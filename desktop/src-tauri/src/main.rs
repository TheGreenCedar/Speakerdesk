#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]
use std::path::PathBuf;
use std::sync::{
    atomic::{AtomicU16, Ordering},
    Arc, Mutex,
};
use tauri::webview::DownloadEvent;
use tauri::{Manager, WebviewUrl, WebviewWindowBuilder};
use tauri_plugin_dialog::DialogExt;
use tauri_plugin_shell::{
    process::{CommandChild, CommandEvent},
    ShellExt,
};
mod exports;
#[cfg(target_os = "macos")]
mod menus;
mod update_control;
mod update_location;
mod updates;
use exports::{ExportDownloads, ExportJob, Finished};

fn export_nonce(url: &tauri::Url, port: u16) -> Option<String> {
    if port == 0
        || url.scheme() != "http"
        || url.host_str() != Some("127.0.0.1")
        || url.port() != Some(port)
        || !url.username().is_empty()
        || url.password().is_some()
        || url.fragment().is_some()
        || !exports::allowed_path(url.path())
    {
        return None;
    }
    let query: Vec<_> = url.query_pairs().collect();
    if query.len() != 1 || query[0].0 != "download" || !exports::valid_nonce(&query[0].1) {
        return None;
    }
    Some(query[0].1.to_string())
}

fn export_status(webview: &tauri::Webview, nonce: &str, phase: &str) {
    if exports::valid_nonce(nonce) {
        let _ = webview.eval(format!(
            "window.speakerdeskExportStatus?.('{nonce}', '{phase}')"
        ));
    }
}

fn choose_export(webview: tauri::Webview, job: Arc<ExportJob>) {
    export_status(&webview, &job.nonce, "choosing");
    let parent = webview.window();
    webview
        .app_handle()
        .dialog()
        .file()
        .set_parent(&parent)
        .set_file_name(job.filename.clone())
        .save_file(move |path| {
            let downloads = webview.app_handle().state::<ExportDownloads>();
            let Some(path) = path else {
                if downloads.cancel(&job) {
                    export_status(&webview, &job.nonce, "cancelled");
                }
                return;
            };
            let Ok(path) = path.into_path() else {
                if downloads.cancel(&job) {
                    export_status(&webview, &job.nonce, "failed");
                }
                return;
            };
            if !downloads.start_write(&job) {
                return;
            }
            export_status(&webview, &job.nonce, "writing");
            std::thread::spawn(move || {
                let saved = job.save_to(&path).is_ok();
                let downloads = webview.app_handle().state::<ExportDownloads>();
                let current = if saved {
                    downloads.release(&job)
                } else {
                    downloads.cancel(&job)
                };
                if current {
                    export_status(&webview, &job.nonce, if saved { "saved" } else { "failed" });
                }
            });
        });
}

#[derive(Default)]
struct RuntimeState {
    child: Option<CommandChild>,
    generation: u64,
    alive: bool,
    quitting: bool,
}
struct RuntimeChild(Mutex<RuntimeState>);
struct RuntimeSettings {
    storage: PathBuf,
    capture_helper: PathBuf,
    executable: PathBuf,
    port: Arc<AtomicU16>,
    models: PathBuf,
}

impl RuntimeSettings {
    fn new(
        storage: PathBuf,
        models: Option<PathBuf>,
        capture_helper: PathBuf,
        executable: PathBuf,
        port: Arc<AtomicU16>,
    ) -> Self {
        let models = models.unwrap_or_else(|| storage.join("models"));
        Self {
            storage,
            models,
            capture_helper,
            executable,
            port,
        }
    }
    fn command<R: tauri::Runtime>(
        &self,
        app: &tauri::AppHandle<R>,
    ) -> tauri_plugin_shell::process::Command {
        app.shell()
            .command(&self.executable)
            .set_process_group(true)
            .args([
                std::ffi::OsStr::new("--native-host"),
                self.storage.as_os_str(),
                self.models.as_os_str(),
            ])
            .env("SPEAKERDESK_HOME", &self.storage)
            .env("SPEAKERDESK_CAPTURE_HELPER", &self.capture_helper)
            .env(
                "SPEAKERDESK_ECHO_LIBRARY",
                self.capture_helper
                    .with_file_name("libspeakerdesk_echo.dylib"),
            )
            .set_raw_out(true)
    }

    fn validate_artifacts(&self) -> Result<(), String> {
        let resources = self
            .capture_helper
            .parent()
            .ok_or("Missing app resources")?;
        let contents = resources.parent().ok_or("Missing app bundle")?;
        let contents = contents.canonicalize().map_err(|_| "Missing app bundle")?;
        for (path, executable) in [
            (self.executable.clone(), true),
            (self.capture_helper.clone(), true),
            (resources.join("speakerdesk-capture.capability.json"), false),
            (resources.join("libspeakerdesk_echo.dylib"), false),
            (resources.join("mlx/mlx.metallib"), false),
            (contents.join("Frameworks/libmlx.dylib"), false),
        ] {
            let metadata = std::fs::symlink_metadata(&path).map_err(|_| {
                format!(
                    "A required native app resource is missing: {}",
                    path.display()
                )
            })?;
            if !metadata.is_file() || metadata.len() == 0 {
                return Err(format!(
                    "A native app resource is unusable: {}",
                    path.display()
                ));
            }
            if !path
                .canonicalize()
                .map_err(|_| "A native app resource is unavailable")?
                .starts_with(&contents)
            {
                return Err(format!(
                    "A native app resource escapes the bundle: {}",
                    path.display()
                ));
            }
            #[cfg(unix)]
            {
                use std::os::unix::fs::MetadataExt;
                if (metadata.uid() != 0 && metadata.uid() != unsafe { libc::getuid() })
                    || metadata.mode() & 0o022 != 0
                    || (executable && metadata.mode() & 0o111 == 0)
                {
                    return Err(format!(
                        "A native app resource has unsafe permissions: {}",
                        path.display()
                    ));
                }
            }
            #[cfg(not(unix))]
            let _ = executable;
        }
        speakerdesk_host::capture::CaptureSupervisor::check_helper(&self.capture_helper)
            .map_err(|error| error.to_string())
    }
}

fn write_runtime(app: &tauri::AppHandle, generation: u64, bytes: &[u8]) -> Result<(), String> {
    let runtime = app.state::<RuntimeChild>();
    let mut state = runtime.0.lock().unwrap_or_else(|p| p.into_inner());
    if generation != state.generation || !state.alive {
        return Err("The local service is unavailable".into());
    }
    state
        .child
        .as_mut()
        .ok_or("The local service is unavailable")?
        .write(bytes)
        .map_err(|e| e.to_string())
}

fn start_runtime(app: &tauri::AppHandle) -> Result<(), String> {
    let runtime = app.state::<RuntimeChild>();
    let settings = app.state::<RuntimeSettings>();
    let mut state = runtime.0.lock().unwrap_or_else(|p| p.into_inner());
    // A missing event stream is not proof that the existing service has exited.
    if state.alive || state.child.is_some() {
        return Err("A local service is already running".into());
    }
    let (mut rx, child) = settings.command(app).spawn().map_err(|e| e.to_string())?;
    state.generation = state
        .generation
        .checked_add(1)
        .ok_or("Service generation exhausted")?;
    let generation = state.generation;
    state.child = Some(child);
    state.alive = true;
    state.quitting = false;
    settings.port.store(0, Ordering::Relaxed);
    let port = settings.port.clone();
    let handle = app.clone();
    let updates = app.state::<updates::Updates>().inner().clone();
    drop(state);
    tauri::async_runtime::spawn(async move {
        let mut opened = false;
        let mut lines = update_control::Lines::default();
        let mut terminated = false;
        while let Some(event) = rx.recv().await {
            match event {
                CommandEvent::Stdout(bytes) => {
                    for line in lines.push(&bytes) {
                        if let Some(packet) = line.strip_prefix(b"SPEAKERDESK_UPDATE=") {
                            updates.packet(generation, packet);
                        } else if !opened {
                            if let Some(url) = line
                                .strip_prefix(b"SPEAKERDESK_URL=")
                                .and_then(|value| std::str::from_utf8(value).ok())
                            {
                                if let Ok(parsed) = url.parse::<tauri::Url>() {
                                    if parsed.scheme() != "http"
                                        || parsed.host_str() != Some("127.0.0.1")
                                        || parsed.port().is_none()
                                        || !parsed.username().is_empty()
                                        || parsed.password().is_some()
                                        || parsed.query().is_some()
                                        || parsed.fragment().is_some()
                                        || parsed.path() != "/"
                                    {
                                        continue;
                                    }
                                    port.store(parsed.port().unwrap(), Ordering::Relaxed);
                                    if let Some(window) = handle.get_webview_window("main") {
                                        match window.navigate(parsed) {
                                            Ok(()) => {
                                                opened = true;
                                                updates.event(updates::Event::Started(generation));
                                            }
                                            Err(error) => eprintln!("Window error: {error}"),
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
                CommandEvent::Stderr(bytes) => eprintln!("{}", String::from_utf8_lossy(&bytes)),
                CommandEvent::Terminated(payload) => {
                    terminated = true;
                    let runtime = handle.state::<RuntimeChild>();
                    let mut state = runtime.0.lock().unwrap_or_else(|p| p.into_inner());
                    let quitting = state.generation == generation && state.quitting;
                    if state.generation == generation {
                        state.alive = false;
                        state.child = None;
                    }
                    drop(state);
                    if quitting {
                        handle.exit(if payload.code == Some(0) { 0 } else { 1 });
                        continue;
                    }
                    updates.event(updates::Event::Terminated(
                        generation,
                        payload.code == Some(0) && payload.signal.is_none(),
                    ));
                    if !opened {
                        handle.dialog().message("Speakerdesk could not start its local processing service. Close the app and try again.").title("Unable to start").show(|_| {});
                    }
                }
                CommandEvent::Error(_) => updates.event(updates::Event::TransportLost(generation)),
                _ => {}
            }
        }
        if !terminated {
            updates.event(updates::Event::TransportLost(generation));
        }
    });
    Ok(())
}

fn recover_runtime(app: &tauri::AppHandle) -> Result<(), String> {
    app.state::<RuntimeSettings>().validate_artifacts()?;
    start_runtime(app)
}

fn recovery_description(app: &tauri::AppHandle) -> String {
    let settings = app.state::<RuntimeSettings>();
    let backup = Some(&settings.executable).and_then(|exe| {
        exe.ancestors()
            .find(|path| path.extension().is_some_and(|ext| ext == "app"))
            .and_then(|bundle| {
                bundle
                    .parent()
                    .map(|parent| parent.join("Speakerdesk (previous version).app"))
            })
    });
    match backup.filter(|path| path.exists()) {
        Some(path) => format!("The previous app is preserved at {}. Reopen that app or install the approved DMG; your recording storage is separate.", path.display()),
        None => "If the updater reports a preserved previous-app path, reopen that app. Otherwise reinstall the approved DMG; your recording storage is separate.".into(),
    }
}
fn main() {
    if std::env::args_os().nth(1).as_deref() == Some("--native-host".as_ref()) {
        let args: Vec<_> = std::env::args_os().skip(2).collect();
        if args.len() != 2 {
            eprintln!("native host requires explicit storage and model directories");
            std::process::exit(1);
        }
        if let Err(error) = speakerdesk_host::native::run(
            args[0].clone().into(),
            args[1].clone().into(),
            std::env::current_exe().expect("application executable"),
        ) {
            eprintln!("Native host failed: {error}");
            std::process::exit(1);
        }
        return;
    }
    if std::env::args_os().nth(1).as_deref() == Some("--inference".as_ref()) {
        let Some(model_root) = std::env::args_os().nth(2) else {
            eprintln!("inference requires an explicit model directory");
            std::process::exit(1);
        };
        if let Err(error) = speakerdesk_models::serve(model_root.into()) {
            eprintln!("{error}");
            std::process::exit(1);
        }
        return;
    }
    let application = tauri::Builder::default()
        .plugin(tauri_plugin_shell::init())
        .plugin(tauri_plugin_dialog::init())
        .plugin(tauri_plugin_updater::Builder::new().build())
        .manage(RuntimeChild(Mutex::new(RuntimeState::default())))
        .manage(ExportDownloads::default())
        .setup(|app| {
            #[cfg(target_os = "macos")]
            menus::install(app)?;
            let storage=std::env::var_os("SPEAKERDESK_HOME").map(std::path::PathBuf::from).unwrap_or(app.path().app_data_dir()?);
            let models=std::env::var_os("SPEAKERDESK_NATIVE_MODELS").map(PathBuf::from);
            std::fs::create_dir_all(&storage)?;
            app.state::<ExportDownloads>().protect_storage(&storage)?;
            let port = Arc::new(AtomicU16::new(0));
            let navigation_port = port.clone();
            let download_port = port.clone();
            WebviewWindowBuilder::new(app,"main",WebviewUrl::App("index.html".into()))
                .title("Speakerdesk").inner_size(1280.0,800.0).min_inner_size(1000.0,680.0)
                .initialization_script(if cfg!(target_os = "macos") {
                    "window.speakerdeskNativeExport = true; window.speakerdeskNativeUpdater = true; window.speakerdeskNativeMenu = true;"
                } else {
                    "window.speakerdeskNativeExport = true; window.speakerdeskNativeUpdater = true;"
                })
                .on_navigation(move |url| {
                    let current = navigation_port.load(Ordering::Relaxed);
                    if current == 0 { return url.scheme()=="tauri" || url.host_str()==Some("tauri.localhost"); }
                    url.host_str()==Some("127.0.0.1") && url.port()==Some(current)
                })
                .on_download(move |webview,event| {
                    match event {
                        DownloadEvent::Requested{url,destination} => {
                            let Some(nonce) = export_nonce(&url, download_port.load(Ordering::Relaxed)) else { return false; };
                            let filename=destination.file_name().unwrap_or_default().to_string_lossy();
                            match webview.app_handle().state::<ExportDownloads>().begin(url.as_str(), &nonce, &filename) {
                                Ok(path) => { *destination = path; export_status(&webview, &nonce, "downloading"); return true; }
                                Err(_) => { export_status(&webview, &nonce, "failed"); return false; }
                            }
                        }
                        DownloadEvent::Finished{url,success,..} => {
                            let finished=webview.app_handle().state::<ExportDownloads>().finished(url.as_str(),success);
                            match finished {
                                Finished::Ready(job) => choose_export(webview,job),
                                Finished::Failed(nonce) => export_status(&webview,&nonce,"failed"),
                                Finished::Ignored => {},
                            }
                        }
                        _ => {},
                    }
                    true
                }).build()?;
            let capture_helper=app.path().resource_dir()?.join("speakerdesk-capture");
            let executable = std::env::current_exe()?;
            app.manage(RuntimeSettings::new(storage, models, capture_helper, executable, port));
            app.manage(updates::Updates::start(app.handle().clone()));
            start_runtime(app.handle()).map_err(std::io::Error::other)?;
            Ok(())
        })
        .build(tauri::generate_context!()).expect("Speakerdesk startup failed");
    application.run(|app, event| {
        if let tauri::RunEvent::ExitRequested { api, .. } = event {
            if app.state::<updates::Updates>().replacing_bundle()
                || (app.state::<updates::Updates>().protects_exit()
                && app.state::<RuntimeChild>().0.lock().unwrap_or_else(|p| p.into_inner()).alive) {
                api.prevent_exit();
                app.dialog().message("Speakerdesk is waiting for its local service to stop safely before updating. Keep the app open; no recording work will be killed by the updater.").title("Update in progress").show(|_| {});
                return;
            }
            app.state::<ExportDownloads>().shutdown();
            {
                let runtime = app.state::<RuntimeChild>();
                let mut state = runtime.0.lock().unwrap_or_else(|p| p.into_inner());
                if state.alive {
                    api.prevent_exit();
                    if !state.quitting {
                        state.quitting = true;
                        if let Some(child) = &mut state.child { let _ = child.write(b"shutdown\n"); }
                        let generation = state.generation;
                        let handle = app.clone();
                        tauri::async_runtime::spawn(async move {
                            tokio::time::sleep(std::time::Duration::from_secs(45)).await;
                            let runtime = handle.state::<RuntimeChild>();
                            let mut state = runtime.0.lock().unwrap_or_else(|p| p.into_inner());
                            if state.alive && state.quitting && state.generation == generation {
                                if let Some(child) = state.child.take() { let _ = child.kill(); }
                            }
                        });
                    }
                }
            }
        }
    });
}

#[cfg(all(test, unix))]
mod runtime_tests {
    use super::*;
    use std::{fs, os::unix::fs::PermissionsExt, sync::atomic::AtomicU64};
    static NEXT: AtomicU64 = AtomicU64::new(0);
    struct Bundle(PathBuf);
    impl Bundle {
        fn new() -> Self {
            let path = std::env::temp_dir().join(format!(
                "speakerdesk-native-startup-{}-{}",
                std::process::id(),
                NEXT.fetch_add(1, Ordering::Relaxed)
            ));
            fs::create_dir(&path).unwrap();
            for relative in [
                "Contents/MacOS/speakerdesk",
                "Contents/Resources/speakerdesk-capture",
                "Contents/Resources/speakerdesk-capture.capability.json",
                "Contents/Resources/libspeakerdesk_echo.dylib",
                "Contents/Resources/mlx/mlx.metallib",
                "Contents/Frameworks/libmlx.dylib",
            ] {
                let file = path.join(relative);
                fs::create_dir_all(file.parent().unwrap()).unwrap();
                fs::write(&file, b"native boundary fixture").unwrap();
                fs::set_permissions(file, fs::Permissions::from_mode(0o755)).unwrap();
            }
            fs::write(path.join("Contents/Resources/speakerdesk-capture.capability.json"),
                serde_json::to_vec(&serde_json::json!({
                    "helper_sha256":"1bb259b6f174a8e110b54a5713d09965effe3ce5de48f1747e5925e493d9025b",
                    "export_contract":"apple_processed_mono16k_v1",
                    "capability_version":"apple_input_node_output_route_v1"
                })).unwrap()).unwrap();
            Self(path)
        }
        fn settings(&self, models: Option<PathBuf>) -> RuntimeSettings {
            RuntimeSettings::new(
                self.0.join("isolated data"),
                models,
                self.0.join("Contents/Resources/speakerdesk-capture"),
                self.0.join("Contents/MacOS/speakerdesk"),
                Arc::new(AtomicU16::new(0)),
            )
        }
    }
    impl Drop for Bundle {
        fn drop(&mut self) {
            let _ = fs::remove_dir_all(&self.0);
        }
    }

    #[test]
    fn forced_runtime_shutdown_terminates_owned_descendants() {
        let bundle = Bundle::new();
        let app = tauri::test::mock_builder()
            .plugin(tauri_plugin_shell::init())
            .build(tauri::test::mock_context(tauri::test::noop_assets()))
            .unwrap();
        fs::write(
            bundle.0.join("Contents/MacOS/speakerdesk"),
            b"#!/bin/sh\n/bin/sleep 30 &\nprintf '%s\\n' \"$!\"\nwait\n",
        )
        .unwrap();
        let (mut events, child) = bundle.settings(None).command(app.handle()).spawn().unwrap();
        let descendant = tauri::async_runtime::block_on(async {
            tokio::time::timeout(std::time::Duration::from_secs(5), async {
                let mut bytes = Vec::new();
                while let Some(event) = events.recv().await {
                    if let CommandEvent::Stdout(chunk) = event {
                        bytes.extend(chunk);
                        if bytes.contains(&b'\n') {
                            return String::from_utf8(bytes)
                                .unwrap()
                                .trim()
                                .parse::<i32>()
                                .unwrap();
                        }
                    }
                }
                panic!("Host did not report its owned descendant")
            })
            .await
        });
        child.kill().unwrap();
        let pid = descendant.unwrap();
        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(2);
        while unsafe { libc::kill(pid, 0) } == 0 && std::time::Instant::now() < deadline {
            std::thread::sleep(std::time::Duration::from_millis(10));
        }
        let survived = unsafe { libc::kill(pid, 0) } == 0;
        if survived {
            // Clean up the real fixture process even when this regression fails.
            unsafe { libc::kill(pid, libc::SIGKILL) };
        }
        assert!(
            !survived,
            "Forced host shutdown left its descendant running"
        );
    }

    #[test]
    fn native_startup_executes_owned_host_with_default_or_explicit_models() {
        let bundle = Bundle::new();
        let app = tauri::test::mock_builder()
            .plugin(tauri_plugin_shell::init())
            .build(tauri::test::mock_context(tauri::test::noop_assets()))
            .unwrap();
        let exe = bundle.0.join("Contents/MacOS/speakerdesk");
        // Process replay only: records the actual shell command arguments/environment.
        fs::write(&exe, b"#!/bin/sh\nprintf '%s\\n' \"$@\" \"$SPEAKERDESK_HOME\" \"$SPEAKERDESK_CAPTURE_HELPER\" \"$SPEAKERDESK_ECHO_LIBRARY\"\n").unwrap();
        for name in [
            "speakerdesk-capture",
            "speakerdesk-capture.capability.json",
            "libspeakerdesk_echo.dylib",
        ] {
            fs::remove_file(bundle.0.join("Contents/Resources").join(name)).unwrap();
        }
        for models in [None, Some(bundle.0.join("other models"))] {
            let settings = bundle.settings(models.clone());
            assert!(
                settings.validate_artifacts().is_err(),
                "Incomplete capture resources block recovery, not the ordinary editor host"
            );
            let mut command: std::process::Command = settings.command(app.handle()).into();
            command
                .env_remove("DYLD_LIBRARY_PATH")
                .env_remove("PYTHONHOME")
                .env_remove("PYTHONPATH");
            let output = command.output().unwrap();
            assert!(
                output.status.success(),
                "The ordinary host must launch while capture is unavailable"
            );
            let lines: Vec<_> = String::from_utf8(output.stdout)
                .unwrap()
                .lines()
                .map(str::to_owned)
                .collect();
            assert_eq!(
                lines,
                vec![
                    "--native-host".to_owned(),
                    settings.storage.display().to_string(),
                    models
                        .unwrap_or_else(|| settings.storage.join("models"))
                        .display()
                        .to_string(),
                    settings.storage.display().to_string(),
                    settings.capture_helper.display().to_string(),
                    settings
                        .capture_helper
                        .with_file_name("libspeakerdesk_echo.dylib")
                        .display()
                        .to_string()
                ]
            );
        }
        assert!(!bundle.0.join("Contents/MacOS/speakerdesk-runtime").exists());
    }

    #[test]
    fn recovery_rejects_missing_unsafe_or_escaping_native_resources() {
        let bundle = Bundle::new();
        let settings = bundle.settings(None);
        for relative in [
            "Contents/MacOS/speakerdesk",
            "Contents/Resources/speakerdesk-capture",
            "Contents/Resources/speakerdesk-capture.capability.json",
            "Contents/Resources/libspeakerdesk_echo.dylib",
            "Contents/Resources/mlx/mlx.metallib",
            "Contents/Frameworks/libmlx.dylib",
        ] {
            let file = bundle.0.join(relative);
            let bytes = fs::read(&file).unwrap();
            fs::remove_file(&file).unwrap();
            assert!(settings.validate_artifacts().is_err(), "{relative}");
            fs::write(&file, bytes).unwrap();
            fs::set_permissions(&file, fs::Permissions::from_mode(0o777)).unwrap();
            assert!(settings.validate_artifacts().is_err(), "{relative}");
            fs::set_permissions(&file, fs::Permissions::from_mode(0o755)).unwrap();
        }
        settings.validate_artifacts().unwrap();
        fs::write(&settings.capture_helper, b"replaced helper").unwrap();
        assert!(settings
            .validate_artifacts()
            .unwrap_err()
            .contains("differs"));
        fs::write(&settings.capture_helper, b"native boundary fixture").unwrap();
        let outside = Bundle::new();
        fs::remove_dir_all(bundle.0.join("Contents/Frameworks")).unwrap();
        std::os::unix::fs::symlink(
            outside.0.join("Contents/Frameworks"),
            bundle.0.join("Contents/Frameworks"),
        )
        .unwrap();
        assert!(settings
            .validate_artifacts()
            .unwrap_err()
            .contains("escapes"));
    }
}
