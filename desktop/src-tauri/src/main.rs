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
            drop(downloads);
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
}
struct RuntimeChild(Mutex<RuntimeState>);
struct RuntimeSettings {
    storage: PathBuf,
    capture_helper: PathBuf,
    executable: PathBuf,
    port: Arc<AtomicU16>,
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
    let (mut rx, child) = app
        .shell()
        .sidecar("speakerdesk-runtime")
        .map_err(|e| e.to_string())?
        .env("SPEAKERDESK_HOME", &settings.storage)
        .env("SPEAKERDESK_CAPTURE_HELPER", &settings.capture_helper)
        .set_raw_out(true)
        .spawn()
        .map_err(|e| e.to_string())?;
    state.generation = state
        .generation
        .checked_add(1)
        .ok_or("Service generation exhausted")?;
    let generation = state.generation;
    state.child = Some(child);
    state.alive = true;
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
                    if state.generation == generation {
                        state.alive = false;
                        state.child = None;
                    }
                    drop(state);
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
    let settings = app.state::<RuntimeSettings>();
    let executable = &settings.executable;
    let sidecar = executable
        .parent()
        .ok_or("Missing app bundle")?
        .join("speakerdesk-runtime");
    let capture = &settings.capture_helper;
    for path in [executable, &sidecar, capture] {
        let metadata =
            std::fs::metadata(path).map_err(|_| "The existing app bundle is not usable")?;
        if !metadata.is_file() {
            return Err("The existing app bundle is not usable".into());
        }
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            if metadata.permissions().mode() & 0o111 == 0 {
                return Err("The existing app bundle is not executable".into());
            }
        }
    }
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
    let application = tauri::Builder::default()
        .plugin(tauri_plugin_shell::init())
        .plugin(tauri_plugin_dialog::init())
        .plugin(tauri_plugin_updater::Builder::new().build())
        .manage(RuntimeChild(Mutex::new(RuntimeState::default())))
        .manage(ExportDownloads::default())
        .setup(|app| {
            let storage=std::env::var_os("SPEAKERDESK_HOME").map(std::path::PathBuf::from).unwrap_or(app.path().app_data_dir()?);
            std::fs::create_dir_all(&storage)?;
            app.state::<ExportDownloads>().protect_storage(&storage)?;
            let port = Arc::new(AtomicU16::new(0));
            let navigation_port = port.clone();
            let download_port = port.clone();
            WebviewWindowBuilder::new(app,"main",WebviewUrl::App("index.html".into()))
                .title("Speakerdesk").inner_size(1280.0,800.0).min_inner_size(1000.0,680.0)
                .initialization_script("window.speakerdeskNativeExport = true; window.speakerdeskNativeUpdater = true;")
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
            app.manage(RuntimeSettings { storage, capture_helper, executable, port });
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
            if let Some(mut child) = app.state::<RuntimeChild>().0.lock().unwrap_or_else(|p| p.into_inner()).child.take() {
                let _ = child.write(b"shutdown\n");
                std::thread::spawn(move || {
                    std::thread::sleep(std::time::Duration::from_secs(5));
                    let _ = child.kill();
                });
            }
        }
    });
}
