#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]
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

struct RuntimeChild(Mutex<Option<CommandChild>>);
fn main() {
    let application = tauri::Builder::default()
        .plugin(tauri_plugin_shell::init())
        .plugin(tauri_plugin_dialog::init())
        .manage(RuntimeChild(Mutex::new(None)))
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
                .initialization_script("window.speakerdeskNativeExport = true;")
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
            let (mut rx,child)=app.shell().sidecar("speakerdesk-runtime")?
                .env("SPEAKERDESK_HOME",storage.to_string_lossy().to_string())
                .env("SPEAKERDESK_CAPTURE_HELPER",capture_helper.to_string_lossy().to_string()).spawn()?;
            *app.state::<RuntimeChild>().0.lock().unwrap()=Some(child);
            let handle=app.handle().clone();
            tauri::async_runtime::spawn(async move {
                let mut opened=false;
                while let Some(event)=rx.recv().await {
                    match event {
                        CommandEvent::Stdout(bytes) => {
                            let line=String::from_utf8_lossy(&bytes);
                            if !opened {
                                if let Some(url)=line.trim().strip_prefix("SPEAKERDESK_URL=") {
                                    if let Ok(parsed)=url.parse::<tauri::Url>() {
                                        port.store(parsed.port().unwrap_or(0),Ordering::Relaxed);
                                        if let Some(window)=handle.get_webview_window("main") {
                                            match window.navigate(parsed) {
                                                Ok(())=>{opened=true;eprintln!("Speakerdesk backend ready at {url}");},
                                                Err(err)=>eprintln!("Window error: {err}")
                                            }
                                        }
                                    }
                                }
                            }
                        },
                        CommandEvent::Stderr(bytes) => eprintln!("{}",String::from_utf8_lossy(&bytes)),
                        CommandEvent::Terminated(_) if !opened => {
                            handle.dialog().message("Speakerdesk could not start its local processing service. Close the app and try again.").title("Unable to start").blocking_show();
                        },
                        _=>{}
                    }
                }
            });
            Ok(())
        })
        .build(tauri::generate_context!()).expect("Speakerdesk startup failed");
    application.run(|app, event| {
        if matches!(event, tauri::RunEvent::ExitRequested { .. }) {
            app.state::<ExportDownloads>().shutdown();
            if let Some(mut child) = app.state::<RuntimeChild>().0.lock().unwrap().take() {
                let _ = child.write(b"shutdown\n");
                std::thread::spawn(move || {
                    std::thread::sleep(std::time::Duration::from_secs(5));
                    let _ = child.kill();
                });
            }
        }
    });
}
