#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]
use std::sync::{Mutex, Arc, atomic::{AtomicU16, Ordering}};
use tauri::{Manager, WebviewUrl, WebviewWindowBuilder};
use tauri::webview::DownloadEvent;
use tauri_plugin_shell::{ShellExt, process::{CommandChild, CommandEvent}};
use tauri_plugin_dialog::DialogExt;

struct RuntimeChild(Mutex<Option<CommandChild>>);
fn main() {
    let application = tauri::Builder::default()
        .plugin(tauri_plugin_shell::init())
        .plugin(tauri_plugin_dialog::init())
        .manage(RuntimeChild(Mutex::new(None)))
        .setup(|app| {
            let port = Arc::new(AtomicU16::new(0));
            let navigation_port = port.clone();
            WebviewWindowBuilder::new(app,"main",WebviewUrl::App("index.html".into()))
                .title("Speakerdesk").inner_size(1280.0,800.0).min_inner_size(1000.0,680.0)
                .on_navigation(move |url| {
                    let current = navigation_port.load(Ordering::Relaxed);
                    if current == 0 { return url.scheme()=="tauri" || url.host_str()==Some("tauri.localhost"); }
                    url.host_str()==Some("127.0.0.1") && url.port()==Some(current)
                })
                .on_download(|webview,event| {
                    if let DownloadEvent::Requested{destination,..}=event {
                        let filename=destination.file_name().unwrap_or_default().to_string_lossy();
                        if let Some(path)=webview.app_handle().dialog().file().set_file_name(filename).blocking_save_file() {
                            if let Ok(path)=path.into_path(){*destination=path;return true;}
                        }
                        return false;
                    }
                    true
                }).build()?;
            let storage=std::env::var_os("SPEAKERDESK_HOME").map(std::path::PathBuf::from).unwrap_or(app.path().app_data_dir()?);
            std::fs::create_dir_all(&storage)?;
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
    application.run(|app,event| {
        if matches!(event,tauri::RunEvent::ExitRequested{..}) {
            if let Some(mut child)=app.state::<RuntimeChild>().0.lock().unwrap().take() {
                let _=child.write(b"shutdown\n");
                std::thread::spawn(move || {
                    std::thread::sleep(std::time::Duration::from_secs(5));
                    let _=child.kill();
                });
            }
        }
    });
}
