//! Disposable subprocess protocol replay. No acoustic model or hardware proof.
use speakerdesk_core::error::ChildError;
use speakerdesk_core::ipc::{
    AsrText, Envelope, EnvelopeBody, Handle, MetricsOut, Request, Response, read_envelope,
    write_envelope,
};
use std::collections::HashSet;
use std::io::{Read, Write};
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;
fn main() {
    assert_eq!(std::env::args().nth(1).as_deref(), Some("--inference"));
    let output = Arc::new(Mutex::new(std::io::stdout()));
    let mut active: Option<(u64, Arc<AtomicBool>)> = None;
    let mut sessions = HashSet::new();
    let mut next_session = 1;
    let mut race: Option<PathBuf> = None;
    let mut race_error = false;
    while let Ok(envelope) = read_envelope(&mut std::io::stdin()) {
        let id = envelope.request_id;
        let EnvelopeBody::Request(request) = envelope.body else {
            std::process::exit(2);
        };
        let result = match request {
            Request::Hello { .. } => Ok(Response::Ready {
                build_id: "fixture".into(),
                device: "replay".into(),
                models: Vec::new(),
            }),
            Request::VadOpen { .. } | Request::DiarOpen { .. } => {
                let handle = Handle {
                    generation: 0,
                    session: next_session,
                };
                next_session += 1;
                sessions.insert(handle.session);
                Ok(Response::Handle(handle))
            }
            Request::VadFeed { handle, .. } | Request::DiarFeed { handle, .. } => {
                assert!(sessions.contains(&handle.session));
                if race_error {
                    Err(ChildError::failed("fixture model error", false))
                } else {
                    Ok(Response::Vad(speakerdesk_core::ipc::VadFeedOut {
                        frames: Vec::new(),
                        received: 0,
                        observed: 0,
                    }))
                }
            }
            Request::VadClose { handle } | Request::DiarClose { handle } => {
                assert!(sessions.remove(&handle.session), "session was not resident");
                Ok(Response::Ack)
            }
            Request::Metrics => Ok(Response::Metrics(MetricsOut {
                peak_mlx_bytes: sessions.len() as u64,
                rss_bytes: std::process::id() as u64,
            })),
            Request::Transcribe { language, .. } if language == "exit" => std::process::exit(17),
            Request::Transcribe { language, .. } if language == "malformed" => {
                output
                    .lock()
                    .unwrap()
                    .write_all(&u32::MAX.to_be_bytes())
                    .unwrap();
                std::process::exit(0);
            }
            Request::Transcribe { language, .. }
                if language.starts_with("race:") || language.starts_with("race-error:") =>
            {
                race_error = language.starts_with("race-error:");
                race = Some(PathBuf::from(language.split_once(':').unwrap().1));
                reply(
                    &output,
                    id,
                    Ok(Response::Text(AsrText {
                        text: "armed".into(),
                        raw_text: "armed".into(),
                        token_count: 1,
                    })),
                );
                continue;
            }
            Request::Transcribe { language, .. } if language.starts_with("block-input:") => {
                reply(
                    &output,
                    id,
                    Ok(Response::Text(AsrText {
                        text: "armed".into(),
                        raw_text: "armed".into(),
                        token_count: 1,
                    })),
                );
                let mut prefix = [0; 4];
                std::io::stdin().read_exact(&mut prefix).unwrap();
                std::fs::write(language.strip_prefix("block-input:").unwrap(), b"started").unwrap();
                loop {
                    std::thread::park();
                }
            }
            Request::Transcribe { language, .. }
                if language.starts_with("slow:") || language.starts_with("hang:") =>
            {
                let (mode, marker) = language.split_once(':').unwrap();
                let hang = mode == "hang";
                let marker = PathBuf::from(marker);
                let flag = Arc::new(AtomicBool::new(false));
                active = Some((id, flag.clone()));
                let writer = output.clone();
                std::thread::spawn(move || {
                    std::fs::write(&marker, b"started").unwrap();
                    if hang {
                        loop {
                            std::thread::park();
                        }
                    }
                    while !flag.load(Ordering::Acquire) {
                        std::thread::sleep(Duration::from_millis(1));
                    }
                    std::fs::write(marker.with_extension("cancelled"), b"cancelled").unwrap();
                    wait_release(&marker);
                    reply(&writer, id, Err(ChildError::Cancelled));
                });
                continue;
            }
            Request::Transcribe { .. } => Ok(Response::Text(AsrText {
                text: "Replay transcript".into(),
                raw_text: "Replay transcript".into(),
                token_count: 2,
            })),
            Request::Cancel { request_id } => {
                if let Some((running, flag)) = &active
                    && *running == request_id
                {
                    flag.store(true, Ordering::Release);
                }
                Ok(Response::Ack)
            }
            Request::Shutdown => {
                write_envelope(
                    &mut *output.lock().unwrap(),
                    Envelope {
                        request_id: id,
                        body: EnvelopeBody::Response(Ok(Response::Ack)),
                    },
                )
                .unwrap();
                return;
            }
            _ => Err(ChildError::Unimplemented),
        };
        if matches!(result, Ok(Response::Handle(_) | Response::Vad(_)) | Err(_))
            && let Some(marker) = race.take()
        {
            let flag = Arc::new(AtomicBool::new(false));
            active = Some((id, flag.clone()));
            let writer = output.clone();
            std::thread::spawn(move || {
                std::fs::write(&marker, b"started").unwrap();
                while !flag.load(Ordering::Acquire) {
                    std::thread::sleep(Duration::from_millis(1));
                }
                std::fs::write(marker.with_extension("cancelled"), b"cancelled").unwrap();
                wait_release(&marker);
                // Deliberately race completion with Cancel: successful state was already created.
                reply(&writer, id, result);
            });
        } else {
            reply(&output, id, result);
        }
    }
}

fn wait_release(marker: &std::path::Path) {
    while !marker.with_extension("release").exists() {
        std::thread::sleep(Duration::from_millis(1));
    }
}

fn reply(output: &Mutex<std::io::Stdout>, id: u64, result: Result<Response, ChildError>) {
    write_envelope(
        &mut *output.lock().unwrap(),
        Envelope {
            request_id: id,
            body: EnvelopeBody::Response(result),
        },
    )
    .unwrap();
}
