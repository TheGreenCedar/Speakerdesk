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
#[path = "inference-fixture/capture.rs"]
mod capture;
fn main() {
    if std::env::args().nth(1).as_deref() == Some("--capture") {
        capture::run();
        return;
    }
    if std::env::args().nth(1).as_deref() == Some("--native-host") {
        let args: Vec<_> = std::env::args_os().skip(2).collect();
        if let Err(error) = speakerdesk_host::native::run(
            args[0].clone().into(),
            args[1].clone().into(),
            std::env::current_exe().unwrap(),
        ) {
            eprintln!("{error}");
            std::process::exit(1);
        }
        return;
    }
    assert_eq!(std::env::args().nth(1).as_deref(), Some("--inference"));
    let output = Arc::new(Mutex::new(std::io::stdout()));
    let mut active: Option<(u64, Arc<AtomicBool>)> = None;
    let mut sessions = HashSet::new();
    let mut diar_progress = std::collections::HashMap::<u64, (u64, u64)>::new();
    let mut vad_progress = std::collections::HashMap::<u64, (u64, u64)>::new();
    let mut live_diar = HashSet::new();
    let mut next_session = 1;
    let mut race: Option<PathBuf> = None;
    let mut race_error = false;
    let root = PathBuf::from(std::env::args_os().nth(2).unwrap());
    let mut held_live_feed = false;
    let mut held_live_asr = false;
    while let Ok(envelope) = read_envelope(&mut std::io::stdin()) {
        let id = envelope.request_id;
        let EnvelopeBody::Request(request) = envelope.body else {
            std::process::exit(2);
        };
        let hold_asr = matches!(request, Request::Transcribe { .. })
            && root.join("fixture-live").exists()
            && root.join("hold-live-asr").exists()
            && !held_live_asr;
        let hold_eof = matches!(request, Request::VadFeed { last: true, .. })
            && root.join("fixture-live").exists()
            && root.join("hold-live-eof").exists();
        if matches!(request, Request::Transcribe { .. })
            && root.join("fixture-live").exists()
            && root.join("exit-live-asr-once").exists()
        {
            std::fs::remove_file(root.join("exit-live-asr-once")).unwrap();
            std::process::exit(19);
        }
        let hold_live = matches!(request, Request::VadFeed { .. })
            && root.join("fixture-live").exists()
            && root.join("hold-live-feed").exists()
            && !held_live_feed;
        let result = match request {
            Request::Hello { build_id } => {
                if root.join("fixture-live").exists() {
                    std::fs::write(
                        root.join("fixture-child-pid"),
                        std::process::id().to_string(),
                    )
                    .unwrap();
                    writeln!(
                        std::fs::OpenOptions::new()
                            .create(true)
                            .append(true)
                            .open(root.join("fixture-child-pids"))
                            .unwrap(),
                        "{}",
                        std::process::id()
                    )
                    .unwrap();
                }
                Ok(Response::Ready {
                    build_id: if build_id == env!("CARGO_PKG_VERSION") {
                        build_id
                    } else {
                        "fixture".into()
                    },
                    device: "replay".into(),
                    models: Vec::new(),
                })
            }
            Request::SetupCheck { models } => {
                // Only the native-host lifecycle replay asks for setup; retain its owned PID.
                let root = PathBuf::from(std::env::args_os().nth(2).unwrap());
                std::fs::write(
                    root.join("fixture-child-pid"),
                    std::process::id().to_string(),
                )
                .unwrap();
                Ok(Response::Setup(
                    models
                        .into_iter()
                        .map(|model| speakerdesk_core::ipc::SetupModelReport {
                            // Protocol replay only: no tensors are initialized in this fixture.
                            model: match model.as_str() {
                                "silero-speech" => "mlx-community/silero-vad-v6@2ebf4a5".into(),
                                "nemotron" => "mlx-community/Nemotron-3-Diarization@59ed2dbfc1346dcea9d423c71306a3a2499c568f".into(),
                                "cohere-speech" => "spokedotso/cohere-transcribe-03-2026-mlx-4bit@064e51eab6db47066cbeaa85e2894b5691bd8d12".into(),
                                _ => format!("replay:{model}"),
                            },
                            ok: true,
                            timing_ms: 0,
                        })
                        .collect(),
                ))
            }
            Request::VadOpen { .. }
            | Request::DiarOpen {
                preset: speakerdesk_core::ipc::DiarPreset::Offline,
            } => {
                let handle = Handle {
                    generation: 0,
                    session: next_session,
                };
                next_session += 1;
                sessions.insert(handle.session);
                Ok(Response::Handle(handle))
            }
            Request::DiarOpen {
                preset: speakerdesk_core::ipc::DiarPreset::Low,
            } => {
                let handle = Handle {
                    generation: 0,
                    session: next_session,
                };
                next_session += 1;
                sessions.insert(handle.session);
                live_diar.insert(handle.session);
                Ok(Response::Handle(handle))
            }
            Request::VadFeed { handle, pcm, last } => {
                assert!(sessions.contains(&handle.session));
                if race_error {
                    Err(ChildError::failed("fixture model error", false))
                } else {
                    let (received, observed) = vad_progress.entry(handle.session).or_default();
                    *received += pcm.len() as u64;
                    let end = if last {
                        *received
                    } else {
                        *received / 512 * 512
                    };
                    let mut frames = Vec::new();
                    while *observed < end {
                        let next = (*observed + 512).min(end);
                        frames.push(speakerdesk_core::ipc::VadFrame {
                            start: *observed,
                            end: next,
                            raw_p: 0.99,
                            norm_p: 0.99,
                            gain: 1.,
                            constant: false,
                        });
                        *observed = next;
                    }
                    Ok(Response::Vad(speakerdesk_core::ipc::VadFeedOut {
                        frames,
                        received: *received,
                        observed: *observed,
                    }))
                }
            }
            Request::DiarFeed { handle, pcm, last } => {
                assert!(sessions.contains(&handle.session));
                assert!(
                    pcm.len() <= 16000 * 10,
                    "whole-recording IPC was not removed"
                );
                if race_error || pcm.iter().any(|x| !x.is_finite()) {
                    Err(ChildError::failed("fixture model error", false))
                } else {
                    let (received, observed) = diar_progress.entry(handle.session).or_default();
                    *received += pcm.len() as u64;
                    // Deliberately retain every observation until EOF.
                    let end = if last || live_diar.contains(&handle.session) {
                        *received / 160 * 160
                    } else {
                        *observed
                    };
                    let turns = if end > *observed {
                        vec![speakerdesk_core::ipc::DiarTurn {
                            start: *observed,
                            end,
                            speaker: "speaker_0".into(),
                        }]
                    } else {
                        Vec::new()
                    };
                    *observed = end;
                    Ok(Response::Diar(speakerdesk_core::ipc::DiarFeedOut {
                        turns,
                        received: *received,
                        observed: *observed,
                    }))
                }
            }
            Request::VadClose { handle } | Request::DiarClose { handle } => {
                // Production Runtime closes idempotently. Cancellation can close a discarded
                // successful feed before Engine::drop releases its retained handle.
                sessions.remove(&handle.session);
                diar_progress.remove(&handle.session);
                vad_progress.remove(&handle.session);
                live_diar.remove(&handle.session);
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
        if hold_live || hold_eof || hold_asr {
            if hold_asr {
                held_live_asr = true;
            } else {
                held_live_feed = true;
            }
            let flag = Arc::new(AtomicBool::new(false));
            active = Some((id, flag.clone()));
            let writer = output.clone();
            let root = root.clone();
            std::thread::spawn(move || {
                let (marker, hold) = if hold_asr {
                    ("edit-asr-started", "hold-live-asr")
                } else if hold_eof {
                    ("recovery-eof-started", "hold-live-eof")
                } else {
                    ("recovery-feed-started", "hold-live-feed")
                };
                std::fs::write(root.join(marker), b"started").unwrap();
                while root.join(hold).exists() && !flag.load(Ordering::Acquire) {
                    std::thread::sleep(Duration::from_millis(1));
                }
                reply(
                    &writer,
                    id,
                    if flag.load(Ordering::Acquire) {
                        Err(ChildError::Cancelled)
                    } else {
                        result
                    },
                );
            });
            continue;
        }
        if matches!(
            result,
            Ok(Response::Handle(_) | Response::Vad(_) | Response::Diar(_)) | Err(_)
        ) && let Some(marker) = race.take()
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
