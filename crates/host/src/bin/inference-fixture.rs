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
    let mut context_index = 0usize;
    let mut held_live_feed = false;
    let mut held_live_asr = false;
    let mut held_context: Option<usize> = None;
    while let Ok(envelope) = read_envelope(&mut std::io::stdin()) {
        let id = envelope.request_id;
        let EnvelopeBody::Request(request) = envelope.body else {
            std::process::exit(2);
        };
        let script: Option<serde_json::Value> = std::fs::read(root.join("context-replay.json"))
            .ok()
            .map(|bytes| serde_json::from_slice(&bytes).unwrap());
        let context_asr = match &request {
            Request::Transcribe { pcm, .. } => script.as_ref().is_some_and(|script| {
                script.get(context_index).is_some_and(|context| {
                    context["request_samples"]
                        .as_u64()
                        .is_none_or(|samples| samples == pcm.len() as u64)
                })
            }),
            _ => false,
        };
        let prior_calls = std::fs::read_to_string(root.join("transcribe-languages"))
            .unwrap_or_default()
            .lines()
            .count();
        let hold_asr = matches!(request, Request::Transcribe { .. })
            && (script
                .as_ref()
                .is_none_or(|script| script[0]["request_samples"].is_null())
                || context_asr)
            && std::fs::read_to_string(root.join("hold-asr-call"))
                .ok()
                .and_then(|v| v.parse::<usize>().ok())
                .is_none_or(|n| n == prior_calls)
            && root.join("fixture-live").exists()
            && root.join("hold-live-asr").exists()
            && (!held_live_asr || (context_asr && held_context != Some(context_index)))
            && std::fs::read_to_string(root.join("hold-context-index"))
                .ok()
                .and_then(|s| s.parse::<usize>().ok())
                .is_none_or(|index| index == context_index);
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
        let hold_voice =
            matches!(request, Request::EmbedVoice { .. }) && root.join("hold-live-voice").exists();
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
                                "coarse-alignment" if root.join("fixture-ctc").exists() => speakerdesk_core::alignment::CHECKPOINT.into(),
                                "whisper-language" => speakerdesk_core::languages::LID_CHECKPOINT.into(),
                                "redimnet2-b6" if root.join("fixture-voice").exists() => "112dd8f4f836abdf8a420e66a5e2885cf8ec64ab:sd-pcm-v1:native-coreml-v1:CPU_AND_GPU".into(),
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
            Request::LanguageProbe { pcm } => {
                writeln!(
                    std::fs::OpenOptions::new()
                        .create(true)
                        .append(true)
                        .open(root.join("language-probe-calls"))
                        .unwrap(),
                    "{}",
                    pcm.len()
                )
                .unwrap();
                let context_probe = std::fs::read(root.join("context-replay.json"))
                    .ok()
                    .and_then(|bytes| serde_json::from_slice::<serde_json::Value>(&bytes).ok())
                    .and_then(|contexts| contexts.get(context_index)?.get("probe").cloned());
                let scores = context_probe
                    .map(|probe| serde_json::from_value(probe).unwrap())
                    .unwrap_or_else(|| {
                        std::fs::read(root.join("replay-probe.json"))
                            .map(|bytes| serde_json::from_slice(&bytes).unwrap())
                            .unwrap_or_else(|_| speakerdesk_core::ipc::LanguageProbeOut {
                                probabilities: vec![0.99, 0.01],
                                codes: vec!["en".into(), "de".into()],
                            })
                    });
                Ok(Response::Probe(scores))
            }
            Request::Transcribe { language, pcm, .. } => {
                use sha2::{Digest, Sha256};
                let digest = format!(
                    "{:x}",
                    Sha256::digest(pcm.iter().flat_map(|v| v.to_le_bytes()).collect::<Vec<_>>())
                );
                writeln!(
                    std::fs::OpenOptions::new()
                        .create(true)
                        .append(true)
                        .open(root.join("transcribe-crops"))
                        .unwrap(),
                    "{}",
                    serde_json::json!({"samples":pcm.len(),"sha256":digest,"language":language})
                )
                .unwrap();
                writeln!(
                    std::fs::OpenOptions::new()
                        .create(true)
                        .append(true)
                        .open(root.join("transcribe-languages"))
                        .unwrap(),
                    "{language}"
                )
                .unwrap();
                let text = if context_asr {
                    let contexts = script.as_ref().expect("scripted context");
                    let context = &contexts[context_index];
                    context_index += 1;
                    writeln!(
                        std::fs::OpenOptions::new()
                            .create(true)
                            .append(true)
                            .open(root.join("context-crops"))
                            .unwrap(),
                        "{} {} {}",
                        pcm.len(),
                        pcm.iter().take_while(|v| **v == 0.).count(),
                        pcm.iter().rev().take_while(|v| **v == 0.).count()
                    )
                    .unwrap();
                    context["text"]
                        .as_str()
                        .expect("scripted bounded context")
                        .to_owned()
                } else {
                    std::fs::read_to_string(root.join("replay-transcript"))
                        .unwrap_or_else(|_| "Replay transcript".into())
                };
                Ok(Response::Text(AsrText {
                    text: text.clone(),
                    raw_text: text,
                    token_count: 2,
                }))
            }
            Request::CtcScores { pcm, token_ids }
                if script.as_ref().is_some_and(|script| {
                    script[0]["request_samples"].is_null()
                        || context_index
                            .checked_sub(1)
                            .and_then(|index| script.get(index))
                            .is_some_and(|context| {
                                context["end_sample"].as_i64().unwrap()
                                    - context["start_sample"].as_i64().unwrap()
                                    == pcm.len() as i64
                            })
                }) =>
            {
                // Explicit synthetic score replay: word onsets are authored independently
                // of the native context planner/assembler. No acoustic quality claim.
                let contexts: serde_json::Value = serde_json::from_slice(
                    &std::fs::read(root.join("context-replay.json")).unwrap(),
                )
                .unwrap();
                let context = &contexts[context_index - 1];
                let start = context["start_sample"].as_i64().unwrap();
                assert_eq!(
                    pcm.len() as i64,
                    context["end_sample"].as_i64().unwrap() - start
                );
                let vocabulary =
                    std::fs::read_to_string(root.join("coarse-alignment/tokens.txt")).unwrap();
                let lookup: std::collections::HashMap<String, i32> = vocabulary
                    .lines()
                    .map(|line| {
                        let (text, id) = line.rsplit_once(' ').unwrap();
                        (text.into(), id.parse().unwrap())
                    })
                    .collect();
                let frames = (pcm.len() - 400) / 320 + 1;
                let width = token_ids.len();
                let blank = token_ids.iter().position(|id| *id == 0).unwrap();
                let mut gathered = vec![-40.; frames * width];
                for row in gathered.chunks_mut(width) {
                    row[blank] = 0.;
                }
                let words = context["words"].as_array().unwrap();
                for (index, word) in words.iter().enumerate() {
                    let onset = word[1].as_i64().unwrap();
                    let mut text = word[0].as_str().unwrap().to_owned();
                    if index + 1 < words.len() {
                        text.push_str(context["separator"].as_str().unwrap_or(" "));
                    }
                    for (offset, c) in text.chars().enumerate() {
                        let sample = onset + offset as i64 * 640;
                        let frame = ((sample - start) / 320) as usize;
                        let id = lookup[&c.to_string()];
                        let column = token_ids.iter().position(|v| *v == id).unwrap();
                        gathered[frame * width..(frame + 1) * width].fill(-40.);
                        gathered[frame * width + column] = 0.;
                    }
                }
                Ok(Response::Ctc(speakerdesk_core::ipc::CtcScoresOut {
                    valid_frames: frames as u32,
                    audio_num_samples: pcm.len() as u32,
                    frame_hop_samples: 320,
                    frame_origin_samples: 0,
                    token_ids,
                    model: speakerdesk_core::timing::MODEL.into(),
                    vocabulary_sha256: speakerdesk_core::alignment::TOKENS_SHA256.into(),
                    gathered,
                }))
            }
            Request::EmbedVoice { pcm } if root.join("fixture-voice").exists() => {
                writeln!(
                    std::fs::OpenOptions::new()
                        .create(true)
                        .append(true)
                        .open(root.join("voice-calls"))
                        .unwrap(),
                    "{} {}",
                    pcm.len(),
                    pcm[0]
                )
                .unwrap();
                let mut vector = vec![0.; 192];
                vector[0] = 1.;
                Ok(Response::Embedding(speakerdesk_core::ipc::VoiceEmbedding {
                    model:speakerdesk_core::voice::VoiceModel::new("aufklarer/ReDimNet2-B6-CoreML","112dd8f4f836abdf8a420e66a5e2885cf8ec64ab:sd-pcm-v1:native-coreml-v1:CPU_AND_GPU","cf8cc300e240efaed56cb931bd7715ded41b66bf55af7e9a19d0b17e7fe986c9",192,16000).unwrap(),
                    clip:speakerdesk_core::voice::ClipEmbedding {vector,clean:true},
                }))
            }
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
        if hold_live || hold_eof || hold_asr || hold_voice {
            let context_hold = if hold_asr && context_asr {
                Some(context_index - 1)
            } else {
                None
            };
            if hold_asr {
                held_live_asr = true;
                held_context = context_hold;
            } else {
                held_live_feed = true;
            }
            let flag = Arc::new(AtomicBool::new(false));
            active = Some((id, flag.clone()));
            let writer = output.clone();
            let root = root.clone();
            std::thread::spawn(move || {
                let (marker, hold) = if hold_voice {
                    ("voice-started", "hold-live-voice")
                } else if hold_asr {
                    ("edit-asr-started", "hold-live-asr")
                } else if hold_eof {
                    ("recovery-eof-started", "hold-live-eof")
                } else {
                    ("recovery-feed-started", "hold-live-feed")
                };
                std::fs::write(root.join(marker), b"started").unwrap();
                if let Some(index) = context_hold {
                    std::fs::write(root.join(format!("context-started-{index}")), b"started")
                        .unwrap();
                }
                while root.join(hold).exists()
                    && !flag.load(Ordering::Acquire)
                    && context_hold
                        .is_none_or(|index| !root.join(format!("release-context-{index}")).exists())
                {
                    std::thread::sleep(Duration::from_millis(1));
                }
                if hold_asr
                    && flag.load(Ordering::Acquire)
                    && root.join("hold-cancel-terminal").exists()
                {
                    std::fs::write(
                        root.join("asr-cancelled"),
                        b"control acknowledged; terminal held",
                    )
                    .unwrap();
                    while root.join("hold-cancel-terminal").exists() {
                        std::thread::sleep(Duration::from_millis(1));
                    }
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
