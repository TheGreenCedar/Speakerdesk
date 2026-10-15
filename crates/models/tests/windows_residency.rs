#![cfg(windows)]

use speakerdesk_core::{
    error::ChildError,
    ipc::{DiarPreset, Handle, Request, Response},
    memo::sha256_hex,
};
use speakerdesk_models::{
    cohere, dispatch::Runtime, operation::Operation, redimnet, silero, sortformer, whisper_lid,
    windows_runtime::Backend,
};
use std::{fs, io::Read, path::PathBuf, time::Instant};

fn directory(name: &str) -> PathBuf {
    std::env::var_os(name).map(PathBuf::from).expect(name)
}

#[test]
#[ignore = "requires pinned Windows Silero and private runtime; opens but never modifies the graph"]
fn loaded_model_retains_its_artifact_admission() {
    let model = directory("SPEAKERDESK_TEST_SILERO");
    let backend = Backend::from_directory(&directory("SPEAKERDESK_TEST_WINDOWS_RUNTIME")).unwrap();
    let network = silero::network::Network::load(&model, &backend).unwrap();
    let writer = fs::OpenOptions::new()
        .write(true)
        .open(model.join("silero.onnx"));
    assert!(
        writer.is_err(),
        "loaded model released its verification handle"
    );
    drop(network);
    assert!(
        fs::OpenOptions::new()
            .write(true)
            .open(model.join("silero.onnx"))
            .is_ok()
    );
}

fn call(runtime: &mut Runtime, request: Request) -> Response {
    runtime
        .execute(request, &Operation::native().unwrap())
        .unwrap()
}

fn handle(response: Response) -> Handle {
    let Response::Handle(handle) = response else {
        panic!("expected model handle: {response:?}")
    };
    assert_eq!(handle.generation, 0);
    handle
}

fn transcribe(pcm: &[f32]) -> Request {
    Request::Transcribe {
        pcm: pcm.to_vec(),
        language: "en".into(),
        padding: (0, 0),
    }
}

#[test]
#[ignore = "requires seven admitted Windows models, independent Moss references and adjacent private runtime"]
fn resident_primary_models_preserve_transcription_across_maximum_separation() {
    let root = directory("SPEAKERDESK_TEST_MODELS");
    let references = directory("SPEAKERDESK_TEST_MOSS_REFERENCES");
    let started = Instant::now();
    let mut runtime = Runtime::new(root.clone()).unwrap();
    let Response::Ready { device, models, .. } = call(
        &mut runtime,
        Request::Hello {
            build_id: "residency".into(),
        },
    ) else {
        panic!("expected readiness")
    };
    assert_eq!(device, "directml");
    assert!(models.is_empty());
    let staging = root.join(".staging");
    assert!(
        !staging.exists(),
        "residency model tree must start without staging"
    );
    let cancelled = Operation::native().unwrap();
    cancelled.cancel();
    assert_eq!(
        runtime.execute(Request::PrepareAlignment, &cancelled),
        Err(ChildError::Cancelled)
    );
    assert_eq!(
        runtime.execute(Request::PrepareSeparation, &Operation::native().unwrap()),
        Err(ChildError::failed(
            "Windows separation requires downloaded prebuilt artifacts",
            false
        ))
    );
    assert!(
        !staging.exists(),
        "cancelled/refused preparation created staging"
    );
    let ids = [
        "silero-speech",
        "nemotron",
        "cohere-speech",
        "whisper-language",
        "coarse-alignment",
        "redimnet2-b6",
        "mossformer2-16k",
    ];
    let identities: Vec<String> = vec![
        silero::WEIGHT_PIN.into(),
        sortformer::WEIGHT_PIN.into(),
        cohere::WEIGHT_PIN.into(),
        whisper_lid::WEIGHT_PIN.into(),
        speakerdesk_core::timing_policy::WINDOWS.checkpoint.into(),
        redimnet::network_windows::REVISION.into(),
        speakerdesk_core::separation::MODEL.into(),
    ];
    let Response::Setup(reports) = call(
        &mut runtime,
        Request::SetupCheck {
            models: ids.iter().map(|id| (*id).into()).collect(),
        },
    ) else {
        panic!("expected setup reports")
    };
    assert_eq!(reports.len(), 7);
    assert!(reports.iter().all(|report| report.ok));
    assert_eq!(
        reports
            .iter()
            .map(|report| report.model.clone())
            .collect::<Vec<_>>(),
        identities
    );
    let Response::Ready { device, models, .. } = call(
        &mut runtime,
        Request::Hello {
            build_id: "residency".into(),
        },
    ) else {
        panic!("expected readiness")
    };
    assert_eq!(device, "directml");
    assert_eq!(models, identities);
    eprintln!(
        "Seven resident Windows models loaded through Runtime: {:?}",
        started.elapsed()
    );

    let pins: serde_json::Value =
        serde_json::from_str(include_str!("../src/windows-model-artifacts.json")).unwrap();
    for id in ids {
        let model = pins
            .as_array()
            .unwrap()
            .iter()
            .find(|model| model["id"] == id)
            .expect("admitted model inventory");
        for (name, pin) in model["files"]
            .as_object()
            .unwrap()
            .iter()
            .chain(model["prepared"].as_object().into_iter().flatten())
        {
            let path = root.join(id).join(name);
            let mut reader = fs::File::open(&path)
                .unwrap_or_else(|error| panic!("unreadable artifact {}: {error}", path.display()));
            assert_eq!(
                reader.metadata().unwrap().len(),
                pin["size"].as_u64().unwrap()
            );
            reader.read_exact(&mut [0_u8; 1]).unwrap();
            drop(reader);
            assert!(
                fs::OpenOptions::new().write(true).open(&path).is_err(),
                "unlocked admitted artifact: {}",
                path.display()
            );
        }
    }

    let wav = include_bytes!("data/two-voice.wav");
    let expected: serde_json::Value =
        serde_json::from_str(include_str!("data/cohere-reference.json")).unwrap();
    let whisper_reference: serde_json::Value =
        serde_json::from_str(include_str!("data/whisper-reference.json")).unwrap();
    let ctc_reference: serde_json::Value =
        serde_json::from_str(include_str!("data/ctc-reference.json")).unwrap();
    assert_eq!(sha256_hex(wav), whisper_reference["wav_sha256"]);
    assert_eq!(sha256_hex(wav), ctc_reference["wav_sha256"]);
    let mut wav = hound::WavReader::new(std::io::Cursor::new(wav)).unwrap();
    assert_eq!(
        (
            wav.spec().channels,
            wav.spec().sample_rate,
            wav.spec().bits_per_sample
        ),
        (1, 16000, 16)
    );
    let pcm: Vec<_> = wav
        .samples::<i16>()
        .map(|sample| f32::from(sample.unwrap()) / 32768.)
        .collect();
    assert_eq!(
        pcm.len(),
        whisper_reference["cases"][0]["samples"].as_u64().unwrap() as usize
    );
    let transcription = handle(call(
        &mut runtime,
        Request::TranscribeOpen {
            pcm: pcm.clone(),
            language: "en".into(),
            padding: (0, 0),
        },
    ));
    assert_eq!(
        call(
            &mut runtime,
            Request::TranscribeStep {
                handle: transcription
            }
        ),
        Response::TranscriptionProgress(None)
    );
    let cancelled = Operation::native().unwrap();
    cancelled.cancel();
    assert_eq!(
        runtime.execute(
            Request::TranscribeStep {
                handle: transcription
            },
            &cancelled
        ),
        Err(ChildError::Cancelled)
    );
    // This proves pre-cancel isolation; it does not exercise interruption inside ORT.
    let wrong = Handle {
        session: transcription.session + 1,
        ..transcription
    };
    assert_eq!(
        runtime.execute(
            Request::TranscribeStep { handle: wrong },
            &Operation::native().unwrap()
        ),
        Err(ChildError::failed("unknown transcription session", false))
    );

    // Keep primary decoder KV tensors resident through all six auxiliary models.
    let vad = handle(call(&mut runtime, Request::VadOpen { conditioning: None }));
    let Response::Vad(vad_result) = call(
        &mut runtime,
        Request::VadFeed {
            handle: vad,
            pcm: pcm.clone(),
            last: true,
        },
    ) else {
        panic!("expected VAD")
    };
    assert_eq!(vad_result.received, pcm.len() as u64);
    assert_eq!(vad_result.observed, pcm.len() as u64);
    assert!(!vad_result.frames.is_empty());
    assert_eq!(
        call(&mut runtime, Request::VadClose { handle: vad }),
        Response::Ack
    );
    assert_eq!(
        runtime.execute(
            Request::VadFeed {
                handle: vad,
                pcm: vec![],
                last: true
            },
            &Operation::native().unwrap()
        ),
        Err(ChildError::failed("unknown Silero session", false))
    );
    let diar = handle(call(
        &mut runtime,
        Request::DiarOpen {
            preset: DiarPreset::Offline,
        },
    ));
    let Response::Diar(diar_result) = call(
        &mut runtime,
        Request::DiarFeed {
            handle: diar,
            pcm: pcm.clone(),
            last: true,
        },
    ) else {
        panic!("expected diarization")
    };
    diar_result.validate(pcm.len() as u64, 0).unwrap();
    assert_eq!(diar_result.observed, (pcm.len() / 160 * 160) as u64);
    assert!(!diar_result.turns.is_empty());
    assert_eq!(
        call(&mut runtime, Request::DiarClose { handle: diar }),
        Response::Ack
    );
    assert_eq!(
        runtime.execute(
            Request::DiarFeed {
                handle: diar,
                pcm: vec![],
                last: true
            },
            &Operation::native().unwrap()
        ),
        Err(ChildError::failed("unknown Nemotron session", false))
    );
    let Response::Embedding(embedding) =
        call(&mut runtime, Request::EmbedVoice { pcm: pcm.clone() })
    else {
        panic!("expected embedding")
    };
    assert_eq!(embedding.model, redimnet::network_windows::identity());
    assert!(embedding.clip.clean);
    assert_eq!(embedding.clip.vector.len(), 192);
    assert!(embedding.clip.vector.iter().all(|value| value.is_finite()));
    let Response::Probe(probe) = call(&mut runtime, Request::LanguageProbe { pcm: pcm.clone() })
    else {
        panic!("expected language probe")
    };
    assert_eq!(probe.codes.len(), 99);
    assert_eq!(probe.probabilities.len(), 99);
    assert!(
        probe
            .probabilities
            .iter()
            .all(|value| value.is_finite() && (0. ..=1.).contains(value))
    );
    let probabilities: Vec<_> = probe
        .codes
        .into_iter()
        .zip(probe.probabilities)
        .map(|(code, score)| (code, f64::from(score)))
        .collect();
    let decision = speakerdesk_core::language::LanguagePolicy::new()
        .decide(&probabilities, &[])
        .unwrap();
    assert_eq!(decision.language.as_deref(), Some("en"));
    assert_eq!(
        decision.detection.reason,
        speakerdesk_core::language::DetectionReason::Detected
    );
    let token_ids: Vec<i32> = serde_json::from_value(ctc_reference["token_ids"].clone()).unwrap();
    let Response::Ctc(scores) = call(
        &mut runtime,
        Request::CtcScores {
            pcm: pcm.clone(),
            token_ids: token_ids.clone(),
        },
    ) else {
        panic!("expected CTC")
    };
    assert_eq!(
        scores.model,
        speakerdesk_core::timing_policy::WINDOWS.model_sha256
    );
    assert_eq!(
        scores.vocabulary_sha256,
        speakerdesk_core::alignment::TOKENS_SHA256
    );
    assert_eq!(scores.audio_num_samples as usize, pcm.len());
    assert_eq!(scores.valid_frames as usize, (pcm.len() - 400) / 320 + 1);
    assert_eq!(
        (scores.frame_hop_samples, scores.frame_origin_samples),
        (320, 0)
    );
    assert_eq!(scores.token_ids, token_ids);
    assert_eq!(
        scores.gathered.len(),
        scores.valid_frames as usize * scores.token_ids.len()
    );
    assert!(
        scores
            .gathered
            .iter()
            .all(|value| value.is_finite() && *value <= 0.)
    );
    assert!(
        scores
            .gathered
            .chunks_exact(scores.token_ids.len())
            .all(
                |row| row.iter().map(|&value| f64::from(value).exp()).sum::<f64>()
                    <= 0.001_f64.exp()
            )
    );

    let manifest = fs::read(references.join("references.json")).unwrap();
    assert_eq!(
        sha256_hex(&manifest),
        "d3a60ce3f6bfd11dc59efe9dca384ad2150b3fceaff852aadbc4c6d41795a1bd"
    );
    let manifest: serde_json::Value = serde_json::from_slice(&manifest).unwrap();
    let case = manifest["cases"]
        .as_array()
        .unwrap()
        .iter()
        .find(|case| case["samples"] == 392000)
        .expect("maximum independent window");
    let read = |record: &serde_json::Value| {
        let bytes = fs::read(references.join(record["path"].as_str().unwrap())).unwrap();
        assert_eq!(sha256_hex(&bytes), record["sha256"]);
        assert_eq!(bytes.len(), 392000 * 4);
        bytes
            .chunks_exact(4)
            .map(|bytes| f32::from_le_bytes(bytes.try_into().unwrap()))
            .collect::<Vec<_>>()
    };
    let mixture = read(&case["input"]);
    let separation = handle(call(&mut runtime, Request::SeparateOpen { pcm: mixture }));
    for phase in 0..48 {
        eprintln!("Resident separation phase {phase}");
        assert_eq!(
            call(&mut runtime, Request::SeparateStep { handle: separation }),
            Response::SeparationProgress(None)
        );
        if phase == 0 {
            assert_eq!(
                call(
                    &mut runtime,
                    Request::TranscribeStep {
                        handle: transcription
                    }
                ),
                Response::TranscriptionProgress(None)
            );
        }
    }
    let Response::SeparationProgress(Some(separated)) =
        call(&mut runtime, Request::SeparateStep { handle: separation })
    else {
        panic!("expected final separation")
    };
    separated.validate(392000).unwrap();
    for (index, actual) in separated.channels.iter().enumerate() {
        let reference = read(&case["outputs"][index]);
        assert_eq!(actual.len(), reference.len());
        assert!(actual.iter().all(|value| value.is_finite()));
        let mut maximum = 0_f64;
        let mut difference_energy = 0_f64;
        let mut reference_energy = 0_f64;
        for (&value, &expected) in actual.iter().zip(&reference) {
            let difference = f64::from(value) - f64::from(expected);
            maximum = maximum.max(difference.abs());
            difference_energy += difference * difference;
            reference_energy += f64::from(expected).powi(2);
        }
        let normalized_rms = (difference_energy / reference_energy.max(1e-20)).sqrt();
        eprintln!("Resident separation channel {index}: max {maximum}, NRMS {normalized_rms}");
        assert!(maximum <= 2e-3 && normalized_rms <= 2e-3);
    }
    assert_eq!(
        call(&mut runtime, Request::SeparateClose { handle: separation }),
        Response::Ack
    );
    assert_eq!(
        runtime.execute(
            Request::SeparateStep { handle: separation },
            &Operation::native().unwrap()
        ),
        Err(ChildError::failed("unknown separation session", false))
    );
    let mut resumed = None;
    // The native decoder caps output at 448 tokens, then returns its final text.
    for _ in 0..449 {
        match call(
            &mut runtime,
            Request::TranscribeStep {
                handle: transcription,
            },
        ) {
            Response::TranscriptionProgress(Some(result)) => {
                resumed = Some(result);
                break;
            }
            Response::TranscriptionProgress(None) => {}
            other => panic!("expected transcription progress: {other:?}"),
        }
    }
    let resumed = resumed.expect("resident decoder exceeded its bounded continuation");
    assert_eq!(resumed.text, expected["text"].as_str().unwrap());
    assert_eq!(resumed.raw_text, resumed.text);
    assert_eq!(
        resumed.token_count as usize,
        expected["tokens"].as_array().unwrap().len()
    );
    assert_eq!(
        runtime.execute(
            Request::TranscribeStep {
                handle: transcription
            },
            &Operation::native().unwrap()
        ),
        Err(ChildError::failed("unknown transcription session", false))
    );
    let closed = handle(call(
        &mut runtime,
        Request::TranscribeOpen {
            pcm: pcm[..1].to_vec(),
            language: "en".into(),
            padding: (0, 0),
        },
    ));
    assert_eq!(
        call(&mut runtime, Request::TranscribeClose { handle: closed }),
        Response::Ack
    );
    assert_eq!(
        runtime.execute(
            Request::TranscribeStep { handle: closed },
            &Operation::native().unwrap()
        ),
        Err(ChildError::failed("unknown transcription session", false))
    );
    let Response::Text(subsequent) = call(&mut runtime, transcribe(&pcm)) else {
        panic!("expected transcription")
    };
    assert_eq!(subsequent, resumed);
    eprintln!(
        "Seven-model Runtime continuation completed: {:?}",
        started.elapsed()
    );
    use windows_sys::Win32::System::{
        ProcessStatus::{GetProcessMemoryInfo, PROCESS_MEMORY_COUNTERS_EX},
        Threading::GetCurrentProcess,
    };
    let mut memory: PROCESS_MEMORY_COUNTERS_EX = unsafe { std::mem::zeroed() };
    let bytes = std::mem::size_of_val(&memory) as u32;
    memory.cb = bytes;
    assert_ne!(
        unsafe {
            GetProcessMemoryInfo(
                GetCurrentProcess(),
                (&mut memory as *mut PROCESS_MEMORY_COUNTERS_EX).cast(),
                bytes,
            )
        },
        0,
        "GetProcessMemoryInfo: {}",
        std::io::Error::last_os_error()
    );
    assert!(memory.PrivateUsage > 0 && memory.PeakPagefileUsage >= memory.PrivateUsage);
    eprintln!(
        "WINDOWS_RESIDENCY_MEMORY {}",
        serde_json::json!({
            "pid":std::process::id(), "current_bytes":memory.PrivateUsage,
            "lifetime_peak_bytes":memory.PeakPagefileUsage,
        })
    );
}
