//! Native qualification only; host lifecycle/replay tests own publication and child recovery.
#![cfg(target_os = "macos")]

use serde::Deserialize;
use serde_json::{Value, json};
use speakerdesk_core::{
    audio::pcm16_bytes,
    error::ChildError,
    ipc::{AsrText, Handle, Request, Response},
    memo::sha256_hex,
};
use speakerdesk_models::dispatch::Runtime;
use speakerdesk_models::operation::Operation;
use std::{fs, io::Write, path::PathBuf, time::Instant};

#[derive(Deserialize)]
struct Manifest {
    cases: Vec<Case>,
}
#[derive(Deserialize)]
struct Case {
    id: String,
    path: PathBuf,
    wav_sha256: String,
    // Missing references are explicit new quality cases, never a skipped parity failure.
    expected: Option<[Wave; 2]>,
}
#[derive(Deserialize)]
struct Wave {
    path: PathBuf,
    wav_sha256: String,
}

fn wav(path: &std::path::Path, digest: Option<&str>) -> Vec<f32> {
    if let Some(digest) = digest {
        assert_eq!(sha256_hex(&fs::read(path).unwrap()), digest);
    }
    let mut reader = hound::WavReader::open(path).unwrap();
    let spec = reader.spec();
    assert_eq!(
        (spec.channels, spec.sample_rate, spec.bits_per_sample),
        (1, 16000, 16)
    );
    assert_eq!(spec.sample_format, hound::SampleFormat::Int);
    reader
        .samples::<i16>()
        .map(|s| f32::from(s.unwrap()) / 32768.)
        .collect()
}

fn call(runtime: &mut Runtime, request: Request, label: &str, phases: &mut Vec<Value>) -> Response {
    let started = Instant::now();
    let response = runtime.execute(request, &Operation::native().unwrap());
    let seconds = started.elapsed().as_secs_f64();
    eprintln!("native phase {label}: {seconds:.9}s");
    phases.push(json!({"phase": label, "seconds": seconds}));
    response.unwrap()
}
fn handle(response: Response) -> Handle {
    match response {
        Response::Handle(handle) => handle,
        other => panic!("expected handle: {other:?}"),
    }
}
fn transcribe(pcm: &[f32]) -> Request {
    Request::Transcribe {
        pcm: pcm.to_vec(),
        language: "en".into(),
        padding: (0, 0),
    }
}
fn open_transcribe(pcm: &[f32]) -> Request {
    Request::TranscribeOpen {
        pcm: pcm.to_vec(),
        language: "en".into(),
        padding: (0, 0),
    }
}
fn golden(text: &AsrText, reference: &Value) {
    assert_eq!(text.text, reference["text"].as_str().unwrap());
    assert_eq!(text.raw_text, text.text);
    assert_eq!(
        text.token_count as usize,
        reference["tokens"].as_array().unwrap().len()
    );
}
fn feed(
    runtime: &mut Runtime,
    vad: Handle,
    pcm: &[f32],
    observed: &mut u64,
    phases: &mut Vec<Value>,
) {
    let response = call(
        runtime,
        Request::VadFeed {
            handle: vad,
            pcm: pcm[..512].to_vec(),
            last: false,
        },
        "primary_vad",
        phases,
    );
    let Response::Vad(output) = response else {
        panic!("expected VAD")
    };
    *observed += 512;
    assert_eq!(output.observed, *observed);
    assert!(!output.frames.is_empty());
}

#[test]
#[ignore = "requires pinned native models, retained public PCM references and Metal; root owns execution"]
fn native_phases_preserve_public_outputs_and_allow_primary_work() {
    let qualification = Instant::now();
    let manifest_path =
        PathBuf::from(std::env::var_os("SPEAKERDESK_CONTINUATION_CASES").expect("public manifest"));
    let manifest_bytes = fs::read(&manifest_path).unwrap();
    let manifest: Manifest = serde_json::from_slice(&manifest_bytes).unwrap();
    assert!((3..=5).contains(&manifest.cases.len()));
    assert_eq!(
        manifest
            .cases
            .iter()
            .filter(|case| case.expected.is_some())
            .count(),
        3
    );
    let output = std::env::var_os("SPEAKERDESK_CONTINUATION_OUTPUT").map(PathBuf::from);
    if let Some(output) = &output {
        fs::create_dir(output).expect("fresh PCM output directory");
    }
    let mut result = fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(std::env::var_os("SPEAKERDESK_CONTINUATION_RESULT").expect("fresh result path"))
        .unwrap();
    let root =
        PathBuf::from(std::env::var_os("SPEAKERDESK_TEST_MODELS").expect("pinned model root"));
    let started = Instant::now();
    let mut runtime = Runtime::new(root.clone()).unwrap();
    let mut phases =
        vec![json!({"phase": "runtime_init", "seconds": started.elapsed().as_secs_f64()})];
    call(
        &mut runtime,
        Request::SetupCheck {
            models: vec![
                "silero-speech".into(),
                "cohere-speech".into(),
                "mossformer2-16k".into(),
            ],
        },
        "model_load",
        &mut phases,
    );
    let fixed = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/data/two-voice.wav");
    let reference: Value =
        serde_json::from_str(include_str!("data/cohere-reference.json")).unwrap();
    let primary = wav(&fixed, Some(reference["wav_sha256"].as_str().unwrap()));
    let vad = handle(call(
        &mut runtime,
        Request::VadOpen { conditioning: None },
        "primary_vad_open",
        &mut phases,
    ));
    let mut observed = 0;
    let mut cases = Vec::new();
    let mut first_pcm = None;
    for (index, case) in manifest.cases.into_iter().enumerate() {
        let pcm = wav(&case.path, Some(&case.wav_sha256));
        let samples = pcm.len();
        if first_pcm.is_none() {
            first_pcm = Some(pcm.clone());
        }
        let mut local = Vec::new();
        let whole = Instant::now();
        let owner = handle(call(
            &mut runtime,
            Request::SeparateOpen { pcm },
            "moss_open",
            &mut local,
        ));
        let mut steps = 0;
        let audio = loop {
            let response = call(
                &mut runtime,
                Request::SeparateStep { handle: owner },
                &format!("moss_step_{steps}"),
                &mut local,
            );
            steps += 1;
            let Response::SeparationProgress(audio) = response else {
                panic!("expected separation progress")
            };
            if let Some(audio) = audio {
                break audio;
            }
            if steps == 1 {
                feed(&mut runtime, vad, &primary, &mut observed, &mut local);
                if index == 0 {
                    let Response::Text(text) = call(
                        &mut runtime,
                        transcribe(&primary),
                        "primary_asr_during_moss",
                        &mut local,
                    ) else {
                        panic!("expected primary text")
                    };
                    golden(&text, &reference);
                }
            }
            assert!(steps < 64, "separation did not finish");
        };
        assert_eq!(steps, 49);
        audio.validate(samples).unwrap();
        call(
            &mut runtime,
            Request::SeparateClose { handle: owner },
            "moss_close",
            &mut local,
        );
        let elapsed = whole.elapsed().as_secs_f64();
        let audio = speakerdesk_core::separation::prepare_audio(audio, samples).unwrap();
        let mut channels = Vec::new();
        for (channel, samples) in audio.channels.iter().enumerate() {
            let bytes = pcm16_bytes(samples).unwrap();
            if let Some(expected) = &case.expected {
                let expected = wav(&expected[channel].path, Some(&expected[channel].wav_sha256));
                let expected = pcm16_bytes(&expected).unwrap();
                let mut differences = bytes.len().abs_diff(expected.len()) / 2;
                let mut max_abs = 0u16;
                for (actual, prior) in bytes.chunks_exact(2).zip(expected.chunks_exact(2)) {
                    let actual = i16::from_le_bytes(actual.try_into().unwrap());
                    let prior = i16::from_le_bytes(prior.try_into().unwrap());
                    differences += usize::from(actual != prior);
                    max_abs = max_abs.max(actual.abs_diff(prior));
                }
                eprintln!(
                    "{} channel{channel}: PCM16 differences={differences}, max_abs_lsb={max_abs}",
                    case.id
                );
                assert!(
                    bytes == expected,
                    "{} channel{channel}: retained PCM16 changed ({differences} samples, max {max_abs} LSB)",
                    case.id
                );
            }
            if let Some(output) = &output {
                let path = output.join(format!("case-{index}-channel-{channel}.wav"));
                let mut writer = hound::WavWriter::create(
                    &path,
                    hound::WavSpec {
                        channels: 1,
                        sample_rate: 16000,
                        bits_per_sample: 16,
                        sample_format: hound::SampleFormat::Int,
                    },
                )
                .unwrap();
                for sample in bytes.chunks_exact(2) {
                    writer
                        .write_sample(i16::from_le_bytes(sample.try_into().unwrap()))
                        .unwrap();
                }
                writer.finalize().unwrap();
                channels.push(json!({"channel":channel,"path":path,"wav_sha256":sha256_hex(&fs::read(&path).unwrap()),"pcm_sha256":sha256_hex(&bytes),"samples":samples.len()}));
            } else {
                channels.push(json!({"channel":channel,"pcm_sha256":sha256_hex(&bytes),"samples":samples.len()}));
            }
        }
        cases.push(json!({"id":case.id,"source":case.path,"source_sha256":case.wav_sha256,"scope":if case.expected.is_some(){"retained_pcm16_parity"}else{"new_quality_unlabelled_channels"},"elapsed_seconds":elapsed,"phases":local,"channels":channels}));
    }
    let whole = Instant::now();
    let aux = handle(call(
        &mut runtime,
        open_transcribe(&primary),
        "cohere_open_encoder",
        &mut phases,
    ));
    let mut steps = 0;
    let text = loop {
        let Response::TranscriptionProgress(text) = call(
            &mut runtime,
            Request::TranscribeStep { handle: aux },
            &format!("cohere_step_{steps}"),
            &mut phases,
        ) else {
            panic!("expected transcription progress")
        };
        steps += 1;
        if let Some(text) = text {
            break text;
        }
        if steps == 1 {
            feed(&mut runtime, vad, &primary, &mut observed, &mut phases);
            let Response::Text(text) = call(
                &mut runtime,
                transcribe(&primary),
                "primary_asr_during_cohere",
                &mut phases,
            ) else {
                panic!("expected primary text")
            };
            golden(&text, &reference);
        }
        assert!(steps < 450, "transcription did not finish");
    };
    let cohere_seconds = whole.elapsed().as_secs_f64();
    golden(&text, &reference);
    call(
        &mut runtime,
        Request::TranscribeClose { handle: aux },
        "cohere_close",
        &mut phases,
    );

    // Mirror the existing serve/supervisor cleanup contract after cancellation.
    // Runtime's early cancellation guard intentionally does not close the owned session.
    for separation in [true, false] {
        let request = if separation {
            Request::SeparateOpen {
                pcm: first_pcm.as_ref().unwrap().clone(),
            }
        } else {
            open_transcribe(&primary)
        };
        let owner = handle(call(&mut runtime, request, "cancel_open", &mut phases));
        let step = if separation {
            Request::SeparateStep { handle: owner }
        } else {
            Request::TranscribeStep { handle: owner }
        };
        assert_eq!(
            runtime.execute(step.clone(), &{
                let operation = Operation::native().unwrap();
                operation.cancel();
                operation
            }),
            Err(ChildError::Cancelled)
        );
        let close = if separation {
            Request::SeparateClose { handle: owner }
        } else {
            Request::TranscribeClose { handle: owner }
        };
        assert_eq!(
            call(&mut runtime, close.clone(), "cancel_close", &mut phases),
            Response::Ack
        );
        assert_eq!(
            call(&mut runtime, close, "cancel_close_again", &mut phases),
            Response::Ack
        );
        assert!(
            runtime
                .execute(step, &Operation::native().unwrap())
                .is_err()
        );
        let request = if separation {
            Request::SeparateOpen { pcm: vec![0.; 16] }
        } else {
            open_transcribe(&[])
        };
        let fresh = handle(call(
            &mut runtime,
            request,
            "reopen_after_cancel",
            &mut phases,
        ));
        assert_ne!(fresh, owner);
        let close = if separation {
            Request::SeparateClose { handle: fresh }
        } else {
            Request::TranscribeClose { handle: fresh }
        };
        call(&mut runtime, close, "reopened_close", &mut phases);
        feed(&mut runtime, vad, &primary, &mut observed, &mut phases);
    }
    call(
        &mut runtime,
        Request::VadClose { handle: vad },
        "primary_vad_close",
        &mut phases,
    );
    let all_phases: Vec<_> = phases
        .iter()
        .chain(
            cases
                .iter()
                .flat_map(|case| case["phases"].as_array().unwrap()),
        )
        .collect();
    let by_time = |a: &&Value, b: &&Value| {
        a["seconds"]
            .as_f64()
            .unwrap()
            .total_cmp(&b["seconds"].as_f64().unwrap())
    };
    let max_phase = all_phases.iter().copied().max_by(by_time).unwrap();
    let max_auxiliary = all_phases
        .iter()
        .copied()
        .filter(|p| {
            p["phase"]
                .as_str()
                .is_some_and(|s| s.starts_with("moss_") || s.starts_with("cohere_"))
        })
        .max_by(by_time)
        .unwrap();
    serde_json::to_writer_pretty(&mut result, &json!({"manifest":manifest_path,"manifest_sha256":sha256_hex(&manifest_bytes),"model_root":root,"cases":cases,"phases":phases,"cohere":text,"cohere_elapsed_seconds":cohere_seconds,"qualification_elapsed_seconds":qualification.elapsed().as_secs_f64(),"maximum_timed_call":max_phase,"maximum_auxiliary_call":max_auxiliary,"limitations":"Direct native Runtime, serial interleaving; no capture/IPC latency, automatic cancellation race, acoustic identity labels or concurrent execution claim. Cold model load is separate; lazy Moss frontend work is charged to its first evaluated step."})).unwrap();
    writeln!(result).unwrap();
}
