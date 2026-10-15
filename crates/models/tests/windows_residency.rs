#![cfg(windows)]

use speakerdesk_models::{
    cohere, mossformer2, operation::Operation, redimnet, silero, sortformer,
    windows_runtime::Backend,
};
use std::{fs, path::PathBuf, time::Instant};

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

#[test]
#[ignore = "requires five pinned Windows models, independent Moss references and a DirectML device"]
fn resident_primary_models_preserve_transcription_across_maximum_separation() {
    let root = directory("SPEAKERDESK_TEST_MODELS");
    let references = directory("SPEAKERDESK_TEST_MOSS_REFERENCES");
    let backend = Backend::from_directory(&directory("SPEAKERDESK_TEST_WINDOWS_RUNTIME")).unwrap();
    let started = Instant::now();
    let mut asr = cohere::network::Network::load(&root.join("cohere-speech"), &backend).unwrap();
    let mut vad = silero::network::Network::load(&root.join("silero-speech"), &backend).unwrap();
    let mut diar = sortformer::network::Network::load(&root.join("nemotron"), &backend).unwrap();
    let mut voice =
        redimnet::network_windows::Network::load(&root.join("redimnet2-b6"), &backend).unwrap();
    let mut separation =
        mossformer2::network::Network::load(&root.join("mossformer2-16k"), &backend).unwrap();
    eprintln!(
        "Five resident Windows models loaded: {:?}",
        started.elapsed()
    );

    let pins: serde_json::Value =
        serde_json::from_str(include_str!("../src/windows-model-artifacts.json")).unwrap();
    for model in pins.as_array().unwrap() {
        for name in model["files"].as_object().unwrap().keys() {
            let path = root.join(model["id"].as_str().unwrap()).join(name);
            assert!(
                fs::OpenOptions::new().write(true).open(&path).is_err(),
                "unlocked admitted artifact: {}",
                path.display()
            );
        }
    }

    let mut wav =
        hound::WavReader::new(std::io::Cursor::new(include_bytes!("data/two-voice.wav"))).unwrap();
    let pcm: Vec<_> = wav
        .samples::<i16>()
        .map(|sample| f32::from(sample.unwrap()) / 32768.)
        .collect();
    let expected: serde_json::Value =
        serde_json::from_str(include_str!("data/cohere-reference.json")).unwrap();
    let operation = Operation::native().unwrap();
    let mut transcription = asr.begin(&pcm, "en", (0, 0), &operation).unwrap();
    assert!(asr.step(&mut transcription, &operation).unwrap().is_none());

    // Keep primary weights and decoder KV state resident through auxiliary work.
    let vad_result = silero::session::Session::new(None)
        .unwrap()
        .feed(&mut vad, &pcm, true, &operation)
        .unwrap();
    assert_eq!(vad_result.received, pcm.len() as u64);
    assert!(!vad_result.frames.is_empty());
    let diar_result = diar.probabilities(&pcm, &operation).unwrap();
    assert_eq!(diar_result.len(), pcm.len() / 160);
    assert!(diar_result.iter().flatten().all(|value| value.is_finite()));
    let embedding = voice.embed(&pcm, &operation).unwrap();
    assert!(embedding.clean);
    assert_eq!(embedding.vector.len(), 192);

    let manifest = fs::read(references.join("references.json")).unwrap();
    assert_eq!(
        speakerdesk_core::memo::sha256_hex(&manifest),
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
        assert_eq!(speakerdesk_core::memo::sha256_hex(&bytes), record["sha256"]);
        assert_eq!(bytes.len(), 392000 * 4);
        bytes
            .chunks_exact(4)
            .map(|bytes| f32::from_le_bytes(bytes.try_into().unwrap()))
            .collect::<Vec<_>>()
    };
    let mixture = read(&case["input"]);
    let mut separated = separation.begin(&mixture, &operation).unwrap();
    for phase in 0..48 {
        eprintln!("Resident separation phase {phase}");
        assert!(
            separation
                .step(&mut separated, &operation)
                .unwrap()
                .is_none()
        );
        if phase == 0 {
            assert!(asr.step(&mut transcription, &operation).unwrap().is_none());
        }
    }
    let channels = separation
        .step(&mut separated, &operation)
        .unwrap()
        .unwrap();
    for (index, actual) in channels.iter().enumerate() {
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
    let resumed = loop {
        if let Some(result) = asr.step(&mut transcription, &operation).unwrap() {
            break result;
        }
    };
    assert_eq!(resumed.text, expected["text"].as_str().unwrap());
    assert_eq!(resumed.raw_text, resumed.text);
    assert_eq!(
        resumed.token_count as usize,
        expected["tokens"].as_array().unwrap().len()
    );
    let subsequent = asr.transcribe(&pcm, "en", (0, 0), &operation).unwrap();
    assert_eq!(subsequent, resumed);
    eprintln!("Resident continuation completed: {:?}", started.elapsed());
}
