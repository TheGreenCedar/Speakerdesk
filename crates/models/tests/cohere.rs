#![cfg(target_os = "macos")]
use speakerdesk_models::cohere::network::Network;
use std::{path::PathBuf, sync::atomic::AtomicBool, time::Instant};

#[test]
#[ignore = "requires SPEAKERDESK_TEST_COHERE pinned checkpoint and Metal"]
fn explicit_language_transcribes_real_audio_with_request_local_caches() {
    speakerdesk_models::metallib::prepare(None).unwrap();
    let directory =
        PathBuf::from(std::env::var_os("SPEAKERDESK_TEST_COHERE").expect("model directory"));
    let started = Instant::now();
    let mut network = Network::load(&directory).unwrap();
    eprintln!("Cohere load {:?}", started.elapsed());
    let mut reader = hound::WavReader::open(
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/data/two-voice.wav"),
    )
    .unwrap();
    let pcm: Vec<f32> = reader
        .samples::<i16>()
        .map(|s| f32::from(s.unwrap()) / 32768.0)
        .collect();
    let reference: serde_json::Value =
        serde_json::from_str(include_str!("data/cohere-reference.json")).unwrap();
    for run in 0..2 {
        let start = Instant::now();
        let result = network
            .transcribe(&pcm, "en", (0, 0), &AtomicBool::new(false))
            .unwrap();
        eprintln!("Cohere run {run}: {:?}, {:?}", start.elapsed(), result);
        assert_eq!(result.text, reference["text"].as_str().unwrap());
        assert_eq!(result.text, result.raw_text);
        assert_eq!(
            result.token_count as usize,
            reference["tokens"].as_array().unwrap().len()
        );
        assert!(
            network
                .transcribe(&pcm, "en", (0, 0), &AtomicBool::new(true))
                .is_err()
        );
        assert!(
            network
                .transcribe(&pcm, "auto", (0, 0), &AtomicBool::new(false))
                .is_err()
        );
    }
    assert!(
        network
            .transcribe(&[f32::NAN], "en", (0, 0), &AtomicBool::new(false))
            .is_err()
    );
    assert!(
        network
            .transcribe(&[], "en", (1, 0), &AtomicBool::new(false))
            .is_err()
    );
    assert!(
        network
            .transcribe(&[0.0; 159], "en", (0, 0), &AtomicBool::new(false))
            .unwrap()
            .text
            .is_empty()
    );
    assert!(
        network
            .transcribe(
                &vec![0.0; 35 * 16000 + 1],
                "en",
                (0, 0),
                &AtomicBool::new(false)
            )
            .is_err()
    );
}
