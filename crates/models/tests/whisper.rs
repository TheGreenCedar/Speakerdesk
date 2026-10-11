#![cfg(target_os = "macos")]
use speakerdesk_models::whisper_lid::network::Network;
use std::{path::PathBuf, sync::atomic::AtomicBool, time::Instant};

#[test]
#[ignore = "requires SPEAKERDESK_TEST_WHISPER pinned checkpoint and Metal"]
fn independent_language_probes_match_all_reference_scores() {
    speakerdesk_models::metallib::prepare(None).unwrap();
    let directory =
        PathBuf::from(std::env::var_os("SPEAKERDESK_TEST_WHISPER").expect("model directory"));
    let mut network = Network::load(&directory).unwrap();
    let mut reader = hound::WavReader::open(
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/data/two-voice.wav"),
    )
    .unwrap();
    let pcm: Vec<f32> = reader
        .samples::<i16>()
        .map(|s| f32::from(s.unwrap()) / 32768.)
        .collect();
    let reference: serde_json::Value =
        serde_json::from_str(include_str!("data/whisper-reference.json")).unwrap();
    let cases = [
        &pcm[..],
        &pcm[..48000],
        &pcm[16000..16549],
        &[0f32; 512][..],
    ];
    for _ in 0..2 {
        for (audio, expected) in cases.iter().zip(reference["cases"].as_array().unwrap()) {
            let started = Instant::now();
            let result = network.probe(audio, &AtomicBool::new(false)).unwrap();
            assert_eq!(result.codes.len(), 99);
            assert_eq!(result.probabilities.len(), 99);
            let differences: Vec<_> = result
                .codes
                .iter()
                .zip(&result.probabilities)
                .map(|(code, p)| {
                    (f64::from(*p) - expected["probabilities"][code].as_f64().unwrap()).abs()
                })
                .collect();
            let maximum = differences.into_iter().fold(0.0, f64::max);
            eprintln!(
                "Whisper {} {:?}, max score difference {maximum}",
                expected["name"],
                started.elapsed()
            );
            assert!(maximum < 1e-4, "score drift {maximum}");
            assert!((result.probabilities.iter().sum::<f32>() - 1.).abs() < 1e-5);
            assert_eq!(
                network.probe(audio, &AtomicBool::new(true)),
                Err(speakerdesk_core::error::ChildError::Cancelled)
            );
        }
    }
    assert!(network.probe(&[f32::NAN], &AtomicBool::new(false)).is_err());
    assert!(network.probe(&[], &AtomicBool::new(false)).is_err());
}
