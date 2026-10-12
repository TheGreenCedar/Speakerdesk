#![cfg(target_os = "macos")]
use serde::Deserialize;
use speakerdesk_models::mossformer2::network::Network;
use std::{path::Path, sync::atomic::AtomicBool};

#[derive(Deserialize)]
struct Reference {
    checkpoint_sha256: String,
    cases: Vec<Case>,
}
#[derive(Deserialize)]
struct Case {
    name: String,
    samples: usize,
    input: String,
    output: [String; 2],
}

fn floats(path: &Path) -> Vec<f32> {
    let bytes = std::fs::read(path).unwrap();
    assert_eq!(bytes.len() % 4, 0);
    bytes
        .chunks_exact(4)
        .map(|bytes| f32::from_le_bytes(bytes.try_into().unwrap()))
        .collect()
}

#[test]
#[ignore = "requires converted pinned weights and independently generated PyTorch forward outputs"]
fn full_window_forward_matches_pinned_upstream_on_public_audio_and_boundaries() {
    let directory =
        std::env::var_os("SPEAKERDESK_MOSSFORMER_MODEL").expect("converted model directory");
    let reference =
        std::env::var_os("SPEAKERDESK_MOSSFORMER_REFERENCE").expect("reference directory");
    let reference = Path::new(&reference);
    let expected: Reference =
        serde_json::from_slice(&std::fs::read(reference.join("reference.json")).unwrap()).unwrap();
    assert_eq!(
        expected.checkpoint_sha256,
        speakerdesk_models::mossformer2::SOURCE_SHA256
    );
    assert!(!expected.cases.is_empty());
    let mut network = Network::load(Path::new(&directory)).unwrap();
    for case in expected.cases {
        let input = floats(&reference.join(&case.input));
        assert_eq!(input.len(), case.samples);
        let actual = network.separate(&input, &AtomicBool::new(false)).unwrap();
        for channel in 0..2 {
            let expected = floats(&reference.join(&case.output[channel]));
            assert_eq!(expected.len(), case.samples);
            assert_eq!(actual[channel].len(), case.samples);
            assert!(actual[channel].iter().all(|x| x.is_finite()));
            let max_error = actual[channel]
                .iter()
                .zip(&expected)
                .map(|(a, b)| (a - b).abs())
                .fold(0f32, f32::max);
            let error_energy: f64 = actual[channel]
                .iter()
                .zip(&expected)
                .map(|(a, b)| (*a as f64 - *b as f64).powi(2))
                .sum();
            let signal_energy: f64 = expected.iter().map(|x| (*x as f64).powi(2)).sum();
            let normalized_rms = (error_energy / signal_energy.max(1e-20)).sqrt();
            eprintln!(
                "{} channel{channel}: max_abs={max_error} normalized_rms={normalized_rms}",
                case.name
            );
            assert!(
                max_error <= 2e-3 && normalized_rms <= 2e-3,
                "{} channel{channel} differs from upstream",
                case.name
            );
        }
    }
    assert!(matches!(
        network.separate(&[0f32; 32000], &AtomicBool::new(true)),
        Err(speakerdesk_core::error::ChildError::Cancelled)
    ));
    for invalid in [
        vec![],
        vec![0f32; 15],
        vec![f32::NAN; 16],
        vec![0f32; 392001],
    ] {
        assert!(network.separate(&invalid, &AtomicBool::new(false)).is_err());
    }
}
