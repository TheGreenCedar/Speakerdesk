#![cfg(target_os = "macos")]
use speakerdesk_models::sortformer::{
    frames_to_turns,
    network::{Network, OFFLINE_WINDOW_SAMPLES},
};
use std::{path::PathBuf, sync::atomic::AtomicBool, time::Instant};

#[test]
#[ignore = "requires SPEAKERDESK_TEST_NEMOTRON pinned checkpoint and Metal"]
fn first_offline_window_matches_pinned_graph_and_valid_tail() {
    speakerdesk_models::metallib::prepare(None).unwrap();
    let directory =
        PathBuf::from(std::env::var_os("SPEAKERDESK_TEST_NEMOTRON").expect("model directory"));
    let start = Instant::now();
    let mut network = Network::load(&directory).unwrap();
    let load = start.elapsed();
    let data = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/data");
    let mut reader = hound::WavReader::open(data.join("two-voice.wav")).unwrap();
    let pcm: Vec<f32> = reader
        .samples::<i16>()
        .map(|s| f32::from(s.unwrap()) / 32768.0)
        .collect();
    let reference: serde_json::Value =
        serde_json::from_str(include_str!("data/nemotron-reference.json")).unwrap();
    let expected: Vec<Vec<f32>> =
        serde_json::from_value(reference["probabilities"].clone()).unwrap();
    let expected_turns: Vec<speakerdesk_core::ipc::Turn> =
        serde_json::from_value(reference["turns"].clone()).unwrap();
    for run in 0..2 {
        let start = Instant::now();
        let probabilities = network
            .probabilities(&pcm, &AtomicBool::new(false))
            .unwrap();
        let elapsed = start.elapsed();
        assert_eq!(probabilities.len(), pcm.len() / 160);
        assert_eq!(probabilities.len(), expected.len());
        let error = probabilities
            .iter()
            .zip(&expected)
            .flat_map(|(a, b)| a.iter().zip(b))
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        eprintln!(
            "Nemotron run {run}: {elapsed:?}, load {load:?}, max probability error {error}, turns {:?}",
            frames_to_turns(&probabilities)
        );
        assert_eq!(frames_to_turns(&probabilities), expected_turns);
    }
    assert!(network.probabilities(&pcm, &AtomicBool::new(true)).is_err());
    for samples in [0, 159, 160, 15 * 160, 16 * 160, 16 * 160 + 159] {
        let probabilities = network
            .probabilities(&vec![0.0; samples], &AtomicBool::new(false))
            .unwrap();
        assert_eq!(probabilities.len(), samples / 160);
        assert!(frames_to_turns(&probabilities).is_empty());
    }
    assert!(
        network
            .probabilities(
                &vec![0.0; OFFLINE_WINDOW_SAMPLES + 1],
                &AtomicBool::new(false)
            )
            .is_err()
    );
    assert!(
        network
            .probabilities(&[f32::NAN; 160], &AtomicBool::new(false))
            .is_err()
    );
}
