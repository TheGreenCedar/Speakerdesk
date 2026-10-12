//! Acoustic comparison against the pinned upstream Python graph, not a mock.
#![cfg(target_os = "macos")]

use speakerdesk_models::silero::{
    FRAME_SAMPLES,
    network::{Network, State},
    peak_gain,
};
use std::{path::PathBuf, time::Instant};

#[test]
#[ignore = "requires SPEAKERDESK_TEST_SILERO pointing to the pinned model directory and Metal"]
fn pinned_graph_matches_independent_reference() {
    speakerdesk_models::metallib::prepare(None).unwrap();
    let directory =
        PathBuf::from(std::env::var_os("SPEAKERDESK_TEST_SILERO").expect("model directory"));
    let started = Instant::now();
    let mut network = Network::load(&directory).unwrap();
    let load = started.elapsed();
    let data = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/data");
    let mut reader = hound::WavReader::open(data.join("two-voice.wav")).unwrap();
    let pcm: Vec<f32> = reader
        .samples::<i16>()
        .map(|x| f32::from(x.unwrap()) / 32768.0)
        .collect();
    let reference: serde_json::Value =
        serde_json::from_str(include_str!("data/silero-reference.json")).unwrap();
    let same_backend: serde_json::Value =
        serde_json::from_str(include_str!("data/silero-reference-macos15.json")).unwrap();
    assert_eq!(pcm.len() as u64, reference["samples"].as_u64().unwrap());
    assert_eq!(
        pcm.len().div_ceil(FRAME_SAMPLES),
        reference["frames"].as_array().unwrap().len()
    );
    assert_eq!(
        reference["wav_sha256"],
        speakerdesk_core::memo::sha256_hex(&std::fs::read(data.join("two-voice.wav")).unwrap())
    );
    let mut max_error = 0.0f32;
    for run in 0..2 {
        let started = Instant::now();
        let mut raw_state = State::default();
        let mut norm_state = State::default();
        let mut actual_latch = false;
        let mut expected_latch = false;
        let mut decision_changes = 0;
        for ((chunk, expected), equal_backend) in pcm
            .chunks(FRAME_SAMPLES)
            .zip(reference["frames"].as_array().unwrap())
            .zip(same_backend["frames"].as_array().unwrap())
        {
            let mut frame = [0.0; FRAME_SAMPLES];
            frame[..chunk.len()].copy_from_slice(chunk);
            let gain = peak_gain(chunk.iter().map(|s| s.abs()).fold(0.0, f32::max));
            let raw_p = network.step(&frame, &mut raw_state).unwrap();
            for sample in &mut frame {
                *sample *= gain;
            }
            let norm_p = network.step(&frame, &mut norm_state).unwrap();
            for (key, actual) in [("raw_p", raw_p), ("norm_p", norm_p)] {
                let error = (actual - expected[key].as_f64().unwrap() as f32).abs();
                max_error = max_error.max(error);
                assert!((actual - equal_backend[key].as_f64().unwrap() as f32).abs() < 1e-4);
            }
            actual_latch =
                speakerdesk_models::silero::hysteresis_flag(actual_latch, raw_p.max(norm_p), false)
                    .latch;
            expected_latch = speakerdesk_models::silero::hysteresis_flag(
                expected_latch,
                expected["raw_p"]
                    .as_f64()
                    .unwrap()
                    .max(expected["norm_p"].as_f64().unwrap()) as f32,
                false,
            )
            .latch;
            decision_changes += usize::from(actual_latch != expected_latch);
        }
        eprintln!(
            "Silero run {run}: {:?}, load {load:?}, max probability error {max_error}, changed speech decisions {decision_changes}",
            started.elapsed()
        );
        assert_eq!(decision_changes, 0);
    }
    packet_and_session_boundaries(&mut network, &pcm);
}

fn packet_and_session_boundaries(network: &mut Network, pcm: &[f32]) {
    use speakerdesk_models::silero::session::Session;
    use std::sync::atomic::AtomicBool;
    let running = AtomicBool::new(false);
    let mut whole = Session::new(None).unwrap();
    let expected = whole.feed(network, pcm, true, &running).unwrap();
    assert_eq!(expected.observed, pcm.len() as u64);
    assert_eq!(
        expected.frames.last().unwrap().end - expected.frames.last().unwrap().start,
        37
    );
    for size in [1, 511, 512, 513, 16_000] {
        let mut session = Session::new(None).unwrap();
        let mut frames = Vec::new();
        let mut received = 0;
        for chunk in pcm.chunks(size) {
            received += chunk.len() as u64;
            let feed = session.feed(network, chunk, false, &running).unwrap();
            assert_eq!(feed.received, received);
            assert_eq!(feed.observed, received / 512 * 512);
            frames.extend(feed.frames);
        }
        assert!(
            session
                .feed(network, &[], false, &running)
                .unwrap()
                .frames
                .is_empty()
        );
        let tail = session.feed(network, &[], true, &running).unwrap();
        assert_eq!(tail.observed, tail.received);
        frames.extend(tail.frames);
        assert_eq!(frames, expected.frames, "partition size {size}");
        assert!(session.feed(network, &[0.0], false, &running).is_err());
    }
    let mut cancelled = Session::new(None).unwrap();
    let mut other = Session::new(None).unwrap();
    cancelled
        .feed(network, &pcm[..512], false, &running)
        .unwrap();
    let first = other.feed(network, &pcm[..511], false, &running).unwrap();
    assert_eq!(first.observed, 0);
    assert!(matches!(
        cancelled.feed(network, &pcm[512..], false, &AtomicBool::new(true)),
        Err(speakerdesk_core::error::ChildError::Cancelled)
    ));
    assert!(cancelled.feed(network, &[], true, &running).is_err());
    assert_eq!(
        other
            .feed(network, &pcm[511..], true, &running)
            .unwrap()
            .frames,
        expected.frames
    );
}
