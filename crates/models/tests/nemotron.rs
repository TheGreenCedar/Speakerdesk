#![cfg(target_os = "macos")]
use speakerdesk_models::sortformer::{frames_to_turns, network::Network, session::Session};
use std::{path::PathBuf, sync::atomic::AtomicBool, time::Instant};

#[test]
#[ignore = "requires pinned checkpoint, independent long reference and Metal"]
fn multiple_offline_windows_match_pinned_reference() {
    speakerdesk_models::metallib::prepare(None).unwrap();
    let models = PathBuf::from(std::env::var_os("SPEAKERDESK_TEST_NEMOTRON").unwrap());
    let proof = PathBuf::from(std::env::var_os("SPEAKERDESK_TEST_NEMOTRON_LONG").unwrap());
    let reference: serde_json::Value =
        serde_json::from_slice(&std::fs::read(proof.join("reference.json")).unwrap()).unwrap();
    let expected: Vec<[f32; 8]> =
        serde_json::from_value(reference["probabilities"].clone()).unwrap();
    assert_eq!(
        reference["source_revision"],
        "feb25a37b07923bae556e59111995071d66afa0d"
    );
    assert_eq!(
        reference["wav_sha256"],
        "3297950fe04a7a7310436f2afaa2f6a0f49163c6c5a176871a3337a90d70f6a8"
    );
    let mut wav = hound::WavReader::open(proof.join("three-windows.wav")).unwrap();
    let pcm: Vec<f32> = wav
        .samples::<i16>()
        .map(|s| s.unwrap() as f32 / 32768.)
        .collect();
    let mut network = Network::load(&models).unwrap();
    let got = network
        .probabilities(&pcm, &AtomicBool::new(false))
        .unwrap();
    assert_eq!(got.len(), pcm.len() / 160);
    assert_eq!(expected.len(), got.len());
    assert_eq!(
        reference["valid_frames"].as_u64().unwrap() as usize,
        got.len()
    );
    let maximum = got
        .iter()
        .zip(&expected)
        .flat_map(|(a, b)| a.iter().zip(b))
        .map(|(a, b)| (a - b).abs())
        .fold(0f32, f32::max);
    assert_eq!(maximum, 0., "pinned BF16 probabilities differ");
    use sha2::{Digest, Sha256};
    assert_eq!(
        format!(
            "{:x}",
            Sha256::digest(std::fs::read(proof.join("three-windows.wav")).unwrap())
        ),
        reference["wav_sha256"].as_str().unwrap()
    );
    assert_eq!(
        reference["runtime_sha256"]["libmlx.dylib"],
        "0c9db159aedd2951bb77610c8b028c4ef79bc64ccc4c73b07a4452a60612ac03"
    );
    assert_eq!(
        reference["runtime_sha256"]["mlx.metallib"],
        "2d2e729831a995a39778164c93ea1b2e8d87f7d8b1e717386d3d816f46f416b2"
    );
    for splits in [
        &[1, 159, 160, 511, 512, 486_399, 17][..],
        &[16000, 435_201, 1][..],
    ] {
        let mut session = Session::new(speakerdesk_core::ipc::DiarPreset::Offline).unwrap();
        let mut cursor = 0;
        let mut index = 0;
        let mut observed = 0;
        let mut activity = vec![[false; 8]; expected.len()];
        loop {
            let end = (cursor + splits[index % splits.len()]).min(pcm.len());
            let last = cursor == pcm.len();
            let feed = session
                .feed(
                    &mut network,
                    &pcm[cursor..end],
                    last,
                    &AtomicBool::new(false),
                )
                .unwrap();
            feed.validate(end as u64, observed).unwrap();
            observed = feed.observed;
            for turn in feed.turns {
                let speaker: usize = turn
                    .speaker
                    .strip_prefix("speaker_")
                    .unwrap()
                    .parse()
                    .unwrap();
                for row in &mut activity[(turn.start / 160) as usize..(turn.end / 160) as usize] {
                    row[speaker] = true;
                }
            }
            cursor = end;
            index += 1;
            if last {
                break;
            }
        }
        assert_eq!(observed as usize, pcm.len() / 160 * 160);
        assert_eq!(
            activity,
            expected
                .iter()
                .map(|row| row.map(|p| p > 0.5))
                .collect::<Vec<_>>()
        );
        assert!(
            session
                .feed(&mut network, &[], true, &AtomicBool::new(false))
                .is_err()
        );
    }
}

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
    for samples in [435_199, 435_200, 435_201, 486_400, 486_559] {
        assert_eq!(
            network
                .probabilities(&vec![0.; samples], &AtomicBool::new(false))
                .unwrap()
                .len(),
            samples / 160
        );
    }
    assert!(
        network
            .probabilities(&[f32::NAN; 160], &AtomicBool::new(false))
            .is_err()
    );
}

#[test]
#[ignore = "requires pinned checkpoints, independent Low reference and Metal"]
fn interleaved_presets_and_cancelled_state_are_independent() {
    use speakerdesk_core::ipc::DiarPreset;
    use std::sync::atomic::Ordering;
    speakerdesk_models::metallib::prepare(None).unwrap();
    let model = PathBuf::from(std::env::var_os("SPEAKERDESK_TEST_NEMOTRON").unwrap());
    let proof = PathBuf::from(std::env::var_os("SPEAKERDESK_TEST_NEMOTRON_LONG").unwrap());
    let reference: serde_json::Value =
        serde_json::from_slice(&std::fs::read(proof.join("low.json")).unwrap()).unwrap();
    let expected: Vec<[f32; 8]> =
        serde_json::from_value(reference["probabilities"].clone()).unwrap();
    assert_eq!(reference["preset"], "low");
    let mut reader = hound::WavReader::open(
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/data/two-voice.wav"),
    )
    .unwrap();
    let pcm: Vec<f32> = reader
        .samples::<i16>()
        .map(|s| s.unwrap() as f32 / 32768.)
        .collect();
    assert_eq!(expected.len(), pcm.len() / 160);
    let mut network = Network::load(&model).unwrap();
    let mut low = Session::new(DiarPreset::Low).unwrap();
    let mut offline = Session::new(DiarPreset::Offline).unwrap();
    let cancel = AtomicBool::new(false);
    let first = low
        .feed(&mut network, &pcm[..72_000], false, &cancel)
        .unwrap();
    assert!(first.observed > 0 && first.observed < first.received);
    let delayed = offline
        .feed(&mut network, &pcm[..72_000], false, &cancel)
        .unwrap();
    assert_eq!(delayed.observed, 0);
    cancel.store(true, Ordering::Release);
    assert_eq!(
        offline
            .feed(&mut network, &pcm[72_000..], true, &cancel)
            .unwrap_err(),
        speakerdesk_core::error::ChildError::Cancelled
    );
    cancel.store(false, Ordering::Release);
    assert!(offline.feed(&mut network, &[], true, &cancel).is_err());
    let last = low
        .feed(&mut network, &pcm[72_000..], true, &cancel)
        .unwrap();
    last.validate(pcm.len() as u64, first.observed).unwrap();
    let mut activity = vec![[false; 8]; expected.len()];
    for turn in first.turns.into_iter().chain(last.turns) {
        let speaker: usize = turn
            .speaker
            .strip_prefix("speaker_")
            .unwrap()
            .parse()
            .unwrap();
        for row in &mut activity[(turn.start / 160) as usize..(turn.end / 160) as usize] {
            row[speaker] = true;
        }
    }
    assert_eq!(
        activity,
        expected
            .iter()
            .map(|row| row.map(|p| p > 0.5))
            .collect::<Vec<_>>()
    );
    let fresh = network.probabilities(&pcm, &cancel).unwrap();
    let short: serde_json::Value =
        serde_json::from_str(include_str!("data/nemotron-reference.json")).unwrap();
    let expected_turns: Vec<speakerdesk_core::ipc::Turn> =
        serde_json::from_value(short["turns"].clone()).unwrap();
    assert_eq!(frames_to_turns(&fresh), expected_turns);
}
