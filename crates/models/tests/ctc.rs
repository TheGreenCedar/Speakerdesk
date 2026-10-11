#![cfg(target_os = "macos")]
use speakerdesk_core::error::ChildError;
use speakerdesk_models::ctc::network::Network;
use std::{path::PathBuf, sync::atomic::AtomicBool, time::Instant};

#[test]
#[ignore = "requires SPEAKERDESK_TEST_CTC pinned converted checkpoint and Metal"]
fn supplied_columns_match_independent_full_vocabulary_scores() {
    speakerdesk_models::metallib::prepare(None).unwrap();
    mlx_rs::memory::set_cache_limit(0).unwrap();
    let directory =
        PathBuf::from(std::env::var_os("SPEAKERDESK_TEST_CTC").expect("model directory"));
    let mut network = Network::load(&directory).unwrap();
    let mut wav = hound::WavReader::open(
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/data/two-voice.wav"),
    )
    .unwrap();
    let pcm: Vec<f32> = wav
        .samples::<i16>()
        .map(|s| f32::from(s.unwrap()) / 32768.)
        .collect();
    let reference: serde_json::Value =
        serde_json::from_str(include_str!("data/ctc-reference.json")).unwrap();
    let ids: Vec<i32> = reference["token_ids"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_i64().unwrap() as i32)
        .collect();
    let cases = [
        &pcm[..],
        &pcm[16000..16400],
        &pcm[16000..16719],
        &pcm[16000..16720],
        &pcm[16000..16549],
        &[0f32; 720][..],
    ];
    for _ in 0..2 {
        for (audio, expected) in cases.iter().zip(reference["cases"].as_array().unwrap()) {
            let started = Instant::now();
            let scores = network
                .scores(audio, &ids, &AtomicBool::new(false))
                .unwrap();
            assert_eq!(
                scores.valid_frames as u64,
                expected["frames"].as_u64().unwrap()
            );
            assert_eq!(scores.audio_num_samples as usize, audio.len());
            assert_eq!(scores.frame_hop_samples, 320);
            assert_eq!(scores.frame_origin_samples, 0);
            assert_eq!(scores.model, reference["model_sha256"].as_str().unwrap());
            assert_eq!(
                scores.vocabulary_sha256,
                reference["tokens_sha256"].as_str().unwrap()
            );
            assert_eq!(scores.token_ids, ids);
            let values: Vec<f64> = expected["gathered"]
                .as_array()
                .unwrap()
                .iter()
                .flat_map(|row| row.as_array().unwrap())
                .map(|v| v.as_f64().unwrap())
                .collect();
            assert_eq!(scores.gathered.len(), values.len());
            let delta = scores
                .gathered
                .iter()
                .zip(&values)
                .map(|(a, b)| (f64::from(*a) - b).abs())
                .fold(0.0, f64::max);
            eprintln!(
                "CTC {} {:?}, max score delta {delta}",
                expected["name"],
                started.elapsed()
            );
            assert!(delta < 1e-4, "score drift {delta}");
            assert_eq!(
                network.scores(audio, &ids, &AtomicBool::new(true)),
                Err(ChildError::Cancelled)
            );
        }
    }
    let empty = network
        .scores(&pcm[..399], &ids, &AtomicBool::new(false))
        .unwrap();
    assert_eq!(empty.valid_frames, 0);
    assert!(empty.gathered.is_empty());
    for (audio, ids) in [
        (&[][..], &[0][..]),
        (&[f32::NAN][..], &[0][..]),
        (&pcm[..400], &[][..]),
        (&pcm[..400], &[-1][..]),
        (&pcm[..400], &[9812][..]),
    ] {
        assert!(network.scores(audio, ids, &AtomicBool::new(false)).is_err());
    }
    assert!(
        network
            .scores(&vec![0.; 392001], &[0], &AtomicBool::new(false))
            .is_err()
    );
}
