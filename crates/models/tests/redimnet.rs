#![cfg(target_os = "macos")]

use sha2::{Digest, Sha256};
use speakerdesk_core::error::ChildError;
use speakerdesk_core::voice::IDENTITY_REUSE_COSINE;
use speakerdesk_models::redimnet::{
    MIN_SAMPLES,
    coreml::{ComputePolicy, Network, verify_artifact},
};
use std::{path::PathBuf, sync::atomic::AtomicBool, time::Instant};

#[test]
#[ignore = "requires SPEAKERDESK_TEST_VOICE and a fresh SPEAKERDESK_VOICE_RESULT output path"]
fn native_coreml_policies_return_stable_normalized_embeddings() {
    let root = PathBuf::from(std::env::var_os("SPEAKERDESK_TEST_VOICE").expect("model directory"));
    let output = PathBuf::from(std::env::var_os("SPEAKERDESK_VOICE_RESULT").expect("result path"));
    assert!(!output.exists());
    let reference: serde_json::Value =
        serde_json::from_str(include_str!("data/redimnet-mlx-reference.json")).unwrap();
    let wav = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/data/two-voice.wav");
    assert_eq!(
        format!("{:x}", Sha256::digest(std::fs::read(&wav).unwrap())),
        reference["wav_sha256"]
    );
    let mut reader = hound::WavReader::open(wav).unwrap();
    let pcm: Vec<_> = reader
        .samples::<i16>()
        .map(|s| f32::from(s.unwrap()) / 32768.)
        .collect();
    let cases = [&pcm[..64000], &pcm[80000..]];
    let mut results = vec![];
    for policy in [ComputePolicy::CpuAndGpu, ComputePolicy::All] {
        if let Ok(selected) = std::env::var("SPEAKERDESK_TEST_COMPUTE_POLICY") {
            assert!(["CPU_AND_GPU", "ALL"].contains(&selected.as_str()));
            if selected != policy.name() {
                continue;
            }
        }
        let started = Instant::now();
        let network = Network::load(&root, policy).unwrap();
        let load_seconds = started.elapsed().as_secs_f64();
        let mut calls = vec![];
        for round in 0..3 {
            for (speaker, audio) in cases.iter().enumerate() {
                let start = Instant::now();
                let embedding = network.embed(audio, &AtomicBool::new(false)).unwrap();
                let seconds = start.elapsed().as_secs_f64();
                assert!(embedding.clean);
                assert_eq!(embedding.vector.len(), 192);
                assert!(embedding.vector.iter().all(|v| v.is_finite()));
                assert!((embedding.vector.iter().map(|v| v * v).sum::<f64>() - 1.).abs() < 1e-12);
                let expected = &reference["cases"][speaker];
                assert_eq!(audio.len(), expected["samples"].as_u64().unwrap() as usize);
                let cosine = embedding
                    .vector
                    .iter()
                    .zip(expected["vector"].as_array().unwrap())
                    .map(|(a, b)| a * b.as_f64().unwrap())
                    .sum::<f64>();
                assert!(
                    cosine >= IDENTITY_REUSE_COSINE,
                    "Core ML disagrees with independent legacy MLX reference: {cosine}"
                );
                if round > 0 {
                    let first: &serde_json::Value = &calls[speaker];
                    let cosine = embedding
                        .vector
                        .iter()
                        .zip(first["vector"].as_array().unwrap())
                        .map(|(a, b)| a * b.as_f64().unwrap())
                        .sum::<f64>();
                    assert!(
                        cosine >= IDENTITY_REUSE_COSINE,
                        "resident predictor changed: {cosine}"
                    );
                }
                calls.push(serde_json::json!({"round":round,"speaker":speaker,"samples":audio.len(),"seconds":seconds,"legacy_cosine":cosine,"vector":embedding.vector}));
            }
        }
        assert_eq!(
            network.embed(cases[0], &AtomicBool::new(true)).unwrap_err(),
            ChildError::Cancelled
        );
        for bad in [
            vec![],
            vec![0.; MIN_SAMPLES - 1],
            vec![0.; 160001],
            vec![f32::NAN; MIN_SAMPLES],
        ] {
            assert!(network.embed(&bad, &AtomicBool::new(false)).is_err());
        }
        for rejected in [vec![0.; MIN_SAMPLES], vec![1.; MIN_SAMPLES]] {
            let result = network.embed(&rejected, &AtomicBool::new(false)).unwrap();
            assert!(!result.clean && result.vector.is_empty());
        }
        results.push(serde_json::json!({"policy":policy.name(),"model":network.identity(),"load_seconds":load_seconds,"calls":calls}));
    }
    std::fs::write(output, serde_json::to_vec_pretty(&results).unwrap()).unwrap();
}

#[test]
#[ignore = "requires SPEAKERDESK_TEST_VOICE; mutates only a temporary artifact copy"]
fn artifact_boundary_rejects_links_extras_and_corruption() {
    use std::os::unix::fs::symlink;
    let source =
        PathBuf::from(std::env::var_os("SPEAKERDESK_TEST_VOICE").expect("model directory"));
    let temp = tempfile::tempdir().unwrap();
    let root = temp.path().join("model");
    std::fs::create_dir(&root).unwrap();
    let manifest: serde_json::Value =
        serde_json::from_str(include_str!("../../../speakerdesk/redimnet2_artifact.json")).unwrap();
    for name in manifest["files"].as_object().unwrap().keys() {
        let path = root.join(name);
        std::fs::create_dir_all(path.parent().unwrap()).unwrap();
        std::fs::copy(source.join(name), path).unwrap();
    }
    verify_artifact(&root).unwrap();
    let alias = temp.path().join("linked-root");
    symlink(&root, &alias).unwrap();
    assert!(verify_artifact(&alias).is_err());
    let compiled = root.join("ReDimNet2B6.mlmodelc");
    let retained = root.join("retained-compiled");
    std::fs::rename(&compiled, &retained).unwrap();
    symlink(&retained, &compiled).unwrap();
    assert!(verify_artifact(&root).is_err());
    std::fs::remove_file(&compiled).unwrap();
    std::fs::rename(&retained, &compiled).unwrap();
    for directory in [false, true] {
        let extra = compiled.join("unexpected");
        if directory {
            std::fs::create_dir(&extra).unwrap();
        } else {
            std::fs::write(&extra, []).unwrap();
        }
        assert!(verify_artifact(&root).is_err());
        if directory {
            std::fs::remove_dir(extra).unwrap();
        } else {
            std::fs::remove_file(extra).unwrap();
        }
    }
    let name = manifest["files"]
        .as_object()
        .unwrap()
        .keys()
        .find(|name| name.starts_with("ReDimNet2B6.mlmodelc/"))
        .unwrap();
    let path = root.join(name);
    let bytes = std::fs::read(&path).unwrap();
    std::fs::remove_file(&path).unwrap();
    symlink(source.join(name), &path).unwrap();
    assert!(verify_artifact(&root).is_err());
    std::fs::remove_file(&path).unwrap();
    let mut corrupted = bytes.clone();
    corrupted[0] ^= 1;
    std::fs::write(&path, corrupted).unwrap();
    assert!(verify_artifact(&root).is_err());
    std::fs::write(&path, bytes).unwrap();
    verify_artifact(&root).unwrap();
}
