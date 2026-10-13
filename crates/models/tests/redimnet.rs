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
    for policy in [
        ComputePolicy::CpuAndGpu,
        ComputePolicy::CpuAndNeuralEngine,
        ComputePolicy::All,
    ] {
        if let Ok(selected) = std::env::var("SPEAKERDESK_TEST_COMPUTE_POLICY") {
            assert!(["CPU_AND_GPU", "CPU_AND_NEURAL_ENGINE", "ALL"].contains(&selected.as_str()));
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
#[ignore = "requires SPEAKERDESK_VOICE_CASES manifest, SPEAKERDESK_TEST_VOICE, selected compute policy and a fresh SPEAKERDESK_VOICE_RESULT"]
fn public_crop_voice_embeddings() {
    use std::io::Write;
    let manifest_path =
        PathBuf::from(std::env::var_os("SPEAKERDESK_VOICE_CASES").expect("public crop manifest"));
    let bytes = std::fs::read(&manifest_path).unwrap();
    let manifest: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
    let cases = manifest["cases"].as_array().expect("manifest cases");
    assert!(!cases.is_empty());
    let root = PathBuf::from(std::env::var_os("SPEAKERDESK_TEST_VOICE").expect("model directory"));
    let output = PathBuf::from(std::env::var_os("SPEAKERDESK_VOICE_RESULT").expect("result path"));
    assert!(!output.exists());
    let selected =
        std::env::var("SPEAKERDESK_TEST_COMPUTE_POLICY").expect("explicit compute policy");
    let policy = [
        ComputePolicy::CpuAndGpu,
        ComputePolicy::CpuAndNeuralEngine,
        ComputePolicy::All,
    ]
    .into_iter()
    .find(|policy| policy.name() == selected)
    .expect("CPU_AND_GPU, CPU_AND_NEURAL_ENGINE or ALL");
    let started = Instant::now();
    let network = Network::load(&root, policy).unwrap();
    let load_seconds = started.elapsed().as_secs_f64();
    let mut calls = Vec::with_capacity(cases.len());
    for case in cases {
        let id = case["id"].as_str().expect("case id");
        let path = PathBuf::from(case["path"].as_str().expect("case WAV path"));
        let path = if path.is_absolute() {
            path
        } else {
            manifest_path.parent().unwrap().join(path)
        };
        let audio = std::fs::read(&path).unwrap();
        let mut reader = hound::WavReader::new(std::io::Cursor::new(&audio)).unwrap();
        assert_eq!(reader.spec().channels, 1);
        assert_eq!(reader.spec().sample_rate, 16000);
        assert_eq!(reader.spec().bits_per_sample, 16);
        assert_eq!(reader.spec().sample_format, hound::SampleFormat::Int);
        let pcm: Vec<_> = reader
            .samples::<i16>()
            .map(|sample| f32::from(sample.unwrap()) / 32768.)
            .collect();
        let start = usize::try_from(case["start_sample"].as_u64().expect("start sample")).unwrap();
        let end = usize::try_from(case["end_sample"].as_u64().expect("end sample")).unwrap();
        assert!(start < end && end <= pcm.len(), "invalid crop: {id}");
        let started = Instant::now();
        let embedding = network
            .embed(&pcm[start..end], &AtomicBool::new(false))
            .unwrap();
        let seconds = started.elapsed().as_secs_f64();
        calls.push(serde_json::json!({"id":id,"path":path,"wav_sha256":format!("{:x}",Sha256::digest(&audio)),
            "start_sample":start,"end_sample":end,"seconds":seconds,"clean":embedding.clean,"vector":embedding.vector}));
    }
    let result = serde_json::json!({"manifest_sha256":format!("{:x}",Sha256::digest(&bytes)),
        "policy":policy.name(),"model":network.identity(),"load_seconds":load_seconds,"calls":calls,
        "limitation":"Configured compute units; actual ANE placement, recognition accuracy and live concurrency are not established by this probe."});
    std::fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(output)
        .unwrap()
        .write_all(&serde_json::to_vec_pretty(&result).unwrap())
        .unwrap();
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

#[test]
#[ignore = "requires disposable SPEAKERDESK_VOICE_FIXTURES, SPEAKERDESK_TEST_VOICE and a fresh SPEAKERDESK_VOICE_RESULT"]
fn frozen_recognition_policy_on_regenerated_synthetic_corpus() {
    use speakerdesk_core::voice::{Calibration, VoiceProfile, normalized, propose_match};
    use std::collections::BTreeMap;
    let fixtures =
        PathBuf::from(std::env::var_os("SPEAKERDESK_VOICE_FIXTURES").expect("fixture root"));
    let root = PathBuf::from(std::env::var_os("SPEAKERDESK_TEST_VOICE").expect("model directory"));
    let output = PathBuf::from(std::env::var_os("SPEAKERDESK_VOICE_RESULT").expect("result path"));
    assert!(!output.exists());
    let manifest_bytes = std::fs::read(fixtures.join("manifest.json")).unwrap();
    let manifest: serde_json::Value = serde_json::from_slice(&manifest_bytes).unwrap();
    assert_eq!(manifest["dataset_id"], "macos-compact-tts-development-v1");
    assert_eq!(manifest["fixture_source"], "synthetic");
    let groups = manifest["groups"].as_array().unwrap();
    assert_eq!(groups.len(), 36);
    let frozen: serde_json::Value =
        serde_json::from_str(include_str!("../../../speakerdesk/voice_calibration.json")).unwrap();
    assert_eq!(
        format!("{:x}", Sha256::digest(&manifest_bytes)),
        frozen["fixture_manifest_sha256"]
    );
    let mut reports = vec![];
    for policy in [ComputePolicy::CpuAndGpu, ComputePolicy::All] {
        let network = Network::load(&root, policy).unwrap();
        let model = network.identity();
        let mut rows = vec![];
        for group in groups {
            let path = fixtures.join(group["file"].as_str().unwrap());
            let bytes = std::fs::read(&path).unwrap();
            let mut reader = hound::WavReader::new(std::io::Cursor::new(&bytes)).unwrap();
            assert_eq!(reader.spec().sample_rate, 16000);
            assert_eq!(reader.spec().channels, 1);
            let pcm: Vec<_> = reader
                .samples::<i16>()
                .map(|s| f32::from(s.unwrap()) / 32768.)
                .collect();
            let mut vectors = vec![];
            let mut seconds = vec![];
            for clip in group["clips"].as_array().unwrap() {
                let left = (clip["start"].as_f64().unwrap() * 16000.).round() as usize;
                let right = (clip["end"].as_f64().unwrap() * 16000.).round() as usize;
                let started = Instant::now();
                let embedding = network
                    .embed(&pcm[left..right], &AtomicBool::new(false))
                    .unwrap();
                seconds.push(started.elapsed().as_secs_f64());
                assert!(embedding.clean);
                vectors.push(embedding.vector);
            }
            assert_eq!(vectors.len(), 2);
            rows.push(serde_json::json!({"id":group["id"],"speaker_id":group["speaker_id"],"split":group["split"],"audio_sha256":format!("{:x}",Sha256::digest(bytes)),"vectors":vectors,"seconds":seconds}));
        }
        let vectors = |row: &serde_json::Value| {
            serde_json::from_value::<Vec<Vec<f64>>>(row["vectors"].clone()).unwrap()
        };
        let mut enrolled: BTreeMap<String, Vec<f64>> = BTreeMap::new();
        for row in rows.iter().filter(|row| row["split"] == "enrollment") {
            let centroid = enrolled
                .entry(row["speaker_id"].as_str().unwrap().to_owned())
                .or_insert_with(|| vec![0.; 192]);
            for vector in vectors(row) {
                for (total, value) in centroid.iter_mut().zip(vector) {
                    *total += value;
                }
            }
        }
        assert_eq!(enrolled.len(), 4);
        let profiles: Vec<_> = enrolled
            .into_iter()
            .map(|(person_id, vector)| VoiceProfile {
                person_id,
                version: "synthetic-corpus".into(),
                enrolled_at: 0.,
                model: model.clone(),
                centroid: normalized(&vector, 192).unwrap(),
                clips: vec![],
                consent: "explicit_remember_voice".into(),
            })
            .collect();
        let cal = &frozen["calibration"];
        let calibration = Calibration::new(
            model.clone(),
            cal["dataset_id"].as_str().unwrap(),
            cal["threshold"].as_f64().unwrap(),
            cal["margin"].as_f64().unwrap(),
            cal["genuine_trials"].as_u64().unwrap(),
            cal["impostor_trials"].as_u64().unwrap(),
            cal["false_accept_rate"].as_f64().unwrap(),
            cal["false_reject_rate"].as_f64().unwrap(),
            cal["minimum_clips"].as_u64().unwrap() as usize,
        )
        .unwrap();
        let mut trials = vec![];
        for row in rows.iter().filter(|row| row["split"] != "enrollment") {
            let found = propose_match(&model, &calibration, &vectors(row), &profiles).unwrap();
            let person = found.as_ref().map(|found| found.person_id.as_str());
            trials.push(serde_json::json!({"id":row["id"],"split":row["split"],"expected":row["speaker_id"],"matched":person,"correct":person==row["speaker_id"].as_str(),"score":found.as_ref().map(|f|f.score),"margin":found.as_ref().map(|f|f.margin)}));
        }
        reports.push(serde_json::json!({"policy":policy.name(),"model":model,"frozen_threshold":cal["threshold"],"frozen_margin":cal["margin"],"rows":rows,"trials":trials}));
    }
    let result = serde_json::json!({"manifest_sha256":format!("{:x}",Sha256::digest(manifest_bytes)),"limitation":"Regenerated synthetic corpus; frozen policy, no tuning; not human population recognition accuracy or legacy profile relabeling authority.","reports":reports});
    std::fs::write(output, serde_json::to_vec_pretty(&result).unwrap()).unwrap();
    for report in result["reports"].as_array().unwrap() {
        assert_eq!(report["trials"].as_array().unwrap().len(), 28);
        assert!(
            report["trials"]
                .as_array()
                .unwrap()
                .iter()
                .all(|trial| trial["correct"] == true),
            "frozen policy failed: {}",
            report["policy"]
        );
    }
}
