#![cfg(windows)]

use speakerdesk_core::{
    error::ChildError,
    voice::{IDENTITY_REUSE_COSINE, caller_may_reuse_identity},
};
use speakerdesk_models::{
    operation::Operation,
    redimnet::{CLIP_ABS, MAX_SAMPLES, MIN_SAMPLES, network_windows::Network},
    windows_runtime::Backend,
};
use std::{path::PathBuf, time::Instant};

#[path = "support/ordinary_voice.rs"]
mod ordinary_voice;

#[test]
#[ignore = "requires pinned SPEAKERDESK_TEST_VOICE, SPEAKERDESK_TEST_WINDOWS_RUNTIME and DirectML"]
fn pinned_windows_graph_preserves_independent_voice_vectors_and_waveform_policy() {
    let model = PathBuf::from(std::env::var_os("SPEAKERDESK_TEST_VOICE").expect("model directory"));
    let runtime = PathBuf::from(
        std::env::var_os("SPEAKERDESK_TEST_WINDOWS_RUNTIME").expect("private runtime directory"),
    );
    let backend = Backend::from_directory(&runtime).unwrap();
    let mut network = Network::load(&model, &backend).unwrap();
    let reference: serde_json::Value =
        serde_json::from_str(include_str!("data/redimnet-mlx-reference.json")).unwrap();
    let wav = include_bytes!("data/two-voice.wav");
    assert_eq!(
        speakerdesk_core::memo::sha256_hex(wav),
        reference["wav_sha256"].as_str().unwrap()
    );
    let mut reader = hound::WavReader::new(std::io::Cursor::new(wav)).unwrap();
    let pcm: Vec<_> = reader
        .samples::<i16>()
        .map(|sample| f32::from(sample.unwrap()) / 32768.)
        .collect();
    let cases = [&pcm[..64000], &pcm[80000..]];
    let mut previous: Vec<Vec<f64>> = Vec::new();
    for round in 0..3 {
        for (speaker, audio) in cases.iter().enumerate() {
            assert_eq!(
                audio.len(),
                reference["cases"][speaker]["samples"].as_u64().unwrap() as usize
            );
            let operation = Operation::native().unwrap();
            let started = Instant::now();
            let result = network.embed(audio, &operation).unwrap();
            assert!(result.clean);
            assert_eq!(result.vector.len(), 192);
            assert!(result.vector.iter().all(|value| value.is_finite()));
            assert!(
                (result.vector.iter().map(|value| value * value).sum::<f64>() - 1.).abs() < 1e-12
            );
            let cosine: f64 = result
                .vector
                .iter()
                .zip(reference["cases"][speaker]["vector"].as_array().unwrap())
                .map(|(actual, expected)| actual * expected.as_f64().unwrap())
                .sum();
            assert!(
                caller_may_reuse_identity(cosine),
                "independent saved-identity gate >= {IDENTITY_REUSE_COSINE}: {cosine}"
            );
            if round == 0 {
                previous.push(result.vector);
            } else {
                let repeated: f64 = result
                    .vector
                    .iter()
                    .zip(&previous[speaker])
                    .map(|(a, b)| a * b)
                    .sum();
                assert!(
                    repeated >= IDENTITY_REUSE_COSINE,
                    "resident repeat changed: {repeated}"
                );
            }
            eprintln!(
                "ReDimNet Windows round {round} speaker {speaker}: {:?}, legacy cosine {cosine}, identity {:?}",
                started.elapsed(),
                network.identity()
            );
        }
    }
    let running = Operation::native().unwrap();
    for rejected in [
        vec![0.; MIN_SAMPLES],
        vec![CLIP_ABS; MIN_SAMPLES],
        vec![-CLIP_ABS; MIN_SAMPLES],
    ] {
        let result = network.embed(&rejected, &running).unwrap();
        assert!(!result.clean && result.vector.is_empty());
    }
    for invalid in [
        vec![],
        vec![0.; MIN_SAMPLES - 1],
        vec![0.; MAX_SAMPLES + 1],
        vec![f32::NAN; MIN_SAMPLES],
        vec![f32::INFINITY; MIN_SAMPLES],
        vec![f32::NEG_INFINITY; MIN_SAMPLES],
    ] {
        assert!(network.embed(&invalid, &running).is_err());
    }
    let cancelled = Operation::native().unwrap();
    cancelled.cancel();
    assert_eq!(
        network.embed(cases[0], &cancelled).unwrap_err(),
        ChildError::Cancelled
    );
    // A cancelled request must not poison the resident session's next operation.
    let fresh = network
        .embed(cases[0], &Operation::native().unwrap())
        .unwrap();
    let cosine: f64 = fresh
        .vector
        .iter()
        .zip(&previous[0])
        .map(|(a, b)| a * b)
        .sum();
    assert!(cosine >= IDENTITY_REUSE_COSINE);
}

#[test]
#[ignore = "requires SPEAKERDESK_VOICE_CASES, SPEAKERDESK_TEST_VOICE, SPEAKERDESK_TEST_WINDOWS_RUNTIME and a fresh SPEAKERDESK_VOICE_RESULT"]
fn public_crop_voice_embeddings() {
    use std::{collections::BTreeSet, io::Write};

    let manifest_path =
        PathBuf::from(std::env::var_os("SPEAKERDESK_VOICE_CASES").expect("public crop manifest"));
    let bytes = std::fs::read(&manifest_path).unwrap();
    let manifest: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
    let cases = manifest["cases"].as_array().expect("manifest cases");
    assert!(!cases.is_empty());
    let model = PathBuf::from(std::env::var_os("SPEAKERDESK_TEST_VOICE").expect("model directory"));
    let runtime = PathBuf::from(
        std::env::var_os("SPEAKERDESK_TEST_WINDOWS_RUNTIME").expect("private runtime directory"),
    );
    let output = PathBuf::from(std::env::var_os("SPEAKERDESK_VOICE_RESULT").expect("result path"));
    assert!(!output.exists(), "result must be a fresh path");
    let backend = Backend::from_directory(&runtime).unwrap();
    let started = Instant::now();
    let mut network = Network::load(&model, &backend).unwrap();
    let load_seconds = started.elapsed().as_secs_f64();
    let mut calls = Vec::with_capacity(cases.len());
    let mut ids = BTreeSet::new();
    for case in cases {
        let id = case["id"].as_str().expect("case id");
        assert!(
            !id.is_empty() && ids.insert(id),
            "empty or duplicate case id: {id}"
        );
        let path = PathBuf::from(case["path"].as_str().expect("case WAV path"));
        let path = if path.is_absolute() {
            path
        } else {
            manifest_path.parent().unwrap().join(path)
        };
        let audio = std::fs::read(&path).unwrap();
        let wav_sha256 = speakerdesk_core::memo::sha256_hex(&audio);
        assert_eq!(
            wav_sha256,
            case["wav_sha256"].as_str().expect("pinned WAV hash"),
            "public crop waveform differs: {id}"
        );
        let mut reader = hound::WavReader::new(std::io::Cursor::new(&audio)).unwrap();
        assert_eq!(reader.spec().channels, 1, "mono PCM required: {id}");
        assert_eq!(
            reader.spec().sample_rate,
            16000,
            "16 kHz PCM required: {id}"
        );
        assert_eq!(reader.spec().bits_per_sample, 16, "PCM16 required: {id}");
        assert_eq!(reader.spec().sample_format, hound::SampleFormat::Int);
        let pcm: Vec<_> = reader
            .samples::<i16>()
            .map(|sample| f32::from(sample.unwrap()) / 32768.)
            .collect();
        let start = usize::try_from(case["start_sample"].as_u64().expect("start sample")).unwrap();
        let end = usize::try_from(case["end_sample"].as_u64().expect("end sample")).unwrap();
        assert!(start < end && end <= pcm.len(), "invalid crop: {id}");
        assert!(
            (MIN_SAMPLES..=MAX_SAMPLES).contains(&(end - start)),
            "crop outside the admitted voice duration: {id}"
        );
        let generated_query = case["role"] == "generated_query";
        let prepared = if generated_query {
            speakerdesk_core::voice::separation::query_pcm(&pcm[start..end]).unwrap()
        } else {
            pcm[start..end].to_vec()
        };
        let prepared_pcm_sha256 = speakerdesk_core::memo::sha256_hex(
            &prepared
                .iter()
                .flat_map(|sample| sample.to_le_bytes())
                .collect::<Vec<_>>(),
        );
        let operation = Operation::native().unwrap();
        let started = Instant::now();
        let embedding = network.embed(&prepared, &operation).unwrap();
        let seconds = started.elapsed().as_secs_f64();
        calls.push(
            serde_json::json!({"id":id,"path":path,"wav_sha256":wav_sha256,
            "start_sample":start,"end_sample":end,"seconds":seconds,
            "generated_query_headroom":generated_query,"prepared_pcm_sha256":prepared_pcm_sha256,
            "clean":embedding.clean,"vector":embedding.vector}),
        );
    }
    let result = serde_json::json!({
        "manifest_sha256":speakerdesk_core::memo::sha256_hex(&bytes),
        "policy":"DirectML:ORT_DISABLE_ALL","model":network.identity(),
        "load_seconds":load_seconds,"calls":calls,
        "limitation":"Fixed public crops and native embeddings; recognition decisions and assembled query preparation are evaluated separately."
    });
    std::fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(output)
        .unwrap()
        .write_all(&serde_json::to_vec_pretty(&result).unwrap())
        .unwrap();
}

#[test]
#[ignore = "requires frozen SPEAKERDESK_VOICE_QUALIFICATION manifests and native Windows embedding receipts"]
fn frozen_natural_separated_voice_decisions() {
    use speakerdesk_core::voice::{
        Calibration, VoiceModel, normalized,
        separation::{Policy, Reference, Target, centroid, propose, query_pcm},
    };
    use std::{collections::BTreeMap, io::Write, path::Path};

    fn load(
        root: &Path,
        split: &str,
        policy: &serde_json::Value,
        model: &VoiceModel,
    ) -> (
        Vec<serde_json::Value>,
        BTreeMap<String, serde_json::Value>,
        String,
    ) {
        let name = format!("{split}-fixed-crops.json");
        let bytes = std::fs::read(root.join(&name)).unwrap();
        let digest = speakerdesk_core::memo::sha256_hex(&bytes);
        assert_eq!(digest, policy["case_pins"][&name]);
        let manifest: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
        let (original, receipt) = if split == "development" {
            ("prepared-query-cases.json", "prepared-query-vectors-2.json")
        } else {
            (
                "holdout/labelled-query-cases.json",
                "holdout/query-vectors.json",
            )
        };
        assert_eq!(
            manifest["original_manifest_sha256"],
            policy["source_pins"][original]
        );
        assert_eq!(
            manifest["original_query_receipt_sha256"],
            policy["source_query_receipts"][receipt]
        );
        let result_bytes = std::fs::read(root.join(format!("{split}-windows-v1.json"))).unwrap();
        let result_hash = speakerdesk_core::memo::sha256_hex(&result_bytes);
        let result: serde_json::Value = serde_json::from_slice(&result_bytes).unwrap();
        assert_eq!(result["manifest_sha256"], digest);
        assert_eq!(result["model"], serde_json::to_value(model).unwrap());
        assert_eq!(result["policy"], "DirectML:ORT_DISABLE_ALL");
        let cases = manifest["cases"].as_array().unwrap().clone();
        let rows = result["calls"].as_array().unwrap();
        assert_eq!(rows.len(), cases.len());
        let mut calls = BTreeMap::new();
        for row in rows {
            let id = row["id"].as_str().unwrap().to_owned();
            assert!(
                calls.insert(id, row.clone()).is_none(),
                "duplicate result row"
            );
        }
        let mut ids = std::collections::BTreeSet::new();
        for case in &cases {
            let id = case["id"].as_str().unwrap();
            assert!(ids.insert(id), "duplicate manifest row");
            let row = calls.get(id).expect("missing native result row");
            for key in ["wav_sha256", "start_sample", "end_sample"] {
                assert_eq!(row[key], case[key], "result/crop mismatch: {id} {key}");
            }
            let generated = case["role"] == "generated_query";
            assert_eq!(row["generated_query_headroom"], generated);
            let audio = std::fs::read(root.join(case["path"].as_str().unwrap())).unwrap();
            assert_eq!(
                speakerdesk_core::memo::sha256_hex(&audio),
                case["wav_sha256"]
            );
            let mut reader = hound::WavReader::new(std::io::Cursor::new(audio)).unwrap();
            assert_eq!(
                (
                    reader.spec().channels,
                    reader.spec().sample_rate,
                    reader.spec().bits_per_sample
                ),
                (1, 16000, 16)
            );
            assert_eq!(reader.spec().sample_format, hound::SampleFormat::Int);
            let pcm: Vec<_> = reader
                .samples::<i16>()
                .map(|v| f32::from(v.unwrap()) / 32768.)
                .collect();
            let start = usize::try_from(case["start_sample"].as_u64().unwrap()).unwrap();
            let end = usize::try_from(case["end_sample"].as_u64().unwrap()).unwrap();
            assert!(start < end && end <= pcm.len());
            assert!((MIN_SAMPLES..=MAX_SAMPLES).contains(&(end - start)));
            let prepared = if generated {
                query_pcm(&pcm[start..end]).unwrap()
            } else {
                pcm[start..end].to_vec()
            };
            let hash = speakerdesk_core::memo::sha256_hex(
                &prepared
                    .iter()
                    .flat_map(|v| v.to_le_bytes())
                    .collect::<Vec<_>>(),
            );
            assert_eq!(
                row["prepared_pcm_sha256"], hash,
                "prepared input mismatch: {id}"
            );
            let vector: Vec<f64> = serde_json::from_value(row["vector"].clone()).unwrap();
            if row["clean"].as_bool().unwrap() {
                normalized(&vector, model.dimension).unwrap();
            } else {
                assert!(vector.is_empty(), "unclean output must not carry a vector");
            }
        }
        (cases, calls, result_hash)
    }

    fn vectors(row: &serde_json::Value) -> Vec<Vec<f64>> {
        if row["clean"] == true {
            vec![serde_json::from_value(row["vector"].clone()).unwrap()]
        } else {
            Vec::new()
        }
    }

    let root = PathBuf::from(
        std::env::var_os("SPEAKERDESK_VOICE_QUALIFICATION").expect("qualification directory"),
    );
    let bytes = std::fs::read(root.join("prospective-policy-fixed-crops.json")).unwrap();
    let policy_hash = speakerdesk_core::memo::sha256_hex(&bytes);
    assert_eq!(
        policy_hash,
        "8a146574442c1040ad9717c6ff9018fca10952d5d6366ddefda90293422b4dd1"
    );
    let frozen: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
    let model = speakerdesk_models::redimnet::network_windows::identity();
    assert_eq!(frozen["model"], serde_json::to_value(&model).unwrap());
    let policy = Policy {
        separator_model: speakerdesk_core::separation::MODEL.into(),
        // Source development counts are frozen metadata, not a claim about Windows observations.
        calibration: Calibration::new(
            model.clone(),
            "librispeech-clean-generated-query-v1:development-6-speakers",
            frozen["threshold"].as_f64().unwrap(),
            frozen["margin"].as_f64().unwrap(),
            6,
            36,
            0.,
            0.,
            frozen["reference_minimum_clips"].as_u64().unwrap() as usize,
        )
        .unwrap(),
    };
    // Fixed from the retained Mac holdout/scores.json (SHA 9b1d2a...23d67c), before reading Windows scores.
    let abstention = "5639-40744-0034_4507-16021-0058-ch1";
    let output = root.join("decisions-windows-v1.json");
    assert!(!output.exists(), "decisions must be a fresh file");
    let mut references = Vec::new();
    let mut decisions = Vec::new();
    let mut result_hashes = BTreeMap::new();
    for (split, people, queries, controls) in [("development", 6, 6, 3), ("holdout", 4, 4, 0)] {
        let (cases, calls, result_hash) = load(&root, split, &frozen, &model);
        result_hashes.insert(format!("{split}-windows-v1.json"), result_hash);
        let mut enrolled = BTreeMap::<String, Vec<Vec<f64>>>::new();
        let mut reference_hashes = std::collections::BTreeSet::new();
        for case in cases.iter().filter(|case| case["role"] == "reference") {
            assert!(reference_hashes.insert(case["wav_sha256"].as_str().unwrap()));
            let speaker = case["speaker"].as_u64().unwrap().to_string();
            let row = &calls[case["id"].as_str().unwrap()];
            assert_eq!(row["clean"], true, "an independent reference was rejected");
            enrolled.entry(speaker).or_default().extend(vectors(row));
        }
        assert_eq!(enrolled.len(), people);
        for (speaker, clips) in enrolled {
            assert_eq!(clips.len(), 2, "two independent reference clips per person");
            assert!(
                !references
                    .iter()
                    .any(|r: &Reference| r.person_id.as_ref() == Some(&speaker)),
                "development/holdout speakers overlap"
            );
            references.push(Reference {
                target: Target::Person {
                    person_id: speaker.clone(),
                    profile_version: "public-fixture-v1".into(),
                },
                person_id: Some(speaker),
                centroid: centroid(&model, &clips).unwrap(),
            });
        }
        assert_eq!(
            references.len(),
            if split == "development" { 6 } else { 10 }
        );
        assert_eq!(
            cases
                .iter()
                .filter(|case| case["role"] == "generated_query")
                .count(),
            queries
        );
        assert_eq!(
            cases
                .iter()
                .filter(|case| case["role"] == "nonspeech_control")
                .count(),
            controls
        );
        assert_eq!(cases.len(), people * 2 + queries + controls);
        let mut episodes = BTreeMap::<String, [Option<&serde_json::Value>; 2]>::new();
        for case in cases
            .iter()
            .filter(|case| case["role"] == "generated_query")
        {
            let (episode, channel) = case["id"].as_str().unwrap().rsplit_once("-ch").unwrap();
            let channel: usize = channel.parse().unwrap();
            assert!(channel < 2);
            if let Some(recorded) = case["episode"].as_str() {
                assert_eq!(episode, recorded);
            }
            if let Some(recorded) = case["channel"].as_u64() {
                assert_eq!(channel as u64, recorded);
            }
            assert!(
                episodes.entry(episode.into()).or_insert([None, None])[channel]
                    .replace(case)
                    .is_none()
            );
        }
        let mut groups = Vec::new();
        for (episode, pair) in episodes {
            let [Some(left), Some(right)] = pair else {
                panic!("incomplete channel pair: {episode}")
            };
            groups.push(vec![left, right]);
        }
        groups.extend(
            cases
                .iter()
                .filter(|case| case["role"] == "nonspeech_control")
                .map(|case| vec![case]),
        );
        for group in groups {
            let mut channels = [Vec::new(), Vec::new()];
            for (channel, case) in group.iter().enumerate() {
                channels[channel] = vectors(&calls[case["id"].as_str().unwrap()]);
            }
            let found = propose(
                &policy,
                &policy.separator_model,
                &model,
                &channels,
                &references,
            )
            .unwrap();
            for (channel, case) in group.iter().enumerate() {
                let id = case["id"].as_str().unwrap();
                let actual = found[channel].as_ref().map(|m| match &m.target {
                    Target::Person { person_id, .. } => person_id.clone(),
                    Target::Track(_) => unreachable!(),
                });
                let truth = case["speaker"].as_u64().map(|s| s.to_string());
                let expected = if split == "holdout" && id == abstention {
                    None
                } else {
                    truth.clone()
                };
                // Report every cosine, but core propose() alone owns threshold, margin and collision decisions.
                let scores: Vec<_> = channels[channel].first().map(|v| {
                    let v = normalized(v, model.dimension).unwrap();
                    references.iter().map(|r| serde_json::json!({"speaker":r.person_id,
                        "score":v.iter().zip(&r.centroid).map(|(a,b)|a*b).sum::<f64>().clamp(-1.,1.)})).collect()
                }).unwrap_or_default();
                decisions.push(serde_json::json!({"split":split,"id":id,"true_speaker":truth,"expected":expected,
                    "matched":actual,"correct":actual==expected,"wrong_accepted":actual.is_some()&&actual!=truth,
                    "clean":calls[id]["clean"],"scores":scores,"score":found[channel].as_ref().map(|m|m.score),
                    "margin":found[channel].as_ref().map(|m|m.margin)}));
            }
        }
    }
    let report = serde_json::json!({"policy_sha256":policy_hash,"policy":frozen,"model":model,
        "expected_abstention":abstention,
        "expected_abstention_source_sha256":"9b1d2ae2f0c26724173408051e8fc5631c940b248a389e6a4161bbb68223d67c",
        "embedding_receipt_sha256":result_hashes,"decisions":decisions,
        "scope":"Prepared Mac separated-query PCM and Windows reference/query embeddings; frozen prospective policy; no threshold fitting or production activation."});
    std::fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(&output)
        .unwrap()
        .write_all(&serde_json::to_vec_pretty(&report).unwrap())
        .unwrap();
    assert_eq!(report["decisions"].as_array().unwrap().len(), 13);
    assert!(
        report["decisions"]
            .as_array()
            .unwrap()
            .iter()
            .all(|row| row["correct"] == true && row["wrong_accepted"] == false),
        "frozen decisions changed; observations retained at {}",
        output.display()
    );
}

#[test]
#[ignore = "requires frozen SPEAKERDESK_ORDINARY_VOICE_QUALIFICATION manifests and native Windows embeddings"]
fn frozen_natural_ordinary_voice_decisions() {
    let root = PathBuf::from(
        std::env::var_os("SPEAKERDESK_ORDINARY_VOICE_QUALIFICATION")
            .expect("ordinary qualification directory"),
    );
    let report = ordinary_voice::score(
        &root,
        speakerdesk_models::redimnet::network_windows::identity(),
        "ordinary-windows-v1.json",
        "ordinary-decisions-windows-v1.json",
        "DirectML:ORT_DISABLE_ALL",
    );
    assert_eq!(report["model"], report["policy"]["model"]);
    assert!(
        report["all_inputs_clean"] == true
            && report["profiles_admitted"] == 6
            && report["decisions"]
                .as_array()
                .unwrap()
                .iter()
                .all(|d| d["correct"] == true && d["wrong_accepted"] == false),
        "ordinary qualification failed; observations retained at {}",
        root.join("ordinary-decisions-windows-v1.json").display()
    );
}
