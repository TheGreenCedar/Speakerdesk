//! Shared offline scoring for the frozen ordinary Windows gate and Mac comparison.
use speakerdesk_models::redimnet::{MAX_SAMPLES, MIN_SAMPLES};

pub fn score(
    root: &std::path::Path,
    model: speakerdesk_core::voice::VoiceModel,
    embedding_receipt: &str,
    decision_receipt: &str,
    compute_policy: &str,
) -> serde_json::Value {
    use speakerdesk_core::voice::{
        Calibration, VoiceClip, VoiceProfile, make_profile, normalized, propose_match,
    };
    use std::{
        collections::{BTreeMap, BTreeSet},
        io::Write,
    };

    let output = root.join(decision_receipt);
    assert!(!output.exists(), "decisions must be a fresh file");
    let policy_bytes = std::fs::read(root.join("ordinary-policy.json")).unwrap();
    let policy_hash = speakerdesk_core::memo::sha256_hex(&policy_bytes);
    assert_eq!(
        policy_hash,
        "0c21f21805f2988297bfd128abb2935767308c875dc60442346e26455be9239d"
    );
    let frozen: serde_json::Value = serde_json::from_slice(&policy_bytes).unwrap();

    let manifest_bytes = std::fs::read(root.join("ordinary-cases.json")).unwrap();
    let manifest_hash = speakerdesk_core::memo::sha256_hex(&manifest_bytes);
    assert_eq!(
        manifest_hash,
        "85df77b04cfb1330a5859c70cd24d27a86466beddfd74b0d28fd97df8d57f635"
    );
    assert_eq!(frozen["manifest_sha256"], manifest_hash);
    let manifest: serde_json::Value = serde_json::from_slice(&manifest_bytes).unwrap();
    let cases = manifest["cases"].as_array().unwrap();
    let trials = manifest["trials"].as_array().unwrap();
    assert_eq!((cases.len(), trials.len()), (44, 16));
    for (path, hash) in manifest["source_pins"].as_object().unwrap() {
        // Relocated prior manifests are receipts, not inputs to this scorer.
        if path.starts_with("source-rows/") {
            assert_eq!(
                speakerdesk_core::memo::sha256_hex(&std::fs::read(root.join(path)).unwrap()),
                *hash
            );
        }
    }
    let known: BTreeSet<_> = manifest["known_speakers"]
        .as_array()
        .unwrap()
        .iter()
        .map(|s| s.as_str().unwrap().to_owned())
        .collect();
    let unknown: BTreeSet<_> = manifest["unknown_speakers"]
        .as_array()
        .unwrap()
        .iter()
        .map(|s| s.as_str().unwrap().to_owned())
        .collect();
    assert_eq!((known.len(), unknown.len()), (6, 4));
    assert!(known.is_disjoint(&unknown));
    let result_bytes = std::fs::read(root.join(embedding_receipt)).unwrap();
    let result: serde_json::Value = serde_json::from_slice(&result_bytes).unwrap();
    assert_eq!(result["manifest_sha256"], manifest_hash);
    assert_eq!(result["model"], serde_json::to_value(&model).unwrap());
    assert_eq!(result["policy"], compute_policy);
    let mut calls = BTreeMap::new();
    for row in result["calls"].as_array().unwrap() {
        assert!(
            calls.insert(row["id"].as_str().unwrap(), row).is_none(),
            "duplicate native result"
        );
    }
    assert_eq!(calls.len(), cases.len());
    let mut by_id = BTreeMap::new();
    let mut waveform_hashes = BTreeSet::new();
    let mut clean = true;
    for case in cases {
        let id = case["id"].as_str().unwrap();
        assert!(by_id.insert(id, case).is_none(), "duplicate manifest ID");
        assert!(
            waveform_hashes.insert(case["wav_sha256"].as_str().unwrap()),
            "reference/query audio reused"
        );
        let row = calls[id];
        for key in ["wav_sha256", "start_sample", "end_sample"] {
            assert_eq!(row[key], case[key], "native crop differs: {id} {key}");
        }
        assert_eq!(
            row["generated_query_headroom"], false,
            "ordinary PCM must stay unchanged"
        );
        let relative = std::path::Path::new(case["path"].as_str().unwrap());
        assert!(
            relative
                .components()
                .all(|c| matches!(c, std::path::Component::Normal(_)))
        );
        let audio = std::fs::read(root.join(relative)).unwrap();
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
            .map(|s| f32::from(s.unwrap()) / 32768.)
            .collect();
        let start = usize::try_from(case["start_sample"].as_u64().unwrap()).unwrap();
        let end = usize::try_from(case["end_sample"].as_u64().unwrap()).unwrap();
        assert!(start < end && end <= pcm.len());
        assert!((MIN_SAMPLES..=MAX_SAMPLES).contains(&(end - start)));
        let prepared: Vec<_> = pcm[start..end]
            .iter()
            .flat_map(|s| s.to_le_bytes())
            .collect();
        assert_eq!(
            row["prepared_pcm_sha256"],
            speakerdesk_core::memo::sha256_hex(&prepared)
        );
        let vector: Vec<f64> = serde_json::from_value(row["vector"].clone()).unwrap();
        if row["clean"].as_bool().unwrap() {
            normalized(&vector, model.dimension).unwrap();
        } else {
            assert!(vector.is_empty());
            clean = false;
        }
    }
    let calibration = Calibration::new(
        model.clone(),
        "librispeech-clean-ordinary-windows-v1:prospective",
        frozen["threshold"].as_f64().unwrap(),
        frozen["margin"].as_f64().unwrap(),
        12,
        4,
        0.,
        0.,
        frozen["minimum_clips"].as_u64().unwrap() as usize,
    )
    .unwrap(); // Rates are prospective gate metadata; observations are reported below.
    let mut profiles = Vec::new();
    for speaker in &known {
        let refs: Vec<_> = cases
            .iter()
            .filter(|c| {
                c["role"] == "reference" && c["speaker"].as_u64().unwrap().to_string() == *speaker
            })
            .collect();
        assert_eq!(refs.len(), 2);
        let mut vectors = Vec::new();
        let mut clips = Vec::new();
        for case in refs {
            let id = case["id"].as_str().unwrap();
            if calls[id]["clean"] == true {
                vectors.push(serde_json::from_value(calls[id]["vector"].clone()).unwrap());
            }
            clips.push(
                VoiceClip::try_new(
                    "ordinary-public-pilot",
                    speaker,
                    id,
                    case["start_sample"].as_u64().unwrap() as f64 / 16000.,
                    case["end_sample"].as_u64().unwrap() as f64 / 16000.,
                )
                .unwrap(),
            );
        }
        if vectors.len() == 2 {
            let draft = make_profile(&model, &vectors, &clips).unwrap();
            profiles.push(VoiceProfile {
                person_id: speaker.clone(),
                version: "ordinary-public-v1".into(),
                enrolled_at: 0.,
                model: draft.model,
                centroid: draft.centroid,
                clips: draft.clips,
                consent: draft.consent,
            });
        }
    }
    let mut used_queries = BTreeSet::new();
    let mut trial_ids = BTreeSet::new();
    let mut decisions = Vec::new();
    let mut genuine = 0;
    let mut negatives = 0;
    for trial in trials {
        assert!(trial_ids.insert(trial["id"].as_str().unwrap()));
        let speaker = trial["speaker"].as_u64().unwrap().to_string();
        let expected = trial["expected"].as_str();
        if known.contains(&speaker) {
            assert_eq!(expected, Some(speaker.as_str()));
            genuine += 1;
        } else {
            assert!(unknown.contains(&speaker) && expected.is_none());
            negatives += 1;
        }
        let ids = trial["cases"].as_array().unwrap();
        assert_eq!(ids.len(), 2);
        let mut vectors = Vec::new();
        let mut query_clean = true;
        for id in ids {
            let id = id.as_str().unwrap();
            assert!(used_queries.insert(id), "query reused in multiple trials");
            let case = by_id[id];
            assert_eq!(case["speaker"], trial["speaker"]);
            assert_eq!(
                case["role"],
                if expected.is_some() {
                    "ordinary_query"
                } else {
                    "unknown_query"
                }
            );
            if calls[id]["clean"] == true {
                vectors.push(serde_json::from_value(calls[id]["vector"].clone()).unwrap());
            } else {
                query_clean = false;
            }
        }
        let found = propose_match(&model, &calibration, &vectors, &profiles).unwrap();
        let matched = found.as_ref().map(|m| m.person_id.as_str());
        // Diagnostic ranks use the same production ranking function; propose_match owns the decision.
        let scores: Vec<_> = speakerdesk_core::voice::rank_matches(&model, &vectors, &profiles)
            .unwrap().into_iter().map(|(score, lowest, profile)|
                serde_json::json!({"speaker":profile.person_id,"score":score,"lowest":lowest})).collect();
        decisions.push(serde_json::json!({"id":trial["id"], "speaker":speaker,
            "query_ids":ids, "clean":query_clean, "expected":expected, "matched":matched, "scores":scores,
            "correct":query_clean && matched==expected, "wrong_accepted":matched.is_some() && matched!=expected,
            "score":found.as_ref().map(|m|m.score), "margin":found.as_ref().map(|m|m.margin)}));
    }
    assert_eq!((genuine, negatives, used_queries.len()), (12, 4, 32));
    let report = serde_json::json!({"policy_sha256":policy_hash, "policy":frozen,
        "manifest_sha256":manifest_hash, "model":model,
        "embedding_receipt_sha256":speakerdesk_core::memo::sha256_hex(&result_bytes),
        "all_inputs_clean":clean, "profiles_admitted":profiles.len(), "decisions":decisions,
        "compute_policy":compute_policy,
        "scope":"Same frozen ordinary pilot inputs and thresholds; policy retains its original Windows prospective identity, model identifies the actual reference/query space. Production policy unchanged."});
    std::fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(&output)
        .unwrap()
        .write_all(&serde_json::to_vec_pretty(&report).unwrap())
        .unwrap();
    report
}
