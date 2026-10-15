//! Frozen cross-backend behavior replay, not an acoustic timing-accuracy oracle.
use super::*;
use std::{collections::BTreeSet, fs, io::Write, path::Path};

const PROTOCOL: &[u8] =
    include_bytes!("../../../../experiments/windows/ctc/windows-qualification-v1.json");
const CAPTURE: &[u8] = include_bytes!("../../../models/tests/ctc_windows_qualification.rs");
const NEMOTRON: &str = "399bf3b851aca8d0062e60519ba9e8da8762138cb77a6139acd5ce3714a79c1b";

fn verified(path: &Path, expected: &str) -> Vec<u8> {
    let bytes = fs::read(path).unwrap_or_else(|error| panic!("{}: {error}", path.display()));
    assert_eq!(sha256_hex(&bytes), expected, "{}", path.display());
    bytes
}

fn raw_scores(scores: &CtcScoresOut) -> Vec<u8> {
    scores
        .gathered
        .iter()
        .flat_map(|value| value.to_le_bytes())
        .collect()
}

fn independent_scores(reference: &Value, scores: &CtcScoresOut) -> Value {
    let expected = reference["gathered"].as_array().unwrap();
    assert_eq!(expected.len(), scores.valid_frames as usize);
    assert_eq!(reference["samples"], scores.audio_num_samples);
    let mut maximum = 0_f64;
    let mut total = 0.;
    let mut count = 0;
    for (frame, row) in expected.iter().enumerate() {
        let row = row.as_array().unwrap();
        assert_eq!(row.len(), 10);
        for (column, value) in row.iter().enumerate() {
            let expected = value.as_f64().unwrap() as f32;
            let actual = scores.gathered[frame * scores.token_ids.len() + column];
            let difference = (f64::from(actual) - f64::from(expected)).abs();
            maximum = maximum.max(difference);
            total += difference;
            count += 1;
        }
    }
    json!({"max_abs":maximum,"mean_abs":total/count as f64,"bound":1e-4,"passes_original_bound":maximum < 1e-4,"columns":10})
}

fn capture_receipt(directory: &Path, expected: &str, manifest: &str, model: &str) -> Value {
    let receipt: Value =
        serde_json::from_slice(&verified(&directory.join("receipt.json"), expected)).unwrap();
    assert_eq!(receipt["manifest_sha256"], manifest);
    assert_eq!(receipt["protocol_sha256"], sha256_hex(PROTOCOL));
    assert_eq!(receipt["capture_test_sha256"], sha256_hex(CAPTURE));
    assert_eq!(receipt["model"], model);
    for key in [
        "passed",
        "repeat_bitexact",
        "invalid_requests_rejected",
        "pre_cancel_isolated",
    ] {
        assert_eq!(receipt[key], true, "{key}");
    }
    assert_eq!(receipt["calls"].as_array().unwrap().len(), 38);
    receipt
}

fn scores(
    directory: &Path,
    receipt: &Value,
    case: &Value,
    index: usize,
    model: &str,
) -> (CtcScoresOut, String) {
    let id = case["id"].as_str().unwrap();
    assert!(
        id.chars()
            .all(|character| character.is_ascii_alphanumeric() || matches!(character, '-' | '_'))
    );
    let bytes = fs::read(directory.join(format!("{id}.json"))).unwrap();
    let scores: CtcScoresOut = serde_json::from_slice(&bytes).unwrap();
    let samples = case["samples"].as_u64().unwrap() as usize;
    assert_eq!(scores.audio_num_samples as usize, samples);
    assert_eq!(scores.valid_frames as usize, (samples - 400) / 320 + 1);
    assert_eq!(
        (scores.frame_hop_samples, scores.frame_origin_samples),
        (320, 0)
    );
    assert_eq!(scores.model, model);
    assert_eq!(scores.vocabulary_sha256, TOKENS_SHA256);
    assert_eq!(json!(scores.token_ids), case["token_ids"]);
    assert_eq!(
        scores.token_ids.iter().collect::<BTreeSet<_>>().len(),
        scores.token_ids.len()
    );
    assert!(scores.token_ids.contains(&0));
    assert!(scores.token_ids.iter().all(|&id| (0..9812).contains(&id)));
    assert_eq!(
        scores.gathered.len(),
        scores.valid_frames as usize * scores.token_ids.len()
    );
    assert!(
        scores
            .gathered
            .iter()
            .all(|value| value.is_finite() && *value <= 0.)
    );
    assert!(
        scores
            .gathered
            .chunks_exact(scores.token_ids.len())
            .all(
                |row| row.iter().map(|&value| f64::from(value).exp()).sum::<f64>()
                    <= 0.001_f64.exp()
            )
    );
    let raw_hash = sha256_hex(&raw_scores(&scores));
    for repeat in 0..2 {
        let call = &receipt["calls"][index + repeat * 19];
        assert_eq!(call["id"], id);
        assert_eq!(call["repeat"], repeat);
        assert_eq!(call["score_sha256"], raw_hash);
        verified(
            &directory.join(format!("{id}-repeat-{repeat}.f32")),
            &raw_hash,
        );
    }
    (scores, sha256_hex(&bytes))
}

fn cell_word_consistency(alignment: &Value, failures: &mut Vec<String>, label: &str) {
    let characters = alignment["characters"].as_array().unwrap();
    for word in alignment["words"].as_array().unwrap() {
        if word["status"] != "aligned" {
            if !word["start_sample"].is_null() || !word["end_sample"].is_null() {
                failures.push(format!("{label}: unresolved word has sample bounds"));
            }
            continue;
        }
        let cells: Vec<_> = characters
            .iter()
            .filter(|cell| {
                cell["independent_acoustic_timestamp"] != false
                    && word["start_char"].as_u64().unwrap() <= cell["start_char"].as_u64().unwrap()
                    && cell["end_char"].as_u64().unwrap() <= word["end_char"].as_u64().unwrap()
            })
            .collect();
        if cells.is_empty()
            || cells.iter().any(|cell| cell["status"] != "aligned")
            || cells.first().map(|cell| &cell["start_sample"]) != Some(&word["start_sample"])
            || cells.last().map(|cell| &cell["end_sample"]) != Some(&word["end_sample"])
        {
            failures.push(format!(
                "{label}: word envelope differs from its accepted cells: {}",
                word["text"]
            ));
        }
    }
}

fn projection(alignment: &Value, activity: Value) -> Value {
    let anchor = &alignment["audio_anchor"];
    let mut segment: crate::document::Segment = serde_json::from_value(json!({
        "id":"frozen-replay","start":anchor["start_sample"].as_i64().unwrap() as f64 / 16000.,
        "end":anchor["end_sample"].as_i64().unwrap() as f64 / 16000.,"speaker":"unassigned",
        "text":alignment["raw_text"],"language":alignment["language"],
        "machine_revision":2,"canonical_machine_revision":2,"audio_revision":3,"language_epoch":1,
        "capture_source":{"source_id":"system","source_revision":1,"catalog_sha256":"frozen-ctc-replay"},
        "speaker_activity":{"audio_revision":3,"observed_end_sample":anchor["end_sample"],"regions":activity}
    })).unwrap();
    attach(&mut segment, alignment.clone());
    segment.reading_turns.unwrap_or_else(|| json!([]))
}

fn activities(source: &Value) -> Vec<(String, Value)> {
    let start = source["audio_anchor"]["start_sample"].as_i64().unwrap();
    let end = source["audio_anchor"]["end_sample"].as_i64().unwrap();
    let middle = start + (end - start) / 2;
    let region =
        |a, b, speakers: &[&str]| json!({"start_sample":a,"end_sample":b,"speakers":speakers});
    let mut result = vec![("single".into(), json!([region(start, end, &["speaker_0"])]))];
    let left = start.max(middle - 4000);
    let right = end.min(middle + 4000);
    let mut overlap = vec![];
    if start < left {
        overlap.push(region(start, left, &["speaker_0"]));
    }
    overlap.push(region(left, right, &["speaker_0", "speaker_1"]));
    if right < end {
        overlap.push(region(right, end, &["speaker_1"]));
    }
    result.push(("midpoint-overlap".into(), json!(overlap)));
    let mut transitions = BTreeSet::from([middle]);
    for word in source["words"].as_array().unwrap() {
        for key in ["start_sample", "end_sample"] {
            if let Some(boundary) = word[key].as_i64() {
                for offset in [-4000, -320, 0, 320, 4000] {
                    let boundary = boundary + offset;
                    if start < boundary && boundary < end {
                        transitions.insert(boundary);
                    }
                }
            }
        }
    }
    for boundary in transitions {
        result.push((
            format!("transition-{boundary}"),
            json!([
                region(start, boundary, &["speaker_0"]),
                region(boundary, end, &["speaker_1"])
            ]),
        ));
    }
    result
}

#[test]
#[ignore = "requires sealed 19-case native Mac/Windows captures and fresh report; no inference"]
fn native_windows_scores_preserve_frozen_supplied_text_behavior() {
    let env_path = |name| std::path::PathBuf::from(std::env::var_os(name).expect(name));
    let cases_dir = env_path("SPEAKERDESK_CTC_QUALIFICATION");
    let mac_dir = env_path("SPEAKERDESK_CTC_MAC_CAPTURE");
    let windows_dir = env_path("SPEAKERDESK_CTC_WINDOWS_CAPTURE");
    let manifest_hash = std::env::var("SPEAKERDESK_CTC_CASES_SHA256").unwrap();
    let manifest: Value =
        serde_json::from_slice(&verified(&cases_dir.join("cases.json"), &manifest_hash)).unwrap();
    let protocol: Value = serde_json::from_slice(PROTOCOL).unwrap();
    let independent = Path::new(env!("CARGO_MANIFEST_DIR")).join("../models/tests/data");
    let reference: Value = serde_json::from_slice(&verified(
        &independent.join("ctc-reference.json"),
        protocol["original_reference_sha256"].as_str().unwrap(),
    ))
    .unwrap();
    let diarization: Value = serde_json::from_slice(&verified(
        &independent.join("nemotron-reference.json"),
        NEMOTRON,
    ))
    .unwrap();
    assert_eq!(diarization["wav_sha256"], reference["wav_sha256"]);
    let wav = verified(
        &independent.join("two-voice.wav"),
        reference["wav_sha256"].as_str().unwrap(),
    );
    let mut reader = hound::WavReader::new(std::io::Cursor::new(wav)).unwrap();
    assert_eq!(
        (
            reader.spec().channels,
            reader.spec().sample_rate,
            reader.spec().bits_per_sample
        ),
        (1, 16000, 16)
    );
    let independent_pcm: Vec<_> = reader
        .samples::<i16>()
        .map(|sample| f32::from(sample.unwrap()) / 32768.)
        .collect();
    assert_eq!(
        independent_pcm.len(),
        diarization["samples"].as_u64().unwrap() as usize
    );
    let mut turns: Vec<crate::ipc::Turn> =
        serde_json::from_value(diarization["turns"].clone()).unwrap();
    for turn in &mut turns {
        turn.start += 1.;
        turn.end += 1.;
    }
    assert_eq!(manifest["protocol_sha256"], sha256_hex(PROTOCOL));
    assert_eq!(
        manifest["preparation_sha256"],
        sha256_hex(include_bytes!(
            "../../../../experiments/windows/ctc/prepare_windows_qualification.py"
        ))
    );
    assert_eq!(MAC.model_sha256, protocol["mac_model_sha256"]);
    assert_eq!(WINDOWS.model_sha256, protocol["graph_sha256"]);
    let tokens = verified(&cases_dir.join("tokens.txt"), TOKENS_SHA256);
    let vocabulary = Vocabulary::parse(std::str::from_utf8(&tokens).unwrap()).unwrap();
    let mac_hash = std::env::var("SPEAKERDESK_CTC_MAC_RECEIPT_SHA256").unwrap();
    let windows_hash = std::env::var("SPEAKERDESK_CTC_WINDOWS_RECEIPT_SHA256").unwrap();
    let mac_receipt = capture_receipt(&mac_dir, &mac_hash, &manifest_hash, MAC.model_sha256);
    let windows_receipt = capture_receipt(
        &windows_dir,
        &windows_hash,
        &manifest_hash,
        WINDOWS.model_sha256,
    );
    assert_eq!(
        mac_receipt["runtime"]["libmlx.dylib"],
        protocol["mlx_sha256"]
    );
    assert_eq!(
        mac_receipt["runtime"]["mlx.metallib"],
        "2d2e729831a995a39778164c93ea1b2e8d87f7d8b1e717386d3d816f46f416b2"
    );
    assert_eq!(windows_receipt["runtime"]["provider"], "DirectML");
    assert_eq!(windows_receipt["runtime"]["private_runtime_admitted"], true);
    assert_eq!(
        windows_receipt["runtime"]["onnxruntime.dll"],
        protocol["onnxruntime_sha256"]
    );
    assert_eq!(
        windows_receipt["runtime"]["DirectML.dll"],
        protocol["directml_sha256"]
    );
    let cases = manifest["cases"].as_array().unwrap();
    assert_eq!(cases.len(), 19);
    let mut seen = BTreeSet::new();
    let mut reports = vec![];
    let mut failures = vec![];
    let mut output = fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(env_path("SPEAKERDESK_CTC_CORE_REPORT"))
        .expect("fresh core replay report path");
    for (index, case) in cases.iter().enumerate() {
        let id = case["id"].as_str().unwrap();
        assert!(seen.insert(id));
        let relative = Path::new(case["path"].as_str().unwrap());
        assert!(
            relative
                .components()
                .all(|part| matches!(part, std::path::Component::Normal(_)))
        );
        let bytes = verified(
            &cases_dir.join(relative),
            case["pcm_sha256"].as_str().unwrap(),
        );
        assert_eq!(bytes.len(), case["samples"].as_u64().unwrap() as usize * 4);
        let pcm: Vec<_> = bytes
            .chunks_exact(4)
            .map(|bytes| f32::from_le_bytes(bytes.try_into().unwrap()))
            .collect();
        assert!(pcm.iter().all(|value| value.is_finite()));
        let text = case["text"].as_str().unwrap();
        assert_eq!(sha256_hex(text.as_bytes()), case["text_sha256"]);
        let (mac, mac_file) = scores(&mac_dir, &mac_receipt, case, index, MAC.model_sha256);
        let (windows, windows_file) = scores(
            &windows_dir,
            &windows_receipt,
            case,
            index,
            WINDOWS.model_sha256,
        );
        let scalar_reference = if let Some(name) = id.strip_prefix("original-") {
            assert_eq!(json!(&mac.token_ids[..10]), reference["token_ids"]);
            assert_eq!(json!(&windows.token_ids[..10]), reference["token_ids"]);
            assert_eq!(case["source"]["wav_sha256"], reference["wav_sha256"]);
            let case = reference["cases"]
                .as_array()
                .unwrap()
                .iter()
                .find(|case| case["name"] == name)
                .unwrap();
            json!({"mac":independent_scores(case,&mac),"windows":independent_scores(case,&windows)})
        } else {
            Value::Null
        };
        let errors: Vec<_> = mac
            .gathered
            .iter()
            .zip(&windows.gathered)
            .map(|(&a, &b)| (f64::from(a) - f64::from(b)).abs())
            .collect();
        let align = |scores| {
            let mut models = crate::import::tests::Scripted::new(vec![], vec![]);
            models.alignment = Some((text.into(), scores));
            vocabulary
                .align(
                    &models,
                    &pcm,
                    text,
                    16000,
                    case["language"].as_str().unwrap(),
                )
                .unwrap()
        };
        let source = align(mac);
        let native = align(windows);
        assert_eq!(descriptor_for_alignment(&source), Some(&MAC));
        assert_eq!(descriptor_for_alignment(&native), Some(&WINDOWS));
        assert_eq!(native["timing_accuracy_calibrated_languages"], json!([]));
        assert_eq!(native["score_calibration_id"], WINDOWS.functional_policy_id);
        let mut case_failures = vec![];
        if id == "original-whole" {
            assert_eq!(pcm, independent_pcm);
            for (name, alignment) in [("Mac", &source), ("Windows", &native)] {
                if alignment["status"] != "aligned"
                    || alignment["complete"] != true
                    || alignment["characters"].as_array().unwrap().is_empty()
                    || alignment["words"].as_array().unwrap().is_empty()
                {
                    case_failures.push(format!(
                        "{name}: original whole must contain complete aligned cells and words"
                    ));
                }
            }
        }
        for key in [
            "raw_text",
            "text_sha256",
            "audio_anchor",
            "offset_unit",
            "timing_kind",
            "language",
            "normalization_policy_id",
            "timing_unit_policy_id",
            "upstream_normalization_revision",
            "normalization_spans",
            "unsupported",
            "acoustic_aliases",
            "status",
            "complete",
            "reason",
            "non_speech_proof",
            "linguistic_word_claim",
            "alignment_backend",
        ] {
            if source[key] != native[key] {
                case_failures.push(format!("alignment/{key}"));
            }
        }
        let budget = if id == "original-whole" { 0 } else { 320 };
        let mut maximum_samples = 0;
        for group in ["characters", "words"] {
            let a = source[group].as_array().unwrap();
            let b = native[group].as_array().unwrap();
            if a.len() != b.len() {
                case_failures.push(format!("{group}/inventory_length"));
            }
            for (index, (a, b)) in a.iter().zip(b).enumerate() {
                for key in [
                    "text",
                    "raw_text",
                    "token_id",
                    "start_char",
                    "end_char",
                    "start_utf16",
                    "end_utf16",
                    "unit_kind",
                    "status",
                    "independent_acoustic_timestamp",
                    "acoustic_target_status",
                    "acoustic_aliases",
                ] {
                    if a[key] != b[key] {
                        case_failures.push(format!("{group}/{index}/{key}"));
                    }
                }
                for key in [
                    "frame",
                    "start_sample",
                    "end_sample",
                    "target_cell_start_sample",
                    "target_cell_end_sample",
                ] {
                    match (a[key].as_i64(), b[key].as_i64()) {
                        (Some(a), Some(b)) => {
                            let distance = (a - b).abs() * if key == "frame" { 320 } else { 1 };
                            maximum_samples = maximum_samples.max(distance);
                            if distance > budget {
                                case_failures
                                    .push(format!("{group}/{index}/{key}: {distance}>{budget}"));
                            }
                        }
                        _ if a[key] != b[key] => {
                            case_failures.push(format!("{group}/{index}/{key}: presence"))
                        }
                        _ => {}
                    }
                }
            }
        }
        cell_word_consistency(&source, &mut case_failures, "Mac");
        cell_word_consistency(&native, &mut case_failures, "Windows");
        let mut projections = vec![];
        let mut frozen_activities = activities(&source);
        if id == "original-whole" {
            let end = 16000 + independent_pcm.len() as i64;
            let regions = crate::live::activity_regions(&turns, 16000, end);
            frozen_activities.push(("independent-two-voice".into(), json!(regions.iter().map(|region| json!({
                "start_sample":region.start_sample,"end_sample":region.end_sample,"speakers":region.speakers
            })).collect::<Vec<_>>())));
        }
        for (name, activity) in frozen_activities {
            let a = projection(&source, activity.clone());
            let b = projection(&native, activity.clone());
            let a_turns = a.as_array().unwrap();
            let b_turns = b.as_array().unwrap();
            if id == "original-whole" {
                if a_turns.is_empty() || b_turns.is_empty() {
                    case_failures.push(format!(
                        "{name}: original whole projection must be nonempty"
                    ));
                }
                for (label, turns) in [("Mac", a_turns), ("Windows", b_turns)] {
                    let owned: BTreeSet<_> = turns
                        .iter()
                        .filter(|turn| {
                            turn["attribution"] == "single"
                                && turn["start"].as_f64().is_some()
                                && turn["end"].as_f64().is_some()
                        })
                        .filter_map(|turn| turn["speaker"].as_str())
                        .collect();
                    if name == "single" && !owned.contains("speaker_0") {
                        case_failures.push(format!("{name}/{label}: missing owned timed span"));
                    }
                    if name == "independent-two-voice"
                        && owned != BTreeSet::from(["speaker_0", "speaker_1"])
                    {
                        case_failures.push(format!("{name}/{label}: expected both independently observed speakers, got {owned:?}"));
                    }
                }
            }
            if a_turns.len() != b_turns.len() {
                case_failures.push(format!("{name}: turn count"));
            }
            for (index, (a, b)) in a_turns.iter().zip(b_turns).enumerate() {
                for key in [
                    "text",
                    "start_offset",
                    "end_offset",
                    "speaker",
                    "attribution",
                ] {
                    if a[key] != b[key] {
                        case_failures.push(format!("{name}/{index}/{key}"));
                    }
                }
                for key in ["start", "end"] {
                    match (a[key].as_f64(), b[key].as_f64()) {
                        (Some(a), Some(b)) if ((a - b) * 16000.).abs().round() as i64 > budget => {
                            case_failures.push(format!("{name}/{index}/{key}: bound"))
                        }
                        (None, None) => {}
                        (Some(_), Some(_)) => {}
                        _ => case_failures.push(format!("{name}/{index}/{key}: presence")),
                    }
                }
            }
            projections.push(json!({"activity":name,"regions":activity,"mac":a,"windows":b}));
        }
        for failure in &case_failures {
            failures.push(format!("{id}/{failure}"));
        }
        eprintln!(
            "{id}: {} failures, maximum boundary difference {maximum_samples} samples",
            case_failures.len()
        );
        reports.push(json!({"id":id,"input_sha256":case["pcm_sha256"],"mac_scores_json_sha256":mac_file,"windows_scores_json_sha256":windows_file,
            "cross_backend_score_diagnostic":{"max_abs":errors.iter().copied().fold(0.,f64::max),"mean_abs":errors.iter().sum::<f64>()/errors.len() as f64},
            "independent_ten_column_scalar_diagnostic":scalar_reference,
            "boundary_budget_samples":budget,"maximum_boundary_difference_samples":maximum_samples,"mac_alignment":source,"windows_alignment":native,"projections":projections,"failures":case_failures}));
    }
    let report = json!({"protocol_sha256":sha256_hex(PROTOCOL),"manifest_sha256":manifest_hash,
        "mac_receipt_sha256":mac_hash,"windows_receipt_sha256":windows_hash,"core_replay_sha256":sha256_hex(include_bytes!("windows_qualification.rs")),
        "original_reference_sha256":protocol["original_reference_sha256"],"independent_diarization_reference_sha256":NEMOTRON,
        "limit":"Actual native score captures replayed through current core. This is cross-backend compatibility, not a Windows child/product run, independently labelled word-boundary accuracy or a replacement for the failed 1e-4 scalar gate.",
        "cases":reports,"failures":failures});
    writeln!(output, "{}", serde_json::to_string_pretty(&report).unwrap()).unwrap();
    output.sync_all().unwrap();
    assert!(
        failures.is_empty(),
        "{} behavior differences; see fresh core report",
        failures.len()
    );
}
