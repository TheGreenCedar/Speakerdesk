//! Supplemental current-core replay, not a Windows child or timing-accuracy oracle.
use super::*;
use std::{collections::BTreeSet, fs, io::Write, path::Path};

const SOURCE_RECEIPT: &str = "09bffd98402729b59a68d5924f62f328958c22aff4fab86119f830134a4f603b";
const VERIFIER: &str = "8a906b1fc3a967855c864cd1298e631037c0787b60a262775519d058deb8731c";
const NATIVE_RECEIPT: &str = "3f3ed8282c6f9586b9a24fa8a6bbe59b9ff27ff18e0106cc9d3e7018d066c3bf";
const WINDOWS_RUNTIME: &str = "6c916621c1807eba630388868bff8eb4cc9718b9fa17464fc9368eb14ad0f33e";
const DIRECTML: &str = "4a263e851f513f7c60a2b5bb08c3ed92102e3007f241c6ca5d63abf29bf04358";
const MHA_GRAPH: &str = "233ac8980dab9c142ef9fe91ab1d3b5d6309a9549f59a284e50199f7b1081699";

fn verified(path: &Path, digest: &str) -> Vec<u8> {
    let bytes = fs::read(path).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
    assert_eq!(sha256_hex(&bytes), digest, "{}", path.display());
    bytes
}

fn score_bytes(scores: &CtcScoresOut) -> Vec<u8> {
    scores
        .gathered
        .iter()
        .flat_map(|v| v.to_le_bytes())
        .collect()
}

// Confidence is diagnostic. Preserve every other field, including raw offsets,
// characters, frames, word sample bounds, status, reason, aliases and provenance.
fn behavior(value: &Value) -> Value {
    match value {
        Value::Object(fields) => Value::Object(
            fields
                .iter()
                .filter(|(key, _)| {
                    !matches!(
                        key.as_str(),
                        "log_probability" | "minimum_token_log_probability"
                    )
                })
                .map(|(key, value)| (key.clone(), behavior(value)))
                .collect(),
        ),
        Value::Array(values) => Value::Array(values.iter().map(behavior).collect()),
        _ => value.clone(),
    }
}

fn differences(source: &Value, native: &Value, path: &str, result: &mut Vec<Value>) {
    if source == native {
        return;
    }
    match (source, native) {
        (Value::Object(a), Value::Object(b)) => {
            for key in a.keys().chain(b.keys()).collect::<BTreeSet<_>>() {
                differences(&source[key], &native[key], &format!("{path}/{key}"), result);
            }
        }
        (Value::Array(a), Value::Array(b)) if a.len() == b.len() => {
            for (index, (a, b)) in a.iter().zip(b).enumerate() {
                differences(a, b, &format!("{path}/{index}"), result);
            }
        }
        _ => result.push(json!({"path":path,"source":source,"native":native})),
    }
}

#[test]
#[ignore = "requires frozen supplemental source and admitted native scores; no model calls"]
fn supplemental_native_scores_preserve_current_core_supplied_text_timing() {
    let source_dir = std::env::var_os("SPEAKERDESK_CTC_TIMING_SOURCE")
        .map(std::path::PathBuf::from)
        .expect("SPEAKERDESK_CTC_TIMING_SOURCE");
    let native_dir = std::env::var_os("SPEAKERDESK_CTC_TIMING_NATIVE")
        .map(std::path::PathBuf::from)
        .expect("SPEAKERDESK_CTC_TIMING_NATIVE");
    let report_path = std::env::var_os("SPEAKERDESK_CTC_TIMING_REPORT")
        .map(std::path::PathBuf::from)
        .expect("SPEAKERDESK_CTC_TIMING_REPORT (fresh path)");
    let receipt: Value = serde_json::from_slice(&verified(
        &source_dir.join("source-receipt.json"),
        SOURCE_RECEIPT,
    ))
    .unwrap();
    assert_eq!(receipt["original_10_columns_bitexact"], true);
    let read_source =
        |name: &str, key: &str| verified(&source_dir.join(name), receipt[key].as_str().unwrap());
    let source_json = read_source("source-scores.json", "source_scores_sha256");
    let source: CtcScoresOut = serde_json::from_slice(&source_json).unwrap();
    let source_raw = read_source("source-scores.f32", "source_score_f32_sha256");
    assert_eq!(score_bytes(&source), source_raw);
    let tokens = read_source("tokens.txt", "tokens_sha256");
    let vocabulary = Vocabulary::parse(std::str::from_utf8(&tokens).unwrap()).unwrap();
    read_source("case.json", "case_sha256");
    read_source("audio.f32", "normalized_input_sha256");
    read_source("capture.py", "capture_script_sha256");
    let ids = fs::read(source_dir.join("ids.i64")).unwrap();
    let expected_ids: Vec<u8> = source
        .token_ids
        .iter()
        .flat_map(|&id| i64::from(id).to_le_bytes())
        .collect();
    assert_eq!(ids, expected_ids);
    assert_eq!(json!(source.token_ids), receipt["token_ids"]);
    assert_eq!(source.token_ids.len(), 30);
    assert_eq!(source.valid_frames, 454);
    assert_eq!(source.audio_num_samples, 145445);
    assert_eq!(source.frame_hop_samples, 320);
    assert_eq!(source.frame_origin_samples, 0);
    assert_eq!(source.model, timing::MODEL);
    assert_eq!(source.vocabulary_sha256, TOKENS_SHA256);

    let independent = Path::new(env!("CARGO_MANIFEST_DIR")).join("../models/tests/data");
    let ctc: Value = serde_json::from_slice(&verified(
        &independent.join("ctc-reference.json"),
        receipt["source_reference_sha256"].as_str().unwrap(),
    ))
    .unwrap();
    assert_eq!(ctc["token_ids"], json!(&source.token_ids[..10]));
    let original = ctc["cases"]
        .as_array()
        .unwrap()
        .iter()
        .find(|case| case["name"] == "whole")
        .unwrap();
    for (frame, row) in original["gathered"].as_array().unwrap().iter().enumerate() {
        for (column, value) in row.as_array().unwrap().iter().enumerate() {
            assert_eq!(
                source.gathered[frame * 30 + column].to_bits(),
                (value.as_f64().unwrap() as f32).to_bits(),
                "original independent score frame {frame}, column {column}"
            );
        }
    }
    let cohere: Value = serde_json::from_slice(&verified(
        &independent.join("cohere-reference.json"),
        receipt["cohere_reference_sha256"].as_str().unwrap(),
    ))
    .unwrap();
    let text = receipt["text"].as_str().unwrap();
    assert_eq!(cohere["text"], text);
    assert_eq!(
        sha256_hex(text.as_bytes()),
        "c23b496aa31ce65941fef721230ea01c55992b5aa40edc8865d8f355278b739c"
    );
    let wav = verified(
        &independent.join("two-voice.wav"),
        receipt["waveform_sha256"].as_str().unwrap(),
    );
    let mut reader = hound::WavReader::new(std::io::Cursor::new(wav)).unwrap();
    assert_eq!(reader.spec().channels, 1);
    assert_eq!(reader.spec().sample_rate, 16000);
    assert_eq!(reader.spec().bits_per_sample, 16);
    assert_eq!(reader.spec().sample_format, hound::SampleFormat::Int);
    let pcm: Vec<f32> = reader
        .samples::<i16>()
        .map(|v| f32::from(v.unwrap()) / 32768.)
        .collect();
    assert_eq!(pcm.len(), source.audio_num_samples as usize);

    // This receipt is emitted only after shared verify_probe admits physical
    // graph/weights, actual runtime DLLs, exact inputs and execution profiles.
    let admission: Value = serde_json::from_slice(&verified(
        &native_dir.join("native-receipt.json"),
        NATIVE_RECEIPT,
    ))
    .unwrap();
    assert_eq!(admission["source_receipt_sha256"], SOURCE_RECEIPT);
    assert_eq!(admission["verification_script_sha256"], VERIFIER);
    assert_eq!(admission["repeat_bitexact"], true);
    assert_eq!(admission["case_sha256"], receipt["case_sha256"]);
    assert_eq!(admission["token_ids"], receipt["token_ids"]);
    assert_eq!(admission["provider"], "dml");
    assert_eq!(admission["runtime_sha256"], WINDOWS_RUNTIME);
    assert_eq!(admission["model_sha256"], MHA_GRAPH);
    for (name, pin) in [
        ("onnxruntime.dll", WINDOWS_RUNTIME),
        ("DirectML.dll", DIRECTML),
    ] {
        let loaded = admission["loaded_runtime_libraries"]
            .as_array()
            .unwrap()
            .iter()
            .find(|item| item["name"] == name)
            .unwrap();
        assert_eq!(loaded["sha256"], pin);
    }
    let native_json = verified(
        &native_dir.join("native-scores.json"),
        admission["native_scores_sha256"].as_str().unwrap(),
    );
    let native: CtcScoresOut = serde_json::from_slice(&native_json).unwrap();
    assert_eq!(
        score_bytes(&native),
        verified(
            &native_dir.join("native-scores.f32"),
            admission["native_score_f32_sha256"].as_str().unwrap()
        )
    );
    let metadata = |bytes: &[u8]| {
        let mut value: Value = serde_json::from_slice(bytes).unwrap();
        value.as_object_mut().unwrap().remove("gathered");
        value
    };
    assert_eq!(metadata(&source_json), metadata(&native_json));
    assert!(native.gathered.iter().all(|v| v.is_finite() && *v <= 0.));

    let replay = |scores| {
        let mut models = crate::import::tests::Scripted::new(Vec::new(), Vec::new());
        models.alignment = Some((text.into(), scores));
        vocabulary.align(&models, &pcm, text, 16000, "en").unwrap()
    };
    let source_alignment = replay(source.clone());
    let native_alignment = replay(native.clone());
    let source_behavior = behavior(&source_alignment);
    let native_behavior = behavior(&native_alignment);
    let mut changes = Vec::new();
    differences(&source_behavior, &native_behavior, "", &mut changes);
    let errors: Vec<f64> = source
        .gathered
        .iter()
        .zip(&native.gathered)
        .map(|(a, b)| (f64::from(*a) - f64::from(*b)).abs())
        .collect();
    let maximum = errors.iter().copied().fold(0., f64::max);
    let report = json!({
        "scope":"Current-core replay of source/native scores. No actual Windows child execution or independent timing-accuracy golden. Current core emits Mac-specific provenance tags; these are not Windows provenance. Behavioral equality does not waive the unchanged original 1e-4 score gate or qualify production.",
        "source_receipt_sha256":SOURCE_RECEIPT,"native_admission":admission,
        "origin_samples":16000,"authoritative_text":text,
        "source_alignment":source_alignment,"native_alignment":native_alignment,
        "behavior_differences":changes,
        "score_diagnostic":{"max_abs":maximum,"mean_abs":errors.iter().sum::<f64>() / errors.len() as f64,
            "original_strict_bound":1e-4,"passes_original_bound":maximum < 1e-4}
    });
    let rendered = serde_json::to_string_pretty(&report).unwrap();
    let mut output = fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(&report_path)
        .unwrap();
    writeln!(output, "{rendered}").unwrap();
    eprintln!("{rendered}\nReplay report: {}", report_path.display());
    for alignment in [&source_alignment, &native_alignment] {
        assert_eq!(
            alignment["alignment_backend"],
            "speakerdesk-rust-ctc-segmentation-1.7.4"
        );
        assert!(!alignment["characters"].as_array().unwrap().is_empty());
    }
    assert_eq!(source_alignment["raw_text"], text);
    assert_eq!(native_alignment["raw_text"], text);
    assert_eq!(
        source_alignment["audio_anchor"],
        json!({"start_sample":16000,"end_sample":161445})
    );
    assert_eq!(source_behavior, native_behavior, "see full replay report");
}
