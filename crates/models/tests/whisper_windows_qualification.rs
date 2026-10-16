//! Prospective Windows contract. The original independent scalar test is unchanged.
#![cfg(any(windows, target_os = "macos"))]

use serde_json::{Value, json};
use speakerdesk_core::{
    error::ChildError,
    ipc::LanguageProbeOut,
    language::{LanguagePolicy, is_confident_detection},
    memo::sha256_hex,
};
use speakerdesk_models::{operation::Operation, whisper_lid::network::Network};
use std::{
    fs,
    io::{Seek, SeekFrom, Write},
    path::Path,
    time::Instant,
};

#[path = "support/qualification_runtime.rs"]
mod qualification_runtime;

const CONTRACT: &str =
    include_str!("../../../experiments/windows/whisper/windows-qualification-v2.json");

fn digest(path: &Path) -> Result<String, String> {
    fs::read(path)
        .map(|bytes| sha256_hex(&bytes))
        .map_err(|e| e.to_string())
}

fn env_path(name: &str) -> Result<std::path::PathBuf, String> {
    std::env::var_os(name)
        .map(Into::into)
        .ok_or_else(|| format!("Missing {name}"))
}

fn core(
    policy: &mut LanguagePolicy,
    codes: &[String],
    probabilities: &[f32],
    speaker: &str,
) -> Value {
    let speakers = vec![speaker.to_owned()];
    let scores = codes
        .iter()
        .cloned()
        .zip(probabilities.iter().map(|p| f64::from(*p)))
        .collect::<Vec<_>>();
    let decision = policy.decide(&scores, &speakers).unwrap();
    json!({"winner":decision.detection.candidates[0].language,"language":decision.language,
        "review":decision.review,"reason":decision.detection.reason.as_str(),
        "remembered":policy.remembered(&speakers)})
}

fn pair(winner: &str, probability: f32, runner: &str) -> (Vec<String>, Vec<f32>) {
    (
        vec![winner.into(), runner.into()],
        vec![probability, 1. - probability],
    )
}

fn policy_sequences() -> Value {
    let mut policy = LanguagePolicy::new();
    let mut rows = Vec::new();
    for (winner, probability, runner, expected) in [
        (
            "en",
            0.95,
            "fr",
            json!({"winner":"en","language":"en","review":false,"reason":"detected","remembered":"en"}),
        ),
        (
            "fr",
            0.60,
            "en",
            json!({"winner":"fr","language":"fr","review":true,"reason":"best_effort","remembered":"en"}),
        ),
        (
            "ru",
            0.95,
            "en",
            json!({"winner":"ru","language":null,"review":true,"reason":"unsupported","remembered":null}),
        ),
        (
            "hi",
            0.55,
            "fr",
            json!({"winner":"hi","language":null,"review":true,"reason":"needs_language","remembered":null}),
        ),
    ] {
        let (codes, scores) = pair(winner, probability, runner);
        let observed = core(&mut policy, &codes, &scores, "speaker-a");
        assert_eq!(observed, expected);
        rows.push(observed);
    }
    let (codes, scores) = pair("en", 0.95, "fr");
    core(&mut policy, &codes, &scores, "speaker-a");
    let (codes, scores) = pair("fr", 0.60, "en");
    let isolated = core(&mut policy, &codes, &scores, "speaker-b");
    assert!(isolated["remembered"].is_null());
    assert_eq!(policy.remembered(&["speaker-a".into()]), Some("en"));
    rows.push(isolated);
    let below = 0.9_f32;
    let above = f32::from_bits(below.to_bits() + 1);
    assert!(f64::from(below) < 0.90 && f64::from(above) > 0.90);
    for (probability, review) in [(below, true), (above, false), (0.5, true)] {
        let (codes, scores) = pair("en", probability, "fr");
        let observed = core(&mut LanguagePolicy::new(), &codes, &scores, "fresh");
        assert_eq!(observed["review"], review);
        assert_eq!(observed["winner"], "en");
        rows.push(observed);
    }
    // With normalized scores and p>=0.90, the 0.20 margin guard is already met.
    // Values around the margin alone remain best effort below the probability guard.
    for probability in [0.6_f32, f32::from_bits(0.6_f32.to_bits() - 1)] {
        let (codes, scores) = pair("en", probability, "fr");
        let observed = core(&mut LanguagePolicy::new(), &codes, &scores, "fresh");
        assert_eq!(observed["reason"], "best_effort");
        rows.push(observed);
    }
    json!(rows)
}

#[test]
fn actual_core_policy_preserves_thresholds_unsupported_context_and_ties() {
    policy_sequences();
}

fn checked_output(output: &LanguageProbeOut, codes: &Value) -> Result<(), String> {
    if output.codes.len() != 99
        || serde_json::to_value(&output.codes).unwrap() != *codes
        || output.probabilities.len() != 99
        || output
            .probabilities
            .iter()
            .any(|p| !p.is_finite() || !(0.0..=1.0).contains(p))
        || (output
            .probabilities
            .iter()
            .map(|p| f64::from(*p))
            .sum::<f64>()
            - 1.)
            .abs()
            > 1e-5
    {
        return Err("Invalid native 99-language result".into());
    }
    Ok(())
}

fn score_hash(output: &LanguageProbeOut) -> String {
    sha256_hex(
        &output
            .probabilities
            .iter()
            .flat_map(|p| p.to_le_bytes())
            .collect::<Vec<_>>(),
    )
}

fn read_audio(directory: &Path, case: &Value) -> Result<Vec<f32>, String> {
    let name = case["file"].as_str().ok_or("Missing PCM filename")?;
    if Path::new(name).components().count() != 1 || name.contains(['/', '\\']) {
        return Err("PCM must be a case-local file".into());
    }
    let bytes = fs::read(directory.join(name)).map_err(|e| e.to_string())?;
    if sha256_hex(&bytes) != case["sha256"]
        || bytes.len() != case["samples"].as_u64().ok_or("Missing sample count")? as usize * 4
    {
        return Err("Frozen PCM identity differs".into());
    }
    let audio = bytes
        .chunks_exact(4)
        .map(|bytes| f32::from_le_bytes(bytes.try_into().unwrap()))
        .collect::<Vec<_>>();
    if audio.is_empty() || audio.iter().any(|value| !value.is_finite()) {
        return Err("Frozen PCM is empty or nonfinite".into());
    }
    Ok(audio)
}

#[cfg(windows)]
fn loaded_runtime() -> Result<Value, String> {
    use windows_sys::Win32::System::LibraryLoader::{GetModuleFileNameW, GetModuleHandleW};
    let mut hashes = serde_json::Map::new();
    for name in ["onnxruntime.dll", "DirectML.dll"] {
        let wide = name.encode_utf16().chain(Some(0)).collect::<Vec<_>>();
        let module = unsafe { GetModuleHandleW(wide.as_ptr()) };
        if module.is_null() {
            return Err(format!("Required runtime is not loaded: {name}"));
        }
        let mut path = vec![0; 32768];
        let length =
            unsafe { GetModuleFileNameW(module, path.as_mut_ptr(), path.len() as u32) } as usize;
        if length == 0 || length >= path.len() {
            return Err("Runtime module path unavailable".into());
        }
        let path = std::path::PathBuf::from(
            String::from_utf16(&path[..length]).map_err(|e| e.to_string())?,
        );
        hashes.insert(name.into(), json!(digest(&path)?));
    }
    Ok(Value::Object(hashes))
}

fn persist(file: &mut fs::File, report: &Value) -> Result<(), String> {
    file.set_len(0).map_err(|e| e.to_string())?;
    file.seek(SeekFrom::Start(0)).map_err(|e| e.to_string())?;
    file.write_all(
        serde_json::to_string_pretty(report)
            .map_err(|e| e.to_string())?
            .as_bytes(),
    )
    .map_err(|e| e.to_string())?;
    file.write_all(b"\n").map_err(|e| e.to_string())?;
    file.sync_all().map_err(|e| e.to_string())
}

fn returned(output: &LanguageProbeOut) -> Value {
    json!({"codes":output.codes,"probabilities":output.probabilities,
        "probability_bits":output.probabilities.iter().map(|p|p.to_bits()).collect::<Vec<_>>(),
        "scores_sha256":score_hash(output)})
}

fn capture(file: &mut fs::File, report: &mut Value) -> Result<(), String> {
    let contract: Value = serde_json::from_str(CONTRACT).unwrap();
    let cases_path = env_path("SPEAKERDESK_WHISPER_CASES")?;
    let directory = cases_path.parent().ok_or("Missing case directory")?;
    let cases: Value = serde_json::from_slice(&fs::read(&cases_path).map_err(|e| e.to_string())?)
        .map_err(|e| e.to_string())?;
    if cases["contract_sha256"] != sha256_hex(CONTRACT.as_bytes())
        || cases["cases"].as_array().ok_or("Missing cases")?.len() != 56
    {
        return Err("Qualification contract/case count differs".into());
    }
    report["cases_sha256"] = json!(digest(&cases_path)?);
    report["policy_sequences"] = policy_sequences();
    #[cfg(windows)]
    {
        let reference = env_path("SPEAKERDESK_WHISPER_MAC_REFERENCE")?;
        let mac: Value = serde_json::from_slice(&fs::read(&reference).map_err(|e| e.to_string())?)
            .map_err(|e| e.to_string())?;
        if mac["platform"] != "macos"
            || mac["passed_capture"] != true
            || mac["cases_sha256"] != report["cases_sha256"]
            || mac["contract_sha256"] != report["contract_sha256"]
        {
            return Err("Mac capture must be frozen before Windows inference".into());
        }
        report["mac_reference_sha256"] = json!(digest(&reference)?);
    }
    let model = env_path("SPEAKERDESK_TEST_WHISPER")?;
    #[cfg(target_os = "macos")]
    {
        report["runtime"] = qualification_runtime::prepare_mlx();
        let mut identities = serde_json::Map::new();
        for (name, expected) in contract["mac_models"].as_object().unwrap() {
            let actual = digest(&model.join(name))?;
            if actual != expected.as_str().unwrap() {
                return Err(format!("Mac artifact pin differs: {name}"));
            }
            identities.insert(name.clone(), json!(actual));
        }
        report["models"] = Value::Object(identities);
    }
    persist(file, report)?;
    #[cfg(windows)]
    let backend = speakerdesk_models::windows_runtime::Backend::from_directory(&env_path(
        "SPEAKERDESK_TEST_WINDOWS_RUNTIME",
    )?)
    .map_err(|e| e.to_string())?;
    #[cfg(windows)]
    let mut network = Network::load(&model, &backend).map_err(|e| e.to_string())?;
    #[cfg(target_os = "macos")]
    let mut network = Network::load(&model).map_err(|e| e.to_string())?;
    #[cfg(windows)]
    {
        let mut identities = serde_json::Map::new();
        for (name, pin) in contract["windows_models"].as_object().unwrap() {
            let actual = digest(&model.join(name))?;
            if actual != pin["sha256"].as_str().unwrap() {
                return Err(format!("Windows graph pin differs: {name}"));
            }
            identities.insert(name.clone(), json!(actual));
        }
        report["models"] = Value::Object(identities);
        report["runtime"] = loaded_runtime()?;
    }
    persist(file, report)?;
    let mut first = std::collections::BTreeMap::new();
    for run in 0..2 {
        for case in cases["cases"].as_array().unwrap() {
            let id = case["id"].as_str().ok_or("Missing case ID")?;
            report["active_case"] = json!({"id":id,"run":run,"pcm_sha256":case["sha256"]});
            let audio = read_audio(directory, case)?;
            let started = Instant::now();
            let operation = Operation::native().map_err(|e| e.to_string())?;
            let output = network
                .probe(&audio, &operation)
                .map_err(|e| format!("{id}, run {run}: {e}"))?;
            report["successful_probes"] = json!(report["successful_probes"].as_u64().unwrap() + 1);
            let mut row = returned(&output);
            row["id"] = json!(id);
            row["run"] = json!(run);
            row["seconds"] = json!(started.elapsed().as_secs_f64());
            report["rows"].as_array_mut().unwrap().push(row);
            // Returned bits survive a failed validation, repeat assertion or unwind.
            persist(file, report)?;
            checked_output(&output, &cases["codes"])?;
            let hash = score_hash(&output);
            if run == 0 {
                first.insert(id.to_owned(), hash.clone());
            } else if first.get(id) != Some(&hash) {
                return Err(format!("Native repeat changed: {id}"));
            }
            let decision = core(
                &mut LanguagePolicy::new(),
                &output.codes,
                &output.probabilities,
                "independent",
            );
            let scores = output
                .codes
                .iter()
                .cloned()
                .zip(output.probabilities.iter().map(|p| f64::from(*p)))
                .collect::<Vec<_>>();
            let routed = LanguagePolicy::new()
                .decide(&scores, &["independent".into()])
                .unwrap();
            let p = routed.detection.probability.unwrap();
            let margin = routed.detection.margin.unwrap();
            let row = report["rows"].as_array_mut().unwrap().last_mut().unwrap();
            row["core"] = decision;
            row["confident"] = json!(is_confident_detection(p, margin));
            row["probability"] = json!(p);
            row["margin"] = json!(margin);
            row["probability_threshold_distance"] = json!(p - 0.90);
            row["margin_threshold_distance"] = json!(margin - 0.20);
            row["top3"] = json!(
                routed
                    .detection
                    .candidates
                    .iter()
                    .map(|c| json!({"code":c.language,"probability":c.probability}))
                    .collect::<Vec<_>>()
            );
            if let Some(expected) = contract["legacy"]["cases"]
                .as_array()
                .unwrap()
                .iter()
                .find(|c| c["id"] == id)
            {
                if row["core"] != expected["expected_core"] {
                    return Err(format!("Original route contract changed: {id}"));
                }
            }
            report["active_case"] = Value::Null;
        }
    }
    for case in cases["cases"]
        .as_array()
        .unwrap()
        .iter()
        .filter(|c| c["recipe"]["bitexact_to"].is_string())
    {
        let base = case["recipe"]["bitexact_to"].as_str().unwrap();
        if first[case["id"].as_str().unwrap()] != first[base] {
            return Err("Context truncation/padding metamorphic result changed".into());
        }
    }
    let audio = read_audio(directory, &cases["cases"][4])?;
    let cancelled = Operation::native().map_err(|e| e.to_string())?;
    cancelled.cancel();
    let result = network.probe(&audio, &cancelled);
    if result.is_ok() {
        report["successful_probes"] = json!(report["successful_probes"].as_u64().unwrap() + 1);
    }
    let observed = match &result {
        Ok(output) => returned(output),
        Err(error) => json!({"error":error.to_string()}),
    };
    report["controls"]
        .as_array_mut()
        .unwrap()
        .push(json!({"name":"pre_cancelled_before_graph_entry","result":observed}));
    persist(file, report)?;
    if result != Err(ChildError::Cancelled) {
        return Err("Pre-cancelled probe succeeded".into());
    }
    for (name, invalid) in [
        ("empty", &[][..]),
        ("nan", &[f32::NAN][..]),
        ("infinity", &[f32::INFINITY][..]),
    ] {
        let operation = Operation::native().map_err(|e| e.to_string())?;
        let result = network.probe(invalid, &operation);
        if result.is_ok() {
            report["successful_probes"] = json!(report["successful_probes"].as_u64().unwrap() + 1);
        }
        let observed = match &result {
            Ok(output) => returned(output),
            Err(error) => json!({"error":error.to_string()}),
        };
        report["controls"]
            .as_array_mut()
            .unwrap()
            .push(json!({"name":name,"result":observed}));
        persist(file, report)?;
        if result.is_ok() {
            return Err("Invalid PCM succeeded".into());
        }
    }
    let operation = Operation::native().map_err(|e| e.to_string())?;
    let output = network
        .probe(&audio, &operation)
        .map_err(|e| e.to_string())?;
    report["successful_probes"] = json!(report["successful_probes"].as_u64().unwrap() + 1);
    report["controls"]
        .as_array_mut()
        .unwrap()
        .push(json!({"name":"after_failure","result":returned(&output)}));
    persist(file, report)?;
    if score_hash(&output) != first[cases["cases"][4]["id"].as_str().unwrap()] {
        return Err("Failure/cancellation leaked into a fresh request".into());
    }
    report["isolation"] = json!({"pre_cancelled":true,"invalid_pcm":true,"after_failure_bitexact":true,
        "cancellation_scope":"pre-cancel only; in-flight termination/reuse remains an assembled cancellation gate",
        "repeat_order":"all56 then all56; interleaved sources"});
    Ok(())
}

#[test]
#[ignore = "requires frozen56 PCM cases, pinned native model/runtime and a fresh capture path; never activates dispatcher"]
fn native_adapter_captures_frozen_windows_contract() {
    let path = env_path("SPEAKERDESK_WHISPER_RESULT").unwrap();
    let mut file = fs::File::create_new(&path).expect("fresh native capture output");
    let mut report = json!({"schema_version":1,"platform":std::env::consts::OS,"passed_capture":false,
        "contract_sha256":sha256_hex(CONTRACT.as_bytes()),"successful_probes":0,"rows":[],"controls":[],
        "test_source_sha256":sha256_hex(include_bytes!("whisper_windows_qualification.rs")),
        "policy_source_sha256":sha256_hex(include_bytes!("../../core/src/language.rs")),
        "errors":[]});
    #[cfg(target_os = "macos")]
    {
        report["adapter_source_sha256"] =
            json!(sha256_hex(include_bytes!("../src/whisper_lid/network.rs")));
    }
    #[cfg(windows)]
    {
        report["adapter_source_sha256"] = json!(sha256_hex(include_bytes!(
            "../src/whisper_lid/network_windows.rs"
        )));
    }
    persist(&mut file, &report).unwrap();
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        capture(&mut file, &mut report)
    }));
    match result {
        Ok(Ok(())) => report["passed_capture"] = json!(true),
        Ok(Err(error)) => report["errors"] = json!([error]),
        Err(payload) => {
            report["errors"] = json!([payload
                .downcast_ref::<String>()
                .cloned()
                .or_else(|| payload.downcast_ref::<&str>().map(|s| s.to_string()))
                .unwrap_or_else(|| "native capture panicked".into())])
        }
    }
    persist(&mut file, &report).unwrap();
    assert_eq!(report["passed_capture"], true, "{report}");
}
