#![cfg(any(target_os = "macos", windows))]
use serde_json::{Value, json};
use speakerdesk_core::{error::ChildError, ipc::CtcScoresOut, memo::sha256_hex};
use speakerdesk_models::ctc::network::Network;
use std::{fs, io::Write, path::PathBuf, time::Instant};

#[path = "support/qualification_runtime.rs"]
mod qualification_runtime;

fn score(
    network: &mut Network,
    pcm: &[f32],
    ids: &[i32],
    cancelled: bool,
) -> Result<CtcScoresOut, ChildError> {
    #[cfg(target_os = "macos")]
    {
        network.scores(pcm, ids, &std::sync::atomic::AtomicBool::new(cancelled))
    }
    #[cfg(windows)]
    {
        let operation = speakerdesk_models::operation::Operation::native().unwrap();
        if cancelled {
            operation.cancel();
        }
        network.scores(pcm, ids, &operation)
    }
}

#[test]
#[ignore = "requires prospectively frozen CTC cases, pinned native runtime/model and fresh output"]
fn capture_actual_ctc_adapter_for_windows_functional_qualification() {
    let path = |name| PathBuf::from(std::env::var_os(name).expect(name));
    let cases_dir = path("SPEAKERDESK_CTC_QUALIFICATION");
    let manifest = fs::read(cases_dir.join("cases.json")).unwrap();
    let manifest_hash = sha256_hex(&manifest);
    assert_eq!(
        manifest_hash,
        std::env::var("SPEAKERDESK_CTC_CASES_SHA256").unwrap()
    );
    let data: Value = serde_json::from_slice(&manifest).unwrap();
    let protocol = include_bytes!("../../../experiments/windows/ctc/windows-qualification-v1.json");
    assert_eq!(data["protocol_sha256"], sha256_hex(protocol));
    let output = path("SPEAKERDESK_CTC_CAPTURE");
    fs::create_dir(&output).expect("capture output must be a fresh directory");
    let model = path("SPEAKERDESK_TEST_CTC");
    #[cfg(windows)]
    let runtime_directory = path("SPEAKERDESK_TEST_WINDOWS_RUNTIME");
    #[cfg(windows)]
    let backend =
        speakerdesk_models::windows_runtime::Backend::from_directory(&runtime_directory).unwrap();
    #[cfg(target_os = "macos")]
    let (mut network, runtime) = {
        let runtime = qualification_runtime::prepare_mlx();
        (Network::load(&model).unwrap(), runtime)
    };
    #[cfg(windows)]
    let (mut network, runtime) = {
        let policy: Value = serde_json::from_slice(protocol).unwrap();
        let runtime_hash =
            sha256_hex(&fs::read(runtime_directory.join("onnxruntime.dll")).unwrap());
        let directml_hash = sha256_hex(&fs::read(runtime_directory.join("DirectML.dll")).unwrap());
        assert_eq!(runtime_hash, policy["onnxruntime_sha256"]);
        assert_eq!(directml_hash, policy["directml_sha256"]);
        (
            Network::load(&model, &backend).unwrap(),
            json!({"provider":"DirectML", "private_runtime_admitted":true,"onnxruntime.dll":runtime_hash,"DirectML.dll":directml_hash}),
        )
    };
    let cases = data["cases"].as_array().unwrap();
    assert_eq!(cases.len(), 19);
    let mut first: Vec<CtcScoresOut> = Vec::new();
    let mut calls = Vec::new();
    let outcome = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        for repeat in 0..2 {
            for (index, case) in cases.iter().enumerate() {
                let id = case["id"].as_str().unwrap();
                let input = case["path"].as_str().unwrap();
                assert!(
                    std::path::Path::new(input)
                        .components()
                        .all(|c| matches!(c, std::path::Component::Normal(_)))
                );
                let bytes = fs::read(cases_dir.join(input)).unwrap();
                assert_eq!(sha256_hex(&bytes), case["pcm_sha256"]);
                assert_eq!(bytes.len(), case["samples"].as_u64().unwrap() as usize * 4);
                assert_eq!(
                    sha256_hex(case["text"].as_str().unwrap().as_bytes()),
                    case["text_sha256"]
                );
                let pcm: Vec<_> = bytes
                    .chunks_exact(4)
                    .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
                    .collect();
                let ids: Vec<i32> = serde_json::from_value(case["token_ids"].clone()).unwrap();
                let started = Instant::now();
                let result = score(&mut network, &pcm, &ids, false);
                fs::write(
                    output.join(format!("{id}-repeat-{repeat}.json")),
                    serde_json::to_vec(&result).unwrap(),
                )
                .unwrap();
                let scores = result.unwrap();
                let raw: Vec<_> = scores
                    .gathered
                    .iter()
                    .flat_map(|v| v.to_le_bytes())
                    .collect();
                fs::write(output.join(format!("{id}-repeat-{repeat}.f32")), &raw).unwrap();
                calls.push(json!({"id":id,"repeat":repeat,"score_sha256":sha256_hex(&raw),"elapsed_ms":started.elapsed().as_millis()}));
                if repeat == 0 {
                    fs::write(
                        output.join(format!("{id}.json")),
                        serde_json::to_vec(&scores).unwrap(),
                    )
                    .unwrap();
                }
                assert_eq!(scores.valid_frames as usize, (pcm.len() - 400) / 320 + 1);
                assert_eq!(scores.audio_num_samples as usize, pcm.len());
                assert_eq!(
                    (scores.frame_hop_samples, scores.frame_origin_samples),
                    (320, 0)
                );
                assert_eq!(scores.token_ids, ids);
                assert_eq!(
                    scores.vocabulary_sha256,
                    speakerdesk_models::ctc::TOKENS_SHA256
                );
                assert_eq!(
                    scores.gathered.len(),
                    scores.valid_frames as usize * ids.len()
                );
                assert!(scores.gathered.iter().all(|v| v.is_finite() && *v <= 0.));
                assert!(
                    scores.gathered.chunks_exact(ids.len()).all(|row| row
                        .iter()
                        .map(|&v| f64::from(v).exp())
                        .sum::<f64>()
                        <= 0.001_f64.exp())
                );
                #[cfg(target_os = "macos")]
                assert_eq!(scores.model, speakerdesk_core::timing::MODEL);
                #[cfg(windows)]
                assert_eq!(
                    scores.model,
                    speakerdesk_models::ctc::network::ARTIFACT_SHA256
                );
                if repeat == 0 {
                    first.push(scores.clone());
                } else {
                    assert_eq!(scores, first[index], "request isolation/repeat: {id}");
                    assert_eq!(
                        raw,
                        first[index]
                            .gathered
                            .iter()
                            .flat_map(|v| v.to_le_bytes())
                            .collect::<Vec<_>>(),
                        "bit-exact repeat: {id}"
                    );
                }
                eprintln!("CTC {id} repeat {repeat}: {:?}", started.elapsed());
            }
        }
        for (pcm, ids) in [
            (vec![], vec![0]),
            (vec![f32::NAN], vec![0]),
            (vec![0.; 392001], vec![0]),
            (vec![0.; 400], vec![]),
            (vec![0.; 400], vec![-1]),
            (vec![0.; 400], vec![9812]),
        ] {
            assert!(score(&mut network, &pcm, &ids, false).is_err());
        }
        assert_eq!(
            score(&mut network, &[0.; 720], &[0], true),
            Err(ChildError::Cancelled)
        );
        let short = score(&mut network, &[0.; 399], &[0], false).unwrap();
        assert_eq!(short.valid_frames, 0);
        assert!(short.gathered.is_empty());
        let case = &cases[0];
        let bytes = fs::read(cases_dir.join(case["path"].as_str().unwrap())).unwrap();
        let pcm: Vec<_> = bytes
            .chunks_exact(4)
            .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
            .collect();
        let post_cancel = score(&mut network, &pcm, &first[0].token_ids, false);
        fs::write(
            output.join("post-pre-cancel.json"),
            serde_json::to_vec(&post_cancel).unwrap(),
        )
        .unwrap();
        assert_eq!(
            post_cancel.unwrap(),
            first[0],
            "cancellation poisoned next request"
        );
    }));
    let passed = outcome.is_ok();
    let receipt = json!({"passed":passed,"manifest_sha256":manifest_hash,"protocol_sha256":sha256_hex(protocol),"capture_test_sha256":sha256_hex(include_bytes!("ctc_windows_qualification.rs")),"model":first.first().map(|s|&s.model),"runtime":runtime,"calls":calls,"repeat_bitexact":passed,"invalid_requests_rejected":passed,"pre_cancel_isolated":passed});
    let mut file = fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(output.join("receipt.json"))
        .unwrap();
    file.write_all(&serde_json::to_vec_pretty(&receipt).unwrap())
        .unwrap();
    file.sync_all().unwrap();
    if let Err(failure) = outcome {
        std::panic::resume_unwind(failure);
    }
}
