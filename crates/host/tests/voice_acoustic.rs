//! Opt-in public acoustic measurement through the real child. This is not a mock
//! or a recognition acceptance test; the independently labelled output is scored
//! separately before selecting a production policy.
use serde_json::json;
use speakerdesk_core::{models::Models, separation, voice::separation as identity};
use speakerdesk_host::inference::InferenceSupervisor;
use std::{io::Write, path::PathBuf, time::Instant};
use tokio_util::sync::CancellationToken;

#[test]
#[ignore = "requires a real desktop child, disposable models, public voice case manifest and fresh output"]
fn public_generated_voice_inputs_through_the_real_child() {
    let manifest = PathBuf::from(std::env::var_os("SPEAKERDESK_VOICE_CASES").unwrap());
    let output = PathBuf::from(std::env::var_os("SPEAKERDESK_VOICE_RESULT").unwrap());
    let mut output = std::fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(output)
        .unwrap();
    let bytes = std::fs::read(&manifest).unwrap();
    let cases: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
    let host = InferenceSupervisor::new(
        std::env::var_os("SPEAKERDESK_TEST_CHILD").unwrap(),
        env!("CARGO_PKG_VERSION"),
        std::env::var_os("SPEAKERDESK_TEST_MODELS").unwrap(),
    )
    .unwrap();
    let models = host.models(CancellationToken::new());
    let report = models
        .setup_check(&[
            "silero-speech".into(),
            "cohere-speech".into(),
            "redimnet2-b6".into(),
        ])
        .unwrap();
    assert_eq!(report.len(), 3);
    assert!(report.iter().all(|entry| entry.ok), "{report:?}");
    let mut calls = Vec::new();
    for case in cases["cases"].as_array().unwrap() {
        let path = PathBuf::from(case["path"].as_str().unwrap());
        let path = if path.is_absolute() {
            path
        } else {
            manifest.parent().unwrap().join(path)
        };
        let mut reader = hound::WavReader::open(&path).unwrap();
        assert_eq!(
            (
                reader.spec().channels,
                reader.spec().sample_rate,
                reader.spec().bits_per_sample
            ),
            (1, 16000, 16)
        );
        let pcm: Vec<_> = reader
            .samples::<i16>()
            .map(|v| f32::from(v.unwrap()) / 32768.)
            .collect();
        let start = Instant::now();
        let (candidate, speech, query) = if case["role"] == "generated_query" {
            let (candidate, speech) =
                separation::decode_voice_with_evidence(&models, &pcm, "en").unwrap();
            let query = identity::query_range(pcm.len(), &speech);
            (Some(candidate), speech, query)
        } else {
            (
                None,
                Vec::new(),
                Some((
                    case["start_sample"].as_u64().unwrap() as usize,
                    case["end_sample"].as_u64().unwrap() as usize,
                )),
            )
        };
        let embedding = query.map(|(start, end)| {
            assert!(start < end && end <= pcm.len());
            let crop = &pcm[start..end];
            let adjusted = if candidate.is_some() {
                identity::query_pcm(crop).unwrap()
            } else {
                crop.to_vec()
            };
            models.embed_voice(&adjusted).unwrap()
        });
        calls.push(json!({"id":case["id"],"path":path,"wav_sha256":speakerdesk_core::memo::sha256_hex(&std::fs::read(&path).unwrap()),"candidate":candidate,"speech_regions":speech,"query_range":query,"embedding":embedding,"seconds":start.elapsed().as_secs_f64()}));
    }
    let pid = host.child_pid().expect("real child ran");
    host.shutdown();
    assert!(host.child_pid().is_none());
    #[cfg(unix)]
    assert_eq!(
        unsafe { libc::kill(pid as libc::pid_t, 0) },
        -1,
        "owned child was not reaped"
    );
    output.write_all(&serde_json::to_vec_pretty(&json!({"scope":"real child query preparation and embeddings; score independently","manifest_sha256":speakerdesk_core::memo::sha256_hex(&bytes),"calls":calls})).unwrap()).unwrap();
}
