#![cfg(feature = "fixtures")]
use speakerdesk_core::error::ChildError;
use speakerdesk_core::models::Models;
use speakerdesk_host::inference::InferenceSupervisor;
use std::time::{Duration, Instant};
use tokio_util::sync::CancellationToken;
#[path = "support/process.rs"]
mod process_support;
use process_support::ObservedProcess;
fn supervisor() -> std::sync::Arc<InferenceSupervisor> {
    InferenceSupervisor::new(
        env!("CARGO_BIN_EXE_inference-fixture"),
        "fixture",
        std::env::temp_dir(),
    )
    .unwrap()
}

#[test]
fn cleanup_close_is_idempotent_and_keeps_unrelated_sessions_resident() {
    let host = supervisor();
    let models = host.models(CancellationToken::new());
    let vad = models.vad_open(None).unwrap();
    let diar = models
        .diar_open(speakerdesk_core::ipc::DiarPreset::Low)
        .unwrap();
    let other = models.vad_open(None).unwrap();
    let pid = host.child_pid().unwrap();
    let observed = ObservedProcess::new(pid);
    models.vad_close(vad).unwrap();
    models.vad_close(vad).unwrap();
    models.diar_close(diar).unwrap();
    models.diar_close(diar).unwrap();
    assert_eq!(
        models.vad_feed(other, &[0.; 512], true).unwrap().observed,
        512
    );
    assert_eq!(host.child_pid(), Some(pid));
    models.vad_close(other).unwrap();
    host.shutdown();
    reaped(&observed);
}

#[test]
fn auxiliary_sessions_yield_to_primary_work_and_retire_on_cancel_or_child_loss() {
    use speakerdesk_core::ipc::Handle;
    for separator in [true, false] {
        let temp = tempfile::tempdir().unwrap();
        let root = temp.path();
        std::fs::write(root.join("fixture-overlap"), b"bounded protocol replay").unwrap();
        let hold = root.join(if separator {
            "hold-live-separation"
        } else {
            "hold-live-asr"
        });
        std::fs::write(&hold, b"keep auxiliary completion pending").unwrap();
        let host =
            InferenceSupervisor::new(env!("CARGO_BIN_EXE_inference-fixture"), "fixture", root)
                .unwrap();
        let ready = host.models(CancellationToken::new());
        let vad = ready.vad_open(None).unwrap();
        let pcm = vec![0.2; 1600];
        let open = |models: &dyn Models| {
            if separator {
                models.separate_open(&pcm)
            } else {
                models.transcribe_open(&pcm, "en", (0, 0))
            }
        };
        let pending = |models: &dyn Models, handle| {
            if separator {
                models.separate_step(handle).map(|v| v.is_none())
            } else {
                models.transcribe_step(handle).map(|v| v.is_none())
            }
        };
        let close = |models: &dyn Models, handle| {
            if separator {
                models.separate_close(handle)
            } else {
                models.transcribe_close(handle)
            }
        };
        let handle = open(&ready).unwrap();
        let pid = host.child_pid().unwrap();
        let observed = ObservedProcess::new(pid);
        assert!(pending(&ready, handle).unwrap());
        assert_eq!(ready.metrics().unwrap().peak_mlx_bytes, 2);
        assert_eq!(
            ready.vad_feed(vad, &[0.; 512], false).unwrap().observed,
            512
        );
        assert_eq!(
            ready.transcribe(&[0.; 160], "en", (0, 0)).unwrap().text,
            "Replay transcript"
        );
        let wrong = Handle {
            session: handle.session + 100,
            ..handle
        };
        assert!(pending(&ready, wrong).is_err());
        close(&ready, wrong).unwrap();
        assert!(
            pending(&ready, handle).unwrap(),
            "wrong handle preserves owner"
        );
        assert_eq!(host.child_pid(), Some(pid));
        std::fs::remove_file(&hold).unwrap();
        if separator {
            let output = ready.separate_step(handle).unwrap().unwrap();
            assert_eq!(output.channels, [vec![0.4; 1600], vec![0.6; 1600]]);
        } else {
            assert_eq!(
                ready.transcribe_step(handle).unwrap().unwrap().text,
                "Replay transcript"
            );
        }
        assert!(
            pending(&ready, handle).is_err(),
            "terminal step retires state"
        );
        close(&ready, handle).unwrap();
        assert_eq!(ready.metrics().unwrap().peak_mlx_bytes, 1);

        let token = CancellationToken::new();
        let cancelled = host.models(token.clone());
        let owned = open(&cancelled).unwrap();
        token.cancel();
        assert_eq!(pending(&cancelled, owned), Err(ChildError::Cancelled));
        close(&cancelled, owned).unwrap();
        close(&cancelled, owned).unwrap();
        assert_eq!(ready.metrics().unwrap().peak_mlx_bytes, 1);

        // A cancelled successful open and a cancelled nonterminal step both own
        // resident state until the supervisor explicitly closes the discarded result.
        std::fs::write(&hold, b"pending for cancellation race").unwrap();
        for step in [false, true] {
            let token = CancellationToken::new();
            let active = host.models(token.clone());
            let owned = step.then(|| open(&active).unwrap());
            let marker = root.join(if step { "step-race" } else { "open-race" });
            ready
                .transcribe(&[], &format!("race:{}", marker.display()), (0, 0))
                .unwrap();
            let pcm = pcm.clone();
            let call = std::thread::spawn(move || match (separator, owned) {
                (true, None) => active.separate_open(&pcm).map(|_| ()),
                (false, None) => active.transcribe_open(&pcm, "en", (0, 0)).map(|_| ()),
                (true, Some(handle)) => active.separate_step(handle).map(|_| ()),
                (false, Some(handle)) => active.transcribe_step(handle).map(|_| ()),
            });
            wait_for(&marker);
            token.cancel();
            wait_for(&marker.with_extension("cancelled"));
            std::fs::write(marker.with_extension("release"), b"release terminal reply").unwrap();
            assert_eq!(call.join().unwrap(), Err(ChildError::Cancelled));
            assert_eq!(
                ready.metrics().unwrap().peak_mlx_bytes,
                1,
                "discarded state leaked"
            );
        }

        let lost = open(&ready).unwrap();
        host.restart(&CancellationToken::new()).unwrap();
        reaped(&observed);
        close(&ready, lost).unwrap();
        assert_eq!(
            host.child_pid(),
            None,
            "cleanup must not respawn lost child"
        );
        let new = ready.vad_open(None).unwrap();
        let new_pid = host.child_pid().unwrap();
        let new_observed = ObservedProcess::new(new_pid);
        assert!(
            pending(&ready, lost)
                .unwrap_err()
                .to_string()
                .contains("stale")
        );
        assert!(
            close(&ready, lost)
                .unwrap_err()
                .to_string()
                .contains("stale")
        );
        assert_eq!(ready.vad_feed(new, &[0.; 512], true).unwrap().observed, 512);
        assert_eq!(host.child_pid(), Some(new_pid));
        ready.vad_close(new).unwrap();
        host.shutdown();
        reaped(&new_observed);
    }
}

#[test]
fn installation_restart_reaps_the_resident_child_and_invalidates_old_handles() {
    let host = supervisor();
    let cancel = CancellationToken::new();
    let models = host.models(cancel.clone());
    let old = models.vad_open(None).unwrap();
    let pid = host.child_pid().unwrap();
    let observed = ObservedProcess::new(pid);
    host.restart(&cancel).unwrap();
    reaped(&observed);
    assert_eq!(host.child_pid(), None);
    assert!(
        models
            .vad_feed(old, &[0.; 512], false)
            .unwrap_err()
            .to_string()
            .contains("stale model session")
    );
    let new = models.vad_open(None).unwrap();
    assert_ne!(old.generation, new.generation);
    models.vad_close(new).unwrap();
    let pid = host.child_pid().unwrap();
    let observed = ObservedProcess::new(pid);
    host.shutdown();
    reaped(&observed);
}

#[test]
fn recording_larger_than_ipc_frame_uses_bounded_feeds_and_includes_eof() {
    let host = supervisor();
    let models = host.models(CancellationToken::new());
    let mut pcm = vec![0.; speakerdesk_core::ipc::MAX_FRAME_BYTES / 4 + 159];
    let turns = models.diarize_window(&pcm).unwrap();
    assert_eq!(
        turns,
        vec![speakerdesk_core::ipc::Turn {
            start: 0.,
            end: (pcm.len() / 160 * 160) as f64 / 16000.,
            speaker: "speaker_0".into()
        }]
    );
    assert_eq!(
        models.metrics().unwrap().peak_mlx_bytes,
        0,
        "session must close"
    );
    pcm[160_001] = f32::NAN;
    assert!(
        models
            .diarize_window(&pcm)
            .unwrap_err()
            .to_string()
            .contains("fixture model error")
    );
    assert_eq!(
        models.metrics().unwrap().peak_mlx_bytes,
        0,
        "failed feed must close"
    );
    let pid = host.child_pid().unwrap();
    let observed = ObservedProcess::new(pid);
    host.shutdown();
    reaped(&observed);
}

#[test]
#[ignore = "requires SPEAKERDESK_TEST_CHILD desktop binary and SPEAKERDESK_TEST_MODELS disposable model root"]
fn actual_desktop_child_returns_native_voice_identity_and_reaps() {
    let host = InferenceSupervisor::new(
        std::env::var_os("SPEAKERDESK_TEST_CHILD").expect("desktop binary"),
        env!("CARGO_PKG_VERSION"),
        std::env::var_os("SPEAKERDESK_TEST_MODELS").expect("model directory"),
    )
    .unwrap();
    let models = host.models(CancellationToken::new());
    let report = models.setup_check(&["redimnet2-b6".into()]).unwrap();
    let revision =
        "112dd8f4f836abdf8a420e66a5e2885cf8ec64ab:sd-pcm-v1:native-coreml-v1:CPU_AND_GPU";
    assert_eq!(report.len(), 1);
    assert!(report[0].ok);
    assert_eq!(report[0].model, revision);
    let pid = host.child_pid().unwrap();
    let observed = ObservedProcess::new(pid);
    let reference: serde_json::Value = serde_json::from_str(include_str!(
        "../../models/tests/data/redimnet-mlx-reference.json"
    ))
    .unwrap();
    let mut reader = hound::WavReader::open(
        std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../models/tests/data/two-voice.wav"),
    )
    .unwrap();
    let pcm: Vec<f32> = reader
        .samples::<i16>()
        .map(|s| f32::from(s.unwrap()) / 32768.)
        .collect();
    let output = models.embed_voice(&pcm[..64000]).unwrap();
    assert_eq!(output.model.revision, revision);
    assert_eq!(output.model.model_id, "aufklarer/ReDimNet2-B6-CoreML");
    assert_eq!(
        output.model.artifact_sha256,
        "cf8cc300e240efaed56cb931bd7715ded41b66bf55af7e9a19d0b17e7fe986c9"
    );
    assert_eq!(output.model.dimension, 192);
    assert_eq!(output.model.sample_rate, 16000);
    assert!(output.clip.clean);
    assert_eq!(output.clip.vector.len(), 192);
    let cosine: f64 = output
        .clip
        .vector
        .iter()
        .zip(reference["cases"][0]["vector"].as_array().unwrap())
        .map(|(a, b)| a * b.as_f64().unwrap())
        .sum();
    assert!(cosine >= speakerdesk_core::voice::IDENTITY_REUSE_COSINE);
    assert!(models.embed_voice(&[]).is_err());
    let silence = models.embed_voice(&[0.; 32000]).unwrap();
    assert!(!silence.clip.clean && silence.clip.vector.is_empty());
    assert_eq!(silence.model, output.model);
    assert_eq!(host.child_pid(), Some(pid));
    host.shutdown();
    reaped(&observed);
}

#[test]
#[ignore = "requires SPEAKERDESK_TEST_CHILD desktop binary and SPEAKERDESK_TEST_MODELS disposable model root"]
fn actual_desktop_child_runs_speech_models_and_reaps() {
    let child = std::env::var_os("SPEAKERDESK_TEST_CHILD").expect("desktop binary");
    let model_root = std::env::var_os("SPEAKERDESK_TEST_MODELS").expect("model directory");
    let host = InferenceSupervisor::new(child, env!("CARGO_PKG_VERSION"), model_root).unwrap();
    let models = host.models(CancellationToken::new());
    let first = models.vad_open(None).unwrap();
    let second = models.vad_open(None).unwrap();
    assert_ne!(first, second);
    let buffered = models.vad_feed(first, &[0.0; 511], false).unwrap();
    assert_eq!((buffered.received, buffered.observed), (511, 0));
    assert!(buffered.frames.is_empty());
    let full = models.vad_feed(first, &[0.0; 38], false).unwrap();
    assert_eq!((full.received, full.observed), (549, 512));
    assert_eq!(full.frames.len(), 1);
    let tail = models.vad_feed(first, &[], true).unwrap();
    assert_eq!((tail.received, tail.observed), (549, 549));
    assert_eq!((tail.frames[0].start, tail.frames[0].end), (512, 549));
    models.vad_close(first).unwrap();
    let other = models.vad_feed(second, &[0.0; 512], true).unwrap();
    assert_eq!((other.frames[0].start, other.observed), (0, 512));
    models.vad_close(second).unwrap();
    let mut reader = hound::WavReader::open(
        std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../models/tests/data/two-voice.wav"),
    )
    .unwrap();
    let pcm: Vec<f32> = reader
        .samples::<i16>()
        .map(|s| f32::from(s.unwrap()) / 32768.0)
        .collect();
    let turns = models.diarize_window(&pcm).unwrap();
    assert_eq!(
        turns,
        vec![
            speakerdesk_core::ipc::Turn {
                start: 0.03,
                end: 4.05,
                speaker: "speaker_0".into()
            },
            speakerdesk_core::ipc::Turn {
                start: 5.0,
                end: 8.86,
                speaker: "speaker_1".into()
            },
        ]
    );
    let text = models.transcribe(&pcm, "en", (0, 0)).unwrap();
    assert_eq!(
        text.text,
        "I checked the recording this morning. The transcript needs a speaker name. I can review it after lunch. Please save the final version."
    );
    assert_eq!(text.token_count, 31);
    let document = speakerdesk_core::import::import_recording(&models, &pcm, "en", None).unwrap();
    let exported = speakerdesk_core::export::export(&document, "txt").unwrap();
    eprintln!("Actual child import transcript: {}", exported.body);
    assert!(
        exported
            .body
            .contains("I checked the recording this morning")
    );
    assert!(exported.body.contains("Please save the final version"));
    let spoken: Vec<_> = document
        .segments
        .iter()
        .filter(|s| !s.text.is_empty())
        .collect();
    assert_eq!(spoken.len(), 2, "no separately decoded VAD tails");
    for (index, row) in spoken.iter().enumerate() {
        assert_eq!(
            row.speaker, "unassigned",
            "unaligned tail context cannot establish word ownership"
        );
        assert_eq!(
            row.rest["speaker_candidates"],
            serde_json::json!([format!("speaker_{index}")])
        );
        assert_eq!(row.rest["voice_eligible"], false);
        assert_eq!(row.review, Some(true));
    }
    let pid = host.child_pid().unwrap();
    let observed = ObservedProcess::new(pid);
    host.shutdown();
    reaped(&observed);
}
fn wait_for(path: &std::path::Path) {
    let deadline = Instant::now() + Duration::from_secs(5);
    while !path.exists() {
        assert!(
            Instant::now() < deadline,
            "fixture did not reach {}",
            path.display()
        );
        std::thread::sleep(Duration::from_millis(1));
    }
}
fn reaped(observed: &ObservedProcess) {
    #[cfg(unix)]
    {
        let result = unsafe { libc::waitpid(observed.pid, std::ptr::null_mut(), libc::WNOHANG) };
        assert_eq!(
            result, -1,
            "the supervisor must have already reaped its child"
        );
        assert_eq!(
            std::io::Error::last_os_error().raw_os_error(),
            Some(libc::ECHILD)
        );
    }
    assert!(
        observed.has_exited(),
        "owned inference child survived cleanup"
    );
}
#[test]
fn resident_child_restarts_after_crash_and_rejects_stale_handles() {
    let host = supervisor();
    let models = host.models(CancellationToken::new());
    let handle = models.vad_open(None).unwrap();
    let pid = host.child_pid().unwrap();
    let observed = ObservedProcess::new(pid);
    assert_eq!(
        models.transcribe(&[0.0; 16], "en", (0, 0)).unwrap().text,
        "Replay transcript"
    );
    assert_eq!(host.child_pid(), Some(pid));
    assert_eq!(
        models.embed_voice(&[]).unwrap_err(),
        ChildError::Unimplemented
    );
    assert_eq!(
        host.child_pid(),
        Some(pid),
        "a model error must not unload unrelated sessions"
    );
    assert!(models.transcribe(&[], "exit", (0, 0)).is_err());
    reaped(&observed);
    let new_handle = models.vad_open(None).unwrap();
    assert_ne!(handle.generation, new_handle.generation);
    assert!(
        models
            .vad_close(handle)
            .unwrap_err()
            .to_string()
            .contains("stale")
    );
    models.vad_close(new_handle).unwrap();
    let last_pid = host.child_pid().unwrap();
    let last_observed = ObservedProcess::new(last_pid);
    host.shutdown();
    reaped(&last_observed);
}
#[test]
fn cancellation_waits_for_terminal_reply_and_keeps_other_sessions() {
    let host = supervisor();
    let ready = host.models(CancellationToken::new());
    let handle = ready.vad_open(None).unwrap();
    let pid = host.child_pid().unwrap();
    let observed = ObservedProcess::new(pid);
    let token = CancellationToken::new();
    let active = host.models(token.clone());
    let temp = tempfile::tempdir().unwrap();
    let marker = temp.path().join("slow");
    let language = format!("slow:{}", marker.display());
    let call = std::thread::spawn(move || active.transcribe(&[], &language, (0, 0)));
    wait_for(&marker);
    token.cancel();
    wait_for(&marker.with_extension("cancelled"));
    assert!(!call.is_finished(), "ack must not complete the operation");
    std::fs::write(marker.with_extension("release"), b"release").unwrap();
    assert_eq!(call.join().unwrap(), Err(ChildError::Cancelled));
    assert_eq!(host.child_pid(), Some(pid));
    ready.vad_close(handle).unwrap();
    let queued = CancellationToken::new();
    queued.cancel();
    assert_eq!(
        host.models(queued).transcribe(&[], "en", (0, 0)),
        Err(ChildError::Cancelled)
    );
    assert_eq!(
        ready.transcribe(&[], "en", (0, 0)).unwrap().text,
        "Replay transcript"
    );
    host.shutdown();
    reaped(&observed);
}
#[test]
fn malformed_frames_and_bad_handshakes_are_reaped() {
    let host = supervisor();
    let models = host.models(CancellationToken::new());
    models.metrics().unwrap();
    let pid = host.child_pid().unwrap();
    let observed = ObservedProcess::new(pid);
    assert!(models.transcribe(&[], "malformed", (0, 0)).is_err());
    reaped(&observed);
    host.shutdown();
    let wrong = InferenceSupervisor::new(
        env!("CARGO_BIN_EXE_inference-fixture"),
        "different-build",
        std::env::temp_dir(),
    )
    .unwrap();
    assert!(wrong.models(CancellationToken::new()).metrics().is_err());
    assert_eq!(wrong.child_pid(), None);
    wrong.shutdown();
}
#[test]
fn uncooperative_cancel_kills_and_reaps_before_releasing_the_operation() {
    let host = supervisor();
    host.models(CancellationToken::new()).metrics().unwrap();
    let pid = host.child_pid().unwrap();
    let observed = ObservedProcess::new(pid);
    let token = CancellationToken::new();
    let models = host.models(token.clone());
    let temp = tempfile::tempdir().unwrap();
    let marker = temp.path().join("hang");
    let language = format!("hang:{}", marker.display());
    let call = std::thread::spawn(move || models.transcribe(&[], &language, (0, 0)));
    wait_for(&marker);
    token.cancel();
    assert_eq!(call.join().unwrap(), Err(ChildError::Timeout));
    reaped(&observed);
    assert_eq!(host.child_pid(), None);
    host.shutdown();
}

#[test]
fn cancellation_interrupts_a_partially_written_request_and_reaps() {
    let host = supervisor();
    let ready = host.models(CancellationToken::new());
    let temp = tempfile::tempdir().unwrap();
    let marker = temp.path().join("blocked");
    ready
        .transcribe(&[], &format!("block-input:{}", marker.display()), (0, 0))
        .unwrap();
    let pid = host.child_pid().unwrap();
    let observed = ObservedProcess::new(pid);
    let token = CancellationToken::new();
    let models = host.models(token.clone());
    let call = std::thread::spawn(move || models.transcribe(&vec![0.0; 1_000_000], "en", (0, 0)));
    wait_for(&marker); // The child consumed the frame prefix and will never consume its payload.
    let started = Instant::now();
    token.cancel();
    assert_eq!(call.join().unwrap(), Err(ChildError::Cancelled));
    assert!(started.elapsed() < Duration::from_secs(10));
    reaped(&observed);
    host.shutdown();
}

#[test]
fn cancelled_operations_can_close_sessions_and_discarded_outputs_are_closed() {
    use speakerdesk_core::ipc::DiarPreset;
    let host = supervisor();
    let ready = host.models(CancellationToken::new());
    let other = ready.vad_open(None).unwrap();
    let pid = host.child_pid().unwrap();
    let observed = ObservedProcess::new(pid);
    let token = CancellationToken::new();
    let models = host.models(token.clone());
    let owned = models.vad_open(None).unwrap();
    token.cancel();
    models.vad_close(owned).unwrap();
    assert_eq!(ready.metrics().unwrap().peak_mlx_bytes, 1);
    for (index, (diar, feed, error)) in [
        (false, false, false),
        (true, false, false),
        (false, true, false),
        (true, true, false),
        (false, true, true),
    ]
    .into_iter()
    .enumerate()
    {
        let token = CancellationToken::new();
        let models = host.models(token.clone());
        let handle = if feed {
            Some(if diar {
                models.diar_open(DiarPreset::Low).unwrap()
            } else {
                models.vad_open(None).unwrap()
            })
        } else {
            None
        };
        let temp = tempfile::tempdir().unwrap();
        let marker = temp.path().join(format!("race-{index}"));
        let mode = if error { "race-error" } else { "race" };
        ready
            .transcribe(&[], &format!("{mode}:{}", marker.display()), (0, 0))
            .unwrap();
        let call = std::thread::spawn(move || match (diar, handle) {
            (false, None) => models.vad_open(None).map(|_| ()),
            (true, None) => models.diar_open(DiarPreset::Low).map(|_| ()),
            (false, Some(handle)) => models.vad_feed(handle, &[0.0; 512], false).map(|_| ()),
            (true, Some(handle)) => models.diar_feed(handle, &[0.0; 512], false).map(|_| ()),
        });
        wait_for(&marker);
        token.cancel();
        wait_for(&marker.with_extension("cancelled"));
        std::fs::write(marker.with_extension("release"), b"release").unwrap();
        assert_eq!(call.join().unwrap(), Err(ChildError::Cancelled));
        assert_eq!(host.child_pid(), Some(pid));
        assert_eq!(
            ready.metrics().unwrap().peak_mlx_bytes,
            1,
            "cancelled session leaked"
        );
    }
    ready.vad_close(other).unwrap();
    host.shutdown();
    reaped(&observed);
}

#[test]
#[ignore = "requires built desktop child, pinned models, and independent long Nemotron reference"]
fn actual_child_keeps_long_offline_and_live_diarization_sessions_separate() {
    use speakerdesk_core::ipc::DiarPreset;
    let host = InferenceSupervisor::new(
        std::env::var_os("SPEAKERDESK_TEST_CHILD").unwrap(),
        env!("CARGO_PKG_VERSION"),
        std::env::var_os("SPEAKERDESK_TEST_MODELS").unwrap(),
    )
    .unwrap();
    let models = host.models(CancellationToken::new());
    let proof =
        std::path::PathBuf::from(std::env::var_os("SPEAKERDESK_TEST_NEMOTRON_LONG").unwrap());
    let reference: serde_json::Value =
        serde_json::from_slice(&std::fs::read(proof.join("reference.json")).unwrap()).unwrap();
    let expected: Vec<[f32; 8]> =
        serde_json::from_value(reference["probabilities"].clone()).unwrap();
    let mut wav = hound::WavReader::open(proof.join("three-windows.wav")).unwrap();
    let pcm: Vec<f32> = wav
        .samples::<i16>()
        .map(|s| s.unwrap() as f32 / 32768.)
        .collect();
    let live = models.diar_open(DiarPreset::Low).unwrap();
    let buffered = models.diar_feed(live, &[0.; 512], false).unwrap();
    assert_eq!((buffered.received, buffered.observed), (512, 0));
    let turns = models.diarize_window(&pcm).unwrap();
    assert_eq!(expected.len(), pcm.len() / 160);
    let mut activity = vec![[false; 8]; expected.len()];
    for turn in turns {
        let speaker: usize = turn
            .speaker
            .strip_prefix("speaker_")
            .unwrap()
            .parse()
            .unwrap();
        for row in
            &mut activity[(turn.start * 100.).round() as usize..(turn.end * 100.).round() as usize]
        {
            row[speaker] = true;
        }
    }
    assert_eq!(
        activity,
        expected
            .iter()
            .map(|row| row.map(|p| p > 0.5))
            .collect::<Vec<_>>()
    );
    let tail = models.diar_feed(live, &[], true).unwrap();
    assert_eq!((tail.received, tail.observed), (512, 480));
    assert!(tail.turns.is_empty());
    models.diar_close(live).unwrap();
    let bad = models.diar_open(DiarPreset::Low).unwrap();
    assert!(models.diar_feed(bad, &[f32::NAN], false).is_err());
    assert!(
        models
            .diar_feed(bad, &[], true)
            .unwrap_err()
            .to_string()
            .contains("unknown Nemotron session")
    );
    models.diar_close(bad).unwrap();
    let pid = host.child_pid().unwrap();
    let observed = ObservedProcess::new(pid);
    host.shutdown();
    reaped(&observed);
}
