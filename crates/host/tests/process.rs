#![cfg(feature = "fixtures")]
use speakerdesk_core::error::ChildError;
use speakerdesk_core::models::Models;
use speakerdesk_host::inference::InferenceSupervisor;
use std::time::{Duration, Instant};
use tokio_util::sync::CancellationToken;
fn supervisor() -> std::sync::Arc<InferenceSupervisor> {
    InferenceSupervisor::new(
        env!("CARGO_BIN_EXE_inference-fixture"),
        "fixture",
        std::env::temp_dir(),
    )
    .unwrap()
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
    let pid = host.child_pid().unwrap();
    host.shutdown();
    reaped(pid);
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
fn reaped(pid: u32) {
    #[cfg(unix)]
    {
        let result = unsafe { libc::waitpid(pid as i32, std::ptr::null_mut(), libc::WNOHANG) };
        assert_eq!(
            result, -1,
            "the supervisor must have already reaped its child"
        );
        assert_eq!(
            std::io::Error::last_os_error().raw_os_error(),
            Some(libc::ECHILD)
        );
        assert_eq!(unsafe { libc::kill(pid as i32, 0) }, -1);
    }
}
#[test]
fn resident_child_restarts_after_crash_and_rejects_stale_handles() {
    let host = supervisor();
    let models = host.models(CancellationToken::new());
    let handle = models.vad_open(None).unwrap();
    let pid = host.child_pid().unwrap();
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
    reaped(pid);
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
    host.shutdown();
    reaped(last_pid);
}
#[test]
fn cancellation_waits_for_terminal_reply_and_keeps_other_sessions() {
    let host = supervisor();
    let ready = host.models(CancellationToken::new());
    let handle = ready.vad_open(None).unwrap();
    let pid = host.child_pid().unwrap();
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
    reaped(pid);
}
#[test]
fn malformed_frames_and_bad_handshakes_are_reaped() {
    let host = supervisor();
    let models = host.models(CancellationToken::new());
    models.metrics().unwrap();
    let pid = host.child_pid().unwrap();
    assert!(models.transcribe(&[], "malformed", (0, 0)).is_err());
    reaped(pid);
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
    let token = CancellationToken::new();
    let models = host.models(token.clone());
    let temp = tempfile::tempdir().unwrap();
    let marker = temp.path().join("hang");
    let language = format!("hang:{}", marker.display());
    let call = std::thread::spawn(move || models.transcribe(&[], &language, (0, 0)));
    wait_for(&marker);
    token.cancel();
    assert_eq!(call.join().unwrap(), Err(ChildError::Timeout));
    reaped(pid);
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
    let token = CancellationToken::new();
    let models = host.models(token.clone());
    let call = std::thread::spawn(move || models.transcribe(&vec![0.0; 1_000_000], "en", (0, 0)));
    wait_for(&marker); // The child consumed the frame prefix and will never consume its payload.
    let started = Instant::now();
    token.cancel();
    assert_eq!(call.join().unwrap(), Err(ChildError::Cancelled));
    assert!(started.elapsed() < Duration::from_secs(10));
    reaped(pid);
    host.shutdown();
}

#[test]
fn cancelled_operations_can_close_sessions_and_discarded_outputs_are_closed() {
    use speakerdesk_core::ipc::DiarPreset;
    let host = supervisor();
    let ready = host.models(CancellationToken::new());
    let other = ready.vad_open(None).unwrap();
    let pid = host.child_pid().unwrap();
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
    reaped(pid);
}
