#![cfg(windows)]

use std::{path::PathBuf, time::Duration};

use speakerdesk_host::capture::{
    CaptureConfig, CapturePhase, CaptureSession, CaptureSnapshot, CaptureSupervisor, Command,
};

#[path = "support/process.rs"]
mod process;

async fn wait_for(
    session: &mut CaptureSession,
    predicate: impl Fn(&CaptureSnapshot) -> bool,
) -> Result<CaptureSnapshot, String> {
    tokio::time::timeout(Duration::from_secs(12), async {
        loop {
            let snapshot = session.updates.borrow().clone();
            if let Some(error) = &snapshot.error {
                return Err(error.clone());
            }
            if predicate(&snapshot) {
                return Ok(snapshot);
            }
            session.updates.changed().await.map_err(|e| e.to_string())?;
        }
    })
    .await
    .map_err(|_| format!("Capture stalled: {:?}", *session.updates.borrow()))?
}

#[tokio::test]
#[ignore = "Records default Windows devices into temporary WAVs; requires SPEAKERDESK_TEST_CAPTURE and SPEAKERDESK_TEST_ECHO"]
async fn native_capture_pause_resume_finalizes_aligned_sources_and_reaps_helper() {
    let helper =
        PathBuf::from(std::env::var_os("SPEAKERDESK_TEST_CAPTURE").expect("native helper"));
    let echo =
        PathBuf::from(std::env::var_os("SPEAKERDESK_TEST_ECHO").expect("native echo library"));
    for (microphone, system) in [(true, false), (false, true), (true, true)] {
        let folder = tempfile::tempdir().unwrap();
        let mut session = CaptureSupervisor::spawn(
            &helper,
            CaptureConfig {
                job_id: "0123456789abcdef0123456789abcdef".into(),
                folder: folder.path().into(),
                microphone,
                system,
                echo_library: Some(echo.clone()),
            },
        )
        .await
        .unwrap();
        let child = process::ObservedProcess::new(session.pid);
        // Always finish the owned session before reporting a lifecycle failure.
        let interaction = async {
            wait_for(&mut session, |s| {
                s.phase == CapturePhase::Recording && s.committed_sample >= 16_000
            })
            .await?;
            session.command(Command::Pause).map_err(|e| e.to_string())?;
            let paused = wait_for(&mut session, |s| s.phase == CapturePhase::Paused).await?;
            tokio::time::sleep(Duration::from_millis(300)).await;
            if *session.updates.borrow() != paused {
                return Err("Paused capture changed its committed state".into());
            }
            session
                .command(Command::Resume)
                .map_err(|e| e.to_string())?;
            wait_for(&mut session, |s| {
                s.phase == CapturePhase::Recording
                    && s.committed_sample >= paused.committed_sample + 16_000
            })
            .await?;
            Ok::<_, String>(paused)
        }
        .await;
        let result = session.finish().await.unwrap();
        assert!(child.has_exited(), "owned native helper outlived Stop");
        let diagnostic =
            std::fs::read_to_string(folder.path().join("capture.log")).unwrap_or_default();
        assert!(
            interaction.is_ok(),
            "microphone={microphone} system={system}: {interaction:?}; {result:?}; {diagnostic}"
        );
        assert_eq!(
            result.phase,
            CapturePhase::Stopped,
            "{result:?}; {diagnostic}"
        );
        assert_eq!(result.raw_sample, result.committed_sample);
        assert!(result.committed_sample >= 32_000);
        assert_eq!(result.late_samples, 0);
        assert!(
            result.processing.is_none(),
            "WASAPI must not acquire Apple processing authority"
        );
        assert!(!folder.path().join("capture-processing.json").exists());
        if microphone && system {
            assert!(result.source_catalog.is_some());
        }
        for (name, channels, enabled) in [
            ("audio.wav", 1, true),
            ("microphone.wav", 1, microphone),
            ("microphone_clean.wav", 1, microphone && system),
            ("system.wav", 1, system),
            ("system_reference.wav", 2, system),
        ] {
            if !enabled {
                continue;
            }
            let mut saved = hound::WavReader::open(folder.path().join(name)).unwrap();
            assert_eq!(saved.spec().sample_rate, 16_000, "{name}");
            assert_eq!(saved.spec().channels, channels, "{name}");
            assert_eq!(saved.duration(), result.committed_sample, "{name}");
            assert_eq!(
                saved.samples::<i16>().map(Result::unwrap).count(),
                result.committed_sample as usize * channels as usize,
                "{name}"
            );
        }
        eprintln!(
            "Native capture microphone={microphone} system={system}: {} committed samples; Pause/Resume/Stop, finalized WAVs and owned child exit passed",
            result.committed_sample
        );
    }
}
