use std::{
    fs,
    path::{Path, PathBuf},
    time::Duration,
};

use base64::{Engine, engine::general_purpose::STANDARD};
use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use tempfile::TempDir;

use super::{CaptureConfig, CapturePhase, CaptureSupervisor, Command, parse_event};
use super::{
    processing::{EXPORT, Helper},
    writer::SourceWriter,
};

const JOB: &str = "0123456789abcdef0123456789abcdef";

fn helper(folder: &Path, qualified: bool, script: &str) -> PathBuf {
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        fs::set_permissions(folder, fs::Permissions::from_mode(0o700)).unwrap();
    }
    let path = folder.join("capture-helper");
    fs::write(&path, script).unwrap();
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        fs::set_permissions(&path, fs::Permissions::from_mode(0o700)).unwrap();
    }
    if qualified {
        fs::write(
            folder.join("capture-helper.capability.json"),
            serde_json::to_vec(&json!({
            "helper_sha256":format!("{:x}",Sha256::digest(script.as_bytes())),
            "export_contract":EXPORT,"capability_version":"apple_input_node_output_route_v1"}))
            .unwrap(),
        )
        .unwrap();
    }
    path
}

fn config(folder: &Path, dual: bool) -> CaptureConfig {
    CaptureConfig {
        job_id: JOB.into(),
        folder: folder.into(),
        microphone: true,
        system: dual,
    }
}

fn receipt(path: &Path) -> Value {
    let format = json!({"sample_rate":48000,"channels":1,"layout":"mono"});
    json!({"type":"capture_processing","schema_version":1,"job_id":JOB,
        "helper_sha256":format!("{:x}",Sha256::digest(fs::read(path).unwrap())),
        "processor":"apple_voice_processing_io","export_contract":EXPORT,
        "output":"processed_microphone","stage":"retained_source_pcm","postprocessing":"identity",
        "stream_epoch":0,"format":{"sample_rate":16000,"channels":1,"layout":"mono"},
        "capability_version":"apple_input_node_output_route_v1",
        "route":{"node":"inputNode","bus":0,"tap_scope":"output"},"state_phase":"after_start",
        "channel_mapping":"mono_direct","callback_format":format,
        "io_formats":{"input_output":format,"output_input":format},
        "readbacks":{"input_enabled":true,"output_enabled":true,"engine_running":true,
            "bypassed":false,"manual_rendering":false,"input_muted":false,"agc_enabled":false}})
}

fn accept(writer: &mut SourceWriter, value: &Value) -> Result<(), super::CaptureError> {
    writer.accept(parse_event(&value.to_string()).unwrap())
}

fn audio(source: &str, start: f64, values: &[f32], channels: usize) -> Value {
    let bytes: Vec<_> = values.iter().flat_map(|v| v.to_le_bytes()).collect();
    json!({"type":"audio","source":source,"time":start/16000.,"rate":16000,
        "channels":channels,"pcm":STANDARD.encode(bytes)})
}

fn pcm(path: &Path) -> Vec<u8> {
    hound::WavReader::open(path)
        .unwrap()
        .into_samples::<i16>()
        .flat_map(|s| s.unwrap().to_le_bytes())
        .collect()
}

#[test]
fn retained_pcm_matches_independent_v066_source_mixer() {
    let fixture: Value =
        serde_json::from_str(include_str!("../../tests/testdata/capture-v066.json")).unwrap();
    assert_eq!(
        fixture["reference_commit"],
        "91f7593fc9c890c69befd6663cdcdeb0072d5f70"
    );
    for case in fixture["cases"].as_array().unwrap() {
        let folder = TempDir::new().unwrap();
        let dual = case["sources"].as_array().unwrap().len() == 2;
        let path = helper(folder.path(), dual, "CPU replay; no hardware qualification");
        let mut writer =
            SourceWriter::create(config(folder.path(), dual), Helper::inspect(&path).unwrap())
                .unwrap();
        if dual {
            accept(&mut writer, &receipt(&path)).unwrap();
        }
        for event in case["events"].as_array().unwrap() {
            accept(&mut writer, event).unwrap();
        }
        let snapshot = writer.finish();
        assert!(
            snapshot.error.is_none(),
            "{}: {:?}",
            case["name"],
            snapshot.error
        );
        assert_eq!(
            u64::from(snapshot.committed_sample),
            case["frames"].as_u64().unwrap()
        );
        assert_eq!(
            snapshot.late_samples,
            case["late_samples"].as_u64().unwrap()
        );
        for (name, expected) in case["expected"].as_object().unwrap() {
            let file = folder.path().join(format!("{name}.wav"));
            assert_eq!(
                u64::from(hound::WavReader::open(&file).unwrap().spec().channels),
                expected["channels"].as_u64().unwrap()
            );
            let retained = pcm(&file);
            assert_eq!(
                retained,
                STANDARD
                    .decode(expected["pcm16_base64"].as_str().unwrap())
                    .unwrap(),
                "{} / {name}",
                case["name"]
            );
            assert_eq!(
                format!("{:x}", Sha256::digest(&retained)),
                expected["sha256"]
            );
        }
        if dual {
            assert_eq!(
                pcm(&folder.path().join("microphone.wav")),
                pcm(&folder.path().join("microphone_clean.wav"))
            );
            let processing = snapshot.processing.unwrap();
            assert_eq!(
                processing.sha256,
                format!(
                    "{:x}",
                    Sha256::digest(
                        fs::read(folder.path().join("capture-processing.json")).unwrap()
                    )
                )
            );
            assert_eq!(processing.metadata["classification"], super::PROCESSED);
        }
    }
}

#[test]
fn processing_requires_owned_unchanged_helper_and_pre_pcm_boundary() {
    for mutation in [
        "job", "helper", "epoch", "route", "readback", "mapping", "format", "extra",
    ] {
        let folder = TempDir::new().unwrap();
        let path = helper(folder.path(), true, "fixture");
        let mut packet = receipt(&path);
        match mutation {
            "job" => packet["job_id"] = json!("f".repeat(32)),
            "helper" => packet["helper_sha256"] = json!("0".repeat(64)),
            "epoch" => packet["stream_epoch"] = json!(1),
            "route" => packet["route"]["node"] = json!("outputNode"),
            "readback" => packet["readbacks"]["input_muted"] = json!(true),
            "mapping" => packet["channel_mapping"] = json!("downmixed"),
            "format" => packet["io_formats"]["output_input"]["sample_rate"] = json!(44100),
            "extra" => packet["classification"] = json!(super::PROCESSED),
            _ => unreachable!(),
        }
        let mut writer =
            SourceWriter::create(config(folder.path(), true), Helper::inspect(&path).unwrap())
                .unwrap();
        assert!(accept(&mut writer, &packet).is_err(), "{mutation}");
        assert!(writer.snapshot.processing.is_none());
        assert!(!folder.path().join("capture-processing.json").exists());
    }
    for boundary in ["duplicate", "pcm_first", "clock_first", "raw"] {
        let folder = TempDir::new().unwrap();
        let path = helper(folder.path(), true, "fixture");
        let mut writer = SourceWriter::create(
            config(folder.path(), boundary != "raw"),
            Helper::inspect(&path).unwrap(),
        )
        .unwrap();
        match boundary {
            "duplicate" => accept(&mut writer, &receipt(&path)).unwrap(),
            "pcm_first" => {
                assert!(accept(&mut writer, &audio("microphone", 0., &[0.1], 1)).is_err())
            }
            "clock_first" => {
                assert!(accept(&mut writer, &json!({"type":"clock","time":1.})).is_err())
            }
            _ => (),
        }
        // A bad PCM event is itself terminal in run_writer; direct tests don't resume it.
        if !["pcm_first", "clock_first"].contains(&boundary) {
            assert!(accept(&mut writer, &receipt(&path)).is_err(), "{boundary}");
        }
    }
    let folder = TempDir::new().unwrap();
    let path = helper(folder.path(), false, "fixture");
    assert!(
        SourceWriter::create(config(folder.path(), true), Helper::inspect(&path).unwrap()).is_err()
    );
    assert!(!folder.path().join("audio.wav").exists());
    helper(folder.path(), true, "qualified fixture");
    fs::write(&path, "replaced helper").unwrap();
    assert!(Helper::inspect(&path).is_err());
}

#[test]
fn invalid_packets_preserve_the_valid_buffered_tail() {
    for mutation in [
        "source",
        "channels",
        "rate",
        "nan",
        "infinity",
        "incomplete",
        "base64",
        "oversize",
        "jump",
        "negative",
    ] {
        let folder = TempDir::new().unwrap();
        let path = helper(folder.path(), false, "fixture");
        let mut writer = SourceWriter::create(
            config(folder.path(), false),
            Helper::inspect(&path).unwrap(),
        )
        .unwrap();
        accept(&mut writer, &audio("microphone", 0., &[0.25; 513], 1)).unwrap();
        let mut packet = audio("microphone", 513., &[0.5], 1);
        match mutation {
            "source" => packet["source"] = json!("system"),
            "channels" => packet["channels"] = json!(2),
            "rate" => packet["rate"] = json!(48000),
            "nan" => packet = audio("microphone", 513., &[f32::NAN], 1),
            "infinity" => packet = audio("microphone", 513., &[f32::INFINITY], 1),
            "incomplete" => packet["pcm"] = json!(STANDARD.encode([0, 1, 2])),
            "base64" => packet["pcm"] = json!("?"),
            "oversize" => packet = audio("microphone", 513., &vec![0.; 16001], 1),
            "jump" => packet["time"] = json!(5.0001),
            "negative" => packet["time"] = json!(-1.),
            _ => unreachable!(),
        }
        assert!(accept(&mut writer, &packet).is_err(), "{mutation}");
        let snapshot = writer.finish();
        assert_eq!(snapshot.committed_sample, 513, "{mutation}");
        assert_eq!(
            pcm(&folder.path().join("audio.wav")),
            [0_u8, 32].repeat(513)
        );
    }
}

#[test]
fn qualified_epoch_changes_retain_tail_and_cannot_resume_as_raw() {
    for kind in ["paused", "format_changed"] {
        let folder = TempDir::new().unwrap();
        let path = helper(folder.path(), true, "fixture");
        let mut writer =
            SourceWriter::create(config(folder.path(), true), Helper::inspect(&path).unwrap())
                .unwrap();
        accept(&mut writer, &receipt(&path)).unwrap();
        accept(&mut writer, &audio("microphone", 0., &[0.25; 511], 1)).unwrap();
        assert!(
            accept(
                &mut writer,
                &json!({"type":kind,"source":"microphone","time":511./16000.})
            )
            .is_err()
        );
        let result = writer.finish();
        assert_eq!(result.committed_sample, 511);
        assert!(result.processing.is_some());
        assert_eq!(
            pcm(&folder.path().join("microphone.wav")),
            pcm(&folder.path().join("microphone_clean.wav"))
        );
    }
}

#[test]
fn recording_files_are_created_exclusively() {
    let folder = TempDir::new().unwrap();
    let path = helper(folder.path(), false, "fixture");
    fs::write(folder.path().join("audio.wav"), b"keep").unwrap();
    assert!(
        SourceWriter::create(
            config(folder.path(), false),
            Helper::inspect(&path).unwrap()
        )
        .is_err()
    );
    assert_eq!(fs::read(folder.path().join("audio.wav")).unwrap(), b"keep");
}

#[cfg(unix)]
fn assert_reaped(pid: u32) {
    assert_eq!(
        unsafe { libc::kill(pid as i32, 0) },
        -1,
        "owned helper {pid} is still alive"
    );
    assert_eq!(
        std::io::Error::last_os_error().raw_os_error(),
        Some(libc::ESRCH)
    );
}

#[tokio::test]
#[cfg(unix)]
async fn real_helper_partial_line_pause_and_stop_are_drained_and_reaped() {
    let folder = TempDir::new().unwrap();
    let path = helper(
        folder.path(),
        false,
        "#!/bin/sh\nIFS= read -r start\nprintf '%s' '{\"type\":\"record'\nIFS= read -r pause\nprintf '%s\\n' 'ing\",\"microphone\":true,\"system\":false}' '{\"type\":\"paused\",\"time\":0}'\nIFS= read -r stop\nprintf '%s\\n' '{\"type\":\"stopped\",\"time\":0}'\n",
    );
    let mut session = CaptureSupervisor::spawn_with_deadline(
        &path,
        config(folder.path(), false),
        Duration::from_secs(2),
    )
    .await
    .unwrap();
    session.command(Command::Pause).unwrap();
    tokio::time::timeout(Duration::from_secs(3), async {
        while session.updates.borrow().phase != CapturePhase::Paused {
            session.updates.changed().await.unwrap();
        }
    })
    .await
    .unwrap();
    let pid = session.pid;
    let result = session.finish().await.unwrap();
    assert_eq!(result.phase, CapturePhase::Stopped, "{:?}", result.error);
    assert_reaped(pid);
}

#[tokio::test]
#[cfg(unix)]
async fn real_helper_failure_matrix_preserves_accepted_tail_and_reaps() {
    for case in ["eof", "partial", "error", "oversize", "stall"] {
        let folder = TempDir::new().unwrap();
        let packet = audio("microphone", 0., &[0.25; 513], 1).to_string();
        let failure = match case {
            "eof" => "exit 0".to_string(),
            "partial" => "printf '%s' '{broken'".to_string(),
            "error" => {
                "printf '%s\\n' '{\"type\":\"error\",\"error\":\"fixture failure\"}'".to_string()
            }
            "oversize" => format!("printf '%s\\n' '{}'", "x".repeat(176 * 1024 + 1)),
            "stall" => "exec sleep 30".to_string(),
            _ => unreachable!(),
        };
        let script =
            format!("#!/bin/sh\nIFS= read -r start\nprintf '%s\\n' '{packet}'\n{failure}\n");
        let path = helper(folder.path(), false, &script);
        let session = CaptureSupervisor::spawn_with_deadline(
            &path,
            config(folder.path(), false),
            Duration::from_millis(250),
        )
        .await
        .unwrap();
        let pid = session.pid;
        // Start and Stop are FIFO; the helper emits its accepted audio before its failure.
        let result = tokio::time::timeout(Duration::from_secs(5), session.finish())
            .await
            .unwrap()
            .unwrap();
        assert_eq!(result.phase, CapturePhase::Failed, "{case}");
        assert_eq!(result.committed_sample, 513, "{case}: {:?}", result.error);
        assert_eq!(
            pcm(&folder.path().join("audio.wav")),
            [0_u8, 32].repeat(513),
            "{case}"
        );
        assert_reaped(pid);
    }
}

#[tokio::test]
#[cfg(unix)]
async fn inherited_stderr_cannot_delay_capture_cleanup() {
    let folder = TempDir::new().unwrap();
    let descendant_file = folder.path().join("descendant.pid");
    let script = format!(
        "#!/bin/sh\nIFS= read -r start\nsleep 30 </dev/null >/dev/null &\nprintf '%s' \"$!\" > '{}'\nprintf '%s\\n' '{{\"type\":\"stopped\",\"time\":0}}'\nexit 0\n",
        descendant_file.display()
    );
    let path = helper(folder.path(), false, &script);
    let session = CaptureSupervisor::spawn_with_deadline(
        &path,
        config(folder.path(), false),
        Duration::from_millis(250),
    )
    .await
    .unwrap();
    let pid = session.pid;
    let mut finish = Box::pin(session.finish());
    let observed = tokio::time::timeout(Duration::from_secs(2), &mut finish).await;
    let completed = observed.is_ok();
    // Only this fixture's recorded descendant is signalled, even on the red path.
    let descendant = fs::read_to_string(descendant_file)
        .unwrap()
        .parse::<i32>()
        .unwrap();
    unsafe {
        libc::kill(descendant, libc::SIGKILL);
    }
    let result = match observed {
        Ok(result) => result,
        Err(_) => tokio::time::timeout(Duration::from_secs(2), finish)
            .await
            .unwrap(),
    }
    .unwrap();
    assert_reaped(pid);
    assert!(
        completed,
        "reaped capture was still waiting for inherited stderr"
    );
    assert_eq!(result.phase, CapturePhase::Stopped, "{:?}", result.error);
}

#[tokio::test]
#[cfg(unix)]
async fn real_wav_write_failure_reaps_helper_and_retains_last_commit() {
    let folder = TempDir::new().unwrap();
    let packet = audio("microphone", 4000., &[0.5; 513], 1).to_string();
    let script = format!(
        "#!/bin/sh\nIFS= read -r start\nprintf '%s\\n' '{packet}' '{{\"type\":\"clock\",\"time\":1.0}}'\nexec sleep 30\n"
    );
    let path = helper(folder.path(), false, &script);
    let cfg = config(folder.path(), false);
    let helper = Helper::inspect(&path).unwrap();
    let mut writer = SourceWriter::create(cfg.clone(), helper.clone()).unwrap();
    accept(&mut writer, &audio("microphone", 0., &[0.25; 4000], 1)).unwrap();
    accept(&mut writer, &json!({"type":"clock","time":0.75})).unwrap();
    assert_eq!(writer.snapshot.committed_sample, 4000);
    writer.make_audio_read_only();
    let log = super::processing::private_create(&folder.path().join("capture.log")).unwrap();
    let session =
        CaptureSupervisor::start(helper, writer, log, cfg, Duration::from_secs(2)).unwrap();
    let pid = session.pid;
    let result = tokio::time::timeout(Duration::from_secs(4), session.finish())
        .await
        .unwrap()
        .unwrap();
    assert_eq!(result.phase, CapturePhase::Failed);
    assert_eq!(result.committed_sample, 4000);
    assert!(
        result
            .error
            .as_deref()
            .unwrap()
            .contains("Bad file descriptor"),
        "{:?}",
        result.error
    );
    assert_eq!(
        pcm(&folder.path().join("audio.wav")),
        [0_u8, 32].repeat(4000)
    );
    assert_reaped(pid);
}
