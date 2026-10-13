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

#[test]
#[cfg(target_os = "macos")]
#[ignore = "Requires the pinned native echo component built by scripts/setup_echo.py"]
fn raw_dual_capture_opens_and_preserves_partial_pause_tail() {
    let folder = TempDir::new().unwrap();
    let path = helper(
        folder.path(),
        false,
        "Raw helper protocol replay; no devices",
    );
    let mut cfg = config(folder.path(), true);
    cfg.echo_library = Some(native_echo_library());
    let mut writer = SourceWriter::create(cfg, Helper::inspect(&path).unwrap())
        .expect("raw dual-source capture must use the native echo component");
    accept(&mut writer, &json!({"type":"recording"})).unwrap();
    accept(&mut writer, &audio("microphone", 0., &[0.25; 513], 1)).unwrap();
    accept(&mut writer, &audio("system", 0., &[0.; 1026], 2)).unwrap();
    accept(&mut writer, &json!({"type":"paused","time":513./16000.})).unwrap();
    accept(&mut writer, &json!({"type":"stopped","time":513./16000.})).unwrap();
    let result = writer.finish();
    assert!(result.error.is_none(), "{:?}", result.error);
    assert_eq!(result.committed_sample, 513);
    assert!(
        result.processing.is_none(),
        "raw AEC must not acquire Apple qualification"
    );
    for name in ["audio.wav", "microphone.wav", "microphone_clean.wav"] {
        assert_eq!(
            pcm(&folder.path().join(name)),
            [0, 32].repeat(513),
            "{name}"
        );
    }
}

#[cfg(target_os = "macos")]
fn native_echo_library() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../desktop/capture/echo/libspeakerdesk_echo.dylib")
}

#[test]
#[cfg(target_os = "macos")]
#[ignore = "Requires independent v0.6.6 echo replay root in SPEAKERDESK_ECHO_REFERENCE"]
fn raw_echo_matches_independent_v066_retention() {
    use std::io::BufRead;
    let reference = PathBuf::from(
        std::env::var_os("SPEAKERDESK_ECHO_REFERENCE").expect("independent replay root"),
    );
    let manifest: Value =
        serde_json::from_slice(&fs::read(reference.join("manifest.json")).unwrap()).unwrap();
    assert_eq!(
        manifest["revision"],
        "91f7593fc9c890c69befd6663cdcdeb0072d5f70"
    );
    let library = native_echo_library();
    assert_eq!(
        format!("{:x}", Sha256::digest(fs::read(&library).unwrap())),
        manifest["library_sha256"].as_str().unwrap()
    );
    // Retain output alongside the independently generated reference for the DSP
    // comparison. This test asserts exact raw/no-echo retention, not echo quality.
    let output = tempfile::Builder::new()
        .prefix("native-")
        .tempdir_in(&reference)
        .unwrap()
        .keep();
    eprintln!("Native capture replay output: {}", output.display());
    for case in manifest["cases"].as_array().unwrap() {
        let name = case["name"].as_str().unwrap();
        let folder = output.join(name);
        fs::create_dir(&folder).unwrap();
        let path = helper(
            &folder,
            false,
            "Independent raw capture protocol replay; no devices",
        );
        let mut cfg = config(&folder, true);
        cfg.echo_library = Some(library.clone());
        let mut writer = SourceWriter::create(cfg, Helper::inspect(&path).unwrap()).unwrap();
        let events = fs::File::open(reference.join(case["events"].as_str().unwrap())).unwrap();
        for line in std::io::BufReader::new(events).lines() {
            let event: Value = serde_json::from_str(&line.unwrap()).unwrap();
            if event["type"] == "resume" {
                writer.resume().unwrap();
            } else {
                accept(&mut writer, &event).unwrap();
            }
        }
        let result = writer.finish();
        assert!(result.error.is_none(), "{name}: {:?}", result.error);
        assert_eq!(
            u64::from(result.raw_sample),
            case["samples"].as_u64().unwrap(),
            "{name}"
        );
        assert_eq!(result.committed_sample, result.raw_sample, "{name}");
        assert!(result.processing.is_none());
        assert!(result.source_catalog.is_some());
        for (track, expected) in case["expected"].as_object().unwrap() {
            let path = folder.join(format!("{track}.wav"));
            let audio = hound::WavReader::open(&path).unwrap();
            assert_eq!(
                u64::from(audio.spec().channels),
                expected["channels"].as_u64().unwrap()
            );
            assert_eq!(
                u64::from(audio.duration()),
                case["samples"].as_u64().unwrap()
            );
            if matches!(track.as_str(), "microphone" | "system" | "system_reference")
                || matches!(
                    name,
                    "no_echo" | "near_only" | "inverted_no_echo" | "partial_pause"
                )
            {
                assert_eq!(
                    format!("{:x}", Sha256::digest(pcm(&path))),
                    expected["pcm16_sha256"].as_str().unwrap(),
                    "{name}/{track}"
                );
            }
        }
    }
}

#[test]
#[cfg(unix)]
fn runtime_artifacts_reject_untrusted_files_and_missing_echo() {
    use std::os::unix::fs::PermissionsExt;
    let folder = TempDir::new().unwrap();
    let path = helper(folder.path(), true, "Owned capability replay");
    let manifest = folder.path().join("capture-helper.capability.json");
    fs::set_permissions(&manifest, fs::Permissions::from_mode(0o664)).unwrap();
    assert!(
        Helper::inspect(&path).is_err(),
        "group-writable qualification must be rejected"
    );
    let library = folder.path().join("echo.dylib");
    fs::write(&library, b"not a native library").unwrap();
    fs::set_permissions(&library, fs::Permissions::from_mode(0o664)).unwrap();
    assert!(matches!(
        speakerdesk_echo::EchoMixer::start(&library),
        Err(speakerdesk_echo::EchoError::IncompatibleVersion)
    ));
    fs::set_permissions(&library, fs::Permissions::from_mode(0o600)).unwrap();
    assert!(matches!(
        speakerdesk_echo::EchoMixer::start(&library),
        Err(speakerdesk_echo::EchoError::IncompatibleVersion)
    ));
    assert!(matches!(
        speakerdesk_echo::EchoMixer::start(&folder.path().join("missing.dylib")),
        Err(speakerdesk_echo::EchoError::Missing)
    ));
}

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
            "export_contract":EXPORT,"capability_version":"apple_input_node_output_route_v1","progress_contract":"audio_queue_fence_v1"}))
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
        echo_library: None,
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
    let mut value = value.clone();
    // PCM fixtures predate this transport envelope; their samples/clocks stay unchanged.
    if value["type"] == "clock" && value.get("producer_fence").is_none() {
        value["producer_fence"] = json!("audio_queue_fence_v1");
        value["sequence"] = json!(writer.next_fence_sequence());
    }
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
        if case["late_samples"].as_u64().unwrap() > 0 {
            // This reference records the old loss. Keep it unchanged as a negative control.
            assert!(snapshot.error.is_some());
            assert_eq!(snapshot.repaired_from_sample, Some(0));
            assert!(snapshot.derived_rebuilt);
            assert_eq!(
                u64::from(snapshot.committed_sample),
                case["frames"].as_u64().unwrap()
            );
            let event = case["events"]
                .as_array()
                .unwrap()
                .iter()
                .find(|event| event["source"] == "system")
                .unwrap();
            let original = STANDARD.decode(event["pcm"].as_str().unwrap()).unwrap();
            let original: Vec<f32> = original
                .as_chunks::<4>()
                .0
                .iter()
                .map(|bytes| f32::from_le_bytes(*bytes))
                .collect();
            let start = (event["time"].as_f64().unwrap() * 16000.).round_ties_even() as usize;
            let retained = pcm(&folder.path().join("system_reference.wav"));
            let original_prefix = speakerdesk_core::audio::pcm16_bytes(&original[..200]).unwrap();
            assert_eq!(
                &retained[start * 4..(start + 100) * 4],
                original_prefix,
                "late original source samples must survive"
            );
            let old = STANDARD
                .decode(
                    case["expected"]["system_reference"]["pcm16_base64"]
                        .as_str()
                        .unwrap(),
                )
                .unwrap();
            assert_ne!(
                &retained[start * 4..(start + 100) * 4],
                &old[start * 4..(start + 100) * 4],
                "old dropped-PCM negative control"
            );
            continue;
        }
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
fn qualified_clock_requires_a_monotonic_producer_fence() {
    let folder = TempDir::new().unwrap();
    let path = helper(folder.path(), true, "fence replay");
    let mut writer =
        SourceWriter::create(config(folder.path(), true), Helper::inspect(&path).unwrap()).unwrap();
    accept(&mut writer, &receipt(&path)).unwrap();
    assert!(
        writer
            .accept(parse_event(r#"{"type":"clock","time":0.75}"#).unwrap())
            .is_err()
    );
    let fence =
        json!({"type":"clock","time":0.75,"producer_fence":"audio_queue_fence_v1","sequence":1});
    writer
        .accept(parse_event(&fence.to_string()).unwrap())
        .unwrap();
    assert!(
        writer
            .accept(parse_event(&fence.to_string()).unwrap())
            .is_err()
    );
}

#[test]
fn late_pcm_repairs_sources_and_rebuilds_playable_audio_without_trusting_old_words() {
    for dual in [false, true] {
        let folder = TempDir::new().unwrap();
        let path = helper(folder.path(), dual, "late PCM replay");
        let mut writer =
            SourceWriter::create(config(folder.path(), dual), Helper::inspect(&path).unwrap())
                .unwrap();
        if dual {
            accept(&mut writer, &receipt(&path)).unwrap();
        }
        // Old independent clock behavior could commit silence before this source arrives.
        accept(&mut writer, &json!({"type":"clock","time":0.75})).unwrap();
        assert_eq!(writer.snapshot.raw_sample, 4000);
        let source = if dual { "system" } else { "microphone" };
        accept(&mut writer, &audio(source, 1000., &[0.25; 513], 1)).unwrap();
        assert_eq!(writer.snapshot.committed_sample, 0);
        assert_eq!(writer.snapshot.repaired_from_sample, Some(0));
        assert!(!writer.snapshot.derived_rebuilt);
        accept(&mut writer, &json!({"type":"stopped","time":0.25})).unwrap();
        let snapshot = writer.finish();
        assert!(snapshot.derived_rebuilt, "{:?}", snapshot.error);
        assert!(snapshot.error.is_some());
        assert_eq!(snapshot.late_samples, 513);
        assert_eq!(snapshot.committed_sample, 4000);
        let mut expected = vec![0u8; 4000 * 2];
        expected[1000 * 2..1513 * 2].copy_from_slice(&[0, 32].repeat(513));
        assert_eq!(pcm(&folder.path().join(format!("{source}.wav"))), expected);
        assert_eq!(pcm(&folder.path().join("audio.wav")), expected);
    }
}

#[test]
fn failed_rebuild_truncates_stale_processed_audio_and_retains_original_packet_journal() {
    let folder = TempDir::new().unwrap();
    let path = helper(folder.path(), false, "repair failure replay");
    let mut writer = SourceWriter::create(
        config(folder.path(), false),
        Helper::inspect(&path).unwrap(),
    )
    .unwrap();
    accept(&mut writer, &json!({"type":"clock","time":0.75})).unwrap();
    accept(&mut writer, &audio("microphone", 1000., &[0.25; 513], 1)).unwrap();
    // Force only derived publication to fail; corrected originals must remain readable.
    fs::write(folder.path().join("audio.repair.wav"), b"occupied").unwrap();
    let snapshot = writer.finish();
    assert!(!snapshot.derived_rebuilt);
    assert_eq!(snapshot.committed_sample, 0);
    assert!(
        snapshot
            .error
            .as_deref()
            .unwrap()
            .contains("repair did not finish")
    );
    assert!(folder.path().join("capture-late-pcm.jsonl").exists());
    assert!(pcm(&folder.path().join("audio.wav")).is_empty());
    assert_eq!(
        &pcm(&folder.path().join("microphone.wav"))[2000..3026],
        &[0, 32].repeat(513)
    );
}

#[test]
fn late_pcm_and_raw_write_failure_still_invalidate_old_derived_audio() {
    let folder = TempDir::new().unwrap();
    let path = helper(folder.path(), false, "late PCM plus raw write failure");
    let mut writer = SourceWriter::create(
        config(folder.path(), false),
        Helper::inspect(&path).unwrap(),
    )
    .unwrap();
    accept(&mut writer, &json!({"type":"clock","time":0.75})).unwrap();
    accept(&mut writer, &audio("microphone", 1000., &[0.25; 513], 1)).unwrap();
    writer.make_file_read_only("microphone");
    accept(&mut writer, &audio("microphone", 4000., &[0.25; 513], 1)).unwrap();
    assert!(accept(&mut writer, &json!({"type":"stopped","time":4513./16000.})).is_err());
    let snapshot = writer.finish();
    assert!(!snapshot.derived_rebuilt);
    assert!(snapshot.error.is_some());
    assert!(pcm(&folder.path().join("audio.wav")).is_empty());
    assert!(folder.path().join("capture-late-pcm.jsonl").exists());
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
fn qualified_format_changes_retain_tail_and_cannot_resume_as_raw() {
    let folder = TempDir::new().unwrap();
    let path = helper(folder.path(), true, "fixture");
    let mut writer =
        SourceWriter::create(config(folder.path(), true), Helper::inspect(&path).unwrap()).unwrap();
    accept(&mut writer, &receipt(&path)).unwrap();
    accept(&mut writer, &audio("microphone", 0., &[0.25; 511], 1)).unwrap();
    assert!(
        accept(
            &mut writer,
            &json!({"type":"format_changed","source":"microphone","time":511./16000.})
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

#[test]
fn qualified_resume_requires_owned_arm_and_exact_fresh_observation() {
    for case in [
        "unsolicited",
        "missing",
        "duplicate",
        "job",
        "route",
        "format",
        "readback",
        "epoch",
        "delayed-recording",
    ] {
        let folder = TempDir::new().unwrap();
        let path = helper(folder.path(), true, "qualified protocol replay");
        let mut writer =
            SourceWriter::create(config(folder.path(), true), Helper::inspect(&path).unwrap())
                .unwrap();
        let original = receipt(&path);
        accept(&mut writer, &original).unwrap();
        accept(&mut writer, &json!({"type":"recording"})).unwrap();
        accept(&mut writer, &audio("microphone", 0., &[0.25; 511], 1)).unwrap();
        accept(&mut writer, &json!({"type":"paused","time":511./16000.})).unwrap();
        let persisted = fs::read(folder.path().join("capture-processing.json")).unwrap();
        if case == "unsolicited" {
            assert!(accept(&mut writer, &original).is_err());
            assert!(accept(&mut writer, &audio("microphone", 511., &[0.5; 17], 1)).is_err());
        } else {
            writer.resume().unwrap();
            if case == "missing" {
                assert!(accept(&mut writer, &json!({"type":"recording"})).is_err());
                assert!(accept(&mut writer, &audio("microphone", 511., &[0.5; 17], 1)).is_err());
            } else if case == "delayed-recording" {
                accept(&mut writer, &original).unwrap();
                // A qualified mic may flow while SCStream.startCapture still awaits.
                for index in 0..28 {
                    let start = 511 + index * 4000;
                    accept(
                        &mut writer,
                        &audio("microphone", f64::from(start), &[0.25; 4000], 1),
                    )
                    .unwrap();
                    accept(
                        &mut writer,
                        &json!({"type":"clock","time":f64::from(start+4000)/16000.}),
                    )
                    .unwrap();
                }
                assert_eq!(writer.snapshot.phase, CapturePhase::Resuming);
                assert!(writer.snapshot.committed_sample >= 104_000);
                accept(&mut writer, &json!({"type":"recording"})).unwrap();
                accept(
                    &mut writer,
                    &json!({"type":"stopped","time":112511./16000.}),
                )
                .unwrap();
            } else {
                let mut fresh = original.clone();
                match case {
                    "duplicate" => {
                        accept(&mut writer, &fresh).unwrap();
                    }
                    "job" => fresh["job_id"] = json!("aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"),
                    "route" => fresh["route"]["bus"] = json!(1),
                    "format" => fresh["callback_format"]["sample_rate"] = json!(16000),
                    "readback" => fresh["readbacks"]["agc_enabled"] = json!(true),
                    "epoch" => fresh["stream_epoch"] = json!(1),
                    _ => unreachable!(),
                }
                assert!(accept(&mut writer, &fresh).is_err(), "{case}");
            }
        }
        let finished = writer.finish();
        assert_eq!(
            finished.committed_sample,
            if case == "delayed-recording" {
                112511
            } else {
                511
            },
            "{case}"
        );
        assert_eq!(
            fs::read(folder.path().join("capture-processing.json")).unwrap(),
            persisted
        );
        let samples = pcm(&folder.path().join("microphone_clean.wav"));
        assert_eq!(&samples[..1022], &[0, 32].repeat(511));
    }
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
async fn qualified_pause_resume_requalifies_same_graph_and_keeps_exact_tails() {
    // Protocol replay only: the helper owns no devices. Each resume repeats the
    // same freshly observed producer contract before any resumed PCM.
    let folder = TempDir::new().unwrap();
    let directory = folder.path().display();
    let script = format!(
        "#!/bin/sh\nIFS= read -r start\ncat '{directory}/receipt.json' '{directory}/stage-0.jsonl'\nstage=0\nwhile IFS= read -r command; do\ncase \"$command\" in\n*pause*) cat '{directory}/paused-'\"$stage\"'.jsonl' ;;\n*resume*) stage=$((stage+1)); cat '{directory}/receipt.json' '{directory}/stage-'\"$stage\"'.jsonl' ;;\n*stop*) cat '{directory}/stopped-'\"$stage\"'.jsonl'; exit 0 ;;\nesac\ndone\n"
    );
    let path = helper(folder.path(), true, &script);
    fs::write(
        folder.path().join("receipt.json"),
        format!("{}\n", receipt(&path)),
    )
    .unwrap();
    let mut endpoint = 0;
    let mut expected = Vec::new();
    let mut expected_stereo = Vec::new();
    for (stage, (count, sample)) in [(511, 0.25_f32), (513, 0.5), (17, -0.25)]
        .into_iter()
        .enumerate()
    {
        let microphone = vec![sample; count];
        let stereo: Vec<_> = (0..count).flat_map(|_| [sample, -sample]).collect();
        let lines = [
            json!({"type":"recording"}),
            audio("microphone", endpoint as f64, &microphone, 1),
            audio("system", endpoint as f64, &stereo, 2),
        ];
        fs::write(
            folder.path().join(format!("stage-{stage}.jsonl")),
            lines
                .iter()
                .map(|line| format!("{line}\n"))
                .collect::<String>(),
        )
        .unwrap();
        endpoint += count;
        expected.extend(((sample * 32768.) as i16).to_le_bytes().repeat(count));
        for _ in 0..count {
            expected_stereo.extend(((sample * 32768.) as i16).to_le_bytes());
            expected_stereo.extend(((-sample * 32768.) as i16).to_le_bytes());
        }
        for kind in ["paused", "stopped"] {
            fs::write(
                folder.path().join(format!("{kind}-{stage}.jsonl")),
                format!("{}\n", json!({"type":kind,"time":endpoint as f64/16000.})),
            )
            .unwrap();
        }
    }
    let mut session = CaptureSupervisor::spawn_with_deadline(
        &path,
        config(folder.path(), true),
        Duration::from_secs(2),
    )
    .await
    .unwrap();
    let pid = session.pid;
    let mut failure = None;
    let mut initial = None;
    for stage in 0..3 {
        for phase in [CapturePhase::Recording, CapturePhase::Paused] {
            if phase == CapturePhase::Paused {
                session.command(Command::Pause).unwrap();
            }
            let observed = tokio::time::timeout(Duration::from_secs(3), async {
                loop {
                    let state = session.updates.borrow().clone();
                    if state.phase == phase || state.error.is_some() {
                        break state;
                    }
                    session.updates.changed().await.unwrap();
                }
            })
            .await
            .unwrap();
            if observed.error.is_some() {
                failure = observed.error;
                break;
            }
            if phase == CapturePhase::Paused {
                assert_eq!(observed.committed_sample, [511, 1024, 1041][stage]);
                let current = fs::read(folder.path().join("capture-processing.json")).unwrap();
                if let Some(initial) = &initial {
                    assert_eq!(&current, initial);
                } else {
                    initial = Some(current);
                }
            }
        }
        if failure.is_some() {
            break;
        }
        if stage < 2 {
            session.command(Command::Resume).unwrap();
        }
    }
    let result = session.finish().await.unwrap();
    assert_reaped(pid);
    assert!(
        failure.is_none(),
        "qualified Pause/Resume failed: {failure:?}"
    );
    assert_eq!(result.phase, CapturePhase::Stopped, "{:?}", result.error);
    assert_eq!(result.committed_sample, 1041);
    assert_eq!(result.late_samples, 0);
    for name in ["microphone.wav", "microphone_clean.wav"] {
        assert_eq!(pcm(&folder.path().join(name)), expected);
    }
    assert_eq!(
        pcm(&folder.path().join("system_reference.wav")),
        expected_stereo
    );
    assert_eq!(pcm(&folder.path().join("system.wav")), expected);
    let mixed: Vec<_> = [(511, 16384_i16), (513, 32767), (17, -16384)]
        .into_iter()
        .flat_map(|(count, value)| value.to_le_bytes().repeat(count))
        .collect();
    assert_eq!(pcm(&folder.path().join("audio.wav")), mixed);
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
    let writer_config = cfg.clone();
    let writer_helper = helper.clone();
    let make_writer = move || {
        let mut writer = SourceWriter::create(writer_config, writer_helper)?;
        accept(&mut writer, &audio("microphone", 0., &[0.25; 4000], 1))?;
        accept(&mut writer, &json!({"type":"clock","time":0.75}))?;
        assert_eq!(writer.snapshot.committed_sample, 4000);
        writer.make_audio_read_only();
        Ok(writer)
    };
    let log = super::processing::private_create(&folder.path().join("capture.log")).unwrap();
    let session = CaptureSupervisor::start(helper, make_writer, log, cfg, Duration::from_secs(2))
        .await
        .unwrap();
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
