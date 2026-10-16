//! Bounded simulated-room DSP diagnostic, not speaker/headset acoustic qualification.
//! No playback, recording, ASR, learned room, lag search or fitted output gain.
//! Run the same frozen test on both platforms, with a fresh output directory:
//! SPEAKERDESK_TEST_ECHO=/absolute/native/library \
//! SPEAKERDESK_ECHO_QUALITY_FIXTURES=/private/voice-ordinary-v1 \
//! SPEAKERDESK_ECHO_QUALITY_OUTPUT=/private/fresh-echo-quality \
//! cargo test --locked -p speakerdesk-host --test echo_quality -- --ignored --nocapture
#![cfg(any(target_os = "macos", windows))]

use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use speakerdesk_echo::{EchoMixer, RATE};
use std::{fs, io::Write, path::Path};

const MANIFEST: &str = "85df77b04cfb1330a5859c70cd24d27a86466beddfd74b0d28fd97df8d57f635";
const FAR: (&str, &str, usize) = (
    "4077-13754-0003",
    "c7a783b45d4fac62025b17c4f43f8b72bfc32b09eb3ad4c3d57ceda4bfc76c49",
    90880,
);
const NEAR: (&str, &str, usize) = (
    "5142-33396-0063",
    "1a0efbcb86807b14d1667d91fcfb38502a8b380ee1c8c0e3448db8b29bad82bf",
    40240,
);
const SAMPLES: usize = 320073;
const EXCLUDE: usize = 64000;
const GAP: usize = 3200;
const RMS: f64 = 0.08;
const FIR: [(usize, f32); 4] = [(1120, 0.55), (1200, -0.15), (1376, 0.08), (1696, 0.04)];
const TILES: [usize; 6] = [159, 4000, 513, 16000, 17, 511];

fn sha(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}

fn write_new(path: &Path, bytes: &[u8]) {
    let mut file = fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(path)
        .unwrap();
    file.write_all(bytes).unwrap();
}

fn raw(directory: &Path, name: &str, values: impl IntoIterator<Item = f32>) -> Value {
    let bytes: Vec<u8> = values.into_iter().flat_map(f32::to_le_bytes).collect();
    write_new(&directory.join(name), &bytes);
    json!({"file":name,"dtype":"little_endian_float32","bytes":bytes.len(),"sha256":sha(&bytes)})
}

fn source(root: &Path, output: &Path, manifest: &Value, pin: (&str, &str, usize)) -> Vec<f32> {
    let case = manifest["cases"]
        .as_array()
        .unwrap()
        .iter()
        .find(|case| case["id"] == pin.0)
        .unwrap();
    let basename = format!("{}.wav", pin.1);
    assert_eq!(case["path"], format!("audio/{basename}"));
    assert_eq!(case["wav_sha256"], pin.1);
    assert_eq!(case["start_sample"], 0);
    assert_eq!(case["end_sample"], pin.2);
    let bytes = fs::read(root.join("audio").join(&basename)).unwrap();
    assert_eq!(sha(&bytes), pin.1);
    write_new(&output.join(basename), &bytes);
    let mut wav = hound::WavReader::new(std::io::Cursor::new(bytes)).unwrap();
    assert_eq!(wav.spec().channels, 1);
    assert_eq!(wav.spec().sample_rate, RATE as u32);
    assert_eq!(wav.spec().bits_per_sample, 16);
    assert_eq!(wav.spec().sample_format, hound::SampleFormat::Int);
    let pcm: Vec<f32> = wav
        .samples::<i16>()
        .map(|sample| f32::from(sample.unwrap()) / 32768.)
        .collect();
    assert_eq!(pcm.len(), pin.2);
    // One fixed full-clip RMS normalization; no normalization of DSP output.
    let energy = pcm.iter().map(|&v| f64::from(v).powi(2)).sum::<f64>();
    assert!(energy > 0.);
    let gain = RMS / (energy / pcm.len() as f64).sqrt();
    (0..SAMPLES)
        .map(|index| {
            let position = index % (pcm.len() + GAP);
            pcm.get(position)
                .map_or(0., |&v| (f64::from(v) * gain) as f32)
        })
        .collect()
}

#[test]
#[ignore = "requires pinned public voices and the actual native EchoMixer component"]
fn native_echo_quality_on_frozen_simulated_room() {
    let env = |name| std::path::PathBuf::from(std::env::var_os(name).expect(name));
    let fixture = env("SPEAKERDESK_ECHO_QUALITY_FIXTURES");
    let output = env("SPEAKERDESK_ECHO_QUALITY_OUTPUT");
    let library = env("SPEAKERDESK_TEST_ECHO");
    let manifest_bytes = fs::read(fixture.join("ordinary-cases.json")).unwrap();
    assert_eq!(sha(&manifest_bytes), MANIFEST);
    let manifest: Value = serde_json::from_slice(&manifest_bytes).unwrap();
    // Existing paths are refused, including empty directories. Inputs are never rewritten.
    fs::create_dir(&output).expect("SPEAKERDESK_ECHO_QUALITY_OUTPUT must be a fresh directory");
    write_new(&output.join("ordinary-cases.json"), &manifest_bytes);
    let recipe = json!({
        "schema_version":1,"kind":"fixed_public_voice_simulated_room_diagnostic_v1",
        "source_manifest_sha256":MANIFEST,"far":{"id":FAR.0,"wav_sha256":FAR.1,"samples":FAR.2},
        "near":{"id":NEAR.0,"wav_sha256":NEAR.1,"samples":NEAR.2},
        "sample_rate":RATE,"samples":SAMPLES,"source_rms":RMS,"repeat_zero_gap_samples":GAP,
        "reference":"left=far,right=0","system":"0.5*far",
        "echo_fir_delay_samples_and_gain":FIR,"tile_samples":TILES,
        "cases":["far_only","local_only","double_talk"],
        "double_talk_near_start_sample":EXCLUDE,"measurement_start_sample":EXCLUDE,
        "near_phase":"source period follows absolute sample index, including muted double-talk warmup",
        "measurement_end_sample":SAMPLES,"alignment":"original sample coordinates; no fitted lag or gain",
        "metrics":"error_reduction_db=10*log10(sum((microphone-near)^2)/sum((clean-near)^2)); near_error_nrms=sqrt(sum((clean-near)^2)/sum(near^2)); energy floor 1e-30 only for log10",
        "gates":{"all":"exact lengths, finite output, bit-exact system",
            "local_only":"bit-exact clean near at every sample",
            "far_only_min_echo_reduction_db":6.0,
            "double_talk":"output-target error energy < input-target error energy; target NRMS < 1"},
        "limit":"Simulated FIR diagnostic only. No physical speaker, microphone, headset, room or acoustic-quality qualification."
    });
    let recipe_bytes = serde_json::to_vec_pretty(&recipe).unwrap();
    write_new(&output.join("recipe.json"), &recipe_bytes);
    let far_source = source(&fixture, &output, &manifest, FAR);
    let near_source = source(&fixture, &output, &manifest, NEAR);
    let mut results = Vec::new();
    for name in ["far_only", "local_only", "double_talk"] {
        let directory = output.join(name);
        fs::create_dir(&directory).unwrap();
        let far = if name == "local_only" {
            vec![0.; SAMPLES]
        } else {
            far_source.clone()
        };
        let near: Vec<f32> = near_source
            .iter()
            .enumerate()
            .map(|(index, &v)| {
                if name == "far_only" || (name == "double_talk" && index < EXCLUDE) {
                    0.
                } else {
                    v
                }
            })
            .collect();
        let echo: Vec<f32> = (0..SAMPLES)
            .map(|index| {
                FIR.iter().fold(0., |sum, &(delay, gain)| {
                    sum + index.checked_sub(delay).map_or(0., |i| far[i] * gain)
                })
            })
            .collect();
        let microphone: Vec<f32> = near.iter().zip(&echo).map(|(a, b)| a + b).collect();
        let system: Vec<f32> = far.iter().map(|v| 0.5 * v).collect();
        let reference: Vec<[f32; 2]> = far.iter().map(|&v| [v, 0.]).collect();
        assert!(
            microphone
                .iter()
                .chain(&far)
                .all(|v| v.is_finite() && v.abs() < 1.),
            "recipe must not invoke clamping"
        );
        let inputs = json!({
            "near_target":raw(&directory,"near.f32",near.iter().copied()),
            "echo":raw(&directory,"echo.f32",echo.iter().copied()),
            "microphone":raw(&directory,"microphone.f32",microphone.iter().copied()),
            "system":raw(&directory,"system.f32",system.iter().copied()),
            "reference":raw(&directory,"reference-stereo.f32",reference.iter().flatten().copied())
        });
        let mut mixer = EchoMixer::start(&library).unwrap();
        let (mut clean, mut original_system) = (Vec::new(), Vec::new());
        let (mut position, mut tile) = (0, 0);
        while position < SAMPLES {
            let end = (position + TILES[tile % TILES.len()]).min(SAMPLES);
            let processed = mixer
                .add(
                    &microphone[position..end],
                    &system[position..end],
                    &reference[position..end],
                    false,
                )
                .unwrap();
            clean.extend(processed.clean);
            original_system.extend(processed.system);
            position = end;
            tile += 1;
        }
        let tail = mixer.add(&[], &[], &[], true).unwrap();
        clean.extend(tail.clean);
        original_system.extend(tail.system);
        let artifacts = json!({"clean":raw(&directory,"clean.f32",clean.iter().copied()),
            "returned_system":raw(&directory,"returned-system.f32",original_system.iter().copied())});
        let geometry = clean.len() == SAMPLES && original_system.len() == SAMPLES;
        let finite = clean.iter().chain(&original_system).all(|v| v.is_finite());
        let system_exact = original_system
            .iter()
            .map(|v| v.to_bits())
            .eq(system.iter().map(|v| v.to_bits()));
        let near_exact = clean
            .iter()
            .map(|v| v.to_bits())
            .eq(near.iter().map(|v| v.to_bits()));
        let mut input_error = 0.;
        let mut output_error = 0.;
        let mut target_energy = 0.;
        let mut maximum = 0_f64;
        if geometry && finite {
            for index in EXCLUDE..SAMPLES {
                let target = f64::from(near[index]);
                input_error += (f64::from(microphone[index]) - target).powi(2);
                let error = f64::from(clean[index]) - target;
                output_error += error * error;
                target_energy += target * target;
                maximum = maximum.max(error.abs());
            }
        }
        let reduction_db = 10. * (input_error.max(1e-30) / output_error.max(1e-30)).log10();
        let nrms = (target_energy > 0.).then(|| (output_error / target_energy).sqrt());
        let quality = match name {
            "far_only" => reduction_db >= 6.,
            "local_only" => near_exact,
            _ => output_error < input_error && nrms.is_some_and(|value| value < 1.),
        };
        results.push(
            json!({"case":name,"inputs":inputs,"outputs":artifacts,"geometry_exact":geometry,
            "finite":finite,"system_bitexact":system_exact,"near_bitexact":near_exact,
            "measurement_samples":SAMPLES-EXCLUDE,"input_error_energy":input_error,
            "output_error_energy":output_error,"target_energy":target_energy,
            "error_reduction_db":reduction_db,"near_error_nrms":nrms,"near_error_max_abs":maximum,
            "passed":geometry && finite && system_exact && quality}),
        );
    }
    let report = json!({"recipe":recipe,"recipe_sha256":sha(&recipe_bytes),
        "diagnostic_source_sha256":sha(include_bytes!("echo_quality.rs")),
        "platform":std::env::consts::OS,"architecture":std::env::consts::ARCH,
        "native_library_path":library.canonicalize().unwrap(),"native_library_sha256":sha(&fs::read(&library).unwrap()),
        "cases":results});
    let report_bytes = serde_json::to_vec_pretty(&report).unwrap();
    write_new(&output.join("report.json"), &report_bytes);
    eprintln!(
        "{}\nEcho quality report: {}",
        String::from_utf8(report_bytes).unwrap(),
        output.join("report.json").display()
    );
    assert!(
        report["cases"]
            .as_array()
            .unwrap()
            .iter()
            .all(|case| case["passed"] == true),
        "frozen simulated-room diagnostic failed; retain its report"
    );
}
