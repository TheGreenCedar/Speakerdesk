#![cfg(windows)]

use speakerdesk_core::error::ChildError;
use speakerdesk_models::{
    mossformer2::network::Network, operation::Operation, windows_runtime::Backend,
};
use std::{
    fs,
    path::Path,
    path::PathBuf,
    time::{Duration, Instant},
};

fn reference(root: &Path, record: &serde_json::Value, samples: usize) -> Vec<f32> {
    let bytes = fs::read(root.join(record["path"].as_str().unwrap())).unwrap();
    assert_eq!(speakerdesk_core::memo::sha256_hex(&bytes), record["sha256"]);
    assert_eq!(bytes.len(), samples * 4);
    bytes
        .chunks_exact(4)
        .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
        .collect()
}

#[test]
#[ignore = "Requires SPEAKERDESK_TEST_MOSS and SPEAKERDESK_TEST_WINDOWS_RUNTIME on a DirectML device"]
fn cancellation_interrupts_a_native_graph_and_the_next_request_can_run() {
    let model = PathBuf::from(std::env::var_os("SPEAKERDESK_TEST_MOSS").expect("model directory"));
    let runtime = PathBuf::from(
        std::env::var_os("SPEAKERDESK_TEST_WINDOWS_RUNTIME").expect("runtime directory"),
    );
    let backend = Backend::from_directory(&runtime).unwrap();
    speakerdesk_models::windows_runtime::verify_model(&model, "mossformer2-16k").unwrap();
    let mut graph = backend
        .load_without_memory_arena(&model.join("flash-00.onnx"))
        .unwrap();
    let input =
        ort::value::Tensor::from_array(([1, 48999, 512], vec![0.01_f32; 48999 * 512])).unwrap();
    let operation = Operation::native().unwrap();
    let options = operation.run_options().unwrap();
    std::thread::scope(|scope| {
        let cancel = scope.spawn(|| {
            std::thread::sleep(Duration::from_millis(50));
            operation.cancel();
        });
        let started = Instant::now();
        let result = graph.run_with_options(ort::inputs!["x" => &input], options);
        let elapsed = started.elapsed();
        cancel.join().unwrap();
        assert!(result.is_err(), "native graph did not observe termination");
        assert!(
            elapsed >= Duration::from_millis(50),
            "cancellation preceded native execution; this did not exercise an in-flight call"
        );
        eprintln!("Native graph termination returned after {elapsed:?}");
    });
    assert_eq!(operation.check().unwrap_err(), ChildError::Cancelled);
    let recovery = Operation::native().unwrap();
    let output = graph
        .run_with_options(ort::inputs!["x" => &input], recovery.run_options().unwrap())
        .unwrap();
    let (shape, data) = output["x_out"].try_extract_tensor::<f32>().unwrap();
    assert_eq!(shape.as_ref(), &[1, 48999, 512]);
    assert!(data.iter().all(|value| value.is_finite()));
}

#[test]
#[ignore = "Requires SPEAKERDESK_TEST_MOSS, SPEAKERDESK_TEST_MOSS_REFERENCES and SPEAKERDESK_TEST_WINDOWS_RUNTIME"]
fn windows_separation_preserves_full_windows_and_cancelled_state_is_not_reused() {
    let model = PathBuf::from(std::env::var_os("SPEAKERDESK_TEST_MOSS").expect("model directory"));
    let runtime = PathBuf::from(
        std::env::var_os("SPEAKERDESK_TEST_WINDOWS_RUNTIME").expect("runtime directory"),
    );
    let references = PathBuf::from(
        std::env::var_os("SPEAKERDESK_TEST_MOSS_REFERENCES")
            .expect("independent raw reference directory"),
    );
    let manifest = fs::read(references.join("references.json")).unwrap();
    // Sealed raw upstream forwards: original source/checkpoint and every waveform hash are included.
    assert_eq!(
        speakerdesk_core::memo::sha256_hex(&manifest),
        "d3a60ce3f6bfd11dc59efe9dca384ad2150b3fceaff852aadbc4c6d41795a1bd"
    );
    let manifest: serde_json::Value = serde_json::from_slice(&manifest).unwrap();
    let cases = manifest["cases"].as_array().unwrap();
    assert_eq!(cases.len(), 2);
    let backend = Backend::from_directory(&runtime).unwrap();
    let started = Instant::now();
    let mut network = Network::load(&model, &backend).unwrap();
    eprintln!(
        "MossFormer Windows graph validation: {:?}",
        started.elapsed()
    );
    for case in cases {
        let samples = case["samples"].as_u64().unwrap() as usize;
        let pcm = reference(&references, &case["input"], samples);
        let operation = Operation::native().unwrap();
        let started = Instant::now();
        eprintln!("MossFormer Windows {}: encoder", case["id"]);
        let mut session = network.begin(&pcm, &operation).unwrap();
        for phase in 0..48 {
            eprintln!("MossFormer Windows {}: stage {phase}", case["id"]);
            assert!(network.step(&mut session, &operation).unwrap().is_none());
        }
        eprintln!("MossFormer Windows {}: finish", case["id"]);
        let channels = network.step(&mut session, &operation).unwrap().unwrap();
        assert!(network.step(&mut session, &operation).is_err());
        for (index, actual) in channels.iter().enumerate() {
            let expected = reference(&references, &case["outputs"][index], samples);
            assert_eq!(actual.len(), samples);
            assert!(actual.iter().all(|v| v.is_finite()));
            let (mut maximum, mut difference_energy, mut expected_energy) = (0_f64, 0_f64, 0_f64);
            for (&actual, &expected) in actual.iter().zip(&expected) {
                let difference = f64::from(actual) - f64::from(expected);
                maximum = maximum.max(difference.abs());
                difference_energy += difference * difference;
                expected_energy += f64::from(expected).powi(2);
            }
            let normalized_rms = (difference_energy / expected_energy.max(1e-20)).sqrt();
            eprintln!(
                "MossFormer Windows {} channel {index}: max {maximum}, normalized RMS {normalized_rms}, full window {:?}",
                case["id"],
                started.elapsed()
            );
            assert!(
                maximum <= 2e-3 && normalized_rms <= 2e-3,
                "original independent waveform bounds exceeded"
            );
        }
        let mut interrupted = network.begin(&pcm, &Operation::native().unwrap()).unwrap();
        let cancelled = Operation::native().unwrap();
        cancelled.cancel();
        assert_eq!(
            network.step(&mut interrupted, &cancelled).unwrap_err(),
            ChildError::Cancelled
        );
        assert!(
            network
                .step(&mut interrupted, &Operation::native().unwrap())
                .is_err()
        );
    }
}
