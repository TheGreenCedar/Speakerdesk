#![cfg(target_os = "macos")]

use speakerdesk_models::ctc::network::Network;
use std::{ffi::CString, path::PathBuf, sync::atomic::AtomicBool};

/// Validation only: capture changes latency and memory and must run separately.
#[test]
#[ignore = "requires pinned CTC, MTL_CAPTURE_ENABLED=1 and a fresh SPEAKERDESK_GPU_TRACE path"]
fn ctc_emits_metal_commands() {
    speakerdesk_models::metallib::prepare(None).unwrap();
    mlx_rs::memory::set_cache_limit(0).unwrap();
    let mut network = Network::load(&PathBuf::from(
        std::env::var_os("SPEAKERDESK_TEST_CTC").expect("model directory"),
    ))
    .unwrap();
    let mut reader = hound::WavReader::open(
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/data/two-voice.wav"),
    )
    .unwrap();
    let pcm: Vec<f32> = reader
        .samples::<i16>()
        .take(720)
        .map(|s| f32::from(s.unwrap()) / 32768.)
        .collect();
    let cancelled = AtomicBool::new(false);
    let warm = network.scores(&pcm, &[0, 4, 4328], &cancelled).unwrap();
    let traced = capture(|| network.scores(&pcm, &[0, 4, 4328], &cancelled).unwrap());
    assert_eq!(warm, traced);
    assert_eq!(traced.valid_frames, 2);
}

#[test]
#[ignore = "requires pinned voice model, MTL_CAPTURE_ENABLED=1 and a fresh SPEAKERDESK_GPU_TRACE path"]
fn redimnet_emits_metal_commands() {
    use speakerdesk_models::redimnet::coreml::{ComputePolicy, Network};
    speakerdesk_models::metallib::prepare(None).unwrap();
    let network = Network::load(
        &PathBuf::from(std::env::var_os("SPEAKERDESK_TEST_VOICE").expect("model directory")),
        ComputePolicy::CpuAndGpu,
    )
    .unwrap();
    let mut reader = hound::WavReader::open(
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/data/two-voice.wav"),
    )
    .unwrap();
    let pcm: Vec<f32> = reader
        .samples::<i16>()
        .take(64000)
        .map(|s| f32::from(s.unwrap()) / 32768.)
        .collect();
    let cancelled = AtomicBool::new(false);
    let warm = network.embed(&pcm, &cancelled).unwrap();
    let traced = capture(|| network.embed(&pcm, &cancelled).unwrap());
    let after = network.embed(&pcm, &cancelled).unwrap();
    let cosine = |other: &speakerdesk_core::voice::ClipEmbedding| {
        warm.vector
            .iter()
            .zip(&other.vector)
            .map(|(a, b)| a * b)
            .sum::<f64>()
    };
    eprintln!(
        "Core ML captured cosine {}, after capture cosine {}",
        cosine(&traced),
        cosine(&after)
    );
    assert!(cosine(&traced) >= 0.9999 && cosine(&after) >= 0.9999);
    assert!(traced.clean);
}

fn capture<T>(predict: impl FnOnce() -> T) -> T {
    let path = PathBuf::from(std::env::var_os("SPEAKERDESK_GPU_TRACE").expect("trace path"));
    assert!(path.is_absolute() && !path.exists());
    assert_eq!(std::env::var("MTL_CAPTURE_ENABLED").unwrap(), "1");
    let stream = mlx_rs::Stream::gpu();
    // SAFETY: both APIs borrow live stream/path values for the duration of the call.
    unsafe {
        assert_eq!(mlx_sys::mlx_synchronize(stream.as_ptr()), 0);
        let c_path = CString::new(path.to_str().unwrap()).unwrap();
        assert_eq!(mlx_sys::mlx_metal_start_capture(c_path.as_ptr()), 0);
    }
    struct Capture;
    impl Drop for Capture {
        fn drop(&mut self) {
            // SAFETY: this test alone owns the successfully started process capture.
            let status = unsafe { mlx_sys::mlx_metal_stop_capture() };
            if !std::thread::panicking() {
                assert_eq!(status, 0);
            }
        }
    }
    let capture = Capture;
    let traced = predict();
    // SAFETY: the stream remains alive until synchronization returns.
    assert_eq!(unsafe { mlx_sys::mlx_synchronize(stream.as_ptr()) }, 0);
    drop(capture);
    assert!(path.is_dir());
    assert!(std::fs::metadata(path.join("capture")).unwrap().len() > 0);
    traced
}
