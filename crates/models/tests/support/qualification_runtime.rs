//! Runtime identity for the prospective native-model captures.
#[cfg(target_os = "macos")]
pub fn prepare_mlx() -> serde_json::Value {
    use speakerdesk_core::memo::sha256_hex;
    use std::{ffi::CStr, path::PathBuf};
    unsafe extern "C" {
        fn _dyld_image_count() -> u32;
        fn _dyld_get_image_name(index: u32) -> *const std::ffi::c_char;
    }
    let directory =
        PathBuf::from(std::env::var_os("SPEAKERDESK_TEST_MLX").expect("pinned MLX directory"));
    let kernels = directory.join("mlx.metallib");
    let kernel_hash = sha256_hex(&std::fs::read(&kernels).unwrap());
    assert_eq!(
        kernel_hash,
        "2d2e729831a995a39778164c93ea1b2e8d87f7d8b1e717386d3d816f46f416b2"
    );
    speakerdesk_models::metallib::prepare(Some(&kernels)).unwrap();
    mlx_rs::memory::set_cache_limit(0).unwrap();
    let images: Vec<_> = unsafe {
        (0.._dyld_image_count())
            .map(|i| PathBuf::from(CStr::from_ptr(_dyld_get_image_name(i)).to_str().unwrap()))
            .filter(|p| p.file_name().is_some_and(|n| n == "libmlx.dylib"))
            .collect()
    };
    assert_eq!(images.len(), 1, "exactly one loaded MLX runtime");
    let runtime_hash = sha256_hex(&std::fs::read(&images[0]).unwrap());
    assert_eq!(
        runtime_hash,
        "0c9db159aedd2951bb77610c8b028c4ef79bc64ccc4c73b07a4452a60612ac03"
    );
    serde_json::json!({"libmlx.dylib":runtime_hash,"mlx.metallib":kernel_hash})
}
