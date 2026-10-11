use std::path::Path;

use speakerdesk_core::error::ChildError;

/// Initialize Metal before advertising any GPU capability. Model readiness is separate.
#[cfg(target_os = "macos")]
pub fn prepare(path: Option<&Path>) -> Result<(), ChildError> {
    if let Some(path) = path {
        if !path.is_file() {
            return Err(ChildError::failed(
                format!("Metal library missing: {}", path.display()),
                false,
            ));
        }
        let path = path
            .to_str()
            .ok_or_else(|| ChildError::failed("Metal library path is not UTF-8", false))?;
        mlx_rs::metal::set_metallib_path(path).map_err(initialization_error)?;
    }
    mlx_rs::Device::set_default(&mlx_rs::Device::gpu());
    let input = mlx_rs::Array::from_slice(&[1.0_f32, 2.0, 3.0, 4.0], &[2, 2]);
    let output = input.matmul(&input).map_err(initialization_error)?;
    output.eval().map_err(initialization_error)?;
    if output.as_slice::<f32>() != [7.0, 10.0, 15.0, 22.0] {
        return Err(ChildError::failed(
            "Metal initialization returned invalid output",
            false,
        ));
    }
    Ok(())
}

#[cfg(target_os = "macos")]
fn initialization_error(error: mlx_rs::error::Exception) -> ChildError {
    ChildError::failed(format!("Metal initialization failed: {error}"), false)
}

#[cfg(not(target_os = "macos"))]
pub fn prepare(_: Option<&Path>) -> Result<(), ChildError> {
    Err(ChildError::MetalUnavailable)
}

#[cfg(all(test, target_os = "macos"))]
mod tests {
    #[test]
    fn explicit_missing_resource_fails_before_initialization() {
        let missing = std::path::Path::new("/speakerdesk-missing-test-resource/mlx.metallib");
        assert!(
            super::prepare(Some(missing))
                .unwrap_err()
                .to_string()
                .contains("Metal library missing")
        );
    }
}
