use std::path::Path;

use speakerdesk_core::error::ChildError;

/// The macOS child calls `mlx_rs::metal::set_metallib_path` before any MLX use.
pub fn prepare(path: Option<&Path>) -> Result<(), ChildError> {
    if cfg!(not(target_os = "macos")) {
        let _ = path;
        return Ok(());
    }
    let _ = path;
    Err(ChildError::MetalUnavailable)
}
