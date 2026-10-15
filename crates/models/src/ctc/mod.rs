//! Omnilingual CTC 300M. Weights converted from the committed ONNX manifest.
//! Upstream components: vendored `speakerdesk/mlx_ctc_components`.
use speakerdesk_core::error::ChildError;
use speakerdesk_core::ipc::CtcScoresOut;

pub mod manifest;
#[cfg(target_os = "macos")]
pub mod network;
#[cfg(windows)]
#[path = "network_windows.rs"]
pub mod network;
#[cfg(target_os = "macos")]
pub mod prepare;

pub const SOURCE_SHA256: &str = "e7c4e54ee4c4c47829cc6667d5d00ed8ea7bef1dcfeef0fce766f77752a2726c";
pub const TOKENS_SHA256: &str = speakerdesk_core::alignment::TOKENS_SHA256;

pub fn ctc_scores(_pcm: &[f32], _token_ids: &[i32]) -> Result<CtcScoresOut, ChildError> {
    Err(ChildError::Unimplemented)
}
