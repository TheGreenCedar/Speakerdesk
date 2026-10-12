//! Cohere Transcribe 4-bit. Upstream: `appautomaton/mlx-speech@4577ad28`.
//! Weights: `spokedotso/cohere-transcribe-03-2026-mlx-4bit@064e51e`.
use speakerdesk_core::error::ChildError;
use speakerdesk_core::ipc::AsrText;

mod greedy;
mod preemphasis;

#[cfg(target_os = "macos")]
mod decoder;
#[cfg(target_os = "macos")]
mod encoder;
#[cfg(target_os = "macos")]
mod features;
#[cfg(target_os = "macos")]
mod layers;
#[cfg(target_os = "macos")]
pub mod network;

pub use greedy::EOS_TOKEN_ID;
pub use preemphasis::PREEMPHASIS;

pub const WEIGHT_PIN: &str =
    "spokedotso/cohere-transcribe-03-2026-mlx-4bit@064e51eab6db47066cbeaa85e2894b5691bd8d12";

pub fn transcribe(
    _pcm: &[f32],
    _language: &str,
    _padding: (u32, u32),
) -> Result<AsrText, ChildError> {
    Err(ChildError::Unimplemented)
}
