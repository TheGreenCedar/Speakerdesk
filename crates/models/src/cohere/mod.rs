//! Cohere Transcribe 4-bit. Upstream: `appautomaton/mlx-speech@4577ad28`.
//! Weights: `spokedotso/cohere-transcribe-03-2026-mlx-4bit@064e51e`.
use speakerdesk_core::error::ChildError;
use speakerdesk_core::ipc::AsrText;

pub fn transcribe(
    _pcm: &[f32],
    _language: &str,
    _padding: (u32, u32),
) -> Result<AsrText, ChildError> {
    Err(ChildError::Unimplemented)
}
