//! ReDimNet2 B6 MIL executor. Ported from `mlx_voice_mil.py`.
//! Weights: `aufklarer/ReDimNet2-B6-CoreML@112dd8f`.
use speakerdesk_core::error::ChildError;

pub fn embed_voice(_pcm: &[f32]) -> Result<Vec<f32>, ChildError> {
    Err(ChildError::Unimplemented)
}
