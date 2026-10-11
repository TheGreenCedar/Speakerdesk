//! ReDimNet2 B6 MIL executor. Ported from `mlx_voice_mil.py`.
//! Weights: `aufklarer/ReDimNet2-B6-CoreML@112dd8f`.
mod waveform;

use speakerdesk_core::error::ChildError;

pub use waveform::{
    CLIP_ABS, IDENTITY_REUSE_COSINE, INPUT_SAMPLES, MAX_SAMPLES, MIN_SAMPLES, PrepareError, RATE,
    SILENCE_RMS, is_clipped, is_silent, prepared_audio, reuses_identity, waveform_rms,
};

pub fn embed_voice(_pcm: &[f32]) -> Result<Vec<f32>, ChildError> {
    Err(ChildError::Unimplemented)
}
