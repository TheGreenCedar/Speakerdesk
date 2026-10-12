//! ReDimNet2 B6 through the system Core ML runtime; waveform policy stays in Rust.
//! Weights: `aufklarer/ReDimNet2-B6-CoreML@112dd8f`.
mod waveform;

#[cfg(target_os = "macos")]
pub mod coreml;

use speakerdesk_core::error::ChildError;

pub use waveform::{
    CLIP_ABS, INPUT_SAMPLES, MAX_SAMPLES, MIN_SAMPLES, PrepareError, RATE, SILENCE_RMS, is_clipped,
    is_silent, prepared_audio, waveform_rms,
};

pub fn embed_voice(_pcm: &[f32]) -> Result<speakerdesk_core::ipc::VoiceEmbedding, ChildError> {
    Err(ChildError::Unimplemented)
}
