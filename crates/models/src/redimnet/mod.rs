//! ReDimNet2 B6; waveform policy stays in Rust across native runtimes.
mod waveform;

#[cfg(target_os = "macos")]
pub mod coreml;

#[cfg(windows)]
pub mod network_windows;

use speakerdesk_core::error::ChildError;

pub use waveform::{
    CLIP_ABS, INPUT_SAMPLES, MAX_SAMPLES, MIN_SAMPLES, PrepareError, RATE, SILENCE_RMS, is_clipped,
    is_silent, prepared_audio, waveform_rms,
};

pub fn embed_voice(_pcm: &[f32]) -> Result<speakerdesk_core::ipc::VoiceEmbedding, ChildError> {
    Err(ChildError::Unimplemented)
}
