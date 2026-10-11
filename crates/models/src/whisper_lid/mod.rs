//! Whisper-tiny LID. Upstream: `Blaizzy/mlx-audio@feb25a37`.
//! Weights: `mlx-community/whisper-tiny-asr-fp16@77fa3f5`.
use speakerdesk_core::error::ChildError;
use speakerdesk_core::ipc::LanguageProbeOut;

pub fn language_probe(_pcm: &[f32]) -> Result<LanguageProbeOut, ChildError> {
    Err(ChildError::Unimplemented)
}
