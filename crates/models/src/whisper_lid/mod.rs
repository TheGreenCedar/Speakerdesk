//! Whisper-tiny LID. Upstream: `Blaizzy/mlx-audio@feb25a37`.
//! Weights: `mlx-community/whisper-tiny-asr-fp16@77fa3f5`.
#[cfg(target_os = "macos")]
mod features;
#[cfg(target_os = "macos")]
pub mod network;
mod select;
mod tokens;

pub use speakerdesk_core::languages::LID_CHECKPOINT as WEIGHT_PIN;

pub use select::{
    LanguageSelection, LanguageSelectionError, TopMargin, argmax_language_token, select_language,
    softmax_language_tokens, top_margin,
};
pub use tokens::{LANGUAGE_COUNT, LanguageTokenError, LanguageTokens, parse_language_tokens};

use speakerdesk_core::error::ChildError;
use speakerdesk_core::ipc::LanguageProbeOut;

pub fn language_probe(_pcm: &[f32]) -> Result<LanguageProbeOut, ChildError> {
    Err(ChildError::Unimplemented)
}
