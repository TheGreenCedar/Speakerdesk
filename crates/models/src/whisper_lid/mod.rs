//! Whisper-tiny LID. Upstream: `Blaizzy/mlx-audio@feb25a37`.
//! Weights: `mlx-community/whisper-tiny-asr-fp16@77fa3f5`.
mod select;
mod tokens;

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

#[cfg(test)]
mod tests {
    /// Full decoder parity needs the macOS weights
    /// `mlx-community/whisper-tiny-asr-fp16@77fa3f5`
    /// (`77fa3f52b482ec80d086df55f893065a9baab172`). This child does not call mlx-rs.
    #[test]
    #[ignore = "macOS weights mlx-community/whisper-tiny-asr-fp16@77fa3f5"]
    fn parity_whisper_lid() {}
}
