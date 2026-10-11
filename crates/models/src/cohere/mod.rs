//! Cohere Transcribe 4-bit. Upstream: `appautomaton/mlx-speech@4577ad28`.
//! Weights: `spokedotso/cohere-transcribe-03-2026-mlx-4bit@064e51e`.
//!
//! `transcribe` stays unimplemented until Metal and the 4-bit checkpoint are
//! available. The helpers below are the pieces that do not need weights.
use speakerdesk_core::error::ChildError;
use speakerdesk_core::ipc::AsrText;

mod greedy;
mod preemphasis;

pub use greedy::{EOS_TOKEN_ID, greedy_stop};
pub use preemphasis::{PREEMPHASIS, apply_preemphasis};

pub fn transcribe(
    _pcm: &[f32],
    _language: &str,
    _padding: (u32, u32),
) -> Result<AsrText, ChildError> {
    Err(ChildError::Unimplemented)
}

#[cfg(test)]
mod tests {
    /// Full-model parity against mlx-speech greedy text. Linux cannot run mlx-rs.
    /// Weights: `spokedotso/cohere-transcribe-03-2026-mlx-4bit@064e51e`.
    /// Upstream: `appautomaton/mlx-speech@4577ad28`.
    #[test]
    #[ignore = "weights spokedotso/cohere-transcribe-03-2026-mlx-4bit@064e51e and upstream appautomaton/mlx-speech@4577ad28"]
    fn parity_cohere() {
        // Encoder, decoder, tokenizer, dither, log-mel, and 4-bit weights stay on Metal.
    }
}
