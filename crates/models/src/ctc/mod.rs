//! Omnilingual CTC 300M. Weights converted from the committed ONNX manifest.
//! Upstream components: vendored `speakerdesk/mlx_ctc_components`.
use speakerdesk_core::error::ChildError;
use speakerdesk_core::ipc::CtcScoresOut;

pub mod manifest;

pub fn ctc_scores(_pcm: &[f32], _token_ids: &[i32]) -> Result<CtcScoresOut, ChildError> {
    Err(ChildError::Unimplemented)
}

#[cfg(test)]
mod tests {
    /// Safetensors reference weights are not loaded here. When that gate opens,
    /// compare scores and require max absolute delta <= 1e-4.
    #[test]
    #[ignore = "safetensors gate |delta| <= 1e-4"]
    fn parity_ctc() {
        panic!("Not implemented: real model comparison has not run.");
    }
}
