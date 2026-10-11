//! Omnilingual CTC 300M. Weights converted from the committed ONNX manifest.
//! Upstream components: vendored `speakerdesk/mlx_ctc_components`.
use speakerdesk_core::error::ChildError;
use speakerdesk_core::ipc::CtcScoresOut;

pub fn ctc_scores(_pcm: &[f32], _token_ids: &[i32]) -> Result<CtcScoresOut, ChildError> {
    Err(ChildError::Unimplemented)
}
