//! Silero VAD v6. Upstream: `Blaizzy/mlx-audio@feb25a37`.
//! Weights: `mlx-community/silero-vad-v6@2ebf4a5`. The silero agent replaces these bodies.
use speakerdesk_core::error::ChildError;
use speakerdesk_core::ipc::{GainPolicy, Handle, VadFrame};

pub fn vad_open(_conditioning: Option<GainPolicy>) -> Result<Handle, ChildError> {
    Err(ChildError::Unimplemented)
}

pub fn vad_feed(_handle: Handle, _pcm: &[f32], _last: bool) -> Result<Vec<VadFrame>, ChildError> {
    Err(ChildError::Unimplemented)
}

pub fn vad_close(_handle: Handle) -> Result<(), ChildError> {
    Err(ChildError::Unimplemented)
}
