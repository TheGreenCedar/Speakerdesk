//! Nemotron-3 diarization. Upstream: `Blaizzy/mlx-audio@feb25a37`.
//! Weights: `mlx-community/Nemotron-3-Diarization@59ed2db`.
use std::path::Path;

use speakerdesk_core::error::ChildError;
use speakerdesk_core::ipc::{DiarFeedOut, DiarPreset, Handle, Turn};

pub fn diar_open(_preset: DiarPreset) -> Result<Handle, ChildError> {
    Err(ChildError::Unimplemented)
}

pub fn diar_feed(_handle: Handle, _pcm: &[f32], _last: bool) -> Result<DiarFeedOut, ChildError> {
    Err(ChildError::Unimplemented)
}

pub fn diar_close(_handle: Handle) -> Result<(), ChildError> {
    Err(ChildError::Unimplemented)
}

pub fn diarize_file(_path: &Path) -> Result<Vec<Turn>, ChildError> {
    Err(ChildError::Unimplemented)
}

pub fn diarize_window(_pcm: &[f32]) -> Result<Vec<Turn>, ChildError> {
    Err(ChildError::Unimplemented)
}
