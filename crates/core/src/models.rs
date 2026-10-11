use std::path::Path;

use crate::error::ChildError;
use crate::ipc::{
    AsrText, CtcScoresOut, DiarFeedOut, DiarPreset, GainPolicy, Handle, LanguageProbeOut,
    MetricsOut, SetupModelReport, Turn, VadFeedOut,
};
/// Sync model seam. `ChildModels`, `FakeModels`, and `ReplayModels` implement it.
/// Model calls must not run under the job-store lock.
pub trait Models: Send + Sync {
    fn vad_open(&self, conditioning: Option<GainPolicy>) -> Result<Handle, ChildError>;
    fn vad_feed(&self, handle: Handle, pcm: &[f32], last: bool) -> Result<VadFeedOut, ChildError>;
    fn vad_close(&self, handle: Handle) -> Result<(), ChildError>;
    fn diar_open(&self, preset: DiarPreset) -> Result<Handle, ChildError>;
    fn diar_feed(&self, handle: Handle, pcm: &[f32], last: bool)
    -> Result<DiarFeedOut, ChildError>;
    fn diar_close(&self, handle: Handle) -> Result<(), ChildError>;
    fn diarize_file(&self, path: &Path) -> Result<Vec<Turn>, ChildError>;
    fn diarize_window(&self, pcm: &[f32]) -> Result<Vec<Turn>, ChildError>;
    fn language_probe(&self, pcm: &[f32]) -> Result<LanguageProbeOut, ChildError>;
    fn transcribe(
        &self,
        pcm: &[f32],
        language: &str,
        padding: (u32, u32),
    ) -> Result<AsrText, ChildError>;
    fn ctc_scores(&self, pcm: &[f32], token_ids: &[i32]) -> Result<CtcScoresOut, ChildError>;
    fn embed_voice(&self, pcm: &[f32]) -> Result<Vec<f32>, ChildError>;
    fn setup_check(&self, models: &[String]) -> Result<Vec<SetupModelReport>, ChildError>;
    fn metrics(&self) -> Result<MetricsOut, ChildError>;
}

/// Scripted double for unit tests. Every call returns [`ChildError::Unimplemented`]
/// unless a test replaces the type with its own impl.
pub struct UnimplementedModels;

impl Models for UnimplementedModels {
    fn vad_open(&self, _conditioning: Option<GainPolicy>) -> Result<Handle, ChildError> {
        Err(ChildError::Unimplemented)
    }
    fn vad_feed(
        &self,
        _handle: Handle,
        _pcm: &[f32],
        _last: bool,
    ) -> Result<VadFeedOut, ChildError> {
        Err(ChildError::Unimplemented)
    }
    fn vad_close(&self, _handle: Handle) -> Result<(), ChildError> {
        Err(ChildError::Unimplemented)
    }
    fn diar_open(&self, _preset: DiarPreset) -> Result<Handle, ChildError> {
        Err(ChildError::Unimplemented)
    }
    fn diar_feed(
        &self,
        _handle: Handle,
        _pcm: &[f32],
        _last: bool,
    ) -> Result<DiarFeedOut, ChildError> {
        Err(ChildError::Unimplemented)
    }
    fn diar_close(&self, _handle: Handle) -> Result<(), ChildError> {
        Err(ChildError::Unimplemented)
    }
    fn diarize_file(&self, _path: &Path) -> Result<Vec<Turn>, ChildError> {
        Err(ChildError::Unimplemented)
    }
    fn diarize_window(&self, _pcm: &[f32]) -> Result<Vec<Turn>, ChildError> {
        Err(ChildError::Unimplemented)
    }
    fn language_probe(&self, _pcm: &[f32]) -> Result<LanguageProbeOut, ChildError> {
        Err(ChildError::Unimplemented)
    }
    fn transcribe(
        &self,
        _pcm: &[f32],
        _language: &str,
        _padding: (u32, u32),
    ) -> Result<AsrText, ChildError> {
        Err(ChildError::Unimplemented)
    }
    fn ctc_scores(&self, _pcm: &[f32], _token_ids: &[i32]) -> Result<CtcScoresOut, ChildError> {
        Err(ChildError::Unimplemented)
    }
    fn embed_voice(&self, _pcm: &[f32]) -> Result<Vec<f32>, ChildError> {
        Err(ChildError::Unimplemented)
    }
    fn setup_check(&self, _models: &[String]) -> Result<Vec<SetupModelReport>, ChildError> {
        Err(ChildError::Unimplemented)
    }
    fn metrics(&self) -> Result<MetricsOut, ChildError> {
        Err(ChildError::Unimplemented)
    }
}
