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
    fn diarize_window(&self, pcm: &[f32]) -> Result<Vec<Turn>, ChildError> {
        let handle = self.diar_open(DiarPreset::Offline)?;
        let result = (|| {
            let mut turns = Vec::new();
            let mut received = 0;
            let mut observed = 0;
            for chunk in pcm.chunks(16000 * 10).chain(std::iter::once(&[][..])) {
                received += chunk.len() as u64;
                let feed = self.diar_feed(handle, chunk, chunk.is_empty())?;
                feed.validate(received, observed)?;
                observed = feed.observed;
                turns.extend(feed.turns.into_iter().map(|turn| Turn {
                    start: turn.start as f64 / 16000.,
                    end: turn.end as f64 / 16000.,
                    speaker: turn.speaker,
                }));
            }
            if observed != received / 160 * 160 {
                return Err(ChildError::failed(
                    "incomplete final diarization observations",
                    false,
                ));
            }
            Ok(turns)
        })();
        let cleanup = self.diar_close(handle);
        result.and_then(|turns| cleanup.map(|()| turns))
    }
    fn language_probe(&self, pcm: &[f32]) -> Result<LanguageProbeOut, ChildError>;
    fn transcribe(
        &self,
        pcm: &[f32],
        language: &str,
        padding: (u32, u32),
    ) -> Result<AsrText, ChildError>;
    fn ctc_scores(&self, pcm: &[f32], token_ids: &[i32]) -> Result<CtcScoresOut, ChildError>;
    fn embed_voice(&self, pcm: &[f32]) -> Result<crate::ipc::VoiceEmbedding, ChildError>;
    fn setup_check(&self, models: &[String]) -> Result<Vec<SetupModelReport>, ChildError>;
    fn separate(&self, _pcm: &[f32]) -> Result<crate::ipc::SeparatedAudio, ChildError> {
        Err(ChildError::Unimplemented)
    }
    fn separate_open(&self, _pcm: &[f32]) -> Result<Handle, ChildError> {
        Err(ChildError::Unimplemented)
    }
    fn separate_step(
        &self,
        _handle: Handle,
    ) -> Result<Option<crate::ipc::SeparatedAudio>, ChildError> {
        Err(ChildError::Unimplemented)
    }
    fn separate_close(&self, _handle: Handle) -> Result<(), ChildError> {
        Err(ChildError::Unimplemented)
    }
    fn transcribe_open(
        &self,
        _pcm: &[f32],
        _language: &str,
        _padding: (u32, u32),
    ) -> Result<Handle, ChildError> {
        Err(ChildError::Unimplemented)
    }
    fn transcribe_step(&self, _handle: Handle) -> Result<Option<AsrText>, ChildError> {
        Err(ChildError::Unimplemented)
    }
    fn transcribe_close(&self, _handle: Handle) -> Result<(), ChildError> {
        Err(ChildError::Unimplemented)
    }
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
    fn embed_voice(&self, _pcm: &[f32]) -> Result<crate::ipc::VoiceEmbedding, ChildError> {
        Err(ChildError::Unimplemented)
    }
    fn setup_check(&self, _models: &[String]) -> Result<Vec<SetupModelReport>, ChildError> {
        Err(ChildError::Unimplemented)
    }
    fn metrics(&self) -> Result<MetricsOut, ChildError> {
        Err(ChildError::Unimplemented)
    }
}
