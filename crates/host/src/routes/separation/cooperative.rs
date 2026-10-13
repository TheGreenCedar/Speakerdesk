//! Capture-only adapter: live inference uses the base connection, never this callback.
use crate::inference::ChildModels;
use speakerdesk_core::{error::ChildError, ipc::*, models::Models};
use std::sync::Mutex;

pub(super) struct Cooperative<F> {
    models: ChildModels,
    pump: Mutex<F>,
}
impl<F: FnMut() -> Result<(), ChildError> + Send> Cooperative<F> {
    pub(super) fn new(models: ChildModels, pump: F) -> Self {
        Self {
            models,
            pump: Mutex::new(pump),
        }
    }
    fn yield_live(&self) -> Result<(), ChildError> {
        (self.pump.lock().expect("serial live pump"))()
    }
}
impl<F: FnMut() -> Result<(), ChildError> + Send> Models for Cooperative<F> {
    fn vad_open(&self, conditioning: Option<GainPolicy>) -> Result<Handle, ChildError> {
        self.yield_live()?;
        self.models.vad_open(conditioning)
    }
    fn vad_feed(&self, handle: Handle, pcm: &[f32], last: bool) -> Result<VadFeedOut, ChildError> {
        self.yield_live()?;
        self.models.vad_feed(handle, pcm, last)
    }
    fn vad_close(&self, handle: Handle) -> Result<(), ChildError> {
        self.models.vad_close(handle)
    }
    fn diar_open(&self, preset: DiarPreset) -> Result<Handle, ChildError> {
        self.yield_live()?;
        self.models.diar_open(preset)
    }
    fn diar_feed(
        &self,
        handle: Handle,
        pcm: &[f32],
        last: bool,
    ) -> Result<DiarFeedOut, ChildError> {
        self.yield_live()?;
        self.models.diar_feed(handle, pcm, last)
    }
    fn diar_close(&self, handle: Handle) -> Result<(), ChildError> {
        self.models.diar_close(handle)
    }
    fn language_probe(&self, pcm: &[f32]) -> Result<LanguageProbeOut, ChildError> {
        self.yield_live()?;
        self.models.language_probe(pcm)
    }
    fn ctc_scores(&self, pcm: &[f32], token_ids: &[i32]) -> Result<CtcScoresOut, ChildError> {
        self.yield_live()?;
        self.models.ctc_scores(pcm, token_ids)
    }
    fn embed_voice(&self, pcm: &[f32]) -> Result<VoiceEmbedding, ChildError> {
        self.yield_live()?;
        self.models.embed_voice(pcm)
    }
    fn setup_check(&self, models: &[String]) -> Result<Vec<SetupModelReport>, ChildError> {
        self.yield_live()?;
        self.models.setup_check(models)
    }
    fn metrics(&self) -> Result<MetricsOut, ChildError> {
        self.models.metrics()
    }
    fn transcribe(
        &self,
        pcm: &[f32],
        language: &str,
        padding: (u32, u32),
    ) -> Result<AsrText, ChildError> {
        self.yield_live()?;
        let handle = self.models.transcribe_open(pcm, language, padding)?;
        let result = (|| {
            loop {
                self.yield_live()?;
                if let Some(text) = self.models.transcribe_step(handle)? {
                    return Ok(text);
                }
            }
        })();
        let cleanup = self.models.transcribe_close(handle);
        result.and_then(|text| cleanup.map(|()| text))
    }
}
