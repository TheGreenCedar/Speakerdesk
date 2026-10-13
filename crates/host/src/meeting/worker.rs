//! One serial live engine, including exceptional disk-backed state reconstruction.
use super::{Progress, conflict, update};
use crate::{
    capture::CaptureSnapshot,
    inference::ChildModels,
    routes::{AppState, OperationGuard},
};
use serde::Deserialize;
use serde_json::{Value, json};
use speakerdesk_core::{
    audio::RecordingReader,
    document::Job,
    error::{ApiError, ChildError, StoreError},
    ids::JobId,
    live::{
        AudioNotice, EditedBoundary, Engine, EngineConfig, EngineEvent, Inbox, LiveCommand,
        LiveError, RecoveryBoundary, SourceRuntime, StreamRecovery, apply_canonical_candidate,
        edited_boundary,
    },
};
use std::{
    collections::{BTreeMap, VecDeque},
    sync::{Arc, Mutex},
    time::Instant,
};
use tokio::sync::{mpsc, oneshot, watch};

#[derive(Clone, Deserialize)]
struct LanguageEpoch {
    epoch: i64,
    language: String,
    start_sample: i64,
}

#[derive(Clone)]
pub(super) struct RecoveryPlan {
    pub boundary: StreamRecovery,
    history: Vec<LanguageEpoch>,
}
impl RecoveryPlan {
    pub fn prepare(job: &mut Job, through: i64) -> Result<Self, ApiError> {
        if job
            .extra
            .get("live_stream_generation")
            .and_then(Value::as_u64)
            .unwrap_or(0)
            != 0
        {
            return Err(conflict(
                "Automatic live recovery was already attempted. Stop to retain audio and edits.",
            ));
        }
        let history: Vec<LanguageEpoch> = serde_json::from_value(
            job.extra
                .get("language_history")
                .cloned()
                .unwrap_or(Value::Null),
        )
        .map_err(|_| conflict("Saved language history is invalid."))?;
        if history.is_empty()
            || history[0].start_sample != 0
            || history.iter().enumerate().any(|(i, e)| {
                e.epoch != i as i64
                    || e.start_sample < 0
                    || e.start_sample > through
                    || (i > 0 && e.start_sample < history[i - 1].start_sample)
            })
        {
            return Err(conflict(
                "Saved language history does not cover the recording.",
            ));
        }
        let document = job
            .document
            .as_mut()
            .ok_or_else(|| conflict("Meeting document is missing."))?;
        // Empty machine placeholders have no accepted words to preserve. Rebuild that tail;
        // explicit user edits (including intentionally empty text) remain authoritative.
        document.segments.retain(|row| {
            !row.text.trim().is_empty()
                || row
                    .protected_fields
                    .as_ref()
                    .is_some_and(|fields| !fields.is_empty())
        });
        let dual = job
            .extra
            .get("sources")
            .and_then(Value::as_array)
            .is_some_and(|s| s.len() == 2);
        let mut source_ends =
            BTreeMap::from([("microphone_clean".to_owned(), 0), ("system".to_owned(), 0)]);
        let mut start = 0;
        for row in &document.segments {
            let original_start = row
                .rest
                .get("start_sample")
                .and_then(Value::as_i64)
                .unwrap_or(0);
            let end = row
                .rest
                .get("canonical_text_end_sample")
                .and_then(Value::as_i64)
                .unwrap_or(original_start);
            if original_start < 0 || end < original_start || end > through {
                return Err(conflict("Saved passages exceed retained audio."));
            }
            if dual {
                let source = row
                    .capture_source
                    .as_ref()
                    .and_then(|v| v.get("source_id"))
                    .and_then(Value::as_str)
                    .and_then(|id| source_ends.get_mut(id))
                    .ok_or_else(|| conflict("A saved passage is missing its capture source."))?;
                *source = (*source).max(end);
            } else {
                start = start.max(end);
            }
        }
        let epoch_at = |sample| {
            history
                .iter()
                .rev()
                .find(|e| e.start_sample <= sample)
                .expect("origin")
                .epoch
        };
        let source_boundaries: BTreeMap<String, RecoveryBoundary> = if dual {
            start = *source_ends.values().min().expect("two sources");
            source_ends
                .into_iter()
                .map(|(source, end)| {
                    (
                        source,
                        RecoveryBoundary {
                            start_sample: end,
                            language_epoch: epoch_at(end),
                        },
                    )
                })
                .collect()
        } else {
            BTreeMap::new()
        };
        let epoch = epoch_at(start);
        for row in &mut document.segments {
            if row.rest.get("canonical_state").and_then(Value::as_str) == Some("open") {
                row.rest.insert("canonical_state".into(), json!("sealed"));
                row.rest.insert("finalized".into(), json!(true));
                row.rest.insert(
                    "canonical_unresolved".into(),
                    json!("inference_restart_boundary"),
                );
                row.review = Some(true);
                if row.protected_fields.as_ref().is_none_or(Vec::is_empty) {
                    row.refinement_state = Some("unresolved".into());
                }
            }
        }
        job.extra.insert("live_stream_generation".into(), json!(1));
        let source_from_samples: BTreeMap<_, _> = source_boundaries
            .iter()
            .map(|(source, cut)| (source.clone(), cut.start_sample))
            .collect();
        job.extra.insert("live_recovery".into(),json!({"status":"replaying","generation":1,"from_sample":start,"source_from_samples":source_from_samples,"replay_through_sample":through}));
        job.extra.remove("live_inference_error");
        job.message = "Restoring live transcription from saved audio. Recording continues.".into();
        Ok(Self {
            boundary: StreamRecovery {
                generation: 1,
                start_sample: start,
                through_sample: through,
                language_epoch: epoch,
                source_boundaries,
            },
            history,
        })
    }
}

enum Engines {
    Single(Box<Engine<ChildModels>>),
    Sources(SourceRuntime<ChildModels>),
}
impl Engines {
    fn inbox(&self) -> Arc<Mutex<Inbox>> {
        match self {
            Self::Single(e) => e.inbox(),
            Self::Sources(e) => e.inbox(),
        }
    }
    fn poll(&mut self) -> Result<Option<Vec<EngineEvent>>, LiveError> {
        match self {
            Self::Single(e) => e.poll(),
            Self::Sources(e) => e.poll(),
        }
    }
    fn continue_edited(
        &mut self,
        boundary: &EditedBoundary,
    ) -> Result<Vec<EngineEvent>, LiveError> {
        match self {
            Self::Single(e) => e.continue_edited(boundary),
            Self::Sources(e) => e.continue_edited(boundary),
        }
    }
    fn handle(&mut self, command: LiveCommand) -> Result<Vec<EngineEvent>, LiveError> {
        match self {
            Self::Single(e) => e.handle(command),
            Self::Sources(e) => e.handle(command),
        }
    }
}
pub(super) struct Worker {
    inbox: Arc<Mutex<Inbox>>,
    wake: mpsc::Sender<()>,
    pub result: watch::Receiver<Option<Result<(), LiveError>>>,
    task: tokio::task::JoinHandle<()>,
}
impl Worker {
    pub fn wake(&self) {
        let _ = self.wake.try_send(());
    }
    pub fn push(&self, command: LiveCommand) -> Result<(), LiveError> {
        self.inbox.lock().expect("inbox").push(command)?;
        let _ = self.wake.try_send(());
        Ok(())
    }
    pub fn failed(&self) -> Option<LiveError> {
        self.result
            .borrow()
            .as_ref()
            .and_then(|r| r.as_ref().err())
            .cloned()
    }
    pub async fn join(self) -> Result<Result<(), LiveError>, StoreError> {
        self.task
            .await
            .map_err(|e| StoreError::message(format!("Live worker cleanup failed: {e}")))?;
        self.result
            .borrow()
            .clone()
            .ok_or_else(|| StoreError::message("Live worker exited without a cleanup result"))
    }
}
fn stream_update(
    app: &AppState,
    id: &JobId,
    op: &OperationGuard,
    generation: u32,
    change: impl FnOnce(&mut Job) -> Result<(), ApiError>,
) -> Result<Job, LiveError> {
    update(app, id, op, |job| {
        if job
            .extra
            .get("live_stream_generation")
            .and_then(Value::as_u64)
            .unwrap_or(0)
            != u64::from(generation)
        {
            return Err(conflict("Live stream was superseded."));
        }
        change(job)
    })
    .map_err(|e| LiveError::Detail(e.message))
}
#[allow(clippy::too_many_arguments)]
fn publish(
    engine: &mut Engines,
    app: &AppState,
    id: &JobId,
    op: &OperationGuard,
    generation: u32,
    progress: &watch::Sender<Progress>,
    events: Vec<EngineEvent>,
    rolling: &crate::routes::refinement::Rolling,
) -> Result<(bool, Option<usize>), LiveError> {
    let mut finished = false;
    let mut changed = None;
    let mut events = VecDeque::from(events);
    let mut boundaries = BTreeMap::new();
    while let Some(event) = events.pop_front() {
        match event {
            EngineEvent::CanonicalRevision { candidate, .. } => {
                if boundaries.get(&candidate.id).is_some_and(|cut| {
                    candidate
                        .rest
                        .get("end_sample")
                        .and_then(Value::as_i64)
                        .is_some_and(|end| end > *cut)
                }) {
                    continue;
                }
                let mut boundary = None;
                stream_update(app, id, op, generation, |job| {
                    let document = job
                        .document
                        .as_ref()
                        .ok_or_else(|| conflict("Meeting document is missing."))?;
                    // Accepted estimates own this original slot. A later canonical
                    // snapshot/recovery must not recreate its mixed source row.
                    if document.segments.iter().any(|row| {
                        row.separation
                            .as_ref()
                            .is_some_and(|voice| voice.original_id == candidate.id)
                    }) {
                        return Ok(());
                    }
                    boundary = edited_boundary(document, &candidate);
                    if boundary.is_some() {
                        let prior = job
                            .document
                            .as_mut()
                            .expect("document")
                            .segments
                            .iter_mut()
                            .find(|row| row.id == candidate.id)
                            .expect("edited passage");
                        prior.rest.insert("canonical_state".into(), json!("sealed"));
                        prior.rest.insert("finalized".into(), json!(true));
                        return Ok(());
                    }
                    let mut merged = apply_canonical_candidate(document, &candidate);
                    let mut next = document
                        .speakers
                        .keys()
                        .filter(|key| key.starts_with("speaker_"))
                        .count()
                        + 1;
                    for (key, label) in &mut merged.speakers {
                        if key.starts_with("speaker_") && !document.speakers.contains_key(key) {
                            while document
                                .speakers
                                .values()
                                .any(|v| v == &format!("Speaker {next}"))
                            {
                                next += 1;
                            }
                            *label = format!("Speaker {next}");
                            next += 1;
                        }
                    }
                    if candidate
                        .rest
                        .get("canonical_state")
                        .and_then(Value::as_str)
                        == Some("sealed")
                        && document
                            .segments
                            .iter()
                            .find(|row| row.id == candidate.id)
                            .is_none_or(|prior| {
                                prior.rest.get("canonical_state").and_then(Value::as_str)
                                    != Some("sealed")
                                    || prior.audio_revision != candidate.audio_revision
                            })
                        && let Some(index) = merged
                            .segments
                            .iter()
                            .position(|row| row.id == candidate.id)
                    {
                        rolling.changed(index);
                    }
                    if let Some(index) = merged
                        .segments
                        .iter()
                        .position(|row| row.id == candidate.id)
                    {
                        changed = Some(changed.map_or(index, |old: usize| old.min(index)));
                    }
                    job.document = Some(merged);
                    Ok(())
                })?;
                if let Some(boundary) = boundary {
                    boundaries.insert(boundary.utterance_id.clone(), boundary.end_sample);
                    let continuation = engine.continue_edited(&boundary)?;
                    for event in continuation.into_iter().rev() {
                        events.push_front(event);
                    }
                }
            }
            EngineEvent::Progress {
                processed_seconds, ..
            } => {
                progress.send_modify(|p| p.processed = p.processed.max(processed_seconds));
            }
            EngineEvent::FlushAck {
                request_id,
                received_sample,
                available_sample,
                speech_observed_sample,
                fast_sequence,
            } => {
                stream_update(app, id, op, generation, |job| {
                    job.extra.insert("pause_flush".into(),json!({"request_id":request_id,"status":"complete","received_sample":received_sample,"available_sample":available_sample,"speech_observed_sample":speech_observed_sample,"fast_sequence":fast_sequence}));
                    Ok(())
                })?;
            }
            EngineEvent::LanguageRegistered {
                generation: epoch, ..
            } => {
                stream_update(app, id, op, generation, |job| {
                    let old = job
                        .extra
                        .get("language_acknowledged_revision")
                        .and_then(Value::as_i64)
                        .unwrap_or(0);
                    job.extra.insert(
                        "language_acknowledged_revision".into(),
                        json!(old.max(epoch)),
                    );
                    Ok(())
                })?;
            }
            EngineEvent::CaptureFinished {
                canonical_observed_sample,
                canonical_uncertain_samples,
                ..
            } => {
                stream_update(app, id, op, generation, |job| {
                    job.extra.insert(
                        "canonical_observed_sample".into(),
                        json!(canonical_observed_sample),
                    );
                    job.extra.insert(
                        "canonical_uncertain_samples".into(),
                        json!(canonical_uncertain_samples),
                    );
                    Ok(())
                })?;
                finished = true;
            }
            EngineEvent::Finished { .. } => finished = true,
        }
    }
    Ok((finished, changed))
}

#[allow(clippy::too_many_arguments)]
pub(super) async fn worker(
    app: AppState,
    id: JobId,
    language: String,
    dual: bool,
    capture: CaptureSnapshot,
    operation: Arc<OperationGuard>,
    progress: watch::Sender<Progress>,
    recovery: Option<RecoveryPlan>,
    rolling: crate::routes::refinement::Rolling,
) -> Result<Worker, LiveError> {
    let models = app
        .models()
        .live_models(&operation.cancellation)
        .ok_or("Live models are unavailable")?;
    let alignment = app.models().live_alignment();
    let (init_tx, init_rx) = oneshot::channel();
    let (wake, mut receiver) = mpsc::channel(1);
    let (result_tx, result) = watch::channel(None);
    let task = tokio::task::spawn_blocking(move || {
        let run = (|| {
            let generation = recovery.as_ref().map_or(0, |r| r.boundary.generation);
            let initial_language = recovery
                .as_ref()
                .map_or(language, |r| r.history[0].language.clone());
            let mut config = EngineConfig::new(id.as_str(), initial_language);
            config.recovery = recovery.as_ref().map(|r| r.boundary.clone());
            let folder = app.data_dir().join(id.as_str());
            if let Some(receipt) = capture.processing {
                config.processed_microphone_contract = Some(receipt.sha256);
            }
            if let Some(catalog) = capture.source_catalog {
                use sha2::{Digest, Sha256};
                config.capture_catalog_sha256 = Some(format!(
                    "{:x}",
                    Sha256::digest(serde_json::to_vec(&catalog).expect("catalog JSON"))
                ));
            }
            let mut engine = if dual {
                let mic = RecordingReader::open(folder.join("microphone_clean.wav"))
                    .map_err(|e| e.to_string())?;
                let system =
                    RecordingReader::open(folder.join("system.wav")).map_err(|e| e.to_string())?;
                Engines::Sources(SourceRuntime::new(config, models, [mic, system])?)
            } else {
                Engines::Single(Box::new(Engine::single(
                    config,
                    models,
                    folder.join("audio.wav"),
                )?))
            };
            if let Some(alignment) = alignment {
                match &mut engine {
                    Engines::Single(engine) => engine.set_word_alignment(alignment),
                    Engines::Sources(engine) => engine.set_word_alignment(alignment),
                }
            }
            init_tx
                .send(engine.inbox())
                .map_err(|_| "Live owner stopped during initialization")?;
            if let Some(plan) = recovery {
                let start = Instant::now();
                let mut cursor = 0;
                let mut epoch = 0;
                while cursor < plan.boundary.through_sample || epoch + 1 < plan.history.len() {
                    if operation.cancellation.is_cancelled() {
                        return Err(LiveError::Child(ChildError::Cancelled));
                    }
                    if epoch + 1 < plan.history.len()
                        && plan.history[epoch + 1].start_sample == cursor
                    {
                        epoch += 1;
                        let stamp = &plan.history[epoch];
                        let events = engine.handle(LiveCommand::Language {
                            language: stamp.language.clone(),
                            generation: stamp.epoch,
                            start_sample: cursor,
                        })?;
                        publish(
                            &mut engine,
                            &app,
                            &id,
                            &operation,
                            generation,
                            &progress,
                            events,
                            &rolling,
                        )?;
                        continue;
                    }
                    let next = (cursor + 16000).min(plan.boundary.through_sample).min(
                        plan.history
                            .get(epoch + 1)
                            .map_or(i64::MAX, |e| e.start_sample),
                    );
                    if next <= cursor {
                        return Err("Recovery language clock did not advance".into());
                    }
                    let stamp = &plan.history[epoch];
                    let events = engine.handle(LiveCommand::Audio(AudioNotice {
                        start_sample: cursor,
                        end_sample: next,
                        language_epoch: stamp.epoch,
                        language: stamp.language.clone(),
                    }))?;
                    publish(
                        &mut engine,
                        &app,
                        &id,
                        &operation,
                        generation,
                        &progress,
                        events,
                        &rolling,
                    )?;
                    cursor = next;
                }
                let events = engine.handle(LiveCommand::Flush {
                    request_id: None,
                    through_sample: Some(cursor),
                })?;
                publish(
                    &mut engine,
                    &app,
                    &id,
                    &operation,
                    generation,
                    &progress,
                    events,
                    &rolling,
                )?;
                stream_update(&app, &id, &operation, generation, |job| {
                    job.extra["live_recovery"]["status"] = json!("recovered");
                    job.extra["live_recovery"]["elapsed_ms"] =
                        json!(start.elapsed().as_millis() as u64);
                    job.message="Live transcription restored. Earlier passages and speaker identities are retained; review the recovery boundary.".into();
                    Ok(())
                })?;
                progress.send_modify(|p| {
                    p.recovering = false;
                    p.error = None;
                });
            }
            let mut accuracy = crate::routes::refinement::Runner::new(rolling.clone());
            let mut enrichment = Enrichment::new(app.clone());
            let inbox = engine.inbox();
            loop {
                if operation.cancellation.is_cancelled() {
                    return Err(LiveError::Child(ChildError::Cancelled));
                }
                let mut progressed = false;
                if let Some(events) = engine.poll()? {
                    progressed = true;
                    let (finished, changed) = publish(
                        &mut engine,
                        &app,
                        &id,
                        &operation,
                        generation,
                        &progress,
                        events,
                        &rolling,
                    )?;
                    if let Some(index) = changed {
                        enrichment.identity.changed();
                        enrichment.separation.changed(index);
                    }
                    if finished {
                        // Live EOF is fully published before the remaining saved work. The
                        // same recording lease owns cancellation and every final model call.
                        while enrichment.step(&operation, &id)?
                            || accuracy.step(&app, &operation)?
                        {}
                        return Ok(());
                    }
                }
                let idle = {
                    let inbox = inbox.lock().expect("inbox");
                    !inbox.has_pending_control() && inbox.queued_audio_span().is_none()
                };
                if idle {
                    // A running child request is not preemptible. Never admit another
                    // auxiliary phase while saved capture audio/control needs draining.
                    let enriched = enrichment.step(&operation, &id)?;
                    progressed |= enriched;
                    if !enriched {
                        let refined = accuracy.step(&app, &operation)?;
                        progressed |= refined;
                    }
                }
                if !progressed && receiver.blocking_recv().is_none() {
                    break;
                }
            }
            Err("Live command owner disappeared".into())
        })();
        result_tx.send_replace(Some(run));
    });
    match init_rx.await {
        Ok(inbox) => Ok(Worker {
            inbox,
            wake,
            result,
            task,
        }),
        Err(_) => {
            task.await.map_err(|e| e.to_string())?;
            Err(result
                .borrow()
                .as_ref()
                .and_then(|r| r.as_ref().err())
                .cloned()
                .unwrap_or_else(|| "Live worker could not initialize".into()))
        }
    }
}

/// The capture worker is the only owner: at most one identity/separation phase
/// is pending, with live polling between phases and cleanup on recovery/shutdown.
struct Enrichment {
    app: AppState,
    identity: crate::people_routes::LiveVoiceChecks,
    separation: crate::routes::separation::LiveSeparation,
}
impl Enrichment {
    fn new(app: AppState) -> Self {
        Self {
            app,
            identity: crate::people_routes::LiveVoiceChecks::new(true),
            separation: crate::routes::separation::LiveSeparation::new(),
        }
    }
    fn step(&mut self, operation: &OperationGuard, id: &JobId) -> Result<bool, LiveError> {
        use speakerdesk_core::store::JobStore;
        let job = self
            .app
            .store()
            .get(id, false)
            .map_err(|e| LiveError::Detail(e.to_string()))?;
        let Some(job) = job else { return Ok(false) };
        if !self.separation.pending() {
            let models = self.app.models();
            match operation.check_live_voice_step(&mut self.identity, models.as_ref(), &job) {
                Ok(true) => return Ok(true),
                Ok(false) => {}
                Err(error) => {
                    if matches!(
                        error,
                        LiveError::Child(
                            ChildError::ProcessLost { .. }
                                | ChildError::Timeout
                                | ChildError::Cancelled
                        )
                    ) {
                        return Err(error);
                    }
                    tracing::warn!(message = %error, "live voice check stopped");
                    self.identity.interrupt(&mut self.app.store());
                }
            }
        }
        if !self.identity.pending() {
            return self.separation.step(&self.app, operation, &job);
        }
        Ok(false)
    }
}
impl Drop for Enrichment {
    fn drop(&mut self) {
        self.identity.interrupt(&mut self.app.store());
        self.separation.interrupt(&self.app);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use speakerdesk_core::store::JobStore;

    #[test]
    fn recovery_boundary_follows_displayed_text_coverage_after_edit_and_later_model() {
        use speakerdesk_core::edit::{PassageEditInput, edit_passage};
        for edited in [false, true] {
            let id = JobId::parse("eeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee").unwrap();
            let mut job = super::super::meeting_job(id, "coverage", "en", &["microphone".into()]);
            let original: speakerdesk_core::document::Segment = serde_json::from_value(json!({
                "id":"open","start":0.,"end":7.,"speaker":"speaker_0","text":"Words decoded through six seconds",
                "machine_revision":1,"canonical_state":"open","start_sample":0,"end_sample":112000,
                "text_audio_anchor":{"start_sample":0,"end_sample":96000},"canonical_text_end_sample":96000
            })).unwrap();
            job.document
                .as_mut()
                .unwrap()
                .segments
                .push(original.clone());
            job.document
                .as_mut()
                .unwrap()
                .speakers
                .insert("speaker_0".into(), "Speaker 1".into());
            if edited {
                let edited = edit_passage(PassageEditInput {
                    status:"recording", duration:8., revision:1, document:job.document.as_ref(), segment_id:"open",
                    body:&json!({"segment_revision":1,"changes":{"text":"Human words","start":0.1,"end":6.5}}), now:1.,
                }).unwrap();
                assert!(edited.segment.rest.get("text_audio_anchor").is_none());
                job.document = Some(edited.document);
                let mut later = original;
                later.end = 8.;
                later.text = "Later machine words".into();
                later.rest.insert("end_sample".into(), json!(128000));
                later
                    .rest
                    .insert("canonical_text_end_sample".into(), json!(128000));
                job.document = Some(apply_canonical_candidate(
                    job.document.as_ref().unwrap(),
                    &later,
                ));
            }
            let saved = job.document.as_ref().unwrap().segments[0].clone();
            let plan = RecoveryPlan::prepare(&mut job, 128000).unwrap();
            assert_eq!(
                plan.boundary.start_sample, 96000,
                "must replay the not-yet-decoded tail"
            );
            let row = &job.document.as_ref().unwrap().segments[0];
            assert_eq!(
                (row.text.as_str(), row.start, row.end),
                (saved.text.as_str(), saved.start, saved.end)
            );
        }
    }

    #[test]
    fn source_boundaries_follow_capture_bindings_not_corrected_speaker_ids() {
        let id = JobId::parse("eeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee").unwrap();
        let mut job =
            super::super::meeting_job(id, "sources", "en", &["microphone".into(), "system".into()]);
        job.document.as_mut().unwrap().segments = serde_json::from_value(json!([
            {"id":"mic","start":0.,"end":10.,"speaker":"speaker_32","text":"mic","canonical_text_end_sample":160000,"capture_source":{"source_id":"microphone_clean"}},
            {"id":"system","start":0.,"end":8.,"speaker":"speaker_0","text":"system","canonical_text_end_sample":128000,"capture_source":{"source_id":"system"}}
        ])).unwrap();
        let plan = RecoveryPlan::prepare(&mut job, 192037).unwrap();
        assert_eq!(plan.boundary.start_sample, 128000);
        assert_eq!(
            plan.boundary.source_boundaries["microphone_clean"].start_sample,
            160000
        );
        assert_eq!(
            plan.boundary.source_boundaries["system"].start_sample,
            128000
        );
        assert_eq!(
            job.extra["live_recovery"]["source_from_samples"]["system"],
            128000
        );
    }

    #[test]
    fn recovery_transaction_preserves_accepted_history_and_rejects_old_worker_publication() {
        let root = tempfile::tempdir().unwrap();
        let app = AppState::open(root.path()).unwrap();
        let id = JobId::parse("eeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee").unwrap();
        let mut job =
            super::super::meeting_job(id.clone(), "recovery", "en", &["microphone".into()]);
        job.status = "ready".into();
        job.duration = Some(3.);
        let doc = job.document.as_mut().unwrap();
        doc.speakers.insert("speaker_0".into(), "Casey".into());
        doc.segments=serde_json::from_value(json!([
            {"id":"accepted","start":0.1,"end":1.0,"speaker":"speaker_0","text":"Human words","protected_fields":["text","start"],"canonical_state":"open","end_sample":16000,"canonical_text_end_sample":16000,"capture_source":{"source_id":"microphone_clean","catalog_sha256":"trusted","source_revision":1}},
            {"id":"empty-machine-tail","start":1.,"end":2.,"speaker":"speaker_0","text":"","canonical_state":"open","end_sample":32000}
        ])).unwrap();
        job.extra.insert("language_history".into(),json!([{"epoch":0,"language":"en","start_sample":0},{"epoch":1,"language":"de","start_sample":16000}]));
        let op = app
            .begin_recording_operation(&mut app.store(), &mut job)
            .unwrap();
        let mut plan = None;
        let recovered = stream_update(&app, &id, &op, 0, |job| {
            plan = Some(RecoveryPlan::prepare(job, 32037)?);
            Ok(())
        })
        .unwrap();
        let plan = plan.unwrap();
        assert_eq!(plan.boundary.start_sample, 16000);
        assert_eq!(plan.boundary.language_epoch, 1);
        let doc = recovered.document.as_ref().unwrap();
        assert_eq!(doc.segments.len(), 1);
        assert_eq!(doc.segments[0].text, "Human words");
        assert_eq!(doc.segments[0].start, 0.1);
        assert_eq!(doc.segments[0].end, 1.);
        assert_eq!(doc.segments[0].rest["canonical_state"], "sealed");
        assert_eq!(
            doc.segments[0].capture_source.as_ref().unwrap()["source_revision"],
            1
        );
        assert_eq!(doc.speakers["speaker_0"], "Casey");
        assert!(
            stream_update(&app, &id, &op, 0, |job| {
                job.message = "late result".into();
                Ok(())
            })
            .is_err()
        );
        assert_eq!(app.store().get(&id, false).unwrap().unwrap(), recovered);
        assert!(RecoveryPlan::prepare(&mut recovered.clone(), 32037).is_err());
    }
}
