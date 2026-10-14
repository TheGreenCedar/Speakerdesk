//! Saved overlap work uses the existing serial worker and transactional job owner.
use super::*;
use speakerdesk_core::{
    document::Segment,
    separation::{self as domain, Candidate},
};

const RETAINED: &str = "Voice separation stopped. Existing words and audio are retained.";

#[derive(Deserialize)]
struct Begin {
    revision: i64,
    segment_id: String,
}

pub(super) async fn begin(
    State(app): State<AppState>,
    AxumPath(jid): AxumPath<String>,
    body: Bytes,
) -> Response {
    blocking(move || match start(&app, &jid, &body) {
        Ok(job) => (StatusCode::ACCEPTED, Json(stored_job_json(&job))).into_response(),
        Err(error) => from_api(error),
    })
    .await
}

fn start(app: &AppState, jid: &str, body: &[u8]) -> Result<Job, ApiError> {
    let request: Begin = serde_json::from_slice(body)
        .map_err(|_| ApiError::new(400, "Choose a saved passage to separate."))?;
    let id = require_job_id(jid)?;
    let lease = app
        .admission
        .lease(Activity::Refinement)
        .map_err(|e| ApiError::new(409, e.message()))?;
    let mut store = app.store();
    let mut job = load_job(&store, &id, false)?;
    if request.revision != job.revision || meeting_id(&app.shared).is_some() {
        return Err(ApiError::new(
            409,
            "The recording changed or capture is still active. Save and try again.",
        ));
    }
    let segment = job
        .document
        .as_ref()
        .and_then(|d| d.segments.iter().find(|s| s.id == request.segment_id))
        .cloned()
        .ok_or_else(|| ApiError::new(404, NOT_FOUND))?;
    domain::eligible(&job, &segment)?;
    if segment.separation_candidate.is_some() {
        return Err(ApiError::new(
            409,
            "Use or discard the existing voice estimates before trying again.",
        ));
    }
    crate::native::passage_recording_path(app.data_dir(), &job, &segment)
        .map_err(|_| ApiError::new(409, "The original passage audio is unavailable."))?;
    let episode = random_hex(16);
    job.extra.remove("separation_error");
    job.extra.insert("passage_separation".into(),json!({"id":episode,"segment_id":segment.id,"phase":"Separating voices and transcribing each estimate…"}));
    job.message = "Separating overlapping voices. Your original words remain editable.".into();
    job.revision += 1;
    job.updated = unix_now();
    let operation = app.begin_refinement_operation(&mut store, &mut job)?;
    drop(store);
    let reply = job.clone();
    let models = app.models();
    app.worker.enqueue(move || {
        let _lease = lease;
        execute(&operation, models.as_ref(), &job, &segment, &episode, false);
    });
    Ok(reply)
}

fn unattempted(job: &Job, segment: &Segment) -> bool {
    segment.separation_candidate.is_none()
        && segment
            .protected_fields
            .as_ref()
            .is_none_or(|fields| fields.is_empty())
        && domain::eligible(job, segment).is_ok()
        && job
            .extra
            .get("separation_attempts")
            .and_then(|attempts| attempts.get(&segment.id))
            .and_then(|attempts| attempts.get(domain::source_key(job, segment)))
            .and_then(Value::as_str)
            != Some(domain::MODEL)
}

/// A saved operation's successor rechecks admission and current source at dequeue time.
pub(super) fn automatic(
    shared: &Arc<Shared>,
    admission: &UpdateAdmission,
    id: &JobId,
    predecessor_generation: u64,
) {
    let models = shared.models.lock().expect("models").clone();
    if !models.separation_available() {
        return;
    }
    let Ok(_lease) = admission.lease(Activity::Refinement) else {
        return;
    };
    let mut store = shared.store.lock().expect("store");
    let Ok(Some(mut job)) = store.get(id, false) else {
        return;
    };
    if job.status != "ready"
        || job.extra.contains_key("passage_separation")
        || job
            .extra
            .get("operation_generation")
            .and_then(Value::as_u64)
            != Some(predecessor_generation)
    {
        return;
    }
    let ids: Vec<_> = job
        .document
        .iter()
        .flat_map(|document| &document.segments)
        .filter(|segment| unattempted(&job, segment))
        .map(|segment| segment.id.clone())
        .collect();
    if ids.is_empty() {
        return;
    }
    let Ok(operation) = begin_operation(
        shared,
        admission,
        &mut store,
        &mut job,
        Activity::Refinement,
    ) else {
        return;
    };
    drop(store);
    for segment_id in ids {
        if operation.cancellation.is_cancelled() {
            break;
        }
        let episode = random_hex(16);
        let started = shared.store.lock().expect("store").update_job(id, |job| {
            if !operation.current(job) {
                return Err(ApiError::new(409, "Separation was superseded."));
            }
            let segment = job
                .document
                .as_ref()
                .and_then(|document| document.segments.iter().find(|segment| segment.id == segment_id))
                .filter(|segment| unattempted(job, segment))
                .cloned()
                .ok_or_else(|| ApiError::new(409, "The passage no longer needs automatic separation."))?;
            let source_key = domain::source_key(job, &segment);
            job.extra
                .entry("separation_attempts")
                .or_insert_with(|| json!({}))
                .as_object_mut()
                .expect("owned attempt map")
                .entry(segment.id.clone())
                .or_insert_with(|| json!({}))[&source_key] = json!(domain::MODEL);
            job.extra.remove("separation_error");
            job.extra.insert("passage_separation".into(), json!({"id":episode,"segment_id":segment.id,"phase":"Separating voices and transcribing each estimate…"}));
            job.message = "Separating overlapping voices. Your original words remain editable.".into();
            Ok::<_, ApiError>(segment)
        });
        let Ok(Some((original, segment))) = started else {
            continue;
        };
        execute(
            &operation,
            models.as_ref(),
            &original,
            &segment,
            &episode,
            true,
        );
    }
}

fn valid_episode(id: &str) -> bool {
    id.len() == 32 && id.bytes().all(|b| b.is_ascii_hexdigit())
}

mod cooperative;

struct SeparatorSession {
    models: crate::inference::ChildModels,
    handle: speakerdesk_core::ipc::Handle,
}
impl Drop for SeparatorSession {
    fn drop(&mut self) {
        use speakerdesk_core::models::Models;
        let _ = self.models.separate_close(self.handle);
    }
}
struct LivePending {
    original: Job,
    segment: Segment,
    episode: String,
    audio: Option<speakerdesk_core::ipc::SeparatedAudio>,
    voices: Vec<domain::CandidateVoice>,
    separator: Option<SeparatorSession>,
    samples: usize,
    speech_ranges: [Vec<(usize, usize)>; 2],
}

/// Capture owns one bounded original passage and at most its two output channels.
/// Discovery is a document cursor, never a queue of PCM or inference tasks.
pub(crate) struct LiveSeparation {
    scan_from: Option<usize>,
    pending: Option<LivePending>,
}
impl LiveSeparation {
    pub(crate) fn new() -> Self {
        Self {
            scan_from: Some(0),
            pending: None,
        }
    }
    pub(crate) fn changed(&mut self, index: usize) {
        self.scan_from = Some(self.scan_from.map_or(index, |old| old.min(index)));
    }
    pub(crate) fn pending(&self) -> bool {
        self.pending.is_some()
    }
    pub(crate) fn interrupt(&mut self, app: &AppState) {
        if let Some(p) = self.pending.take() {
            let _ = app.store().update_job(&p.original.id, |job| {
                if job
                    .extra
                    .get("passage_separation")
                    .and_then(|v| v["id"].as_str())
                    == Some(&p.episode)
                {
                    interrupt(job);
                }
                Ok::<_, ApiError>(())
            });
            remove_assets(app.data_dir(), &p.original, &p.episode);
        }
    }
    pub(crate) fn step<F: FnMut() -> Result<(), speakerdesk_core::error::ChildError> + Send>(
        &mut self,
        app: &AppState,
        operation: &OperationGuard,
        job: &Job,
        pump: &mut F,
    ) -> Result<bool, speakerdesk_core::live::LiveError> {
        let models = app.models();
        if operation.cancellation.is_cancelled() {
            self.interrupt(app);
            return Ok(false);
        }
        if self.pending.is_none() {
            if !models.separation_available() {
                return Ok(false);
            }
            let Some(start) = self.scan_from.take() else {
                return Ok(false);
            };
            let mut store = app.store();
            let Some(current) = store
                .get(&job.id, false)
                .map_err(|e| speakerdesk_core::live::LiveError::Detail(e.to_string()))?
            else {
                return Ok(false);
            };
            if !operation.current(&current) {
                return Ok(false);
            }
            let selected = current
                .document
                .iter()
                .flat_map(|d| d.segments.iter().enumerate())
                .skip(start)
                .find(|(_, s)| {
                    s.separation_candidate.is_none()
                        && domain::eligible_live(&current, s).is_ok()
                        && current
                            .extra
                            .get("separation_attempts")
                            .and_then(|attempts| attempts.get(&s.id))
                            .and_then(|attempts| attempts.get(domain::source_key(&current, s)))
                            .and_then(Value::as_str)
                            != Some(domain::MODEL)
                })
                .map(|(i, s)| (i, s.clone()));
            let Some((index, segment)) = selected else {
                return Ok(false);
            };
            self.scan_from = Some(index + 1);
            let episode = random_hex(16);
            let key = domain::source_key(&current, &segment);
            let (mut original, ()) = store.update_job(&current.id, |job| {
                job.extra.entry("separation_attempts").or_insert_with(|| json!({}))
                    .as_object_mut().expect("owned attempt map").entry(segment.id.clone())
                    .or_insert_with(|| json!({}))[&key] = json!(domain::MODEL);
                job.extra.remove("separation_error");
                job.extra.insert("passage_separation".into(), json!({"id":episode,"segment_id":segment.id,"phase":"Separating overlapping voices…"}));
                Ok::<_, ApiError>(())
            }).map_err(|e| speakerdesk_core::live::LiveError::Detail(e.message))?.ok_or(speakerdesk_core::live::LiveError::Message("Recording is unavailable"))?;
            original.document = None;
            self.pending = Some(LivePending {
                original,
                segment,
                episode,
                audio: None,
                voices: Vec::with_capacity(2),
                separator: None,
                samples: 0,
                speech_ranges: std::array::from_fn(|_| Vec::new()),
            });
        }
        let p = self.pending.as_mut().expect("selected passage");
        let current = app
            .store()
            .get(&p.original.id, false)
            .map_err(|e| speakerdesk_core::live::LiveError::Detail(e.to_string()))?;
        let fresh = current.as_ref().is_some_and(|job| {
            operation.current(job)
                && job
                    .extra
                    .get("passage_separation")
                    .and_then(|v| v["id"].as_str())
                    == Some(&p.episode)
                && job
                    .document
                    .as_ref()
                    .and_then(|d| d.segments.iter().find(|s| s.id == p.segment.id))
                    .is_some_and(|s| {
                        domain::eligible_live(job, s).is_ok()
                            && domain::source_key(job, s)
                                == domain::source_key(&p.original, &p.segment)
                    })
        });
        if !fresh {
            self.interrupt(app);
            return Ok(true);
        }
        let language = p
            .segment
            .rest
            .get("language_mode")
            .and_then(Value::as_str)
            .unwrap_or(&p.original.language);
        let result = if p.audio.is_none() {
            (|| {
                use speakerdesk_core::models::Models;
                if let Some(session) = &p.separator {
                    if let Some(audio) = session.models.separate_step(session.handle)? {
                        p.audio = Some(domain::prepare_audio(audio, p.samples)?);
                        drop(p.separator.take());
                    }
                } else {
                    let path = crate::native::passage_recording_path(
                        app.data_dir(),
                        &p.original,
                        &p.segment,
                    )
                    .map_err(|e| {
                        speakerdesk_core::error::ChildError::failed(e.to_string(), false)
                    })?;
                    let pcm = audio::read_pcm16_range(path, p.segment.start, p.segment.end)
                        .map_err(|e| {
                            speakerdesk_core::error::ChildError::failed(e.to_string(), false)
                        })?;
                    let child = models
                        .live_models(&operation.cancellation)
                        .ok_or(speakerdesk_core::error::ChildError::Unimplemented)?;
                    let handle = child.separate_open(&pcm)?;
                    p.samples = pcm.len();
                    p.separator = Some(SeparatorSession {
                        models: child,
                        handle,
                    });
                }
                Ok(())
            })()
        } else if p.voices.len() < 2 {
            (|| {
                let child = models
                    .live_models(&operation.cancellation)
                    .ok_or(speakerdesk_core::error::ChildError::Unimplemented)?;
                let cooperative = cooperative::Cooperative::new(child, pump);
                let channel = p.voices.len();
                domain::decode_voice_with_evidence(
                    &cooperative,
                    &p.audio.as_ref().expect("separated audio").channels[channel],
                    language,
                )
                .map(|(voice, ranges)| {
                    p.voices.push(voice);
                    p.speech_ranges[channel] = ranges;
                })
            })()
        } else {
            let p = self.pending.take().expect("completed passage");
            let result = write_audio(
                app.data_dir(),
                &p.original,
                &p.episode,
                p.audio.as_ref().expect("separated audio"),
                &operation.cancellation,
            )
            .map(|()| Candidate {
                id: p.episode.clone(),
                source_key: domain::source_key(&p.original, &p.segment),
                original: Box::new(p.segment.clone()),
                voices: p.voices.try_into().expect("two voices"),
                model: p.audio.expect("separated audio").model,
                speech_ranges: bounded_speech_ranges(p.speech_ranges, p.samples),
            });
            finish(
                operation,
                &p.original,
                &p.segment,
                &p.episode,
                true,
                true,
                result,
                models.as_ref(),
            );
            return Ok(true);
        };
        if let Err(error) = result {
            let p = self.pending.take().expect("failed passage");
            if matches!(
                error,
                speakerdesk_core::error::ChildError::ProcessLost { .. }
                    | speakerdesk_core::error::ChildError::Timeout
            ) {
                // A new child generation may retry once through capture's existing
                // bounded recovery; a model/data failure remains a recorded attempt.
                let key = domain::source_key(&p.original, &p.segment);
                let _ = app.store().update_job(&p.original.id, |job| {
                    if operation.current(job) {
                        if let Some(attempts) = job
                            .extra
                            .get_mut("separation_attempts")
                            .and_then(|a| a.get_mut(&p.segment.id))
                            .and_then(Value::as_object_mut)
                        {
                            attempts.remove(&key);
                        }
                    }
                    Ok::<_, ApiError>(())
                });
            }
            finish(
                operation,
                &p.original,
                &p.segment,
                &p.episode,
                true,
                true,
                Err(error.to_string()),
                models.as_ref(),
            );
            if matches!(
                error,
                speakerdesk_core::error::ChildError::ProcessLost { .. }
                    | speakerdesk_core::error::ChildError::Timeout
                    | speakerdesk_core::error::ChildError::Cancelled
            ) {
                return Err(error.into());
            }
        }
        Ok(true)
    }
}
fn raw_path(root: &Path, job: &Job, episode: &str, channel: u8) -> std::io::Result<PathBuf> {
    if !valid_episode(episode) || channel > 1 {
        return Err(std::io::Error::other("Invalid separated audio identity"));
    }
    Ok(media_write_directory(root, job)
        .join("separated")
        .join(format!("{episode}-{channel}.wav")))
}
pub(crate) fn asset_path(
    root: &Path,
    job: &Job,
    episode: &str,
    channel: u8,
) -> std::io::Result<PathBuf> {
    let path = raw_path(root, job, episode, channel)?;
    confined_file(
        root,
        path.strip_prefix(root)
            .map_err(std::io::Error::other)?
            .to_str()
            .ok_or_else(|| std::io::Error::other("Invalid audio path"))?,
    )
}
fn remove_assets(root: &Path, job: &Job, episode: &str) {
    for channel in 0..2 {
        if let Ok(path) = raw_path(root, job, episode, channel) {
            let parent = path.parent().expect("owned audio directory");
            if owned_directory(root, parent) {
                let _ = fs::remove_file(path.with_extension("part"));
                let _ = fs::remove_file(path);
            }
        }
    }
}
fn owned_directory(root: &Path, directory: &Path) -> bool {
    fs::symlink_metadata(directory).is_ok_and(|m| m.is_dir() && !m.is_symlink())
        && fs::canonicalize(directory).is_ok_and(|p| p.starts_with(root))
}
/// Reconcile only this feature's files after recovering the durable job state.
pub(super) fn cleanup(root: &Path, store: &SqliteJobStore) -> Result<(), StoreError> {
    for summary in store.summaries()? {
        if ![
            root.join(summary.id.as_str()).join("separated"),
            root.join("derived")
                .join(summary.id.as_str())
                .join("separated"),
        ]
        .iter()
        .any(|directory| owned_directory(root, directory))
        {
            continue;
        }
        let Some(job) = store.get(&summary.id, false)? else {
            continue;
        };
        let directory = media_write_directory(root, &job).join("separated");
        if !owned_directory(root, &directory) {
            continue;
        }
        let entries = fs::read_dir(&directory).map_err(|e| StoreError::message(e.to_string()))?;
        for entry in entries {
            let entry = entry.map_err(|e| StoreError::message(e.to_string()))?;
            let path = entry.path();
            let Some(stem) = path.file_stem().and_then(|s| s.to_str()) else {
                continue;
            };
            let Some((episode, channel)) = stem.rsplit_once('-') else {
                continue;
            };
            if !valid_episode(episode) || !matches!(channel, "0" | "1") {
                continue;
            }
            let partial = path.extension().is_some_and(|ext| ext == "part");
            if (partial
                || path.extension().is_some_and(|ext| ext == "wav") && !references(&job, episode))
                && entry.file_type().is_ok_and(|kind| kind.is_file())
            {
                fs::remove_file(path).map_err(|e| StoreError::message(e.to_string()))?;
            }
        }
    }
    Ok(())
}
fn references(job: &Job, episode: &str) -> bool {
    job.document.as_ref().is_some_and(|d| {
        d.segments.iter().any(|s| {
            s.separation_candidate
                .as_ref()
                .is_some_and(|c| c.id == episode)
                || s.separation
                    .as_ref()
                    .is_some_and(|v| v.episode_id == episode)
        })
    })
}

fn execute(
    operation: &OperationGuard,
    models: &dyn ApplicationModels,
    original: &Job,
    segment: &Segment,
    episode: &str,
    automatic: bool,
) {
    let source_key = domain::source_key(original, segment);
    let result: Result<Candidate, String> = (|| {
        if operation.cancellation.is_cancelled() {
            return Err("Separation cancelled.".into());
        }
        let path =
            crate::native::passage_recording_path(&operation.shared.data_dir, original, segment)
                .map_err(|e| e.to_string())?;
        let pcm =
            audio::read_pcm16_range(path, segment.start, segment.end).map_err(|e| e.to_string())?;
        let language = segment
            .rest
            .get("language_mode")
            .and_then(Value::as_str)
            .unwrap_or(&original.language);
        let (audio, voices, speech_ranges) =
            models.separate_passage_with_evidence(&pcm, language, &operation.cancellation)?;
        audio.validate(pcm.len()).map_err(|e| e.to_string())?;
        write_audio(
            &operation.shared.data_dir,
            original,
            episode,
            &audio,
            &operation.cancellation,
        )?;
        Ok(Candidate {
            id: episode.into(),
            source_key: source_key.clone(),
            original: Box::new(segment.clone()),
            voices,
            model: audio.model,
            speech_ranges: bounded_speech_ranges(speech_ranges, pcm.len()),
        })
    })();
    finish(
        operation, original, segment, episode, automatic, false, result, models,
    );
}

fn bounded_speech_ranges(
    ranges: [Vec<(usize, usize)>; 2],
    length: usize,
) -> [Vec<(usize, usize)>; 2] {
    ranges.map(|ranges| {
        if ranges.len() <= 256
            && ranges
                .iter()
                .all(|(start, end)| start < end && *end <= length)
        {
            ranges
        } else {
            Vec::new()
        }
    })
}

fn write_audio(
    root: &Path,
    original: &Job,
    episode: &str,
    audio: &speakerdesk_core::ipc::SeparatedAudio,
    cancellation: &CancellationToken,
) -> Result<(), String> {
    for (channel, pcm) in audio.channels.iter().enumerate() {
        if cancellation.is_cancelled() {
            return Err("Separation cancelled.".into());
        }
        let path = raw_path(root, original, episode, channel as u8).map_err(|e| e.to_string())?;
        let directory = path.parent().expect("owned audio directory");
        fs::create_dir_all(directory).map_err(|e| e.to_string())?;
        let authority = fs::canonicalize(root).map_err(|e| e.to_string())?;
        if !fs::canonicalize(directory)
            .map_err(|e| e.to_string())?
            .starts_with(&authority)
            || fs::symlink_metadata(directory)
                .map_err(|e| e.to_string())?
                .is_symlink()
        {
            return Err("Separated audio directory is not owned by this app.".into());
        }
        let partial = path.with_extension("part");
        // New unpredictable identities never replace an existing file or follow its link.
        fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&partial)
            .map_err(|e| e.to_string())?;
        let mut writer = audio::RecordingWriter::create(&partial).map_err(|e| e.to_string())?;
        writer.write_samples(pcm).map_err(|e| e.to_string())?;
        drop(writer);
        fs::File::open(&partial)
            .and_then(|f| f.sync_all())
            .map_err(|e| e.to_string())?;
        fs::rename(&partial, &path).map_err(|e| e.to_string())?;
        fs::File::open(directory)
            .and_then(|f| f.sync_all())
            .map_err(|e| e.to_string())?;
    }
    Ok(())
}

fn finish(
    operation: &OperationGuard,
    original: &Job,
    segment: &Segment,
    episode: &str,
    automatic: bool,
    live: bool,
    result: Result<Candidate, String>,
    _models: &dyn ApplicationModels,
) {
    let source_key = domain::source_key(original, segment);
    let mut store = operation.shared.store.lock().expect("jobs");
    let published=store.update_job(&original.id,|job| {
        if !operation.current(job) || job.extra.get("passage_separation").and_then(|v|v["id"].as_str())!=Some(episode) {
            return Err(ApiError::new(409,"Separation was cancelled or superseded."));
        }
        job.extra.remove("passage_separation");
        let current=job.document.as_ref().and_then(|d|d.segments.iter().find(|s|s.id==segment.id));
        if current.is_none_or(|row|domain::source_key(job,row)!=source_key
            || automatic && !row.protected_fields.as_ref().is_none_or(|fields|fields.is_empty())) {
            job.extra.insert("separation_error".into(),json!("The passage changed during separation. Your edits are retained; retry the current passage."));
            job.message=RETAINED.into();
            return Ok(false);
        }
        match &result {
            Ok(candidate) => {
                if automatic && candidate.voices.iter().any(|voice| voice.text.trim().is_empty() || !speakerdesk_core::languages::is_supported(&voice.language)) {
                    job.extra.insert("separation_error".into(), json!("Both voice estimates need usable words. The original passage is retained."));
                    job.message = RETAINED.into();
                    return Ok(false);
                }
                job.document.as_mut().expect("current document").segments.iter_mut().find(|s|s.id==segment.id).expect("current passage").separation_candidate=Some(Box::new(candidate.clone()));
                // Nonempty ASR is not evidence that two generated channels contain
                // distinct voices. Keep the original and its attributed reading turns.
                if !live { job.message="Voice estimates ready. Compare them with the original recording.".into(); }
                Ok(true)
            }
            Err(error) => {
                tracing::warn!(%error,"saved separation failed");
                job.extra.insert("separation_error".into(),json!("Separation did not finish. Original words and audio are retained; check model setup and retry."));
                job.message=RETAINED.into();
                Ok(false)
            }
        }
    });
    drop(store);
    if !matches!(published, Ok(Some((_, true)))) {
        remove_assets(&operation.shared.data_dir, original, episode);
    }
}

pub(super) fn interrupt(job: &mut Job) {
    if job.extra.remove("passage_separation").is_some() {
        job.message = RETAINED.into();
        job.extra.insert(
            "separation_error".into(),
            json!("Separation stopped. Original words and audio are retained."),
        );
    }
}

#[cfg(test)]
mod automatic_tests {
    use super::*;
    use std::sync::{
        atomic::{AtomicUsize, Ordering},
        mpsc,
    };

    struct CountingModel(AtomicUsize);
    impl ApplicationModels for CountingModel {
        fn infer(&self, _: &Job, _: &CancellationToken) -> Result<Document, String> {
            unreachable!()
        }
        fn separation_available(&self) -> bool {
            true
        }
        fn separate_passage(
            &self,
            _: &[f32],
            _: &str,
            _: &CancellationToken,
        ) -> Result<
            (
                speakerdesk_core::ipc::SeparatedAudio,
                [domain::CandidateVoice; 2],
            ),
            String,
        > {
            self.0.fetch_add(1, Ordering::SeqCst);
            Err("scripted separation failure".into())
        }
    }

    #[test]
    fn queued_work_skips_capture_and_setup_without_consuming_the_attempt() {
        for activity in [Activity::Setup, Activity::recording()] {
            let root = tempfile::tempdir().unwrap();
            let app = AppState::open(root.path()).unwrap();
            let job: Job = serde_json::from_value(json!({"id":"aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa","name":"mix","status":"ready","language":"en","duration":2.,"created":0.,"updated":0.,"operation_generation":7,"document":{"speakers":{},"segments":[{"id":"mixed","start":0.,"end":2.,"speaker":"multiple_speakers","text":"original words","audio_revision":0,"speaker_activity":{"audio_revision":0,"observed_end_sample":32000,"regions":[{"start_sample":0,"end_sample":32000,"speakers":["speaker_0","speaker_1"]}]}}]}})).unwrap();
            let folder = root.path().join(job.id.as_str());
            fs::create_dir(&folder).unwrap();
            let mut writer = audio::RecordingWriter::create(folder.join("audio.wav")).unwrap();
            writer.write_samples(&vec![0.125; 32000]).unwrap();
            drop(writer);
            app.insert_job(job.clone());
            let models = Arc::new(CountingModel(AtomicUsize::new(0)));
            app.set_models(models.clone());
            let (started, waiting) = mpsc::channel();
            let (release, blocked) = mpsc::channel();
            app.worker.enqueue(move || {
                started.send(()).unwrap();
                blocked.recv().unwrap();
            });
            waiting.recv().unwrap();
            app.queue_separation(job.id.clone(), 7);
            let lease = app.admission.lease(activity).unwrap();
            let recording = if activity == Activity::recording() {
                let mut other = job.clone();
                other.id = JobId::parse("bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb").unwrap();
                Some(
                    app.begin_recording_operation(&mut app.store(), &mut other)
                        .unwrap(),
                )
            } else {
                None
            };
            release.send(()).unwrap();
            app.drain();
            let skipped = app.store().get(&job.id, false).unwrap().unwrap();
            assert_eq!(models.0.load(Ordering::SeqCst), 0);
            assert!(!skipped.extra.contains_key("separation_attempts"));
            assert_eq!(skipped.document, job.document);
            drop(recording);
            drop(lease);
            app.queue_separation(job.id.clone(), 7);
            app.drain();
            let attempted = app.store().get(&job.id, false).unwrap().unwrap();
            assert_eq!(models.0.load(Ordering::SeqCst), 1);
            assert!(attempted.extra.contains_key("separation_attempts"));
            assert_eq!(attempted.document, job.document);
        }
    }
}

pub(super) async fn accept(
    State(app): State<AppState>,
    AxumPath((jid, episode)): AxumPath<(String, String)>,
) -> Response {
    blocking(move || {
        let result = (|| {
            let id = require_job_id(&jid)?;
            let job = load_job(&app.store(), &id, false)?;
            let passage = job
                .document
                .as_ref()
                .and_then(|d| {
                    d.segments.iter().find(|s| {
                        s.separation_candidate
                            .as_ref()
                            .is_some_and(|c| c.id == episode)
                    })
                })
                .ok_or_else(|| ApiError::new(404, "Voice estimates are no longer available."))?;
            let candidate = passage
                .separation_candidate
                .as_ref()
                .expect("selected candidate")
                .as_ref()
                .clone();
            let (start, end) = domain::sample_bounds(passage)?;
            let mut channels = Vec::with_capacity(2);
            for channel in 0..2 {
                let path = asset_path(app.data_dir(), &job, &episode, channel).map_err(|_| {
                    ApiError::new(
                        409,
                        "Separated audio unavailable. The original passage is retained.",
                    )
                })?;
                let pcm = audio::read_pcm16(path).map_err(|_| {
                    ApiError::new(
                        409,
                        "Separated audio unavailable. The original passage is retained.",
                    )
                })?;
                if pcm.len() as u64 != end - start {
                    return Err(ApiError::new(
                        409,
                        "Separated audio length changed. The original passage is retained.",
                    ));
                }
                channels.push(pcm);
            }
            let passage_id = passage.id.clone();
            let models = app.models();
            let recognize =
                bounded_speech_ranges(candidate.speech_ranges.clone(), channels[0].len())
                    .iter()
                    .any(|ranges| !ranges.is_empty())
                    && models.separated_voice_policy().is_some()
                    && app.store().preference("voice_recognition")?.as_deref() != Some("false");
            if !recognize {
                return app
                    .store()
                    .update_job(&id, |current| {
                        domain::accept(current, &passage_id, &episode)?;
                        current.message = "Separated voices saved.".into();
                        Ok::<_, ApiError>(())
                    })?
                    .map(|(job, _)| job)
                    .ok_or_else(|| ApiError::new(404, NOT_FOUND));
            }
            let lease = app
                .admission
                .lease(Activity::Refinement)
                .map_err(|e| ApiError::new(409, e.message()))?;
            let mut store = app.store();
            let mut current = load_job(&store, &id, false)?;
            domain::accept(&mut current, &passage_id, &episode)?;
            current.message = "Separated voices saved.".into();
            current.revision += 1;
            current.updated = unix_now();
            let operation = app.begin_refinement_operation(&mut store, &mut current)?;
            drop(store);
            let reply = current.clone();
            app.worker.enqueue(move || {
                let _lease = lease;
                recognize_accepted(
                    &operation,
                    models.as_ref(),
                    &current,
                    &candidate,
                    speakerdesk_core::ipc::SeparatedAudio {
                        model: candidate.model.clone(),
                        channels: channels.try_into().expect("two channels"),
                    },
                );
            });
            Ok(reply)
        })();
        respond(result)
    })
    .await
}
fn recognize_accepted(
    operation: &OperationGuard,
    models: &dyn ApplicationModels,
    original: &Job,
    candidate: &Candidate,
    audio: speakerdesk_core::ipc::SeparatedAudio,
) {
    let ranges = bounded_speech_ranges(candidate.speech_ranges.clone(), audio.channels[0].len());
    if ranges.iter().all(Vec::is_empty) {
        return;
    }
    let mut state = {
        let mut store = operation.shared.store.lock().expect("store");
        let Ok(Some(current)) = store.get(&original.id, false) else {
            return;
        };
        if !operation.current(&current) {
            return;
        }
        match super::separation_identity::Identity::begin(
            models,
            &mut store,
            &current,
            &candidate.original,
            &candidate.id,
            &audio,
            &ranges,
        ) {
            Ok(Some(state)) => state,
            Ok(None) => return,
            Err(error) => {
                tracing::warn!(%error,"could not start accepted voice recognition");
                return;
            }
        }
    };
    let mut completed = false;
    loop {
        let snapshot = {
            let store = operation.shared.store.lock().expect("store");
            match (
                store.get(&original.id, false),
                super::separation_identity::Snapshot::load(&store, models),
            ) {
                (Ok(Some(current)), Ok(snapshot)) if operation.current(&current) => {
                    (current, snapshot)
                }
                _ => break,
            }
        };
        match state.step(
            models,
            &operation.shared.data_dir,
            &operation.cancellation,
            &snapshot.0,
            &snapshot.1,
            &audio,
        ) {
            Ok(true) => {}
            Ok(false) => {
                completed = true;
                break;
            }
            Err(error) => {
                tracing::warn!(%error,"accepted voice recognition interrupted");
                break;
            }
        }
    }
    let mut store = operation.shared.store.lock().expect("store");
    let snapshot = super::separation_identity::Snapshot::load(&store, models).ok();
    let _ = store.update_job(&original.id, |current| {
        if completed && operation.current(current) {
            if let Some(snapshot) = &snapshot {
                state.apply(current, snapshot);
            } else {
                state.interrupt(current);
            }
        } else {
            state.interrupt(current);
        }
        Ok::<_, ApiError>(())
    });
}

pub(super) async fn discard(
    State(app): State<AppState>,
    AxumPath((jid, episode)): AxumPath<(String, String)>,
) -> Response {
    blocking(move || {
        let result = (|| {
            let id = require_job_id(&jid)?;
            let job = app
                .store()
                .update_job(&id, |job| {
                    let row = job
                        .document
                        .as_mut()
                        .and_then(|d| {
                            d.segments.iter_mut().find(|s| {
                                s.separation_candidate
                                    .as_ref()
                                    .is_some_and(|c| c.id == episode)
                            })
                        })
                        .ok_or_else(|| {
                            ApiError::new(404, "Voice estimates are no longer available.")
                        })?;
                    row.separation_candidate = None;
                    job.message = "Original passage kept.".into();
                    Ok::<_, ApiError>(())
                })?
                .map(|(job, _)| job)
                .ok_or_else(|| ApiError::new(404, NOT_FOUND))?;
            remove_assets(app.data_dir(), &job, &episode);
            Ok(job)
        })();
        respond(result)
    })
    .await
}
pub(super) async fn restore(
    State(app): State<AppState>,
    AxumPath((jid, episode)): AxumPath<(String, String)>,
) -> Response {
    blocking(move || {
        let result = (|| {
            let id = require_job_id(&jid)?;
            app.store()
                .update_job(&id, |job| {
                    domain::restore(job, &episode)?;
                    job.message = "Original passage restored.".into();
                    Ok::<_, ApiError>(())
                })?
                .map(|(job, _)| job)
                .ok_or_else(|| ApiError::new(404, NOT_FOUND))
        })();
        respond(result)
    })
    .await
}
fn respond(result: Result<Job, ApiError>) -> Response {
    match result {
        Ok(job) => Json(stored_job_json(&job)).into_response(),
        Err(error) => from_api(error),
    }
}
pub(super) async fn history(
    State(app): State<AppState>,
    AxumPath((jid, episode)): AxumPath<(String, String)>,
) -> Response {
    blocking(move || {
        let result = (|| {
            let job = load_job(&app.store(), &require_job_id(&jid)?, false)?;
            job.extra
                .get("separation_history")
                .and_then(|h| h.get(&episode))
                .cloned()
                .ok_or_else(|| ApiError::new(404, NOT_FOUND))
        })();
        match result {
            Ok(value) => Json(value).into_response(),
            Err(error) => from_api(error),
        }
    })
    .await
}
pub(super) async fn playback(
    State(app): State<AppState>,
    AxumPath((jid, episode, channel)): AxumPath<(String, String, u8)>,
    request: Request,
) -> Response {
    let path = tokio::task::spawn_blocking(move || {
        let job = load_job(&app.store(), &require_job_id(&jid).ok()?, false).ok()?;
        if !references(&job, &episode) {
            return None;
        }
        asset_path(app.data_dir(), &job, &episode, channel).ok()
    })
    .await;
    let Ok(Some(path)) = path else {
        return StatusCode::NOT_FOUND.into_response();
    };
    match tower_http::services::ServeFile::new(path)
        .try_call(request)
        .await
    {
        Ok(response) => response.map(axum::body::Body::new),
        Err(_) => StatusCode::INTERNAL_SERVER_ERROR.into_response(),
    }
}

pub(super) async fn passage_audio(
    State(app): State<AppState>,
    AxumPath((jid, pid)): AxumPath<(String, String)>,
    request: Request,
) -> Response {
    let path = tokio::task::spawn_blocking(move || {
        let job = load_job(&app.store(), &require_job_id(&jid).ok()?, false).ok()?;
        let passage = job
            .document
            .as_ref()?
            .segments
            .iter()
            .find(|s| s.id == pid)?;
        crate::native::passage_recording_path(app.data_dir(), &job, passage).ok()
    })
    .await;
    serve_audio(path.ok().flatten(), request).await
}

pub(super) async fn original_audio(
    State(app): State<AppState>,
    AxumPath((jid, episode)): AxumPath<(String, String)>,
    request: Request,
) -> Response {
    let path = tokio::task::spawn_blocking(move || {
        let job = load_job(&app.store(), &require_job_id(&jid).ok()?, false).ok()?;
        let history: domain::History =
            serde_json::from_value(job.extra.get("separation_history")?.get(&episode)?.clone())
                .ok()?;
        crate::native::passage_recording_path(app.data_dir(), &job, &history.original).ok()
    })
    .await;
    serve_audio(path.ok().flatten(), request).await
}
async fn serve_audio(path: Option<PathBuf>, request: Request) -> Response {
    let Some(path) = path else {
        return StatusCode::NOT_FOUND.into_response();
    };
    match tower_http::services::ServeFile::new(path)
        .try_call(request)
        .await
    {
        Ok(response) => response.map(axum::body::Body::new),
        Err(_) => StatusCode::INTERNAL_SERVER_ERROR.into_response(),
    }
}
