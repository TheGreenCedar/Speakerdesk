//! Saved overlap work uses the existing serial worker and transactional job owner.
use super::*;
use speakerdesk_core::{
    document::Segment,
    separation::{self as domain, Candidate},
};

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
    let path = crate::native::passage_recording_path(app.data_dir(), &job, &segment)
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
        execute(&operation, models.as_ref(), &job, &segment, &path, &episode);
    });
    Ok(reply)
}

fn valid_episode(id: &str) -> bool {
    id.len() == 32 && id.bytes().all(|b| b.is_ascii_hexdigit())
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
    path: &Path,
    episode: &str,
) {
    let source_key = domain::source_key(original, segment);
    let result: Result<Candidate, String> = (|| {
        if operation.cancellation.is_cancelled() {
            return Err("Separation cancelled.".into());
        }
        let pcm =
            audio::read_pcm16_range(path, segment.start, segment.end).map_err(|e| e.to_string())?;
        let language = segment
            .rest
            .get("language_mode")
            .and_then(Value::as_str)
            .unwrap_or(&original.language);
        let (audio, voices) = models.separate_passage(&pcm, language, &operation.cancellation)?;
        audio.validate(pcm.len()).map_err(|e| e.to_string())?;
        for (channel, pcm) in audio.channels.iter().enumerate() {
            if operation.cancellation.is_cancelled() {
                return Err("Separation cancelled.".into());
            }
            let path = raw_path(&operation.shared.data_dir, original, episode, channel as u8)
                .map_err(|e| e.to_string())?;
            let directory = path.parent().expect("owned audio directory");
            fs::create_dir_all(directory).map_err(|e| e.to_string())?;
            let root = fs::canonicalize(&operation.shared.data_dir).map_err(|e| e.to_string())?;
            if !fs::canonicalize(directory)
                .map_err(|e| e.to_string())?
                .starts_with(&root)
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
        Ok(Candidate {
            id: episode.into(),
            source_key: source_key.clone(),
            original: Box::new(segment.clone()),
            voices,
            model: audio.model,
        })
    })();
    let mut store = operation.shared.store.lock().expect("jobs");
    let published=store.update_job(&original.id,|job| {
        if !operation.current(job) || job.extra.get("passage_separation").and_then(|v|v["id"].as_str())!=Some(episode) {
            return Err(ApiError::new(409,"Separation was cancelled or superseded."));
        }
        job.extra.remove("passage_separation");
        let current=job.document.as_ref().and_then(|d|d.segments.iter().find(|s|s.id==segment.id));
        if current.is_none_or(|row|domain::source_key(job,row)!=source_key) {
            job.extra.insert("separation_error".into(),json!("The passage changed during separation. Your edits are retained; retry the current passage."));
            return Ok(false);
        }
        match &result {
            Ok(candidate) => {
                job.document.as_mut().expect("current document").segments.iter_mut().find(|s|s.id==segment.id).expect("current passage").separation_candidate=Some(Box::new(candidate.clone()));
                job.message="Voice estimates ready. Listen to both before replacing the original passage.".into();
                Ok(true)
            }
            Err(error) => {
                tracing::warn!(%error,"saved separation failed");
                job.extra.insert("separation_error".into(),json!("Separation did not finish. Original words and audio are retained; check model setup and retry."));
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
        job.extra.insert(
            "separation_error".into(),
            json!("Separation stopped. Original words and audio are retained."),
        );
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
            let (start, end) = domain::sample_bounds(passage)?;
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
            }
            let passage_id = passage.id.clone();
            app.store()
                .update_job(&id, |current| {
                    domain::accept(current, &passage_id, &episode)?;
                    current.message = "Separated voices saved.".into();
                    Ok::<_, ApiError>(())
                })?
                .map(|(job, _)| job)
                .ok_or_else(|| ApiError::new(404, NOT_FOUND))
        })();
        respond(result)
    })
    .await
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
