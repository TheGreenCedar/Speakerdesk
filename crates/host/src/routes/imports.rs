//! Import orchestration from upload staging through preparation, queued execution and progress.
//!
//! Imports may queue together, unlike recording/refinement admission. Every queued
//! closure keeps its admission lease until it returns, and operation registration
//! remains held while cancellation drains. Progress and resume publication live in
//! `import_progress`; cancellation invalidates the saved generation before notifying
//! the worker. The shared `/cancel` endpoint also retains saved-refinement cancellation.

use std::fs;
use std::path::Path;
use std::sync::Arc;

use axum::Json;
use axum::body::Bytes;
use axum::extract::{FromRequest, Multipart, Path as AxumPath, Request, State};
use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use serde_json::{Map, Value, json};
use speakerdesk_core::audio::{self, AudioError};
use speakerdesk_core::document::{Document, Job, JobKind};
use speakerdesk_core::error::ApiError;
use speakerdesk_core::ids::JobId;
use speakerdesk_core::store::JobStore;
use speakerdesk_core::{languages, limits, messages, transcript};
use tokio_util::sync::CancellationToken;

use super::{
    ACTIVE, AppState, NOT_FOUND, OperationGuard, RunningOperation, Shared, blocking, confined_file,
    error_response, from_api, import_progress, language_allowed, load_job, media_write_directory,
    meeting_id, new_job_id, refinement, require_job_id, secure_filename, separation, store_error,
    store_failure, stored_job_json, unix_now,
};
use crate::admission::Activity;

const PREPARING_AUDIO: &str = "Preparing local audio…";
const READY_FOR_INFERENCE: &str = "Ready for local inference or manual editing.";
const WAITING_FOR_INFERENCE: &str = "Waiting for local inference…";
const STARTING_INFERENCE: &str = "Starting local inference…";

const AUDIO_EXTENSIONS: &[&str] = &[
    ".wav", ".mp3", ".m4a", ".flac", ".ogg", ".aiff", ".aif", ".mp4", ".aac", ".webm",
];

/// Turns an uploaded file into a duration in seconds.
pub trait AudioPrep: Send + Sync {
    fn prepare(&self, directory: &Path, job: &Job) -> Result<f64, String>;
    fn prepare_cancellable(
        &self,
        directory: &Path,
        job: &Job,
        cancelled: &dyn Fn() -> bool,
    ) -> Result<f64, String> {
        if cancelled() {
            return Err("Audio preparation cancelled.".into());
        }
        let result = self.prepare(directory, job);
        if cancelled() {
            return Err("Audio preparation cancelled.".into());
        }
        result
    }
}

/// Copies a canonical WAV, or decodes anything else, via `speakerdesk_core::audio`.
#[derive(Debug, Default)]
pub struct WavPrepare;

impl AudioPrep for WavPrepare {
    fn prepare(&self, directory: &Path, job: &Job) -> Result<f64, String> {
        self.prepare_cancellable(directory, job, &|| false)
    }
    fn prepare_cancellable(
        &self,
        directory: &Path,
        job: &Job,
        cancelled: &dyn Fn() -> bool,
    ) -> Result<f64, String> {
        let Some(source_name) = job.source_file.as_deref() else {
            return Err(messages::AUDIO_PREP_FAILED.to_string());
        };
        let folder = media_write_directory(directory, job);
        fs::create_dir_all(&folder).map_err(|e| e.to_string())?;
        // The operation exclusively owns preparation. A prior process may have
        // exited between decode and atomic installation; retain source/audio.
        for entry in fs::read_dir(&folder).map_err(|e| e.to_string())? {
            let entry = entry.map_err(|e| e.to_string())?;
            let name = entry.file_name();
            let name = name.to_string_lossy();
            let kind = entry.file_type().map_err(|e| e.to_string())?;
            if name.starts_with(".speakerdesk-audio-")
                && (name.ends_with(".decode") || name.ends_with(".part"))
                && kind.is_file()
            {
                fs::remove_file(entry.path()).map_err(|e| e.to_string())?;
            } else if kind.is_dir()
                && name.strip_prefix(".speakerdesk-audio-").is_some_and(|id| {
                    id.len() == 36
                        && id.bytes().enumerate().all(|(index, byte)| {
                            if matches!(index, 8 | 13 | 18 | 23) {
                                byte == b'-'
                            } else {
                                byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte)
                            }
                        })
                })
            {
                // afconvert uses an equally private per-attempt directory.
                fs::remove_dir_all(entry.path()).map_err(|e| e.to_string())?;
            }
        }
        let source_root = job
            .extra
            .get("legacy_media_root")
            .and_then(Value::as_str)
            .map(Path::new)
            .unwrap_or(directory)
            .join(job.id.as_str());
        let source = confined_file(&source_root, source_name).map_err(|e| e.to_string())?;
        let destination = folder.join("audio.wav");
        match audio::normalize_cancellable(&source, &destination, cancelled) {
            Ok(duration) => Ok(duration),
            Err(AudioError::Message(message)) => Err(message.to_string()),
            Err(_) => Err(messages::AUDIO_PREP_FAILED.to_string()),
        }
    }
}

pub(super) async fn upload(State(app): State<AppState>, request: Request) -> Response {
    let mut multipart = match Multipart::from_request(request, &()).await {
        Ok(multipart) => multipart,
        Err(_) => return error_response(StatusCode::BAD_REQUEST, messages::BATCH_SIZE),
    };
    let mut language = languages::AUTOMATIC.to_string();
    let mut files = Vec::new();
    loop {
        let field = match multipart.next_field().await {
            Ok(Some(field)) => field,
            Ok(None) => break,
            Err(_) => return error_response(StatusCode::BAD_REQUEST, messages::INVALID_REQUEST),
        };
        match field.name() {
            Some("language") => match field.text().await {
                Ok(value) => language = value,
                Err(_) => {
                    return error_response(StatusCode::BAD_REQUEST, messages::INVALID_REQUEST);
                }
            },
            Some("files") => {
                let filename = field.file_name().unwrap_or("").to_string();
                match field.bytes().await {
                    Ok(bytes) => files.push((filename, bytes)),
                    Err(_) => {
                        return error_response(StatusCode::BAD_REQUEST, messages::INVALID_REQUEST);
                    }
                }
            }
            _ => {}
        }
    }
    blocking(move || {
        let lease = match app.admission.lease(Activity::Import) {
            Ok(lease) => Arc::new(lease),
            Err(error) => return error_response(StatusCode::CONFLICT, error.message()),
        };
        match accept_uploads(&app, &language, files) {
            Ok(created) => {
                for job in &created {
                    let lease = lease.clone();
                    let shared = Arc::clone(&app.shared);
                    let id = job.id.clone();
                    let generation = job
                        .extra
                        .get("operation_generation")
                        .and_then(Value::as_u64)
                        .unwrap_or(0);
                    let cancellation = CancellationToken::new();
                    app.shared.operations.lock().expect("operations").insert(
                        id.clone(),
                        RunningOperation {
                            generation,
                            cancellation: cancellation.clone(),
                            activity: Activity::Import,
                        },
                    );
                    app.worker.enqueue(move || {
                        let _lease = lease;
                        let _operation = OperationGuard {
                            shared: shared.clone(),
                            id: id.clone(),
                            generation,
                            cancellation,
                            parent: None,
                        };
                        prepare_job(&shared, &id);
                    });
                }
                (
                    StatusCode::CREATED,
                    Json(created.iter().map(stored_job_json).collect::<Vec<_>>()),
                )
                    .into_response()
            }
            Err(error) => from_api(error),
        }
    })
    .await
}

struct Upload {
    filename: String,
    bytes: Bytes,
}

fn accept_uploads(
    app: &AppState,
    language: &str,
    files: Vec<(String, Bytes)>,
) -> Result<Vec<Job>, ApiError> {
    if files.is_empty() || files.len() > limits::MAX_BATCH_FILES {
        return Err(ApiError::new(400, messages::BATCH_SIZE));
    }
    if !language_allowed(language) {
        return Err(ApiError::new(400, messages::SELECT_LANGUAGE));
    }
    let mut uploads = Vec::with_capacity(files.len());
    for (filename, bytes) in files {
        let suffix = path_suffix(&filename);
        if !AUDIO_EXTENSIONS.contains(&suffix.as_str()) {
            return Err(ApiError::new(400, messages::SUPPORTED_FORMATS));
        }
        uploads.push(Upload { filename, bytes });
    }
    let created_at = unix_now();
    let mut created = Vec::with_capacity(uploads.len());
    let mut staged = Vec::new();
    for upload in uploads {
        let id = match new_job_id() {
            Ok(id) => id,
            Err(error) => {
                remove_staged(&app.shared.data_dir, &staged);
                return Err(store_error(&error));
            }
        };
        let suffix = path_suffix(&upload.filename);
        staged.push(id.clone());
        if let Err(error) = write_source(&app.shared.data_dir, &id, &suffix, &upload.bytes) {
            remove_staged(&app.shared.data_dir, &staged);
            tracing::error!(%error, "could not store an upload");
            return Err(ApiError::new(500, messages::LOCAL_OPERATION_FAILED));
        }
        let name = secure_filename(&upload.filename);
        let job = Job {
            id,
            name: if name.is_empty() {
                "Recording".to_string()
            } else {
                name
            },
            kind: JobKind::Import,
            status: "preparing".to_string(),
            message: PREPARING_AUDIO.to_string(),
            language: language.to_string(),
            source_file: Some(format!("source{suffix}")),
            duration: None,
            revision: 0,
            created: created_at,
            updated: created_at,
            document: None,
            extra: Map::new(),
        };
        created.push(job);
    }
    let result = app
        .store()
        .insert_uploads(&created, ACTIVE, limits::MAX_BATCH_FILES as u64);
    match result {
        Ok(true) => Ok(created),
        Ok(false) => {
            remove_staged(&app.shared.data_dir, &staged);
            Err(ApiError::new(429, messages::QUEUE_FULL))
        }
        Err(error) => {
            remove_staged(&app.shared.data_dir, &staged);
            Err(store_error(&error))
        }
    }
}

fn prepare_job(shared: &Shared, id: &JobId) {
    let job = {
        let directory = shared.store.lock().expect("jobs");
        match directory.get(id, false) {
            Ok(Some(job)) => job,
            _ => return,
        }
    };
    if !matches!(job.status.as_str(), "preparing" | "queued") {
        return;
    }
    let cancellation = shared
        .operations
        .lock()
        .expect("operations")
        .get(id)
        .filter(|operation| {
            job.extra
                .get("operation_generation")
                .and_then(Value::as_u64)
                .unwrap_or(0)
                == operation.generation
        })
        .map(|operation| operation.cancellation.clone())
        .unwrap_or_default();
    let prepare = Arc::clone(&*shared.prepare.lock().expect("prepare"));
    let result =
        prepare.prepare_cancellable(&shared.data_dir, &job, &|| cancellation.is_cancelled());
    let _ = shared
        .store
        .lock()
        .expect("jobs")
        .update_job(id, |current| {
            if current.status != job.status
                || current.extra.get("operation_generation")
                    != job.extra.get("operation_generation")
            {
                return Err(ApiError::new(
                    409,
                    "Audio preparation is no longer current.",
                ));
            }
            match result {
                Ok(duration) => {
                    if current.status != "queued" {
                        current.status = "uploaded".to_string();
                    }
                    current.duration = Some(duration);
                    current.message = READY_FOR_INFERENCE.to_string();
                }
                Err(message) => {
                    current.status = "failed".to_string();
                    current.message = message;
                }
            }
            Ok::<_, ApiError>(())
        });
}

pub(super) async fn run(State(app): State<AppState>, AxumPath(jid): AxumPath<String>) -> Response {
    blocking(move || {
        let lease = match app.admission.lease(Activity::Import) {
            Ok(lease) => lease,
            Err(error) => return error_response(StatusCode::CONFLICT, error.message()),
        };
        let id = match require_job_id(&jid) {
            Ok(id) => id,
            Err(error) => return from_api(error),
        };
        let mut directory = app.shared.store.lock().expect("jobs");
        if app.admission.closed() {
            return error_response(StatusCode::CONFLICT, messages::UPDATE_RESERVED);
        }
        let mut job = match load_job(&directory, &id, false) {
            Ok(job) => job,
            Err(error) => return from_api(error),
        };
        if ACTIVE.contains(&job.status.as_str())
            || app
                .shared
                .operations
                .lock()
                .expect("operations")
                .contains_key(&id)
        {
            return error_response(StatusCode::CONFLICT, messages::JOB_ALREADY_RUNNING);
        }
        if job.extra.get("deletion_pending").and_then(Value::as_bool) == Some(true) {
            return error_response(
                StatusCode::CONFLICT,
                "Retry Delete to finish removing this recording.",
            );
        }
        if has_segments(&job) && !job.extra.contains_key("import_checkpoint") {
            return error_response(StatusCode::CONFLICT, messages::RERUN_WOULD_OVERWRITE);
        }
        if meeting_id(&app.shared).is_some() {
            return error_response(StatusCode::CONFLICT, messages::FINISH_MEETING_BEFORE_IMPORT);
        }
        let issues = app.shared.issues.lock().expect("issues").clone();
        if !issues.is_empty() {
            return error_response(StatusCode::CONFLICT, &issues.join(" "));
        }
        if job.language == languages::AUTOMATIC && !app.automatic_language_ready() {
            return error_response(StatusCode::CONFLICT, messages::LID_MODEL_MISSING);
        }
        let generation = job
            .extra
            .get("operation_generation")
            .and_then(Value::as_u64)
            .unwrap_or(0)
            + 1;
        job.extra
            .insert("operation_generation".into(), json!(generation));
        job.status = "queued".to_string();
        job.message = WAITING_FOR_INFERENCE.to_string();
        job.updated = unix_now();
        if let Err(error) = directory.put(&job, None) {
            return store_failure(&error);
        }
        let cancellation = CancellationToken::new();
        app.shared.operations.lock().expect("operations").insert(
            id.clone(),
            RunningOperation {
                generation,
                cancellation: cancellation.clone(),
                activity: Activity::Import,
            },
        );
        drop(directory);
        let body = stored_job_json(&job);
        let shared = Arc::clone(&app.shared);
        let admission = app.admission.clone();
        let queue = app.worker.queue_handle();
        let queued = id.clone();
        app.worker.enqueue(move || {
            execute_import(&shared, &queued, generation, &cancellation);
            drop(lease);
            queue.enqueue(move || {
                separation::automatic(&shared, &admission, &queued, generation);
            });
        });
        (StatusCode::ACCEPTED, Json(body)).into_response()
    })
    .await
}

pub(super) fn execute_import(
    shared: &Arc<Shared>,
    id: &JobId,
    generation: u64,
    cancellation: &CancellationToken,
) {
    let operation = OperationGuard {
        shared: shared.clone(),
        id: id.clone(),
        generation,
        cancellation: cancellation.clone(),
        parent: None,
    };
    let current = shared.store.lock().expect("store").get(id, false);
    if !matches!(current, Ok(Some(ref job)) if job.status == "queued" && job.extra.get("operation_generation").and_then(Value::as_u64) == Some(generation))
    {
        return;
    }
    if !audio_ready(shared, id) {
        return;
    }
    let started = shared.store.lock().expect("jobs").update_job(id, |job| {
        if cancellation.is_cancelled()
            || job.status != "queued"
            || job
                .extra
                .get("operation_generation")
                .and_then(Value::as_u64)
                != Some(generation)
        {
            return Err(ApiError::new(
                409,
                "The inference operation is no longer current.",
            ));
        }
        job.status = "processing".to_string();
        job.message = STARTING_INFERENCE.to_string();
        Ok::<_, ApiError>(())
    });
    let Ok(Some((job, ()))) = started else {
        return;
    };
    let models = Arc::clone(&*shared.models.lock().expect("models"));
    let inferred = import_progress::infer(shared, &operation, &job, models.as_ref())
        .and_then(|document| checked_document(document, job.duration.unwrap_or(0.0)));
    let mut store = shared.store.lock().expect("store");
    let result = store.update_job(id, |current| {
        if cancellation.is_cancelled()
            || current.status != "processing"
            || (current.revision != job.revision
                && !current.extra.contains_key("import_checkpoint"))
            || current
                .extra
                .get("operation_generation")
                .and_then(Value::as_u64)
                != Some(generation)
        {
            return Err(ApiError::new(
                409,
                "The inference operation is no longer current.",
            ));
        }
        match inferred {
            Ok(document) => {
                if current.extra.contains_key("import_checkpoint") {
                    if let Some(retained) = &mut current.document {
                        retained.provenance = document.provenance;
                        retained.warnings = document.warnings;
                    }
                } else {
                    current.document = Some(document);
                }
                current.extra.remove("import_checkpoint");
                current.extra.remove("import_resumable");
                current.status = "ready".into();
                current.message = messages::TRANSCRIPT_READY.into();
            }
            Err(message) => {
                current.status = "failed".into();
                current.message = if message.is_empty() {
                    messages::PROCESSING_FAILED.into()
                } else {
                    message
                };
            }
        }
        Ok::<_, ApiError>(())
    });
    let published = match result {
        Ok(Some((job, ()))) if job.status == "ready" => Some(job),
        Err(error) => {
            tracing::warn!(message = %error.message, "import result was not published");
            None
        }
        _ => None,
    };
    drop(store);
    if published.is_some() {
        operation.check_saved_voices(models.as_ref());
    }
}

fn audio_ready(shared: &Shared, id: &JobId) -> bool {
    let missing = {
        let directory = shared.store.lock().expect("jobs");
        match directory.get(id, false) {
            Ok(Some(job)) => duration_missing(job.duration),
            _ => return false,
        }
    };
    if missing {
        prepare_job(shared, id);
    }
    let directory = shared.store.lock().expect("jobs");
    match directory.get(id, false) {
        Ok(Some(job)) => job.status != "failed" && !duration_missing(job.duration),
        _ => false,
    }
}

pub(super) async fn cancel(
    State(app): State<AppState>,
    AxumPath(jid): AxumPath<String>,
) -> Response {
    blocking(move || {
        let id = match require_job_id(&jid) {
            Ok(id) => id,
            Err(error) => return from_api(error),
        };
        let mut store = app.store();
        match store.get(&id, false) {
            Ok(Some(job)) if job.kind == JobKind::Import && job.status == "uploaded" => {
                return Json(stored_job_json(&job)).into_response();
            }
            Err(error) => return store_failure(&error),
            _ => (),
        }
        let result = store.update_job(&id, |job| {
            let running = app
                .shared
                .operations
                .lock()
                .expect("operations")
                .get(&id)
                .cloned();
            let refinement = running.as_ref().is_some_and(|op| {
                op.activity == Activity::Refinement
                    && job
                        .extra
                        .get("operation_generation")
                        .and_then(Value::as_u64)
                        == Some(op.generation)
            });
            if !refinement
                && (job.kind != JobKind::Import
                    || !["preparing", "queued", "processing"].contains(&job.status.as_str()))
            {
                return Err(ApiError::new(
                    409,
                    "There is no import transcription to cancel.",
                ));
            }
            let generation = job
                .extra
                .get("operation_generation")
                .and_then(Value::as_u64)
                .unwrap_or(0);
            job.extra
                .insert("operation_generation".into(), json!(generation + 1));
            refinement::interrupt(job);
            separation::interrupt(job);
            if job
                .extra
                .remove("passage_retry")
                .is_some_and(|value| !value.is_null())
            {
                job.message = "Passage retry cancelled. Existing words are retained.".into();
            }
            if !refinement {
                job.status = "uploaded".into();
                job.message =
                    "Processing cancelled. Uploaded audio and saved words are kept.".into();
            }
            let cancellation = running
                .filter(|op| op.generation == generation)
                .map(|op| op.cancellation);
            Ok::<_, ApiError>(cancellation)
        });
        match result {
            Ok(Some((job, cancellation))) => {
                // Invalidate publication durably before notifying the worker.
                if let Some(cancellation) = cancellation {
                    cancellation.cancel();
                }
                Json(stored_job_json(&job)).into_response()
            }
            Ok(None) => error_response(StatusCode::NOT_FOUND, NOT_FOUND),
            Err(error) => from_api(error),
        }
    })
    .await
}

pub(super) fn checked_document(document: Document, duration: f64) -> Result<Document, String> {
    let value = serde_json::to_value(&document).map_err(|error| error.to_string())?;
    transcript::validate(&value, duration).map_err(|error| error.message)
}

fn duration_missing(duration: Option<f64>) -> bool {
    duration.is_none_or(|value| value == 0.0)
}

fn has_segments(job: &Job) -> bool {
    job.document
        .as_ref()
        .is_some_and(|document| !document.segments.is_empty())
}

fn remove_staged(root: &Path, ids: &[JobId]) {
    for id in ids {
        remove_job_dir(root, id);
    }
}

fn write_source(root: &Path, id: &JobId, suffix: &str, bytes: &[u8]) -> std::io::Result<()> {
    let folder = root.join(id.as_str());
    fs::create_dir_all(&folder)?;
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        let _ = fs::set_permissions(&folder, fs::Permissions::from_mode(0o700));
    }
    fs::write(folder.join(format!("source{suffix}")), bytes)
}

fn remove_job_dir(root: &Path, id: &JobId) {
    let folder = root.join(id.as_str());
    if folder.starts_with(root) {
        let _ = fs::remove_dir_all(folder);
    }
}

fn path_suffix(filename: &str) -> String {
    let name = filename.rsplit(['/', '\\']).next().unwrap_or(filename);
    let Some(index) = name.rfind('.') else {
        return String::new();
    };
    if index == 0 {
        return String::new();
    }
    name[index..].to_ascii_lowercase()
}
