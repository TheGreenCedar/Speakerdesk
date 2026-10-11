//! HTTP routes for import jobs.
//!
//! Handlers speak the v0.6.6 job API. Mutating requests compare
//! `X-Speakerdesk-Token` with [`speakerdesk_core::security::tokens_match`].
//! Transcript writes go through [`speakerdesk_core::edit`]. Import runs are
//! queued on [`crate::worker::JobWorker`]. Tests supply a directory and
//! [`ImportModels`] so nothing here downloads weights or spawns MLX.

use std::fs;
use std::io::Read;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
use std::time::{SystemTime, UNIX_EPOCH};

use axum::Json;
use axum::Router;
use axum::body::Bytes;
use axum::extract::{
    DefaultBodyLimit, FromRequest, Multipart, Path as AxumPath, Query, Request, State,
};
use axum::http::header::{self, HeaderName, HeaderValue};
use axum::http::{HeaderMap, Method, StatusCode};
use axum::middleware::{self, Next};
use axum::response::{IntoResponse, Response};
use axum::routing::{get, patch, post, put};
use serde::Deserialize;
use serde_json::{Map, Value, json};
use speakerdesk_core::audio::{self, AudioError};
use speakerdesk_core::document::{Document, Job, JobKind, JobSummary};
use speakerdesk_core::edit::{self, PassageEditInput, SaveInput};
use speakerdesk_core::error::{ApiError, StoreError};
use speakerdesk_core::ids::JobId;
use speakerdesk_core::security::{self, MAX_UPLOAD_BYTES};
use speakerdesk_core::{export, languages, limits, messages, transcript};

use crate::http::{security_headers, token_header};
use crate::worker::JobWorker;

const NOT_FOUND: &str = "The requested URL was not found on the server. If you entered the URL manually please check your spelling and try again.";
const PREPARING_AUDIO: &str = "Preparing local audio…";
const READY_FOR_INFERENCE: &str = "Ready for local inference or manual editing.";
const WAITING_FOR_INFERENCE: &str = "Waiting for local inference…";
const STARTING_INFERENCE: &str = "Starting local inference…";
const CONFIG_NOTE: &str = "Model execution is checked when inference starts. Configuration is not proof of working inference.";
const MODEL_NAME: &str = "nvidia/Nemotron-3-Diarization";
const MODEL_SPEAKER_LIMIT: i64 = 8;
const ACTIVE: &[&str] = &[
    "preparing",
    "queued",
    "processing",
    "starting",
    "recording",
    "paused",
    "finishing",
];
const PRIVATE_FIELDS: &[&str] = &[
    "rolling_sources",
    "refinement_history",
    "fast_history",
    "fast_previous_revision",
    "boundary_candidate",
    "boundary_candidates",
    "rolling_inflight",
    "fast_retained_candidate",
];
const EXPORT_KINDS: &[&str] = &["txt", "srt", "vtt", "json"];
const AUDIO_EXTENSIONS: &[&str] = &[
    ".wav", ".mp3", ".m4a", ".flac", ".ogg", ".aiff", ".aif", ".mp4", ".aac", ".webm",
];

/// In-memory jobs. Route tests use this, or another [`JobDirectory`], instead of SQLite.
pub trait JobDirectory: Send {
    fn get(&mut self, id: &JobId, metadata: bool) -> Result<Option<Job>, StoreError>;
    fn summaries(&mut self) -> Result<Vec<JobSummary>, StoreError>;
    fn count_status(&mut self, statuses: &[&str]) -> Result<u64, StoreError>;
    fn put(&mut self, job: &Job) -> Result<(), StoreError>;
    fn delete(&mut self, id: &JobId) -> Result<bool, StoreError>;
}

/// Import inference behind `POST /api/jobs/:jid/run`.
///
/// Implementations must not download models or spawn MLX. The default reports
/// that setup is unfinished.
pub trait ImportModels: Send + Sync {
    fn infer(&self, job: &Job) -> Result<Document, String>;
}

/// Turns an uploaded file into a duration in seconds.
pub trait AudioPrep: Send + Sync {
    fn prepare(&self, directory: &Path, job: &Job) -> Result<f64, String>;
}

/// Local stand-in for the job table.
#[derive(Debug, Default)]
pub struct MemoryJobs {
    jobs: Vec<Job>,
}

impl JobDirectory for MemoryJobs {
    fn get(&mut self, id: &JobId, metadata: bool) -> Result<Option<Job>, StoreError> {
        let Some(job) = self.jobs.iter().find(|job| &job.id == id) else {
            return Ok(None);
        };
        let mut job = job.clone();
        if metadata {
            job.document = None;
            for key in PRIVATE_FIELDS {
                job.extra.remove(*key);
            }
        }
        Ok(Some(job))
    }

    fn summaries(&mut self) -> Result<Vec<JobSummary>, StoreError> {
        let mut jobs = self.jobs.clone();
        jobs.sort_by(|left, right| {
            right
                .created
                .total_cmp(&left.created)
                .then_with(|| right.id.as_str().cmp(left.id.as_str()))
        });
        Ok(jobs.into_iter().map(summary_of).collect())
    }

    fn count_status(&mut self, statuses: &[&str]) -> Result<u64, StoreError> {
        let count = self
            .jobs
            .iter()
            .filter(|job| statuses.contains(&job.status.as_str()))
            .count();
        Ok(u64::try_from(count).unwrap_or(u64::MAX))
    }

    fn put(&mut self, job: &Job) -> Result<(), StoreError> {
        if let Some(existing) = self.jobs.iter_mut().find(|stored| stored.id == job.id) {
            *existing = job.clone();
        } else {
            self.jobs.push(job.clone());
        }
        Ok(())
    }

    fn delete(&mut self, id: &JobId) -> Result<bool, StoreError> {
        let before = self.jobs.len();
        self.jobs.retain(|job| &job.id != id);
        Ok(self.jobs.len() != before)
    }
}

/// Infer returns the setup message and does not touch a model runtime.
#[derive(Debug, Default)]
pub struct UnavailableModels;

impl ImportModels for UnavailableModels {
    fn infer(&self, _job: &Job) -> Result<Document, String> {
        Err(messages::SETUP_BEFORE_TRANSCRIPTION.to_string())
    }
}

/// Copies a canonical WAV, or decodes anything else, via `speakerdesk_core::audio`.
#[derive(Debug, Default)]
pub struct WavPrepare;

impl AudioPrep for WavPrepare {
    fn prepare(&self, directory: &Path, job: &Job) -> Result<f64, String> {
        let Some(source_name) = job.source_file.as_deref() else {
            return Err(messages::AUDIO_PREP_FAILED.to_string());
        };
        let folder = directory.join(job.id.as_str());
        let source = folder.join(source_name);
        let destination = folder.join("audio.wav");
        match audio::normalize(&source, &destination) {
            Ok(duration) => Ok(duration),
            Err(AudioError::Message(message)) => Err(message.to_string()),
            Err(_) => Err(messages::AUDIO_PREP_FAILED.to_string()),
        }
    }
}

struct Shared {
    directory: Mutex<Box<dyn JobDirectory + Send>>,
    models: Mutex<Arc<dyn ImportModels>>,
    prepare: Mutex<Arc<dyn AudioPrep>>,
    meeting: Mutex<Option<JobId>>,
    issues: Mutex<Vec<String>>,
    default_language: Mutex<String>,
    data_dir: PathBuf,
}

/// Shared route state. Clone is cheap; the worker and directory are shared.
#[derive(Clone)]
pub struct AppState {
    token: Arc<str>,
    shared: Arc<Shared>,
    worker: Arc<JobWorker>,
}

impl AppState {
    pub fn new() -> Self {
        Self::with_directory(Box::new(MemoryJobs::default()))
    }

    pub fn with_directory(directory: Box<dyn JobDirectory + Send>) -> Self {
        let data_dir = std::env::temp_dir().join(format!("speakerdesk-{}", random_hex(8)));
        let _ = fs::create_dir_all(&data_dir);
        Self {
            token: Arc::from(random_hex(32)),
            shared: Arc::new(Shared {
                directory: Mutex::new(directory),
                models: Mutex::new(Arc::new(UnavailableModels)),
                prepare: Mutex::new(Arc::new(WavPrepare)),
                meeting: Mutex::new(None),
                issues: Mutex::new(default_issues()),
                default_language: Mutex::new(languages::AUTOMATIC.to_string()),
                data_dir,
            }),
            worker: Arc::new(JobWorker::new()),
        }
    }

    pub fn token(&self) -> &str {
        &self.token
    }

    pub fn data_dir(&self) -> &Path {
        &self.shared.data_dir
    }

    pub fn drain(&self) {
        self.worker.drain();
    }

    pub fn insert_job(&self, job: Job) {
        self.shared
            .directory
            .lock()
            .expect("jobs")
            .put(&job)
            .expect("insert job");
    }

    pub fn set_meeting(&self, id: Option<JobId>) {
        *self.shared.meeting.lock().expect("meeting") = id;
    }

    pub fn set_issues(&self, issues: Vec<String>) {
        *self.shared.issues.lock().expect("issues") = issues;
    }

    pub fn set_models(&self, models: Arc<dyn ImportModels>) {
        *self.shared.models.lock().expect("models") = models;
    }
}

impl Default for AppState {
    fn default() -> Self {
        Self::new()
    }
}

/// Placeholder kept for the people and meeting route agents.
#[derive(Debug, Default)]
pub struct Routes;

impl Routes {
    pub fn new() -> Self {
        Self
    }
}

/// Job routes, including token checks and the v0.6.6 security headers.
pub fn router(state: AppState) -> Router {
    Router::new()
        .merge(crate::people_routes::routes())
        .merge(crate::meeting::routes())
        .route("/api/config", get(config))
        .route("/api/jobs", get(list_jobs).post(create_jobs))
        .route("/api/jobs/{jid}", get(job_detail).delete(delete_job))
        .route("/api/jobs/{jid}/transcript", put(save_transcript))
        .route("/api/jobs/{jid}/segments/{sid}", patch(edit_passage))
        .route("/api/jobs/{jid}/export/{kind}", get(export_job))
        .route("/api/jobs/{jid}/run", post(run_job))
        .layer(DefaultBodyLimit::max(MAX_UPLOAD_BYTES))
        .layer(middleware::from_fn_with_state(state.clone(), guard))
        .with_state(state)
}

async fn guard(State(app): State<AppState>, request: Request, next: Next) -> Response {
    let host = request
        .headers()
        .get(header::HOST)
        .and_then(|value| value.to_str().ok())
        .unwrap_or("");
    if !security::host_allowed(host) {
        return with_security(error_response(
            StatusCode::BAD_REQUEST,
            messages::INVALID_REQUEST,
        ));
    }
    if is_mutating(request.method()) && !token_ok(&app, request.headers()) {
        return with_security(error_response(
            StatusCode::FORBIDDEN,
            messages::TOKEN_MISMATCH,
        ));
    }
    let mut response = next.run(request).await;
    apply_security(&mut response);
    response
}

async fn config(State(app): State<AppState>) -> Response {
    let issues = app.shared.issues.lock().expect("issues").clone();
    let default_language = app
        .shared
        .default_language
        .lock()
        .expect("language")
        .clone();
    Json(json!({
        "languages": language_choices(),
        "default_language": default_language,
        "readiness": readiness_json(&issues),
        "decoder": decoder_name(),
    }))
    .into_response()
}

async fn list_jobs(State(app): State<AppState>) -> Response {
    let mut directory = app.shared.directory.lock().expect("jobs");
    match directory.summaries() {
        Ok(summaries) => Json(summaries).into_response(),
        Err(error) => store_failure(&error),
    }
}

async fn create_jobs(State(app): State<AppState>, request: Request) -> Response {
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
    match accept_uploads(&app, &language, files) {
        Ok(created) => {
            for job in &created {
                let shared = Arc::clone(&app.shared);
                let id = job.id.clone();
                app.worker.enqueue(move || prepare_job(&shared, &id));
            }
            (
                StatusCode::CREATED,
                Json(created.iter().map(stored_job_json).collect::<Vec<_>>()),
            )
                .into_response()
        }
        Err(error) => from_api(error),
    }
}

async fn job_detail(
    State(app): State<AppState>,
    AxumPath(jid): AxumPath<String>,
    Query(query): Query<DetailQuery>,
) -> Response {
    let known = match query.known_revision.as_deref() {
        Some(raw) => match parse_known_revision(raw) {
            Ok(value) => Some(value),
            Err(()) => {
                return error_response(StatusCode::BAD_REQUEST, messages::KNOWN_REVISION);
            }
        },
        None => None,
    };
    let id = match require_job_id(&jid) {
        Ok(id) => id,
        Err(error) => return from_api(error),
    };
    let mut directory = app.shared.directory.lock().expect("jobs");
    let metadata = known.is_some();
    let job = match load_job(&mut **directory, &id, metadata) {
        Ok(job) => job,
        Err(error) => return from_api(error),
    };
    let (job, unchanged) = if let Some(known) = known {
        let unchanged = known == i128::from(job.revision);
        if unchanged {
            (job, Some(true))
        } else {
            match load_job(&mut **directory, &id, false) {
                Ok(full) => (full, Some(false)),
                Err(error) => return from_api(error),
            }
        }
    } else {
        (job, None)
    };
    let owned = meeting_id(&app.shared).as_ref() == Some(&id);
    Json(detail_json(
        &job,
        known.is_some() && unchanged == Some(true),
        unchanged,
        owned,
    ))
    .into_response()
}

async fn save_transcript(
    State(app): State<AppState>,
    AxumPath(jid): AxumPath<String>,
    body: Bytes,
) -> Response {
    let id = match require_job_id(&jid) {
        Ok(id) => id,
        Err(error) => return from_api(error),
    };
    let body = match json_body(&body) {
        Ok(body) => body,
        Err(error) => return from_api(error),
    };
    let mut directory = app.shared.directory.lock().expect("jobs");
    let mut job = match load_job(&mut **directory, &id, false) {
        Ok(job) => job,
        Err(error) => return from_api(error),
    };
    let assignments = map_field(&job.extra, "speaker_assignments");
    let checks = map_field(&job.extra, "voice_checks");
    let outcome = match edit::save_transcript(SaveInput {
        status: &job.status,
        duration: job.duration,
        revision: job.revision,
        saved: job.document.as_ref(),
        body: &body,
        now: unix_now(),
        speaker_assignments: &assignments,
        voice_checks: &checks,
    }) {
        Ok(outcome) => outcome,
        Err(error) => return from_api(error),
    };
    job.document = Some(outcome.document);
    job.revision = outcome.revision;
    job.status = outcome.status.to_string();
    job.message = outcome.message.to_string();
    job.extra.insert(
        "speaker_assignments".to_string(),
        Value::Object(outcome.speaker_assignments),
    );
    job.extra.insert(
        "voice_checks".to_string(),
        Value::Object(outcome.voice_checks),
    );
    if outcome.clear_voice_suggestions {
        job.extra.remove("voice_suggestions");
    }
    job.updated = unix_now();
    if let Err(error) = directory.put(&job) {
        return store_failure(&error);
    }
    Json(stored_job_json(&job)).into_response()
}

async fn edit_passage(
    State(app): State<AppState>,
    AxumPath((jid, sid)): AxumPath<(String, String)>,
    body: Bytes,
) -> Response {
    let id = match require_job_id(&jid) {
        Ok(id) => id,
        Err(error) => return from_api(error),
    };
    let body = match json_body(&body) {
        Ok(body) => body,
        Err(error) => return from_api(error),
    };
    let mut directory = app.shared.directory.lock().expect("jobs");
    let mut job = match load_job(&mut **directory, &id, false) {
        Ok(job) => job,
        Err(error) => return from_api(error),
    };
    let outcome = match edit::edit_passage(PassageEditInput {
        status: &job.status,
        duration: job.duration.unwrap_or(0.0),
        revision: job.revision,
        document: job.document.as_ref(),
        segment_id: &sid,
        body: &body,
        now: unix_now(),
    }) {
        Ok(outcome) => outcome,
        Err(error) => return from_api(error),
    };
    job.document = Some(outcome.document);
    job.revision = outcome.revision;
    job.updated = unix_now();
    let segment = outcome.segment;
    if let Err(error) = directory.put(&job) {
        return store_failure(&error);
    }
    Json(json!({
        "segment": segment,
        "revision": outcome.revision,
    }))
    .into_response()
}

async fn export_job(
    State(app): State<AppState>,
    AxumPath((jid, kind)): AxumPath<(String, String)>,
) -> Response {
    if !EXPORT_KINDS.contains(&kind.as_str()) {
        return error_response(StatusCode::NOT_FOUND, NOT_FOUND);
    }
    let id = match require_job_id(&jid) {
        Ok(id) => id,
        Err(error) => return from_api(error),
    };
    let mut directory = app.shared.directory.lock().expect("jobs");
    let job = match load_job(&mut **directory, &id, false) {
        Ok(job) => job,
        Err(error) => return from_api(error),
    };
    let Some(mut document) = job.document.clone() else {
        return error_response(StatusCode::CONFLICT, messages::EXPORT_NEEDS_TRANSCRIPT);
    };
    if kind == "json"
        && let Some(assignments) = job.extra.get("speaker_assignments")
    {
        document
            .rest
            .insert("speaker_assignments".to_string(), assignments.clone());
    }
    match export::export(&document, &kind) {
        Ok(rendered) => export_response(rendered, download_name(&job.name, &kind)),
        Err(_) => error_response(StatusCode::PAYLOAD_TOO_LARGE, messages::EXPORT_TOO_LARGE),
    }
}

async fn delete_job(State(app): State<AppState>, AxumPath(jid): AxumPath<String>) -> Response {
    let id = match require_job_id(&jid) {
        Ok(id) => id,
        Err(error) => return from_api(error),
    };
    let mut directory = app.shared.directory.lock().expect("jobs");
    let job = match load_job(&mut **directory, &id, false) {
        Ok(job) => job,
        Err(error) => return from_api(error),
    };
    if ACTIVE.contains(&job.status.as_str()) || meeting_id(&app.shared).as_ref() == Some(&id) {
        return error_response(StatusCode::CONFLICT, messages::DELETE_BUSY);
    }
    if let Err(error) = directory.delete(&id) {
        return store_failure(&error);
    }
    drop(directory);
    remove_job_dir(&app.shared.data_dir, &id);
    Json(json!({ "deleted": true })).into_response()
}

async fn run_job(State(app): State<AppState>, AxumPath(jid): AxumPath<String>) -> Response {
    let id = match require_job_id(&jid) {
        Ok(id) => id,
        Err(error) => return from_api(error),
    };
    let mut directory = app.shared.directory.lock().expect("jobs");
    let mut job = match load_job(&mut **directory, &id, false) {
        Ok(job) => job,
        Err(error) => return from_api(error),
    };
    if ACTIVE.contains(&job.status.as_str()) {
        return error_response(StatusCode::CONFLICT, messages::JOB_ALREADY_RUNNING);
    }
    if has_segments(&job) {
        return error_response(StatusCode::CONFLICT, messages::RERUN_WOULD_OVERWRITE);
    }
    if meeting_id(&app.shared).is_some() {
        return error_response(StatusCode::CONFLICT, messages::FINISH_MEETING_BEFORE_IMPORT);
    }
    let issues = app.shared.issues.lock().expect("issues").clone();
    if !issues.is_empty() {
        return error_response(StatusCode::CONFLICT, &issues.join(" "));
    }
    job.status = "queued".to_string();
    job.message = WAITING_FOR_INFERENCE.to_string();
    job.updated = unix_now();
    if let Err(error) = directory.put(&job) {
        return store_failure(&error);
    }
    drop(directory);
    let body = stored_job_json(&job);
    let shared = Arc::clone(&app.shared);
    let queued = id.clone();
    app.worker.enqueue(move || execute_import(&shared, &queued));
    (StatusCode::ACCEPTED, Json(body)).into_response()
}

#[derive(Debug, Deserialize)]
struct DetailQuery {
    known_revision: Option<String>,
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
    let mut directory = app.shared.directory.lock().expect("jobs");
    let active = match directory.count_status(ACTIVE) {
        Ok(active) => active,
        Err(error) => return Err(store_error(&error)),
    };
    if active.saturating_add(uploads.len() as u64) > limits::MAX_BATCH_FILES as u64 {
        return Err(ApiError::new(429, messages::QUEUE_FULL));
    }
    let created_at = unix_now();
    let mut created = Vec::with_capacity(uploads.len());
    let mut staged = Vec::new();
    for upload in uploads {
        let id = match new_job_id() {
            Ok(id) => id,
            Err(error) => {
                rollback(&mut **directory, &app.shared.data_dir, &staged);
                return Err(store_error(&error));
            }
        };
        let suffix = path_suffix(&upload.filename);
        if let Err(error) = write_source(&app.shared.data_dir, &id, &suffix, &upload.bytes) {
            rollback(&mut **directory, &app.shared.data_dir, &staged);
            tracing::error!(%error, "could not store an upload");
            return Err(ApiError::new(500, messages::LOCAL_OPERATION_FAILED));
        }
        staged.push(id.clone());
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
        if let Err(error) = directory.put(&job) {
            rollback(&mut **directory, &app.shared.data_dir, &staged);
            return Err(store_error(&error));
        }
        created.push(job);
    }
    Ok(created)
}

fn prepare_job(shared: &Shared, id: &JobId) {
    let job = {
        let mut directory = shared.directory.lock().expect("jobs");
        match directory.get(id, false) {
            Ok(Some(job)) => job,
            _ => return,
        }
    };
    let prepare = Arc::clone(&*shared.prepare.lock().expect("prepare"));
    let result = prepare.prepare(&shared.data_dir, &job);
    let mut directory = shared.directory.lock().expect("jobs");
    let Ok(Some(mut job)) = directory.get(id, false) else {
        return;
    };
    match result {
        Ok(duration) => {
            job.status = "uploaded".to_string();
            job.duration = Some(duration);
            job.message = READY_FOR_INFERENCE.to_string();
        }
        Err(message) => {
            job.status = "failed".to_string();
            job.message = message;
        }
    }
    job.updated = unix_now();
    let _ = directory.put(&job);
}

fn execute_import(shared: &Shared, id: &JobId) {
    if !audio_ready(shared, id) {
        return;
    }
    {
        let mut directory = shared.directory.lock().expect("jobs");
        let Ok(Some(mut job)) = directory.get(id, false) else {
            return;
        };
        if job.status == "failed" {
            return;
        }
        job.status = "processing".to_string();
        job.message = STARTING_INFERENCE.to_string();
        job.updated = unix_now();
        if directory.put(&job).is_err() {
            return;
        }
    }
    let job = {
        let mut directory = shared.directory.lock().expect("jobs");
        match directory.get(id, false) {
            Ok(Some(job)) => job,
            _ => return,
        }
    };
    let models = Arc::clone(&*shared.models.lock().expect("models"));
    let inferred = models.infer(&job);
    let mut directory = shared.directory.lock().expect("jobs");
    let Ok(Some(mut job)) = directory.get(id, false) else {
        return;
    };
    match inferred {
        Ok(document) => match checked_document(document, job.duration.unwrap_or(0.0)) {
            Ok(document) => {
                job.document = Some(document);
                job.revision += 1;
                job.status = "ready".to_string();
                job.message = messages::TRANSCRIPT_READY.to_string();
            }
            Err(message) => {
                job.status = "failed".to_string();
                job.message = message;
            }
        },
        Err(message) => {
            job.status = "failed".to_string();
            job.message = if message.is_empty() {
                messages::PROCESSING_FAILED.to_string()
            } else {
                message
            };
        }
    }
    job.updated = unix_now();
    let _ = directory.put(&job);
}

fn audio_ready(shared: &Shared, id: &JobId) -> bool {
    let missing = {
        let mut directory = shared.directory.lock().expect("jobs");
        match directory.get(id, false) {
            Ok(Some(job)) => duration_missing(job.duration),
            _ => return false,
        }
    };
    if missing {
        prepare_job(shared, id);
    }
    let mut directory = shared.directory.lock().expect("jobs");
    match directory.get(id, false) {
        Ok(Some(job)) => job.status != "failed" && !duration_missing(job.duration),
        _ => false,
    }
}

fn checked_document(document: Document, duration: f64) -> Result<Document, String> {
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

fn load_job(directory: &mut dyn JobDirectory, id: &JobId, metadata: bool) -> Result<Job, ApiError> {
    match directory.get(id, metadata) {
        Ok(Some(job)) => Ok(job),
        Ok(None) => Err(ApiError::new(404, NOT_FOUND)),
        Err(error) => Err(store_error(&error)),
    }
}

fn require_job_id(jid: &str) -> Result<JobId, ApiError> {
    JobId::parse(jid).map_err(|_| ApiError::new(404, NOT_FOUND))
}

fn meeting_id(shared: &Shared) -> Option<JobId> {
    shared.meeting.lock().expect("meeting").clone()
}

fn rollback(directory: &mut dyn JobDirectory, root: &Path, ids: &[JobId]) {
    for id in ids {
        let _ = directory.delete(id);
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

fn stored_job_json(job: &Job) -> Value {
    let mut view = job.clone();
    for key in PRIVATE_FIELDS {
        view.extra.remove(*key);
    }
    let mut value = serde_json::to_value(&view).expect("job serializes");
    if let Some(object) = value.as_object_mut() {
        object.entry("duration").or_insert(Value::Null);
        object.entry("document").or_insert(Value::Null);
    }
    value
}

fn detail_json(job: &Job, metadata: bool, unchanged: Option<bool>, owned: bool) -> Value {
    let mut value = stored_job_json(job);
    if metadata && let Some(object) = value.as_object_mut() {
        object.remove("document");
    }
    if let Some(unchanged) = unchanged {
        value["unchanged"] = Value::Bool(unchanged);
    }
    value["inference_owned"] = Value::Bool(owned);
    value
}

fn summary_of(job: Job) -> JobSummary {
    JobSummary {
        id: job.id,
        name: job.name,
        kind: Some(job.kind),
        source_file: job.source_file,
        created: job.created,
        updated: job.updated,
        status: job.status,
        message: Some(job.message),
        language: Some(job.language),
        duration: job.duration,
        revision: Some(job.revision),
    }
}

fn map_field(extra: &Map<String, Value>, key: &str) -> Map<String, Value> {
    extra
        .get(key)
        .and_then(Value::as_object)
        .cloned()
        .unwrap_or_default()
}

fn json_body(bytes: &[u8]) -> Result<Value, ApiError> {
    if bytes.is_empty() {
        return Err(ApiError::new(400, messages::INVALID_REQUEST));
    }
    serde_json::from_slice(bytes).map_err(|_| ApiError::new(400, messages::INVALID_REQUEST))
}

fn parse_known_revision(raw: &str) -> Result<i128, ()> {
    if raw.is_empty() || raw.len() > 20 || !raw.bytes().all(|byte| byte.is_ascii_digit()) {
        return Err(());
    }
    raw.parse().map_err(|_| ())
}

fn language_allowed(language: &str) -> bool {
    language == languages::AUTOMATIC || languages::is_supported(language)
}

fn language_choices() -> Value {
    let mut choices = Map::new();
    choices.insert(
        languages::AUTOMATIC.to_string(),
        Value::String("Automatic".to_string()),
    );
    for (code, name) in languages::LANGUAGES {
        choices.insert((*code).to_string(), Value::String((*name).to_string()));
    }
    Value::Object(choices)
}

fn readiness_json(issues: &[String]) -> Value {
    let automatic = !issues
        .iter()
        .any(|issue| issue == messages::LID_MODEL_MISSING);
    json!({
        "configured": issues.is_empty(),
        "issues": issues,
        "automatic_language": automatic,
        "model": MODEL_NAME,
        "speaker_limit": MODEL_SPEAKER_LIMIT,
        "device": "mlx",
        "note": CONFIG_NOTE,
    })
}

fn default_issues() -> Vec<String> {
    [
        messages::MISSING_DIARIZATION,
        messages::MISSING_ASR,
        messages::DIAR_PYTHON,
        messages::ASR_PYTHON,
        messages::SPEECH_MODEL_MISSING,
        messages::LID_MODEL_MISSING,
        messages::TIMING_NOT_READY,
    ]
    .into_iter()
    .map(str::to_string)
    .collect()
}

fn decoder_name() -> &'static str {
    if command_on_path("ffmpeg") {
        "FFmpeg"
    } else if command_on_path("afconvert") {
        "macOS AudioToolbox"
    } else {
        "PCM WAV only"
    }
}

fn command_on_path(name: &str) -> bool {
    let Some(path) = std::env::var_os("PATH") else {
        return false;
    };
    std::env::split_paths(&path).any(|dir| dir.join(name).is_file())
}

fn download_name(job_name: &str, kind: &str) -> String {
    let stem = Path::new(job_name)
        .file_stem()
        .and_then(|stem| stem.to_str())
        .unwrap_or("transcript");
    let secured = secure_filename(&format!("{stem}.{kind}"));
    if secured.is_empty() {
        format!("transcript.{kind}")
    } else {
        secured
    }
}

fn secure_filename(name: &str) -> String {
    let base = Path::new(name)
        .file_name()
        .and_then(|value| value.to_str())
        .unwrap_or(name);
    let mut out = String::new();
    let mut spaced = false;
    for ch in base.chars() {
        if ch.is_ascii_alphanumeric() || ch == '.' || ch == '-' {
            if spaced && !out.is_empty() {
                out.push('_');
            }
            spaced = false;
            out.push(ch);
        } else if ch.is_whitespace() {
            spaced = true;
        }
    }
    while out.starts_with(['.', '_']) {
        out.remove(0);
    }
    out
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

fn export_response(rendered: export::Export, filename: String) -> Response {
    let mut response = (StatusCode::OK, rendered.body).into_response();
    if let Ok(content_type) = HeaderValue::from_str(rendered.content_type) {
        response
            .headers_mut()
            .insert(header::CONTENT_TYPE, content_type);
    }
    let disposition = format!("attachment; filename=\"{filename}\"");
    if let Ok(disposition) = HeaderValue::from_str(&disposition) {
        response
            .headers_mut()
            .insert(header::CONTENT_DISPOSITION, disposition);
    }
    response
}

fn token_ok(app: &AppState, headers: &HeaderMap) -> bool {
    let provided = headers
        .get(token_header())
        .map(HeaderValue::as_bytes)
        .unwrap_or(b"");
    security::tokens_match(app.token.as_bytes(), provided)
}

fn is_mutating(method: &Method) -> bool {
    matches!(
        *method,
        Method::POST | Method::PUT | Method::PATCH | Method::DELETE
    )
}

fn error_response(status: StatusCode, message: &str) -> Response {
    (status, Json(json!({ "error": message }))).into_response()
}

fn from_api(error: ApiError) -> Response {
    let status = StatusCode::from_u16(error.status).unwrap_or(StatusCode::INTERNAL_SERVER_ERROR);
    error_response(status, &error.message)
}

fn store_error(error: &StoreError) -> ApiError {
    tracing::error!(%error, "job store request failed");
    ApiError::new(500, messages::LOCAL_OPERATION_FAILED)
}

fn store_failure(error: &StoreError) -> Response {
    from_api(store_error(error))
}

fn with_security(mut response: Response) -> Response {
    apply_security(&mut response);
    response
}

fn apply_security(response: &mut Response) {
    for (name, value) in security_headers() {
        response.headers_mut().insert(
            HeaderName::from_static(name),
            HeaderValue::from_static(value),
        );
    }
}

fn unix_now() -> f64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|duration| duration.as_secs_f64())
        .unwrap_or(0.0)
}

fn new_job_id() -> Result<JobId, StoreError> {
    JobId::parse(random_hex(16)).map_err(|error| StoreError::message(error.to_string()))
}

fn random_hex(bytes: usize) -> String {
    let mut raw = vec![0_u8; bytes];
    let mut file = fs::File::open("/dev/urandom").expect("urandom");
    file.read_exact(&mut raw).expect("urandom");
    if bytes == 16 {
        raw[6] = (raw[6] & 0x0f) | 0x40;
        raw[8] = (raw[8] & 0x3f) | 0x80;
    }
    let mut hex = String::with_capacity(raw.len() * 2);
    for byte in raw {
        const HEX: &[u8; 16] = b"0123456789abcdef";
        hex.push(HEX[(byte >> 4) as usize] as char);
        hex.push(HEX[(byte & 0x0f) as usize] as char);
    }
    hex
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::{Arc, Condvar, Mutex};
    use std::time::Duration;

    use serde_json::{Map, Value, json};
    use speakerdesk_core::document::{Document, Job, JobKind, Segment};
    use speakerdesk_core::ids::JobId;
    use speakerdesk_core::{messages, security};
    use tokio::io::{AsyncReadExt, AsyncWriteExt};
    use tokio::net::TcpStream;

    use super::{AppState, ImportModels, router};

    const JOB_A: &str = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa";
    const JOB_B: &str = "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb";

    struct Running {
        base: String,
        client: reqwest::Client,
        state: AppState,
        shutdown: Option<tokio::sync::oneshot::Sender<()>>,
        server: tokio::task::JoinHandle<Result<(), std::io::Error>>,
    }

    impl Running {
        async fn spawn(state: AppState) -> Self {
            let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
            let port = listener.local_addr().unwrap().port();
            let app = router(state.clone());
            let (shutdown, signal) = tokio::sync::oneshot::channel();
            let server = tokio::spawn(async move {
                axum::serve(listener, app)
                    .with_graceful_shutdown(async move {
                        let _ = signal.await;
                    })
                    .await
            });
            let running = Self {
                base: format!("http://127.0.0.1:{port}"),
                client: reqwest::Client::builder()
                    .pool_max_idle_per_host(0)
                    .build()
                    .unwrap(),
                state,
                shutdown: Some(shutdown),
                server,
            };
            for _ in 0..50 {
                if running
                    .client
                    .get(running.url("/api/config"))
                    .send()
                    .await
                    .is_ok()
                {
                    break;
                }
                tokio::time::sleep(Duration::from_millis(10)).await;
            }
            running
        }

        fn url(&self, path: &str) -> String {
            format!("{}{path}", self.base)
        }

        fn authed(&self, builder: reqwest::RequestBuilder) -> reqwest::RequestBuilder {
            builder.header(crate::http::token_header(), self.state.token())
        }

        async fn shutdown(mut self) {
            let directory = self.state.data_dir().to_path_buf();
            if let Some(shutdown) = self.shutdown.take() {
                let _ = shutdown.send(());
            }
            let _ = tokio::time::timeout(Duration::from_secs(3), self.server).await;
            let _ = std::fs::remove_dir_all(directory);
        }
    }

    fn segment(text: &str, machine_revision: i64) -> Segment {
        Segment {
            id: "one".into(),
            start: 0.0,
            end: 1.0,
            speaker: "speaker_0".into(),
            text: text.into(),
            language: None,
            language_detection: None,
            transcription_review: None,
            reading_turns: None,
            reading_turn_provenance: None,
            alignment: None,
            speaker_activity: None,
            acoustic_evidence: None,
            refinement_state: None,
            machine_revision: Some(machine_revision),
            protected_fields: None,
            review_resolution: None,
            retry_candidate: None,
            canonical_utterance_id: None,
            audio_revision: None,
            review: None,
            capture_source: None,
            rest: Map::new(),
        }
    }

    fn document(text: &str, machine_revision: i64) -> Document {
        Document {
            schema_version: 1,
            speakers: BTreeMap::from([("speaker_0".into(), "Speaker 1".into())]),
            segments: vec![segment(text, machine_revision)],
            provenance: None,
            warnings: Vec::new(),
            diarization: Vec::new(),
            edited_at: None,
            rest: Map::new(),
        }
    }

    fn job(id: &str, status: &str, created: f64) -> Job {
        Job {
            id: JobId::parse(id).unwrap(),
            name: "Standup.wav".into(),
            kind: JobKind::Import,
            status: status.into(),
            message: "Saved.".into(),
            language: "en".into(),
            source_file: Some("source.wav".into()),
            duration: Some(10.0),
            revision: 7,
            created,
            updated: created,
            document: Some(document("Keep these words.", 2)),
            extra: Map::from_iter([("fast_history".into(), json!({"old": "secret"}))]),
        }
    }

    fn tiny_wav() -> Vec<u8> {
        let samples: [i16; 16] = [
            0, 1000, 0, -1000, 0, 1000, 0, -1000, 0, 1000, 0, -1000, 0, 1000, 0, -1000,
        ];
        let data_len = (samples.len() * 2) as u32;
        let mut bytes = Vec::new();
        bytes.extend_from_slice(b"RIFF");
        bytes.extend_from_slice(&(36 + data_len).to_le_bytes());
        bytes.extend_from_slice(b"WAVEfmt ");
        bytes.extend_from_slice(&16u32.to_le_bytes());
        bytes.extend_from_slice(&1u16.to_le_bytes());
        bytes.extend_from_slice(&1u16.to_le_bytes());
        bytes.extend_from_slice(&16_000u32.to_le_bytes());
        bytes.extend_from_slice(&32_000u32.to_le_bytes());
        bytes.extend_from_slice(&2u16.to_le_bytes());
        bytes.extend_from_slice(&16u16.to_le_bytes());
        bytes.extend_from_slice(b"data");
        bytes.extend_from_slice(&data_len.to_le_bytes());
        for sample in samples {
            bytes.extend_from_slice(&sample.to_le_bytes());
        }
        bytes
    }

    fn multipart(language: &str, files: &[(&str, &[u8])]) -> (String, Vec<u8>) {
        let boundary = "----speakerdesk-boundary";
        let mut body = Vec::new();
        body.extend_from_slice(
            format!(
                "--{boundary}\r\nContent-Disposition: form-data; name=\"language\"\r\n\r\n{language}\r\n"
            )
            .as_bytes(),
        );
        for (name, bytes) in files {
            body.extend_from_slice(
                format!(
                    "--{boundary}\r\nContent-Disposition: form-data; name=\"files\"; filename=\"{name}\"\r\nContent-Type: application/octet-stream\r\n\r\n"
                )
                .as_bytes(),
            );
            body.extend_from_slice(bytes);
            body.extend_from_slice(b"\r\n");
        }
        body.extend_from_slice(format!("--{boundary}--\r\n").as_bytes());
        (format!("multipart/form-data; boundary={boundary}"), body)
    }

    struct GateModels {
        entered: Arc<AtomicUsize>,
        release: Arc<(Mutex<bool>, Condvar)>,
        document: Document,
    }

    impl ImportModels for GateModels {
        fn infer(&self, _job: &Job) -> Result<Document, String> {
            self.entered.fetch_add(1, Ordering::SeqCst);
            let (lock, cond) = &*self.release;
            let mut ready = lock.lock().expect("release");
            while !*ready {
                ready = cond.wait(ready).expect("release");
            }
            Ok(self.document.clone())
        }
    }

    async fn wait_until(flag: &AtomicUsize, expected: usize) {
        for _ in 0..100 {
            if flag.load(Ordering::SeqCst) >= expected {
                return;
            }
            tokio::time::sleep(Duration::from_millis(10)).await;
        }
        panic!(
            "timed out waiting for {expected} import runs, saw {}",
            flag.load(Ordering::SeqCst)
        );
    }

    #[tokio::test]
    async fn http_router_serves_config_with_security_headers() {
        let app = crate::http::router();
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let port = listener.local_addr().unwrap().port();
        let (shutdown, signal) = tokio::sync::oneshot::channel();
        let server = tokio::spawn(async move {
            axum::serve(listener, app)
                .with_graceful_shutdown(async move {
                    let _ = signal.await;
                })
                .await
        });
        let response = reqwest::Client::new()
            .get(format!("http://127.0.0.1:{port}/api/config"))
            .send()
            .await
            .unwrap();
        assert_eq!(response.status(), 200);
        assert_eq!(
            response
                .headers()
                .get("content-security-policy")
                .unwrap()
                .to_str()
                .unwrap(),
            security::CONTENT_SECURITY_POLICY
        );
        assert_eq!(
            response
                .headers()
                .get("cache-control")
                .unwrap()
                .to_str()
                .unwrap(),
            security::CACHE_CONTROL
        );
        let body: Value = response.json().await.unwrap();
        assert_eq!(body["languages"]["en"], "English");
        assert_eq!(body["languages"]["auto"], "Automatic");
        assert_eq!(body["readiness"]["note"], super::CONFIG_NOTE);
        assert_eq!(body["readiness"]["configured"], false);
        let _ = shutdown.send(());
        let _ = tokio::time::timeout(Duration::from_secs(3), server).await;
    }

    #[tokio::test]
    async fn mutating_routes_require_the_session_token() {
        let running = Running::spawn(AppState::new()).await;
        let (content_type, body) = multipart("en", &[("clip.wav", tiny_wav().as_slice())]);
        let missing = running
            .client
            .post(running.url("/api/jobs"))
            .header("content-type", content_type)
            .body(body)
            .send()
            .await
            .unwrap();
        assert_eq!(missing.status(), 403);
        assert_eq!(
            missing.json::<Value>().await.unwrap()["error"],
            messages::TOKEN_MISMATCH
        );

        let wrong = running
            .client
            .delete(running.url(&format!("/api/jobs/{JOB_A}")))
            .header(crate::http::token_header(), "not-the-token")
            .send()
            .await
            .unwrap();
        assert_eq!(wrong.status(), 403);
        assert_eq!(
            wrong
                .headers()
                .get("x-content-type-options")
                .unwrap()
                .to_str()
                .unwrap(),
            security::X_CONTENT_TYPE_OPTIONS
        );

        let open = running
            .client
            .get(running.url("/api/jobs"))
            .send()
            .await
            .unwrap();
        assert_eq!(open.status(), 200);
        running.shutdown().await;
    }

    #[tokio::test]
    async fn unknown_host_is_rejected_and_loopback_is_allowed() {
        let running = Running::spawn(AppState::new()).await;
        let address = running.base.trim_start_matches("http://");
        let mut stream = TcpStream::connect(address).await.unwrap();
        stream
            .write_all(
                b"GET /api/config HTTP/1.1\r\nHost: example.com\r\nConnection: close\r\n\r\n",
            )
            .await
            .unwrap();
        let mut buffer = Vec::new();
        stream.read_to_end(&mut buffer).await.unwrap();
        let text = String::from_utf8_lossy(&buffer);
        assert!(text.starts_with("HTTP/1.1 400"));
        assert!(text.contains(messages::INVALID_REQUEST));
        assert!(
            text.to_ascii_lowercase()
                .contains("content-security-policy")
        );
        running.shutdown().await;
    }

    #[tokio::test]
    async fn job_ids_that_are_not_32_lowercase_hex_are_not_found() {
        let running = Running::spawn(AppState::new()).await;
        for path in [
            "/api/jobs/abc",
            "/api/jobs/AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA",
            &format!("/api/jobs/{}/export/txt", "A".repeat(32)),
        ] {
            let response = running.client.get(running.url(path)).send().await.unwrap();
            assert_eq!(response.status(), 404, "{path}");
            assert!(
                response.json::<Value>().await.unwrap()["error"]
                    .as_str()
                    .unwrap()
                    .contains("not found")
            );
        }
        let missing = running
            .client
            .get(running.url(&format!("/api/jobs/{JOB_A}")))
            .send()
            .await
            .unwrap();
        assert_eq!(missing.status(), 404);
        running.shutdown().await;
    }

    #[tokio::test]
    async fn lists_jobs_and_hides_private_fields_on_detail() {
        let state = AppState::new();
        state.insert_job(job(JOB_A, "ready", 1.0));
        state.insert_job(job(JOB_B, "failed", 2.0));
        let running = Running::spawn(state).await;
        let listed = running
            .client
            .get(running.url("/api/jobs"))
            .send()
            .await
            .unwrap();
        let listed: Vec<Value> = listed.json().await.unwrap();
        assert_eq!(listed[0]["id"], JOB_B);
        assert_eq!(listed[1]["id"], JOB_A);
        assert_eq!(listed[1]["status"], "ready");
        assert!(listed[0].get("fast_history").is_none());

        let detail = running
            .client
            .get(running.url(&format!("/api/jobs/{JOB_A}")))
            .send()
            .await
            .unwrap();
        let detail: Value = detail.json().await.unwrap();
        assert!(detail.get("fast_history").is_none());
        assert_eq!(
            detail["document"]["segments"][0]["text"],
            "Keep these words."
        );
        assert_eq!(detail["inference_owned"], false);
        assert!(detail.get("unchanged").is_none());

        let same = running
            .client
            .get(running.url(&format!("/api/jobs/{JOB_A}?known_revision=7")))
            .send()
            .await
            .unwrap();
        let same: Value = same.json().await.unwrap();
        assert_eq!(same["unchanged"], true);
        assert!(same.get("document").is_none());

        let moved = running
            .client
            .get(running.url(&format!("/api/jobs/{JOB_A}?known_revision=6")))
            .send()
            .await
            .unwrap();
        let moved: Value = moved.json().await.unwrap();
        assert_eq!(moved["unchanged"], false);
        assert!(moved.get("document").is_some());

        let bad = running
            .client
            .get(running.url(&format!("/api/jobs/{JOB_A}?known_revision=nope")))
            .send()
            .await
            .unwrap();
        assert_eq!(bad.status(), 400);
        assert_eq!(
            bad.json::<Value>().await.unwrap()["error"],
            messages::KNOWN_REVISION
        );
        running.shutdown().await;
    }

    #[tokio::test]
    async fn upload_prepares_audio_without_a_model() {
        let state = AppState::new();
        let running = Running::spawn(state).await;
        let wav = tiny_wav();
        let (content_type, body) = multipart("en", &[("My Clip.wav", wav.as_slice())]);
        let created = running
            .authed(running.client.post(running.url("/api/jobs")))
            .header("content-type", content_type)
            .body(body)
            .send()
            .await
            .unwrap();
        assert_eq!(created.status(), 201);
        let created: Vec<Value> = created.json().await.unwrap();
        assert_eq!(created[0]["status"], "preparing");
        assert_eq!(created[0]["message"], super::PREPARING_AUDIO);
        assert_eq!(created[0]["name"], "My_Clip.wav");
        let id = created[0]["id"].as_str().unwrap().to_string();
        running.state.drain();
        let ready = running
            .client
            .get(running.url(&format!("/api/jobs/{id}")))
            .send()
            .await
            .unwrap();
        let ready: Value = ready.json().await.unwrap();
        assert_eq!(ready["status"], "uploaded");
        assert!(ready["duration"].as_f64().unwrap() > 0.0);
        assert_eq!(ready["message"], super::READY_FOR_INFERENCE);

        let bad_type = running
            .authed(running.client.post(running.url("/api/jobs")))
            .header("content-type", multipart("en", &[("notes.txt", b"nope")]).0)
            .body(multipart("en", &[("notes.txt", b"nope")]).1)
            .send()
            .await
            .unwrap();
        assert_eq!(bad_type.status(), 400);
        assert_eq!(
            bad_type.json::<Value>().await.unwrap()["error"],
            messages::SUPPORTED_FORMATS
        );
        running.shutdown().await;
    }

    #[tokio::test]
    async fn stale_transcript_and_passage_writes_conflict() {
        let state = AppState::new();
        state.insert_job(job(JOB_A, "ready", 1.0));
        let running = Running::spawn(state).await;
        let stale = running
            .authed(
                running
                    .client
                    .put(running.url(&format!("/api/jobs/{JOB_A}/transcript"))),
            )
            .json(&json!({
                "revision": 6,
                "document": {
                    "speakers": {"speaker_0": "Speaker 1"},
                    "segments": [{
                        "id": "one", "start": 0.0, "end": 1.0,
                        "speaker": "speaker_0", "text": "Too late."
                    }]
                }
            }))
            .send()
            .await
            .unwrap();
        assert_eq!(stale.status(), 409);
        assert_eq!(
            stale.json::<Value>().await.unwrap()["error"],
            messages::TRANSCRIPT_STALE
        );

        let saved = running
            .authed(
                running
                    .client
                    .put(running.url(&format!("/api/jobs/{JOB_A}/transcript"))),
            )
            .json(&json!({
                "revision": 7,
                "document": {
                    "speakers": {"speaker_0": "Speaker 1"},
                    "segments": [{
                        "id": "one", "start": 0.0, "end": 1.0,
                        "speaker": "speaker_0", "text": "Edited words."
                    }]
                }
            }))
            .send()
            .await
            .unwrap();
        assert_eq!(saved.status(), 200);
        let saved: Value = saved.json().await.unwrap();
        assert_eq!(saved["revision"], 8);
        assert_eq!(saved["document"]["segments"][0]["text"], "Edited words.");
        assert_eq!(saved["message"], "Saved locally.");

        let stale_passage = running
            .authed(
                running
                    .client
                    .patch(running.url(&format!("/api/jobs/{JOB_A}/segments/one"))),
            )
            .json(&json!({"segment_revision": 2, "changes": {"text": "Stale."}}))
            .send()
            .await
            .unwrap();
        assert_eq!(stale_passage.status(), 409);
        assert_eq!(
            stale_passage.json::<Value>().await.unwrap()["error"],
            messages::PASSAGE_STALE
        );

        let current = running
            .client
            .get(running.url(&format!("/api/jobs/{JOB_A}")))
            .send()
            .await
            .unwrap()
            .json::<Value>()
            .await
            .unwrap();
        let machine = current["document"]["segments"][0]["machine_revision"]
            .as_i64()
            .unwrap();
        let edited = running
            .authed(
                running
                    .client
                    .patch(running.url(&format!("/api/jobs/{JOB_A}/segments/one"))),
            )
            .json(&json!({"segment_revision": machine, "changes": {"text": "My correction."}}))
            .send()
            .await
            .unwrap();
        assert_eq!(edited.status(), 200);
        let edited: Value = edited.json().await.unwrap();
        assert_eq!(edited["segment"]["text"], "My correction.");
        assert_eq!(edited["revision"], 9);
        running.shutdown().await;
    }

    #[tokio::test]
    async fn export_and_delete_follow_the_job_state() {
        let state = AppState::new();
        state.insert_job(job(JOB_A, "ready", 1.0));
        let mut busy = job(JOB_B, "processing", 2.0);
        busy.document = None;
        state.insert_job(busy);
        let running = Running::spawn(state).await;

        let missing = running
            .client
            .get(running.url(&format!("/api/jobs/{JOB_B}/export/txt")))
            .send()
            .await
            .unwrap();
        assert_eq!(missing.status(), 409);
        assert_eq!(
            missing.json::<Value>().await.unwrap()["error"],
            messages::EXPORT_NEEDS_TRANSCRIPT
        );
        let unknown = running
            .client
            .get(running.url(&format!("/api/jobs/{JOB_A}/export/pdf")))
            .send()
            .await
            .unwrap();
        assert_eq!(unknown.status(), 404);

        let exported = running
            .client
            .get(running.url(&format!("/api/jobs/{JOB_A}/export/txt")))
            .send()
            .await
            .unwrap();
        assert_eq!(exported.status(), 200);
        let text = exported.text().await.unwrap();
        assert!(text.contains("Keep these words."));
        assert!(text.contains("Speaker 1"));

        let blocked = running
            .authed(
                running
                    .client
                    .delete(running.url(&format!("/api/jobs/{JOB_B}"))),
            )
            .send()
            .await
            .unwrap();
        assert_eq!(blocked.status(), 409);
        assert_eq!(
            blocked.json::<Value>().await.unwrap()["error"],
            messages::DELETE_BUSY
        );

        running
            .state
            .set_meeting(Some(JobId::parse(JOB_A).unwrap()));
        let owned = running
            .authed(
                running
                    .client
                    .delete(running.url(&format!("/api/jobs/{JOB_A}"))),
            )
            .send()
            .await
            .unwrap();
        assert_eq!(owned.status(), 409);
        running.state.set_meeting(None);
        let deleted = running
            .authed(
                running
                    .client
                    .delete(running.url(&format!("/api/jobs/{JOB_A}"))),
            )
            .send()
            .await
            .unwrap();
        assert_eq!(deleted.status(), 200);
        assert_eq!(deleted.json::<Value>().await.unwrap()["deleted"], true);
        let gone = running
            .client
            .get(running.url(&format!("/api/jobs/{JOB_A}")))
            .send()
            .await
            .unwrap();
        assert_eq!(gone.status(), 404);
        running.shutdown().await;
    }

    #[tokio::test]
    async fn run_queues_one_import_and_refuses_a_busy_or_unready_job() {
        let state = AppState::new();
        let mut first = job(JOB_A, "uploaded", 1.0);
        first.document = None;
        first.revision = 0;
        first.duration = Some(2.0);
        let mut second = job(JOB_B, "uploaded", 2.0);
        second.document = None;
        second.revision = 0;
        second.duration = Some(2.0);
        state.insert_job(first);
        state.insert_job(second);
        let running = Running::spawn(state).await;

        let unready = running
            .authed(
                running
                    .client
                    .post(running.url(&format!("/api/jobs/{JOB_A}/run"))),
            )
            .send()
            .await
            .unwrap();
        assert_eq!(unready.status(), 409);
        let unready: Value = unready.json().await.unwrap();
        assert!(
            unready["error"]
                .as_str()
                .unwrap()
                .contains(messages::MISSING_DIARIZATION)
        );

        let release = Arc::new((Mutex::new(false), Condvar::new()));
        let entered = Arc::new(AtomicUsize::new(0));
        running.state.set_issues(Vec::new());
        running.state.set_models(Arc::new(GateModels {
            entered: Arc::clone(&entered),
            release: Arc::clone(&release),
            document: document("Hello from the fake model.", 0),
        }));
        let queued = running
            .authed(
                running
                    .client
                    .post(running.url(&format!("/api/jobs/{JOB_A}/run"))),
            )
            .send()
            .await
            .unwrap();
        assert_eq!(queued.status(), 202);
        assert_eq!(queued.json::<Value>().await.unwrap()["status"], "queued");
        let also = running
            .authed(
                running
                    .client
                    .post(running.url(&format!("/api/jobs/{JOB_B}/run"))),
            )
            .send()
            .await
            .unwrap();
        assert_eq!(also.status(), 202);
        wait_until(&entered, 1).await;
        tokio::time::sleep(Duration::from_millis(40)).await;
        assert_eq!(entered.load(Ordering::SeqCst), 1);
        let while_busy = running
            .client
            .get(running.url(&format!("/api/jobs/{JOB_B}")))
            .send()
            .await
            .unwrap()
            .json::<Value>()
            .await
            .unwrap();
        assert_eq!(while_busy["status"], "queued");

        {
            let mut ready = release.0.lock().unwrap();
            *ready = true;
            release.1.notify_all();
        }
        running.state.drain();
        let done = running
            .client
            .get(running.url(&format!("/api/jobs/{JOB_A}")))
            .send()
            .await
            .unwrap()
            .json::<Value>()
            .await
            .unwrap();
        assert_eq!(done["status"], "ready");
        assert_eq!(done["message"], messages::TRANSCRIPT_READY);
        assert_eq!(
            done["document"]["segments"][0]["text"],
            "Hello from the fake model."
        );
        let second_done = running
            .client
            .get(running.url(&format!("/api/jobs/{JOB_B}")))
            .send()
            .await
            .unwrap()
            .json::<Value>()
            .await
            .unwrap();
        assert_eq!(second_done["status"], "ready");

        let again = running
            .authed(
                running
                    .client
                    .post(running.url(&format!("/api/jobs/{JOB_A}/run"))),
            )
            .send()
            .await
            .unwrap();
        assert_eq!(again.status(), 409);
        assert_eq!(
            again.json::<Value>().await.unwrap()["error"],
            messages::RERUN_WOULD_OVERWRITE
        );

        running
            .state
            .set_meeting(Some(JobId::parse(JOB_B).unwrap()));
        let mut third = job(&"c".repeat(32), "uploaded", 3.0);
        third.document = None;
        third.duration = Some(2.0);
        running.state.insert_job(third);
        let meeting = running
            .authed(
                running
                    .client
                    .post(running.url(&format!("/api/jobs/{}/run", "c".repeat(32)))),
            )
            .send()
            .await
            .unwrap();
        assert_eq!(meeting.status(), 409);
        assert_eq!(
            meeting.json::<Value>().await.unwrap()["error"],
            messages::FINISH_MEETING_BEFORE_IMPORT
        );
        running.shutdown().await;
    }

    #[tokio::test]
    async fn a_full_queue_rejects_another_upload() {
        let state = AppState::new();
        for index in 0..16 {
            let id = format!("{index:032x}");
            let mut queued = job(&id, "preparing", index as f64);
            queued.document = None;
            state.insert_job(queued);
        }
        let running = Running::spawn(state).await;
        let (content_type, body) = multipart("en", &[("clip.wav", tiny_wav().as_slice())]);
        let response = running
            .authed(running.client.post(running.url("/api/jobs")))
            .header("content-type", content_type)
            .body(body)
            .send()
            .await
            .unwrap();
        assert_eq!(response.status(), 429);
        assert_eq!(
            response.json::<Value>().await.unwrap()["error"],
            messages::QUEUE_FULL
        );
        running.shutdown().await;
    }
}
