//! HTTP routes for import jobs.
//!
//! Handlers speak the v0.6.6 job API. Mutating requests compare
//! `X-Speakerdesk-Token` with [`speakerdesk_core::security::tokens_match`].
//! Transcript writes go through [`speakerdesk_core::edit`]. Import runs are
//! queued on [`crate::worker::JobWorker`]. Tests use temporary SQLite stores and
//! [`ApplicationModels`] so nothing here downloads weights or spawns MLX.

use std::collections::HashMap;
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
use speakerdesk_core::document::{Document, Job, JobKind, Segment};
use speakerdesk_core::edit::{self, PassageEditInput, SaveInput};
use speakerdesk_core::error::{ApiError, StoreError};
use speakerdesk_core::ids::JobId;
use speakerdesk_core::security::{self, MAX_UPLOAD_BYTES};
use speakerdesk_core::store::{JobStore, SqliteJobStore};
use speakerdesk_core::{export, languages, limits, messages, transcript};
use tokio_util::sync::CancellationToken;

use crate::admission::{Activity, Admission as UpdateAdmission};
use crate::http::{security_headers, token_header};
use crate::worker::JobWorker;

mod import_progress;
pub(crate) mod refinement;
mod retry;
pub(crate) mod separation;
pub(crate) mod separation_identity;

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
    "legacy_media_root",
    "operation_generation",
    "refinement_generation",
    "rolling_sources",
    "refinement_history",
    "separation_history",
    "separation_attempts",
    "fast_history",
    "fast_previous_revision",
    "boundary_candidate",
    "boundary_candidates",
    "rolling_inflight",
    "fast_retained_candidate",
    "import_checkpoint",
];
const EXPORT_KINDS: &[&str] = &["txt", "srt", "vtt", "json"];
const AUDIO_EXTENSIONS: &[&str] = &[
    ".wav", ".mp3", ".m4a", ".flac", ".ogg", ".aiff", ".aif", ".mp4", ".aac", ".webm",
];

/// Import inference behind `POST /api/jobs/:jid/run`.
///
/// The production implementation uses the resident inference supervisor.
pub trait ApplicationModels: Send + Sync {
    /// A real resident child connection for the serial live worker.
    fn live_models(
        &self,
        _cancellation: &CancellationToken,
    ) -> Option<crate::inference::ChildModels> {
        None
    }
    fn live_alignment(&self) -> Option<Arc<speakerdesk_core::alignment::Vocabulary>> {
        None
    }
    fn infer(&self, job: &Job, cancellation: &CancellationToken) -> Result<Document, String>;
    fn import_identity(&self) -> Option<String> {
        None
    }
    fn infer_progress(
        &self,
        job: &Job,
        cancellation: &CancellationToken,
        _resume: Option<&speakerdesk_core::import::ImportResume>,
        _publish: &mut dyn FnMut(
            speakerdesk_core::import::ImportProgress<'_>,
        )
            -> Result<Vec<Segment>, speakerdesk_core::import::ImportError>,
    ) -> Result<Document, String> {
        self.infer(job, cancellation)
    }
    fn retry_passage(
        &self,
        _pcm: &[f32],
        _language: &str,
        _processed_microphone_catalog: Option<&str>,
        _cancellation: &CancellationToken,
    ) -> Result<speakerdesk_core::import::PassageCandidate, String> {
        Err("Passage retry is unavailable in this build.".into())
    }
    fn separate_passage(
        &self,
        _pcm: &[f32],
        _language: &str,
        _cancellation: &CancellationToken,
    ) -> Result<
        (
            speakerdesk_core::ipc::SeparatedAudio,
            [speakerdesk_core::separation::CandidateVoice; 2],
        ),
        String,
    > {
        Err("Separation model setup is required.".into())
    }
    fn separation_available(&self) -> bool {
        false
    }
    fn separate_passage_with_evidence(
        &self,
        pcm: &[f32],
        language: &str,
        cancellation: &CancellationToken,
    ) -> Result<
        (
            speakerdesk_core::ipc::SeparatedAudio,
            [speakerdesk_core::separation::CandidateVoice; 2],
            [Vec<(usize, usize)>; 2],
        ),
        String,
    > {
        self.separate_passage(pcm, language, cancellation)
            .map(|(audio, voices)| (audio, voices, std::array::from_fn(|_| Vec::new())))
    }
    /// Capture yields to live audio between separation and each bounded decode.
    fn separate_audio(
        &self,
        pcm: &[f32],
        cancellation: &CancellationToken,
    ) -> Result<speakerdesk_core::ipc::SeparatedAudio, speakerdesk_core::error::ChildError> {
        let models = self
            .live_models(cancellation)
            .ok_or(speakerdesk_core::error::ChildError::Unimplemented)?;
        let audio = speakerdesk_core::models::Models::separate(&models, pcm)?;
        speakerdesk_core::separation::prepare_audio(audio, pcm.len())
    }
    fn decode_separated_voice(
        &self,
        pcm: &[f32],
        language: &str,
        cancellation: &CancellationToken,
    ) -> Result<speakerdesk_core::separation::CandidateVoice, speakerdesk_core::error::ChildError>
    {
        let models = self
            .live_models(cancellation)
            .ok_or(speakerdesk_core::error::ChildError::Unimplemented)?;
        speakerdesk_core::separation::decode_voice(&models, pcm, language)
    }
    fn voice_policy(&self) -> Option<speakerdesk_core::voice::Calibration> {
        None
    }
    fn separated_voice_policy(&self) -> Option<speakerdesk_core::voice::separation::Policy> {
        None
    }
    fn embed_voice(
        &self,
        _pcm: &[f32],
        _cancellation: &CancellationToken,
    ) -> Result<speakerdesk_core::ipc::VoiceEmbedding, String> {
        Err("Voice recognition is not available in this build.".into())
    }
    fn embed_live_voice(
        &self,
        pcm: &[f32],
        cancellation: &CancellationToken,
    ) -> Result<speakerdesk_core::ipc::VoiceEmbedding, speakerdesk_core::error::ChildError> {
        if let Some(models) = self.live_models(cancellation) {
            speakerdesk_core::models::Models::embed_voice(&models, pcm)
        } else {
            self.embed_voice(pcm, cancellation)
                .map_err(|e| speakerdesk_core::error::ChildError::failed(e, false))
        }
    }
}

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

/// Infer returns the setup message and does not touch a model runtime.
#[derive(Debug, Default)]
pub struct UnavailableModels;

impl ApplicationModels for UnavailableModels {
    fn infer(
        &self,
        _job: &Job,
        _cancellation: &tokio_util::sync::CancellationToken,
    ) -> Result<Document, String> {
        Err(messages::SETUP_BEFORE_TRANSCRIPTION.to_string())
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

#[derive(Clone)]
struct RunningOperation {
    generation: u64,
    cancellation: CancellationToken,
    activity: Activity,
}

struct Shared {
    store: Mutex<SqliteJobStore>,
    _owner: fs::File,
    pub(crate) meeting_manager: Arc<Mutex<crate::meeting::MeetingManager>>,
    models: Mutex<Arc<dyn ApplicationModels>>,
    operations: Mutex<HashMap<JobId, RunningOperation>>,
    prepare: Mutex<Arc<dyn AudioPrep>>,
    issues: Mutex<Vec<String>>,
    reports: Mutex<Vec<speakerdesk_core::ipc::SetupModelReport>>,
    data_dir: PathBuf,
}

/// Shared route state. Clone is cheap; the worker and directory are shared.
#[derive(Clone)]
pub struct AppState {
    pub(crate) native: Arc<std::sync::OnceLock<Arc<crate::native::NativeModels>>>,
    token: Arc<str>,
    shared: Arc<Shared>,
    worker: Arc<JobWorker>,
    pub(crate) admission: UpdateAdmission,
    pub(crate) updates: Arc<Mutex<crate::native_updates::RuntimeUpdates>>,
}

impl AppState {
    pub fn open(root: impl AsRef<Path>) -> Result<Self, StoreError> {
        Self::open_with_legacy(root, None)
    }

    pub fn open_with_legacy(
        root: impl AsRef<Path>,
        legacy_db: Option<&Path>,
    ) -> Result<Self, StoreError> {
        fs::create_dir_all(root.as_ref()).map_err(|e| StoreError::message(e.to_string()))?;
        let data_dir = fs::canonicalize(root).map_err(|e| StoreError::message(e.to_string()))?;
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            fs::set_permissions(&data_dir, fs::Permissions::from_mode(0o700))
                .map_err(|e| StoreError::message(e.to_string()))?;
        }
        let owner = fs::OpenOptions::new()
            .create(true)
            .truncate(false)
            .read(true)
            .write(true)
            .open(data_dir.join("native.lock"))
            .map_err(|e| StoreError::message(e.to_string()))?;
        owner
            .try_lock()
            .map_err(|e| StoreError::message(format!("Cannot own native storage: {e}")))?;
        let database = data_dir.join("speakerdesk.sqlite");
        let legacy_db = match legacy_db {
            Some(path) => Some(path.to_path_buf()),
            None if !database
                .try_exists()
                .map_err(|e| StoreError::message(e.to_string()))? =>
            {
                legacy_database(&data_dir)?
            }
            None => None,
        };
        if let Some(legacy_db) = legacy_db {
            speakerdesk_core::store::import_legacy(&legacy_db, &database)?;
        }
        let mut store = SqliteJobStore::open(database)?;
        store.recover(ACTIVE)?;
        recover_deletions(&data_dir, &mut store)?;
        separation::cleanup(&data_dir, &store)?;
        Ok(Self {
            native: Arc::new(std::sync::OnceLock::new()),
            token: Arc::from(random_hex(32)),
            shared: Arc::new(Shared {
                store: Mutex::new(store),
                _owner: owner,
                meeting_manager: Arc::new(Mutex::new(crate::meeting::MeetingManager::new())),
                models: Mutex::new(Arc::new(UnavailableModels)),
                operations: Mutex::new(HashMap::new()),
                prepare: Mutex::new(Arc::new(WavPrepare)),
                issues: Mutex::new(default_issues()),
                reports: Mutex::new(Vec::new()),
                data_dir,
            }),
            worker: Arc::new(JobWorker::new()),
            admission: UpdateAdmission::new(),
            updates: Arc::new(Mutex::new(crate::native_updates::RuntimeUpdates::default())),
        })
    }

    #[cfg(test)]
    pub(crate) fn new() -> Self {
        Self::open(std::env::temp_dir().join(format!("speakerdesk-{}", random_hex(8)))).unwrap()
    }

    pub(crate) fn store(&self) -> std::sync::MutexGuard<'_, SqliteJobStore> {
        self.shared.store.lock().expect("store")
    }

    pub(crate) fn meeting_manager(&self) -> Arc<Mutex<crate::meeting::MeetingManager>> {
        Arc::clone(&self.shared.meeting_manager)
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

    pub(crate) fn queue_separation(&self, id: JobId, generation: u64) {
        let shared = self.shared.clone();
        let admission = self.admission.clone();
        self.worker.queue_handle().enqueue(move || {
            separation::automatic(&shared, &admission, &id, generation);
        });
    }

    pub fn insert_job(&self, job: Job) {
        self.shared
            .store
            .lock()
            .expect("jobs")
            .put(&job, None)
            .expect("insert job");
    }

    pub fn set_issues(&self, issues: Vec<String>) {
        *self.shared.issues.lock().expect("issues") = issues;
    }

    pub fn set_models(&self, models: Arc<dyn ApplicationModels>) {
        *self.shared.models.lock().expect("models") = models;
    }

    pub(crate) fn models(&self) -> Arc<dyn ApplicationModels> {
        self.shared.models.lock().expect("models").clone()
    }

    pub(crate) fn begin_refinement_operation(
        &self,
        store: &mut SqliteJobStore,
        job: &mut Job,
    ) -> Result<OperationGuard, ApiError> {
        self.begin_operation(store, job, Activity::Refinement)
    }

    pub(crate) fn begin_recording_operation(
        &self,
        store: &mut SqliteJobStore,
        job: &mut Job,
    ) -> Result<OperationGuard, ApiError> {
        self.begin_operation(store, job, Activity::recording())
    }

    fn begin_operation(
        &self,
        store: &mut SqliteJobStore,
        job: &mut Job,
        activity: Activity,
    ) -> Result<OperationGuard, ApiError> {
        begin_operation(&self.shared, &self.admission, store, job, activity)
    }

    pub(crate) fn model_status(
        &self,
    ) -> (Vec<String>, Vec<speakerdesk_core::ipc::SetupModelReport>) {
        (
            self.shared.issues.lock().expect("issues").clone(),
            self.shared.reports.lock().expect("reports").clone(),
        )
    }
    pub(crate) fn set_model_reports(&self, reports: Vec<speakerdesk_core::ipc::SetupModelReport>) {
        *self.shared.reports.lock().expect("reports") = reports;
        self.set_issues(Vec::new());
    }

    pub(crate) fn automatic_language_ready(&self) -> bool {
        self.shared
            .reports
            .lock()
            .expect("reports")
            .iter()
            .any(|report| report.ok && report.model == languages::LID_CHECKPOINT)
    }

    pub(crate) fn alignment_ready(&self) -> bool {
        self.shared
            .reports
            .lock()
            .expect("reports")
            .iter()
            .any(|report| report.ok && report.model == speakerdesk_core::alignment::CHECKPOINT)
    }

    pub(crate) fn cancel_all(&self) -> Result<(), StoreError> {
        self.admission.close();
        let mut store = self.store();
        let operations = self.shared.operations.lock().expect("operations").clone();
        let tokens: Vec<_> = operations
            .values()
            .map(|op| op.cancellation.clone())
            .collect();
        let result = (|| {
            for (id, operation) in operations {
                let generation = operation.generation;
                let token = operation.cancellation;
                let active = store.get(&id, false)?.is_some_and(|job| {
                    (matches!(
                        operation.activity,
                        Activity::Refinement | Activity::Recording { .. }
                    ) || ["queued", "processing"].contains(&job.status.as_str()))
                        && job
                            .extra
                            .get("operation_generation")
                            .and_then(Value::as_u64)
                            == Some(generation)
                });
                if !active {
                    token.cancel();
                    continue;
                }
                store.update_job(&id, |job| {
                    if job
                        .extra
                        .get("operation_generation")
                        .and_then(Value::as_u64)
                        == Some(generation)
                    {
                        job.extra
                            .insert("operation_generation".into(), json!(generation + 1));
                        refinement::interrupt(job);
                        separation::interrupt(job);
                        if job
                            .extra
                            .remove("passage_retry")
                            .is_some_and(|value| !value.is_null())
                        {
                            job.message =
                                "Passage retry interrupted. Existing words are retained.".into();
                        }
                        if ["queued", "processing"].contains(&job.status.as_str()) {
                            job.status = "uploaded".into();
                            job.message = "Transcription interrupted. Saved audio is kept.".into();
                        }
                        if matches!(operation.activity, Activity::Recording { .. }) {
                            job.status = "failed".into();
                            job.message =
                                "Recording interrupted. Captured audio and edits are retained."
                                    .into();
                        }
                    }
                    Ok::<_, StoreError>(())
                })?;
                token.cancel();
            }
            Ok(())
        })();
        if result.is_err() {
            // A failed transaction cannot grant publication. Cancel under the same store lock
            // before any worker can reload the unchanged generation, then report the failure.
            for token in tokens {
                token.cancel();
            }
        }
        result
    }
}

fn begin_operation(
    shared: &Arc<Shared>,
    admission: &UpdateAdmission,
    store: &mut SqliteJobStore,
    job: &mut Job,
    activity: Activity,
) -> Result<OperationGuard, ApiError> {
    if job.extra.get("deletion_pending").and_then(Value::as_bool) == Some(true) {
        return Err(ApiError::new(
            409,
            "Retry Delete to finish removing this recording.",
        ));
    }
    if admission.closed() {
        return Err(ApiError::new(409, messages::UPDATE_RESERVED));
    }
    if store.count_status(ACTIVE)? > 0
        || shared
            .operations
            .lock()
            .expect("operations")
            .iter()
            .any(|(id, op)| {
                matches!(activity, Activity::Recording { .. })
                    || matches!(op.activity, Activity::Recording { .. })
                    || id == &job.id
            })
    {
        return Err(ApiError::new(
            409,
            "Finish the active recording or transcription before starting this operation.",
        ));
    }
    let generation = job
        .extra
        .get("operation_generation")
        .and_then(Value::as_u64)
        .unwrap_or(0)
        .checked_add(1)
        .ok_or_else(|| ApiError::new(409, "Operation generation exhausted"))?;
    job.extra
        .insert("operation_generation".into(), json!(generation));
    store.put(job, None)?;
    let cancellation = CancellationToken::new();
    shared.operations.lock().expect("operations").insert(
        job.id.clone(),
        RunningOperation {
            generation,
            cancellation: cancellation.clone(),
            activity,
        },
    );
    Ok(OperationGuard {
        shared: shared.clone(),
        id: job.id.clone(),
        generation,
        cancellation,
        parent: None,
    })
}

fn legacy_database(root: &Path) -> Result<Option<PathBuf>, StoreError> {
    let mut candidates = vec![
        root.join("recordings/jobs.sqlite"),
        root.join("jobs.sqlite"),
    ];
    // The packaged Python app used Speakerdesk before Tauri adopted its bundle identifier.
    if root.ends_with("Library/Application Support/com.speakerdesk.desktop") {
        candidates.push(
            root.with_file_name("Speakerdesk")
                .join("recordings/jobs.sqlite"),
        );
    }
    for path in candidates {
        match fs::symlink_metadata(&path) {
            Ok(_) => return Ok(Some(path)),
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => {}
            Err(error) => return Err(StoreError::message(error.to_string())),
        }
    }
    Ok(None)
}

/// Job routes, including token checks and the v0.6.6 security headers.
pub fn router(state: AppState) -> Router {
    Router::new()
        .merge(crate::people_routes::routes())
        .merge(crate::meeting::routes())
        .merge(crate::native::routes())
        .merge(crate::native_updates::routes())
        .route("/api/config", get(config))
        .route("/api/jobs", get(list_jobs).post(create_jobs))
        .route("/api/jobs/{jid}", get(job_detail).delete(delete_job))
        .route("/api/jobs/{jid}/transcript", put(save_transcript))
        .route("/api/jobs/{jid}/segments/{sid}", patch(edit_passage))
        .route("/api/jobs/{jid}/export/{kind}", get(export_job))
        .route("/api/jobs/{jid}/run", post(run_job))
        .route("/api/jobs/{jid}/retry", post(retry::retry_passage))
        .route("/api/jobs/{jid}/separation", post(separation::begin))
        .route(
            "/api/jobs/{jid}/separation/{episode}/accept",
            post(separation::accept),
        )
        .route(
            "/api/jobs/{jid}/separation/{episode}/discard",
            post(separation::discard),
        )
        .route(
            "/api/jobs/{jid}/separation/{episode}/restore",
            post(separation::restore),
        )
        .route(
            "/api/jobs/{jid}/separation/{episode}/history",
            get(separation::history),
        )
        .route(
            "/api/jobs/{jid}/separation/{episode}/original/audio",
            get(separation::original_audio),
        )
        .route(
            "/api/jobs/{jid}/separation/{episode}/{channel}/audio",
            get(separation::playback),
        )
        .route(
            "/api/jobs/{jid}/passages/{pid}/audio",
            get(separation::passage_audio),
        )
        .route("/api/jobs/{jid}/cancel", post(cancel_job))
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
    if let Some(origin) = request.headers().get(header::ORIGIN)
        && origin.to_str().ok() != Some(format!("http://{host}").as_str())
    {
        return with_security(error_response(
            StatusCode::FORBIDDEN,
            messages::INVALID_REQUEST,
        ));
    }
    let _activity = if is_mutating(request.method()) && request.uri().path() != "/api/updates" {
        // Long inference handlers take their own lease; ordinary edits may run during setup.
        let activity = if (request.method() == Method::POST && request.uri().path() == "/api/jobs")
            || request.uri().path().starts_with("/api/meetings")
        {
            Activity::Import
        } else {
            Activity::Mutation
        };
        match app.admission.lease(activity) {
            Ok(lease) => Some(lease),
            Err(error) => {
                return with_security(error_response(StatusCode::CONFLICT, error.message()));
            }
        }
    } else {
        None
    };
    let mut response = next.run(request).await;
    apply_security(&mut response);
    response
}

async fn config(State(app): State<AppState>) -> Response {
    blocking(move || {
        let issues = app.shared.issues.lock().expect("issues").clone();
        let default_language = match app.store().preference("default_language") {
            Ok(value) => value.unwrap_or_else(|| languages::AUTOMATIC.to_string()),
            Err(error) => return store_failure(&error),
        };
        Json(json!({
            "languages": language_choices(),
            "default_language": default_language,
            "theme": app.store().preference("theme").ok().flatten()
                .filter(|theme| matches!(theme.as_str(), "system" | "light" | "dark"))
                .unwrap_or_else(|| "system".into()),
            "readiness": readiness_json(&issues, app.automatic_language_ready(), app.alignment_ready()),
            "decoder": if cfg!(target_os = "macos") { "Symphonia and macOS AudioToolbox" } else { "Symphonia" },
        }))
        .into_response()
    })
    .await
}

async fn list_jobs(State(app): State<AppState>) -> Response {
    blocking(move || {
        let directory = app.shared.store.lock().expect("jobs");
        match directory.summaries() {
            Ok(summaries) => Json(summaries).into_response(),
            Err(error) => store_failure(&error),
        }
    })
    .await
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

async fn job_detail(
    State(app): State<AppState>,
    AxumPath(jid): AxumPath<String>,
    Query(query): Query<DetailQuery>,
) -> Response {
    blocking(move || {
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
        let directory = app.shared.store.lock().expect("jobs");
        let metadata = known.is_some();
        let job = match load_job(&directory, &id, metadata) {
            Ok(job) => job,
            Err(error) => return from_api(error),
        };
        let (job, unchanged) = if let Some(known) = known {
            let unchanged = known == i128::from(job.revision);
            if unchanged {
                (job, Some(true))
            } else {
                match load_job(&directory, &id, false) {
                    Ok(full) => (full, Some(false)),
                    Err(error) => return from_api(error),
                }
            }
        } else {
            (job, None)
        };
        let owned = app
            .shared
            .operations
            .lock()
            .expect("operations")
            .contains_key(&id);
        Json(detail_json(
            &job,
            known.is_some() && unchanged == Some(true),
            unchanged,
            owned,
        ))
        .into_response()
    })
    .await
}

async fn save_transcript(
    State(app): State<AppState>,
    AxumPath(jid): AxumPath<String>,
    body: Bytes,
) -> Response {
    blocking(move || {
        let id = match require_job_id(&jid) {
            Ok(id) => id,
            Err(error) => return from_api(error),
        };
        let body = match json_body(&body) {
            Ok(body) => body,
            Err(error) => return from_api(error),
        };
        let result = app.store().update_job(&id, |job| {
            if job.extra.get("import_resumable").and_then(Value::as_bool) == Some(true) {
                return Err(ApiError::new(
                    409,
                    "Use passage corrections while import transcription is unfinished.",
                ));
            }
            let assignments = map_field(&job.extra, "speaker_assignments");
            let checks = map_field(&job.extra, "voice_checks");
            let outcome = edit::save_transcript(SaveInput {
                status: &job.status,
                duration: job.duration,
                revision: job.revision,
                saved: job.document.as_ref(),
                body: &body,
                now: unix_now(),
                speaker_assignments: &assignments,
                voice_checks: &checks,
            })?;
            let accepted_repair = if job.extra.get("refinement_status").and_then(Value::as_str)
                == Some("unresolved")
            {
                job.document.as_ref().and_then(|prior| {
                    let rows: std::collections::HashMap<&str, &Segment> = prior
                        .segments
                        .iter()
                        .map(|row| (row.id.as_str(), row))
                        .collect();
                    outcome
                        .document
                        .segments
                        .iter()
                        .find(|row| {
                            rows.get(row.id.as_str())
                                .is_some_and(|old| accepted_retry(old, row))
                        })
                        .map(|row| {
                            (
                                row.id.clone(),
                                row.retry_candidate.as_ref().expect("accepted retry")["id"]
                                    .as_str()
                                    .expect("retry id")
                                    .to_owned(),
                                speakerdesk_core::live::segment_version(row),
                            )
                        })
                })
            } else {
                None
            };
            if outcome.language_context_changed {
                let generation = job
                    .extra
                    .get("language_context_generation")
                    .and_then(Value::as_u64)
                    .unwrap_or(0);
                job.extra.insert(
                    "language_context_generation".into(),
                    json!(generation.checked_add(1).ok_or_else(|| ApiError::new(
                        409,
                        "Language context generation exhausted."
                    ))?),
                );
            }
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
            Ok::<_, ApiError>(accepted_repair)
        });
        match result {
            Ok(Some((job, accepted_repair))) => {
                if let Some(repair) = accepted_repair {
                    let app = app.clone();
                    let id = job.id.to_string();
                    let generation = job
                        .extra
                        .get("operation_generation")
                        .and_then(Value::as_u64)
                        .unwrap_or(0);
                    let queue = app.worker.queue_handle();
                    queue.enqueue(move || {
                        refinement::resume_after_repair(&app, &id, repair, generation)
                    });
                }
                Json(stored_job_json(&job)).into_response()
            }
            Ok(None) => error_response(StatusCode::NOT_FOUND, NOT_FOUND),
            Err(error) => from_api(error),
        }
    })
    .await
}

/// Acceptance must come from the server's fresh retained candidate, never an
/// uploaded receipt or an unrelated edit that happens to use the same words.
fn accepted_retry(old: &Segment, row: &Segment) -> bool {
    let (Some(before), Some(after)) = (&old.retry_candidate, &row.retry_candidate) else {
        return false;
    };
    let Some(language) = before["language"]
        .as_str()
        .filter(|code| languages::is_supported(code))
    else {
        return false;
    };
    before["id"].is_string()
        && before["id"] == after["id"]
        && before["accepted_at"].is_null()
        && after["accepted_at"].is_number()
        && before["error"].is_null()
        && before["transcription_review"]["partial_text"] != true
        && before["original_text"] == old.text
        && before["text"]
            .as_str()
            .is_some_and(|text| !text.trim().is_empty() && text == row.text)
        && row.language.as_deref() == Some(language)
        && row.language_detection.as_ref().is_some_and(|decision| {
            decision["mode"] == "manual" && decision["reason"] == "passage_retry"
        })
        && before["start"] == old.start
        && before["end"] == old.end
        && before["speaker"] == old.speaker
        && row.start == old.start
        && row.end == old.end
        && row.speaker == old.speaker
        && row.capture_source == old.capture_source
        && row.audio_revision == old.audio_revision
        && row.rest.get("language_epoch") == old.rest.get("language_epoch")
        && row.rest.get("language_mode") == old.rest.get("language_mode")
        && row
            .protected_fields
            .as_ref()
            .is_some_and(|fields| fields.iter().any(|field| field == "text"))
}

async fn edit_passage(
    State(app): State<AppState>,
    AxumPath((jid, sid)): AxumPath<(String, String)>,
    body: Bytes,
) -> Response {
    blocking(move || {
        let id = match require_job_id(&jid) {
            Ok(id) => id,
            Err(error) => return from_api(error),
        };
        let body = match json_body(&body) {
            Ok(body) => body,
            Err(error) => return from_api(error),
        };
        let result = app.store().update_job(&id, |job| {
            let outcome = edit::edit_passage(PassageEditInput {
                status: if job.kind == JobKind::Import
                    && job.extra.get("import_resumable").and_then(Value::as_bool) == Some(true)
                {
                    "ready"
                } else {
                    &job.status
                },
                duration: job.duration.unwrap_or(0.0),
                revision: job.revision,
                document: job.document.as_ref(),
                segment_id: &sid,
                body: &body,
                now: unix_now(),
            })?;
            if outcome.language_context_changed {
                let generation = job
                    .extra
                    .get("language_context_generation")
                    .and_then(Value::as_u64)
                    .unwrap_or(0);
                job.extra.insert(
                    "language_context_generation".into(),
                    json!(generation.checked_add(1).ok_or_else(|| ApiError::new(
                        409,
                        "Language context generation exhausted."
                    ))?),
                );
            }
            job.document = Some(outcome.document);
            job.revision = outcome.revision;
            Ok::<_, ApiError>(outcome.segment)
        });
        match result {
            Ok(Some((job, segment))) => {
                Json(json!({"segment": segment, "revision": job.revision})).into_response()
            }
            Ok(None) => error_response(StatusCode::NOT_FOUND, NOT_FOUND),
            Err(error) => from_api(error),
        }
    })
    .await
}

async fn export_job(
    State(app): State<AppState>,
    AxumPath((jid, kind)): AxumPath<(String, String)>,
) -> Response {
    blocking(move || {
        if !EXPORT_KINDS.contains(&kind.as_str()) {
            return error_response(StatusCode::NOT_FOUND, NOT_FOUND);
        }
        let id = match require_job_id(&jid) {
            Ok(id) => id,
            Err(error) => return from_api(error),
        };
        let directory = app.shared.store.lock().expect("jobs");
        let job = match load_job(&directory, &id, false) {
            Ok(job) => job,
            Err(error) => return from_api(error),
        };
        drop(directory);
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
    })
    .await
}

async fn delete_job(State(app): State<AppState>, AxumPath(jid): AxumPath<String>) -> Response {
    blocking(move || {
        let id = match require_job_id(&jid) {
            Ok(id) => id,
            Err(error) => return from_api(error),
        };
        let mut directory = app.shared.store.lock().expect("jobs");
        let mut job = match load_job(&directory, &id, false) {
            Ok(job) => job,
            Err(error) if error.status == 404 => {
                return Json(json!({"deleted":true})).into_response();
            }
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
            return error_response(StatusCode::CONFLICT, messages::DELETE_BUSY);
        }
        job.status = "deleting".into();
        job.message = "Deleting recording…".into();
        job.extra.insert("deletion_pending".into(), json!(true));
        job.revision += 1;
        if let Err(error) = directory.put(&job, None) {
            return store_failure(&error);
        }
        drop(directory);
        let removed = remove_owned_recording(&app.shared.data_dir, &job);
        let mut directory = app.store();
        match removed {
            Ok(()) => match directory.delete(&id) {
                Ok(_) => Json(json!({"deleted":true})).into_response(),
                Err(error) => store_failure(&error),
            },
            Err(error) => {
                if let Ok(Some(mut current)) = directory.get(&id, false) {
                    current.status = "delete_failed".into();
                    current.message =
                        format!("Could not finish deleting recording: {error}. Retry Delete.");
                    current.revision += 1;
                    if let Err(error) = directory.put(&current, None) {
                        return store_failure(&error);
                    }
                }
                error_response(
                    StatusCode::INTERNAL_SERVER_ERROR,
                    "Could not remove all recording data. Retry Delete to finish cleanup.",
                )
            }
        }
    })
    .await
}

fn remove_owned_recording(root: &Path, job: &Job) -> std::io::Result<()> {
    crate::recovery::RecoveryStore::open(root)?.discard_job(job.id.as_str())?;
    let folder = media_write_directory(root, job);
    match fs::symlink_metadata(&folder) {
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(()),
        Err(error) => return Err(error),
        Ok(metadata) if metadata.is_symlink() || !metadata.is_dir() => {
            return Err(std::io::Error::other(
                "Recording media is not an owned directory",
            ));
        }
        Ok(_) => (),
    }
    match fs::remove_dir_all(folder) {
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => Ok(()),
        result => result,
    }
}

fn recover_deletions(root: &Path, store: &mut SqliteJobStore) -> Result<(), StoreError> {
    for summary in store
        .summaries()?
        .into_iter()
        .filter(|job| matches!(job.status.as_str(), "deleting" | "delete_failed"))
    {
        let Some(mut job) = store.get(&summary.id, false)? else {
            continue;
        };
        if job.extra.get("deletion_pending").and_then(Value::as_bool) != Some(true) {
            continue;
        }
        match remove_owned_recording(root, &job) {
            Ok(()) => {
                store.delete(&job.id)?;
            }
            Err(error) => {
                job.status = "delete_failed".into();
                job.message =
                    format!("Could not finish deleting recording: {error}. Retry Delete.");
                store.put(&job, None)?;
            }
        }
    }
    Ok(())
}

async fn run_job(State(app): State<AppState>, AxumPath(jid): AxumPath<String>) -> Response {
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

async fn cancel_job(State(app): State<AppState>, AxumPath(jid): AxumPath<String>) -> Response {
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

pub(crate) struct OperationGuard {
    shared: Arc<Shared>,
    id: JobId,
    generation: u64,
    pub(crate) cancellation: CancellationToken,
    // A child scope holds registration alive but never unregisters its owner.
    parent: Option<Arc<OperationGuard>>,
}

impl OperationGuard {
    pub(crate) fn child_scope(self: &Arc<Self>) -> Arc<Self> {
        Arc::new(Self {
            shared: self.shared.clone(),
            id: self.id.clone(),
            generation: self.generation,
            cancellation: self.cancellation.child_token(),
            parent: Some(self.clone()),
        })
    }
    pub(crate) fn generation(&self) -> u64 {
        self.generation
    }

    /// Capture's serial owner keeps its recording activity and operation generation.
    /// This does not admit another refinement worker or relax saved/manual routes.
    pub(crate) fn check_live_voice_step(
        &self,
        checks: &mut crate::people_routes::LiveVoiceChecks,
        models: &dyn ApplicationModels,
        job: &Job,
    ) -> Result<bool, speakerdesk_core::live::LiveError> {
        if !self.current(job)
            || !matches!(
                job.status.as_str(),
                "recording" | "paused" | "finishing" | "ready"
            )
        {
            checks.interrupt(&mut self.shared.store.lock().expect("store"));
            return Ok(false);
        }
        checks.step(&self.shared.store, &self.shared.data_dir, models, self, job)
    }

    /// Retain the same operation through optional checks of an already editable result.
    /// Caller runs on the existing blocking worker, after capture/inference has joined.
    pub(crate) fn check_saved_voices(&self, models: &dyn ApplicationModels) {
        let store = self.shared.store.lock().expect("store");
        let job = match store.get(&self.id, false) {
            Ok(Some(job)) if job.status == "ready" && self.current(&job) => job,
            Ok(_) => return,
            Err(error) => {
                tracing::warn!(%error,"could not check voices on saved transcript");
                return;
            }
        };
        let mut operations = self.shared.operations.lock().expect("operations");
        let Some(active) = operations
            .get_mut(&self.id)
            .filter(|active| active.generation == self.generation)
        else {
            return;
        };
        active.activity = Activity::Refinement;
        drop(operations);
        drop(store);
        crate::people_routes::check_saved_voices(
            &self.shared.store,
            &self.shared.data_dir,
            models,
            self,
            &job,
        );
    }

    pub(crate) fn current(&self, job: &Job) -> bool {
        !self.cancellation.is_cancelled()
            && job
                .extra
                .get("operation_generation")
                .and_then(Value::as_u64)
                == Some(self.generation)
    }
}

impl Drop for OperationGuard {
    fn drop(&mut self) {
        if self.parent.is_some() {
            return;
        }
        let mut operations = self.shared.operations.lock().expect("operations");
        if operations
            .get(&self.id)
            .is_some_and(|current| current.generation == self.generation)
        {
            operations.remove(&self.id);
        }
    }
}

fn execute_import(
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

fn load_job(directory: &SqliteJobStore, id: &JobId, metadata: bool) -> Result<Job, ApiError> {
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
    shared
        .operations
        .lock()
        .expect("operations")
        .iter()
        .find(|(_, op)| matches!(op.activity, Activity::Recording { .. }))
        .map(|(id, _)| id.clone())
}

fn remove_staged(root: &Path, ids: &[JobId]) {
    for id in ids {
        remove_job_dir(root, id);
    }
}

pub(crate) fn media_write_directory(root: &Path, job: &Job) -> PathBuf {
    if job.extra.contains_key("legacy_media_root") {
        root.join("derived").join(job.id.as_str())
    } else {
        root.join(job.id.as_str())
    }
}

pub(crate) fn confined_file(root: &Path, name: &str) -> std::io::Result<PathBuf> {
    let root = root.canonicalize()?;
    let path = root.join(name).canonicalize()?;
    if !path.starts_with(&root) || !path.is_file() {
        return Err(std::io::Error::new(
            std::io::ErrorKind::PermissionDenied,
            "Audio path escapes its recording directory",
        ));
    }
    Ok(path)
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

pub(crate) fn stored_job_json(job: &Job) -> Value {
    let mut view = job.clone();
    for key in PRIVATE_FIELDS {
        view.extra.remove(*key);
    }
    let mut value = serde_json::to_value(&view).expect("job serializes");
    if let Some(object) = value.as_object_mut() {
        object.entry("duration").or_insert(Value::Null);
        object.entry("document").or_insert(Value::Null);
        if refinement::available(job) {
            object.insert("saved_refinement_available".into(), Value::Bool(true));
        }
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

fn readiness_json(issues: &[String], automatic: bool, alignment: bool) -> Value {
    json!({
        "configured": issues.is_empty(),
        "issues": issues,
        "automatic_language": issues.is_empty() && automatic,
        "word_alignment": issues.is_empty() && alignment,
        "live_capture": false,
        "model": MODEL_NAME,
        "speaker_limit": MODEL_SPEAKER_LIMIT,
        "device": if issues.is_empty() { "metal" } else { "unavailable" },
        "note": CONFIG_NOTE,
    })
}

fn default_issues() -> Vec<String> {
    [
        messages::MISSING_DIARIZATION,
        messages::MISSING_ASR,
        messages::SPEECH_MODEL_MISSING,
    ]
    .into_iter()
    .map(str::to_string)
    .collect()
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

pub(crate) fn new_job_id() -> Result<JobId, StoreError> {
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

pub(crate) async fn blocking(work: impl FnOnce() -> Response + Send + 'static) -> Response {
    tokio::task::spawn_blocking(work)
        .await
        .unwrap_or_else(|error| {
            tracing::error!(%error, "host operation failed");
            error_response(
                StatusCode::INTERNAL_SERVER_ERROR,
                messages::LOCAL_OPERATION_FAILED,
            )
        })
}

#[cfg(test)]
mod tests {
    mod retry;
    use speakerdesk_core::store::JobStore;
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

    use super::{AppState, ApplicationModels, router};

    const JOB_A: &str = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa";
    const JOB_B: &str = "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb";

    #[tokio::test]
    async fn rejected_upload_admission_creates_no_stuck_jobs_or_owned_media() {
        let state = AppState::new();
        let setup = state.admission.lease(super::Activity::Setup).unwrap();
        let running = Running::spawn(state.clone()).await;
        let (content_type, body) = multipart("en", &[("clip.wav", tiny_wav().as_slice())]);
        let rejected = running
            .authed(running.client.post(running.url("/api/jobs")))
            .header("content-type", content_type.clone())
            .body(body.clone())
            .send()
            .await
            .unwrap();
        assert_eq!(rejected.status(), 409);
        assert!(state.store().summaries().unwrap().is_empty());
        assert!(
            !std::fs::read_dir(state.data_dir())
                .unwrap()
                .any(|entry| entry.unwrap().file_type().unwrap().is_dir())
        );
        drop(setup);
        let accepted = running
            .authed(running.client.post(running.url("/api/jobs")))
            .header("content-type", content_type)
            .body(body)
            .send()
            .await
            .unwrap();
        assert_eq!(accepted.status(), 201);
        state.drain();
        assert_eq!(state.store().summaries().unwrap()[0].status, "uploaded");
        running.shutdown().await;
    }

    #[test]
    fn cancelling_a_worker_scope_keeps_its_recording_owner_registered() {
        let app = AppState::new();
        let mut job = job(JOB_A, "ready", 0.);
        app.insert_job(job.clone());
        let owner = Arc::new(
            app.begin_recording_operation(&mut app.store(), &mut job)
                .unwrap(),
        );
        let child = owner.child_scope();
        child.cancellation.cancel();
        assert!(!owner.cancellation.is_cancelled());
        drop(child);
        assert!(app.shared.operations.lock().unwrap().contains_key(&job.id));
        let child = owner.child_scope();
        drop(owner);
        assert!(app.shared.operations.lock().unwrap().contains_key(&job.id));
        drop(child);
        assert!(!app.shared.operations.lock().unwrap().contains_key(&job.id));
    }

    #[tokio::test]
    async fn deletion_failure_retains_intent_and_restart_finishes_owned_media_cleanup() {
        let state = AppState::new();
        let id = JobId::parse(JOB_A).unwrap();
        state.insert_job(job(JOB_A, "ready", 1.));
        let recovery = crate::recovery::RecoveryStore::open(state.data_dir()).unwrap();
        let draft = json!({"schema_version":1,"job_id":JOB_A,"has_unsaved":true,
            "document":{"segments":[{"text":"Private correction"}]}});
        recovery.save(&"d".repeat(32), 1, &draft).unwrap();
        let media = state.data_dir().join(JOB_A);
        // A conflicting file is deterministic even under privileged test users.
        std::fs::write(&media, b"not a media directory").unwrap();
        let running = Running::spawn(state.clone()).await;
        let response = running
            .authed(
                running
                    .client
                    .delete(running.url(&format!("/api/jobs/{JOB_A}"))),
            )
            .send()
            .await
            .unwrap();
        assert_eq!(response.status(), 500);
        let retained = state.store().get(&id, false).unwrap().unwrap();
        assert_eq!(retained.status, "delete_failed");
        assert_eq!(retained.extra["deletion_pending"], true);
        assert!(media.is_file());
        assert!(recovery.list().unwrap().is_empty());
        assert!(recovery.save(&"e".repeat(32), 2, &draft).unwrap().is_none());
        assert!(
            state
                .store()
                .update_job(&id, |job| {
                    job.name = "wrong".into();
                    Ok::<_, speakerdesk_core::error::StoreError>(())
                })
                .is_err()
        );
        std::fs::remove_file(&media).unwrap();
        std::fs::create_dir(&media).unwrap();
        std::fs::write(media.join("audio.wav"), b"retained audio").unwrap();
        super::recover_deletions(state.data_dir(), &mut state.store()).unwrap();
        assert!(state.store().get(&id, false).unwrap().is_none());
        assert!(!media.exists());
        let response = running
            .authed(
                running
                    .client
                    .delete(running.url(&format!("/api/jobs/{JOB_A}"))),
            )
            .send()
            .await
            .unwrap();
        assert_eq!(response.status(), 200);
        running.shutdown().await;
    }

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
            separation_candidate: None,
            separation: None,
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

    #[test]
    fn only_fresh_explicit_accepted_repair_authorizes_accuracy_continuation() {
        for case in [
            "accepted",
            "same_words",
            "unrelated",
            "discarded",
            "duplicate",
            "stale_candidate",
            "forged",
            "error",
            "partial",
            "bounds",
        ] {
            let mut old = segment("Previous words", 2);
            old.rest.insert("language_epoch".into(), json!(4));
            old.rest.insert("language_mode".into(), json!("auto"));
            old.retry_candidate = Some(
                json!({"id":"owned-retry","start":old.start,"end":old.end,"speaker":old.speaker,"original_text":old.text,
                "language":"de","text":if case == "same_words" { "Previous words" } else { "Repaired words" }}),
            );
            match case {
                "duplicate" => old.retry_candidate.as_mut().unwrap()["accepted_at"] = json!(1.),
                "stale_candidate" => {
                    old.retry_candidate.as_mut().unwrap()["original_text"] = json!("Older revision")
                }
                "forged" => old.retry_candidate = None,
                "error" => {
                    old.retry_candidate.as_mut().unwrap()["error"] = json!("Failed model candidate")
                }
                "partial" => {
                    old.retry_candidate.as_mut().unwrap()["transcription_review"] =
                        json!({"partial_text":true})
                }
                _ => {}
            }
            let saved = Document {
                segments: vec![old.clone()],
                ..document("unused", 2)
            };
            let mut incoming = serde_json::to_value(&saved).unwrap();
            if case != "discarded" {
                incoming["segments"][0]["text"] = json!(if case == "same_words" {
                    "Previous words"
                } else {
                    "Repaired words"
                });
                incoming["segments"][0]["language"] = json!("de");
                incoming["segments"][0]["language_detection"] = if case == "unrelated" {
                    json!({"mode":"manual","reason":"override"})
                } else {
                    json!({"mode":"manual","reason":"passage_retry"})
                };
            }
            if case == "bounds" {
                incoming["segments"][0]["start"] = json!(0.1);
            }
            if case == "forged" {
                incoming["segments"][0]["retry_candidate"] = json!({"id":"uploaded","accepted_at":1.,"language":"de","text":"Repaired words"});
            }
            let outcome =
                speakerdesk_core::edit::save_transcript(speakerdesk_core::edit::SaveInput {
                    status: "ready",
                    duration: Some(10.),
                    revision: 7,
                    saved: Some(&saved),
                    body: &json!({"revision":7,"document":incoming}),
                    now: 12.,
                    speaker_assignments: &Map::new(),
                    voice_checks: &Map::new(),
                })
                .unwrap();
            let row = &outcome.document.segments[0];
            assert_eq!(
                super::accepted_retry(&old, row),
                matches!(case, "accepted" | "same_words"),
                "{case}"
            );
            assert_eq!(row.rest["language_mode"], "auto");
            assert_eq!(row.rest["language_epoch"], 4);
            if matches!(case, "accepted" | "same_words") {
                assert!(
                    outcome.language_context_changed,
                    "accepted language interpretation invalidates main summary"
                );
            }
            if case == "same_words" {
                assert_eq!(row.protected_fields, Some(vec!["text".into()]));
                assert!(row.retry_candidate.as_ref().unwrap()["accepted_at"].is_number());
            }
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

    impl ApplicationModels for GateModels {
        fn infer(
            &self,
            _job: &Job,
            _cancellation: &tokio_util::sync::CancellationToken,
        ) -> Result<Document, String> {
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

    #[cfg(unix)]
    #[test]
    fn fresh_native_storage_is_private() {
        use std::os::unix::fs::PermissionsExt;
        let state = AppState::new();
        assert_eq!(
            std::fs::metadata(state.data_dir())
                .unwrap()
                .permissions()
                .mode()
                & 0o777,
            0o700
        );
    }

    #[tokio::test]
    async fn migrated_audio_is_read_only_and_delete_keeps_rollback_media() {
        let state = AppState::new();
        let root = state.data_dir().to_path_buf();
        let id = JobId::parse(JOB_A).unwrap();
        let folder = root.join(JOB_A);
        std::fs::create_dir_all(&folder).unwrap();
        let audio = tiny_wav();
        std::fs::write(folder.join("source.wav"), &audio).unwrap();
        std::fs::write(folder.join("audio.wav"), &audio).unwrap();
        let mut stored = job(JOB_A, "ready", 1.0);
        stored.extra.insert("legacy_media_root".into(), json!(root));
        let derived = root.join("derived").join(JOB_A);
        std::fs::create_dir_all(&derived).unwrap();
        let stale = derived.join(".speakerdesk-audio-09093390-fcfb-45d3-a834-dac0e2fefb4e");
        std::fs::create_dir(&stale).unwrap();
        std::fs::write(stale.join("audio.wav"), b"interrupted conversion").unwrap();
        let decoded = derived.join(".speakerdesk-audio-interrupted.decode");
        std::fs::write(&decoded, b"interrupted decode").unwrap();
        super::AudioPrep::prepare(&super::WavPrepare, &root, &stored).unwrap();
        assert!(!stale.exists());
        assert!(!decoded.exists());
        assert!(root.join("derived").join(JOB_A).join("audio.wav").is_file());
        assert_eq!(std::fs::read(folder.join("audio.wav")).unwrap(), audio);
        state.insert_job(stored);
        let running = Running::spawn(state.clone()).await;
        let deleted = running
            .authed(
                running
                    .client
                    .delete(running.url(&format!("/api/jobs/{id}"))),
            )
            .send()
            .await
            .unwrap();
        assert_eq!(deleted.status(), 200);
        assert_eq!(std::fs::read(folder.join("audio.wav")).unwrap(), audio);
        assert_eq!(std::fs::read(folder.join("source.wav")).unwrap(), audio);
        assert!(!root.join("derived").join(JOB_A).exists());
        running.shutdown().await;
    }

    #[test]
    fn late_import_results_cannot_replace_edits_or_new_operations() {
        for invalidate in [false, true] {
            let state = AppState::new();
            let mut stored = job(JOB_A, "queued", 1.0);
            stored.document = None;
            stored.extra.insert("operation_generation".into(), json!(1));
            let id = stored.id.clone();
            state.insert_job(stored);
            let entered = Arc::new(AtomicUsize::new(0));
            let release = Arc::new((Mutex::new(false), Condvar::new()));
            state.set_models(Arc::new(GateModels {
                entered: entered.clone(),
                release: release.clone(),
                document: document("Late model words", 0),
            }));
            let shared = state.shared.clone();
            let thread = std::thread::spawn(move || {
                super::execute_import(&shared, &id, 1, &tokio_util::sync::CancellationToken::new())
            });
            let deadline = std::time::Instant::now() + Duration::from_secs(3);
            while entered.load(Ordering::SeqCst) == 0 {
                assert!(std::time::Instant::now() < deadline);
                std::thread::sleep(Duration::from_millis(1));
            }
            let id = JobId::parse(JOB_A).unwrap();
            state
                .store()
                .update_job(&id, |job| {
                    job.document = Some(document("Protected correction", 1));
                    if invalidate {
                        job.extra.insert("operation_generation".into(), json!(2));
                    }
                    Ok::<_, speakerdesk_core::error::StoreError>(())
                })
                .unwrap();
            *release.0.lock().unwrap() = true;
            release.1.notify_all();
            thread.join().unwrap();
            let saved = state.store().get(&id, false).unwrap().unwrap();
            assert_eq!(
                saved.document.unwrap().segments[0].text,
                "Protected correction"
            );
            assert_eq!(saved.revision, 9);
        }
    }

    #[tokio::test]
    async fn cancel_persists_before_notifying_the_import_worker() {
        struct GatePrepare {
            entered: Arc<AtomicUsize>,
            release: Arc<(Mutex<bool>, Condvar)>,
        }
        impl super::AudioPrep for GatePrepare {
            fn prepare(&self, _: &std::path::Path, _: &Job) -> Result<f64, String> {
                self.entered.fetch_add(1, Ordering::SeqCst);
                let (lock, wake) = &*self.release;
                drop(
                    wake.wait_while(lock.lock().unwrap(), |ready| !*ready)
                        .unwrap(),
                );
                Ok(10.0)
            }
        }
        for during_prepare in [false, true] {
            let state = AppState::new();
            state.set_issues(Vec::new());
            let mut stored = job(JOB_A, "uploaded", 1.0);
            stored.document = None;
            if during_prepare {
                stored.duration = None;
            }
            let id = stored.id.clone();
            state.insert_job(stored);
            let entered = Arc::new(AtomicUsize::new(0));
            let release = Arc::new((Mutex::new(false), Condvar::new()));
            if during_prepare {
                *state.shared.prepare.lock().unwrap() = Arc::new(GatePrepare {
                    entered: entered.clone(),
                    release: release.clone(),
                });
            }
            state.set_models(Arc::new(GateModels {
                entered: entered.clone(),
                release: release.clone(),
                document: document("Cancelled late output", 0),
            }));
            let running = Running::spawn(state.clone()).await;
            let response = running
                .authed(
                    running
                        .client
                        .post(running.url(&format!("/api/jobs/{JOB_A}/run"))),
                )
                .send()
                .await
                .unwrap();
            assert_eq!(response.status(), 202);
            wait_until(&entered, 1).await;
            let cancellation = state.shared.operations.lock().unwrap()[&id]
                .cancellation
                .clone();
            let cancelled = running
                .authed(
                    running
                        .client
                        .post(running.url(&format!("/api/jobs/{JOB_A}/cancel"))),
                )
                .send()
                .await
                .unwrap();
            assert_eq!(cancelled.status(), 200);
            assert!(cancellation.is_cancelled());
            let persisted = state.store().get(&id, false).unwrap().unwrap();
            assert_eq!(persisted.extra["operation_generation"], 2);
            assert_eq!(persisted.status, "uploaded");
            assert!(persisted.document.is_none());
            let repeated = running
                .authed(
                    running
                        .client
                        .post(running.url(&format!("/api/jobs/{JOB_A}/cancel"))),
                )
                .send()
                .await
                .unwrap();
            assert_eq!(repeated.status(), 200);
            assert_eq!(
                state.store().get(&id, false).unwrap().unwrap().revision,
                persisted.revision
            );
            assert!(
                state.shared.operations.lock().unwrap().contains_key(&id),
                "acknowledgement is not completion"
            );
            let (retry, delete) = tokio::join!(
                running
                    .authed(
                        running
                            .client
                            .post(running.url(&format!("/api/jobs/{JOB_A}/run")))
                    )
                    .send(),
                running
                    .authed(
                        running
                            .client
                            .delete(running.url(&format!("/api/jobs/{JOB_A}")))
                    )
                    .send(),
            );
            *release.0.lock().unwrap() = true;
            release.1.notify_all();
            state.drain();
            assert_eq!(
                retry.unwrap().status(),
                409,
                "retry must wait for terminal cleanup"
            );
            assert_eq!(
                delete.unwrap().status(),
                409,
                "audio is still owned by the worker"
            );
            assert!(state.shared.operations.lock().unwrap().is_empty());
            assert_eq!(
                state.store().get(&id, false).unwrap().unwrap().status,
                "uploaded",
                "cancelled preparation must not revive processing"
            );
            assert!(
                state
                    .store()
                    .get(&id, false)
                    .unwrap()
                    .unwrap()
                    .document
                    .is_none()
            );
            let retry = running
                .authed(
                    running
                        .client
                        .post(running.url(&format!("/api/jobs/{JOB_A}/run"))),
                )
                .send()
                .await
                .unwrap();
            assert_eq!(retry.status(), 202);
            state.drain();
            running.shutdown().await;
        }
    }

    #[tokio::test]
    async fn http_router_serves_config_with_security_headers() {
        let app = crate::http::router(AppState::new());
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
    async fn auto_readiness_requires_the_initialized_checkpoint_without_blocking_manual_imports() {
        use speakerdesk_core::{ipc::SetupModelReport, languages::LID_CHECKPOINT};
        let state = AppState::new();
        state.set_issues(Vec::new());
        let mut automatic = job(JOB_A, "uploaded", 1.0);
        automatic.language = "auto".into();
        automatic.document = None;
        state.insert_job(automatic);
        let mut manual = job(JOB_B, "uploaded", 2.0);
        manual.document = None;
        state.insert_job(manual);
        let running = Running::spawn(state.clone()).await;
        for (checkpoint, ok, expected) in [
            ("unknown-checkpoint", true, false),
            (LID_CHECKPOINT, true, true),
            (LID_CHECKPOINT, false, false),
        ] {
            state.set_model_reports(vec![SetupModelReport {
                model: checkpoint.into(),
                ok,
                timing_ms: 0,
            }]);
            let config: Value = running
                .client
                .get(running.url("/api/config"))
                .send()
                .await
                .unwrap()
                .json()
                .await
                .unwrap();
            assert_eq!(config["readiness"]["configured"], true);
            assert_eq!(config["readiness"]["automatic_language"], expected);
            let setup: Value = running
                .client
                .get(running.url("/api/setup"))
                .send()
                .await
                .unwrap()
                .json()
                .await
                .unwrap();
            assert_eq!(setup["core_ready"], true);
            assert_eq!(setup["automatic_language"], expected);
        }
        let rejected = running
            .authed(
                running
                    .client
                    .post(running.url(&format!("/api/jobs/{JOB_A}/run"))),
            )
            .send()
            .await
            .unwrap();
        assert_eq!(rejected.status(), 409);
        assert!(
            rejected
                .text()
                .await
                .unwrap()
                .contains(messages::LID_MODEL_MISSING)
        );
        let accepted = running
            .authed(
                running
                    .client
                    .post(running.url(&format!("/api/jobs/{JOB_B}/run"))),
            )
            .send()
            .await
            .unwrap();
        assert!(accepted.status().is_success());
        state.drain();
        running.shutdown().await;
    }

    #[tokio::test]
    async fn desktop_routes_serve_the_editor_audio_and_persistent_preferences() {
        let state = AppState::new();
        state.insert_job(job(JOB_A, "uploaded", 1.0));
        let folder = state.data_dir().join(JOB_A);
        std::fs::create_dir_all(&folder).unwrap();
        let wav = tiny_wav();
        std::fs::write(folder.join("audio.wav"), &wav).unwrap();
        let running = Running::spawn(state.clone()).await;
        let index = running.client.get(running.url("/")).send().await.unwrap();
        assert!(
            index.headers()["content-type"]
                .to_str()
                .unwrap()
                .starts_with("text/html")
        );
        let html = index.text().await.unwrap();
        assert!(html.contains("id=\"editor\""));
        assert!(html.contains(state.token()));
        assert!(!html.contains("{{token}}"));
        let script = running
            .client
            .get(running.url("/static/app.js"))
            .send()
            .await
            .unwrap();
        assert!(
            script.headers()["content-type"]
                .to_str()
                .unwrap()
                .starts_with("text/javascript")
        );
        assert!(
            script
                .text()
                .await
                .unwrap()
                .contains("function languageReady(mode)")
        );
        let audio_url = running.url(&format!("/api/jobs/{JOB_A}/audio"));
        let head = running.client.head(&audio_url).send().await.unwrap();
        assert_eq!(head.status(), 200);
        assert_eq!(head.headers()["content-length"], wav.len().to_string());
        assert!(head.bytes().await.unwrap().is_empty());
        let range = running
            .client
            .get(&audio_url)
            .header("Range", "bytes=0-15")
            .send()
            .await
            .unwrap();
        assert_eq!(range.status(), 206);
        assert_eq!(range.bytes().await.unwrap(), wav[..16]);
        let clips = running
            .client
            .get(running.url(&format!("/api/jobs/{JOB_A}/speakers/speaker_0/voice-clips")))
            .send()
            .await
            .unwrap();
        assert_eq!(clips.status(), 200);
        let clips: Value = clips.json().await.unwrap();
        assert_eq!(clips["revision"], 7);
        assert_eq!(clips["status"], "unavailable");
        assert!(clips["message"].as_str().unwrap().contains("qualified"));
        assert_eq!(clips["clips"], json!([]));
        assert_eq!(clips["recommended_ids"], json!([]));
        assert_eq!(clips["minimum_clips"], 2);
        assert_eq!(
            running
                .client
                .get(running.url(&format!("/api/jobs/{JOB_A}/speakers/missing/voice-clips")))
                .send()
                .await
                .unwrap()
                .status(),
            404
        );
        let save = running
            .authed(
                running
                    .client
                    .patch(running.url("/api/preferences/language")),
            )
            .json(&json!({"language":"fr"}))
            .send()
            .await
            .unwrap();
        assert_eq!(save.status(), 200);
        assert_eq!(
            state
                .store()
                .preference("default_language")
                .unwrap()
                .as_deref(),
            Some("fr")
        );
        assert_eq!(
            running
                .client
                .get(running.url("/api/config"))
                .send()
                .await
                .unwrap()
                .json::<Value>()
                .await
                .unwrap()["default_language"],
            "fr"
        );
        for path in ["/", "/static/app.js", "/api/jobs"] {
            assert_eq!(
                running
                    .client
                    .get(running.url(path))
                    .header("Origin", "https://example.com")
                    .send()
                    .await
                    .unwrap()
                    .status(),
                403
            );
        }
        state.admission.reserve(1, 1).unwrap();
        let blocked = running
            .authed(
                running
                    .client
                    .patch(running.url("/api/preferences/language")),
            )
            .json(&json!({"language":"en"}))
            .send()
            .await
            .unwrap();
        assert_eq!(blocked.status(), 409);
        state.admission.release(1, 1).unwrap();
        running.shutdown().await;
    }

    #[test]
    fn shutdown_write_failure_cancels_recording_before_releasing_the_store() {
        let state = AppState::new();
        let mut recording = job(JOB_A, "ready", 1.0);
        recording.kind = JobKind::Meeting;
        // SQLite can store this revision, but the next mutation cannot increment it.
        recording.revision = i64::MAX;
        let operation = state
            .begin_recording_operation(&mut state.store(), &mut recording)
            .unwrap();
        let before = state.store().get(&recording.id, false).unwrap().unwrap();
        assert!(
            state
                .cancel_all()
                .unwrap_err()
                .to_string()
                .contains("Revision overflow")
        );
        assert!(operation.cancellation.is_cancelled());
        assert!(!operation.current(&before));
        assert_eq!(
            state.store().get(&recording.id, false).unwrap().unwrap(),
            before
        );
        assert!(state.admission.closed());
    }

    #[test]
    fn shutdown_cancels_only_current_active_generations_and_preserves_terminal_results() {
        let state = AppState::new();
        let mut completed = job(JOB_A, "ready", 1.0);
        completed
            .extra
            .insert("operation_generation".into(), json!(4));
        state.insert_job(completed);
        let before = state
            .store()
            .get(&JobId::parse(JOB_A).unwrap(), false)
            .unwrap()
            .unwrap();
        let mut active = job(JOB_B, "processing", 1.0);
        active.extra.insert("operation_generation".into(), json!(7));
        state.insert_job(active);
        for (id, generation) in [(JOB_A, 4), (JOB_B, 7)] {
            state.shared.operations.lock().unwrap().insert(
                JobId::parse(id).unwrap(),
                super::RunningOperation {
                    generation,
                    cancellation: tokio_util::sync::CancellationToken::new(),
                    activity: super::Activity::Import,
                },
            );
        }
        state.cancel_all().unwrap();
        assert!(state.admission.closed());
        let store = state.store();
        assert_eq!(
            store
                .get(&JobId::parse(JOB_A).unwrap(), false)
                .unwrap()
                .unwrap(),
            before
        );
        let active = store
            .get(&JobId::parse(JOB_B).unwrap(), false)
            .unwrap()
            .unwrap();
        assert_eq!(active.status, "uploaded");
        assert_eq!(active.extra["operation_generation"], 8);
        assert!(
            state
                .shared
                .operations
                .lock()
                .unwrap()
                .values()
                .all(|op| op.cancellation.is_cancelled())
        );
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

        let recording = {
            let mut store = running.state.store();
            let mut busy = store
                .get(&JobId::parse(JOB_B).unwrap(), false)
                .unwrap()
                .unwrap();
            busy.status = "ready".into();
            store.put(&busy, None).unwrap();
            let mut job = store
                .get(&JobId::parse(JOB_A).unwrap(), false)
                .unwrap()
                .unwrap();
            running
                .state
                .begin_recording_operation(&mut store, &mut job)
                .unwrap()
        };
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
        drop(recording);
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
        assert_eq!(
            running.state.admission.reserve(1, 1),
            Err(crate::admission::AdmissionError::Busy)
        );
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

        let recording = {
            let mut store = running.state.store();
            let mut job = store
                .get(&JobId::parse(JOB_B).unwrap(), false)
                .unwrap()
                .unwrap();
            running
                .state
                .begin_recording_operation(&mut store, &mut job)
                .unwrap()
        };
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
        drop(recording);
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
