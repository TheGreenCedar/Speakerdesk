//! Disk-backed live recording. SQLite owns jobs; this manager owns only runtime handles.
mod runtime;
mod worker;

use crate::{
    admission::Activity,
    capture::CaptureSupervisor,
    routes::{AppState, OperationGuard, blocking, stored_job_json},
};
use axum::{
    Json, Router,
    body::Bytes,
    extract::{Path, State},
    http::StatusCode,
    response::{IntoResponse, Response},
    routing::{get, patch, post},
};
use serde::Deserialize;
use serde_json::{Map, Value, json};
use speakerdesk_core::{
    document::{Document, Job, JobKind},
    error::{ApiError, StoreError},
    ids::JobId,
    languages, limits, messages,
    store::JobStore,
};
use std::{
    collections::{BTreeMap, HashSet},
    fs,
    path::PathBuf,
};
use tokio::sync::{mpsc, oneshot, watch};
use tokio_util::sync::CancellationToken;

#[derive(Default)]
pub struct MeetingManager {
    start_gate: std::sync::Arc<tokio::sync::Mutex<()>>,
    helper: Option<PathBuf>,
    echo_library: Option<PathBuf>,
    helper_error: Option<String>,
    cleanup_error: Option<String>,
    active: Option<Active>,
}
struct Active {
    id: JobId,
    commands: mpsc::Sender<Control>,
    progress: watch::Receiver<Progress>,
    task: Option<tokio::task::JoinHandle<Result<(), StoreError>>>,
}
#[derive(Clone, Default)]
struct Progress {
    processed: f64,
    levels: BTreeMap<String, f32>,
    error: Option<String>,
    recovering: bool,
}
struct Control {
    action: Action,
    reply: oneshot::Sender<Result<Value, ApiError>>,
}
enum Action {
    Pause,
    Resume,
    Stop,
    Accuracy { resume: bool },
    Language { language: String, revision: i64 },
}

impl MeetingManager {
    pub fn new() -> Self {
        Self::default()
    }
}

pub(crate) async fn configure_capture(
    app: &AppState,
    helper: Option<PathBuf>,
    echo_library: Option<PathBuf>,
) {
    let checked = tokio::task::spawn_blocking(move || {
        helper.map(|path| {
            CaptureSupervisor::check_helper(&path)
                .map(|_| path)
                .map_err(|e| e.to_string())
        })
    })
    .await;
    let manager = app.meeting_manager();
    let mut manager = manager.lock().expect("meeting");
    manager.echo_library = echo_library;
    match checked {
        Ok(Some(Ok(path))) => {
            manager.helper = Some(path);
            manager.helper_error = None;
        }
        Ok(Some(Err(error))) => manager.helper_error = Some(error),
        Err(error) => manager.helper_error = Some(error.to_string()),
        Ok(None) => manager.helper_error = Some("The capture helper is not configured.".into()),
    }
}

/// Called after durable operation invalidation. Joining proves capture, writer and model cleanup.
pub(crate) async fn shutdown(app: &AppState) -> Result<(), StoreError> {
    let gate = app
        .meeting_manager()
        .lock()
        .expect("meeting")
        .start_gate
        .clone();
    let _start_gate = gate.lock_owned().await;
    let (task, error) = {
        let manager = app.meeting_manager();
        let mut manager = manager.lock().expect("meeting");
        (
            manager
                .active
                .as_mut()
                .and_then(|active| active.task.take()),
            manager.cleanup_error.clone(),
        )
    };
    if let Some(task) = task {
        task.await
            .map_err(|e| StoreError::message(e.to_string()))??;
    }
    error.map_or(Ok(()), |e| Err(StoreError::message(e)))
}

pub(crate) fn routes() -> Router<AppState> {
    Router::new()
        .route("/api/meeting", get(status))
        .route("/api/meetings", post(start))
        .route("/api/meetings/{jid}/{action}", post(control))
        .route("/api/meetings/{jid}/language", patch(language))
        .route(
            "/api/jobs/{jid}/refinement/{action}",
            post(crate::routes::refinement::action),
        )
        .route(
            "/api/jobs/{jid}/refinement/revisions/{window}",
            get(crate::routes::refinement::history),
        )
}
fn error(error: ApiError) -> Response {
    (
        StatusCode::from_u16(error.status).unwrap_or(StatusCode::INTERNAL_SERVER_ERROR),
        Json(error.body()),
    )
        .into_response()
}
fn conflict(message: impl Into<String>) -> ApiError {
    ApiError::new(409, message)
}
fn number(job: &Job, key: &str) -> i64 {
    job.extra.get(key).and_then(Value::as_i64).unwrap_or(0)
}

pub(crate) async fn refinement_action(app: &AppState, jid: &str, action: &str) -> Option<Response> {
    let commands = {
        let manager = app.meeting_manager();
        let manager = manager.lock().expect("meeting");
        manager
            .active
            .as_ref()
            .filter(|active| {
                active.id.as_str() == jid
                    && active.task.as_ref().is_some_and(|task| !task.is_finished())
            })
            .map(|active| active.commands.clone())
    }?;
    if !["resume", "pause"].contains(&action) {
        return Some(error(ApiError::new(404, "Not found.")));
    }
    let (reply, response) = oneshot::channel();
    if commands
        .send(Control {
            action: Action::Accuracy {
                resume: action == "resume",
            },
            reply,
        })
        .await
        .is_err()
    {
        return Some(error(conflict("This meeting is finishing.")));
    }
    Some(match response.await {
        Ok(Ok(job)) => (
            if action == "resume" {
                StatusCode::ACCEPTED
            } else {
                StatusCode::OK
            },
            Json(job),
        )
            .into_response(),
        Ok(Err(e)) => error(e),
        Err(_) => error(conflict("This meeting is finishing.")),
    })
}

async fn status(State(app): State<AppState>) -> Response {
    blocking(move || {
        let (id, progress, available, helper_error) = {
            let manager = app.meeting_manager();
            let manager = manager.lock().expect("meeting");
            (manager.active.as_ref().map(|a| a.id.clone()), manager.active.as_ref().map(|a| a.progress.borrow().clone()).unwrap_or_default(), manager.helper.is_some(), manager.helper_error.clone())
        };
        let job = match id { Some(id) => match app.store().get(&id, false) { Ok(job) => job, Err(e) => return error(e.into()) }, None => None };
        let duration = job.as_ref().and_then(|j| j.duration).unwrap_or(0.0);
        Json(json!({"id": job.as_ref().map(|j| j.id.as_str()), "status":job.as_ref().map_or("idle", |j| j.status.as_str()),
            "sources":job.as_ref().and_then(|j|j.extra.get("sources")), "duration":duration,
            "processed_seconds":progress.processed, "pending_seconds":(duration-progress.processed).max(0.0), "levels":progress.levels,
            "error":progress.error.or(helper_error), "capture_available":available,
            "language":job.as_ref().map(|j| &j.language),
            "language_revision":job.as_ref().map_or(0,|j|number(j,"language_revision")),
            "language_acknowledged_revision":job.as_ref().map_or(0,|j|number(j,"language_acknowledged_revision")),
            "language_from_sample":job.as_ref().and_then(|j|j.extra.get("language_history")).and_then(Value::as_array).and_then(|h|h.last()).and_then(|e|e.get("start_sample")).cloned().unwrap_or(json!(0)),
            "pause_flush":job.as_ref().and_then(|j|j.extra.get("pause_flush")), "refinement_status":job.as_ref().and_then(|j|j.extra.get("refinement_status"))})).into_response()
    }).await
}

#[derive(Deserialize)]
struct Start {
    name: Option<String>,
    language: Option<String>,
    sources: Vec<String>,
}
async fn start(State(app): State<AppState>, body: Bytes) -> Response {
    let request: Start = match serde_json::from_slice(&body) {
        Ok(v) => v,
        Err(_) => return error(ApiError::new(400, "Invalid meeting request.")),
    };
    let gate = app
        .meeting_manager()
        .lock()
        .expect("meeting")
        .start_gate
        .clone();
    let start_gate = gate.lock_owned().await;
    if app
        .meeting_manager()
        .lock()
        .expect("meeting")
        .active
        .as_ref()
        .and_then(|a| a.task.as_ref())
        .is_some_and(|t| !t.is_finished())
    {
        return error(conflict(messages::MEETING_ACTIVE));
    }
    let completed = {
        let manager = app.meeting_manager();
        let mut manager = manager.lock().expect("meeting");
        manager
            .active
            .as_mut()
            .filter(|a| a.task.as_ref().is_some_and(|t| t.is_finished()))
            .and_then(|a| a.task.take())
    };
    if let Some(task) = completed {
        let result = task
            .await
            .map_err(|e| e.to_string())
            .and_then(|r| r.map_err(|e| e.to_string()));
        if let Err(message) = result {
            app.meeting_manager().lock().expect("meeting").cleanup_error = Some(message);
        }
    }
    let handle = tokio::runtime::Handle::current();
    blocking(move || {
        // The admitted mutation outlives a disconnected HTTP client.
        let _start_gate = start_gate;
        let result = (|| {
            if request.sources.is_empty()
                || request.sources.len() > 2
                || request.sources.iter().any(|s| !["microphone", "system"].contains(&s.as_str()))
                || request.sources.iter().collect::<HashSet<_>>().len() != request.sources.len()
            {
                return Err(ApiError::new(400, "Choose microphone, system audio, or both."));
            }
            let lease = app.admission.lease(Activity::recording())
                .map_err(|e| conflict(e.message()))?;
            if app.meeting_manager().lock().expect("meeting").cleanup_error.is_some() {
                return Err(conflict("Previous recording cleanup failed. Restart the application before recording again."));
            }
            if !app.model_status().0.is_empty()
                || app.models().live_models(&CancellationToken::new()).is_none()
            {
                return Err(conflict(messages::OPEN_SETTINGS_SETUP));
            }
            let helper = app.meeting_manager().lock().expect("meeting").helper.clone()
                .ok_or_else(|| conflict("The capture helper is not available."))?;
            let free_bytes = fs4::available_space(app.data_dir())
                .map_err(|e| ApiError::new(500, e.to_string()))?;
            if free_bytes < limits::MEETING_FREE_BYTES {
                return Err(conflict(messages::MEETING_FREE_SPACE));
            }
            let language = request.language.unwrap_or_else(|| {
                app.store().preference("default_language").ok().flatten()
                    .unwrap_or_else(|| "auto".into())
            });
            validate_language(&app, &language)?;
            let name = request.name.unwrap_or_else(|| "New meeting".into());
            let name = name.trim();
            if name.is_empty() || name.chars().count() > 160 {
                return Err(ApiError::new(400, "Meeting name must contain 1–160 characters."));
            }
            let id = crate::routes::new_job_id()?;
            let folder = app.data_dir().join(id.as_str());
            let mut builder = fs::DirBuilder::new();
            #[cfg(unix)]
            {
                use std::os::unix::fs::DirBuilderExt;
                builder.mode(0o700);
            }
            builder.create(&folder).map_err(|e| ApiError::new(500, e.to_string()))?;
            let mut job = meeting_job(id.clone(), name, &language, &request.sources);
            let operation = match app.begin_recording_operation(&mut app.store(), &mut job) {
                Ok(op) => op,
                Err(e) => {
                    let _ = fs::remove_dir(&folder);
                    return Err(e);
                }
            };
            let response = stored_job_json(&job);
            let (commands, receiver) = mpsc::channel(8);
            let (progress_tx, progress) = watch::channel(Progress::default());
            let manager = app.meeting_manager();
            let mut manager = manager.lock().expect("meeting");
            let task = handle.spawn(runtime::record(
                app.clone(), id.clone(), language, request.sources, helper,
                std::sync::Arc::new(operation), lease, receiver, progress_tx,
            ));
            manager.active = Some(Active { id, commands, progress, task: Some(task) });
            Ok(response)
        })();
        match result {
            Ok(value) => (StatusCode::ACCEPTED, Json(value)).into_response(),
            Err(e) => error(e),
        }
    })
    .await
}

async fn send(app: &AppState, jid: &str, action: Action) -> Result<Value, ApiError> {
    let sender = {
        let manager = app.meeting_manager();
        let manager = manager.lock().expect("meeting");
        manager
            .active
            .as_ref()
            .filter(|a| a.id.as_str() == jid && a.task.as_ref().is_some_and(|t| !t.is_finished()))
            .map(|a| a.commands.clone())
            .ok_or_else(|| conflict("This meeting is no longer recording."))?
    };
    let (reply, receiver) = oneshot::channel();
    sender
        .try_send(Control { action, reply })
        .map_err(|_| ApiError::new(429, "Meeting controls are busy."))?;
    receiver
        .await
        .map_err(|_| conflict("This meeting has finished."))?
}
async fn control(
    State(app): State<AppState>,
    Path((jid, action)): Path<(String, String)>,
) -> Response {
    let action = match action.as_str() {
        "pause" => Action::Pause,
        "resume" => Action::Resume,
        "stop" => Action::Stop,
        _ => return error(ApiError::new(404, "Unknown meeting action.")),
    };
    match send(&app, &jid, action).await {
        Ok(v) => (StatusCode::ACCEPTED, Json(v)).into_response(),
        Err(e) => error(e),
    }
}
#[derive(Deserialize)]
struct Language {
    language: String,
    language_revision: i64,
}
async fn language(State(app): State<AppState>, Path(jid): Path<String>, body: Bytes) -> Response {
    let request: Language = match serde_json::from_slice(&body) {
        Ok(v) => v,
        Err(_) => return error(ApiError::new(400, "Invalid language request.")),
    };
    if let Err(e) = validate_language(&app, &request.language) {
        return error(e);
    }
    match send(
        &app,
        &jid,
        Action::Language {
            language: request.language,
            revision: request.language_revision,
        },
    )
    .await
    {
        Ok(v) => Json(v).into_response(),
        Err(e) => error(e),
    }
}
fn validate_language(app: &AppState, language: &str) -> Result<(), ApiError> {
    if language == languages::AUTOMATIC {
        if !app.automatic_language_ready() {
            return Err(conflict(messages::LID_MODEL_MISSING));
        }
    } else if !languages::is_supported(language) {
        return Err(ApiError::new(400, "Unsupported language."));
    }
    Ok(())
}
fn meeting_job(id: JobId, name: &str, language: &str, sources: &[String]) -> Job {
    let history = json!([{"epoch":0,"generation":0,"language":language,"start_sample":0}]);
    let now = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs_f64();
    Job {id,name:name.into(),kind:JobKind::Meeting,status:"starting".into(),message:"Starting meeting…".into(),language:language.into(),source_file:Some("audio.wav".into()),duration:Some(0.0),revision:0,created:now,updated:now,
        document:Some(Document{schema_version:1,speakers:BTreeMap::new(),segments:Vec::new(),provenance:Some(json!({"kind":"local_inference","mode":"live_phrase_windows","language":language,"language_history":history,"timing":"Speech crop boundaries; supported sealed passages use supplied-text word timing. Uncertain or overlapping speech needs review. Sealed passages are rechecked with fresh bounded context; saved corrections are protected."})),warnings:vec!["Phrase boundaries and overlapping speech need review.".into()],diarization:Vec::new(),edited_at:None,rest:Map::new()}),
        extra:serde_json::from_value(json!({"sources":sources,"language_revision":0,"language_acknowledged_revision":0,"language_epoch":0,"language_history":history,"refinement_status":"waiting","refinement_generation":0,"rolling_refinement":{"version":1,"mode":"live","completed":{},"failed":{}}})).expect("object")}
}
fn update(
    app: &AppState,
    id: &JobId,
    operation: &OperationGuard,
    change: impl FnOnce(&mut Job) -> Result<(), ApiError>,
) -> Result<Job, ApiError> {
    app.store()
        .update_job(id, |job| {
            if !operation.current(job) {
                return Err(conflict("Recording operation was cancelled."));
            }
            change(job)
        })?
        .map(|(job, ())| job)
        .ok_or_else(|| ApiError::new(404, "Meeting not found."))
}
