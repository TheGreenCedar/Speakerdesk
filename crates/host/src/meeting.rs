//! Live meeting routes.
//!
//! The handlers speak the v0.6.6 meeting API and record worker commands on
//! [`speakerdesk_core::live::Inbox`]. They do not spawn the Swift capture
//! helper or an MLX process. The parent router checks the session token.

use std::collections::HashSet;
use std::fs;
use std::io::Read;
use std::path::Path;
use std::sync::{Arc, Mutex};
use std::time::{SystemTime, UNIX_EPOCH};

use axum::Json;
use axum::body::Bytes;
use axum::extract::{Path as AxumPath, State};
use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use axum::routing::{get, patch, post};
use serde_json::{Map, Value, json};
use speakerdesk_core::document::{Document, Job, JobKind};
use speakerdesk_core::ids::JobId;
use speakerdesk_core::live::{AudioNotice, Inbox, LiveCommand};
use speakerdesk_core::{languages, limits, messages};

use crate::capture::Command;
use crate::routes::AppState;

const NOT_FOUND: &str = "The requested URL was not found on the server. If you entered the URL manually please check your spelling and try again.";
const TITLE_MAX_CHARS: usize = 160;
const PACKET_LIMIT: usize = 120;
const LIVE_STATUSES: &[&str] = &["starting", "recording", "paused", "finishing"];
/// `language_detection.LID_CHECKPOINT` (`repo` + `@` + `revision`).
const LID_CHECKPOINT: &str =
    "mlx-community/whisper-tiny-asr-fp16@77fa3f52b482ec80d086df55f893065a9baab172";

/// How a JSON field was supplied.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Supplied {
    Missing,
    Invalid,
    Value(String),
}

/// Body of `POST /api/meetings` after the source-list check.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct StartRequest {
    pub name: Supplied,
    pub language: Supplied,
    pub sources: Vec<String>,
}

/// Preconditions `POST /api/meetings` can see.
///
/// `None` means [`AppState`] does not expose that flag. The matching
/// [`MeetingManager`] check then returns the frozen refusal.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct StartObservation {
    pub import_running: Option<bool>,
    pub setup_ready: Option<bool>,
    pub capture_available: bool,
    pub free_bytes: u64,
}

/// Preconditions for saved-audio refinement.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RefineObservation {
    pub import_running: Option<bool>,
    pub setup_ready: Option<bool>,
}

/// Refusal or validation failure. `message` is a frozen v0.6.6 string.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MeetingError {
    pub status: StatusCode,
    pub message: &'static str,
}

impl MeetingError {
    pub const fn new(status: StatusCode, message: &'static str) -> Self {
        Self { status, message }
    }

    pub const fn conflict(message: &'static str) -> Self {
        Self::new(StatusCode::CONFLICT, message)
    }

    pub const fn bad(message: &'static str) -> Self {
        Self::new(StatusCode::BAD_REQUEST, message)
    }

    pub const fn missing() -> Self {
        Self::new(StatusCode::NOT_FOUND, NOT_FOUND)
    }

    pub const fn backlog(message: &'static str) -> Self {
        Self::new(StatusCode::TOO_MANY_REQUESTS, message)
    }
}

impl IntoResponse for MeetingError {
    fn into_response(self) -> Response {
        (self.status, Json(json!({ "error": self.message }))).into_response()
    }
}

/// In-memory live meeting. One host process can own several test apps.
pub struct MeetingManager {
    active: Option<Job>,
    default_language: String,
    inbox: Inbox,
    packet_count: usize,
    received_samples: i64,
    duration: f64,
    processed: f64,
    levels: Map<String, Value>,
    error: Option<String>,
    capture_started: bool,
    capture_alive: bool,
    worker_alive: bool,
    two_pass: bool,
    stopped: bool,
    shutdown_sent: bool,
    capture_log: Vec<Command>,
    epoch: i64,
}

impl Default for MeetingManager {
    fn default() -> Self {
        Self::new()
    }
}

impl MeetingManager {
    pub fn new() -> Self {
        Self {
            active: None,
            default_language: languages::AUTOMATIC.to_string(),
            inbox: Inbox::new(1, 0).expect("live inbox bounds"),
            packet_count: 0,
            received_samples: 0,
            duration: 0.0,
            processed: 0.0,
            levels: Map::new(),
            error: None,
            capture_started: false,
            capture_alive: false,
            worker_alive: false,
            two_pass: false,
            stopped: false,
            shutdown_sent: false,
            capture_log: Vec::new(),
            epoch: 0,
        }
    }

    pub fn default_language(&self) -> &str {
        &self.default_language
    }

    pub fn capture_commands(&self) -> &[Command] {
        &self.capture_log
    }

    /// Gates visible from the public [`AppState`] surface.
    ///
    /// Setup readiness and the import worker are private, so both are `None`.
    /// This build does not include the Swift capture helper.
    pub const fn host_gates(free_bytes: u64) -> StartObservation {
        StartObservation {
            import_running: None,
            setup_ready: None,
            capture_available: false,
            free_bytes,
        }
    }

    /// Saved refinement cannot see setup or the import worker either.
    pub const fn host_refine_gates() -> RefineObservation {
        RefineObservation {
            import_running: None,
            setup_ready: None,
        }
    }

    /// `None` and `Some(true)` both refuse: a hidden worker counts as busy.
    pub const fn import_block(import_running: Option<bool>) -> Option<&'static str> {
        match import_running {
            Some(false) => None,
            Some(true) | None => Some(messages::MEETING_WAIT_IMPORT),
        }
    }

    /// `None` and `Some(false)` both refuse with the setup message.
    pub const fn setup_block(setup_ready: Option<bool>) -> Option<&'static str> {
        match setup_ready {
            Some(true) => None,
            Some(false) | None => Some(messages::OPEN_SETTINGS_SETUP),
        }
    }

    /// Disk gate for `limits::MEETING_FREE_BYTES` (512 MiB).
    pub const fn free_space_block(free_bytes: u64) -> Option<&'static str> {
        if free_bytes < limits::MEETING_FREE_BYTES {
            Some(messages::MEETING_FREE_SPACE)
        } else {
            None
        }
    }

    /// Import inference, as seen by saved-audio resume.
    pub const fn inference_busy_block(import_running: Option<bool>) -> Option<&'static str> {
        match import_running {
            Some(false) => None,
            Some(true) | None => Some(messages::WAIT_FOR_LOCAL_INFERENCE),
        }
    }

    /// Model setup, as seen by saved-audio resume.
    pub const fn refine_setup_block(setup_ready: Option<bool>) -> Option<&'static str> {
        match setup_ready {
            Some(true) => None,
            Some(false) | None => Some(messages::SETUP_BEFORE_REFINE),
        }
    }

    /// `GET /api/meeting`.
    pub fn status(&self) -> Value {
        let job = self.active.as_ref();
        json!({
            "id": job.map(|job| job.id.as_str()),
            "status": job.map(|job| job.status.as_str()).unwrap_or("idle"),
            "sources": job.and_then(|job| job.extra.get("sources")).cloned().unwrap_or_else(|| json!([])),
            "duration": self.duration,
            "processed_seconds": self.processed,
            "pending_seconds": (self.duration - self.processed).max(0.0),
            "levels": self.levels,
            "error": self.error,
            "capture_available": false,
            "language": job.map(|job| job.language.as_str()),
            "language_revision": job.map(|job| number_field(job, "language_revision")).unwrap_or(0),
            "language_acknowledged_revision": job.map(|job| number_field(job, "language_acknowledged_revision")).unwrap_or(0),
            "language_from_sample": job.map(last_language_sample).unwrap_or(0),
            "pause_flush": job.and_then(|job| job.extra.get("pause_flush")).cloned(),
            "refinement_status": job
                .and_then(|job| job.extra.get("refinement_status"))
                .and_then(Value::as_str)
                .unwrap_or(if job.is_some() { "waiting" } else { "idle" }),
        })
    }

    /// Python `MeetingManager.start`, without launching capture or models.
    pub fn start(
        &mut self,
        request: &StartRequest,
        observed: StartObservation,
        folder: &Path,
    ) -> Result<Job, MeetingError> {
        if self.runtime_owned() {
            return Err(MeetingError::conflict(messages::MEETING_ACTIVE));
        }
        if let Some(message) = Self::import_block(observed.import_running) {
            return Err(MeetingError::conflict(message));
        }
        let name = match &request.name {
            Supplied::Missing => "Meeting".to_string(),
            Supplied::Invalid => return Err(MeetingError::bad(messages::MEETING_TITLE)),
            Supplied::Value(name) => name.clone(),
        };
        let language = match &request.language {
            Supplied::Missing => self.default_language.clone(),
            Supplied::Invalid => return Err(MeetingError::bad(messages::MEETING_TITLE)),
            Supplied::Value(language) => language.clone(),
        };
        if !valid_title(&name) || !language_choice(&language) {
            return Err(MeetingError::bad(messages::MEETING_TITLE));
        }
        if !valid_routes(&request.sources) {
            return Err(MeetingError::bad(messages::MEETING_ROUTE));
        }
        if !observed.capture_available {
            return Err(MeetingError::conflict(messages::LIVE_CAPTURE_MISSING));
        }
        if both_sources(&request.sources) {
            return Err(MeetingError::conflict(messages::ECHO_MISSING));
        }
        if let Some(message) = Self::setup_block(observed.setup_ready) {
            return Err(MeetingError::conflict(message));
        }
        if let Some(message) = Self::free_space_block(observed.free_bytes) {
            return Err(MeetingError::conflict(message));
        }
        self.open_meeting(name.trim(), &language, &request.sources, folder)
    }

    /// The live worker reported ready. No model process is started here.
    pub fn note_worker_ready(&mut self, two_pass: bool) -> Result<(), MeetingError> {
        if self.active.is_none() {
            return Err(MeetingError::conflict(messages::MEETING_INACTIVE));
        }
        self.worker_alive = true;
        self.two_pass = two_pass;
        Ok(())
    }

    /// Capture helper status line (`recording` or `paused`).
    pub fn note_capture_phase(&mut self, phase: &str) -> Result<Job, MeetingError> {
        if !matches!(phase, "recording" | "paused") {
            return Err(MeetingError::bad(messages::INVALID_REQUEST));
        }
        if self.active.is_none() {
            return Err(MeetingError::conflict(messages::MEETING_INACTIVE));
        }
        self.capture_started = true;
        self.capture_alive = true;
        let request_id = self.acknowledge_pause(phase);
        if !self.stopped {
            let message = if phase == "recording" {
                "Recording"
            } else {
                "Paused"
            };
            let job = self.active.as_mut().expect("meeting");
            job.status = phase.to_string();
            job.message = message.to_string();
            job.updated = unix_now();
        }
        if phase == "paused"
            && self.two_pass
            && let Some(request_id) = request_id
        {
            self.inbox
                .push(LiveCommand::Flush {
                    request_id: Some(request_id),
                    through_sample: Some(self.received_samples),
                })
                .map_err(|_| {
                    MeetingError::new(
                        StatusCode::INTERNAL_SERVER_ERROR,
                        messages::LOCAL_OPERATION_FAILED,
                    )
                })?;
        }
        self.active.clone().ok_or_else(MeetingError::missing)
    }

    /// One capture packet. The live inbox coalesces contiguous audio.
    pub fn retain_audio(&mut self, samples: i64) -> Result<(), MeetingError> {
        if self.active.is_none() {
            return Err(MeetingError::conflict(messages::MEETING_INACTIVE));
        }
        if samples <= 0 {
            return Err(MeetingError::bad(messages::CAPTURE_DURATION_INVALID));
        }
        if self.packet_count >= PACKET_LIMIT {
            return Err(MeetingError::backlog(messages::LIVE_LANGUAGE_BACKLOG));
        }
        let start_sample = self.received_samples;
        let end_sample = start_sample.saturating_add(samples);
        let language = self.active.as_ref().expect("meeting").language.clone();
        self.inbox
            .push(LiveCommand::Audio(AudioNotice {
                start_sample,
                end_sample,
                language_epoch: self.epoch,
                language,
            }))
            .map_err(|_| {
                MeetingError::new(
                    StatusCode::INTERNAL_SERVER_ERROR,
                    messages::LOCAL_OPERATION_FAILED,
                )
            })?;
        self.received_samples = end_sample;
        self.duration = end_sample as f64 / speakerdesk_core::live::RATE as f64;
        self.packet_count += 1;
        if let Some(job) = self.active.as_mut() {
            job.duration = Some(self.duration);
            job.updated = unix_now();
        }
        Ok(())
    }

    /// Record an in-flight refinement id so pause can cancel it.
    pub fn note_refinement(&mut self, operation_id: &str) -> Result<(), MeetingError> {
        let Some(job) = self.active.as_mut() else {
            return Err(MeetingError::conflict(messages::MEETING_INACTIVE));
        };
        job.extra.insert(
            "rolling_inflight".to_string(),
            json!({ "operation_id": operation_id }),
        );
        Ok(())
    }

    /// Whether pause recorded this refinement id on the live inbox.
    pub fn refinement_cancelled(&self, operation_id: &str) -> bool {
        self.inbox
            .cancelled_request(&speakerdesk_core::live::RefineCommand {
                operation_id: operation_id.to_string(),
                language_epoch: self.epoch,
                canonical_sealed: false,
            })
    }

    /// Next coalesced command for the live worker.
    pub fn poll_worker(&mut self) -> Result<Option<LiveCommand>, MeetingError> {
        self.inbox.take().map_err(|_| {
            MeetingError::new(
                StatusCode::INTERNAL_SERVER_ERROR,
                messages::LOCAL_OPERATION_FAILED,
            )
        })
    }

    /// `POST /api/meetings/{jid}/{action}`.
    pub fn control(&mut self, jid: &str, action: &str) -> Result<Job, MeetingError> {
        if !matches!(action, "pause" | "resume" | "stop") {
            return Err(MeetingError::missing());
        }
        if self
            .active
            .as_ref()
            .is_none_or(|job| job.id.as_str() != jid)
        {
            return Err(MeetingError::conflict(messages::MEETING_INACTIVE));
        }
        let status = self.active.as_ref().expect("meeting").status.clone();
        let required = match action {
            "pause" => Some("recording"),
            "resume" => Some("paused"),
            _ => None,
        };
        if !is_live(&status) || required.is_some_and(|needed| status != needed) {
            return Err(MeetingError::conflict(messages::MEETING_NOT_READY));
        }
        if self.capture_started && !self.capture_alive {
            self.stopped = true;
            return Err(MeetingError::conflict(messages::CAPTURE_STOPPED));
        }
        if !self.capture_started {
            if action == "stop" {
                return self.finish_before_capture();
            }
            return Err(MeetingError::conflict(messages::WAIT_FOR_RECORDING_START));
        }
        match action {
            "stop" => self.stop_capture(),
            "pause" => self.pause_capture(),
            "resume" => self.resume_capture(),
            _ => Err(MeetingError::missing()),
        }
    }

    /// `PATCH /api/meetings/{jid}/language`.
    pub fn change_language(
        &mut self,
        jid: &str,
        body: &Map<String, Value>,
        setup_ready: Option<bool>,
    ) -> Result<Value, MeetingError> {
        let (language, revision) = live_language_request(body)?;
        if self
            .active
            .as_ref()
            .is_none_or(|job| job.id.as_str() != jid)
        {
            return Err(MeetingError::conflict(messages::MEETING_INACTIVE));
        }
        let status = self.active.as_ref().expect("meeting").status.clone();
        if !matches!(status.as_str(), "recording" | "paused") || self.stopped || !self.worker_alive
        {
            return Err(MeetingError::conflict(messages::LANGUAGE_WHILE_RECORDING));
        }
        let current_revision =
            number_field(self.active.as_ref().expect("meeting"), "language_revision");
        if revision != current_revision {
            return Err(MeetingError::conflict(messages::LIVE_LANGUAGE_STALE));
        }
        let current_language = self.active.as_ref().expect("meeting").language.clone();
        if language == current_language {
            self.default_language.clone_from(&language);
            return self.language_body();
        }
        if let Some(message) = Self::setup_block(setup_ready) {
            return Err(MeetingError::conflict(message));
        }
        if self.packet_count >= PACKET_LIMIT {
            return Err(MeetingError::backlog(messages::LIVE_LANGUAGE_BACKLOG));
        }
        let start_sample = (self.duration * speakerdesk_core::live::RATE as f64).round() as i64;
        let generation = revision + 1;
        self.apply_language(&language, generation, start_sample);
        self.packet_count += 1;
        self.epoch = generation;
        self.inbox
            .push(LiveCommand::Language {
                language,
                generation,
                start_sample,
            })
            .map_err(|_| {
                MeetingError::new(
                    StatusCode::INTERNAL_SERVER_ERROR,
                    messages::LOCAL_OPERATION_FAILED,
                )
            })?;
        self.language_body()
    }

    /// `POST /api/jobs/{jid}/refinement/{action}`.
    pub fn refinement(
        &mut self,
        jid: &str,
        action: &str,
        observed: RefineObservation,
        folder: &Path,
    ) -> Result<Job, MeetingError> {
        match action {
            "pause" => self.pause_refinement(jid),
            "resume" => self.resume_refinement(jid, observed, folder),
            _ => Err(MeetingError::missing()),
        }
    }

    fn pause_refinement(&mut self, jid: &str) -> Result<Job, MeetingError> {
        if self
            .active
            .as_ref()
            .is_none_or(|job| job.id.as_str() != jid)
        {
            return Err(MeetingError::conflict(messages::NOT_REFINING));
        }
        let status = self.active.as_ref().expect("meeting").status.clone();
        let operation = self.take_inflight();
        let job = self.active.as_mut().expect("meeting");
        job.extra
            .insert("refinement_status".to_string(), json!("paused"));
        if let Some(plan) = job
            .extra
            .get_mut("rolling_refinement")
            .and_then(Value::as_object_mut)
        {
            plan.insert("cancelled".to_string(), Value::Bool(true));
        }
        job.updated = unix_now();
        let final_pause = !is_live(&status);
        if let Some(operation_id) = operation.filter(|id| !id.is_empty()) {
            self.inbox
                .push(LiveCommand::CancelRefinement { operation_id })
                .map_err(|_| {
                    MeetingError::new(
                        StatusCode::INTERNAL_SERVER_ERROR,
                        messages::LOCAL_OPERATION_FAILED,
                    )
                })?;
        }
        if final_pause && !self.shutdown_sent {
            self.inbox.push(LiveCommand::Shutdown).map_err(|_| {
                MeetingError::new(
                    StatusCode::INTERNAL_SERVER_ERROR,
                    messages::LOCAL_OPERATION_FAILED,
                )
            })?;
            self.shutdown_sent = true;
        }
        Ok(self.active.clone().expect("meeting"))
    }

    fn resume_refinement(
        &mut self,
        jid: &str,
        observed: RefineObservation,
        folder: &Path,
    ) -> Result<Job, MeetingError> {
        if self
            .active
            .as_ref()
            .is_some_and(|job| job.id.as_str() == jid)
            && self.worker_alive
            && self.two_pass
        {
            return self.mark_refinement_waiting();
        }
        if self.active.is_some()
            || self.capture_alive
            || self.worker_alive
            || Self::inference_busy_block(observed.import_running).is_some()
        {
            return Err(MeetingError::conflict(messages::WAIT_FOR_LOCAL_INFERENCE));
        }
        let id = JobId::parse(jid).map_err(|_| MeetingError::missing())?;
        if !folder.join(jid).join("audio.wav").is_file() {
            return Err(MeetingError::conflict(messages::NO_SAVED_AUDIO));
        }
        if let Some(message) = Self::refine_setup_block(observed.setup_ready) {
            return Err(MeetingError::conflict(message));
        }
        let mut job = meeting_job(id, "Meeting", languages::AUTOMATIC, &[]);
        job.status = "ready".to_string();
        job.message = "Refining saved passages…".to_string();
        job.extra
            .insert("refinement_status".to_string(), json!("waiting"));
        self.reset_runtime();
        self.worker_alive = true;
        self.two_pass = true;
        self.active = Some(job.clone());
        Ok(job)
    }

    fn mark_refinement_waiting(&mut self) -> Result<Job, MeetingError> {
        let job = self.active.as_mut().expect("meeting");
        job.extra
            .insert("refinement_status".to_string(), json!("waiting"));
        if let Some(plan) = job
            .extra
            .get_mut("rolling_refinement")
            .and_then(Value::as_object_mut)
        {
            plan.insert("cancelled".to_string(), Value::Bool(false));
        }
        job.updated = unix_now();
        Ok(job.clone())
    }

    fn acknowledge_pause(&mut self, phase: &str) -> Option<String> {
        if phase != "paused" {
            return None;
        }
        let two_pass = self.two_pass;
        let through = self.received_samples;
        let job = self.active.as_mut()?;
        let flush = job.extra.get_mut("pause_flush")?.as_object_mut()?;
        if flush.get("state").and_then(Value::as_str) != Some("awaiting_capture") {
            return None;
        }
        let request_id = flush
            .get("request_id")
            .and_then(Value::as_str)
            .unwrap_or_default()
            .to_string();
        let state = if two_pass {
            "pending"
        } else {
            "capture_complete"
        };
        flush.insert("state".to_string(), json!(state));
        flush.insert("through_sample".to_string(), json!(through));
        two_pass.then_some(request_id)
    }

    fn apply_language(&mut self, language: &str, generation: i64, start_sample: i64) {
        let job = self.active.as_mut().expect("meeting");
        job.language = language.to_string();
        job.revision += 1;
        job.updated = unix_now();
        job.extra
            .insert("language_revision".to_string(), json!(generation));
        job.extra
            .insert("language_epoch".to_string(), json!(generation));
        let mut history = job
            .extra
            .get("language_history")
            .and_then(Value::as_array)
            .cloned()
            .unwrap_or_default();
        history.push(json!({
            "epoch": generation,
            "generation": generation,
            "language": language,
            "start_sample": start_sample,
        }));
        job.extra.insert(
            "language_history".to_string(),
            Value::Array(history.clone()),
        );
        if let Some(provenance) = job
            .document
            .as_mut()
            .and_then(|document| document.provenance.as_mut())
            .and_then(Value::as_object_mut)
        {
            provenance.insert("language".to_string(), json!("per_window"));
            provenance.insert("language_history".to_string(), Value::Array(history));
            if language == languages::AUTOMATIC {
                provenance.insert(
                    "language_detector".to_string(),
                    Value::String(LID_CHECKPOINT.to_string()),
                );
            }
        }
        self.default_language = language.to_string();
    }

    fn language_body(&self) -> Result<Value, MeetingError> {
        let mut value =
            serde_json::to_value(self.active.as_ref().expect("meeting")).map_err(|_| {
                MeetingError::new(
                    StatusCode::INTERNAL_SERVER_ERROR,
                    messages::LOCAL_OPERATION_FAILED,
                )
            })?;
        value["default_language"] = json!(self.default_language);
        Ok(value)
    }

    fn finish_before_capture(&mut self) -> Result<Job, MeetingError> {
        self.stopped = true;
        let job = self.active.as_mut().expect("meeting");
        job.status = "failed".to_string();
        job.message = messages::MEETING_STOPPED_BEFORE_CAPTURE.to_string();
        job.updated = unix_now();
        let job = job.clone();
        self.active = None;
        self.worker_alive = false;
        self.capture_alive = false;
        Ok(job)
    }

    fn stop_capture(&mut self) -> Result<Job, MeetingError> {
        self.stopped = true;
        self.capture_log.push(Command::Stop);
        self.inbox.push(LiveCommand::Stop).map_err(|_| {
            MeetingError::new(
                StatusCode::INTERNAL_SERVER_ERROR,
                messages::LOCAL_OPERATION_FAILED,
            )
        })?;
        let job = self.active.as_mut().expect("meeting");
        job.status = "finishing".to_string();
        job.message = "Finishing transcript…".to_string();
        job.updated = unix_now();
        Ok(job.clone())
    }

    fn pause_capture(&mut self) -> Result<Job, MeetingError> {
        let awaiting = self
            .active
            .as_ref()
            .and_then(|job| job.extra.get("pause_flush"))
            .and_then(|flush| flush.get("state"))
            .and_then(Value::as_str)
            == Some("awaiting_capture");
        if awaiting {
            return Ok(self.active.clone().expect("meeting"));
        }
        let request_id = random_hex(16);
        let job = self.active.as_mut().expect("meeting");
        job.extra.insert(
            "pause_flush".to_string(),
            json!({
                "request_id": request_id,
                "state": "awaiting_capture",
            }),
        );
        job.updated = unix_now();
        self.capture_log.push(Command::Pause);
        Ok(self.active.clone().expect("meeting"))
    }

    fn resume_capture(&mut self) -> Result<Job, MeetingError> {
        self.capture_log.push(Command::Resume);
        Ok(self.active.clone().expect("meeting"))
    }

    fn take_inflight(&mut self) -> Option<String> {
        let job = self.active.as_mut()?;
        let inflight = job.extra.remove("rolling_inflight")?;
        inflight
            .get("operation_id")
            .and_then(Value::as_str)
            .map(str::to_string)
    }

    fn open_meeting(
        &mut self,
        name: &str,
        language: &str,
        sources: &[String],
        folder: &Path,
    ) -> Result<Job, MeetingError> {
        let id = new_job_id().map_err(|_| {
            MeetingError::new(
                StatusCode::INTERNAL_SERVER_ERROR,
                messages::LOCAL_OPERATION_FAILED,
            )
        })?;
        let job = meeting_job(id, name, language, sources);
        let dest = folder.join(job.id.as_str());
        if let Err(error) = fs::create_dir_all(&dest) {
            tracing::error!(%error, "could not create a meeting folder");
            return Err(MeetingError::new(
                StatusCode::INTERNAL_SERVER_ERROR,
                messages::LOCAL_OPERATION_FAILED,
            ));
        }
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            let _ = fs::set_permissions(&dest, fs::Permissions::from_mode(0o700));
        }
        self.reset_runtime();
        self.default_language = language.to_string();
        self.active = Some(job.clone());
        Ok(job)
    }

    fn reset_runtime(&mut self) {
        self.inbox = Inbox::new(1, 0).expect("live inbox bounds");
        self.packet_count = 0;
        self.received_samples = 0;
        self.duration = 0.0;
        self.processed = 0.0;
        self.levels.clear();
        self.error = None;
        self.capture_started = false;
        self.capture_alive = false;
        self.worker_alive = false;
        self.two_pass = false;
        self.stopped = false;
        self.shutdown_sent = false;
        self.capture_log.clear();
        self.epoch = 0;
    }

    fn runtime_owned(&self) -> bool {
        self.capture_alive
            || self.worker_alive
            || self.active.as_ref().is_some_and(|job| is_live(&job.status))
    }
}

/// Meeting routes. Keep this signature; the job router merges it.
pub fn routes() -> axum::Router<crate::routes::AppState> {
    axum::Router::new()
        .route("/api/meeting", get(meeting_status))
        .route("/api/meetings", post(start_meeting))
        .route("/api/meetings/{jid}/{action}", post(control_meeting))
        .route(
            "/api/meetings/{jid}/language",
            patch(change_meeting_language),
        )
        .route(
            "/api/jobs/{jid}/refinement/{action}",
            post(refinement_action),
        )
}

fn manager_for(app: &AppState) -> Arc<Mutex<MeetingManager>> {
    app.meeting_manager()
}

fn with_manager<T>(app: &AppState, body: impl FnOnce(&mut MeetingManager) -> T) -> T {
    let manager = manager_for(app);
    let mut guard = manager.lock().expect("meeting manager");
    body(&mut guard)
}

async fn meeting_status(State(app): State<AppState>) -> Response {
    Json(with_manager(&app, |manager| manager.status())).into_response()
}

async fn start_meeting(State(app): State<AppState>, body: Bytes) -> Response {
    let request = match parse_start_body(&body) {
        Ok(request) => request,
        Err(error) => return error.into_response(),
    };
    let free_bytes = crate::setup::directory_free_bytes(app.data_dir()).unwrap_or(0);
    let result = with_manager(&app, |manager| {
        manager.start(
            &request,
            MeetingManager::host_gates(free_bytes),
            app.data_dir(),
        )
    });
    match result {
        Ok(job) => {
            remember(&app, &job);
            (StatusCode::CREATED, Json(job_value(&job))).into_response()
        }
        Err(error) => error.into_response(),
    }
}

async fn control_meeting(
    State(app): State<AppState>,
    AxumPath((jid, action)): AxumPath<(String, String)>,
) -> Response {
    let result = with_manager(&app, |manager| manager.control(&jid, &action));
    match result {
        Ok(job) => {
            remember(&app, &job);
            (StatusCode::ACCEPTED, Json(job_value(&job))).into_response()
        }
        Err(error) => error.into_response(),
    }
}

async fn change_meeting_language(
    State(app): State<AppState>,
    AxumPath(jid): AxumPath<String>,
    body: Bytes,
) -> Response {
    let parsed = match parse_language_body(&body) {
        Ok(parsed) => parsed,
        Err(error) => return error.into_response(),
    };
    let result = with_manager(&app, |manager| {
        let result = manager.change_language(&jid, &parsed, None);
        if result.is_ok()
            && let Some(job) = manager.active.clone()
        {
            remember(&app, &job);
        }
        result
    });
    match result {
        Ok(value) => Json(value).into_response(),
        Err(error) => error.into_response(),
    }
}

async fn refinement_action(
    State(app): State<AppState>,
    AxumPath((jid, action)): AxumPath<(String, String)>,
) -> Response {
    let result = with_manager(&app, |manager| {
        manager.refinement(
            &jid,
            &action,
            MeetingManager::host_refine_gates(),
            app.data_dir(),
        )
    });
    match result {
        Ok(job) => {
            remember(&app, &job);
            (StatusCode::ACCEPTED, Json(job_value(&job))).into_response()
        }
        Err(error) => error.into_response(),
    }
}

fn remember(app: &AppState, job: &Job) {
    app.insert_job(job.clone());
    if is_live(&job.status) {
        app.set_meeting(Some(job.id.clone()));
    } else {
        app.set_meeting(None);
    }
}

fn parse_start_body(bytes: &[u8]) -> Result<StartRequest, MeetingError> {
    let value = parse_json(bytes)?;
    let Some(body) = value.as_object() else {
        return Err(MeetingError::bad(messages::MEETING_SOURCES));
    };
    Ok(StartRequest {
        name: supplied_string(body.get("name")),
        language: supplied_string(body.get("language")),
        sources: source_list(body.get("sources"))?,
    })
}

fn parse_language_body(bytes: &[u8]) -> Result<Map<String, Value>, MeetingError> {
    if bytes.is_empty() {
        return Err(MeetingError::bad(messages::CHOOSE_LIVE_LANGUAGE));
    }
    let value = parse_json(bytes)?;
    value
        .as_object()
        .cloned()
        .ok_or_else(|| MeetingError::bad(messages::CHOOSE_LIVE_LANGUAGE))
}

fn parse_json(bytes: &[u8]) -> Result<Value, MeetingError> {
    if bytes.is_empty() {
        return Err(MeetingError::bad(messages::INVALID_REQUEST));
    }
    serde_json::from_slice(bytes).map_err(|_| MeetingError::bad(messages::INVALID_REQUEST))
}

fn supplied_string(value: Option<&Value>) -> Supplied {
    match value {
        None => Supplied::Missing,
        Some(Value::String(text)) => Supplied::Value(text.clone()),
        Some(_) => Supplied::Invalid,
    }
}

fn source_list(value: Option<&Value>) -> Result<Vec<String>, MeetingError> {
    let Some(Value::Array(items)) = value else {
        return Err(MeetingError::bad(messages::MEETING_SOURCES));
    };
    let mut sources = Vec::with_capacity(items.len());
    for item in items {
        let Some(text) = item.as_str() else {
            return Err(MeetingError::bad(messages::MEETING_SOURCES));
        };
        sources.push(text.to_string());
    }
    if sources.iter().collect::<HashSet<_>>().len() != sources.len() {
        return Err(MeetingError::bad(messages::MEETING_SOURCES));
    }
    Ok(sources)
}

fn live_language_request(body: &Map<String, Value>) -> Result<(String, i64), MeetingError> {
    let Some(language) = body.get("language").and_then(Value::as_str) else {
        return Err(MeetingError::bad(messages::LIVE_LANGUAGE_REVISION));
    };
    if !language_choice(language) {
        return Err(MeetingError::bad(messages::LIVE_LANGUAGE_REVISION));
    }
    let Some(revision) = body.get("language_revision").and_then(Value::as_i64) else {
        return Err(MeetingError::bad(messages::LIVE_LANGUAGE_REVISION));
    };
    if revision < 0 {
        return Err(MeetingError::bad(messages::LIVE_LANGUAGE_REVISION));
    }
    Ok((language.to_string(), revision))
}

fn valid_title(name: &str) -> bool {
    !name.trim().is_empty() && name.chars().count() <= TITLE_MAX_CHARS
}

fn language_choice(language: &str) -> bool {
    language == languages::AUTOMATIC || languages::is_supported(language)
}

fn valid_routes(sources: &[String]) -> bool {
    !sources.is_empty()
        && sources
            .iter()
            .all(|source| source == "microphone" || source == "system")
}

fn both_sources(sources: &[String]) -> bool {
    let unique: HashSet<&str> = sources.iter().map(String::as_str).collect();
    unique.contains("microphone") && unique.contains("system")
}

fn is_live(status: &str) -> bool {
    LIVE_STATUSES.contains(&status)
}

fn number_field(job: &Job, key: &str) -> i64 {
    job.extra.get(key).and_then(Value::as_i64).unwrap_or(0)
}

fn last_language_sample(job: &Job) -> i64 {
    job.extra
        .get("language_history")
        .and_then(Value::as_array)
        .and_then(|history| history.last())
        .and_then(|epoch| epoch.get("start_sample"))
        .and_then(Value::as_i64)
        .unwrap_or(0)
}

fn meeting_job(id: JobId, name: &str, language: &str, sources: &[String]) -> Job {
    let now = unix_now();
    let history = json!([{
        "epoch": 0,
        "generation": 0,
        "language": language,
        "start_sample": 0,
    }]);
    let mut warnings =
        vec!["Phrase boundaries may cut words. Overlapping speech needs review.".to_string()];
    if language == languages::AUTOMATIC {
        warnings.push(
            "Uncertain language uses recent established context or a marked supported-language guess when available. Choose the meeting language if Auto has no supported context. Original audio is preserved."
                .to_string(),
        );
    }
    let detector = if language == languages::AUTOMATIC {
        Value::String(LID_CHECKPOINT.to_string())
    } else {
        Value::Null
    };
    let document = Document {
        schema_version: 1,
        speakers: std::collections::BTreeMap::from([(
            "speaker_0".to_string(),
            "Speaker 1".to_string(),
        )]),
        segments: Vec::new(),
        provenance: Some(json!({
            "kind": "pending_inference",
            "mode": "live_phrase_windows",
            "language": language,
            "language_detector": detector,
            "models": ["nvidia/Nemotron-3-Diarization", "CohereLabs/cohere-transcribe-03-2026"],
            "timing": "Retained audio regions with NVIDIA speaker labels and Cohere phrase text. No word timestamps.",
            "language_history": history,
        })),
        warnings,
        diarization: Vec::new(),
        edited_at: None,
        rest: Map::new(),
    };
    let mut extra = Map::new();
    extra.insert("sources".to_string(), json!(sources));
    extra.insert("language_revision".to_string(), json!(0));
    extra.insert("language_acknowledged_revision".to_string(), json!(0));
    extra.insert("language_epoch".to_string(), json!(0));
    extra.insert("language_history".to_string(), history);
    extra.insert("refinement_status".to_string(), json!("waiting"));
    extra.insert("refinement_history".to_string(), json!({}));
    extra.insert("fast_history".to_string(), json!({}));
    extra.insert("refinement_unresolved".to_string(), json!([]));
    extra.insert("rolling_sources".to_string(), json!({}));
    extra.insert(
        "rolling_refinement".to_string(),
        json!({
            "completed_sample": 0,
            "pending": [],
            "cancelled": false,
        }),
    );
    Job {
        id,
        name: name.to_string(),
        kind: JobKind::Meeting,
        status: "starting".to_string(),
        message: "Starting meeting…".to_string(),
        language: language.to_string(),
        source_file: Some("audio.wav".to_string()),
        duration: Some(0.0),
        revision: 0,
        created: now,
        updated: now,
        document: Some(document),
        extra,
    }
}

fn job_value(job: &Job) -> Value {
    serde_json::to_value(job)
        .unwrap_or_else(|_| json!({ "error": messages::LOCAL_OPERATION_FAILED }))
}

fn unix_now() -> f64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|duration| duration.as_secs_f64())
        .unwrap_or(0.0)
}

fn new_job_id() -> std::io::Result<JobId> {
    let hex = random_hex(16);
    JobId::parse(hex)
        .map_err(|error| std::io::Error::new(std::io::ErrorKind::InvalidData, error.to_string()))
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
    use std::time::Duration;

    use serde_json::{Value, json};
    use speakerdesk_core::live::LiveCommand;
    use speakerdesk_core::{languages, limits, messages};

    use axum::http::StatusCode;

    use super::{
        MeetingError, MeetingManager, RefineObservation, StartObservation, StartRequest, Supplied,
    };
    use crate::capture::Command;
    use crate::routes::{AppState, router};
    use crate::setup::directory_free_bytes;

    const JOB: &str = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa";

    fn ready(free_bytes: u64) -> StartObservation {
        StartObservation {
            import_running: Some(false),
            setup_ready: Some(true),
            capture_available: true,
            free_bytes,
        }
    }

    fn mic(name: &str, language: &str) -> StartRequest {
        StartRequest {
            name: Supplied::Value(name.to_string()),
            language: Supplied::Value(language.to_string()),
            sources: vec!["microphone".to_string()],
        }
    }

    fn both() -> StartRequest {
        StartRequest {
            sources: vec!["microphone".to_string(), "system".to_string()],
            ..mic("Meeting", "en")
        }
    }

    struct TempDir(std::path::PathBuf);

    impl TempDir {
        fn new() -> Self {
            let path = std::env::temp_dir().join(format!(
                "speakerdesk-meeting-{}-{}",
                std::process::id(),
                random_suffix()
            ));
            std::fs::create_dir_all(&path).unwrap();
            Self(path)
        }
    }

    impl Drop for TempDir {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.0);
        }
    }

    fn random_suffix() -> String {
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos()
            .to_string()
    }

    fn message(error: MeetingError) -> &'static str {
        error.message
    }

    fn open_recording(
        manager: &mut MeetingManager,
        folder: &std::path::Path,
        two_pass: bool,
    ) -> String {
        let job = manager
            .start(
                &mic("Standup", "en"),
                ready(limits::MEETING_FREE_BYTES),
                folder,
            )
            .unwrap();
        manager.note_worker_ready(two_pass).unwrap();
        manager.note_capture_phase("recording").unwrap();
        job.id.as_str().to_string()
    }

    #[test]
    fn unobserved_gates_return_the_frozen_messages() {
        assert_eq!(limits::MEETING_FREE_BYTES, 512 * 1024 * 1024);
        assert_eq!(
            MeetingManager::import_block(None),
            Some(messages::MEETING_WAIT_IMPORT)
        );
        assert_eq!(
            MeetingManager::import_block(Some(true)),
            Some(messages::MEETING_WAIT_IMPORT)
        );
        assert_eq!(MeetingManager::import_block(Some(false)), None);

        assert_eq!(
            MeetingManager::setup_block(None),
            Some(messages::OPEN_SETTINGS_SETUP)
        );
        assert_eq!(
            MeetingManager::setup_block(Some(false)),
            Some(messages::OPEN_SETTINGS_SETUP)
        );
        assert_eq!(MeetingManager::setup_block(Some(true)), None);

        assert_eq!(
            MeetingManager::free_space_block(limits::MEETING_FREE_BYTES - 1),
            Some(messages::MEETING_FREE_SPACE)
        );
        assert_eq!(
            MeetingManager::free_space_block(limits::MEETING_FREE_BYTES),
            None
        );
        assert_eq!(
            MeetingManager::inference_busy_block(None),
            Some(messages::WAIT_FOR_LOCAL_INFERENCE)
        );
        assert_eq!(
            MeetingManager::refine_setup_block(None),
            Some(messages::SETUP_BEFORE_REFINE)
        );
        assert_eq!(MeetingManager::refine_setup_block(Some(true)), None);

        let hidden = MeetingManager::host_gates(0);
        assert_eq!(hidden.import_running, None);
        assert_eq!(hidden.setup_ready, None);
        assert!(!hidden.capture_available);
        assert_eq!(
            MeetingManager::free_space_block(hidden.free_bytes),
            Some(messages::MEETING_FREE_SPACE)
        );
        assert_eq!(
            MeetingManager::host_refine_gates(),
            RefineObservation {
                import_running: None,
                setup_ready: None,
            }
        );
    }

    #[test]
    fn host_gates_carry_the_directory_free_byte_count() {
        let state = AppState::new();
        let free = directory_free_bytes(state.data_dir()).unwrap();
        let gates = MeetingManager::host_gates(free);
        assert_eq!(gates.free_bytes, free);
        assert!(free >= limits::MEETING_FREE_BYTES);
        assert_eq!(MeetingManager::free_space_block(gates.free_bytes), None);
        let _ = std::fs::remove_dir_all(state.data_dir());
    }

    #[test]
    fn start_checks_follow_the_v0_6_6_order() {
        let folder = TempDir::new();
        let mut manager = MeetingManager::new();

        let hidden = manager
            .start(
                &mic("", "en"),
                MeetingManager::host_gates(limits::MEETING_FREE_BYTES),
                &folder.0,
            )
            .unwrap_err();
        assert_eq!(message(hidden), messages::MEETING_WAIT_IMPORT);

        let busy = ready(limits::MEETING_FREE_BYTES);
        let mut busy_import = busy;
        busy_import.import_running = Some(true);
        assert_eq!(
            message(
                manager
                    .start(&mic("", "nope"), busy_import, &folder.0)
                    .unwrap_err()
            ),
            messages::MEETING_WAIT_IMPORT
        );

        assert_eq!(
            message(
                manager
                    .start(
                        &mic("  ", "en"),
                        ready(limits::MEETING_FREE_BYTES),
                        &folder.0
                    )
                    .unwrap_err()
            ),
            messages::MEETING_TITLE
        );
        assert_eq!(
            message(
                manager
                    .start(
                        &mic(&"n".repeat(161), "en"),
                        ready(limits::MEETING_FREE_BYTES),
                        &folder.0
                    )
                    .unwrap_err()
            ),
            messages::MEETING_TITLE
        );
        assert_eq!(
            message(
                manager
                    .start(
                        &mic("Standup", "xx"),
                        ready(limits::MEETING_FREE_BYTES),
                        &folder.0
                    )
                    .unwrap_err()
            ),
            messages::MEETING_TITLE
        );
        let numbered = StartRequest {
            name: Supplied::Invalid,
            ..mic("Standup", "en")
        };
        assert_eq!(
            message(
                manager
                    .start(&numbered, ready(limits::MEETING_FREE_BYTES), &folder.0)
                    .unwrap_err()
            ),
            messages::MEETING_TITLE
        );

        let unknown = StartRequest {
            sources: vec!["headset".to_string()],
            ..mic("Standup", "en")
        };
        assert_eq!(
            message(
                manager
                    .start(&unknown, ready(limits::MEETING_FREE_BYTES), &folder.0)
                    .unwrap_err()
            ),
            messages::MEETING_ROUTE
        );
        let empty = StartRequest {
            sources: Vec::new(),
            ..mic("Standup", "en")
        };
        assert_eq!(
            message(
                manager
                    .start(&empty, ready(limits::MEETING_FREE_BYTES), &folder.0)
                    .unwrap_err()
            ),
            messages::MEETING_ROUTE
        );

        let mut no_capture = ready(limits::MEETING_FREE_BYTES);
        no_capture.capture_available = false;
        assert_eq!(
            message(
                manager
                    .start(&mic("Standup", "en"), no_capture, &folder.0)
                    .unwrap_err()
            ),
            messages::LIVE_CAPTURE_MISSING
        );
        assert_eq!(
            message(manager.start(&both(), no_capture, &folder.0).unwrap_err()),
            messages::LIVE_CAPTURE_MISSING
        );
        assert_eq!(
            message(
                manager
                    .start(&both(), ready(limits::MEETING_FREE_BYTES), &folder.0)
                    .unwrap_err()
            ),
            messages::ECHO_MISSING
        );

        let mut unready = ready(limits::MEETING_FREE_BYTES);
        unready.setup_ready = None;
        assert_eq!(
            message(
                manager
                    .start(&mic("Standup", "en"), unready, &folder.0)
                    .unwrap_err()
            ),
            messages::OPEN_SETTINGS_SETUP
        );
        unready.setup_ready = Some(false);
        assert_eq!(
            message(
                manager
                    .start(
                        &mic("Standup", "en"),
                        StartObservation {
                            free_bytes: 0,
                            ..unready
                        },
                        &folder.0
                    )
                    .unwrap_err()
            ),
            messages::OPEN_SETTINGS_SETUP
        );

        assert_eq!(
            message(
                manager
                    .start(
                        &mic("Standup", "en"),
                        ready(limits::MEETING_FREE_BYTES - 1),
                        &folder.0
                    )
                    .unwrap_err()
            ),
            messages::MEETING_FREE_SPACE
        );
        assert!(std::fs::read_dir(&folder.0).unwrap().next().is_none());

        let job = manager
            .start(
                &mic("  Standup  ", languages::AUTOMATIC),
                ready(limits::MEETING_FREE_BYTES),
                &folder.0,
            )
            .unwrap();
        assert_eq!(job.status, "starting");
        assert_eq!(job.message, "Starting meeting…");
        assert_eq!(job.name, "Standup");
        assert_eq!(job.language, languages::AUTOMATIC);
        assert_eq!(job.extra["sources"], json!(["microphone"]));
        assert_eq!(job.extra["refinement_status"], "waiting");
        assert!(folder.0.join(job.id.as_str()).is_dir());
        assert_eq!(manager.status()["status"], "starting");
        assert_eq!(manager.status()["capture_available"], false);
        assert!(manager.capture_commands().is_empty());

        let again = manager
            .start(&mic("", "en"), ready(limits::MEETING_FREE_BYTES), &folder.0)
            .unwrap_err();
        assert_eq!(message(again), messages::MEETING_ACTIVE);
        assert_eq!(again.status, StatusCode::CONFLICT);
    }

    #[test]
    fn a_missing_meeting_folder_does_not_stick() {
        let folder = TempDir::new();
        let blocker = folder.0.join("blocked");
        std::fs::write(&blocker, b"x").unwrap();
        let mut manager = MeetingManager::new();
        let error = manager
            .start(
                &mic("Standup", "en"),
                ready(limits::MEETING_FREE_BYTES),
                &blocker,
            )
            .unwrap_err();
        assert_eq!(error.status, StatusCode::INTERNAL_SERVER_ERROR);
        assert_eq!(message(error), messages::LOCAL_OPERATION_FAILED);
        assert_eq!(manager.status()["status"], "idle");
    }

    #[test]
    fn control_pause_and_stop_use_the_live_inbox_without_a_helper_process() {
        let folder = TempDir::new();
        let mut manager = MeetingManager::new();
        let id = open_recording(&mut manager, &folder.0, true);
        manager.retain_audio(1_600).unwrap();

        let early = MeetingManager::new().control(&id, "pause").unwrap_err();
        assert_eq!(message(early), messages::MEETING_INACTIVE);

        let unknown = manager.control(&id, "nope").unwrap_err();
        assert_eq!(unknown.status, StatusCode::NOT_FOUND);

        let starting = {
            let mut fresh = MeetingManager::new();
            let job = fresh
                .start(
                    &mic("Standup", "en"),
                    ready(limits::MEETING_FREE_BYTES),
                    &folder.0,
                )
                .unwrap();
            let paused = fresh.control(job.id.as_str(), "pause").unwrap_err();
            assert_eq!(message(paused), messages::MEETING_NOT_READY);
            fresh.control(job.id.as_str(), "stop").unwrap()
        };
        assert_eq!(starting.status, "failed");
        assert_eq!(starting.message, messages::MEETING_STOPPED_BEFORE_CAPTURE);

        let paused = manager.control(&id, "pause").unwrap();
        assert_eq!(paused.status, "recording");
        assert_eq!(paused.extra["pause_flush"]["state"], "awaiting_capture");
        assert_eq!(manager.capture_commands(), &[Command::Pause]);
        let duplicate = manager.control(&id, "pause").unwrap();
        assert_eq!(manager.capture_commands(), &[Command::Pause]);
        assert_eq!(duplicate.extra["pause_flush"]["state"], "awaiting_capture");

        let acknowledged = manager.note_capture_phase("paused").unwrap();
        assert_eq!(acknowledged.status, "paused");
        assert_eq!(acknowledged.message, "Paused");
        assert_eq!(acknowledged.extra["pause_flush"]["state"], "pending");
        assert!(matches!(
            manager.poll_worker().unwrap(),
            Some(LiveCommand::Audio(_))
        ));
        assert!(matches!(
            manager.poll_worker().unwrap(),
            Some(LiveCommand::Flush {
                through_sample: Some(1_600),
                ..
            })
        ));

        manager.control(&id, "resume").unwrap();
        assert_eq!(manager.capture_commands().last(), Some(&Command::Resume));
        manager.note_capture_phase("recording").unwrap();
        let stopped = manager.control(&id, "stop").unwrap();
        assert_eq!(stopped.status, "finishing");
        assert_eq!(stopped.message, "Finishing transcript…");
        assert!(matches!(
            manager.poll_worker().unwrap(),
            Some(LiveCommand::Stop)
        ));
    }

    #[test]
    fn a_dead_capture_reports_the_frozen_stop_message() {
        let folder = TempDir::new();
        let mut manager = MeetingManager::new();
        let id = open_recording(&mut manager, &folder.0, false);
        manager.capture_alive = false;
        let error = manager.control(&id, "pause").unwrap_err();
        assert_eq!(message(error), messages::CAPTURE_STOPPED);
        assert!(manager.stopped);
    }

    #[test]
    fn language_changes_enqueue_a_live_command_or_the_backlog_message() {
        let folder = TempDir::new();
        let mut manager = MeetingManager::new();
        let id = open_recording(&mut manager, &folder.0, true);
        manager.retain_audio(1_600).unwrap();

        let missing = super::parse_language_body(b"[]").unwrap_err();
        assert_eq!(message(missing), messages::CHOOSE_LIVE_LANGUAGE);

        let bad = manager
            .change_language(
                &id,
                &map_of(&json!({"language": "xx", "language_revision": 0})),
                Some(true),
            )
            .unwrap_err();
        assert_eq!(message(bad), messages::LIVE_LANGUAGE_REVISION);
        let negative = manager
            .change_language(
                &id,
                &map_of(&json!({"language": "fr", "language_revision": -1})),
                Some(true),
            )
            .unwrap_err();
        assert_eq!(message(negative), messages::LIVE_LANGUAGE_REVISION);

        let stale = manager
            .change_language(
                &id,
                &map_of(&json!({"language": "fr", "language_revision": 4})),
                Some(true),
            )
            .unwrap_err();
        assert_eq!(message(stale), messages::LIVE_LANGUAGE_STALE);

        let same = manager
            .change_language(
                &id,
                &map_of(&json!({"language": "en", "language_revision": 0})),
                Some(false),
            )
            .unwrap();
        assert_eq!(same["language_revision"], 0);
        assert_eq!(same["default_language"], "en");

        let unready = manager
            .change_language(
                &id,
                &map_of(&json!({"language": "fr", "language_revision": 0})),
                None,
            )
            .unwrap_err();
        assert_eq!(message(unready), messages::OPEN_SETTINGS_SETUP);

        for _ in 0..119 {
            manager.retain_audio(1).unwrap();
        }
        let blocked = manager
            .change_language(
                &id,
                &map_of(&json!({"language": "fr", "language_revision": 0})),
                Some(true),
            )
            .unwrap_err();
        assert_eq!(blocked.status, StatusCode::TOO_MANY_REQUESTS);
        assert_eq!(message(blocked), messages::LIVE_LANGUAGE_BACKLOG);
        assert_eq!(manager.status()["language"], "en");
        assert_eq!(manager.status()["language_revision"], 0);

        let mut open = MeetingManager::new();
        let id = open_recording(&mut open, &folder.0, true);
        let changed = open
            .change_language(
                &id,
                &map_of(&json!({"language": "fr", "language_revision": 0})),
                Some(true),
            )
            .unwrap();
        assert_eq!(changed["language"], "fr");
        assert_eq!(changed["language_revision"], 1);
        assert_eq!(changed["default_language"], "fr");
        assert_eq!(changed["language_history"][1]["start_sample"], 0);
        assert_eq!(open.default_language(), "fr");
        assert!(matches!(
            open.poll_worker().unwrap(),
            Some(LiveCommand::Language { language, generation: 1, start_sample: 0 }) if language == "fr"
        ));

        let idle = open
            .change_language(
                "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb",
                &map_of(&json!({"language": "en", "language_revision": 1})),
                Some(true),
            )
            .unwrap_err();
        assert_eq!(message(idle), messages::MEETING_INACTIVE);

        open.note_worker_ready(true).unwrap();
        let mut starting = MeetingManager::new();
        let starting_id = starting
            .start(
                &mic("Standup", "en"),
                ready(limits::MEETING_FREE_BYTES),
                &folder.0,
            )
            .unwrap()
            .id
            .as_str()
            .to_string();
        starting.note_worker_ready(true).unwrap();
        let too_soon = starting
            .change_language(
                &starting_id,
                &map_of(&json!({"language": "fr", "language_revision": 0})),
                Some(true),
            )
            .unwrap_err();
        assert_eq!(message(too_soon), messages::LANGUAGE_WHILE_RECORDING);
    }

    #[test]
    fn refinement_pause_cancels_the_live_command_and_resume_checks_audio() {
        let folder = TempDir::new();
        let mut manager = MeetingManager::new();
        let id = open_recording(&mut manager, &folder.0, true);
        manager.note_refinement("window-1").unwrap();

        let other = manager
            .refinement(JOB, "pause", MeetingManager::host_refine_gates(), &folder.0)
            .unwrap_err();
        assert_eq!(message(other), messages::NOT_REFINING);
        let unknown = manager
            .refinement(&id, "drop", MeetingManager::host_refine_gates(), &folder.0)
            .unwrap_err();
        assert_eq!(unknown.status, StatusCode::NOT_FOUND);

        let paused = manager
            .refinement(&id, "pause", MeetingManager::host_refine_gates(), &folder.0)
            .unwrap();
        assert_eq!(paused.extra["refinement_status"], "paused");
        assert!(paused.extra.get("rolling_inflight").is_none());
        assert_eq!(paused.extra["rolling_refinement"]["cancelled"], true);
        assert!(manager.refinement_cancelled("window-1"));

        let resumed = manager
            .refinement(
                &id,
                "resume",
                MeetingManager::host_refine_gates(),
                &folder.0,
            )
            .unwrap();
        assert_eq!(resumed.extra["refinement_status"], "waiting");
        assert_eq!(resumed.extra["rolling_refinement"]["cancelled"], false);

        let mut idle = MeetingManager::new();
        let waiting = idle
            .refinement(
                JOB,
                "resume",
                MeetingManager::host_refine_gates(),
                &folder.0,
            )
            .unwrap_err();
        assert_eq!(message(waiting), messages::WAIT_FOR_LOCAL_INFERENCE);

        let observed = RefineObservation {
            import_running: Some(false),
            setup_ready: Some(true),
        };
        let missing_id = idle
            .refinement("abc", "resume", observed, &folder.0)
            .unwrap_err();
        assert_eq!(missing_id.status, StatusCode::NOT_FOUND);
        let no_audio = idle
            .refinement(JOB, "resume", observed, &folder.0)
            .unwrap_err();
        assert_eq!(message(no_audio), messages::NO_SAVED_AUDIO);

        std::fs::create_dir_all(folder.0.join(JOB)).unwrap();
        std::fs::write(folder.0.join(JOB).join("audio.wav"), b"RIFF").unwrap();
        let unready = RefineObservation {
            import_running: Some(false),
            setup_ready: None,
        };
        let setup = idle
            .refinement(JOB, "resume", unready, &folder.0)
            .unwrap_err();
        assert_eq!(message(setup), messages::SETUP_BEFORE_REFINE);

        let saved = idle.refinement(JOB, "resume", observed, &folder.0).unwrap();
        assert_eq!(saved.status, "ready");
        assert_eq!(saved.message, "Refining saved passages…");
        assert_eq!(saved.id.as_str(), JOB);
        assert!(idle.worker_alive);
        let blocked = idle
            .start(
                &mic("Next", "en"),
                ready(limits::MEETING_FREE_BYTES),
                &folder.0,
            )
            .unwrap_err();
        assert_eq!(message(blocked), messages::MEETING_ACTIVE);
    }

    fn map_of(value: &Value) -> serde_json::Map<String, Value> {
        value.as_object().unwrap().clone()
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
                    .get(running.url("/api/meeting"))
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

    #[tokio::test]
    async fn meeting_routes_honor_frozen_errors_and_do_not_start_capture() {
        let state = AppState::new();
        let running = Running::spawn(state).await;

        let idle = running
            .client
            .get(running.url("/api/meeting"))
            .send()
            .await
            .unwrap();
        assert_eq!(idle.status(), 200);
        let idle: Value = idle.json().await.unwrap();
        assert_eq!(idle["status"], "idle");
        assert_eq!(idle["id"], Value::Null);
        assert_eq!(idle["sources"], json!([]));
        assert_eq!(idle["capture_available"], false);
        assert_eq!(idle["refinement_status"], "idle");
        assert_eq!(idle["language_revision"], 0);

        let anonymous = running
            .client
            .post(running.url("/api/meetings"))
            .json(&json!({"name": "Standup", "language": "en", "sources": ["microphone"]}))
            .send()
            .await
            .unwrap();
        assert_eq!(anonymous.status(), 403);
        assert_eq!(
            anonymous.json::<Value>().await.unwrap()["error"],
            messages::TOKEN_MISMATCH
        );

        for path in [
            "/api/meetings/aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa/pause",
            "/api/jobs/aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa/refinement/pause",
        ] {
            let blocked = running.client.post(running.url(path)).send().await.unwrap();
            assert_eq!(blocked.status(), 403, "{path}");
        }
        let blocked_language = running
            .client
            .patch(running.url("/api/meetings/aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa/language"))
            .json(&json!({"language": "fr", "language_revision": 0}))
            .send()
            .await
            .unwrap();
        assert_eq!(blocked_language.status(), 403);

        let duplicates = running
            .authed(running.client.post(running.url("/api/meetings")))
            .json(&json!({"name": "Standup", "language": "en", "sources": ["microphone", "microphone"]}))
            .send()
            .await
            .unwrap();
        assert_eq!(duplicates.status(), 400);
        assert_eq!(
            duplicates.json::<Value>().await.unwrap()["error"],
            messages::MEETING_SOURCES
        );

        let shaped = running
            .authed(running.client.post(running.url("/api/meetings")))
            .json(&json!({"name": "", "language": "xx", "sources": ["microphone"]}))
            .send()
            .await
            .unwrap();
        assert_eq!(shaped.status(), 409);
        assert_eq!(
            shaped.json::<Value>().await.unwrap()["error"],
            messages::MEETING_WAIT_IMPORT
        );
        let jobs = running
            .client
            .get(running.url("/api/jobs"))
            .send()
            .await
            .unwrap();
        assert_eq!(jobs.json::<Value>().await.unwrap(), json!([]));

        let id = {
            let manager = super::manager_for(&running.state);
            let mut manager = manager.lock().expect("meeting");
            let job = manager
                .start(
                    &mic("Standup", "en"),
                    ready(limits::MEETING_FREE_BYTES),
                    running.state.data_dir(),
                )
                .unwrap();
            job.id.as_str().to_string()
        };
        let active = running
            .authed(running.client.post(running.url("/api/meetings")))
            .json(&json!({"name": "Other", "language": "en", "sources": ["microphone"]}))
            .send()
            .await
            .unwrap();
        assert_eq!(active.status(), 409);
        assert_eq!(
            active.json::<Value>().await.unwrap()["error"],
            messages::MEETING_ACTIVE
        );
        let status = running
            .client
            .get(running.url("/api/meeting"))
            .send()
            .await
            .unwrap();
        let status: Value = status.json().await.unwrap();
        assert_eq!(status["id"], id);
        assert_eq!(status["status"], "starting");

        let not_action = running
            .authed(
                running
                    .client
                    .post(running.url(&format!("/api/meetings/{id}/drop"))),
            )
            .send()
            .await
            .unwrap();
        assert_eq!(not_action.status(), 404);
        let wrong_method = running
            .authed(
                running
                    .client
                    .post(running.url(&format!("/api/meetings/{id}/language"))),
            )
            .send()
            .await
            .unwrap();
        assert_eq!(wrong_method.status(), 405);

        let stopped = running
            .authed(
                running
                    .client
                    .post(running.url(&format!("/api/meetings/{id}/stop"))),
            )
            .send()
            .await
            .unwrap();
        assert_eq!(stopped.status(), 202);
        let stopped: Value = stopped.json().await.unwrap();
        assert_eq!(stopped["status"], "failed");
        assert_eq!(stopped["message"], messages::MEETING_STOPPED_BEFORE_CAPTURE);
        let idle = running
            .client
            .get(running.url("/api/meeting"))
            .send()
            .await
            .unwrap();
        assert_eq!(idle.json::<Value>().await.unwrap()["status"], "idle");
        let stored = running
            .client
            .get(running.url(&format!("/api/jobs/{id}")))
            .send()
            .await
            .unwrap();
        assert_eq!(stored.status(), 200);
        let stored: Value = stored.json().await.unwrap();
        assert_eq!(stored["message"], messages::MEETING_STOPPED_BEFORE_CAPTURE);
        assert!(stored.get("fast_history").is_none());

        running.shutdown().await;
    }

    #[tokio::test]
    async fn language_and_refinement_routes_refuse_without_a_live_worker() {
        let state = AppState::new();
        let running = Running::spawn(state).await;
        let id = {
            let manager = super::manager_for(&running.state);
            let mut manager = manager.lock().expect("meeting");
            open_recording(&mut manager, running.state.data_dir(), true)
        };

        let not_object = running
            .authed(
                running
                    .client
                    .patch(running.url(&format!("/api/meetings/{id}/language"))),
            )
            .json(&json!([]))
            .send()
            .await
            .unwrap();
        assert_eq!(not_object.status(), 400);
        assert_eq!(
            not_object.json::<Value>().await.unwrap()["error"],
            messages::CHOOSE_LIVE_LANGUAGE
        );

        let stale = running
            .authed(
                running
                    .client
                    .patch(running.url(&format!("/api/meetings/{id}/language"))),
            )
            .json(&json!({"language": "fr", "language_revision": 3}))
            .send()
            .await
            .unwrap();
        assert_eq!(stale.status(), 409);
        assert_eq!(
            stale.json::<Value>().await.unwrap()["error"],
            messages::LIVE_LANGUAGE_STALE
        );

        let hidden_setup = running
            .authed(
                running
                    .client
                    .patch(running.url(&format!("/api/meetings/{id}/language"))),
            )
            .json(&json!({"language": "fr", "language_revision": 0}))
            .send()
            .await
            .unwrap();
        assert_eq!(hidden_setup.status(), 409);
        assert_eq!(
            hidden_setup.json::<Value>().await.unwrap()["error"],
            messages::OPEN_SETTINGS_SETUP
        );

        let same = running
            .authed(
                running
                    .client
                    .patch(running.url(&format!("/api/meetings/{id}/language"))),
            )
            .json(&json!({"language": "en", "language_revision": 0}))
            .send()
            .await
            .unwrap();
        assert_eq!(same.status(), 200);
        let same: Value = same.json().await.unwrap();
        assert_eq!(same["language"], "en");
        assert_eq!(same["default_language"], "en");

        let not_refining = running
            .authed(
                running
                    .client
                    .post(running.url(&format!("/api/jobs/{JOB}/refinement/pause"))),
            )
            .send()
            .await
            .unwrap();
        assert_eq!(not_refining.status(), 409);
        assert_eq!(
            not_refining.json::<Value>().await.unwrap()["error"],
            messages::NOT_REFINING
        );

        let paused = running
            .authed(
                running
                    .client
                    .post(running.url(&format!("/api/jobs/{id}/refinement/pause"))),
            )
            .send()
            .await
            .unwrap();
        assert_eq!(paused.status(), 202);
        assert_eq!(
            paused.json::<Value>().await.unwrap()["refinement_status"],
            "paused"
        );

        let resumed = running
            .authed(
                running
                    .client
                    .post(running.url(&format!("/api/jobs/{id}/refinement/resume"))),
            )
            .send()
            .await
            .unwrap();
        assert_eq!(resumed.status(), 202);
        assert_eq!(
            resumed.json::<Value>().await.unwrap()["refinement_status"],
            "waiting"
        );

        let missing_action = running
            .authed(
                running
                    .client
                    .post(running.url(&format!("/api/jobs/{id}/refinement/drop"))),
            )
            .send()
            .await
            .unwrap();
        assert_eq!(missing_action.status(), 404);

        running.shutdown().await;

        let state = AppState::new();
        let running = Running::spawn(state).await;
        let waiting = running
            .authed(
                running
                    .client
                    .post(running.url(&format!("/api/jobs/{JOB}/refinement/resume"))),
            )
            .send()
            .await
            .unwrap();
        assert_eq!(waiting.status(), 409);
        assert_eq!(
            waiting.json::<Value>().await.unwrap()["error"],
            messages::WAIT_FOR_LOCAL_INFERENCE
        );
        running.shutdown().await;
    }
}
