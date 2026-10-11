//! People, recognition, and meeting-identity routes for v0.6.6.
//!
//! [`AppState`](crate::routes::AppState) has no people table. Names, voice
//! profiles, and the recognition preference stay in this module, keyed by the
//! state's data directory. Identity routes load job documents from
//! `identity-jobs/{id}.json` under that directory, then keep edits in memory.
//! Suggestion text comes from [`speakerdesk_core::people`].

use std::collections::{BTreeMap, BTreeSet};
use std::fs::File;
use std::io::Read;
use std::path::{Path, PathBuf};
use std::sync::{Mutex, OnceLock};
use std::time::{SystemTime, UNIX_EPOCH};

use axum::Json;
use axum::Router;
use axum::body::Bytes;
use axum::extract::{Path as AxumPath, State};
use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use axum::routing::{delete, get, patch, post};
use serde_json::{Map, Value, json};
use speakerdesk_core::ids::JobId;
use speakerdesk_core::{messages, people, voice};

use crate::routes::AppState;

const NOT_FOUND: &str = "The requested URL was not found on the server. If you entered the URL manually please check your spelling and try again.";
const NAMING_BUSY: &[&str] = &["preparing", "queued", "processing", "finishing"];
const VOICE_READY: &str =
    "Saved voices are recognized once when a new speaker appears. You can correct any name.";
const VOICE_UNQUALIFIED: &str = "Voice recognition needs a qualified GPU/ANE runtime. Saved names and voices are kept. Install a Speakerdesk update with qualified voice recognition.";
const PROFILE_KEPT: &str = "Your saved voice is kept. Voice recognition is currently unavailable.";

pub fn routes() -> Router<AppState> {
    Router::new()
        .route("/api/people", get(list_people).post(create_person))
        .route("/api/people/{pid}", patch(rename_person))
        .route("/api/people/{pid}/voice", delete(forget_voice))
        .route(
            "/api/recognition",
            get(recognition_status).patch(update_recognition),
        )
        .route(
            "/api/jobs/{jid}/identity-suggestions",
            get(identity_suggestions),
        )
        .route(
            "/api/jobs/{jid}/identity-suggestions/{sid}/dismiss",
            post(dismiss_suggestion),
        )
        .route(
            "/api/jobs/{jid}/speakers/{track}/identity",
            post(confirm_identity),
        )
}

#[derive(Debug)]
struct Catalog {
    people: people::People,
    voices: voice::VoiceBook,
    recognition_enabled: bool,
    jobs: BTreeMap<String, Value>,
}

impl Default for Catalog {
    fn default() -> Self {
        Self {
            people: people::People::default(),
            voices: voice::VoiceBook::default(),
            recognition_enabled: true,
            jobs: BTreeMap::new(),
        }
    }
}

fn catalogs() -> &'static Mutex<BTreeMap<PathBuf, Catalog>> {
    static CATALOGS: OnceLock<Mutex<BTreeMap<PathBuf, Catalog>>> = OnceLock::new();
    CATALOGS.get_or_init(|| Mutex::new(BTreeMap::new()))
}

fn with_catalog<T>(app: &AppState, mutate: impl FnOnce(&mut Catalog, &Path) -> T) -> T {
    let dir = app.data_dir().to_path_buf();
    let mut guard = catalogs().lock().expect("people");
    let catalog = guard.entry(dir.clone()).or_default();
    mutate(catalog, &dir)
}

fn job_file(dir: &Path, jid: &str) -> PathBuf {
    dir.join("identity-jobs").join(format!("{jid}.json"))
}

async fn list_people(State(app): State<AppState>) -> Response {
    Json(with_catalog(&app, |catalog, _dir| people_view(catalog))).into_response()
}

async fn create_person(State(app): State<AppState>, body: Bytes) -> Response {
    let body = match json_object(&body) {
        Ok(body) => body,
        Err(error) => return error.response(),
    };
    match with_catalog(&app, |catalog, _dir| add_person(catalog, &body)) {
        Ok(person) => (StatusCode::CREATED, Json(person)).into_response(),
        Err(error) => error.response(),
    }
}

async fn rename_person(
    State(app): State<AppState>,
    AxumPath(pid): AxumPath<String>,
    body: Bytes,
) -> Response {
    let body = match json_object(&body) {
        Ok(body) => body,
        Err(error) => return error.response(),
    };
    match with_catalog(&app, |catalog, _dir| rename_in(catalog, &pid, &body)) {
        Ok(person) => Json(person).into_response(),
        Err(error) => error.response(),
    }
}

async fn forget_voice(State(app): State<AppState>, AxumPath(pid): AxumPath<String>) -> Response {
    match with_catalog(&app, |catalog, _dir| forget_in(catalog, &pid)) {
        Ok(body) => Json(body).into_response(),
        Err(error) => error.response(),
    }
}

async fn recognition_status(State(app): State<AppState>) -> Response {
    let enabled = with_catalog(&app, |catalog, _dir| catalog.recognition_enabled);
    Json(json!({ "enabled": enabled })).into_response()
}

async fn update_recognition(State(app): State<AppState>, body: Bytes) -> Response {
    let body = match json_object(&body) {
        Ok(body) => body,
        Err(error) => return error.response(),
    };
    match with_catalog(&app, |catalog, dir| set_recognition(catalog, dir, &body)) {
        Ok(enabled) => Json(json!({ "enabled": enabled })).into_response(),
        Err(error) => error.response(),
    }
}

async fn identity_suggestions(
    State(app): State<AppState>,
    AxumPath(jid): AxumPath<String>,
) -> Response {
    match with_catalog(&app, |catalog, dir| suggestions_for(catalog, dir, &jid)) {
        Ok(body) => Json(body).into_response(),
        Err(error) => error.response(),
    }
}

async fn dismiss_suggestion(
    State(app): State<AppState>,
    AxumPath((jid, sid)): AxumPath<(String, String)>,
    body: Bytes,
) -> Response {
    let body = match json_object(&body) {
        Ok(body) => body,
        Err(error) => return error.response(),
    };
    match with_catalog(&app, |catalog, dir| {
        dismiss_in(catalog, dir, &jid, &sid, &body)
    }) {
        Ok(job) => Json(job).into_response(),
        Err(error) => error.response(),
    }
}

async fn confirm_identity(
    State(app): State<AppState>,
    AxumPath((jid, track)): AxumPath<(String, String)>,
    body: Bytes,
) -> Response {
    let body = match json_object(&body) {
        Ok(body) => body,
        Err(error) => return error.response(),
    };
    match with_catalog(&app, |catalog, dir| {
        confirm_in(catalog, dir, &jid, &track, &body)
    }) {
        Ok(job) => Json(job).into_response(),
        Err(error) => error.response(),
    }
}

fn people_view(catalog: &Catalog) -> Value {
    let available = voice::runtime_available(None, None, false);
    let saved: BTreeSet<&str> = catalog
        .voices
        .profiles()
        .iter()
        .map(|profile| profile.person_id.as_str())
        .collect();
    let mut rows: Vec<people::Person> = catalog.people.iter().cloned().collect();
    rows.sort_by(|left, right| {
        left.name
            .to_lowercase()
            .cmp(&right.name.to_lowercase())
            .then_with(|| left.id.cmp(&right.id))
    });
    let people = rows
        .iter()
        .map(|person| listed_person(person, saved.contains(person.id.as_str())))
        .collect::<Vec<_>>();
    let mut body = readiness_fields(available);
    body["people"] = json!(people);
    body
}

fn readiness_fields(available: bool) -> Value {
    if available {
        json!({
            "voice_available": true,
            "voice_message": VOICE_READY,
            "readiness_code": "runtime_ready",
            "next_action": Value::Null,
        })
    } else {
        json!({
            "voice_available": false,
            "voice_message": VOICE_UNQUALIFIED,
            "readiness_code": "runtime_unqualified",
            "next_action": "updates",
        })
    }
}

fn listed_person(person: &people::Person, saved: bool) -> Value {
    let mut value = person_json(person);
    let object = value.as_object_mut().expect("person object");
    object.insert("voice_saved".to_string(), json!(saved));
    object.insert("voice_compatible".to_string(), json!(false));
    if saved {
        object.insert(
            "voice_profile_state".to_string(),
            json!("stored_unavailable"),
        );
        object.insert("voice_profile_message".to_string(), json!(PROFILE_KEPT));
        object.insert("voice_profile_next_action".to_string(), Value::Null);
    } else {
        object.insert("voice_profile_state".to_string(), json!("none"));
        object.insert("voice_profile_message".to_string(), Value::Null);
        object.insert("voice_profile_next_action".to_string(), Value::Null);
    }
    value
}

fn person_json(person: &people::Person) -> Value {
    let mut value = json!({
        "id": person.id,
        "name": person.name,
        "created": person.created,
    });
    if let Some(updated) = person.updated {
        value["updated"] = json!(updated);
    }
    value
}

fn add_person(catalog: &mut Catalog, body: &Map<String, Value>) -> Result<Value, HttpError> {
    let name = name_text(body)?;
    match catalog.people.create(name, unix_now()) {
        Ok(person) => Ok(person_json(&person)),
        Err(error) => Err(from_people(error)),
    }
}

fn rename_in(
    catalog: &mut Catalog,
    pid: &str,
    body: &Map<String, Value>,
) -> Result<Value, HttpError> {
    let name = name_text(body)?;
    match catalog.people.rename(pid, name, unix_now()) {
        Ok(person) => Ok(person_json(&person)),
        Err(error) => Err(from_people(error)),
    }
}

fn forget_in(catalog: &mut Catalog, pid: &str) -> Result<Value, HttpError> {
    match catalog.voices.forget(&catalog.people, pid) {
        Ok(()) => Ok(json!({ "forgotten": true })),
        Err(voice::VoiceError::People(error)) => Err(from_people(error)),
        Err(error) => Err(HttpError::new(StatusCode::BAD_REQUEST, error.to_string())),
    }
}

fn name_text(body: &Map<String, Value>) -> Result<&str, HttpError> {
    match body.get("name") {
        Some(Value::String(name)) => Ok(name.as_str()),
        _ => Err(HttpError::new(
            StatusCode::BAD_REQUEST,
            messages::PERSON_NAME,
        )),
    }
}

fn set_recognition(
    catalog: &mut Catalog,
    dir: &Path,
    body: &Map<String, Value>,
) -> Result<bool, HttpError> {
    let enabled = match body.get("enabled") {
        Some(Value::Bool(enabled)) => *enabled,
        _ => {
            return Err(HttpError::new(
                StatusCode::BAD_REQUEST,
                messages::RECOGNITION_CHOICE,
            ));
        }
    };
    catalog.recognition_enabled = enabled;
    if !enabled {
        import_disk_jobs(catalog, dir);
        for job in catalog.jobs.values_mut() {
            cancel_open_checks(job);
        }
    }
    Ok(enabled)
}

fn cancel_open_checks(job: &mut Value) {
    let Some(checks) = job
        .get_mut("voice_checks")
        .and_then(|value| value.as_object_mut())
    else {
        return;
    };
    for check in checks.values_mut() {
        if check.get("status").and_then(|status| status.as_str()) == Some(voice::CHECK_CHECKING)
            && let Some(object) = check.as_object_mut()
        {
            object.insert("status".to_string(), json!(voice::CHECK_CANCELLED));
        }
    }
}

fn suggestions_for(catalog: &mut Catalog, dir: &Path, jid: &str) -> Result<Value, HttpError> {
    let job = load_job(catalog, dir, jid)?;
    let revision = job_revision(&job)?;
    let suggestions = listed_suggestions(catalog, &job);
    Ok(json!({ "revision": revision, "suggestions": suggestions }))
}

fn dismiss_in(
    catalog: &mut Catalog,
    dir: &Path,
    jid: &str,
    sid: &str,
    body: &Map<String, Value>,
) -> Result<Value, HttpError> {
    let mut job = load_job(catalog, dir, jid)?;
    editing_job(&job, body, None)?;
    let current = listed_suggestions(catalog, &job);
    let known = current
        .iter()
        .any(|item| item.get("id").and_then(|value| value.as_str()) == Some(sid));
    if !known {
        return Err(HttpError::new(
            StatusCode::CONFLICT,
            messages::SUGGESTION_STALE,
        ));
    }
    let mut dismissed = array_field(&job, "dismissed_identity_suggestions")?;
    dismissed.push(json!(sid));
    if dismissed.len() > 256 {
        let extra = dismissed.len() - 256;
        dismissed.drain(0..extra);
    }
    job["dismissed_identity_suggestions"] = json!(dismissed);
    let revision = job_revision(&job)?;
    job["revision"] = json!(revision + 1);
    catalog.jobs.insert(jid.to_string(), job.clone());
    Ok(job)
}

fn confirm_in(
    catalog: &mut Catalog,
    dir: &Path,
    jid: &str,
    track: &str,
    body: &Map<String, Value>,
) -> Result<Value, HttpError> {
    let mut job = load_job(catalog, dir, jid)?;
    editing_job(&job, body, Some(track))?;
    let suggestion = selected_suggestion(catalog, &job, body, track)?;
    let (person_id, name) = resolve_name(&catalog.people, body, suggestion.as_ref())?;
    if let Some(suggestion) = suggestion.as_ref() {
        reject_mismatch(suggestion, person_id.as_deref(), &name)?;
    }
    let source = suggestion
        .as_ref()
        .and_then(|item| item.get("kind"))
        .and_then(|value| value.as_str())
        .unwrap_or("manual");
    let evidence = suggestion
        .as_ref()
        .map(suggestion_evidence)
        .unwrap_or(Value::Null);
    let assignment = people::SpeakerAssignment {
        person_id: person_id.clone(),
        meeting_id: jid.to_string(),
        track_id: track.to_string(),
        name: name.clone(),
        confirmed: Some(true),
        source: source.to_string(),
    };
    set_speaker(&mut job, track, &name)?;
    {
        let assignments = ensure_object(&mut job, "speaker_assignments")?;
        assignments.insert(
            track.to_string(),
            assignment_json(&assignment, unix_now(), evidence),
        );
    }
    let kept = without_track(&job, track)?;
    job["voice_suggestions"] = json!(kept);
    {
        let checks = ensure_object(&mut job, "voice_checks")?;
        checks.insert(
            track.to_string(),
            json!({ "id": new_id(), "status": "manual" }),
        );
    }
    let revision = job_revision(&job)?;
    job["revision"] = json!(revision + 1);
    catalog.jobs.insert(jid.to_string(), job.clone());
    Ok(job)
}

fn selected_suggestion(
    catalog: &Catalog,
    job: &Value,
    body: &Map<String, Value>,
    track: &str,
) -> Result<Option<Value>, HttpError> {
    let Some(suggestion_id) = optional_text(body.get("suggestion_id")) else {
        return Ok(None);
    };
    let found = listed_suggestions(catalog, job).into_iter().find(|item| {
        item.get("id").and_then(|value| value.as_str()) == Some(suggestion_id)
            && item.get("track_id").and_then(|value| value.as_str()) == Some(track)
    });
    found
        .map(Some)
        .ok_or_else(|| HttpError::new(StatusCode::CONFLICT, messages::SUGGESTION_STALE))
}

fn resolve_name(
    book: &people::People,
    body: &Map<String, Value>,
    suggestion: Option<&Value>,
) -> Result<(Option<String>, String), HttpError> {
    if let Some(pid) = optional_text(body.get("person_id")) {
        let person = book.get(pid).map_err(from_people)?;
        return Ok((Some(pid.to_string()), person.name.clone()));
    }
    let raw = if let Some(suggestion) = suggestion {
        suggestion
            .get("name")
            .and_then(|value| value.as_str())
            .unwrap_or("")
    } else if let Some(Value::String(name)) = body.get("name") {
        name.as_str()
    } else {
        return Err(HttpError::new(
            StatusCode::BAD_REQUEST,
            messages::PERSON_NAME,
        ));
    };
    match people::person_name(raw) {
        Ok(name) => Ok((None, name)),
        Err(error) => Err(from_people(error)),
    }
}

fn reject_mismatch(
    suggestion: &Value,
    person_id: Option<&str>,
    name: &str,
) -> Result<(), HttpError> {
    let suggested_person = optional_text(suggestion.get("person_id"));
    let person_mismatch = suggested_person.is_some_and(|expected| Some(expected) != person_id);
    let suggested_name = suggestion
        .get("name")
        .and_then(|value| value.as_str())
        .unwrap_or("");
    if person_mismatch || !same_name(name, suggested_name) {
        return Err(HttpError::new(
            StatusCode::BAD_REQUEST,
            messages::CONFIRM_OR_CORRECT,
        ));
    }
    Ok(())
}

fn same_name(left: &str, right: &str) -> bool {
    left.to_lowercase() == right.to_lowercase()
}

fn suggestion_evidence(suggestion: &Value) -> Value {
    let mut evidence = Map::new();
    if let Some(value) = suggestion.get("evidence") {
        evidence.insert("evidence".to_string(), value.clone());
    }
    if let Some(value) = suggestion.get("match") {
        evidence.insert("match".to_string(), value.clone());
    }
    Value::Object(evidence)
}

fn assignment_json(
    assignment: &people::SpeakerAssignment,
    confirmed_at: f64,
    evidence: Value,
) -> Value {
    json!({
        "person_id": assignment.person_id,
        "meeting_id": assignment.meeting_id,
        "track_id": assignment.track_id,
        "name": assignment.name,
        "confirmed": assignment.confirmed.unwrap_or(true),
        "confirmed_at": confirmed_at,
        "source": assignment.source,
        "evidence": evidence,
    })
}

fn set_speaker(job: &mut Value, track: &str, name: &str) -> Result<(), HttpError> {
    let speakers = job
        .get_mut("document")
        .and_then(|document| document.get_mut("speakers"))
        .and_then(|speakers| speakers.as_object_mut())
        .ok_or_else(|| HttpError::new(StatusCode::BAD_REQUEST, messages::INVALID_REQUEST))?;
    speakers.insert(track.to_string(), json!(name));
    Ok(())
}

fn listed_suggestions(catalog: &Catalog, job: &Value) -> Vec<Value> {
    let available = voice::runtime_available(None, None, false);
    let intros = people::introduction_suggestions(&suggestion_job(job));
    let intro_tracks: BTreeSet<String> = intros.iter().map(|item| item.track_id.clone()).collect();
    let mut listed: Vec<Value> = intros.iter().map(suggestion_json).collect();
    listed.extend(voice_suggestions(catalog, job, &intro_tracks, available));
    listed
}

fn suggestion_json(suggestion: &people::IntroductionSuggestion) -> Value {
    json!({
        "id": suggestion.id,
        "track_id": suggestion.track_id,
        "name": suggestion.name,
        "kind": suggestion.kind,
        "evidence": {
            "segment_id": suggestion.evidence.segment_id,
            "quote": suggestion.evidence.quote,
            "start": suggestion.evidence.start,
            "end": suggestion.evidence.end,
        },
    })
}

fn suggestion_job(job: &Value) -> people::SuggestionJob {
    serde_json::from_value(job.clone()).unwrap_or(people::SuggestionJob {
        document: None,
        dismissed_identity_suggestions: Vec::new(),
    })
}

fn voice_suggestions(
    catalog: &Catalog,
    job: &Value,
    intro_tracks: &BTreeSet<String>,
    available: bool,
) -> Vec<Value> {
    if !available {
        return Vec::new();
    }
    let Some(items) = job.get("voice_suggestions").and_then(Value::as_array) else {
        return Vec::new();
    };
    let dismissed = job
        .get("dismissed_identity_suggestions")
        .and_then(Value::as_array);
    let mut kept = Vec::new();
    for suggestion in items {
        let Some(person_id) = suggestion.get("person_id").and_then(Value::as_str) else {
            continue;
        };
        let Ok(person) = catalog.people.get(person_id) else {
            continue;
        };
        let Some(track) = suggestion.get("track_id").and_then(Value::as_str) else {
            continue;
        };
        if intro_tracks.contains(track) || track_assigned(job, track) {
            continue;
        }
        let Some(id) = suggestion.get("id").and_then(Value::as_str) else {
            continue;
        };
        if dismissed.is_some_and(|items| items.iter().any(|item| item.as_str() == Some(id))) {
            continue;
        }
        let Some(version) = suggestion
            .pointer("/match/profile_version")
            .and_then(Value::as_str)
        else {
            continue;
        };
        let profile_ok = catalog
            .voices
            .profiles()
            .iter()
            .any(|profile| profile.person_id == person_id && profile.version == version);
        if !profile_ok {
            continue;
        }
        let mut copy = suggestion.clone();
        copy["name"] = json!(person.name);
        kept.push(copy);
    }
    kept
}

fn track_assigned(job: &Value, track: &str) -> bool {
    job.get("speaker_assignments")
        .and_then(Value::as_object)
        .and_then(|map| map.get(track))
        .is_some_and(|value| !value.is_null())
}

fn editing_job(
    job: &Value,
    body: &Map<String, Value>,
    track: Option<&str>,
) -> Result<(), HttpError> {
    let status = job
        .get("status")
        .and_then(Value::as_str)
        .ok_or_else(|| HttpError::new(StatusCode::BAD_REQUEST, messages::INVALID_REQUEST))?;
    let document = job.get("document").filter(|value| value.is_object());
    if NAMING_BUSY.contains(&status) || document.is_none() {
        return Err(HttpError::new(
            StatusCode::CONFLICT,
            messages::NAME_BEFORE_FINAL,
        ));
    }
    let revision = job_revision(job)?;
    if body.get("revision").and_then(Value::as_i64) != Some(revision) {
        return Err(HttpError::new(StatusCode::CONFLICT, messages::NAME_STALE));
    }
    if let Some(track) = track {
        let speakers = document
            .and_then(|value| value.get("speakers"))
            .and_then(Value::as_object);
        if speakers.is_none_or(|map| !map.contains_key(track)) {
            return Err(not_found());
        }
    }
    Ok(())
}

fn load_job(catalog: &mut Catalog, dir: &Path, jid: &str) -> Result<Value, HttpError> {
    if JobId::parse(jid).is_err() {
        return Err(not_found());
    }
    if let Some(job) = catalog.jobs.get(jid) {
        return Ok(job.clone());
    }
    match std::fs::read(job_file(dir, jid)) {
        Ok(bytes) => match serde_json::from_slice::<Value>(&bytes) {
            Ok(value) if value.is_object() => {
                catalog.jobs.insert(jid.to_string(), value.clone());
                Ok(value)
            }
            Ok(_) => Err(HttpError::new(
                StatusCode::BAD_REQUEST,
                messages::INVALID_REQUEST,
            )),
            Err(_) => Err(HttpError::new(
                StatusCode::INTERNAL_SERVER_ERROR,
                messages::LOCAL_OPERATION_FAILED,
            )),
        },
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => Err(not_found()),
        Err(_) => Err(HttpError::new(
            StatusCode::INTERNAL_SERVER_ERROR,
            messages::LOCAL_OPERATION_FAILED,
        )),
    }
}

fn import_disk_jobs(catalog: &mut Catalog, dir: &Path) {
    let Ok(entries) = std::fs::read_dir(dir.join("identity-jobs")) else {
        return;
    };
    for entry in entries.flatten() {
        let path = entry.path();
        if path.extension().and_then(|ext| ext.to_str()) != Some("json") {
            continue;
        }
        let Some(name) = path.file_stem().and_then(|stem| stem.to_str()) else {
            continue;
        };
        if JobId::parse(name).is_err() || catalog.jobs.contains_key(name) {
            continue;
        }
        let Ok(bytes) = std::fs::read(&path) else {
            continue;
        };
        if let Ok(value) = serde_json::from_slice::<Value>(&bytes)
            && value.is_object()
        {
            catalog.jobs.insert(name.to_string(), value);
        }
    }
}

fn ensure_object<'a>(
    job: &'a mut Value,
    key: &str,
) -> Result<&'a mut Map<String, Value>, HttpError> {
    if job.get(key).is_none() {
        job[key] = json!({});
    }
    job.get_mut(key)
        .and_then(Value::as_object_mut)
        .ok_or_else(|| HttpError::new(StatusCode::BAD_REQUEST, messages::INVALID_REQUEST))
}

fn without_track(job: &Value, track: &str) -> Result<Vec<Value>, HttpError> {
    let Some(list) = job.get("voice_suggestions") else {
        return Ok(Vec::new());
    };
    let Some(list) = list.as_array() else {
        return Err(HttpError::new(
            StatusCode::BAD_REQUEST,
            messages::INVALID_REQUEST,
        ));
    };
    let mut kept = Vec::new();
    for item in list {
        let item_track = item
            .get("track_id")
            .and_then(Value::as_str)
            .ok_or_else(|| HttpError::new(StatusCode::BAD_REQUEST, messages::INVALID_REQUEST))?;
        if item_track != track {
            kept.push(item.clone());
        }
    }
    Ok(kept)
}

fn array_field(job: &Value, key: &str) -> Result<Vec<Value>, HttpError> {
    match job.get(key) {
        None => Ok(Vec::new()),
        Some(Value::Array(items)) => Ok(items.clone()),
        Some(_) => Err(HttpError::new(
            StatusCode::BAD_REQUEST,
            messages::INVALID_REQUEST,
        )),
    }
}

fn job_revision(job: &Value) -> Result<i64, HttpError> {
    job.get("revision")
        .and_then(Value::as_i64)
        .ok_or_else(|| HttpError::new(StatusCode::BAD_REQUEST, messages::INVALID_REQUEST))
}

fn optional_text(value: Option<&Value>) -> Option<&str> {
    value
        .and_then(Value::as_str)
        .filter(|text| !text.is_empty())
}

fn json_object(bytes: &[u8]) -> Result<Map<String, Value>, HttpError> {
    if bytes.is_empty() {
        return Err(HttpError::new(
            StatusCode::BAD_REQUEST,
            messages::INVALID_REQUEST,
        ));
    }
    match serde_json::from_slice::<Value>(bytes) {
        Ok(Value::Object(map)) => Ok(map),
        _ => Err(HttpError::new(
            StatusCode::BAD_REQUEST,
            messages::INVALID_REQUEST,
        )),
    }
}

struct HttpError {
    status: StatusCode,
    message: String,
}

impl HttpError {
    fn new(status: StatusCode, message: impl Into<String>) -> Self {
        Self {
            status,
            message: message.into(),
        }
    }

    fn response(self) -> Response {
        (self.status, Json(json!({ "error": self.message }))).into_response()
    }
}

fn from_people(error: people::PeopleError) -> HttpError {
    let (status, message) = match error {
        people::PeopleError::Name => (StatusCode::BAD_REQUEST, messages::PERSON_NAME),
        people::PeopleError::Cap => (StatusCode::BAD_REQUEST, messages::PEOPLE_CAP),
        people::PeopleError::Missing => (StatusCode::NOT_FOUND, messages::PERSON_MISSING),
        people::PeopleError::RememberConsent => {
            (StatusCode::BAD_REQUEST, messages::REMEMBER_CONSENT)
        }
        people::PeopleError::RememberProfile => {
            (StatusCode::BAD_REQUEST, messages::REMEMBER_PROFILE)
        }
    };
    HttpError::new(status, message)
}

fn not_found() -> HttpError {
    HttpError::new(StatusCode::NOT_FOUND, NOT_FOUND)
}

fn unix_now() -> f64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|duration| duration.as_secs_f64())
        .unwrap_or(0.0)
}

fn new_id() -> String {
    random_hex(16)
}

fn random_hex(bytes: usize) -> String {
    let mut raw = vec![0_u8; bytes];
    let mut file = File::open("/dev/urandom").expect("urandom");
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
    use speakerdesk_core::messages;

    use crate::routes::AppState;

    async fn serve(app: axum::Router) -> Running {
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
        let running = Running {
            base: format!("http://127.0.0.1:{port}"),
            client: reqwest::Client::new(),
            shutdown: Some(shutdown),
            server,
        };
        for _ in 0..50 {
            if running
                .client
                .get(running.url("/api/people"))
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

    struct Running {
        base: String,
        client: reqwest::Client,
        shutdown: Option<tokio::sync::oneshot::Sender<()>>,
        server: tokio::task::JoinHandle<Result<(), std::io::Error>>,
    }

    impl Running {
        fn url(&self, path: &str) -> String {
            format!("{}{path}", self.base)
        }

        async fn shutdown(mut self) {
            if let Some(shutdown) = self.shutdown.take() {
                let _ = shutdown.send(());
            }
            let _ = tokio::time::timeout(Duration::from_secs(3), self.server).await;
        }
    }

    fn write_job(state: &AppState, id: &str, job: Value) {
        let path = super::job_file(state.data_dir(), id);
        std::fs::create_dir_all(path.parent().unwrap()).unwrap();
        std::fs::write(path, serde_json::to_vec(&job).unwrap()).unwrap();
    }

    fn intro_job(id: &str) -> Value {
        json!({
            "id": id,
            "name": "Meeting",
            "kind": "meeting",
            "status": "ready",
            "message": "",
            "language": "en",
            "created": 1.0,
            "updated": 1.0,
            "revision": 0,
            "duration": 12.0,
            "document": {
                "schema_version": 1,
                "speakers": {"speaker_0": "Speaker 1"},
                "segments": [{
                    "id": "intro",
                    "speaker": "speaker_0",
                    "start": 0.0,
                    "end": 3.0,
                    "text": "I'm Priya. Welcome to the meeting.",
                    "review": false
                }],
                "provenance": {"kind": "local_inference"}
            },
            "voice_checks": {
                "speaker_1": {"id": "check-1", "status": "checking"}
            }
        })
    }

    #[tokio::test]
    async fn route_table_accepts_people_methods_and_parent_rejects_bad_tokens() {
        // `crate::router` is `speakerdesk_host::router`: the parent applies the token guard.
        let running = serve(crate::router()).await;
        let job = "a".repeat(32);
        let person = "b".repeat(32);
        let suggestion = "c".repeat(32);
        let people = running
            .client
            .get(running.url("/api/people"))
            .send()
            .await
            .unwrap();
        assert_ne!(people.status(), 405);
        assert_eq!(people.status(), 200);
        let recognition = running
            .client
            .get(running.url("/api/recognition"))
            .send()
            .await
            .unwrap();
        assert_ne!(recognition.status(), 405);
        assert_eq!(recognition.status(), 200);
        let suggestions = running
            .client
            .get(running.url(&format!("/api/jobs/{job}/identity-suggestions")))
            .send()
            .await
            .unwrap();
        assert_ne!(suggestions.status(), 405);
        let mutations = [
            ("POST", "/api/people".to_string()),
            ("PATCH", format!("/api/people/{person}")),
            ("DELETE", format!("/api/people/{person}/voice")),
            ("PATCH", "/api/recognition".to_string()),
            (
                "POST",
                format!("/api/jobs/{job}/identity-suggestions/{suggestion}/dismiss"),
            ),
            (
                "POST",
                format!("/api/jobs/{job}/speakers/speaker_0/identity"),
            ),
        ];
        for (method, path) in mutations {
            let response = running
                .client
                .request(method.parse().unwrap(), running.url(&path))
                .header(crate::http::token_header(), "not-the-token")
                .json(&json!({}))
                .send()
                .await
                .unwrap();
            assert_ne!(response.status(), 405, "{method} {path}");
            assert_eq!(response.status(), 403, "{method} {path}");
            assert_eq!(
                response.json::<Value>().await.unwrap()["error"],
                messages::TOKEN_MISMATCH
            );
        }
        running.shutdown().await;
    }

    #[tokio::test]
    async fn people_and_identity_routes_follow_the_people_module() {
        let state = AppState::new();
        let running = serve(super::routes().with_state(state.clone())).await;
        let blank = running
            .client
            .post(running.url("/api/people"))
            .json(&json!({ "name": "   " }))
            .send()
            .await
            .unwrap();
        assert_eq!(blank.status(), 400);
        assert_eq!(
            blank.json::<Value>().await.unwrap()["error"],
            messages::PERSON_NAME
        );
        let long = running
            .client
            .post(running.url("/api/people"))
            .json(&json!({ "name": "a".repeat(101) }))
            .send()
            .await
            .unwrap();
        assert_eq!(long.status(), 400);
        assert_eq!(
            long.json::<Value>().await.unwrap()["error"],
            messages::PERSON_NAME
        );

        let created = running
            .client
            .post(running.url("/api/people"))
            .json(&json!({ "name": "  Priya  " }))
            .send()
            .await
            .unwrap();
        assert_eq!(created.status(), 201);
        let created: Value = created.json().await.unwrap();
        assert_eq!(created["name"], "Priya");
        assert!(created.get("voice_saved").is_none());
        let pid = created["id"].as_str().unwrap().to_string();
        running
            .client
            .post(running.url("/api/people"))
            .json(&json!({ "name": "alex" }))
            .send()
            .await
            .unwrap();

        let listed = running
            .client
            .get(running.url("/api/people"))
            .send()
            .await
            .unwrap()
            .json::<Value>()
            .await
            .unwrap();
        assert_eq!(listed["people"][0]["name"], "alex");
        assert_eq!(listed["people"][1]["name"], "Priya");
        assert_eq!(listed["people"][1]["id"], pid);
        assert_eq!(listed["voice_available"], false);
        assert_eq!(listed["readiness_code"], "runtime_unqualified");
        assert_eq!(listed["next_action"], "updates");
        assert_eq!(listed["people"][1]["voice_saved"], false);
        assert_eq!(listed["people"][1]["voice_profile_state"], "none");
        assert!(listed["people"][1]["voice_profile_message"].is_null());
        assert!(!listed.to_string().contains("centroid"));

        let renamed = running
            .client
            .patch(running.url(&format!("/api/people/{pid}")))
            .json(&json!({ "name": "Dr. Priya" }))
            .send()
            .await
            .unwrap();
        assert_eq!(renamed.status(), 200);
        let renamed: Value = renamed.json().await.unwrap();
        assert_eq!(renamed["name"], "Dr. Priya");
        assert!(renamed["updated"].is_number());
        let missing = running
            .client
            .patch(running.url(&format!("/api/people/{}", "c".repeat(32))))
            .json(&json!({ "name": "Nope" }))
            .send()
            .await
            .unwrap();
        assert_eq!(missing.status(), 404);
        assert_eq!(
            missing.json::<Value>().await.unwrap()["error"],
            messages::PERSON_MISSING
        );
        let forgotten = running
            .client
            .delete(running.url(&format!("/api/people/{pid}/voice")))
            .send()
            .await
            .unwrap();
        assert_eq!(forgotten.status(), 200);
        assert_eq!(forgotten.json::<Value>().await.unwrap()["forgotten"], true);
        let forgotten_missing = running
            .client
            .delete(running.url(&format!("/api/people/{}/voice", "d".repeat(32))))
            .send()
            .await
            .unwrap();
        assert_eq!(forgotten_missing.status(), 404);

        let recognition = running
            .client
            .get(running.url("/api/recognition"))
            .send()
            .await
            .unwrap()
            .json::<Value>()
            .await
            .unwrap();
        assert_eq!(recognition["enabled"], true);
        let bad_choice = running
            .client
            .patch(running.url("/api/recognition"))
            .json(&json!({ "enabled": "no" }))
            .send()
            .await
            .unwrap();
        assert_eq!(bad_choice.status(), 400);
        assert_eq!(
            bad_choice.json::<Value>().await.unwrap()["error"],
            messages::RECOGNITION_CHOICE
        );

        let meeting = "a".repeat(32);
        write_job(&state, &meeting, intro_job(&meeting));
        let off = running
            .client
            .patch(running.url("/api/recognition"))
            .json(&json!({ "enabled": false }))
            .send()
            .await
            .unwrap()
            .json::<Value>()
            .await
            .unwrap();
        assert_eq!(off["enabled"], false);
        let suggestions = running
            .client
            .get(running.url(&format!("/api/jobs/{meeting}/identity-suggestions")))
            .send()
            .await
            .unwrap()
            .json::<Value>()
            .await
            .unwrap();
        assert_eq!(suggestions["revision"], 0);
        assert_eq!(suggestions["suggestions"][0]["name"], "Priya");
        assert_eq!(
            suggestions["suggestions"][0]["kind"],
            "explicit_introduction"
        );
        assert_eq!(
            suggestions["suggestions"][0]["evidence"]["quote"],
            "I'm Priya. Welcome to the meeting."
        );
        let sid = suggestions["suggestions"][0]["id"]
            .as_str()
            .unwrap()
            .to_string();

        let stale = running
            .client
            .post(running.url(&format!("/api/jobs/{meeting}/speakers/speaker_0/identity")))
            .json(&json!({ "revision": 4, "suggestion_id": sid }))
            .send()
            .await
            .unwrap();
        assert_eq!(stale.status(), 409);
        assert_eq!(
            stale.json::<Value>().await.unwrap()["error"],
            messages::NAME_STALE
        );
        let confirmed = running
            .client
            .post(running.url(&format!("/api/jobs/{meeting}/speakers/speaker_0/identity")))
            .json(&json!({ "revision": 0, "suggestion_id": sid }))
            .send()
            .await
            .unwrap();
        assert_eq!(confirmed.status(), 200);
        let confirmed: Value = confirmed.json().await.unwrap();
        assert_eq!(confirmed["document"]["speakers"]["speaker_0"], "Priya");
        assert!(confirmed["speaker_assignments"]["speaker_0"]["person_id"].is_null());
        assert_eq!(
            confirmed["speaker_assignments"]["speaker_0"]["source"],
            "explicit_introduction"
        );
        assert_eq!(
            confirmed["speaker_assignments"]["speaker_0"]["evidence"]["evidence"]["segment_id"],
            "intro"
        );
        assert_eq!(
            confirmed["voice_checks"]["speaker_1"]["status"],
            "cancelled"
        );
        assert_eq!(confirmed["voice_checks"]["speaker_0"]["status"], "manual");
        assert_eq!(confirmed["revision"], 1);
        let people = running
            .client
            .get(running.url("/api/people"))
            .send()
            .await
            .unwrap()
            .json::<Value>()
            .await
            .unwrap();
        assert_eq!(people["people"].as_array().unwrap().len(), 2);

        let stale_suggestion = running
            .client
            .post(running.url(&format!("/api/jobs/{meeting}/speakers/speaker_0/identity")))
            .json(&json!({ "revision": 1, "suggestion_id": sid }))
            .send()
            .await
            .unwrap();
        assert_eq!(stale_suggestion.status(), 409);
        assert_eq!(
            stale_suggestion.json::<Value>().await.unwrap()["error"],
            messages::SUGGESTION_STALE
        );

        let other = "e".repeat(32);
        write_job(&state, &other, intro_job(&other));
        let other_suggestions = running
            .client
            .get(running.url(&format!("/api/jobs/{other}/identity-suggestions")))
            .send()
            .await
            .unwrap()
            .json::<Value>()
            .await
            .unwrap();
        let other_sid = other_suggestions["suggestions"][0]["id"]
            .as_str()
            .unwrap()
            .to_string();
        let stale_dismiss = running
            .client
            .post(running.url(&format!(
                "/api/jobs/{other}/identity-suggestions/{other_sid}/dismiss"
            )))
            .json(&json!({ "revision": 3 }))
            .send()
            .await
            .unwrap();
        assert_eq!(stale_dismiss.status(), 409);
        assert_eq!(
            stale_dismiss.json::<Value>().await.unwrap()["error"],
            messages::NAME_STALE
        );
        let dismissed = running
            .client
            .post(running.url(&format!(
                "/api/jobs/{other}/identity-suggestions/{other_sid}/dismiss"
            )))
            .json(&json!({ "revision": 0 }))
            .send()
            .await
            .unwrap();
        assert_eq!(dismissed.status(), 200);
        assert_eq!(dismissed.json::<Value>().await.unwrap()["revision"], 1);
        let gone = running
            .client
            .get(running.url(&format!("/api/jobs/{other}/identity-suggestions")))
            .send()
            .await
            .unwrap()
            .json::<Value>()
            .await
            .unwrap();
        assert_eq!(gone["suggestions"].as_array().unwrap().len(), 0);

        let busy_id = "f".repeat(32);
        let mut busy = intro_job(&busy_id);
        busy["status"] = json!("preparing");
        write_job(&state, &busy_id, busy);
        let blocked = running
            .client
            .post(running.url(&format!("/api/jobs/{busy_id}/speakers/speaker_0/identity")))
            .json(&json!({ "revision": 0, "name": "Alex" }))
            .send()
            .await
            .unwrap();
        assert_eq!(blocked.status(), 409);
        assert_eq!(
            blocked.json::<Value>().await.unwrap()["error"],
            messages::NAME_BEFORE_FINAL
        );

        let mismatch_id = "1".repeat(32);
        write_job(&state, &mismatch_id, intro_job(&mismatch_id));
        let mismatch_suggestions = running
            .client
            .get(running.url(&format!("/api/jobs/{mismatch_id}/identity-suggestions")))
            .send()
            .await
            .unwrap()
            .json::<Value>()
            .await
            .unwrap();
        let mismatch_sid = mismatch_suggestions["suggestions"][0]["id"]
            .as_str()
            .unwrap();
        let mismatch = running
            .client
            .post(running.url(&format!(
                "/api/jobs/{mismatch_id}/speakers/speaker_0/identity"
            )))
            .json(&json!({ "revision": 0, "suggestion_id": mismatch_sid, "person_id": pid }))
            .send()
            .await
            .unwrap();
        assert_eq!(mismatch.status(), 400);
        assert_eq!(
            mismatch.json::<Value>().await.unwrap()["error"],
            messages::CONFIRM_OR_CORRECT
        );
        let unknown_person = running
            .client
            .post(running.url(&format!(
                "/api/jobs/{mismatch_id}/speakers/speaker_0/identity"
            )))
            .json(&json!({ "revision": 0, "person_id": "2".repeat(32) }))
            .send()
            .await
            .unwrap();
        assert_eq!(unknown_person.status(), 404);
        assert_eq!(
            unknown_person.json::<Value>().await.unwrap()["error"],
            messages::PERSON_MISSING
        );
        let manual = running
            .client
            .post(running.url(&format!(
                "/api/jobs/{mismatch_id}/speakers/speaker_0/identity"
            )))
            .json(&json!({ "revision": 0, "name": "Alex" }))
            .send()
            .await
            .unwrap();
        assert_eq!(manual.status(), 200);
        let manual: Value = manual.json().await.unwrap();
        assert_eq!(manual["document"]["speakers"]["speaker_0"], "Alex");
        assert_eq!(
            manual["speaker_assignments"]["speaker_0"]["source"],
            "manual"
        );
        assert!(manual["speaker_assignments"]["speaker_0"]["person_id"].is_null());

        let linked_id = "3".repeat(32);
        write_job(&state, &linked_id, intro_job(&linked_id));
        let linked = running
            .client
            .post(running.url(&format!(
                "/api/jobs/{linked_id}/speakers/speaker_0/identity"
            )))
            .json(&json!({ "revision": 0, "person_id": pid, "name": "ignored" }))
            .send()
            .await
            .unwrap();
        assert_eq!(linked.status(), 200);
        let linked: Value = linked.json().await.unwrap();
        assert_eq!(linked["document"]["speakers"]["speaker_0"], "Dr. Priya");
        assert_eq!(linked["speaker_assignments"]["speaker_0"]["person_id"], pid);
        assert_eq!(
            linked["speaker_assignments"]["speaker_0"]["meeting_id"],
            linked_id
        );

        let unknown = running
            .client
            .get(running.url(&format!(
                "/api/jobs/{}/identity-suggestions",
                "4".repeat(32)
            )))
            .send()
            .await
            .unwrap();
        assert_eq!(unknown.status(), 404);
        assert_ne!(unknown.status(), 405);
        running.shutdown().await;
        let _ = std::fs::remove_dir_all(state.data_dir());
    }
}
