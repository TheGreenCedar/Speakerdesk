//! People and identity routes share the application SQLite store.

use std::collections::{BTreeMap, BTreeSet};
use std::fs::File;
use std::io::Read;
use std::time::{SystemTime, UNIX_EPOCH};

use axum::Json;
use axum::Router;
use axum::body::Bytes;
use axum::extract::{Path as AxumPath, State};
use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use axum::routing::{delete, get, patch, post};
use serde_json::{Map, Value, json};
use speakerdesk_core::error::StoreError;
use speakerdesk_core::ids::JobId;
use speakerdesk_core::store::{JobStore, SqliteJobStore};
use speakerdesk_core::{messages, people, voice};

use crate::routes::AppState;

const NOT_FOUND: &str = "The requested URL was not found on the server. If you entered the URL manually please check your spelling and try again.";
const NAMING_BUSY: &[&str] = &["preparing", "queued", "processing", "finishing"];
const VOICE_READY: &str = "Saved voices can recognize speakers in new recordings. Confirm a speaker before saving their voice.";
const VOICE_UNQUALIFIED: &str = "Voice recognition needs a qualified GPU/ANE runtime. Saved names and voices are kept. Install a Speakerdesk update with qualified voice recognition.";
mod automatic;
mod voice_http;
pub(crate) use automatic::check_saved_voices;

const PROFILE_KEPT: &str = "Your saved voice is kept. Voice recognition is currently unavailable.";

pub fn routes() -> Router<AppState> {
    Router::new()
        .route("/api/people", get(list_people).post(create_person))
        .route("/api/people/{pid}", patch(rename_person))
        .route(
            "/api/people/{pid}/voice",
            delete(forget_voice).post(voice_http::remember_voice),
        )
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
        .route(
            "/api/jobs/{jid}/speakers/{track}/voice-clips",
            get(voice_http::voice_clips),
        )
        .route(
            "/api/jobs/{jid}/speakers/{track}/voice-preview",
            get(voice_http::voice_preview),
        )
        .route(
            "/api/jobs/{jid}/speakers/{track}/voice-suggestion",
            post(voice_http::voice_suggestion),
        )
}

struct Catalog {
    people: people::People,
    versions: BTreeMap<String, String>,
    profiles: BTreeMap<String, Value>,
    policy: Option<voice::Calibration>,
}

impl Catalog {
    fn load(store: &SqliteJobStore, policy: Option<voice::Calibration>) -> Result<Self, HttpError> {
        Ok(Self {
            people: store.people()?,
            versions: store.voice_versions()?,
            profiles: store.voice_profiles()?,
            policy,
        })
    }
}

async fn database(
    app: AppState,
    status: StatusCode,
    work: impl FnOnce(&mut SqliteJobStore) -> Result<Value, HttpError> + Send + 'static,
) -> Response {
    crate::routes::blocking(move || match work(&mut app.store()) {
        Ok(value) => (status, Json(value)).into_response(),
        Err(error) => error.response(),
    })
    .await
}

async fn list_people(State(app): State<AppState>) -> Response {
    let policy = app.models().voice_policy();
    database(app, StatusCode::OK, move |store| {
        Ok(people_view(&Catalog::load(store, policy)?))
    })
    .await
}

async fn create_person(State(app): State<AppState>, body: Bytes) -> Response {
    database(app, StatusCode::CREATED, move |store| {
        let body = json_object(&body)?;
        let name = name_text(&body)?;
        store.edit_person(|people| {
            let person = people.create(name, unix_now()).map_err(from_people)?;
            let value = person_json(&person);
            Ok((person, value))
        })
    })
    .await
}

async fn rename_person(
    State(app): State<AppState>,
    AxumPath(pid): AxumPath<String>,
    body: Bytes,
) -> Response {
    database(app, StatusCode::OK, move |store| {
        let body = json_object(&body)?;
        let name = name_text(&body)?;
        store.edit_person(|people| {
            let person = people.rename(&pid, name, unix_now()).map_err(from_people)?;
            let value = person_json(&person);
            Ok((person, value))
        })
    })
    .await
}

async fn forget_voice(State(app): State<AppState>, AxumPath(pid): AxumPath<String>) -> Response {
    database(app, StatusCode::OK, move |store| {
        store.people()?.get(&pid).map_err(from_people)?;
        store.forget_voice(&pid)?;
        Ok(json!({"forgotten": true}))
    })
    .await
}

async fn recognition_status(State(app): State<AppState>) -> Response {
    database(app, StatusCode::OK, |store| {
        Ok(json!({"enabled": store.preference("voice_recognition")?.as_deref() != Some("false")}))
    })
    .await
}

async fn update_recognition(State(app): State<AppState>, body: Bytes) -> Response {
    database(app, StatusCode::OK, move |store| {
        let body = json_object(&body)?;
        let enabled = body
            .get("enabled")
            .and_then(Value::as_bool)
            .ok_or_else(|| HttpError::new(StatusCode::BAD_REQUEST, messages::RECOGNITION_CHOICE))?;
        store.set_recognition(enabled)?;
        Ok(json!({"enabled": enabled}))
    })
    .await
}

async fn identity_suggestions(
    State(app): State<AppState>,
    AxumPath(jid): AxumPath<String>,
) -> Response {
    let policy = app.models().voice_policy();
    database(app, StatusCode::OK, move |store| {
        let job = load_job(store, &jid)?;
        Ok(json!({"revision": job_revision(&job)?, "suggestions": listed_suggestions(&Catalog::load(store, policy)?, &job)}))
    }).await
}

async fn dismiss_suggestion(
    State(app): State<AppState>,
    AxumPath((jid, sid)): AxumPath<(String, String)>,
    body: Bytes,
) -> Response {
    let policy = app.models().voice_policy();
    database(app, StatusCode::OK, move |store| {
        let body = json_object(&body)?;
        update_identity(store, &jid, policy, |catalog, job| {
            dismiss_in(catalog, job, &sid, &body)
        })
    })
    .await
}

async fn confirm_identity(
    State(app): State<AppState>,
    AxumPath((jid, track)): AxumPath<(String, String)>,
    body: Bytes,
) -> Response {
    let policy = app.models().voice_policy();
    database(app, StatusCode::OK, move |store| {
        let body = json_object(&body)?;
        update_identity(store, &jid, policy, |catalog, job| {
            confirm_in(catalog, job, &jid, &track, &body)
        })
    })
    .await
}

fn update_identity(
    store: &mut SqliteJobStore,
    jid: &str,
    policy: Option<voice::Calibration>,
    update: impl FnOnce(&Catalog, Value) -> Result<Value, HttpError>,
) -> Result<Value, HttpError> {
    let id = JobId::parse(jid).map_err(|_| not_found())?;
    let catalog = Catalog::load(store, policy)?;
    let (job, ()) = store
        .update_job(&id, |job| {
            let value = update(&catalog, serde_json::to_value(&*job).map_err(invalid_job)?)?;
            *job = serde_json::from_value(value).map_err(invalid_job)?;
            Ok::<_, HttpError>(())
        })?
        .ok_or_else(not_found)?;
    Ok(crate::routes::stored_job_json(&job))
}

fn invalid_job(error: serde_json::Error) -> HttpError {
    tracing::error!(%error, "stored document could not be decoded");
    HttpError::new(
        StatusCode::INTERNAL_SERVER_ERROR,
        messages::LOCAL_OPERATION_FAILED,
    )
}

fn people_view(catalog: &Catalog) -> Value {
    let available = catalog.policy.is_some();
    let mut rows: Vec<people::Person> = catalog.people.iter().cloned().collect();
    rows.sort_by(|left, right| {
        left.name
            .to_lowercase()
            .cmp(&right.name.to_lowercase())
            .then_with(|| left.id.cmp(&right.id))
    });
    let people = rows
        .iter()
        .map(|person| {
            listed_person(
                person,
                catalog.profiles.get(&person.id),
                catalog.policy.as_ref(),
            )
        })
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

fn listed_person(
    person: &people::Person,
    profile: Option<&Value>,
    policy: Option<&voice::Calibration>,
) -> Value {
    let saved = profile.is_some();
    let mut value = person_json(person);
    let object = value.as_object_mut().expect("person object");
    object.insert("voice_saved".to_string(), json!(saved));
    let compatible = profile
        .zip(policy)
        .is_some_and(|(profile, policy)| profile.get("model") == Some(&json!(policy.model)));
    object.insert("voice_compatible".into(), json!(compatible));
    let (state, message, action) = if !saved {
        ("none", None, None)
    } else if policy.is_none() {
        ("stored_unavailable", Some(PROFILE_KEPT), None)
    } else if compatible {
        ("compatible", None, None)
    } else {
        (
            "incompatible_runtime",
            Some(
                "Remember this voice again before it can recognize this person. Your saved voice and name are kept until you choose to replace the voice.",
            ),
            Some("remember_voice"),
        )
    };
    object.insert("voice_profile_state".into(), json!(state));
    object.insert("voice_profile_message".into(), json!(message));
    object.insert("voice_profile_next_action".into(), json!(action));
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

fn name_text(body: &Map<String, Value>) -> Result<&str, HttpError> {
    match body.get("name") {
        Some(Value::String(name)) => Ok(name.as_str()),
        _ => Err(HttpError::new(
            StatusCode::BAD_REQUEST,
            messages::PERSON_NAME,
        )),
    }
}

fn dismiss_in(
    catalog: &Catalog,
    mut job: Value,
    sid: &str,
    body: &Map<String, Value>,
) -> Result<Value, HttpError> {
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
    Ok(job)
}

fn confirm_in(
    catalog: &Catalog,
    mut job: Value,
    jid: &str,
    track: &str,
    body: &Map<String, Value>,
) -> Result<Value, HttpError> {
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
    let available = catalog.policy.is_some();
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
            .versions
            .get(person_id)
            .is_some_and(|stored| stored == version);
        if !profile_ok
            || catalog.policy.as_ref().is_none_or(|policy| {
                suggestion.pointer("/match/model") != Some(&json!(policy.model))
                    || suggestion.pointer("/match/calibration")
                        != Some(&voice_http::policy_json(policy))
            })
        {
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

fn load_job(store: &SqliteJobStore, jid: &str) -> Result<Value, HttpError> {
    let id = JobId::parse(jid).map_err(|_| not_found())?;
    let job = store.get(&id, false)?.ok_or_else(not_found)?;
    serde_json::to_value(job).map_err(invalid_job)
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

impl From<StoreError> for HttpError {
    fn from(error: StoreError) -> Self {
        tracing::error!(%error, "identity store operation failed");
        Self::new(
            StatusCode::INTERNAL_SERVER_ERROR,
            messages::LOCAL_OPERATION_FAILED,
        )
    }
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
    use speakerdesk_core::store::JobStore;
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

    fn write_job(state: &AppState, _id: &str, job: Value) {
        state.insert_job(serde_json::from_value(job).unwrap());
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
    async fn voice_editor_routes_publish_check_and_reopen_one_store() {
        let root = tempfile::tempdir().unwrap();
        let state = AppState::open(root.path()).unwrap();
        let replay = std::sync::Arc::new(VoiceReplay::default());
        state.set_models(replay.clone());
        let jid = "7".repeat(32);
        voice_job(&state, &jid);
        let running = voice_server(&state).await;
        let pid = enroll_person(&running, &jid).await;
        let body = json!({"revision":1,"meeting_id":jid,"track_id":"speaker_0","segment_ids":["a","b"],"consent":true});
        let endpoint = running.url(&format!("/api/people/{pid}/voice"));
        let mut denied = body.clone();
        denied["consent"] = json!(false);
        assert_eq!(
            running
                .client
                .post(&endpoint)
                .json(&denied)
                .send()
                .await
                .unwrap()
                .status(),
            400
        );
        denied = body.clone();
        denied["revision"] = json!(0);
        assert_eq!(
            running
                .client
                .post(&endpoint)
                .json(&denied)
                .send()
                .await
                .unwrap()
                .status(),
            409
        );
        assert_eq!(replay.calls.load(std::sync::atomic::Ordering::SeqCst), 0);
        let choices: Value = running
            .client
            .get(running.url(&format!("/api/jobs/{jid}/speakers/speaker_0/voice-clips")))
            .send()
            .await
            .unwrap()
            .json()
            .await
            .unwrap();
        assert_eq!(choices["status"], "ready");
        assert_eq!(choices["recommended_ids"], json!(["a", "b"]));
        let preview = running
            .client
            .get(running.url(&format!(
                "/api/jobs/{jid}/speakers/speaker_0/voice-preview?revision=1&clip=b"
            )))
            .header("Range", "bytes=0-43")
            .send()
            .await
            .unwrap();
        assert_eq!(preview.status(), 206);
        let header = preview.bytes().await.unwrap();
        assert_eq!(&header[..4], b"RIFF");
        assert_eq!(
            u32::from_le_bytes(header[40..44].try_into().unwrap()),
            9 * 16000 * 2
        );
        assert_eq!(
            running
                .client
                .get(running.url(&format!(
                    "/api/jobs/{jid}/speakers/speaker_0/voice-preview?revision=0&clip=b"
                )))
                .send()
                .await
                .unwrap()
                .status(),
            409
        );
        let saved = running
            .client
            .post(&endpoint)
            .json(&body)
            .send()
            .await
            .unwrap();
        assert_eq!(saved.status(), 200, "{}", saved.text().await.unwrap());
        assert_eq!(replay.calls.load(std::sync::atomic::Ordering::SeqCst), 2);
        assert_eq!(
            *replay.starts.lock().unwrap(),
            vec![3276. / 32768., 6553. / 32768.]
        );
        let original = state.store().voice_profiles().unwrap()[&pid].clone();
        assert_eq!(
            original["model"],
            json!(crate::native::native_voice_policy().model)
        );
        assert_eq!(original["clips"].as_array().unwrap().len(), 2);
        assert_eq!(original["consent"], "explicit_remember_voice");
        assert_eq!(
            running
                .client
                .patch(running.url("/api/recognition"))
                .json(&json!({"enabled":false}))
                .send()
                .await
                .unwrap()
                .status(),
            200
        );
        let second = "8".repeat(32);
        voice_job(&state, &second);
        let checked = running
            .client
            .post(running.url(&format!(
                "/api/jobs/{second}/speakers/speaker_0/voice-suggestion"
            )))
            .json(&json!({"revision":0,"segment_ids":["a","b"]}))
            .send()
            .await
            .unwrap();
        assert_eq!(checked.status(), 200);
        let checked: Value = checked.json().await.unwrap();
        assert_eq!(checked["matched"], true);
        assert_eq!(
            checked["job"]["document"]["speakers"]["speaker_0"],
            "Speaker 1"
        );
        let suggestions: Value = running
            .client
            .get(running.url(&format!("/api/jobs/{second}/identity-suggestions")))
            .send()
            .await
            .unwrap()
            .json()
            .await
            .unwrap();
        assert_eq!(suggestions["suggestions"].as_array().unwrap().len(), 1);
        assert_eq!(suggestions["suggestions"][0]["person_id"], pid);
        let applied = running.client.post(running.url(&format!("/api/jobs/{second}/speakers/speaker_0/identity"))).json(&json!({"revision":2,"person_id":pid,"suggestion_id":suggestions["suggestions"][0]["id"]})).send().await.unwrap();
        assert_eq!(applied.status(), 200);
        assert_eq!(
            applied.json::<Value>().await.unwrap()["document"]["speakers"]["speaker_0"],
            "Ada"
        );
        running.shutdown().await;
        drop(state);
        let state = AppState::open(root.path()).unwrap();
        state.set_models(replay);
        let running = voice_server(&state).await;
        let setup: Value = running
            .client
            .get(running.url("/api/setup"))
            .send()
            .await
            .unwrap()
            .json()
            .await
            .unwrap();
        assert_eq!(setup["voice"]["enabled"], false);
        assert_eq!(setup["voice"]["available"], true);
        let people: Value = running
            .client
            .get(running.url("/api/people"))
            .send()
            .await
            .unwrap()
            .json()
            .await
            .unwrap();
        assert_eq!(people["people"][0]["voice_compatible"], true);
        assert_eq!(state.store().voice_profiles().unwrap()[&pid], original);
        state.set_models(std::sync::Arc::new(crate::routes::UnavailableModels));
        let kept: Value = running
            .client
            .get(running.url("/api/people"))
            .send()
            .await
            .unwrap()
            .json()
            .await
            .unwrap();
        assert_eq!(
            kept["people"][0]["voice_profile_state"],
            "stored_unavailable"
        );
        assert_eq!(state.store().voice_profiles().unwrap()[&pid], original);
        assert_eq!(
            running
                .client
                .delete(running.url(&format!("/api/people/{pid}/voice")))
                .send()
                .await
                .unwrap()
                .status(),
            200
        );
        assert!(state.store().voice_profiles().unwrap().is_empty());
        assert_eq!(
            state.store().people().unwrap().get(&pid).unwrap().name,
            "Ada"
        );
        running.shutdown().await;
    }

    #[tokio::test]
    async fn blocked_voice_work_respects_edits_forget_replacement_cancel_and_shutdown() {
        use std::sync::{Arc, Condvar, Mutex, mpsc};
        for case in [
            "words",
            "bounds",
            "source",
            "catalog",
            "forget",
            "replace",
            "cancel",
            "shutdown",
            "check_save",
            "check_cancel",
            "check_shutdown",
            "check_audio_failure",
            "check_model_failure",
            "manual_off_on",
        ] {
            let root = tempfile::tempdir().unwrap();
            let state = AppState::open(root.path()).unwrap();
            let (entered, receive) = mpsc::sync_channel(1);
            let gate = Arc::new((Mutex::new(false), Condvar::new()));
            let replay = Arc::new(VoiceReplay {
                gate: Some(VoiceGate {
                    entered,
                    released: gate.clone(),
                }),
                wrong_identity: case == "check_model_failure",
                ..Default::default()
            });
            state.set_models(replay);
            let jid = "9".repeat(32);
            if matches!(case, "source" | "catalog") {
                source_voice_job(&state, &jid);
            } else {
                voice_job(&state, &jid);
            }
            let running = voice_server(&state).await;
            let checking = case.starts_with("check") || case == "manual_off_on";
            let pid = if checking {
                String::new()
            } else {
                enroll_person(&running, &jid).await
            };
            let url = if checking {
                running.url(&format!(
                    "/api/jobs/{jid}/speakers/speaker_0/voice-suggestion"
                ))
            } else {
                running.url(&format!("/api/people/{pid}/voice"))
            };
            let client = running.client.clone();
            let body = json!({"revision":if checking{0}else{1},"meeting_id":jid,"track_id":"speaker_0","segment_ids":["a","b"],"consent":true});
            let pending =
                tokio::spawn(async move { client.post(url).json(&body).send().await.unwrap() });
            tokio::task::spawn_blocking(move || {
                receive.recv_timeout(Duration::from_secs(5)).unwrap()
            })
            .await
            .unwrap();
            assert!(state.admission.busy());
            let mut replacement = None;
            match case {
                "source" | "catalog" => {
                    state
                        .store()
                        .update_job(&speakerdesk_core::ids::JobId::parse(&jid).unwrap(), |job| {
                            if case == "source" {
                                job.document.as_mut().unwrap().segments[0]
                                    .capture_source
                                    .as_mut()
                                    .unwrap()["source_id"] = json!("system");
                            } else {
                                job.extra.get_mut("capture_source_catalog").unwrap()["policy"] =
                                    json!("changed");
                            }
                            Ok::<_, speakerdesk_core::error::StoreError>(())
                        })
                        .unwrap();
                }
                "words" | "bounds" => {
                    let changes = if case == "words" {
                        json!({"text":"Protected during voice extraction."})
                    } else {
                        json!({"start":0.1})
                    };
                    let response = running
                        .client
                        .patch(running.url(&format!("/api/jobs/{jid}/segments/a")))
                        .json(&json!({"segment_revision":0,"changes":changes}))
                        .send()
                        .await
                        .unwrap();
                    assert_eq!(response.status(), 200);
                }
                "forget" => {
                    assert_eq!(
                        running
                            .client
                            .delete(running.url(&format!("/api/people/{pid}/voice")))
                            .send()
                            .await
                            .unwrap()
                            .status(),
                        200
                    );
                }
                "replace" => {
                    let other = "a".repeat(32);
                    voice_job(&state, &other);
                    assert_eq!(
                        running
                            .client
                            .post(
                                running
                                    .url(&format!("/api/jobs/{other}/speakers/speaker_0/identity"))
                            )
                            .json(&json!({"revision":0,"person_id":pid}))
                            .send()
                            .await
                            .unwrap()
                            .status(),
                        200
                    );
                    let saved=running.client.post(running.url(&format!("/api/people/{pid}/voice"))).json(&json!({"revision":1,"meeting_id":other,"track_id":"speaker_0","segment_ids":["a","b"],"consent":true})).send().await.unwrap();
                    assert_eq!(saved.status(), 200);
                    replacement = Some(state.store().voice_profiles().unwrap()[&pid].clone());
                }
                "cancel" | "check_cancel" => {
                    assert_eq!(
                        running
                            .client
                            .post(running.url(&format!("/api/jobs/{jid}/cancel")))
                            .send()
                            .await
                            .unwrap()
                            .status(),
                        200
                    );
                }
                "shutdown" | "check_shutdown" => {
                    state.cancel_all().unwrap();
                }
                "check_save" => {
                    let current: Value = running
                        .client
                        .get(running.url(&format!("/api/jobs/{jid}")))
                        .send()
                        .await
                        .unwrap()
                        .json()
                        .await
                        .unwrap();
                    let mut document = current["document"].clone();
                    document["segments"][0]["text"] = json!("Saved while checking.");
                    assert_eq!(
                        running
                            .client
                            .put(running.url(&format!("/api/jobs/{jid}/transcript")))
                            .json(&json!({"revision":current["revision"],"document":document}))
                            .send()
                            .await
                            .unwrap()
                            .status(),
                        200
                    );
                }
                "manual_off_on" => {
                    for enabled in [false, true] {
                        assert_eq!(
                            running
                                .client
                                .patch(running.url("/api/recognition"))
                                .json(&json!({"enabled":enabled}))
                                .send()
                                .await
                                .unwrap()
                                .status(),
                            200
                        );
                    }
                }
                "check_audio_failure" => {
                    std::fs::OpenOptions::new()
                        .write(true)
                        .open(state.data_dir().join(&jid).join("audio.wav"))
                        .unwrap()
                        .set_len(48)
                        .unwrap();
                }
                "check_model_failure" => {}
                _ => unreachable!(),
            }
            // Cancellation cannot release admission while this model call still owns work.
            assert!(state.admission.busy());
            assert!(state.admission.ensure_idle().is_err());
            *gate.0.lock().unwrap() = true;
            gate.1.notify_all();
            let response = tokio::time::timeout(Duration::from_secs(5), pending)
                .await
                .unwrap()
                .unwrap();
            let status = response.status();
            let result: Value = response.json().await.unwrap();
            let accepted = matches!(case, "words" | "manual_off_on");
            assert_eq!(
                status,
                if accepted { 200 } else { 409 },
                "case {case}: {result}"
            );
            assert!(!state.admission.busy());
            let job = state
                .store()
                .get(&speakerdesk_core::ids::JobId::parse(&jid).unwrap(), false)
                .unwrap()
                .unwrap();
            assert_eq!(job.status, "ready");
            if case == "words" {
                assert_eq!(
                    job.document.unwrap().segments[0].text,
                    "Protected during voice extraction."
                );
            }
            if checking {
                assert_ne!(job.extra["voice_checks"]["speaker_0"]["status"], "checking");
            }
            let profiles = state.store().voice_profiles().unwrap();
            if let Some(replacement) = replacement {
                assert_eq!(profiles[&pid], replacement);
            } else {
                assert_eq!(profiles.len(), usize::from(case == "words"), "case {case}");
            }
            running.shutdown().await;
        }
    }

    #[tokio::test]
    async fn completed_import_checks_saved_voices_once_without_enrollment() {
        use speakerdesk_core::{ids::JobId, store::JobStore};
        use std::sync::{Arc, atomic::Ordering};
        for case in [
            "match",
            "disabled",
            "checked",
            "incompatible",
            "introduction",
        ] {
            let root = tempfile::tempdir().unwrap();
            let state = AppState::open(root.path()).unwrap();
            state.set_issues(Vec::new());
            state.set_models(Arc::new(VoiceReplay::default()));
            let enrollment = "7".repeat(32);
            voice_job(&state, &enrollment);
            let running = voice_server(&state).await;
            let pid = enroll_person(&running, &enrollment).await;
            let saved = running.client.post(running.url(&format!("/api/people/{pid}/voice")))
                .json(&json!({"revision":1,"meeting_id":enrollment,"track_id":"speaker_0","segment_ids":["a","b"],"consent":true}))
                .send().await.unwrap();
            assert_eq!(saved.status(), 200);
            if case == "disabled" {
                state.store().set_recognition(false).unwrap();
            }
            if case == "incompatible" {
                let mut profile = state.store().voice_profiles().unwrap()[&pid].clone();
                profile["model"]["revision"] = json!("legacy-space");
                let profile = serde_json::from_value(profile).unwrap();
                let mut store = state.store();
                let generation = store.voice_generation(&pid).unwrap().unwrap();
                assert!(
                    store
                        .publish_voice(
                            &JobId::parse(&enrollment).unwrap(),
                            generation,
                            &profile,
                            |_| Ok::<_, speakerdesk_core::error::StoreError>(true)
                        )
                        .unwrap()
                );
            }
            let profiles = state.store().voice_profiles().unwrap();
            let jid = "8".repeat(32);
            voice_job(&state, &jid);
            let id = JobId::parse(&jid).unwrap();
            let mut job = state.store().get(&id, false).unwrap().unwrap();
            let mut document = job.document.take().unwrap();
            if case == "introduction" {
                document.segments[0].text = "My name is Priya.".into();
            }
            if case == "checked" {
                job.extra.insert(
                    "voice_checks".into(),
                    json!({"speaker_0":{"status":"manual","unknown":"keep"}}),
                );
            }
            job.status = "uploaded".into();
            job.language = "en".into();
            state.insert_job(job);
            let replay = Arc::new(VoiceReplay {
                document: Some(document),
                ..Default::default()
            });
            state.set_models(replay.clone());
            let response = running
                .client
                .post(running.url(&format!("/api/jobs/{jid}/run")))
                .send()
                .await
                .unwrap();
            assert_eq!(
                response.status(),
                202,
                "case {case}: {}",
                response.text().await.unwrap()
            );
            let waiting = state.clone();
            tokio::task::spawn_blocking(move || waiting.drain())
                .await
                .unwrap();
            let job = state.store().get(&id, false).unwrap().unwrap();
            let checks = &job.extra.get("voice_checks").cloned().unwrap_or_default();
            let expected_calls = if matches!(case, "match" | "introduction") {
                2
            } else {
                0
            };
            assert_eq!(
                replay.calls.load(Ordering::SeqCst),
                expected_calls,
                "case {case}"
            );
            assert_eq!(job.status, "ready");
            if case == "match" {
                let assignment = &job.extra["speaker_assignments"]["speaker_0"];
                assert_eq!(assignment["person_id"], pid);
                assert_eq!(assignment["source"], "automatic_voice");
                assert_eq!(assignment["confirmed"], false);
                assert_eq!(assignment["evidence"]["clips"].as_array().unwrap().len(), 2);
                assert_eq!(job.document.as_ref().unwrap().speakers["speaker_0"], "Ada");
                assert_eq!(checks["speaker_0"]["status"], "matched");
                assert_eq!(job.revision, 4); // processing, transcript, checking, decision
            } else {
                assert!(job.extra.get("speaker_assignments").is_none());
                assert_eq!(
                    job.document.as_ref().unwrap().speakers["speaker_0"],
                    "Speaker 1"
                );
                if case == "introduction" {
                    assert_eq!(checks["speaker_0"]["status"], "unknown");
                }
                if case == "checked" {
                    assert_eq!(checks["speaker_0"]["unknown"], "keep");
                }
            }
            assert_eq!(state.store().voice_profiles().unwrap(), profiles);
            assert!(!state.admission.busy());
            assert_eq!(
                running
                    .client
                    .post(running.url(&format!("/api/jobs/{jid}/run")))
                    .send()
                    .await
                    .unwrap()
                    .status(),
                409
            );
            assert_eq!(replay.calls.load(Ordering::SeqCst), expected_calls);
            running.shutdown().await;
        }
    }

    #[tokio::test]
    async fn automatic_checks_reconcile_concurrent_changes_and_keep_admission_until_return() {
        use speakerdesk_core::{ids::JobId, store::JobStore};
        use std::sync::{Arc, Condvar, Mutex, atomic::Ordering, mpsc};
        for case in [
            "words",
            "bounds",
            "name",
            "forget",
            "unrelated_legacy",
            "replace",
            "off_on",
            "cancel",
            "shutdown",
            "model",
            "audio",
        ] {
            let root = tempfile::tempdir().unwrap();
            let state = AppState::open(root.path()).unwrap();
            state.set_issues(Vec::new());
            state.set_models(Arc::new(VoiceReplay::default()));
            let enrollment = "7".repeat(32);
            voice_job(&state, &enrollment);
            let running = voice_server(&state).await;
            let pid = enroll_person(&running, &enrollment).await;
            assert_eq!(running.client.post(running.url(&format!("/api/people/{pid}/voice")))
                .json(&json!({"revision":1,"meeting_id":enrollment,"track_id":"speaker_0","segment_ids":["a","b"],"consent":true}))
                .send().await.unwrap().status(), 200);
            let legacy = if case == "unrelated_legacy" {
                let legacy_id = "6".repeat(32);
                voice_job(&state, &legacy_id);
                let other_pid = enroll_person(&running, &legacy_id).await;
                let mut store = state.store();
                let mut profile: speakerdesk_core::voice::VoiceProfile =
                    serde_json::from_value(store.voice_profiles().unwrap()[&pid].clone()).unwrap();
                profile.person_id = other_pid.clone();
                profile.model.revision = "legacy-space".into();
                let generation = store.voice_generation(&other_pid).unwrap().unwrap();
                assert!(
                    store
                        .publish_voice(
                            &JobId::parse(&legacy_id).unwrap(),
                            generation,
                            &profile,
                            |_| Ok::<_, speakerdesk_core::error::StoreError>(true)
                        )
                        .unwrap()
                );
                Some(other_pid)
            } else {
                None
            };
            let jid = "8".repeat(32);
            voice_job(&state, &jid);
            let id = JobId::parse(&jid).unwrap();
            let mut job = state.store().get(&id, false).unwrap().unwrap();
            let mut document = job.document.take().unwrap();
            // A second eligible track proves cancellation also drains unstarted checks.
            document
                .speakers
                .insert("speaker_1".into(), "Speaker 2".into());
            let mut second = document.segments.clone();
            for segment in &mut second {
                segment.id.push('2');
                segment.speaker = "speaker_1".into();
            }
            document.segments.extend(second);
            job.status = "uploaded".into();
            job.language = "en".into();
            job.extra.remove("voice_checks");
            job.extra
                .insert("retained_metadata".into(), json!({"keep":true}));
            state.insert_job(job);
            let (entered, receive) = mpsc::sync_channel(1);
            let gate = Arc::new((Mutex::new(false), Condvar::new()));
            let replay = Arc::new(VoiceReplay {
                document: Some(document),
                gate: Some(VoiceGate {
                    entered,
                    released: gate.clone(),
                }),
                wrong_identity: case == "model",
                ..Default::default()
            });
            state.set_models(replay.clone());
            assert_eq!(
                running
                    .client
                    .post(running.url(&format!("/api/jobs/{jid}/run")))
                    .send()
                    .await
                    .unwrap()
                    .status(),
                202
            );
            tokio::task::spawn_blocking(move || {
                receive.recv_timeout(Duration::from_secs(5)).unwrap()
            })
            .await
            .unwrap();
            let current = state.store().get(&id, false).unwrap().unwrap();
            assert_eq!(current.status, "ready");
            let detail: Value = running
                .client
                .get(running.url(&format!("/api/jobs/{jid}")))
                .send()
                .await
                .unwrap()
                .json()
                .await
                .unwrap();
            assert_eq!(detail["inference_owned"], true);
            assert!(state.admission.busy());
            match case {
                "words" | "bounds" => {
                    let changes = if case == "words" {
                        json!({"text":"Protected words."})
                    } else {
                        json!({"start":0.1})
                    };
                    assert_eq!(
                        running
                            .client
                            .patch(running.url(&format!("/api/jobs/{jid}/segments/a")))
                            .json(&json!({"segment_revision":0,"changes":changes}))
                            .send()
                            .await
                            .unwrap()
                            .status(),
                        200
                    );
                }
                "name" => {
                    assert_eq!(
                        running
                            .client
                            .post(
                                running
                                    .url(&format!("/api/jobs/{jid}/speakers/speaker_0/identity"))
                            )
                            .json(&json!({"revision":current.revision,"name":"Chosen name"}))
                            .send()
                            .await
                            .unwrap()
                            .status(),
                        200
                    );
                }
                "forget" => {
                    assert_eq!(
                        running
                            .client
                            .delete(running.url(&format!("/api/people/{pid}/voice")))
                            .send()
                            .await
                            .unwrap()
                            .status(),
                        200
                    );
                }
                "unrelated_legacy" => {
                    assert_eq!(
                        running
                            .client
                            .delete(running.url(&format!("/api/people/{}/voice", legacy.unwrap())))
                            .send()
                            .await
                            .unwrap()
                            .status(),
                        200
                    );
                }
                "replace" => {
                    let mut store = state.store();
                    let mut profile: speakerdesk_core::voice::VoiceProfile =
                        serde_json::from_value(store.voice_profiles().unwrap()[&pid].clone())
                            .unwrap();
                    profile.version = "replacement-profile".into();
                    let generation = store.voice_generation(&pid).unwrap().unwrap();
                    assert!(
                        store
                            .publish_voice(&id, generation, &profile, |_| Ok::<
                                _,
                                speakerdesk_core::error::StoreError,
                            >(
                                true
                            ))
                            .unwrap()
                    );
                }
                "off_on" => {
                    for enabled in [false, true] {
                        assert_eq!(
                            running
                                .client
                                .patch(running.url("/api/recognition"))
                                .json(&json!({"enabled":enabled}))
                                .send()
                                .await
                                .unwrap()
                                .status(),
                            200
                        );
                    }
                }
                "cancel" => {
                    assert_eq!(
                        running
                            .client
                            .post(running.url(&format!("/api/jobs/{jid}/cancel")))
                            .send()
                            .await
                            .unwrap()
                            .status(),
                        200
                    );
                }
                "shutdown" => {
                    state.cancel_all().unwrap();
                }
                "audio" => {
                    std::fs::OpenOptions::new()
                        .write(true)
                        .open(state.data_dir().join(&jid).join("audio.wav"))
                        .unwrap()
                        .set_len(48)
                        .unwrap();
                }
                "model" => {}
                _ => unreachable!(),
            }
            let profiles = state.store().voice_profiles().unwrap();
            assert!(state.admission.ensure_idle().is_err());
            let during = state.store().get(&id, false).unwrap().unwrap();
            if matches!(case, "cancel" | "shutdown") {
                assert!(
                    during.extra["operation_generation"].as_u64()
                        > current.extra["operation_generation"].as_u64()
                );
            }
            *gate.0.lock().unwrap() = true;
            gate.1.notify_all();
            let waiting = state.clone();
            tokio::task::spawn_blocking(move || waiting.drain())
                .await
                .unwrap();
            let saved = state.store().get(&id, false).unwrap().unwrap();
            assert_eq!(saved.status, "ready");
            assert_eq!(saved.extra["retained_metadata"]["keep"], true);
            assert!(!state.admission.busy());
            assert_eq!(state.store().voice_profiles().unwrap(), profiles);
            for track in ["speaker_0", "speaker_1"] {
                assert_ne!(
                    saved.extra["voice_checks"][track]["status"], "checking",
                    "case {case}, {track}"
                );
            }
            let doc = saved.document.unwrap();
            if case == "name" {
                assert_eq!(doc.speakers["speaker_0"], "Chosen name");
            } else if !matches!(case, "words" | "unrelated_legacy") {
                assert_eq!(doc.speakers["speaker_0"], "Speaker 1", "case {case}");
            }
            if case == "words" {
                assert_eq!(doc.segments[0].text, "Protected words.");
            }
            let expected_calls = match case {
                "words" | "bounds" | "name" | "unrelated_legacy" => 4,
                "forget" | "replace" | "off_on" | "model" => 2,
                _ => 1,
            };
            assert_eq!(
                replay.calls.load(Ordering::SeqCst),
                expected_calls,
                "case {case}"
            );
            let expected = match case {
                "words" | "unrelated_legacy" => "matched",
                "name" => "manual",
                "model" | "audio" => "unavailable",
                _ => "cancelled",
            };
            assert_eq!(
                saved.extra["voice_checks"]["speaker_0"]["status"], expected,
                "case {case}"
            );
            if case == "unrelated_legacy" {
                assert_eq!(
                    saved.extra["voice_checks"]["speaker_1"]["status"],
                    "matched"
                );
                assert_eq!(
                    saved.extra["speaker_assignments"]["speaker_1"]["person_id"],
                    pid
                );
            }
            running.shutdown().await;
        }
    }

    #[derive(Default)]
    // Deterministic model double: exercises HTTP/store races, never acoustic quality.
    struct VoiceReplay {
        document: Option<speakerdesk_core::document::Document>,
        calls: std::sync::atomic::AtomicUsize,
        starts: std::sync::Mutex<Vec<f32>>,
        gate: Option<VoiceGate>,
        wrong_identity: bool,
    }
    struct VoiceGate {
        entered: std::sync::mpsc::SyncSender<()>,
        released: std::sync::Arc<(std::sync::Mutex<bool>, std::sync::Condvar)>,
    }
    impl crate::routes::ApplicationModels for VoiceReplay {
        fn infer(
            &self,
            _: &speakerdesk_core::document::Job,
            _: &tokio_util::sync::CancellationToken,
        ) -> Result<speakerdesk_core::document::Document, String> {
            self.document
                .clone()
                .ok_or_else(|| "Voice replay only".into())
        }
        fn voice_policy(&self) -> Option<speakerdesk_core::voice::Calibration> {
            Some(crate::native::native_voice_policy())
        }
        fn embed_voice(
            &self,
            pcm: &[f32],
            _: &tokio_util::sync::CancellationToken,
        ) -> Result<speakerdesk_core::ipc::VoiceEmbedding, String> {
            let count = self.calls.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
            assert_eq!(pcm.len(), 48000);
            self.starts.lock().unwrap().push(pcm[0]);
            if count == 0
                && let Some(gate) = &self.gate
            {
                gate.entered.send(()).unwrap();
                let (lock, wake) = &*gate.released;
                let (released, timeout) = wake
                    .wait_timeout_while(lock.lock().unwrap(), Duration::from_secs(10), |ready| {
                        !*ready
                    })
                    .unwrap();
                assert!(
                    *released && !timeout.timed_out(),
                    "test gate was never released"
                );
            }
            let mut vector = vec![0.; 192];
            vector[0] = 1.;
            let mut model = crate::native::native_voice_policy().model;
            if self.wrong_identity {
                model.revision = "changed".into();
            }
            Ok(speakerdesk_core::ipc::VoiceEmbedding {
                model,
                clip: speakerdesk_core::voice::ClipEmbedding {
                    vector,
                    clean: true,
                },
            })
        }
    }

    fn voice_job(state: &AppState, jid: &str) {
        let mut job = intro_job(jid);
        job["kind"] = json!("import");
        job["duration"] = json!(9.);
        job["document"]["segments"] = json!([
            {"id":"a","speaker":"speaker_0","start":0.,"end":3.,"text":"First passage.","review":false},
            {"id":"b","speaker":"speaker_0","start":4.,"end":7.,"text":"Second passage.","review":false}
        ]);
        write_job(state, jid, job);
        let directory = state.data_dir().join(jid);
        std::fs::create_dir_all(&directory).unwrap();
        let mut wav = hound::WavWriter::create(
            directory.join("audio.wav"),
            hound::WavSpec {
                channels: 1,
                sample_rate: 16000,
                bits_per_sample: 16,
                sample_format: hound::SampleFormat::Int,
            },
        )
        .unwrap();
        for sample in 0..9 * 16000 {
            wav.write_sample(if sample < 4 * 16000 {
                3276_i16
            } else {
                6553_i16
            })
            .unwrap();
        }
        wav.finalize().unwrap();
    }
    fn source_voice_job(state: &AppState, jid: &str) {
        voice_job(state, jid);
        let catalog = json!({"schema_version":1,"job_id":jid,"sample_rate":16000,"clock":"recording_samples","policy":"aligned_aec_microphone_and_original_system_v1","sources":[{"source_id":"microphone_clean","basename":"microphone_clean.wav","revision":1},{"source_id":"system","basename":"system.wav","revision":1}]});
        let hash = speakerdesk_core::memo::sha256_hex(&serde_json::to_vec(&catalog).unwrap());
        let id = speakerdesk_core::ids::JobId::parse(jid).unwrap();
        let mut job = state.store().get(&id, false).unwrap().unwrap();
        job.extra
            .insert("sources".into(), json!(["microphone", "system"]));
        job.extra.insert("capture_source_catalog".into(), catalog);
        for (row, source) in job
            .document
            .as_mut()
            .unwrap()
            .segments
            .iter_mut()
            .zip(["microphone_clean", "system"])
        {
            row.capture_source =
                Some(json!({"catalog_sha256":hash,"source_id":source,"source_revision":1}));
        }
        state.insert_job(job);
        for (name, value) in [("microphone_clean.wav", 1000i16), ("system.wav", 2000i16)] {
            let mut wav = hound::WavWriter::create(
                state.data_dir().join(jid).join(name),
                hound::WavSpec {
                    channels: 1,
                    sample_rate: 16000,
                    bits_per_sample: 16,
                    sample_format: hound::SampleFormat::Int,
                },
            )
            .unwrap();
            for _ in 0..9 * 16000 {
                wav.write_sample(value).unwrap();
            }
            wav.finalize().unwrap();
        }
    }

    #[tokio::test]
    async fn source_bound_voice_preview_enrollment_and_manual_check_use_each_original_crop() {
        let state = AppState::new();
        let replay = std::sync::Arc::new(VoiceReplay::default());
        state.set_models(replay.clone());
        let jid = "6".repeat(32);
        source_voice_job(&state, &jid);
        let running = voice_server(&state).await;
        let pid = enroll_person(&running, &jid).await;
        for (clip, start, sample) in [("a", 0u64, 1000i16), ("b", 4u64 * 16000, 2000i16)] {
            let offset = 44 + 2 * start;
            let result = running
                .client
                .get(running.url(&format!(
                    "/api/jobs/{jid}/speakers/speaker_0/voice-preview?revision=1&clip={clip}"
                )))
                .header("Range", format!("bytes={offset}-{}", offset + 1))
                .send()
                .await
                .unwrap();
            assert_eq!(result.status(), 206);
            assert_eq!(
                result.bytes().await.unwrap().as_ref(),
                &sample.to_le_bytes()
            );
        }
        let saved=running.client.post(running.url(&format!("/api/people/{pid}/voice"))).json(&json!({"revision":1,"meeting_id":jid,"track_id":"speaker_0","segment_ids":["a","b"],"consent":true})).send().await.unwrap();
        assert_eq!(saved.status(), 200, "{}", saved.text().await.unwrap());
        let second = "5".repeat(32);
        source_voice_job(&state, &second);
        let checked = running
            .client
            .post(running.url(&format!(
                "/api/jobs/{second}/speakers/speaker_0/voice-suggestion"
            )))
            .json(&json!({"revision":0,"segment_ids":["a","b"]}))
            .send()
            .await
            .unwrap();
        assert_eq!(checked.status(), 200);
        assert_eq!(checked.json::<Value>().await.unwrap()["matched"], true);
        assert_eq!(
            *replay.starts.lock().unwrap(),
            vec![
                1000. / 32768.,
                2000. / 32768.,
                1000. / 32768.,
                2000. / 32768.
            ]
        );
        running.shutdown().await;
    }

    async fn voice_server(state: &AppState) -> Running {
        let mut running = serve(crate::router(state.clone())).await;
        let headers = reqwest::header::HeaderMap::from_iter([(
            crate::http::token_header()
                .parse::<reqwest::header::HeaderName>()
                .unwrap(),
            state.token().parse().unwrap(),
        )]);
        running.client = reqwest::Client::builder()
            .default_headers(headers)
            .build()
            .unwrap();
        running
    }
    async fn enroll_person(running: &Running, jid: &str) -> String {
        let person: Value = running
            .client
            .post(running.url("/api/people"))
            .json(&json!({"name":"Ada"}))
            .send()
            .await
            .unwrap()
            .json()
            .await
            .unwrap();
        let pid = person["id"].as_str().unwrap().to_owned();
        let assigned = running
            .client
            .post(running.url(&format!("/api/jobs/{jid}/speakers/speaker_0/identity")))
            .json(&json!({"revision":0,"person_id":pid}))
            .send()
            .await
            .unwrap();
        assert_eq!(assigned.status(), 200);
        pid
    }

    #[tokio::test]
    async fn identity_changes_reach_the_job_and_export_authority() {
        let state = AppState::new();
        let root = state.data_dir().to_path_buf();
        assert!(
            AppState::open(&root).is_err(),
            "one process owns the storage root"
        );
        let id = "9".repeat(32);
        let mut seeded = intro_job(&id);
        seeded["fast_history"] = json!({"private": "retained"});
        seeded["operation_generation"] = json!(4);
        state.insert_job(serde_json::from_value(seeded).unwrap());
        let running = serve(crate::routes::router(state.clone())).await;
        let response = running
            .client
            .post(running.url(&format!("/api/jobs/{id}/speakers/speaker_0/identity")))
            .header(crate::http::token_header(), state.token())
            .json(&json!({"revision": 0, "name": "Ada"}))
            .send()
            .await
            .unwrap();
        assert_eq!(response.status(), 200);
        let named: Value = response.json().await.unwrap();
        assert!(named.get("fast_history").is_none());
        assert!(named.get("operation_generation").is_none());
        let saved: Value = running
            .client
            .get(running.url(&format!("/api/jobs/{id}")))
            .send()
            .await
            .unwrap()
            .json()
            .await
            .unwrap();
        assert_eq!(saved["document"]["speakers"]["speaker_0"], "Ada");
        assert_eq!(saved["revision"], 1);
        let exported = running
            .client
            .get(running.url(&format!("/api/jobs/{id}/export/txt")))
            .send()
            .await
            .unwrap()
            .text()
            .await
            .unwrap();
        assert!(exported.contains("Ada:"));
        let edited = running
            .client
            .patch(running.url(&format!("/api/jobs/{id}/segments/intro")))
            .header(crate::http::token_header(), state.token())
            .json(&json!({"segment_revision": 0, "changes": {"text": "Edited words", "end": 2.5}}))
            .send()
            .await
            .unwrap();
        assert_eq!(edited.status(), 200);
        assert_eq!(edited.json::<Value>().await.unwrap()["revision"], 2);
        let created = running
            .client
            .post(running.url("/api/people"))
            .header(crate::http::token_header(), state.token())
            .json(&json!({"name": "Ada"}))
            .send()
            .await
            .unwrap();
        assert_eq!(created.status(), 201);
        running.shutdown().await;
        drop(state);
        let reopened = AppState::open(&root).unwrap();
        let internal = reopened
            .store()
            .get(&speakerdesk_core::ids::JobId::parse(&id).unwrap(), false)
            .unwrap()
            .unwrap();
        assert_eq!(internal.extra["fast_history"]["private"], "retained");
        let running = serve(crate::routes::router(reopened.clone())).await;
        let saved: Value = running
            .client
            .get(running.url(&format!("/api/jobs/{id}")))
            .send()
            .await
            .unwrap()
            .json()
            .await
            .unwrap();
        assert_eq!(saved["revision"], 3);
        assert_eq!(saved["voice_checks"]["speaker_1"]["status"], "cancelled");
        assert_eq!(saved["document"]["speakers"]["speaker_0"], "Ada");
        assert_eq!(saved["document"]["segments"][0]["text"], "Edited words");
        assert_eq!(saved["document"]["segments"][0]["end"], 2.5);
        let listed: Value = running
            .client
            .get(running.url("/api/people"))
            .send()
            .await
            .unwrap()
            .json()
            .await
            .unwrap();
        assert_eq!(listed["people"][0]["name"], "Ada");
        for kind in ["txt", "srt", "vtt", "json"] {
            let text = running
                .client
                .get(running.url(&format!("/api/jobs/{id}/export/{kind}")))
                .send()
                .await
                .unwrap()
                .text()
                .await
                .unwrap();
            assert!(text.contains("Edited words") && text.contains("Ada"));
        }
        let stale = running
            .client
            .put(running.url(&format!("/api/jobs/{id}/transcript")))
            .header(crate::http::token_header(), reopened.token())
            .json(&json!({"revision": 1, "document": saved["document"]}))
            .send()
            .await
            .unwrap();
        assert_eq!(stale.status(), 409);
        running.shutdown().await;
        drop(reopened);
        let reopened = AppState::open(&root).unwrap();
        assert_eq!(
            reopened
                .store()
                .get(&speakerdesk_core::ids::JobId::parse(&id).unwrap(), false)
                .unwrap()
                .unwrap()
                .revision,
            3
        );
        drop(reopened);
        std::fs::remove_dir_all(root).unwrap();
    }

    #[tokio::test]
    async fn route_table_accepts_people_methods_and_parent_rejects_bad_tokens() {
        // `crate::router` is `speakerdesk_host::router`: the parent applies the token guard.
        let running = serve(crate::router(AppState::new())).await;
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
        assert_eq!(suggestions["revision"], 1);
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
            .json(&json!({ "revision": 1, "suggestion_id": sid }))
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
        assert_eq!(confirmed["revision"], 2);
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
            .json(&json!({ "revision": 2, "suggestion_id": sid }))
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
