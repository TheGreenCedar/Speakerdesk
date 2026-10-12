//! The existing desktop updater protocol, attached to the native host's admission.
use std::io::Write;

use axum::{
    Json, Router,
    body::Bytes,
    extract::State,
    http::StatusCode,
    response::{IntoResponse, Response},
    routing::{get, patch},
};
use serde::Deserialize;
use serde_json::{Value, json};

use crate::{admission::Admission, routes::AppState};

pub const PREFIX: &str = "SPEAKERDESK_UPDATE=";
pub const MAX_FRAME: usize = 8192;
const MAX_ID: u64 = (1 << 53) - 1;
const STALE: &str = "This update preparation is no longer current.";

pub(crate) struct RuntimeUpdates {
    status: Value,
    connected: bool,
    pending_install: bool,
    editor_ready: bool,
    last_preparation: u64,
}

impl Default for RuntimeUpdates {
    fn default() -> Self {
        Self {
            status: json!({"id":0,"state":"unavailable","error":"App updates are available in the desktop app."}),
            connected: false,
            pending_install: false,
            editor_ready: false,
            last_preparation: 0,
        }
    }
}

pub(crate) fn emit(value: &Value) -> std::io::Result<()> {
    let frame = format!("{PREFIX}{value}\n");
    if frame.len() > MAX_FRAME {
        return Err(std::io::Error::other("Oversized update control"));
    }
    let mut out = std::io::stdout().lock();
    out.write_all(frame.as_bytes())?;
    out.flush()
}

impl RuntimeUpdates {
    pub fn connect(&mut self) {
        self.connected = true;
        self.status = json!({"id":0,"state":"idle"});
    }

    fn view(&self, admission: &Admission) -> Value {
        let mut value = self.status.clone();
        value["reserved"] = json!(admission.reserved() || value["reserved"] == true);
        value
    }

    /// Only a validated native shutdown returns true. The caller still reaps the child.
    pub fn handle(&mut self, admission: &Admission, message: Value) -> bool {
        let Some(id) = message["id"].as_u64().filter(|id| *id <= MAX_ID) else {
            return false;
        };
        if Some(id) != self.status["id"].as_u64() {
            return false;
        }
        let preparation = message["preparation"]
            .as_u64()
            .filter(|p| *p > 0 && *p <= MAX_ID);
        match message["op"].as_str() {
            Some("status") => {
                let Some(state) = message["state"].as_str() else {
                    return false;
                };
                if ![
                    "unavailable",
                    "idle",
                    "checking",
                    "current",
                    "available",
                    "downloading",
                    "verifying",
                    "ready",
                    "preparing",
                    "preparing_install",
                    "installing",
                    "stopping",
                    "shutting_down",
                    "restart_required",
                    "cancelled",
                    "error",
                ]
                .contains(&state)
                {
                    return false;
                }
                let new_preparation =
                    self.pending_install && self.status.get("preparation").is_none();
                if let Some(preparation) = preparation {
                    if new_preparation {
                        if preparation <= self.last_preparation
                            || !["preparing", "preparing_install"].contains(&state)
                        {
                            return false;
                        }
                    } else if self.status["preparation"].as_u64() != Some(preparation) {
                        return false;
                    }
                } else if self.pending_install || admission.reserved() {
                    return false;
                }
                for key in ["reserved", "cancellable"] {
                    if message.get(key).is_some_and(|v| !v.is_boolean()) {
                        return false;
                    }
                }
                for key in ["version", "error", "notes"] {
                    if message.get(key).is_some_and(|v| {
                        v.as_str()
                            .is_none_or(|s| s.len() > if key == "notes" { 4096 } else { 512 })
                    }) {
                        return false;
                    }
                }
                for key in ["downloaded_bytes", "total_bytes"] {
                    if message
                        .get(key)
                        .is_some_and(|v| v.as_u64().is_none_or(|n| n > MAX_ID))
                    {
                        return false;
                    }
                }
                if new_preparation {
                    self.last_preparation = preparation.unwrap();
                }
                let mut status = json!({"id":id,"state":state});
                for key in [
                    "reserved",
                    "cancellable",
                    "version",
                    "error",
                    "notes",
                    "downloaded_bytes",
                    "total_bytes",
                    "preparation",
                ] {
                    if let Some(value) = message.get(key) {
                        status[key] = value.clone();
                    }
                }
                self.status = status;
                if ["cancelled", "error", "available", "ready"].contains(&state)
                    && !admission.reserved()
                {
                    self.pending_install = false;
                }
            }
            Some(op @ ("reserve" | "release" | "shutdown"))
                if id > 0
                    && preparation.is_some()
                    && preparation == self.status["preparation"].as_u64() =>
            {
                let preparation = preparation.unwrap();
                let result = match op {
                    "reserve" if self.pending_install && self.editor_ready => {
                        admission.reserve(id, preparation)
                    }
                    "release" => admission.release(id, preparation),
                    "shutdown" => admission.shutdown_handshake(id, preparation),
                    _ => Err(crate::admission::AdmissionError::Stale),
                };
                if op == "shutdown" && result.is_ok() {
                    self.status["state"] = json!("shutting_down");
                    return true;
                }
                if op == "release" && result.is_ok() {
                    self.pending_install = false;
                    self.editor_ready = false;
                }
                let reply = match op {
                    "reserve" => "reserved",
                    "release" => "released",
                    _ => "shutdown_ready",
                };
                let _ = emit(
                    &json!({"id":id,"preparation":preparation,"op":reply,"ok":result.is_ok()}),
                );
            }
            _ => {}
        }
        false
    }

    fn request(&mut self, admission: &Admission, body: Operation) -> Result<Value, &'static str> {
        if !self.connected {
            return Err("App updates are available in the desktop app.");
        }
        let state = self.status["state"].as_str().unwrap_or("unavailable");
        let guarded = admission.reserved() || self.status["reserved"] == true;
        if ["check", "download", "install"].contains(&body.op.as_str()) && guarded {
            return Err(
                "Wait for the protected update operation to finish or be safely cancelled.",
            );
        }
        if ["download", "install"].contains(&body.op.as_str()) {
            admission.ensure_idle().map_err(|e| e.message())?;
        }
        if !["editor_ready", "editor_error"].contains(&body.op.as_str())
            && (body.id.is_some() || body.preparation.is_some() || body.error.is_some())
        {
            return Err("Invalid update request.");
        }
        match body.op.as_str() {
            "check" => {
                if [
                    "checking",
                    "downloading",
                    "verifying",
                    "preparing",
                    "preparing_install",
                    "stopping",
                    "installing",
                    "shutting_down",
                ]
                .contains(&state)
                {
                    return Err("Wait for the current update operation or cancel it.");
                }
                let id = self.status["id"]
                    .as_u64()
                    .unwrap_or(0)
                    .checked_add(1)
                    .filter(|id| *id <= MAX_ID)
                    .ok_or(STALE)?;
                self.pending_install = false;
                self.editor_ready = false;
                self.status = json!({"id":id,"state":"checking"});
            }
            "download" if state == "available" => {
                self.status["state"] = json!("downloading");
            }
            "install" if state == "ready" && !self.pending_install => {
                self.pending_install = true;
                self.editor_ready = false;
                self.status.as_object_mut().unwrap().remove("preparation");
                self.status["state"] = json!("preparing");
            }
            "editor_ready" | "editor_error" => {
                if !self.pending_install
                    || !["preparing", "preparing_install"].contains(&state)
                    || body.id != self.status["id"].as_u64()
                    || body.preparation.is_none()
                    || body.preparation != self.status["preparation"].as_u64()
                    || body.error.as_ref().is_some_and(|s| s.len() > 512)
                    || (body.op == "editor_ready" && body.error.is_some())
                {
                    return Err(STALE);
                }
                self.editor_ready = body.op == "editor_ready";
            }
            "cancel"
                if self.status["id"].as_u64().is_some_and(|n| n > 0)
                    && !["installing", "shutting_down", "restart_required"].contains(&state)
                    && (state != "stopping" || self.status["cancellable"] == true) => {}
            _ => return Err("The requested update operation is unavailable in the current state."),
        }
        let mut packet = json!({"op":body.op,"id":self.status["id"]});
        if let Some(preparation) = body.preparation {
            packet["preparation"] = json!(preparation);
        }
        if let Some(error) = body.error {
            packet["error"] = json!(error);
        }
        emit(&packet).map_err(|_| "The desktop update connection is unavailable.")?;
        Ok(self.view(admission))
    }
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Operation {
    op: String,
    id: Option<u64>,
    preparation: Option<u64>,
    error: Option<String>,
}

pub(crate) fn routes() -> Router<AppState> {
    Router::new()
        .route("/api/updates", get(status).post(operation))
        .route("/api/preferences/update-notice", patch(dismiss))
        .layer(axum::extract::DefaultBodyLimit::max(MAX_FRAME))
}

async fn status(State(app): State<AppState>) -> Response {
    crate::routes::blocking(move || {
        let mut view = app.updates.lock().expect("updates").view(&app.admission);
        match app.store().preference("dismissed_update_version") {
            Ok(value) => {
                view["dismissed_version"] = json!(value.unwrap_or_default());
                Json(view).into_response()
            }
            Err(_) => failure("Could not read update preferences."),
        }
    })
    .await
}

async fn operation(State(app): State<AppState>, body: Bytes) -> Response {
    let Ok(body) = serde_json::from_slice::<Operation>(&body) else {
        return failure("Invalid update request.");
    };
    crate::routes::blocking(move || {
        match app
            .updates
            .lock()
            .expect("updates")
            .request(&app.admission, body)
        {
            Ok(status) => (StatusCode::ACCEPTED, Json(status)).into_response(),
            Err(error) => failure(error),
        }
    })
    .await
}

async fn dismiss(State(app): State<AppState>, body: Bytes) -> Response {
    let Ok(body) = serde_json::from_slice::<Value>(&body) else {
        return failure("Invalid update notice.");
    };
    crate::routes::blocking(move || {
        let updates = app.updates.lock().expect("updates");
        let view = updates.view(&app.admission);
        let Some(version) = view["version"].as_str().filter(|s| !s.is_empty()) else {
            return failure("No available update notice.");
        };
        if body.as_object().is_none_or(|v| v.len() != 1)
            || body["version"] != version
            || view["state"] != "available"
            || view["reserved"] == true
        {
            return failure("Dismiss the currently available update version.");
        }
        match app
            .store()
            .set_preference("dismissed_update_version", version)
        {
            Ok(()) => Json(json!({"dismissed_version":version})).into_response(),
            Err(_) => failure("Could not save update preference."),
        }
    })
    .await
}

fn failure(error: &str) -> Response {
    (StatusCode::CONFLICT, Json(json!({"error":error}))).into_response()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::admission::Activity;

    fn operation(op: &str) -> Operation {
        Operation {
            op: op.into(),
            id: None,
            preparation: None,
            error: None,
        }
    }

    #[test]
    fn native_control_reserves_the_actual_work_owner_and_requires_current_editor_ack() {
        let admission = Admission::new();
        let mut updates = RuntimeUpdates::default();
        assert!(updates.request(&admission, operation("check")).is_err());
        updates.connect();
        updates.request(&admission, operation("check")).unwrap();
        updates.handle(
            &admission,
            json!({"op":"status","id":1,"state":"available"}),
        );
        let lease = admission.lease(Activity::Import).unwrap();
        assert!(updates.request(&admission, operation("download")).is_err());
        drop(lease);
        updates.request(&admission, operation("download")).unwrap();
        updates.handle(&admission, json!({"op":"status","id":1,"state":"ready"}));
        updates.request(&admission, operation("install")).unwrap();
        updates.handle(
            &admission,
            json!({"op":"status","id":1,"state":"preparing","preparation":1}),
        );
        updates.handle(&admission, json!({"op":"reserve","id":1,"preparation":1}));
        assert!(!admission.reserved());
        let mut ready = operation("editor_ready");
        ready.id = Some(1);
        ready.preparation = Some(2);
        assert!(updates.request(&admission, ready).is_err());
        let mut ready = operation("editor_ready");
        ready.id = Some(1);
        ready.preparation = Some(1);
        updates.request(&admission, ready).unwrap();
        let lease = admission.lease(Activity::Import).unwrap();
        updates.handle(&admission, json!({"op":"reserve","id":1,"preparation":1}));
        assert!(!admission.reserved());
        drop(lease);
        updates.handle(&admission, json!({"op":"reserve","id":1,"preparation":1}));
        assert!(admission.reserved());
        assert!(admission.lease(Activity::Import).is_err());
        assert!(!updates.handle(&admission, json!({"op":"release","id":1,"preparation":2})));
        assert!(admission.reserved());
        assert!(updates.handle(&admission, json!({"op":"shutdown","id":1,"preparation":1})));
        assert!(admission.closed());
        assert!(updates.request(&admission, operation("cancel")).is_err());
    }
}
