//! Native-owned copies survive host loss and its changing HTTP origin.
use crate::RuntimeChild;
use speakerdesk_host::recovery::{RecoveryStore, MAX_BYTES};
use std::{
    path::Path,
    sync::Mutex,
    time::{Duration, SystemTime, UNIX_EPOCH},
};
use tauri::{AppHandle, Manager};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct Request {
    sequence: u64,
    generation: u64,
    closing: bool,
}
#[derive(Default)]
struct State {
    sequence: u64,
    pending: Option<Request>,
    approved: bool,
    last_saved: Option<(u64, String)>,
}
impl State {
    fn begin(&mut self, generation: u64, closing: bool) -> Option<Request> {
        if self.approved
            || self.pending.is_some_and(|r| r.closing)
            || (!closing && self.pending.is_some())
        {
            return None;
        }
        self.sequence += 1;
        let request = Request {
            sequence: self.sequence,
            generation,
            closing,
        };
        self.pending = Some(request);
        Some(request)
    }
    fn owns(&self, request: Request, generation: u64) -> bool {
        self.pending == Some(request) && request.generation == generation
    }
    fn finish(&mut self, request: Request, generation: u64, result: &str) -> bool {
        if !self.owns(request, generation) {
            return false;
        }
        self.pending = None;
        self.last_saved = Some((generation, result.into()));
        if request.closing {
            self.approved = true;
        }
        true
    }
    fn fail(&mut self, request: Request) -> bool {
        if self.pending != Some(request) {
            return false;
        }
        self.pending = None;
        true
    }
    fn failure_callback(&mut self, request: Request, generation: u64) -> Option<bool> {
        if !self.fail(request) {
            return None;
        }
        if request.closing || request.generation == generation {
            Some(request.closing)
        } else {
            None
        }
    }
}
#[derive(Debug, PartialEq)]
enum Completion {
    Ignored,
    Periodic,
    Close,
}
pub struct Recovery {
    session: u128,
    store: RecoveryStore,
    state: Mutex<State>,
    writer: Mutex<()>,
}
impl Recovery {
    pub fn open(storage: &Path) -> std::io::Result<Self> {
        let stamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_nanos();
        Ok(Self {
            session: stamp ^ ((std::process::id() as u128) << 64),
            store: RecoveryStore::open(storage)?,
            state: Mutex::new(State::default()),
            writer: Mutex::new(()),
        })
    }
    pub fn approved(&self) -> bool {
        self.state
            .lock()
            .unwrap_or_else(|p| p.into_inner())
            .approved
    }
    fn complete(
        &self,
        request: Request,
        result: &str,
        host_page: bool,
        generation: impl Fn() -> u64,
    ) -> Result<Completion, String> {
        // Disk writes serialize, but timeout/close ownership remains responsive.
        let _writer = self.writer.lock().unwrap_or_else(|p| p.into_inner());
        let state = self.state.lock().unwrap_or_else(|p| p.into_inner());
        if !state.owns(request, generation()) {
            return Ok(Completion::Ignored);
        }
        let unchanged = !request.closing
            && state
                .last_saved
                .as_ref()
                .is_some_and(|(g, s)| *g == request.generation && s == result);
        drop(state);
        if !unchanged {
            if result.len() > MAX_BYTES {
                return Err("The private draft recovery copy is larger than 5 MB.".into());
            }
            let value: serde_json::Value = serde_json::from_str(result)
                .map_err(|_| "The editor did not return a draft recovery copy.")?;
            if value.get("error").is_some()
                || (host_page && value.get("receiver_ready") == Some(&serde_json::json!(false)))
            {
                return Err("The open editor could not provide its recovery copy.".into());
            }
            let page = match value.get("page_id").and_then(serde_json::Value::as_str) {
                Some(page)
                    if page.len() == 32
                        && page
                            .bytes()
                            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte)) =>
                {
                    u128::from_str_radix(page, 16).map_err(|_| "Invalid recovery page identity.")?
                }
                None if value.get("job_id").is_none_or(serde_json::Value::is_null) => 0,
                _ => return Err("The editor returned an invalid recovery page identity.".into()),
            };
            let session = format!("{:032x}", self.session ^ page);
            self.store
                .save(&session, request.sequence, &value)
                .map_err(|_| "Could not save the private draft recovery copy.")?;
        }
        let mut state = self.state.lock().unwrap_or_else(|p| p.into_inner());
        if !state.finish(request, generation(), result) {
            return Ok(Completion::Ignored);
        }
        Ok(if request.closing {
            Completion::Close
        } else {
            Completion::Periodic
        })
    }
}
fn generation(app: &AppHandle) -> u64 {
    app.state::<RuntimeChild>()
        .0
        .lock()
        .unwrap_or_else(|p| p.into_inner())
        .generation
}
pub fn start(app: AppHandle) {
    tauri::async_runtime::spawn(async move {
        loop {
            tokio::time::sleep(Duration::from_secs(1)).await;
            if app.get_webview_window("main").is_none() {
                break;
            }
            request(&app, false);
        }
    });
}
pub fn close(app: &AppHandle) {
    request(app, true);
}
fn request(app: &AppHandle, closing: bool) {
    let Some(window) = app.get_webview_window("main") else {
        return;
    };
    let current = generation(app);
    let recovery = app.state::<Recovery>();
    let request = recovery
        .state
        .lock()
        .unwrap_or_else(|p| p.into_inner())
        .begin(current, closing);
    let Some(request) = request else { return };
    let script=format!("(() => {{ try {{ return window.speakerdeskRecoverySnapshot ? window.speakerdeskRecoverySnapshot({closing}) : {{schema_version:1,job_id:null,has_unsaved:false,receiver_ready:false}}; }} catch(error) {{ return {{error:String(error)}}; }} }})()");
    let handle = app.clone();
    if window
        .eval_with_callback(script, move |result| {
            let handle = handle.clone();
            tauri::async_runtime::spawn_blocking(move || complete(&handle, request, &result));
        })
        .is_err()
    {
        failed(
            app,
            request,
            "Could not read the open editor. Keep Speakerdesk open and copy your corrections.",
        );
        return;
    }
    let handle = app.clone();
    tauri::async_runtime::spawn(async move {
        tokio::time::sleep(Duration::from_secs(5)).await;
        failed(&handle,request,"Draft recovery did not respond. Speakerdesk stays open; copy your corrections before trying again.");
    });
}
fn complete(app: &AppHandle, request: Request, result: &str) {
    let host_page = app
        .get_webview_window("main")
        .and_then(|window| window.url().ok())
        .is_some_and(|url| url.scheme() == "http");
    match app
        .state::<Recovery>()
        .complete(request, result, host_page, || generation(app))
    {
        Ok(Completion::Ignored) => (),
        Ok(completion) => {
            if let Some(window) = app.get_webview_window("main") {
                let _ = window.eval("window.speakerdeskRecoveryAcknowledged?.()");
            }
            if completion == Completion::Close {
                app.exit(0);
            }
        }
        Err(error) => failed(
            app,
            request,
            &format!("{error} Speakerdesk stays open. Copy your corrections before trying again."),
        ),
    }
}
fn failed(app: &AppHandle, request: Request, message: &str) {
    let recovery = app.state::<Recovery>();
    let closing = recovery
        .state
        .lock()
        .unwrap_or_else(|p| p.into_inner())
        .failure_callback(request, generation(app));
    let Some(closing) = closing else { return };
    if let Some(window) = app.get_webview_window("main") {
        let encoded = serde_json::to_string(message).unwrap();
        let _ = window.eval(format!(
            "window.speakerdeskRecoveryFailed?.({encoded}, {})",
            closing
        ));
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn close_timeout_releases_its_old_page_but_periodic_failure_ignores_replacement() {
        let mut state = State::default();
        let periodic = state.begin(1, false).unwrap();
        assert_eq!(state.failure_callback(periodic, 2), None);
        assert!(state.pending.is_none());
        let close = state.begin(1, true).unwrap();
        assert_eq!(state.failure_callback(close, 2), Some(true));
        assert!(!state.approved);
        assert!(state.pending.is_none());
        let newer = state.begin(2, true).unwrap();
        assert_eq!(state.failure_callback(close, 2), None);
        assert_eq!(state.pending, Some(newer));
    }
    #[test]
    fn stale_callbacks_cannot_write_or_approve_close_and_restart_keeps_old_copies() {
        let root = std::env::temp_dir().join(format!(
            "speakerdesk-native-recovery-{}-{}",
            std::process::id(),
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        std::fs::create_dir(&root).unwrap();
        let recovery = Recovery::open(&root).unwrap();
        let dirty=serde_json::json!({"schema_version":1,"receiver_ready":true,"page_id":"a".repeat(32),"job_id":"c".repeat(32),"has_unsaved":true,"document":{"segments":[{"text":"Exact human correction"}]}}).to_string();
        let periodic = recovery.state.lock().unwrap().begin(1, false).unwrap();
        let close = recovery.state.lock().unwrap().begin(1, true).unwrap();
        assert_eq!(
            recovery.complete(periodic, &dirty, true, || 1).unwrap(),
            Completion::Ignored
        );
        assert!(recovery.store.list().unwrap().is_empty());
        assert!(!recovery.approved());
        assert_eq!(
            recovery.complete(close, &dirty, true, || 2).unwrap(),
            Completion::Ignored
        );
        assert!(recovery.store.list().unwrap().is_empty());
        assert!(!recovery.approved());
        recovery.state.lock().unwrap().fail(close);
        let periodic = recovery.state.lock().unwrap().begin(1, false).unwrap();
        assert_eq!(
            recovery.complete(periodic, &dirty, true, || 1).unwrap(),
            Completion::Periodic
        );
        let receipt = recovery.store.list().unwrap()[0]["id"]
            .as_str()
            .unwrap()
            .to_owned();
        let again = recovery.state.lock().unwrap().begin(1, false).unwrap();
        recovery.complete(again, &dirty, true, || 1).unwrap();
        assert_eq!(
            recovery.store.read(&receipt).unwrap()["sequence"],
            periodic.sequence,
            "Unchanged periodic copies avoid another fsync"
        );
        let mut late_dirty: serde_json::Value = serde_json::from_str(&dirty).unwrap();
        late_dirty["document"]["segments"][0]["text"] =
            serde_json::json!("Newest words while the replacement service starts");
        let old_page_new_generation = recovery.state.lock().unwrap().begin(2, false).unwrap();
        recovery
            .complete(old_page_new_generation, &late_dirty.to_string(), true, || 2)
            .unwrap();
        let clean=serde_json::json!({"schema_version":1,"receiver_ready":true,"page_id":"b".repeat(32),"job_id":"c".repeat(32),"has_unsaved":false}).to_string();
        let new = recovery.state.lock().unwrap().begin(2, false).unwrap();
        recovery.complete(new, &clean, true, || 2).unwrap();
        assert_eq!(
            recovery.store.list().unwrap().len(),
            1,
            "Clean new page cannot erase old page even after its snapshot used the new runtime generation"
        );
        assert_eq!(
            recovery.store.read(&receipt).unwrap()["snapshot"]["document"]["segments"][0]["text"],
            "Newest words while the replacement service starts"
        );
        let final_close = recovery.state.lock().unwrap().begin(2, true).unwrap();
        assert!(recovery
            .complete(final_close, "invalid", true, || 2)
            .is_err());
        assert!(!recovery.approved());
        recovery.state.lock().unwrap().fail(final_close);
        let mut new_dirty: serde_json::Value = serde_json::from_str(&dirty).unwrap();
        new_dirty["page_id"] = serde_json::json!("b".repeat(32));
        let final_close = recovery.state.lock().unwrap().begin(2, true).unwrap();
        assert_eq!(
            recovery
                .complete(final_close, &new_dirty.to_string(), true, || 2)
                .unwrap(),
            Completion::Close
        );
        assert!(recovery.approved());
        assert_eq!(recovery.store.list().unwrap().len(), 2);
        std::fs::remove_dir_all(root).unwrap();
    }
}
