//! Rust owns update policy and verified bytes. The loopback page can request only
//! fixed operations; it receives no updater IPC capability, URL, key, or file path.
use crate::exports::ExportDownloads;
use crate::update_control::{Control, Phase, MAX_FRAME_BYTES, MAX_REQUEST_ID};
use serde::Deserialize;
use serde_json::{json, Value};
use std::sync::{
    atomic::{AtomicBool, Ordering},
    Arc,
};
use std::time::{Duration, Instant};
use tauri::{AppHandle, Manager};
use tauri_plugin_dialog::DialogExt;
use tauri_plugin_updater::{Update, UpdaterExt};
use tokio::sync::mpsc;

const ENDPOINT: &str = "https://rootandruntime.com/downloads/speakerdesk/updater/latest.json";
const MAX_UPDATE_BYTES: u64 = 256 * 1024 * 1024;
const PREPARE_TIMEOUT: Duration = Duration::from_secs(30);
const STOP_TIMEOUT: Duration = Duration::from_secs(45);

#[derive(Deserialize, Debug)]
#[serde(deny_unknown_fields)]
pub struct Packet {
    op: String,
    id: u64,
    #[serde(default)]
    ok: Option<bool>,
    #[serde(default)]
    #[serde(rename = "error")]
    _error: Option<String>,
    #[serde(default)]
    preparation: Option<u64>,
}
pub enum Event {
    Packet(u64, Packet),
    Terminated(u64, bool),
    TransportLost(u64),
    Started(u64),
    Checked(u64, u64, Result<Option<Update>, String>),
    Progress(u64, u64, u64, Option<u64>),
    DownloadFinished(u64, u64),
    Downloaded(u64, u64, Result<Vec<u8>, String>),
    Installed(u64, Result<(), String>),
    Timeout(u64, Phase, u64),
}
#[derive(Clone)]
pub struct Updates {
    sender: mpsc::UnboundedSender<Event>,
    guarded: Arc<AtomicBool>,
    replacing: Arc<AtomicBool>,
}
impl Updates {
    pub fn start(app: AppHandle) -> Self {
        let (sender, receiver) = mpsc::unbounded_channel();
        let guarded = Arc::new(AtomicBool::new(false));
        let replacing = Arc::new(AtomicBool::new(false));
        let updates = Self {
            sender: sender.clone(),
            guarded: guarded.clone(),
            replacing: replacing.clone(),
        };
        tauri::async_runtime::spawn(
            Coordinator::new(app, sender, guarded, replacing).run(receiver),
        );
        updates
    }
    pub fn packet(&self, generation: u64, bytes: &[u8]) {
        if bytes.len() > MAX_FRAME_BYTES {
            return;
        }
        if let Ok(packet) = serde_json::from_slice::<Packet>(bytes) {
            if packet.id > 0
                && packet.id <= MAX_REQUEST_ID
                && matches!(
                    packet.op.as_str(),
                    "check"
                        | "download"
                        | "install"
                        | "cancel"
                        | "editor_ready"
                        | "editor_error"
                        | "reserved"
                        | "released"
                        | "shutdown_ready"
                )
            {
                let _ = self.sender.send(Event::Packet(generation, packet));
            }
        }
    }
    pub fn event(&self, event: Event) {
        let _ = self.sender.send(event);
    }
    pub fn protects_exit(&self) -> bool {
        self.guarded.load(Ordering::Acquire)
    }
    pub fn replacing_bundle(&self) -> bool {
        self.replacing.load(Ordering::Acquire)
    }
}

struct Coordinator {
    app: AppHandle,
    sender: mpsc::UnboundedSender<Event>,
    guarded: Arc<AtomicBool>,
    replacing: Arc<AtomicBool>,
    control: Control,
    generation: u64,
    selected: Option<Update>,
    verified: Option<Vec<u8>>,
    task: Option<tauri::async_runtime::JoinHandle<()>>,
    downloaded: u64,
    total: Option<u64>,
    error: Option<String>,
    export_reserved: bool,
    recovery: bool,
    epoch: u64,
}
impl Coordinator {
    fn new(
        app: AppHandle,
        sender: mpsc::UnboundedSender<Event>,
        guarded: Arc<AtomicBool>,
        replacing: Arc<AtomicBool>,
    ) -> Self {
        Self {
            app,
            sender,
            guarded,
            replacing,
            control: Control::default(),
            generation: 0,
            selected: None,
            verified: None,
            task: None,
            downloaded: 0,
            total: None,
            error: None,
            export_reserved: false,
            recovery: false,
            epoch: 0,
        }
    }
    async fn run(mut self, mut receiver: mpsc::UnboundedReceiver<Event>) {
        while let Some(event) = receiver.recv().await {
            match event {
                Event::Started(generation) => {
                    self.generation = generation;
                    self.control = Control::default();
                    if self.recovery {
                        self.control.phase = Phase::Error;
                        self.error = Some("The update could not be installed. Speakerdesk recovered its local service; you can check and retry.".into());
                        self.recovery = false;
                    } else {
                        self.control.phase = if verification_configured(&self.app) { Phase::Idle } else { Phase::Unavailable };
                        self.error = (!verification_configured(&self.app)).then(|| "Updates are unavailable in this build because its verification configuration is incomplete.".into());
                    }
                    self.status();
                }
                Event::Packet(generation, packet) if generation == self.generation => self.packet(packet),
                Event::Checked(id, epoch, result) if epoch == self.epoch && self.control.current(id) && self.control.phase == Phase::Checking => {
                    self.task = None;
                    match result {
                        Ok(Some(update)) if allowed_archive(&update) => {
                            self.selected = Some(update); self.control.phase = Phase::Available;
                        }
                        Ok(Some(_)) => self.fail("The release did not identify an approved Speakerdesk update archive."),
                        Ok(None) => self.control.phase = Phase::Current,
                        Err(error) => self.fail(&error),
                    }
                    self.status();
                }
                Event::Progress(id, epoch, downloaded, total) if epoch == self.epoch && self.control.current(id) && self.control.phase == Phase::Downloading => {
                    if downloaded > MAX_UPDATE_BYTES || total.is_some_and(|n| n > MAX_UPDATE_BYTES) {
                        self.abort_download("The update exceeds the supported download size.");
                    } else { self.downloaded = downloaded; self.total = total; }
                    self.status();
                }
                Event::DownloadFinished(id, epoch) if epoch == self.epoch && self.control.current(id) => {
                    self.control.finished_download(id); self.status();
                }
                Event::Downloaded(id, epoch, result) if epoch == self.epoch && self.control.current(id) && matches!(self.control.phase, Phase::Downloading | Phase::Verifying) => {
                    self.task = None;
                    match result {
                        Ok(bytes) if bytes.len() as u64 <= MAX_UPDATE_BYTES && self.control.verified(id) => {
                            self.downloaded = bytes.len() as u64; self.verified = Some(bytes);
                        }
                        Ok(_) => self.fail("The update exceeds the supported download size."),
                        Err(error) => self.fail(&format!("The update was not verified. Check again and retry. {error}")),
                    }
                    self.status();
                }
                Event::Timeout(id, phase, epoch) if epoch == self.epoch && self.control.current(id) && self.control.phase == phase => {
                    match phase {
                        Phase::Preparing => {
                            self.control.phase = Phase::Ready;
                            self.error = Some("Update preparation timed out; your edits were kept. Try again when editing is finished.".into());
                            self.guarded.store(false, Ordering::Release); self.status();
                        }
                        Phase::Reserving => self.release("The local service did not confirm that it was idle. Your update has not been installed."),
                        Phase::Releasing | Phase::Stopping => self.block("Speakerdesk is waiting for its local service to finish safely. No update was installed. Keep the app open; recording data has not been replaced."),
                        _ => {},
                    }
                }
                Event::Terminated(generation, clean) if generation == self.generation => {
                    self.control.terminated(clean);
                    if self.control.install() {
                        self.install();
                    } else if self.control.phase == Phase::Blocked && self.control.safe_exit() {
                        self.recover();
                    } else if matches!(self.control.phase, Phase::Stopping | Phase::Blocked | Phase::Reserving | Phase::Releasing) {
                        self.block("The local service stopped without a confirmed safe shutdown. The app was not updated. Close Speakerdesk and reopen the existing app to recover its service.");
                    } else {
                        self.stop_task(); self.verified = None; self.selected = None;
                        self.fail("Speakerdesk's local service stopped. Close the app and reopen it.");
                        self.status();
                    }
                }
                Event::TransportLost(generation) if generation == self.generation => {
                    if self.guarded.load(Ordering::Acquire) {
                        self.block("Speakerdesk lost contact with its local service before confirming safe termination. No update will be installed.");
                    } else { self.stop_task(); self.fail("Speakerdesk lost contact with its local service. Close the app and reopen it."); }
                }
                Event::Installed(id, result) if self.control.current(id) && self.control.phase == Phase::Installing => {
                    self.task = None;
                    self.replacing.store(false, Ordering::Release);
                    match result {
                        Ok(()) => { self.guarded.store(false, Ordering::Release); self.app.restart(); },
                        Err(error) => {
                            self.error = Some(format!("The update was not installed: {}", clipped(&error, 384)));
                            self.recover();
                        }
                    }
                }
                _ => {},
            }
        }
        self.stop_task();
    }
    fn packet(&mut self, packet: Packet) {
        let id = packet.id;
        match packet.op.as_str() {
            "check" if self.control.check(id) => {
                self.stop_task();
                self.selected = None;
                self.verified = None;
                self.downloaded = 0;
                self.total = None;
                self.error = None;
                if !verification_configured(&self.app) {
                    self.control.phase = Phase::Unavailable;
                    self.error = Some("Updates are unavailable in this build because its verification configuration is incomplete.".into());
                } else {
                    let app = self.app.clone();
                    let sender = self.sender.clone();
                    let epoch = self.epoch;
                    self.task = Some(tauri::async_runtime::spawn(async move {
                        let result = match app
                            .updater_builder()
                            .endpoints(vec![ENDPOINT.parse().unwrap()])
                            .and_then(|builder| builder.timeout(Duration::from_secs(15)).build())
                        {
                            Ok(updater) => updater.check().await.map_err(|e| e.to_string()),
                            Err(error) => Err(error.to_string()),
                        };
                        let _ = sender.send(Event::Checked(id, epoch, result));
                    }));
                }
                self.status();
            }
            "download" if self.selected.is_some() && self.control.download(id) => {
                self.next_epoch();
                let epoch = self.epoch;
                self.error = None;
                self.verified = None;
                self.downloaded = 0;
                self.total = None;
                let mut update = self.selected.as_ref().unwrap().clone();
                update.timeout = Some(Duration::from_secs(300));
                let sender = self.sender.clone();
                let finished = sender.clone();
                self.task = Some(tauri::async_runtime::spawn(async move {
                    let progress = sender.clone();
                    let mut bytes = 0u64;
                    let mut last = Instant::now();
                    let result = update
                        .download(
                            move |chunk, total| {
                                bytes = bytes.saturating_add(chunk as u64);
                                if last.elapsed() >= Duration::from_millis(200)
                                    || bytes > MAX_UPDATE_BYTES
                                    || total.is_some_and(|n| n > MAX_UPDATE_BYTES)
                                {
                                    let _ = progress.send(Event::Progress(id, epoch, bytes, total));
                                    last = Instant::now();
                                }
                            },
                            move || {
                                let _ = finished.send(Event::DownloadFinished(id, epoch));
                            },
                        )
                        .await
                        .map_err(|e| e.to_string());
                    let _ = sender.send(Event::Downloaded(id, epoch, result));
                }));
                self.status();
            }
            "install" if self.verified.is_some() && self.control.prepare(id) => {
                self.next_epoch();
                self.error = None;
                // Establish the fresh preparation token before any terminal
                // refusal, so the runtime can clear its pending install safely.
                self.status();
                let location = std::env::current_exe()
                    .map_err(|_| "Speakerdesk's installed app location could not be verified.")
                    .and_then(|exe| crate::update_location::preflight(&exe));
                if let Err(error) = location {
                    self.fail(error);
                    self.status();
                    return;
                }
                self.guarded.store(true, Ordering::Release);
                self.timeout(Phase::Preparing, PREPARE_TIMEOUT);
                self.status();
            }
            "editor_ready"
                if self.control.preparation_matches(id, packet.preparation)
                    && self.control.editor_ready(id, packet.preparation) =>
            {
                if !self.app.state::<ExportDownloads>().reserve_update(id) {
                    self.control.phase = Phase::Ready;
                    self.error = Some(
                        "Finish or cancel the pending export before installing this update.".into(),
                    );
                    self.guarded.store(false, Ordering::Release);
                    self.status();
                    return;
                }
                self.export_reserved = true;
                self.next_epoch();
                if self
                    .send(json!({"op":"reserve", "id":id, "preparation":self.control.preparation}))
                    .is_err()
                {
                    self.block("Speakerdesk could not contact its local service to reserve an update. No update was installed.");
                    return;
                }
                self.timeout(Phase::Reserving, PREPARE_TIMEOUT);
                self.status();
            }
            "editor_error"
                if self.control.preparation_matches(id, packet.preparation)
                    && self.control.current(id)
                    && self.control.phase == Phase::Preparing =>
            {
                self.control.phase = Phase::Ready;
                self.error = Some("Finish editing and save your transcript before installing. Your edits have been kept.".into());
                self.guarded.store(false, Ordering::Release);
                self.status();
            }
            "reserved"
                if self.control.preparation_matches(id, packet.preparation)
                    && self.control.current(id)
                    && self.control.phase == Phase::Reserving =>
            {
                if packet.ok == Some(true) && self.control.reserved(id, packet.preparation) {
                    self.next_epoch();
                    if self
                        .send(json!({"op":"shutdown", "id":id, "preparation":self.control.preparation}))
                        .is_err()
                    {
                        self.block("Speakerdesk could not confirm delivery of its shutdown request. No update will be installed.");
                    } else {
                        self.timeout(Phase::Stopping, STOP_TIMEOUT);
                        self.status();
                    }
                } else {
                    self.release("Finish recording, processing, model setup and pending saves before installing this update.");
                }
            }
            "released"
                if self.control.preparation_matches(id, packet.preparation)
                    && self.control.current(id)
                    && self.control.phase == Phase::Releasing =>
            {
                if packet.ok == Some(true) {
                    self.release_export();
                    self.control.phase = Phase::Ready;
                    self.guarded.store(false, Ordering::Release);
                    self.status();
                } else {
                    self.block("Speakerdesk could not confirm that update protection was released. No update was installed; keep the app open.");
                }
            }
            "shutdown_ready"
                if self.control.preparation_matches(id, packet.preparation)
                    && self.control.shutdown_ack(
                        id,
                        packet.preparation,
                        packet.ok == Some(true),
                    ) =>
            {
                if packet.ok != Some(true) {
                    self.block("The local service could not finish its cleanup safely. No update will be installed. Recording data has not been replaced.");
                }
                // Terminated is a separate native event. An ACK never installs.
            }
            "cancel" if self.control.current(id) && self.control.cancellable() => {
                if self.control.phase == Phase::Reserving {
                    self.release("Update installation cancelled. Your data has been kept.");
                } else {
                    self.stop_task();
                    if self.verified.is_some() {
                        self.control.phase = Phase::Ready;
                    } else if self.selected.is_some() {
                        self.control.phase = Phase::Available;
                    } else {
                        self.control.phase = Phase::Idle;
                    }
                    self.error = None;
                    self.guarded.store(false, Ordering::Release);
                    self.status();
                }
            }
            _ => {}
        }
    }
    fn status(&self) {
        let mut status = json!({"op":"status", "id":self.control.id, "state":self.control.phase.status(),
            "downloaded_bytes":self.downloaded, "reserved":self.guarded.load(Ordering::Acquire),
            "cancellable":self.control.cancellable()});
        if self.control.preparation > 0 && self.control.preparation_id == self.control.id {
            status["preparation"] = json!(self.control.preparation);
        }
        if let Some(total) = self.total {
            status["total_bytes"] = json!(total);
        }
        if let Some(update) = &self.selected {
            status["version"] = json!(clipped(&update.version, 48));
            if let Some(notes) = &update.body {
                status["notes"] = json!(clipped(notes, 512));
            }
        }
        if let Some(error) = &self.error {
            status["error"] = json!(clipped(error, 512));
        }
        let _ = self.send(status);
    }
    fn send(&self, value: Value) -> Result<(), String> {
        let mut bytes = b"SPEAKERDESK_UPDATE=".to_vec();
        bytes.extend(serde_json::to_vec(&value).map_err(|e| e.to_string())?);
        if bytes.len() + 1 > MAX_FRAME_BYTES {
            return Err("Oversized native update control".into());
        }
        bytes.push(b'\n');
        crate::write_runtime(&self.app, self.generation, &bytes)
    }
    fn fail(&mut self, error: &str) {
        self.control.phase = Phase::Error;
        self.verified = None;
        self.error = Some(clipped(error, 512));
    }
    fn next_epoch(&mut self) {
        self.epoch = self
            .epoch
            .checked_add(1)
            .expect("Update operation sequence exhausted");
    }
    fn stop_task(&mut self) {
        self.next_epoch();
        if let Some(task) = self.task.take() {
            task.abort();
        }
    }
    fn abort_download(&mut self, error: &str) {
        self.stop_task();
        self.fail(error);
    }
    fn timeout(&self, phase: Phase, timeout: Duration) {
        let id = self.control.id;
        let sender = self.sender.clone();
        let epoch = self.epoch;
        tauri::async_runtime::spawn(async move {
            tokio::time::sleep(timeout).await;
            let _ = sender.send(Event::Timeout(id, phase, epoch));
        });
    }
    fn release(&mut self, error: &str) {
        self.next_epoch();
        self.control.phase = Phase::Releasing;
        self.error = Some(error.into());
        if self
            .send(json!({"op":"release", "id":self.control.id, "preparation":self.control.preparation}))
            .is_err()
        {
            self.block("Speakerdesk could not confirm that update protection was released. No update was installed.");
        } else {
            self.timeout(Phase::Releasing, PREPARE_TIMEOUT);
            self.status();
        }
    }
    fn release_export(&mut self) {
        if self.export_reserved {
            self.app
                .state::<ExportDownloads>()
                .release_update(self.control.id);
            self.export_reserved = false;
        }
    }
    fn block(&mut self, error: &str) {
        self.control.phase = Phase::Blocked;
        self.error = Some(error.into());
        self.guarded.store(true, Ordering::Release);
        self.status();
        self.app
            .dialog()
            .message(error)
            .title("Update paused safely")
            .show(|_| {});
    }
    fn install(&mut self) {
        let Some(update) = self.selected.take() else {
            self.recover();
            return;
        };
        let Some(bytes) = self.verified.take() else {
            self.recover();
            return;
        };
        self.replacing.store(true, Ordering::Release);
        self.status();
        let id = self.control.id;
        let sender = self.sender.clone();
        // The official installer may ask for authorization on the main thread;
        // keep the main event loop free while the blocking replacement runs.
        tauri::async_runtime::spawn_blocking(move || {
            let result = update.install(&bytes).map_err(|e| e.to_string());
            let _ = sender.send(Event::Installed(id, result));
        });
    }
    fn recover(&mut self) {
        self.verified = None;
        self.selected = None;
        if !self.control.safe_exit() {
            self.block("The previous service state is unknown. Speakerdesk will not launch a second service or install the update.");
            return;
        }
        if let Err(error) = crate::recover_runtime(&self.app) {
            let paths = crate::recovery_description(&self.app);
            let installation_error = self
                .error
                .as_deref()
                .unwrap_or("The update could not be installed.");
            self.block(&format!(
                "Speakerdesk could not restore its local service. {} {} {paths}",
                clipped(&installation_error, 1024),
                clipped(&error, 256)
            ));
            return;
        }
        self.release_export();
        self.guarded.store(false, Ordering::Release);
        self.recovery = true;
        // Started from the new process resets the request namespace. All old
        // stdout/termination messages carry the old native generation.
    }
}

fn clipped(value: &str, chars: usize) -> String {
    value.chars().take(chars).collect()
}
fn verification_configured(app: &AppHandle) -> bool {
    let Some(config) = app.config().plugins.0.get("updater") else {
        return false;
    };
    config["pubkey"]
        .as_str()
        .is_some_and(|key| !key.trim().is_empty())
        && config["requireSignedVersion"].as_bool() == Some(true)
        && config["allowDowngrades"].as_bool() == Some(false)
        && [
            "dangerousInsecureTransportProtocol",
            "dangerousAcceptInvalidCerts",
            "dangerousAcceptInvalidHostnames",
        ]
        .iter()
        .all(|key| config[*key].as_bool() != Some(true))
}

fn allowed_archive(update: &Update) -> bool {
    let url = &update.download_url;
    let filename = format!("Speakerdesk_{}_AppleSilicon.app.tar.gz", update.version);
    update.version.len() <= 48
        && url.scheme() == "https"
        && url.host_str() == Some("rootandruntime.com")
        && url.port().is_none()
        && url.username().is_empty()
        && url.password().is_none()
        && url.query().is_none()
        && url.fragment().is_none()
        && url.path()
            == format!(
                "/downloads/speakerdesk/releases/{}/{}",
                update.version, filename
            )
}
