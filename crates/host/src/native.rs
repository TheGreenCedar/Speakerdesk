//! Production Rust service used by the existing desktop shell.
use axum::{
    Json, Router,
    body::{Body, Bytes},
    extract::{Path as RoutePath, Request, State},
    http::{StatusCode, header},
    response::{Html, IntoResponse, Response},
    routing::{get, patch},
};
use serde_json::{Value, json};
use speakerdesk_core::{
    audio,
    document::{Document, Job},
    ids::JobId,
    languages,
    models::Models,
    store::JobStore,
};
use std::{
    io::Write,
    path::{Path, PathBuf},
    sync::{Arc, Mutex, RwLock},
};
use tokio::io::{AsyncBufReadExt, AsyncReadExt, BufReader};
use tokio_util::sync::CancellationToken;

use crate::{
    admission::Activity,
    inference::InferenceSupervisor,
    native_updates,
    routes::{AppState, ApplicationModels, blocking, confined_file, media_write_directory},
};

#[derive(rust_embed::RustEmbed)]
#[folder = "../../speakerdesk/static/"]
struct Assets;

pub(crate) struct NativeModels {
    supervisor: Arc<InferenceSupervisor>,
    root: PathBuf,
    alignment: RwLock<Option<Arc<speakerdesk_core::alignment::Vocabulary>>>,
    voice: RwLock<Option<speakerdesk_core::voice::Calibration>>,
    model_root: PathBuf,
    setup: crate::setup::SetupService,
    task: Mutex<Option<(CancellationToken, tokio::task::JoinHandle<()>)>>,
}

impl ApplicationModels for NativeModels {
    fn retry_passage(
        &self,
        pcm: &[f32],
        language: &str,
        cancellation: &CancellationToken,
    ) -> Result<speakerdesk_core::import::PassageCandidate, String> {
        speakerdesk_core::import::retry_passage(
            &self.supervisor.models(cancellation.clone()),
            pcm,
            0.,
            pcm.len() as f64 / 16000.,
            language,
        )
        .map_err(|error| error.to_string())
    }
    fn voice_policy(&self) -> Option<speakerdesk_core::voice::Calibration> {
        self.voice.read().expect("voice capability").clone()
    }
    fn embed_voice(
        &self,
        pcm: &[f32],
        cancellation: &CancellationToken,
    ) -> Result<speakerdesk_core::ipc::VoiceEmbedding, String> {
        self.supervisor
            .models(cancellation.clone())
            .embed_voice(pcm)
            .map_err(|e| e.to_string())
    }
    fn infer(&self, job: &Job, cancellation: &CancellationToken) -> Result<Document, String> {
        let path = recording_path(&self.root, job).map_err(|e| e.to_string())?;
        let pcm = audio::read_pcm16(path).map_err(|e| e.to_string())?;
        let alignment = self.alignment.read().expect("alignment capability").clone();
        speakerdesk_core::import::import_recording(
            &self.supervisor.models(cancellation.clone()),
            &pcm,
            &job.language,
            alignment.as_deref(),
        )
        .map_err(|e| e.to_string())
    }
}

impl NativeModels {
    fn start(
        self: &Arc<Self>,
        app: AppState,
        download: bool,
    ) -> Result<(), crate::admission::AdmissionError> {
        let mut task = self.task.lock().expect("setup task");
        if task.as_ref().is_some_and(|(_, task)| !task.is_finished()) {
            return Ok(());
        }
        let lease = app.admission.lease(Activity::Setup)?;
        let cancel = CancellationToken::new();
        let token = cancel.clone();
        let native = self.clone();
        native.setup.begin(&crate::setup::artifacts());
        native.setup.phase(if download {
            "Checking installed models"
        } else {
            "Initializing native speech models"
        });
        if !download {
            app.set_issues(vec!["Initializing native speech models…".into()]);
        }
        let handle = tokio::spawn(async move {
            let _lease = lease;
            let result = if download {
                native.install(&app, &token).await
            } else {
                native
                    .initialize_async(app.clone(), token.clone(), false)
                    .await
            };
            if let Err(error) = &result {
                // Do not leave a first-run failure looking like an indefinitely pending load.
                if app.model_status().1.is_empty() {
                    app.set_issues(vec![error.clone()]);
                }
            }
            native.setup.finish(result);
        });
        *task = Some((cancel, handle));
        Ok(())
    }
    fn cancel(&self) {
        if let Some((cancel, task)) = self.task.lock().expect("setup task").as_ref()
            && !task.is_finished()
        {
            self.setup.phase("Cancelling model setup");
            cancel.cancel();
        }
    }
    async fn join(&self) -> Result<(), tokio::task::JoinError> {
        let task = self.task.lock().expect("setup task").take();
        if let Some((cancel, task)) = task {
            cancel.cancel();
            task.await?;
        }
        Ok(())
    }
    async fn initialize_async(
        self: &Arc<Self>,
        app: AppState,
        cancel: CancellationToken,
        all: bool,
    ) -> Result<(), String> {
        let native = self.clone();
        tokio::task::spawn_blocking(move || native.initialize(&app, &cancel, all))
            .await
            .map_err(|e| e.to_string())?
    }
    fn initialize(
        &self,
        app: &AppState,
        cancel: &CancellationToken,
        all: bool,
    ) -> Result<(), String> {
        let models = self.supervisor.models(cancel.clone());
        let mut reports = Vec::new();
        let mut failures = Vec::new();
        let mut core_failures = Vec::new();
        let mut alignment = None;
        let mut voice = None;
        for (index, artifact) in crate::setup::artifacts().iter().enumerate() {
            if cancel.is_cancelled() {
                return Err("Model setup cancelled".into());
            }
            let identity = match artifact.id.as_str() {
                "silero-speech" => format!("{}@{}", artifact.repo, &artifact.revision[..7]),
                "coarse-alignment" => speakerdesk_core::alignment::CHECKPOINT.into(),
                "redimnet2-b6" => native_voice_policy().model.revision,
                _ => format!("{}@{}", artifact.repo, artifact.revision),
            };
            self.setup.phase(format!("Initializing {}", artifact.name));
            let loaded = models
                .setup_check(std::slice::from_ref(&artifact.id))
                .map_err(|e| e.to_string())
                .and_then(|mut loaded| {
                    if loaded.len() != 1 || !loaded[0].ok || loaded[0].model != identity {
                        return Err("Model initialization returned an unexpected identity".into());
                    }
                    if artifact.id == "coarse-alignment" {
                        let tokens = std::fs::read_to_string(
                            self.model_root.join("coarse-alignment/tokens.txt"),
                        )
                        .map_err(|e| e.to_string())?;
                        alignment = Some(Arc::new(speakerdesk_core::alignment::Vocabulary::parse(
                            &tokens,
                        )?));
                    }
                    if artifact.id == "redimnet2-b6" {
                        voice = Some(native_voice_policy());
                    }
                    Ok(loaded.remove(0))
                });
            match loaded {
                Ok(report) => reports.push(report),
                Err(error) => {
                    let message = format!("{}: {error}", artifact.name);
                    if index < 3 {
                        core_failures.push(message.clone());
                    }
                    failures.push(message);
                    reports.push(speakerdesk_core::ipc::SetupModelReport {
                        model: identity,
                        ok: false,
                        timing_ms: 0,
                    });
                }
            }
        }
        if cancel.is_cancelled() || app.admission.closed() {
            return Err("Model setup cancelled".into());
        }
        *self.alignment.write().expect("alignment capability") = alignment;
        *self.voice.write().expect("voice capability") = voice;
        app.set_model_reports(reports);
        app.set_issues(core_failures.clone());
        let errors = if all { failures } else { core_failures };
        if errors.is_empty() {
            Ok(())
        } else {
            Err(errors.join(" "))
        }
    }
    async fn install(
        self: &Arc<Self>,
        app: &AppState,
        cancel: &CancellationToken,
    ) -> Result<(), String> {
        let installer = crate::setup::Installer::open(&self.model_root)
            .await
            .map_err(|e| e.to_string())?;
        let mut completed = 0;
        for model in crate::setup::artifacts() {
            if !installer
                .installed(&model, cancel)
                .await
                .map_err(|e| e.to_string())?
            {
                let staging = installer
                    .stage(&model, completed, &self.setup, cancel)
                    .await
                    .map_err(|e| e.to_string())?;
                if installer
                    .needs_preparation(&model, &staging, cancel)
                    .await
                    .map_err(|e| e.to_string())?
                {
                    self.setup.phase("Preparing native timing weights");
                    let supervisor = self.supervisor.clone();
                    let cancellation = cancel.clone();
                    tokio::task::spawn_blocking(move || {
                        supervisor.prepare_alignment(&cancellation)
                    })
                    .await
                    .map_err(|e| e.to_string())?
                    .map_err(|e| e.to_string())?;
                }
                installer
                    .activate(&model, &staging, cancel)
                    .await
                    .map_err(|e| e.to_string())?;
            }
            completed += model.download_bytes();
            self.setup.bytes(completed);
        }
        self.setup.phase("Restarting the model worker");
        let supervisor = self.supervisor.clone();
        let cancellation = cancel.clone();
        tokio::task::spawn_blocking(move || supervisor.restart(&cancellation))
            .await
            .map_err(|e| e.to_string())?
            .map_err(|e| e.to_string())?;
        self.initialize_async(app.clone(), cancel.clone(), true)
            .await
    }
}

pub(crate) fn native_voice_policy() -> speakerdesk_core::voice::Calibration {
    use speakerdesk_core::voice::{Calibration, VoiceModel};
    // Frozen synthetic policy verified on native Core ML; legacy profile spaces stay distinct.
    let model = VoiceModel::new(
        "aufklarer/ReDimNet2-B6-CoreML",
        "112dd8f4f836abdf8a420e66a5e2885cf8ec64ab:sd-pcm-v1:native-coreml-v1:CPU_AND_GPU",
        "cf8cc300e240efaed56cb931bd7715ded41b66bf55af7e9a19d0b17e7fe986c9",
        192,
        16000,
    )
    .expect("pinned voice identity");
    Calibration::new(
        model,
        "macos-compact-tts-development-v1:calibration",
        0.7264125328398843,
        0.10003429964182987,
        8,
        4,
        0.,
        0.,
        2,
    )
    .expect("measured voice policy")
}

pub(crate) fn recording_path(root: &Path, job: &Job) -> std::io::Result<PathBuf> {
    let native = media_write_directory(root, job).join("audio.wav");
    if native.exists() {
        return confined_file(
            root,
            native
                .strip_prefix(root)
                .map_err(std::io::Error::other)?
                .to_str()
                .ok_or_else(|| std::io::Error::other("Invalid recording path"))?,
        );
    }
    if let Some(legacy) = job.extra.get("legacy_media_root").and_then(Value::as_str) {
        return confined_file(Path::new(legacy), &format!("{}/audio.wav", job.id));
    }
    Err(std::io::Error::new(
        std::io::ErrorKind::NotFound,
        "Recording audio is unavailable",
    ))
}

pub fn run(
    storage: PathBuf,
    model_root: PathBuf,
    executable: PathBuf,
) -> Result<(), Box<dyn std::error::Error>> {
    let runtime = tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()?;
    runtime.block_on(serve(storage, model_root, executable))
}

async fn serve(
    storage: PathBuf,
    model_root: PathBuf,
    executable: PathBuf,
) -> Result<(), Box<dyn std::error::Error>> {
    let app = AppState::open(storage)?;
    if app.store().preference("default_language")?.is_none() {
        app.store().set_preference("default_language", "en")?;
    }
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await?;
    let address = listener.local_addr()?;
    let supervisor =
        InferenceSupervisor::new(executable, env!("CARGO_PKG_VERSION"), model_root.clone())?;
    let native = Arc::new(NativeModels {
        supervisor: supervisor.clone(),
        root: app.data_dir().to_path_buf(),
        model_root,
        alignment: RwLock::new(None),
        voice: RwLock::new(None),
        setup: crate::setup::SetupService::default(),
        task: Mutex::new(None),
    });
    app.set_models(native.clone());
    app.native
        .set(native.clone())
        .ok()
        .expect("native setup configured once");
    app.updates.lock().expect("updates").connect();
    native.start(app.clone(), false)?;
    let stop = CancellationToken::new();
    let server_stop = stop.clone();
    let router = crate::router(app.clone());
    let mut server = tokio::spawn(async move {
        axum::serve(listener, router)
            .with_graceful_shutdown(server_stop.cancelled_owned())
            .await
    });
    let output_error = writeln!(std::io::stdout(), "SPEAKERDESK_URL=http://{address}/")
        .and_then(|_| std::io::stdout().flush())
        .err();
    let mut input = BufReader::new(tokio::io::stdin());
    let mut update_shutdown = false;
    let control_result = loop {
        if let Some(error) = output_error {
            break Err(error);
        }
        let mut line = Vec::new();
        let count = match (&mut input)
            .take((native_updates::MAX_FRAME + 1) as u64)
            .read_until(b'\n', &mut line)
            .await
        {
            Ok(count) => count,
            Err(error) => break Err(error),
        };
        if count > native_updates::MAX_FRAME {
            break Err(std::io::Error::other("Oversized native control frame"));
        }
        if count == 0 || line == b"shutdown\n" {
            break Ok(());
        }
        let Some(body) = line.strip_prefix(native_updates::PREFIX.as_bytes()) else {
            continue;
        };
        let Ok(message) = serde_json::from_slice(body) else {
            continue;
        };
        if app
            .updates
            .lock()
            .expect("updates")
            .handle(&app.admission, message)
        {
            update_shutdown = true;
            break Ok(());
        }
    };
    app.admission.close();
    stop.cancel();
    native.cancel();
    let close_app = app.clone();
    let close_supervisor = supervisor.clone();
    let cleanup = tokio::task::spawn_blocking(move || {
        let cancelled = close_app.cancel_all();
        close_supervisor.shutdown();
        close_app.drain();
        cancelled
    });
    // Reaping is independent of clients finishing uploads or audio responses.
    let drained = tokio::time::timeout(crate::inference::CONTROL_TIMEOUT, &mut server).await;
    let drained = match drained {
        Ok(result) => result?.is_ok(),
        Err(_) => {
            server.abort();
            let _ = server.await;
            false
        }
    };
    let cleanup = cleanup.await?;
    native.join().await?;
    if update_shutdown {
        native_updates::emit(
            &app.admission
                .confirm_exit(drained && cleanup.is_ok() && supervisor.child_pid().is_none())?,
        )?;
    }
    cleanup?;
    control_result?;
    if !drained {
        return Err(std::io::Error::other(
            "Native HTTP service did not drain before the control deadline",
        )
        .into());
    }
    Ok(())
}

pub(crate) fn routes() -> Router<AppState> {
    Router::new()
        .route("/", get(index))
        .route("/static/{*path}", get(asset))
        .route("/api/jobs/{jid}/audio", get(playback))
        .route("/api/preferences/language", patch(language))
        .route("/api/setup", get(setup).post(start_setup))
        .route("/api/setup/cancel", axum::routing::post(cancel_setup))
        .route("/api/notices", get(notices))
}

async fn index(State(app): State<AppState>) -> Html<String> {
    Html(
        include_str!("../../../speakerdesk/templates/index.html").replace("{{token}}", app.token()),
    )
}

async fn asset(RoutePath(path): RoutePath<String>) -> Response {
    let Some(asset) = Assets::get(&path) else {
        return StatusCode::NOT_FOUND.into_response();
    };
    let content_type = match Path::new(&path).extension().and_then(|s| s.to_str()) {
        Some("js") => "text/javascript; charset=utf-8",
        Some("css") => "text/css; charset=utf-8",
        Some("svg") => "image/svg+xml",
        Some("woff2") => "font/woff2",
        Some("png") => "image/png",
        _ => "application/octet-stream",
    };
    (
        [(header::CONTENT_TYPE, content_type)],
        asset.data.into_owned(),
    )
        .into_response()
}

async fn playback(
    State(app): State<AppState>,
    RoutePath(jid): RoutePath<String>,
    request: Request,
) -> Response {
    let path = tokio::task::spawn_blocking(move || {
        let id = JobId::parse(jid).ok()?;
        let job = app.store().get(&id, false).ok()??;
        recording_path(app.data_dir(), &job).ok()
    })
    .await;
    let Ok(Some(path)) = path else {
        return StatusCode::NOT_FOUND.into_response();
    };
    match tower_http::services::ServeFile::new(path)
        .try_call(request)
        .await
    {
        Ok(response) => response.map(Body::new),
        Err(_) => StatusCode::INTERNAL_SERVER_ERROR.into_response(),
    }
}

async fn language(State(app): State<AppState>, body: Bytes) -> Response {
    let Ok(body) = serde_json::from_slice::<Value>(&body) else {
        return failure("Choose a supported language mode.");
    };
    let Some(language) = body["language"]
        .as_str()
        .filter(|s| *s == "auto" || languages::is_supported(s))
        .map(str::to_owned)
    else {
        return failure("Choose a supported language mode.");
    };
    blocking(
        move || match app.store().set_preference("default_language", &language) {
            Ok(()) => Json(json!({"default_language":language})).into_response(),
            Err(_) => failure("Could not save the default language."),
        },
    )
    .await
}

async fn setup(State(app): State<AppState>) -> Response {
    blocking(move || {
        let enabled = match app.store().preference("voice_recognition") {
            Ok(value) => value.as_deref()!=Some("false"),
            Err(_) => return StatusCode::INTERNAL_SERVER_ERROR.into_response(),
        };
        let (issues, reports) = app.model_status();
        let core_ready = issues.is_empty();
        let voice = app.models().voice_policy().is_some();
        let automatic = core_ready && app.automatic_language_ready();
        let timing = core_ready && app.alignment_ready();
        let ready = core_ready && automatic && timing && voice;
        let progress = app.native.get().map(|native| native.setup.progress()).unwrap_or_default();
        let busy = progress.status=="downloading";
        let supported = cfg!(all(target_os="macos",target_arch="aarch64"));
        let can_download = supported && app.native.get().is_some() && !app.admission.busy() && !app.admission.closed() && !app.admission.reserved();
        let catalog = crate::setup::artifacts();
        let models: Vec<_> = catalog.iter().enumerate().map(|(index, model)| json!({"id":model.id,"name":model.name,"installed":reports.get(index).is_some_and(|r| r.ok),"bytes":model.download_bytes(),"checkpoint":reports.get(index).map(|r| &r.model)})).collect();
        let alignment = &catalog[4];
        let status = if busy {"downloading"} else if progress.error.is_some() {"failed"} else if ready {"ready"} else {"idle"};
        Json(json!({
            "supported":supported,"ready":ready,"core_ready":core_ready,"automatic_language":automatic,
            "status":status,"phase":progress.phase,"error":progress.error,
            "models":models,"total_bytes":catalog.iter().map(|m| m.download_bytes()).sum::<u64>(),
            "downloaded_bytes":progress.downloaded_bytes,"required_free_bytes":catalog.iter().map(|m| m.stored_bytes()).sum::<u64>(),
            "can_download":can_download,"can_cancel":busy && app.native.get().is_some(),
            "alignment":{"id":alignment.id,"status":if busy {"downloading"}else if timing {"ready"}else {"idle"},"ready":timing,"can_download":false,
                "supported_languages":languages::LANGUAGES.iter().map(|(code,_)| *code).collect::<Vec<_>>(),"timing_accuracy_calibrated_languages":["en"],"total_bytes":alignment.download_bytes(),"stored_bytes":alignment.stored_bytes(),"required_free_bytes":alignment.stored_bytes(),
                "message":"Supplied-text timing preserves authoritative transcript words. Use Download models below to finish setup for all local models.","phase":progress.phase,"downloaded_bytes":0},
            "voice":{"status":if voice {"ready"}else {"unavailable"},"enabled":enabled,"available":voice,"released":false,
                "message":if voice {"Native voice enrollment and manual and automatic checks are available."}else {"The native voice model did not initialize. Saved voices and names are kept."}},
            "message": if busy {String::new()} else if !core_ready {issues.join(" ")} else {format!("Native preview. Automatic language: {}. Supplied-text timing: {}. Live capture is not implemented yet.",if automatic {"available"}else {"unavailable"},if timing {"available"}else {"unavailable"})}
        })).into_response()
    }).await
}

async fn start_setup(State(app): State<AppState>) -> Response {
    let Some(native) = app.native.get().cloned() else {
        return failure("Native model setup is unavailable in this runtime");
    };
    match native.start(app, true) {
        Ok(()) => Json(json!({"status":"downloading"})).into_response(),
        Err(error) => (
            StatusCode::CONFLICT,
            Json(json!({"error":error.to_string()})),
        )
            .into_response(),
    }
}
async fn cancel_setup(State(app): State<AppState>) -> Response {
    if let Some(native) = app.native.get() {
        native.cancel();
    }
    Json(json!({"status":"cancelling"})).into_response()
}
async fn notices() -> Response {
    (
        [
            (header::CONTENT_TYPE, "text/plain; charset=utf-8"),
            (
                header::CONTENT_DISPOSITION,
                "attachment; filename=Speakerdesk-third-party-notices.txt",
            ),
        ],
        include_str!("../../../packaging/THIRD_PARTY_NOTICES.txt"),
    )
        .into_response()
}
fn failure(message: &str) -> Response {
    (StatusCode::BAD_REQUEST, Json(json!({"error":message}))).into_response()
}

#[cfg(test)]
mod setup_tests {
    use super::*;
    #[tokio::test]
    async fn actual_router_allows_edits_and_cancellation_until_setup_cleanup_finishes() {
        // Model work is a held replay. Requests use the production router and file-backed store.
        let temp = tempfile::tempdir().unwrap();
        let app = AppState::open(temp.path().join("data")).unwrap();
        let jid = "12345678901234567890123456789012";
        let document = json!({"speakers":{"speaker_0":"Speaker 1"},"segments":[{"id":"one","start":0.,"end":1.,"speaker":"speaker_0","text":"Original words."}]});
        app.insert_job(serde_json::from_value(json!({"id":jid,"name":"Replay","status":"ready","language":"en","duration":1.,"created":0.,"updated":0.,"revision":0,"document":document})).unwrap());
        let invalid_root = temp.path().join("invalid-model-directory");
        std::fs::write(&invalid_root, b"fixture prevents network access").unwrap();
        let supervisor =
            InferenceSupervisor::new(std::env::current_exe().unwrap(), "test", &invalid_root)
                .unwrap();
        let native = Arc::new(NativeModels {
            supervisor: supervisor.clone(),
            root: app.data_dir().into(),
            model_root: invalid_root,
            alignment: RwLock::new(None),
            voice: RwLock::new(None),
            setup: crate::setup::SetupService::default(),
            task: Mutex::new(None),
        });
        app.native.set(native.clone()).ok().unwrap();
        let cancel = CancellationToken::new();
        let token = cancel.clone();
        let cleanup = Arc::new(tokio::sync::Notify::new());
        let gate = cleanup.clone();
        let lease = app.admission.lease(Activity::Setup).unwrap();
        native.setup.begin(&crate::setup::artifacts());
        *native.task.lock().unwrap() = Some((
            cancel.clone(),
            tokio::spawn(async move {
                let _lease = lease;
                token.cancelled().await;
                gate.notified().await;
            }),
        ));
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let address = listener.local_addr().unwrap();
        let router = crate::router(app.clone());
        let stop = CancellationToken::new();
        let stopped = stop.clone();
        let server = tokio::spawn(async move {
            axum::serve(listener, router)
                .with_graceful_shutdown(stopped.cancelled_owned())
                .await
                .unwrap();
        });
        let client = reqwest::Client::new();
        let url = format!("http://{address}");
        let post = |path: &str| {
            client
                .post(format!("{url}{path}"))
                .header("X-Speakerdesk-Token", app.token())
        };
        assert_eq!(
            post("/api/setup").send().await.unwrap().status(),
            200,
            "duplicate setup should reach the owner"
        );
        let mut edited = document;
        edited["segments"][0]["text"] = json!("Saved while setup waits.");
        let saved = client
            .put(format!("{url}/api/jobs/{jid}/transcript"))
            .header("X-Speakerdesk-Token", app.token())
            .json(&json!({"revision":0,"document":edited}))
            .send()
            .await
            .unwrap();
        assert_eq!(saved.status(), 200, "{}", saved.text().await.unwrap());
        assert_eq!(
            post(&format!("/api/jobs/{jid}/run"))
                .send()
                .await
                .unwrap()
                .status(),
            409
        );
        assert_eq!(
            post("/api/setup/cancel").send().await.unwrap().status(),
            200
        );
        assert!(cancel.is_cancelled());
        assert!(
            app.admission.busy(),
            "cancel acknowledgement must retain admission through cleanup"
        );
        assert!(app.admission.reserve(1, 1).is_err());
        assert_eq!(
            app.store()
                .get(&JobId::parse(jid).unwrap(), false)
                .unwrap()
                .unwrap()
                .revision,
            1
        );
        cleanup.notify_one();
        native.join().await.unwrap();
        assert!(!app.admission.busy());
        assert_eq!(
            post("/api/setup").send().await.unwrap().status(),
            200,
            "fresh setup must not hold an Import lease in middleware"
        );
        let task = native.task.lock().unwrap().take().unwrap().1;
        task.await.unwrap();
        assert_eq!(native.setup.progress().status, "failed");
        assert!(native.setup.progress().error.is_some());
        stop.cancel();
        server.await.unwrap();
        tokio::task::spawn_blocking(move || supervisor.shutdown())
            .await
            .unwrap();
    }
}
