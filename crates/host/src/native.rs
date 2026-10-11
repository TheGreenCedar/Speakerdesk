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
    sync::{Arc, OnceLock},
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

struct ImportInference {
    supervisor: Arc<InferenceSupervisor>,
    root: PathBuf,
    alignment: Arc<OnceLock<speakerdesk_core::alignment::Vocabulary>>,
    voice: Arc<OnceLock<speakerdesk_core::voice::Calibration>>,
}

impl ApplicationModels for ImportInference {
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
        self.voice.get().cloned()
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
        // shortcut: first Nemotron window only; remove when offline AOSC/FIFO is implemented.
        if job.duration.is_none_or(|duration| duration > 27.2) {
            return Err("This native preview supports imports up to 27.2 seconds. Longer diarization is not implemented yet.".into());
        }
        let path = recording_path(&self.root, job).map_err(|e| e.to_string())?;
        let pcm = audio::read_pcm16(path).map_err(|e| e.to_string())?;
        speakerdesk_core::import::import_recording(
            &self.supervisor.models(cancellation.clone()),
            &pcm,
            &job.language,
            self.alignment.get(),
        )
        .map_err(|e| e.to_string())
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
    let alignment_directory = model_root.join("coarse-alignment");
    let alignment = Arc::new(OnceLock::new());
    let voice = Arc::new(OnceLock::new());
    let supervisor = InferenceSupervisor::new(executable, env!("CARGO_PKG_VERSION"), model_root)?;
    app.set_models(Arc::new(ImportInference {
        supervisor: supervisor.clone(),
        root: app.data_dir().to_path_buf(),
        alignment: alignment.clone(),
        voice: voice.clone(),
    }));
    app.set_issues(vec!["Initializing native speech models…".into()]);
    app.updates.lock().expect("updates").connect();
    let cancel = CancellationToken::new();
    let load_app = app.clone();
    let load_models = supervisor.models(cancel.clone());
    let lease = app.admission.lease(Activity::Setup)?;
    let initialize = tokio::task::spawn_blocking(move || {
        let _lease = lease;
        match load_models.setup_check(&[
            "silero-speech".into(),
            "nemotron".into(),
            "cohere-speech".into(),
        ]) {
            Ok(mut reports) if reports.len() == 3 && reports.iter().all(|report| report.ok) => {
                // Auto is optional: a missing language model must not block manual imports.
                match load_models.setup_check(&["whisper-language".into()]) {
                    Ok(language) => reports.extend(language),
                    Err(error) => {
                        tracing::warn!(%error, "automatic language model did not initialize");
                        reports.push(speakerdesk_core::ipc::SetupModelReport {
                            model: languages::LID_CHECKPOINT.into(),
                            ok: false,
                            timing_ms: 0,
                        });
                    }
                }
                let timing = std::fs::read_to_string(alignment_directory.join("tokens.txt"))
                    .map_err(|e| e.to_string())
                    .and_then(|tokens| speakerdesk_core::alignment::Vocabulary::parse(&tokens))
                    .and_then(|vocabulary| {
                        let report = load_models
                            .setup_check(&["coarse-alignment".into()])
                            .map_err(|e| e.to_string())?;
                        if report.len() != 1
                            || !report[0].ok
                            || report[0].model != speakerdesk_core::alignment::CHECKPOINT
                        {
                            return Err("Native alignment did not initialize".into());
                        }
                        let _ = alignment.set(vocabulary);
                        Ok(report)
                    });
                match timing {
                    Ok(timing) => reports.extend(timing),
                    Err(error) => {
                        tracing::warn!(%error,"supplied-text timing did not initialize");
                        reports.push(speakerdesk_core::ipc::SetupModelReport {
                            model: speakerdesk_core::alignment::CHECKPOINT.into(),
                            ok: false,
                            timing_ms: 0,
                        });
                    }
                }
                match load_models.setup_check(&["redimnet2-b6".into()]) {
                    Ok(loaded)
                        if loaded.len() == 1
                            && loaded[0].ok
                            && loaded[0].model == native_voice_policy().model.revision =>
                    {
                        let _ = voice.set(native_voice_policy());
                        reports.extend(loaded);
                    }
                    other => tracing::warn!(?other, "native voice model did not initialize"),
                }
                load_app.set_model_reports(reports)
            }
            Ok(_) => {
                load_app.set_issues(vec!["The native speech models did not initialize.".into()])
            }
            Err(error) => load_app.set_issues(vec![format!("Native model setup failed: {error}")]),
        }
    });
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
    cancel.cancel();
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
    initialize.await?;
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
        .route("/api/setup", get(setup).post(setup_unavailable))
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
    let voice_available = app.models().voice_policy().is_some();
    let models: Vec<_> = ["Speech detection", "Speaker diarization", "Transcription", "Automatic language", "Supplied-text timing"].iter().enumerate().map(|(index, name)| json!({"id":index,"name":name,"installed":reports.get(index).is_some_and(|r| r.ok),"bytes":0,"checkpoint":reports.get(index).map(|r| &r.model)})).collect();
    Json(
        json!({"supported":cfg!(all(target_os="macos",target_arch="aarch64")),"ready":issues.is_empty(),"core_ready":issues.is_empty(),"automatic_language":issues.is_empty() && app.automatic_language_ready(),"status":if issues.is_empty(){"ready"}else{"failed"},"error":issues.join(" "),"models":models,"total_bytes":0,"downloaded_bytes":0,"can_download":false,
        "voice":{"status":if voice_available{"ready"}else{"unavailable"},"enabled":enabled,"available":voice_available,"released":false,"message":if voice_available{"Native voice enrollment and manual checking are available. Automatic checks are still pending."}else{"The native voice model did not initialize. Saved voices and names are kept."}},
        "message":format!("Native preview: imports up to 27.2 seconds. Automatic language: {}. Supplied-text timing: {}. Live capture is not implemented yet.",if app.automatic_language_ready(){"available"}else{"unavailable"},if app.alignment_ready(){"available"}else{"unavailable"})}),
    ).into_response()
    }).await
}

async fn setup_unavailable() -> Response {
    failure(
        "Native model downloads are not implemented yet. This development build uses its explicit model directory.",
    )
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
