//! Real router and SQLite lifecycle with scripted audio, not acoustic quality proof.
use std::{
    path::Path,
    sync::{
        Arc, Condvar, Mutex,
        atomic::{AtomicBool, AtomicUsize, Ordering},
    },
    time::Duration,
};

use serde_json::{Value, json};
use speakerdesk_core::{
    audio,
    document::{Document, Job},
    ids::JobId,
    import::PassageCandidate,
    ipc::SeparatedAudio,
    separation::{CandidateVoice, MODEL},
    store::{JobStore, SqliteJobStore},
    voice,
};
use speakerdesk_host::routes::{AppState, ApplicationModels};
use tokio_util::sync::CancellationToken;

const JOB: &str = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa";
const INTERRUPTED: &str = "cccccccccccccccccccccccccccccccc";
const SEPARATION_STOPPED: &str = "Voice separation stopped. Existing words and audio are retained.";

struct Replay {
    entered: AtomicBool,
    release: (Mutex<bool>, Condvar),
    fail: bool,
    retried: AtomicBool,
    document: Option<Document>,
    available: bool,
    empty_voice: bool,
    calls: AtomicUsize,
}

impl Replay {
    fn new(blocked: bool, fail: bool) -> Self {
        Self {
            entered: AtomicBool::new(false),
            release: (Mutex::new(!blocked), Condvar::new()),
            fail,
            retried: AtomicBool::new(false),
            document: None,
            available: true,
            empty_voice: false,
            calls: AtomicUsize::new(0),
        }
    }
    fn release(&self) {
        *self.release.0.lock().unwrap() = true;
        self.release.1.notify_all();
    }
}

impl ApplicationModels for Replay {
    fn infer(&self, _: &Job, _: &CancellationToken) -> Result<Document, String> {
        Ok(self
            .document
            .clone()
            .expect("manual separation never reruns the import"))
    }
    fn separation_available(&self) -> bool {
        self.available
    }
    fn separate_passage(
        &self,
        pcm: &[f32],
        language: &str,
        _: &CancellationToken,
    ) -> Result<(SeparatedAudio, [CandidateVoice; 2]), String> {
        assert_eq!(pcm.len(), 32000);
        assert_eq!(language, "en");
        self.calls.fetch_add(1, Ordering::SeqCst);
        self.entered.store(true, Ordering::SeqCst);
        let mut released = self.release.0.lock().unwrap();
        while !*released {
            released = self.release.1.wait(released).unwrap();
        }
        if self.fail {
            return Err("scripted model failure".into());
        }
        Ok((
            SeparatedAudio {
                model: MODEL.into(),
                channels: [vec![0.25; pcm.len()], vec![0.5; pcm.len()]],
            },
            std::array::from_fn(|channel| CandidateVoice {
                text: if self.empty_voice && channel == 1 {
                    String::new()
                } else {
                    format!("estimated voice {channel}")
                },
                language: "en".into(),
                review: false,
                transcription_review: None,
            }),
        ))
    }
    fn retry_passage(
        &self,
        pcm: &[f32],
        language: &str,
        _: Option<&str>,
        _: &CancellationToken,
    ) -> Result<PassageCandidate, String> {
        assert_eq!(
            pcm,
            vec![0.25; 16000],
            "retry must crop Voice A, using its local clock"
        );
        self.retried.store(true, Ordering::SeqCst);
        Ok(PassageCandidate {
            start: 0.,
            end: 1.,
            text: "retried voice A".into(),
            language: language.into(),
            review: false,
            transcription_review: None,
            retry_parts: Vec::new(),
        })
    }
}

struct ReleaseOnDrop(Arc<Replay>);
impl Drop for ReleaseOnDrop {
    fn drop(&mut self) {
        self.0.release();
    }
}

struct Host {
    state: AppState,
    client: reqwest::Client,
    base: String,
    shutdown: tokio::sync::oneshot::Sender<()>,
    server: tokio::task::JoinHandle<Result<(), std::io::Error>>,
}

impl Host {
    async fn start(state: AppState) -> Self {
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let base = format!("http://{}", listener.local_addr().unwrap());
        let router = speakerdesk_host::router(state.clone());
        let (shutdown, signal) = tokio::sync::oneshot::channel();
        let server = tokio::spawn(async move {
            axum::serve(listener, router)
                .with_graceful_shutdown(async {
                    let _ = signal.await;
                })
                .await
        });
        Self {
            state,
            client: reqwest::Client::builder()
                .pool_max_idle_per_host(0)
                .build()
                .unwrap(),
            base,
            shutdown,
            server,
        }
    }
    async fn request(
        &self,
        method: reqwest::Method,
        suffix: &str,
        body: Option<Value>,
    ) -> reqwest::Response {
        let mut request = self
            .client
            .request(method, format!("{}/api/jobs/{JOB}{suffix}", self.base))
            .header(speakerdesk_host::http::token_header(), self.state.token());
        if let Some(body) = body {
            request = request.json(&body);
        }
        request.send().await.unwrap()
    }
    async fn job(&self) -> Value {
        self.request(reqwest::Method::GET, "", None)
            .await
            .error_for_status()
            .unwrap()
            .json()
            .await
            .unwrap()
    }
    async fn begin(&self) -> Value {
        let job = self.job().await;
        let response = self
            .request(
                reqwest::Method::POST,
                "/separation",
                Some(json!({"revision":job["revision"],"segment_id":"mixed"})),
            )
            .await;
        assert_eq!(response.status(), 202);
        response.json().await.unwrap()
    }
    async fn post(&self, suffix: &str, expected: u16) -> Value {
        let response = self.request(reqwest::Method::POST, suffix, None).await;
        assert_eq!(response.status(), expected);
        response.json().await.unwrap()
    }
    async fn stop(self) {
        let _ = self.shutdown.send(());
        tokio::time::timeout(Duration::from_secs(3), self.server)
            .await
            .unwrap()
            .unwrap()
            .unwrap();
    }
}

fn fixture(root: &Path, half_sample_start: bool) -> AppState {
    let state = AppState::open(root).unwrap();
    let start = if half_sample_start {
        2. + 1. / 256.
    } else {
        2.
    };
    let end = start + 2.;
    let job: Job = serde_json::from_value(json!({"id":JOB,"name":"overlap","status":"ready","language":"en","duration":6.,"created":0.,"updated":0.,"revision":7,"document":{"speakers":{"speaker_0":"Ada","speaker_1":"Bea","multiple_speakers":"Overlapping speakers"},"segments":[{"id":"mixed","start":start,"end":end,"speaker":"multiple_speakers","text":"protected original words","protected_fields":["text"],"machine_revision":2,"audio_revision":0,"alignment":{"mixed":"never copy"},"reading_turns":[{"mixed":"never copy"}],"speaker_activity":{"audio_revision":0,"observed_end_sample":65000,"regions":[{"start_sample":32000,"end_sample":65000,"speakers":["speaker_0","speaker_1"]}]}},{"id":"adjacent","start":4.1,"end":6.,"speaker":"speaker_0","text":"adjacent words"}]}})).unwrap();
    let folder = root.join(JOB);
    std::fs::create_dir_all(&folder).unwrap();
    let mut writer = audio::RecordingWriter::create(folder.join("audio.wav")).unwrap();
    writer.write_samples(&vec![0.125; 96000]).unwrap();
    state.insert_job(job);
    state
}

fn segment<'a>(job: &'a Value, id: &str) -> &'a Value {
    job["document"]["segments"]
        .as_array()
        .unwrap()
        .iter()
        .find(|row| row["id"] == id)
        .unwrap()
}

fn stored(root: &Path) -> Job {
    SqliteJobStore::open(root.join("speakerdesk.sqlite"))
        .unwrap()
        .get(&JobId::parse(JOB).unwrap(), false)
        .unwrap()
        .unwrap()
}

fn automatic_fixture(root: &Path, blocked: bool, mode: &str) -> (AppState, Arc<Replay>) {
    let state = fixture(root, false);
    let mut job = stored(root);
    let mut document = job.document.take().unwrap();
    document.segments[0].text = "original mixed words".into();
    if mode != "protected" {
        document.segments[0].protected_fields = None;
    }
    job.status = "uploaded".into();
    state.insert_job(job);
    state.set_issues(Vec::new());
    let mut replay = Replay::new(blocked, mode == "failure");
    replay.document = Some(document);
    replay.available = mode != "unavailable";
    replay.empty_voice = mode == "empty_voice";
    let replay = Arc::new(replay);
    state.set_models(replay.clone());
    (state, replay)
}

async fn wait_separation(replay: &Replay) {
    tokio::time::timeout(Duration::from_secs(3), async {
        while !replay.entered.load(Ordering::SeqCst) {
            tokio::task::yield_now().await;
        }
    })
    .await
    .unwrap();
}

#[tokio::test]
async fn automatic_import_separation_preserves_protection_cancellation_and_unrelated_edits() {
    for mode in [
        "protected",
        "unavailable",
        "edit",
        "cancel",
        "stale_generation",
        "failure",
        "empty_voice",
        "unrelated",
    ] {
        let temp = tempfile::tempdir().unwrap();
        let skip = matches!(mode, "protected" | "unavailable");
        let (state, replay) = automatic_fixture(temp.path(), !skip, mode);
        let _release_on_failure = ReleaseOnDrop(replay.clone());
        let host = Host::start(state).await;
        host.post("/run", 202).await;
        if !skip {
            wait_separation(&replay).await;
            match mode {
                "edit" => {
                    let response = host.request(reqwest::Method::PATCH, "/segments/mixed", Some(json!({"segment_revision":2,"changes":{"text":"my concurrent correction"}}))).await;
                    assert_eq!(response.status(), 200);
                }
                "cancel" => {
                    let cancelled = host.post("/cancel", 200).await;
                    assert!(cancelled.get("passage_separation").is_none());
                    assert_eq!(cancelled["message"], SEPARATION_STOPPED);
                }
                "stale_generation" => {
                    let mut current = stored(temp.path());
                    let generation = current.extra["operation_generation"].as_u64().unwrap();
                    current
                        .extra
                        .insert("operation_generation".into(), json!(generation + 1));
                    current.extra.remove("passage_separation");
                    current.message = "A newer operation owns this message.".into();
                    host.state.insert_job(current);
                }
                "unrelated" => {
                    let current = host.job().await;
                    let mut document = current["document"].clone();
                    document["segments"][1]["text"] = json!("my neighbouring correction");
                    document["speakers"]["speaker_0"] = json!("Renamed Ada");
                    let response = host
                        .request(
                            reqwest::Method::PUT,
                            "/transcript",
                            Some(json!({"revision":current["revision"],"document":document})),
                        )
                        .await;
                    assert_eq!(response.status(), 200);
                }
                _ => {}
            }
            replay.release();
        }
        host.state.drain();
        let finished = host.job().await;
        assert_eq!(finished["status"], "ready", "{mode}");
        assert!(finished.get("passage_separation").is_none(), "{mode}");
        if mode == "stale_generation" {
            assert_eq!(finished["message"], "A newer operation owns this message.");
        } else if matches!(mode, "edit" | "cancel" | "failure" | "empty_voice") {
            assert!(
                finished["separation_error"]
                    .as_str()
                    .is_some_and(|reason| !reason.is_empty()),
                "{mode}"
            );
            assert_eq!(
                finished["message"], SEPARATION_STOPPED,
                "{mode}: the editor status must report that separation stopped"
            );
        }
        assert_eq!(
            replay.calls.load(Ordering::SeqCst),
            usize::from(!skip),
            "{mode}"
        );
        assert!(
            finished["document"]["segments"]
                .as_array()
                .unwrap()
                .iter()
                .all(|row| row.get("separation_candidate").is_none()),
            "automatic work must never leave an acceptance task: {mode}"
        );
        if mode == "unrelated" {
            let rows = finished["document"]["segments"].as_array().unwrap();
            assert_eq!(rows.len(), 3);
            assert_eq!(rows[0]["text"], "estimated voice 0");
            assert_eq!(rows[1]["text"], "estimated voice 1");
            assert_eq!(rows[2]["id"], "adjacent");
            assert_eq!(rows[2]["text"], "my neighbouring correction");
            assert_eq!(finished["document"]["speakers"]["speaker_0"], "Renamed Ada");
            for row in &rows[..2] {
                assert_eq!(
                    row["speaker"], "unassigned",
                    "channel order cannot identify the local tracks"
                );
                assert_eq!(row["review"], true);
                assert_eq!(row["review_resolution"], "separated_estimate");
            }
        } else {
            let rows = finished["document"]["segments"].as_array().unwrap();
            assert_eq!(rows.len(), 2, "{mode}");
            assert_eq!(rows[0]["id"], "mixed", "{mode}");
            assert_eq!(
                rows[0]["text"],
                if mode == "edit" {
                    "my concurrent correction"
                } else {
                    "original mixed words"
                },
                "{mode}"
            );
            assert_eq!(rows[1]["text"], "adjacent words", "{mode}");
            let directory = temp.path().join(JOB).join("separated");
            if directory.exists() {
                assert_eq!(
                    std::fs::read_dir(directory).unwrap().count(),
                    0,
                    "{mode}: rejected automatic work cannot retain generated assets"
                );
            }
        }
        assert!(temp.path().join(JOB).join("audio.wav").is_file());
        host.stop().await;
        if matches!(mode, "cancel" | "failure") {
            let state = AppState::open(temp.path()).unwrap();
            let mut previous = stored(temp.path());
            let mut replay = Replay::new(false, false);
            replay.document = previous.document.take();
            let replay = Arc::new(replay);
            state.set_models(replay.clone());
            state.set_issues(Vec::new());
            previous.status = "uploaded".into();
            state.insert_job(previous);
            let host = Host::start(state).await;
            host.post("/run", 202).await;
            host.state.drain();
            assert_eq!(
                replay.calls.load(Ordering::SeqCst),
                0,
                "{mode}: restart cannot repeat the attempted source"
            );
            assert_eq!(
                host.job().await["document"]["segments"][0]["text"],
                "original mixed words"
            );
            host.stop().await;
        }
    }
}

#[tokio::test]
async fn automatic_rows_export_restart_restore_and_do_not_repeat_the_same_source() {
    let temp = tempfile::tempdir().unwrap();
    let (state, replay) = automatic_fixture(temp.path(), false, "ordinary");
    let original_audio = std::fs::read(temp.path().join(JOB).join("audio.wav")).unwrap();
    let host = Host::start(state).await;
    host.post("/run", 202).await;
    host.state.drain();
    let split = host.job().await;
    let rows = split["document"]["segments"].as_array().unwrap();
    assert_eq!(rows.len(), 3);
    let episode = rows[0]["separation"]["episode_id"]
        .as_str()
        .unwrap()
        .to_owned();
    for channel in 0..2 {
        assert_eq!(rows[channel]["separation"]["channel"], channel);
        assert_eq!(rows[channel]["separation"]["origin_sample"], 32000);
        assert_eq!(rows[channel]["separation"]["length_samples"], 32000);
        assert!(rows[channel].get("alignment").is_none());
        assert!(rows[channel].get("reading_turns").is_none());
        let asset = temp
            .path()
            .join(JOB)
            .join("separated")
            .join(format!("{episode}-{channel}.wav"));
        let samples = audio::read_pcm16(asset).unwrap();
        assert_eq!(samples, vec![if channel == 0 { 0.25 } else { 0.5 }; 32000]);
        let response = host
            .request(
                reqwest::Method::GET,
                &format!("/separation/{episode}/{channel}/audio"),
                None,
            )
            .await;
        assert_eq!(response.status(), 200);
        assert!(response.bytes().await.unwrap().len() > 44);
    }
    for kind in ["txt", "srt", "vtt", "json"] {
        let response = host
            .request(reqwest::Method::GET, &format!("/export/{kind}"), None)
            .await;
        assert_eq!(response.status(), 200);
        let body = response.text().await.unwrap();
        assert!(body.find("estimated voice 0").unwrap() < body.find("estimated voice 1").unwrap());
        assert!(!body.contains("original mixed words"));
        if kind != "json" {
            assert!(body.contains("Voice A") && body.contains("Voice B"));
        }
    }
    assert_eq!(replay.calls.load(Ordering::SeqCst), 1);
    host.stop().await;

    let state = AppState::open(temp.path()).unwrap();
    let replay = Arc::new(Replay::new(false, false));
    state.set_models(replay.clone());
    let host = Host::start(state).await;
    assert_eq!(host.job().await["document"], split["document"]);
    let restored = host
        .post(&format!("/separation/{episode}/restore"), 200)
        .await;
    assert_eq!(
        restored["document"]["segments"][0]["text"],
        "original mixed words"
    );
    for _ in 0..2 {
        let current = host.job().await;
        let response = host
            .request(
                reqwest::Method::PUT,
                "/transcript",
                Some(json!({"revision":current["revision"],"document":current["document"]})),
            )
            .await;
        assert_eq!(response.status(), 200);
        let response = host
            .request(reqwest::Method::GET, "/export/json", None)
            .await;
        assert_eq!(response.status(), 200);
    }
    host.state.drain();
    assert_eq!(
        replay.calls.load(Ordering::SeqCst),
        0,
        "GET, save, export and restore cannot start model work"
    );
    host.stop().await;

    let state = AppState::open(temp.path()).unwrap();
    let mut restored_job = stored(temp.path());
    let mut replay = Replay::new(false, false);
    replay.document = restored_job.document.take();
    let replay = Arc::new(replay);
    state.set_models(replay.clone());
    state.set_issues(Vec::new());
    // Re-enter the import Ready boundary with the persisted original source.
    restored_job.status = "uploaded".into();
    state.insert_job(restored_job);
    let host = Host::start(state).await;
    host.post("/run", 202).await;
    host.state.drain();
    let finished = host.job().await;
    assert_eq!(
        replay.calls.load(Ordering::SeqCst),
        0,
        "the persisted attempt must survive restore, saves and restart"
    );
    assert_eq!(
        finished["document"]["segments"].as_array().unwrap().len(),
        2
    );
    assert_eq!(
        finished["document"]["segments"][0]["text"],
        "original mixed words"
    );
    assert_eq!(
        std::fs::read(temp.path().join(JOB).join("audio.wav")).unwrap(),
        original_audio
    );
    host.stop().await;
}

#[tokio::test]
async fn candidate_accept_restart_edit_retry_and_exports_keep_the_two_audio_clocks() {
    let temp = tempfile::tempdir().unwrap();
    let state = fixture(temp.path(), true);
    state.set_models(Arc::new(Replay::new(false, false)));
    let host = Host::start(state).await;
    let original = host.job().await["document"]["segments"][0].clone();
    host.begin().await;
    host.state.drain();
    let pending = host.job().await;
    let episode = pending["document"]["segments"][0]["separation_candidate"]["id"]
        .as_str()
        .unwrap()
        .to_owned();
    assert_eq!(pending["document"]["segments"][0]["text"], original["text"]);
    for kind in ["txt", "srt", "vtt", "json"] {
        let body = host
            .request(reqwest::Method::GET, &format!("/export/{kind}"), None)
            .await
            .error_for_status()
            .unwrap()
            .text()
            .await
            .unwrap();
        assert!(body.contains("protected original words"));
        assert!(
            !body.contains("estimated voice"),
            "pending candidates cannot become {kind} export content"
        );
    }
    host.stop().await;
    let host = Host::start(AppState::open(temp.path()).unwrap()).await;
    assert_eq!(
        host.job().await["document"]["segments"][0]["separation_candidate"]["id"],
        episode
    );
    let accepted = host
        .post(&format!("/separation/{episode}/accept"), 200)
        .await;
    assert_eq!(accepted["message"], "Separated voices saved.");
    let rows = accepted["document"]["segments"].as_array().unwrap();
    assert_eq!(rows.len(), 3);
    assert_eq!(rows[2]["id"], "adjacent");
    assert_eq!(rows[0]["separation"]["origin_sample"], 32062);
    assert_eq!(rows[1]["separation"]["length_samples"], 32000);
    assert_eq!(rows[0]["speaker"], "unassigned");
    assert!(
        rows[..2]
            .iter()
            .all(|row| row.get("alignment").is_none() && row.get("reading_turns").is_none())
    );
    let mut forged = accepted["document"].clone();
    forged["segments"][0]["separation"]["channel"] = json!(1);
    forged["segments"][0]["separation"]["origin_sample"] = json!(0);
    forged["segments"][1]
        .as_object_mut()
        .unwrap()
        .remove("separation");
    forged["segments"].as_array_mut().unwrap()[..2].reverse();
    let response = host
        .request(
            reqwest::Method::PUT,
            "/transcript",
            Some(json!({"revision":accepted["revision"],"document":forged})),
        )
        .await;
    assert_eq!(
        response.status(),
        200,
        "an unchanged quantized episode remains saveable"
    );
    let saved: Value = response.json().await.unwrap();
    for channel in 0..2 {
        assert_eq!(
            saved["document"]["segments"][channel]["separation"],
            rows[channel]["separation"]
        );
    }
    for kind in ["txt", "srt", "vtt", "json"] {
        let body = host
            .request(reqwest::Method::GET, &format!("/export/{kind}"), None)
            .await
            .error_for_status()
            .unwrap()
            .text()
            .await
            .unwrap();
        assert!(!body.contains("protected original words"));
        assert!(body.find("estimated voice 0").unwrap() < body.find("estimated voice 1").unwrap());
        if kind != "json" {
            assert!(body.contains("Voice A") && body.contains("Voice B"));
        }
    }
    host.stop().await;
    let state = AppState::open(temp.path()).unwrap();
    let replay = Arc::new(Replay::new(false, false));
    state.set_models(replay.clone());
    let host = Host::start(state).await;
    let accepted = host.job().await;
    let row_id = accepted["document"]["segments"][0]["id"].as_str().unwrap();
    let sibling = accepted["document"]["segments"][1].clone();
    let sibling_id = sibling["id"].as_str().unwrap();
    let response = host.request(reqwest::Method::PATCH, &format!("/segments/{row_id}"), Some(json!({"segment_revision":0,"changes":{"text":"my correction","speaker":"speaker_0","start":2.5,"end":3.5}}))).await;
    assert_eq!(response.status(), 200);
    let edited = host.job().await;
    assert_eq!(segment(&edited, sibling_id), &sibling);
    let voice: voice::AudioSegment =
        serde_json::from_value(segment(&edited, row_id).clone()).unwrap();
    assert!(voice::speaker_audio_ranges(&voice, "speaker_0").is_empty());
    assert!(!voice::eligible_segment(&voice, "speaker_0"));
    let response = host
        .request(
            reqwest::Method::POST,
            "/retry",
            Some(json!({"revision":edited["revision"],"segment_id":row_id,"language":"en"})),
        )
        .await;
    assert_eq!(response.status(), 202);
    host.state.drain();
    assert!(replay.retried.load(Ordering::SeqCst));
    let retried = host.job().await;
    assert_eq!(
        segment(&retried, row_id)["retry_candidate"]["text"],
        "retried voice A"
    );
    assert_eq!(segment(&retried, sibling_id), &sibling);
    host.post(&format!("/separation/{episode}/restore"), 409)
        .await;
    let history = host
        .request(
            reqwest::Method::GET,
            &format!("/separation/{episode}/history"),
            None,
        )
        .await
        .error_for_status()
        .unwrap()
        .json::<Value>()
        .await
        .unwrap();
    assert_eq!(history["original"], original);
    assert_eq!(segment(&host.job().await, row_id)["text"], "my correction");
    host.stop().await;
}

#[tokio::test]
async fn edited_cancelled_and_failed_attempts_never_publish_assets_or_replace_words() {
    for mode in ["edit", "cancel", "failure"] {
        let temp = tempfile::tempdir().unwrap();
        let state = fixture(temp.path(), false);
        let replay = Arc::new(Replay::new(true, mode == "failure"));
        let _release_on_failure = ReleaseOnDrop(replay.clone());
        state.set_models(replay.clone());
        let host = Host::start(state).await;
        let started = host.begin().await;
        let episode = started["passage_separation"]["id"].as_str().unwrap();
        tokio::time::timeout(Duration::from_secs(3), async {
            while !replay.entered.load(Ordering::SeqCst) {
                tokio::task::yield_now().await;
            }
        })
        .await
        .unwrap();
        if mode == "edit" {
            let response = host.request(reqwest::Method::PATCH, "/segments/mixed", Some(json!({"segment_revision":2,"changes":{"text":"changed during separation"}}))).await;
            assert_eq!(response.status(), 200);
        } else if mode == "cancel" {
            host.post("/cancel", 200).await;
        }
        replay.release();
        host.state.drain();
        let stopped = host.job().await;
        assert_eq!(stopped["status"], "ready");
        assert!(stopped.get("passage_separation").is_none());
        assert!(stopped.get("separation_error").is_some());
        assert!(
            stopped["document"]["segments"][0]
                .get("separation_candidate")
                .is_none()
        );
        assert_eq!(
            stopped["document"]["segments"][0]["text"],
            if mode == "edit" {
                "changed during separation"
            } else {
                "protected original words"
            }
        );
        for channel in 0..2 {
            assert!(
                !temp
                    .path()
                    .join(JOB)
                    .join("separated")
                    .join(format!("{episode}-{channel}.wav"))
                    .exists()
            );
        }
        assert!(temp.path().join(JOB).join("audio.wav").is_file());
        host.stop().await;
    }
}

#[tokio::test]
async fn missing_generated_asset_blocks_atomic_acceptance() {
    let temp = tempfile::tempdir().unwrap();
    let state = fixture(temp.path(), false);
    state.set_models(Arc::new(Replay::new(false, false)));
    let host = Host::start(state).await;
    host.begin().await;
    host.state.drain();
    let before = host.job().await;
    let episode = before["document"]["segments"][0]["separation_candidate"]["id"]
        .as_str()
        .unwrap();
    std::fs::remove_file(
        temp.path()
            .join(JOB)
            .join("separated")
            .join(format!("{episode}-1.wav")),
    )
    .unwrap();
    host.post(&format!("/separation/{episode}/accept"), 409)
        .await;
    assert_eq!(host.job().await["document"], before["document"]);
    let playback = host
        .request(
            reqwest::Method::GET,
            &format!("/separation/{episode}/1/audio"),
            None,
        )
        .await;
    assert_eq!(playback.status(), 404);
    let kept = host
        .post(&format!("/separation/{episode}/discard"), 200)
        .await;
    assert_eq!(kept["message"], "Original passage kept.");
    assert_eq!(
        kept["document"]["segments"][0]["text"],
        "protected original words"
    );
    host.begin().await;
    host.state.drain();
    let pending = host.job().await;
    let next_episode = pending["document"]["segments"][0]["separation_candidate"]["id"]
        .as_str()
        .unwrap();
    let accepted = host
        .post(&format!("/separation/{next_episode}/accept"), 200)
        .await;
    assert_eq!(accepted["message"], "Separated voices saved.");
    let restored = host
        .post(&format!("/separation/{next_episode}/restore"), 200)
        .await;
    assert_eq!(restored["message"], "Original passage restored.");
    assert_eq!(
        restored["document"]["segments"][0],
        before["document"]["segments"][0]["separation_candidate"]["original"]
    );
    assert_eq!(
        restored["document"]["segments"][1],
        before["document"]["segments"][1]
    );
    host.stop().await;
}

#[tokio::test]
async fn restart_stops_only_the_interrupted_attempt_and_keeps_original_audio() {
    let temp = tempfile::tempdir().unwrap();
    let state = fixture(temp.path(), false);
    let initial = Host::start(state.clone()).await;
    let mut stored: Job = serde_json::from_value(initial.job().await).unwrap();
    initial.stop().await;
    stored.message = "Separating overlapping voices. Your original words remain editable.".into();
    stored.extra.insert(
        "passage_separation".into(),
        json!({"id":INTERRUPTED,"segment_id":"mixed"}),
    );
    state.insert_job(stored);
    let directory = temp.path().join(JOB).join("separated");
    std::fs::create_dir_all(&directory).unwrap();
    for filename in [
        format!("{INTERRUPTED}-0.part"),
        format!("{INTERRUPTED}-1.wav"),
        "private-reference.wav".into(),
    ] {
        std::fs::write(directory.join(filename), b"owned attempt data").unwrap();
    }
    let original = std::fs::read(temp.path().join(JOB).join("audio.wav")).unwrap();
    drop(state);
    let host = Host::start(AppState::open(temp.path()).unwrap()).await;
    let job = host.job().await;
    assert_eq!(job["status"], "ready");
    assert_eq!(job["message"], SEPARATION_STOPPED);
    assert!(job.get("passage_separation").is_none());
    assert!(job.get("separation_error").is_some());
    assert!(!directory.join(format!("{INTERRUPTED}-0.part")).exists());
    assert!(!directory.join(format!("{INTERRUPTED}-1.wav")).exists());
    assert!(directory.join("private-reference.wav").is_file());
    assert_eq!(
        std::fs::read(temp.path().join(JOB).join("audio.wav")).unwrap(),
        original
    );
    host.stop().await;
}
