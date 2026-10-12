//! Router/store acceptance with retained model output, not acoustic inference.
use super::*;
use speakerdesk_core::import::PassageCandidate;
use tokio_util::sync::CancellationToken;

struct Replay {
    entered: AtomicUsize,
    release: (Mutex<bool>, Condvar),
    expected: Vec<f32>,
    fail: bool,
}
impl ApplicationModels for Replay {
    fn infer(&self, _: &Job, _: &CancellationToken) -> Result<Document, String> {
        unreachable!("retry must not run whole-recording inference")
    }
    fn retry_passage(
        &self,
        pcm: &[f32],
        language: &str,
        _: &CancellationToken,
    ) -> Result<PassageCandidate, String> {
        assert_eq!(
            pcm, self.expected,
            "exact bounded PCM including fractional endpoints"
        );
        assert_eq!(language, "fr");
        self.entered.fetch_add(1, Ordering::SeqCst);
        let mut released = self.release.0.lock().unwrap();
        while !*released {
            released = self.release.1.wait(released).unwrap();
        }
        if self.fail {
            return Err("retained worker failure".into());
        }
        Ok(PassageCandidate {
            start: 0.,
            end: pcm.len() as f64 / 16000.,
            text: "Bonjour tout le monde.".into(),
            language: language.into(),
            review: false,
            transcription_review: None,
            retry_parts: vec![json!({"start":0.,"end":1.,"text":"Bonjour tout le monde."})],
        })
    }
}
impl Replay {
    fn release(&self) {
        *self.release.0.lock().unwrap() = true;
        self.release.1.notify_all();
    }
}
struct ReleaseOnDrop(Arc<Replay>);
impl Drop for ReleaseOnDrop {
    fn drop(&mut self) {
        self.0.release();
    }
}

fn fixture(fail: bool) -> (AppState, Arc<Replay>) {
    let state = AppState::new();
    let mut saved = job(JOB_A, "ready", 1.);
    let doc = saved.document.as_mut().unwrap();
    doc.segments[0].start = 2.00004;
    doc.segments[0].end = 4.00009;
    doc.segments[0].alignment = Some(json!({"kind":"fixture timing"}));
    doc.segments[0].rest.insert(
        "reading_alignment".into(),
        json!({"kind":"fixture supplied-text timing"}),
    );
    let mut other = segment("Unrelated words.", 1);
    other.id = "two".into();
    other.start = 5.;
    other.end = 6.;
    doc.segments.push(other);
    let folder = state.data_dir().join(JOB_A);
    std::fs::create_dir_all(&folder).unwrap();
    let pcm: Vec<f32> = (0..160000)
        .map(|i| ((i % 1000) as f32 - 500.) / 32768.)
        .collect();
    let mut wav = hound::WavWriter::create(
        folder.join("audio.wav"),
        hound::WavSpec {
            channels: 1,
            sample_rate: 16000,
            bits_per_sample: 16,
            sample_format: hound::SampleFormat::Int,
        },
    )
    .unwrap();
    for sample in &pcm {
        wav.write_sample((sample * 32768.) as i16).unwrap();
    }
    wav.finalize().unwrap();
    let models = Arc::new(Replay {
        entered: AtomicUsize::new(0),
        release: (Mutex::new(false), Condvar::new()),
        expected: pcm[32001..64001].to_vec(),
        fail,
    });
    state.insert_job(saved);
    state.set_models(models.clone());
    (state, models)
}

fn source_fixture() -> (AppState, Arc<Replay>) {
    let (state, replay) = fixture(false);
    let folder = state.data_dir().join(JOB_A);
    std::fs::rename(folder.join("audio.wav"), folder.join("system.wav")).unwrap();
    let mut wav = hound::WavWriter::create(
        folder.join("audio.wav"),
        hound::WavSpec {
            channels: 1,
            sample_rate: 16000,
            bits_per_sample: 16,
            sample_format: hound::SampleFormat::Int,
        },
    )
    .unwrap();
    for _ in 0..160000 {
        wav.write_sample(12345i16).unwrap();
    }
    wav.finalize().unwrap();
    let mut job = state
        .store()
        .get(&JobId::parse(JOB_A).unwrap(), false)
        .unwrap()
        .unwrap();
    let catalog = json!({"schema_version":1,"job_id":JOB_A,"sample_rate":16000,"clock":"recording_samples","policy":"aligned_aec_microphone_and_original_system_v1","sources":[{"source_id":"microphone_clean","basename":"microphone_clean.wav","revision":1},{"source_id":"system","basename":"system.wav","revision":1}]});
    let hash = speakerdesk_core::memo::sha256_hex(&serde_json::to_vec(&catalog).unwrap());
    job.extra
        .insert("sources".into(), json!(["microphone", "system"]));
    job.extra.insert("capture_source_catalog".into(), catalog);
    job.document.as_mut().unwrap().segments[0].capture_source =
        Some(json!({"catalog_sha256":hash,"source_id":"system","source_revision":1}));
    state.insert_job(job);
    (state, replay)
}

async fn start(running: &Running, revision: i64) -> reqwest::Response {
    running
        .authed(
            running
                .client
                .post(running.url(&format!("/api/jobs/{JOB_A}/retry"))),
        )
        .json(&json!({"revision":revision,"segment_id":"one","language":"fr"}))
        .send()
        .await
        .unwrap()
}

#[tokio::test]
async fn retry_candidate_survives_unrelated_edits_and_is_accepted_explicitly() {
    let (state, models) = source_fixture();
    let running = Running::spawn(state.clone()).await;
    let _release = ReleaseOnDrop(models.clone());
    assert_eq!(start(&running, 6).await.status(), 409);
    let response = start(&running, 7).await;
    assert_eq!(response.status(), 202);
    let pending: Value = response.json().await.unwrap();
    assert_eq!(pending["status"], "ready");
    assert_eq!(pending["revision"], 8);
    wait_until(&models.entered, 1).await;
    assert_eq!(start(&running, 8).await.status(), 409);
    let mut doc = pending["document"].clone();
    doc["segments"][1]["text"] = json!("Protected unrelated edit.");
    let edited = running
        .authed(
            running
                .client
                .put(running.url(&format!("/api/jobs/{JOB_A}/transcript"))),
        )
        .json(&json!({"revision":8,"document":doc}))
        .send()
        .await
        .unwrap();
    assert_eq!(edited.status(), 200);
    models.release();
    state.drain();
    let completed = state
        .store()
        .get(&JobId::parse(JOB_A).unwrap(), false)
        .unwrap()
        .unwrap();
    assert_eq!(completed.revision, 10);
    assert!(!completed.extra.contains_key("passage_retry"));
    let doc = completed.document.as_ref().unwrap();
    assert_eq!(doc.segments[0].text, "Keep these words.");
    assert_eq!(doc.segments[1].text, "Protected unrelated edit.");
    let candidate = doc.segments[0].retry_candidate.as_ref().unwrap();
    assert!(doc.segments[0].alignment.is_some());
    assert!(doc.segments[0].rest.contains_key("reading_alignment"));
    assert_eq!(candidate["start"], 2.00004);
    assert_eq!(candidate["end"], 4.00009);
    assert_eq!(candidate["speaker"], "speaker_0");
    assert_eq!(candidate["original_text"], "Keep these words.");
    assert!(candidate["id"].is_string());
    assert!(candidate["created"].is_number());
    assert_eq!(candidate["retry_parts"][0]["start"], 0.);
    let mut doc = serde_json::to_value(doc).unwrap();
    doc["segments"][0]["text"] = candidate["text"].clone();
    doc["segments"][0]["language"] = json!("fr");
    doc["segments"][0]["language_detection"] = json!({"mode":"manual","reason":"passage_retry"});
    let accepted = running
        .authed(
            running
                .client
                .put(running.url(&format!("/api/jobs/{JOB_A}/transcript"))),
        )
        .json(&json!({"revision":10,"document":doc}))
        .send()
        .await
        .unwrap();
    assert_eq!(accepted.status(), 200);
    let saved: Value = accepted.json().await.unwrap();
    assert_eq!(
        saved["document"]["segments"][0]["retry_candidate"]["replaced_text"],
        "Keep these words."
    );
    assert_eq!(saved["document"]["segments"][0]["language"], "fr");
    assert!(saved["document"]["segments"][0]["alignment"].is_null());
    assert!(
        saved["document"]["segments"][0]
            .get("reading_alignment")
            .is_none()
    );
    let stale = running
        .authed(
            running
                .client
                .put(running.url(&format!("/api/jobs/{JOB_A}/transcript"))),
        )
        .json(&json!({"revision":10,"document":doc}))
        .send()
        .await
        .unwrap();
    assert_eq!(stale.status(), 409);
    let disk =
        speakerdesk_core::store::SqliteJobStore::open(state.data_dir().join("speakerdesk.sqlite"))
            .unwrap()
            .get(&JobId::parse(JOB_A).unwrap(), false)
            .unwrap()
            .unwrap();
    assert_eq!(
        serde_json::to_value(disk.document).unwrap(),
        saved["document"]
    );
    running.shutdown().await;
}

#[tokio::test]
async fn retry_stale_cancel_failure_and_restart_boundaries() {
    for case in [
        "text",
        "bounds",
        "speaker",
        "language",
        "audio",
        "source",
        "catalog",
        "cancel",
        "shutdown",
        "failure",
        "queued_cancel",
    ] {
        let (state, models) = if matches!(case, "source" | "catalog") {
            source_fixture()
        } else {
            fixture(case == "failure")
        };
        let running = Running::spawn(state.clone()).await;
        let _release = ReleaseOnDrop(models.clone());
        let queue_gate = Arc::new((Mutex::new(false), Condvar::new()));
        if case == "queued_cancel" {
            let gate = queue_gate.clone();
            state.worker.enqueue(move || {
                let mut ready = gate.0.lock().unwrap();
                while !*ready {
                    ready = gate.1.wait(ready).unwrap();
                }
            });
        }
        assert_eq!(start(&running, 7).await.status(), 202, "{case}");
        if case != "queued_cancel" {
            wait_until(&models.entered, 1).await;
        }
        let id = JobId::parse(JOB_A).unwrap();
        let admitted = state.store().get(&id, false).unwrap().unwrap();
        if case == "cancel" || case == "queued_cancel" {
            let response = running
                .authed(
                    running
                        .client
                        .post(running.url(&format!("/api/jobs/{JOB_A}/cancel"))),
                )
                .send()
                .await
                .unwrap();
            assert_eq!(response.status(), 200);
        } else if case == "shutdown" {
            state.cancel_all().unwrap();
        } else if case != "failure" {
            state
                .store()
                .update_job(&id, |job| {
                    let s = &mut job.document.as_mut().unwrap().segments[0];
                    match case {
                        "text" => s.text = "Protected user edit.".into(),
                        "bounds" => s.end = 3.,
                        "speaker" => s.speaker = "unassigned".into(),
                        "language" => s.language = Some("es".into()),
                        "audio" => s.audio_revision = Some(2),
                        "source" => {
                            s.capture_source.as_mut().unwrap()["source_id"] =
                                json!("microphone_clean")
                        }
                        "catalog" => {
                            job.extra.get_mut("capture_source_catalog").unwrap()["policy"] =
                                json!("changed")
                        }
                        _ => unreachable!(),
                    }
                    Ok::<_, speakerdesk_core::error::StoreError>(())
                })
                .unwrap();
        }
        if ["cancel", "shutdown", "queued_cancel"].contains(&case) {
            let cancelled = state.store().get(&id, false).unwrap().unwrap();
            assert_ne!(
                cancelled.extra["operation_generation"],
                admitted.extra["operation_generation"]
            );
            assert!(!cancelled.extra.contains_key("passage_retry"));
            assert_eq!(cancelled.status, "ready");
            assert!(
                state.admission.busy(),
                "admission must wait for worker cleanup"
            );
        }
        *queue_gate.0.lock().unwrap() = true;
        queue_gate.1.notify_all();
        models.release();
        state.drain();
        let completed = state.store().get(&id, false).unwrap().unwrap();
        assert!(!completed.extra.contains_key("passage_retry"), "{case}");
        assert_eq!(completed.status, "ready");
        assert!(!state.admission.busy());
        let segment = &completed.document.as_ref().unwrap().segments[0];
        if case == "failure" {
            assert_eq!(
                segment.retry_candidate.as_ref().unwrap()["error"],
                messages::RETRY_RETAINED
            );
        } else {
            assert!(segment.retry_candidate.is_none(), "{case}");
        }
        assert_eq!(
            segment.text,
            if case == "text" {
                "Protected user edit."
            } else {
                "Keep these words."
            }
        );
        assert_eq!(
            models.entered.load(Ordering::SeqCst),
            usize::from(case != "queued_cancel")
        );
        running.shutdown().await;
    }
    let (state, models) = fixture(false);
    models.release();
    let id = JobId::parse(JOB_A).unwrap();
    let mut legacy = job(JOB_B, "ready", 2.);
    legacy.extra.insert("passage_retry".into(), Value::Null);
    state.insert_job(legacy);
    let legacy = state
        .store()
        .get(&JobId::parse(JOB_B).unwrap(), false)
        .unwrap()
        .unwrap();
    state
        .store()
        .update_job(&id, |job| {
            job.extra
                .insert("passage_retry".into(), json!({"id":"interrupted"}));
            Ok::<_, speakerdesk_core::error::StoreError>(())
        })
        .unwrap();
    let before = state.store().get(&id, false).unwrap().unwrap();
    let path = state.data_dir().to_path_buf();
    drop(state);
    let reopened = AppState::open(path.clone()).unwrap();
    let after = reopened.store().get(&id, false).unwrap().unwrap();
    assert_eq!(
        reopened
            .store()
            .get(&JobId::parse(JOB_B).unwrap(), false)
            .unwrap()
            .unwrap(),
        legacy
    );
    assert_eq!(after.status, "ready");
    assert_eq!(after.document, before.document);
    assert!(!after.extra.contains_key("passage_retry"));
    assert_eq!(after.revision, before.revision + 1);
    drop(reopened);
    let again = AppState::open(path.clone()).unwrap();
    assert_eq!(again.store().get(&id, false).unwrap().unwrap(), after);
    drop(again);
    std::fs::remove_dir_all(path).unwrap();
}
