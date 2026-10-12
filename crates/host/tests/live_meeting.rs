#![cfg(all(feature = "fixtures", unix))]
//! Mounted native HTTP + owned helper/model subprocess replay. No hardware or acoustic claim.
use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use std::{fs, os::unix::fs::PermissionsExt, path::Path, process::Stdio, time::Duration};
use tokio::{
    io::{AsyncBufReadExt, AsyncWriteExt, BufReader},
    process::{Child, ChildStdin, Command},
};

struct Running {
    child: Child,
    input: ChildStdin,
    url: String,
    token: String,
    client: reqwest::Client,
}
impl Running {
    async fn open(root: &Path, dual: bool) -> Self {
        fs::write(root.join("fixture-live"), b"native-host protocol replay").unwrap();
        let helper = root.join("capture-helper");
        let quote = |s: &str| format!("'{}'", s.replace('\'', "'\\''"));
        fs::write(
            &helper,
            format!(
                "#!/bin/sh\nexec {} --capture {}\n",
                quote(env!("CARGO_BIN_EXE_inference-fixture")),
                quote(root.to_str().unwrap())
            ),
        )
        .unwrap();
        fs::set_permissions(&helper, fs::Permissions::from_mode(0o700)).unwrap();
        if dual {
            let digest = format!("{:x}", Sha256::digest(fs::read(&helper).unwrap()));
            fs::write(root.join("capture-helper.capability.json"),json!({"helper_sha256":digest,"export_contract":"apple_processed_mono16k_v1","capability_version":"apple_input_node_output_route_v1"}).to_string()).unwrap();
            let format = json!({"sample_rate":48000,"channels":1,"layout":"mono"});
            fs::write(root.join("receipt.json"),json!({"type":"capture_processing","schema_version":1,"job_id":"replaced-by-owned-start","helper_sha256":digest,
                "processor":"apple_voice_processing_io","export_contract":"apple_processed_mono16k_v1","output":"processed_microphone","stage":"retained_source_pcm","postprocessing":"identity",
                "stream_epoch":0,"format":{"sample_rate":16000,"channels":1,"layout":"mono"},"capability_version":"apple_input_node_output_route_v1",
                "route":{"node":"inputNode","bus":0,"tap_scope":"output"},"state_phase":"after_start","channel_mapping":"mono_direct","callback_format":format,"io_formats":{"input_output":format,"output_input":format},
                "readbacks":{"input_enabled":true,"output_enabled":true,"engine_running":true,"bypassed":false,"manual_rendering":false,"input_muted":false,"agc_enabled":false}}).to_string()).unwrap();
        }
        let mut child = Command::new(env!("CARGO_BIN_EXE_inference-fixture"))
            .arg("--native-host")
            .arg(root.join("data"))
            .arg(root)
            .env("SPEAKERDESK_CAPTURE_HELPER", &helper)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::inherit())
            .kill_on_drop(true)
            .spawn()
            .unwrap();
        let input = child.stdin.take().unwrap();
        let mut output = BufReader::new(child.stdout.take().unwrap()).lines();
        let line = tokio::time::timeout(Duration::from_secs(10), output.next_line())
            .await
            .unwrap()
            .unwrap()
            .unwrap();
        let url = line.strip_prefix("SPEAKERDESK_URL=").unwrap().to_owned();
        // Keep draining control output, including shutdown acknowledgements.
        tokio::spawn(async move { while output.next_line().await.ok().flatten().is_some() {} });
        let client = reqwest::Client::builder()
            .timeout(Duration::from_secs(20))
            .build()
            .unwrap();
        let html = client.get(&url).send().await.unwrap().text().await.unwrap();
        let token = html
            .split("name=\"speakerdesk-token\" content=\"")
            .nth(1)
            .unwrap()
            .split('"')
            .next()
            .unwrap()
            .into();
        let running = Self {
            child,
            input,
            url,
            token,
            client,
        };
        running
            .wait("api/config", |v| v["readiness"]["configured"] == true)
            .await;
        assert_eq!(running.get("api/meeting").await["capture_available"], true);
        running
    }
    async fn get(&self, path: &str) -> Value {
        self.client
            .get(format!("{}{path}", self.url))
            .send()
            .await
            .unwrap()
            .error_for_status()
            .unwrap()
            .json()
            .await
            .unwrap()
    }
    async fn wait(&self, path: &str, predicate: impl Fn(&Value) -> bool) -> Value {
        tokio::time::timeout(Duration::from_secs(15), async {
            loop {
                let value = self.get(path).await;
                if predicate(&value) {
                    return value;
                }
                tokio::time::sleep(Duration::from_millis(15)).await;
            }
        })
        .await
        .unwrap_or_else(|_| panic!("timed out at {path}"))
    }
    async fn post(&self, path: &str, body: Value) -> reqwest::Response {
        self.client
            .post(format!("{}{path}", self.url))
            .header("X-Speakerdesk-Token", &self.token)
            .json(&body)
            .send()
            .await
            .unwrap()
    }
    async fn start(&self, sources: Value) -> String {
        let response = self
            .post(
                "api/meetings",
                json!({"name":"Disposable live replay","language":"en","sources":sources}),
            )
            .await;
        assert_eq!(response.status(), 202, "{}", response.text().await.unwrap());
        // The only current meeting is returned by the public status surface.
        self.wait("api/meeting", |v| v["id"].is_string()).await["id"]
            .as_str()
            .unwrap()
            .into()
    }
    async fn stop(&self, id: &str) -> Value {
        let response = self
            .post(&format!("api/meetings/{id}/stop"), json!({}))
            .await;
        assert_eq!(response.status(), 202, "{}", response.text().await.unwrap());
        self.wait(&format!("api/jobs/{id}"), |v| {
            ["ready", "failed"].contains(&v["status"].as_str().unwrap_or(""))
                && v["inference_owned"] == false
        })
        .await
    }
    async fn close(mut self, root: &Path) {
        let capture = fs::read_to_string(root.join("capture-pid"))
            .ok()
            .map(|s| s.parse::<i32>().unwrap());
        let inference = fs::read_to_string(root.join("fixture-child-pid"))
            .ok()
            .map(|s| s.parse::<i32>().unwrap());
        self.input.write_all(b"shutdown\n").await.unwrap();
        assert!(
            tokio::time::timeout(Duration::from_secs(25), self.child.wait())
                .await
                .unwrap()
                .unwrap()
                .success()
        );
        for pid in capture.into_iter().chain(inference) {
            assert_eq!(
                unsafe { libc::kill(pid, 0) },
                -1,
                "owned process {pid} survived host cleanup"
            );
        }
        if let Ok(pids) = fs::read_to_string(root.join("fixture-child-pids")) {
            for pid in pids.lines().map(|s| s.parse::<i32>().unwrap()) {
                assert_eq!(
                    unsafe { libc::kill(pid, 0) },
                    -1,
                    "owned recovery child survived cleanup"
                );
            }
        }
    }
}
fn trigger(root: &Path, stage: u32) {
    fs::write(root.join(format!("stage-{stage}")), b"replay").unwrap();
}

async fn wait_file(path: &Path) {
    tokio::time::timeout(Duration::from_secs(10), async {
        while !path.exists() {
            tokio::time::sleep(Duration::from_millis(10)).await;
        }
    })
    .await
    .unwrap();
}

#[tokio::test]
async fn exceptional_recovery_preserves_history_and_is_bounded_through_failure_or_shutdown() {
    for mode in [
        "recover",
        "dual-recover",
        "fail-recovery",
        "shutdown-recovery",
    ] {
        let temp = tempfile::tempdir().unwrap();
        let root = temp.path();
        let dual = mode == "dual-recover";
        let host = Running::open(root, dual).await;
        trigger(root, 1);
        let id = host
            .start(if dual {
                json!(["microphone", "system"])
            } else {
                json!(["microphone"])
            })
            .await;
        let path = format!("api/jobs/{id}");
        let first = host
            .wait(&path, |v| {
                v["document"]["segments"]
                    .as_array()
                    .is_some_and(|rows| rows.iter().any(|r| r["text"] == "Replay transcript"))
                    && v["duration"].as_f64().unwrap_or(0.) >= 6.5
            })
            .await;
        let row = &first["document"]["segments"][0];
        let old_id = row["id"].clone();
        let old_speaker = row["speaker"].as_str().unwrap().to_owned();
        let save=host.client.patch(format!("{}{path}/segments/{}",host.url,old_id.as_str().unwrap())).header("X-Speakerdesk-Token",&host.token)
            .json(&json!({"segment_revision":row["machine_revision"],"changes":{"text":"Protected before child loss","start":0.1}})).send().await.unwrap();
        assert_eq!(save.status(), 200, "{}", save.text().await.unwrap());
        // Protecting an open row schedules its continuation, which may publish
        // between this GET and naming. Retry only that genuine revision conflict.
        tokio::time::timeout(Duration::from_secs(5), async {
            loop {
                let current = host.get(&path).await;
                let named = host
                    .post(
                        &format!("{path}/speakers/{old_speaker}/identity"),
                        json!({"revision":current["revision"],"name":"Casey","person_id":null}),
                    )
                    .await;
                let status = named.status();
                if status == 200 {
                    break;
                }
                let body: Value = named.json().await.unwrap();
                assert_eq!(status, 409, "{body}");
                assert_eq!(body["error"], speakerdesk_core::messages::NAME_STALE);
                tokio::time::sleep(Duration::from_millis(10)).await;
            }
        })
        .await
        .expect("pre-loss naming did not reach a current revision");
        // Persist a real pre-loss language boundary; replay must reconstruct it.
        let changed = host
            .client
            .patch(format!("{}api/meetings/{id}/language", host.url))
            .header("X-Speakerdesk-Token", &host.token)
            .json(&json!({"language":"fr","language_revision":0}))
            .send()
            .await
            .unwrap();
        assert_eq!(changed.status(), 200);
        host.wait("api/meeting", |v| v["language_acknowledged_revision"] == 1)
            .await;
        let old = host.get(&path).await;
        let history = old["language_history"].clone();
        let old_pid: i32 = fs::read_to_string(root.join("fixture-child-pid"))
            .unwrap()
            .parse()
            .unwrap();
        fs::write(root.join("hold-live-feed"), b"block recovered worker only").unwrap();
        assert_eq!(unsafe { libc::kill(old_pid, libc::SIGKILL) }, 0);
        trigger(root, 2);
        wait_file(&root.join("recovery-feed-started")).await;
        let replay = host
            .wait(&path, |v| v["live_recovery"]["status"] == "replaying")
            .await;
        assert_eq!(replay["live_stream_generation"], 1);
        assert_eq!(replay["language_history"], history);
        let new_pid: i32 = fs::read_to_string(root.join("fixture-child-pid"))
            .unwrap()
            .parse()
            .unwrap();
        assert_ne!(new_pid, old_pid);
        assert_eq!(unsafe { libc::kill(old_pid, 0) }, -1);
        let before = replay["document"].clone();
        trigger(root, 3);
        host.wait("api/meeting", |v| {
            v["duration"].as_f64().unwrap_or(0.) >= 20.5
        })
        .await;
        if mode == "recover" {
            assert_eq!(
                host.post(&format!("api/meetings/{id}/pause"), json!({}))
                    .await
                    .status(),
                202
            );
            host.wait("api/meeting", |v| {
                v["status"] == "paused" && v["pause_flush"]["status"] == "pending"
            })
            .await;
            assert_eq!(
                host.post(&format!("api/meetings/{id}/resume"), json!({}))
                    .await
                    .status(),
                409
            );
        }
        match mode {
            "shutdown-recovery" => {
                host.close(root).await;
                fs::remove_file(root.join("hold-live-feed")).unwrap();
                let reopened = Running::open(root, dual).await;
                let stored = reopened.get(&path).await;
                assert_eq!(stored["status"], "failed");
                assert_eq!(stored["document"], before);
                reopened.close(root).await;
            }
            "fail-recovery" => {
                assert_eq!(unsafe { libc::kill(new_pid, libc::SIGKILL) }, 0);
                host.wait(&path, |v| v["live_recovery"]["status"] == "failed")
                    .await;
                trigger(root, 4);
                host.wait("api/meeting", |v| {
                    v["duration"].as_f64().unwrap_or(0.) >= 27.5
                })
                .await;
                let failed = host.stop(&id).await;
                assert_eq!(failed["status"], "failed");
                assert_eq!(failed["document"], before);
                assert_eq!(
                    fs::read_to_string(root.join("fixture-child-pids"))
                        .unwrap()
                        .lines()
                        .count(),
                    2,
                    "failure spun up another child"
                );
                host.close(root).await;
            }
            _ => {
                fs::remove_file(root.join("hold-live-feed")).unwrap();
                host.wait(&path, |v| v["live_recovery"]["status"] == "recovered")
                    .await;
                if mode == "recover" {
                    let paused = host
                        .wait("api/meeting", |v| v["pause_flush"]["status"] == "complete")
                        .await;
                    assert_eq!(paused["pause_flush"]["received_sample"], 336000);
                    assert_eq!(
                        host.post(&format!("api/meetings/{id}/resume"), json!({}))
                            .await
                            .status(),
                        202
                    );
                }
                let final_job = host.stop(&id).await;
                assert_eq!(final_job["status"], "ready", "{final_job}");
                assert_eq!(final_job["canonical_observed_sample"], 336513);
                assert_eq!(final_job["language_history"], history);
                let rows = final_job["document"]["segments"].as_array().unwrap();
                let ids: std::collections::HashSet<_> =
                    rows.iter().map(|r| r["id"].as_str().unwrap()).collect();
                assert_eq!(ids.len(), rows.len());
                let old = rows.iter().find(|r| r["id"] == old_id).unwrap();
                assert_eq!(old["text"], "Protected before child loss");
                assert_eq!(old["start"], 0.1);
                assert_eq!(old["speaker"], old_speaker);
                assert_eq!(old["canonical_state"], "sealed");
                assert_eq!(old["review"], true);
                assert_eq!(final_job["document"]["speakers"][&old_speaker], "Casey");
                let boundary = final_job["live_recovery"]["from_sample"].as_i64().unwrap();
                let new: Vec<_> = rows
                    .iter()
                    .filter(|r| r["stream_generation"] == 1)
                    .collect();
                assert!(!new.is_empty());
                for row in new {
                    assert!(row["start_sample"].as_i64().unwrap() >= boundary);
                    assert_ne!(row["speaker"], old_speaker);
                    if dual {
                        assert!(row["capture_source"]["catalog_sha256"].is_string());
                        assert_eq!(row["capture_source"]["source_revision"], 1);
                    }
                }
                assert!(
                    final_job["document"]["speakers"]
                        .as_object()
                        .unwrap()
                        .values()
                        .all(|v| v != "Speaker 65" && v != "Speaker 97")
                );
                assert!(final_job["live_recovery"]["elapsed_ms"].is_u64());
                host.close(root).await;
                let reopened = Running::open(root, dual).await;
                assert_eq!(reopened.get(&path).await["document"], final_job["document"]);
                let export = reopened
                    .client
                    .get(format!("{}{path}/export/txt", reopened.url))
                    .send()
                    .await
                    .unwrap()
                    .text()
                    .await
                    .unwrap();
                assert!(export.contains("Protected before child loss") && export.contains("Casey"));
                reopened.close(root).await;
            }
        }
    }
}

#[tokio::test]
async fn recovery_covers_asr_and_final_eof_child_loss() {
    for eof in [false, true] {
        let temp = tempfile::tempdir().unwrap();
        let root = temp.path();
        let host = Running::open(root, false).await;
        trigger(root, 1);
        let id = host.start(json!(["microphone"])).await;
        let path = format!("api/jobs/{id}");
        host.wait(&path, |v| {
            v["document"]["segments"]
                .as_array()
                .is_some_and(|s| s.iter().any(|r| r["text"] == "Replay transcript"))
        })
        .await;
        let old_pid: i32 = fs::read_to_string(root.join("fixture-child-pid"))
            .unwrap()
            .parse()
            .unwrap();
        let final_job = if eof {
            fs::write(root.join("hold-live-eof"), b"block final model flush").unwrap();
            assert_eq!(
                host.post(&format!("api/meetings/{id}/stop"), json!({}))
                    .await
                    .status(),
                202
            );
            wait_file(&root.join("recovery-eof-started")).await;
            assert_eq!(unsafe { libc::kill(old_pid, libc::SIGKILL) }, 0);
            fs::remove_file(root.join("hold-live-eof")).unwrap();
            host.wait(&path, |v| {
                ["ready", "failed"].contains(&v["status"].as_str().unwrap_or(""))
                    && v["inference_owned"] == false
            })
            .await
        } else {
            fs::write(root.join("exit-live-asr-once"), b"fail inside recognizer").unwrap();
            trigger(root, 2);
            host.wait(&path, |v| v["live_recovery"]["status"] == "recovered")
                .await;
            host.stop(&id).await
        };
        assert_eq!(final_job["status"], "ready", "{final_job}");
        assert_eq!(final_job["live_recovery"]["status"], "recovered");
        assert_eq!(
            final_job["canonical_observed_sample"],
            if eof { 112513 } else { 224513 }
        );
        assert_eq!(
            fs::read_to_string(root.join("fixture-child-pids"))
                .unwrap()
                .lines()
                .count(),
            2
        );
        assert_eq!(unsafe { libc::kill(old_pid, 0) }, -1);
        host.close(root).await;
    }
}

#[tokio::test]
async fn editing_open_live_words_preserves_the_edit_and_publishes_continued_speech() {
    for mode in [
        "text", "bounds", "inflight", "empty", "language", "recovery",
    ] {
        let temp = tempfile::tempdir().unwrap();
        let root = temp.path();
        if mode == "empty" {
            fs::write(root.join("first-stage-samples"), b"32000").unwrap();
        }
        let host = Running::open(root, false).await;
        trigger(root, 1);
        let id = host.start(json!(["microphone"])).await;
        let path = format!("api/jobs/{id}");
        host.wait(&path, |v| {
            v["document"]["segments"].as_array().is_some_and(|rows| {
                rows.iter().any(|r| {
                    if mode == "empty" {
                        r["text"] == ""
                    } else {
                        r["text"] == "Replay transcript"
                    }
                })
            })
        })
        .await;
        host.wait("api/meeting", |v| {
            v["processed_seconds"].as_f64().unwrap_or(0.) >= if mode == "empty" { 1.4 } else { 6.4 }
        })
        .await;
        let first = if mode == "inflight" {
            fs::write(root.join("hold-live-asr"), b"hold next decoder response").unwrap();
            trigger(root, 2);
            wait_file(&root.join("edit-asr-started")).await;
            host.get(&path).await
        } else {
            host.get(&path).await
        };
        let row = &first["document"]["segments"][0];
        let sid = row["id"].as_str().unwrap();
        let coverage = row["canonical_text_end_sample"]
            .as_i64()
            .unwrap_or_else(|| row["end_sample"].as_i64().unwrap());
        assert_eq!(row["canonical_state"], "open", "{mode}");
        let changes = if mode == "bounds" || mode == "inflight" {
            json!({"text":"Human corrected introduction","start":0.1,"end":6.})
        } else {
            json!({"text":"Human corrected introduction"})
        };
        let edited = host
            .client
            .patch(format!("{}{path}/segments/{sid}", host.url))
            .header("X-Speakerdesk-Token", &host.token)
            .json(&json!({"segment_revision":row["machine_revision"], "changes":changes}))
            .send()
            .await
            .unwrap();
        assert_eq!(
            edited.status(),
            200,
            "{mode}: {}",
            edited.text().await.unwrap()
        );
        let saved = host.get(&path).await;
        assert_eq!(
            host.post(
                &format!("{path}/speakers/speaker_0/identity"),
                json!({"revision":saved["revision"],"name":"Casey","person_id":null})
            )
            .await
            .status(),
            200
        );
        if mode == "inflight" {
            fs::remove_file(root.join("hold-live-asr")).unwrap();
        }
        if mode == "language" {
            let response = host
                .client
                .patch(format!("{}api/meetings/{id}/language", host.url))
                .header("X-Speakerdesk-Token", &host.token)
                .json(&json!({"language":"de","language_revision":0}))
                .send()
                .await
                .unwrap();
            assert_eq!(response.status(), 200);
            host.wait("api/meeting", |v| v["language_acknowledged_revision"] == 1)
                .await;
        }
        if mode == "recovery" {
            let pid: i32 = fs::read_to_string(root.join("fixture-child-pid"))
                .unwrap()
                .parse()
                .unwrap();
            assert_eq!(unsafe { libc::kill(pid, libc::SIGKILL) }, 0);
        }
        trigger(root, 2);
        trigger(root, 3);
        let end = if mode == "empty" { 256513 } else { 336513 };
        host.wait("api/meeting", |v| {
            v["duration"].as_f64().unwrap_or(0.) >= (end - 8513) as f64 / 16000.
        })
        .await;
        let final_job = host.stop(&id).await;
        assert_eq!(final_job["status"], "ready", "{mode}: {final_job}");
        let rows = final_job["document"]["segments"].as_array().unwrap();
        let original = rows.iter().find(|r| r["id"] == sid).unwrap();
        assert_eq!(original["text"], "Human corrected introduction");
        if mode == "bounds" || mode == "inflight" {
            assert_eq!(original["start"], 0.1);
            assert_eq!(original["end"], 6.);
        }
        assert_eq!(final_job["document"]["speakers"]["speaker_0"], "Casey");
        let continuations: Vec<_> = rows
            .iter()
            .filter(|r| r["id"] != sid && r["text"] == "Replay transcript")
            .collect();
        assert!(
            !continuations.is_empty(),
            "{mode}: continued speech disappeared: {final_job}"
        );
        assert!(
            continuations
                .iter()
                .all(|r| r["start_sample"].as_i64().unwrap() >= coverage)
        );
        if mode != "language" {
            assert_eq!(continuations[0]["start_sample"], coverage);
        }
        if mode == "language" {
            assert!(continuations.iter().any(|r| r["language"] == "de"));
        }
        if mode == "recovery" {
            assert_eq!(final_job["live_recovery"]["status"], "recovered");
        }
        let ids: std::collections::HashSet<_> =
            rows.iter().map(|r| r["id"].as_str().unwrap()).collect();
        assert_eq!(ids.len(), rows.len());
        assert_eq!(final_job["canonical_observed_sample"], end);
        host.close(root).await;
        if mode == "inflight" {
            let reopened = Running::open(root, false).await;
            assert_eq!(reopened.get(&path).await["document"], final_job["document"]);
            let text = reopened
                .client
                .get(format!("{}{path}/export/txt", reopened.url))
                .send()
                .await
                .unwrap()
                .text()
                .await
                .unwrap();
            assert!(
                text.contains("Human corrected introduction")
                    && text.contains("Replay transcript")
                    && text.contains("Casey")
            );
            let stale = reopened
                .client
                .put(format!("{}{path}/transcript", reopened.url))
                .header("X-Speakerdesk-Token", &reopened.token)
                .json(&json!({"revision":first["revision"], "document":first["document"]}))
                .send()
                .await
                .unwrap();
            assert_eq!(stale.status(), 409);
            reopened.close(root).await;
        }
    }
}

#[tokio::test]
async fn concurrent_starts_and_terminal_replacement_create_only_owned_jobs() {
    let temp = tempfile::tempdir().unwrap();
    let root = temp.path();
    let host = Running::open(root, false).await;
    for expected in 1..=2 {
        let request = json!({"name":"Concurrent replay","language":"en","sources":["microphone"]});
        let (a, b) = tokio::join!(
            host.post("api/meetings", request.clone()),
            host.post("api/meetings", request)
        );
        let mut statuses = vec![a.status().as_u16(), b.status().as_u16()];
        statuses.sort();
        assert_eq!(statuses, vec![202, 409]);
        assert_eq!(
            host.get("api/jobs").await.as_array().unwrap().len(),
            expected
        );
        let id = host.get("api/meeting").await["id"]
            .as_str()
            .unwrap()
            .to_owned();
        let finished = host.stop(&id).await;
        assert_eq!(finished["status"], "ready", "{finished}");
    }
    host.close(root).await;
}

#[tokio::test]
async fn live_saved_audio_candidates_preserve_edits_language_eof_and_reopen() {
    let temp = tempfile::tempdir().unwrap();
    let root = temp.path();
    let host = Running::open(root, false).await;
    trigger(root, 1);
    let id = host.start(json!(["microphone"])).await;
    let path = format!("api/jobs/{id}");
    let first = host
        .wait(&path, |v| {
            v["document"]["segments"]
                .as_array()
                .is_some_and(|s| s.iter().any(|s| s["text"] == "Replay transcript"))
        })
        .await;
    assert!(first.get("operation_generation").is_none());
    assert!(first.get("rolling_refinement").is_none());
    assert_eq!(
        host.post(
            "api/meetings",
            json!({"language":"en","sources":["system"]})
        )
        .await
        .status(),
        409
    );
    let segment_id = first["document"]["segments"][0]["id"].clone();
    let end = first["document"]["segments"][0]["end"].as_f64().unwrap() - 0.1;
    let saved=host.client.patch(format!("{}{path}/segments/{}",host.url,segment_id.as_str().unwrap())).header("X-Speakerdesk-Token",&host.token).json(&json!({"segment_revision":first["document"]["segments"][0]["machine_revision"],"changes":{"text":"Words corrected during capture","start":0.1,"end":end}})).send().await.unwrap();
    assert_eq!(saved.status(), 200, "{}", saved.text().await.unwrap());
    let current = host.get(&path).await;
    let named = host
        .post(
            &format!("{path}/speakers/speaker_0/identity"),
            json!({"revision":current["revision"],"name":"Casey","person_id":null}),
        )
        .await;
    assert_eq!(named.status(), 200, "{}", named.text().await.unwrap());
    trigger(root, 2);
    host.wait("api/meeting", |v| {
        v["processed_seconds"].as_f64().unwrap_or(0.) >= 12.
    })
    .await;
    let current = host.get(&path).await;
    let row = current["document"]["segments"]
        .as_array()
        .unwrap()
        .iter()
        .find(|s| s["id"] == segment_id)
        .unwrap();
    assert_eq!(row["text"], "Words corrected during capture");
    assert_eq!(row["start"], 0.1);
    assert_eq!(row["end"], end);
    assert_eq!(current["document"]["speakers"]["speaker_0"], "Casey");
    assert_eq!(
        host.post(&format!("api/meetings/{id}/pause"), json!({}))
            .await
            .status(),
        202
    );
    host.wait("api/meeting", |v| {
        v["status"] == "paused" && v["pause_flush"]["status"] == "complete"
    })
    .await;
    let changed = host
        .client
        .patch(format!("{}api/meetings/{id}/language", host.url))
        .header("X-Speakerdesk-Token", &host.token)
        .json(&json!({"language":"fr","language_revision":0}))
        .send()
        .await
        .unwrap();
    assert_eq!(changed.status(), 200);
    let changed: Value = changed.json().await.unwrap();
    assert_eq!(changed["language_history"][1]["start_sample"], 224000);
    assert_eq!(changed["default_language"], "fr");
    assert_eq!(
        host.client
            .patch(format!("{}api/meetings/{id}/language", host.url))
            .header("X-Speakerdesk-Token", &host.token)
            .json(&json!({"language":"en","language_revision":0}))
            .send()
            .await
            .unwrap()
            .status(),
        409
    );
    host.wait("api/meeting", |v| v["language_acknowledged_revision"] == 1)
        .await;
    assert_eq!(
        host.post(&format!("api/meetings/{id}/resume"), json!({}))
            .await
            .status(),
        202
    );
    trigger(root, 3);
    host.wait("api/meeting", |v| {
        v["duration"].as_f64().unwrap_or(0.) >= 20.5
    })
    .await;
    let final_job = host.stop(&id).await;
    assert_eq!(final_job["status"], "ready", "{final_job}");
    assert_eq!(final_job["duration"], 336513. / 16000.);
    assert_eq!(final_job["canonical_observed_sample"], 336513);
    let rows = final_job["document"]["segments"].as_array().unwrap();
    assert!(rows.iter().any(|s| s["language"] == "en"));
    assert!(rows.iter().any(|s| s["language"] == "fr"));
    let exported = host
        .client
        .get(format!("{}{path}/export/txt", host.url))
        .send()
        .await
        .unwrap()
        .text()
        .await
        .unwrap();
    assert!(exported.contains("Words corrected during capture"));
    assert!(exported.contains("Casey"));
    host.close(root).await;
    let reopened = Running::open(root, false).await;
    let after = reopened.get(&path).await;
    assert_eq!(after["document"], final_job["document"]);
    assert_eq!(after["revision"], final_job["revision"]);
    let stale = reopened
        .client
        .put(format!("{}{path}/transcript", reopened.url))
        .header("X-Speakerdesk-Token", &reopened.token)
        .json(&json!({"revision":first["revision"],"document":first["document"]}))
        .send()
        .await
        .unwrap();
    assert_eq!(stale.status(), 409);
    assert_eq!(reopened.get(&path).await["document"], after["document"]);
    reopened.close(root).await;
}

#[tokio::test]
async fn live_source_bindings_qualified_pause_and_whole_host_shutdown_are_honest() {
    for mode in [
        "dual-stop",
        "dual-pause",
        "shutdown",
        "capture-error",
        "inference-error",
    ] {
        let temp = tempfile::tempdir().unwrap();
        let root = temp.path();
        let dual = mode.starts_with("dual");
        let host = Running::open(root, dual).await;
        trigger(root, 1);
        let id = host
            .start(if dual {
                json!(["microphone", "system"])
            } else {
                json!(["system"])
            })
            .await;
        let path = format!("api/jobs/{id}");
        let live = host
            .wait(&path, |v| {
                v["document"]["segments"]
                    .as_array()
                    .is_some_and(|s| !s.is_empty())
            })
            .await;
        if dual {
            let live = host
                .wait(&path, |v| {
                    v["document"]["segments"]
                        .as_array()
                        .is_some_and(|s| s.len() >= 2)
                })
                .await;
            let rows = live["document"]["segments"].as_array().unwrap();
            let sources: std::collections::BTreeSet<_> = rows
                .iter()
                .map(|s| s["capture_source"]["source_id"].as_str().unwrap())
                .collect();
            assert_eq!(
                sources,
                std::collections::BTreeSet::from(["microphone_clean", "system"])
            );
            assert_ne!(rows[0]["id"], rows[1]["id"]);
            assert_ne!(rows[0]["speaker"], rows[1]["speaker"]);
            for row in rows {
                assert_eq!(row["capture_source"]["source_revision"], 1);
                assert_eq!(
                    row["capture_source"]["catalog_sha256"]
                        .as_str()
                        .unwrap()
                        .len(),
                    64
                );
            }
            assert!(live["capture_processing"]["sha256"].is_string());
        }
        match mode {
            "dual-stop" => {
                assert_eq!(host.stop(&id).await["status"], "ready");
                host.close(root).await;
            }
            "dual-pause" => {
                assert_eq!(
                    host.post(&format!("api/meetings/{id}/pause"), json!({}))
                        .await
                        .status(),
                    202
                );
                let failed = host
                    .wait(&path, |v| {
                        v["status"] == "failed" && v["inference_owned"] == false
                    })
                    .await;
                assert!(failed["message"].as_str().unwrap().contains("Pause"));
                assert!(failed["duration"].as_f64().unwrap() >= 7.);
                assert_eq!(
                    host.post(&format!("api/meetings/{id}/resume"), json!({}))
                        .await
                        .status(),
                    409
                );
                host.close(root).await;
            }
            "capture-error" => {
                fs::write(root.join("fail-capture"), b"fail").unwrap();
                let failed = host
                    .wait(&path, |v| {
                        v["status"] == "failed" && v["inference_owned"] == false
                    })
                    .await;
                assert!(
                    !failed["document"]["segments"]
                        .as_array()
                        .unwrap()
                        .is_empty()
                );
                host.close(root).await;
            }
            "inference-error" => {
                let child: i32 = fs::read_to_string(root.join("fixture-child-pid"))
                    .unwrap()
                    .parse()
                    .unwrap();
                assert_eq!(unsafe { libc::kill(child, libc::SIGKILL) }, 0);
                trigger(root, 2);
                host.wait(&path, |v| v["live_recovery"]["status"] == "recovered")
                    .await;
                trigger(root, 3);
                host.wait("api/meeting", |v| {
                    v["duration"].as_f64().unwrap_or(0.) >= 20.5
                })
                .await;
                let failed = host.stop(&id).await;
                assert_eq!(failed["status"], "ready");
                assert!(
                    !failed["document"]["segments"]
                        .as_array()
                        .unwrap()
                        .is_empty()
                );
                assert_eq!(failed["duration"], 336513. / 16000.);
                host.close(root).await;
            }
            _ => {
                let row = &live["document"]["segments"][0];
                let saved=host.client.patch(format!("{}{path}/segments/{}",host.url,row["id"].as_str().unwrap())).header("X-Speakerdesk-Token",&host.token).json(&json!({"segment_revision":row["machine_revision"],"changes":{"text":"Keep this correction through shutdown"}})).send().await.unwrap();
                assert_eq!(saved.status(), 200, "{}", saved.text().await.unwrap());
                host.close(root).await;
                let reopened = Running::open(root, false).await;
                let after = reopened.get(&path).await;
                assert_eq!(after["status"], "failed");
                assert_eq!(
                    after["document"]["segments"][0]["text"],
                    "Keep this correction through shutdown"
                );
                assert!(after["duration"].as_f64().unwrap() >= 7.);
                reopened.close(root).await;
            }
        }
    }
}

#[tokio::test]
async fn completed_live_transcript_checks_saved_voices_without_owning_or_replacing_words() {
    for mode in ["match", "disabled", "cancel", "shutdown"] {
        let temp = tempfile::tempdir().unwrap();
        let root = temp.path();
        fs::write(
            root.join("fixture-voice"),
            b"explicit replay of native voice identity",
        )
        .unwrap();
        let host = Running::open(root, false).await;
        trigger(root, 1);
        trigger(root, 2);
        let first = host.start(json!(["microphone"])).await;
        host.wait(&format!("api/jobs/{first}"), |v| {
            v["duration"].as_f64().unwrap_or(0.) >= 13.5
        })
        .await;
        let saved = host.stop(&first).await;
        assert_eq!(saved["status"], "ready");
        assert!(
            !root.join("voice-calls").exists(),
            "no profile must mean no automatic enrollment/inference"
        );
        let person = host
            .post("api/people", json!({"name":"Ada"}))
            .await
            .json::<Value>()
            .await
            .unwrap();
        let pid = person["person"]["id"]
            .as_str()
            .or_else(|| person["id"].as_str())
            .expect("created person");
        let named = host
            .post(
                &format!("api/jobs/{first}/speakers/speaker_0/identity"),
                json!({"revision":saved["revision"],"person_id":pid}),
            )
            .await;
        assert_eq!(named.status(), 200);
        let named = named.json::<Value>().await.unwrap();
        let clips = host
            .get(&format!("api/jobs/{first}/speakers/speaker_0/voice-clips"))
            .await;
        assert_eq!(clips["status"], "ready", "{mode}: {clips}");
        let enroll=host.post(&format!("api/people/{pid}/voice"),json!({"revision":named["revision"],"meeting_id":first,"track_id":"speaker_0","segment_ids":clips["recommended_ids"],"consent":true})).await;
        assert_eq!(enroll.status(), 200, "{}", enroll.text().await.unwrap());
        let people_before = host.get("api/people").await;
        assert_eq!(
            fs::read_to_string(root.join("voice-calls"))
                .unwrap()
                .lines()
                .count(),
            2
        );
        if mode == "disabled" {
            let response = host
                .client
                .patch(format!("{}api/recognition", host.url))
                .header("X-Speakerdesk-Token", &host.token)
                .json(&json!({"enabled":false}))
                .send()
                .await
                .unwrap();
            assert_eq!(response.status(), 200);
        } else {
            fs::write(
                root.join("hold-live-voice"),
                b"block optional check after ready",
            )
            .unwrap();
        }
        let second = host.start(json!(["microphone"])).await;
        let path = format!("api/jobs/{second}");
        host.wait(&path, |v| v["duration"].as_f64().unwrap_or(0.) >= 13.5)
            .await;
        let stopping = host
            .post(&format!("api/meetings/{second}/stop"), json!({}))
            .await;
        assert_eq!(stopping.status(), 202);
        let ready = host.wait(&path, |v| v["status"] == "ready").await;
        let original_text = ready["document"]["segments"][0]["text"].clone();
        assert!(!original_text.as_str().unwrap().is_empty());
        if mode == "disabled" {
            host.wait(&path, |v| v["inference_owned"] == false).await;
            assert_eq!(
                fs::read_to_string(root.join("voice-calls"))
                    .unwrap()
                    .lines()
                    .count(),
                2
            );
        } else {
            wait_file(&root.join("voice-started")).await;
            let blocked = host.get(&path).await;
            assert_eq!(blocked["inference_owned"], true);
            let passage = &blocked["document"]["segments"][0];
            let edited=host.client.patch(format!("{}{path}/segments/{}",host.url,passage["id"].as_str().unwrap()))
                .header("X-Speakerdesk-Token",&host.token)
                .json(&json!({"segment_revision":passage["machine_revision"],"changes":{"text":"Edited while optional voice matching is running."}}))
                .send().await.unwrap();
            assert_eq!(
                edited.status(),
                200,
                "Ready must remain editable during optional inference"
            );
            let competing = host
                .post(
                    "api/meetings",
                    json!({"name":"must wait","language":"en","sources":["microphone"]}),
                )
                .await;
            assert_eq!(competing.status(), 409);
            if mode == "shutdown" {
                host.close(root).await;
                continue;
            }
            if mode == "cancel" {
                let cancelled = host.post(&format!("{path}/cancel"), json!({})).await;
                assert_eq!(cancelled.status(), 200);
            } else {
                fs::remove_file(root.join("hold-live-voice")).unwrap();
            }
            let done = host.wait(&path, |v| v["inference_owned"] == false).await;
            assert_eq!(done["status"], "ready");
            assert_eq!(
                done["document"]["segments"][0]["text"],
                "Edited while optional voice matching is running."
            );
            assert_eq!(
                done["voice_checks"]["speaker_0"]["status"],
                if mode == "match" {
                    "matched"
                } else {
                    "cancelled"
                }
            );
            if mode == "match" {
                assert_eq!(done["speaker_assignments"]["speaker_0"]["person_id"], pid);
                assert_eq!(done["speaker_assignments"]["speaker_0"]["confirmed"], false);
            } else {
                assert!(done["speaker_assignments"]["speaker_0"].is_null());
            }
        }
        assert_eq!(
            host.get("api/people").await,
            people_before,
            "checking never enrolls or changes the voice catalog"
        );
        host.close(root).await;
    }
}
