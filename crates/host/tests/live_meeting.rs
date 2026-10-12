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
        Self::open_capture(root, dual, None).await
    }
    async fn open_capture(root: &Path, dual: bool, echo: Option<(&Path, &str)>) -> Self {
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
        if dual && echo.is_none() {
            let digest = format!("{:x}", Sha256::digest(fs::read(&helper).unwrap()));
            fs::write(root.join("capture-helper.capability.json"),json!({"helper_sha256":digest,"export_contract":"apple_processed_mono16k_v1","capability_version":"apple_input_node_output_route_v1"}).to_string()).unwrap();
            let format = json!({"sample_rate":48000,"channels":1,"layout":"mono"});
            fs::write(root.join("receipt.json"),json!({"type":"capture_processing","schema_version":1,"job_id":"replaced-by-owned-start","helper_sha256":digest,
                "processor":"apple_voice_processing_io","export_contract":"apple_processed_mono16k_v1","output":"processed_microphone","stage":"retained_source_pcm","postprocessing":"identity",
                "stream_epoch":0,"format":{"sample_rate":16000,"channels":1,"layout":"mono"},"capability_version":"apple_input_node_output_route_v1",
                "route":{"node":"inputNode","bus":0,"tap_scope":"output"},"state_phase":"after_start","channel_mapping":"mono_direct","callback_format":format,"io_formats":{"input_output":format,"output_input":format},
                "readbacks":{"input_enabled":true,"output_enabled":true,"engine_running":true,"bypassed":false,"manual_rendering":false,"input_muted":false,"agc_enabled":false}}).to_string()).unwrap();
        }
        let mut command = Command::new(env!("CARGO_BIN_EXE_inference-fixture"));
        command
            .arg("--native-host")
            .arg(root.join("data"))
            .arg(root)
            .env("SPEAKERDESK_CAPTURE_HELPER", &helper)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::inherit())
            .kill_on_drop(true)
            .env_remove("SPEAKERDESK_TEST_ECHO_FAILURE");
        if let Some((library, mode)) = echo {
            command.env("SPEAKERDESK_ECHO_LIBRARY", library);
            if mode != "real" {
                command.env("SPEAKERDESK_TEST_ECHO_FAILURE", mode);
            }
        }
        let mut child = command.spawn().unwrap();
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
        let mut last = Value::Null;
        tokio::time::timeout(Duration::from_secs(15), async {
            loop {
                let value = self.get(path).await;
                if predicate(&value) {
                    return value;
                }
                last = value;
                tokio::time::sleep(Duration::from_millis(15)).await;
            }
        })
        .await
        .unwrap_or_else(|_| {
            let rows = last["document"]["segments"].as_array().into_iter().flatten().take(8)
                .map(|row| json!({
                    "id":row["id"],"canonical_state":row["canonical_state"],
                    "start_sample":row["start_sample"],"end_sample":row["end_sample"],
                    "text_end_sample":row["canonical_text_end_sample"],
                    "text_bytes":row["text"].as_str().map(str::len),
                    "language_epoch":row["language_epoch"],"machine_revision":row["machine_revision"],
                    "review":row["review"],"source":row["capture_source"],
                })).collect::<Vec<_>>();
            panic!("timed out at {path}; last state: {}", json!({
                "id":last["id"],"status":last["status"],"duration":last["duration"],
                "configured":last["readiness"]["configured"],"inference_owned":last["inference_owned"],
                "language_acknowledged_revision":last["language_acknowledged_revision"],
                "pause_flush":last["pause_flush"],"live_recovery":last["live_recovery"],
                "message":last["message"].as_str().map(|message|message.chars().take(200).collect::<String>()),
                "rows":rows,
            }));
        })
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
    async fn name_live_speaker(&self, path: &str, track: &str, name: &str) {
        // Live publication may advance the revision between GET and naming.
        tokio::time::timeout(Duration::from_secs(5), async {
            loop {
                let current = self.get(path).await;
                let response = self
                    .post(
                        &format!("{path}/speakers/{track}/identity"),
                        json!({"revision":current["revision"],"name":name,"person_id":null}),
                    )
                    .await;
                let status = response.status();
                if status == 200 {
                    break;
                }
                let body: Value = response.json().await.unwrap();
                assert_eq!(status, 409, "{body}");
                assert_eq!(body["error"], speakerdesk_core::messages::NAME_STALE);
                tokio::time::sleep(Duration::from_millis(10)).await;
            }
        })
        .await
        .expect("live naming did not reach a current revision");
    }
    async fn start_live_only(&self, sources: Value) -> String {
        let id = self.start(sources).await;
        let response = self
            .post(&format!("api/jobs/{id}/refinement/pause"), json!({}))
            .await;
        assert_eq!(response.status(), 200, "{}", response.text().await.unwrap());
        id
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
            .start_live_only(if dual {
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
        host.name_live_speaker(&path, &old_speaker, "Casey").await;
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
        // Finish edit feedback before killing the child. The already sealed
        // protected prefix must stay unchanged; a different open French tail
        // is what recovery interrupts and marks for review.
        trigger(root, 2);
        // Two seven-second stages, less the writer's half-second watermark,
        // rounded down to a complete Silero frame: no later PCM is committed.
        let pre_loss_observed = (14 * 16_000 - 8_000) / 512 * 512;
        let pre_loss = host
            .wait(&path, |job| {
                let Some(rows) = job["document"]["segments"].as_array() else {
                    return false;
                };
                job["duration"].as_f64().unwrap_or(0.) >= 13.5
                    && rows
                        .iter()
                        .any(|row| row["id"] == old_id && row["canonical_state"] == "sealed")
                    && rows
                        .iter()
                        .filter(|row| {
                            row["language_epoch"] == 1
                                && row["canonical_state"] == "open"
                                && row["text"] == "Replay transcript"
                                && row["end_sample"].as_i64() == Some(pre_loss_observed)
                                && row["canonical_text_end_sample"]
                                    .as_i64()
                                    .zip(row["start_sample"].as_i64())
                                    .is_some_and(|(text_end, start)| text_end > start)
                        })
                        .count()
                        == if dual { 2 } else { 1 }
            })
            .await;
        let pre_loss_rows = pre_loss["document"]["segments"].as_array().unwrap();
        let protected_before = pre_loss_rows
            .iter()
            .find(|row| row["id"] == old_id)
            .unwrap()
            .clone();
        let interrupted = pre_loss_rows
            .iter()
            .filter(|row| {
                row["canonical_state"] == "open" && !row["text"].as_str().unwrap_or("").is_empty()
            })
            .cloned()
            .collect::<Vec<_>>();
        assert!(!interrupted.is_empty());
        let history = pre_loss["language_history"].clone();
        let old_pid: i32 = fs::read_to_string(root.join("fixture-child-pid"))
            .unwrap()
            .parse()
            .unwrap();
        fs::write(root.join("hold-live-feed"), b"block recovered worker only").unwrap();
        assert_eq!(unsafe { libc::kill(old_pid, libc::SIGKILL) }, 0);
        trigger(root, 3);
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
        let replay_rows = before["segments"].as_array().unwrap();
        assert_eq!(
            replay_rows.iter().find(|row| row["id"] == old_id).unwrap(),
            &protected_before,
            "{mode}: recovery must preserve an already sealed protected prefix"
        );
        for prior in &interrupted {
            let row = replay_rows
                .iter()
                .find(|row| row["id"] == prior["id"])
                .unwrap();
            assert_eq!(row["canonical_state"], "sealed", "{mode}: {row}");
            assert_eq!(row["review"], true, "{mode}: {row}");
            assert_eq!(row["canonical_unresolved"], "inference_restart_boundary");
            for key in [
                "text",
                "start",
                "end",
                "speaker",
                "machine_revision",
                "capture_source",
            ] {
                assert_eq!(
                    row[key], prior[key],
                    "{mode}: interrupted passage field {key}"
                );
            }
        }
        trigger(root, 4);
        host.wait("api/meeting", |v| {
            v["duration"].as_f64().unwrap_or(0.) >= 27.5
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
                trigger(root, 5);
                host.wait("api/meeting", |v| {
                    v["duration"].as_f64().unwrap_or(0.) >= 34.5
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
                    assert_eq!(paused["pause_flush"]["received_sample"], 448000);
                    assert_eq!(
                        host.post(&format!("api/meetings/{id}/resume"), json!({}))
                            .await
                            .status(),
                        202
                    );
                }
                let final_job = host.stop(&id).await;
                assert_eq!(final_job["status"], "ready", "{final_job}");
                assert_eq!(final_job["canonical_observed_sample"], 448513);
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
                assert_eq!(
                    old, &protected_before,
                    "{mode}: later recovery publication must preserve the sealed edited prefix"
                );
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
        let id = host.start_live_only(json!(["microphone"])).await;
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
        let id = host.start_live_only(json!(["microphone"])).await;
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
        host.name_live_speaker(&path, "speaker_0", "Casey").await;
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
    let id = host.start_live_only(json!(["microphone"])).await;
    let path = format!("api/jobs/{id}");
    let first = host
        .wait(&path, |v| {
            v["document"]["segments"]
                .as_array()
                .is_some_and(|s| s.iter().any(|s| s["text"] == "Replay transcript"))
        })
        .await;
    assert!(first.get("operation_generation").is_none());
    assert_eq!(first["rolling_refinement"]["mode"], "live");
    assert_eq!(first["refinement_status"], "paused");
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
    host.name_live_speaker(&path, "speaker_0", "Casey").await;
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
#[cfg(target_os = "macos")]
#[ignore = "Requires pinned echo component and test ABI fixture in SPEAKERDESK_ECHO_FAULT_LIBRARY"]
async fn raw_dual_capture_routes_sources_and_retains_failure_endpoints_on_reopen() {
    let native = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../desktop/capture/echo/libspeakerdesk_echo.dylib");
    let fault = std::path::PathBuf::from(
        std::env::var_os("SPEAKERDESK_ECHO_FAULT_LIBRARY").expect("test-only ABI fixture"),
    );
    for case in ["real", "process", "reset", "abi", "missing"] {
        let temp = tempfile::tempdir().unwrap();
        let root = temp.path();
        let missing = root.join("missing-echo.dylib");
        let library = match case {
            "real" => &native,
            "missing" => &missing,
            _ => &fault,
        };
        let host = Running::open_capture(root, true, Some((library, case))).await;
        let response = host
            .post(
                "api/meetings",
                json!({"name":"Raw DSP replay","language":"en","sources":["microphone","system"]}),
            )
            .await;
        assert_eq!(
            response.status(),
            202,
            "{case}: {}",
            response.text().await.unwrap()
        );
        let started: Value = response.json().await.unwrap();
        let id = started["id"].as_str().unwrap();
        let path = format!("api/jobs/{id}");
        let finished = if matches!(case, "abi" | "missing") {
            host.wait(&path, |v| {
                v["status"] == "failed" && v["inference_owned"] == false
            })
            .await
        } else {
            trigger(root, 1);
            if case == "process" {
                host.wait(&path, |v| {
                    v["status"] == "failed" && v["inference_owned"] == false
                })
                .await
            } else {
                // Includes the native lookahead; pause must flush the full raw endpoint.
                host.wait(&path, |v| v["duration"].as_f64().unwrap_or(0.) >= 5.9)
                    .await;
                assert_eq!(
                    host.post(&format!("api/meetings/{id}/pause"), json!({}))
                        .await
                        .status(),
                    202
                );
                if case == "reset" {
                    host.wait(&path, |v| {
                        v["status"] == "failed" && v["inference_owned"] == false
                    })
                    .await
                } else {
                    host.wait(&path, |v| {
                        v["status"] == "paused" && v["pause_flush"]["status"] == "complete"
                    })
                    .await;
                    assert_eq!(
                        host.post(&format!("api/meetings/{id}/resume"), json!({}))
                            .await
                            .status(),
                        202
                    );
                    host.wait(&path, |v| v["status"] == "recording").await;
                    trigger(root, 2);
                    host.wait(&path, |v| v["duration"].as_f64().unwrap_or(0.) >= 12.9)
                        .await;
                    host.stop(id).await
                }
            }
        };
        let folder = root.join("data").join(id);
        assert!(
            finished["capture_processing"].is_null(),
            "raw AEC is not Apple processing"
        );
        assert!(!folder.join("capture-processing.json").exists());
        if matches!(case, "abi" | "missing") {
            assert!(
                !folder.join("audio.wav").exists(),
                "runtime rejection precedes PCM creation"
            );
            let expected = if case == "abi" {
                speakerdesk_core::messages::ECHO_INCOMPATIBLE
            } else {
                speakerdesk_core::messages::ECHO_MISSING
            };
            assert!(
                finished["message"].as_str().unwrap().contains(expected),
                "{finished}"
            );
        } else {
            let coverage = &finished["capture_audio_coverage"];
            let raw = coverage["raw_samples"].as_u64().unwrap();
            let clean = coverage["clean_samples"].as_u64().unwrap();
            assert_eq!(
                finished["duration"].as_f64().unwrap(),
                clean as f64 / 16000.
            );
            assert_eq!(
                finished["capture_source_catalog"]["policy"],
                "aligned_aec_microphone_and_original_system_v1"
            );
            for track in [
                "microphone",
                "system",
                "system_reference",
                "microphone_clean",
                "audio",
            ] {
                let wav = hound::WavReader::open(folder.join(format!("{track}.wav"))).unwrap();
                let frames = if matches!(track, "audio" | "microphone_clean") {
                    clean
                } else {
                    raw
                };
                assert_eq!(u64::from(wav.duration()), frames, "{case}/{track}");
            }
            // Original tracks remain exact even if native processing or reset consumes
            // input and then fails. Neither is relabelled as cleaned playback.
            for track in ["microphone", "system"] {
                for (i, sample) in hound::WavReader::open(folder.join(format!("{track}.wav")))
                    .unwrap()
                    .into_samples::<i16>()
                    .enumerate()
                {
                    let value = 0.2_f32 * (i as f32 * 0.071).sin();
                    let expected = (f64::from(value) * 32768.)
                        .round_ties_even()
                        .clamp(-32768., 32767.) as i16;
                    assert_eq!(sample.unwrap(), expected, "{case}/{track}/{i}");
                }
            }
            if case == "real" {
                assert_eq!(finished["status"], "ready");
                assert_eq!(
                    raw, 224513,
                    "Pause/Resume and final partial frame retain each sample once"
                );
                assert_eq!(clean, raw);
                assert_eq!(
                    fs::read(folder.join("microphone.wav")).unwrap(),
                    fs::read(folder.join("microphone_clean.wav")).unwrap(),
                    "tonal no-echo fallback preserves microphone"
                );
                let owners: std::collections::BTreeSet<_> = finished["document"]["segments"]
                    .as_array()
                    .unwrap()
                    .iter()
                    .map(|row| row["capture_source"]["source_id"].as_str().unwrap())
                    .collect();
                assert_eq!(
                    owners,
                    std::collections::BTreeSet::from(["microphone_clean", "system"])
                );
            } else {
                assert!(raw > clean && clean > 0, "{case}: {coverage}");
                let expected = if case == "process" {
                    "rejected an audio frame"
                } else {
                    "could not reset"
                };
                assert!(
                    finished["message"].as_str().unwrap().contains(expected),
                    "{finished}"
                );
            }
            let playback = host
                .client
                .get(format!("{}api/jobs/{id}/audio", host.url))
                .send()
                .await
                .unwrap()
                .error_for_status()
                .unwrap()
                .bytes()
                .await
                .unwrap();
            assert_eq!(
                playback.as_ref(),
                fs::read(folder.join("audio.wav")).unwrap()
            );
        }
        host.close(root).await;
        let reopened = Running::open_capture(root, true, Some((library, case))).await;
        let saved = reopened.get(&path).await;
        assert_eq!(saved["status"], finished["status"], "{case}");
        assert_eq!(saved["duration"], finished["duration"]);
        assert_eq!(
            saved["capture_audio_coverage"],
            finished["capture_audio_coverage"]
        );
        assert_eq!(
            saved["capture_source_catalog"],
            finished["capture_source_catalog"]
        );
        assert_eq!(saved["document"], finished["document"]);
        reopened.close(root).await;
    }
}

#[tokio::test]
async fn qualified_live_resume_keeps_sources_and_owns_pending_shutdown() {
    for case in ["twice", "device-failure", "stop-pending", "late-recording"] {
        let temp = tempfile::tempdir().unwrap();
        let root = temp.path();
        let host = Running::open(root, true).await;
        trigger(root, 1);
        let id = host.start_live_only(json!(["microphone", "system"])).await;
        let path = format!("api/jobs/{id}");
        host.wait(&path, |v| v["duration"].as_f64().unwrap_or(0.) >= 6.5)
            .await;
        assert_eq!(
            host.post(&format!("api/meetings/{id}/pause"), json!({}))
                .await
                .status(),
            202
        );
        let paused = host
            .wait(&path, |v| {
                v["status"] == "paused" && v["pause_flush"]["status"] == "complete"
            })
            .await;
        let folder = root.join("data").join(&id);
        let receipt = fs::read(folder.join("capture-processing.json")).unwrap();
        let catalog = paused["capture_source_catalog"].clone();
        fs::write(root.join("hold-resume"), b"replay barrier").unwrap();
        if case == "device-failure" {
            fs::write(root.join("fail-resume"), b"replay").unwrap();
        }
        if case == "late-recording" {
            fs::write(root.join("late-recording"), b"replay").unwrap();
        }
        assert_eq!(
            host.post(&format!("api/meetings/{id}/resume"), json!({}))
                .await
                .status(),
            202
        );
        wait_file(&root.join("resume-started")).await;
        assert_eq!(
            host.post(&format!("api/meetings/{id}/resume"), json!({}))
                .await
                .status(),
            409,
            "{case}"
        );
        let pending = host.get(&path).await;
        assert_eq!(pending["message"], "Resuming recording…");
        assert_eq!(pending["inference_owned"], true);
        let finished = if matches!(case, "stop-pending" | "late-recording") {
            assert_eq!(
                host.post(&format!("api/meetings/{id}/stop"), json!({}))
                    .await
                    .status(),
                202
            );
            if case == "late-recording" {
                wait_file(&root.join("late-recording-sent")).await;
                let draining = host
                    .wait(&path, |v| v["duration"].as_f64().unwrap_or(0.) >= 7.5)
                    .await;
                assert_eq!(draining["status"], "finishing");
                assert_eq!(draining["inference_owned"], true);
                fs::write(root.join("release-stop"), b"replay").unwrap();
            }
            host.wait(&path, |v| {
                v["status"] == "ready" && v["inference_owned"] == false
            })
            .await
        } else {
            fs::write(root.join("release-resume"), b"replay").unwrap();
            if case == "device-failure" {
                let failed = host
                    .wait(&path, |v| {
                        v["status"] == "failed" && v["inference_owned"] == false
                    })
                    .await;
                assert!(
                    failed["message"]
                        .as_str()
                        .unwrap()
                        .contains("device changed"),
                    "{failed}"
                );
                failed
            } else {
                host.wait(&path, |v| v["status"] == "recording").await;
                for stage in 2..=3 {
                    if stage == 3 {
                        assert_eq!(
                            host.post(&format!("api/meetings/{id}/resume"), json!({}))
                                .await
                                .status(),
                            202
                        );
                        host.wait(&path, |v| v["status"] == "recording").await;
                    }
                    trigger(root, stage);
                    host.wait(&path, |v| {
                        v["duration"].as_f64().unwrap_or(0.) >= f64::from(stage * 7) - 0.5
                    })
                    .await;
                    assert_eq!(
                        host.post(&format!("api/meetings/{id}/pause"), json!({}))
                            .await
                            .status(),
                        202
                    );
                    host.wait(&path, |v| {
                        v["status"] == "paused" && v["pause_flush"]["status"] == "complete"
                    })
                    .await;
                }
                host.stop(&id).await
            }
        };
        let samples = match case {
            "twice" => 336000,
            "late-recording" => 128000,
            _ => 112000,
        };
        assert_eq!(finished["duration"], f64::from(samples) / 16000., "{case}");
        assert_eq!(finished["capture_source_catalog"], catalog);
        assert_eq!(
            fs::read(folder.join("capture-processing.json")).unwrap(),
            receipt
        );
        let microphone = hound::WavReader::open(folder.join("microphone.wav"))
            .unwrap()
            .into_samples::<i16>()
            .collect::<Result<Vec<_>, _>>()
            .unwrap();
        let clean = hound::WavReader::open(folder.join("microphone_clean.wav"))
            .unwrap()
            .into_samples::<i16>()
            .collect::<Result<Vec<_>, _>>()
            .unwrap();
        assert_eq!(microphone, clean);
        assert_eq!(clean.len(), samples as usize);
        for (index, &sample) in clean.iter().enumerate() {
            let value = 0.2_f32 * (index as f32 * 0.071).sin();
            let expected = (f64::from(value) * 32768.)
                .round_ties_even()
                .clamp(-32768., 32767.) as i16;
            assert_eq!(sample, expected, "{case} sample {index}");
        }
        let sources: std::collections::BTreeSet<_> = finished["document"]["segments"]
            .as_array()
            .unwrap()
            .iter()
            .map(|row| row["capture_source"]["source_id"].as_str().unwrap())
            .collect();
        assert_eq!(
            sources,
            std::collections::BTreeSet::from(["microphone_clean", "system"])
        );
        host.close(root).await;
    }
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
            .start_live_only(if dual {
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
                host.wait(&path, |v| {
                    v["status"] == "paused" && v["pause_flush"]["status"] == "complete"
                })
                .await;
                let saved = host.stop(&id).await;
                assert_eq!(saved["status"], "ready");
                assert_eq!(saved["duration"], 7.);
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
        let first = host.start_live_only(json!(["microphone"])).await;
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
        let second = host.start_live_only(json!(["microphone"])).await;
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

#[tokio::test]
async fn ordinary_live_language_evidence_survives_routing_edits_and_restart() {
    for case in [
        "short",
        "long",
        "manual",
        "mixed",
        "uncertain",
        "edit_epoch",
        "recovery",
    ] {
        let temp = tempfile::tempdir().unwrap();
        let root = temp.path();
        if case == "uncertain" {
            fs::write(
                root.join("replay-probe.json"),
                json!({"probabilities":[0.55,0.45],"codes":["en","de"]}).to_string(),
            )
            .unwrap();
        }
        let host = Running::open(root, false).await;
        trigger(root, 1);
        let response=host.post("api/meetings",json!({"name":"Ordinary live language evidence replay","language":if case=="manual"{"fr"}else{"auto"},"sources":["microphone"]})).await;
        assert_eq!(response.status(), 202, "{}", response.text().await.unwrap());
        let id = host.wait("api/meeting", |v| v["id"].is_string()).await["id"]
            .as_str()
            .unwrap()
            .to_owned();
        let path = format!("api/jobs/{id}");
        assert_eq!(
            host.post(&format!("{path}/refinement/pause"), json!({}))
                .await
                .status(),
            200
        );
        let stages = if matches!(case, "long" | "mixed") {
            4
        } else {
            1
        };
        for stage in 1..=stages {
            if stage > 1 {
                trigger(root, stage);
            }
            host.wait(&path, |v| {
                v["duration"].as_f64().unwrap_or(0.) >= f64::from(stage * 7) - 0.5
            })
            .await;
        }
        assert_eq!(
            host.post(&format!("api/meetings/{id}/pause"), json!({}))
                .await
                .status(),
            202
        );
        let paused = host
            .wait(&path, |v| {
                v["status"] == "paused" && v["pause_flush"]["status"] == "complete"
            })
            .await;
        let prefix = paused["document"]["segments"][0].clone();
        if matches!(case, "edit_epoch" | "recovery") {
            let edited=host.client.patch(format!("{}{path}/segments/{}",host.url,prefix["id"].as_str().unwrap())).header("X-Speakerdesk-Token",&host.token)
                .json(&json!({"segment_revision":prefix["machine_revision"],"changes":{"text":"Preserved language prefix","start":0.1}})).send().await.unwrap();
            assert_eq!(edited.status(), 200, "{}", edited.text().await.unwrap());
            if case == "recovery" {
                let pid: i32 = fs::read_to_string(root.join("fixture-child-pid"))
                    .unwrap()
                    .parse()
                    .unwrap();
                assert_eq!(unsafe { libc::kill(pid, libc::SIGKILL) }, 0);
            }
            assert_eq!(
                host.post(&format!("api/meetings/{id}/resume"), json!({}))
                    .await
                    .status(),
                202
            );
            trigger(root, 2);
            host.wait(&path, |v| v["duration"].as_f64().unwrap_or(0.) >= 13.5)
                .await;
            if case == "edit_epoch" {
                let changed = host
                    .client
                    .patch(format!("{}api/meetings/{id}/language", host.url))
                    .header("X-Speakerdesk-Token", &host.token)
                    .json(&json!({"language":"fr","language_revision":0}))
                    .send()
                    .await
                    .unwrap();
                assert_eq!(changed.status(), 200, "{}", changed.text().await.unwrap());
                host.wait("api/meeting", |v| v["language_acknowledged_revision"] == 1)
                    .await;
                trigger(root, 3);
                host.wait(&path, |v| v["duration"].as_f64().unwrap_or(0.) >= 20.5)
                    .await;
            }
        }
        if case == "mixed" {
            // The acknowledged pause has already published the long English
            // prefix. Resumed audio decodes its remaining tail in German, regardless
            // of how many probes or balanced cuts were needed before the pause.
            assert_eq!(prefix["language"], "en");
            assert!(prefix["canonical_text_end_sample"].as_i64().unwrap() > 392000);
            assert!(
                prefix["language_detection"]["probes"]
                    .as_array()
                    .unwrap()
                    .iter()
                    .all(|probe| probe["language"] == "en")
            );
            fs::write(
                root.join("replay-probe.json"),
                json!({"probabilities":[0.01,0.99],"codes":["en","de"]}).to_string(),
            )
            .unwrap();
            assert_eq!(
                host.post(&format!("api/meetings/{id}/resume"), json!({}))
                    .await
                    .status(),
                202
            );
            trigger(root, 5);
            host.wait(&path, |v| v["duration"].as_f64().unwrap_or(0.) >= 34.5)
                .await;
        }
        let done = host.stop(&id).await;
        assert_eq!(done["status"], "ready", "{case}: {done}");
        let rows = done["document"]["segments"].as_array().unwrap();
        assert!(!rows.is_empty());
        for row in rows {
            let detection = &row["language_detection"];
            let manual = row["language_mode"] != "auto";
            assert_eq!(
                detection["mode"],
                if manual { "manual" } else { "auto" },
                "{case}: {row}"
            );
            assert_ne!(
                row["language"], "auto",
                "captured mode is never a detected language"
            );
            if manual {
                assert_eq!(row["language"], row["language_mode"]);
                assert_eq!(detection["reason"], "override");
            } else {
                let probes = detection["probes"]
                    .as_array()
                    .expect("retained actual routing probes");
                assert!(!probes.is_empty());
                for probe in probes {
                    let a = probe["start_sample"].as_i64().unwrap();
                    let b = probe["end_sample"].as_i64().unwrap();
                    // Protected display bounds may differ from the original acoustic scope.
                    let floor = if row["text"] == "Preserved language prefix" {
                        0
                    } else {
                        (row["start"].as_f64().unwrap() * 16000.).round() as i64
                    };
                    let end = (row["end"].as_f64().unwrap() * 16000.).round() as i64;
                    assert!(
                        floor <= a && a < b && b <= end,
                        "{case}: {probe} outside {floor}..{end}"
                    );
                    if let Some(context) = probe["decision"]["context_end_sample"].as_i64() {
                        assert!(0 <= context && context <= a);
                    }
                }
            }
        }
        if case == "mixed" {
            let row = &rows[0];
            assert!(row["language"].is_null(), "{row}");
            assert_eq!(row["review"], true);
            assert_ne!(row["language_detection"]["reason"], "detected");
            let probes = row["language_detection"]["probes"].as_array().unwrap();
            assert!(
                probes.iter().any(|p| p["language"] == "en")
                    && probes.iter().any(|p| p["language"] == "de")
            );
            let retained = probes
                .iter()
                .take_while(|p| p["language"] == "en")
                .cloned()
                .collect::<Vec<_>>();
            let original = prefix["language_detection"]["probes"].as_array().unwrap();
            assert!(!retained.is_empty() && retained.len() < original.len());
            assert_eq!(
                retained.as_slice(),
                &original[..retained.len()],
                "accepted prefix keeps the actual previous language decisions"
            );
            let cutoff = retained.last().unwrap()["end_sample"].as_i64().unwrap();
            assert!(
                probes[retained.len()..]
                    .iter()
                    .all(|probe| probe["language"] == "de"
                        && probe["start_sample"].as_i64().unwrap() >= cutoff)
            );
        } else if case == "uncertain" {
            assert_eq!(rows[0]["review"], true);
            assert_eq!(rows[0]["language_detection"]["reason"], "best_effort");
            assert_eq!(
                rows[0]["language_detection"]["probability"],
                f64::from(0.55_f32)
            );
        } else if case == "long" {
            assert_eq!(rows[0]["language"], "en");
            assert!(
                rows[0]["language_detection"]["probes"]
                    .as_array()
                    .unwrap()
                    .iter()
                    .any(|p| p["start_sample"].as_i64().unwrap() >= 14 * 16000)
            );
        }
        if matches!(case, "edit_epoch" | "recovery") {
            let retained = rows.iter().find(|r| r["id"] == prefix["id"]).unwrap();
            assert_eq!(retained["text"], "Preserved language prefix");
            assert_eq!(retained["language_detection"], prefix["language_detection"]);
            assert!(rows.iter().any(|r| r["start"].as_f64().unwrap() >= 6.));
            if case == "edit_epoch" {
                assert!(rows.iter().any(|r| r["language_epoch"] == 1
                    && r["language_mode"] == "fr"
                    && r["language"] == "fr"));
            }
        }
        let calls = fs::read_to_string(root.join("language-probe-calls")).unwrap_or_default();
        // Read/poll/export must not probe merely to serialize decisions.
        assert_eq!(host.get(&path).await["document"], done["document"]);
        host.close(root).await;
        let reopened = Running::open(root, false).await;
        assert_eq!(reopened.get(&path).await["document"], done["document"]);
        assert_eq!(
            fs::read_to_string(root.join("language-probe-calls")).unwrap_or_default(),
            calls
        );
        reopened.close(root).await;
    }
}

#[tokio::test]
async fn saved_long_accuracy_reaches_bounded_context_models_and_retains_missing_timing() {
    // Native helper/model protocol replay. Missing CTC must retain prior words,
    // but the long route must actually attempt bounded decoding rather than reject duration.
    let temp = tempfile::tempdir().unwrap();
    let root = temp.path();
    let host = Running::open(root, false).await;
    trigger(root, 1);
    let id = host.start_live_only(json!(["microphone"])).await;
    let path = format!("api/jobs/{id}");
    for stage in 1..=4 {
        if stage > 1 {
            trigger(root, stage);
        }
        host.wait(&path, |v| {
            v["duration"].as_f64().unwrap_or(0.) >= f64::from(stage * 7) - 0.5
        })
        .await;
    }
    let before = host.stop(&id).await;
    assert_eq!(before["status"], "ready");
    assert!(
        before["document"]["segments"]
            .as_array()
            .unwrap()
            .iter()
            .any(|row| row["end"].as_f64().unwrap() - row["start"].as_f64().unwrap() > 24.5)
    );
    let calls = fs::read_to_string(root.join("transcribe-languages"))
        .unwrap_or_default()
        .lines()
        .count();
    let response = host
        .post(&format!("{path}/refinement/resume"), json!({}))
        .await;
    assert_eq!(response.status(), 202, "{}", response.text().await.unwrap());
    let after = host.wait(&path, |v| v["inference_owned"] == false).await;
    assert!(
        fs::read_to_string(root.join("transcribe-languages"))
            .unwrap_or_default()
            .lines()
            .count()
            > calls,
        "long saved accuracy work must reach bounded model crops"
    );
    assert_eq!(after["refinement_status"], "unresolved");
    assert_eq!(
        after["document"], before["document"],
        "unqualified boundaries keep prior words and timing"
    );
    host.close(root).await;
}

#[tokio::test]
async fn saved_long_context_publication_timing_and_late_terminal_matrix() {
    for case in [
        "german",
        "chinese",
        "mixed",
        "uncertain",
        "complete",
        "edit",
        "pause",
    ] {
        let temp = tempfile::tempdir().unwrap();
        let root = temp.path();
        fs::write(
            root.join("fixture-ctc"),
            b"synthetic overlap ownership scores",
        )
        .unwrap();
        fs::create_dir(root.join("coarse-alignment")).unwrap();
        fs::write(
            root.join("coarse-alignment/tokens.txt"),
            include_str!("testdata/ctc-tokens.txt"),
        )
        .unwrap();
        let host = Running::open(root, false).await;
        trigger(root, 1);
        let captured_mode = if case == "chinese" { "zh" } else { "auto" };
        let response=host.post("api/meetings",json!({"name":"Long context ownership replay","language":captured_mode,"sources":["microphone"]})).await;
        assert_eq!(response.status(), 202, "{}", response.text().await.unwrap());
        let id = host.wait("api/meeting", |v| v["id"].is_string()).await["id"]
            .as_str()
            .unwrap()
            .to_owned();
        let path = format!("api/jobs/{id}");
        assert_eq!(
            host.post(&format!("{path}/refinement/pause"), json!({}))
                .await
                .status(),
            200
        );
        for stage in 1..=4 {
            if stage > 1 {
                trigger(root, stage);
            }
            host.wait(&path, |v| {
                v["duration"].as_f64().unwrap_or(0.) >= f64::from(stage * 7) - 0.5
            })
            .await;
        }
        let before = host.stop(&id).await;
        let old = &before["document"]["segments"][0];
        assert_eq!(old["language_mode"], captured_mode);
        assert_eq!(old["end"], 28. + 513. / 16000.);
        // Authored raw contexts and absolute CTC event positions. The duplicate
        // bridge is the same event; the two repeats are distinct events.
        fs::write(root.join("context-replay.json"),json!([
            {"text":"start repeat bridge extra","start_sample":0,"end_sample":272000,
             "words":[["start",16000],["repeat",192000],["bridge",208000],["extra",256000]]},
            {"text":"repeat bridge finish repeat","start_sample":176000,"end_sample":448513,
             "words":[["repeat",192000],["bridge",208000],["finish",352000],["repeat",416000]]},
            {"text":"repeat","start_sample":400000,"end_sample":448513,"words":[["repeat",416000]]}
        ]).to_string()).unwrap();
        let german = matches!(case, "german" | "mixed" | "uncertain" | "edit" | "pause");
        if german {
            fs::write(
                root.join("replay-probe.json"),
                json!({"probabilities":[0.99,0.01],"codes":["de","en"]}).to_string(),
            )
            .unwrap();
            fs::write(root.join("context-replay.json"),json!([
                {"text":"anfang wieder brücke extra","start_sample":0,"end_sample":272000,
                 "words":[["anfang",16000],["wieder",192000],["brücke",208000],["extra",256000]]},
                {"text":"wieder brücke ende wieder","start_sample":176000,"end_sample":448513,
                 "words":[["wieder",192000],["brücke",208000],["ende",352000],["wieder",416000]]},
                {"text":"wieder","start_sample":400000,"end_sample":448513,"words":[["wieder",416000]]}
            ]).to_string()).unwrap();
        }
        if case == "chinese" {
            fs::write(root.join("context-replay.json"),json!([
                {"text":"开始重复桥梁额外","separator":"","start_sample":0,"end_sample":272000,
                 "words":[["开始",16000],["重复",192000],["桥梁",208000],["额外",256000]]},
                {"text":"重复桥梁结束重复","separator":"","start_sample":176000,"end_sample":448513,
                 "words":[["重复",192000],["桥梁",208000],["结束",352000],["重复",416000]]},
                {"text":"重复","separator":"","start_sample":400000,"end_sample":448513,"words":[["重复",416000]]}
            ]).to_string()).unwrap();
        }
        if matches!(case, "mixed" | "uncertain") {
            let file = root.join("context-replay.json");
            let mut contexts: Value = serde_json::from_slice(&fs::read(&file).unwrap()).unwrap();
            contexts[1]["probe"] = if case == "mixed" {
                json!({"probabilities":[0.99,0.01],"codes":["fr","de"]})
            } else {
                json!({"probabilities":[0.55,0.45],"codes":["de","fr"]})
            };
            fs::write(file, contexts.to_string()).unwrap();
        }
        let completes = matches!(case, "complete" | "german" | "chinese");
        let held = matches!(case, "edit" | "pause");
        if held {
            fs::write(root.join("hold-live-asr"), b"hold final context").unwrap();
            fs::write(root.join("hold-context-index"), b"2").unwrap();
        }
        let begin = host
            .post(&format!("{path}/refinement/resume"), json!({}))
            .await;
        assert_eq!(begin.status(), 202, "{}", begin.text().await.unwrap());
        if held {
            wait_file(&root.join("edit-asr-started")).await;
            assert_eq!(
                host.get(&path).await["document"],
                before["document"],
                "no partial anchor publications"
            );
            if case == "edit" {
                let edited=host.client.patch(format!("{}{path}/segments/{}",host.url,old["id"].as_str().unwrap()))
                    .header("X-Speakerdesk-Token",&host.token).json(&json!({"segment_revision":old["machine_revision"],"changes":{"text":"Kept while the final context was running","start":0.1}})).send().await.unwrap();
                assert_eq!(edited.status(), 200, "{}", edited.text().await.unwrap());
            } else {
                assert_eq!(
                    host.post(&format!("{path}/refinement/pause"), json!({}))
                        .await
                        .status(),
                    200
                );
            }
            fs::remove_file(root.join("hold-live-asr")).unwrap();
        }
        let after = host.wait(&path, |v| v["inference_owned"] == false).await;
        let row = &after["document"]["segments"][0];
        if completes {
            assert_eq!(after["refinement_status"], "complete", "{after}");
            let expected = if case == "chinese" {
                "开始重复桥梁结束重复"
            } else if german {
                "anfang wieder brücke ende wieder"
            } else {
                "start repeat bridge finish repeat"
            };
            assert_eq!(row["text"], expected);
            let language = if case == "chinese" {
                "zh"
            } else if german {
                "de"
            } else {
                "en"
            };
            assert_eq!(row["language"], language);
            assert_eq!(row["language_mode"], captured_mode);
            assert_eq!(row["reading_alignment"]["language"], language);
            assert_eq!(
                row["reading_alignment"]["score_calibration_id"],
                if language != "en" {
                    speakerdesk_core::timing_policy::FUNCTIONAL_POLICY_ID
                } else {
                    speakerdesk_core::timing::CALIBRATION
                }
            );
            if language != "en" {
                assert_eq!(
                    row["reading_alignment"]["qualified_scope"],
                    "multilingual_functional_ctc_units"
                );
                assert_eq!(
                    row["reading_alignment"]["timing_accuracy_calibrated_languages"],
                    json!([])
                );
                assert_eq!(
                    row["reading_turn_provenance"]["calibration_id"],
                    speakerdesk_core::timing_policy::FUNCTIONAL_POLICY_ID
                );
            }
            assert_eq!(row["id"], old["id"]);
            assert_eq!(row["speaker"], old["speaker"]);
            assert_eq!(row["language_epoch"], old["language_epoch"]);
            assert_eq!(
                row["machine_revision"].as_i64(),
                old["machine_revision"].as_i64().map(|v| v + 1)
            );
            assert!(!row["reading_turns"].as_array().unwrap().is_empty());
            assert_eq!(
                row["reading_turn_provenance"]["machine_revision"],
                row["machine_revision"]
            );
            assert_eq!(
                row["reading_alignment"]["words"].as_array().unwrap().len(),
                if case == "chinese" { 10 } else { 5 }
            );
            for word in row["reading_alignment"]["words"].as_array().unwrap() {
                let a = word["start_char"].as_u64().unwrap() as usize;
                let b = word["end_char"].as_u64().unwrap() as usize;
                assert_eq!(
                    expected.chars().skip(a).take(b - a).collect::<String>(),
                    word["text"].as_str().unwrap()
                );
                assert_eq!(
                    word["start_utf16"].as_u64().unwrap() as usize,
                    expected.chars().take(a).map(char::len_utf16).sum::<usize>()
                );
                assert_eq!(
                    word["end_utf16"].as_u64().unwrap() as usize,
                    expected.chars().take(b).map(char::len_utf16).sum::<usize>()
                );
            }
            assert_eq!(
                fs::read_to_string(root.join("context-crops"))
                    .unwrap()
                    .lines()
                    .map(|line| line
                        .split_whitespace()
                        .next()
                        .unwrap()
                        .parse::<usize>()
                        .unwrap())
                    .collect::<Vec<_>>(),
                vec![275200, 275713, 51713]
            );
            let evidence = serde_json::to_value(&after["document"]).unwrap();
            let crops = fs::read(root.join("context-crops")).unwrap();
            let probes = fs::read(root.join("language-probe-calls")).ok();
            host.close(root).await;
            let reopened = Running::open(root, false).await;
            assert_eq!(reopened.get(&path).await["document"], evidence);
            assert_eq!(
                reopened
                    .post(&format!("{path}/refinement/resume"), json!({}))
                    .await
                    .status(),
                202
            );
            let repeated = reopened
                .wait(&path, |v| v["inference_owned"] == false)
                .await;
            assert_eq!(repeated["document"], evidence);
            assert_eq!(fs::read(root.join("context-crops")).unwrap(), crops);
            assert_eq!(fs::read(root.join("language-probe-calls")).ok(), probes);
            let exported = reopened
                .client
                .get(format!("{}{path}/export/txt", reopened.url))
                .send()
                .await
                .unwrap()
                .error_for_status()
                .unwrap()
                .text()
                .await
                .unwrap();
            assert!(exported.contains(expected));
            reopened.close(root).await;
        } else {
            if case == "edit" {
                assert_eq!(row["text"], "Kept while the final context was running");
                assert_eq!(row["start"], 0.1);
            } else if case == "pause" {
                assert_eq!(after["document"], before["document"]);
                assert_eq!(after["refinement_status"], "paused");
            } else {
                assert_eq!(
                    after["document"], before["document"],
                    "{case} must retain previous legible words"
                );
                assert_eq!(after["refinement_status"], "unresolved", "{case}");
            }
            host.close(root).await;
        }
    }
}

#[tokio::test]
async fn saved_auto_accuracy_uses_fresh_probe_and_keeps_captured_mode_after_restart() {
    let temp = tempfile::tempdir().unwrap();
    let root = temp.path();
    let host = Running::open(root, false).await;
    trigger(root, 1);
    let response = host
        .post(
            "api/meetings",
            json!({
                "name":"Disposable Auto routing replay","language":"auto","sources":["microphone"]
            }),
        )
        .await;
    assert_eq!(response.status(), 202, "{}", response.text().await.unwrap());
    let id = host.wait("api/meeting", |v| v["id"].is_string()).await["id"]
        .as_str()
        .unwrap()
        .to_owned();
    let path = format!("api/jobs/{id}");
    assert_eq!(
        host.post(&format!("{path}/refinement/pause"), json!({}))
            .await
            .status(),
        200
    );
    host.wait(&path, |v| v["duration"].as_f64().unwrap_or(0.) >= 6.5)
        .await;
    let before = host.stop(&id).await;
    assert_eq!(before["status"], "ready");
    let old = &before["document"]["segments"][0];
    assert_eq!(old["language_mode"], "auto");
    assert_eq!(old["language"], "en");
    fs::write(
        root.join("replay-probe.json"),
        // Retained real score precision exercises JSON/SQLite checkpoint stability.
        json!({"probabilities":[f64::from(0.9886158_f32),f64::from(0.0020635556_f32)],"codes":["de","en"]}).to_string(),
    )
    .unwrap();
    fs::write(
        root.join("replay-transcript"),
        "Neue Worte aus dem gespeicherten Audio.",
    )
    .unwrap();
    let calls_before = fs::read_to_string(root.join("language-probe-calls"))
        .unwrap()
        .lines()
        .count();
    let language_calls_before = fs::read_to_string(root.join("transcribe-languages"))
        .unwrap()
        .lines()
        .count();
    let resume = host
        .post(&format!("{path}/refinement/resume"), json!({}))
        .await;
    assert_eq!(resume.status(), 202, "{}", resume.text().await.unwrap());
    let after = host.wait(&path, |v| v["inference_owned"] == false).await;
    assert_eq!(after["refinement_status"], "complete", "{after}");
    let row = after["document"]["segments"]
        .as_array()
        .unwrap()
        .iter()
        .find(|r| r["id"] == old["id"])
        .unwrap();
    assert_eq!(row["text"], "Neue Worte aus dem gespeicherten Audio.");
    assert_eq!(row["language"], "de");
    assert_eq!(row["language_mode"], "auto");
    assert_eq!(row["language_epoch"], old["language_epoch"]);
    assert_eq!(row["language_detection"]["mode"], "auto");
    assert!(
        fs::read_to_string(root.join("language-probe-calls"))
            .unwrap()
            .lines()
            .count()
            > calls_before
    );
    let languages = fs::read_to_string(root.join("transcribe-languages")).unwrap();
    assert!(
        languages
            .lines()
            .skip(language_calls_before)
            .all(|language| language == "de")
    );
    assert_eq!(
        after["document"]["segments"].as_array().unwrap().len(),
        before["document"]["segments"].as_array().unwrap().len()
    );
    host.close(root).await;
    let reopened = Running::open(root, false).await;
    assert_eq!(reopened.get(&path).await["document"], after["document"]);
    let probes_after = fs::read_to_string(root.join("language-probe-calls")).unwrap();
    let asr_after = fs::read_to_string(root.join("transcribe-languages")).unwrap();
    let repeat = reopened
        .post(&format!("{path}/refinement/resume"), json!({}))
        .await;
    assert_eq!(repeat.status(), 202, "{}", repeat.text().await.unwrap());
    let repeated = reopened
        .wait(&path, |v| v["inference_owned"] == false)
        .await;
    assert_eq!(repeated["refinement_status"], "complete");
    assert_eq!(
        repeated["document"], after["document"],
        "completed Auto anchors must not republish after SQLite reopen"
    );
    assert_eq!(
        fs::read_to_string(root.join("language-probe-calls")).unwrap(),
        probes_after
    );
    assert_eq!(
        fs::read_to_string(root.join("transcribe-languages")).unwrap(),
        asr_after
    );
    reopened.close(root).await;
}

#[tokio::test]
async fn rolling_accuracy_keeps_capture_running_and_owns_cancel_until_terminal() {
    let temp = tempfile::tempdir().unwrap();
    let root = temp.path();
    // Language changes seal jointly observed audio: lcm(VAD 512, diarization 160).
    let first_samples = 44 * 2560;
    fs::write(root.join("first-stage-samples"), first_samples.to_string()).unwrap();
    let host = Running::open(root, false).await;
    let id = host.start(json!(["microphone"])).await;
    let path = format!("api/jobs/{id}");
    // Disable automatic dispatch before producing the first sealed anchor.
    let paused = host
        .post(&format!("{path}/refinement/pause"), json!({}))
        .await;
    assert_eq!(
        paused.status(),
        200,
        "live accuracy must have working controls: {}",
        paused.text().await.unwrap()
    );
    trigger(root, 1);
    host.wait(&path, |job| job["duration"].as_f64().unwrap_or(0.) >= 6.5)
        .await;
    assert_eq!(
        host.post(&format!("api/meetings/{id}/pause"), json!({}))
            .await
            .status(),
        202
    );
    host.wait(&path, |job| job["pause_flush"]["status"] == "complete")
        .await;
    let boundary = host
        .client
        .patch(format!("{}api/meetings/{id}/language", host.url))
        .header("X-Speakerdesk-Token", &host.token)
        .json(&json!({"language":"fr","language_revision":0}))
        .send()
        .await
        .unwrap();
    assert_eq!(boundary.status(), 200, "{}", boundary.text().await.unwrap());
    let before = host
        .wait(&path, |job| {
            job["language_acknowledged_revision"] == 1
                && job["document"]["segments"].as_array().is_some_and(|rows| {
                    rows.iter().any(|row| {
                        row["canonical_state"] == "sealed" && row["text"] == "Replay transcript"
                    })
                })
        })
        .await;
    let prior = before["document"]["segments"][0].clone();
    let sid = prior["id"].as_str().unwrap();
    fs::write(
        root.join("replay-transcript"),
        "Accuracy candidate for the sealed prefix.",
    )
    .unwrap();
    fs::write(
        root.join("hold-live-asr"),
        b"hold rolling accuracy response",
    )
    .unwrap();
    fs::write(root.join("hold-cancel-terminal"), b"ack is not terminal").unwrap();
    let resumed = host
        .post(&format!("{path}/refinement/resume"), json!({}))
        .await;
    assert_eq!(resumed.status(), 202, "{}", resumed.text().await.unwrap());
    wait_file(&root.join("edit-asr-started")).await;
    assert_eq!(
        host.post(&format!("api/meetings/{id}/resume"), json!({}))
            .await
            .status(),
        202
    );
    trigger(root, 2);
    let recording = host
        .wait(&path, |job| job["duration"].as_f64().unwrap_or(0.) >= 13.5)
        .await;
    assert_eq!(recording["status"], "recording");
    assert_eq!(recording["document"]["segments"][0]["text"], prior["text"]);
    assert!(
        fs::metadata(root.join("data").join(&id).join("audio.wav"))
            .unwrap()
            .len()
            >= 44 + 216000 * 2,
        "capture must commit new saved PCM while accuracy is held"
    );
    let edited = host.client.patch(format!("{}{path}/segments/{sid}", host.url))
        .header("X-Speakerdesk-Token", &host.token)
        .json(&json!({"segment_revision":prior["machine_revision"],"changes":{"text":"Protected rolling words"}})).send().await.unwrap();
    assert_eq!(edited.status(), 200, "{}", edited.text().await.unwrap());
    assert_eq!(
        host.post(&format!("{path}/refinement/pause"), json!({}))
            .await
            .status(),
        200
    );
    wait_file(&root.join("asr-cancelled")).await;
    assert_eq!(host.get(&path).await["inference_owned"], true);
    assert_eq!(
        host.post(&format!("{path}/refinement/resume"), json!({}))
            .await
            .status(),
        409,
        "the active accuracy generation remains owned through its terminal response"
    );
    fs::remove_file(root.join("hold-cancel-terminal")).unwrap();
    fs::remove_file(root.join("hold-live-asr")).unwrap();
    host.wait("api/meeting", |meeting| {
        meeting["processed_seconds"].as_f64().unwrap_or(0.) >= 13.
    })
    .await;
    let final_job = host.stop(&id).await;
    assert_eq!(final_job["status"], "ready");
    assert_eq!(final_job["refinement_status"], "paused");
    let retained = final_job["document"]["segments"]
        .as_array()
        .unwrap()
        .iter()
        .find(|row| row["id"] == prior["id"])
        .unwrap();
    assert_eq!(
        retained["text"], "Protected rolling words",
        "late cancelled accuracy cannot overwrite edits"
    );
    assert_eq!(
        final_job["canonical_observed_sample"],
        first_samples + 112000 + 513
    );
    host.close(root).await;
    let reopened = Running::open(root, false).await;
    assert_eq!(reopened.get(&path).await["document"], final_job["document"]);
    reopened.close(root).await;
}

#[tokio::test]
async fn rolling_accuracy_default_sources_language_recovery_and_shutdown_matrix() {
    for case in ["default", "dual", "epoch", "loss", "shutdown"] {
        let temp = tempfile::tempdir().unwrap();
        let root = temp.path();
        fs::write(root.join("first-stage-samples"), (44 * 2560).to_string()).unwrap();
        let dual = case == "dual";
        if dual {
            fs::write(root.join("distinct-system"), b"independent source samples").unwrap();
        }
        let host = Running::open(root, dual).await;
        let sources = if dual {
            json!(["microphone", "system"])
        } else {
            json!(["microphone"])
        };
        let id = if case == "default" {
            host.start(sources).await
        } else {
            host.start_live_only(sources).await
        };
        let path = format!("api/jobs/{id}");
        if case == "default" {
            assert_eq!(host.get(&path).await["refinement_status"], "waiting");
        }
        trigger(root, 1);
        host.wait(&path, |job| job["duration"].as_f64().unwrap_or(0.) >= 6.5)
            .await;
        assert_eq!(
            host.post(&format!("api/meetings/{id}/pause"), json!({}))
                .await
                .status(),
            202
        );
        host.wait(&path, |job| job["pause_flush"]["status"] == "complete")
            .await;
        let changed = host
            .client
            .patch(format!("{}api/meetings/{id}/language", host.url))
            .header("X-Speakerdesk-Token", &host.token)
            .json(&json!({"language":"fr","language_revision":0}))
            .send()
            .await
            .unwrap();
        assert_eq!(changed.status(), 200, "{}", changed.text().await.unwrap());
        let before = host
            .wait(&path, |job| {
                job["language_acknowledged_revision"] == 1
                    && job["document"]["segments"].as_array().is_some_and(|rows| {
                        !rows.is_empty()
                            && rows.iter().all(|row| row["canonical_state"] == "sealed")
                    })
            })
            .await;
        let priors = before["document"]["segments"].as_array().unwrap().clone();
        if case != "default" {
            fs::write(root.join("transcribe-crops"), b"").unwrap();
            fs::write(
                root.join("replay-transcript"),
                "Fresh rolling accuracy words.",
            )
            .unwrap();
            fs::write(root.join("hold-live-asr"), b"owned rolling request").unwrap();
            assert_eq!(
                host.post(&format!("{path}/refinement/resume"), json!({}))
                    .await
                    .status(),
                202
            );
            wait_file(&root.join("edit-asr-started")).await;
            if case == "shutdown" {
                host.close(root).await;
                fs::remove_file(root.join("hold-live-asr")).unwrap();
                let reopened = Running::open(root, false).await;
                let interrupted = reopened.get(&path).await;
                assert_eq!(interrupted["refinement_status"], "paused");
                assert_eq!(interrupted["document"], before["document"]);
                reopened.close(root).await;
                continue;
            }
            if case == "epoch" {
                use speakerdesk_core::{
                    ids::JobId,
                    store::{JobStore, SqliteJobStore},
                };
                let mut store = SqliteJobStore::open(root.join("data/speakerdesk.sqlite")).unwrap();
                let mut job = store
                    .get(&JobId::parse(&id).unwrap(), false)
                    .unwrap()
                    .unwrap();
                job.extra.get_mut("language_history").unwrap()[0]["language"] = json!("de");
                store.put(&job, None).unwrap();
            }
            if case == "loss" {
                let pid = fs::read_to_string(root.join("fixture-child-pid"))
                    .unwrap()
                    .parse::<i32>()
                    .unwrap();
                assert_eq!(unsafe { libc::kill(pid, libc::SIGKILL) }, 0);
                fs::remove_file(root.join("hold-live-asr")).unwrap();
                host.wait(&path, |job| job["live_recovery"]["status"] == "recovered")
                    .await;
            } else {
                host.name_live_speaker(
                    &path,
                    priors[0]["speaker"].as_str().unwrap(),
                    "Rolling speaker",
                )
                .await;
                // A future captured epoch does not invalidate the sealed English anchor.
                let changed = host
                    .client
                    .patch(format!("{}api/meetings/{id}/language", host.url))
                    .header("X-Speakerdesk-Token", &host.token)
                    .json(&json!({"language":"de","language_revision":1}))
                    .send()
                    .await
                    .unwrap();
                assert_eq!(changed.status(), 200, "{}", changed.text().await.unwrap());
                fs::remove_file(root.join("hold-live-asr")).unwrap();
            }
        }
        let accepted = host
            .wait(&path, |job| {
                if case == "epoch" {
                    job["refinement_status"] == "unresolved"
                } else {
                    priors.iter().all(|prior| {
                        job["document"]["segments"]
                            .as_array()
                            .unwrap()
                            .iter()
                            .any(|row| {
                                row["id"] == prior["id"] && row["refinement_state"] == "refined"
                            })
                    })
                }
            })
            .await;
        for prior in &priors {
            let row = accepted["document"]["segments"]
                .as_array()
                .unwrap()
                .iter()
                .find(|row| row["id"] == prior["id"])
                .unwrap();
            assert_eq!(row["capture_source"], prior["capture_source"]);
            assert_eq!(row["language_epoch"], prior["language_epoch"]);
            assert_eq!(row["language_mode"], "en");
            if case == "epoch" {
                assert_eq!(row, prior, "changed captured epoch rejects late accuracy");
            } else {
                assert!(
                    row["machine_revision"].as_i64().unwrap()
                        > prior["machine_revision"].as_i64().unwrap()
                        || case == "default"
                );
            }
        }
        if dual {
            let calls = fs::read_to_string(root.join("transcribe-crops"))
                .unwrap()
                .lines()
                .map(|line| serde_json::from_str::<Value>(line).unwrap())
                .collect::<Vec<_>>();
            let mut digests = Vec::new();
            for prior in &priors {
                let source = prior["capture_source"]["source_id"].as_str().unwrap();
                let samples = hound::WavReader::open(
                    root.join("data").join(&id).join(format!("{source}.wav")),
                )
                .unwrap()
                .into_samples::<i16>()
                .map(Result::unwrap)
                .skip(prior["start_sample"].as_u64().unwrap() as usize)
                .take(
                    (prior["end_sample"].as_u64().unwrap()
                        - prior["start_sample"].as_u64().unwrap()) as usize,
                )
                .map(|sample| f32::from(sample) / 32768.)
                .collect::<Vec<_>>();
                let digest = format!(
                    "{:x}",
                    Sha256::digest(
                        samples
                            .iter()
                            .flat_map(|sample| sample.to_le_bytes())
                            .collect::<Vec<_>>()
                    )
                );
                assert!(
                    calls
                        .iter()
                        .any(|call| call["samples"] == samples.len() && call["sha256"] == digest),
                    "accuracy must use {source}"
                );
                digests.push(digest);
            }
            assert_ne!(
                digests[0], digests[1],
                "distinct sources make mixed-audio mistakes observable"
            );
        }
        assert_eq!(
            host.post(&format!("api/meetings/{id}/resume"), json!({}))
                .await
                .status(),
            202
        );
        trigger(root, 2);
        host.wait(&path, |job| job["duration"].as_f64().unwrap_or(0.) >= 13.5)
            .await;
        let final_job = host.stop(&id).await;
        assert_eq!(final_job["status"], "ready", "{case}: {final_job}");
        for prior in &priors {
            let row = final_job["document"]["segments"]
                .as_array()
                .unwrap()
                .iter()
                .find(|row| row["id"] == prior["id"])
                .unwrap();
            let refined = accepted["document"]["segments"]
                .as_array()
                .unwrap()
                .iter()
                .find(|row| row["id"] == prior["id"])
                .unwrap();
            assert_eq!(
                row, refined,
                "subsequent live/final output must preserve accepted anchor"
            );
        }
        host.close(root).await;
        let calls = fs::read_to_string(root.join("transcribe-languages")).unwrap();
        let reopened = Running::open(root, dual).await;
        assert_eq!(reopened.get(&path).await["document"], final_job["document"]);
        if case != "epoch" {
            assert_eq!(
                reopened
                    .post(&format!("{path}/refinement/resume"), json!({}))
                    .await
                    .status(),
                202
            );
            reopened
                .wait(&path, |job| {
                    job["refinement_status"] == "complete" && job["inference_owned"] == false
                })
                .await;
            assert_eq!(
                fs::read_to_string(root.join("transcribe-languages")).unwrap(),
                calls,
                "completed anchors survive restart without repeated model work"
            );
        }
        reopened.close(root).await;
    }
}

#[tokio::test]
async fn rolling_long_context_yields_to_live_audio_before_its_last_crop() {
    let temp = tempfile::tempdir().unwrap();
    let root = temp.path();
    fs::write(root.join("fixture-ctc"), b"synthetic overlap ownership").unwrap();
    fs::create_dir(root.join("coarse-alignment")).unwrap();
    fs::write(
        root.join("coarse-alignment/tokens.txt"),
        include_str!("testdata/ctc-tokens.txt"),
    )
    .unwrap();
    let host = Running::open(root, false).await;
    let id = host.start_live_only(json!(["microphone"])).await;
    let path = format!("api/jobs/{id}");
    for stage in 1..=4 {
        trigger(root, stage);
        host.wait(&path, |job| {
            job["duration"].as_f64().unwrap_or(0.) >= f64::from(stage * 7) - 0.5
        })
        .await;
    }
    assert_eq!(
        host.post(&format!("api/meetings/{id}/pause"), json!({}))
            .await
            .status(),
        202
    );
    host.wait(&path, |job| job["pause_flush"]["status"] == "complete")
        .await;
    let response = host
        .client
        .patch(format!("{}api/meetings/{id}/language", host.url))
        .header("X-Speakerdesk-Token", &host.token)
        .json(&json!({"language":"fr","language_revision":0}))
        .send()
        .await
        .unwrap();
    assert_eq!(response.status(), 200, "{}", response.text().await.unwrap());
    let before = host
        .wait(&path, |job| job["language_acknowledged_revision"] == 1)
        .await;
    let old = before["document"]["segments"][0].clone();
    assert_eq!(old["end_sample"], 448000);
    fs::write(root.join("context-replay.json"),json!([
        {"text":"start repeat bridge extra","request_samples":275200,"start_sample":0,"end_sample":272000,
         "words":[["start",16000],["repeat",192000],["bridge",208000],["extra",256000]]},
        {"text":"repeat bridge finish repeat","request_samples":275200,"start_sample":176000,"end_sample":448000,
         "words":[["repeat",192000],["bridge",208000],["finish",352000],["repeat",416000]]}
    ]).to_string()).unwrap();
    fs::write(root.join("hold-live-asr"), b"hold each scripted context").unwrap();
    assert_eq!(
        host.post(&format!("{path}/refinement/resume"), json!({}))
            .await
            .status(),
        202
    );
    wait_file(&root.join("context-started-0")).await;
    assert_eq!(
        host.post(&format!("api/meetings/{id}/resume"), json!({}))
            .await
            .status(),
        202
    );
    trigger(root, 5);
    host.wait(&path, |job| job["duration"].as_f64().unwrap_or(0.) >= 34.5)
        .await;
    fs::write(
        root.join("release-context-0"),
        b"finish first physical context",
    )
    .unwrap();
    wait_file(&root.join("context-started-1")).await;
    let progress = host.get("api/meeting").await;
    assert!(
        progress["processed_seconds"].as_f64().unwrap() > 28.,
        "live saved-audio observations must advance before the next accuracy context: {progress}"
    );
    let held = host.get(&path).await;
    assert_eq!(
        held["document"]["segments"][0]["text"], old["text"],
        "only a complete anchor may replace published words"
    );
    fs::write(root.join("release-context-1"), b"finish final context").unwrap();
    let accepted = host
        .wait(&path, |job| {
            job["document"]["segments"]
                .as_array()
                .unwrap()
                .iter()
                .any(|row| row["id"] == old["id"] && row["refinement_state"] == "refined")
        })
        .await;
    let row = accepted["document"]["segments"]
        .as_array()
        .unwrap()
        .iter()
        .find(|row| row["id"] == old["id"])
        .unwrap()
        .clone();
    assert_eq!(row["text"], "start repeat bridge finish repeat");
    assert!(row["reading_turns"].is_array());
    assert_eq!(
        fs::read_to_string(root.join("context-crops"))
            .unwrap()
            .lines()
            .count(),
        2
    );
    fs::remove_file(root.join("hold-live-asr")).unwrap();
    let final_job = host.stop(&id).await;
    assert_eq!(final_job["status"], "ready", "{final_job}");
    assert_eq!(
        final_job["document"]["segments"]
            .as_array()
            .unwrap()
            .iter()
            .find(|r| r["id"] == old["id"])
            .unwrap(),
        &row
    );
    assert_eq!(final_job["canonical_observed_sample"], 560513);
    host.close(root).await;
}

#[tokio::test]
async fn rolling_accuracy_pause_is_serviced_while_final_models_drain() {
    let temp = tempfile::tempdir().unwrap();
    let root = temp.path();
    let host = Running::open(root, false).await;
    let id = host.start(json!(["microphone"])).await;
    let path = format!("api/jobs/{id}");
    trigger(root, 1);
    host.wait("api/meeting", |job| {
        job["processed_seconds"].as_f64().unwrap_or(0.) >= 6.4
    })
    .await;
    let calls = fs::read_to_string(root.join("transcribe-languages"))
        .unwrap()
        .lines()
        .count();
    fs::write(root.join("hold-asr-call"), (calls + 1).to_string()).unwrap();
    fs::write(
        root.join("hold-live-asr"),
        b"hold final accuracy after final live ASR",
    )
    .unwrap();
    fs::write(root.join("hold-cancel-terminal"), b"terminal ownership").unwrap();
    assert_eq!(
        host.post(&format!("api/meetings/{id}/stop"), json!({}))
            .await
            .status(),
        202
    );
    wait_file(&root.join("edit-asr-started")).await;
    let draining = host.get(&path).await;
    assert_eq!(draining["message"], "Audio saved. Finishing transcript…");
    assert_eq!(draining["status"], "finishing");
    assert_eq!(draining["inference_owned"], true);
    assert_eq!(draining["capture_audio_coverage"]["clean_samples"], 112513);
    let response = host
        .post(&format!("{path}/refinement/pause"), json!({}))
        .await;
    assert_eq!(response.status(), 200, "{}", response.text().await.unwrap());
    wait_file(&root.join("asr-cancelled")).await;
    let cancelling = host.get(&path).await;
    assert_eq!(cancelling["status"], "finishing");
    assert_eq!(cancelling["inference_owned"], true);
    assert_eq!(cancelling["refinement_status"], "paused");
    assert_eq!(cancelling["message"], "Audio saved. Finishing transcript…");
    fs::remove_file(root.join("hold-cancel-terminal")).unwrap();
    fs::remove_file(root.join("hold-live-asr")).unwrap();
    let finished = host
        .wait(&path, |job| {
            job["status"] == "ready" && job["inference_owned"] == false
        })
        .await;
    assert_eq!(finished["refinement_status"], "paused");
    assert_eq!(finished["canonical_observed_sample"], 112513);
    assert_eq!(
        finished["document"]["segments"][0]["text"],
        "Replay transcript"
    );
    host.close(root).await;
}

#[tokio::test]
async fn saved_accuracy_pass_publishes_child_candidate_and_keeps_history_after_restart() {
    let temp = tempfile::tempdir().unwrap();
    let root = temp.path();
    let host = Running::open(root, false).await;
    trigger(root, 1);
    let id = host.start_live_only(json!(["microphone"])).await;
    let path = format!("api/jobs/{id}");
    host.wait(&path, |job| job["duration"].as_f64().unwrap_or(0.) >= 6.5)
        .await;
    let before = host.stop(&id).await;
    assert_eq!(before["status"], "ready");
    let old = &before["document"]["segments"][0];
    assert_eq!(old["text"], "Replay transcript");
    fs::write(
        root.join("replay-transcript"),
        "Revised saved recording words.",
    )
    .unwrap();
    let response = host
        .post(&format!("{path}/refinement/resume"), json!({}))
        .await;
    assert_eq!(response.status(), 202, "{}", response.text().await.unwrap());
    let after = host
        .wait(&path, |job| {
            job["refinement_status"] == "complete" && job["inference_owned"] == false
        })
        .await;
    assert_eq!(after["status"], "ready");
    let row = &after["document"]["segments"][0];
    assert_eq!(row["id"], old["id"]);
    assert_eq!(row["text"], "Revised saved recording words.");
    assert!(row["machine_revision"].as_i64().unwrap() > old["machine_revision"].as_i64().unwrap());
    assert_eq!(
        after["document"]["segments"].as_array().unwrap().len(),
        before["document"]["segments"].as_array().unwrap().len()
    );
    let history = host
        .get(&format!(
            "{path}/refinement/revisions/{}",
            row["refinement_window"].as_str().unwrap()
        ))
        .await;
    assert_eq!(
        history["segments"],
        json!([{
            "id":old["id"], "start":old["start"], "end":old["end"],
            "speaker":old["speaker"], "text":old["text"]
        }]),
        "history contains previous words, not duplicated acoustic/timing evidence"
    );
    host.close(root).await;
    let reopened = Running::open(root, false).await;
    let saved = reopened.get(&path).await;
    assert_eq!(saved["document"], after["document"]);
    assert_eq!(saved["refinement_status"], "complete");
    let repeat = reopened
        .post(&format!("{path}/refinement/resume"), json!({}))
        .await;
    assert_eq!(repeat.status(), 202, "{}", repeat.text().await.unwrap());
    let finished = reopened
        .wait(&path, |job| {
            job["refinement_status"] == "complete" && job["inference_owned"] == false
        })
        .await;
    assert_eq!(
        finished["document"], after["document"],
        "completed anchors must not publish twice"
    );
    reopened.close(root).await;
}

#[tokio::test]
async fn saved_accuracy_pause_failure_and_late_candidate_matrix() {
    for case in [
        "edit",
        "pause",
        "cancel",
        "failure",
        "restart",
        "source",
        "epoch",
        "unrelated",
    ] {
        let temp = tempfile::tempdir().unwrap();
        let root = temp.path();
        let host = Running::open(root, matches!(case, "source" | "unrelated")).await;
        trigger(root, 1);
        let id = host
            .start_live_only(if matches!(case, "source" | "unrelated") {
                json!(["microphone", "system"])
            } else {
                json!(["microphone"])
            })
            .await;
        let path = format!("api/jobs/{id}");
        host.wait(&path, |job| job["duration"].as_f64().unwrap_or(0.) >= 6.5)
            .await;
        let before = host.stop(&id).await;
        assert_eq!(before["status"], "ready", "{case}");
        let old = &before["document"]["segments"][0];
        let row_id = old["id"].as_str().unwrap();
        fs::write(root.join("replay-transcript"), "New accuracy words.").unwrap();
        if case == "failure" {
            fs::write(root.join("exit-live-asr-once"), b"fail saved inference").unwrap();
        } else {
            fs::write(root.join("hold-live-asr"), b"hold saved inference").unwrap();
            if case == "pause" {
                fs::write(root.join("hold-cancel-terminal"), b"hold cleanup").unwrap();
            }
        }
        let begin = host
            .post(&format!("{path}/refinement/resume"), json!({}))
            .await;
        assert_eq!(
            begin.status(),
            202,
            "{case}: {}",
            begin.text().await.unwrap()
        );
        if case == "failure" {
            let failed = host
                .wait(&path, |job| {
                    job["refinement_status"] == "unresolved" && job["inference_owned"] == false
                })
                .await;
            assert_eq!(failed["document"], before["document"]);
        } else {
            wait_file(&root.join("edit-asr-started")).await;
            assert_eq!(
                host.post(&format!("{path}/refinement/resume"), json!({}))
                    .await
                    .status(),
                409,
                "{case}: admission remains held"
            );
            match case {
                "edit" => {
                    let response=host.client.patch(format!("{}{path}/segments/{row_id}",host.url)).header("X-Speakerdesk-Token",&host.token)
                        .json(&json!({"segment_revision":old["machine_revision"],"changes":{"text":"Protected during accuracy work","start":0.1}})).send().await.unwrap();
                    assert_eq!(response.status(), 200, "{}", response.text().await.unwrap());
                    let current = host.get(&path).await;
                    let name=host.post(&format!("{path}/speakers/{}/identity",old["speaker"].as_str().unwrap()),json!({"revision":current["revision"],"name":"Preserved name","person_id":null})).await;
                    assert_eq!(name.status(), 200, "{}", name.text().await.unwrap());
                }
                "unrelated" => {
                    let current = host.get(&path).await;
                    let mut document = current["document"].clone();
                    assert_eq!(document["segments"].as_array().unwrap().len(), 2);
                    document["segments"][1]["text"] = json!("Protected unrelated source words");
                    document["speakers"][old["speaker"].as_str().unwrap()] =
                        json!("Preserved name");
                    let saved = host
                        .client
                        .put(format!("{}{path}/transcript", host.url))
                        .header("X-Speakerdesk-Token", &host.token)
                        .json(&json!({"revision":current["revision"],"document":document}))
                        .send()
                        .await
                        .unwrap();
                    assert_eq!(saved.status(), 200, "{}", saved.text().await.unwrap());
                }
                "pause" => {
                    assert_eq!(
                        host.post(&format!("{path}/refinement/pause"), json!({}))
                            .await
                            .status(),
                        200
                    );
                    wait_file(&root.join("asr-cancelled")).await;
                    assert_eq!(host.get(&path).await["inference_owned"], true);
                    assert_eq!(
                        host.post(&format!("{path}/refinement/resume"), json!({}))
                            .await
                            .status(),
                        409,
                        "cancel acknowledgement is not terminal cleanup"
                    );
                    fs::remove_file(root.join("hold-cancel-terminal")).unwrap();
                }
                "cancel" => {
                    assert_eq!(
                        host.post(&format!("{path}/cancel"), json!({}))
                            .await
                            .status(),
                        200
                    );
                }
                "restart" => {
                    host.close(root).await;
                    fs::remove_file(root.join("hold-live-asr")).unwrap();
                    let reopened = Running::open(root, false).await;
                    let saved = reopened.get(&path).await;
                    assert_eq!(saved["refinement_status"], "paused");
                    assert_eq!(saved["document"], before["document"]);
                    assert_eq!(
                        reopened
                            .post(&format!("{path}/refinement/resume"), json!({}))
                            .await
                            .status(),
                        202
                    );
                    reopened
                        .wait(&path, |job| {
                            job["refinement_status"] == "complete"
                                && job["inference_owned"] == false
                        })
                        .await;
                    reopened.close(root).await;
                    continue;
                }
                "source" | "epoch" => {
                    use speakerdesk_core::{ids::JobId, store::SqliteJobStore};
                    let mut store =
                        SqliteJobStore::open(root.join("data/speakerdesk.sqlite")).unwrap();
                    store
                        .update_job(&JobId::parse(&id).unwrap(), |job| {
                            let row = job
                                .document
                                .as_mut()
                                .unwrap()
                                .segments
                                .iter_mut()
                                .find(|s| s.id == row_id)
                                .unwrap();
                            if case == "source" {
                                row.capture_source.as_mut().unwrap()["source_revision"] = json!(2);
                            } else {
                                row.rest.insert("language_epoch".into(), json!(99));
                            }
                            Ok::<_, speakerdesk_core::error::StoreError>(())
                        })
                        .unwrap();
                }
                _ => unreachable!(),
            }
            fs::remove_file(root.join("hold-live-asr")).unwrap();
            let retained = host
                .wait(&path, |job| job["inference_owned"] == false)
                .await;
            let row = retained["document"]["segments"]
                .as_array()
                .unwrap()
                .iter()
                .find(|s| s["id"] == old["id"])
                .unwrap();
            assert_eq!(
                row["text"],
                if case == "unrelated" {
                    json!("New accuracy words.")
                } else if case == "edit" {
                    json!("Protected during accuracy work")
                } else {
                    old["text"].clone()
                },
                "{case}"
            );
            assert_eq!(retained["status"], "ready");
            if case == "edit" {
                assert_eq!(row["start"], 0.1);
                assert_eq!(
                    retained["document"]["speakers"][old["speaker"].as_str().unwrap()],
                    "Preserved name"
                );
            }
            if case == "unrelated" {
                let rows = retained["document"]["segments"].as_array().unwrap();
                let before_rows = before["document"]["segments"].as_array().unwrap();
                assert_eq!(
                    rows.iter().map(|s| &s["id"]).collect::<Vec<_>>(),
                    before_rows.iter().map(|s| &s["id"]).collect::<Vec<_>>(),
                    "source rows keep their stored order"
                );
                let other_id = &before_rows[1]["id"];
                assert_eq!(
                    rows.iter().find(|s| &s["id"] == other_id).unwrap()["text"],
                    "Protected unrelated source words"
                );
                assert_eq!(
                    retained["document"]["speakers"][old["speaker"].as_str().unwrap()],
                    "Preserved name"
                );
                assert_eq!(retained["refinement_status"], "complete");
            }
            if matches!(case, "edit" | "source" | "epoch" | "unrelated") {
                host.close(root).await;
                continue;
            }
            assert_eq!(retained["refinement_status"], "paused");
        }
        assert_eq!(
            host.post(&format!("{path}/refinement/resume"), json!({}))
                .await
                .status(),
            202
        );
        let retried = host
            .wait(&path, |job| {
                job["refinement_status"] == "complete" && job["inference_owned"] == false
            })
            .await;
        assert_eq!(
            retried["document"]["segments"][0]["text"], "New accuracy words.",
            "{case}"
        );
        host.close(root).await;
    }
}
