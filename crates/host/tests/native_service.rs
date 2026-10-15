#![cfg(feature = "fixtures")]
//! Real native HTTP/control service, with a disposable inference protocol replay.
//! This exercises process cleanup, not acoustic inference or desktop UI behavior.
use serde_json::{Value, json};
use std::process::Stdio;
use std::time::Duration;
use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};
use tokio::process::Command;
#[path = "support/process.rs"]
mod process_support;
use process_support::ObservedProcess;

#[tokio::test]
async fn native_service_reaps_on_eof_bad_control_stalled_upload_and_update_shutdown() {
    for mode in ["eof", "oversized", "stalled-upload", "update"] {
        let temp = tempfile::tempdir().unwrap();
        let mut host = Command::new(env!("CARGO_BIN_EXE_inference-fixture"))
            .arg("--native-host")
            .arg(temp.path().join("data"))
            .arg(temp.path())
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::inherit())
            .kill_on_drop(true)
            .spawn()
            .unwrap();
        let mut input = host.stdin.take().unwrap();
        let mut output = BufReader::new(host.stdout.take().unwrap()).lines();
        let line = tokio::time::timeout(Duration::from_secs(5), output.next_line())
            .await
            .unwrap()
            .unwrap()
            .unwrap();
        let url = line.strip_prefix("SPEAKERDESK_URL=").unwrap();
        let client = reqwest::Client::new();
        tokio::time::timeout(Duration::from_secs(5), async {
            loop {
                let config: Value = client
                    .get(format!("{url}api/config"))
                    .send()
                    .await
                    .unwrap()
                    .json()
                    .await
                    .unwrap();
                if config["readiness"]["configured"] == true {
                    break;
                }
                tokio::time::sleep(Duration::from_millis(10)).await;
            }
        })
        .await
        .unwrap();
        let child = ObservedProcess::new(
            std::fs::read_to_string(temp.path().join("fixture-child-pid"))
                .unwrap()
                .parse()
                .unwrap(),
        );
        let html = client.get(url).send().await.unwrap().text().await.unwrap();
        let token = html
            .split("name=\"speakerdesk-token\" content=\"")
            .nth(1)
            .unwrap()
            .split('"')
            .next()
            .unwrap();
        let mut stalled = None;
        match mode {
            "eof" => drop(input),
            "oversized" => {
                input.write_all(&vec![b'x'; 8193]).await.unwrap();
                drop(input);
            }
            "stalled-upload" => {
                let address = url.strip_prefix("http://").unwrap().trim_end_matches('/');
                let mut stream = tokio::net::TcpStream::connect(address).await.unwrap();
                let request = format!(
                    "POST /api/jobs HTTP/1.1\r\nHost: {address}\r\nX-Speakerdesk-Token: {token}\r\nContent-Type: multipart/form-data; boundary=fixture\r\nContent-Length: 100000\r\nExpect: 100-continue\r\n\r\n"
                );
                stream.write_all(request.as_bytes()).await.unwrap();
                let mut stream = BufReader::new(stream);
                let mut response = String::new();
                tokio::time::timeout(Duration::from_secs(5), stream.read_line(&mut response))
                    .await
                    .unwrap()
                    .unwrap();
                assert!(response.contains("100 Continue"), "{response}");
                stalled = Some(stream);
                input.write_all(b"shutdown\n").await.unwrap();
            }
            "update" => {
                let updates = format!("{url}api/updates");
                for (body, control) in [
                    (
                        json!({"op":"check"}),
                        json!({"op":"status","id":1,"state":"ready"}),
                    ),
                    (
                        json!({"op":"install"}),
                        json!({"op":"status","id":1,"state":"preparing","preparation":1}),
                    ),
                    (
                        json!({"op":"editor_ready","id":1,"preparation":1}),
                        json!({"op":"reserve","id":1,"preparation":1}),
                    ),
                ] {
                    let reply = client
                        .post(&updates)
                        .header("X-Speakerdesk-Token", token)
                        .json(&body)
                        .send()
                        .await
                        .unwrap();
                    assert!(
                        reply.status().is_success(),
                        "{}",
                        reply.text().await.unwrap()
                    );
                    // The emitted request proves the corresponding HTTP transition completed.
                    let line = tokio::time::timeout(Duration::from_secs(5), output.next_line())
                        .await
                        .unwrap()
                        .unwrap()
                        .unwrap();
                    assert!(line.starts_with("SPEAKERDESK_UPDATE="), "{line}");
                    input
                        .write_all(format!("SPEAKERDESK_UPDATE={control}\n").as_bytes())
                        .await
                        .unwrap();
                    // Read back the applied status before the next HTTP operation.
                    if control["op"] == "status" {
                        tokio::time::timeout(Duration::from_secs(5), async {
                            loop {
                                let status: Value = client
                                    .get(&updates)
                                    .send()
                                    .await
                                    .unwrap()
                                    .json()
                                    .await
                                    .unwrap();
                                if status["state"] == control["state"] {
                                    break;
                                }
                                tokio::task::yield_now().await;
                            }
                        })
                        .await
                        .unwrap();
                    }
                }
                let reserved = tokio::time::timeout(Duration::from_secs(5), output.next_line())
                    .await
                    .unwrap()
                    .unwrap()
                    .unwrap();
                assert!(reserved.contains("\"op\":\"reserved\""), "{reserved}");
                assert!(reserved.contains("\"ok\":true"), "{reserved}");
                input
                    .write_all(
                        b"SPEAKERDESK_UPDATE={\"op\":\"shutdown\",\"id\":1,\"preparation\":1}\n",
                    )
                    .await
                    .unwrap();
                let ready = tokio::time::timeout(Duration::from_secs(5), output.next_line())
                    .await
                    .unwrap()
                    .unwrap()
                    .unwrap();
                assert!(ready.contains("\"op\":\"shutdown_ready\""), "{ready}");
                assert!(ready.contains("\"ok\":true"), "{ready}");
                assert!(
                    child.has_exited(),
                    "shutdown_ready was emitted before the inference child exited and was reaped"
                );
            }
            _ => unreachable!(),
        }
        // Child cleanup must not wait for the ten-second HTTP drain deadline.
        tokio::time::timeout(Duration::from_secs(5), async {
            while !child.has_exited() {
                tokio::time::sleep(Duration::from_millis(10)).await;
            }
        })
        .await
        .expect("native host left its inference child alive or unreaped");
        let status = tokio::time::timeout(Duration::from_secs(15), host.wait())
            .await
            .unwrap()
            .unwrap();
        assert_eq!(
            status.success(),
            ["eof", "update"].contains(&mode),
            "{mode}: {status}"
        );
        drop(stalled);
        assert!(
            client.get(url).send().await.is_err(),
            "listener survived {mode}"
        );
    }
}
