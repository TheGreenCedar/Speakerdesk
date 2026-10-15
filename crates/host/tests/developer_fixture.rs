#![cfg(all(feature = "fixtures", unix))]
//! The developer CLI must serve the real editor API against disposable storage.

use std::os::unix::fs::PermissionsExt;
use std::path::PathBuf;
use std::process::Stdio;
use std::time::Duration;

use rusqlite::{Connection, OpenFlags};
use serde_json::{Value, json};
use tokio::io::{AsyncBufReadExt, BufReader};
use tokio::process::Command;

const JOB_ID: &str = "0123456789abcdef0123456789abcdef";
const WAIT: Duration = Duration::from_secs(5);

#[tokio::test]
async fn people_fixture_serves_authenticated_edits_persists_and_cleans_up() {
    let temporary_root = tempfile::tempdir().unwrap();
    let mut process = Command::new(env!("CARGO_BIN_EXE_speakerdesk-dev"))
        .args(["--fixture", "people"])
        .env("TMPDIR", temporary_root.path())
        .stdout(Stdio::piped())
        .stderr(Stdio::inherit())
        .kill_on_drop(true)
        .spawn()
        .unwrap();
    let mut output = BufReader::new(process.stdout.take().unwrap()).lines();
    let (url, directory) = tokio::time::timeout(WAIT, async {
        let url = output.next_line().await.unwrap().unwrap();
        let directory = output.next_line().await.unwrap().unwrap();
        (
            url.strip_prefix("SPEAKERDESK_URL=").unwrap().to_owned(),
            PathBuf::from(directory.strip_prefix("SPEAKERDESK_DATA_DIR=").unwrap()),
        )
    })
    .await
    .expect("fixture publishes its address and disposable library");
    assert!(url.starts_with("http://127.0.0.1:"));
    assert!(directory.starts_with(std::fs::canonicalize(temporary_root.path()).unwrap()));
    assert_eq!(
        std::fs::metadata(&directory).unwrap().permissions().mode() & 0o777,
        0o700
    );

    let client = reqwest::Client::builder().timeout(WAIT).build().unwrap();
    let html = client
        .get(&url)
        .send()
        .await
        .unwrap()
        .error_for_status()
        .unwrap()
        .text()
        .await
        .unwrap();
    let token = html
        .split("name=\"speakerdesk-token\" content=\"")
        .nth(1)
        .unwrap()
        .split('"')
        .next()
        .unwrap();
    client
        .get(format!("{url}static/app.js"))
        .send()
        .await
        .unwrap()
        .error_for_status()
        .unwrap();
    let detail_url = format!("{url}api/jobs/{JOB_ID}");
    let seeded: Value = client
        .get(&detail_url)
        .send()
        .await
        .unwrap()
        .error_for_status()
        .unwrap()
        .json()
        .await
        .unwrap();
    assert_eq!(seeded["status"], "ready");
    assert!(seeded["name"].as_str().unwrap().contains("Fictional"));
    assert_eq!(seeded["document"]["segments"].as_array().unwrap().len(), 3);

    let playback = client
        .get(format!("{detail_url}/audio"))
        .send()
        .await
        .unwrap()
        .error_for_status()
        .unwrap();
    assert_eq!(playback.headers()["accept-ranges"], "bytes");
    let wav = playback.bytes().await.unwrap();
    let mut audio = hound::WavReader::new(std::io::Cursor::new(&wav)).unwrap();
    assert_eq!(audio.spec().channels, 1);
    assert_eq!(audio.spec().sample_rate, 16_000);
    assert_eq!(audio.spec().bits_per_sample, 16);
    assert_eq!(audio.duration(), 18 * 16_000);
    assert!(audio.samples::<i16>().all(|sample| sample.unwrap() == 0));
    let passage_audio = client
        .get(format!("{detail_url}/passages/intro/audio"))
        .header("Range", "bytes=0-43")
        .send()
        .await
        .unwrap();
    assert_eq!(passage_audio.status(), 206);
    assert_eq!(passage_audio.bytes().await.unwrap().as_ref(), &wav[..44]);

    let identity_url = format!("{detail_url}/speakers/speaker_0/identity");
    let name = json!({"revision": seeded["revision"], "name": "Mira"});
    let denied = client.post(&identity_url).json(&name).send().await.unwrap();
    assert_eq!(denied.status(), 403);
    let named: Value = client
        .post(&identity_url)
        .header(speakerdesk_host::http::token_header(), token)
        .json(&name)
        .send()
        .await
        .unwrap()
        .error_for_status()
        .unwrap()
        .json()
        .await
        .unwrap();
    assert_eq!(named["revision"], seeded["revision"].as_i64().unwrap() + 1);
    assert_eq!(named["document"]["speakers"]["speaker_0"], "Mira");

    let mut document = named["document"].clone();
    document["segments"][0]["text"] = json!("The fictional launch is on Friday.");
    let saved: Value = client
        .put(format!("{detail_url}/transcript"))
        .header(speakerdesk_host::http::token_header(), token)
        .json(&json!({"revision": named["revision"], "document": document}))
        .send()
        .await
        .unwrap()
        .error_for_status()
        .unwrap()
        .json()
        .await
        .unwrap();
    let reopened: Value = client
        .get(&detail_url)
        .send()
        .await
        .unwrap()
        .error_for_status()
        .unwrap()
        .json()
        .await
        .unwrap();
    assert_eq!(reopened["document"], saved["document"]);
    let exported = client
        .get(format!("{detail_url}/export/txt"))
        .send()
        .await
        .unwrap()
        .error_for_status()
        .unwrap()
        .text()
        .await
        .unwrap();
    assert!(exported.contains("Mira:"), "{exported}");
    assert!(exported.contains("The fictional launch is on Friday."));

    // Reopen SQLite independently so a memory-only host cannot satisfy the check.
    let database = Connection::open_with_flags(
        directory.join("speakerdesk.sqlite"),
        OpenFlags::SQLITE_OPEN_READ_ONLY,
    )
    .unwrap();
    let (persisted_document, persisted_revision): (String, i64) = database
        .query_row(
            "SELECT document, revision FROM jobs WHERE id = ?1",
            [JOB_ID],
            |row| Ok((row.get(0)?, row.get(1)?)),
        )
        .unwrap();
    assert_eq!(
        serde_json::from_str::<Value>(&persisted_document).unwrap(),
        saved["document"]
    );
    assert_eq!(persisted_revision, saved["revision"].as_i64().unwrap());
    assert_eq!(
        database
            .query_row("SELECT COUNT(*) FROM voice_profiles", [], |row| row
                .get::<_, i64>(0))
            .unwrap(),
        0,
        "naming the fixture speaker never enrolls a voice"
    );
    drop(database);

    let pid = i32::try_from(process.id().unwrap()).unwrap();
    assert_eq!(unsafe { libc::kill(pid, libc::SIGINT) }, 0);
    let status = tokio::time::timeout(WAIT, process.wait())
        .await
        .expect("Ctrl-C shuts down the fixture")
        .unwrap();
    assert!(status.success(), "{status}");
    assert!(!directory.exists(), "fixture must remove its own library");
    assert!(temporary_root.path().exists());
    assert!(client.get(&url).send().await.is_err());
}
