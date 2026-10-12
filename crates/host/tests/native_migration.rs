#![cfg(feature = "fixtures")]
//! Upgrade discovery through the production host, using only disposable storage.
use rusqlite::{Connection, OpenFlags, params};
use serde_json::{Value, json};
use speakerdesk_core::store::{JobStore, SqliteJobStore};
use speakerdesk_host::routes::{AppState, AudioPrep, WavPrepare};
use std::{path::Path, process::Stdio, time::Duration};
use tokio::{
    io::{AsyncBufReadExt, AsyncWriteExt, BufReader},
    process::{Child, Command},
};

const JOB_ID: &str = "0123456789abcdef0123456789abcdef";
const PERSON_ID: &str = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa";

fn spawn(storage: &Path, models: &Path) -> Child {
    Command::new(env!("CARGO_BIN_EXE_inference-fixture"))
        .arg("--native-host")
        .arg(storage)
        .arg(models)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .kill_on_drop(true)
        .spawn()
        .unwrap()
}

struct Host {
    child: Child,
    url: String,
    token: String,
    client: reqwest::Client,
}

impl Host {
    async fn start(storage: &Path, models: &Path) -> Self {
        let mut child = spawn(storage, models);
        let line = tokio::time::timeout(Duration::from_secs(5), async {
            BufReader::new(child.stdout.as_mut().unwrap())
                .lines()
                .next_line()
                .await
                .unwrap()
                .expect("native host did not start")
        })
        .await
        .unwrap();
        let url = line.strip_prefix("SPEAKERDESK_URL=").unwrap().to_owned();
        let client = reqwest::Client::new();
        let html = client.get(&url).send().await.unwrap().text().await.unwrap();
        let token = html
            .split("name=\"speakerdesk-token\" content=\"")
            .nth(1)
            .unwrap()
            .split('"')
            .next()
            .unwrap()
            .to_owned();
        Self {
            child,
            url,
            token,
            client,
        }
    }

    async fn json(&self, path: &str) -> Value {
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

    async fn stop(mut self) {
        self.child
            .stdin
            .take()
            .unwrap()
            .write_all(b"shutdown\n")
            .await
            .unwrap();
        let result = tokio::time::timeout(Duration::from_secs(15), self.child.wait_with_output())
            .await
            .unwrap()
            .unwrap();
        assert!(
            result.status.success(),
            "{}",
            String::from_utf8_lossy(&result.stderr)
        );
    }
}

fn seed_legacy(path: &Path) -> Vec<u8> {
    std::fs::create_dir_all(path.parent().unwrap()).unwrap();
    let db = Connection::open(path).unwrap();
    db.execute_batch(
        "CREATE TABLE jobs(id TEXT PRIMARY KEY, payload TEXT NOT NULL);
         CREATE TABLE people(id TEXT PRIMARY KEY, payload TEXT NOT NULL);
         CREATE TABLE preferences(key TEXT PRIMARY KEY, value TEXT NOT NULL);
         CREATE TABLE voice_profiles(person_id TEXT PRIMARY KEY, payload TEXT NOT NULL);
         INSERT INTO preferences VALUES('default_language', 'auto');",
    )
    .unwrap();
    let document = json!({"schema_version":1,"speakers":{"speaker_0":"Ada"},
        "segments":[{"id":"one","start":0.,"end":1.,"speaker":"speaker_0","text":"Original words."}]});
    db.execute(
        "INSERT INTO jobs VALUES(?1,?2)",
        params![
            JOB_ID,
            json!({"id":JOB_ID,"name":"Legacy recording","status":"ready","language":"en",
                "source_file":"source.wav","duration":1.,"revision":2,"created":1.,"updated":2.,"document":document})
                .to_string()
        ],
    )
    .unwrap();
    db.execute(
        "INSERT INTO people VALUES(?1,?2)",
        params![
            PERSON_ID,
            json!({"id":PERSON_ID,"name":"Ada","created":1.,"updated":2.}).to_string()
        ],
    )
    .unwrap();
    db.execute(
        "INSERT INTO voice_profiles VALUES(?1,?2)",
        params![
            PERSON_ID,
            json!({"person_id":PERSON_ID,"version":"cccccccccccccccccccccccccccccccc",
                "enrolled_at":3.,"consent":"explicit_remember_voice",
                "model":{"model_id":"legacy-redimnet","revision":"legacy-v1",
                    "artifact_sha256":"b".repeat(64),"dimension":2,"sample_rate":16000},
                "centroid":[1.,0.],"clips":[]})
            .to_string()
        ],
    )
    .unwrap();
    drop(db);
    let folder = path.parent().unwrap().join(JOB_ID);
    std::fs::create_dir_all(&folder).unwrap();
    let source = folder.join("source.wav");
    let mut audio = hound::WavWriter::create(
        &source,
        hound::WavSpec {
            channels: 1,
            sample_rate: 16000,
            bits_per_sample: 16,
            sample_format: hound::SampleFormat::Int,
        },
    )
    .unwrap();
    for _ in 0..16000 {
        audio.write_sample(123_i16).unwrap();
    }
    audio.finalize().unwrap();
    std::fs::copy(&source, folder.join("audio.wav")).unwrap();
    std::fs::read(path).unwrap()
}

#[tokio::test]
async fn native_startup_discovers_legacy_and_preserves_edits_and_media_on_restart() {
    for layout in ["packaged", "direct", "tauri-default"] {
        let temp = tempfile::tempdir().unwrap();
        let storage = if layout == "tauri-default" {
            temp.path()
                .join("Library/Application Support/com.speakerdesk.desktop")
        } else {
            temp.path().join("native")
        };
        let legacy = match layout {
            "packaged" => storage.join("recordings/jobs.sqlite"),
            "direct" => storage.join("jobs.sqlite"),
            _ => storage
                .with_file_name("Speakerdesk")
                .join("recordings/jobs.sqlite"),
        };
        let before = seed_legacy(&legacy);
        let legacy_media = legacy.parent().unwrap().join(JOB_ID);
        let audio = std::fs::read(legacy_media.join("audio.wav")).unwrap();
        let host = Host::start(&storage, temp.path()).await;
        let jobs = host.json("api/jobs").await;
        assert_eq!(jobs.as_array().unwrap().len(), 1, "{layout}");
        assert_eq!(jobs[0]["id"], JOB_ID, "{layout}");
        let people = host.json("api/people").await;
        assert_eq!(people["people"].as_array().unwrap().len(), 1, "{layout}");
        assert_eq!(people["people"][0]["id"], PERSON_ID, "{layout}");
        assert_eq!(people["people"][0]["name"], "Ada", "{layout}");
        assert_eq!(people["people"][0]["voice_saved"], true, "{layout}");
        assert_eq!(host.json("api/config").await["default_language"], "auto");
        let mut job = host.json(&format!("api/jobs/{JOB_ID}")).await;
        job["document"]["segments"][0]["text"] = json!("Saved after migration.");
        let saved: Value = host
            .client
            .put(format!("{}api/jobs/{JOB_ID}/transcript", host.url))
            .header("X-Speakerdesk-Token", &host.token)
            .json(&json!({"revision":job["revision"],"document":job["document"]}))
            .send()
            .await
            .unwrap()
            .error_for_status()
            .unwrap()
            .json()
            .await
            .unwrap();
        assert_eq!(
            saved["document"]["segments"][0]["text"],
            "Saved after migration."
        );
        host.stop().await;

        let store = SqliteJobStore::open(storage.join("speakerdesk.sqlite")).unwrap();
        let job = store
            .get(&speakerdesk_core::ids::JobId::parse(JOB_ID).unwrap(), false)
            .unwrap()
            .unwrap();
        assert_eq!(
            job.extra["legacy_media_root"],
            legacy
                .parent()
                .unwrap()
                .canonicalize()
                .unwrap()
                .to_str()
                .unwrap()
        );
        WavPrepare.prepare(&storage, &job).unwrap();
        assert_eq!(
            std::fs::read(storage.join("derived").join(JOB_ID).join("audio.wav")).unwrap(),
            audio
        );
        drop(store);

        let host = Host::start(&storage, temp.path()).await;
        assert_eq!(
            host.json(&format!("api/jobs/{JOB_ID}")).await["document"]["segments"][0]["text"],
            "Saved after migration."
        );
        assert_eq!(
            host.json("api/people").await["people"][0]["voice_saved"],
            true
        );
        assert_eq!(
            host.client
                .get(format!("{}api/jobs/{JOB_ID}/audio", host.url))
                .send()
                .await
                .unwrap()
                .error_for_status()
                .unwrap()
                .bytes()
                .await
                .unwrap()
                .as_ref(),
            audio
        );
        host.client
            .delete(format!("{}api/jobs/{JOB_ID}", host.url))
            .header("X-Speakerdesk-Token", &host.token)
            .send()
            .await
            .unwrap()
            .error_for_status()
            .unwrap();
        host.stop().await;
        assert!(!storage.join("derived").join(JOB_ID).exists());
        assert_eq!(std::fs::read(&legacy).unwrap(), before);
        for name in ["source.wav", "audio.wav"] {
            assert_eq!(std::fs::read(legacy_media.join(name)).unwrap(), audio);
        }
        let db = Connection::open_with_flags(
            storage.join("speakerdesk.sqlite"),
            OpenFlags::SQLITE_OPEN_READ_ONLY,
        )
        .unwrap();
        assert_eq!(
            db.query_row("SELECT count(*) FROM migrations", [], |row| row
                .get::<_, i64>(0))
                .unwrap(),
            1
        );
        assert_eq!(
            db.query_row(
                "SELECT centroid FROM voice_profiles WHERE person_id=?1",
                [PERSON_ID],
                |row| row.get::<_, Vec<u8>>(0)
            )
            .unwrap(),
            [1_f64.to_le_bytes(), 0_f64.to_le_bytes()].concat()
        );
    }
}

#[tokio::test]
async fn native_startup_failed_migration_does_not_create_a_store_and_can_retry() {
    let temp = tempfile::tempdir().unwrap();
    let storage = temp.path().join("native");
    let legacy = storage.join("recordings/jobs.sqlite");
    std::fs::create_dir_all(legacy.parent().unwrap()).unwrap();
    std::fs::write(&legacy, b"broken legacy SQLite").unwrap();
    let result = tokio::time::timeout(
        Duration::from_secs(5),
        spawn(&storage, temp.path()).wait_with_output(),
    )
    .await
    .unwrap()
    .unwrap();
    assert!(!result.status.success());
    assert!(
        String::from_utf8_lossy(&result.stderr)
            .contains("Legacy database schema cannot be imported")
    );
    assert!(!storage.join("speakerdesk.sqlite").exists());
    assert_eq!(std::fs::read(&legacy).unwrap(), b"broken legacy SQLite");
    assert!(!std::fs::read_dir(&storage).unwrap().any(|entry| {
        entry
            .unwrap()
            .file_name()
            .to_string_lossy()
            .contains(".partial")
    }));
    #[cfg(unix)]
    {
        std::fs::remove_file(&legacy).unwrap();
        std::os::unix::fs::symlink("missing-backup.sqlite", &legacy).unwrap();
        let result = tokio::time::timeout(
            Duration::from_secs(5),
            spawn(&storage, temp.path()).wait_with_output(),
        )
        .await
        .unwrap()
        .unwrap();
        assert!(!result.status.success());
        assert!(!storage.join("speakerdesk.sqlite").exists());
        assert_eq!(
            std::fs::read_link(&legacy).unwrap(),
            Path::new("missing-backup.sqlite")
        );
    }
    let abandoned = storage.join("speakerdesk.sqlite.interrupted.partial");
    std::fs::write(&abandoned, b"unpublished migration").unwrap();
    std::fs::remove_file(&legacy).unwrap();
    seed_legacy(&legacy);
    let host = Host::start(&storage, temp.path()).await;
    assert_eq!(host.json("api/jobs").await[0]["id"], JOB_ID);
    host.stop().await;
    assert_eq!(std::fs::read(abandoned).unwrap(), b"unpublished migration");
}

#[tokio::test]
async fn native_startup_keeps_existing_native_state_and_requires_its_owner_lock() {
    let temp = tempfile::tempdir().unwrap();
    let storage = temp.path().join("native");
    let state = AppState::open(&storage).unwrap();
    let legacy = storage.join("recordings/jobs.sqlite");
    let before = seed_legacy(&legacy);
    let result = tokio::time::timeout(
        Duration::from_secs(5),
        spawn(&storage, temp.path()).wait_with_output(),
    )
    .await
    .unwrap()
    .unwrap();
    assert!(!result.status.success());
    assert!(String::from_utf8_lossy(&result.stderr).contains("Cannot own native storage"));
    drop(state);
    let host = Host::start(&storage, temp.path()).await;
    assert_eq!(host.json("api/jobs").await, json!([]));
    assert_eq!(host.json("api/people").await["people"], json!([]));
    assert_eq!(host.json("api/config").await["default_language"], "en");
    host.stop().await;
    assert_eq!(std::fs::read(&legacy).unwrap(), before);
}
