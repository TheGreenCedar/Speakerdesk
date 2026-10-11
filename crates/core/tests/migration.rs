//! Migration tests on the five real v0.6.6 `tests/goldens/legacy-db/*.sqlite`
//! fixtures: product projection parity, legacy byte-identity, idempotence,
//! failure atomicity, recovery mapping, unknown-key coverage and voice-profile
//! centroid round-trip.

use std::fs;
use std::path::{Path, PathBuf};

use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use speakerdesk_core::model::Job;
use speakerdesk_core::store::Store;
use speakerdesk_core::store::legacy::{map_payload, migrate};

const NOW: f64 = 1_700_000_000.0;

fn fixtures() -> Vec<PathBuf> {
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../tests/goldens/legacy-db");
    let mut files: Vec<PathBuf> = std::fs::read_dir(&root)
        .unwrap()
        .filter_map(|e| {
            let p = e.unwrap().path();
            (p.extension().and_then(|x| x.to_str()) == Some("sqlite")).then_some(p)
        })
        .collect();
    files.sort();
    files
}

struct TempHome(PathBuf);

impl TempHome {
    /// `<tmp>/<unique>/recordings/jobs.sqlite`
    fn with_fixture(fixture: &Path) -> Self {
        let dir = std::env::temp_dir().join(format!(
            "speakerdesk-migrate-test-{}-{}-{}",
            std::process::id(),
            fixture.file_stem().unwrap().to_str().unwrap(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(dir.join("recordings")).unwrap();
        fs::copy(fixture, dir.join("recordings/jobs.sqlite")).unwrap();
        Self(dir)
    }
}

impl Drop for TempHome {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.0);
    }
}

fn sha256_file(path: &Path) -> String {
    hex::encode(Sha256::digest(fs::read(path).unwrap()))
}

/// Legacy payload + job_progress overlay, exactly as JobStore._decode reads.
fn legacy_jobs(db: &Path) -> Vec<Value> {
    let conn = rusqlite::Connection::open(db).unwrap();
    let mut stmt = conn
        .prepare(
            "SELECT j.payload, p.duration, p.updated FROM jobs j
             LEFT JOIN job_progress p ON p.id=j.id",
        )
        .unwrap();
    stmt.query_map([], |row| {
        Ok((
            row.get::<_, String>(0)?,
            row.get::<_, Option<f64>>(1)?,
            row.get::<_, Option<f64>>(2)?,
        ))
    })
    .unwrap()
    .map(|row| {
        let (payload, progress_duration, progress_updated) = row.unwrap();
        let mut job: Value = serde_json::from_str(&payload).unwrap();
        if let Some(duration) = progress_duration {
            let obj = job.as_object_mut().unwrap();
            let base = obj.get("duration").and_then(Value::as_f64).unwrap_or(0.0);
            obj.insert("duration".to_string(), json!(base.max(duration)));
            let updated = obj.get("updated").and_then(Value::as_f64).unwrap_or(0.0);
            obj.insert(
                "updated".to_string(),
                json!(updated.max(progress_updated.unwrap_or(0.0))),
            );
        }
        job
    })
    .collect()
}

/// The product projection the migration would produce — mapping, typed
/// deserialization, recovery, then the serialized payload (the importer
/// applies the same pipeline).
fn expected_job(payload: &Value) -> Value {
    let mut raw = payload.clone();
    map_payload(&mut raw);
    let mut job: Job = serde_json::from_value(raw).unwrap();
    speakerdesk_core::store::recover_job(&mut job, NOW).unwrap();
    serde_json::to_value(&job).unwrap()
}

fn legacy_table(db: &Path, table: &str, key_col: &str) -> Vec<(String, Value)> {
    let conn = rusqlite::Connection::open(db).unwrap();
    let mut stmt = conn
        .prepare(&format!("SELECT {key_col}, payload FROM {table}"))
        .unwrap();
    stmt.query_map([], |row| {
        Ok((
            row.get::<_, String>(0)?,
            serde_json::from_str::<Value>(&row.get::<_, String>(1)?).unwrap(),
        ))
    })
    .unwrap()
    .collect::<Result<Vec<_>, _>>()
    .unwrap()
}

#[test]
fn all_fixtures_migrate() {
    for fixture in fixtures() {
        let name = fixture.file_stem().unwrap().to_str().unwrap().to_string();
        let home = TempHome::with_fixture(&fixture);
        let legacy = home.0.join("recordings/jobs.sqlite");
        let before = sha256_file(&legacy);

        let outcome = migrate(&home.0, NOW).unwrap().expect(&name);
        assert!(home.0.join("speakerdesk.sqlite").exists(), "{name}");
        assert!(
            !home.0.join("speakerdesk.sqlite.partial").exists(),
            "{name}: partial left behind"
        );
        assert_eq!(sha256_file(&legacy), before, "{name}: legacy mutated");
        assert!(
            outcome.unknown_keys.is_empty(),
            "{name}: unknown keys {:?}",
            outcome.unknown_keys
        );

        // Job-by-job product parity.
        let store = Store::open(&home.0).unwrap();
        let expected = legacy_jobs(&legacy);
        assert_eq!(outcome.jobs, expected.len(), "{name}");
        for want in expected {
            let want = expected_job(&want);
            let id = want["id"].as_str().unwrap().to_string();
            let got = store.get(&id, false).unwrap().expect(&name);
            assert_eq!(
                serde_json::to_value(&got).unwrap(),
                want,
                "{name}: job {id}"
            );
        }

        // People / voice profiles / preferences.
        let expected_people: Vec<Value> = legacy_table(&legacy, "people", "id")
            .into_iter()
            .map(|(_, p)| p)
            .collect();
        let mut got_people: Vec<Value> = store
            .list_people()
            .unwrap()
            .iter()
            .map(|p| serde_json::to_value(p).unwrap())
            .collect();
        let mut want_people = expected_people;
        got_people.sort_by_key(|p| p["id"].to_string());
        want_people.sort_by_key(|p| p["id"].to_string());
        assert_eq!(got_people, want_people, "{name}: people");

        let expected_voices = legacy_table(&legacy, "voice_profiles", "person_id");
        assert_eq!(outcome.voice_profiles, expected_voices.len(), "{name}");
        for (person_id, profile) in expected_voices {
            let (got, got_centroid) = store.get_voice_profile(&person_id).unwrap().expect(&name);
            // Centroid round-trip: f32 storage vs f64 JSON — compare at f32.
            let want_centroid: Vec<f32> = profile["centroid"]
                .as_array()
                .unwrap()
                .iter()
                .map(|v| v.as_f64().unwrap() as f32)
                .collect();
            assert_eq!(got_centroid, want_centroid, "{name}: centroid bit-exact");
            let mut want_rest = profile.clone();
            want_rest.as_object_mut().unwrap().remove("centroid");
            assert_eq!(
                serde_json::to_value(&got).unwrap(),
                want_rest,
                "{name}: profile payload"
            );
        }

        let expected_prefs = {
            let conn = rusqlite::Connection::open(&legacy).unwrap();
            let mut stmt = conn.prepare("SELECT key,value FROM preferences").unwrap();
            stmt.query_map([], |row| {
                Ok((row.get::<_, String>(0)?, row.get::<_, String>(1)?))
            })
            .unwrap()
            .collect::<Result<Vec<_>, _>>()
            .unwrap()
        };
        let mut got_prefs = store.list_preferences().unwrap();
        let mut want_prefs = expected_prefs;
        got_prefs.sort();
        want_prefs.sort();
        assert_eq!(got_prefs, want_prefs, "{name}: preferences");

        // Idempotent second run.
        assert!(
            migrate(&home.0, NOW).unwrap().is_none(),
            "{name}: not idempotent"
        );
        assert_eq!(sha256_file(&legacy), before, "{name}: legacy mutated (2nd)");
    }
}

#[test]
fn active_run_recovers_as_failed() {
    let fixture = fixtures()
        .into_iter()
        .find(|p| p.file_stem().unwrap() == "active-run")
        .expect("active-run fixture");
    let home = TempHome::with_fixture(&fixture);
    migrate(&home.0, NOW).unwrap().unwrap();
    let store = Store::open(&home.0).unwrap();
    let jobs = store.summaries().unwrap();
    for job in &jobs {
        let job = serde_json::to_value(job).unwrap();
        assert_eq!(job["status"], json!("failed"));
        assert_eq!(
            job["message"],
            json!("The app stopped during processing. Retry this job.")
        );
        assert_eq!(job["updated"], json!(NOW));
    }
}

#[test]
fn corrupt_payload_leaves_no_database() {
    let fixture = fixtures()
        .into_iter()
        .find(|p| p.file_stem().unwrap() == "import-ready")
        .expect("import-ready fixture");
    let home = TempHome::with_fixture(&fixture);
    let legacy = home.0.join("recordings/jobs.sqlite");
    {
        // The legacy triggers validate JSON on write, so inject the failure
        // with a valid-JSON payload whose id violates the new CHECK.
        let conn = rusqlite::Connection::open(&legacy).unwrap();
        conn.execute(
            "INSERT INTO jobs VALUES ('short', '{\"id\":\"short\",\"kind\":\"import\",\"name\":\"x\",\"status\":\"ready\",\"created\":1.0,\"updated\":1.0,\"revision\":0}')",
            [],
        )
        .unwrap();
    }
    let err = migrate(&home.0, NOW).unwrap_err();
    assert!(
        err.to_string().contains("jobs.sqlite"),
        "error names the legacy path: {err}"
    );
    assert!(!home.0.join("speakerdesk.sqlite").exists());
    assert!(!home.0.join("speakerdesk.sqlite.partial").exists());
}

#[test]
fn no_legacy_database_is_noop() {
    let dir =
        std::env::temp_dir().join(format!("speakerdesk-migrate-empty-{}", std::process::id()));
    let _ = fs::remove_dir_all(&dir);
    fs::create_dir_all(&dir).unwrap();
    assert!(migrate(&dir, NOW).unwrap().is_none());
    fs::remove_dir_all(&dir).unwrap();
}
