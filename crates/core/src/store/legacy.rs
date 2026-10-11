use std::collections::HashMap;
use std::path::{Path, PathBuf};

use rusqlite::{Connection, OpenFlags, params};
use serde_json::{Map, Value};
use sha2::{Digest, Sha256};

use super::sqlite::{insert_job, prepare_connection, unix_time};
use crate::document::Job;
use crate::error::StoreError;

/// One-way import into a new database. The legacy file is opened read-only.
/// A failed import deletes only the `*.partial` file and its sidecars.
pub fn import_legacy(legacy_db: &Path, destination: &Path) -> Result<(), StoreError> {
    if destination.exists() {
        return Err(StoreError::message(format!(
            "Destination already exists: {}",
            destination.display()
        )));
    }
    let dump = read_legacy(legacy_db)?;
    let partial = partial_path(destination)?;
    remove_sqlite_family(&partial);
    if let Err(err) =
        write_dump(&partial, legacy_db, &dump).and_then(|()| publish(&partial, destination))
    {
        remove_sqlite_family(&partial);
        return Err(err);
    }
    Ok(())
}

struct LegacyDump {
    jobs: Vec<Job>,
    people: Vec<PersonRow>,
    preferences: Vec<(String, String)>,
    voices: Vec<VoiceRow>,
    sha256: String,
}

struct LegacySchema {
    people: bool,
    preferences: bool,
    voices: bool,
    progress: bool,
    progress_updated: bool,
}

struct PersonRow {
    id: String,
    name: String,
    created: f64,
    updated: f64,
}

struct VoiceRow {
    person_id: String,
    model_id: String,
    revision: String,
    artifact_sha256: String,
    dimension: i64,
    centroid: Vec<u8>,
    profile_version: i64,
    consented_at: Option<f64>,
    enrollment: String,
}

struct Progress {
    duration: f64,
    updated: Option<f64>,
}

fn read_legacy(path: &Path) -> Result<LegacyDump, StoreError> {
    let sha256 = file_sha256(path)?;
    let connection = open_readonly(path)?;
    let schema = inspect_schema(&connection, path)?;
    let progress = if schema.progress {
        read_progress(&connection, path, schema.progress_updated)?
    } else {
        HashMap::new()
    };
    Ok(LegacyDump {
        jobs: read_jobs(&connection, path, &progress)?,
        people: if schema.people {
            read_people(&connection, path)?
        } else {
            Vec::new()
        },
        preferences: if schema.preferences {
            read_preferences(&connection, path)?
        } else {
            Vec::new()
        },
        voices: if schema.voices {
            read_voices(&connection, path)?
        } else {
            Vec::new()
        },
        sha256,
    })
}

/// Opens `path` with the read-only flag only. Do not add a write or create flag.
fn open_readonly(path: &Path) -> Result<Connection, StoreError> {
    let connection = Connection::open_with_flags(path, OpenFlags::SQLITE_OPEN_READ_ONLY)
        .map_err(|err| schema_error(path, &err.to_string()))?;
    connection
        .pragma_update(None, "query_only", "ON")
        .map_err(|err| schema_error(path, &err.to_string()))?;
    Ok(connection)
}

fn inspect_schema(connection: &Connection, path: &Path) -> Result<LegacySchema, StoreError> {
    require_table(connection, path, "jobs", &["id", "payload"], true)?;
    let people = require_table(connection, path, "people", &["id", "payload"], false)?.is_some();
    let preferences =
        require_table(connection, path, "preferences", &["key", "value"], false)?.is_some();
    let voices = require_table(
        connection,
        path,
        "voice_profiles",
        &["person_id", "payload"],
        false,
    )?
    .is_some();
    let progress_columns =
        require_table(connection, path, "job_progress", &["id", "duration"], false)?;
    let progress_updated = progress_columns
        .as_ref()
        .is_some_and(|columns| columns.iter().any(|column| column == "updated"));
    Ok(LegacySchema {
        people,
        preferences,
        voices,
        progress: progress_columns.is_some(),
        progress_updated,
    })
}

fn require_table(
    connection: &Connection,
    path: &Path,
    table: &str,
    required: &[&str],
    mandatory: bool,
) -> Result<Option<Vec<String>>, StoreError> {
    let columns =
        table_columns(connection, table).map_err(|err| schema_error(path, &err.to_string()))?;
    match columns {
        None if mandatory => Err(schema_error(path, &format!("missing {table}"))),
        None => Ok(None),
        Some(columns) if has_columns(&columns, required) => Ok(Some(columns)),
        Some(_) => Err(schema_error(path, &format!("{table} columns"))),
    }
}

fn has_columns(columns: &[String], required: &[&str]) -> bool {
    required
        .iter()
        .all(|name| columns.iter().any(|column| column == name))
}

fn table_columns(
    connection: &Connection,
    table: &str,
) -> Result<Option<Vec<String>>, rusqlite::Error> {
    let count: i64 = connection.query_row(
        "SELECT COUNT(*) FROM sqlite_master WHERE type = 'table' AND name = ?1",
        [table],
        |row| row.get(0),
    )?;
    if count == 0 {
        return Ok(None);
    }
    let mut statement = connection.prepare(&format!("PRAGMA table_info({table})"))?;
    let names = statement
        .query_map([], |row| row.get::<_, String>(1))?
        .collect::<Result<Vec<_>, _>>()?;
    Ok(Some(names))
}

fn read_jobs(
    connection: &Connection,
    path: &Path,
    progress: &HashMap<String, Progress>,
) -> Result<Vec<Job>, StoreError> {
    let mut statement = connection
        .prepare("SELECT id, payload FROM jobs")
        .map_err(|err| schema_error(path, &err.to_string()))?;
    let mut rows = statement
        .query([])
        .map_err(|err| schema_error(path, &err.to_string()))?;
    let mut jobs = Vec::new();
    while let Some(row) = rows
        .next()
        .map_err(|err| schema_error(path, &err.to_string()))?
    {
        let id: String = row
            .get(0)
            .map_err(|err| schema_error(path, &err.to_string()))?;
        let payload: String = row
            .get(1)
            .map_err(|err| schema_error(path, &err.to_string()))?;
        jobs.push(parse_legacy_job(path, &id, &payload, progress.get(&id))?);
    }
    Ok(jobs)
}

fn parse_legacy_job(
    path: &Path,
    id: &str,
    payload: &str,
    progress: Option<&Progress>,
) -> Result<Job, StoreError> {
    let mut job: Job =
        serde_json::from_str(payload).map_err(|_| schema_error(path, "job payload"))?;
    if job.id.as_str() != id {
        return Err(schema_error(path, "job id"));
    }
    if let Some(progress) = progress {
        job.duration = Some(job.duration.unwrap_or(0.0).max(progress.duration));
        if let Some(updated) = progress.updated {
            job.updated = job.updated.max(updated);
        }
    }
    Ok(job)
}

fn read_progress(
    connection: &Connection,
    path: &Path,
    with_updated: bool,
) -> Result<HashMap<String, Progress>, StoreError> {
    let sql = if with_updated {
        "SELECT id, duration, updated FROM job_progress"
    } else {
        "SELECT id, duration, NULL FROM job_progress"
    };
    let mut statement = connection
        .prepare(sql)
        .map_err(|err| schema_error(path, &err.to_string()))?;
    let mut rows = statement
        .query([])
        .map_err(|err| schema_error(path, &err.to_string()))?;
    let mut progress = HashMap::new();
    while let Some(row) = rows
        .next()
        .map_err(|err| schema_error(path, &err.to_string()))?
    {
        let id: String = row
            .get(0)
            .map_err(|err| schema_error(path, &err.to_string()))?;
        let duration: f64 = row
            .get(1)
            .map_err(|err| schema_error(path, &err.to_string()))?;
        let updated: Option<f64> = row
            .get(2)
            .map_err(|err| schema_error(path, &err.to_string()))?;
        progress.insert(id, Progress { duration, updated });
    }
    Ok(progress)
}

fn read_people(connection: &Connection, path: &Path) -> Result<Vec<PersonRow>, StoreError> {
    let mut statement = connection
        .prepare("SELECT id, payload FROM people")
        .map_err(|err| schema_error(path, &err.to_string()))?;
    let mut rows = statement
        .query([])
        .map_err(|err| schema_error(path, &err.to_string()))?;
    let mut people = Vec::new();
    while let Some(row) = rows
        .next()
        .map_err(|err| schema_error(path, &err.to_string()))?
    {
        let id: String = row
            .get(0)
            .map_err(|err| schema_error(path, &err.to_string()))?;
        let payload: String = row
            .get(1)
            .map_err(|err| schema_error(path, &err.to_string()))?;
        people.push(parse_person(path, &id, &payload)?);
    }
    Ok(people)
}

fn parse_person(path: &Path, id: &str, payload: &str) -> Result<PersonRow, StoreError> {
    let value: Value = serde_json::from_str(payload).map_err(|_| schema_error(path, "person"))?;
    let Some(object) = value.as_object() else {
        return Err(schema_error(path, "person"));
    };
    if object
        .get("id")
        .and_then(Value::as_str)
        .is_some_and(|found| found != id)
    {
        return Err(schema_error(path, "person id"));
    }
    let Some(name) = object.get("name").and_then(Value::as_str) else {
        return Err(schema_error(path, "person name"));
    };
    let name = name.trim();
    if name.is_empty() {
        return Err(schema_error(path, "person name"));
    }
    let Some(created) = object
        .get("created")
        .and_then(Value::as_f64)
        .filter(|created| created.is_finite())
    else {
        return Err(schema_error(path, "person"));
    };
    let updated = object
        .get("updated")
        .and_then(Value::as_f64)
        .filter(|updated| updated.is_finite())
        .unwrap_or(created);
    Ok(PersonRow {
        id: id.to_string(),
        name: name.to_string(),
        created,
        updated,
    })
}

fn read_preferences(
    connection: &Connection,
    path: &Path,
) -> Result<Vec<(String, String)>, StoreError> {
    let mut statement = connection
        .prepare("SELECT key, value FROM preferences")
        .map_err(|err| schema_error(path, &err.to_string()))?;
    let mut rows = statement
        .query([])
        .map_err(|err| schema_error(path, &err.to_string()))?;
    let mut preferences = Vec::new();
    while let Some(row) = rows
        .next()
        .map_err(|err| schema_error(path, &err.to_string()))?
    {
        let key: String = row
            .get(0)
            .map_err(|err| schema_error(path, &err.to_string()))?;
        let value: String = row
            .get(1)
            .map_err(|err| schema_error(path, &err.to_string()))?;
        preferences.push((key, value));
    }
    Ok(preferences)
}

fn read_voices(connection: &Connection, path: &Path) -> Result<Vec<VoiceRow>, StoreError> {
    let mut statement = connection
        .prepare("SELECT person_id, payload FROM voice_profiles")
        .map_err(|err| schema_error(path, &err.to_string()))?;
    let mut rows = statement
        .query([])
        .map_err(|err| schema_error(path, &err.to_string()))?;
    let mut voices = Vec::new();
    while let Some(row) = rows
        .next()
        .map_err(|err| schema_error(path, &err.to_string()))?
    {
        let person_id: String = row
            .get(0)
            .map_err(|err| schema_error(path, &err.to_string()))?;
        let payload: String = row
            .get(1)
            .map_err(|err| schema_error(path, &err.to_string()))?;
        voices.push(parse_voice(path, &person_id, &payload)?);
    }
    Ok(voices)
}

fn parse_voice(path: &Path, person_id: &str, payload: &str) -> Result<VoiceRow, StoreError> {
    let value: Value =
        serde_json::from_str(payload).map_err(|_| schema_error(path, "voice profile"))?;
    let Some(object) = value.as_object() else {
        return Err(schema_error(path, "voice profile"));
    };
    if object
        .get("person_id")
        .and_then(Value::as_str)
        .is_some_and(|found| found != person_id)
    {
        return Err(schema_error(path, "voice profile person"));
    }
    let Some(model) = object.get("model").and_then(Value::as_object) else {
        return Err(schema_error(path, "voice model"));
    };
    let Some(model_id) = text_field(model, "model_id") else {
        return Err(schema_error(path, "voice model"));
    };
    let Some(revision) = text_field(model, "revision") else {
        return Err(schema_error(path, "voice model"));
    };
    let Some(artifact_sha256) = text_field(model, "artifact_sha256") else {
        return Err(schema_error(path, "voice model"));
    };
    if !valid_sha256(&artifact_sha256) {
        return Err(schema_error(path, "voice model"));
    }
    let Some(dimension) = object
        .get("model")
        .and_then(|model| model.get("dimension"))
        .and_then(Value::as_i64)
        .filter(|dimension| (1..=4096).contains(dimension))
    else {
        return Err(schema_error(path, "voice model"));
    };
    let Some(centroid) = centroid_bytes(object.get("centroid"), dimension) else {
        return Err(schema_error(path, "voice centroid"));
    };
    Ok(VoiceRow {
        person_id: person_id.to_string(),
        model_id,
        revision,
        artifact_sha256,
        dimension,
        centroid,
        profile_version: object.get("version").and_then(Value::as_i64).unwrap_or(1),
        consented_at: object.get("enrolled_at").and_then(Value::as_f64),
        enrollment: payload.to_string(),
    })
}

fn text_field(object: &Map<String, Value>, key: &str) -> Option<String> {
    let text = object.get(key).and_then(Value::as_str)?;
    if text.is_empty() {
        None
    } else {
        Some(text.to_string())
    }
}

fn valid_sha256(value: &str) -> bool {
    value.len() == 64
        && value
            .bytes()
            .all(|byte| byte.is_ascii_hexdigit() && !byte.is_ascii_uppercase())
}

fn centroid_bytes(value: Option<&Value>, dimension: i64) -> Option<Vec<u8>> {
    let values = value?.as_array()?;
    if values.len() != usize::try_from(dimension).ok()? {
        return None;
    }
    let mut bytes = Vec::with_capacity(values.len() * 8);
    for item in values {
        let sample = item.as_f64()?;
        if !sample.is_finite() {
            return None;
        }
        bytes.extend_from_slice(&sample.to_le_bytes());
    }
    Some(bytes)
}

fn write_dump(partial: &Path, legacy_path: &Path, dump: &LegacyDump) -> Result<(), StoreError> {
    let mut connection = Connection::open(partial)?;
    prepare_connection(&mut connection)?;
    let transaction = connection.transaction()?;
    for job in &dump.jobs {
        insert_job(&transaction, job, None)?;
    }
    for person in &dump.people {
        transaction.execute(
            "INSERT INTO people (id, name, created, updated) VALUES (?1, ?2, ?3, ?4)",
            params![person.id, person.name, person.created, person.updated],
        )?;
    }
    for (key, value) in &dump.preferences {
        transaction.execute(
            "INSERT OR REPLACE INTO preferences (key, value) VALUES (?1, ?2)",
            params![key, value],
        )?;
    }
    for voice in &dump.voices {
        transaction.execute(
            "INSERT INTO voice_profiles (
                person_id, model_id, revision, artifact_sha256, dimension, centroid,
                profile_version, consented_at, enrollment
             ) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9)",
            params![
                voice.person_id,
                voice.model_id,
                voice.revision,
                voice.artifact_sha256,
                voice.dimension,
                voice.centroid,
                voice.profile_version,
                voice.consented_at,
                voice.enrollment,
            ],
        )?;
    }
    transaction.execute(
        "INSERT INTO migrations (
            legacy_path, legacy_sha256, migrated_at, app_version, counts
         ) VALUES (?1, ?2, ?3, ?4, ?5)",
        params![
            legacy_path.display().to_string(),
            dump.sha256,
            unix_time(),
            env!("CARGO_PKG_VERSION"),
            dump.counts_json(),
        ],
    )?;
    transaction.commit()?;
    let busy: i64 =
        connection.query_row("PRAGMA wal_checkpoint(TRUNCATE)", [], |row| row.get(0))?;
    if busy != 0 {
        return Err(StoreError::message(format!(
            "Could not checkpoint imported database: {}",
            partial.display()
        )));
    }
    connection
        .close()
        .map_err(|(_, err)| StoreError::from(err))?;
    Ok(())
}

impl LegacyDump {
    fn counts_json(&self) -> String {
        serde_json::json!({
            "jobs": self.jobs.len(),
            "people": self.people.len(),
            "preferences": self.preferences.len(),
            "voice_profiles": self.voices.len(),
        })
        .to_string()
    }
}

fn publish(partial: &Path, destination: &Path) -> Result<(), StoreError> {
    fsync_file(partial)?;
    remove_sidecars(partial);
    std::fs::rename(partial, destination).map_err(|err| {
        StoreError::message(format!(
            "Could not publish imported database {}: {err}",
            destination.display()
        ))
    })?;
    if let Some(parent) = destination
        .parent()
        .filter(|parent| !parent.as_os_str().is_empty())
    {
        let _ = fsync_dir(parent);
    }
    Ok(())
}

fn file_sha256(path: &Path) -> Result<String, StoreError> {
    let bytes = std::fs::read(path).map_err(|err| schema_error(path, &err.to_string()))?;
    Ok(hex::encode(Sha256::digest(bytes)))
}

fn partial_path(destination: &Path) -> Result<PathBuf, StoreError> {
    let Some(name) = destination.file_name() else {
        return Err(StoreError::message(format!(
            "Destination has no file name: {}",
            destination.display()
        )));
    };
    let mut partial_name = name.to_os_string();
    partial_name.push(".partial");
    Ok(destination.with_file_name(partial_name))
}

fn sidecar(path: &Path, suffix: &str) -> PathBuf {
    let mut name = path.file_name().unwrap_or_default().to_os_string();
    name.push(suffix);
    path.with_file_name(name)
}

fn remove_sidecars(path: &Path) {
    let _ = std::fs::remove_file(sidecar(path, "-wal"));
    let _ = std::fs::remove_file(sidecar(path, "-shm"));
}

fn remove_sqlite_family(path: &Path) {
    let _ = std::fs::remove_file(path);
    remove_sidecars(path);
}

fn fsync_file(path: &Path) -> Result<(), StoreError> {
    let file = std::fs::File::open(path).map_err(io_error)?;
    file.sync_all().map_err(io_error)?;
    Ok(())
}

fn fsync_dir(path: &Path) -> Result<(), StoreError> {
    let file = std::fs::File::open(path).map_err(io_error)?;
    file.sync_all().map_err(io_error)?;
    Ok(())
}

fn io_error(err: std::io::Error) -> StoreError {
    StoreError::message(err.to_string())
}

fn schema_error(path: &Path, detail: &str) -> StoreError {
    StoreError::message(format!(
        "Legacy database schema cannot be imported: {} ({detail})",
        path.display()
    ))
}

#[cfg(test)]
mod tests {
    use std::path::Path;

    use rusqlite::params;
    use serde_json::json;
    use sha2::{Digest, Sha256};

    use super::import_legacy;
    use crate::document::JobKind;
    use crate::ids::JobId;
    use crate::store::JobStore;
    use crate::store::sqlite::SqliteJobStore;

    const JOB_ID: &str = "0123456789abcdef0123456789abcdef";
    const PERSON_ID: &str = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa";

    struct TempDir(std::path::PathBuf);

    impl TempDir {
        fn new(label: &str) -> Self {
            let path = std::env::temp_dir().join(format!(
                "speakerdesk-{label}-{}-{}",
                std::process::id(),
                uuid::Uuid::new_v4()
            ));
            std::fs::create_dir_all(&path).unwrap();
            Self(path)
        }
    }

    impl Drop for TempDir {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.0);
        }
    }

    fn dir_names(dir: &Path) -> Vec<String> {
        let mut names = std::fs::read_dir(dir)
            .unwrap()
            .map(|entry| entry.unwrap().file_name().to_string_lossy().into_owned())
            .collect::<Vec<_>>();
        names.sort();
        names
    }

    #[test]
    fn imports_minimal_legacy_jobs_without_changing_source_bytes() {
        let dir = TempDir::new("legacy-minimal");
        let legacy = dir.0.join("jobs.sqlite");
        let destination = dir.0.join("speakerdesk.sqlite");
        let payload = json!({
            "id": JOB_ID,
            "name": "Clip",
            "status": "ready",
            "message": "done",
            "language": "en",
            "source_file": "clip.wav",
            "duration": 3.5,
            "revision": 2,
            "created": 1.25,
            "updated": 2.5,
            "document": {
                "schema_version": 1,
                "speakers": {"speaker_0": "Speaker 1"},
                "segments": []
            },
            "speaker_assignments": {"speaker_0": {"name": "Ada"}},
            "rolling_inflight": {"id": "window"}
        })
        .to_string();
        let connection = rusqlite::Connection::open(&legacy).unwrap();
        connection
            .execute_batch("CREATE TABLE jobs (id TEXT PRIMARY KEY, payload TEXT NOT NULL)")
            .unwrap();
        connection
            .execute(
                "INSERT INTO jobs (id, payload) VALUES (?1, ?2)",
                params![JOB_ID, payload],
            )
            .unwrap();
        drop(connection);

        let before = std::fs::read(&legacy).unwrap();
        import_legacy(&legacy, &destination).unwrap();
        let after = std::fs::read(&legacy).unwrap();
        assert_eq!(before, after);
        assert_eq!(
            dir_names(&dir.0),
            vec!["jobs.sqlite".to_string(), "speakerdesk.sqlite".to_string()]
        );

        let store = SqliteJobStore::open(&destination).unwrap();
        let job = store
            .get(&JobId::parse(JOB_ID).unwrap(), false)
            .unwrap()
            .unwrap();
        assert_eq!(job.name, "Clip");
        assert_eq!(job.kind, JobKind::Import);
        assert_eq!(job.status, "ready");
        assert_eq!(job.message, "done");
        assert_eq!(job.source_file.as_deref(), Some("clip.wav"));
        assert_eq!(job.duration, Some(3.5));
        assert_eq!(job.updated, 2.5);
        assert_eq!(
            job.document.as_ref().unwrap().speakers["speaker_0"],
            "Speaker 1"
        );
        assert_eq!(job.extra["speaker_assignments"]["speaker_0"]["name"], "Ada");
        assert_eq!(job.extra["rolling_inflight"]["id"], "window");
        let metadata = store
            .get(&JobId::parse(JOB_ID).unwrap(), true)
            .unwrap()
            .unwrap();
        assert!(metadata.document.is_none());
        assert!(metadata.extra.get("rolling_inflight").is_none());
        assert!(metadata.extra.get("speaker_assignments").is_some());
    }

    #[test]
    fn rejects_a_legacy_schema_it_cannot_parse_and_leaves_the_source() {
        let dir = TempDir::new("legacy-schema");
        let legacy = dir.0.join("jobs.sqlite");
        let destination = dir.0.join("speakerdesk.sqlite");
        let connection = rusqlite::Connection::open(&legacy).unwrap();
        connection
            .execute_batch("CREATE TABLE jobs (id TEXT PRIMARY KEY)")
            .unwrap();
        drop(connection);
        let before = std::fs::read(&legacy).unwrap();

        let error = import_legacy(&legacy, &destination).unwrap_err();
        assert!(
            error.to_string().contains(&legacy.display().to_string()),
            "{error}"
        );
        assert_eq!(std::fs::read(&legacy).unwrap(), before);
        assert!(!destination.exists());
        assert_eq!(dir_names(&dir.0), vec!["jobs.sqlite".to_string()]);
    }

    #[test]
    fn deletes_only_the_partial_when_a_payload_cannot_be_parsed() {
        let dir = TempDir::new("legacy-payload");
        let legacy = dir.0.join("jobs.sqlite");
        let destination = dir.0.join("speakerdesk.sqlite");
        let connection = rusqlite::Connection::open(&legacy).unwrap();
        connection
            .execute_batch("CREATE TABLE jobs (id TEXT PRIMARY KEY, payload TEXT NOT NULL)")
            .unwrap();
        connection
            .execute(
                "INSERT INTO jobs (id, payload) VALUES (?1, ?2)",
                params![JOB_ID, "{"],
            )
            .unwrap();
        drop(connection);
        let before = std::fs::read(&legacy).unwrap();

        let error = import_legacy(&legacy, &destination).unwrap_err();
        assert!(
            error.to_string().contains(&legacy.display().to_string()),
            "{error}"
        );
        assert_eq!(std::fs::read(&legacy).unwrap(), before);
        assert!(!destination.exists());
        assert_eq!(dir_names(&dir.0), vec!["jobs.sqlite".to_string()]);
    }

    #[test]
    fn maps_people_preferences_voice_profiles_and_checkpoints() {
        let dir = TempDir::new("legacy-rich");
        let legacy = dir.0.join("jobs.sqlite");
        let destination = dir.0.join("speakerdesk.sqlite");
        let sha = "ab".repeat(32);
        let connection = rusqlite::Connection::open(&legacy).unwrap();
        connection
            .execute_batch(
                "CREATE TABLE jobs (id TEXT PRIMARY KEY, payload TEXT NOT NULL);
                 CREATE TABLE people (id TEXT PRIMARY KEY, payload TEXT NOT NULL);
                 CREATE TABLE preferences (key TEXT PRIMARY KEY, value TEXT NOT NULL);
                 CREATE TABLE voice_profiles (person_id TEXT PRIMARY KEY, payload TEXT NOT NULL);
                 CREATE TABLE job_progress (id TEXT PRIMARY KEY, duration REAL NOT NULL, updated REAL NOT NULL);",
            )
            .unwrap();
        let job = json!({
            "id": JOB_ID,
            "name": "Imported",
            "kind": "meeting",
            "status": "ready",
            "created": 1.0,
            "updated": 2.0,
            "duration": 1.0
        })
        .to_string();
        let person = json!({
            "id": PERSON_ID,
            "name": " Ada ",
            "created": 4.0,
            "updated": 8.0
        })
        .to_string();
        let voice = json!({
            "person_id": PERSON_ID,
            "version": "cccccccccccccccccccccccccccccccc",
            "enrolled_at": 10.5,
            "consent": "explicit_remember_voice",
            "model": {
                "model_id": "redimnet",
                "revision": "rev-1",
                "artifact_sha256": sha,
                "dimension": 2,
                "sample_rate": 16000
            },
            "centroid": [1.0, 0.0],
            "clips": []
        })
        .to_string();
        connection
            .execute(
                "INSERT INTO jobs (id, payload) VALUES (?1, ?2)",
                params![JOB_ID, job],
            )
            .unwrap();
        connection
            .execute(
                "INSERT INTO people (id, payload) VALUES (?1, ?2)",
                params![PERSON_ID, person],
            )
            .unwrap();
        connection
            .execute(
                "INSERT INTO preferences (key, value) VALUES ('default_language', 'auto')",
                [],
            )
            .unwrap();
        connection
            .execute(
                "INSERT INTO voice_profiles (person_id, payload) VALUES (?1, ?2)",
                params![PERSON_ID, voice],
            )
            .unwrap();
        connection
            .execute(
                "INSERT INTO job_progress (id, duration, updated) VALUES (?1, 4.5, 20.0)",
                [JOB_ID],
            )
            .unwrap();
        drop(connection);
        let before = std::fs::read(&legacy).unwrap();

        import_legacy(&legacy, &destination).unwrap();
        assert_eq!(std::fs::read(&legacy).unwrap(), before);

        {
            let store = SqliteJobStore::open(&destination).unwrap();
            let job = store
                .get(&JobId::parse(JOB_ID).unwrap(), false)
                .unwrap()
                .unwrap();
            assert_eq!(job.kind, JobKind::Meeting);
            assert_eq!(job.duration, Some(4.5));
            assert_eq!(job.updated, 20.0);
        }

        let connection = rusqlite::Connection::open(&destination).unwrap();
        let name: String = connection
            .query_row(
                "SELECT name FROM people WHERE id = ?1",
                [PERSON_ID],
                |row| row.get(0),
            )
            .unwrap();
        assert_eq!(name, "Ada");
        let language: String = connection
            .query_row(
                "SELECT value FROM preferences WHERE key = 'default_language'",
                [],
                |row| row.get(0),
            )
            .unwrap();
        assert_eq!(language, "auto");
        let (model_id, dimension, centroid, version, consented_at, enrollment): (
            String,
            i64,
            Vec<u8>,
            i64,
            f64,
            String,
        ) = connection
            .query_row(
                "SELECT model_id, dimension, centroid, profile_version, consented_at, enrollment \
                 FROM voice_profiles WHERE person_id = ?1",
                [PERSON_ID],
                |row| {
                    Ok((
                        row.get(0)?,
                        row.get(1)?,
                        row.get(2)?,
                        row.get(3)?,
                        row.get(4)?,
                        row.get(5)?,
                    ))
                },
            )
            .unwrap();
        assert_eq!(model_id, "redimnet");
        assert_eq!(dimension, 2);
        assert_eq!(version, 1);
        assert_eq!(consented_at, 10.5);
        assert!(enrollment.contains("explicit_remember_voice"));
        let mut expected = Vec::new();
        expected.extend_from_slice(&1.0f64.to_le_bytes());
        expected.extend_from_slice(&0.0f64.to_le_bytes());
        assert_eq!(centroid, expected);

        let (stored_path, digest, counts): (String, String, String) = connection
            .query_row(
                "SELECT legacy_path, legacy_sha256, counts FROM migrations",
                [],
                |row| Ok((row.get(0)?, row.get(1)?, row.get(2)?)),
            )
            .unwrap();
        assert!(stored_path.contains("jobs.sqlite"));
        assert_eq!(digest, hex::encode(Sha256::digest(&before)));
        let counts: serde_json::Value = serde_json::from_str(&counts).unwrap();
        assert_eq!(counts["jobs"], 1);
        assert_eq!(counts["people"], 1);
        assert_eq!(counts["preferences"], 1);
        assert_eq!(counts["voice_profiles"], 1);
    }
}
