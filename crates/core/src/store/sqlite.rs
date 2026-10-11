//! Rusqlite `JobStore` for `speakerdesk.sqlite`.
//!
//! Indexed columns are the job projection. The transcript lives in `document`.
//! Other product fields (assignments, recovery, suggestions) live in `meeting`.
//! Python `PRIVATE_FIELDS` live in `private` and are omitted from metadata reads.

use std::path::Path;
use std::time::{SystemTime, UNIX_EPOCH};

use rusqlite::{Connection, OptionalExtension, TransactionBehavior, params, types::Type};
use serde_json::{Map, Value};

use super::{JobStore, apply_pragmas, migrations, validate_checkpoint_duration};
use crate::document::{Document, Job, JobKind, JobSummary};
use crate::error::StoreError;
use crate::ids::JobId;
use crate::languages;
use crate::messages;

const PRIVATE_FIELDS: &[&str] = &[
    "legacy_media_root",
    "operation_generation",
    "rolling_sources",
    "refinement_history",
    "fast_history",
    "fast_previous_revision",
    "boundary_candidate",
    "boundary_candidates",
    "rolling_inflight",
    "fast_retained_candidate",
];

const JOB_COLUMNS: &str = "\
id, kind, name, status, message, language, source_file, duration, \
checkpoint_duration, revision, created, updated, document, meeting, private";

const INSERT_JOB: &str = "\
INSERT OR REPLACE INTO jobs (
  id, kind, name, status, message, language, source_file, duration,
  checkpoint_duration, revision, created, updated, document, meeting, private
) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, ?12, ?13, ?14, ?15)";

pub struct SqliteJobStore {
    connection: Connection,
}

struct StoredJob {
    job: Job,
    checkpoint: Option<f64>,
}

impl SqliteJobStore {
    pub fn open(path: impl AsRef<Path>) -> Result<Self, StoreError> {
        let mut connection = Connection::open(path)?;
        prepare_connection(&mut connection)?;
        Ok(Self { connection })
    }

    pub fn open_in_memory() -> Result<Self, StoreError> {
        let mut connection = Connection::open_in_memory()?;
        prepare_connection(&mut connection)?;
        Ok(Self { connection })
    }

    pub fn insert_uploads(
        &mut self,
        jobs: &[Job],
        active: &[&str],
        limit: u64,
    ) -> Result<bool, StoreError> {
        let tx = self
            .connection
            .transaction_with_behavior(TransactionBehavior::Immediate)?;
        let statuses =
            serde_json::to_string(active).map_err(|e| StoreError::message(e.to_string()))?;
        let count: i64 = tx.query_row(
            "SELECT count(*) FROM jobs WHERE status IN (SELECT value FROM json_each(?1))",
            [statuses],
            |row| row.get(0),
        )?;
        if (count as u64).saturating_add(jobs.len() as u64) > limit {
            return Ok(false);
        }
        for job in jobs {
            insert_job(&tx, job, None)?;
        }
        if let Some(job) = jobs.last() {
            save_default_language(&tx, &job.language)?;
        }
        tx.commit()?;
        Ok(true)
    }

    pub fn delete(&mut self, id: &JobId) -> Result<bool, StoreError> {
        Ok(self
            .connection
            .execute("DELETE FROM jobs WHERE id = ?1", [id.as_str()])?
            != 0)
    }

    /// The callback only transforms this document; no I/O or model work may run here.
    pub fn update_job<R, E: From<StoreError>>(
        &mut self,
        id: &JobId,
        update: impl FnOnce(&mut Job) -> Result<R, E>,
    ) -> Result<Option<(Job, R)>, E> {
        let tx = self
            .connection
            .transaction_with_behavior(TransactionBehavior::Immediate)
            .map_err(StoreError::from)?;
        let Some(stored) = fetch_job(&tx, id.as_str())? else {
            return Ok(None);
        };
        let checkpoint = stored.checkpoint;
        let mut job = for_view(stored, false);
        let revision = job.revision;
        let result = update(&mut job)?;
        job.revision = revision
            .checked_add(1)
            .ok_or_else(|| StoreError::message("Revision overflow"))?;
        job.updated = unix_time();
        insert_job(&tx, &job, checkpoint)?;
        tx.commit().map_err(StoreError::from)?;
        Ok(Some((job, result)))
    }

    pub fn people(&self) -> Result<crate::people::People, StoreError> {
        load_people(&self.connection)
    }

    pub fn edit_person<R, E: From<StoreError>>(
        &mut self,
        edit: impl FnOnce(&mut crate::people::People) -> Result<(crate::people::Person, R), E>,
    ) -> Result<R, E> {
        let tx = self
            .connection
            .transaction_with_behavior(TransactionBehavior::Immediate)
            .map_err(StoreError::from)?;
        let (person, result) = edit(&mut load_people(&tx)?)?;
        tx.execute("INSERT INTO people(id,name,created,updated) VALUES(?1,?2,?3,?4) ON CONFLICT(id) DO UPDATE SET name=excluded.name,updated=excluded.updated",
            params![person.id,person.name,person.created,person.updated.unwrap_or(person.created)])
            .map_err(StoreError::from)?;
        tx.commit().map_err(StoreError::from)?;
        Ok(result)
    }

    pub fn voice_versions(&self) -> Result<std::collections::BTreeMap<String, String>, StoreError> {
        self.connection
            .prepare("SELECT person_id, CAST(profile_version AS TEXT) FROM voice_profiles")?
            .query_map([], |row| Ok((row.get(0)?, row.get(1)?)))?
            .collect::<Result<_, _>>()
            .map_err(StoreError::from)
    }

    pub fn forget_voice(&mut self, person_id: &str) -> Result<(), StoreError> {
        self.connection
            .execute("DELETE FROM voice_profiles WHERE person_id=?1", [person_id])?;
        Ok(())
    }

    pub fn preference(&self, key: &str) -> Result<Option<String>, StoreError> {
        self.connection
            .query_row("SELECT value FROM preferences WHERE key=?1", [key], |row| {
                row.get(0)
            })
            .optional()
            .map_err(StoreError::from)
    }

    pub fn set_preference(&mut self, key: &str, value: &str) -> Result<(), StoreError> {
        self.connection.execute("INSERT INTO preferences(key,value) VALUES(?1,?2) ON CONFLICT(key) DO UPDATE SET value=excluded.value", [key, value])?;
        Ok(())
    }

    pub fn set_recognition(&mut self, enabled: bool) -> Result<(), StoreError> {
        let tx = self
            .connection
            .transaction_with_behavior(TransactionBehavior::Immediate)?;
        tx.execute("INSERT INTO preferences(key,value) VALUES('voice_recognition',?1) ON CONFLICT(key) DO UPDATE SET value=excluded.value", [enabled.to_string()])?;
        if !enabled {
            // Only jobs with an outstanding check need to be deserialized.
            let ids: Vec<String> = tx.prepare("SELECT id FROM jobs WHERE EXISTS (SELECT 1 FROM json_each(json_extract(meeting, '$.voice_checks')) WHERE json_extract(value, '$.status')='checking')")?
                .query_map([], |row| row.get(0))?.collect::<Result<_,_>>()?;
            for id in ids {
                let Some(mut stored) = fetch_job(&tx, &id)? else {
                    continue;
                };
                if let Some(checks) = stored
                    .job
                    .extra
                    .get_mut("voice_checks")
                    .and_then(Value::as_object_mut)
                {
                    for check in checks.values_mut() {
                        if check.get("status").and_then(Value::as_str) == Some("checking") {
                            check["status"] = Value::String("cancelled".into());
                        }
                    }
                }
                stored.job.revision += 1;
                stored.job.updated = unix_time();
                insert_job(&tx, &stored.job, stored.checkpoint)?;
            }
        }
        tx.commit()?;
        Ok(())
    }
}

fn load_people(connection: &Connection) -> Result<crate::people::People, StoreError> {
    let entries = connection
        .prepare("SELECT id,name,created,updated FROM people")?
        .query_map([], |row| {
            Ok(crate::people::Person {
                id: row.get(0)?,
                name: row.get(1)?,
                created: row.get(2)?,
                updated: row.get(3)?,
            })
        })?
        .collect::<Result<Vec<_>, _>>()?;
    Ok(crate::people::People::from_stored(entries))
}

pub(crate) fn prepare_connection(connection: &mut Connection) -> Result<(), StoreError> {
    apply_pragmas(connection)?;
    migrations().to_latest(connection)?;
    Ok(())
}

pub(crate) fn unix_time() -> f64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|duration| duration.as_secs_f64())
        .unwrap_or(0.0)
}

impl JobStore for SqliteJobStore {
    fn initialize(&mut self) -> Result<(), StoreError> {
        prepare_connection(&mut self.connection)
    }

    fn get(&self, id: &JobId, metadata: bool) -> Result<Option<Job>, StoreError> {
        let Some(stored) = fetch_job(&self.connection, id.as_str())? else {
            return Ok(None);
        };
        Ok(Some(for_view(stored, metadata)))
    }

    fn summaries(&self) -> Result<Vec<JobSummary>, StoreError> {
        let mut statement = self.connection.prepare(
            "SELECT id, name, kind, source_file, created, updated, status, message, language, \
             duration, revision, checkpoint_duration FROM jobs ORDER BY created DESC",
        )?;
        let rows = statement.query_map([], summary_from_row)?;
        rows.collect::<Result<Vec<_>, _>>()
            .map_err(StoreError::from)
    }

    fn count_status(&self, statuses: &[&str]) -> Result<u64, StoreError> {
        if statuses.is_empty() {
            return Ok(0);
        }
        let mut sql = String::from("SELECT COUNT(*) FROM jobs WHERE status IN (");
        for index in 0..statuses.len() {
            if index > 0 {
                sql.push(',');
            }
            sql.push('?');
        }
        sql.push(')');
        let count: i64 = self.connection.query_row(
            &sql,
            rusqlite::params_from_iter(statuses.iter().copied()),
            |row| row.get(0),
        )?;
        u64::try_from(count).map_err(|err| StoreError::message(err.to_string()))
    }

    fn put(&mut self, job: &Job, default_language: Option<&str>) -> Result<(), StoreError> {
        let transaction = self
            .connection
            .transaction_with_behavior(TransactionBehavior::Immediate)?;
        let checkpoint = checkpoint_of(&transaction, job.id.as_str())?;
        let mut stored = job.clone();
        if let Some(checkpoint) = checkpoint {
            stored.duration = Some(stored.duration.unwrap_or(0.0).max(checkpoint));
        }
        stored.updated = unix_time();
        insert_job(&transaction, &stored, None)?;
        if let Some(language) = default_language {
            save_default_language(&transaction, language)?;
        }
        transaction.commit()?;
        Ok(())
    }

    fn checkpoint_duration(&mut self, id: &JobId, duration: f64) -> Result<bool, StoreError> {
        validate_checkpoint_duration(duration)?;
        let transaction = self
            .connection
            .transaction_with_behavior(TransactionBehavior::Immediate)?;
        let row = transaction
            .query_row(
                "SELECT duration, checkpoint_duration FROM jobs WHERE id = ?1",
                [id.as_str()],
                |row| Ok((row.get::<_, Option<f64>>(0)?, row.get::<_, Option<f64>>(1)?)),
            )
            .optional()?;
        let Some((stored_duration, checkpoint)) = row else {
            return Ok(false);
        };
        let current = stored_duration
            .unwrap_or(0.0)
            .max(checkpoint.unwrap_or(0.0));
        if duration > current {
            transaction.execute(
                "UPDATE jobs SET checkpoint_duration = ?1, updated = ?2 WHERE id = ?3",
                params![duration, unix_time(), id.as_str()],
            )?;
        }
        transaction.commit()?;
        Ok(true)
    }

    fn recover(&mut self, active: &[&str]) -> Result<(), StoreError> {
        let transaction = self
            .connection
            .transaction_with_behavior(TransactionBehavior::Immediate)?;
        let jobs = fetch_all(&transaction)?;
        for stored in jobs {
            if let Some(job) = recover_one(stored, active) {
                insert_job(&transaction, &job, None)?;
            }
        }
        transaction.commit()?;
        Ok(())
    }
}

pub(crate) fn insert_job(
    connection: &Connection,
    job: &Job,
    checkpoint: Option<f64>,
) -> Result<(), StoreError> {
    let (meeting, private) = split_extra(&job.extra)?;
    let document = document_json(&job.document)?;
    connection.execute(
        INSERT_JOB,
        params![
            job.id.as_str(),
            job.kind.as_str(),
            job.name,
            job.status,
            job.message,
            job.language,
            job.source_file,
            job.duration,
            checkpoint,
            job.revision,
            job.created,
            job.updated,
            document,
            meeting,
            private,
        ],
    )?;
    Ok(())
}

fn save_default_language(connection: &Connection, language: &str) -> Result<(), StoreError> {
    if language != languages::AUTOMATIC && !languages::is_supported(language) {
        return Err(StoreError::message(messages::CHOOSE_LANGUAGE_MODE));
    }
    connection.execute(
        "INSERT OR REPLACE INTO preferences (key, value) VALUES ('default_language', ?1)",
        [language],
    )?;
    Ok(())
}

fn checkpoint_of(connection: &Connection, id: &str) -> Result<Option<f64>, StoreError> {
    let value = connection
        .query_row(
            "SELECT checkpoint_duration FROM jobs WHERE id = ?1",
            [id],
            |row| row.get::<_, Option<f64>>(0),
        )
        .optional()?;
    Ok(value.flatten())
}

fn fetch_job(connection: &Connection, id: &str) -> Result<Option<StoredJob>, StoreError> {
    let mut statement =
        connection.prepare(&format!("SELECT {JOB_COLUMNS} FROM jobs WHERE id = ?1"))?;
    let mut rows = statement.query([id])?;
    let Some(row) = rows.next()? else {
        return Ok(None);
    };
    Ok(Some(stored_from_row(row)?))
}

fn fetch_all(connection: &Connection) -> Result<Vec<StoredJob>, StoreError> {
    let mut statement = connection.prepare(&format!("SELECT {JOB_COLUMNS} FROM jobs"))?;
    let mut rows = statement.query([])?;
    let mut jobs = Vec::new();
    while let Some(row) = rows.next()? {
        jobs.push(stored_from_row(row)?);
    }
    Ok(jobs)
}

fn stored_from_row(row: &rusqlite::Row<'_>) -> Result<StoredJob, rusqlite::Error> {
    let id: String = row.get("id")?;
    let kind: String = row.get("kind")?;
    let document: Option<String> = row.get("document")?;
    let meeting: Option<String> = row.get("meeting")?;
    let private_json: Option<String> = row.get("private")?;
    let mut extra = Map::new();
    if let Some(meeting) = meeting.as_deref() {
        extra.extend(parse_map(meeting)?);
    }
    if let Some(private_json) = private_json.as_deref() {
        extra.extend(parse_map(private_json)?);
    }
    let job = Job {
        id: parse_job_id(&id)?,
        name: row.get("name")?,
        kind: parse_kind(&kind)?,
        status: row.get("status")?,
        message: row.get("message")?,
        language: row.get("language")?,
        source_file: row.get("source_file")?,
        duration: row.get("duration")?,
        revision: row.get("revision")?,
        created: row.get("created")?,
        updated: row.get("updated")?,
        document: document.as_deref().map(parse_document).transpose()?,
        extra,
    };
    Ok(StoredJob {
        checkpoint: row.get("checkpoint_duration")?,
        job,
    })
}

fn summary_from_row(row: &rusqlite::Row<'_>) -> Result<JobSummary, rusqlite::Error> {
    let id: String = row.get("id")?;
    let kind: String = row.get("kind")?;
    let duration: Option<f64> = row.get("duration")?;
    let checkpoint: Option<f64> = row.get("checkpoint_duration")?;
    Ok(JobSummary {
        id: parse_job_id(&id)?,
        name: row.get("name")?,
        kind: Some(parse_kind(&kind)?),
        source_file: row.get("source_file")?,
        created: row.get("created")?,
        updated: row.get("updated")?,
        status: row.get("status")?,
        message: Some(row.get("message")?),
        language: Some(row.get("language")?),
        duration: overlay_duration(duration, checkpoint),
        revision: Some(row.get("revision")?),
    })
}

fn for_view(stored: StoredJob, metadata: bool) -> Job {
    let mut job = overlay_checkpoint(stored);
    if metadata {
        job.document = None;
        for key in PRIVATE_FIELDS {
            job.extra.remove(*key);
        }
    }
    job
}

fn overlay_checkpoint(mut stored: StoredJob) -> Job {
    stored.job.duration = overlay_duration(stored.job.duration, stored.checkpoint);
    stored.job
}

fn overlay_duration(duration: Option<f64>, checkpoint: Option<f64>) -> Option<f64> {
    match checkpoint {
        Some(checkpoint) => Some(duration.unwrap_or(0.0).max(checkpoint)),
        None => duration,
    }
}

fn recover_one(stored: StoredJob, active: &[&str]) -> Option<Job> {
    let mut changed = stored.checkpoint.is_some();
    let mut job = overlay_checkpoint(stored);
    if active.contains(&job.status.as_str()) {
        job.status = "failed".to_string();
        job.message = match job.kind {
            JobKind::Meeting => messages::MEETING_INTERRUPTED.to_string(),
            JobKind::Import => messages::IMPORT_INTERRUPTED.to_string(),
        };
        job.updated = unix_time();
        changed = true;
    }
    if pause_rolling_refinement(&mut job.extra) {
        changed = true;
    }
    changed.then_some(job)
}

fn pause_rolling_refinement(extra: &mut Map<String, Value>) -> bool {
    let Some(plan) = extra.get("rolling_refinement") else {
        return false;
    };
    if !json_truthy(plan) {
        return false;
    }
    let status = extra.get("refinement_status").and_then(Value::as_str);
    if !matches!(status, Some("waiting") | Some("refining")) {
        return false;
    }
    queue_inflight_windows(extra);
    extra.insert(
        "refinement_status".to_string(),
        Value::String("paused".to_string()),
    );
    extra.remove("rolling_inflight");
    true
}

fn queue_inflight_windows(extra: &mut Map<String, Value>) {
    let Some(pending) = extra
        .get_mut("rolling_refinement")
        .and_then(Value::as_object_mut)
        .and_then(|plan| plan.get_mut("pending"))
        .and_then(Value::as_array_mut)
    else {
        return;
    };
    for window in pending.iter_mut() {
        let Some(object) = window.as_object_mut() else {
            continue;
        };
        if object.get("status").and_then(Value::as_str) == Some("in_flight") {
            object.insert("status".to_string(), Value::String("queued".to_string()));
        }
    }
}

fn json_truthy(value: &Value) -> bool {
    match value {
        Value::Null | Value::Bool(false) => false,
        Value::Bool(true) | Value::Object(_) => true,
        Value::Number(number) => number.as_f64().is_some_and(|value| value != 0.0),
        Value::String(text) => !text.is_empty(),
        Value::Array(items) => !items.is_empty(),
    }
}

fn split_extra(extra: &Map<String, Value>) -> Result<(Option<String>, Option<String>), StoreError> {
    let mut meeting = Map::new();
    let mut private = Map::new();
    for (key, value) in extra {
        if PRIVATE_FIELDS.contains(&key.as_str()) {
            private.insert(key.clone(), value.clone());
        } else {
            meeting.insert(key.clone(), value.clone());
        }
    }
    Ok((object_json(&meeting)?, object_json(&private)?))
}

fn object_json(map: &Map<String, Value>) -> Result<Option<String>, StoreError> {
    if map.is_empty() {
        Ok(None)
    } else {
        serde_json::to_string(map)
            .map(Some)
            .map_err(|err| StoreError::message(err.to_string()))
    }
}

fn document_json(document: &Option<Document>) -> Result<Option<String>, StoreError> {
    match document {
        Some(document) => serde_json::to_string(document)
            .map(Some)
            .map_err(|err| StoreError::message(err.to_string())),
        None => Ok(None),
    }
}

fn parse_map(text: &str) -> Result<Map<String, Value>, rusqlite::Error> {
    serde_json::from_str(text).map_err(|err| sql_conversion(err.to_string()))
}

fn parse_document(text: &str) -> Result<Document, rusqlite::Error> {
    serde_json::from_str(text).map_err(|err| sql_conversion(err.to_string()))
}

fn parse_kind(kind: &str) -> Result<JobKind, rusqlite::Error> {
    match kind {
        "import" => Ok(JobKind::Import),
        "meeting" => Ok(JobKind::Meeting),
        _ => Err(sql_conversion(format!("Unknown job kind: {kind}"))),
    }
}

fn parse_job_id(id: &str) -> Result<JobId, rusqlite::Error> {
    JobId::parse(id).map_err(|err| sql_conversion(err.to_string()))
}

fn sql_conversion(message: String) -> rusqlite::Error {
    rusqlite::Error::FromSqlConversionFailure(
        0,
        Type::Text,
        Box::new(std::io::Error::other(message)),
    )
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;

    use serde_json::json;

    use super::SqliteJobStore;
    use crate::document::{Document, Job, JobKind};
    use crate::ids::JobId;
    use crate::messages;
    use crate::store::JobStore;

    const IMPORT_ID: &str = "0123456789abcdef0123456789abcdef";
    const MEETING_ID: &str = "11111111111111111111111111111111";

    fn job(id: &str, kind: JobKind, status: &str, created: f64) -> Job {
        Job {
            id: JobId::parse(id).unwrap(),
            name: "Standup".to_string(),
            kind,
            status: status.to_string(),
            message: String::new(),
            language: "en".to_string(),
            source_file: Some("audio.wav".to_string()),
            duration: Some(1.0),
            revision: 3,
            created,
            updated: created,
            document: Some(Document {
                schema_version: 1,
                speakers: BTreeMap::from([("speaker_0".to_string(), "Ada".to_string())]),
                segments: Vec::new(),
                provenance: None,
                warnings: vec!["Keep".to_string()],
                diarization: Vec::new(),
                edited_at: None,
                rest: serde_json::Map::new(),
            }),
            extra: serde_json::Map::from_iter([
                (
                    "speaker_assignments".to_string(),
                    json!({"speaker_0": {"name": "Ada"}}),
                ),
                ("rolling_inflight".to_string(), json!({"id": "window"})),
            ]),
        }
    }

    fn open() -> SqliteJobStore {
        let mut store = SqliteJobStore::open_in_memory().unwrap();
        store.initialize().unwrap();
        store
    }

    #[test]
    fn put_get_round_trips_document_and_private_fields() {
        let mut store = open();
        let job = job(IMPORT_ID, JobKind::Import, "ready", 1.0);
        store.put(&job, None).unwrap();
        let loaded = store.get(&job.id, false).unwrap().unwrap();
        assert!(loaded.updated > job.updated);
        assert_eq!(loaded.document, job.document);
        assert_eq!(loaded.extra, job.extra);
        assert_eq!(loaded.duration, Some(1.0));
        assert_eq!(loaded.kind, JobKind::Import);

        let metadata = store.get(&job.id, true).unwrap().unwrap();
        assert!(metadata.document.is_none());
        assert!(metadata.extra.get("rolling_inflight").is_none());
        assert!(metadata.extra.get("speaker_assignments").is_some());
    }

    #[test]
    fn summaries_are_newest_first_and_count_status_filters() {
        let mut store = open();
        store
            .put(&job(IMPORT_ID, JobKind::Import, "ready", 1.0), None)
            .unwrap();
        store
            .put(&job(MEETING_ID, JobKind::Meeting, "processing", 5.0), None)
            .unwrap();
        let summaries = store.summaries().unwrap();
        assert_eq!(summaries.len(), 2);
        assert_eq!(summaries[0].id.as_str(), MEETING_ID);
        assert_eq!(summaries[1].id.as_str(), IMPORT_ID);
        assert_eq!(summaries[0].duration, Some(1.0));
        assert_eq!(store.count_status(&["processing"]).unwrap(), 1);
        assert_eq!(store.count_status(&["ready", "processing"]).unwrap(), 2);
        assert_eq!(store.count_status(&[]).unwrap(), 0);
        assert_eq!(store.count_status(&["missing"]).unwrap(), 0);
    }

    #[test]
    fn checkpoint_duration_advances_only_when_greater_and_put_folds_it() {
        let mut store = open();
        let job = job(IMPORT_ID, JobKind::Import, "recording", 1.0);
        let missing = JobId::parse(MEETING_ID).unwrap();
        assert!(!store.checkpoint_duration(&missing, 1.0).unwrap());
        let invalid = store.checkpoint_duration(&job.id, f64::NAN).unwrap_err();
        assert_eq!(invalid.to_string(), messages::CAPTURE_DURATION_INVALID);

        store.put(&job, None).unwrap();
        assert!(store.checkpoint_duration(&job.id, 4.0).unwrap());
        assert_eq!(
            store.get(&job.id, false).unwrap().unwrap().duration,
            Some(4.0)
        );
        let updated = store.get(&job.id, false).unwrap().unwrap().updated;
        assert!(store.checkpoint_duration(&job.id, 4.0).unwrap());
        assert!(store.checkpoint_duration(&job.id, 3.5).unwrap());
        let current = store.get(&job.id, false).unwrap().unwrap();
        assert_eq!(current.duration, Some(4.0));
        assert_eq!(current.updated, updated);
        assert_eq!(store.summaries().unwrap()[0].duration, Some(4.0));

        let mut replacement = job.clone();
        replacement.duration = Some(2.0);
        store.put(&replacement, None).unwrap();
        let folded = store.get(&job.id, false).unwrap().unwrap();
        assert_eq!(folded.duration, Some(4.0));
        let checkpoint: Option<f64> = store
            .connection
            .query_row(
                "SELECT checkpoint_duration FROM jobs WHERE id = ?1",
                [IMPORT_ID],
                |row| row.get(0),
            )
            .unwrap();
        assert_eq!(checkpoint, None);
    }

    #[test]
    fn invalid_default_language_rolls_back_the_job() {
        let mut store = open();
        let job = job(IMPORT_ID, JobKind::Import, "ready", 1.0);
        let err = store.put(&job, Some("latin")).unwrap_err();
        assert_eq!(err.to_string(), messages::CHOOSE_LANGUAGE_MODE);
        assert!(store.get(&job.id, false).unwrap().is_none());

        store.put(&job, Some("auto")).unwrap();
        let value: String = store
            .connection
            .query_row(
                "SELECT value FROM preferences WHERE key = 'default_language'",
                [],
                |row| row.get(0),
            )
            .unwrap();
        assert_eq!(value, "auto");
    }

    #[test]
    fn recover_fails_active_jobs_and_pauses_rolling_refinement() {
        let mut store = open();
        let mut meeting = job(MEETING_ID, JobKind::Meeting, "recording", 2.0);
        meeting.extra.insert(
            "rolling_refinement".to_string(),
            json!({"version": 1, "pending": [{"id": "w", "status": "in_flight"}]}),
        );
        meeting
            .extra
            .insert("refinement_status".to_string(), json!("refining"));
        store.put(&meeting, None).unwrap();

        let mut import = job(IMPORT_ID, JobKind::Import, "processing", 1.0);
        import
            .extra
            .insert("refinement_status".to_string(), json!("waiting"));
        store.put(&import, None).unwrap();

        let ready_id = "22222222222222222222222222222222";
        let mut ready = job(ready_id, JobKind::Import, "ready", 3.0);
        ready.extra.insert(
            "rolling_refinement".to_string(),
            json!({"pending": [{"status": "in_flight"}]}),
        );
        ready
            .extra
            .insert("refinement_status".to_string(), json!("waiting"));
        store.put(&ready, None).unwrap();

        let queued_id = "33333333333333333333333333333333";
        store
            .put(&job(queued_id, JobKind::Import, "queued", 4.0), None)
            .unwrap();
        store
            .connection
            .execute("UPDATE jobs SET updated = 12.5", [])
            .unwrap();

        store.recover(&["recording", "processing"]).unwrap();

        let meeting = store
            .get(&JobId::parse(MEETING_ID).unwrap(), false)
            .unwrap()
            .unwrap();
        assert_eq!(meeting.status, "failed");
        assert_eq!(meeting.message, messages::MEETING_INTERRUPTED);
        assert!(meeting.updated > 12.5);
        assert_eq!(meeting.extra.get("refinement_status").unwrap(), "paused");
        assert!(meeting.extra.get("rolling_inflight").is_none());
        assert_eq!(
            meeting.extra["rolling_refinement"]["pending"][0]["status"],
            "queued"
        );

        let import = store
            .get(&JobId::parse(IMPORT_ID).unwrap(), false)
            .unwrap()
            .unwrap();
        assert_eq!(import.status, "failed");
        assert_eq!(import.message, messages::IMPORT_INTERRUPTED);
        assert_eq!(import.extra.get("refinement_status").unwrap(), "waiting");

        let ready = store
            .get(&JobId::parse(ready_id).unwrap(), false)
            .unwrap()
            .unwrap();
        assert_eq!(ready.status, "ready");
        assert_eq!(ready.updated, 12.5);
        assert_eq!(ready.extra.get("refinement_status").unwrap(), "paused");
        assert_eq!(
            ready.extra["rolling_refinement"]["pending"][0]["status"],
            "queued"
        );

        let queued = store
            .get(&JobId::parse(queued_id).unwrap(), false)
            .unwrap()
            .unwrap();
        assert_eq!(queued.status, "queued");
        assert_eq!(queued.message, "");
    }

    #[test]
    fn recover_folds_checkpoint_duration_into_the_job() {
        let mut store = open();
        let job = job(IMPORT_ID, JobKind::Import, "ready", 1.0);
        store.put(&job, None).unwrap();
        assert!(store.checkpoint_duration(&job.id, 6.5).unwrap());
        store.recover(&[]).unwrap();
        let loaded = store.get(&job.id, false).unwrap().unwrap();
        assert_eq!(loaded.status, "ready");
        assert_eq!(loaded.duration, Some(6.5));
        let checkpoint: Option<f64> = store
            .connection
            .query_row(
                "SELECT checkpoint_duration FROM jobs WHERE id = ?1",
                [IMPORT_ID],
                |row| row.get(0),
            )
            .unwrap();
        assert_eq!(checkpoint, None);
    }
}
