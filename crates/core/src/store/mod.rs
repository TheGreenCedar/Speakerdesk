//! `core::store` — the new `speakerdesk.sqlite` job store.
//!
//! Replaces the v0.6.6 `jobs`/`job_metadata`/`job_progress` trio: metadata is
//! columnar, the transcript document is its own column, engine-only state
//! (`private`) is a separate column, and live-duration checkpoints are
//! `checkpoint_*` columns overlaid on read exactly like `job_progress`.
//!
//! Everything in and out is typed: `Job`, `JobSummary`, `Person`,
//! `VoiceProfile`. `serde_json::Value` appears only at the column
//! serialization boundary (the payload is split into `meeting`/`document`/
//! `private` JSON columns and reassembled on read).

pub mod legacy;

use std::fs::OpenOptions;
use std::path::Path;
use std::sync::Mutex;
use std::time::Duration;

use rusqlite::{Connection, OptionalExtension, params};
use rusqlite_migration::{M, Migrations};
use serde_json::{Map, Value, json};

use crate::model::{
    Job, JobKind, JobStatus, JobSummary, PRIVATE_KEYS, Person, RefinementStatus, RollingRefinement,
    RollingWindowStatus, VoiceProfile,
};
use crate::transcript::LANGUAGES;

/// `language_detection.LANGUAGE_CHOICES` membership — 'auto' plus the
/// supported meeting-language vocabulary.
fn language_choice(value: &str) -> bool {
    value == "auto" || LANGUAGES.iter().any(|(code, _)| *code == value)
}

/// Job keys that map to `jobs` columns; everything else product goes into
/// `meeting` JSON.
pub(crate) const COLUMN_KEYS: &[&str] = &[
    "id",
    "kind",
    "name",
    "status",
    "message",
    "language",
    "source_file",
    "duration",
    "revision",
    "created",
    "updated",
];

pub(crate) const SCHEMA: &str = "
CREATE TABLE jobs (
    id TEXT PRIMARY KEY CHECK (length(id) = 32),
    kind TEXT NOT NULL CHECK (kind IN ('import','meeting')),
    name TEXT NOT NULL,
    status TEXT NOT NULL,
    message TEXT,
    language TEXT,
    source_file TEXT,
    duration REAL,
    checkpoint_duration REAL,
    checkpoint_updated REAL,
    revision INTEGER NOT NULL DEFAULT 0,
    created REAL NOT NULL,
    updated REAL NOT NULL,
    document TEXT,
    meeting TEXT,
    private TEXT
);
CREATE INDEX jobs_status ON jobs(status);
CREATE INDEX jobs_created ON jobs(created);
CREATE TABLE people (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    created REAL NOT NULL,
    updated REAL
);
CREATE TABLE voice_profiles (
    person_id TEXT PRIMARY KEY REFERENCES people(id) ON DELETE CASCADE,
    model_id TEXT NOT NULL,
    revision TEXT NOT NULL,
    artifact_sha256 TEXT NOT NULL,
    dimension INTEGER NOT NULL,
    centroid BLOB NOT NULL,
    profile_version TEXT NOT NULL,
    consented_at REAL NOT NULL,
    enrollment TEXT NOT NULL
);
CREATE TABLE preferences (key TEXT PRIMARY KEY, value TEXT NOT NULL);
CREATE TABLE migrations (
    id INTEGER PRIMARY KEY CHECK (id = 1),
    legacy_path TEXT NOT NULL,
    legacy_sha256 TEXT NOT NULL,
    migrated_at REAL NOT NULL,
    app_version TEXT NOT NULL,
    counts TEXT NOT NULL
);
";

/// Errors from the store layer.
#[derive(Debug, thiserror::Error)]
pub enum StoreError {
    #[error("sqlite: {0}")]
    Sqlite(#[from] rusqlite::Error),
    #[error("migration: {0}")]
    Migration(#[from] rusqlite_migration::Error),
    #[error("json: {0}")]
    Json(#[from] serde_json::Error),
    #[error("io: {0}")]
    Io(#[from] std::io::Error),
    #[error("{0}")]
    Store(String),
}

pub type Result<T> = std::result::Result<T, StoreError>;

fn bad(message: &str) -> StoreError {
    StoreError::Store(message.to_string())
}

/// Create (not truncate) `path` with mode 0600, then open it.
fn create_private_file(path: &Path) -> Result<()> {
    let mut options = OpenOptions::new();
    options.write(true).create_new(true);
    #[cfg(unix)]
    {
        use std::os::unix::fs::OpenOptionsExt;
        options.mode(0o600);
    }
    options.open(path).map(|_| ()).map_err(Into::into)
}

fn configure(conn: &Connection) -> Result<()> {
    conn.pragma_update(None, "journal_mode", "WAL")?;
    conn.pragma_update(None, "synchronous", "FULL")?;
    conn.pragma_update(None, "foreign_keys", "ON")?;
    conn.busy_timeout(Duration::from_secs(10))?;
    Ok(())
}

/// Run schema migrations on a connection.
fn migrate_schema(conn: &mut Connection) -> Result<()> {
    let migrations = Migrations::new(vec![M::up(SCHEMA)]);
    migrations.to_latest(conn)?;
    Ok(())
}

/// `rolling_refinement.RollingPlan(state, recover=True).snapshot()` —
/// validate the checkpoint and requeue in-flight windows.
fn recover_rolling_plan(state: &RollingRefinement) -> Result<RollingRefinement> {
    const CORE_SAMPLES: i64 = 18 * 16_000;
    const CONTEXT_LIMIT: i64 = 24 * 16_000;
    const MAX_PENDING: usize = 2;
    let mut state = state.clone();
    if state.version != 1 || state.completed_sample < 0 {
        return Err(bad("Invalid rolling refinement checkpoint."));
    }
    if state.pending.len() > MAX_PENDING {
        return Err(bad("Invalid rolling refinement checkpoint."));
    }
    let mut previous = state.completed_sample;
    for window in state.pending.iter_mut() {
        let (Some(context_start), Some(context_end)) =
            (window.context_start_sample, window.context_end_sample)
        else {
            return Err(bad("Invalid rolling refinement checkpoint."));
        };
        let ok = window.start_sample == previous
            && previous < window.end_sample
            && window.end_sample <= previous + CORE_SAMPLES
            && context_end - context_start <= CONTEXT_LIMIT;
        if !ok {
            return Err(bad("Noncontiguous or oversized refinement request."));
        }
        previous = window.end_sample;
        if window.status == RollingWindowStatus::InFlight {
            window.status = RollingWindowStatus::Queued;
        }
    }
    Ok(state)
}

/// Apply the v0.6.6 `JobStore.recover` rules to one decoded job payload.
/// Returns true when the payload changed.
pub fn recover_job(job: &mut Job, now: f64) -> Result<bool> {
    let mut changed = false;
    if job.status.is_active() {
        job.status = JobStatus::Failed;
        job.message = Some(
            if job.kind == Some(JobKind::Meeting) {
                "The meeting was interrupted. Captured audio and transcript were preserved."
            } else {
                "The app stopped during processing. Retry this job."
            }
            .to_string(),
        );
        job.updated = now;
        changed = true;
    }
    if job.rolling_refinement.is_some()
        && matches!(
            job.refinement_status,
            Some(RefinementStatus::Waiting | RefinementStatus::Refining)
        )
    {
        let recovered = recover_rolling_plan(job.rolling_refinement.as_ref().unwrap())?;
        job.rolling_refinement = Some(recovered);
        job.refinement_status = Some(RefinementStatus::Paused);
        job.private.rolling_inflight = None;
        changed = true;
    }
    Ok(changed)
}

/// `JobStore._decode` progress overlay on an assembled payload map.
fn overlay_progress(
    job: &mut Map<String, Value>,
    checkpoint_duration: Option<f64>,
    checkpoint_updated: Option<f64>,
) {
    if let Some(duration) = checkpoint_duration {
        let base = job.get("duration").and_then(Value::as_f64).unwrap_or(0.0);
        job.insert("duration".to_string(), json!(base.max(duration)));
        let updated = job.get("updated").and_then(Value::as_f64).unwrap_or(0.0);
        let progress = checkpoint_updated.unwrap_or(0.0);
        job.insert("updated".to_string(), json!(updated.max(progress)));
    }
}

/// The `jobs` table plus people, voice profiles and preferences.
pub struct Store {
    conn: Mutex<Connection>,
}

/// Split the serialized job payload into (meeting, document, private) JSON
/// column values. Columns are read straight from the typed `Job`.
fn split_job(job: &Job) -> (Option<String>, String, Option<String>) {
    let payload = job.to_payload();
    let mut meeting = Map::new();
    let mut private = None;
    let mut document = "null".to_string();
    if let Value::Object(map) = payload {
        for (k, v) in map {
            if COLUMN_KEYS.contains(&k.as_str()) {
                continue;
            } else if k == "document" {
                document = serde_json::to_string(&v).unwrap_or_default();
            } else if PRIVATE_KEYS.contains(&k.as_str()) {
                continue;
            } else {
                meeting.insert(k, v);
            }
        }
    }
    let private_value = serde_json::to_value(&job.private).unwrap_or_default();
    if let Value::Object(map) = private_value
        && !map.is_empty()
    {
        private = Some(serde_json::to_string(&Value::Object(map)).unwrap_or_default());
    }
    (
        (!meeting.is_empty())
            .then(|| serde_json::to_string(&Value::Object(meeting)).unwrap_or_default()),
        document,
        private,
    )
}

/// Merge a stored row back into the v0.6.6 payload shape, then deserialize
/// into the typed `Job`.
fn assemble_job(
    column_job: Map<String, Value>,
    meeting: Option<String>,
    document: Option<String>,
    private: Option<String>,
    progress: (Option<f64>, Option<f64>),
    metadata: bool,
) -> Result<Job> {
    let mut job = column_job;
    if let Some(Value::Object(map)) = meeting
        .as_deref()
        .map(serde_json::from_str::<Value>)
        .transpose()?
    {
        for (k, v) in map {
            job.insert(k, v);
        }
    }
    if !metadata {
        if let Some(document) = document {
            let parsed: Value = serde_json::from_str(&document)?;
            job.insert("document".to_string(), parsed);
        }
        if let Some(Value::Object(map)) = private
            .as_deref()
            .map(serde_json::from_str::<Value>)
            .transpose()?
        {
            for (k, v) in map {
                job.insert(k, v);
            }
        }
    }
    overlay_progress(&mut job, progress.0, progress.1);
    Ok(serde_json::from_value(Value::Object(job))?)
}

/// The `jobs` table's columnar fields for one row.
struct JobColumns<'a> {
    id: &'a str,
    kind: &'a str,
    name: &'a str,
    status: &'a str,
    message: Option<&'a str>,
    language: Option<&'a str>,
    source_file: Option<&'a str>,
    duration: Option<f64>,
    revision: i64,
    created: f64,
    updated: f64,
}

/// Columnar job fields assembled into a payload map (the `kind` import
/// omission lives here).
fn column_map(c: &JobColumns<'_>) -> Map<String, Value> {
    let mut job = Map::new();
    job.insert("id".to_string(), json!(c.id));
    // v0.6.6 writes `kind` only on meeting jobs; keep that payload shape.
    if c.kind != "import" {
        job.insert("kind".to_string(), json!(c.kind));
    }
    job.insert("name".to_string(), json!(c.name));
    job.insert("status".to_string(), json!(c.status));
    if let Some(v) = c.message {
        job.insert("message".to_string(), json!(v));
    }
    if let Some(v) = c.language {
        job.insert("language".to_string(), json!(v));
    }
    if let Some(v) = c.source_file {
        job.insert("source_file".to_string(), json!(v));
    }
    job.insert(
        "duration".to_string(),
        c.duration.map_or(Value::Null, |d| json!(d)),
    );
    job.insert("revision".to_string(), json!(c.revision));
    job.insert("created".to_string(), json!(c.created));
    job.insert("updated".to_string(), json!(c.updated));
    job
}

/// Write one `jobs` row (columns + meeting/document/private split) inside a
/// transaction. The checkpoint columns are cleared, matching `JobStore.put`
/// folding.
pub(crate) fn write_job_row(
    tx: &rusqlite::Transaction,
    job: &Job,
) -> std::result::Result<(), StoreError> {
    let (meeting, document, private) = split_job(job);
    tx.execute(
        "INSERT OR REPLACE INTO jobs
         (id,kind,name,status,message,language,source_file,duration,revision,created,updated,document,meeting,private,checkpoint_duration,checkpoint_updated)
         VALUES (?1,?2,?3,?4,?5,?6,?7,?8,?9,?10,?11,?12,?13,?14,NULL,NULL)",
        params![
            job.id,
            // v0.6.6 omits `kind` on import payloads; the column stores it.
            match job.kind {
                Some(JobKind::Meeting) => "meeting",
                _ => "import",
            },
            job.name,
            job.status.as_str(),
            job.message,
            job.language,
            job.source_file,
            job.duration,
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

/// The `enrollment` blob: every non-column profile key (absent vs present
/// exactly as the payload had it) with `sample_rate` lifted out of `model`.
pub(crate) fn enrollment_payload(
    profile: &VoiceProfile,
) -> std::result::Result<String, StoreError> {
    let mut enrollment = serde_json::to_value(profile)?;
    if let Some(obj) = enrollment.as_object_mut() {
        for key in ["person_id", "version", "enrolled_at", "model"] {
            obj.remove(key);
        }
        obj.insert("sample_rate".to_string(), json!(profile.model.sample_rate));
    }
    Ok(serde_json::to_string(&enrollment)?)
}

/// Insert or replace a person row inside a transaction.
pub(crate) fn write_person_row(
    tx: &rusqlite::Transaction,
    person: &Person,
) -> std::result::Result<(), StoreError> {
    tx.execute(
        "INSERT OR REPLACE INTO people (id,name,created,updated) VALUES (?1,?2,?3,?4)",
        params![person.id, person.name, person.created, person.updated],
    )?;
    Ok(())
}

/// Insert or replace a voice profile row (centroid as f32-le BLOB) inside a
/// transaction.
pub(crate) fn write_voice_profile_row(
    tx: &rusqlite::Transaction,
    profile: &VoiceProfile,
    centroid: &[f32],
) -> std::result::Result<(), StoreError> {
    let blob: Vec<u8> = centroid.iter().flat_map(|f| f.to_le_bytes()).collect();
    tx.execute(
        "INSERT OR REPLACE INTO voice_profiles
         (person_id,model_id,revision,artifact_sha256,dimension,centroid,profile_version,consented_at,enrollment)
         VALUES (?1,?2,?3,?4,?5,?6,?7,?8,?9)",
        params![
            profile.person_id,
            profile.model.model_id,
            profile.model.revision,
            profile.model.artifact_sha256,
            profile.model.dimension,
            blob,
            profile.version,
            profile.enrolled_at,
            enrollment_payload(profile)?,
        ],
    )?;
    Ok(())
}

impl Store {
    /// Open (or create) `<storage>/speakerdesk.sqlite` with the §5 pragmas
    /// and schema. The file is created mode 0600.
    pub fn open(storage: &Path) -> Result<Self> {
        let path = storage.join("speakerdesk.sqlite");
        if !path.exists() {
            create_private_file(&path)?;
        }
        let mut conn = Connection::open(&path)?;
        configure(&conn)?;
        migrate_schema(&mut conn)?;
        Ok(Self {
            conn: Mutex::new(conn),
        })
    }

    /// `JobStore.get` — the typed job (`metadata=false` loads document and
    /// private state; `metadata=true` leaves `document=None` and private
    /// state empty, the `job_metadata` projection).
    pub fn get(&self, jid: &str, metadata: bool) -> Result<Option<Job>> {
        let conn = self.conn.lock().unwrap();
        let row = conn
            .query_row(
                "SELECT kind,name,status,message,language,source_file,duration,revision,created,updated,document,meeting,private,checkpoint_duration,checkpoint_updated
                 FROM jobs WHERE id=?1",
                params![jid],
                |row| {
                    Ok((
                        row.get::<_, String>(0)?,
                        row.get::<_, String>(1)?,
                        row.get::<_, String>(2)?,
                        row.get::<_, Option<String>>(3)?,
                        row.get::<_, Option<String>>(4)?,
                        row.get::<_, Option<String>>(5)?,
                        row.get::<_, Option<f64>>(6)?,
                        row.get::<_, i64>(7)?,
                        row.get::<_, f64>(8)?,
                        row.get::<_, f64>(9)?,
                        row.get::<_, Option<String>>(10)?,
                        row.get::<_, Option<String>>(11)?,
                        row.get::<_, Option<String>>(12)?,
                        row.get::<_, Option<f64>>(13)?,
                        row.get::<_, Option<f64>>(14)?,
                    ))
                },
            )
            .optional()?;
        let Some((
            kind,
            name,
            status,
            message,
            language,
            source_file,
            duration,
            revision,
            created,
            updated,
            document,
            meeting,
            private,
            checkpoint_duration,
            checkpoint_updated,
        )) = row
        else {
            return Ok(None);
        };
        let columns = column_map(&JobColumns {
            id: jid,
            kind: &kind,
            name: &name,
            status: &status,
            message: message.as_deref(),
            language: language.as_deref(),
            source_file: source_file.as_deref(),
            duration,
            revision,
            created,
            updated,
        });
        Ok(Some(assemble_job(
            columns,
            meeting,
            document,
            private,
            (checkpoint_duration, checkpoint_updated),
            metadata,
        )?))
    }

    /// `JobStore.summaries` — metadata rows ordered by created DESC,
    /// projected to the summary fields with the progress overlay.
    pub fn summaries(&self) -> Result<Vec<JobSummary>> {
        let conn = self.conn.lock().unwrap();
        let mut stmt = conn.prepare(
            "SELECT id,kind,name,status,message,language,source_file,duration,revision,created,updated,meeting,checkpoint_duration,checkpoint_updated FROM jobs ORDER BY created DESC",
        )?;
        let rows = stmt
            .query_map([], |row| {
                Ok((
                    row.get::<_, String>(0)?,
                    row.get::<_, String>(1)?,
                    row.get::<_, String>(2)?,
                    row.get::<_, String>(3)?,
                    row.get::<_, Option<String>>(4)?,
                    row.get::<_, Option<String>>(5)?,
                    row.get::<_, Option<String>>(6)?,
                    row.get::<_, Option<f64>>(7)?,
                    row.get::<_, i64>(8)?,
                    row.get::<_, f64>(9)?,
                    row.get::<_, f64>(10)?,
                    row.get::<_, Option<String>>(11)?,
                    row.get::<_, Option<f64>>(12)?,
                    row.get::<_, Option<f64>>(13)?,
                ))
            })?
            .collect::<rusqlite::Result<Vec<_>>>()?;
        let mut out = Vec::with_capacity(rows.len());
        for (
            id,
            kind,
            name,
            status,
            message,
            language,
            source_file,
            duration,
            revision,
            created,
            updated,
            meeting,
            checkpoint_duration,
            checkpoint_updated,
        ) in rows
        {
            let mut job = column_map(&JobColumns {
                id: &id,
                kind: &kind,
                name: &name,
                status: &status,
                message: message.as_deref(),
                language: language.as_deref(),
                source_file: source_file.as_deref(),
                duration,
                revision,
                created,
                updated,
            });
            // Metadata overlay: meeting product keys minus private/document.
            if let Some(Value::Object(map)) = meeting
                .as_deref()
                .map(serde_json::from_str::<Value>)
                .transpose()?
            {
                for (k, v) in map {
                    job.insert(k, v);
                }
            }
            overlay_progress(&mut job, checkpoint_duration, checkpoint_updated);
            // The SUMMARY_FIELDS projection on the typed summary.
            let status: JobStatus =
                serde_json::from_value(job.get("status").cloned().unwrap_or(Value::Null))?;
            out.push(JobSummary {
                id,
                name,
                kind: match kind.as_str() {
                    "meeting" => Some(JobKind::Meeting),
                    _ => None,
                },
                source_file,
                created,
                updated: job
                    .get("updated")
                    .and_then(Value::as_f64)
                    .unwrap_or(updated),
                status,
                message,
                language,
                duration: job.get("duration").and_then(Value::as_f64),
                revision,
            });
        }
        Ok(out)
    }

    /// `JobStore.count_status`.
    pub fn count_status(&self, statuses: &[JobStatus]) -> Result<i64> {
        let conn = self.conn.lock().unwrap();
        let placeholders = statuses.iter().map(|_| "?").collect::<Vec<_>>().join(",");
        let sql = format!("SELECT count(*) FROM jobs WHERE status IN ({placeholders})");
        let mut stmt = conn.prepare(&sql)?;
        let count = stmt.query_row(
            rusqlite::params_from_iter(statuses.iter().map(|s| s.as_str())),
            |row| row.get::<_, i64>(0),
        )?;
        Ok(count)
    }

    /// `JobStore.put` — fold any duration checkpoint, stamp `updated`, and
    /// write the row (meeting/document/private split) in one transaction.
    /// `job.updated` (and the folded `duration`) mutate in place, like
    /// v0.6.6's `put` stamping the payload it received.
    pub fn put(&self, job: &mut Job, now: f64) -> Result<()> {
        self.put_with_language(job, now, None)
    }

    /// `JobStore.put` with the optional `default_language` preference write
    /// in the same commit (v0.6.6 `save_default_language` inside `put`).
    pub fn put_with_language(
        &self,
        job: &mut Job,
        now: f64,
        default_language: Option<&str>,
    ) -> Result<()> {
        crate::model::check_opaque_bounds_job(job).map_err(|e| bad(&e))?;
        let conn = self.conn.lock().unwrap();
        let tx = conn.unchecked_transaction()?;
        let checkpoint: Option<f64> = tx
            .query_row(
                "SELECT checkpoint_duration FROM jobs WHERE id=?1",
                params![job.id],
                |row| row.get(0),
            )
            .optional()?;
        if let Some(checkpoint) = checkpoint {
            job.duration = Some(job.duration.unwrap_or(0.0).max(checkpoint));
        }
        job.updated = now;
        write_job_row(&tx, job)?;
        // The selected future-audio epoch and restart default are one commit.
        if let Some(language) = default_language {
            if !language_choice(language) {
                return Err(bad("Choose a supported language mode."));
            }
            tx.execute(
                "INSERT OR REPLACE INTO preferences (key, value) VALUES ('default_language', ?1)",
                params![language],
            )?;
        }
        tx.commit()?;
        Ok(())
    }

    /// `JobStore.checkpoint_duration` — fold a live duration checkpoint.
    /// Returns false when the job does not exist.
    pub fn checkpoint_duration(&self, jid: &str, duration: f64, now: f64) -> Result<bool> {
        if !duration.is_finite() || duration < 0.0 {
            return Err(bad("Capture duration must be finite and nonnegative."));
        }
        let conn = self.conn.lock().unwrap();
        let tx = conn.unchecked_transaction()?;
        let row = tx
            .query_row(
                "SELECT duration, checkpoint_duration FROM jobs WHERE id=?1",
                params![jid],
                |row| Ok((row.get::<_, Option<f64>>(0)?, row.get::<_, Option<f64>>(1)?)),
            )
            .optional()?;
        let Some((base, checkpoint)) = row else {
            return Ok(false);
        };
        if duration > base.unwrap_or(0.0).max(checkpoint.unwrap_or(0.0)) {
            tx.execute(
                "UPDATE jobs SET checkpoint_duration=?2, checkpoint_updated=?3 WHERE id=?1",
                params![jid, duration, now],
            )?;
        }
        tx.commit()?;
        Ok(true)
    }

    /// Delete a job row (v0.6.6 `DELETE /api/jobs/<jid>` store part).
    pub fn delete(&self, jid: &str) -> Result<()> {
        let conn = self.conn.lock().unwrap();
        conn.execute("DELETE FROM jobs WHERE id=?1", params![jid])?;
        Ok(())
    }

    /// `JobStore.recover` — fold crash checkpoints, then map interrupted
    /// work to `failed`/`paused` with the exact v0.6.6 messages.
    pub fn recover(&self, now: f64) -> Result<()> {
        let conn = self.conn.lock().unwrap();
        let tx = conn.unchecked_transaction()?;
        let mut stmt = tx.prepare(
            "SELECT id,kind,name,status,message,language,source_file,duration,revision,created,updated,document,meeting,private,checkpoint_duration,checkpoint_updated FROM jobs",
        )?;
        let rows = stmt
            .query_map([], |row| {
                Ok((
                    row.get::<_, String>(0)?,
                    row.get::<_, String>(1)?,
                    row.get::<_, String>(2)?,
                    row.get::<_, String>(3)?,
                    row.get::<_, Option<String>>(4)?,
                    row.get::<_, Option<String>>(5)?,
                    row.get::<_, Option<String>>(6)?,
                    row.get::<_, Option<f64>>(7)?,
                    row.get::<_, i64>(8)?,
                    row.get::<_, f64>(9)?,
                    row.get::<_, f64>(10)?,
                    row.get::<_, Option<String>>(11)?,
                    row.get::<_, Option<String>>(12)?,
                    row.get::<_, Option<String>>(13)?,
                    row.get::<_, Option<f64>>(14)?,
                    row.get::<_, Option<f64>>(15)?,
                ))
            })?
            .collect::<rusqlite::Result<Vec<_>>>()?;
        drop(stmt);
        for (
            id,
            kind,
            name,
            status,
            message,
            language,
            source_file,
            duration,
            revision,
            created,
            updated,
            document,
            meeting,
            private,
            checkpoint_duration,
            checkpoint_updated,
        ) in rows
        {
            let columns = column_map(&JobColumns {
                id: &id,
                kind: &kind,
                name: &name,
                status: &status,
                message: message.as_deref(),
                language: language.as_deref(),
                source_file: source_file.as_deref(),
                duration,
                revision,
                created,
                updated,
            });
            let mut job = assemble_job(
                columns,
                meeting,
                document,
                private,
                (checkpoint_duration, checkpoint_updated),
                false,
            )?;
            let mut changed = checkpoint_duration.is_some();
            if recover_job(&mut job, now)? {
                changed = true;
            }
            if changed {
                write_job_row(&tx, &job)?;
                tx.execute(
                    "UPDATE jobs SET checkpoint_duration=NULL, checkpoint_updated=NULL WHERE id=?1",
                    params![id],
                )?;
            }
        }
        tx.commit()?;
        Ok(())
    }

    // ---- people / voice profiles / preferences (importer-facing CRUD) ----

    /// Insert or replace a person row.
    pub fn put_person(&self, person: &Person) -> Result<()> {
        let conn = self.conn.lock().unwrap();
        let tx = conn.unchecked_transaction()?;
        write_person_row(&tx, person)?;
        tx.commit()?;
        Ok(())
    }

    /// All people ordered by created.
    pub fn list_people(&self) -> Result<Vec<Person>> {
        let conn = self.conn.lock().unwrap();
        let mut stmt =
            conn.prepare("SELECT id,name,created,updated FROM people ORDER BY created")?;
        let rows = stmt
            .query_map([], |row| {
                Ok(Person {
                    id: row.get::<_, String>(0)?,
                    name: row.get::<_, String>(1)?,
                    created: row.get::<_, f64>(2)?,
                    updated: row.get::<_, Option<f64>>(3)?,
                })
            })?
            .collect::<rusqlite::Result<Vec<_>>>()?;
        Ok(rows)
    }

    /// Insert or replace a voice profile (centroid stored as f32-le BLOB).
    /// `profile` is the typed payload minus the centroid array.
    pub fn put_voice_profile(&self, profile: &VoiceProfile, centroid: &[f32]) -> Result<()> {
        let conn = self.conn.lock().unwrap();
        let tx = conn.unchecked_transaction()?;
        write_voice_profile_row(&tx, profile, centroid)?;
        tx.commit()?;
        Ok(())
    }

    /// One voice profile as the typed payload plus its f32 centroid.
    pub fn get_voice_profile(&self, person_id: &str) -> Result<Option<(VoiceProfile, Vec<f32>)>> {
        let conn = self.conn.lock().unwrap();
        let row = conn
            .query_row(
                "SELECT model_id,revision,artifact_sha256,dimension,centroid,profile_version,consented_at,enrollment
                 FROM voice_profiles WHERE person_id=?1",
                params![person_id],
                |row| {
                    Ok((
                        row.get::<_, String>(0)?,
                        row.get::<_, String>(1)?,
                        row.get::<_, String>(2)?,
                        row.get::<_, i64>(3)?,
                        row.get::<_, Vec<u8>>(4)?,
                        row.get::<_, String>(5)?,
                        row.get::<_, f64>(6)?,
                        row.get::<_, String>(7)?,
                    ))
                },
            )
            .optional()?;
        let Some((
            model_id,
            revision,
            artifact_sha256,
            dimension,
            centroid,
            profile_version,
            consented_at,
            enrollment,
        )) = row
        else {
            return Ok(None);
        };
        let floats: Vec<f32> = centroid
            .as_chunks::<4>()
            .0
            .iter()
            .map(|c| f32::from_le_bytes(*c))
            .collect();
        let enrollment: Value = serde_json::from_str(&enrollment)?;
        let mut payload = enrollment;
        let sample_rate = payload
            .as_object_mut()
            .and_then(|o| o.remove("sample_rate"))
            .unwrap_or_else(|| json!(16_000));
        let obj = payload.as_object_mut().unwrap();
        // `sample_rate` lives inside `model` in the v0.6.6 payload shape.
        obj.insert("person_id".to_string(), json!(person_id));
        obj.insert("version".to_string(), json!(profile_version));
        obj.insert("enrolled_at".to_string(), json!(consented_at));
        obj.insert(
            "model".to_string(),
            json!({
                "model_id": model_id,
                "revision": revision,
                "artifact_sha256": artifact_sha256,
                "dimension": dimension,
                "sample_rate": sample_rate,
            }),
        );
        let profile: VoiceProfile = serde_json::from_value(payload)?;
        Ok(Some((profile, floats)))
    }

    /// All voice profiles.
    pub fn list_voice_profiles(&self) -> Result<Vec<(VoiceProfile, Vec<f32>)>> {
        let conn = self.conn.lock().unwrap();
        let mut stmt = conn.prepare("SELECT person_id FROM voice_profiles")?;
        let ids: Vec<String> = stmt
            .query_map([], |row| row.get::<_, String>(0))?
            .collect::<rusqlite::Result<Vec<_>>>()?;
        drop(stmt);
        drop(conn);
        let mut out = Vec::new();
        for id in ids {
            if let Some(profile) = self.get_voice_profile(&id)? {
                out.push(profile);
            }
        }
        Ok(out)
    }

    /// `language_preferences.default_language` — 'auto' unless a supported
    /// choice was saved.
    pub fn default_language(&self) -> Result<String> {
        Ok(self
            .get_preference("default_language")?
            .filter(|v| language_choice(v))
            .unwrap_or_else(|| "auto".to_string()))
    }

    /// `preferences` key/value — value is the raw stored string.
    pub fn set_preference(&self, key: &str, value: &str) -> Result<()> {
        let conn = self.conn.lock().unwrap();
        conn.execute(
            "INSERT OR REPLACE INTO preferences (key,value) VALUES (?1,?2)",
            params![key, value],
        )?;
        Ok(())
    }

    pub fn get_preference(&self, key: &str) -> Result<Option<String>> {
        let conn = self.conn.lock().unwrap();
        Ok(conn
            .query_row(
                "SELECT value FROM preferences WHERE key=?1",
                params![key],
                |row| row.get::<_, String>(0),
            )
            .optional()?)
    }

    /// All preferences as (key, value) pairs.
    pub fn list_preferences(&self) -> Result<Vec<(String, String)>> {
        let conn = self.conn.lock().unwrap();
        let mut stmt = conn.prepare("SELECT key,value FROM preferences")?;
        let rows = stmt
            .query_map([], |row| {
                Ok((row.get::<_, String>(0)?, row.get::<_, String>(1)?))
            })?
            .collect::<rusqlite::Result<Vec<_>>>()?;
        Ok(rows)
    }

    /// Record a completed migration (single-row `migrations` table).
    pub fn record_migration(
        &self,
        legacy_path: &str,
        legacy_sha256: &str,
        migrated_at: f64,
        app_version: &str,
        counts: &legacy::MigrationOutcome,
    ) -> Result<()> {
        let conn = self.conn.lock().unwrap();
        conn.execute(
            "INSERT OR REPLACE INTO migrations (id,legacy_path,legacy_sha256,migrated_at,app_version,counts) VALUES (1,?1,?2,?3,?4,?5)",
            params![legacy_path, legacy_sha256, migrated_at, app_version, serde_json::to_string(counts)?],
        )?;
        Ok(())
    }
}
