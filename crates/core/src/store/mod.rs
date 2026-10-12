//! New database at `<storage>/speakerdesk.sqlite`. The legacy `jobs.sqlite` is never written.
mod legacy;
mod sqlite;

pub use legacy::import_legacy;
pub use sqlite::SqliteJobStore;

use std::time::Duration;

use rusqlite::Connection;
use rusqlite_migration::{M, Migrations};

use crate::document::{Job, JobSummary};
use crate::error::StoreError;
use crate::ids::JobId;
use crate::messages;

pub const JOURNAL_MODE: &str = "WAL";
pub const SYNCHRONOUS: &str = "FULL";

pub const SCHEMA_V1: &str = "
CREATE TABLE jobs (
  id TEXT PRIMARY KEY CHECK(length(id) = 32),
  kind TEXT NOT NULL CHECK(kind IN ('import', 'meeting')),
  name TEXT NOT NULL,
  status TEXT NOT NULL,
  message TEXT NOT NULL,
  language TEXT NOT NULL,
  source_file TEXT,
  duration REAL,
  checkpoint_duration REAL,
  revision INTEGER NOT NULL,
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
  updated REAL NOT NULL
);
CREATE TABLE voice_profiles (
  person_id TEXT PRIMARY KEY REFERENCES people(id) ON DELETE CASCADE,
  model_id TEXT NOT NULL,
  revision TEXT NOT NULL,
  artifact_sha256 TEXT NOT NULL,
  dimension INTEGER NOT NULL,
  centroid BLOB NOT NULL,
  profile_version INTEGER NOT NULL,
  consented_at REAL,
  enrollment TEXT NOT NULL
);
CREATE TABLE preferences (
  key TEXT PRIMARY KEY,
  value TEXT NOT NULL
);
CREATE TABLE migrations (
  legacy_path TEXT PRIMARY KEY,
  legacy_sha256 TEXT NOT NULL,
  migrated_at REAL NOT NULL,
  app_version TEXT NOT NULL,
  counts TEXT NOT NULL
);
";

pub fn migrations() -> Migrations<'static> {
    Migrations::new(vec![M::up(SCHEMA_V1), M::up("
ALTER TABLE voice_profiles RENAME TO old_voice_profiles;
CREATE TABLE voice_profiles (
 person_id TEXT PRIMARY KEY REFERENCES people(id) ON DELETE CASCADE,
 model_id TEXT NOT NULL, revision TEXT NOT NULL, artifact_sha256 TEXT NOT NULL,
 dimension INTEGER NOT NULL, centroid BLOB NOT NULL, profile_version TEXT NOT NULL,
 consented_at REAL, enrollment TEXT NOT NULL
);
INSERT INTO voice_profiles SELECT person_id,model_id,revision,artifact_sha256,dimension,centroid,
 COALESCE(json_extract(enrollment, '$.version'), CAST(profile_version AS TEXT)),consented_at,enrollment FROM old_voice_profiles;
DROP TABLE old_voice_profiles;
"), M::up("ALTER TABLE people ADD COLUMN voice_generation INTEGER NOT NULL DEFAULT 0;")])
}

pub fn apply_pragmas(connection: &Connection) -> Result<(), StoreError> {
    connection.pragma_update(None, "journal_mode", JOURNAL_MODE)?;
    connection.pragma_update(None, "synchronous", SYNCHRONOUS)?;
    connection.pragma_update(None, "foreign_keys", "ON")?;
    connection.busy_timeout(Duration::from_millis(10_000))?;
    Ok(())
}

pub fn validate_checkpoint_duration(duration: f64) -> Result<(), StoreError> {
    if duration.is_finite() && duration >= 0.0 {
        Ok(())
    } else {
        Err(StoreError::message(messages::CAPTURE_DURATION_INVALID))
    }
}

pub trait JobStore {
    fn initialize(&mut self) -> Result<(), StoreError>;
    fn get(&self, id: &JobId, metadata: bool) -> Result<Option<Job>, StoreError>;
    fn summaries(&self) -> Result<Vec<JobSummary>, StoreError>;
    fn count_status(&self, statuses: &[&str]) -> Result<u64, StoreError>;
    fn put(&mut self, job: &Job, default_language: Option<&str>) -> Result<(), StoreError>;
    fn checkpoint_duration(&mut self, id: &JobId, duration: f64) -> Result<bool, StoreError>;
    fn recover(&mut self, active: &[&str]) -> Result<(), StoreError>;
}

#[cfg(test)]
mod tests {
    use super::{apply_pragmas, migrations, validate_checkpoint_duration};

    #[test]
    fn schema_v1_applies_on_an_empty_database() {
        let mut connection = rusqlite::Connection::open_in_memory().unwrap();
        apply_pragmas(&connection).unwrap();
        migrations().to_latest(&mut connection).unwrap();
        let version: i64 = connection
            .query_row("PRAGMA user_version", [], |row| row.get(0))
            .unwrap();
        assert_eq!(version, 3);
        let tables: Vec<String> = connection
            .prepare("SELECT name FROM sqlite_master WHERE type = 'table' ORDER BY name")
            .unwrap()
            .query_map([], |row| row.get(0))
            .unwrap()
            .collect::<Result<_, _>>()
            .unwrap();
        assert!(tables.iter().any(|name| name == "jobs"));
        assert!(tables.iter().any(|name| name == "voice_profiles"));
    }

    #[test]
    fn checkpoint_duration_rejects_non_finite() {
        assert!(validate_checkpoint_duration(-1.0).is_err());
        assert!(validate_checkpoint_duration(f64::NAN).is_err());
        assert!(validate_checkpoint_duration(1.25).is_ok());
    }
}
