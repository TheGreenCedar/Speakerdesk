//! Private draft copies. These receipts never update canonical transcripts.
use crate::routes::AppState;
use axum::{
    Json, Router,
    extract::{Path as RoutePath, State},
    http::StatusCode,
    response::{IntoResponse, Response},
    routing::get,
};
use serde_json::{Value, json};
use std::{
    fs::{self, OpenOptions},
    io::{Read, Write},
    path::{Path, PathBuf},
};

pub const MAX_BYTES: usize = 5 * 1024 * 1024;
fn hex(value: &str) -> bool {
    value.len() == 32
        && value
            .bytes()
            .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b))
}
fn valid_id(id: &str) -> bool {
    id.split_once('-')
        .is_some_and(|(session, job)| hex(session) && hex(job))
}
fn invalid() -> std::io::Error {
    std::io::Error::new(
        std::io::ErrorKind::InvalidData,
        "Invalid or oversized draft recovery copy",
    )
}

#[derive(Clone)]
pub struct RecoveryStore {
    root: PathBuf,
}
impl RecoveryStore {
    pub fn open(storage: &Path) -> std::io::Result<Self> {
        let root = storage.join("draft-recovery");
        if let Ok(metadata) = fs::symlink_metadata(&root) {
            if metadata.is_symlink() || !metadata.is_dir() {
                return Err(invalid());
            }
        }
        fs::create_dir_all(&root)?;
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            fs::set_permissions(&root, fs::Permissions::from_mode(0o700))?;
        }
        Ok(Self { root })
    }
    fn path(&self, id: &str) -> std::io::Result<PathBuf> {
        if !valid_id(id) {
            return Err(invalid());
        }
        Ok(self.root.join(format!("{id}.json")))
    }
    fn lock(&self) -> std::io::Result<fs::File> {
        let path = self.root.join(".writer.lock");
        if fs::symlink_metadata(&path).is_ok_and(|m| m.is_symlink() || !m.is_file()) {
            return Err(invalid());
        }
        let mut options = OpenOptions::new();
        options.create(true).truncate(false).read(true).write(true);
        #[cfg(unix)]
        {
            use std::os::unix::fs::OpenOptionsExt;
            options.mode(0o600);
        }
        let file = options.open(path)?;
        for _ in 0..20 {
            match fs4::FileExt::try_lock(&file) {
                Ok(()) => return Ok(file),
                Err(fs4::TryLockError::WouldBlock) => {
                    std::thread::sleep(std::time::Duration::from_millis(25))
                }
                Err(fs4::TryLockError::Error(error)) => return Err(error),
            }
        }
        Err(std::io::Error::new(
            std::io::ErrorKind::WouldBlock,
            "Draft recovery is busy. Try again.",
        ))
    }
    pub fn save(
        &self,
        session: &str,
        sequence: u64,
        snapshot: &Value,
    ) -> std::io::Result<Option<String>> {
        if !hex(session) || snapshot.get("schema_version") != Some(&json!(1)) {
            return Err(invalid());
        }
        let Some(job) = snapshot.get("job_id").and_then(Value::as_str) else {
            return Ok(None);
        };
        if !hex(job) {
            return Err(invalid());
        }
        let _lock = self.lock()?;
        if self.root.join(format!(".deleted-{job}")).try_exists()? {
            return Ok(None);
        }
        let id = format!("{session}-{job}");
        if snapshot.get("has_unsaved").and_then(Value::as_bool) != Some(true) {
            self.discard_unlocked(&id)?;
            return Ok(None);
        }
        let receipt = json!({"schema_version":1,"session":session,"sequence":sequence,"saved_at":std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap_or_default().as_secs(),"snapshot":snapshot});
        let bytes = serde_json::to_vec(&receipt).map_err(|_| invalid())?;
        if bytes.len() > MAX_BYTES {
            return Err(invalid());
        }
        let destination = self.path(&id)?;
        if fs::symlink_metadata(&destination).is_ok_and(|m| m.is_symlink() || !m.is_file()) {
            return Err(invalid());
        }
        let temporary = self.root.join(format!(".{id}-{sequence}.tmp"));
        let mut options = OpenOptions::new();
        options.write(true).create_new(true);
        #[cfg(unix)]
        {
            use std::os::unix::fs::OpenOptionsExt;
            options.mode(0o600);
        }
        let result: std::io::Result<()> = (|| {
            let mut file = options.open(&temporary)?;
            file.write_all(&bytes)?;
            file.sync_all()?;
            fs::rename(&temporary, &destination)?;
            fs::File::open(&self.root)?.sync_all()?;
            Ok(())
        })();
        if result.is_err() {
            let _ = fs::remove_file(&temporary);
        }
        result?;
        Ok(Some(id))
    }
    pub fn read(&self, id: &str) -> std::io::Result<Value> {
        let path = self.path(id)?;
        let metadata = fs::symlink_metadata(&path)?;
        if !metadata.is_file() || metadata.is_symlink() || metadata.len() > MAX_BYTES as u64 {
            return Err(invalid());
        }
        let mut bytes = Vec::new();
        fs::File::open(path)?
            .take(MAX_BYTES as u64 + 1)
            .read_to_end(&mut bytes)?;
        if bytes.len() > MAX_BYTES {
            return Err(invalid());
        }
        serde_json::from_slice(&bytes).map_err(|_| invalid())
    }
    pub fn list(&self) -> std::io::Result<Vec<Value>> {
        let mut result = Vec::new();
        for entry in fs::read_dir(&self.root)? {
            let entry = entry?;
            let filename = entry.file_name();
            let Some(id) = filename.to_str().and_then(|s| s.strip_suffix(".json")) else {
                continue;
            };
            if !valid_id(id) {
                continue;
            }
            if let Ok(receipt) = self.read(id) {
                result.push(json!({"id":id,"name":receipt["snapshot"]["name"],"job_id":receipt["snapshot"]["job_id"],"saved_at":receipt["saved_at"],"revision":receipt["snapshot"]["revision"]}));
            }
        }
        result.sort_by_key(|value| std::cmp::Reverse(value["saved_at"].as_u64().unwrap_or(0)));
        Ok(result)
    }
    fn discard_unlocked(&self, id: &str) -> std::io::Result<()> {
        let path = self.path(id)?;
        match fs::remove_file(path) {
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => Ok(()),
            result => result,
        }
    }
    pub fn discard(&self, id: &str) -> std::io::Result<()> {
        let _lock = self.lock()?;
        self.discard_unlocked(id)
    }
    /// A durable deletion marker suppresses late native callbacks in every session.
    pub fn discard_job(&self, job: &str) -> std::io::Result<()> {
        if !hex(job) {
            return Err(invalid());
        }
        let _lock = self.lock()?;
        let marker = self.root.join(format!(".deleted-{job}"));
        let mut options = OpenOptions::new();
        options.create_new(true).write(true);
        #[cfg(unix)]
        {
            use std::os::unix::fs::OpenOptionsExt;
            options.mode(0o600);
        }
        match options.open(&marker) {
            Ok(file) => file.sync_all()?,
            Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => {
                let metadata = fs::symlink_metadata(&marker)?;
                if metadata.is_symlink() || !metadata.is_file() {
                    return Err(invalid());
                }
            }
            Err(error) => return Err(error),
        }
        for entry in fs::read_dir(&self.root)? {
            let entry = entry?;
            let filename = entry.file_name();
            if let Some(id) = filename
                .to_str()
                .and_then(|name| name.strip_suffix(".json"))
            {
                if valid_id(id) && id.ends_with(job) {
                    self.discard_unlocked(id)?;
                }
            }
        }
        fs::File::open(&self.root)?.sync_all()?;
        Ok(())
    }
}

pub fn routes() -> Router<AppState> {
    Router::new()
        .route("/api/recovery", get(list))
        .route("/api/recovery/{id}", get(read).delete(discard))
}
fn failure(error: std::io::Error) -> Response {
    let status = if error.kind() == std::io::ErrorKind::NotFound {
        StatusCode::NOT_FOUND
    } else {
        StatusCode::INTERNAL_SERVER_ERROR
    };
    (status,Json(json!({"error":"Draft recovery copy is unavailable. Keep the current window open.","detail":error.to_string()}))).into_response()
}
async fn list(State(app): State<AppState>) -> Response {
    crate::routes::blocking(move || {
        match RecoveryStore::open(app.data_dir()).and_then(|s| s.list()) {
            Ok(receipts) => Json(json!({"receipts":receipts})).into_response(),
            Err(error) => failure(error),
        }
    })
    .await
}
async fn read(State(app): State<AppState>, RoutePath(id): RoutePath<String>) -> Response {
    crate::routes::blocking(move || {
        match RecoveryStore::open(app.data_dir()).and_then(|s| s.read(&id)) {
            Ok(receipt) => Json(receipt).into_response(),
            Err(error) => failure(error),
        }
    })
    .await
}
async fn discard(State(app): State<AppState>, RoutePath(id): RoutePath<String>) -> Response {
    crate::routes::blocking(move || {
        match RecoveryStore::open(app.data_dir()).and_then(|s| s.discard(&id)) {
            Ok(()) => Json(json!({"discarded":true})).into_response(),
            Err(error) => failure(error),
        }
    })
    .await
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn private_receipts_survive_new_sessions_without_canonical_writes() {
        let root = std::env::temp_dir().join(format!(
            "speakerdesk-recovery-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        fs::create_dir(&root).unwrap();
        let store = RecoveryStore::open(&root).unwrap();
        let a = "a".repeat(32);
        let b = "b".repeat(32);
        let job = "c".repeat(32);
        let dirty = json!({"schema_version":1,"job_id":job,"name":"Meeting","has_unsaved":true,"document":{"segments":[{"text":"Exact unsaved words"}]},"passage_drafts":[["p",{"text":"Correction"}]],"timing_drafts":[["invalid",""]]});
        let id = store.save(&a, 1, &dirty).unwrap().unwrap();
        assert_eq!(
            store.read(&id).unwrap()["snapshot"]["document"]["segments"][0]["text"],
            "Exact unsaved words"
        );
        let clean = json!({"schema_version":1,"job_id":job,"has_unsaved":false});
        store.save(&b, 1, &clean).unwrap();
        assert_eq!(store.list().unwrap().len(), 1);
        let oversized = json!({"schema_version":1,"job_id":job,"has_unsaved":true,"text":"x".repeat(MAX_BYTES)});
        assert!(store.save(&a, 2, &oversized).is_err());
        assert_eq!(store.read(&id).unwrap()["sequence"], 1);
        assert!(store.read("../../outside").is_err());
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            assert_eq!(
                fs::metadata(&store.root).unwrap().permissions().mode() & 0o777,
                0o700
            );
            assert_eq!(
                fs::metadata(store.path(&id).unwrap())
                    .unwrap()
                    .permissions()
                    .mode()
                    & 0o777,
                0o600
            );
        }
        store.save(&a, 3, &clean).unwrap();
        assert!(store.list().unwrap().is_empty());
        assert!(!root.join("speakerdesk.sqlite").exists());
        fs::remove_dir_all(root).unwrap();
    }
    #[test]
    fn deleted_jobs_cannot_be_recreated_by_other_writers_or_sessions() {
        let root = std::env::temp_dir().join(format!(
            "speakerdesk-recovery-delete-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        fs::create_dir(&root).unwrap();
        let first = RecoveryStore::open(&root).unwrap();
        let second = RecoveryStore::open(&root).unwrap();
        let session = "a".repeat(32);
        let job = "b".repeat(32);
        let other = "c".repeat(32);
        let dirty = |job: &str| json!({"schema_version":1,"job_id":job,"has_unsaved":true,"document":{"text":"Unsaved private words"}});
        first.save(&session, 1, &dirty(&job)).unwrap();
        first.save(&session, 2, &dirty(&other)).unwrap();
        second.discard_job(&job).unwrap();
        assert_eq!(first.save(&"d".repeat(32), 3, &dirty(&job)).unwrap(), None);
        assert_eq!(first.list().unwrap().len(), 1);
        assert_eq!(first.list().unwrap()[0]["job_id"], other);
        fs::remove_dir_all(root).unwrap();
    }
}
