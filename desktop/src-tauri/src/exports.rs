//! Bounded, owned export staging. The webview callback never waits for a dialog.
use std::fs::{self, DirBuilder, File};
use std::io::{self, Read, Write};
#[cfg(unix)]
use std::os::unix::fs::{DirBuilderExt, PermissionsExt};
use std::path::{Path, PathBuf};
use std::sync::{
    atomic::{AtomicU64, Ordering},
    Arc, Mutex,
};
use std::time::{SystemTime, UNIX_EPOCH};

pub const MAX_EXPORT_BYTES: u64 = 16 * 1024 * 1024;
static SEQUENCE: AtomicU64 = AtomicU64::new(0);

pub fn valid_nonce(value: &str) -> bool {
    value.len() == 32
        && value
            .bytes()
            .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b))
}

pub fn allowed_path(path: &str) -> bool {
    if path == "/api/notices" {
        return true;
    }
    let parts: Vec<_> = path.split('/').collect();
    parts.len() == 6
        && parts[0].is_empty()
        && parts[1] == "api"
        && parts[2] == "jobs"
        && valid_nonce(parts[3])
        && parts[4] == "export"
        && matches!(parts[5], "txt" | "srt" | "vtt" | "json")
}

struct PrivateDirectory(PathBuf);
impl PrivateDirectory {
    fn create(parent: &Path) -> io::Result<Self> {
        for _ in 0..8 {
            let time = SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap_or_default()
                .as_nanos();
            let sequence = SEQUENCE.fetch_add(1, Ordering::Relaxed);
            let path = parent.join(format!(
                ".speakerdesk-export-{}-{time}-{sequence}",
                std::process::id()
            ));
            let mut builder = DirBuilder::new();
            #[cfg(unix)]
            builder.mode(0o700);
            match builder.create(&path) {
                Ok(()) => return Ok(Self(path)),
                Err(error) if error.kind() == io::ErrorKind::AlreadyExists => continue,
                Err(error) => return Err(error),
            }
        }
        Err(io::Error::new(
            io::ErrorKind::AlreadyExists,
            "Export staging unavailable",
        ))
    }
}
impl Drop for PrivateDirectory {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.0);
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Phase {
    Downloading,
    Choosing,
    Writing,
    Committed,
    Cancelled,
}
struct JobState {
    phase: Phase,
    sibling: Option<PathBuf>,
}
pub struct ExportJob {
    pub url: String,
    pub nonce: String,
    pub filename: String,
    staging: PrivateDirectory,
    protected_root: Option<PathBuf>,
    gate: Mutex<JobState>,
}
impl ExportJob {
    pub fn payload(&self) -> PathBuf {
        self.staging.0.join("payload")
    }
    fn cancel(&self) {
        let sibling = {
            let mut state = self.gate.lock().unwrap_or_else(|p| p.into_inner());
            state.phase = Phase::Cancelled;
            state.sibling.take()
        };
        let _ = fs::remove_dir_all(&self.staging.0);
        if let Some(path) = sibling {
            let _ = fs::remove_dir_all(path);
        }
    }
    fn writing(&self) -> bool {
        self.gate.lock().unwrap_or_else(|p| p.into_inner()).phase == Phase::Writing
    }
    pub fn save_to(&self, destination: &Path) -> io::Result<()> {
        if !destination.is_absolute() || destination.starts_with(&self.staging.0) {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "Invalid export destination",
            ));
        }
        let parent = destination
            .parent()
            .ok_or_else(|| {
                io::Error::new(io::ErrorKind::InvalidInput, "Missing destination folder")
            })?
            .canonicalize()?;
        if self
            .protected_root
            .as_ref()
            .is_some_and(|root| parent.starts_with(root))
        {
            return Err(io::Error::new(
                io::ErrorKind::PermissionDenied,
                "Choose a folder outside Speakerdesk's recording storage",
            ));
        }
        let sibling = PrivateDirectory::create(&parent)?;
        {
            let mut state = self.gate.lock().unwrap_or_else(|p| p.into_inner());
            if state.phase != Phase::Writing {
                return Err(io::Error::new(
                    io::ErrorKind::Interrupted,
                    "Export cancelled",
                ));
            }
            state.sibling = Some(sibling.0.clone());
        }
        let candidate = sibling.0.join("payload");
        let mut source = File::open(self.payload())?;
        let mut output = File::options()
            .write(true)
            .create_new(true)
            .open(&candidate)?;
        #[cfg(unix)]
        output.set_permissions(fs::Permissions::from_mode(0o600))?;
        let mut total = 0u64;
        let mut buffer = [0u8; 64 * 1024];
        loop {
            if !self.writing() {
                return Err(io::Error::new(
                    io::ErrorKind::Interrupted,
                    "Export cancelled",
                ));
            }
            let count = source.read(&mut buffer)?;
            if count == 0 {
                break;
            }
            total += count as u64;
            if total > MAX_EXPORT_BYTES {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    "Export exceeds 16 MiB",
                ));
            }
            output.write_all(&buffer[..count])?;
        }
        output.flush()?;
        output.sync_all()?;
        drop(output);
        // Cancellation and the final atomic replacement are mutually ordered.
        let mut state = self.gate.lock().unwrap_or_else(|p| p.into_inner());
        if state.phase != Phase::Writing {
            return Err(io::Error::new(
                io::ErrorKind::Interrupted,
                "Export cancelled",
            ));
        }
        fs::rename(&candidate, destination)?;
        state.phase = Phase::Committed;
        state.sibling = None;
        Ok(())
    }
}

#[derive(Default)]
struct State {
    pending: Option<Arc<ExportJob>>,
    stopped: bool,
    update_reservation: Option<u64>,
    protected_root: Option<PathBuf>,
}
#[derive(Default)]
pub struct ExportDownloads {
    state: Mutex<State>,
}
pub enum Finished {
    Ignored,
    Failed(String),
    Ready(Arc<ExportJob>),
}
impl ExportDownloads {
    pub fn protect_storage(&self, path: &Path) -> io::Result<()> {
        self.state
            .lock()
            .unwrap_or_else(|p| p.into_inner())
            .protected_root = Some(path.canonicalize()?);
        Ok(())
    }
    pub fn begin(&self, url: &str, nonce: &str, filename: &str) -> io::Result<PathBuf> {
        let mut state = self.state.lock().unwrap_or_else(|p| p.into_inner());
        if state.stopped
            || state.update_reservation.is_some()
            || state.pending.is_some()
            || !valid_nonce(nonce)
        {
            return Err(io::Error::new(
                io::ErrorKind::WouldBlock,
                "Another export is still pending",
            ));
        }
        let filename: String = filename
            .chars()
            .filter(|c| !c.is_control() && *c != '/' && *c != '\\')
            .take(180)
            .collect();
        let job = Arc::new(ExportJob {
            url: url.into(),
            nonce: nonce.into(),
            filename: if filename.is_empty() {
                "Speakerdesk.txt".into()
            } else {
                filename
            },
            staging: PrivateDirectory::create(&std::env::temp_dir())?,
            protected_root: state.protected_root.clone(),
            gate: Mutex::new(JobState {
                phase: Phase::Downloading,
                sibling: None,
            }),
        });
        let destination = job.payload();
        state.pending = Some(job);
        Ok(destination)
    }
    pub fn finished(&self, url: &str, success: bool) -> Finished {
        let job = {
            let state = self.state.lock().unwrap_or_else(|p| p.into_inner());
            match &state.pending {
                Some(job) if job.url == url => job.clone(),
                _ => return Finished::Ignored,
            }
        };
        let mut gate = job.gate.lock().unwrap_or_else(|p| p.into_inner());
        if gate.phase != Phase::Downloading {
            return Finished::Ignored;
        }
        let valid = success
            && fs::symlink_metadata(job.payload())
                .is_ok_and(|m| m.is_file() && m.len() <= MAX_EXPORT_BYTES);
        if !valid {
            drop(gate);
            self.cancel(&job);
            return Finished::Failed(job.nonce.clone());
        }
        gate.phase = Phase::Choosing;
        drop(gate);
        Finished::Ready(job)
    }
    pub fn start_write(&self, job: &Arc<ExportJob>) -> bool {
        let state = self.state.lock().unwrap_or_else(|p| p.into_inner());
        if !state
            .pending
            .as_ref()
            .is_some_and(|current| Arc::ptr_eq(current, job))
        {
            return false;
        }
        let mut gate = job.gate.lock().unwrap_or_else(|p| p.into_inner());
        if gate.phase != Phase::Choosing {
            return false;
        }
        gate.phase = Phase::Writing;
        true
    }
    pub fn release(&self, job: &Arc<ExportJob>) -> bool {
        let mut state = self.state.lock().unwrap_or_else(|p| p.into_inner());
        if !state
            .pending
            .as_ref()
            .is_some_and(|current| Arc::ptr_eq(current, job))
        {
            return false;
        }
        state.pending = None;
        true
    }
    pub fn cancel(&self, job: &Arc<ExportJob>) -> bool {
        job.cancel();
        self.release(job)
    }
    /// Reserve only an idle export subsystem; never cancel a user's pending save.
    pub fn reserve_update(&self, id: u64) -> bool {
        let mut state = self.state.lock().unwrap_or_else(|p| p.into_inner());
        if id == 0 || state.stopped || state.pending.is_some() {
            return false;
        }
        match state.update_reservation {
            Some(current) => current == id,
            None => {
                state.update_reservation = Some(id);
                true
            }
        }
    }
    pub fn release_update(&self, id: u64) -> bool {
        let mut state = self.state.lock().unwrap_or_else(|p| p.into_inner());
        if state.update_reservation != Some(id) {
            return false;
        }
        state.update_reservation = None;
        true
    }
    pub fn shutdown(&self) {
        let job = {
            let mut state = self.state.lock().unwrap_or_else(|p| p.into_inner());
            state.stopped = true;
            state.pending.take()
        };
        if let Some(job) = job {
            job.cancel();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    fn nonce() -> &'static str {
        "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
    }
    fn ready(exports: &ExportDownloads, url: &str, bytes: &[u8]) -> Arc<ExportJob> {
        let payload = exports.begin(url, nonce(), "meeting.txt").unwrap();
        assert!(!payload.exists());
        fs::write(payload, bytes).unwrap();
        match exports.finished(url, true) {
            Finished::Ready(job) => job,
            _ => panic!("expected ready export"),
        }
    }
    #[test]
    fn allowed_routes_and_nonce_do_not_admit_audio_or_other_downloads() {
        assert!(allowed_path(&format!("/api/jobs/{}/export/txt", nonce())));
        assert!(allowed_path("/api/notices"));
        for path in [
            "/api/jobs/../export/txt",
            "/api/jobs/a/audio",
            "/api/jobs/a/export/txt",
            "/api/jobs/aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa/export/exe",
            "/api/notices/extra",
        ] {
            assert!(!allowed_path(path));
        }
        for value in [
            "",
            "../audio",
            "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA",
            "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa;",
        ] {
            assert!(!valid_nonce(value));
        }
    }
    #[test]
    fn one_owner_survives_duplicate_finish_and_cancel_then_retries() {
        let exports = ExportDownloads::default();
        let job = ready(&exports, "first", b"private words");
        assert!(exports.begin("second", nonce(), "next.txt").is_err());
        assert!(matches!(exports.finished("first", true), Finished::Ignored));
        assert!(matches!(
            exports.finished("stale", false),
            Finished::Ignored
        ));
        let directory = job.staging.0.clone();
        assert!(exports.cancel(&job));
        assert!(!directory.exists());
        let next = ready(&exports, "next", b"retry");
        assert!(!exports.cancel(&job));
        assert!(exports.start_write(&next));
        exports.cancel(&next);
    }
    #[test]
    fn failed_or_oversized_download_cleans_only_owned_staging() {
        let exports = ExportDownloads::default();
        let payload = exports.begin("failure", nonce(), "meeting.txt").unwrap();
        let parent = payload.parent().unwrap().to_owned();
        assert!(matches!(
            exports.finished("failure", false),
            Finished::Failed(_)
        ));
        assert!(!parent.exists());
        let payload = exports.begin("large", nonce(), "meeting.txt").unwrap();
        File::create(&payload)
            .unwrap()
            .set_len(MAX_EXPORT_BYTES + 1)
            .unwrap();
        assert!(matches!(
            exports.finished("large", true),
            Finished::Failed(_)
        ));
        assert!(!payload.exists());
    }
    #[test]
    fn successful_save_and_empty_save_are_atomic_and_leave_no_sibling() {
        let folder = PrivateDirectory::create(&std::env::temp_dir()).unwrap();
        let target = folder.0.join("saved.txt");
        fs::write(&target, b"old file").unwrap();
        for bytes in [b"new private words".as_slice(), b"".as_slice()] {
            let exports = ExportDownloads::default();
            let job = ready(&exports, "save", bytes);
            assert!(exports.start_write(&job));
            job.save_to(&target).unwrap();
            assert_eq!(fs::read(&target).unwrap(), bytes);
            assert!(exports.release(&job));
            assert_eq!(fs::read_dir(&folder.0).unwrap().count(), 1);
        }
    }
    #[test]
    fn failed_write_preserves_existing_destination_and_cleans_sibling() {
        let folder = PrivateDirectory::create(&std::env::temp_dir()).unwrap();
        let target = folder.0.join("saved.txt");
        fs::write(&target, b"existing important file").unwrap();
        let exports = ExportDownloads::default();
        let job = ready(&exports, "save", b"replacement");
        assert!(exports.start_write(&job));
        fs::remove_file(job.payload()).unwrap();
        assert!(job.save_to(&target).is_err());
        exports.cancel(&job);
        assert_eq!(fs::read(&target).unwrap(), b"existing important file");
        assert_eq!(fs::read_dir(&folder.0).unwrap().count(), 1);
    }
    #[test]
    fn shutdown_during_download_or_dialog_rejects_late_callbacks() {
        for dialog in [false, true] {
            let exports = ExportDownloads::default();
            let payload = exports.begin("close", nonce(), "meeting.txt").unwrap();
            fs::write(&payload, b"words").unwrap();
            let held = if dialog {
                match exports.finished("close", true) {
                    Finished::Ready(job) => Some(job),
                    _ => panic!(),
                }
            } else {
                None
            };
            exports.shutdown();
            assert!(!payload.exists());
            assert!(matches!(exports.finished("close", true), Finished::Ignored));
            if let Some(job) = held {
                assert!(!exports.start_write(&job));
            }
            assert!(exports.begin("new", nonce(), "meeting.txt").is_err());
        }
    }
    #[test]
    fn shutdown_during_write_preserves_old_file_and_removes_sibling() {
        let folder = PrivateDirectory::create(&std::env::temp_dir()).unwrap();
        let target = folder.0.join("saved.txt");
        fs::write(&target, b"old").unwrap();
        let exports = ExportDownloads::default();
        let job = ready(&exports, "save", b"new");
        assert!(exports.start_write(&job));
        let sibling = PrivateDirectory::create(&folder.0).unwrap();
        fs::write(sibling.0.join("payload"), b"partial").unwrap();
        job.gate.lock().unwrap().sibling = Some(sibling.0.clone());
        exports.shutdown();
        assert!(!sibling.0.exists());
        assert!(job.save_to(&target).is_err());
        assert_eq!(fs::read(target).unwrap(), b"old");
    }
    #[test]
    fn recording_storage_cannot_be_overwritten_by_export() {
        let folder = PrivateDirectory::create(&std::env::temp_dir()).unwrap();
        let target = folder.0.join("audio.wav");
        fs::write(&target, b"original audio").unwrap();
        let exports = ExportDownloads::default();
        exports.protect_storage(&folder.0).unwrap();
        let job = ready(&exports, "save", b"text");
        assert!(exports.start_write(&job));
        assert!(job.save_to(&target).is_err());
        exports.cancel(&job);
        assert_eq!(fs::read(target).unwrap(), b"original audio");
    }
    #[test]
    fn update_refuses_download_chooser_and_write_without_cancelling_export() {
        for phase in [Phase::Downloading, Phase::Choosing, Phase::Writing] {
            let exports = ExportDownloads::default();
            let payload = exports.begin("save", nonce(), "meeting.txt").unwrap();
            fs::write(&payload, b"private transcript").unwrap();
            let job = exports.state.lock().unwrap().pending.clone().unwrap();
            job.gate.lock().unwrap().phase = phase;
            assert!(!exports.reserve_update(7));
            assert_eq!(fs::read(&payload).unwrap(), b"private transcript");
            assert_eq!(job.gate.lock().unwrap().phase, phase);
            assert!(exports.cancel(&job));
            assert!(exports.reserve_update(7));
        }
    }
    #[test]
    fn update_reservation_blocks_exports_and_stale_release_cannot_open_gate() {
        let exports = ExportDownloads::default();
        assert!(!exports.reserve_update(0));
        assert!(exports.reserve_update(4));
        assert!(exports.reserve_update(4));
        assert!(!exports.reserve_update(5));
        assert!(exports.begin("save", nonce(), "meeting.txt").is_err());
        assert!(!exports.release_update(3));
        assert!(exports.begin("save", nonce(), "meeting.txt").is_err());
        assert!(exports.release_update(4));
        let job = ready(&exports, "save", b"after update cancellation");
        assert!(exports.cancel(&job));
        assert!(exports.reserve_update(5));
        assert!(!exports.release_update(4));
        assert!(exports.begin("save", nonce(), "meeting.txt").is_err());
        assert!(exports.release_update(5));
    }
}
