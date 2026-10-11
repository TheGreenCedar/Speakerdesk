//! Model downloads, Range resume, and the setup state machine.
//!
//! Mirrors the admission and disk checks in `speakerdesk/model_setup.py`.
//! A partial file is resumed with an HTTP Range at its length. Nothing here
//! sends that request.

use std::io;
use std::path::Path;

use reqwest::header::RANGE;
use speakerdesk_core::{limits, messages};
use thiserror::Error;

/// Bytes observed while a model download is in progress.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DownloadProgress {
    pub downloaded_bytes: u64,
    pub total_bytes: u64,
}

impl DownloadProgress {
    pub const fn new(downloaded_bytes: u64, total_bytes: u64) -> Self {
        Self {
            downloaded_bytes,
            total_bytes,
        }
    }
}

/// Setup lifecycle. `Downloading` carries progress; `Failed` keeps the user-facing message.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub enum SetupState {
    #[default]
    Idle,
    Downloading {
        progress: DownloadProgress,
    },
    Ready,
    Failed {
        message: String,
    },
}

/// Gates checked before `POST /api/setup` starts a download.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SetupGates {
    /// Apple Silicon Mac running macOS 15 or later.
    pub apple_silicon: bool,
    /// A meeting or transcription currently holds the runtime.
    pub meeting_busy: bool,
    /// Voice model setup is already downloading.
    pub voice_setup_downloading: bool,
}

impl SetupGates {
    /// No admission blocker. Disk space is checked separately.
    pub const fn clear() -> Self {
        Self {
            apple_silicon: true,
            meeting_busy: false,
            voice_setup_downloading: false,
        }
    }
}

/// Result of asking setup to start.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Start {
    Started,
    AlreadyDownloading,
}

/// Admission or disk refusal. The text is a frozen v0.6.6 message.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Error)]
pub enum SetupError {
    #[error("{0}")]
    Refused(&'static str),
}

/// Owned setup state. Callers supply free-byte counts so tests never touch the network.
#[derive(Debug, Default)]
pub struct SetupService {
    state: SetupState,
}

impl SetupService {
    pub fn new() -> Self {
        Self {
            state: SetupState::Idle,
        }
    }

    pub fn state(&self) -> &SetupState {
        &self.state
    }

    /// Admit a setup start in the same order as `model_setup.start`.
    ///
    /// Apple Silicon, a busy meeting, and an in-progress voice download are
    /// refused. An in-progress model download is left unchanged.
    pub fn start(&mut self, gates: SetupGates) -> Result<Start, SetupError> {
        if !gates.apple_silicon {
            return Err(SetupError::Refused(messages::NEEDS_APPLE_SILICON));
        }
        if gates.meeting_busy {
            return Err(SetupError::Refused(messages::SETUP_BUSY_MEETING));
        }
        if gates.voice_setup_downloading {
            return Err(SetupError::Refused(messages::WAIT_FOR_VOICE_SETUP));
        }
        if matches!(self.state, SetupState::Downloading { .. }) {
            return Ok(Start::AlreadyDownloading);
        }
        self.state = SetupState::Downloading {
            progress: DownloadProgress::new(0, 0),
        };
        Ok(Start::Started)
    }

    /// Record download progress while setup is downloading.
    pub fn record_progress(&mut self, progress: DownloadProgress) -> bool {
        if let SetupState::Downloading { progress: current } = &mut self.state {
            *current = progress;
            true
        } else {
            false
        }
    }

    pub fn finish(&mut self) {
        self.state = SetupState::Ready;
    }

    pub fn fail(&mut self, message: impl Into<String>) {
        self.state = SetupState::Failed {
            message: message.into(),
        };
    }

    /// Refuse setup when free space is under [`limits::SETUP_FREE_BYTES`].
    ///
    /// Failure stores [`messages::FREE_SPACE_SETUP`] on the state machine.
    /// A passing check leaves the current state in place.
    pub fn require_free_space(&mut self, free_bytes: u64) -> Result<(), SetupError> {
        if free_bytes < limits::SETUP_FREE_BYTES {
            self.fail(messages::FREE_SPACE_SETUP);
            Err(SetupError::Refused(messages::FREE_SPACE_SETUP))
        } else {
            Ok(())
        }
    }
}

/// Free bytes available to a normal user on the filesystem that holds `path`.
///
/// Matches `shutil.disk_usage(path).free` (`f_bavail`).
pub fn directory_free_bytes(path: &Path) -> io::Result<u64> {
    fs4::available_space(path)
}

/// How the next model request should continue a partial file.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ResumePlan {
    pub offset: u64,
    /// `bytes={offset}-` when `offset` is greater than zero.
    pub range: Option<String>,
}

impl ResumePlan {
    /// Header plan for a known partial length. Zero starts a fresh request.
    pub fn for_length(offset: u64) -> Self {
        let range = (offset > 0).then(|| format!("bytes={offset}-"));
        Self { offset, range }
    }

    /// Read a `.part` file and plan the next request at its length.
    ///
    /// A missing file starts at byte zero with no Range header, matching
    /// `headers={'Range': f'bytes={offset}-'} if offset else {}`.
    pub fn for_partial(path: &Path) -> io::Result<Self> {
        match std::fs::metadata(path) {
            Ok(metadata) => Ok(Self::for_length(metadata.len())),
            Err(error) if error.kind() == io::ErrorKind::NotFound => Ok(Self::for_length(0)),
            Err(error) => Err(error),
        }
    }

    pub fn range_header(&self) -> Option<&str> {
        self.range.as_deref()
    }

    /// Attach the Range header to a request builder. Does not send it.
    pub fn apply(&self, builder: reqwest::RequestBuilder) -> reqwest::RequestBuilder {
        match self.range_header() {
            Some(value) => builder.header(RANGE, value),
            None => builder,
        }
    }
}

#[cfg(test)]
mod tests {
    use std::fs::File;
    use std::path::PathBuf;
    use std::sync::atomic::{AtomicU64, Ordering};

    use super::{
        DownloadProgress, ResumePlan, SetupError, SetupGates, SetupService, SetupState, Start,
        directory_free_bytes,
    };
    use reqwest::header::RANGE;
    use speakerdesk_core::{limits, messages};

    static TEMP_DIRS: AtomicU64 = AtomicU64::new(0);

    struct TempDir(PathBuf);

    impl TempDir {
        fn new() -> Self {
            let id = TEMP_DIRS.fetch_add(1, Ordering::Relaxed);
            let path =
                std::env::temp_dir().join(format!("speakerdesk-setup-{}-{id}", std::process::id()));
            std::fs::create_dir_all(&path).unwrap();
            Self(path)
        }
    }

    impl Drop for TempDir {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.0);
        }
    }

    fn range_on_request(plan: &ResumePlan) -> Option<String> {
        let request = plan
            .apply(reqwest::Client::new().get("https://example.invalid/model.safetensors"))
            .build()
            .expect("building a request does not download");
        request
            .headers()
            .get(RANGE)
            .map(|value| value.to_str().unwrap().to_string())
    }

    #[test]
    fn preconditions_use_the_frozen_messages() {
        let mut setup = SetupService::new();
        assert_eq!(setup.state(), &SetupState::Idle);

        let unsupported = setup
            .start(SetupGates {
                apple_silicon: false,
                ..SetupGates::clear()
            })
            .unwrap_err();
        assert_eq!(
            unsupported,
            SetupError::Refused(messages::NEEDS_APPLE_SILICON)
        );
        assert_eq!(setup.state(), &SetupState::Idle);

        let busy = setup
            .start(SetupGates {
                meeting_busy: true,
                ..SetupGates::clear()
            })
            .unwrap_err();
        assert_eq!(busy, SetupError::Refused(messages::SETUP_BUSY_MEETING));

        let voice = setup
            .start(SetupGates {
                voice_setup_downloading: true,
                ..SetupGates::clear()
            })
            .unwrap_err();
        assert_eq!(voice, SetupError::Refused(messages::WAIT_FOR_VOICE_SETUP));
        assert_eq!(setup.state(), &SetupState::Idle);
    }

    #[test]
    fn earlier_preconditions_win_and_a_second_start_keeps_progress() {
        let mut setup = SetupService::new();
        let both = setup
            .start(SetupGates {
                apple_silicon: false,
                meeting_busy: true,
                voice_setup_downloading: true,
            })
            .unwrap_err();
        assert_eq!(both, SetupError::Refused(messages::NEEDS_APPLE_SILICON));

        assert_eq!(setup.start(SetupGates::clear()).unwrap(), Start::Started);
        let progress = DownloadProgress::new(12, 40);
        assert!(setup.record_progress(progress));
        assert_eq!(
            setup.start(SetupGates::clear()).unwrap(),
            Start::AlreadyDownloading
        );
        assert_eq!(setup.state(), &SetupState::Downloading { progress });

        let blocked = setup
            .start(SetupGates {
                voice_setup_downloading: true,
                ..SetupGates::clear()
            })
            .unwrap_err();
        assert_eq!(blocked, SetupError::Refused(messages::WAIT_FOR_VOICE_SETUP));
        assert_eq!(setup.state(), &SetupState::Downloading { progress });
    }

    #[test]
    fn free_space_under_the_reserve_fails_with_the_setup_message() {
        assert_eq!(limits::SETUP_FREE_BYTES, 2_400_000_000);
        let mut setup = SetupService::new();
        assert_eq!(setup.start(SetupGates::clear()).unwrap(), Start::Started);
        let progress = DownloadProgress::new(3, 9);
        assert!(setup.record_progress(progress));

        setup.require_free_space(limits::SETUP_FREE_BYTES).unwrap();
        assert_eq!(setup.state(), &SetupState::Downloading { progress });

        let error = setup
            .require_free_space(limits::SETUP_FREE_BYTES - 1)
            .unwrap_err();
        assert_eq!(error, SetupError::Refused(messages::FREE_SPACE_SETUP));
        assert_eq!(error.to_string(), messages::FREE_SPACE_SETUP);
        assert_eq!(
            setup.state(),
            &SetupState::Failed {
                message: messages::FREE_SPACE_SETUP.to_string(),
            }
        );
    }

    #[test]
    fn finish_and_fail_move_the_state_machine() {
        let mut setup = SetupService::new();
        assert!(!setup.record_progress(DownloadProgress::new(1, 1)));
        setup.start(SetupGates::clear()).unwrap();
        setup.finish();
        assert_eq!(setup.state(), &SetupState::Ready);
        setup.fail("Model verification failed. Retry the download.");
        assert_eq!(
            setup.state(),
            &SetupState::Failed {
                message: "Model verification failed. Retry the download.".to_string(),
            }
        );
        assert_eq!(setup.start(SetupGates::clear()).unwrap(), Start::Started);
        assert_eq!(
            setup.state(),
            &SetupState::Downloading {
                progress: DownloadProgress::new(0, 0),
            }
        );
    }

    #[test]
    fn partial_file_plans_a_range_header_at_its_length() {
        let fake_length = 7_u64;
        let dir = TempDir::new();
        let partial = dir.0.join("model.safetensors.part");
        let file = File::create(&partial).unwrap();
        file.set_len(fake_length).unwrap();
        drop(file);

        let plan = ResumePlan::for_partial(&partial).unwrap();
        assert_eq!(plan, ResumePlan::for_length(fake_length));
        assert_eq!(plan.range_header(), Some("bytes=7-"));
        assert_eq!(range_on_request(&plan).as_deref(), Some("bytes=7-"));

        let missing = dir.0.join("absent.part");
        let fresh = ResumePlan::for_partial(&missing).unwrap();
        assert_eq!(fresh.range_header(), None);
        assert_eq!(range_on_request(&fresh), None);

        let empty = dir.0.join("empty.part");
        File::create(&empty).unwrap();
        let empty_plan = ResumePlan::for_partial(&empty).unwrap();
        assert_eq!(empty_plan.offset, 0);
        assert_eq!(range_on_request(&empty_plan), None);
    }

    #[test]
    fn directory_free_bytes_reads_the_filesystem() {
        let dir = TempDir::new();
        let free = directory_free_bytes(&dir.0).unwrap();
        assert!(free > 0);
    }
}
