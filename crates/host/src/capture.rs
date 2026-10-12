//! Owned Swift capture transport and bounded, disk-backed source alignment.

mod processing;
#[cfg(test)]
mod replay_tests;
mod transport;
mod writer;

pub use processing::ProcessingReceipt;
pub use transport::{CaptureSession, CaptureSupervisor};
pub use writer::{CaptureConfig, CapturePhase, CaptureSnapshot};

use std::time::Duration;

use serde::Serialize;
use serde_json::Value;
use speakerdesk_core::{limits, messages};
use thiserror::Error;

/// `capture_processing.PROCESSED`: qualified Apple processed microphone.
pub const PROCESSED: &str = "qualified_apple_active_processed_microphone";

/// Bound from `MeetingManager._stop_capture` (`threading.Timer(15, ...)`).
pub const STALL_STOP: Duration = Duration::from_secs(15);

/// Stdin commands from `desktop/capture/MeetingCapture.swift`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum Command {
    Start {
        microphone: bool,
        system: bool,
        #[serde(skip_serializing_if = "Option::is_none")]
        job_id: Option<String>,
        #[serde(skip_serializing_if = "Option::is_none")]
        capture_export_contract: Option<String>,
    },
    Pause,
    Resume,
    Stop,
}

impl Command {
    pub fn start(microphone: bool, system: bool) -> Self {
        Self::Start {
            microphone,
            system,
            job_id: None,
            capture_export_contract: None,
        }
    }

    /// One JSON object plus the newline the helper reads with `readLine()`.
    pub fn to_line(&self) -> Result<String, serde_json::Error> {
        let mut line = serde_json::to_string(self)?;
        line.push('\n');
        Ok(line)
    }
}

/// Clock-tick disk guard from `live_meeting._run`.
///
/// Returns [`messages::RECORDING_DISK_LOW`] when `free_bytes` is under
/// [`limits::DISK_GUARD_FREE_BYTES`].
pub fn disk_guard_message(free_bytes: u64) -> Option<&'static str> {
    if free_bytes < limits::DISK_GUARD_FREE_BYTES {
        Some(messages::RECORDING_DISK_LOW)
    } else {
        None
    }
}

/// JSON `type` values the desktop capture helper emits.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EventKind {
    CaptureProcessing,
    Audio,
    CaptureRuntime,
    Clock,
    FormatChanged,
    Recording,
    Paused,
    Stopped,
    Error,
    HelperCheck,
}

impl EventKind {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::CaptureProcessing => "capture_processing",
            Self::Audio => "audio",
            Self::CaptureRuntime => "capture_runtime",
            Self::Clock => "clock",
            Self::FormatChanged => "format_changed",
            Self::Recording => "recording",
            Self::Paused => "paused",
            Self::Stopped => "stopped",
            Self::Error => "error",
            Self::HelperCheck => "helper_check",
        }
    }

    fn parse(name: &str) -> Option<Self> {
        Some(match name {
            "capture_processing" => Self::CaptureProcessing,
            "audio" => Self::Audio,
            "capture_runtime" => Self::CaptureRuntime,
            "clock" => Self::Clock,
            "format_changed" => Self::FormatChanged,
            "recording" => Self::Recording,
            "paused" => Self::Paused,
            "stopped" => Self::Stopped,
            "error" => Self::Error,
            "helper_check" => Self::HelperCheck,
            _ => return None,
        })
    }
}

/// One parsed helper stdout line. The body keeps the original JSON object.
#[derive(Debug, Clone, PartialEq)]
pub struct CaptureEvent {
    kind: EventKind,
    body: Value,
}

impl CaptureEvent {
    pub fn kind(&self) -> EventKind {
        self.kind
    }

    pub fn body(&self) -> &Value {
        &self.body
    }
}

#[derive(Debug, Error, PartialEq, Eq)]
pub enum CaptureError {
    #[error("capture event line is not JSON")]
    Json,
    #[error("capture event line has no type")]
    MissingType,
    #[error("unknown capture event {0}")]
    UnknownEvent(String),
    #[error("{0}")]
    Protocol(String),
}

impl From<std::io::Error> for CaptureError {
    fn from(error: std::io::Error) -> Self {
        Self::Protocol(error.to_string())
    }
}

impl From<hound::Error> for CaptureError {
    fn from(error: hound::Error) -> Self {
        Self::Protocol(error.to_string())
    }
}

/// Parses one helper JSON line. Unknown `type` values are rejected.
pub fn parse_event(line: &str) -> Result<CaptureEvent, CaptureError> {
    let body: Value = serde_json::from_str(line.trim()).map_err(|_| CaptureError::Json)?;
    let name = body
        .get("type")
        .and_then(Value::as_str)
        .ok_or(CaptureError::MissingType)?;
    let kind =
        EventKind::parse(name).ok_or_else(|| CaptureError::UnknownEvent(name.to_string()))?;
    Ok(CaptureEvent { kind, body })
}

#[cfg(test)]
mod tests {
    use speakerdesk_core::{limits, messages};

    use super::{Command, EventKind, disk_guard_message, parse_event};

    #[test]
    fn parses_one_recording_json_event() {
        let event = parse_event(r#"{"type":"recording","microphone":true,"system":true}"#)
            .expect("recording line");
        assert_eq!(event.kind(), EventKind::Recording);
        assert_eq!(event.body()["microphone"], true);
        assert_eq!(event.body()["system"], true);
    }

    #[test]
    fn disk_guard_reports_the_low_space_message_under_the_limit() {
        assert_eq!(
            disk_guard_message(limits::DISK_GUARD_FREE_BYTES - 1),
            Some(messages::RECORDING_DISK_LOW)
        );
        assert_eq!(disk_guard_message(limits::DISK_GUARD_FREE_BYTES), None);
    }

    #[test]
    fn commands_are_json_lines() {
        let start = Command::start(true, false);
        assert_eq!(
            start.to_line().expect("start"),
            "{\"type\":\"start\",\"microphone\":true,\"system\":false}\n"
        );
        assert_eq!(
            Command::Pause.to_line().expect("pause"),
            "{\"type\":\"pause\"}\n"
        );
        assert_eq!(
            Command::Resume.to_line().expect("resume"),
            "{\"type\":\"resume\"}\n"
        );
        assert_eq!(
            Command::Stop.to_line().expect("stop"),
            "{\"type\":\"stop\"}\n"
        );
    }
}
