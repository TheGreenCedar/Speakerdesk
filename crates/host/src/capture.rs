//! Swift helper supervisor.
//!
//! Stdin commands stay `start`, `pause`, `resume`, and `stop`. Stdout events
//! match the desktop helper's JSON lines. This module does not spawn Swift.
//!
//! `speakerdesk-echo` is not a host dependency. Route classification stays a
//! pure function here; the echo mixer is linked later.

use std::time::Duration;

use serde::Serialize;
use serde_json::Value;
use speakerdesk_core::{limits, messages};
use thiserror::Error;

/// `capture_processing.PROCESSED`: qualified Apple processed microphone.
pub const PROCESSED: &str = "qualified_apple_active_processed_microphone";

/// `capture_processing.LEGACY`: unknown or raw microphone route.
pub const LEGACY: &str = "legacy_unknown_or_raw";

/// Bound from `MeetingManager._stop_capture` (`threading.Timer(15, ...)`).
pub const STALL_STOP: Duration = Duration::from_secs(15);

/// JSON-lines session with the desktop capture helper.
///
/// The stall bound is [`StallDeadline`]. This type does not launch the helper.
#[derive(Debug, Default)]
pub struct CaptureSupervisor;

impl CaptureSupervisor {
    pub fn new() -> Self {
        Self
    }
}

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

/// How the host retains the microphone after a processing classification.
///
/// The mixer is linked later. This crate does not construct an echo canceller.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MicrophoneRoute {
    /// `PROCESSED` means `microphone_clean = microphone`, with no second AEC pass.
    CopyMicrophone,
    /// Any other classification still needs acoustic echo cancellation.
    NeedsAec,
}

/// `PROCESSED` copies the microphone. Every other classification needs AEC.
pub fn classify_route(classification: &str) -> MicrophoneRoute {
    if classification == PROCESSED {
        MicrophoneRoute::CopyMicrophone
    } else {
        MicrophoneRoute::NeedsAec
    }
}

/// Pure stall check: a stop request is late once `elapsed` reaches 15 seconds.
pub fn stop_deadline_reached(elapsed: Duration) -> bool {
    elapsed >= STALL_STOP
}

/// First `arm` starts the 15 second stop bound. A later `arm` leaves it in place.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct StallDeadline {
    armed_at: Option<Duration>,
}

impl StallDeadline {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn arm(&mut self, now: Duration) {
        if self.armed_at.is_none() {
            self.armed_at = Some(now);
        }
    }

    pub fn reached(&self, now: Duration) -> bool {
        self.armed_at
            .is_some_and(|armed_at| stop_deadline_reached(now.saturating_sub(armed_at)))
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
    use std::time::Duration;

    use speakerdesk_core::{limits, messages};

    use super::{
        Command, EventKind, LEGACY, MicrophoneRoute, PROCESSED, STALL_STOP, StallDeadline,
        classify_route, disk_guard_message, parse_event, stop_deadline_reached,
    };

    #[test]
    fn processed_route_copies_microphone_and_other_routes_need_aec() {
        assert_eq!(classify_route(PROCESSED), MicrophoneRoute::CopyMicrophone);
        assert_eq!(classify_route(LEGACY), MicrophoneRoute::NeedsAec);
        assert_eq!(classify_route(""), MicrophoneRoute::NeedsAec);
    }

    #[test]
    fn stall_deadline_trips_at_fifteen_seconds_and_does_not_reset() {
        assert!(!stop_deadline_reached(STALL_STOP - Duration::from_nanos(1)));
        assert!(stop_deadline_reached(STALL_STOP));
        assert!(stop_deadline_reached(Duration::from_secs(16)));

        let mut deadline = StallDeadline::new();
        assert!(!deadline.reached(Duration::from_secs(100)));
        deadline.arm(Duration::from_secs(10));
        deadline.arm(Duration::from_secs(20));
        assert!(!deadline.reached(Duration::from_secs(24)));
        assert!(deadline.reached(Duration::from_secs(25)));
    }

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
