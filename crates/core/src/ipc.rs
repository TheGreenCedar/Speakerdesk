use std::path::PathBuf;
use std::time::Duration;

use serde::{Deserialize, Serialize};

use crate::error::ChildError;

pub const MAX_FRAME_BYTES: usize = 64 * 1024 * 1024;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct Handle(pub u64);

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct GainPolicy {
    /// When set, Silero's normalized view uses `min(256, max(1, 0.25 / peak))`.
    pub peak: Option<f32>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DiarPreset {
    Low,
    Offline,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct VadFrame {
    pub start: u64,
    pub end: u64,
    pub raw_p: f32,
    pub norm_p: f32,
    pub gain: f32,
    pub constant: bool,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Turn {
    pub start: f64,
    pub end: f64,
    pub speaker: String,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct DiarFeedOut {
    pub turns: Vec<Turn>,
    pub frames_processed: u64,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct LanguageProbeOut {
    pub probabilities: Vec<f32>,
    pub codes: Vec<String>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct AsrText {
    pub text: String,
    pub raw_text: String,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CtcScoresOut {
    pub frames: u32,
    pub gathered: Vec<f32>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SetupModelReport {
    pub model: String,
    pub ok: bool,
    pub timing_ms: u64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct MetricsOut {
    pub peak_mlx_bytes: u64,
    pub rss_bytes: u64,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum Request {
    Hello {
        build_id: String,
    },
    VadOpen {
        conditioning: Option<GainPolicy>,
    },
    VadFeed {
        handle: Handle,
        pcm: Vec<f32>,
        last: bool,
    },
    VadClose {
        handle: Handle,
    },
    DiarOpen {
        preset: DiarPreset,
    },
    DiarFeed {
        handle: Handle,
        pcm: Vec<f32>,
        last: bool,
    },
    DiarClose {
        handle: Handle,
    },
    DiarizeFile {
        path: PathBuf,
    },
    DiarizeWindow {
        pcm: Vec<f32>,
    },
    LanguageProbe {
        pcm: Vec<f32>,
    },
    Transcribe {
        pcm: Vec<f32>,
        language: String,
        padding: (u32, u32),
    },
    CtcScores {
        pcm: Vec<f32>,
        token_ids: Vec<i32>,
    },
    EmbedVoice {
        pcm: Vec<f32>,
    },
    SetupCheck {
        models: Vec<String>,
    },
    Metrics,
    Cancel {
        request_id: u64,
    },
    Shutdown,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum Response {
    Ready { build_id: String, device: String },
    Handle(Handle),
    Vad(Vec<VadFrame>),
    Diar(DiarFeedOut),
    Turns(Vec<Turn>),
    Probe(LanguageProbeOut),
    Text(AsrText),
    Ctc(CtcScoresOut),
    Embedding(Vec<f32>),
    Setup(Vec<SetupModelReport>),
    Metrics(MetricsOut),
    Ack,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Envelope {
    pub request_id: u64,
    pub body: EnvelopeBody,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum EnvelopeBody {
    Request(Request),
    Response(Result<Response, ChildError>),
}

/// `None` means the supervisor scales the limit with audio duration (`DiarizeFile`).
pub fn request_timeout(request: &Request) -> Option<Duration> {
    let seconds = match request {
        Request::Transcribe { .. } => 60,
        Request::LanguageProbe { .. } => 10,
        Request::CtcScores { .. } => 30,
        Request::EmbedVoice { .. } => 10,
        Request::VadFeed { .. } | Request::DiarFeed { .. } => 10,
        Request::SetupCheck { .. } => 120,
        Request::DiarizeFile { .. } => return None,
        Request::Hello { .. }
        | Request::VadOpen { .. }
        | Request::VadClose { .. }
        | Request::DiarOpen { .. }
        | Request::DiarClose { .. }
        | Request::DiarizeWindow { .. }
        | Request::Metrics
        | Request::Cancel { .. }
        | Request::Shutdown => 10,
    };
    Some(Duration::from_secs(seconds))
}
