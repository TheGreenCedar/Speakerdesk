use std::path::PathBuf;
use std::time::Duration;

use serde::{Deserialize, Serialize};

use crate::error::ChildError;

pub const MAX_FRAME_BYTES: usize = 64 * 1024 * 1024;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct Handle {
    pub generation: u64,
    pub session: u64,
}

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
    /// Absolute sample coordinates within this session, never packet offsets.
    pub start: u64,
    pub end: u64,
    pub raw_p: f32,
    pub norm_p: f32,
    pub gain: f32,
    pub constant: bool,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct VadFeedOut {
    pub frames: Vec<VadFrame>,
    /// PCM accepted, including the incomplete frame buffered for the next feed.
    pub received: u64,
    /// Exclusive endpoint of contiguous neural observations.
    pub observed: u64,
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
    pub token_count: u32,
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
    Ready {
        build_id: String,
        device: String,
        models: Vec<String>,
    },
    Handle(Handle),
    Vad(VadFeedOut),
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

pub fn read_envelope<R: std::io::Read>(reader: &mut R) -> std::io::Result<Envelope> {
    let mut length_bytes = [0u8; 4];
    reader.read_exact(&mut length_bytes[..1])?;
    reader
        .read_exact(&mut length_bytes[1..])
        .map_err(|error| std::io::Error::new(std::io::ErrorKind::InvalidData, error))?;
    let length = u32::from_be_bytes(length_bytes) as usize;
    if length > MAX_FRAME_BYTES {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            "frame exceeds 64 MiB",
        ));
    }
    let mut payload = vec![0u8; length];
    reader
        .read_exact(&mut payload)
        .map_err(|error| std::io::Error::new(std::io::ErrorKind::InvalidData, error))?;
    decode_envelope(&payload)
}

pub fn write_envelope<W: std::io::Write>(
    writer: &mut W,
    envelope: Envelope,
) -> std::io::Result<()> {
    let payload = postcard::to_allocvec(&envelope)
        .map_err(|error| std::io::Error::new(std::io::ErrorKind::InvalidData, error.to_string()))?;
    if payload.len() > MAX_FRAME_BYTES {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            "frame exceeds 64 MiB",
        ));
    }
    let length = u32::try_from(payload.len())
        .map_err(|error| std::io::Error::new(std::io::ErrorKind::InvalidData, error.to_string()))?;
    writer.write_all(&length.to_be_bytes())?;
    writer.write_all(&payload)?;
    writer.flush()?;
    Ok(())
}

pub fn decode_envelope(payload: &[u8]) -> std::io::Result<Envelope> {
    let (envelope, trailing) = postcard::take_from_bytes(payload)
        .map_err(|error| std::io::Error::new(std::io::ErrorKind::InvalidData, error.to_string()))?;
    if !trailing.is_empty() {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            "trailing bytes in inference frame",
        ));
    }
    Ok(envelope)
}
