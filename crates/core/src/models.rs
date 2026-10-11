//! `core::models` — the `Models` seam and the parent↔inference-child IPC
//! protocol (plan §4.4).
//!
//! The Tauri host owns orchestration and persistence; one resident
//! `speakerdesk --inference` child owns MLX/Metal access. These types define
//! the wire contract and the synchronous trait the host drives — real
//! (child-process), `FakeModels` and `ReplayModels` implementations arrive
//! with M2/M3; there are no implementations here.
//!
//! Framing (§4.4): `u32` big-endian length + a postcard payload carrying
//! `request_id` and `Result<Payload, ChildError>`. Core stays serde-only —
//! every type derives `Serialize`/`Deserialize` over plain data, so a
//! postcard codec can sit on top without a dependency here.
//!
//! Guesses are marked `GUESS:` in doc comments.

use serde::{Deserialize, Serialize};

/// Correlates a request with its response; the parent's counter.
pub type RequestId = u64;

/// A live streaming handle returned by `vad_open`/`diar_open`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct StreamHandle(pub u64);

/// VAD conditioning — the gain/speech-admission policy the child applies to
/// `vad_feed` PCM. `None` on `vad_open` means the default policy.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct GainPolicy {
    /// SHA-256 of the admission contract the policy was minted against.
    pub contract_sha256: String,
    /// The capture source this policy binds (e.g. `microphone_clean`).
    pub capture_source: String,
    /// GUESS: the policy identity string (admission `input_policy`).
    pub input_policy: String,
}

/// Diarization mode for `diar_open` (§4.4 `Low|Offline`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DiarPreset {
    Low,
    Offline,
}

/// One VAD decision frame emitted by `vad_feed`: a sample range plus the
/// admission observations the engine needs (plan §4.4:
/// `{start,end,raw_p,norm_p,gain,constant}`).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct VadFrame {
    /// GUESS: start sample offset (16 kHz domain).
    pub start: i64,
    /// GUESS: end sample offset.
    pub end: i64,
    /// Raw model probability for the frame.
    pub raw_p: f32,
    /// Normalized (post-conditioning) probability.
    pub norm_p: f32,
    /// Applied gain factor.
    pub gain: f32,
    /// Constant-PCM override flag (flat input detected).
    pub constant: bool,
}

/// One diarization turn — `{start, end, speaker}` in seconds; `speaker` is
/// the model's track label (the host namespaces it per source).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Turn {
    pub start: f64,
    pub end: f64,
    /// GUESS: track label string as Nemotron emits it (e.g. `speaker_0`).
    pub speaker: String,
}

/// `diar_feed` result — newly emitted turns plus progress.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct DiarFeedResult {
    pub turns: Vec<Turn>,
    /// GUESS: audio consumed so far, in samples.
    pub frames_processed: i64,
}

/// `language_probe` result — the 99-way language score vector plus the
/// matching language codes (parallel arrays).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct LanguageProbeResult {
    /// Exactly [`LANGUAGE_COUNT`] scores.
    pub scores: Vec<f32>,
    /// Language codes parallel to `scores`.
    pub codes: Vec<String>,
}

/// Number of language-probe scores (the 99-language detector head).
pub const LANGUAGE_COUNT: usize = 99;

/// `transcribe` result.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TranscriptResult {
    /// Detokenized display text.
    pub text: String,
    /// GUESS: the raw model output before detokenization/cleanup.
    pub raw_text: String,
}

/// `ctc_scores` result — per-frame gathered log-probs (including the blank
/// token) used by reading-turn CTC segmentation.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CtcScoresResult {
    /// Number of frames scored.
    pub frames: i64,
    /// Gathered log-probabilities, frame-major (includes blank).
    /// GUESS: flat `Vec<f32>`; framing shape decided with the codec.
    pub log_probs: Vec<f32>,
}

/// Padding around a `transcribe` clip, in samples (each `0|3200` per §4.4).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct TranscribePadding {
    pub pre: u32,
    pub post: u32,
}

/// Voice-embedding input length (`EmbedVoice` — 6 s at 16 kHz).
pub const EMBED_VOICE_SAMPLES: usize = 96_000;
/// Voice-embedding output dimension (ReDimNet2-B6).
pub const EMBED_VOICE_DIM: usize = 192;

/// One model's `setup_check` verdict.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SetupCheckResult {
    /// Model identifier as passed in `setup_check`.
    pub model: String,
    pub ok: bool,
    /// GUESS: wall-clock check duration in seconds.
    pub seconds: f64,
    /// Present when `ok` is false.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error: Option<String>,
}

/// `metrics` result — child process memory highs.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ModelMetrics {
    /// Peak MLX (Metal) allocation in bytes.
    pub peak_mlx_bytes: u64,
    /// Peak resident set size in bytes.
    pub peak_rss_bytes: u64,
}

/// `hello` response — the child's build and the device it selected.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Ready {
    pub build_id: String,
    /// e.g. `mps`; there is no CPU neural fallback.
    pub device: String,
}

/// Child-reported failure carried in `Result<Payload, ChildError>`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ChildError {
    /// GUESS: machine-readable kind (`model_load`, `inference`, `io`, …);
    /// free-form until the child lands.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub kind: Option<String>,
    pub message: String,
    /// True when the request failed because `cancel` observed it.
    #[serde(default)]
    pub cancelled: bool,
}

impl std::fmt::Display for ChildError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.message)
    }
}

impl std::error::Error for ChildError {}

/// Every request the parent can send (one variant per §4.4 verb).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Request {
    /// Handshake — the child verifies `build_id` matches and replies `Ready`.
    Hello {
        build_id: String,
    },
    VadOpen {
        conditioning: Option<GainPolicy>,
    },
    VadFeed {
        h: StreamHandle,
        pcm: Vec<f32>,
        last: bool,
    },
    VadClose {
        h: StreamHandle,
    },
    DiarOpen {
        preset: DiarPreset,
    },
    DiarFeed {
        h: StreamHandle,
        pcm: Vec<f32>,
        last: bool,
    },
    DiarClose {
        h: StreamHandle,
    },
    DiarizeFile {
        path: std::path::PathBuf,
    },
    DiarizeWindow {
        pcm: Vec<f32>,
    },
    LanguageProbe {
        pcm: Vec<f32>,
    },
    Transcribe {
        pcm: Vec<f32>,
        /// Language code or `auto`.
        language: String,
        padding: TranscribePadding,
    },
    CtcScores {
        pcm: Vec<f32>,
        token_ids: Vec<i64>,
    },
    /// Exactly [`EMBED_VOICE_SAMPLES`] samples.
    EmbedVoice {
        pcm: Vec<f32>,
    },
    /// Per-model checks; `models` are the model identifiers to verify.
    SetupCheck {
        models: Vec<String>,
    },
    Metrics,
    /// Cancel the in-flight request with this id.
    Cancel {
        request_id: RequestId,
    },
    Shutdown,
}

/// Every successful response payload.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Payload {
    Ready(Ready),
    /// New stream handle (`vad_open`/`diar_open`/`diar_close` acked with a
    /// plain `Ok` — GUESS: close returns `Ok`, not a payload).
    Handle(StreamHandle),
    VadFrames(Vec<VadFrame>),
    DiarTurns(DiarFeedResult),
    Turns(Vec<Turn>),
    LanguageProbe(LanguageProbeResult),
    Transcribed(TranscriptResult),
    CtcScores(CtcScoresResult),
    /// [`EMBED_VOICE_DIM`]-length embedding.
    Embedding(Vec<f32>),
    SetupReport(Vec<SetupCheckResult>),
    Metrics(ModelMetrics),
    /// Ack for `cancel`/`shutdown`/stream closes.
    Ok,
}

/// On-the-wire request frame — `u32` big-endian length prefix (added by the
/// transport) + postcard-serialized body.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RequestFrame {
    pub request_id: RequestId,
    pub request: Request,
}

/// On-the-wire response frame.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ResponseFrame {
    pub request_id: RequestId,
    pub result: Result<Payload, ChildError>,
}

/// Failure at the seam: either a child-reported `ChildError` or a transport
/// problem (dead child, framing error, timeout — surfaced by the host-side
/// implementation, not this type).
#[derive(Debug, thiserror::Error)]
pub enum ModelsError {
    #[error("inference child: {0}")]
    Child(#[from] ChildError),
    #[error("inference transport: {0}")]
    Transport(String),
}

/// The synchronous model seam (plan §4.4). One method per request kind;
/// stream state lives inside the implementation behind the returned
/// `StreamHandle`s. The real implementation multiplexes onto a single MLX
/// executor thread inside the child; cancellation is routed to an atomic
/// flag and per-request timeouts/child supervision live in the host.
///
/// GUESS: methods take `&mut self` (single-owner sequencing); `Send` bound
/// left to implementations.
pub trait Models {
    fn hello(&mut self, build_id: &str) -> Result<Ready, ModelsError>;
    fn vad_open(&mut self, conditioning: Option<GainPolicy>) -> Result<StreamHandle, ModelsError>;
    fn vad_feed(
        &mut self,
        h: StreamHandle,
        pcm: &[f32],
        last: bool,
    ) -> Result<Vec<VadFrame>, ModelsError>;
    fn vad_close(&mut self, h: StreamHandle) -> Result<(), ModelsError>;
    fn diar_open(&mut self, preset: DiarPreset) -> Result<StreamHandle, ModelsError>;
    fn diar_feed(
        &mut self,
        h: StreamHandle,
        pcm: &[f32],
        last: bool,
    ) -> Result<DiarFeedResult, ModelsError>;
    fn diar_close(&mut self, h: StreamHandle) -> Result<(), ModelsError>;
    fn diarize_file(&mut self, path: &std::path::Path) -> Result<Vec<Turn>, ModelsError>;
    fn diarize_window(&mut self, pcm: &[f32]) -> Result<Vec<Turn>, ModelsError>;
    fn language_probe(&mut self, pcm: &[f32]) -> Result<LanguageProbeResult, ModelsError>;
    fn transcribe(
        &mut self,
        pcm: &[f32],
        language: &str,
        padding: TranscribePadding,
    ) -> Result<TranscriptResult, ModelsError>;
    fn ctc_scores(
        &mut self,
        pcm: &[f32],
        token_ids: &[i64],
    ) -> Result<CtcScoresResult, ModelsError>;
    /// `pcm` must be exactly [`EMBED_VOICE_SAMPLES`] samples; the result is
    /// [`EMBED_VOICE_DIM`]-long.
    fn embed_voice(&mut self, pcm: &[f32]) -> Result<Vec<f32>, ModelsError>;
    fn setup_check(&mut self, models: &[String]) -> Result<Vec<SetupCheckResult>, ModelsError>;
    fn metrics(&mut self) -> Result<ModelMetrics, ModelsError>;
    fn cancel(&mut self, request_id: RequestId) -> Result<(), ModelsError>;
    fn shutdown(&mut self) -> Result<(), ModelsError>;
}
