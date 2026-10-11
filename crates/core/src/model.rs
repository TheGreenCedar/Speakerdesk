//! Typed product model — the single representation used by `edit`, `store`,
//! `transcript` and `review`. Field names and presence match the v0.6.6
//! persisted payload exactly: `Option` + `skip_serializing_if` is
//! absent-when-unset, `Nullable<T>` distinguishes an explicit JSON `null`
//! from an absent key, and serde-visible fields are declared in alphabetical
//! order so serialization matches the sorted-key JSON the oracle emits.
//!
//! Deny rules: types reachable only through persisted rows deny unknown
//! fields (the ledger strip + census have already run on those), while types
//! reachable from client input (`Document`, `Segment` and their nested
//! structs) tolerate unknown keys — v0.6.6 ignores them at the API boundary
//! and the server restores server-owned fields afterwards.
//!
//! The only opaque `serde_json` fields are the ones ruled opaque in the
//! field ledger: `Segment::provenance`, `Segment::speaker_track_mapping`,
//! `SpeakerAssignment::evidence`, `VoiceSuggestion::{match_,evidence}` and
//! the engine-state maps in `JobPrivate`.

use std::collections::BTreeMap;

use serde::de::Deserializer;
use serde::ser::Serializer;
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};

// ---------------------------------------------------------------------------
// Nullable<T> — three-state field: absent / explicit null / value.

/// `null`-valued keys v0.6.6 writes that must not collapse into "absent".
#[derive(Debug, Clone, Default, PartialEq)]
pub enum Nullable<T> {
    #[default]
    Absent,
    Null,
    Value(T),
}

impl<T> Nullable<T> {
    pub fn is_absent(&self) -> bool {
        matches!(self, Self::Absent)
    }

    /// The stored value, if any (`Null` and `Absent` both yield `None`).
    pub fn as_option(&self) -> Option<&T> {
        match self {
            Self::Value(v) => Some(v),
            _ => None,
        }
    }

    pub fn as_mut_option(&mut self) -> Option<&mut T> {
        match self {
            Self::Value(v) => Some(v),
            _ => None,
        }
    }

    pub fn of(value: T) -> Self {
        Self::Value(value)
    }
}

impl<T: Serialize> Serialize for Nullable<T> {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        match self {
            Self::Value(v) => v.serialize(serializer),
            Self::Null => serializer.serialize_unit(),
            Self::Absent => serializer.serialize_none(),
        }
    }
}

impl<'de, T: Deserialize<'de>> Deserialize<'de> for Nullable<T> {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        // serde calls visit_none for `null`; an absent key uses the field
        // `default` instead and never reaches here.
        Option::<T>::deserialize(deserializer).map(|v| match v {
            Some(v) => Self::Value(v),
            None => Self::Null,
        })
    }
}

/// Lenient `Option<T>` deserialization: the raw `Value` is read first (so
/// stream position stays valid), then converted — a wrong-typed value
/// degrades to `None` instead of failing the whole document.
///
/// Used only on fields the server *always* overwrites on write paths —
/// the PUT restore list and the recomputed fields (`machine_revision`,
/// `protected_fields`, `refinement_state`, `voice_eligible`,
/// `speaker_candidates`, `retry_candidate`). v0.6.6 pops or restores those
/// regardless of what the client sent, so the client value is dead data
/// and must not reject the request. Client-preserved fields stay strict:
/// malformed data there is a type error, not content.
fn de_opt_lenient<'de, D, T>(deserializer: D) -> Result<Option<T>, D::Error>
where
    D: Deserializer<'de>,
    T: Deserialize<'de>,
{
    let value = Value::deserialize(deserializer)?;
    Ok(T::deserialize(value).ok())
}

/// Lenient `Nullable<T>` variant, for server-recomputed fields that v0.6.6
/// may write as explicit `null`.
fn de_nullable_lenient<'de, D, T>(deserializer: D) -> Result<Nullable<T>, D::Error>
where
    D: Deserializer<'de>,
    T: Deserialize<'de>,
{
    let value = Value::deserialize(deserializer)?;
    Ok(match value {
        Value::Null => Nullable::Null,
        value => T::deserialize(value)
            .map(Nullable::Value)
            .unwrap_or(Nullable::Absent),
    })
}

// ---------------------------------------------------------------------------
// Closed string domains (serde `snake_case` matches the wire strings).

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum JobKind {
    Import,
    Meeting,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum JobStatus {
    Preparing,
    Queued,
    Processing,
    Uploaded,
    Ready,
    Failed,
    Starting,
    Recording,
    Paused,
    Finishing,
}

impl JobStatus {
    /// `app.ACTIVE` — statuses that reject edits and map to `failed` on
    /// recovery.
    pub fn is_active(self) -> bool {
        matches!(
            self,
            Self::Preparing
                | Self::Queued
                | Self::Processing
                | Self::Starting
                | Self::Recording
                | Self::Paused
                | Self::Finishing
        )
    }

    /// The `'preparing'|'queued'|'processing'` subset PATCH rejects.
    pub fn blocks_patch(self) -> bool {
        matches!(self, Self::Preparing | Self::Queued | Self::Processing)
    }

    /// The wire string (matches serde `snake_case`).
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Preparing => "preparing",
            Self::Queued => "queued",
            Self::Processing => "processing",
            Self::Uploaded => "uploaded",
            Self::Ready => "ready",
            Self::Failed => "failed",
            Self::Starting => "starting",
            Self::Recording => "recording",
            Self::Paused => "paused",
            Self::Finishing => "finishing",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Attribution {
    Single,
    Overlap,
    Unknown,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CaptureSourceId {
    MicrophoneClean,
    System,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ProvenanceKind {
    LocalInference,
    Imported,
    Manual,
    PendingInference,
}

// NOTE: `segment.refinement_state` is deliberately a pass-through
// `Nullable<String>`, not an enum — v0.6.6 copies the prior row's value
// verbatim on PUT, and stored payloads carry values beyond the
// edited/provisional/refined/unresolved writer vocabulary (goldens seed
// `final`/`invented`).

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RefinementStatus {
    Waiting,
    Refining,
    Paused,
    Unresolved,
    Complete,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ReviewResolution {
    WordsReviewed,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RollingWindowStatus {
    Queued,
    InFlight,
}

// ---------------------------------------------------------------------------
// Shared nested shapes.

/// `{start_sample, end_sample}` — audio anchors and region lists.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SampleRange {
    pub start_sample: i64,
    pub end_sample: i64,
}

/// The capture-source admission binding — `{catalog_sha256, source_id,
/// source_revision}`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CaptureSourceBinding {
    pub catalog_sha256: String,
    pub source_id: CaptureSourceId,
    pub source_revision: i64,
}

/// `speaker_activity.regions[]`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SpeakerActivityRegion {
    pub start_sample: i64,
    pub end_sample: i64,
    #[serde(default)]
    pub speakers: Vec<String>,
}

/// `activity_regions[]` — seconds-domain speaker regions on document rows.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ActivityRegion {
    pub start: f64,
    pub end: f64,
    #[serde(default)]
    pub speakers: Vec<String>,
}

/// `decode_context` — bounded-decode clip bounds and policy id.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct DecodeContext {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub boundary_sample: Option<i64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub end_sample: Option<i64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub policy: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub start_sample: Option<i64>,
}

/// `language_detection` — mode/reason/probability plus optional probes;
/// tolerant of further `.*` keys (the ledger rules the subtree opaque).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct LanguageDetection {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub decision: Option<Map<String, Value>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub mode: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub probability: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub probes: Option<Vec<LanguageProbe>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reason: Option<String>,
}

/// `language_detection.probes[]` — a sampled language decision.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct LanguageProbe {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub decision: Option<Map<String, Value>>,
    #[serde(default)]
    pub end_sample: i64,
    #[serde(default)]
    pub language: Option<String>,
    #[serde(default)]
    pub review: bool,
    #[serde(default)]
    pub start_sample: i64,
}

/// `speaker_activity` — per-utterance speaker ownership evidence.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SpeakerActivity {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub audio_revision: Option<i64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub observed_end_sample: Option<i64>,
    #[serde(default)]
    pub regions: Vec<SpeakerActivityRegion>,
}

/// `transcription_review` — the review cause a passage carries.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TranscriptionReview {
    /// One of `review.REASONS` (kept as a string: unknown reasons are
    /// ignored by `review_reason`, matching v0.6.6).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reason: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub partial_text: Option<bool>,
}

/// `acoustic_evidence` — the kept product keys (the receipt keys
/// `input_policy`/`model_revision`/`classification_scope`/
/// `negative_signal_state`/`model_negative_regions`/
/// `maximum_model_probability`/`observed_until_sample` are dropped at
/// import).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AcousticEvidence {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub audio_revision: Option<i64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub complete: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub decision: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub end_sample: Option<i64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub maximum_probability: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub retained_audio_revision: Option<i64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub source: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub speech_regions: Option<Vec<SampleRange>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub start_sample: Option<i64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub uncertain_regions: Option<Vec<SampleRange>>,
}

/// `reading_turns[]` — a localized turn slice with attribution. Fields are
/// optional so malformed rows deserialize and degrade to the parent
/// passage at export, matching v0.6.6.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ReadingTurn {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub attribution: Option<Attribution>,
    #[serde(default, skip_serializing_if = "Nullable::is_absent")]
    pub end: Nullable<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub end_offset: Option<i64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub speaker: Option<String>,
    #[serde(default, skip_serializing_if = "Nullable::is_absent")]
    pub start: Nullable<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub start_offset: Option<i64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub text: Option<String>,
}

/// `retry_candidate` — a stored retry suggestion bound to its audio.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RetryCandidate {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub accepted_at: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub created: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub end: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub id: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub language: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub original_text: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub replaced_text: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub review: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub review_resolution: Option<ReviewResolution>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub speaker: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub start: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub text: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub transcription_review: Option<TranscriptionReview>,
}

// ---------------------------------------------------------------------------
// Segment — every `product` ledger row under `job.document.segments[]`.
// No `deny_unknown_fields`: client input ignores unknown keys.

#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct Segment {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub acoustic_evidence: Option<AcousticEvidence>,
    #[serde(
        default,
        deserialize_with = "de_opt_lenient",
        skip_serializing_if = "Option::is_none"
    )]
    pub activity_regions: Option<Vec<ActivityRegion>>,
    #[serde(
        default,
        deserialize_with = "de_opt_lenient",
        skip_serializing_if = "Option::is_none"
    )]
    pub audio_anchor: Option<SampleRange>,
    #[serde(
        default,
        deserialize_with = "de_opt_lenient",
        skip_serializing_if = "Option::is_none"
    )]
    pub audio_revision: Option<i64>,
    #[serde(
        default,
        deserialize_with = "de_opt_lenient",
        skip_serializing_if = "Option::is_none"
    )]
    pub audio_state: Option<String>,
    #[serde(
        default,
        deserialize_with = "de_opt_lenient",
        skip_serializing_if = "Option::is_none"
    )]
    pub canonical_machine_revision: Option<i64>,
    #[serde(
        default,
        deserialize_with = "de_opt_lenient",
        skip_serializing_if = "Option::is_none"
    )]
    pub canonical_state: Option<String>,
    #[serde(
        default,
        deserialize_with = "de_opt_lenient",
        skip_serializing_if = "Option::is_none"
    )]
    pub canonical_unresolved: Option<String>,
    #[serde(
        default,
        deserialize_with = "de_opt_lenient",
        skip_serializing_if = "Option::is_none"
    )]
    pub canonical_utterance_id: Option<String>,
    #[serde(
        default,
        deserialize_with = "de_opt_lenient",
        skip_serializing_if = "Option::is_none"
    )]
    pub capture_source: Option<CaptureSourceBinding>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub cohere_raw_text: Option<String>,
    /// Export-local marker: txt output prefixes `Coarse timing` (transcript.py
    /// sets it on projected slices; a stored flag is honored verbatim).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub coarse_timing: Option<bool>,
    /// v0.6.6 writes `null` on unassigned/uncertain rows.
    #[serde(default, skip_serializing_if = "Nullable::is_absent")]
    pub confidence: Nullable<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub decode_context: Option<DecodeContext>,
    pub end: f64,
    #[serde(
        default,
        deserialize_with = "de_opt_lenient",
        skip_serializing_if = "Option::is_none"
    )]
    pub end_sample: Option<i64>,
    #[serde(
        default,
        deserialize_with = "de_opt_lenient",
        skip_serializing_if = "Option::is_none"
    )]
    pub finalized: Option<bool>,
    /// Always assigned by `validate` (`seg-{i}`); documents that bypass it
    /// (export-only fixtures) deserialize with `""`.
    #[serde(default)]
    pub id: String,
    /// v0.6.6 writes `null` when detection is unavailable.
    #[serde(default, skip_serializing_if = "Nullable::is_absent")]
    pub language: Nullable<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub language_detection: Option<LanguageDetection>,
    #[serde(
        default,
        deserialize_with = "de_opt_lenient",
        skip_serializing_if = "Option::is_none"
    )]
    pub language_epoch: Option<i64>,
    #[serde(
        default,
        deserialize_with = "de_opt_lenient",
        skip_serializing_if = "Option::is_none"
    )]
    pub language_generation: Option<i64>,
    #[serde(
        default,
        deserialize_with = "de_opt_lenient",
        skip_serializing_if = "Option::is_none"
    )]
    pub language_mode: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub language_review: Option<bool>,
    #[serde(
        default,
        deserialize_with = "de_opt_lenient",
        skip_serializing_if = "Option::is_none"
    )]
    pub machine_revision: Option<i64>,
    #[serde(
        default,
        deserialize_with = "de_opt_lenient",
        skip_serializing_if = "Option::is_none"
    )]
    pub protected_fields: Option<Vec<String>>,
    /// Opaque bounded client-supplied object (ledger ruling); written `null`
    /// by some import paths.
    #[serde(default, skip_serializing_if = "Nullable::is_absent")]
    pub provenance: Nullable<Map<String, Value>>,
    #[serde(
        default,
        deserialize_with = "de_opt_lenient",
        skip_serializing_if = "Option::is_none"
    )]
    pub reading_turns: Option<Vec<ReadingTurn>>,
    /// Export-local marker: txt output prefixes `Timing unknown; parent
    /// audio` (set on projected slices; a stored flag is honored verbatim).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reading_slice: Option<bool>,
    /// v0.6.6 PUT writes `null` here when the prior row lacks the key, and
    /// copies an arbitrary prior value verbatim — pass-through, not an enum.
    #[serde(
        default,
        deserialize_with = "de_nullable_lenient",
        skip_serializing_if = "Nullable::is_absent"
    )]
    pub refinement_state: Nullable<String>,
    #[serde(
        default,
        deserialize_with = "de_opt_lenient",
        skip_serializing_if = "Option::is_none"
    )]
    pub refinement_window: Option<String>,
    #[serde(
        default,
        deserialize_with = "de_opt_lenient",
        skip_serializing_if = "Option::is_none"
    )]
    pub retry_candidate: Option<RetryCandidate>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub review: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub review_resolution: Option<ReviewResolution>,
    pub speaker: String,
    #[serde(
        default,
        deserialize_with = "de_opt_lenient",
        skip_serializing_if = "Option::is_none"
    )]
    pub speaker_activity: Option<SpeakerActivity>,
    #[serde(
        default,
        deserialize_with = "de_opt_lenient",
        skip_serializing_if = "Option::is_none"
    )]
    pub speaker_candidates: Option<Vec<String>>,
    /// Opaque bounded mapping evidence (ledger ruling).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub speaker_track_mapping: Option<Map<String, Value>>,
    #[serde(
        default,
        deserialize_with = "de_opt_lenient",
        skip_serializing_if = "Option::is_none"
    )]
    pub speech_regions: Option<Vec<SampleRange>>,
    pub start: f64,
    #[serde(
        default,
        deserialize_with = "de_opt_lenient",
        skip_serializing_if = "Option::is_none"
    )]
    pub start_sample: Option<i64>,
    pub text: String,
    #[serde(
        default,
        deserialize_with = "de_opt_lenient",
        skip_serializing_if = "Option::is_none"
    )]
    pub text_audio_anchor: Option<SampleRange>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub timing: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub transcription_review: Option<TranscriptionReview>,
    #[serde(
        default,
        deserialize_with = "de_opt_lenient",
        skip_serializing_if = "Option::is_none"
    )]
    pub voice_eligible: Option<bool>,
}

// ---------------------------------------------------------------------------
// Document.

/// `document.provenance` — product keys; `metrics`/`libraries` are open
/// subtree rows in the ledger, stored as opaque bounded objects.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct DocumentProvenance {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub asr_checkpoint: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub asr_model: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub device: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub diarization_checkpoint: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub diarization_model: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub kind: Option<ProvenanceKind>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub language: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub language_detector: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub libraries: Option<Map<String, Value>>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub metrics: Option<Map<String, Value>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub mode: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub models: Option<Vec<String>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub speaker_limit: Option<i64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub speech_detector: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub timing: Option<String>,
}

#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct Document {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub edited_at: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub provenance: Option<DocumentProvenance>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub schema_version: Option<i64>,
    pub segments: Vec<Segment>,
    pub speakers: BTreeMap<String, String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub warnings: Option<Vec<String>>,
}

// ---------------------------------------------------------------------------
// Job-level product shapes.

/// `speaker_assignments.<track>` — a named speaker bound to a person.
/// Persisted only (never built from client input), so unknown keys error.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SpeakerAssignment {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub assigned_at: Option<f64>,
    pub confirmed: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub confirmed_at: Option<f64>,
    /// Opaque match/clip evidence (the JS panel dumps it whole); explicit
    /// `null` on manual assignments.
    #[serde(default, skip_serializing_if = "Nullable::is_absent")]
    pub evidence: Nullable<Map<String, Value>>,
    pub meeting_id: String,
    pub name: String,
    /// `null` when the speaker was named without choosing a person.
    #[serde(default, skip_serializing_if = "Nullable::is_absent")]
    pub person_id: Nullable<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub source: Option<String>,
    pub track_id: String,
}

/// `voice_checks.<track>` — a pending/manual identity check.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct VoiceCheck {
    pub id: String,
    pub status: String,
}

/// `voice_suggestions[]` — a proposed track→person suggestion.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct VoiceSuggestion {
    /// Opaque evidence payload (open `.*` product row).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub evidence: Option<Value>,
    pub id: String,
    pub kind: String,
    /// Opaque match payload (open `.*` product row).
    #[serde(rename = "match", skip_serializing_if = "Option::is_none")]
    pub match_: Option<Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub person_id: Option<String>,
    pub track_id: String,
}

/// `passage_retry` — an in-flight manual passage retry (`{id, segment_id,
/// language}`); the job key may be an explicit `null` after completion.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PassageRetry {
    pub id: String,
    pub language: String,
    pub segment_id: String,
}

/// `language_history[]` — one language epoch.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct LanguageEpoch {
    pub epoch: i64,
    pub generation: i64,
    pub language: String,
    pub start_sample: i64,
}

/// `rolling_refinement.pending[]` — one planned refinement window.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RollingWindow {
    #[serde(default)]
    pub attempts: i64,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub context_end_sample: Option<i64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub context_start_sample: Option<i64>,
    pub end_sample: i64,
    pub id: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub operation_id: Option<String>,
    pub start_sample: i64,
    pub status: RollingWindowStatus,
}

/// `rolling_refinement` — the persisted rolling-refinement plan.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RollingRefinement {
    #[serde(default)]
    pub cancelled: bool,
    pub completed_sample: i64,
    pub last_scheduled_at: f64,
    #[serde(default)]
    pub pending: Vec<RollingWindow>,
    pub version: i64,
}

/// `capture_audio` — the kept product key (`frames`/`sample_rate` are
/// dropped at import).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CaptureAudio {
    pub state: String,
}

// ---------------------------------------------------------------------------
// Job — product fields plus the flattened private engine state.

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Job {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub capture_audio: Option<CaptureAudio>,
    pub created: f64,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub dismissed_identity_suggestions: Option<Vec<String>>,
    /// Present in every v0.6.6 job payload, `null` while preparing.
    pub document: Option<Document>,
    /// Present in every v0.6.6 job payload, `null` while preparing.
    pub duration: Option<f64>,
    pub id: String,
    /// Absent on imports, `"meeting"` on meetings.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub kind: Option<JobKind>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub language: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub language_acknowledged_revision: Option<i64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub language_epoch: Option<i64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub language_history: Option<Vec<LanguageEpoch>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub language_revision: Option<i64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub message: Option<String>,
    pub name: String,
    /// `null` after a completed retry, absent otherwise.
    #[serde(default, skip_serializing_if = "Nullable::is_absent")]
    pub passage_retry: Nullable<PassageRetry>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub refinement_status: Option<RefinementStatus>,
    pub revision: i64,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub rolling_refinement: Option<RollingRefinement>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub source_file: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub sources: Option<Vec<String>>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub speaker_assignments: Option<BTreeMap<String, SpeakerAssignment>>,
    pub status: JobStatus,
    pub updated: f64,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub voice_checks: Option<BTreeMap<String, VoiceCheck>>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub voice_suggestions: Option<Vec<VoiceSuggestion>>,
    /// Private engine state — stored in the `private` column and merged back
    /// into the payload on read (v0.6.6 keeps these keys at job top level).
    #[serde(flatten)]
    pub private: JobPrivate,
}

/// Job-level `private`-class keys — opaque engine state the user never
/// sees. Each field keeps its v0.6.6 JSON shape; only the top-level kinds
/// are typed.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct JobPrivate {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub boundary_candidate: Option<Value>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub boundary_candidates: Option<Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub canonical_force_fresh_asr: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub canonical_observed_sample: Option<i64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub canonical_refined: Option<Map<String, Value>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub canonical_uncertain_samples: Option<i64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub canonical_utterances: Option<bool>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub fast_history: Option<Map<String, Value>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub fast_previous_revision: Option<Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub fast_retained_candidate: Option<Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub last_fast_sequence: Option<i64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub pause_flush: Option<Value>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub refinement_history: Option<Map<String, Value>>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub refinement_unresolved: Option<Vec<Value>>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub rolling_inflight: Option<Value>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub rolling_sources: Option<Map<String, Value>>,
}

impl Job {
    /// Serialize to the v0.6.6 payload shape (private keys merged at top
    /// level).
    pub fn to_payload(&self) -> Value {
        serde_json::to_value(self).unwrap_or_default()
    }

    /// The `job_metadata` projection: the payload minus `document` and the
    /// private keys.
    pub fn to_metadata_payload(&self) -> Value {
        let mut payload = self.to_payload();
        if let Value::Object(map) = &mut payload {
            map.remove("document");
            for key in PRIVATE_KEYS {
                map.remove(*key);
            }
        }
        payload
    }
}

/// The `job.*` `private`-class keys — must mirror `JobPrivate`'s field
/// names.
pub const PRIVATE_KEYS: &[&str] = &[
    "boundary_candidate",
    "boundary_candidates",
    "canonical_force_fresh_asr",
    "canonical_observed_sample",
    "canonical_refined",
    "canonical_uncertain_samples",
    "canonical_utterances",
    "fast_history",
    "fast_previous_revision",
    "fast_retained_candidate",
    "last_fast_sequence",
    "pause_flush",
    "refinement_history",
    "refinement_unresolved",
    "rolling_inflight",
    "rolling_sources",
];

/// `job_metadata` summary keys, in the exact order v0.6.6 emits them.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct JobSummary {
    pub id: String,
    pub name: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub kind: Option<JobKind>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub source_file: Option<String>,
    pub created: f64,
    pub updated: f64,
    pub status: JobStatus,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub message: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub language: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub duration: Option<f64>,
    pub revision: i64,
}

/// `people` row.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Person {
    pub created: f64,
    pub id: String,
    pub name: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub updated: Option<f64>,
}

/// `voice_profiles` — the payload minus the `centroid` array, which travels
/// separately as the `centroid` BLOB column.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct VoiceProfile {
    #[serde(default)]
    pub clips: Vec<VoiceClip>,
    pub consent: String,
    pub enrolled_at: f64,
    pub model: VoiceModel,
    pub person_id: String,
    pub version: String,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct VoiceModel {
    pub artifact_sha256: String,
    pub dimension: i64,
    pub model_id: String,
    pub revision: String,
    pub sample_rate: i64,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct VoiceClip {
    pub end: f64,
    pub meeting_id: String,
    pub segment_id: String,
    pub start: f64,
    pub track_id: String,
}

// ---------------------------------------------------------------------------
// Bounds for opaque JSON fields. Every opaque `Value`/`Map` entering through
// client input or legacy import is checked: serialized size and nesting
// depth. Engine-state fields in `JobPrivate` get a larger budget; the
// product-facing blobs stay small.

/// Serialized-JSON cap for product opaque objects (`segment.provenance`,
/// `speaker_track_mapping`, `speaker_assignments.*.evidence`,
/// `voice_suggestions.*`, `document.provenance` subtrees).
pub const MAX_OPAQUE_BYTES: usize = 64 * 1024;
/// Serialized-JSON cap for `JobPrivate` engine-state fields.
pub const MAX_PRIVATE_BYTES: usize = 1024 * 1024;
/// Nesting-depth cap for every opaque value.
pub const MAX_OPAQUE_DEPTH: usize = 32;

fn json_depth(value: &Value) -> usize {
    match value {
        Value::Array(list) => 1 + list.iter().map(json_depth).max().unwrap_or(0),
        Value::Object(map) => 1 + map.values().map(json_depth).max().unwrap_or(0),
        _ => 0,
    }
}

fn check_opaque(name: &str, value: &Value, max_bytes: usize) -> Result<(), String> {
    if json_depth(value) > MAX_OPAQUE_DEPTH {
        return Err(format!("{name}: opaque payload nested too deeply"));
    }
    if serde_json::to_string(value)
        .map(|s| s.len())
        .unwrap_or(usize::MAX)
        > max_bytes
    {
        return Err(format!("{name}: opaque payload too large"));
    }
    Ok(())
}

fn check_opaque_opt(name: &str, value: Option<&Value>, max_bytes: usize) -> Result<(), String> {
    if let Some(v) = value {
        check_opaque(name, v, max_bytes)?;
    }
    Ok(())
}

/// Check every opaque field in a document (client-input entry point).
pub fn check_opaque_bounds_document(document: &Document) -> Result<(), String> {
    if let Some(provenance) = &document.provenance {
        for map in [&provenance.metrics, &provenance.libraries]
            .into_iter()
            .flatten()
        {
            check_opaque(
                "document.provenance",
                &Value::Object(map.clone()),
                MAX_OPAQUE_BYTES,
            )?;
        }
    }
    for (i, segment) in document.segments.iter().enumerate() {
        let name = format!("segment[{i}]");
        if let Nullable::Value(map) = &segment.provenance {
            check_opaque(
                &format!("{name}.provenance"),
                &Value::Object(map.clone()),
                MAX_OPAQUE_BYTES,
            )?;
        }
        if let Some(map) = &segment.speaker_track_mapping {
            check_opaque(
                &format!("{name}.speaker_track_mapping"),
                &Value::Object(map.clone()),
                MAX_OPAQUE_BYTES,
            )?;
        }
    }
    Ok(())
}

/// Check every opaque field on a job (legacy-import and store-write entry
/// points).
pub fn check_opaque_bounds_job(job: &Job) -> Result<(), String> {
    if let Some(document) = &job.document {
        check_opaque_bounds_document(document)?;
    }
    if let Some(assignments) = &job.speaker_assignments {
        for (track, assignment) in assignments {
            if let Nullable::Value(map) = &assignment.evidence {
                check_opaque(
                    &format!("speaker_assignments.{track}.evidence"),
                    &Value::Object(map.clone()),
                    MAX_OPAQUE_BYTES,
                )?;
            }
        }
    }
    if let Some(suggestions) = &job.voice_suggestions {
        for suggestion in suggestions {
            check_opaque_opt(
                "voice_suggestions[].match",
                suggestion.match_.as_ref(),
                MAX_OPAQUE_BYTES,
            )?;
            check_opaque_opt(
                "voice_suggestions[].evidence",
                suggestion.evidence.as_ref(),
                MAX_OPAQUE_BYTES,
            )?;
        }
    }
    let p = &job.private;
    for (name, value) in [
        ("boundary_candidate", p.boundary_candidate.as_ref()),
        ("boundary_candidates", p.boundary_candidates.as_ref()),
        ("fast_previous_revision", p.fast_previous_revision.as_ref()),
        (
            "fast_retained_candidate",
            p.fast_retained_candidate.as_ref(),
        ),
        ("pause_flush", p.pause_flush.as_ref()),
        ("rolling_inflight", p.rolling_inflight.as_ref()),
    ] {
        check_opaque_opt(&format!("job.{name}"), value, MAX_PRIVATE_BYTES)?;
    }
    for (name, map) in [
        ("canonical_refined", p.canonical_refined.as_ref()),
        ("fast_history", p.fast_history.as_ref()),
        ("refinement_history", p.refinement_history.as_ref()),
        ("rolling_sources", p.rolling_sources.as_ref()),
    ] {
        if let Some(map) = map {
            check_opaque(
                &format!("job.{name}"),
                &Value::Object(map.clone()),
                MAX_PRIVATE_BYTES,
            )?;
        }
    }
    if let Some(list) = &p.refinement_unresolved {
        check_opaque(
            "job.refinement_unresolved",
            &Value::Array(list.clone()),
            MAX_PRIVATE_BYTES,
        )?;
    }
    Ok(())
}

impl VoiceProfile {
    /// Serialize plus the f32 centroid as a JSON float array — the v0.6.6
    /// `enrollment` payload shape (`sample_rate` nested under `model`).
    pub fn to_payload(&self, centroid: &[f32]) -> Value {
        let mut payload = serde_json::to_value(self).unwrap_or_default();
        if let Value::Object(map) = &mut payload {
            map.insert(
                "centroid".to_string(),
                Value::Array(
                    centroid
                        .iter()
                        .map(|f| {
                            serde_json::Number::from_f64(*f as f64)
                                .map(Value::Number)
                                .unwrap_or(Value::Null)
                        })
                        .collect(),
                ),
            );
        }
        payload
    }
}
