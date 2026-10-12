//! Canonical live engine.
//!
//! Inbox coalescing, utterance admission, short routed revisions, the English
//! authoritative tail, the non-English core plan, and compare-and-swap window
//! reconciliation. Model calls go through [`crate::models::Models`]. This module
//! reads bounded crops from the retained WAV; it does not load MLX.

mod context;
pub use context::Refinement as SavedRefinement;

use std::collections::{BTreeMap, HashMap, VecDeque};
use std::path::Path;
use std::sync::{Arc, Mutex};

use serde_json::{Map, Value};
use sha2::{Digest, Sha256};
use thiserror::Error;

use crate::audio::RecordingReader;
use crate::document::{Document, Segment};
use crate::error::ChildError;
use crate::ipc::{DiarPreset, GainPolicy, Handle, Turn, VadFeedOut};
use crate::language::{
    AcousticEvidence, AsrOutput, AudioState, LanguageDetection, LanguageDetector, Passage,
    Recognizer, SpeechTranscriber, TranscribeOptions,
};
use crate::languages;
use crate::limits;
use crate::models::Models;
use crate::speech::{self, Admission, SampleSpan, SpeechDecision, SpeechError, SpeechLedger};
use crate::timing::{self, ReadingRow, ReadingTurn, TimedWord};

/// 16 kHz recording clock. Alias of [`limits::SAMPLE_RATE`].
pub const RATE: i64 = limits::SAMPLE_RATE as i64;
/// Largest single canonical decode, in samples (`24.5 * RATE`).
pub const MAX_DECODE_SAMPLES: i64 = 392_000;
/// Silence that seals an open utterance.
pub const SILENCE_SAMPLES: i64 = 8_000;
/// Left context retained before the first speech sample.
pub const UTTERANCE_CONTEXT_SAMPLES: i64 = 3_200;
/// Declared capture inputs. Local diarization slots stay inside 32 ids each.
pub const SOURCE_IDS: [&str; 2] = ["microphone_clean", "system"];

const CORE_PLAN_CAP: i64 = 20 * RATE;
const CORE_PLAN_MIN: i64 = 6 * RATE;
const CORE_PLAN_MIN_GAP: i64 = 2_048;
const CORE_PLAN_RADIUS: i64 = RATE;
/// `core_plan.POLICY`.
pub const CORE_PLAN_POLICY: &str = "current_silero_negative_gap_bounded20_radius1_v2";
const CONTEXT_CORE: i64 = 14 * RATE;
const CONTEXT_PAD: i64 = 3 * RATE;
const PHYSICAL_MAX: i64 = 20 * RATE;
/// `context_plan.POLICY`.
pub const CONTEXT_POLICY: &str = "english_bounded14_context3_corroborated_v1";
const AUTHORITATIVE_CORE: i64 = 224_000;
const AUTHORITATIVE_PAD: i64 = 48_000;
const AUTHORITATIVE_MAX: i64 = 320_000;
/// `authoritative_tail.POLICY`.
pub const AUTHORITATIVE_POLICY: &str = "authoritative_tail_last_supported_boundary_v2";
const MAX_CONTROL_MESSAGES: usize = 128;

#[derive(Debug, Error, Clone, PartialEq, Eq)]
pub enum LiveError {
    #[error("{0}")]
    Message(&'static str),
    #[error("{0}")]
    Detail(String),
    #[error(transparent)]
    Child(ChildError),
}

impl From<SpeechError> for LiveError {
    fn from(error: SpeechError) -> Self {
        Self::Detail(error.to_string())
    }
}

impl From<String> for LiveError {
    fn from(error: String) -> Self {
        Self::Detail(error)
    }
}
impl From<&str> for LiveError {
    fn from(error: &str) -> Self {
        Self::Detail(error.into())
    }
}

impl From<ChildError> for LiveError {
    fn from(error: ChildError) -> Self {
        Self::Child(error)
    }
}

impl From<crate::import::ImportError> for LiveError {
    fn from(error: crate::import::ImportError) -> Self {
        error.child.map_or(Self::Detail(error.message), Self::Child)
    }
}

fn sha256_hex(bytes: &[u8]) -> String {
    hex::encode(Sha256::digest(bytes))
}

fn round_samples(seconds: f64) -> i64 {
    (seconds * RATE as f64).round() as i64
}

/// Stable source-local speaker id. `speaker_0` on system audio is `speaker_32`.
pub fn speaker_id(source_id: &str, local_id: &str) -> Result<String, LiveError> {
    let Some(index) = SOURCE_IDS.iter().position(|id| *id == source_id) else {
        return Err(LiveError::Message(
            "Source speaker track exceeds the bounded namespace.",
        ));
    };
    let Some(digits) = local_id.strip_prefix("speaker_") else {
        return Err(LiveError::Message(
            "Source speaker track exceeds the bounded namespace.",
        ));
    };
    if digits.is_empty() || !digits.bytes().all(|byte| byte.is_ascii_digit()) {
        return Err(LiveError::Message(
            "Source speaker track exceeds the bounded namespace.",
        ));
    }
    let local: u32 = digits.parse().unwrap_or(u32::MAX);
    if local >= 32 {
        return Err(LiveError::Message(
            "Source speaker track exceeds the bounded namespace.",
        ));
    }
    Ok(format!("speaker_{}", 32 * index as u32 + local))
}

pub fn utterance_id(job_id: &str, epoch: i64, start_sample: i64) -> String {
    let seed = format!("{job_id}:{epoch}:{start_sample}");
    format!("utterance-{}", &sha256_hex(seed.as_bytes())[..24])
}

fn speaker_label(speaker: &str) -> String {
    if let Some(digits) = speaker.strip_prefix("speaker_")
        && let Ok(index) = digits.parse::<i64>()
    {
        return format!("Speaker {}", index + 1);
    }
    if speaker == "unassigned" {
        "Unknown speaker".to_string()
    } else if speaker == "multiple_speakers" {
        "Multiple speakers".to_string()
    } else {
        "Mixed audio".to_string()
    }
}

fn single_speaker(name: &str) -> bool {
    let Some(digits) = name.strip_prefix("speaker_") else {
        return false;
    };
    !digits.is_empty() && digits.bytes().all(|byte| byte.is_ascii_digit())
}

#[derive(Debug, Clone, PartialEq)]
pub struct AudioNotice {
    pub start_sample: i64,
    pub end_sample: i64,
    pub language_epoch: i64,
    pub language: String,
}

#[derive(Debug, Clone, PartialEq)]
pub enum LiveCommand {
    Audio(AudioNotice),
    Language {
        language: String,
        generation: i64,
        start_sample: i64,
    },
    Shutdown,
    Flush {
        request_id: Option<String>,
        through_sample: Option<i64>,
    },
    Stop,
}

/// Coalesces contiguous saved-audio notices while retaining language/control order.
pub struct Inbox {
    audio_batch_samples: i64,
    messages: VecDeque<LiveCommand>,
    shutdown: bool,
    received: i64,
}

impl Inbox {
    pub fn new(audio_batch_seconds: i64) -> Result<Self, LiveError> {
        if !(1..=6).contains(&audio_batch_seconds) {
            return Err(LiveError::Message("Invalid bounded audio drain cadence."));
        }
        Ok(Self {
            audio_batch_samples: audio_batch_seconds * RATE,
            messages: VecDeque::new(),
            shutdown: false,
            received: 0,
        })
    }

    pub fn push(&mut self, message: LiveCommand) -> Result<(), LiveError> {
        match message {
            LiveCommand::Audio(notice) => {
                if notice.start_sample != self.received
                    || notice.end_sample <= notice.start_sample
                    || notice.end_sample - notice.start_sample > RATE
                {
                    return Err(LiveError::Message("Noncontiguous audio notification."));
                }
                self.received = notice.end_sample;
                let coalesce = self.messages.back_mut().and_then(|last| match last {
                    LiveCommand::Audio(existing)
                        if existing.language_epoch == notice.language_epoch =>
                    {
                        Some(existing)
                    }
                    _ => None,
                });
                if let Some(existing) = coalesce {
                    existing.end_sample = notice.end_sample;
                } else {
                    self.messages.push_back(LiveCommand::Audio(notice));
                }
            }
            LiveCommand::Shutdown => self.shutdown = true,
            LiveCommand::Language { .. } | LiveCommand::Flush { .. } | LiveCommand::Stop => {
                self.messages.push_back(message)
            }
        }
        if self.messages.len() > MAX_CONTROL_MESSAGES {
            return Err(LiveError::Message(
                "Too many pending language/control boundaries.",
            ));
        }
        Ok(())
    }

    /// Contiguous audio before the next control supersedes an open tail.
    pub fn newer_audio_queued(&self, through_sample: i64, epoch: i64) -> bool {
        matches!(self.messages.front(),Some(LiveCommand::Audio(notice)) if notice.language_epoch==epoch && notice.start_sample==through_sample && notice.end_sample>through_sample)
    }
    pub fn queued_audio_span(&self) -> Option<(i64, i64, i64)> {
        self.messages.iter().find_map(|message| match message {
            LiveCommand::Audio(notice) => Some((
                notice.start_sample,
                notice.end_sample,
                notice.language_epoch,
            )),
            _ => None,
        })
    }
    /// Controls are ordered with audio but take priority over background accuracy work.
    pub fn has_pending_control(&self) -> bool {
        self.shutdown
            || self
                .messages
                .iter()
                .any(|message| !matches!(message, LiveCommand::Audio(_)))
    }
    /// Non-blocking take. `None` means the queue is idle.
    pub fn take(&mut self) -> Result<Option<LiveCommand>, LiveError> {
        if self.shutdown {
            return Ok(Some(LiveCommand::Shutdown));
        }
        let Some(mut result) = self.messages.pop_front() else {
            return Ok(None);
        };
        if let LiveCommand::Audio(ref mut notice) = result {
            let batch_end = notice.start_sample + self.audio_batch_samples;
            if notice.end_sample > batch_end {
                let mut rest = notice.clone();
                notice.end_sample = batch_end;
                rest.start_sample = batch_end;
                self.messages.push_front(LiveCommand::Audio(rest));
            }
        }
        Ok(Some(result))
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DecodeStage {
    Live,
    Refined,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum UtteranceState {
    Open,
    Sealed,
}

#[derive(Debug, Clone, PartialEq)]
struct Utterance {
    id: String,
    start_sample: i64,
    end_sample: i64,
    last_speech_sample: i64,
    language_epoch: i64,
    state: UtteranceState,
    text: String,
    machine_revision: i64,
    audio_revision: i64,
    speech_regions: Vec<SampleSpan>,
    protected_fields: Vec<String>,
    refinement_state: String,
    language: Option<String>,
    language_detection: Option<LanguageDetection>,
    review: bool,
    audio_state: Option<AudioState>,
    transcription_review: bool,
    canonical_unresolved: Option<String>,
    speaker_candidates: Vec<String>,
    speaker_activity: Option<timing::SpeakerActivity>,
    text_audio_anchor: Option<timing::AudioAnchor>,
    text_end_sample: Option<i64>,
}

/// VAD speech objects. NVIDIA activity is attached separately and does not split words.
pub struct UtteranceBook {
    job_id: String,
    cursor: i64,
    epoch_start: i64,
    epoch: i64,
    rows: HashMap<String, Utterance>,
    order: Vec<String>,
    active: Option<String>,
    closed: bool,
    uncertain_sample_count: i64,
}

impl UtteranceBook {
    pub fn new(job_id: impl Into<String>) -> Result<Self, LiveError> {
        let job_id = job_id.into();
        if job_id.is_empty() {
            return Err(LiveError::Message(
                "Utterance identity requires a recording.",
            ));
        }
        Ok(Self {
            job_id,
            cursor: 0,
            epoch_start: 0,
            epoch: 0,
            rows: HashMap::new(),
            order: Vec::new(),
            active: None,
            closed: false,
            uncertain_sample_count: 0,
        })
    }

    pub fn cursor(&self) -> i64 {
        self.cursor
    }

    pub fn uncertain_sample_count(&self) -> i64 {
        self.uncertain_sample_count
    }

    pub fn closed(&self) -> bool {
        self.closed
    }

    fn set_end(row: &mut Utterance, end: i64) {
        if row.end_sample == end {
            return;
        }
        let shrinking = end < row.end_sample;
        row.end_sample = end;
        row.audio_revision += 1;
        // New observations cannot revise the already-observed activity prefix.
        // A shortened edited/sealed span keeps only its own retained evidence.
        if shrinking {
            let activity = row.speaker_activity.take();
            clip_retained_activity(row, activity.as_ref());
        }
        row.text_audio_anchor = None;
    }

    fn seal(&mut self, end: i64) {
        let Some(active) = self.active.clone() else {
            return;
        };
        let row = self.rows.get_mut(&active).expect("active utterance");
        let capped = end.min(row.last_speech_sample + UTTERANCE_CONTEXT_SAMPLES);
        Self::set_end(row, capped);
        row.state = UtteranceState::Sealed;
        self.active = None;
    }

    pub fn boundary(&mut self, sample: i64, epoch: i64) -> Result<(), LiveError> {
        if self.closed || sample != self.cursor || epoch != self.epoch + 1 {
            return Err(LiveError::Message(
                "Language boundaries must follow observed audio.",
            ));
        }
        self.seal(sample);
        self.epoch_start = sample;
        self.epoch = epoch;
        Ok(())
    }

    pub fn observe(&mut self, evidence: &Admission) -> Result<(), LiveError> {
        let start = evidence.start_sample;
        let end = evidence.end_sample;
        if self.closed
            || !evidence.complete
            || !matches!(
                evidence.decision,
                SpeechDecision::Speech | SpeechDecision::NoSpeech | SpeechDecision::Uncertain
            )
            || start != self.cursor
            || end <= start
        {
            return Err(LiveError::Message(
                "Canonical utterances require ordered complete speech evidence.",
            ));
        }
        if (evidence.decision == SpeechDecision::Speech) != !evidence.speech_regions.is_empty() {
            return Err(LiveError::Message(
                "Speech decision and observed regions disagree.",
            ));
        }
        let mut position = start;
        for region in &evidence.speech_regions {
            if region.start_sample < position
                || region.end_sample <= region.start_sample
                || region.end_sample > end
            {
                return Err(LiveError::Message("Invalid speech-region ledger."));
            }
            position = region.end_sample;
        }
        for region in &evidence.uncertain_regions {
            self.uncertain_sample_count += region.end_sample - region.start_sample;
        }
        for region in &evidence.speech_regions {
            if self.active.as_ref().is_some_and(|active| {
                region.start_sample - self.rows.get(active).expect("active").last_speech_sample
                    >= SILENCE_SAMPLES
            }) {
                self.seal(region.start_sample);
            }
            if self.active.is_none() {
                let previous_end = self
                    .order
                    .last()
                    .and_then(|id| self.rows.get(id))
                    .map(|row| row.end_sample)
                    .unwrap_or(0);
                let begin = self
                    .epoch_start
                    .max(previous_end)
                    .max(region.start_sample - UTTERANCE_CONTEXT_SAMPLES);
                let id = utterance_id(&self.job_id, self.epoch, begin);
                self.rows.insert(
                    id.clone(),
                    Utterance {
                        id: id.clone(),
                        start_sample: begin,
                        end_sample: region.end_sample,
                        last_speech_sample: region.end_sample,
                        language_epoch: self.epoch,
                        state: UtteranceState::Open,
                        text: String::new(),
                        machine_revision: 0,
                        audio_revision: 0,
                        speech_regions: Vec::new(),
                        protected_fields: Vec::new(),
                        refinement_state: "provisional".to_string(),
                        language: None,
                        language_detection: None,
                        review: false,
                        audio_state: None,
                        transcription_review: false,
                        canonical_unresolved: None,
                        speaker_candidates: Vec::new(),
                        speaker_activity: None,
                        text_audio_anchor: None,
                        text_end_sample: None,
                    },
                );
                self.order.push(id.clone());
                self.active = Some(id);
            }
            let row = self
                .rows
                .get_mut(self.active.as_ref().expect("active"))
                .expect("row");
            if row
                .speech_regions
                .last()
                .is_some_and(|last| last.end_sample == region.start_sample)
            {
                row.speech_regions.last_mut().expect("region").end_sample = region.end_sample;
            } else {
                row.speech_regions.push(*region);
            }
            row.last_speech_sample = region.end_sample;
            Self::set_end(row, region.end_sample);
        }
        if let Some(active) = self.active.clone() {
            let row = self.rows.get_mut(&active).expect("active");
            let capped = end.min(row.last_speech_sample + UTTERANCE_CONTEXT_SAMPLES);
            Self::set_end(row, capped);
            if end - row.last_speech_sample >= SILENCE_SAMPLES {
                self.seal(end);
            }
        }
        self.cursor = end;
        Ok(())
    }

    pub fn finish(&mut self) {
        self.seal(self.cursor);
        self.closed = true;
    }

    /// Ordered utterance endpoints and admitted regions for offline crop planning.
    pub(crate) fn speech_regions(&self) -> impl Iterator<Item = (SampleSpan, &[SampleSpan])> {
        self.order.iter().map(|id| {
            let row = &self.rows[id];
            (
                SampleSpan {
                    start_sample: row.start_sample,
                    end_sample: row.end_sample,
                },
                row.speech_regions.as_slice(),
            )
        })
    }

    fn apply_model(
        &mut self,
        identity: &str,
        text: &str,
        stage: DecodeStage,
        mut complete: bool,
    ) -> Result<(), LiveError> {
        let row = self
            .rows
            .get_mut(identity)
            .ok_or(LiveError::Message("Stale or mismatched utterance result."))?;
        if complete && text.trim().is_empty() {
            complete = false;
        }
        if complete && !row.protected_fields.iter().any(|field| field == "text") {
            row.text = text.to_string();
            row.text_end_sample = Some(row.end_sample);
            row.text_audio_anchor = Some(timing::AudioAnchor {
                start_sample: row.start_sample,
                end_sample: row.end_sample,
            });
            row.machine_revision += 1;
            row.refinement_state =
                if stage == DecodeStage::Refined && row.state == UtteranceState::Sealed {
                    "refined".to_string()
                } else {
                    "provisional".to_string()
                };
        }
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RoutedPart {
    pub start_sample: i64,
    pub end_sample: i64,
    pub text: String,
    pub text_sha256: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RoutedRevision {
    pub text: String,
    pub parts: Vec<RoutedPart>,
}

/// Join successful disjoint same-language raw decodes. Word times stay unset.
pub fn routed_short_revision(passages: &[Passage], start: i64, end: i64) -> Option<RoutedRevision> {
    if passages.len() < 2 || end <= start {
        return None;
    }
    let mut cursor = 0i64;
    let mut language: Option<&str> = None;
    let mut parts = Vec::new();
    for passage in passages {
        if !passage.start.is_finite() || !passage.end.is_finite() {
            return None;
        }
        let first = round_samples(passage.start);
        let last = round_samples(passage.end);
        let raw = passage
            .cohere_raw_text
            .as_deref()
            .unwrap_or(passage.text.as_str());
        let selected = passage.language.as_deref()?;
        let evidence = &passage.acoustic_evidence;
        if first != cursor
            || first >= last
            || last > end - start
            || raw.trim().is_empty()
            || !languages::is_supported(selected)
            || language.is_some_and(|current| current != selected)
            || passage.transcription_review.is_some()
            || passage.audio_state.is_some()
            || evidence.source != speech::SILERO_SOURCE
            || !evidence.complete
            || evidence.decision != SpeechDecision::Speech
            || evidence.start_sample != start + first
            || evidence.end_sample != start + last
            || !evidence.uncertain_regions.is_empty()
            || evidence.speech_regions.is_empty()
        {
            return None;
        }
        language = Some(selected);
        cursor = last;
        parts.push(RoutedPart {
            start_sample: start + first,
            end_sample: start + last,
            text_sha256: sha256_hex(raw.as_bytes()),
            text: raw.to_string(),
        });
    }
    if cursor != end - start {
        return None;
    }
    Some(RoutedRevision {
        text: parts
            .iter()
            .map(|part| part.text.as_str())
            .collect::<Vec<_>>()
            .join(" "),
        parts,
    })
}

pub fn activity_regions(turns: &[Turn], start: i64, end: i64) -> Vec<timing::ActivityRegion> {
    #[cfg(test)]
    tests::record_activity_sweep(start, end, turns.len());
    if start >= end {
        return Vec::new();
    }
    let mut boundaries: BTreeMap<i64, Vec<(&str, i32)>> = BTreeMap::new();
    boundaries.entry(start).or_default();
    boundaries.entry(end).or_default();
    for turn in turns {
        let turn_start = round_samples(turn.start).clamp(start, end);
        let turn_end = round_samples(turn.end).clamp(start, end);
        if turn_start < turn_end {
            boundaries
                .entry(turn_start)
                .or_default()
                .push((&turn.speaker, 1));
            boundaries
                .entry(turn_end)
                .or_default()
                .push((&turn.speaker, -1));
        }
    }
    let mut active = BTreeMap::<&str, i32>::new();
    let mut previous = start;
    let mut regions = Vec::new();
    for (sample, changes) in boundaries {
        if previous < sample {
            regions.push(timing::ActivityRegion {
                start_sample: previous,
                end_sample: sample,
                speakers: active.keys().map(|speaker| (*speaker).to_owned()).collect(),
            });
        }
        for (speaker, delta) in changes {
            let count = active.entry(speaker).or_default();
            *count += delta;
            if *count == 0 {
                active.remove(speaker);
            }
        }
        previous = sample;
    }
    regions
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AnchorRow {
    pub id: String,
    pub start_sample: i64,
    pub end_sample: i64,
    pub machine_revision: i64,
    pub audio_revision: i64,
    pub language_epoch: i64,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DecodeRequest {
    pub utterance_id: String,
    pub machine_revision: i64,
    pub audio_revision: i64,
    pub language_epoch: i64,
    pub core_start_sample: i64,
    pub core_end_sample: i64,
    pub start_sample: i64,
    pub end_sample: i64,
}

fn request_for(
    row: &AnchorRow,
    core_start: i64,
    core_end: i64,
    start: i64,
    end: i64,
) -> DecodeRequest {
    DecodeRequest {
        utterance_id: row.id.clone(),
        machine_revision: row.machine_revision,
        audio_revision: row.audio_revision,
        language_epoch: row.language_epoch,
        core_start_sample: core_start,
        core_end_sample: core_end,
        start_sample: start,
        end_sample: end,
    }
}

/// Physical context windows from `context_plan.py`. Custom edges are optional.
pub fn context_requests(
    row: &AnchorRow,
    edges: Option<&[i64]>,
) -> Result<Vec<DecodeRequest>, LiveError> {
    let (start, end) = (row.start_sample, row.end_sample);
    if start < 0 || end <= start {
        return Err(LiveError::Message("Invalid context audio anchor."));
    }
    let owned;
    let edges = if let Some(edges) = edges {
        edges
    } else {
        let count = ((end - start + CONTEXT_CORE - 1) / CONTEXT_CORE).max(1);
        owned = (0..=count)
            .map(|index| start + (end - start) * index / count)
            .collect::<Vec<_>>();
        owned.as_slice()
    };
    if edges.len() < 2
        || edges.first().copied() != Some(start)
        || edges.last().copied() != Some(end)
        || edges
            .windows(2)
            .any(|pair| pair[1] <= pair[0] || pair[1] - pair[0] > CONTEXT_CORE)
    {
        return Err(LiveError::Message(
            "Context cores must partition bounded original audio.",
        ));
    }
    let mut output = Vec::new();
    for pair in edges.windows(2) {
        let (core_start, core_end) = (pair[0], pair[1]);
        let window_start = start.max(core_start - CONTEXT_PAD);
        let window_end = end.min(core_end + CONTEXT_PAD);
        if window_end - window_start > PHYSICAL_MAX {
            return Err(LiveError::Message("Oversized physical context decode."));
        }
        output.push(request_for(
            row,
            core_start,
            core_end,
            window_start,
            window_end,
        ));
    }
    Ok(output)
}

/// Anchored cores for the authoritative English tail.
pub fn authoritative_requests(row: &AnchorRow) -> Result<Vec<DecodeRequest>, LiveError> {
    let (start, end) = (row.start_sample, row.end_sample);
    if start < 0 || end <= start {
        return Err(LiveError::Message("Invalid original audio anchor."));
    }
    let mut edges = vec![start];
    if end - start > AUTHORITATIVE_MAX {
        let mut cursor = start;
        while cursor + AUTHORITATIVE_CORE < end {
            cursor += AUTHORITATIVE_CORE;
            edges.push(cursor);
        }
    }
    edges.push(end);
    Ok(edges
        .windows(2)
        .map(|pair| {
            let (core_start, core_end) = (pair[0], pair[1]);
            request_for(
                row,
                core_start,
                core_end,
                start.max(core_start - AUTHORITATIVE_PAD),
                end.min(core_end + AUTHORITATIVE_PAD),
            )
        })
        .collect())
}

#[derive(Debug, Clone, PartialEq)]
pub struct TailPart {
    pub request: DecodeRequest,
    pub text: String,
    pub complete: bool,
    pub alignment: Option<Value>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct TailAssembly {
    pub policy: &'static str,
    pub complete: bool,
    pub recognition_complete: bool,
    pub text: Option<String>,
    pub reason: Option<&'static str>,
    pub words: Vec<TimedWord>,
}

/// One selected raw tail. Without validated CTC cells, a multi-part seam stays in review.
pub fn assemble_authoritative(
    row: &AnchorRow,
    requests: &[DecodeRequest],
    parts: &[TailPart],
) -> Result<TailAssembly, LiveError> {
    if parts.is_empty() || parts.len() != requests.len() {
        return Err(LiveError::Message("Missing selected tails."));
    }
    if row.start_sample < 0 || row.end_sample <= row.start_sample {
        return Err(LiveError::Message("Invalid authoritative anchor."));
    }
    let mut cursor = row.start_sample;
    for (request, part) in requests.iter().zip(parts) {
        if part.request != *request
            || request.utterance_id != row.id
            || request.machine_revision != row.machine_revision
            || request.audio_revision != row.audio_revision
            || request.language_epoch != row.language_epoch
            || request.start_sample < row.start_sample
            || request.core_start_sample < request.start_sample
            || request.core_end_sample <= request.core_start_sample
            || request.end_sample < request.core_end_sample
            || request.end_sample > row.end_sample
            || request.end_sample - request.start_sample > AUTHORITATIVE_MAX
            || request.core_start_sample != cursor
        {
            return Err(LiveError::Message("Stale authoritative tail."));
        }
        cursor = request.core_end_sample;
    }
    if cursor != row.end_sample {
        return Err(LiveError::Message("Incomplete original audio partition."));
    }
    let recognition_complete = parts
        .iter()
        .all(|part| part.complete && !part.text.trim().is_empty());
    let base = |reason: Option<&'static str>, complete: bool, text: Option<String>| TailAssembly {
        policy: AUTHORITATIVE_POLICY,
        complete,
        recognition_complete,
        text,
        reason,
        words: Vec::new(),
    };
    if !recognition_complete {
        return Ok(base(
            Some("incomplete_selected_cohere_revision"),
            false,
            None,
        ));
    }
    let whole = parts.iter().all(|part| {
        (part.request.start_sample, part.request.end_sample) == (row.start_sample, row.end_sample)
    });
    if whole {
        return Ok(base(
            None,
            true,
            Some(parts.last().expect("part").text.clone()),
        ));
    }
    let mut composer = context::Composer::default();
    for part in parts {
        if let Err(reason) = composer.push(part.clone()) {
            return Ok(base(Some(reason), false, None));
        }
    }
    match composer.finish() {
        Ok(assembly) => Ok(assembly),
        Err(reason) => Ok(base(Some(reason), false, None)),
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct CoreAnchor {
    pub id: String,
    pub start_sample: i64,
    pub end_sample: i64,
    pub machine_revision: i64,
    pub audio_revision: i64,
    pub language_epoch: i64,
    pub speech_regions: Vec<SampleSpan>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct CoreCut {
    pub sample: i64,
    pub method: &'static str,
    pub negative_receipt: Option<Admission>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct CorePlan {
    pub policy: &'static str,
    pub cap_samples: i64,
    pub radius_samples: i64,
    pub utterance_id: String,
    pub machine_revision: i64,
    pub audio_revision: i64,
    pub language_epoch: i64,
    pub start_sample: i64,
    pub end_sample: i64,
    pub edges: Vec<i64>,
    pub cuts: Vec<CoreCut>,
}

fn negative_gap(evidence: &Admission, start: i64, end: i64) -> bool {
    evidence.complete
        && evidence.decision == SpeechDecision::NoSpeech
        && evidence.input_policy.as_deref() == Some(speech::INPUT_POLICY)
        && evidence.model_revision == speech::SILERO_REVISION
        && evidence.source == speech::SILERO_SOURCE
        && (evidence.start_sample, evidence.end_sample) == (start, end)
        && evidence.speech_regions.is_empty()
        && evidence.uncertain_regions.is_empty()
}

/// Bounded original-audio cores. A verified Silero gap may move the cut; text never does.
pub fn plan_cores(row: &CoreAnchor, mut admission: impl FnMut(i64, i64) -> Admission) -> CorePlan {
    let (start, end) = (row.start_sample, row.end_sample);
    let count = ((end - start + CORE_PLAN_CAP - 1) / CORE_PLAN_CAP).max(1);
    let gaps: Vec<(i64, i64)> = row
        .speech_regions
        .windows(2)
        .filter_map(|pair| {
            let gap = pair[1].start_sample - pair[0].end_sample;
            (gap >= CORE_PLAN_MIN_GAP).then_some((pair[0].end_sample, pair[1].start_sample))
        })
        .collect();
    let mut edges = vec![start];
    let mut cuts = Vec::new();
    for index in 1..count {
        let remaining = count - index;
        let low = (edges[edges.len() - 1] + CORE_PLAN_MIN).max(end - remaining * CORE_PLAN_CAP);
        let high = (edges[edges.len() - 1] + CORE_PLAN_CAP).min(end - remaining * CORE_PLAN_MIN);
        let original_target = start + (end - start) * index / count;
        let target = original_target.clamp(low, high);
        let mut candidates: Vec<(i64, i64, i64)> = gaps
            .iter()
            .filter_map(|(gap_start, gap_end)| {
                let midpoint = (gap_start + gap_end) / 2;
                (low <= midpoint
                    && midpoint <= high
                    && (midpoint - original_target).abs() <= CORE_PLAN_RADIUS)
                    .then_some((*gap_start, *gap_end, midpoint))
            })
            .collect();
        candidates.sort_by_key(|(_, _, midpoint)| ((midpoint - target).abs(), *midpoint));
        candidates.truncate(3);
        let mut chosen = target;
        let mut receipt = None;
        for (gap_start, gap_end, midpoint) in candidates {
            let evidence = admission(gap_start, gap_end);
            if negative_gap(&evidence, gap_start, gap_end) {
                chosen = midpoint;
                receipt = Some(evidence);
                break;
            }
        }
        edges.push(chosen);
        cuts.push(CoreCut {
            sample: chosen,
            method: if receipt.is_some() {
                "verified_model_negative_gap"
            } else {
                "balanced_bounded_fallback"
            },
            negative_receipt: receipt,
        });
    }
    edges.push(end);
    CorePlan {
        policy: CORE_PLAN_POLICY,
        cap_samples: CORE_PLAN_CAP,
        radius_samples: CORE_PLAN_RADIUS,
        utterance_id: row.id.clone(),
        machine_revision: row.machine_revision,
        audio_revision: row.audio_revision,
        language_epoch: row.language_epoch,
        start_sample: start,
        end_sample: end,
        edges,
        cuts,
    }
}

pub fn validate_core_plan(plan: &CorePlan, row: &CoreAnchor) -> Result<Vec<i64>, LiveError> {
    let count = ((row.end_sample - row.start_sample + CORE_PLAN_CAP - 1) / CORE_PLAN_CAP).max(1);
    let edges_ok = plan.policy == CORE_PLAN_POLICY
        && plan.cap_samples == CORE_PLAN_CAP
        && plan.radius_samples == CORE_PLAN_RADIUS
        && plan.utterance_id == row.id
        && plan.machine_revision == row.machine_revision
        && plan.audio_revision == row.audio_revision
        && plan.language_epoch == row.language_epoch
        && plan.start_sample == row.start_sample
        && plan.end_sample == row.end_sample
        && plan.edges.first().copied() == Some(row.start_sample)
        && plan.edges.last().copied() == Some(row.end_sample)
        && plan
            .edges
            .windows(2)
            .all(|pair| pair[1] > pair[0] && pair[1] - pair[0] <= CORE_PLAN_CAP)
        && (plan.edges.len() == 2
            || plan
                .edges
                .windows(2)
                .all(|pair| pair[1] - pair[0] >= CORE_PLAN_MIN))
        && plan.cuts.len() + 2 == plan.edges.len()
        && plan.edges.len() as i64 - 1 == count;
    if !edges_ok {
        return Err(LiveError::Message(
            "Core plan must cover bounded ordered original audio.",
        ));
    }
    let parts = plan.edges.len() - 1;
    for (index, cut) in plan.cuts.iter().enumerate() {
        let step = index as i64 + 1;
        let original_target =
            row.start_sample + (row.end_sample - row.start_sample) * step / parts as i64;
        let low = (plan.edges[index] + CORE_PLAN_MIN)
            .max(row.end_sample - (parts as i64 - step) * CORE_PLAN_CAP);
        let high = (plan.edges[index] + CORE_PLAN_CAP)
            .min(row.end_sample - (parts as i64 - step) * CORE_PLAN_MIN);
        let target = original_target.clamp(low, high);
        if cut.sample != plan.edges[index + 1] {
            return Err(LiveError::Message("Core boundary receipt mismatch."));
        }
        match cut.method {
            "balanced_bounded_fallback" => {
                if cut.negative_receipt.is_some() || cut.sample != target {
                    return Err(LiveError::Message(
                        "Fallback must use the bounded balanced target.",
                    ));
                }
            }
            "verified_model_negative_gap" => {
                let Some(receipt) = &cut.negative_receipt else {
                    return Err(LiveError::Message("Missing core gap evidence."));
                };
                let (gap_start, gap_end) = (receipt.start_sample, receipt.end_sample);
                let between = row
                    .speech_regions
                    .windows(2)
                    .any(|pair| (pair[0].end_sample, pair[1].start_sample) == (gap_start, gap_end));
                if gap_end - gap_start < CORE_PLAN_MIN_GAP
                    || cut.sample != (gap_start + gap_end) / 2
                    || (cut.sample - original_target).abs() > CORE_PLAN_RADIUS
                    || !between
                    || !negative_gap(receipt, gap_start, gap_end)
                {
                    return Err(LiveError::Message(
                        "Unqualified negative gap cannot anchor a core.",
                    ));
                }
            }
            _ => return Err(LiveError::Message("Unknown core boundary method.")),
        }
    }
    Ok(plan.edges.clone())
}

fn segment_bounds(segment: &Segment) -> Result<(i64, i64), LiveError> {
    if !segment.start.is_finite() || !segment.end.is_finite() || segment.start < 0.0 {
        return Err(LiveError::Message(
            "Audio times must be finite and nonnegative.",
        ));
    }
    Ok((round_samples(segment.start), round_samples(segment.end)))
}

/// Source snapshot of the fields a saved accuracy pass is allowed to replace.
pub fn segment_version(segment: &Segment) -> String {
    let value = serde_json::json!({
        "end": segment.end,
        "id": segment.id,
        "machine_revision": segment.machine_revision,
        "protected_fields": segment.protected_fields.as_deref().unwrap_or_default(),
        "speaker": segment.speaker,
        "start": segment.start,
        "text": segment.text,
        "audio_revision": segment.audio_revision,
        "capture_source": segment.capture_source,
        "canonical_utterance_id": segment.canonical_utterance_id,
        "language": segment.language,
        "language_detection": segment.language_detection,
        "language_epoch": segment.rest.get("language_epoch"),
        "language_generation": segment.rest.get("language_generation"),
        "language_mode": segment.rest.get("language_mode"),
        "canonical_machine_revision": segment.rest.get("canonical_machine_revision"),
        "canonical_state": segment.rest.get("canonical_state"),
    });
    sha256_hex(serde_json::to_string(&value).unwrap_or_default().as_bytes())
}

/// Fresh decode of one saved canonical anchor. It owns no live stream state and
/// retains the recorded speaker/activity identity instead of creating new tracks.
pub fn refine_saved_passage<M: Models + ?Sized + 'static>(
    models: &M,
    original: &Segment,
    pcm: &[f32],
    vocabulary: Option<&crate::alignment::Vocabulary>,
) -> Result<Segment, LiveError> {
    let (start, end) = saved_bounds(original)?;
    if end - start > MAX_DECODE_SAMPLES || end - start != pcm.len() as i64 {
        return Err(LiveError::Message(
            "Saved crop exceeds the bounded single-decode limit.",
        ));
    }
    let language = original.rest["language_mode"]
        .as_str()
        .expect("checked mode");
    let processed = processed_catalog(original);
    let (text, detected_language, detection) =
        crate::import::refine_passage_source(models, pcm, language, start, processed)
            .map_err(LiveError::from)?;
    let mut candidate = saved_candidate(original, text, detected_language.clone(), detection, end)?;
    if let Some(vocabulary) = vocabulary {
        if let Some(alignment) = original.rest.get("reading_alignment").filter(|alignment| {
            vocabulary.matches_input(alignment, pcm, &candidate.text, start, &detected_language)
                && context::complete_alignment(TailPart {
                    request: DecodeRequest {
                        utterance_id: candidate
                            .canonical_utterance_id
                            .clone()
                            .expect("saved bounds"),
                        machine_revision: candidate.machine_revision.expect("saved candidate"),
                        audio_revision: candidate.audio_revision.unwrap_or(0),
                        language_epoch: candidate.rest["language_epoch"].as_i64().unwrap_or(0),
                        core_start_sample: start,
                        core_end_sample: end,
                        start_sample: start,
                        end_sample: end,
                    },
                    text: candidate.text.clone(),
                    complete: true,
                    alignment: Some((*alignment).clone()),
                })
        }) {
            crate::alignment::attach(&mut candidate, alignment.clone());
            return Ok(candidate);
        }
        match vocabulary.align(models, pcm, &candidate.text, start, &detected_language) {
            Ok(alignment) => crate::alignment::attach(&mut candidate, alignment),
            Err(
                error @ (ChildError::ProcessLost { .. }
                | ChildError::Timeout
                | ChildError::Cancelled),
            ) => return Err(error.into()),
            Err(_) => {
                candidate.rest.insert(
                    "reading_alignment".into(),
                    serde_json::json!({"status":"unavailable","reason":"model_error"}),
                );
            }
        }
    }
    Ok(candidate)
}

/// Read the saved source in bounded crops, including long canonical anchors.
pub fn refine_saved_recording<M: Models + ?Sized + 'static>(
    models: &M,
    original: &Segment,
    reader: &RecordingReader,
    vocabulary: Option<&crate::alignment::Vocabulary>,
) -> Result<Segment, LiveError> {
    context::refine(models, original, reader, vocabulary)
}

fn saved_bounds(original: &Segment) -> Result<(i64, i64), LiveError> {
    let (start, end) = segment_bounds(original)?;
    let language = original
        .rest
        .get("language_mode")
        .and_then(Value::as_str)
        .unwrap_or("");
    if original.canonical_utterance_id.is_none()
        || original.rest.get("canonical_state").and_then(Value::as_str) != Some("sealed")
        || protected_segment(original)
        || end <= start
        || (language != languages::AUTOMATIC && !languages::is_supported(language))
    {
        return Err(LiveError::Message(
            "This saved passage needs its captured language setting and an unedited sealed anchor.",
        ));
    }
    Ok((start, end))
}

fn processed_catalog(original: &Segment) -> Option<&str> {
    original
        .capture_source
        .as_ref()
        .filter(|s| s["source_id"] == "microphone_clean")
        .and_then(|s| s["catalog_sha256"].as_str())
}

fn saved_candidate(
    original: &Segment,
    text: String,
    detected_language: String,
    detection: Value,
    end: i64,
) -> Result<Segment, LiveError> {
    let mut candidate = original.clone();
    crate::edit::clear_alignment_bundle(&mut candidate);
    candidate.text = text;
    candidate.language = Some(detected_language);
    candidate.language_detection = Some(detection);
    candidate.machine_revision = Some(
        original
            .machine_revision
            .unwrap_or(0)
            .checked_add(1)
            .ok_or(LiveError::Message("Passage revision exhausted."))?,
    );
    let canonical = original
        .rest
        .get("canonical_machine_revision")
        .and_then(Value::as_i64)
        .unwrap_or(original.machine_revision.unwrap_or(0))
        .checked_add(1)
        .ok_or(LiveError::Message("Canonical revision exhausted."))?;
    candidate
        .rest
        .insert("canonical_machine_revision".into(), Value::from(canonical));
    candidate
        .rest
        .insert("canonical_text_end_sample".into(), Value::from(end));
    candidate.rest.remove("canonical_unresolved");
    candidate.rest.remove("cohere_raw_text");
    candidate.transcription_review = None;
    candidate.retry_candidate = None;
    candidate.review_resolution = None;
    candidate.rest.remove("text_audio_anchor");
    candidate.rest.remove("audio_float32_sha256");
    Ok(candidate)
}

fn protected_segment(segment: &Segment) -> bool {
    segment
        .protected_fields
        .as_ref()
        .is_some_and(|fields| !fields.is_empty())
}

/// Publish one complete canonical candidate without changing row order or identity.
/// Stale or protected passages remain untouched; candidate timing already carries
/// its final machine revision before reaching this boundary.
pub fn publish_saved_passage(
    document: &mut Document,
    original: &Segment,
    candidate: &Segment,
) -> Result<bool, LiveError> {
    let Some(current) = document
        .segments
        .iter_mut()
        .find(|row| row.id == original.id)
    else {
        return Ok(false);
    };
    if protected_segment(current) || segment_version(current) != segment_version(original) {
        return Ok(false);
    }
    let (start, end) = segment_bounds(original)?;
    let next_revision = original.machine_revision.unwrap_or(0).checked_add(1);
    let next_canonical = original
        .rest
        .get("canonical_machine_revision")
        .and_then(Value::as_i64)
        .unwrap_or(original.machine_revision.unwrap_or(0))
        .checked_add(1);
    if start < 0
        || end <= start
        || original.canonical_utterance_id.is_none()
        || original.rest.get("canonical_state").and_then(Value::as_str) != Some("sealed")
        || candidate.id != original.id
        || candidate.start != original.start
        || candidate.end != original.end
        || candidate.speaker != original.speaker
        || candidate.capture_source != original.capture_source
        || candidate.audio_revision != original.audio_revision
        || candidate.canonical_utterance_id != original.canonical_utterance_id
        || [
            "language_epoch",
            "language_generation",
            "language_mode",
            "canonical_state",
        ]
        .iter()
        .any(|key| candidate.rest.get(*key) != original.rest.get(*key))
        || next_revision.is_none()
        || candidate.machine_revision != next_revision
        || next_canonical.is_none()
        || candidate
            .rest
            .get("canonical_machine_revision")
            .and_then(Value::as_i64)
            != next_canonical
        || candidate.text.trim().is_empty()
        || candidate.transcription_review.is_some()
    {
        return Err(LiveError::Message(
            "Invalid saved canonical refinement candidate.",
        ));
    }
    let mut written = candidate.clone();
    written.refinement_state = Some("refined".into());
    written.rest.insert("finalized".into(), Value::Bool(true));
    written.rest.insert(
        "audio_anchor".into(),
        serde_json::json!({"start_sample":start,"end_sample":end}),
    );
    written
        .rest
        .insert("refinement_window".into(), Value::from(original.id.clone()));
    *current = written;
    Ok(true)
}

/// A persisted edit to an open passage closes its machine-owned text extent.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EditedBoundary {
    pub utterance_id: String,
    pub end_sample: i64,
}

pub fn edited_boundary(document: &Document, candidate: &Segment) -> Option<EditedBoundary> {
    let prior = document
        .segments
        .iter()
        .find(|row| row.id == candidate.id)?;
    let end = prior.rest.get("canonical_text_boundary_sample")?.as_i64()?;
    let candidate_end = candidate.rest.get("end_sample")?.as_i64()?;
    (candidate_end > end
        || (candidate_end == end
            && candidate
                .rest
                .get("canonical_state")
                .and_then(Value::as_str)
                == Some("open")))
    .then(|| EditedBoundary {
        utterance_id: prior.id.clone(),
        end_sample: end,
    })
}

/// Keep fields [`crate::edit::edit_passage`] marked protected when a canonical candidate arrives.
pub fn apply_canonical_candidate(document: &Document, candidate: &Segment) -> Document {
    if edited_boundary(document, candidate).is_some() {
        return document.clone();
    }
    let mut document = document.clone();
    if let Some(prior) = document
        .segments
        .iter()
        .find(|segment| segment.id == candidate.id)
        .cloned()
    {
        // A completed saved-anchor revision outranks later draft snapshots of that same
        // audio. Refinement never changes the live book's acoustic/identity ownership.
        if prior.refinement_state.as_deref() == Some("refined")
            && prior.canonical_utterance_id == candidate.canonical_utterance_id
            && prior.audio_revision == candidate.audio_revision
            && prior.rest.get("start_sample") == candidate.rest.get("start_sample")
            && prior.rest.get("end_sample") == candidate.rest.get("end_sample")
        {
            return document;
        }
        // Catalog/source identity is server-owned and cannot change under a passage.
        if prior.capture_source != candidate.capture_source {
            return document;
        }
        let mut row = candidate.clone();
        if let Some(boundary) = prior.rest.get("canonical_text_boundary_sample") {
            row.rest
                .insert("canonical_text_boundary_sample".into(), boundary.clone());
        }
        let protected = prior.protected_fields.clone().unwrap_or_default();
        row.protected_fields = if protected.is_empty() {
            prior.protected_fields.clone()
        } else {
            Some(protected.clone())
        };
        for field in &protected {
            match field.as_str() {
                "text" => row.text = prior.text.clone(),
                "speaker" => row.speaker = prior.speaker.clone(),
                "start" => row.start = prior.start,
                "end" => row.end = prior.end,
                _ => {}
            }
        }
        if !protected.is_empty() {
            row.refinement_state = Some("edited".to_string());
            row.machine_revision = prior.machine_revision;
        }
        let kept_text = protected.iter().any(|field| field == "text")
            || (row.text.trim().is_empty() && !prior.text.trim().is_empty());
        if row.text.trim().is_empty() && !prior.text.trim().is_empty() {
            row.text = prior.text.clone();
        }
        if protected.iter().any(|field| field == "text") {
            row.language = prior.language.clone();
            row.language_detection = prior.language_detection.clone();
        }
        if kept_text {
            // A later decode does not extend the audio represented by preserved words.
            for key in ["canonical_text_end_sample", "text_audio_anchor"] {
                row.rest.remove(key);
                if let Some(value) = prior.rest.get(key) {
                    row.rest.insert(key.into(), value.clone());
                }
            }
        }
        if protected
            .iter()
            .any(|field| field == "start" || field == "end")
        {
            row.rest.remove("text_audio_anchor");
        }
        if kept_text || !protected.is_empty() {
            crate::edit::clear_alignment_bundle(&mut row);
        }
        if let Some(slot) = document
            .segments
            .iter_mut()
            .find(|segment| segment.id == row.id)
        {
            *slot = row;
        }
    } else {
        let mut row = candidate.clone();
        row.machine_revision.get_or_insert(0);
        document.segments.push(row);
        document
            .segments
            .sort_by(|left, right| left.start.total_cmp(&right.start));
    }
    if !candidate.speaker.is_empty() {
        document
            .speakers
            .entry(candidate.speaker.clone())
            .or_insert_with(|| speaker_label(&candidate.speaker));
    }
    document
}

#[derive(Debug, Clone, PartialEq)]
pub struct EngineConfig {
    pub job_id: String,
    pub language: String,
    pub language_epoch: i64,
    pub audio_batch_seconds: i64,
    pub fast_sequence: i64,
    pub source_id: Option<String>,
    /// Validated host catalog identity for source-specific candidates.
    pub capture_catalog_sha256: Option<String>,
    /// Exceptional restart: rebuild acoustic state through this saved prefix before polling.
    pub recovery: Option<StreamRecovery>,
    /// Set only after the host verifies the processed microphone's source-bound receipt.
    pub processed_microphone_contract: Option<String>,
}

/// The host owns replay sequencing and the persisted publication boundary. Historical
/// PCM is fed with `handle`; the inbox accepts only audio after `through_sample`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct StreamRecovery {
    pub generation: u32,
    pub start_sample: i64,
    pub through_sample: i64,
    pub language_epoch: i64,
    /// Independent immutable publication cutoffs for the two retained source files.
    pub source_boundaries: BTreeMap<String, RecoveryBoundary>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RecoveryBoundary {
    pub start_sample: i64,
    pub language_epoch: i64,
}

impl StreamRecovery {
    fn validate(&self) -> Result<(), LiveError> {
        if self.generation == 0
            || self.generation.checked_mul(64).is_none()
            || self.start_sample < 0
            || self.through_sample < self.start_sample
            || u32::try_from(self.through_sample).is_err()
            || self.language_epoch < 0
            || self.source_boundaries.iter().any(|(source, cut)| {
                !SOURCE_IDS.contains(&source.as_str())
                    || cut.start_sample < 0
                    || cut.start_sample > self.through_sample
                    || cut.language_epoch < 0
            })
        {
            return Err(LiveError::Message("Invalid recovery publication boundary."));
        }
        Ok(())
    }
}

impl EngineConfig {
    pub fn new(job_id: impl Into<String>, language: impl Into<String>) -> Self {
        Self {
            job_id: job_id.into(),
            language: language.into(),
            language_epoch: 0,
            audio_batch_seconds: 6,
            fast_sequence: 0,
            source_id: None,
            capture_catalog_sha256: None,
            recovery: None,
            processed_microphone_contract: None,
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
struct LanguageStamp {
    start_sample: i64,
    language: String,
    epoch: i64,
}

#[derive(Debug, Clone, PartialEq)]
pub enum EngineEvent {
    CanonicalRevision {
        candidate: Box<Segment>,
        fast_sequence: i64,
    },
    Progress {
        processed_seconds: f64,
        received_seconds: f64,
        flush: bool,
    },
    FlushAck {
        request_id: String,
        received_sample: i64,
        available_sample: i64,
        speech_observed_sample: i64,
        fast_sequence: i64,
    },
    CaptureFinished {
        duration: f64,
        canonical_observed_sample: i64,
        canonical_uncertain_samples: i64,
    },
    LanguageRegistered {
        generation: i64,
        language: String,
        start_sample: i64,
    },
    Finished {
        duration: f64,
    },
}

struct SharedLedger(Arc<Mutex<SpeechLedger>>);

impl AcousticEvidence for SharedLedger {
    fn admit(&mut self, start: i64, end: i64) -> Result<Admission, SpeechError> {
        self.0.lock().expect("speech ledger").admission(start, end)
    }
}

struct ModelDetector<M: Models> {
    models: Arc<M>,
    failure: Arc<Mutex<Option<ChildError>>>,
}

impl<M: Models> LanguageDetector for ModelDetector<M> {
    fn detect(&mut self, pcm: &[f32]) -> Result<Vec<(String, f64)>, crate::language::RouteError> {
        let probe = self.models.language_probe(pcm).map_err(|error| {
            *self.failure.lock().expect("model failure") = Some(error);
            crate::language::RouteError::Detector
        })?;
        if probe.codes.len() != probe.probabilities.len() || probe.codes.is_empty() {
            return Err(crate::language::RouteError::Detector);
        }
        Ok(probe
            .codes
            .into_iter()
            .zip(probe.probabilities.into_iter().map(f64::from))
            .collect())
    }
}

struct ModelRecognizer<M: Models> {
    models: Arc<M>,
    failure: Arc<Mutex<Option<ChildError>>>,
}

impl<M: Models> Recognizer for ModelRecognizer<M> {
    fn transcribe(
        &mut self,
        pcm: &[f32],
        language: &str,
    ) -> Result<AsrOutput, crate::language::RouteError> {
        let text = self
            .models
            .transcribe(pcm, language, (0, 0))
            .map_err(|error| {
                *self.failure.lock().expect("model failure") = Some(error);
                crate::language::RouteError::Recognition
            })?;
        let raw = if text.raw_text.is_empty() {
            text.text
        } else {
            text.raw_text
        };
        let token_count = text.token_count as usize;
        Ok(AsrOutput {
            text: raw,
            token_count,
        })
    }
}

#[derive(Default)]
struct DecodeProgress {
    end_sample: i64,
    audio_revision: i64,
    completed: Option<CompletedCores>,
}

/// Accepted disjoint cores of the currently growing utterance. The final core
/// remains an active tail until later audio requires another bounded cut.
struct CompletedCores {
    end_sample: i64,
    text: String,
    decisions: Vec<crate::language::ProbeEvent>,
    review: bool,
}

/// Canonical resident engine. PCM stays on disk; weights stay behind `M`.
pub struct Engine<M: Models> {
    config: EngineConfig,
    models: Arc<M>,
    inbox: Arc<Mutex<Inbox>>,
    recording: RecordingReader,
    committed: u32,
    received: i64,
    turns: Vec<Turn>,
    diar_observed: i64,
    timeline: Vec<LanguageStamp>,
    capture_finished: bool,
    fast_sequence: i64,
    book: UtteranceBook,
    last_decoded: HashMap<String, DecodeProgress>,
    published: HashMap<String, Segment>,
    next_utterance: usize,
    vad: Option<Handle>,
    diar: Option<Handle>,
    ledger: Arc<Mutex<SpeechLedger>>,
    word_alignment: Option<Arc<crate::alignment::Vocabulary>>,
    events: Vec<EngineEvent>,
}

impl<M: Models> Drop for Engine<M> {
    fn drop(&mut self) {
        if let Some(handle) = self.vad.take() {
            let _ = self.models.vad_close(handle);
        }
        if let Some(handle) = self.diar.take() {
            let _ = self.models.diar_close(handle);
        }
    }
}

impl<M: Models + 'static> Engine<M> {
    pub fn new(
        config: EngineConfig,
        models: Arc<M>,
        inbox: Arc<Mutex<Inbox>>,
        recording: RecordingReader,
    ) -> Result<Self, LiveError> {
        let namespace = if let Some(recovery) = &config.recovery {
            recovery.validate()?;
            format!("{}:stream:{}", config.job_id, recovery.generation)
        } else {
            config.job_id.clone()
        };
        let namespace = if let Some(source) = &config.source_id {
            if !SOURCE_IDS.contains(&source.as_str()) {
                return Err(LiveError::Message("Unknown capture input."));
            }
            format!("{namespace}:{source}")
        } else {
            namespace
        };
        let mut book = UtteranceBook::new(namespace)?;
        if let Some(recovery) = &config.recovery {
            let (start, epoch) = if let Some(source) = &config.source_id {
                let boundary = recovery
                    .source_boundaries
                    .get(source)
                    .ok_or(LiveError::Message(
                        "Recovery requires each source publication boundary.",
                    ))?;
                (boundary.start_sample, boundary.language_epoch)
            } else {
                (recovery.start_sample, recovery.language_epoch)
            };
            book.cursor = start;
            book.epoch_start = start;
            book.epoch = epoch;
        }
        let ledger = if let Some(contract) = &config.processed_microphone_contract {
            if config.source_id.as_deref() != Some("microphone_clean") {
                return Err(LiveError::Message(
                    "Unity conditioning requires the verified microphone source.",
                ));
            }
            SpeechLedger::with_processed_microphone(0, contract)?
        } else {
            SpeechLedger::new(0)?
        };
        let timeline = vec![LanguageStamp {
            start_sample: 0,
            language: config.language.clone(),
            epoch: config.language_epoch,
        }];
        Ok(Self {
            fast_sequence: config.fast_sequence,
            config,
            models,
            inbox,
            recording,
            committed: 0,
            received: 0,
            turns: Vec::new(),
            diar_observed: 0,
            timeline,
            capture_finished: false,
            book,
            last_decoded: HashMap::new(),
            published: HashMap::new(),
            next_utterance: 0,
            vad: None,
            diar: None,
            ledger: Arc::new(Mutex::new(ledger)),
            word_alignment: None,
            events: Vec::new(),
        })
    }

    pub fn single(
        config: EngineConfig,
        models: M,
        recording: impl AsRef<Path>,
    ) -> Result<Self, LiveError> {
        let batch = config.audio_batch_seconds;
        let models = Arc::new(models);
        let inbox = Arc::new(Mutex::new(Inbox::new(batch)?));
        if let Some(recovery) = &config.recovery {
            recovery.validate()?;
            inbox.lock().expect("inbox").received = recovery.through_sample;
        }
        let recording =
            RecordingReader::open(recording).map_err(|e| LiveError::Detail(e.to_string()))?;
        Self::new(config, models, inbox, recording)
    }

    pub fn inbox(&self) -> Arc<Mutex<Inbox>> {
        Arc::clone(&self.inbox)
    }

    /// The host installs the already-validated vocabulary once per worker.
    pub fn set_word_alignment(&mut self, vocabulary: Arc<crate::alignment::Vocabulary>) {
        self.word_alignment = Some(vocabulary);
    }

    pub fn push(&self, command: LiveCommand) -> Result<(), LiveError> {
        self.inbox.lock().expect("inbox").push(command)
    }

    pub fn poll(&mut self) -> Result<Option<Vec<EngineEvent>>, LiveError> {
        let message = self.inbox.lock().expect("inbox").take()?;
        message.map(|command| self.handle(command)).transpose()
    }

    pub fn handle(&mut self, message: LiveCommand) -> Result<Vec<EngineEvent>, LiveError> {
        self.events.clear();
        match message {
            LiveCommand::Audio(notice) => self.on_audio(notice)?,
            LiveCommand::Flush {
                request_id,
                through_sample,
            } => self.on_flush(request_id, through_sample)?,
            LiveCommand::Stop => self.on_stop()?,
            LiveCommand::Language {
                language,
                generation,
                start_sample,
            } => self.on_language(language, generation, start_sample)?,
            LiveCommand::Shutdown => {
                self.close_streams()?;
                self.events.push(EngineEvent::Finished {
                    duration: self.received as f64 / RATE as f64,
                });
            }
        }
        Ok(std::mem::take(&mut self.events))
    }

    /// Publication feedback, serviced by the same serial owner even after final ASR.
    /// Only this known utterance is split; no historical PCM or document replay occurs.
    pub fn continue_edited(
        &mut self,
        boundary: &EditedBoundary,
    ) -> Result<Vec<EngineEvent>, LiveError> {
        self.events.clear();
        let Some(original) = self.book.rows.get(&boundary.utterance_id).cloned() else {
            return Ok(Vec::new());
        };
        let cut = boundary.end_sample;
        if cut <= original.start_sample || cut > original.end_sample {
            return Err(LiveError::Message("Invalid edited text boundary."));
        }
        if original
            .protected_fields
            .iter()
            .any(|field| field == "text")
            && original.end_sample == cut
        {
            return Ok(Vec::new());
        }
        let mut tail = original.clone();
        tail.id = utterance_id(&self.book.job_id, original.language_epoch, cut);
        tail.start_sample = cut;
        tail.speech_regions = original
            .speech_regions
            .iter()
            .filter_map(|span| {
                (span.end_sample > cut).then_some(SampleSpan {
                    start_sample: span.start_sample.max(cut),
                    end_sample: span.end_sample,
                })
            })
            .collect();
        tail.text.clear();
        tail.text_audio_anchor = None;
        tail.text_end_sample = None;
        tail.protected_fields.clear();
        tail.machine_revision = 0;
        tail.audio_revision = 0;
        tail.refinement_state = "provisional".into();
        tail.canonical_unresolved = None;
        tail.language = None;
        tail.language_detection = None;
        tail.audio_state = None;
        tail.transcription_review = false;
        tail.review = false;
        clip_retained_activity(&mut tail, original.speaker_activity.as_ref());
        let old = self
            .book
            .rows
            .get_mut(&boundary.utterance_id)
            .expect("known utterance");
        UtteranceBook::set_end(old, cut);
        old.text_end_sample = Some(cut);
        old.state = UtteranceState::Sealed;
        old.protected_fields.push("text".into());
        old.speech_regions = original
            .speech_regions
            .iter()
            .filter_map(|span| {
                (span.start_sample < cut).then_some(SampleSpan {
                    start_sample: span.start_sample,
                    end_sample: span.end_sample.min(cut),
                })
            })
            .collect();
        old.last_speech_sample = old.last_speech_sample.min(cut);
        clip_retained_activity(old, original.speaker_activity.as_ref());
        let old = old.clone();
        self.publish(old)?;
        self.published.remove(&boundary.utterance_id);
        self.last_decoded.remove(&boundary.utterance_id);
        let was_active = self.book.active.as_ref() == Some(&boundary.utterance_id);
        if was_active {
            self.book.active = None;
        }
        if !tail.speech_regions.is_empty() {
            let index = self
                .book
                .order
                .iter()
                .position(|id| id == &boundary.utterance_id)
                .expect("known order")
                + 1;
            if self.book.rows.contains_key(&tail.id) {
                return Err(LiveError::Message("Continuation identity already exists."));
            }
            self.book.order.insert(index, tail.id.clone());
            if index <= self.next_utterance {
                self.next_utterance += 1;
            }
            if was_active {
                self.book.active = Some(tail.id.clone());
            }
            if tail.state == UtteranceState::Open {
                self.next_utterance = self.next_utterance.min(index);
            }
            let identity = tail.id.clone();
            let decode = tail.state == UtteranceState::Sealed
                || self.capture_finished
                || tail.end_sample - cut >= 6 * RATE;
            self.book.rows.insert(identity.clone(), tail);
            if decode {
                self.decode(&identity)?;
            }
            let tail = self.book.rows[&identity].clone();
            self.publish(tail)?;
            if self.book.rows[&identity].state == UtteranceState::Sealed {
                self.published.remove(&identity);
                self.last_decoded.remove(&identity);
            }
        }
        Ok(std::mem::take(&mut self.events))
    }

    fn on_audio(&mut self, notice: AudioNotice) -> Result<(), LiveError> {
        if self.capture_finished {
            return Err(LiveError::Message("Audio arrived after capture finished."));
        }
        if notice.start_sample != self.received || notice.end_sample <= notice.start_sample {
            return Err(LiveError::Message("Invalid saved audio endpoint."));
        }
        let endpoint = u32::try_from(notice.end_sample)
            .map_err(|_| LiveError::Message("Invalid saved audio endpoint."))?;
        self.recording
            .require_samples(endpoint)
            .map_err(|error| LiveError::Detail(error.to_string()))?;
        self.committed = endpoint;
        if self
            .timeline
            .last()
            .is_some_and(|stamp| stamp.epoch != notice.language_epoch)
        {
            self.timeline.push(LanguageStamp {
                start_sample: notice.start_sample,
                language: notice.language.clone(),
                epoch: notice.language_epoch,
            });
        }
        let mut cursor = notice.start_sample;
        while cursor < notice.end_sample {
            let next = (cursor + RATE).min(notice.end_sample);
            self.feed_range(cursor, next)?;
            let current = next == notice.end_sample
                && !self
                    .inbox
                    .lock()
                    .expect("inbox")
                    .newer_audio_queued(next, notice.language_epoch);
            self.commit(false, current)?;
            cursor = next;
        }
        self.events.push(EngineEvent::Progress {
            processed_seconds: self.processed_sample() as f64 / RATE as f64,
            received_seconds: self.received as f64 / RATE as f64,
            flush: false,
        });
        Ok(())
    }

    fn processed_sample(&self) -> i64 {
        self.received
            .min(self.ledger.lock().expect("speech ledger").end_sample())
            .min(self.diar_observed)
    }

    fn feed_range(&mut self, start: i64, end: i64) -> Result<(), LiveError> {
        if start != self.received {
            return Err(LiveError::Message("Noncontiguous captured audio."));
        }
        let pcm = self.read_pcm(start, end)?;
        self.ensure_streams()?;
        let vad = self.vad.expect("vad");
        let feed = self.models.vad_feed(vad, &pcm, false)?;
        self.admit_frames(end, feed)?;
        let diar = self.diar.expect("diar");
        let diarized = self.models.diar_feed(diar, &pcm, false)?;
        self.admit_diarization(end, diarized)?;
        self.received = end;
        Ok(())
    }

    fn admit_diarization(
        &mut self,
        received: i64,
        feed: crate::ipc::DiarFeedOut,
    ) -> Result<(), LiveError> {
        feed.validate(received as u64, self.diar_observed as u64)?;
        self.diar_observed = feed.observed as i64;
        // Validation confines this batch to [previous observed, new observed).
        // End ordering therefore holds across batches as well as within one.
        let mut turns = feed.turns;
        turns.sort_by_key(|turn| turn.end);
        for turn in turns {
            let speaker = self.prefix_speaker(&turn.speaker)?;
            self.turns.push(Turn {
                start: turn.start as f64 / RATE as f64,
                end: turn.end as f64 / RATE as f64,
                speaker,
            });
        }
        Ok(())
    }

    fn admit_frames(&mut self, received: i64, feed: VadFeedOut) -> Result<(), LiveError> {
        let mut cursor = self.ledger.lock().expect("speech ledger").end_sample();
        if feed.received != received as u64
            || feed.observed > feed.received
            || feed.observed < cursor as u64
        {
            return Err(LiveError::Message("Invalid Silero observation endpoint."));
        }
        for frame in feed.frames {
            let start = i64::try_from(frame.start).unwrap_or(-1);
            let end = i64::try_from(frame.end).unwrap_or(-1);
            if start != cursor
                || end <= start
                || frame.end > feed.observed
                || end - start > speech::FRAME_SAMPLES as i64
            {
                return Err(LiveError::Message(
                    "Invalid or noncontiguous Silero evidence.",
                ));
            }
            let samples = self.read_pcm(start, end)?;
            self.ledger.lock().expect("speech ledger").append_pcm(
                start,
                &samples,
                f64::from(frame.raw_p),
                f64::from(frame.norm_p),
            )?;
            cursor = end;
        }
        if cursor as u64 != feed.observed {
            return Err(LiveError::Message(
                "Invalid or noncontiguous Silero evidence.",
            ));
        }
        Ok(())
    }

    fn on_flush(
        &mut self,
        request_id: Option<String>,
        through_sample: Option<i64>,
    ) -> Result<(), LiveError> {
        if let Some(request_id) = &request_id
            && (request_id.is_empty() || through_sample != Some(self.received))
        {
            return Err(LiveError::Message(
                "Pause flush must follow its exact captured audio endpoint.",
            ));
        }
        self.commit(true, true)?;
        self.events.push(EngineEvent::Progress {
            processed_seconds: self.processed_sample() as f64 / RATE as f64,
            received_seconds: self.received as f64 / RATE as f64,
            flush: true,
        });
        if let Some(request_id) = request_id {
            let observed = self.ledger.lock().expect("speech ledger").end_sample();
            self.events.push(EngineEvent::FlushAck {
                request_id,
                received_sample: self.received,
                available_sample: self.received.min(observed).min(self.diar_observed),
                speech_observed_sample: self.received.min(observed),
                fast_sequence: self.fast_sequence,
            });
        }
        Ok(())
    }

    fn on_stop(&mut self) -> Result<(), LiveError> {
        if self.capture_finished {
            return Ok(());
        }
        let result = self.finish_capture();
        let cleanup = self.close_streams();
        result.and(cleanup)
    }

    fn finish_capture(&mut self) -> Result<(), LiveError> {
        self.ensure_streams()?;
        if let Some(vad) = self.vad {
            let feed = self.models.vad_feed(vad, &[], true)?;
            self.admit_frames(self.received, feed)?;
            if self.ledger.lock().expect("speech ledger").end_sample() != self.received {
                return Err(LiveError::Message("Incomplete final Silero observations."));
            }
        }
        if let Some(diar) = self.diar {
            let feed = self.models.diar_feed(diar, &[], true)?;
            self.admit_diarization(self.received, feed)?;
            if self.diar_observed != self.received / 160 * 160 {
                return Err(LiveError::Message(
                    "Incomplete final diarization observations.",
                ));
            }
        }
        self.capture_finished = true;
        self.commit(true, true)?;
        self.events.push(EngineEvent::Progress {
            processed_seconds: self.processed_sample() as f64 / RATE as f64,
            received_seconds: self.received as f64 / RATE as f64,
            flush: true,
        });
        self.events.push(EngineEvent::CaptureFinished {
            duration: self.received as f64 / RATE as f64,
            canonical_observed_sample: self.book.cursor(),
            canonical_uncertain_samples: self.book.uncertain_sample_count(),
        });
        Ok(())
    }

    fn on_language(
        &mut self,
        language: String,
        generation: i64,
        start_sample: i64,
    ) -> Result<(), LiveError> {
        let current = self.timeline.last().expect("timeline").epoch;
        if generation != current + 1 || start_sample != self.received {
            return Err(LiveError::Message("Invalid ordered language boundary."));
        }
        self.timeline.push(LanguageStamp {
            start_sample,
            language: language.clone(),
            epoch: generation,
        });
        self.commit(false, true)?;
        self.events.push(EngineEvent::LanguageRegistered {
            generation,
            language,
            start_sample,
        });
        Ok(())
    }

    fn ensure_streams(&mut self) -> Result<(), LiveError> {
        if self.vad.is_none() {
            let conditioning = self
                .config
                .processed_microphone_contract
                .as_ref()
                .map(|_| GainPolicy { peak: None });
            let vad = self.models.vad_open(conditioning)?;
            match self.models.diar_open(DiarPreset::Low) {
                Ok(diar) => {
                    self.vad = Some(vad);
                    self.diar = Some(diar);
                }
                Err(error) => {
                    let _ = self.models.vad_close(vad);
                    return Err(error.into());
                }
            }
        }
        Ok(())
    }

    fn close_streams(&mut self) -> Result<(), LiveError> {
        let vad = self
            .vad
            .take()
            .map(|handle| self.models.vad_close(handle))
            .transpose();
        let diar = self
            .diar
            .take()
            .map(|handle| self.models.diar_close(handle))
            .transpose();
        vad?;
        diar?;
        Ok(())
    }

    fn read_pcm(&self, start: i64, end: i64) -> Result<Vec<f32>, LiveError> {
        if start < 0 || end <= start || end - start > MAX_DECODE_SAMPLES {
            return Err(LiveError::Message("Oversized audio read."));
        }
        self.recording
            .read_samples(start as u32, end as u32, self.committed)
            .map_err(|error| LiveError::Detail(error.to_string()))
    }

    fn prefix_speaker(&self, local: &str) -> Result<String, LiveError> {
        let speaker = match &self.config.source_id {
            Some(source) => speaker_id(source, local),
            None => Ok(local.to_string()),
        }?;
        let Some(recovery) = &self.config.recovery else {
            return Ok(speaker);
        };
        let index = speaker
            .strip_prefix("speaker_")
            .and_then(|v| v.parse::<u32>().ok())
            .filter(|v| *v < 64)
            .ok_or(LiveError::Message("Invalid recovered speaker slot."))?;
        let offset = recovery
            .generation
            .checked_mul(64)
            .and_then(|v| v.checked_add(index))
            .ok_or(LiveError::Message("Recovered speaker namespace exhausted."))?;
        Ok(format!("speaker_{offset}"))
    }

    fn language_at(&self, sample: i64) -> LanguageStamp {
        self.timeline
            .iter()
            .rev()
            .find(|stamp| stamp.start_sample <= sample)
            .expect("timeline covers the origin")
            .clone()
    }

    fn commit(&mut self, final_pass: bool, decode_open: bool) -> Result<(), LiveError> {
        let available = {
            let ledger = self.ledger.lock().expect("speech ledger");
            let speech = self.received.min(ledger.end_sample());
            if self.capture_finished {
                speech
            } else {
                speech.min(self.diar_observed)
            }
        };
        loop {
            let boundary = self
                .timeline
                .iter()
                .find(|stamp| stamp.epoch == self.book.epoch + 1);
            if let Some(stamp) = boundary
                && stamp.start_sample == self.book.cursor()
            {
                self.book.boundary(stamp.start_sample, stamp.epoch)?;
                continue;
            }
            let end = boundary.map_or(available, |stamp| available.min(stamp.start_sample));
            if self.book.cursor() >= end {
                break;
            }
            let admission = self
                .ledger
                .lock()
                .expect("speech ledger")
                .admission(self.book.cursor(), end)?;
            self.book.observe(&admission)?;
        }
        if self.capture_finished && available == self.received && !self.book.closed() {
            self.book.finish();
        }
        let order = self.book.order[self.next_utterance..].to_vec();
        for identity in order {
            if !self.book.rows.contains_key(&identity) {
                continue;
            }
            let row = self.book.rows.get(&identity).expect("row");
            let previous = self.published.get(&identity);
            if row.state == UtteranceState::Sealed
                && previous
                    .and_then(|segment| segment.rest.get("canonical_state"))
                    .and_then(Value::as_str)
                    == Some("sealed")
                && previous.and_then(|segment| segment.audio_revision) == Some(row.audio_revision)
            {
                continue;
            }
            let last = self.last_decoded.get(&identity);
            let changed = last.is_none_or(|last| last.audio_revision != row.audio_revision);
            let length = row.end_sample - row.start_sample;
            let mut due = final_pass
                || row.state == UtteranceState::Sealed
                || (length >= 6 * RATE
                    && (last.is_none()
                        || (decode_open
                            && row.end_sample - last.expect("decoded").end_sample >= 3 * RATE)));
            let long_unaligned = length > MAX_DECODE_SAMPLES;
            if long_unaligned && row.state == UtteranceState::Open && !final_pass {
                due = last.is_none()
                    || (decode_open
                        && row.end_sample - last.expect("decoded").end_sample >= 6 * RATE);
            }
            let decode =
                changed && due && !row.protected_fields.iter().any(|field| field == "text");
            self.attach(&identity)?;
            if decode {
                self.decode(&identity)?;
            }
            let row = self.book.rows.get(&identity).expect("row").clone();
            self.publish(row)?;
        }
        while let Some(identity) = self.book.order.get(self.next_utterance) {
            if self.book.rows[identity].state != UtteranceState::Sealed {
                break;
            }
            self.published.remove(identity);
            self.last_decoded.remove(identity);
            self.next_utterance += 1;
        }
        let mut earliest = self.book.cursor().saturating_sub(UTTERANCE_CONTEXT_SAMPLES);
        if let Some(id) = &self.book.active {
            let row = &self.book.rows[id];
            let tail = row
                .speaker_activity
                .as_ref()
                .and_then(|activity| activity.regions.last())
                .map_or(row.start_sample, |region| region.start_sample);
            earliest = earliest.min(tail);
        }
        let expired = self
            .turns
            .partition_point(|turn| round_samples(turn.end) <= earliest);
        self.turns.drain(..expired);
        Ok(())
    }

    fn attach(&mut self, identity: &str) -> Result<(), LiveError> {
        let row = self.book.rows.get_mut(identity).expect("row");
        let observed = row.end_sample.min(self.diar_observed);
        let activity = row
            .speaker_activity
            .get_or_insert_with(|| timing::SpeakerActivity {
                audio_revision: row.audio_revision,
                observed_end_sample: Some(row.start_sample),
                regions: Vec::new(),
            });
        activity.audio_revision = row.audio_revision;
        if observed <= row.start_sample {
            activity.observed_end_sample = Some(observed);
            activity.regions.clear();
            row.speaker_candidates.clear();
            return Ok(());
        }
        if activity
            .observed_end_sample
            .is_some_and(|end| end >= observed)
        {
            return Ok(());
        }
        // The last region may have ended at a temporary VAD watermark inside a
        // diarization turn. Rebuild it with the new suffix, preserving exact turn
        // boundaries rather than introducing a boundary on every audio packet.
        let start = activity
            .regions
            .pop()
            .map_or(row.start_sample, |region| region.start_sample);
        let first = self
            .turns
            .partition_point(|turn| round_samples(turn.end) <= start);
        let regions = activity_regions(&self.turns[first..], start, observed);
        for region in &regions {
            for speaker in &region.speakers {
                if let Err(index) = row.speaker_candidates.binary_search(speaker) {
                    row.speaker_candidates.insert(index, speaker.clone());
                }
            }
        }
        activity.regions.extend(regions);
        activity.observed_end_sample = Some(observed);
        Ok(())
    }

    fn decode(&mut self, identity: &str) -> Result<(), LiveError> {
        let row = self.book.rows.get(identity).expect("row").clone();
        let stamp = self.language_at(row.start_sample);
        let length = row.end_sample - row.start_sample;
        let mut progress = self.last_decoded.remove(identity).unwrap_or_default();
        if length <= MAX_DECODE_SAMPLES {
            self.decode_short(&row, &stamp.language)?;
        } else {
            self.decode_cores(&row, &stamp.language, &mut progress)?;
        }
        let row = self.book.rows.get(identity).expect("row");
        progress.end_sample = row.end_sample;
        progress.audio_revision = row.audio_revision;
        self.last_decoded.insert(identity.to_string(), progress);
        Ok(())
    }

    fn decode_short(&mut self, row: &Utterance, language: &str) -> Result<(), LiveError> {
        let passages = self.transcribe_span(
            row.start_sample,
            row.end_sample,
            language,
            &row.speaker_candidates,
            (0, 0),
        )?;
        let routed = if passages.len() > 1 {
            routed_short_revision(&passages, row.start_sample, row.end_sample)
        } else {
            None
        };
        let length = row.end_sample - row.start_sample;
        let single = passages.len() == 1
            && passages[0].transcription_review.is_none()
            && round_samples(passages[0].start) == 0
            && round_samples(passages[0].end) == length;
        let complete = single || routed.is_some();
        let text = if let Some(routed) = &routed {
            routed.text.clone()
        } else if passages.len() == 1 {
            passages[0]
                .cohere_raw_text
                .clone()
                .unwrap_or_else(|| passages[0].text.clone())
        } else {
            String::new()
        };
        self.book
            .apply_model(&row.id, &text, DecodeStage::Live, complete)?;
        let stored = self.book.rows.get_mut(&row.id).expect("row");
        let mut decisions = Vec::new();
        crate::language::append_passage_decisions(&mut decisions, &passages, row.start_sample);
        if let Some(summary) = crate::language::summarize_decisions(decisions) {
            stored.language = summary.language;
            stored.language_detection = Some(summary.detection);
            stored.review = summary.review || passages.iter().any(|passage| passage.review);
        }
        stored.audio_state = (passages.len() == 1)
            .then(|| passages[0].audio_state)
            .flatten();
        stored.transcription_review = passages
            .iter()
            .any(|passage| passage.transcription_review.is_some());
        if !complete || text.trim().is_empty() {
            stored.canonical_unresolved = Some("incomplete_cohere_revision".to_string());
        } else {
            stored.canonical_unresolved = None;
        }
        Ok(())
    }

    fn decode_cores(
        &mut self,
        row: &Utterance,
        language: &str,
        progress: &mut DecodeProgress,
    ) -> Result<(), LiveError> {
        let start = progress
            .completed
            .as_ref()
            .map_or(row.start_sample, |part| part.end_sample);
        let edges = if row.end_sample - start <= MAX_DECODE_SAMPLES {
            vec![start, row.end_sample]
        } else {
            // Only the unaccepted suffix can move. Do not search or validate gaps
            // behind a completed boundary on subsequent capture notifications.
            let first = row
                .speech_regions
                .partition_point(|span| span.end_sample <= start);
            let core_row = CoreAnchor {
                id: row.id.clone(),
                start_sample: start,
                end_sample: row.end_sample,
                machine_revision: row.machine_revision,
                audio_revision: row.audio_revision,
                language_epoch: row.language_epoch,
                speech_regions: row.speech_regions[first..]
                    .iter()
                    .map(|span| SampleSpan {
                        start_sample: span.start_sample.max(start),
                        end_sample: span.end_sample,
                    })
                    .collect(),
            };
            let ledger = Arc::clone(&self.ledger);
            let plan = plan_cores(&core_row, |start, end| {
                ledger
                    .lock()
                    .expect("speech ledger")
                    .admission(start, end)
                    .unwrap_or_else(|_| pending_admission(start, end))
            });
            validate_core_plan(&plan, &core_row)?
        };
        let anchor = anchor_from(row);
        let mut text = progress
            .completed
            .as_ref()
            .map_or_else(String::new, |part| part.text.clone());
        let mut decisions = progress
            .completed
            .as_ref()
            .map_or_else(Vec::new, |part| part.decisions.clone());
        let mut review = progress.completed.as_ref().is_some_and(|part| part.review);
        let mut complete = true;
        for pair in edges.windows(2) {
            let request = request_for(&anchor, pair[0], pair[1], pair[0], pair[1]);
            let passages = self.transcribe_span(
                request.start_sample,
                request.end_sample,
                language,
                &row.speaker_candidates,
                (0, 0),
            )?;
            crate::language::append_passage_decisions(
                &mut decisions,
                &passages,
                request.start_sample,
            );
            review |= passages.iter().any(|passage| passage.review);
            let tail = tail_part(&request, &passages);
            if !tail.complete {
                // The first unowned core stays retryable. Never freeze an
                // incomplete prefix or publish later words across this hole.
                complete = false;
                break;
            }
            if !text.is_empty() {
                text.push(' ');
            }
            text.push_str(&tail.text);
            if request.end_sample < row.end_sample {
                progress.completed = Some(CompletedCores {
                    end_sample: request.end_sample,
                    text: text.clone(),
                    decisions: decisions.clone(),
                    review,
                });
            }
        }
        self.book
            .apply_model(&row.id, &text, DecodeStage::Live, complete)?;
        let stored = self.book.rows.get_mut(&row.id).expect("row");
        if let Some(summary) = crate::language::summarize_decisions(decisions) {
            stored.language = summary.language;
            stored.language_detection = Some(summary.detection);
            stored.review = review || summary.review;
        }
        stored.audio_state = None;
        stored.transcription_review = !complete;
        if complete && !text.trim().is_empty() {
            stored.canonical_unresolved = None;
        } else {
            stored.canonical_unresolved = Some("incomplete_cohere_revision".to_string());
        }
        Ok(())
    }

    fn transcribe_span(
        &self,
        start: i64,
        end: i64,
        language: &str,
        names: &[String],
        padding: (u32, u32),
    ) -> Result<Vec<Passage>, LiveError> {
        let pcm = self.read_pcm(start, end)?;
        let failure = Arc::new(Mutex::new(None));
        let detector = if language == languages::AUTOMATIC {
            Some(Box::new(ModelDetector {
                models: Arc::clone(&self.models),
                failure: failure.clone(),
            }) as Box<dyn LanguageDetector>)
        } else {
            None
        };
        let mut transcriber = SpeechTranscriber::new(
            language,
            Box::new(SharedLedger(Arc::clone(&self.ledger))),
            detector,
            Box::new(ModelRecognizer {
                models: Arc::clone(&self.models),
                failure: failure.clone(),
            }),
            None,
        )
        .map_err(|error| LiveError::Detail(error.to_string()))?;
        let result = transcriber
            .transcribe(
                &pcm,
                limits::SAMPLE_RATE,
                names,
                TranscribeOptions {
                    max_asr_seconds: 24.5,
                    start_sample: start,
                    asr_padding: padding,
                },
            )
            .map_err(|error| LiveError::Detail(error.to_string()));
        if let Some(error) = failure.lock().expect("model failure").take() {
            return Err(error.into());
        }
        result
    }

    fn publish(&mut self, row: Utterance) -> Result<(), LiveError> {
        let mut candidate = project_segment(&row, &self.language_at(row.start_sample));
        if self
            .config
            .recovery
            .as_ref()
            .is_some_and(|recovery| row.start_sample < recovery.through_sample)
        {
            // Words recovered from the interrupted prefix may overlap a preserved edited
            // display boundary. Keep both authoritative edits and this uncertain tail visible.
            candidate.review = Some(true);
            candidate.rest.insert(
                "canonical_unresolved".into(),
                Value::from("inference_restart_boundary"),
            );
            candidate.refinement_state = Some("unresolved".into());
        }
        candidate.rest.insert(
            "stream_generation".into(),
            Value::from(self.config.recovery.as_ref().map_or(0, |r| r.generation)),
        );
        if let (Some(source), Some(catalog)) =
            (&self.config.source_id, &self.config.capture_catalog_sha256)
        {
            candidate.capture_source = Some(
                serde_json::json!({"catalog_sha256":catalog,"source_id":source,"source_revision":1}),
            );
        }
        if row.state == UtteranceState::Sealed {
            let admission = self
                .ledger
                .lock()
                .expect("speech ledger")
                .admission(row.start_sample, row.end_sample)?;
            candidate.acoustic_evidence = Some(speech::admission_value(&admission));
            let admitted = admission.complete
                && admission.uncertain_regions.is_empty()
                && !admission.evidence_error;
            if !admitted {
                candidate
                    .rest
                    .insert("audio_state".into(), Value::from("unverified_audio"));
            }
            if !row.text.trim().is_empty()
                && row.audio_state.is_none()
                && row.protected_fields.is_empty()
                && admitted
                && let Some(vocabulary) = &self.word_alignment
                && let Some(language) = candidate
                    .language
                    .as_deref()
                    .filter(|l| languages::is_supported(l))
            {
                let length = row.end_sample - row.start_sample;
                if length <= crate::alignment::MAX_SAMPLES as i64 {
                    let pcm = self
                        .recording
                        .read_samples(
                            row.start_sample as u32,
                            row.end_sample as u32,
                            self.committed,
                        )
                        .map_err(|e| LiveError::Detail(e.to_string()))?;
                    match vocabulary.align(
                        self.models.as_ref(),
                        &pcm,
                        &candidate.text,
                        row.start_sample,
                        language,
                    ) {
                        Ok(aligned) => crate::alignment::attach(&mut candidate, aligned),
                        Err(
                            error @ (ChildError::ProcessLost { .. }
                            | ChildError::Timeout
                            | ChildError::Cancelled),
                        ) => return Err(error.into()),
                        Err(_) => {
                            candidate.rest.insert(
                                "reading_alignment".into(),
                                serde_json::json!({"status":"unavailable","reason":"model_error"}),
                            );
                        }
                    }
                } else {
                    candidate.rest.insert(
                        "reading_alignment".into(),
                        serde_json::json!({"status":"unavailable","reason":"crop_too_long"}),
                    );
                }
            }
        }
        if self.published.get(&row.id) == Some(&candidate) {
            return Ok(());
        }
        self.published.insert(row.id.clone(), candidate.clone());
        self.fast_sequence += 1;
        self.events.push(EngineEvent::CanonicalRevision {
            candidate: Box::new(candidate),
            fast_sequence: self.fast_sequence,
        });
        Ok(())
    }
}

fn anchor_from(row: &Utterance) -> AnchorRow {
    AnchorRow {
        id: row.id.clone(),
        start_sample: row.start_sample,
        end_sample: row.end_sample,
        machine_revision: row.machine_revision,
        audio_revision: row.audio_revision,
        language_epoch: row.language_epoch,
    }
}

fn tail_part(request: &DecodeRequest, passages: &[Passage]) -> TailPart {
    let span = request.end_sample - request.start_sample;
    let mut cursor = 0i64;
    let mut complete = !passages.is_empty();
    let mut texts = Vec::new();
    for passage in passages {
        let begin = round_samples(passage.start);
        let end = round_samples(passage.end);
        let text = passage
            .cohere_raw_text
            .clone()
            .unwrap_or_else(|| passage.text.clone());
        let valid = begin == cursor
            && begin < end
            && end <= span
            && passage.transcription_review.is_none()
            && !text.trim().is_empty();
        complete = complete && valid;
        cursor = end;
        texts.push(text);
    }
    complete = complete && cursor == span;
    TailPart {
        request: request.clone(),
        text: texts.join(" "),
        complete,
        alignment: None,
    }
}

fn clip_retained_activity(row: &mut Utterance, activity: Option<&timing::SpeakerActivity>) {
    row.speaker_candidates.clear();
    row.speaker_activity = activity.map(|activity| {
        let regions: Vec<_> = activity
            .regions
            .iter()
            .filter_map(|region| {
                let start = region.start_sample.max(row.start_sample);
                let end = region.end_sample.min(row.end_sample);
                (start < end).then(|| timing::ActivityRegion {
                    start_sample: start,
                    end_sample: end,
                    speakers: region.speakers.clone(),
                })
            })
            .collect();
        for region in &regions {
            for owner in &region.speakers {
                if !row.speaker_candidates.contains(owner) {
                    row.speaker_candidates.push(owner.clone());
                }
            }
        }
        row.speaker_candidates.sort();
        timing::SpeakerActivity {
            audio_revision: row.audio_revision,
            observed_end_sample: activity
                .observed_end_sample
                .map(|end| end.clamp(row.start_sample, row.end_sample)),
            regions,
        }
    });
}

fn project_segment(row: &Utterance, stamp: &LanguageStamp) -> Segment {
    let names = &row.speaker_candidates;
    let single = names.len() == 1 && single_speaker(&names[0]);
    let speaker = if single {
        names[0].clone()
    } else if names.len() > 1 {
        "multiple_speakers".to_string()
    } else {
        "unassigned".to_string()
    };
    let mut reading_turns = None;
    if let (Some(evidence), Some(activity)) = (
        row.text_audio_anchor.as_ref(),
        row.speaker_activity.as_ref(),
    ) {
        let turns = timing::project_turns(&ReadingRow {
            text: row.text.clone(),
            start_sample: row.start_sample,
            end_sample: row.end_sample,
            machine_revision: row.machine_revision,
            audio_revision: row.audio_revision,
            protected_fields: row.protected_fields.clone(),
            text_audio_anchor: Some(evidence.clone()),
            reading_word_evidence: None,
            speaker_activity: Some(activity.clone()),
            installed_provider_identity: None,
        });
        if !turns.is_empty() {
            reading_turns = Some(turns_value(&turns));
        }
    }
    let mut rest = Map::new();
    if let Some(anchor) = &row.text_audio_anchor {
        rest.insert(
            "text_audio_anchor".into(),
            serde_json::json!({"start_sample":anchor.start_sample,"end_sample":anchor.end_sample}),
        );
    }
    if let Some(end) = row.text_end_sample {
        // Unlike current audio evidence, this survives audio growth and edited display
        // bounds. It is not alignment or a claim about edited words' timing.
        rest.insert("canonical_text_end_sample".into(), Value::from(end));
    }
    rest.insert(
        "canonical_state".to_string(),
        Value::String(if row.state == UtteranceState::Sealed {
            "sealed".to_string()
        } else {
            "open".to_string()
        }),
    );
    rest.insert("speaker_candidates".into(), serde_json::json!(names));
    rest.insert("source_speaker_candidates".into(), serde_json::json!(names));
    // Voice crops use the observed region ledger, including partial/unknown tails.
    rest.insert("voice_eligible".into(), Value::Bool(false));
    if let Some(state) = &row.audio_state {
        rest.insert("audio_state".into(), serde_json::json!(state.as_str()));
    }
    rest.insert("start_sample".to_string(), Value::from(row.start_sample));
    rest.insert("end_sample".to_string(), Value::from(row.end_sample));
    rest.insert(
        "language_epoch".to_string(),
        Value::from(row.language_epoch),
    );
    rest.insert(
        "language_mode".to_string(),
        Value::String(stamp.language.clone()),
    );
    rest.insert(
        "canonical_machine_revision".to_string(),
        Value::from(row.machine_revision),
    );
    rest.insert(
        "timing".to_string(),
        Value::String("canonical_vad_audio_anchor".to_string()),
    );
    rest.insert(
        "finalized".to_string(),
        Value::Bool(row.state == UtteranceState::Sealed),
    );
    if let Some(reason) = &row.canonical_unresolved {
        rest.insert(
            "canonical_unresolved".to_string(),
            Value::String(reason.clone()),
        );
    }
    let refinement_state = if row.canonical_unresolved.is_some() {
        "unresolved".to_string()
    } else {
        row.refinement_state.clone()
    };
    Segment {
        id: row.id.clone(),
        start: row.start_sample as f64 / RATE as f64,
        end: row.end_sample as f64 / RATE as f64,
        speaker,
        text: row.text.clone(),
        language: if stamp.language != languages::AUTOMATIC {
            Some(stamp.language.clone())
        } else {
            row.language.clone()
        },
        language_detection: row.language_detection.as_ref().map(crate::import::detection_value),
        transcription_review: row.canonical_unresolved.as_ref().map(|reason| {
            serde_json::json!({"reason": reason, "partial_text": !row.text.trim().is_empty()})
        }),
        reading_turns,
        reading_turn_provenance: None,
        alignment: None,
        speaker_activity: row.speaker_activity.as_ref().map(|activity| {
            serde_json::json!({
                "audio_revision": activity.audio_revision,
                "observed_end_sample": activity.observed_end_sample,
                "regions": activity.regions.iter().map(|r| serde_json::json!({"start_sample":r.start_sample,"end_sample":r.end_sample,"speakers":r.speakers})).collect::<Vec<_>>(),
            })
        }),
        acoustic_evidence: None,
        refinement_state: Some(refinement_state),
        machine_revision: Some(row.machine_revision),
        protected_fields: (!row.protected_fields.is_empty()).then(|| row.protected_fields.clone()),
        review_resolution: None,
        retry_candidate: None,
        canonical_utterance_id: Some(row.id.clone()),
        audio_revision: Some(row.audio_revision),
        review: Some(!single || row.review || row.canonical_unresolved.is_some() || row.transcription_review),
        capture_source: None,
        rest,
    }
}

fn turns_value(turns: &[ReadingTurn]) -> Value {
    Value::Array(
        turns
            .iter()
            .map(|turn| {
                serde_json::json!({
                    "text": turn.text,
                    "speaker": turn.speaker,
                    "attribution": turn.attribution,
                })
            })
            .collect(),
    )
}

fn pending_admission(start: i64, end: i64) -> Admission {
    Admission {
        source: speech::SILERO_SOURCE.to_string(),
        model_revision: speech::SILERO_REVISION.to_string(),
        start_sample: start,
        end_sample: end,
        observed_until_sample: end,
        complete: false,
        decision: SpeechDecision::Pending,
        speech_regions: Vec::new(),
        input_policy: None,
        conditioning_sha256: None,
        maximum_probability: None,
        maximum_model_probability: None,
        uncertain_regions: Vec::new(),
        model_negative_regions: Vec::new(),
        negative_signal_state: None,
        classification_scope: None,
        evidence_error: false,
    }
}

#[cfg(test)]
fn empty_document() -> Document {
    Document {
        schema_version: 1,
        speakers: BTreeMap::new(),
        segments: Vec::new(),
        provenance: None,
        warnings: Vec::new(),
        diarization: Vec::new(),
        edited_at: None,
        rest: Map::new(),
    }
}

/// Two capture sources on one sample clock. Speaker ids are source-prefixed.
pub struct SourceRuntime<M: Models> {
    inbox: Arc<Mutex<Inbox>>,
    engines: Vec<Engine<M>>,
    fast_sequence: i64,
    received: i64,
}

impl<M: Models + 'static> SourceRuntime<M> {
    pub fn inbox(&self) -> Arc<Mutex<Inbox>> {
        Arc::clone(&self.inbox)
    }
    pub fn new(
        config: EngineConfig,
        models: M,
        recordings: [RecordingReader; 2],
    ) -> Result<Self, LiveError> {
        let models = Arc::new(models);
        let inbox = Arc::new(Mutex::new(Inbox::new(config.audio_batch_seconds)?));
        if let Some(recovery) = &config.recovery {
            recovery.validate()?;
            inbox.lock().expect("inbox").received = recovery.through_sample;
        }
        let mut engines = Vec::new();
        for (source_id, recording) in SOURCE_IDS.into_iter().zip(recordings) {
            let mut source_config = config.clone();
            source_config.source_id = Some(source_id.to_string());
            if source_id != "microphone_clean" {
                source_config.processed_microphone_contract = None;
            }
            engines.push(Engine::new(
                source_config,
                Arc::clone(&models),
                Arc::clone(&inbox),
                recording,
            )?);
        }
        Ok(Self {
            fast_sequence: config.fast_sequence,
            inbox,
            engines,
            received: 0,
        })
    }

    pub fn set_word_alignment(&mut self, vocabulary: Arc<crate::alignment::Vocabulary>) {
        for engine in &mut self.engines {
            engine.set_word_alignment(vocabulary.clone());
        }
    }

    pub fn continue_edited(
        &mut self,
        boundary: &EditedBoundary,
    ) -> Result<Vec<EngineEvent>, LiveError> {
        let mut events = Vec::new();
        for engine in &mut self.engines {
            for mut event in engine.continue_edited(boundary)? {
                if let EngineEvent::CanonicalRevision { fast_sequence, .. } = &mut event {
                    self.fast_sequence += 1;
                    *fast_sequence = self.fast_sequence;
                }
                events.push(event);
            }
        }
        Ok(events)
    }

    pub fn push(&self, command: LiveCommand) -> Result<(), LiveError> {
        self.inbox.lock().expect("inbox").push(command)
    }

    pub fn poll(&mut self) -> Result<Option<Vec<EngineEvent>>, LiveError> {
        let message = self.inbox.lock().expect("inbox").take()?;
        message.map(|command| self.handle(command)).transpose()
    }

    pub fn handle(&mut self, message: LiveCommand) -> Result<Vec<EngineEvent>, LiveError> {
        if let LiveCommand::Audio(ref notice) = message {
            if notice.start_sample != self.received {
                return Err(LiveError::Message(
                    "Capture sources require contiguous ordered audio.",
                ));
            }
            let endpoint = u32::try_from(notice.end_sample)
                .map_err(|_| LiveError::Message("Invalid saved audio endpoint."))?;
            for engine in &self.engines {
                engine
                    .recording
                    .require_samples(endpoint)
                    .map_err(|error| LiveError::Detail(error.to_string()))?;
            }
        }
        let mut pending: BTreeMap<&'static str, Vec<EngineEvent>> = BTreeMap::new();
        let mut published = Vec::new();
        for engine in &mut self.engines {
            for event in engine.handle(message.clone())? {
                match event {
                    EngineEvent::CanonicalRevision { .. } => {
                        published.push(event);
                    }
                    EngineEvent::Progress { .. } => {
                        pending.entry("progress").or_default().push(event)
                    }
                    EngineEvent::FlushAck { .. } => pending.entry("flush").or_default().push(event),
                    EngineEvent::CaptureFinished { .. } => {
                        pending.entry("stop").or_default().push(event)
                    }
                    EngineEvent::LanguageRegistered { .. } => {
                        pending.entry("language").or_default().push(event);
                    }
                    EngineEvent::Finished { .. } => {
                        pending.entry("finished").or_default().push(event)
                    }
                }
            }
        }
        let endpoints: Vec<i64> = self.engines.iter().map(|engine| engine.received).collect();
        if endpoints.iter().any(|endpoint| *endpoint != endpoints[0]) {
            return Err(LiveError::Message("Capture source clocks diverged."));
        }
        self.received = endpoints[0];
        let mut events = Vec::new();
        for mut event in published {
            if let EngineEvent::CanonicalRevision { fast_sequence, .. } = &mut event {
                self.fast_sequence += 1;
                *fast_sequence = self.fast_sequence;
            }
            events.push(event);
        }
        if let Some(progress) = pending.get("progress") {
            let processed = progress
                .iter()
                .filter_map(|event| match event {
                    EngineEvent::Progress {
                        processed_seconds, ..
                    } => Some(*processed_seconds),
                    _ => None,
                })
                .fold(f64::INFINITY, f64::min);
            events.push(EngineEvent::Progress {
                processed_seconds: processed,
                received_seconds: self.received as f64 / RATE as f64,
                flush: false,
            });
        }
        match message {
            LiveCommand::Flush {
                request_id: Some(request_id),
                ..
            } => {
                let acks = pending.get("flush").ok_or(LiveError::Message(
                    "Pause flush must follow its exact captured audio endpoint.",
                ))?;
                let available = acks
                    .iter()
                    .filter_map(|event| match event {
                        EngineEvent::FlushAck {
                            available_sample, ..
                        } => Some(*available_sample),
                        _ => None,
                    })
                    .min()
                    .unwrap_or(self.received);
                let speech = acks
                    .iter()
                    .filter_map(|event| match event {
                        EngineEvent::FlushAck {
                            speech_observed_sample,
                            ..
                        } => Some(*speech_observed_sample),
                        _ => None,
                    })
                    .min()
                    .unwrap_or(self.received);
                events.push(EngineEvent::FlushAck {
                    request_id,
                    received_sample: self.received,
                    available_sample: available,
                    speech_observed_sample: speech,
                    fast_sequence: self.fast_sequence,
                });
            }
            LiveCommand::Stop => {
                let finished = pending.get("stop").cloned().unwrap_or_default();
                let observed = finished
                    .iter()
                    .filter_map(|event| match event {
                        EngineEvent::CaptureFinished {
                            canonical_observed_sample,
                            ..
                        } => Some(*canonical_observed_sample),
                        _ => None,
                    })
                    .min()
                    .unwrap_or(self.received);
                let uncertain = finished
                    .iter()
                    .filter_map(|event| match event {
                        EngineEvent::CaptureFinished {
                            canonical_uncertain_samples,
                            ..
                        } => Some(*canonical_uncertain_samples),
                        _ => None,
                    })
                    .max()
                    .unwrap_or(0);
                events.push(EngineEvent::CaptureFinished {
                    duration: self.received as f64 / RATE as f64,
                    canonical_observed_sample: observed,
                    canonical_uncertain_samples: uncertain,
                });
            }
            LiveCommand::Language { .. } => {
                let registered = pending.get("language").cloned().unwrap_or_default();
                if registered.len() == SOURCE_IDS.len()
                    && registered.windows(2).all(|pair| pair[0] == pair[1])
                {
                    if let Some(event) = registered.into_iter().next() {
                        events.push(event);
                    }
                } else if !registered.is_empty() {
                    return Err(LiveError::Message(
                        "Capture source language clocks diverged.",
                    ));
                }
            }
            LiveCommand::Shutdown => events.push(EngineEvent::Finished {
                duration: self.received as f64 / RATE as f64,
            }),
            _ => {}
        }
        Ok(events)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::edit::{PassageEditInput, edit_passage};
    use crate::ipc::{
        AsrText, CtcScoresOut, DiarFeedOut, GainPolicy, LanguageProbeOut, MetricsOut,
        SetupModelReport, VadFrame,
    };
    use crate::language::{DetectionMode, DetectionReason, LanguageDetection};
    use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};

    thread_local! {
        static ACTIVITY_SWEEPS: std::cell::RefCell<Option<Vec<(i64, i64, usize)>>> = const { std::cell::RefCell::new(None) };
    }
    pub(super) fn record_activity_sweep(start: i64, end: i64, turns: usize) {
        ACTIVITY_SWEEPS.with_borrow_mut(|calls| {
            if let Some(calls) = calls {
                calls.push((start, end, turns));
            }
        });
    }

    struct TestRecording {
        path: std::path::PathBuf,
        writer: crate::audio::RecordingWriter,
    }

    impl TestRecording {
        fn new() -> Self {
            let path =
                std::env::temp_dir().join(format!("speakerdesk-live-{}.wav", uuid::Uuid::new_v4()));
            let writer = crate::audio::RecordingWriter::create(&path).unwrap();
            Self { path, writer }
        }
    }

    impl Drop for TestRecording {
        fn drop(&mut self) {
            std::fs::remove_file(&self.path).unwrap();
        }
    }

    struct TestEngine {
        engine: Engine<FakeModels>,
        recording: TestRecording,
    }

    impl TestEngine {
        fn new(config: EngineConfig, models: FakeModels) -> Result<Self, LiveError> {
            let recording = TestRecording::new();
            let engine = Engine::single(config, models, &recording.path)?;
            Ok(Self { engine, recording })
        }
        fn append_pcm(&mut self, pcm: &[f32]) {
            self.recording.writer.write_samples(pcm).unwrap();
        }
    }

    impl std::ops::Deref for TestEngine {
        type Target = Engine<FakeModels>;
        fn deref(&self) -> &Self::Target {
            &self.engine
        }
    }
    impl std::ops::DerefMut for TestEngine {
        fn deref_mut(&mut self) -> &mut Self::Target {
            &mut self.engine
        }
    }

    fn notice(start: i64, end: i64, epoch: i64) -> LiveCommand {
        LiveCommand::Audio(AudioNotice {
            start_sample: start,
            end_sample: end,
            language_epoch: epoch,
            language: "en".to_string(),
        })
    }

    #[test]
    fn inbox_coalesces_same_epoch_audio_and_keeps_epoch_boundaries() {
        let mut inbox = Inbox::new(6).unwrap();
        inbox.push(notice(0, RATE, 0)).unwrap();
        inbox.push(notice(RATE, 2 * RATE, 0)).unwrap();
        assert_eq!(inbox.queued_audio_span(), Some((0, 2 * RATE, 0)));
        inbox.push(notice(2 * RATE, 3 * RATE, 1)).unwrap();
        assert_eq!(inbox.take().unwrap().map(|command| matches!(command, LiveCommand::Audio(ref audio) if audio.end_sample == 2 * RATE)), Some(true));
        assert_eq!(inbox.queued_audio_span(), Some((2 * RATE, 3 * RATE, 1)));
    }

    fn speech_passage(start: f64, end: f64, text: &str, abs_start: i64) -> Passage {
        let first = round_samples(start);
        let last = round_samples(end);
        Passage {
            start,
            end,
            text: text.trim().to_string(),
            cohere_raw_text: Some(text.to_string()),
            language: Some("en".into()),
            language_detection: LanguageDetection {
                mode: DetectionMode::Auto,
                reason: DetectionReason::Detected,
                probability: Some(0.99),
                margin: Some(0.4),
                candidates: Vec::new(),
                detector_error: false,
                context_end_sample: None,
                probes: Vec::new(),
            },
            review: false,
            language_review: false,
            audio_state: None,
            acoustic_evidence: Admission {
                source: speech::SILERO_SOURCE.into(),
                model_revision: speech::SILERO_REVISION.into(),
                start_sample: abs_start + first,
                end_sample: abs_start + last,
                observed_until_sample: abs_start + last,
                complete: true,
                decision: SpeechDecision::Speech,
                speech_regions: vec![SampleSpan {
                    start_sample: abs_start + first,
                    end_sample: abs_start + last,
                }],
                input_policy: Some(speech::INPUT_POLICY.into()),
                conditioning_sha256: None,
                maximum_probability: Some(0.9),
                maximum_model_probability: Some(0.9),
                uncertain_regions: Vec::new(),
                model_negative_regions: Vec::new(),
                negative_signal_state: None,
                classification_scope: None,
                evidence_error: false,
            },
            pcm_evidence: None,
            transcription_review: None,
            cohere_input_padding: None,
        }
    }

    #[test]
    fn routed_short_revision_joins_contiguous_english_and_rejects_a_gap() {
        let passages = vec![
            speech_passage(0.0, 0.5, "left", 1_000),
            speech_passage(0.5, 1.0, "right", 1_000),
        ];
        let routed = routed_short_revision(&passages, 1_000, 1_000 + RATE).unwrap();
        assert_eq!(routed.text, "left right");
        assert_eq!(routed.parts.len(), 2);
        let mut gapped = passages.clone();
        gapped[1].start = 0.75;
        assert!(routed_short_revision(&gapped, 1_000, 1_000 + RATE).is_none());
    }

    fn anchor(length: i64) -> AnchorRow {
        AnchorRow {
            id: "utt".into(),
            start_sample: 0,
            end_sample: length,
            machine_revision: 1,
            audio_revision: 2,
            language_epoch: 0,
        }
    }

    #[test]
    fn context_and_authoritative_requests_follow_the_frozen_edges() {
        let row = anchor(28 * RATE);
        let context = context_requests(&row, None).unwrap();
        assert_eq!(CONTEXT_POLICY, "english_bounded14_context3_corroborated_v1");
        assert_eq!(context.len(), 2);
        assert_eq!(context[0].core_end_sample, 14 * RATE);
        assert_eq!(context[0].end_sample, 17 * RATE);
        assert_eq!(context[1].start_sample, 11 * RATE);
        assert_eq!(context[1].end_sample, 28 * RATE);

        let long = anchor(21 * RATE);
        let tail = authoritative_requests(&long).unwrap();
        assert_eq!(tail.len(), 2);
        assert_eq!(tail[0].core_end_sample, AUTHORITATIVE_CORE);
        assert_eq!(tail[0].end_sample, AUTHORITATIVE_CORE + AUTHORITATIVE_PAD);
        assert_eq!(tail[1].core_start_sample, AUTHORITATIVE_CORE);
        assert_eq!(tail[1].start_sample, AUTHORITATIVE_CORE - AUTHORITATIVE_PAD);
        assert_eq!(tail[1].end_sample, 21 * RATE);
    }

    #[test]
    fn authoritative_whole_tail_keeps_raw_text_and_a_missing_seam_stays_in_review() {
        let row = anchor(RATE);
        let requests = authoritative_requests(&row).unwrap();
        let parts = vec![TailPart {
            request: requests[0].clone(),
            text: "hello there".into(),
            complete: true,
            alignment: None,
        }];
        let whole = assemble_authoritative(&row, &requests, &parts).unwrap();
        assert!(whole.complete);
        assert_eq!(whole.text.as_deref(), Some("hello there"));
        assert_eq!(whole.policy, AUTHORITATIVE_POLICY);

        let long = anchor(21 * RATE);
        let requests = authoritative_requests(&long).unwrap();
        let parts: Vec<TailPart> = requests
            .iter()
            .enumerate()
            .map(|(index, request)| TailPart {
                request: request.clone(),
                text: format!("part-{index}"),
                complete: true,
                alignment: None,
            })
            .collect();
        let seam = assemble_authoritative(&long, &requests, &parts).unwrap();
        assert!(!seam.complete);
        assert!(seam.recognition_complete);
        assert_eq!(seam.reason, Some("unavailable_boundary_timing"));
        assert!(seam.text.is_none());
    }

    // Supplied CTC-cell overlap-ownership replay, not acoustic quality evidence.
    fn overlap_part(request: &DecodeRequest, text: &str, onsets: &[i64]) -> TailPart {
        let mut words = Vec::new();
        let mut characters = Vec::new();
        let raw: Vec<_> = text.chars().collect();
        let units = timing::raw_units(text, crate::timing_policy::TIMING_UNIT_POLICY_ID).unwrap();
        assert_eq!(units.len(), onsets.len());
        for (unit_index, (unit, onset)) in units.iter().zip(onsets).enumerate() {
            let mut offset = unit.start_char;
            for (index, character) in unit.text.chars().enumerate() {
                let sample = onset + index as i64 * 320;
                characters.push(serde_json::json!({"text":character.to_string(),"raw_text":character.to_string(),
                    "token_id":character as u32,"start_char":offset,"end_char":offset+1,
                    "frame":(sample-request.start_sample)/320,"start_sample":sample,"end_sample":sample+320,
                    "log_probability":-0.01,"status":"aligned"}));
                offset += 1;
            }
            words.push(
                serde_json::json!({"text":unit.text,"start_char":unit.start_char,"end_char":offset,
                "start_sample":onset,"end_sample":onset+unit.text.chars().count() as i64*320,"status":"aligned"}),
            );
            let next = units
                .get(unit_index + 1)
                .map_or(raw.len(), |u| u.start_char);
            while offset < next {
                assert!(raw[offset].is_whitespace());
                let sample = onset + (offset - unit.start_char) as i64 * 320;
                characters.push(serde_json::json!({"text":" ","raw_text":raw[offset].to_string(),"token_id":32,
                    "start_char":offset,"end_char":offset+1,"frame":(sample-request.start_sample)/320,
                    "start_sample":sample,"end_sample":sample+320,"log_probability":-0.01,"status":"aligned"}));
                offset += 1;
            }
        }
        TailPart {
            request: request.clone(),
            text: text.into(),
            complete: true,
            alignment: Some(serde_json::json!({
                "raw_text":text,"text_sha256":sha256_hex(text.as_bytes()),
                "audio_anchor":{"start_sample":request.start_sample,"end_sample":request.end_sample},
                "provider_identity_sha256":crate::alignment::NATIVE_PROVIDER,"model_sha256":timing::MODEL,
                "language":"en","provider_id":crate::timing_policy::PROVIDER_ID,
                "normalization_policy_id":crate::timing_policy::NORMALIZATION_POLICY_ID,
                "timing_unit_policy_id":crate::timing_policy::TIMING_UNIT_POLICY_ID,
                "frame_calibration_id":timing::CALIBRATION,"score_calibration_id":timing::CALIBRATION,
                "status":"aligned","complete":true,"characters":characters,"words":words
            })),
        }
    }

    #[test]
    fn authoritative_context_accepts_functional_non_english_envelopes() {
        let mut row = anchor(26 * RATE + 513);
        row.start_sample = 7 * RATE + 137;
        row.end_sample += row.start_sample;
        let requests = authoritative_requests(&row).unwrap();
        let mut parts = vec![
            overlap_part(
                &requests[0],
                "Anfang wieder Brücke Kontext",
                &[
                    8 * RATE + 137,
                    20 * RATE + 137,
                    21 * RATE + 137,
                    23 * RATE + 137,
                ],
            ),
            overlap_part(
                &requests[1],
                "wieder Brücke Ende wieder",
                &[
                    20 * RATE + 137,
                    21 * RATE + 137,
                    27 * RATE + 137,
                    32 * RATE + 137,
                ],
            ),
        ];
        for part in &mut parts {
            let alignment = part.alignment.as_mut().unwrap();
            alignment["language"] = serde_json::json!("de");
            alignment["frame_calibration_id"] =
                serde_json::json!(crate::timing_policy::FUNCTIONAL_POLICY_ID);
            alignment["score_calibration_id"] =
                serde_json::json!(crate::timing_policy::FUNCTIONAL_POLICY_ID);
        }
        let assembled = assemble_authoritative(&row, &requests, &parts).unwrap();
        assert!(
            assembled.complete,
            "functional multilingual CTC must support the same bounded seam: {:?}",
            assembled.reason
        );
        assert_eq!(
            assembled.text.as_deref(),
            Some("Anfang wieder Brücke Ende wieder")
        );
        let mut spaced = parts.clone();
        spaced[1] = overlap_part(
            &requests[1],
            "wieder Brücke\u{a0}\n  Ende wieder",
            &[
                20 * RATE + 137,
                21 * RATE + 137,
                27 * RATE + 137,
                32 * RATE + 137,
            ],
        );
        for key in ["language", "frame_calibration_id", "score_calibration_id"] {
            spaced[1].alignment.as_mut().unwrap()[key] =
                parts[1].alignment.as_ref().unwrap()[key].clone();
        }
        let selected = assemble_authoritative(&row, &requests, &spaced).unwrap();
        assert!(selected.complete, "{:?}", selected.reason);
        assert_eq!(
            selected.text.as_deref(),
            Some("Anfang wieder Brücke\u{a0}\n  Ende wieder")
        );
        for (field, wrong) in [
            ("provider_identity_sha256", "wrong-provider"),
            ("frame_calibration_id", timing::CALIBRATION),
            ("score_calibration_id", timing::CALIBRATION),
            ("normalization_policy_id", "wrong-normalization"),
            ("timing_unit_policy_id", "wrong-units"),
            ("language", "fr"),
        ] {
            let mut changed = parts.clone();
            changed[1].alignment.as_mut().unwrap()[field] = Value::from(wrong);
            let rejected = assemble_authoritative(&row, &requests, &changed).unwrap();
            assert!(!rejected.complete && rejected.text.is_none(), "{field}");
        }

        let mut cjk = vec![
            overlap_part(
                &requests[0],
                "「𠮷本橋」、外",
                &[
                    8 * RATE + 137,
                    20 * RATE + 137,
                    21 * RATE + 137,
                    23 * RATE + 137,
                ],
            ),
            overlap_part(
                &requests[1],
                "本橋」、明日",
                &[
                    20 * RATE + 137,
                    21 * RATE + 137,
                    27 * RATE + 137,
                    32 * RATE + 137,
                ],
            ),
        ];
        for part in &mut cjk {
            let alignment = part.alignment.as_mut().unwrap();
            alignment["language"] = Value::from("ja");
            alignment["frame_calibration_id"] =
                Value::from(crate::timing_policy::FUNCTIONAL_POLICY_ID);
            alignment["score_calibration_id"] =
                Value::from(crate::timing_policy::FUNCTIONAL_POLICY_ID);
        }
        let selected = assemble_authoritative(&row, &requests, &cjk).unwrap();
        assert!(selected.complete, "{:?}", selected.reason);
        assert_eq!(selected.text.as_deref(), Some("「𠮷本橋」、明日"));
        let units = timing::raw_units(
            selected.text.as_deref().unwrap(),
            crate::timing_policy::TIMING_UNIT_POLICY_ID,
        )
        .unwrap();
        assert_eq!(selected.words.len(), units.len());
        for (word, unit) in selected.words.iter().zip(units) {
            assert_eq!(
                (&word.text, word.start_char, word.end_char),
                (&unit.text, unit.start_char, unit.end_char)
            );
        }
    }

    #[test]
    fn authoritative_context_cells_keep_raw_overlap_once_and_distinct_repetitions() {
        let mut row = anchor(26 * RATE + 513);
        row.start_sample = 7 * RATE + 137;
        row.end_sample += row.start_sample;
        let requests = authoritative_requests(&row).unwrap();
        assert_eq!(requests.len(), 2);
        let parts = vec![
            overlap_part(
                &requests[0],
                "Café, repeat bridge context",
                &[
                    8 * RATE + 137,
                    20 * RATE + 137,
                    21 * RATE + 137,
                    23 * RATE + 137,
                ],
            ),
            overlap_part(
                &requests[1],
                "repeat bridge finish repeat",
                &[
                    20 * RATE + 137,
                    21 * RATE + 137,
                    27 * RATE + 137,
                    32 * RATE + 137,
                ],
            ),
        ];
        let assembled = assemble_authoritative(&row, &requests, &parts).unwrap();
        assert!(assembled.complete, "{:?}", assembled.reason);
        assert_eq!(
            assembled.text.as_deref(),
            Some("Café, repeat bridge finish repeat")
        );
        assert_eq!(
            assembled
                .words
                .iter()
                .filter(|w| w.text == "repeat")
                .count(),
            2,
            "distinct repeated acoustic events must both remain"
        );
        assert_eq!(
            assembled
                .words
                .iter()
                .filter(|w| w.text == "bridge")
                .count(),
            1
        );
        assert!(
            assembled
                .words
                .iter()
                .all(|w| w.start_sample.is_none_or(|start| start >= row.start_sample))
        );
        for case in [
            "missing",
            "stale_anchor",
            "shifted",
            "unmapped",
            "incomplete",
        ] {
            let mut changed = parts.clone();
            match case {
                "missing" => changed[1].alignment = None,
                "stale_anchor" => {
                    changed[1].alignment.as_mut().unwrap()["audio_anchor"]["start_sample"] =
                        Value::from(0)
                }
                "shifted" | "unmapped" => {
                    let delta = if case == "unmapped" { -RATE } else { RATE };
                    let alignment = changed[1].alignment.as_mut().unwrap();
                    for group in ["characters", "words"] {
                        for cell in alignment[group].as_array_mut().unwrap() {
                            cell["start_sample"] =
                                Value::from(cell["start_sample"].as_i64().unwrap() + delta);
                            cell["end_sample"] =
                                Value::from(cell["end_sample"].as_i64().unwrap() + delta);
                            if case == "unmapped" && group == "characters" {
                                cell["frame"] =
                                    Value::from(cell["frame"].as_i64().unwrap() + delta / 320);
                            }
                        }
                    }
                }
                "incomplete" => changed[1].complete = false,
                _ => unreachable!(),
            }
            let rejected = assemble_authoritative(&row, &requests, &changed).unwrap();
            assert!(!rejected.complete, "{case}");
            if case == "unmapped" {
                assert_eq!(rejected.reason, Some("boundary_mapping_absent"));
            }
            assert!(
                rejected.text.is_none(),
                "{case}: never fall back to concatenated overlap"
            );
        }
    }

    #[test]
    fn authoritative_context_rejects_valid_but_ambiguous_acoustic_events() {
        let row = anchor(26 * RATE);
        let requests = authoritative_requests(&row).unwrap();
        let mut left = overlap_part(&requests[0], "go", &[13 * RATE]);
        // One stretched source event maps onto two short repeated events. Every
        // cell remains on its own physical crop's grid and has a valid envelope.
        let alignment = left.alignment.as_mut().unwrap();
        alignment["characters"][1]["frame"] = Value::from((13 * RATE + 3200) / 320);
        alignment["characters"][1]["start_sample"] = Value::from(13 * RATE + 3200);
        alignment["characters"][1]["end_sample"] = Value::from(13 * RATE + 3520);
        alignment["words"][0]["end_sample"] = Value::from(13 * RATE + 3520);
        let right = overlap_part(&requests[1], "go go", &[13 * RATE, 13 * RATE + 2880]);
        let result = assemble_authoritative(&row, &requests, &[left, right]).unwrap();
        assert_eq!(result.reason, Some("ambiguous_boundary_mapping"));
        assert!(result.text.is_none());
    }

    fn no_speech(start: i64, end: i64) -> Admission {
        Admission {
            source: speech::SILERO_SOURCE.into(),
            model_revision: speech::SILERO_REVISION.into(),
            start_sample: start,
            end_sample: end,
            observed_until_sample: end,
            complete: true,
            decision: SpeechDecision::NoSpeech,
            speech_regions: Vec::new(),
            input_policy: Some(speech::INPUT_POLICY.into()),
            conditioning_sha256: None,
            maximum_probability: Some(0.1),
            maximum_model_probability: Some(0.1),
            uncertain_regions: Vec::new(),
            model_negative_regions: Vec::new(),
            negative_signal_state: Some(speech::NEGATIVE_SIGNAL_STATE.into()),
            classification_scope: Some(speech::CLASSIFICATION_SCOPE.into()),
            evidence_error: false,
        }
    }

    fn core_anchor(regions: Vec<SampleSpan>) -> CoreAnchor {
        CoreAnchor {
            id: "utt".into(),
            start_sample: 0,
            end_sample: 25 * RATE,
            machine_revision: 3,
            audio_revision: 4,
            language_epoch: 1,
            speech_regions: regions,
        }
    }

    #[test]
    fn core_plan_uses_a_verified_negative_gap_and_otherwise_the_balanced_cut() {
        let balanced = plan_cores(
            &core_anchor(vec![SampleSpan {
                start_sample: 0,
                end_sample: 25 * RATE,
            }]),
            no_speech,
        );
        let edges = validate_core_plan(
            &balanced,
            &core_anchor(vec![SampleSpan {
                start_sample: 0,
                end_sample: 25 * RATE,
            }]),
        )
        .unwrap();
        assert_eq!(edges, vec![0, 200_000, 25 * RATE]);
        assert_eq!(balanced.policy, CORE_PLAN_POLICY);
        assert_eq!(balanced.cuts[0].method, "balanced_bounded_fallback");

        let regions = vec![
            SampleSpan {
                start_sample: 0,
                end_sample: 190_000,
            },
            SampleSpan {
                start_sample: 194_000,
                end_sample: 25 * RATE,
            },
        ];
        let row = core_anchor(regions);
        let planned = plan_cores(&row, no_speech);
        assert_eq!(planned.edges[1], 192_000);
        assert_eq!(planned.cuts[0].method, "verified_model_negative_gap");
        validate_core_plan(&planned, &row).unwrap();
    }

    fn sample_segment(text: &str, revision: i64) -> Segment {
        Segment {
            id: "seg-1".into(),
            start: 0.0,
            end: 1.0,
            speaker: "speaker_0".into(),
            text: text.into(),
            language: Some("en".into()),
            language_detection: None,
            transcription_review: None,
            reading_turns: None,
            reading_turn_provenance: None,
            alignment: None,
            speaker_activity: None,
            acoustic_evidence: None,
            refinement_state: Some("provisional".into()),
            machine_revision: Some(revision),
            protected_fields: None,
            review_resolution: None,
            retry_candidate: None,
            canonical_utterance_id: None,
            audio_revision: Some(0),
            review: Some(false),
            capture_source: None,
            rest: Map::new(),
        }
    }

    fn document_with(segment: Segment) -> Document {
        let mut speakers = BTreeMap::new();
        speakers.insert("speaker_0".into(), "Speaker 1".into());
        Document {
            schema_version: 1,
            speakers,
            segments: vec![segment],
            provenance: None,
            warnings: Vec::new(),
            diarization: Vec::new(),
            edited_at: None,
            rest: Map::new(),
        }
    }

    #[test]
    fn saved_publication_preserves_current_edits_and_rejects_invalid_candidates() {
        let mut old = sample_segment("hello", 0);
        old.canonical_utterance_id = Some(old.id.clone());
        old.rest
            .insert("canonical_state".into(), Value::from("sealed"));
        old.rest
            .insert("canonical_machine_revision".into(), Value::from(0));
        let original = document_with(old.clone());
        let mut candidate = old.clone();
        candidate.text = "replacement".into();
        candidate.machine_revision = Some(1);
        candidate
            .rest
            .insert("canonical_machine_revision".into(), Value::from(1));
        let mut replaced = original.clone();
        assert!(publish_saved_passage(&mut replaced, &old, &candidate).unwrap());
        assert_eq!(replaced.segments[0].text, "replacement");
        assert_eq!(replaced.segments[0].id, old.id);
        assert!(!publish_saved_passage(&mut replaced, &old, &candidate).unwrap());

        let body = serde_json::json!({
            "segment_revision": 0,
            "changes": {"text": "edited words"}
        });
        let edited = edit_passage(PassageEditInput {
            status: "ready",
            duration: 10.0,
            revision: 1,
            document: Some(&original),
            segment_id: "seg-1",
            body: &body,
            now: 2.0,
        })
        .unwrap();
        let mut retained = edited.document.clone();
        assert!(!publish_saved_passage(&mut retained, &old, &candidate).unwrap());
        assert!(!publish_saved_passage(&mut retained, &edited.segment, &candidate).unwrap());
        assert_eq!(retained, edited.document);
        for field in [
            "id",
            "bounds",
            "source",
            "revision",
            "canonical",
            "epoch",
            "text",
        ] {
            let mut bad = candidate.clone();
            match field {
                "id" => bad.id = "different".into(),
                "bounds" => bad.end += 0.1,
                "source" => bad.capture_source = Some(serde_json::json!({"source_id":"system"})),
                "revision" => bad.machine_revision = Some(2),
                "canonical" => {
                    bad.rest
                        .insert("canonical_machine_revision".into(), Value::from(2));
                }
                "epoch" => {
                    bad.rest.insert("language_epoch".into(), Value::from(2));
                }
                "text" => bad.text.clear(),
                _ => unreachable!(),
            }
            let mut unchanged = original.clone();
            assert!(
                publish_saved_passage(&mut unchanged, &old, &bad).is_err(),
                "{field}"
            );
            assert_eq!(unchanged, original);
        }
        let applied = apply_canonical_candidate(&edited.document, &candidate);
        assert_eq!(applied.segments[0].text, "edited words");
        assert_eq!(
            applied.segments[0].refinement_state.as_deref(),
            Some("edited")
        );
    }

    struct FakeModels {
        conditioning: Mutex<Vec<Option<GainPolicy>>>,
        ctc_calls: AtomicUsize,
        transcribe_calls: AtomicUsize,
        probe_calls: AtomicUsize,
        probe_scores: Mutex<VecDeque<Result<LanguageProbeOut, ChildError>>>,
        transcription_languages: Mutex<Vec<String>>,
        transcription_crops: Mutex<Vec<(f32, usize)>>,
        clock_text: AtomicBool,
        transcription_results: Mutex<VecDeque<AsrText>>,
        vad_sessions: Mutex<BTreeMap<u64, (u64, Vec<f32>)>>,
        next_session: AtomicUsize,
        fail_diar_open: AtomicBool,
        fail_vad_eof: AtomicBool,
        diar_sessions: AtomicUsize,
        diar_progress: Mutex<BTreeMap<u64, (u64, u64)>>,
        delayed_diar: AtomicBool,
        speaker_switch: AtomicUsize,
        overlapping_diar: bool,
        diar_emitted: Mutex<Vec<Turn>>,
        alignment: Option<(String, CtcScoresOut)>,
        ctc_error: Option<ChildError>,
    }

    impl FakeModels {
        fn english() -> Self {
            Self {
                conditioning: Mutex::new(Vec::new()),
                ctc_calls: AtomicUsize::new(0),
                transcribe_calls: AtomicUsize::new(0),
                probe_calls: AtomicUsize::new(0),
                probe_scores: Mutex::new(VecDeque::new()),
                transcription_languages: Mutex::new(Vec::new()),
                transcription_crops: Mutex::new(Vec::new()),
                clock_text: AtomicBool::new(false),
                transcription_results: Mutex::new(VecDeque::new()),
                vad_sessions: Mutex::new(BTreeMap::new()),
                next_session: AtomicUsize::new(1),
                fail_diar_open: AtomicBool::new(false),
                fail_vad_eof: AtomicBool::new(false),
                diar_sessions: AtomicUsize::new(0),
                diar_progress: Mutex::new(BTreeMap::new()),
                delayed_diar: AtomicBool::new(false),
                speaker_switch: AtomicUsize::new(usize::MAX),
                overlapping_diar: false,
                diar_emitted: Mutex::new(Vec::new()),
                alignment: None,
                ctc_error: None,
            }
        }
    }

    impl Models for FakeModels {
        fn vad_open(&self, conditioning: Option<GainPolicy>) -> Result<Handle, ChildError> {
            self.conditioning.lock().unwrap().push(conditioning);
            let session = self.next_session.fetch_add(1, Ordering::Relaxed) as u64;
            self.vad_sessions
                .lock()
                .unwrap()
                .insert(session, (0, Vec::new()));
            Ok(Handle {
                generation: 1,
                session,
            })
        }
        fn vad_feed(
            &self,
            handle: Handle,
            pcm: &[f32],
            last: bool,
        ) -> Result<VadFeedOut, ChildError> {
            if last && self.fail_vad_eof.load(Ordering::Relaxed) {
                return Err(ChildError::failed("replayed EOF failure", false));
            }
            let mut sessions = self.vad_sessions.lock().unwrap();
            let (observed, pending) = sessions.get_mut(&handle.session).unwrap();
            pending.extend_from_slice(pcm);
            let received = *observed + pending.len() as u64;
            let mut frames = Vec::new();
            while pending.len() >= speech::FRAME_SAMPLES || (last && !pending.is_empty()) {
                let count = pending.len().min(speech::FRAME_SAMPLES);
                let peak = pending[..count]
                    .iter()
                    .map(|sample| sample.abs())
                    .fold(0.0f32, f32::max);
                let probability = if peak > 0.05 { 0.9 } else { 0.1 };
                frames.push(VadFrame {
                    start: *observed,
                    end: *observed + count as u64,
                    raw_p: probability,
                    norm_p: probability,
                    gain: 1.0,
                    constant: false,
                });
                *observed += count as u64;
                pending.drain(..count);
            }
            Ok(VadFeedOut {
                frames,
                received,
                observed: *observed,
            })
        }
        fn vad_close(&self, handle: Handle) -> Result<(), ChildError> {
            self.vad_sessions.lock().unwrap().remove(&handle.session);
            Ok(())
        }
        fn diar_open(&self, _preset: DiarPreset) -> Result<Handle, ChildError> {
            if self.fail_diar_open.load(Ordering::Relaxed) {
                return Err(ChildError::failed("replayed open failure", false));
            }
            self.diar_sessions.fetch_add(1, Ordering::Relaxed);
            let session = self.next_session.fetch_add(1, Ordering::Relaxed) as u64;
            self.diar_progress.lock().unwrap().insert(session, (0, 0));
            Ok(Handle {
                generation: 1,
                session,
            })
        }
        fn diar_feed(
            &self,
            handle: Handle,
            pcm: &[f32],
            last: bool,
        ) -> Result<DiarFeedOut, ChildError> {
            let mut progress = self.diar_progress.lock().unwrap();
            let (received, previous) = progress.get_mut(&handle.session).unwrap();
            *received += pcm.len() as u64;
            let received = *received;
            let observed = if !last && self.delayed_diar.load(Ordering::Relaxed) {
                *previous
            } else {
                received / 160 * 160
            };
            let start = *previous;
            *previous = observed;
            let mut turns: Vec<_> = if observed > start {
                let split = self.speaker_switch.load(Ordering::Relaxed) as u64;
                [
                    (start, observed.min(split), "speaker_0"),
                    (start.max(split), observed, "speaker_1"),
                ]
                .into_iter()
                .filter(|(start, end, _)| start < end)
                .map(|(start, end, speaker)| crate::ipc::DiarTurn {
                    start,
                    end,
                    speaker: speaker.into(),
                })
                .collect()
            } else {
                Vec::new()
            };
            if self.overlapping_diar
                && let Some(first) = turns.first().cloned()
            {
                turns.push(first);
                // Deliberately not sorted by end: a duplicate full turn precedes
                // a shorter overlapping speaker observation in the same batch.
                turns.push(crate::ipc::DiarTurn {
                    start,
                    end: start + (observed - start).div_ceil(2),
                    speaker: "speaker_2".into(),
                });
            }
            self.diar_emitted
                .lock()
                .unwrap()
                .extend(turns.iter().map(|turn| Turn {
                    start: turn.start as f64 / RATE as f64,
                    end: turn.end as f64 / RATE as f64,
                    speaker: turn.speaker.clone(),
                }));
            Ok(DiarFeedOut {
                turns,
                received,
                observed,
            })
        }
        fn diar_close(&self, handle: Handle) -> Result<(), ChildError> {
            self.diar_progress.lock().unwrap().remove(&handle.session);
            self.diar_sessions.fetch_sub(1, Ordering::Relaxed);
            Ok(())
        }
        fn diarize_window(&self, _pcm: &[f32]) -> Result<Vec<Turn>, ChildError> {
            Err(ChildError::Unimplemented)
        }
        fn language_probe(&self, _pcm: &[f32]) -> Result<LanguageProbeOut, ChildError> {
            self.probe_calls.fetch_add(1, Ordering::SeqCst);
            self.probe_scores
                .lock()
                .unwrap()
                .pop_front()
                .unwrap_or_else(|| {
                    Ok(LanguageProbeOut {
                        probabilities: vec![0.99],
                        codes: vec!["en".into()],
                    })
                })
        }
        fn transcribe(
            &self,
            pcm: &[f32],
            language: &str,
            _padding: (u32, u32),
        ) -> Result<AsrText, ChildError> {
            self.transcribe_calls.fetch_add(1, Ordering::SeqCst);
            self.transcription_crops
                .lock()
                .unwrap()
                .push((pcm[0].abs(), pcm.len()));
            self.transcription_languages
                .lock()
                .unwrap()
                .push(language.into());
            if let Some(result) = self.transcription_results.lock().unwrap().pop_front() {
                return Ok(result);
            }
            if self.clock_text.load(Ordering::Relaxed) {
                // Replay ownership markers at quarter-second offsets. Quantized
                // crop origins remain far from these points at bounded core cuts.
                let start = ((pcm[0].abs() - 0.125) * (128 * RATE) as f32).round() as i64;
                let end = start + pcm.len() as i64;
                let text = (0..60)
                    .filter(|second| {
                        let sample = second * RATE + RATE / 4;
                        start <= sample && sample < end
                    })
                    .map(|second| format!("word{second}"))
                    .collect::<Vec<_>>()
                    .join(" ");
                return Ok(AsrText {
                    text: text.clone(),
                    raw_text: text,
                    token_count: 1,
                });
            }
            let text = self
                .alignment
                .as_ref()
                .map(|(text, _)| text.as_str())
                .unwrap_or(if language == "de" { "eins" } else { "hello" });
            Ok(AsrText {
                text: text.into(),
                raw_text: text.into(),
                token_count: 1,
            })
        }
        fn ctc_scores(&self, pcm: &[f32], token_ids: &[i32]) -> Result<CtcScoresOut, ChildError> {
            self.ctc_calls.fetch_add(1, Ordering::SeqCst);
            if let Some(error) = &self.ctc_error {
                return Err(error.clone());
            }
            let Some((_, scores)) = &self.alignment else {
                return Err(ChildError::Unimplemented);
            };
            assert_eq!(pcm.len(), scores.audio_num_samples as usize);
            let columns: Vec<_> = token_ids
                .iter()
                .map(|id| {
                    scores
                        .token_ids
                        .iter()
                        .position(|known| known == id)
                        .unwrap()
                })
                .collect();
            let mut replay = scores.clone();
            replay.gathered = scores
                .gathered
                .chunks(scores.token_ids.len())
                .flat_map(|row| columns.iter().map(|i| row[*i]))
                .collect();
            replay.token_ids = token_ids.to_vec();
            Ok(replay)
        }
        fn embed_voice(&self, _pcm: &[f32]) -> Result<crate::ipc::VoiceEmbedding, ChildError> {
            Err(ChildError::Unimplemented)
        }
        fn setup_check(&self, _models: &[String]) -> Result<Vec<SetupModelReport>, ChildError> {
            Ok(Vec::new())
        }
        fn metrics(&self) -> Result<MetricsOut, ChildError> {
            Ok(MetricsOut {
                peak_mlx_bytes: 0,
                rss_bytes: 0,
            })
        }
    }

    fn loud(samples: usize) -> Vec<f32> {
        (0..samples)
            .map(|index| if index % 2 == 0 { 0.2 } else { -0.1 })
            .collect()
    }

    fn clock_pcm(samples: i64) -> Vec<f32> {
        (0..samples)
            .map(|sample| {
                let amplitude = 0.125 + sample as f32 / (128 * RATE) as f32;
                if sample % 2 == 0 {
                    amplitude
                } else {
                    -amplitude
                }
            })
            .collect()
    }

    fn quiet(samples: usize) -> Vec<f32> {
        (0..samples)
            .map(|index| if index % 2 == 0 { 0.001 } else { -0.0005 })
            .collect()
    }

    fn drive(engine: &mut TestEngine, pcm: &[f32], language: &str) {
        let start = i64::from(engine.recording.writer.frames());
        engine.append_pcm(pcm);
        let mut cursor = start;
        let end = i64::from(engine.recording.writer.frames());
        while cursor < end {
            let next = (cursor + RATE).min(end);
            engine
                .push(LiveCommand::Audio(AudioNotice {
                    start_sample: cursor,
                    end_sample: next,
                    language_epoch: 0,
                    language: language.to_string(),
                }))
                .unwrap();
            cursor = next;
        }
    }

    fn revisions(events: &[EngineEvent]) -> Vec<&Segment> {
        events
            .iter()
            .filter_map(|event| match event {
                EngineEvent::CanonicalRevision { candidate, .. } => Some(candidate.as_ref()),
                _ => None,
            })
            .collect()
    }

    #[test]
    fn activity_sweep_preserves_clipped_overlaps_duplicates_touching_turns_and_gaps() {
        let turns = [
            (5., 7., "c"),
            (2., 5., "a"),
            (9., 11., "d"),
            (1., 3., "a"),
            (3., 4., "b"),
        ]
        .map(|(start, end, speaker)| Turn {
            start,
            end,
            speaker: speaker.into(),
        });
        let actual = activity_regions(&turns, 2 * RATE, 10 * RATE);
        let expected = [
            (2, 3, vec!["a"]),
            (3, 4, vec!["a", "b"]),
            (4, 5, vec!["a"]),
            (5, 7, vec!["c"]),
            (7, 9, vec![]),
            (9, 10, vec!["d"]),
        ];
        assert_eq!(
            actual,
            expected.map(|(start, end, speakers)| timing::ActivityRegion {
                start_sample: start * RATE,
                end_sample: end * RATE,
                speakers: speakers.into_iter().map(str::to_owned).collect(),
            })
        );
        assert!(activity_regions(&turns, 10, 9).is_empty());
        assert_eq!(
            activity_regions(&[], 1, 2),
            [timing::ActivityRegion {
                start_sample: 1,
                end_sample: 2,
                speakers: vec![]
            }]
        );
    }

    fn accept_live_events(
        document: &mut Document,
        events: Vec<EngineEvent>,
        mut feedback: impl FnMut(&EditedBoundary) -> Vec<EngineEvent>,
    ) {
        let mut events = VecDeque::from(events);
        while let Some(event) = events.pop_front() {
            if let EngineEvent::CanonicalRevision { candidate, .. } = event {
                if let Some(boundary) = edited_boundary(document, &candidate) {
                    for event in feedback(&boundary).into_iter().rev() {
                        events.push_front(event);
                    }
                } else {
                    *document = apply_canonical_candidate(document, &candidate);
                }
            }
        }
    }

    #[test]
    fn edited_continuations_keep_packet_independent_ids_and_language_boundaries() {
        let cut = 99840;
        let language_boundary = 9 * RATE;
        let end = 15 * RATE + 159;
        let mut expected = None;
        for splits in [
            vec![language_boundary, end],
            vec![
                cut + 1,
                cut + 511,
                cut + 1025,
                language_boundary,
                language_boundary + 37,
                end,
            ],
        ] {
            let mut engine =
                TestEngine::new(EngineConfig::new("partitions", "en"), FakeModels::english())
                    .unwrap();
            engine.append_pcm(&loud(end as usize));
            let first = engine.handle(notice(0, cut, 0)).unwrap();
            let mut document = empty_document();
            accept_live_events(&mut document, first, |boundary| {
                engine.continue_edited(boundary).unwrap()
            });
            let row = document.segments[0].clone();
            document = edit_passage(PassageEditInput {status:"recording",duration:7.,revision:1,document:Some(&document),segment_id:&row.id,
                body:&serde_json::json!({"segment_revision":row.machine_revision,"changes":{"text":"Corrected prefix"}}),now:1.}).unwrap().document;
            let mut cursor = cut;
            let mut epoch = 0;
            for stop in splits {
                let events = engine.handle(notice(cursor, stop, epoch)).unwrap();
                accept_live_events(&mut document, events, |boundary| {
                    engine.continue_edited(boundary).unwrap()
                });
                cursor = stop;
                if stop == language_boundary {
                    epoch = 1;
                    let events = engine
                        .handle(LiveCommand::Language {
                            language: "de".into(),
                            generation: 1,
                            start_sample: stop,
                        })
                        .unwrap();
                    accept_live_events(&mut document, events, |boundary| {
                        engine.continue_edited(boundary).unwrap()
                    });
                }
            }
            let events = engine.handle(LiveCommand::Stop).unwrap();
            accept_live_events(&mut document, events, |boundary| {
                engine.continue_edited(boundary).unwrap()
            });
            let count = document.segments.len();
            let events = engine.handle(LiveCommand::Stop).unwrap();
            accept_live_events(&mut document, events, |boundary| {
                engine.continue_edited(boundary).unwrap()
            });
            assert_eq!(document.segments.len(), count);
            assert_eq!(document.segments[0].text, "Corrected prefix");
            let spans: Vec<_> = document
                .segments
                .iter()
                .map(|row| {
                    (
                        row.id.clone(),
                        row.rest["start_sample"].as_i64().unwrap(),
                        row.rest["end_sample"].as_i64().unwrap(),
                        row.text.clone(),
                        row.language.clone(),
                    )
                })
                .collect();
            assert!(
                spans.windows(2).all(|pair| pair[0].2 == pair[1].1),
                "continuations must partition canonical audio: {spans:?}"
            );
            assert!(
                spans
                    .iter()
                    .any(|row| row.1 == language_boundary && row.3 == "eins")
            );
            if let Some(expected) = &expected {
                assert_eq!(&spans, expected);
            } else {
                expected = Some(spans);
            }
            let sealed = document.segments.last().unwrap().clone();
            let edit = edit_passage(PassageEditInput {status:"ready",duration:end as f64/RATE as f64,revision:2,document:Some(&document),segment_id:&sealed.id,
                body:&serde_json::json!({"segment_revision":sealed.machine_revision,"changes":{"text":"Sealed correction"}}),now:2.}).unwrap();
            assert!(
                !edit
                    .segment
                    .rest
                    .contains_key("canonical_text_boundary_sample")
            );
        }
    }

    #[test]
    fn edited_source_eof_uses_each_clipped_speaker_ledger_after_turns_pruning() {
        let cut = 99840;
        let end = 14 * RATE + 159;
        let mut inputs = [TestRecording::new(), TestRecording::new()];
        for input in &mut inputs {
            input.writer.write_samples(&loud(end as usize)).unwrap();
        }
        let mut config = EngineConfig::new("edited-sources", "en");
        config.processed_microphone_contract = Some("a".repeat(64));
        config.capture_catalog_sha256 = Some("b".repeat(64));
        let models = FakeModels::english();
        models.speaker_switch.store(cut as usize, Ordering::Relaxed);
        let mut runtime = SourceRuntime::new(
            config,
            models,
            inputs
                .each_ref()
                .map(|input| RecordingReader::open(&input.path).unwrap()),
        )
        .unwrap();
        let first = runtime.handle(notice(0, cut, 0)).unwrap();
        let mut document = empty_document();
        accept_live_events(&mut document, first, |boundary| {
            runtime.continue_edited(boundary).unwrap()
        });
        let originals = document.segments.clone();
        for row in &originals {
            document = edit_passage(PassageEditInput {status:"recording",duration:7.,revision:1,document:Some(&document),segment_id:&row.id,
                body:&serde_json::json!({"segment_revision":row.machine_revision,"changes":{"text":"Corrected source prefix"}}),now:1.}).unwrap().document;
        }
        let mut pending = runtime.handle(notice(cut, end, 0)).unwrap();
        pending.extend(runtime.handle(LiveCommand::Stop).unwrap());
        assert!(
            runtime
                .engines
                .iter()
                .all(|engine| engine.turns.iter().all(|turn| turn.end > 13.))
        );
        accept_live_events(&mut document, pending, |boundary| {
            runtime.continue_edited(boundary).unwrap()
        });
        assert_eq!(document.segments.len(), 4);
        for (source, old_owner, new_owner) in [
            ("microphone_clean", "speaker_0", "speaker_1"),
            ("system", "speaker_32", "speaker_33"),
        ] {
            let rows: Vec<_> = document
                .segments
                .iter()
                .filter(|row| row.capture_source.as_ref().unwrap()["source_id"] == source)
                .collect();
            assert_eq!(rows.len(), 2);
            assert_eq!(rows[0].speaker, old_owner);
            assert_eq!(rows[1].speaker, new_owner);
            assert_eq!(rows[0].rest["end_sample"], cut);
            assert_eq!(rows[1].rest["start_sample"], cut);
            assert_eq!(rows[1].text, "hello");
        }
    }

    #[test]
    fn edited_open_passage_does_not_swallow_continuing_audio() {
        let mut engine = TestEngine::new(
            EngineConfig::new("edited-open", "en"),
            FakeModels::english(),
        )
        .unwrap();
        engine.append_pcm(&loud((14 * RATE + 159) as usize));
        let mut document = empty_document();
        for candidate in revisions(&engine.handle(notice(0, 6 * RATE + 512, 0)).unwrap()) {
            document = apply_canonical_candidate(&document, candidate);
        }
        let original = document.segments[0].clone();
        let cut = original.rest["canonical_text_end_sample"].as_i64().unwrap();
        document = edit_passage(PassageEditInput {
            status:"recording",duration:7.,revision:1,document:Some(&document),segment_id:&original.id,
            body:&serde_json::json!({"segment_revision":original.machine_revision,"changes":{"text":"Human introduction","start":0.1,"end":6.}}),now:1.,
        }).unwrap().document;
        let mut events = engine
            .handle(notice(6 * RATE + 512, 14 * RATE + 159, 0))
            .unwrap();
        events.extend(engine.handle(LiveCommand::Stop).unwrap());
        let mut events = VecDeque::from(events);
        while let Some(event) = events.pop_front() {
            if let EngineEvent::CanonicalRevision { candidate, .. } = event {
                if let Some(boundary) = edited_boundary(&document, &candidate) {
                    events.extend(engine.continue_edited(&boundary).unwrap());
                } else {
                    document = apply_canonical_candidate(&document, &candidate);
                }
            }
        }
        let edited = document
            .segments
            .iter()
            .find(|r| r.id == original.id)
            .unwrap();
        assert_eq!(
            (edited.text.as_str(), edited.start, edited.end),
            ("Human introduction", 0.1, 6.)
        );
        let tail = document
            .segments
            .iter()
            .find(|r| r.id != original.id && !r.text.is_empty())
            .expect("later speech must have its own passage");
        assert_eq!(tail.rest["start_sample"], cut);
        assert_eq!(tail.rest["end_sample"], 14 * RATE + 159);
    }

    #[test]
    fn live_publication_exposes_text_coverage_separately_from_growing_audio() {
        let mut engine =
            TestEngine::new(EngineConfig::new("coverage", "en"), FakeModels::english()).unwrap();
        engine.append_pcm(&loud((7 * RATE) as usize));
        let first = engine.handle(notice(0, 6 * RATE + 512, 0)).unwrap();
        let decoded = revisions(&first)
            .into_iter()
            .find(|r| !r.text.is_empty())
            .unwrap()
            .clone();
        let decoded_end = decoded.rest["end_sample"].as_i64().unwrap();
        let growing = engine.handle(notice(6 * RATE + 512, 7 * RATE, 0)).unwrap();
        let published = revisions(&growing);
        let row = published.last().unwrap();
        assert!(row.rest["end_sample"].as_i64().unwrap() > decoded_end);
        assert_eq!(row.rest["canonical_text_end_sample"], decoded_end);
        assert!(
            row.rest.get("text_audio_anchor").is_none(),
            "stale timing evidence must still be invalidated"
        );
        assert_eq!(decoded.rest["text_audio_anchor"]["end_sample"], decoded_end);
    }

    #[test]
    fn recovery_rebuilds_acoustics_before_publication_and_keeps_absolute_language_and_eof() {
        for cut in [0, RATE + 37, 2 * RATE] {
            for silence in [false, true] {
                let through = 2 * RATE + 100;
                let end = 3 * RATE + 159;
                let mut config = EngineConfig::new("recovered", "en");
                config.recovery = Some(StreamRecovery {
                    generation: 1,
                    start_sample: cut,
                    through_sample: through,
                    language_epoch: if cut >= RATE { 1 } else { 0 },
                    source_boundaries: BTreeMap::new(),
                });
                let mut engine = TestEngine::new(config, FakeModels::english()).unwrap();
                let mut pcm = loud(end as usize);
                if silence {
                    pcm[..RATE as usize].fill(0.);
                }
                engine.append_pcm(&pcm);
                let mut events = engine.handle(notice(0, RATE, 0)).unwrap();
                events.extend(
                    engine
                        .handle(LiveCommand::Language {
                            language: "de".into(),
                            generation: 1,
                            start_sample: RATE,
                        })
                        .unwrap(),
                );
                events.extend(
                    engine
                        .handle(LiveCommand::Audio(AudioNotice {
                            start_sample: RATE,
                            end_sample: through,
                            language_epoch: 1,
                            language: "de".into(),
                        }))
                        .unwrap(),
                );
                for (start, stop) in [(through, through + RATE), (through + RATE, end)] {
                    engine
                        .push(LiveCommand::Audio(AudioNotice {
                            start_sample: start,
                            end_sample: stop,
                            language_epoch: 1,
                            language: "de".into(),
                        }))
                        .unwrap();
                }
                engine.push(LiveCommand::Stop).unwrap();
                while let Some(next) = engine.poll().unwrap() {
                    events.extend(next);
                }
                let mut document = empty_document();
                for event in &events {
                    if let EngineEvent::CanonicalRevision { candidate, .. } = event {
                        assert!(candidate.rest["start_sample"].as_i64().unwrap() >= cut);
                        assert_eq!(candidate.rest["stream_generation"], 1);
                        assert_ne!(
                            candidate.id,
                            utterance_id(
                                "recovered",
                                candidate.rest["language_epoch"].as_i64().unwrap(),
                                candidate.rest["start_sample"].as_i64().unwrap()
                            )
                        );
                        document = apply_canonical_candidate(&document, candidate);
                    }
                }
                assert!(
                    document
                        .segments
                        .iter()
                        .any(|r| r.text == "eins" && r.speaker == "speaker_64")
                );
                assert!(events.iter().any(|e|matches!(e,EngineEvent::CaptureFinished{canonical_observed_sample,..} if *canonical_observed_sample==end)));
            }
        }
        let mut bad = EngineConfig::new("bad", "en");
        bad.recovery = Some(StreamRecovery {
            generation: u32::MAX,
            start_sample: 0,
            through_sample: 1,
            language_epoch: 0,
            source_boundaries: BTreeMap::new(),
        });
        assert!(TestEngine::new(bad, FakeModels::english()).is_err());
    }

    #[test]
    fn saved_audio_replay_handles_packet_splits_and_does_not_revisit_sealed_utterances() {
        let pcm = [loud(16_037), vec![0.; 16_000], loud(8_013)].concat();
        let mut split =
            TestEngine::new(EngineConfig::new("saved", "en"), FakeModels::english()).unwrap();
        let mut split_document = empty_document();
        let mut cursor = 0;
        for end in [1, 511, 512, 513, 1023, 1024, 16_037, 32_037, pcm.len()] {
            split.append_pcm(&pcm[cursor..end]);
            for event in split.handle(notice(cursor as i64, end as i64, 0)).unwrap() {
                if let EngineEvent::CanonicalRevision { candidate, .. } = event {
                    split_document = apply_canonical_candidate(&split_document, &candidate);
                }
            }
            cursor = end;
        }
        for event in split.handle(LiveCommand::Stop).unwrap() {
            if let EngineEvent::CanonicalRevision { candidate, .. } = event {
                split_document = apply_canonical_candidate(&split_document, &candidate);
            }
        }
        assert_eq!(split.next_utterance, split.book.order.len());
        assert!(split.published.is_empty());
        assert!(split.last_decoded.is_empty());
        let decoded = split.models.transcribe_calls.load(Ordering::Relaxed);
        assert!(
            revisions(
                &split
                    .handle(LiveCommand::Flush {
                        request_id: None,
                        through_sample: None
                    })
                    .unwrap()
            )
            .is_empty()
        );
        assert_eq!(
            decoded,
            split.models.transcribe_calls.load(Ordering::Relaxed)
        );

        // New model state reconstructs from the same retained file, without appending PCM.
        let mut replay = Engine::single(
            EngineConfig::new("saved", "en"),
            FakeModels::english(),
            &split.recording.path,
        )
        .unwrap();
        let mut replay_document = empty_document();
        for event in replay
            .handle(notice(0, pcm.len() as i64, 0))
            .unwrap()
            .into_iter()
            .chain(replay.handle(LiveCommand::Stop).unwrap())
        {
            if let EngineEvent::CanonicalRevision { candidate, .. } = event {
                replay_document = apply_canonical_candidate(&replay_document, &candidate);
            }
        }
        let content = |document: &Document| {
            document
                .segments
                .iter()
                .map(|s| {
                    (
                        s.id.clone(),
                        s.start,
                        s.end,
                        s.speaker.clone(),
                        s.text.clone(),
                    )
                })
                .collect::<Vec<_>>()
        };
        assert_eq!(content(&split_document), content(&replay_document));
        assert_eq!(replay.ledger.lock().unwrap().end_sample(), pcm.len() as i64);
        assert_eq!(replay.diar_observed, pcm.len() as i64 / 160 * 160);
    }

    #[test]
    fn growing_activity_only_revisits_the_affected_observation_tail() {
        // Counted replay, not a timing benchmark. Full input turns remain only
        // in the fake model so its independent full sweep can check every prefix.
        let mut models = FakeModels::english();
        models.overlapping_diar = true;
        models
            .speaker_switch
            .store(32 * RATE as usize, Ordering::Relaxed);
        let mut engine =
            TestEngine::new(EngineConfig::new("activity-growth", "en"), models).unwrap();
        let end = 35 * RATE + 513;
        engine.append_pcm(&loud(end as usize));
        engine.handle(notice(0, 30 * RATE, 0)).unwrap();
        let identity = engine.book.order[0].clone();
        let initial = &engine.book.rows[&identity];
        let affected_start = initial
            .speaker_activity
            .as_ref()
            .unwrap()
            .regions
            .last()
            .unwrap()
            .start_sample;
        let historical_turns = engine.models.diar_emitted.lock().unwrap().len();
        let decoded = engine.models.transcribe_calls.load(Ordering::Relaxed);
        ACTIVITY_SWEEPS.with_borrow_mut(|calls| *calls = Some(Vec::new()));
        for next in [31 * RATE, 32 * RATE, 33 * RATE, 34 * RATE, 35 * RATE, end] {
            // One missing observation batch then catch-up must preserve the
            // previously observed prefix and not claim the received gap as known.
            engine
                .models
                .delayed_diar
                .store(next == 31 * RATE, Ordering::Relaxed);
            let start = engine.received;
            engine.handle(notice(start, next, 0)).unwrap();
            let calls = ACTIVITY_SWEEPS.with_borrow_mut(Option::take);
            let row = &engine.book.rows[&identity];
            let observed = row.end_sample.min(engine.diar_observed);
            let reference = activity_regions(
                &engine.models.diar_emitted.lock().unwrap(),
                row.start_sample,
                observed,
            );
            assert_eq!(
                row.speaker_activity.as_ref().unwrap().regions,
                reference,
                "prefix {next}"
            );
            let owners: std::collections::BTreeSet<_> = reference
                .iter()
                .flat_map(|r| r.speakers.iter().cloned())
                .collect();
            assert_eq!(
                row.speaker_candidates,
                owners.into_iter().collect::<Vec<_>>()
            );
            ACTIVITY_SWEEPS.with_borrow_mut(|slot| *slot = calls);
        }
        let calls = ACTIVITY_SWEEPS.with_borrow_mut(Option::take).unwrap();
        assert_eq!(
            engine.models.transcribe_calls.load(Ordering::Relaxed),
            decoded,
            "these observations precede the next long ASR deadline"
        );
        let final_events = engine.handle(LiveCommand::Stop).unwrap();
        let rows = revisions(&final_events);
        let row = rows.last().unwrap();
        let activity: timing::SpeakerActivity =
            serde_json::from_value(row.speaker_activity.clone().unwrap()).unwrap();
        let observed = end / 160 * 160;
        assert_eq!(activity.observed_end_sample, Some(observed));
        assert!(
            observed < end,
            "the sub-hop EOF tail is still unobserved by diarization"
        );
        assert_eq!(
            activity.regions,
            activity_regions(&engine.models.diar_emitted.lock().unwrap(), 0, observed)
        );
        assert!(
            calls.iter().all(|(start, _, _)| *start >= affected_start),
            "unchanged activity prefix was rebuilt: {calls:?}"
        );
        assert!(
            calls.iter().map(|(_, _, turns)| turns).sum::<usize>() < historical_turns,
            "new observations must not repeatedly inspect all historical turns: {calls:?}"
        );
    }

    #[test]
    fn growing_live_utterance_does_not_decode_completed_audio_again() {
        // Counted model replay over retained PCM, not an elapsed-time benchmark.
        // Monotone amplitudes identify crop origins after PCM16 round-tripping.
        let final_end = 42 * RATE + 1025;
        let pcm = clock_pcm(final_end);
        let mut engine = TestEngine::new(
            EngineConfig::new("growing-work", "en"),
            FakeModels::english(),
        )
        .unwrap();
        engine.models.clock_text.store(true, Ordering::Relaxed);
        engine.append_pcm(&pcm);
        let mut document = empty_document();
        for event in engine.handle(notice(0, 30 * RATE, 0)).unwrap() {
            if let EngineEvent::CanonicalRevision { candidate, .. } = event {
                document = apply_canonical_candidate(&document, &candidate);
            }
        }
        assert_eq!(document.segments.len(), 1);
        let identity = document.segments[0].id.clone();
        let crops = engine.models.transcription_crops.lock().unwrap().clone();
        let completed_end = crops.last().unwrap().0;
        assert!(
            completed_end > crops[0].0,
            "initial bounded cores own a prefix"
        );
        let mut previous_calls = crops.len();
        // At exactly 42s Silero has observed only 41.984s, less than six
        // seconds since the 36s decode. Include the next real VAD frame.
        for (start, end) in [(30 * RATE, 36 * RATE), (36 * RATE, 42 * RATE + 512)] {
            for event in engine.handle(notice(start, end, 0)).unwrap() {
                if let EngineEvent::CanonicalRevision { candidate, .. } = event {
                    document = apply_canonical_candidate(&document, &candidate);
                }
            }
            let crops = engine.models.transcription_crops.lock().unwrap();
            let next = &crops[previous_calls..];
            assert!(
                !next.is_empty(),
                "new speech must still be decoded at received {end}, observed {}",
                engine.ledger.lock().unwrap().end_sample()
            );
            assert!(
                next.iter().all(|(first, _)| *first >= completed_end),
                "accepted historical prefix was decoded again at {end}: {next:?}"
            );
            assert!(
                next.iter()
                    .all(|(_, count)| *count as i64 <= MAX_DECODE_SAMPLES)
            );
            assert!(
                next.iter().map(|(_, count)| *count as i64).sum::<i64>()
                    <= MAX_DECODE_SAMPLES + end - start,
                "growth must decode only a bounded active tail and new samples"
            );
            previous_calls = crops.len();
            assert_eq!(document.segments.len(), 1);
            assert_eq!(document.segments[0].id, identity);
            let expected = (0..end / RATE)
                .map(|i| format!("word{i}"))
                .collect::<Vec<_>>()
                .join(" ");
            assert_eq!(
                document.segments[0].text, expected,
                "exact disjoint acoustic ownership"
            );
        }
        for end in [42 * RATE + 513, 42 * RATE + 1023, final_end] {
            let start = engine.received;
            for event in engine.handle(notice(start, end, 0)).unwrap() {
                if let EngineEvent::CanonicalRevision { candidate, .. } = event {
                    document = apply_canonical_candidate(&document, &candidate);
                }
            }
        }
        for event in engine.handle(LiveCommand::Stop).unwrap() {
            if let EngineEvent::CanonicalRevision { candidate, .. } = event {
                document = apply_canonical_candidate(&document, &candidate);
            }
        }
        assert_eq!(document.segments.len(), 1);
        assert_eq!(document.segments[0].id, identity);
        assert_eq!(document.segments[0].end, final_end as f64 / RATE as f64);
        assert_eq!(document.segments[0].text.split_whitespace().count(), 42);
        assert!(
            engine.models.transcription_crops.lock().unwrap()[previous_calls..]
                .iter()
                .all(|(first, _)| *first >= completed_end)
        );
        assert!(
            engine.last_decoded.is_empty(),
            "sealed progress is released"
        );
    }

    #[test]
    fn growing_live_cores_retry_incomplete_results_without_freezing_or_erasing_words() {
        for failed_prefix in [true, false] {
            let mut engine = TestEngine::new(
                EngineConfig::new("incomplete-core", "en"),
                FakeModels::english(),
            )
            .unwrap();
            engine.models.clock_text.store(true, Ordering::Relaxed);
            let retry_end = 42 * RATE + 512;
            engine.append_pcm(&clock_pcm(retry_end));
            let initial = if failed_prefix { 8 * RATE } else { 30 * RATE };
            let mut document = empty_document();
            for event in engine.handle(notice(0, initial, 0)).unwrap() {
                if let EngineEvent::CanonicalRevision { candidate, .. } = event {
                    document = apply_canonical_candidate(&document, &candidate);
                }
            }
            let previous_text = document.segments[0].text.clone();
            assert!(!previous_text.is_empty());
            engine
                .models
                .transcription_results
                .lock()
                .unwrap()
                .push_back(AsrText {
                    text: "incomplete words".into(),
                    raw_text: "incomplete words".into(),
                    token_count: 448,
                });
            let failed_end = if failed_prefix { 30 * RATE } else { 36 * RATE };
            for event in engine.handle(notice(initial, failed_end, 0)).unwrap() {
                if let EngineEvent::CanonicalRevision { candidate, .. } = event {
                    document = apply_canonical_candidate(&document, &candidate);
                }
            }
            assert_eq!(document.segments[0].text, previous_text);
            assert_eq!(
                document.segments[0].rest["canonical_unresolved"],
                "incomplete_cohere_revision"
            );
            let count = engine.models.transcription_crops.lock().unwrap().len();
            for event in engine.handle(notice(failed_end, retry_end, 0)).unwrap() {
                if let EngineEvent::CanonicalRevision { candidate, .. } = event {
                    document = apply_canonical_candidate(&document, &candidate);
                }
            }
            assert_eq!(
                document.segments[0].text,
                (0..42)
                    .map(|i| format!("word{i}"))
                    .collect::<Vec<_>>()
                    .join(" ")
            );
            let crops = engine.models.transcription_crops.lock().unwrap();
            assert_eq!(
                crops[count].0 == crops[0].0,
                failed_prefix,
                "only an incomplete first core must be retried from the origin"
            );
            assert_eq!(document.segments.len(), 1);
        }
    }

    #[test]
    fn growing_live_cores_keep_decisions_and_reset_at_edit_source_and_language_boundaries() {
        for source in SOURCE_IDS {
            let mut config = EngineConfig::new("growing-source", "auto");
            config.source_id = Some(source.into());
            config.capture_catalog_sha256 = Some("a".repeat(64));
            let mut engine = TestEngine::new(config, FakeModels::english()).unwrap();
            engine.models.clock_text.store(true, Ordering::Relaxed);
            let end = 49 * RATE + 137;
            engine.append_pcm(&clock_pcm(end));
            let mut document = empty_document();
            let events = engine.handle(notice(0, 30 * RATE, 0)).unwrap();
            accept_live_events(&mut document, events, |b| {
                engine.continue_edited(b).unwrap()
            });
            let original = document.segments[0].clone();
            let initial_probes = original.language_detection.as_ref().unwrap()["probes"]
                .as_array()
                .unwrap();
            let retained_probes = initial_probes
                .iter()
                .filter(|p| p["end_sample"].as_i64().unwrap() <= 15 * RATE)
                .cloned()
                .collect::<Vec<_>>();
            assert!(!retained_probes.is_empty());
            engine
                .models
                .probe_scores
                .lock()
                .unwrap()
                .extend((0..7).map(|_| {
                    Ok(LanguageProbeOut {
                        probabilities: vec![0.99, 0.01],
                        codes: vec!["de".into(), "en".into()],
                    })
                }));
            let events = engine.handle(notice(30 * RATE, 36 * RATE, 0)).unwrap();
            accept_live_events(&mut document, events, |b| {
                engine.continue_edited(b).unwrap()
            });
            let mixed = document.segments[0].clone();
            assert_eq!(mixed.id, original.id);
            assert_eq!(mixed.language, None);
            let detection = mixed.language_detection.as_ref().unwrap();
            assert_eq!(detection["reason"], "mixed");
            assert_eq!(
                &detection["probes"].as_array().unwrap()[..retained_probes.len()],
                retained_probes.as_slice()
            );
            assert!(
                detection["probes"]
                    .as_array()
                    .unwrap()
                    .iter()
                    .any(|p| p["language"] == "de"
                        && p["start_sample"].as_i64().unwrap() >= 15 * RATE)
            );
            document = edit_passage(PassageEditInput {
                status: "recording",
                duration: 36.,
                revision: 1,
                document: Some(&document),
                segment_id: &mixed.id,
                body: &serde_json::json!({
                    "segment_revision": mixed.machine_revision,
                    "changes": {"text": "Exact edited prefix", "start": 0.1}
                }),
                now: 1.,
            })
            .unwrap()
            .document;
            document
                .speakers
                .insert(mixed.speaker.clone(), "Saved name".into());
            for (start, stop) in [(36 * RATE, 42 * RATE), (42 * RATE, 48 * RATE)] {
                let events = engine.handle(notice(start, stop, 0)).unwrap();
                accept_live_events(&mut document, events, |b| {
                    engine.continue_edited(b).unwrap()
                });
            }
            let events = engine
                .handle(LiveCommand::Language {
                    language: "fr".into(),
                    generation: 1,
                    start_sample: 48 * RATE,
                })
                .unwrap();
            accept_live_events(&mut document, events, |b| {
                engine.continue_edited(b).unwrap()
            });
            let events = engine
                .handle(LiveCommand::Audio(AudioNotice {
                    start_sample: 48 * RATE,
                    end_sample: end,
                    language_epoch: 1,
                    language: "fr".into(),
                }))
                .unwrap();
            accept_live_events(&mut document, events, |b| {
                engine.continue_edited(b).unwrap()
            });
            let events = engine.handle(LiveCommand::Stop).unwrap();
            accept_live_events(&mut document, events, |b| {
                engine.continue_edited(b).unwrap()
            });
            assert_eq!(document.segments.len(), 3);
            assert_eq!(document.segments[0].text, "Exact edited prefix");
            assert_eq!(document.segments[0].start, 0.1);
            assert_eq!(
                document.segments[0].language_detection,
                mixed.language_detection
            );
            assert_eq!(document.speakers[&mixed.speaker], "Saved name");
            assert_eq!(
                document.segments[1].text,
                (36..48)
                    .map(|i| format!("word{i}"))
                    .collect::<Vec<_>>()
                    .join(" ")
            );
            assert_eq!(document.segments[1].rest["language_mode"], "auto");
            assert_eq!(document.segments[1].language.as_deref(), Some("en"));
            assert!(
                document.segments[1].language_detection.as_ref().unwrap()["probes"]
                    .as_array()
                    .unwrap()
                    .iter()
                    .all(|p| p["start_sample"].as_i64().unwrap() >= 36 * RATE)
            );
            assert_eq!(document.segments[2].text, "word48");
            assert_eq!(document.segments[2].language.as_deref(), Some("fr"));
            assert_eq!(document.segments[2].rest["language_epoch"], 1);
            assert!(
                document
                    .segments
                    .iter()
                    .all(|row| row.capture_source.as_ref().unwrap()["source_id"] == source)
            );
            assert!(engine.last_decoded.is_empty());
        }
    }

    #[test]
    fn live_routed_decisions_keep_absolute_recent_context_and_captured_epoch() {
        let mut engine = TestEngine::new(
            EngineConfig::new("language-clock", "en"),
            FakeModels::english(),
        )
        .unwrap();
        engine.models.probe_scores.lock().unwrap().extend([
            Ok(LanguageProbeOut {
                probabilities: vec![0.99, 0.01],
                codes: vec!["en".into(), "ru".into()],
            }),
            Ok(LanguageProbeOut {
                probabilities: vec![0.55, 0.45],
                codes: vec!["ru".into(), "en".into()],
            }),
        ]);
        engine.append_pcm(&loud((13 * RATE) as usize));
        let mut events = engine.handle(notice(0, 7 * RATE, 0)).unwrap();
        events.extend(
            engine
                .handle(LiveCommand::Language {
                    language: "auto".into(),
                    generation: 1,
                    start_sample: 7 * RATE,
                })
                .unwrap(),
        );
        events.extend(
            engine
                .handle(LiveCommand::Audio(AudioNotice {
                    start_sample: 7 * RATE,
                    end_sample: 13 * RATE,
                    language_epoch: 1,
                    language: "auto".into(),
                }))
                .unwrap(),
        );
        events.extend(engine.handle(LiveCommand::Stop).unwrap());
        let rows = revisions(&events);
        let row = rows
            .iter()
            .rev()
            .find(|row| row.rest["language_epoch"] == 1)
            .unwrap();
        assert_eq!(row.rest["language_mode"], "auto");
        assert_eq!(row.language.as_deref(), Some("en"));
        let detection = row.language_detection.as_ref().unwrap();
        assert_eq!(detection["reason"], "recent_context");
        assert_eq!(detection["context_end_sample"], 10 * RATE);
        assert_eq!(detection["probes"][0]["start_sample"], 7 * RATE);
        assert_eq!(detection["probes"][0]["end_sample"], 10 * RATE);
        assert_eq!(detection["probes"][1]["start_sample"], 10 * RATE);
        assert_eq!(detection["probes"][1]["end_sample"], 13 * RATE);
        assert_eq!(
            detection["probes"][1]["decision"]["context_end_sample"],
            10 * RATE
        );
        assert_eq!(
            detection["probes"][1]["decision"]["candidates"][0]["language"],
            "ru"
        );
        assert_eq!(row.review, Some(true));
        assert_eq!(engine.models.probe_calls.load(Ordering::SeqCst), 2);
        let encoded = serde_json::to_value(row).unwrap();
        assert_eq!(encoded["language_detection"], *detection);
        let manual = rows
            .iter()
            .rev()
            .find(|row| row.rest["language_epoch"] == 0)
            .unwrap();
        assert_eq!(
            manual.language_detection.as_ref().unwrap()["reason"],
            "override"
        );
    }

    #[test]
    fn missing_audio_and_invalid_conditioning_fail_before_inference() {
        let mut engine =
            TestEngine::new(EngineConfig::new("missing", "en"), FakeModels::english()).unwrap();
        assert!(engine.handle(notice(0, 512, 0)).is_err());
        assert_eq!(engine.received, 0);
        assert!(engine.models.conditioning.lock().unwrap().is_empty());
        let mut config = EngineConfig::new("bad-source", "en");
        config.source_id = Some("system".into());
        config.processed_microphone_contract = Some("a".repeat(64));
        assert!(TestEngine::new(config.clone(), FakeModels::english()).is_err());
        config.source_id = Some("microphone_clean".into());
        config.processed_microphone_contract = Some("bad".into());
        assert!(TestEngine::new(config, FakeModels::english()).is_err());
    }

    #[test]
    fn language_boundary_inside_unobserved_final_hop_stays_unknown() {
        let mut engine =
            TestEngine::new(EngineConfig::new("partial", "en"), FakeModels::english()).unwrap();
        engine.append_pcm(&loud(16_037));
        engine.handle(notice(0, 16_037, 0)).unwrap();
        engine
            .handle(LiveCommand::Language {
                language: "de".into(),
                generation: 1,
                start_sample: 16_037,
            })
            .unwrap();
        engine.append_pcm(&loud(50));
        engine.handle(notice(16_037, 16_087, 1)).unwrap();
        engine.handle(LiveCommand::Stop).unwrap();
        assert_eq!(engine.diar_observed, 16_000);
        let tail = engine
            .book
            .rows
            .values()
            .find(|row| row.start_sample == 16_037)
            .unwrap();
        assert!(tail.speaker_candidates.is_empty());
        assert!(tail.speaker_activity.as_ref().unwrap().regions.is_empty());
        assert_eq!(engine.models.diar_sessions.load(Ordering::Relaxed), 0);
    }

    #[test]
    fn delayed_diarization_and_final_only_turns_reach_sealed_passages() {
        let mut engine =
            TestEngine::new(EngineConfig::new("delayed", "en"), FakeModels::english()).unwrap();
        engine.models.delayed_diar.store(true, Ordering::Relaxed);
        let pcm = [loud(48_000), vec![0.; 16_000], loud(16_037)].concat();
        engine.append_pcm(&pcm);
        let audio = engine.handle(notice(0, pcm.len() as i64, 0)).unwrap();
        assert!(engine.ledger.lock().unwrap().end_sample() > 0);
        assert_eq!(engine.diar_observed, 0);
        let pause = engine
            .handle(LiveCommand::Flush {
                request_id: Some("delayed-pause".into()),
                through_sample: Some(pcm.len() as i64),
            })
            .unwrap();
        assert!(pause.iter().any(
            |event| matches!(event, EngineEvent::FlushAck {received_sample, available_sample, ..}
            if *received_sample == 80_037 && *available_sample == 0)
        ));
        assert_eq!(
            engine.book.cursor(),
            0,
            "unobserved speaker evidence cannot seal passages"
        );
        assert!(engine.published.is_empty());
        let events = engine.handle(LiveCommand::Stop).unwrap();
        assert_eq!(engine.diar_observed, 80_000);
        assert_eq!(engine.book.cursor(), 80_037);
        assert_eq!(
            engine.turns,
            vec![Turn {
                start: 0.,
                end: 5.,
                speaker: "speaker_0".into()
            }]
        );
        assert!(!revisions(&events).is_empty());
        for row in engine.book.rows.values() {
            assert_eq!(row.speaker_candidates, vec!["speaker_0"]);
            assert_eq!(
                row.speaker_activity.as_ref().unwrap().observed_end_sample,
                Some(row.end_sample.min(80_000))
            );
        }
        assert!(engine.book.closed());
        for (phase, batch, processed, flush) in [
            ("audio", audio, 0., false),
            ("pause", pause, 0., true),
            ("stop", events, 80_000. / RATE as f64, true),
        ] {
            let progress = batch
                .iter()
                .filter_map(|event| match event {
                    EngineEvent::Progress {
                        processed_seconds,
                        received_seconds,
                        flush,
                    } => Some((*processed_seconds, *received_seconds, *flush)),
                    _ => None,
                })
                .collect::<Vec<_>>();
            assert_eq!(
                progress,
                vec![(processed, 80_037. / RATE as f64, flush)],
                "{phase} progress follows jointly observed audio; the received EOF tail remains distinct"
            );
        }
    }

    #[test]
    fn buffered_vad_tail_and_language_boundary_reach_publication() {
        let mut engine =
            TestEngine::new(EngineConfig::new("buffered", "en"), FakeModels::english()).unwrap();
        engine.append_pcm(&loud(16_037));
        let events = engine.handle(notice(0, 16_037, 0)).unwrap();
        assert_eq!(engine.ledger.lock().unwrap().end_sample(), 15_872);
        assert!(events.iter().any(|event| matches!(event, EngineEvent::Progress {processed_seconds, received_seconds, ..}
            if *processed_seconds == 15_872.0 / 16_000.0 && *received_seconds == 16_037.0 / 16_000.0)));
        engine
            .handle(LiveCommand::Language {
                language: "de".into(),
                generation: 1,
                start_sample: 16_037,
            })
            .unwrap();
        assert_eq!(engine.book.epoch, 0);
        let events = engine.handle(LiveCommand::Stop).unwrap();
        assert_eq!(engine.ledger.lock().unwrap().end_sample(), 16_037);
        assert_eq!(engine.book.epoch, 1);
        assert!(
            revisions(&events)
                .iter()
                .any(|segment| segment.language.as_deref() == Some("en")
                    && segment.end == 16_037.0 / 16_000.0)
        );
        assert!(engine.models.vad_sessions.lock().unwrap().is_empty());
        assert_eq!(engine.models.diar_sessions.load(Ordering::Relaxed), 0);
    }

    #[test]
    fn model_streams_close_on_partial_open_failed_stop_and_drop() {
        for failure in 0..3 {
            let mut engine =
                TestEngine::new(EngineConfig::new("cleanup", "en"), FakeModels::english()).unwrap();
            let models = engine.models.clone();
            if failure == 0 {
                models.fail_diar_open.store(true, Ordering::Relaxed);
            }
            engine.append_pcm(&loud(512));
            let opened = engine.handle(notice(0, 512, 0));
            if failure == 0 {
                assert!(opened.is_err());
            } else {
                opened.unwrap();
            }
            if failure == 1 {
                models.fail_vad_eof.store(true, Ordering::Relaxed);
                assert!(engine.handle(LiveCommand::Stop).is_err());
            }
            if failure < 2 {
                assert!(models.vad_sessions.lock().unwrap().is_empty());
                assert_eq!(models.diar_sessions.load(Ordering::Relaxed), 0);
            }
            drop(engine);
            assert!(models.vad_sessions.lock().unwrap().is_empty());
            assert_eq!(models.diar_sessions.load(Ordering::Relaxed), 0);
        }
    }

    #[test]
    fn canonical_engine_publishes_a_provisional_revision_then_flush_language_and_stop() {
        let mut engine =
            TestEngine::new(EngineConfig::new("job-1", "en"), FakeModels::english()).unwrap();
        drive(&mut engine, &loud(RATE as usize), "en");
        let mut seen = Vec::new();
        while let Some(events) = engine.poll().unwrap() {
            seen.extend(events);
        }
        seen.extend(
            engine
                .handle(LiveCommand::Flush {
                    request_id: Some("pause-1".into()),
                    through_sample: Some(RATE),
                })
                .unwrap(),
        );
        let published = revisions(&seen);
        let provisional = published.last().unwrap();
        assert_eq!(provisional.text, "hello");
        assert_eq!(provisional.speaker, "speaker_0");
        assert_eq!(provisional.refinement_state.as_deref(), Some("provisional"));
        assert_eq!(provisional.language.as_deref(), Some("en"));
        assert!(seen.iter().any(|event| matches!(
            event,
            EngineEvent::FlushAck { request_id, received_sample, .. }
                if request_id == "pause-1" && *received_sample == RATE
        )));
        seen.extend(
            engine
                .handle(LiveCommand::Language {
                    language: "de".into(),
                    generation: 1,
                    start_sample: RATE,
                })
                .unwrap(),
        );
        assert!(seen.iter().any(|event| matches!(
            event,
            EngineEvent::LanguageRegistered { generation: 1, language, start_sample }
                if language == "de" && *start_sample == RATE
        )));
        seen.extend(engine.handle(LiveCommand::Stop).unwrap());
        assert!(seen.iter().any(|event| matches!(
            event,
            EngineEvent::CaptureFinished { canonical_observed_sample, .. }
                if *canonical_observed_sample == RATE
        )));
    }

    #[test]
    fn saved_auto_accuracy_reprobes_and_preserves_uncertain_or_mixed_anchors() {
        let mut old = sample_segment("Earlier legible English words", 7);
        old.start = 1.;
        old.end = 7.;
        old.canonical_utterance_id = Some(old.id.clone());
        old.language = Some("en".into());
        old.language_detection = Some(serde_json::json!({"mode":"auto","reason":"detected"}));
        old.rest.extend(
            serde_json::from_value::<Map<String, Value>>(serde_json::json!({
                "canonical_state":"sealed","language_mode":"auto","language_epoch":3,
                "canonical_machine_revision":7
            }))
            .unwrap(),
        );
        let probe = |language: &str, probability: f32| {
            Ok(LanguageProbeOut {
                probabilities: vec![probability, 1. - probability],
                codes: vec![
                    language.into(),
                    if language == "en" { "de" } else { "en" }.into(),
                ],
            })
        };
        for case in [
            "single",
            "uncertain",
            "unsupported",
            "mixed",
            "detector_error",
        ] {
            let models = FakeModels::english();
            let scores = match case {
                "single" => vec![probe("de", 0.99), probe("de", 0.99)],
                "uncertain" => vec![probe("de", 0.51), probe("de", 0.51)],
                "unsupported" => vec![probe("ru", 0.99), probe("ru", 0.99)],
                "mixed" => vec![probe("en", 0.99), probe("de", 0.99)],
                "detector_error" => {
                    vec![Err(ChildError::failed("replayed detector failure", false)); 2]
                }
                _ => unreachable!(),
            };
            models.probe_scores.lock().unwrap().extend(scores);
            let result = refine_saved_passage(&models, &old, &loud(6 * RATE as usize), None);
            assert_eq!(
                models.probe_calls.load(Ordering::SeqCst),
                2,
                "{case}: captured Auto must probe again instead of forcing the old detected language"
            );
            let mut document = document_with(old.clone());
            if case == "single" {
                let candidate = result.unwrap();
                assert_eq!(candidate.text, "eins");
                assert_eq!(candidate.language.as_deref(), Some("de"));
                assert_eq!(
                    candidate.language_detection.as_ref().unwrap()["mode"],
                    "auto"
                );
                assert_eq!(candidate.rest["language_mode"], "auto");
                assert_eq!(candidate.rest["language_epoch"], 3);
                let detection = candidate.language_detection.as_ref().unwrap();
                assert_eq!(detection["probes"][0]["start_sample"], RATE);
                assert_eq!(detection["probes"][1]["end_sample"], 7 * RATE);
                assert_eq!(*models.transcription_languages.lock().unwrap(), vec!["de"]);
                assert!(publish_saved_passage(&mut document, &old, &candidate).unwrap());
            } else {
                assert!(
                    result.is_err(),
                    "{case}: preserve previous words if this anchor has no complete single-language replacement"
                );
                assert_eq!(document, document_with(old.clone()));
                if case == "mixed" {
                    assert_eq!(
                        *models.transcription_languages.lock().unwrap(),
                        vec!["en", "de"]
                    );
                } else if case != "uncertain" {
                    assert_eq!(models.transcribe_calls.load(Ordering::SeqCst), 0, "{case}");
                }
            }
        }
    }

    #[test]
    fn saved_canonical_accuracy_keeps_timing_identity_and_blocks_stale_or_incomplete_audio() {
        let (vocabulary, text, scores) = crate::alignment::tests::live_fixture();
        let samples = scores.audio_num_samples as usize;
        let mut models = FakeModels::english();
        models.alignment = Some((text.clone(), scores));
        let mut old = sample_segment("Earlier words", 7);
        old.start = 1.;
        old.end = 1. + samples as f64 / RATE as f64;
        old.canonical_utterance_id = Some(old.id.clone());
        old.audio_revision = Some(11);
        old.language = Some("en".into());
        old.capture_source = Some(
            serde_json::json!({"catalog_sha256":"test-catalog","source_id":"system","source_revision":1}),
        );
        old.rest.extend(serde_json::from_value::<Map<String,Value>>(serde_json::json!({"canonical_state":"sealed","language_epoch":3,"language_generation":3,"language_mode":"en","canonical_machine_revision":7})).unwrap());
        old.speaker_activity = Some(
            serde_json::json!({"audio_revision":11,"observed_end_sample":16000+samples,"regions":[{"start_sample":16000,"end_sample":16000+samples,"speakers":["speaker_0"]}]}),
        );
        let candidate =
            refine_saved_passage(&models, &old, &loud(samples), Some(&vocabulary)).unwrap();
        assert_eq!(candidate.text, text);
        assert_eq!(candidate.machine_revision, Some(8));
        let mut document = document_with(old.clone());
        let mut other_source = old.clone();
        other_source.id = "other-source".into();
        other_source.capture_source.as_mut().unwrap()["source_id"] =
            Value::from("microphone_clean");
        document.segments.push(other_source.clone());
        document
            .speakers
            .insert("speaker_0".into(), "Chosen name".into());
        let mut accepted = document.clone();
        assert!(publish_saved_passage(&mut accepted, &old, &candidate).unwrap());
        assert_eq!(
            accepted.segments.iter().map(|s| &s.id).collect::<Vec<_>>(),
            document.segments.iter().map(|s| &s.id).collect::<Vec<_>>()
        );
        let row = accepted.segments.iter().find(|s| s.id == old.id).unwrap();
        assert_eq!(
            accepted
                .segments
                .iter()
                .find(|s| s.id == other_source.id)
                .unwrap(),
            &other_source
        );
        assert_eq!(accepted.speakers["speaker_0"], "Chosen name");
        assert_eq!(row.machine_revision, Some(8));
        assert_eq!(row.audio_revision, Some(11));
        assert_eq!(row.rest["canonical_machine_revision"], 8);
        assert_eq!(
            row.reading_turn_provenance.as_ref().unwrap()["language_epoch"],
            3
        );
        assert_eq!(
            row.reading_turn_provenance.as_ref().unwrap()["capture_source"],
            old.capture_source.clone().unwrap()
        );
        assert!(row.reading_turns.is_some());
        let serialized: Document =
            serde_json::from_value(serde_json::to_value(&accepted).unwrap()).unwrap();
        assert!(
            crate::export::export(&serialized, "srt")
                .unwrap()
                .body
                .contains("Go")
        );
        for mutation in ["source", "audio", "epoch", "language", "mode", "edit"] {
            let mut changed = document.clone();
            let row = &mut changed.segments[0];
            match mutation {
                "source" => {
                    row.capture_source.as_mut().unwrap()["source_revision"] = Value::from(2)
                }
                "audio" => row.audio_revision = Some(12),
                "epoch" => {
                    row.rest.insert("language_epoch".into(), Value::from(4));
                }
                "language" => row.language = Some("fr".into()),
                "mode" => {
                    row.rest.insert("language_mode".into(), Value::from("fr"));
                }
                "edit" => {
                    row.text = "User correction".into();
                    row.protected_fields = Some(vec!["text".into()]);
                }
                _ => unreachable!(),
            }
            let before = changed.clone();
            assert!(
                !publish_saved_passage(&mut changed, &old, &candidate).unwrap(),
                "{mutation}"
            );
            assert_eq!(changed, before, "{mutation}");
        }
        assert_eq!(models.conditioning.lock().unwrap().last(), Some(&None));
        let mut processed = old.clone();
        processed.capture_source.as_mut().unwrap()["source_id"] = Value::from("microphone_clean");
        processed.capture_source.as_mut().unwrap()["catalog_sha256"] = Value::from("a".repeat(64));
        refine_saved_passage(&models, &processed, &loud(samples), None).unwrap();
        assert_eq!(
            models.conditioning.lock().unwrap().last(),
            Some(&Some(GainPolicy { peak: None }))
        );
        let mut auto = old.clone();
        auto.rest
            .insert("language_mode".into(), Value::from("auto"));
        auto.language = Some("fr".into());
        let automatic =
            refine_saved_passage(&models, &auto, &loud(samples), Some(&vocabulary)).unwrap();
        assert_eq!(automatic.language.as_deref(), Some("en"));
        assert_eq!(
            automatic.language_detection.as_ref().unwrap()["mode"],
            "auto"
        );
        assert_eq!(automatic.rest["reading_alignment"]["language"], "en");
        assert_eq!(
            automatic.reading_turn_provenance.as_ref().unwrap()["language_epoch"],
            3
        );
        assert_eq!(automatic.machine_revision, Some(8));
        assert!(automatic.reading_turns.is_some());
        let mut automatic_document = document_with(auto.clone());
        assert!(publish_saved_passage(&mut automatic_document, &auto, &automatic).unwrap());
        let roundtrip: Document =
            serde_json::from_value(serde_json::to_value(automatic_document).unwrap()).unwrap();
        assert!(
            crate::export::export(&roundtrip, "srt")
                .unwrap()
                .body
                .contains("Go")
        );
        auto.rest.remove("language_mode");
        let calls = models.transcribe_calls.load(Ordering::SeqCst);
        assert!(
            refine_saved_passage(&models, &auto, &loud(samples), None).is_err(),
            "a detected language does not prove the captured setting"
        );
        assert_eq!(models.transcribe_calls.load(Ordering::SeqCst), calls);
        let mut partial = old;
        let pcm = [loud(2 * RATE as usize), vec![0.; RATE as usize], loud(160)].concat();
        partial.end = partial.start + pcm.len() as f64 / RATE as f64;
        let complete = FakeModels::english();
        let candidate = refine_saved_passage(&complete, &partial, &pcm, None).unwrap();
        assert_eq!(candidate.text, "hello hello");
        assert_eq!(
            complete
                .transcription_crops
                .lock()
                .unwrap()
                .last()
                .unwrap()
                .1,
            3744,
            "the 160 speech samples have 3200 canonical context plus their preceding VAD frame tail"
        );
        assert!(candidate.transcription_review.is_none());
        let incomplete = FakeModels::english();
        incomplete.transcription_results.lock().unwrap().extend([
            AsrText {
                text: "hello".into(),
                raw_text: "hello".into(),
                token_count: 1,
            },
            AsrText {
                text: "unfinished tail".into(),
                raw_text: "unfinished tail".into(),
                token_count: 448,
            },
        ]);
        assert!(
            refine_saved_passage(&incomplete, &partial, &pcm, None).is_err(),
            "an incomplete second decode cannot replace whole-anchor words"
        );
        partial.end = partial.start + 160. / RATE as f64;
        assert!(
            refine_saved_passage(&FakeModels::english(), &partial, &loud(160), None).is_err(),
            "admitted speech without sufficient physical context remains unresolved"
        );
    }

    #[test]
    fn sealed_live_timing_uses_independent_ctc_replay_once_at_eof() {
        let (vocabulary, text, scores) = crate::alignment::tests::live_fixture();
        let samples = scores.audio_num_samples as usize;
        let mut models = FakeModels::english();
        models.alignment = Some((text.clone(), scores));
        let mut engine = TestEngine::new(EngineConfig::new("timed-live", "en"), models).unwrap();
        engine.set_word_alignment(Arc::new(vocabulary));
        drive(&mut engine, &loud(samples), "en");
        while engine.poll().unwrap().is_some() {}
        assert_eq!(engine.models.ctc_calls.load(Ordering::SeqCst), 0);
        let events = engine.handle(LiveCommand::Stop).unwrap();
        let rows = revisions(&events);
        let row = *rows.last().unwrap();
        assert_eq!(row.text, text);
        assert_eq!(engine.models.ctc_calls.load(Ordering::SeqCst), 1);
        assert!(row.reading_turns.is_some());
        assert_eq!(
            row.rest["reading_alignment"]["audio_anchor"],
            serde_json::json!({"start_sample":0,"end_sample":samples})
        );
        let mut reading: timing::ReadingSegment =
            serde_json::from_value(serde_json::to_value(row).unwrap()).unwrap();
        reading.installed_provider_identity = Some(crate::alignment::NATIVE_PROVIDER.into());
        assert_eq!(timing::validated_turns(&reading), reading.reading_turns);
        let pcm = engine.read_pcm(0, samples as i64).unwrap();
        let before = engine.models.transcribe_calls.load(Ordering::SeqCst);
        let refined = refine_saved_passage(
            engine.models.as_ref(),
            row,
            &pcm,
            engine.word_alignment.as_deref(),
        )
        .unwrap();
        assert_eq!(
            engine.models.transcribe_calls.load(Ordering::SeqCst),
            before + 1
        );
        assert_eq!(engine.models.ctc_calls.load(Ordering::SeqCst), 1);
        assert_eq!(refined.text, row.text);
        assert_eq!(
            refined.machine_revision,
            row.machine_revision.map(|revision| revision + 1)
        );
        assert_eq!(
            refined.reading_turn_provenance.as_ref().unwrap()["machine_revision"],
            refined.machine_revision.unwrap()
        );
        for protected in [false, true] {
            let mut stale = row.clone();
            if protected {
                stale.protected_fields = Some(vec!["text".into()]);
            } else {
                stale.machine_revision = Some(row.machine_revision.unwrap() + 1);
            }
            let mut document = document_with(stale);
            let before = document.clone();
            assert!(!publish_saved_passage(&mut document, row, &refined).unwrap());
            assert_eq!(document, before);
        }

        let mut current = row.clone();
        current.machine_revision = Some(99);
        current.audio_revision = Some(77);
        current
            .rest
            .insert("canonical_machine_revision".into(), Value::from(88));
        current.rest.insert("language_epoch".into(), Value::from(4));
        current.capture_source =
            Some(serde_json::json!({"source_id":"system","source_revision":2}));
        let activity = current.speaker_activity.as_mut().unwrap();
        activity["audio_revision"] = Value::from(77);
        for region in activity["regions"].as_array_mut().unwrap() {
            region["speakers"] = serde_json::json!(["current-speaker"]);
        }
        let projected = refine_saved_passage(
            engine.models.as_ref(),
            &current,
            &pcm,
            engine.word_alignment.as_deref(),
        )
        .unwrap();
        assert_eq!(engine.models.ctc_calls.load(Ordering::SeqCst), 1);
        assert_eq!(projected.machine_revision, Some(100));
        let provenance = projected.reading_turn_provenance.as_ref().unwrap();
        assert_eq!(provenance["canonical_machine_revision"], 89);
        assert_eq!(provenance["audio_revision"], 77);
        assert_eq!(provenance["language_epoch"], 4);
        assert_eq!(
            provenance["capture_source"],
            current.capture_source.clone().unwrap()
        );
        assert_ne!(projected.reading_turns, row.reading_turns);
        let mut reading: timing::ReadingSegment =
            serde_json::from_value(serde_json::to_value(&projected).unwrap()).unwrap();
        reading.installed_provider_identity = Some(crate::alignment::NATIVE_PROVIDER.into());
        assert_eq!(timing::validated_turns(&reading), reading.reading_turns);

        // A rejected reuse must enter the real CTC call and preserve its fatal
        // cancellation result, rather than publishing stale timing as success.
        for case in [
            "pcm",
            "range",
            "asr_text",
            "language",
            "complete",
            "status",
            "raw_text",
            "text_sha256",
            "audio_anchor",
            "provider_identity_sha256",
            "provider_id",
            "model_sha256",
            "normalization_policy_id",
            "timing_unit_policy_id",
            "frame_calibration_id",
            "score_calibration_id",
            "upstream_normalization_revision",
            "candidate_runtime",
            "alignment_backend",
            "acoustic_aliases",
            "normalization_spans",
            "audio_float32_sha256",
            "missing_cell",
            "missing_word",
            "token_id",
            "cell_text",
            "cell_utf16",
            "word_utf16",
            "frame",
            "receptive_window",
            "score",
            "cell_bounds",
            "word_bounds",
            "independent",
            "protected",
        ] {
            let mut old = row.clone();
            let mut supplied = pcm.clone();
            let mut models = FakeModels::english();
            models.alignment = engine.models.alignment.clone();
            models.ctc_error = Some(ChildError::Cancelled);
            match case {
                "pcm" => supplied[0] += 0.001,
                "range" => {
                    old.start += 1.;
                    old.end += 1.;
                }
                "asr_text" => models.alignment.as_mut().unwrap().0.push('!'),
                "language" => {
                    old.rest.insert("language_mode".into(), Value::from("fr"));
                }
                "protected" => old.protected_fields = Some(vec!["text".into()]),
                _ => {
                    let alignment = old.rest.get_mut("reading_alignment").unwrap();
                    match case {
                        "complete" => alignment[case] = Value::from(false),
                        "missing_cell" => {
                            alignment["characters"].as_array_mut().unwrap().remove(1);
                        }
                        "missing_word" => {
                            alignment["words"].as_array_mut().unwrap().remove(0);
                        }
                        "token_id" => alignment["characters"][0]["token_id"] = Value::from(-1),
                        "cell_text" => alignment["characters"][0]["text"] = Value::from("X"),
                        "cell_utf16" => {
                            alignment["characters"][0]["start_utf16"] = Value::from(999)
                        }
                        "word_utf16" => alignment["words"][0]["start_utf16"] = Value::from(999),
                        "frame" => alignment["characters"][0]["frame"] = Value::from(-1),
                        "receptive_window" => {
                            supplied.truncate(10_240);
                            old.end = old.start + supplied.len() as f64 / RATE as f64;
                            alignment["audio_anchor"]["end_sample"] = Value::from(10_240);
                            let bytes: Vec<_> =
                                supplied.iter().flat_map(|v| v.to_le_bytes()).collect();
                            alignment["audio_float32_sha256"] = Value::from(sha256_hex(&bytes));
                            let cell = alignment["characters"]
                                .as_array_mut()
                                .unwrap()
                                .last_mut()
                                .unwrap();
                            cell["frame"] = Value::from(31);
                            cell["start_sample"] = Value::from(9_920);
                            cell["end_sample"] = Value::from(10_240);
                            let word = alignment["words"]
                                .as_array_mut()
                                .unwrap()
                                .last_mut()
                                .unwrap();
                            // This final raw unit has exactly one acoustic cell.
                            word["start_sample"] = Value::from(9_920);
                            word["end_sample"] = Value::from(10_240);
                            assert!(context::complete_alignment(TailPart {
                                request: DecodeRequest {
                                    utterance_id: old.canonical_utterance_id.clone().unwrap(),
                                    machine_revision: old.machine_revision.unwrap(),
                                    audio_revision: old.audio_revision.unwrap_or(0),
                                    language_epoch: row.rest["language_epoch"]
                                        .as_i64()
                                        .unwrap_or(0),
                                    core_start_sample: 0,
                                    core_end_sample: 10_240,
                                    start_sample: 0,
                                    end_sample: 10_240,
                                },
                                text: old.text.clone(),
                                complete: true,
                                alignment: Some(alignment.clone()),
                            }));
                        }
                        "score" => alignment["characters"][0]["log_probability"] = Value::from(1.),
                        "cell_bounds" => {
                            alignment["characters"][0]["start_sample"] = Value::from(-1)
                        }
                        "word_bounds" => alignment["words"][0]["end_sample"] = Value::from(1),
                        "independent" => {
                            alignment["characters"][0]["independent_acoustic_timestamp"] =
                                Value::from(false)
                        }
                        _ => alignment[case] = Value::from("stale"),
                    }
                }
            }
            let result =
                refine_saved_passage(&models, &old, &supplied, engine.word_alignment.as_deref());
            if case == "protected" {
                assert!(result.is_err());
                assert_eq!(models.transcribe_calls.load(Ordering::SeqCst), 0);
                assert_eq!(models.ctc_calls.load(Ordering::SeqCst), 0);
            } else {
                assert_eq!(
                    result.unwrap_err(),
                    LiveError::Child(ChildError::Cancelled),
                    "{case}"
                );
                assert_eq!(models.transcribe_calls.load(Ordering::SeqCst), 1, "{case}");
                assert_eq!(models.ctc_calls.load(Ordering::SeqCst), 1, "{case}");
            }
        }
    }

    #[test]
    fn optional_live_timing_failure_preserves_words_but_child_loss_invalidates_streams() {
        for error in [
            ChildError::failed("CTC replay failure", false),
            ChildError::Unimplemented,
            ChildError::ProcessLost {
                message: "replay child exited".into(),
            },
            ChildError::Timeout,
            ChildError::Cancelled,
        ] {
            let (vocabulary, text, scores) = crate::alignment::tests::live_fixture();
            let samples = scores.audio_num_samples as usize;
            let mut models = FakeModels::english();
            models.alignment = Some((text.clone(), scores));
            models.ctc_error = Some(error.clone());
            let mut engine =
                TestEngine::new(EngineConfig::new("optional-timing", "en"), models).unwrap();
            engine.set_word_alignment(Arc::new(vocabulary));
            drive(&mut engine, &loud(samples), "en");
            while engine.poll().unwrap().is_some() {}
            let result = engine.handle(LiveCommand::Stop);
            if matches!(error, ChildError::Failed { .. } | ChildError::Unimplemented) {
                let events = result.expect("optional timing failure must not suppress saved words");
                let rows = revisions(&events);
                let row = rows.last().unwrap();
                assert_eq!(row.text, text);
                assert_eq!(row.rest["reading_alignment"]["status"], "unavailable");
                assert!(row.reading_turn_provenance.is_none());
                assert!(
                    events
                        .iter()
                        .any(|e| matches!(e, EngineEvent::CaptureFinished { .. }))
                );
            } else {
                assert_eq!(result.unwrap_err(), LiveError::Child(error));
            }
            assert_eq!(engine.models.diar_sessions.load(Ordering::SeqCst), 0);
            assert!(engine.models.vad_sessions.lock().unwrap().is_empty());
        }
    }

    #[test]
    fn sealed_live_observations_keep_verified_solo_crops_and_unknown_tail() {
        let mut engine =
            TestEngine::new(EngineConfig::new("evidence", "en"), FakeModels::english()).unwrap();
        drive(&mut engine, &loud((4 * RATE + 37) as usize), "en");
        while engine.poll().unwrap().is_some() {}
        let events = engine.handle(LiveCommand::Stop).unwrap();
        let rows = revisions(&events);
        let row = rows.last().unwrap();
        let activity = row.speaker_activity.as_ref().unwrap();
        assert_eq!(
            activity["regions"][0]["speakers"],
            serde_json::json!(["speaker_0"])
        );
        assert_eq!(activity["observed_end_sample"], 4 * RATE);
        assert_eq!(row.acoustic_evidence.as_ref().unwrap()["complete"], true);
        let audio: crate::voice::AudioSegment =
            serde_json::from_value(serde_json::to_value(row).unwrap()).unwrap();
        assert_eq!(
            crate::voice::speaker_audio_ranges(&audio, "speaker_0"),
            vec![(0., 4.)]
        );
        let mut old = serde_json::to_value(row).unwrap();
        old["speaker_activity"]
            .as_object_mut()
            .unwrap()
            .remove("regions");
        old.as_object_mut().unwrap().remove("voice_eligible");
        let old: crate::voice::AudioSegment = serde_json::from_value(old).unwrap();
        assert!(crate::voice::speaker_audio_ranges(&old, "speaker_0").is_empty());
        let mut overlap = audio.clone();
        overlap.speaker_candidates = Some(vec!["speaker_0".into(), "speaker_1".into()]);
        assert!(crate::voice::speaker_audio_ranges(&overlap, "speaker_0").is_empty());
    }

    #[test]
    fn unwired_live_timing_never_probes_or_calls_the_acoustic_model() {
        let mut english =
            TestEngine::new(EngineConfig::new("job-en", "en"), FakeModels::english()).unwrap();
        drive(&mut english, &loud((21 * RATE) as usize), "en");
        let mut events = Vec::new();
        while let Some(batch) = english.poll().unwrap() {
            events.extend(batch);
        }
        let published = revisions(&events);
        let last = published.last().unwrap();
        assert_eq!(last.text, "hello");
        assert_eq!(last.refinement_state.as_deref(), Some("provisional"));
        assert_eq!(english.models.ctc_calls.load(Ordering::SeqCst), 0);

        let mut german =
            TestEngine::new(EngineConfig::new("job-de", "de"), FakeModels::english()).unwrap();
        drive(&mut german, &loud((25 * RATE) as usize), "de");
        let mut events = Vec::new();
        while let Some(batch) = german.poll().unwrap() {
            events.extend(batch);
        }
        let published = revisions(&events);
        let last = published.last().unwrap();
        assert_eq!(last.text, "eins eins");
        assert_eq!(last.language.as_deref(), Some("de"));
        assert_eq!(german.models.ctc_calls.load(Ordering::SeqCst), 0);
        assert!(german.models.transcribe_calls.load(Ordering::SeqCst) >= 2);
    }

    #[test]
    fn recovery_keeps_independent_source_cutoffs() {
        let mut inputs = [TestRecording::new(), TestRecording::new()];
        for input in &mut inputs {
            input
                .writer
                .write_samples(&loud((3 * RATE + 37) as usize))
                .unwrap();
        }
        let mut config = EngineConfig::new("source-recovery", "en");
        config.processed_microphone_contract = Some("a".repeat(64));
        config.capture_catalog_sha256 = Some("b".repeat(64));
        config.recovery = Some(StreamRecovery {
            generation: 1,
            start_sample: RATE,
            through_sample: 3 * RATE + 37,
            language_epoch: 0,
            source_boundaries: BTreeMap::from([
                (
                    "microphone_clean".into(),
                    RecoveryBoundary {
                        start_sample: 2 * RATE,
                        language_epoch: 0,
                    },
                ),
                (
                    "system".into(),
                    RecoveryBoundary {
                        start_sample: RATE,
                        language_epoch: 0,
                    },
                ),
            ]),
        });
        let mut runtime = SourceRuntime::new(
            config,
            FakeModels::english(),
            inputs
                .each_ref()
                .map(|r| RecordingReader::open(&r.path).unwrap()),
        )
        .unwrap();
        let mut events = runtime.handle(notice(0, 3 * RATE + 37, 0)).unwrap();
        events.extend(runtime.handle(LiveCommand::Stop).unwrap());
        let mut starts = BTreeMap::new();
        for candidate in revisions(&events) {
            let source = candidate.capture_source.as_ref().unwrap()["source_id"]
                .as_str()
                .unwrap()
                .to_owned();
            let start = candidate.rest["start_sample"].as_i64().unwrap();
            starts
                .entry(source)
                .and_modify(|old: &mut i64| *old = (*old).min(start))
                .or_insert(start);
        }
        assert_eq!(
            starts,
            BTreeMap::from([
                ("microphone_clean".into(), 2 * RATE),
                ("system".into(), RATE)
            ])
        );
    }

    #[test]
    fn two_sources_prefix_speaker_ids_on_one_clock() {
        let mut inputs = [TestRecording::new(), TestRecording::new()];
        let mut config = EngineConfig::new("job-src", "en");
        config.processed_microphone_contract = Some("a".repeat(64));
        let mut runtime = SourceRuntime::new(
            config,
            FakeModels::english(),
            inputs
                .each_ref()
                .map(|input| RecordingReader::open(&input.path).unwrap()),
        )
        .unwrap();
        let speech = loud(RATE as usize);
        let silence = quiet((RATE / 2) as usize);
        for input in &mut inputs {
            input.writer.write_samples(&speech).unwrap();
            input.writer.write_samples(&silence).unwrap();
        }
        runtime.push(notice(0, RATE, 0)).unwrap();
        runtime.push(notice(RATE, RATE + RATE / 2, 0)).unwrap();
        let mut speakers = Vec::new();
        let mut identities = std::collections::BTreeSet::new();
        while let Some(events) = runtime.poll().unwrap() {
            for event in events {
                if let EngineEvent::CanonicalRevision { candidate, .. } = event
                    && candidate.text == "hello"
                {
                    identities.insert(candidate.id.clone());
                    speakers.push(candidate.speaker);
                }
            }
        }
        for event in runtime.handle(LiveCommand::Stop).unwrap() {
            if let EngineEvent::CanonicalRevision { candidate, .. } = event
                && candidate.text == "hello"
            {
                identities.insert(candidate.id.clone());
                speakers.push(candidate.speaker);
            }
        }
        assert_eq!(
            speakers,
            vec!["speaker_0".to_string(), "speaker_32".to_string()]
        );
        assert_eq!(
            identities.len(),
            2,
            "source-local utterances must coexist in one document"
        );
        assert_eq!(
            *runtime.engines[0].models.conditioning.lock().unwrap(),
            [Some(GainPolicy { peak: None }), None]
        );
        assert_eq!(
            runtime.engines[0].ledger.lock().unwrap().gain_mode(),
            speech::GainMode::Unity
        );
        assert_eq!(
            runtime.engines[0].ledger.lock().unwrap().input_policy(),
            speech::PRODUCTION_UNITY_POLICY
        );
        assert_eq!(
            runtime.engines[1].ledger.lock().unwrap().gain_mode(),
            speech::GainMode::QuietSpeech
        );
        assert_eq!(speaker_id("system", "speaker_0").unwrap(), "speaker_32");
        assert!(speaker_id("system", "speaker_32").is_err());
    }

    #[test]
    fn both_saved_sources_are_checked_before_either_model_session_advances() {
        let mut inputs = [TestRecording::new(), TestRecording::new()];
        let mut runtime = SourceRuntime::new(
            EngineConfig::new("missing-source", "en"),
            FakeModels::english(),
            inputs
                .each_ref()
                .map(|input| RecordingReader::open(&input.path).unwrap()),
        )
        .unwrap();
        inputs[0].writer.write_samples(&loud(512)).unwrap();
        inputs[1].writer.write_samples(&loud(511)).unwrap();
        assert!(runtime.handle(notice(0, 512, 0)).is_err());
        for engine in &runtime.engines {
            assert_eq!(engine.received, 0);
            assert!(engine.models.conditioning.lock().unwrap().is_empty());
        }
        inputs[1].writer.write_samples(&[-0.1]).unwrap();
        runtime.handle(notice(0, 512, 0)).unwrap();
        runtime.handle(LiveCommand::Stop).unwrap();
        assert_eq!(runtime.received, 512);
    }

    #[test]
    fn utterances_open_on_speech_and_seal_after_silence() {
        let mut book = UtteranceBook::new("job").unwrap();
        let mut ledger = SpeechLedger::new(0).unwrap();
        for (offset, probability) in [0.9, 0.9, 0.1, 0.1].into_iter().enumerate() {
            let start = offset as i64 * 8_000;
            let pcm: Vec<f32> = (0..8_000)
                .map(|index| if index % 2 == 0 { 0.2 } else { -0.1 })
                .collect();
            let mut cursor = start;
            for chunk in pcm.chunks(speech::FRAME_SAMPLES) {
                ledger
                    .append_pcm(cursor, chunk, probability, probability)
                    .unwrap();
                cursor += chunk.len() as i64;
            }
        }
        let admission = ledger.admission(0, 32_000).unwrap();
        book.observe(&admission).unwrap();
        assert_eq!(book.order.len(), 1);
        let row = book.rows.values().next().unwrap();
        assert_eq!(row.state, UtteranceState::Sealed);
        assert!(row.end_sample <= 16_000 + UTTERANCE_CONTEXT_SAMPLES);
    }
}
