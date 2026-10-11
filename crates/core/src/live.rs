//! Canonical live engine.
//!
//! Inbox coalescing, utterance admission, short routed revisions, the English
//! authoritative tail, the non-English core plan, and compare-and-swap window
//! reconciliation. Model calls go through [`crate::models::Models`]. This module
//! reads bounded crops from the retained WAV; it does not load MLX.

use std::collections::{BTreeMap, HashMap, VecDeque};
use std::path::Path;
use std::sync::{Arc, Mutex};
use std::time::Instant;

use serde_json::{Map, Value};
use sha2::{Digest, Sha256};
use thiserror::Error;

use crate::audio::RecordingReader;
use crate::document::{Document, Segment};
use crate::error::ChildError;
use crate::ipc::{DiarPreset, GainPolicy, Handle, Turn, VadFeedOut};
use crate::language::{
    AcousticEvidence, AsrOutput, AudioState, LanguageDetector, Passage, Recognizer,
    SpeechTranscriber, TranscribeOptions,
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
const FAST_MAX_SAMPLES: i64 = 392_000;
const REFINE_CORE_SAMPLES: i64 = 18 * RATE;
const CANCEL_MEMORY: usize = 16;
const MAX_CONTROL_MESSAGES: usize = 128;

#[derive(Debug, Error, Clone, PartialEq, Eq)]
pub enum LiveError {
    #[error("{0}")]
    Message(&'static str),
    #[error("{0}")]
    Detail(String),
}

impl From<SpeechError> for LiveError {
    fn from(error: SpeechError) -> Self {
        Self::Detail(error.to_string())
    }
}

impl From<ChildError> for LiveError {
    fn from(error: ChildError) -> Self {
        Self::Detail(error.to_string())
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

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RefineCommand {
    pub operation_id: String,
    pub language_epoch: i64,
    pub canonical_sealed: bool,
}

#[derive(Debug, Clone, PartialEq)]
pub enum LiveCommand {
    Audio(AudioNotice),
    Refine(RefineCommand),
    CancelRefinement {
        operation_id: String,
    },
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

/// Coalesces contiguous audio notices and remembers in-flight cancellations.
pub struct Inbox {
    audio_batch_samples: i64,
    refinement_interval_seconds: i64,
    last_refinement_at: Option<Instant>,
    active_refinement: Option<(String, i64)>,
    capture_finished: bool,
    last_taken: Option<&'static str>,
    messages: VecDeque<LiveCommand>,
    refinement: Option<RefineCommand>,
    shutdown: bool,
    historical: bool,
    latest_epoch: i64,
    cancelled: Vec<String>,
    received: i64,
}

impl Inbox {
    pub fn new(
        audio_batch_seconds: i64,
        refinement_interval_seconds: i64,
    ) -> Result<Self, LiveError> {
        if !(1..=6).contains(&audio_batch_seconds) {
            return Err(LiveError::Message("Invalid bounded audio drain cadence."));
        }
        if !(0..=60).contains(&refinement_interval_seconds) {
            return Err(LiveError::Message(
                "Invalid bounded background refinement interval.",
            ));
        }
        Ok(Self {
            audio_batch_samples: audio_batch_seconds * RATE,
            refinement_interval_seconds,
            last_refinement_at: None,
            active_refinement: None,
            capture_finished: false,
            last_taken: None,
            messages: VecDeque::new(),
            refinement: None,
            shutdown: false,
            historical: false,
            latest_epoch: 0,
            cancelled: Vec::new(),
            received: 0,
        })
    }

    pub fn set_historical(&mut self, historical: bool) {
        self.historical = historical;
    }

    pub fn set_latest_epoch(&mut self, epoch: i64) {
        self.latest_epoch = epoch;
    }

    fn remember_cancel(&mut self, operation_id: &str) {
        let mut keep = vec![operation_id.to_string()];
        if let Some((active, _)) = &self.active_refinement
            && self.cancelled.iter().any(|id| id == active)
        {
            keep.push(active.clone());
        }
        let mut retained: Vec<String> = self
            .cancelled
            .iter()
            .filter(|id| !keep.iter().any(|kept| kept == *id))
            .cloned()
            .collect();
        retained.sort();
        let room = CANCEL_MEMORY.saturating_sub(keep.len());
        if retained.len() > room {
            retained = retained.split_off(retained.len() - room);
        }
        retained.extend(keep);
        retained.sort();
        retained.dedup();
        self.cancelled = retained;
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
            LiveCommand::Refine(command) => {
                if self.refinement.is_some() {
                    return Err(LiveError::Message(
                        "Only one dispatched refinement is allowed.",
                    ));
                }
                self.refinement = Some(command);
            }
            LiveCommand::CancelRefinement { operation_id } => {
                if !operation_id.is_empty() {
                    self.remember_cancel(&operation_id);
                }
                if self
                    .refinement
                    .as_ref()
                    .is_some_and(|pending| pending.operation_id == operation_id)
                {
                    self.refinement = None;
                }
            }
            LiveCommand::Language {
                language,
                generation,
                start_sample,
            } => {
                self.latest_epoch = generation;
                if !self.historical {
                    let stale = self
                        .active_refinement
                        .as_ref()
                        .filter(|(_, epoch)| *epoch != self.latest_epoch)
                        .map(|(operation, _)| operation.clone());
                    if let Some(operation) = stale {
                        self.remember_cancel(&operation);
                    }
                }
                if self
                    .refinement
                    .as_ref()
                    .is_some_and(|pending| pending.language_epoch != self.latest_epoch)
                {
                    self.refinement = None;
                }
                self.messages.push_back(LiveCommand::Language {
                    language,
                    generation,
                    start_sample,
                });
            }
            LiveCommand::Shutdown => self.shutdown = true,
            LiveCommand::Flush { .. } | LiveCommand::Stop => self.messages.push_back(message),
        }
        if self.messages.len() > MAX_CONTROL_MESSAGES {
            return Err(LiveError::Message(
                "Too many pending language/control boundaries.",
            ));
        }
        Ok(())
    }

    pub fn cancelled_request(&self, command: &RefineCommand) -> bool {
        let historical = self.historical
            || (command.canonical_sealed && command.language_epoch <= self.latest_epoch);
        self.shutdown
            || self.cancelled.iter().any(|id| id == &command.operation_id)
            || (!historical && command.language_epoch != self.latest_epoch)
    }

    /// Contiguous audio before the next control supersedes an open tail.
    pub fn newer_audio_queued(&self, through_sample: i64, epoch: i64) -> bool {
        match self.messages.front() {
            Some(LiveCommand::Audio(notice)) => {
                notice.language_epoch == epoch
                    && notice.start_sample == through_sample
                    && notice.end_sample > through_sample
            }
            _ => false,
        }
    }

    pub fn finish_refinement(&mut self, operation_id: &str) {
        if self
            .active_refinement
            .as_ref()
            .is_some_and(|(active, _)| active == operation_id)
        {
            self.active_refinement = None;
        }
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

    pub fn pending_refinement(&self) -> Option<&RefineCommand> {
        self.refinement.as_ref()
    }

    /// Non-blocking take. `None` means the queue is idle.
    pub fn take(&mut self) -> Result<Option<LiveCommand>, LiveError> {
        if self.shutdown {
            return Ok(Some(LiveCommand::Shutdown));
        }
        let refinement_due = self.refinement.as_ref().is_some_and(|_| {
            self.refinement_interval_seconds == 0
                || self.capture_finished
                || self.historical
                || self.last_refinement_at.is_none_or(|last| {
                    last.elapsed().as_secs() >= self.refinement_interval_seconds as u64
                })
        });
        let audio_waiting = matches!(self.messages.front(), Some(LiveCommand::Audio(_)));
        if refinement_due
            && (self.messages.is_empty() || (audio_waiting && self.last_taken == Some("audio")))
        {
            let result = self.refinement.take().expect("refinement is due");
            self.last_refinement_at = Some(Instant::now());
            self.active_refinement = Some((result.operation_id.clone(), result.language_epoch));
            self.last_taken = Some("refine");
            return Ok(Some(LiveCommand::Refine(result)));
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
        self.last_taken = Some(match &result {
            LiveCommand::Audio(_) => "audio",
            LiveCommand::Refine(_) => "refine",
            LiveCommand::Language { .. } => "language",
            LiveCommand::Flush { .. } => "flush",
            LiveCommand::Stop => "stop",
            LiveCommand::Shutdown => "shutdown",
            LiveCommand::CancelRefinement { .. } => "cancel",
        });
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
struct MachineVersion {
    text: String,
    complete: bool,
    reason: Option<String>,
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
    machine_versions: Vec<MachineVersion>,
    refinement_state: String,
    language: Option<String>,
    review: bool,
    audio_state: Option<AudioState>,
    transcription_review: bool,
    canonical_unresolved: Option<String>,
    speaker_candidates: Vec<String>,
    speaker_activity: Option<timing::SpeakerActivity>,
    text_audio_anchor: Option<timing::AudioAnchor>,
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
        row.end_sample = end;
        row.audio_revision += 1;
        row.speaker_activity = None;
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
                        machine_versions: Vec::new(),
                        refinement_state: "provisional".to_string(),
                        language: None,
                        review: false,
                        audio_state: None,
                        transcription_review: false,
                        canonical_unresolved: None,
                        speaker_candidates: Vec::new(),
                        speaker_activity: None,
                        text_audio_anchor: None,
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
        row.machine_versions.push(MachineVersion {
            text: text.to_string(),
            complete,
            reason: None,
        });
        if complete && !row.protected_fields.iter().any(|field| field == "text") {
            row.text = text.to_string();
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
    Ok(base(Some("unavailable_boundary_timing"), false, None))
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

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct OwnershipWindow {
    pub id: String,
    pub kind: Option<String>,
    pub start_sample: i64,
    pub end_sample: i64,
}

#[derive(Debug, Clone, PartialEq)]
pub struct ReconcileOutcome {
    pub document: Document,
    pub previous_segments: Vec<Segment>,
    pub protected_ids: Vec<String>,
    pub changed: bool,
}

fn segment_bounds(segment: &Segment) -> Result<(i64, i64), LiveError> {
    if !segment.start.is_finite() || !segment.end.is_finite() || segment.start < 0.0 {
        return Err(LiveError::Message(
            "Audio times must be finite and nonnegative.",
        ));
    }
    Ok((round_samples(segment.start), round_samples(segment.end)))
}

fn intersects(left: (i64, i64), right: (i64, i64)) -> bool {
    left.0 < right.1 && right.0 < left.1
}

/// Source snapshot of the fields a window is allowed to replace.
pub fn segment_version(segment: &Segment) -> String {
    let value = serde_json::json!({
        "end": segment.end,
        "id": segment.id,
        "machine_revision": segment.machine_revision,
        "protected_fields": segment.protected_fields,
        "speaker": segment.speaker,
        "start": segment.start,
        "text": segment.text,
    });
    sha256_hex(serde_json::to_string(&value).unwrap_or_default().as_bytes())
}

fn protected_segment(segment: &Segment) -> bool {
    segment
        .protected_fields
        .as_ref()
        .is_some_and(|fields| !fields.is_empty())
}

fn transcription_blocked(segment: &Segment) -> bool {
    segment.transcription_review.is_some()
}

fn successful_coverage(segment: &Segment) -> bool {
    !segment.text.trim().is_empty() && !transcription_blocked(segment)
}

fn covers(old: &Segment, replacements: &[&Segment]) -> Result<bool, LiveError> {
    let (old_start, old_end) = segment_bounds(old)?;
    let mut intervals: Vec<(i64, i64)> = replacements
        .iter()
        .filter(|candidate| {
            segment_bounds(candidate)
                .ok()
                .is_some_and(|bounds| intersects(bounds, (old_start, old_end)))
                && successful_coverage(candidate)
        })
        .map(|candidate| segment_bounds(candidate).expect("bounds just checked"))
        .collect();
    intervals.sort_unstable();
    let mut cursor = old_start;
    for (start, end) in intervals {
        if start > cursor {
            break;
        }
        cursor = cursor.max(end);
    }
    Ok(cursor >= old_end)
}

/// Compare-and-swap a refinement window. Protected rows and stale versions stay put.
///
/// Human edits reach `protected_fields` through [`crate::edit::edit_passage`].
pub fn reconcile_window(
    document: &Document,
    window: &OwnershipWindow,
    expected: &BTreeMap<String, String>,
    candidates: &[Segment],
    speakers: Option<&BTreeMap<String, String>>,
) -> Result<ReconcileOutcome, LiveError> {
    let owned = (window.start_sample, window.end_sample);
    let maximum = if window.kind.as_deref() == Some("fast_tail") {
        FAST_MAX_SAMPLES
    } else {
        REFINE_CORE_SAMPLES
    };
    if owned.0 < 0 || owned.1 <= owned.0 || owned.1 - owned.0 > maximum {
        return Err(LiveError::Message("Invalid refinement ownership window."));
    }
    let previous: Vec<Segment> = document
        .segments
        .iter()
        .filter(|segment| {
            segment_bounds(segment)
                .ok()
                .is_some_and(|bounds| intersects(bounds, owned))
        })
        .cloned()
        .collect();
    let mut proposed = candidates.to_vec();
    for candidate in &proposed {
        let (start, end) = segment_bounds(candidate)?;
        if start < owned.0 || end <= start || end > owned.1 {
            return Err(LiveError::Message(
                "Refinement text must belong entirely to its audio window.",
            ));
        }
    }
    proposed.sort_by_key(|segment| segment_bounds(segment).unwrap_or((0, 0)));
    if proposed.windows(2).any(|pair| {
        segment_bounds(&pair[0]).unwrap_or((0, 0)).1 > segment_bounds(&pair[1]).unwrap_or((0, 0)).0
    }) {
        return Err(LiveError::Message(
            "Refinement candidates must not duplicate overlapping audio.",
        ));
    }
    let mut blocked: Vec<Segment> = previous
        .iter()
        .filter(|segment| {
            let bounds = segment_bounds(segment).unwrap_or((0, 0));
            protected_segment(segment)
                || expected.get(&segment.id).map(String::as_str)
                    != Some(segment_version(segment).as_str())
                || bounds.0 < owned.0
                || bounds.1 > owned.1
                || bounds.1 <= bounds.0
        })
        .cloned()
        .collect();
    for old in &previous {
        if !old.text.trim().is_empty() && !covers(old, &proposed.iter().collect::<Vec<_>>())? {
            blocked.push(old.clone());
        }
    }
    let mut blocked_ids: Vec<String> = blocked.iter().map(|segment| segment.id.clone()).collect();
    let accepted = loop {
        let accepted: Vec<Segment> = proposed
            .iter()
            .filter(|candidate| {
                !blocked.iter().any(|blocked| {
                    intersects(
                        segment_bounds(candidate).unwrap_or((0, 0)),
                        segment_bounds(blocked).unwrap_or((0, 0)),
                    )
                })
            })
            .cloned()
            .collect();
        let newly: Vec<Segment> = previous
            .iter()
            .filter(|segment| {
                !blocked_ids.iter().any(|id| id == &segment.id)
                    && !segment.text.trim().is_empty()
                    && !covers(segment, &accepted.iter().collect::<Vec<_>>()).unwrap_or(false)
            })
            .cloned()
            .collect();
        if newly.is_empty() {
            break accepted;
        }
        blocked_ids.extend(newly.iter().map(|segment| segment.id.clone()));
        blocked.extend(newly);
    };
    let removed: Vec<String> = previous
        .iter()
        .filter(|segment| !blocked_ids.iter().any(|id| id == &segment.id))
        .map(|segment| segment.id.clone())
        .collect();
    let mut output = document.clone();
    output
        .segments
        .retain(|segment| !removed.iter().any(|id| id == &segment.id));
    let mut used: Vec<String> = output
        .segments
        .iter()
        .map(|segment| segment.id.clone())
        .collect();
    let old_ids: Vec<String> = document
        .segments
        .iter()
        .map(|segment| segment.id.clone())
        .collect();
    let fast = window.kind.as_deref() == Some("fast_tail");
    for candidate in &accepted {
        let mut eligible: Vec<&Segment> = previous
            .iter()
            .filter(|segment| {
                removed.iter().any(|id| id == &segment.id)
                    && !used.iter().any(|id| id == &segment.id)
                    && intersects(
                        segment_bounds(segment).unwrap_or((0, 0)),
                        segment_bounds(candidate).unwrap_or((0, 0)),
                    )
            })
            .collect();
        eligible.sort_by(|left, right| {
            let overlap = |segment: &&Segment| {
                let (start, end) = segment_bounds(segment).unwrap_or((0, 0));
                let (candidate_start, candidate_end) = segment_bounds(candidate).unwrap_or((0, 0));
                (end.min(candidate_end) - start.max(candidate_start)).max(0)
            };
            overlap(right).cmp(&overlap(left))
        });
        let old = eligible.first().copied();
        let sid = old.map(|segment| segment.id.clone()).unwrap_or_else(|| {
            let (start, end) = segment_bounds(candidate).unwrap_or((0, 0));
            format!(
                "refined-{}",
                &sha256_hex(format!("{start}:{end}").as_bytes())[..24]
            )
        });
        if used.iter().any(|id| id == &sid)
            || (old.is_none() && old_ids.iter().any(|id| id == &sid))
        {
            return Err(LiveError::Message("Refinement ID collision."));
        }
        let mut written = candidate.clone();
        written.id = sid.clone();
        written.machine_revision = Some(
            old.and_then(|segment| segment.machine_revision)
                .unwrap_or(0)
                + 1,
        );
        if !fast {
            written.refinement_state = Some(if successful_coverage(&written) {
                "refined".to_string()
            } else {
                "unresolved".to_string()
            });
            written
                .rest
                .insert("finalized".to_string(), Value::Bool(true));
        }
        let (start, end) = segment_bounds(&written)?;
        written.rest.insert(
            "audio_anchor".to_string(),
            serde_json::json!({"start_sample": start, "end_sample": end}),
        );
        written.rest.insert(
            "refinement_window".to_string(),
            Value::String(window.id.clone()),
        );
        output.segments.push(written);
        used.push(sid);
    }
    if let Some(speakers) = speakers {
        for (key, name) in speakers {
            output
                .speakers
                .entry(key.clone())
                .or_insert_with(|| name.clone());
        }
    }
    output.segments.sort_by(|left, right| {
        left.start
            .total_cmp(&right.start)
            .then(left.end.total_cmp(&right.end))
    });
    blocked_ids.sort();
    blocked_ids.dedup();
    let changed = output != *document;
    Ok(ReconcileOutcome {
        document: output,
        previous_segments: previous,
        protected_ids: blocked_ids,
        changed,
    })
}

/// Keep fields [`crate::edit::edit_passage`] marked protected when a canonical candidate arrives.
pub fn apply_canonical_candidate(document: &Document, candidate: &Segment) -> Document {
    let mut document = document.clone();
    if let Some(prior) = document
        .segments
        .iter()
        .find(|segment| segment.id == candidate.id)
        .cloned()
    {
        let mut row = candidate.clone();
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
        if row.text.trim().is_empty() && !prior.text.trim().is_empty() {
            row.text = prior.text.clone();
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
    /// Set only after the host verifies the processed microphone's source-bound receipt.
    pub processed_microphone_contract: Option<String>,
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
    RefinementResult {
        operation_id: String,
        cancelled: bool,
        fast_sequence: Option<i64>,
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

struct ModelDetector<M: Models>(Arc<M>);

impl<M: Models> LanguageDetector for ModelDetector<M> {
    fn detect(&mut self, pcm: &[f32]) -> Result<Vec<(String, f64)>, crate::language::RouteError> {
        let probe = self
            .0
            .language_probe(pcm)
            .map_err(|_| crate::language::RouteError::Detector)?;
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
            .map_err(|_| crate::language::RouteError::Recognition)?;
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

struct CorePart {
    text: String,
    complete: bool,
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
    last_decoded: HashMap<String, (i64, i64)>,
    published: HashMap<String, Segment>,
    next_utterance: usize,
    vad: Option<Handle>,
    diar: Option<Handle>,
    ledger: Arc<Mutex<SpeechLedger>>,
    timing_apis: bool,
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
        // shortcut: enable after live publication is wired to the supplied-text adapter.
        let timing_apis = false;
        let namespace = if let Some(source) = &config.source_id {
            if !SOURCE_IDS.contains(&source.as_str()) {
                return Err(LiveError::Message("Unknown capture input."));
            }
            format!("{}:{source}", config.job_id)
        } else {
            config.job_id.clone()
        };
        let book = UtteranceBook::new(namespace)?;
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
            timing_apis,
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
        let inbox = Arc::new(Mutex::new(Inbox::new(batch, 0)?));
        let recording =
            RecordingReader::open(recording).map_err(|e| LiveError::Detail(e.to_string()))?;
        Self::new(config, models, inbox, recording)
    }

    pub fn inbox(&self) -> Arc<Mutex<Inbox>> {
        Arc::clone(&self.inbox)
    }

    pub fn timing_apis(&self) -> bool {
        self.timing_apis
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
            LiveCommand::Refine(command) => self.on_refine(command)?,
            LiveCommand::Shutdown => {
                self.close_streams()?;
                self.events.push(EngineEvent::Finished {
                    duration: self.received as f64 / RATE as f64,
                });
            }
            LiveCommand::CancelRefinement { .. } => {}
        }
        Ok(std::mem::take(&mut self.events))
    }

    fn on_audio(&mut self, notice: AudioNotice) -> Result<(), LiveError> {
        if self.capture_finished {
            return Err(LiveError::Message("Audio arrived after capture finished."));
        }
        if notice.start_sample != self.received
            || notice.end_sample <= notice.start_sample
            || notice.end_sample > (crate::audio::MAX_DURATION_SECONDS * RATE as f64) as i64
        {
            return Err(LiveError::Message("Invalid saved audio endpoint."));
        }
        self.recording
            .require_samples(notice.end_sample as u32)
            .map_err(|error| LiveError::Detail(error.to_string()))?;
        self.committed = notice.end_sample as u32;
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
            processed_seconds: self.ledger.lock().expect("speech ledger").end_sample() as f64
                / RATE as f64,
            received_seconds: self.received as f64 / RATE as f64,
            flush: false,
        });
        Ok(())
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
        for turn in feed.turns {
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
            processed_seconds: self.received as f64 / RATE as f64,
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
        self.inbox.lock().expect("inbox").capture_finished = true;
        self.commit(true, true)?;
        self.events.push(EngineEvent::Progress {
            processed_seconds: self.received as f64 / RATE as f64,
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

    fn on_refine(&mut self, command: RefineCommand) -> Result<(), LiveError> {
        let cancelled = self
            .inbox
            .lock()
            .expect("inbox")
            .cancelled_request(&command);
        self.inbox
            .lock()
            .expect("inbox")
            .finish_refinement(&command.operation_id);
        self.events.push(EngineEvent::RefinementResult {
            operation_id: command.operation_id,
            cancelled,
            fast_sequence: None,
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
        match &self.config.source_id {
            Some(source) => speaker_id(source, local),
            None => Ok(local.to_string()),
        }
    }

    fn language_at(&self, sample: i64) -> LanguageStamp {
        self.timeline
            .iter()
            .rev()
            .find(|stamp| stamp.start_sample <= sample)
            .expect("timeline covers the origin")
            .clone()
    }

    fn english_timing(&self, mode: &str, detected: Option<&str>) -> bool {
        self.timing_apis && (mode == "en" || (mode == "auto" && detected == Some("en")))
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
            let row = self.book.rows.get(&identity).expect("row").clone();
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
            self.attach(&identity)?;
            let last = self.last_decoded.get(&identity).copied();
            let changed = last.is_none_or(|(_, revision)| revision != row.audio_revision);
            let length = row.end_sample - row.start_sample;
            let stamp = self.language_at(row.start_sample);
            let authoritative = self.english_timing(&stamp.language, row.language.as_deref());
            let limit = if authoritative {
                PHYSICAL_MAX
            } else {
                MAX_DECODE_SAMPLES
            };
            let mut due = final_pass
                || row.state == UtteranceState::Sealed
                || (length >= 6 * RATE
                    && (last.is_none()
                        || (decode_open && row.end_sample - last.expect("decoded").0 >= 3 * RATE)));
            let long_unaligned = length > limit && !authoritative;
            if long_unaligned && row.state == UtteranceState::Open && !final_pass {
                due = last.is_none()
                    || (decode_open && row.end_sample - last.expect("decoded").0 >= 6 * RATE);
            }
            if changed && due {
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
        let earliest = self.book.active.as_ref().map_or(
            self.book.cursor().saturating_sub(UTTERANCE_CONTEXT_SAMPLES),
            |id| self.book.rows[id].start_sample,
        );
        self.turns.retain(|turn| round_samples(turn.end) > earliest);
        Ok(())
    }

    fn attach(&mut self, identity: &str) -> Result<(), LiveError> {
        let row = self.book.rows.get(identity).expect("row").clone();
        let regions = activity_regions(
            &self.turns,
            row.start_sample,
            row.end_sample.min(self.diar_observed),
        );
        let activity = timing::SpeakerActivity {
            audio_revision: row.audio_revision,
            observed_end_sample: Some(row.end_sample.min(self.diar_observed)),
            regions: regions.clone(),
        };
        let mut owners = Vec::new();
        for region in &regions {
            for speaker in &region.speakers {
                if !owners.contains(speaker) {
                    owners.push(speaker.clone());
                }
            }
        }
        owners.sort();
        let row = self.book.rows.get_mut(identity).expect("row");
        row.speaker_activity = Some(activity);
        row.speaker_candidates = owners;
        Ok(())
    }

    fn decode(&mut self, identity: &str) -> Result<(), LiveError> {
        let row = self.book.rows.get(identity).expect("row").clone();
        let stamp = self.language_at(row.start_sample);
        let length = row.end_sample - row.start_sample;
        let authoritative = self.english_timing(&stamp.language, row.language.as_deref());
        let limit = if authoritative {
            PHYSICAL_MAX
        } else {
            MAX_DECODE_SAMPLES
        };
        if length <= limit {
            self.decode_short(&row, &stamp.language)?;
        } else if authoritative {
            self.decode_authoritative(&row, &stamp.language)?;
        } else {
            self.decode_cores(&row, &stamp.language)?;
        }
        let row = self.book.rows.get(identity).expect("row");
        self.last_decoded
            .insert(identity.to_string(), (row.end_sample, row.audio_revision));
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
        if let Some(passage) = passages.first()
            && routed.is_none()
        {
            stored.language = passage.language.clone();
            stored.review = passage.review;
            stored.audio_state = passage.audio_state;
            stored.transcription_review = passage.transcription_review.is_some();
        }
        if let Some(routed) = routed
            && stored.protected_fields.iter().all(|field| field != "text")
            && stored.text == routed.text
        {
            stored.language = passages
                .first()
                .and_then(|passage| passage.language.clone());
        }
        if !complete || text.trim().is_empty() {
            stored.canonical_unresolved = Some("incomplete_cohere_revision".to_string());
        } else {
            stored.canonical_unresolved = None;
        }
        Ok(())
    }

    fn decode_authoritative(&mut self, row: &Utterance, language: &str) -> Result<(), LiveError> {
        let anchor = anchor_from(row);
        let requests = authoritative_requests(&anchor)?;
        let mut parts = Vec::new();
        for request in &requests {
            let left = if request.start_sample > row.start_sample {
                crate::language::CANONICAL_PAD_SAMPLES
            } else {
                0
            };
            let right = if request.end_sample < row.end_sample {
                crate::language::CANONICAL_PAD_SAMPLES
            } else {
                0
            };
            let passages = self.transcribe_span(
                request.start_sample,
                request.end_sample,
                language,
                &row.speaker_candidates,
                (left, right),
            )?;
            let part = tail_part(request, &passages, true);
            parts.push(part);
        }
        let assembled = assemble_authoritative(&anchor, &requests, &parts)?;
        let stored = self.book.rows.get_mut(&row.id).expect("row");
        stored.machine_versions.push(MachineVersion {
            text: assembled.text.clone().unwrap_or_default(),
            complete: assembled.complete,
            reason: assembled.reason.map(str::to_string),
        });
        if assembled.complete {
            if let Some(text) = assembled.text
                && stored.protected_fields.iter().all(|field| field != "text")
            {
                stored.text = text;
                stored.machine_revision += 1;
                stored.text_audio_anchor = Some(timing::AudioAnchor {
                    start_sample: stored.start_sample,
                    end_sample: stored.end_sample,
                });
                stored.refinement_state = "provisional".to_string();
                stored.canonical_unresolved = None;
            }
        } else {
            stored.canonical_unresolved = Some(
                assembled
                    .reason
                    .unwrap_or("incomplete_cohere_revision")
                    .to_string(),
            );
            stored.refinement_state = "unresolved".to_string();
        }
        Ok(())
    }

    fn decode_cores(&mut self, row: &Utterance, language: &str) -> Result<(), LiveError> {
        let core_row = CoreAnchor {
            id: row.id.clone(),
            start_sample: row.start_sample,
            end_sample: row.end_sample,
            machine_revision: row.machine_revision,
            audio_revision: row.audio_revision,
            language_epoch: row.language_epoch,
            speech_regions: row.speech_regions.clone(),
        };
        let plan = {
            let ledger = Arc::clone(&self.ledger);
            plan_cores(&core_row, |start, end| {
                ledger
                    .lock()
                    .expect("speech ledger")
                    .admission(start, end)
                    .unwrap_or_else(|_| pending_admission(start, end))
            })
        };
        let edges = validate_core_plan(&plan, &core_row)?;
        let anchor = anchor_from(row);
        let mut requests = Vec::new();
        for pair in edges.windows(2) {
            requests.push(request_for(&anchor, pair[0], pair[1], pair[0], pair[1]));
        }
        let mut parts = Vec::new();
        for request in &requests {
            let passages = self.transcribe_span(
                request.start_sample,
                request.end_sample,
                language,
                &row.speaker_candidates,
                (0, 0),
            )?;
            let tail = tail_part(request, &passages, false);
            parts.push(CorePart {
                text: tail.text,
                complete: tail.complete,
            });
        }
        let complete = parts.iter().all(|part| part.complete);
        let text = parts
            .iter()
            .map(|part| part.text.as_str())
            .collect::<Vec<_>>()
            .join(" ");
        self.book
            .apply_model(&row.id, &text, DecodeStage::Live, complete)?;
        let stored = self.book.rows.get_mut(&row.id).expect("row");
        if complete && !text.trim().is_empty() {
            stored.canonical_unresolved = None;
            stored.language = Some(language.to_string());
        } else {
            stored.canonical_unresolved = Some("incomplete_cohere_revision".to_string());
        }
        let _ = plan;
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
        let detector = if language == languages::AUTOMATIC {
            Some(Box::new(ModelDetector(Arc::clone(&self.models))) as Box<dyn LanguageDetector>)
        } else {
            None
        };
        let mut transcriber = SpeechTranscriber::new(
            language,
            Box::new(SharedLedger(Arc::clone(&self.ledger))),
            detector,
            Box::new(ModelRecognizer {
                models: Arc::clone(&self.models),
            }),
            None,
        )
        .map_err(|error| LiveError::Detail(error.to_string()))?;
        transcriber
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
            .map_err(|error| LiveError::Detail(error.to_string()))
    }

    fn publish(&mut self, row: Utterance) -> Result<(), LiveError> {
        let candidate = project_segment(&row, &self.language_at(row.start_sample));
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

fn tail_part(request: &DecodeRequest, passages: &[Passage], aligned: bool) -> TailPart {
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
    complete = complete && cursor == span && (!aligned || passages.len() == 1);
    TailPart {
        request: request.clone(),
        text: texts.join(" "),
        complete,
    }
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
    rest.insert(
        "canonical_state".to_string(),
        Value::String(if row.state == UtteranceState::Sealed {
            "sealed".to_string()
        } else {
            "open".to_string()
        }),
    );
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
        language_detection: None,
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
    pub fn new(
        config: EngineConfig,
        models: M,
        recordings: [RecordingReader; 2],
    ) -> Result<Self, LiveError> {
        let models = Arc::new(models);
        let inbox = Arc::new(Mutex::new(Inbox::new(config.audio_batch_seconds, 0)?));
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
                    EngineEvent::CanonicalRevision { .. }
                    | EngineEvent::RefinementResult { .. } => {
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
        let mut inbox = Inbox::new(6, 0).unwrap();
        inbox.push(notice(0, RATE, 0)).unwrap();
        inbox.push(notice(RATE, 2 * RATE, 0)).unwrap();
        assert_eq!(inbox.queued_audio_span(), Some((0, 2 * RATE, 0)));
        inbox.push(notice(2 * RATE, 3 * RATE, 1)).unwrap();
        assert_eq!(inbox.take().unwrap().map(|command| matches!(command, LiveCommand::Audio(ref audio) if audio.end_sample == 2 * RATE)), Some(true));
        assert_eq!(inbox.queued_audio_span(), Some((2 * RATE, 3 * RATE, 1)));
    }

    #[test]
    fn inbox_drops_a_cancelled_refinement_and_a_stale_language_epoch() {
        let mut inbox = Inbox::new(1, 0).unwrap();
        let pending = RefineCommand {
            operation_id: "refine-1".into(),
            language_epoch: 0,
            canonical_sealed: false,
        };
        inbox.push(LiveCommand::Refine(pending.clone())).unwrap();
        inbox
            .push(LiveCommand::CancelRefinement {
                operation_id: "refine-1".into(),
            })
            .unwrap();
        assert!(inbox.pending_refinement().is_none());
        assert!(inbox.cancelled_request(&pending));
        assert!(inbox.take().unwrap().is_none());

        let queued = RefineCommand {
            operation_id: "refine-2".into(),
            language_epoch: 0,
            canonical_sealed: false,
        };
        inbox.push(LiveCommand::Refine(queued)).unwrap();
        inbox
            .push(LiveCommand::Language {
                language: "de".into(),
                generation: 1,
                start_sample: 0,
            })
            .unwrap();
        assert!(inbox.pending_refinement().is_none());
        assert!(!inbox.newer_audio_queued(0, 0));
    }

    #[test]
    fn inbox_prefers_audio_until_one_batch_has_been_taken() {
        let mut inbox = Inbox::new(6, 0).unwrap();
        inbox.push(notice(0, RATE, 0)).unwrap();
        inbox
            .push(LiveCommand::Refine(RefineCommand {
                operation_id: "later".into(),
                language_epoch: 0,
                canonical_sealed: false,
            }))
            .unwrap();
        assert!(matches!(inbox.take().unwrap(), Some(LiveCommand::Audio(_))));
        assert!(matches!(
            inbox.take().unwrap(),
            Some(LiveCommand::Refine(_))
        ));
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
            })
            .collect();
        let seam = assemble_authoritative(&long, &requests, &parts).unwrap();
        assert!(!seam.complete);
        assert!(seam.recognition_complete);
        assert_eq!(seam.reason, Some("unavailable_boundary_timing"));
        assert!(seam.text.is_none());
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

    fn window() -> OwnershipWindow {
        OwnershipWindow {
            id: "win".into(),
            kind: None,
            start_sample: 0,
            end_sample: RATE,
        }
    }

    #[test]
    fn reconcile_window_checks_segment_version_and_edit_passage_protection() {
        let original = document_with(sample_segment("hello", 0));
        let mut expected = BTreeMap::new();
        expected.insert("seg-1".into(), "stale".into());
        let mut candidate = sample_segment("replacement", 0);
        candidate.id = "new".into();
        let blocked =
            reconcile_window(&original, &window(), &expected, &[candidate.clone()], None).unwrap();
        assert_eq!(blocked.document.segments[0].text, "hello");
        assert!(blocked.protected_ids.iter().any(|id| id == "seg-1"));

        expected.insert("seg-1".into(), segment_version(&original.segments[0]));
        let replaced =
            reconcile_window(&original, &window(), &expected, &[candidate], None).unwrap();
        assert_eq!(replaced.document.segments[0].text, "replacement");
        assert_eq!(replaced.document.segments[0].id, "seg-1");
        assert_eq!(replaced.document.segments[0].machine_revision, Some(1));

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
        assert_eq!(
            edited.segment.protected_fields.as_deref(),
            Some(["text".to_string()].as_slice())
        );
        let mut expected = BTreeMap::new();
        expected.insert("seg-1".into(), segment_version(&edited.segment));
        let mut newer = sample_segment("model words", 1);
        newer.id = "new".into();
        let kept =
            reconcile_window(&edited.document, &window(), &expected, &[newer], None).unwrap();
        assert_eq!(kept.document.segments[0].text, "edited words");
        assert!(kept.protected_ids.iter().any(|id| id == "seg-1"));

        let mut arriving = sample_segment("model words", 4);
        arriving.id = edited.segment.id.clone();
        let applied = apply_canonical_candidate(&edited.document, &arriving);
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
        vad_sessions: Mutex<BTreeMap<u64, (u64, Vec<f32>)>>,
        next_session: AtomicUsize,
        fail_diar_open: AtomicBool,
        fail_vad_eof: AtomicBool,
        diar_sessions: AtomicUsize,
        diar_progress: Mutex<BTreeMap<u64, (u64, u64)>>,
        delayed_diar: AtomicBool,
    }

    impl FakeModels {
        fn english() -> Self {
            Self {
                conditioning: Mutex::new(Vec::new()),
                ctc_calls: AtomicUsize::new(0),
                transcribe_calls: AtomicUsize::new(0),
                vad_sessions: Mutex::new(BTreeMap::new()),
                next_session: AtomicUsize::new(1),
                fail_diar_open: AtomicBool::new(false),
                fail_vad_eof: AtomicBool::new(false),
                diar_sessions: AtomicUsize::new(0),
                diar_progress: Mutex::new(BTreeMap::new()),
                delayed_diar: AtomicBool::new(false),
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
                0
            } else {
                received / 160 * 160
            };
            let start = *previous;
            *previous = observed;
            Ok(DiarFeedOut {
                turns: if observed > start {
                    vec![crate::ipc::DiarTurn {
                        start,
                        end: observed,
                        speaker: "speaker_0".into(),
                    }]
                } else {
                    Vec::new()
                },
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
            Ok(LanguageProbeOut {
                probabilities: vec![0.99],
                codes: vec!["en".into()],
            })
        }
        fn transcribe(
            &self,
            _pcm: &[f32],
            language: &str,
            _padding: (u32, u32),
        ) -> Result<AsrText, ChildError> {
            self.transcribe_calls.fetch_add(1, Ordering::SeqCst);
            let text = if language == "de" { "eins" } else { "hello" };
            Ok(AsrText {
                text: text.into(),
                raw_text: text.into(),
                token_count: 1,
            })
        }
        fn ctc_scores(&self, _pcm: &[f32], _token_ids: &[i32]) -> Result<CtcScoresOut, ChildError> {
            self.ctc_calls.fetch_add(1, Ordering::SeqCst);
            Err(ChildError::Unimplemented)
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
        engine.handle(notice(0, pcm.len() as i64, 0)).unwrap();
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
        assert!(!engine.timing_apis());
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
        let models_calls = engine.models.transcribe_calls.load(Ordering::SeqCst);
        engine
            .push(LiveCommand::Refine(RefineCommand {
                operation_id: "refine-9".into(),
                language_epoch: 0,
                canonical_sealed: false,
            }))
            .unwrap();
        let taken = engine.inbox.lock().unwrap().take().unwrap().unwrap();
        engine
            .inbox
            .lock()
            .unwrap()
            .push(LiveCommand::CancelRefinement {
                operation_id: "refine-9".into(),
            })
            .unwrap();
        let result = engine.handle(taken).unwrap();
        assert!(matches!(
            &result[0],
            EngineEvent::RefinementResult { operation_id, cancelled: true, .. }
                if operation_id == "refine-9"
        ));
        assert_eq!(
            engine.models.transcribe_calls.load(Ordering::SeqCst),
            models_calls
        );
    }

    #[test]
    fn unwired_live_timing_never_probes_or_calls_the_acoustic_model() {
        let mut english =
            TestEngine::new(EngineConfig::new("job-en", "en"), FakeModels::english()).unwrap();
        assert!(!english.timing_apis());
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
