//! Language policy and speech transcriber. Owned by the core-speech agent.
//! Thresholds live in [`crate::limits`] and codes in [`crate::languages`].
//!
//! Probabilities are injected. This module does not load Whisper weights.

use std::collections::HashMap;

use thiserror::Error;

use crate::languages::{self, AUTOMATIC};
use crate::limits;
use crate::memo::{self, ExactMemo, MemoScope};
use crate::messages;
use crate::speech::{
    self, Admission, BlockedSpeech, PcmEvidence, SILERO_REVISION, SILERO_SOURCE, SampleSpan,
    SpeechDecision, SpeechError, SpeechLedger,
};

/// Successful supported-language context expires after 60 seconds of audio.
pub const RECENT_CONTEXT_SECONDS: i64 = 60;
pub const RECENT_CONTEXT_SAMPLES: i64 = RECENT_CONTEXT_SECONDS * limits::SAMPLE_RATE as i64;
pub const PROBE_SECONDS: f64 = 3.0;
pub const MIN_COHERE_SAMPLES: usize = 320;
pub const SHORT_CONTEXT_SAMPLES: usize = 3_200;
pub const CANONICAL_PAD_SAMPLES: u32 = 3_200;
pub const MIN_ASR_SECONDS: f64 = 0.75;
pub const DEFAULT_MAX_ASR_SECONDS: f64 = 6.0;
pub const TOKEN_LIMIT: usize = 448;
pub const PAD_POLICY: &str = "canonical_cut_context_200ms_zeros_v1";

#[derive(Debug, Error, Clone, PartialEq, Eq)]
pub enum LanguageError {
    #[error("Invalid language detection probabilities.")]
    InvalidProbabilities,
    #[error("Invalid preceding language context.")]
    InvalidContext,
    #[error("{0}")]
    UnsupportedMode(&'static str),
    #[error("Language routing requires finite 16 kHz mono speech audio and its sample position.")]
    InvalidAudio,
}

#[derive(Debug, Error, Clone, PartialEq, Eq)]
pub enum RouteError {
    #[error("detector failed")]
    Detector,
    #[error("recognition failed")]
    Recognition,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DetectionMode {
    Auto,
    Manual,
}

impl DetectionMode {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Auto => "auto",
            Self::Manual => "manual",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DetectionReason {
    NeedsLanguage,
    Unsupported,
    Detected,
    Mixed,
    BestEffort,
    RecentContext,
    InsufficientSpeech,
    SpeechAdmissionUncertain,
    SpeechEvidencePending,
    Override,
}

impl DetectionReason {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::NeedsLanguage => "needs_language",
            Self::Unsupported => "unsupported",
            Self::Detected => "detected",
            Self::Mixed => "mixed",
            Self::BestEffort => "best_effort",
            Self::RecentContext => "recent_context",
            Self::InsufficientSpeech => "insufficient_speech",
            Self::SpeechAdmissionUncertain => "speech_admission_uncertain",
            Self::SpeechEvidencePending => "speech_evidence_pending",
            Self::Override => "override",
        }
    }

    fn from_blocked(reason: &str) -> Self {
        match reason {
            "speech_admission_uncertain" => Self::SpeechAdmissionUncertain,
            "speech_evidence_pending" => Self::SpeechEvidencePending,
            _ => Self::InsufficientSpeech,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AudioState {
    DigitalSilence,
    ConstantSignal,
    ModelNonSpeech,
    SpeechEvidencePending,
    InsufficientAcousticContext,
    ShortAcousticContext,
}

impl AudioState {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::DigitalSilence => "digital_silence",
            Self::ConstantSignal => "constant_signal",
            Self::ModelNonSpeech => "model_non_speech",
            Self::SpeechEvidencePending => "speech_evidence_pending",
            Self::InsufficientAcousticContext => "insufficient_acoustic_context",
            Self::ShortAcousticContext => "short_acoustic_context",
        }
    }

    fn from_name(name: &str) -> Self {
        match name {
            "digital_silence" => Self::DigitalSilence,
            "constant_signal" => Self::ConstantSignal,
            "model_non_speech" => Self::ModelNonSpeech,
            "insufficient_acoustic_context" => Self::InsufficientAcousticContext,
            "short_acoustic_context" => Self::ShortAcousticContext,
            _ => Self::SpeechEvidencePending,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum LanguageChoice {
    Auto,
    Manual(String),
}

impl LanguageChoice {
    pub fn parse(language: &str) -> Result<Self, LanguageError> {
        if language == AUTOMATIC {
            Ok(Self::Auto)
        } else if languages::is_supported(language) {
            Ok(Self::Manual(language.to_owned()))
        } else {
            Err(LanguageError::UnsupportedMode(
                messages::CHOOSE_LANGUAGE_MODE,
            ))
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct Candidate {
    pub language: String,
    pub probability: f64,
}

#[derive(Debug, Clone, PartialEq)]
pub struct LanguageDetection {
    pub mode: DetectionMode,
    pub reason: DetectionReason,
    pub probability: Option<f64>,
    pub margin: Option<f64>,
    pub candidates: Vec<Candidate>,
    pub detector_error: bool,
    pub context_end_sample: Option<i64>,
    pub context_evidence: Option<LanguageContextEvidence>,
    pub probes: Vec<ProbeEvent>,
    /// At most two selected direct self-donor receipts survive a routed append.
    /// These remain separate from the current crop's language observations.
    pub context_witnesses: Vec<ProbeEvent>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct ProbeEvent {
    pub start_sample: i64,
    pub end_sample: i64,
    pub language: Option<String>,
    pub decision: LanguageDetection,
    pub review: bool,
}

/// Retain decisions from actual routing, including the single-window case where
/// the transcriber has no nested probe list. Live options already use the saved
/// recording clock; only passage display bounds are relative to this crop.
pub(crate) fn append_passage_decisions(
    events: &mut Vec<ProbeEvent>,
    passages: &[Passage],
    start: i64,
) {
    for passage in passages {
        // Verified non-speech carries no spoken-language decision. Unknown speech
        // and failed/unsupported routing still participate in the summary.
        if passage.text.trim().is_empty()
            && matches!(
                passage.audio_state,
                Some(
                    AudioState::DigitalSilence
                        | AudioState::ConstantSignal
                        | AudioState::ModelNonSpeech
                )
            )
        {
            continue;
        }
        if passage.language_detection.probes.is_empty() {
            events.push(ProbeEvent {
                start_sample: start
                    + (passage.start * f64::from(limits::SAMPLE_RATE)).round() as i64,
                end_sample: start + (passage.end * f64::from(limits::SAMPLE_RATE)).round() as i64,
                language: passage.language.clone(),
                decision: passage.language_detection.clone(),
                review: passage.language_review,
            });
        } else {
            events.extend(passage.language_detection.probes.iter().cloned());
        }
    }
}

/// A display summary of retained routing decisions, not a new detection. Mixed
/// crops have no single language or invented aggregate probability; each actual
/// decision remains available at its absolute acoustic position.
pub(crate) fn summarize_decisions(events: Vec<ProbeEvent>) -> Option<PolicyDecision> {
    let first = events.first()?;
    let first_language = events.iter().find_map(|event| event.language.as_deref());
    let mixed = events
        .iter()
        .filter_map(|event| event.language.as_deref())
        .any(|language| Some(language) != first_language);
    let unknown = events.iter().find(|event| event.language.is_none());
    let selected = unknown.unwrap_or_else(|| {
        events
            .iter()
            .min_by(|left, right| {
                // A reviewed decision remains visible even when another probe is confident.
                right.review.cmp(&left.review).then_with(|| {
                    left.decision
                        .probability
                        .unwrap_or(1.)
                        .total_cmp(&right.decision.probability.unwrap_or(1.))
                })
            })
            .expect("nonempty decisions")
    });
    let language = if mixed || unknown.is_some() {
        None
    } else {
        first.language.clone()
    };
    let mut detection = selected.decision.clone();
    if mixed {
        detection.reason = DetectionReason::Mixed;
        detection.probability = None;
        detection.margin = None;
        detection.candidates.clear();
        detection.context_end_sample = None;
        detection.context_evidence = None;
    }
    detection.detector_error = events.iter().any(|event| event.decision.detector_error);
    let review = mixed || unknown.is_some() || events.iter().any(|event| event.review);
    detection.probes = if detection.mode == DetectionMode::Auto {
        events
    } else {
        Vec::new()
    };
    Some(PolicyDecision {
        language,
        review,
        detection,
    })
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LanguageContext {
    pub language: String,
    pub end_sample: i64,
    pub evidence: Option<LanguageContextEvidence>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LanguageContextAnchor {
    pub utterance_id: String,
    pub row_start_sample: i64,
    pub speaker_track_id: Option<String>,
    pub start_sample: i64,
    pub end_sample: i64,
    pub machine_revision: i64,
    pub audio_revision: i64,
    pub probe_sha256: String,
    pub probe_index: Option<usize>,
}

/// Borrowed decisions retain their actual acoustic sources. They never become
/// new direct detections when later passages choose a language context.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LanguageContextEvidence {
    pub scope: &'static str,
    pub speaker_track_id: Option<String>,
    pub language_epoch: i64,
    pub interpretation_generation: u64,
    pub capture_source: Option<serde_json::Value>,
    pub stream_generation: Option<i64>,
    pub anchors: Vec<LanguageContextAnchor>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct PolicyDecision {
    pub language: Option<String>,
    pub review: bool,
    pub detection: LanguageDetection,
}

#[derive(Debug, Clone, PartialEq)]
pub struct TranscriptionReview {
    pub reason: &'static str,
    pub partial_text: bool,
    pub valid_feature_frames: Option<i64>,
    pub decoder_reason: Option<&'static str>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CoherePadding {
    pub policy: &'static str,
    pub leading_samples: u32,
    pub trailing_samples: u32,
    pub physical_start_sample: i64,
    pub physical_end_sample: i64,
}

#[derive(Debug, Clone, PartialEq)]
pub struct AsrOutput {
    pub text: String,
    pub token_count: usize,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Passage {
    pub start: f64,
    pub end: f64,
    pub text: String,
    pub cohere_raw_text: Option<String>,
    pub language: Option<String>,
    pub language_detection: LanguageDetection,
    pub review: bool,
    pub language_review: bool,
    pub audio_state: Option<AudioState>,
    pub acoustic_evidence: Admission,
    pub pcm_evidence: Option<PcmEvidence>,
    pub transcription_review: Option<TranscriptionReview>,
    pub cohere_input_padding: Option<CoherePadding>,
}

#[derive(Debug, Clone, Copy)]
pub struct TranscribeOptions {
    pub max_asr_seconds: f64,
    pub start_sample: i64,
    pub asr_padding: (u32, u32),
}

impl Default for TranscribeOptions {
    fn default() -> Self {
        Self {
            max_asr_seconds: DEFAULT_MAX_ASR_SECONDS,
            start_sample: 0,
            asr_padding: (0, 0),
        }
    }
}

pub trait AcousticEvidence {
    fn admit(&mut self, start: i64, end: i64) -> Result<Admission, SpeechError>;
}

pub trait LanguageDetector {
    fn detect(&mut self, pcm: &[f32]) -> Result<Vec<(String, f64)>, RouteError>;
}

pub trait Recognizer {
    fn transcribe(&mut self, pcm: &[f32], language: &str) -> Result<AsrOutput, RouteError>;
}

#[derive(Debug, Clone)]
pub struct FixedEvidence {
    pub decision: SpeechDecision,
    pub probability: f64,
}

impl AcousticEvidence for FixedEvidence {
    fn admit(&mut self, start: i64, end: i64) -> Result<Admission, SpeechError> {
        Ok(fixed_admission(start, end, self.decision, self.probability))
    }
}

impl AcousticEvidence for SpeechLedger {
    fn admit(&mut self, start: i64, end: i64) -> Result<Admission, SpeechError> {
        self.admission(start, end)
    }
}

#[derive(Debug, Clone)]
pub struct LanguagePolicy {
    current: HashMap<Vec<String>, String>,
}

impl Default for LanguagePolicy {
    fn default() -> Self {
        Self::new()
    }
}

impl LanguagePolicy {
    pub fn new() -> Self {
        Self {
            current: HashMap::new(),
        }
    }

    pub fn reset(&mut self, speaker: &[String]) {
        self.current.remove(speaker);
    }

    pub fn remembered(&self, speaker: &[String]) -> Option<&str> {
        self.current.get(speaker).map(String::as_str)
    }

    pub fn decide(
        &mut self,
        probabilities: &[(String, f64)],
        speaker: &[String],
    ) -> Result<PolicyDecision, LanguageError> {
        memo::validate_language_scores(probabilities)
            .map_err(|_| LanguageError::InvalidProbabilities)?;
        let mut ranked: Vec<(usize, &str, f64)> = probabilities
            .iter()
            .enumerate()
            .map(|(index, (code, probability))| (index, code.as_str(), *probability))
            .collect();
        ranked.sort_by(|left, right| {
            right
                .2
                .partial_cmp(&left.2)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then(left.0.cmp(&right.0))
        });
        let winner = ranked[0].1;
        let probability = ranked[0].2;
        let runner = ranked.get(1).map(|item| item.2).unwrap_or(0.0);
        let margin = probability - runner;
        let confident = is_confident_detection(probability, margin);
        let candidates = ranked
            .iter()
            .take(3)
            .map(|(_, code, score)| Candidate {
                language: (*code).to_owned(),
                probability: *score,
            })
            .collect();
        let mut detection = blank_detection(DetectionMode::Auto, DetectionReason::NeedsLanguage);
        detection.probability = Some(probability);
        detection.margin = Some(margin);
        detection.candidates = candidates;
        if !languages::is_supported(winner) {
            if confident {
                self.reset(speaker);
                detection.reason = DetectionReason::Unsupported;
            }
            return Ok(PolicyDecision {
                language: None,
                review: true,
                detection,
            });
        }
        detection.reason = if confident {
            DetectionReason::Detected
        } else {
            DetectionReason::BestEffort
        };
        if confident {
            self.current.insert(speaker.to_vec(), winner.to_owned());
        }
        Ok(PolicyDecision {
            language: Some(winner.to_owned()),
            review: !confident,
            detection,
        })
    }
}

pub fn is_confident_detection(probability: f64, margin: f64) -> bool {
    probability >= limits::LANGUAGE_AUTO_THRESHOLD && margin >= limits::LANGUAGE_MARGIN_THRESHOLD
}

pub fn detector_issues(verified: bool) -> &'static [&'static str] {
    if verified {
        &[]
    } else {
        const MISSING: [&str; 1] = [messages::LID_MODEL_MISSING];
        &MISSING
    }
}

pub fn probe_edges(len: usize, sample_rate: u32, automatic: bool) -> Vec<usize> {
    if !automatic || sample_rate == 0 {
        return vec![0, len];
    }
    let count = ((len as f64) / (PROBE_SECONDS * f64::from(sample_rate)))
        .ceil()
        .max(1.0) as usize;
    let mut edges = Vec::with_capacity(count + 1);
    for index in 0..count {
        let value = (len as f64) * (index as f64) / (count as f64);
        edges.push(value as usize);
    }
    edges.push(len);
    edges
}

pub fn fixed_admission(
    start: i64,
    end: i64,
    decision: SpeechDecision,
    probability: f64,
) -> Admission {
    Admission {
        source: SILERO_SOURCE.to_owned(),
        model_revision: SILERO_REVISION.to_owned(),
        start_sample: start,
        end_sample: end,
        observed_until_sample: end,
        complete: decision != SpeechDecision::Pending,
        decision,
        speech_regions: if decision == SpeechDecision::Speech {
            vec![SampleSpan {
                start_sample: start,
                end_sample: end,
            }]
        } else {
            Vec::new()
        },
        input_policy: Some(speech::INPUT_POLICY.to_owned()),
        conditioning_sha256: None,
        maximum_probability: Some(probability),
        maximum_model_probability: Some(probability),
        uncertain_regions: Vec::new(),
        model_negative_regions: Vec::new(),
        negative_signal_state: None,
        classification_scope: None,
        evidence_error: false,
    }
}

pub fn accept_admission(evidence: Admission, start: i64, end: i64) -> Admission {
    if admission_is_usable(&evidence, start, end) {
        evidence
    } else {
        rejected_pending(start, end, false)
    }
}

pub struct SpeechTranscriber<'a> {
    language: LanguageChoice,
    policy: LanguagePolicy,
    context: Option<LanguageContext>,
    evidence: Box<dyn AcousticEvidence + 'a>,
    detector: Option<Box<dyn LanguageDetector + 'a>>,
    recognizer: Box<dyn Recognizer + 'a>,
    memo: Option<ExactMemo>,
    probe_scope: MemoScope,
    language_epoch: u64,
    pub cohere_calls: u64,
}

impl<'a> SpeechTranscriber<'a> {
    pub fn new(
        language: &str,
        evidence: Box<dyn AcousticEvidence + 'a>,
        detector: Option<Box<dyn LanguageDetector + 'a>>,
        recognizer: Box<dyn Recognizer + 'a>,
        context: Option<LanguageContext>,
    ) -> Result<Self, LanguageError> {
        let choice = LanguageChoice::parse(language)?;
        let context = match context {
            Some(context) => Some(validate_context(context)?),
            None => None,
        };
        Ok(Self {
            language: choice,
            policy: LanguagePolicy::new(),
            context,
            evidence,
            detector,
            recognizer,
            memo: None,
            probe_scope: MemoScope::new("probe", "default"),
            language_epoch: 0,
            cohere_calls: 0,
        })
    }

    pub fn context(&self) -> Option<&LanguageContext> {
        self.context.as_ref()
    }

    /// The import adapter routes a bounded crop on its local clock. Validate
    /// source context in `new` first; only this owner translates its origin.
    pub(crate) fn translate_context_to_crop(&mut self, origin: i64) -> Result<(), LanguageError> {
        if origin < 0 {
            return Err(LanguageError::InvalidContext);
        }
        if let Some(context) = &mut self.context {
            context.end_sample = context
                .end_sample
                .checked_sub(origin)
                .ok_or(LanguageError::InvalidContext)?;
        }
        Ok(())
    }

    pub fn set_language(&mut self, language: &str) -> Result<(), LanguageError> {
        let choice = LanguageChoice::parse(language)?;
        if self.language != choice {
            self.language = choice;
            self.policy = LanguagePolicy::new();
            self.context = None;
        }
        Ok(())
    }

    pub fn set_probe_memo(&mut self, memo: ExactMemo, scope: MemoScope) {
        self.memo = Some(memo);
        self.probe_scope = scope;
    }

    pub fn set_probe_scope(&mut self, scope: MemoScope) {
        self.probe_scope = scope;
    }

    pub fn set_language_epoch(&mut self, epoch: u64) {
        self.language_epoch = epoch;
    }

    pub fn transcribe(
        &mut self,
        audio: &[f32],
        sample_rate: u32,
        speakers: &[String],
        opts: TranscribeOptions,
    ) -> Result<Vec<Passage>, LanguageError> {
        let mut memo = self.memo.take();
        let result = self.transcribe_inner(audio, sample_rate, speakers, opts, memo.as_mut());
        self.memo = memo;
        result
    }

    pub fn transcribe_sharing(
        &mut self,
        audio: &[f32],
        sample_rate: u32,
        speakers: &[String],
        opts: TranscribeOptions,
        memo: &mut ExactMemo,
    ) -> Result<Vec<Passage>, LanguageError> {
        self.transcribe_inner(audio, sample_rate, speakers, opts, Some(memo))
    }

    fn transcribe_inner(
        &mut self,
        audio: &[f32],
        sample_rate: u32,
        speakers: &[String],
        opts: TranscribeOptions,
        memo: Option<&mut ExactMemo>,
    ) -> Result<Vec<Passage>, LanguageError> {
        validate_audio(audio, sample_rate, &opts)?;
        let mut probes = self.build_probes(audio, sample_rate, speakers, &opts, memo);
        Ok(self.collapse(&mut probes, audio, sample_rate, speakers, &opts))
    }

    fn build_probes(
        &mut self,
        audio: &[f32],
        sample_rate: u32,
        speakers: &[String],
        opts: &TranscribeOptions,
        mut memo: Option<&mut ExactMemo>,
    ) -> Vec<Probe> {
        let edges = probe_edges(
            audio.len(),
            sample_rate,
            matches!(self.language, LanguageChoice::Auto),
        );
        let mut spans = Vec::new();
        for pair in edges.windows(2) {
            let (begin, end) = (pair[0], pair[1]);
            if begin >= end {
                continue;
            }
            let abs_start = opts.start_sample + begin as i64;
            let abs_end = opts.start_sample + end as i64;
            spans.push((
                begin,
                end,
                guarded_admit(self.evidence.as_mut(), abs_start, abs_end),
            ));
        }
        spans
            .into_iter()
            .map(|(begin, end, evidence)| {
                let crop = Crop {
                    pcm: &audio[begin..end],
                    begin,
                    end,
                    start_sample: opts.start_sample,
                    speakers,
                    evidence,
                };
                self.classify(crop, memo.as_deref_mut())
            })
            .collect()
    }

    fn classify(&mut self, crop: Crop<'_>, memo: Option<&mut ExactMemo>) -> Probe {
        let abs_start = crop.start_sample + crop.begin as i64;
        let abs_end = crop.start_sample + crop.end as i64;
        let mode = self.detection_mode();
        if let Some(blocked) =
            speech::blocked_route(crop.evidence.decision, crop.pcm, abs_start, abs_end)
        {
            return blocked_probe(crop.begin, crop.end, mode, self.manual_language(), blocked);
        }
        if crop.pcm.len() < MIN_COHERE_SAMPLES {
            return insufficient_probe(crop.begin, crop.end, mode, self.manual_language());
        }
        if let Some(language) = self.manual_language() {
            return override_probe(crop.begin, crop.end, language);
        }
        let scores = self.load_scores(crop.pcm, abs_start, memo);
        match scores.and_then(|scores| self.policy.decide(&scores, crop.speakers).map_err(|_| ())) {
            Ok(decision) => Probe {
                begin: crop.begin,
                end: crop.end,
                language: decision.language,
                review: decision.review,
                detection: decision.detection,
                audio_state: None,
                pcm_evidence: None,
            },
            Err(()) => detector_error_probe(crop.begin, crop.end),
        }
    }

    fn load_scores(
        &mut self,
        pcm: &[f32],
        start: i64,
        memo: Option<&mut ExactMemo>,
    ) -> Result<Vec<(String, f64)>, ()> {
        let mut detector = self.detector.take();
        let scope = self.probe_scope.clone();
        let epoch = self.language_epoch;
        let loaded = match detector.as_mut() {
            None => Err(()),
            Some(detector) => load_detected(detector.as_mut(), memo, &scope, epoch, pcm, start),
        };
        self.detector = detector;
        loaded
    }

    fn collapse(
        &mut self,
        probes: &mut [Probe],
        audio: &[f32],
        sample_rate: u32,
        speakers: &[String],
        opts: &TranscribeOptions,
    ) -> Vec<Passage> {
        let mut passages = Vec::new();
        let mut index = 0;
        let automatic = self.is_auto();
        while index < probes.len() {
            let window = index;
            index += 1;
            let absolute = opts.start_sample + probes[window].begin as i64;
            note_window(&mut self.context, automatic, &mut probes[window], absolute);
            let mut events = vec![snapshot(&probes[window], opts.start_sample)];
            absorb_followers(&mut Merge {
                context: &self.context,
                probes,
                window,
                index: &mut index,
                events: &mut events,
                opts,
                sample_rate,
                automatic,
            });
            if events.len() > 1 {
                probes[window].detection.probes = events.clone();
            }
            let chosen = probes[window].clone();
            passages.push(self.finish(audio, sample_rate, speakers, opts, &chosen, &events));
        }
        passages
    }

    fn finish(
        &mut self,
        audio: &[f32],
        sample_rate: u32,
        speakers: &[String],
        opts: &TranscribeOptions,
        window: &Probe,
        events: &[ProbeEvent],
    ) -> Passage {
        let mut transcription_review = acoustic_review(window);
        let abs_start = opts.start_sample + window.begin as i64;
        let abs_end = opts.start_sample + window.end as i64;
        let evidence = guarded_admit(self.evidence.as_mut(), abs_start, abs_end);
        let mut raw_text = None;
        let mut text = String::new();
        let mut audio_state = window.audio_state;
        let mut padding = None;
        if should_recognize(window, evidence.decision) {
            let language = window.language.clone().unwrap_or_default();
            let (pcm, pad) = recognize_pcm(audio, window, opts);
            padding = pad;
            self.cohere_calls += 1;
            match self.recognizer.transcribe(&pcm, &language) {
                Ok(output) => {
                    raw_text = Some(output.text.clone());
                    text = output.text.trim().to_owned();
                    if let Some(review) = asr_review(&output, &text, window.end - window.begin) {
                        transcription_review = Some(review);
                    }
                    if !text.is_empty() && window.end - window.begin < SHORT_CONTEXT_SAMPLES {
                        audio_state = Some(AudioState::ShortAcousticContext);
                    }
                }
                Err(_) => {
                    transcription_review = Some(TranscriptionReview {
                        reason: "transcription_failed",
                        partial_text: false,
                        valid_feature_frames: None,
                        decoder_reason: None,
                    });
                }
            }
        }
        if self.is_auto() {
            update_context(
                &mut self.context,
                events,
                &text,
                transcription_review.as_ref(),
            );
        }
        let language_review = window.review;
        let review = window.review || transcription_review.is_some() || speakers.len() > 1;
        Passage {
            start: window.begin as f64 / f64::from(sample_rate),
            end: window.end as f64 / f64::from(sample_rate),
            text,
            cohere_raw_text: raw_text,
            language: window.language.clone(),
            language_detection: window.detection.clone(),
            review,
            language_review,
            audio_state,
            acoustic_evidence: evidence,
            pcm_evidence: window.pcm_evidence.clone(),
            transcription_review,
            cohere_input_padding: padding,
        }
    }

    fn is_auto(&self) -> bool {
        matches!(self.language, LanguageChoice::Auto)
    }

    fn detection_mode(&self) -> DetectionMode {
        if self.is_auto() {
            DetectionMode::Auto
        } else {
            DetectionMode::Manual
        }
    }

    fn manual_language(&self) -> Option<String> {
        match &self.language {
            LanguageChoice::Manual(code) => Some(code.clone()),
            LanguageChoice::Auto => None,
        }
    }
}

#[derive(Clone)]
struct Probe {
    begin: usize,
    end: usize,
    language: Option<String>,
    detection: LanguageDetection,
    review: bool,
    audio_state: Option<AudioState>,
    pcm_evidence: Option<PcmEvidence>,
}

struct Crop<'a> {
    pcm: &'a [f32],
    begin: usize,
    end: usize,
    start_sample: i64,
    speakers: &'a [String],
    evidence: Admission,
}

fn validate_context(context: LanguageContext) -> Result<LanguageContext, LanguageError> {
    if !languages::is_supported(&context.language) || context.end_sample < 0 {
        return Err(LanguageError::InvalidContext);
    }
    Ok(context)
}

fn validate_audio(
    audio: &[f32],
    sample_rate: u32,
    opts: &TranscribeOptions,
) -> Result<(), LanguageError> {
    let limit = limits::PASSAGE_RETRY_MAX_SECONDS * f64::from(limits::SAMPLE_RATE);
    let (left, right) = opts.asr_padding;
    let padding_ok =
        matches!(left, 0 | CANONICAL_PAD_SAMPLES) && matches!(right, 0 | CANONICAL_PAD_SAMPLES);
    let padded = audio.len() as u64 + u64::from(left) + u64::from(right);
    if !padding_ok
        || audio.is_empty()
        || sample_rate != limits::SAMPLE_RATE
        || audio.iter().any(|sample| !sample.is_finite())
        || opts.start_sample < 0
        || !opts.max_asr_seconds.is_finite()
        || opts.max_asr_seconds < MIN_ASR_SECONDS
        || opts.max_asr_seconds > limits::PASSAGE_RETRY_MAX_SECONDS
        || audio.len() as f64 > limit
        || padded as f64 > limit
    {
        return Err(LanguageError::InvalidAudio);
    }
    Ok(())
}

fn admission_is_usable(evidence: &Admission, start: i64, end: i64) -> bool {
    if evidence.source != SILERO_SOURCE
        || evidence.model_revision != SILERO_REVISION
        || evidence.start_sample != start
        || evidence.end_sample != end
        || !evidence.complete
        || !matches!(
            evidence.decision,
            SpeechDecision::Speech | SpeechDecision::NoSpeech | SpeechDecision::Uncertain
        )
    {
        return false;
    }
    let Some(probability) = evidence.maximum_probability else {
        return false;
    };
    if !probability.is_finite() || !(0.0..=1.0).contains(&probability) {
        return false;
    }
    if matches!(
        evidence.decision,
        SpeechDecision::NoSpeech | SpeechDecision::Uncertain
    ) && probability >= speech::ONSET_PROBABILITY
    {
        return false;
    }
    regions_fit(&evidence.speech_regions, start, end)
        && (evidence.decision == SpeechDecision::Speech) == !evidence.speech_regions.is_empty()
}

fn regions_fit(regions: &[SampleSpan], start: i64, end: i64) -> bool {
    let mut cursor = start;
    for region in regions {
        if region.start_sample < cursor
            || region.start_sample >= region.end_sample
            || region.end_sample > end
        {
            return false;
        }
        cursor = region.end_sample;
    }
    true
}

fn rejected_pending(start: i64, end: i64, evidence_error: bool) -> Admission {
    Admission {
        source: SILERO_SOURCE.to_owned(),
        model_revision: SILERO_REVISION.to_owned(),
        start_sample: start,
        end_sample: end,
        observed_until_sample: start,
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
        evidence_error,
    }
}

fn guarded_admit(evidence: &mut dyn AcousticEvidence, start: i64, end: i64) -> Admission {
    match evidence.admit(start, end) {
        Ok(admission) => accept_admission(admission, start, end),
        Err(_) => rejected_pending(start, end, true),
    }
}

fn blank_detection(mode: DetectionMode, reason: DetectionReason) -> LanguageDetection {
    LanguageDetection {
        mode,
        reason,
        probability: None,
        margin: None,
        candidates: Vec::new(),
        detector_error: false,
        context_end_sample: None,
        context_evidence: None,
        probes: Vec::new(),
        context_witnesses: Vec::new(),
    }
}

fn blocked_probe(
    begin: usize,
    end: usize,
    mode: DetectionMode,
    language: Option<String>,
    blocked: BlockedSpeech,
) -> Probe {
    Probe {
        begin,
        end,
        language,
        detection: blank_detection(mode, DetectionReason::from_blocked(blocked.reason)),
        review: true,
        audio_state: Some(AudioState::from_name(blocked.audio_state)),
        pcm_evidence: blocked.pcm,
    }
}

fn insufficient_probe(
    begin: usize,
    end: usize,
    mode: DetectionMode,
    language: Option<String>,
) -> Probe {
    Probe {
        begin,
        end,
        language,
        detection: blank_detection(mode, DetectionReason::InsufficientSpeech),
        review: true,
        audio_state: Some(AudioState::InsufficientAcousticContext),
        pcm_evidence: None,
    }
}

fn override_probe(begin: usize, end: usize, language: String) -> Probe {
    Probe {
        begin,
        end,
        language: Some(language),
        detection: blank_detection(DetectionMode::Manual, DetectionReason::Override),
        review: false,
        audio_state: None,
        pcm_evidence: None,
    }
}

fn detector_error_probe(begin: usize, end: usize) -> Probe {
    let mut detection = blank_detection(DetectionMode::Auto, DetectionReason::NeedsLanguage);
    detection.detector_error = true;
    Probe {
        begin,
        end,
        language: None,
        detection,
        review: true,
        audio_state: None,
        pcm_evidence: None,
    }
}

fn load_detected(
    detector: &mut dyn LanguageDetector,
    memo: Option<&mut ExactMemo>,
    scope: &MemoScope,
    epoch: u64,
    pcm: &[f32],
    start: i64,
) -> Result<Vec<(String, f64)>, ()> {
    let Some(memo) = memo else {
        let scores = detector.detect(pcm).map_err(|_| ())?;
        memo::validate_language_scores(&scores).map_err(|_| ())?;
        return Ok(scores);
    };
    let key = memo::probe_key_f32(epoch, start as u64, pcm);
    memo.probe(scope, key, || {
        detector.detect(pcm).map_err(|_| memo::MemoError::Detector)
    })
    .map_err(|_| ())
}

fn note_window(
    context: &mut Option<LanguageContext>,
    automatic: bool,
    probe: &mut Probe,
    absolute_begin: i64,
) {
    if !automatic {
        return;
    }
    match probe.detection.reason {
        DetectionReason::Unsupported => *context = None,
        DetectionReason::Detected => {
            let differs = context
                .as_ref()
                .is_some_and(|ctx| Some(ctx.language.as_str()) != probe.language.as_deref());
            if differs {
                *context = None;
            }
        }
        DetectionReason::BestEffort | DetectionReason::NeedsLanguage => {
            if let Some(ctx) = fresh_context(context, absolute_begin) {
                apply_rescue(&ctx, probe);
            }
        }
        _ => {}
    }
}

fn rescue_follower(
    context: &Option<LanguageContext>,
    automatic: bool,
    probe: &mut Probe,
    absolute_begin: i64,
) {
    if !automatic
        || !matches!(
            probe.detection.reason,
            DetectionReason::BestEffort | DetectionReason::NeedsLanguage
        )
    {
        return;
    }
    if let Some(ctx) = fresh_context(context, absolute_begin) {
        apply_rescue(&ctx, probe);
    }
}

fn fresh_context(context: &Option<LanguageContext>, start_sample: i64) -> Option<LanguageContext> {
    let ctx = context.as_ref()?;
    let delta = start_sample - ctx.end_sample;
    let main = ctx.evidence.as_ref().is_some_and(|e| {
        matches!(
            e.scope,
            "speaker_main_language" | "meeting_main_language" | "same_anchor_direct_windows"
        ) && e.anchors.len() >= 2
    });
    if main || (0..=RECENT_CONTEXT_SAMPLES).contains(&delta) {
        Some(ctx.clone())
    } else {
        None
    }
}

fn apply_rescue(context: &LanguageContext, probe: &mut Probe) {
    probe.language = Some(context.language.clone());
    probe.detection.reason = DetectionReason::RecentContext;
    probe.detection.context_end_sample = Some(context.end_sample);
    probe.detection.context_evidence = context.evidence.clone();
}

fn snapshot(probe: &Probe, start_sample: i64) -> ProbeEvent {
    let mut decision = probe.detection.clone();
    decision.probes.clear();
    ProbeEvent {
        start_sample: start_sample + probe.begin as i64,
        end_sample: start_sample + probe.end as i64,
        language: probe.language.clone(),
        decision,
        review: probe.review,
    }
}

struct Merge<'a> {
    context: &'a Option<LanguageContext>,
    probes: &'a mut [Probe],
    window: usize,
    index: &'a mut usize,
    events: &'a mut Vec<ProbeEvent>,
    opts: &'a TranscribeOptions,
    sample_rate: u32,
    automatic: bool,
}

fn absorb_followers(merge: &mut Merge<'_>) {
    while *merge.index < merge.probes.len() {
        let absolute = merge.opts.start_sample + merge.probes[*merge.index].begin as i64;
        rescue_follower(
            merge.context,
            merge.automatic,
            &mut merge.probes[*merge.index],
            absolute,
        );
        if !can_merge(
            &merge.probes[merge.window],
            &merge.probes[*merge.index],
            merge.opts.max_asr_seconds,
            merge.sample_rate,
        ) {
            break;
        }
        let follower_end = merge.probes[*merge.index].end;
        let follower_review = merge.probes[*merge.index].review;
        let follower_rank = rank_probability(&merge.probes[*merge.index]);
        merge.events.push(snapshot(
            &merge.probes[*merge.index],
            merge.opts.start_sample,
        ));
        merge.probes[merge.window].end = follower_end;
        merge.probes[merge.window].review |= follower_review;
        if follower_rank < rank_probability(&merge.probes[merge.window]) {
            let mut replacement = merge.probes[*merge.index].detection.clone();
            replacement.probes.clear();
            merge.probes[merge.window].detection = replacement;
        }
        *merge.index += 1;
    }
}

fn can_merge(window: &Probe, following: &Probe, max_seconds: f64, sample_rate: u32) -> bool {
    if window.language.is_none() || following.language != window.language {
        return false;
    }
    if window.audio_state.is_some() || following.audio_state.is_some() {
        return false;
    }
    if window.detection.reason == DetectionReason::Unsupported
        || following.detection.reason == DetectionReason::Unsupported
    {
        return false;
    }
    let span = following.end as f64 - window.begin as f64;
    span <= max_seconds * f64::from(sample_rate)
}

fn rank_probability(probe: &Probe) -> f64 {
    probe.detection.probability.unwrap_or(1.0)
}

fn should_recognize(window: &Probe, decision: SpeechDecision) -> bool {
    window.language.is_some()
        && decision == SpeechDecision::Speech
        && window.detection.reason != DetectionReason::InsufficientSpeech
}

fn acoustic_review(window: &Probe) -> Option<TranscriptionReview> {
    if window.audio_state != Some(AudioState::InsufficientAcousticContext) {
        return None;
    }
    Some(TranscriptionReview {
        reason: "insufficient_acoustic_context",
        partial_text: false,
        valid_feature_frames: Some(((window.end - window.begin) / 160) as i64),
        decoder_reason: None,
    })
}

fn recognize_pcm(
    audio: &[f32],
    window: &Probe,
    opts: &TranscribeOptions,
) -> (Vec<f32>, Option<CoherePadding>) {
    let left = if window.begin == 0 {
        opts.asr_padding.0
    } else {
        0
    };
    let right = if window.end == audio.len() {
        opts.asr_padding.1
    } else {
        0
    };
    let pcm = pad_pcm(&audio[window.begin..window.end], left, right);
    if left == 0 && right == 0 {
        return (pcm, None);
    }
    let padding = CoherePadding {
        policy: PAD_POLICY,
        leading_samples: left,
        trailing_samples: right,
        physical_start_sample: opts.start_sample + window.begin as i64,
        physical_end_sample: opts.start_sample + window.end as i64,
    };
    (pcm, Some(padding))
}

fn pad_pcm(pcm: &[f32], left: u32, right: u32) -> Vec<f32> {
    if left == 0 && right == 0 {
        return pcm.to_vec();
    }
    let mut padded = Vec::with_capacity(pcm.len() + left as usize + right as usize);
    padded.extend(std::iter::repeat_n(0.0, left as usize));
    padded.extend_from_slice(pcm);
    padded.extend(std::iter::repeat_n(0.0, right as usize));
    padded
}

fn asr_review(output: &AsrOutput, text: &str, samples: usize) -> Option<TranscriptionReview> {
    if !text.is_empty() && samples < SHORT_CONTEXT_SAMPLES {
        return Some(TranscriptionReview {
            reason: "short_acoustic_context",
            partial_text: output.token_count >= TOKEN_LIMIT,
            valid_feature_frames: None,
            decoder_reason: (output.token_count >= TOKEN_LIMIT).then_some("token_limit"),
        });
    }
    if output.token_count >= TOKEN_LIMIT {
        return Some(TranscriptionReview {
            reason: "token_limit",
            partial_text: true,
            valid_feature_frames: None,
            decoder_reason: None,
        });
    }
    if text.is_empty() {
        return Some(TranscriptionReview {
            reason: "empty_result",
            partial_text: false,
            valid_feature_frames: None,
            decoder_reason: None,
        });
    }
    None
}

fn update_context(
    context: &mut Option<LanguageContext>,
    events: &[ProbeEvent],
    text: &str,
    review: Option<&TranscriptionReview>,
) {
    let success = !text.is_empty() && review.is_none();
    for event in events {
        match event.decision.reason {
            DetectionReason::Unsupported => *context = None,
            DetectionReason::Detected => {
                if success {
                    if let Some(language) = &event.language {
                        *context = Some(LanguageContext {
                            language: language.clone(),
                            end_sample: event.end_sample,
                            evidence: None,
                        });
                    }
                } else if context
                    .as_ref()
                    .is_some_and(|ctx| Some(ctx.language.as_str()) != event.language.as_deref())
                {
                    *context = None;
                }
            }
            _ => {}
        }
    }
}

#[cfg(test)]
mod tests {
    use std::collections::VecDeque;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::{Arc, Mutex};

    use super::*;
    use crate::limits;
    use crate::messages;
    use crate::speech::{self, SpeechLedger};

    struct ScriptDetector {
        calls: Arc<AtomicUsize>,
        scores: VecDeque<Result<Vec<(String, f64)>, RouteError>>,
    }

    impl LanguageDetector for ScriptDetector {
        fn detect(&mut self, _pcm: &[f32]) -> Result<Vec<(String, f64)>, RouteError> {
            self.calls.fetch_add(1, Ordering::Relaxed);
            self.scores.pop_front().unwrap_or(Err(RouteError::Detector))
        }
    }

    struct ScriptAsr {
        calls: Arc<AtomicUsize>,
        pcm: Arc<Mutex<Vec<Vec<f32>>>>,
        results: VecDeque<Result<AsrOutput, RouteError>>,
    }

    impl Recognizer for ScriptAsr {
        fn transcribe(&mut self, pcm: &[f32], language: &str) -> Result<AsrOutput, RouteError> {
            self.calls.fetch_add(1, Ordering::Relaxed);
            self.pcm
                .lock()
                .unwrap_or_else(|err| err.into_inner())
                .push(pcm.to_vec());
            if let Some(result) = self.results.pop_front() {
                return result;
            }
            Ok(AsrOutput {
                text: format!("  words {language}  "),
                token_count: 1,
            })
        }
    }

    struct Rig {
        router: SpeechTranscriber<'static>,
        detects: Arc<AtomicUsize>,
        asrs: Arc<AtomicUsize>,
        pcm: Arc<Mutex<Vec<Vec<f32>>>>,
    }

    fn rig(
        language: &str,
        scores: Vec<Vec<(String, f64)>>,
        decision: SpeechDecision,
        probability: f64,
        context: Option<LanguageContext>,
        asr: Vec<Result<AsrOutput, RouteError>>,
    ) -> Rig {
        let detects = Arc::new(AtomicUsize::new(0));
        let asrs = Arc::new(AtomicUsize::new(0));
        let pcm = Arc::new(Mutex::new(Vec::new()));
        let router = SpeechTranscriber::new(
            language,
            Box::new(FixedEvidence {
                decision,
                probability,
            }),
            Some(Box::new(ScriptDetector {
                calls: Arc::clone(&detects),
                scores: scores.into_iter().map(Ok).collect(),
            })),
            Box::new(ScriptAsr {
                calls: Arc::clone(&asrs),
                pcm: Arc::clone(&pcm),
                results: asr.into(),
            }),
            context,
        )
        .unwrap();
        Rig {
            router,
            detects,
            asrs,
            pcm,
        }
    }

    fn speech_rig(scores: Vec<Vec<(String, f64)>>, context: Option<LanguageContext>) -> Rig {
        rig(
            "auto",
            scores,
            SpeechDecision::Speech,
            0.8,
            context,
            Vec::new(),
        )
    }

    fn pair(winner: &str, probability: f64, runner: &str) -> Vec<(String, f64)> {
        vec![
            (winner.to_owned(), probability),
            (runner.to_owned(), 1.0 - probability),
        ]
    }

    fn names(items: &[&str]) -> Vec<String> {
        items.iter().map(|item| (*item).to_owned()).collect()
    }

    fn varying(n: usize) -> Vec<f32> {
        (0..n)
            .map(|index| if index % 2 == 0 { 0.1 } else { -0.1 })
            .collect()
    }

    fn speaker() -> Vec<String> {
        names(&["speaker_0"])
    }

    #[test]
    fn thresholds_reasons_and_messages_do_not_drift() {
        assert_eq!(limits::LANGUAGE_AUTO_THRESHOLD, 0.90);
        assert_eq!(limits::LANGUAGE_MARGIN_THRESHOLD, 0.20);
        assert_eq!(limits::PASSAGE_RETRY_MAX_SECONDS, 24.5);
        assert_eq!(limits::SAMPLE_RATE, 16_000);
        assert_eq!(RECENT_CONTEXT_SECONDS, 60);
        assert_eq!(RECENT_CONTEXT_SAMPLES, 60 * 16_000);
        assert_eq!(PROBE_SECONDS, 3.0);
        assert_eq!(MIN_COHERE_SAMPLES, 320);
        assert_eq!(SHORT_CONTEXT_SAMPLES, 3_200);
        assert_eq!(CANONICAL_PAD_SAMPLES, 3_200);
        assert_eq!(TOKEN_LIMIT, 448);
        assert!(is_confident_detection(0.90, 0.20));
        assert!(!is_confident_detection(0.899, 0.20));
        assert!(!is_confident_detection(0.90, 0.199));
        assert!(is_confident_detection(
            limits::LANGUAGE_AUTO_THRESHOLD,
            limits::LANGUAGE_MARGIN_THRESHOLD
        ));
        assert!(!is_confident_detection(
            limits::LANGUAGE_AUTO_THRESHOLD,
            limits::LANGUAGE_MARGIN_THRESHOLD - 0.001
        ));
        assert_eq!(DetectionReason::NeedsLanguage.as_str(), "needs_language");
        assert_eq!(DetectionReason::Unsupported.as_str(), "unsupported");
        assert_eq!(DetectionReason::Detected.as_str(), "detected");
        assert_eq!(DetectionReason::BestEffort.as_str(), "best_effort");
        assert_eq!(DetectionReason::RecentContext.as_str(), "recent_context");
        assert_eq!(
            DetectionReason::InsufficientSpeech.as_str(),
            "insufficient_speech"
        );
        assert_eq!(
            DetectionReason::SpeechAdmissionUncertain.as_str(),
            "speech_admission_uncertain"
        );
        assert_eq!(
            DetectionReason::SpeechEvidencePending.as_str(),
            "speech_evidence_pending"
        );
        assert_eq!(DetectionReason::Override.as_str(), "override");
        assert_eq!(AudioState::DigitalSilence.as_str(), "digital_silence");
        assert_eq!(AudioState::ConstantSignal.as_str(), "constant_signal");
        assert_eq!(AudioState::ModelNonSpeech.as_str(), "model_non_speech");
        assert_eq!(
            AudioState::SpeechEvidencePending.as_str(),
            "speech_evidence_pending"
        );
        assert_eq!(
            messages::LID_MODEL_MISSING,
            "Automatic language detection needs its local model. Download models in Settings, or choose a language override."
        );
        assert_eq!(
            messages::CHOOSE_LANGUAGE_MODE,
            "Choose a supported language mode."
        );
        assert_eq!(detector_issues(false), &[messages::LID_MODEL_MISSING]);
        assert_eq!(
            detector_issues(false)[0],
            "Automatic language detection needs its local model. Download models in Settings, or choose a language override."
        );
        assert_eq!(
            LanguageChoice::parse("nope").unwrap_err().to_string(),
            messages::CHOOSE_LANGUAGE_MODE
        );
        assert_eq!(
            LanguageChoice::parse("ru").unwrap_err().to_string(),
            "Choose a supported language mode."
        );
    }

    #[test]
    fn language_summary_distinguishes_verified_non_speech_from_unknown_speech() {
        for case in ["digital", "constant", "model", "unknown"] {
            let mut spoken = speech_rig(vec![pair("en", 0.99, "de")], None);
            let speech = spoken
                .router
                .transcribe(
                    &varying(3 * 16000),
                    16000,
                    &speaker(),
                    TranscribeOptions {
                        start_sample: 7 * 16000,
                        ..Default::default()
                    },
                )
                .unwrap();
            let decision = match case {
                "model" => SpeechDecision::NoSpeech,
                "unknown" => SpeechDecision::Uncertain,
                _ => SpeechDecision::Speech,
            };
            let mut tail = rig(
                "auto",
                Vec::new(),
                decision,
                if decision == SpeechDecision::Speech {
                    0.9
                } else {
                    0.1
                },
                None,
                Vec::new(),
            );
            let pcm = match case {
                "digital" => vec![0.; 3 * 16000],
                "constant" => vec![0.1; 3 * 16000],
                _ => varying(3 * 16000),
            };
            let silent = tail
                .router
                .transcribe(
                    &pcm,
                    16000,
                    &speaker(),
                    TranscribeOptions {
                        start_sample: 10 * 16000,
                        ..Default::default()
                    },
                )
                .unwrap();
            assert!(silent.iter().all(|row| row.text.is_empty()));
            assert_eq!(tail.detects.load(Ordering::Relaxed), 0);
            let mut events = Vec::new();
            append_passage_decisions(&mut events, &speech, 7 * 16000);
            append_passage_decisions(&mut events, &silent, 10 * 16000);
            let summary = summarize_decisions(events).unwrap();
            if case == "unknown" {
                assert!(summary.language.is_none());
                assert!(summary.review);
                assert_eq!(
                    summary.detection.reason,
                    DetectionReason::SpeechAdmissionUncertain
                );
                assert_eq!(summary.detection.probes.len(), 2);
            } else {
                assert_eq!(summary.language.as_deref(), Some("en"), "{case}");
                assert!(!summary.review, "{case}");
                assert_eq!(summary.detection.reason, DetectionReason::Detected);
                assert_eq!(summary.detection.probes.len(), 1);
            }
        }
    }

    #[test]
    fn policy_uses_contract_thresholds_for_supported_and_unsupported() {
        let mut policy = LanguagePolicy::new();
        let detected = policy.decide(&pair("en", 0.90, "fr"), &speaker()).unwrap();
        assert_eq!(detected.detection.reason, DetectionReason::Detected);
        assert_eq!(detected.language.as_deref(), Some("en"));
        assert!(!detected.review);
        assert_eq!(policy.remembered(&speaker()), Some("en"));

        let weak = policy.decide(&pair("fr", 0.89, "en"), &speaker()).unwrap();
        assert_eq!(weak.detection.reason, DetectionReason::BestEffort);
        assert!(weak.review);
        assert_eq!(weak.detection.candidates[0].language, "fr");

        let unsupported = policy.decide(&pair("ru", 0.90, "en"), &speaker()).unwrap();
        assert_eq!(unsupported.detection.reason, DetectionReason::Unsupported);
        assert!(unsupported.language.is_none());
        assert!(policy.remembered(&speaker()).is_none());

        let abstain = policy.decide(&pair("ru", 0.51, "fr"), &speaker()).unwrap();
        assert_eq!(abstain.detection.reason, DetectionReason::NeedsLanguage);
        assert!(abstain.language.is_none());
        assert_eq!(
            policy
                .decide(&[("en".into(), f64::NAN)], &speaker())
                .unwrap_err()
                .to_string(),
            "Invalid language detection probabilities."
        );
        assert!(
            policy
                .decide(&[("en".into(), 0.8), ("fr".into(), 0.8)], &speaker())
                .is_err()
        );
        assert!(policy.decide(&[], &speaker()).is_err());
    }

    #[test]
    fn confident_switch_merges_only_the_new_language() {
        let mut rig = speech_rig(
            vec![
                pair("en", 0.97, "fr"),
                pair("fr", 0.92, "en"),
                pair("fr", 0.92, "en"),
            ],
            None,
        );
        let rows = rig
            .router
            .transcribe(
                &varying(9 * 16_000),
                16_000,
                &speaker(),
                TranscribeOptions::default(),
            )
            .unwrap();
        assert_eq!(
            rows.iter()
                .map(|row| (row.start, row.end, row.language.as_deref()))
                .collect::<Vec<_>>(),
            vec![(0.0, 3.0, Some("en")), (3.0, 9.0, Some("fr"))]
        );
        assert!(rows.iter().all(|row| !row.review && !row.text.is_empty()));
        assert_eq!(rig.asrs.load(Ordering::Relaxed), 2);
    }

    #[test]
    fn recent_context_rescues_weak_supported_and_unsupported_scores() {
        let mut rig = speech_rig(
            vec![
                pair("en", 0.97, "fr"),
                pair("fr", 0.65, "en"),
                pair("ru", 0.55, "fr"),
            ],
            None,
        );
        let rows = rig
            .router
            .transcribe(
                &varying(9 * 16_000),
                16_000,
                &speaker(),
                TranscribeOptions::default(),
            )
            .unwrap();
        assert_eq!(
            rows.iter()
                .map(|row| row.language.as_deref())
                .collect::<Vec<_>>(),
            vec![Some("en"), Some("en")]
        );
        assert_eq!(
            rows.iter()
                .map(|row| row.language_detection.reason)
                .collect::<Vec<_>>(),
            vec![DetectionReason::Detected, DetectionReason::RecentContext]
        );
        assert_eq!(rows[1].language_detection.candidates[0].language, "ru");
        assert_eq!(rows[1].language_detection.candidates[0].probability, 0.55);
        let probes = &rows[1].language_detection.probes;
        assert_eq!(probes[0].decision.candidates[0].language, "fr");
        assert_eq!(probes[0].decision.candidates[0].probability, 0.65);
        assert_eq!(
            probes
                .iter()
                .map(|probe| (probe.start_sample, probe.end_sample))
                .collect::<Vec<_>>(),
            vec![(48_000, 96_000), (96_000, 144_000)]
        );
        assert_eq!(probes[1].decision.candidates[0].language, "ru");
        assert!(rows[1].review && rows[1].language_review);
        assert_eq!(
            rig.router.context(),
            Some(&LanguageContext {
                language: "en".to_owned(),
                end_sample: 3 * 16_000,
                evidence: None,
            })
        );
    }

    #[test]
    fn best_effort_without_context_and_unsupported_are_not_forced_english() {
        let mut best = speech_rig(vec![pair("fr", 0.51, "en")], None);
        let row = best
            .router
            .transcribe(
                &varying(16_000),
                16_000,
                &speaker(),
                TranscribeOptions::default(),
            )
            .unwrap()
            .remove(0);
        assert_eq!(row.language.as_deref(), Some("fr"));
        assert_eq!(row.text, "words fr");
        assert_eq!(row.language_detection.reason, DetectionReason::BestEffort);
        assert!(row.review);
        assert!(best.router.context().is_none());

        for (scores, reason) in [
            (pair("ru", 0.97, "en"), DetectionReason::Unsupported),
            (pair("ru", 0.51, "fr"), DetectionReason::NeedsLanguage),
        ] {
            let mut rig = speech_rig(vec![scores], None);
            let row = rig
                .router
                .transcribe(
                    &varying(16_000),
                    16_000,
                    &speaker(),
                    TranscribeOptions::default(),
                )
                .unwrap()
                .remove(0);
            assert!(row.language.is_none());
            assert!(row.text.is_empty());
            assert_eq!(row.language_detection.reason, reason);
            assert_eq!(rig.asrs.load(Ordering::Relaxed), 0);
        }
    }

    #[test]
    fn clear_unsupported_drops_context_before_the_next_probe() {
        let mut rig = speech_rig(
            vec![pair("ru", 0.97, "en"), pair("fr", 0.51, "en")],
            Some(LanguageContext {
                language: "en".to_owned(),
                end_sample: 0,
                evidence: None,
            }),
        );
        let rows = rig
            .router
            .transcribe(
                &varying(6 * 16_000),
                16_000,
                &speaker(),
                TranscribeOptions::default(),
            )
            .unwrap();
        assert_eq!(
            rows.iter()
                .map(|row| row.language_detection.reason)
                .collect::<Vec<_>>(),
            vec![DetectionReason::Unsupported, DetectionReason::BestEffort]
        );
        assert_eq!(rows[1].language.as_deref(), Some("fr"));
        assert!(rig.router.context().is_none());
    }

    #[test]
    fn confident_switch_clears_context_before_a_weak_follower_can_reuse_it() {
        let mut rig = speech_rig(
            vec![pair("fr", 0.97, "en"), pair("fr", 0.60, "en")],
            Some(LanguageContext {
                language: "en".to_owned(),
                end_sample: 0,
                evidence: None,
            }),
        );
        let rows = rig
            .router
            .transcribe(
                &varying(6 * 16_000),
                16_000,
                &speaker(),
                TranscribeOptions::default(),
            )
            .unwrap();
        assert_eq!(rows.len(), 1);
        assert_eq!(rows[0].language.as_deref(), Some("fr"));
        assert_eq!(
            rows[0].language_detection.reason,
            DetectionReason::BestEffort
        );
        assert_eq!(
            rows[0].language_detection.probes[0].decision.reason,
            DetectionReason::Detected
        );
        assert_eq!(
            rows[0].language_detection.probes[1].decision.reason,
            DetectionReason::BestEffort
        );
        assert_eq!(
            rig.router.context(),
            Some(&LanguageContext {
                language: "fr".to_owned(),
                end_sample: 3 * 16_000,
                evidence: None,
            })
        );
    }

    #[test]
    fn context_is_inclusive_at_sixty_seconds_and_ignores_the_future() {
        for (start, reason, language) in [
            (16_000, DetectionReason::RecentContext, "fr"),
            (61 * 16_000, DetectionReason::RecentContext, "fr"),
            (61 * 16_000 + 1, DetectionReason::BestEffort, "en"),
            (0, DetectionReason::BestEffort, "en"),
        ] {
            let mut rig = speech_rig(
                vec![pair("en", 0.60, "fr")],
                Some(LanguageContext {
                    language: "fr".to_owned(),
                    end_sample: 16_000,
                    evidence: None,
                }),
            );
            let row = rig
                .router
                .transcribe(
                    &varying(16_000),
                    16_000,
                    &names(&["unknown"]),
                    TranscribeOptions {
                        start_sample: start,
                        ..TranscribeOptions::default()
                    },
                )
                .unwrap()
                .remove(0);
            assert_eq!(row.language_detection.reason, reason, "start {start}");
            assert_eq!(row.language.as_deref(), Some(language), "start {start}");
        }
    }

    #[test]
    fn blank_failed_and_partial_asr_never_establish_context() {
        let cases = [
            Ok(AsrOutput {
                text: "   ".to_owned(),
                token_count: 1,
            }),
            Err(RouteError::Recognition),
            Ok(AsrOutput {
                text: "Partial".to_owned(),
                token_count: 448,
            }),
        ];
        for first in cases {
            let mut rig = rig(
                "auto",
                vec![pair("fr", 0.97, "en"), pair("en", 0.60, "fr")],
                SpeechDecision::Speech,
                0.8,
                None,
                vec![first],
            );
            let rows = rig
                .router
                .transcribe(
                    &varying(6 * 16_000),
                    16_000,
                    &speaker(),
                    TranscribeOptions::default(),
                )
                .unwrap();
            assert_eq!(
                rows[1].language_detection.reason,
                DetectionReason::BestEffort
            );
            assert!(rig.router.context().is_none());
            assert!(rows[0].transcription_review.is_some());
        }
    }

    #[test]
    fn failed_confident_switch_cannot_reuse_contradicted_context() {
        let mut rig = rig(
            "auto",
            vec![pair("fr", 0.97, "en"), pair("en", 0.60, "fr")],
            SpeechDecision::Speech,
            0.8,
            Some(LanguageContext {
                language: "en".to_owned(),
                end_sample: 0,
                evidence: None,
            }),
            vec![Err(RouteError::Recognition)],
        );
        let rows = rig
            .router
            .transcribe(
                &varying(6 * 16_000),
                16_000,
                &speaker(),
                TranscribeOptions::default(),
            )
            .unwrap();
        assert_eq!(
            rows[0].transcription_review.as_ref().unwrap().reason,
            "transcription_failed"
        );
        assert_eq!(
            rows[1].language_detection.reason,
            DetectionReason::BestEffort
        );
        assert!(rig.router.context().is_none());
    }

    #[test]
    fn detector_failure_can_use_context_without_inventing_scores() {
        let mut rig = rig(
            "auto",
            Vec::new(),
            SpeechDecision::Speech,
            0.8,
            Some(LanguageContext {
                language: "fr".to_owned(),
                end_sample: 0,
                evidence: None,
            }),
            Vec::new(),
        );
        let row = rig
            .router
            .transcribe(
                &varying(16_000),
                16_000,
                &names(&["unknown"]),
                TranscribeOptions::default(),
            )
            .unwrap()
            .remove(0);
        assert_eq!(row.text, "words fr");
        assert!(row.review);
        assert_eq!(row.language_detection.mode, DetectionMode::Auto);
        assert_eq!(
            row.language_detection.reason,
            DetectionReason::RecentContext
        );
        assert!(row.language_detection.detector_error);
        assert_eq!(row.language_detection.context_end_sample, Some(0));
        assert!(row.language_detection.probability.is_none());
        assert!(row.language_detection.candidates.is_empty());
    }

    #[test]
    fn manual_override_skips_detection_and_marks_short_or_overlap() {
        let mut rig = rig(
            "fr",
            vec![pair("en", 0.99, "de")],
            SpeechDecision::Speech,
            0.8,
            None,
            Vec::new(),
        );
        let short = rig
            .router
            .transcribe(
                &varying(1_000),
                16_000,
                &names(&["a", "b"]),
                TranscribeOptions::default(),
            )
            .unwrap()
            .remove(0);
        assert_eq!(short.language.as_deref(), Some("fr"));
        assert_eq!(short.language_detection.reason, DetectionReason::Override);
        assert!(!short.language_review);
        assert!(short.review);
        assert_eq!(short.audio_state, Some(AudioState::ShortAcousticContext));
        assert_eq!(
            short.transcription_review.unwrap().reason,
            "short_acoustic_context"
        );
        assert_eq!(rig.detects.load(Ordering::Relaxed), 0);

        let long = rig
            .router
            .transcribe(
                &varying(24 * 16_000),
                16_000,
                &names(&["a", "b"]),
                TranscribeOptions::default(),
            )
            .unwrap()
            .remove(0);
        assert_eq!(
            (long.start, long.end, long.language.as_deref()),
            (0.0, 24.0, Some("fr"))
        );
        assert!(long.review && !long.language_review);
        assert!(long.audio_state.is_none());
        assert_eq!(rig.detects.load(Ordering::Relaxed), 0);
    }

    #[test]
    fn zero_constant_pending_and_uncertain_pcm_do_not_reach_asr() {
        let zeros = vec![0.0; 16_000];
        let mut silent = rig(
            "auto",
            Vec::new(),
            SpeechDecision::NoSpeech,
            0.01,
            None,
            Vec::new(),
        );
        let row = silent
            .router
            .transcribe(&zeros, 16_000, &speaker(), TranscribeOptions::default())
            .unwrap()
            .remove(0);
        assert!(row.language.is_none());
        assert_eq!(
            row.language_detection.reason,
            DetectionReason::InsufficientSpeech
        );
        assert_eq!(row.audio_state, Some(AudioState::ModelNonSpeech));
        assert_eq!(
            row.pcm_evidence.as_ref().unwrap().kind,
            "exact_digital_silence"
        );
        assert_eq!(silent.asrs.load(Ordering::Relaxed), 0);
        assert_eq!(silent.detects.load(Ordering::Relaxed), 0);

        let mut inherited = rig(
            "auto",
            Vec::new(),
            SpeechDecision::Speech,
            0.8,
            None,
            Vec::new(),
        );
        let row = inherited
            .router
            .transcribe(&zeros, 16_000, &speaker(), TranscribeOptions::default())
            .unwrap()
            .remove(0);
        assert_eq!(row.audio_state, Some(AudioState::DigitalSilence));
        assert_eq!(
            row.language_detection.reason,
            DetectionReason::InsufficientSpeech
        );
        assert_eq!(inherited.asrs.load(Ordering::Relaxed), 0);

        let constant = vec![0.04f32; 16_000];
        let mut dc = rig(
            "fr",
            Vec::new(),
            SpeechDecision::Speech,
            0.8,
            None,
            Vec::new(),
        );
        let row = dc
            .router
            .transcribe(&constant, 16_000, &speaker(), TranscribeOptions::default())
            .unwrap()
            .remove(0);
        assert_eq!(row.language.as_deref(), Some("fr"));
        assert_eq!(row.audio_state, Some(AudioState::ConstantSignal));
        assert_eq!(
            row.pcm_evidence.as_ref().unwrap().sample_value,
            Some(f64::from(0.04f32))
        );

        let mut uncertain = rig(
            "auto",
            Vec::new(),
            SpeechDecision::Uncertain,
            0.01,
            None,
            Vec::new(),
        );
        let row = uncertain
            .router
            .transcribe(
                &varying(16_000),
                16_000,
                &speaker(),
                TranscribeOptions::default(),
            )
            .unwrap()
            .remove(0);
        assert_eq!(
            row.language_detection.reason,
            DetectionReason::SpeechAdmissionUncertain
        );
        assert_eq!(row.audio_state, Some(AudioState::SpeechEvidencePending));

        let mut pending = rig(
            "auto",
            Vec::new(),
            SpeechDecision::Pending,
            0.01,
            None,
            Vec::new(),
        );
        let row = pending
            .router
            .transcribe(
                &varying(16_000),
                16_000,
                &speaker(),
                TranscribeOptions::default(),
            )
            .unwrap()
            .remove(0);
        assert_eq!(
            row.language_detection.reason,
            DetectionReason::SpeechEvidencePending
        );
        assert!(!row.acoustic_evidence.evidence_error);
        assert_eq!(pending.asrs.load(Ordering::Relaxed), 0);
    }

    #[test]
    fn ledger_constant_zero_blocks_asr_and_padding_stays_virtual() {
        let mut ledger = SpeechLedger::new(0).unwrap();
        ledger.append_pcm(0, &vec![0.0; 512], 0.99, 0.99).unwrap();
        let detects = Arc::new(AtomicUsize::new(0));
        let asrs = Arc::new(AtomicUsize::new(0));
        let pcm = Arc::new(Mutex::new(Vec::new()));
        let mut router = SpeechTranscriber::new(
            "en",
            Box::new(ledger),
            Some(Box::new(ScriptDetector {
                calls: Arc::clone(&detects),
                scores: VecDeque::new(),
            })),
            Box::new(ScriptAsr {
                calls: Arc::clone(&asrs),
                pcm: Arc::clone(&pcm),
                results: VecDeque::new(),
            }),
            None,
        )
        .unwrap();
        let row = router
            .transcribe(
                &vec![0.0; 512],
                16_000,
                &speaker(),
                TranscribeOptions::default(),
            )
            .unwrap()
            .remove(0);
        assert_eq!(row.audio_state, Some(AudioState::ModelNonSpeech));
        assert_eq!(row.acoustic_evidence.decision, SpeechDecision::NoSpeech);
        assert_eq!(asrs.load(Ordering::Relaxed), 0);

        let mut padded = speech_rig(vec![pair("en", 0.97, "fr")], None);
        let audio = varying(4_000);
        let row = padded
            .router
            .transcribe(
                &audio,
                16_000,
                &speaker(),
                TranscribeOptions {
                    asr_padding: (3_200, 3_200),
                    ..TranscribeOptions::default()
                },
            )
            .unwrap()
            .remove(0);
        let seen = padded.pcm.lock().unwrap_or_else(|err| err.into_inner());
        assert_eq!(seen[0].len(), 4_000 + 6_400);
        assert!(seen[0][..3_200].iter().all(|sample| *sample == 0.0));
        assert_eq!(&seen[0][3_200..7_200], &audio[..]);
        let padding = row.cohere_input_padding.unwrap();
        assert_eq!(padding.physical_end_sample, 4_000);
        assert_eq!(padding.leading_samples, 3_200);
        assert_eq!(padding.policy, PAD_POLICY);
    }

    #[test]
    fn shared_probe_memo_reroutes_the_same_scores_from_each_context() {
        let memo = ExactMemo::new(8, 4_096).unwrap();
        let pcm = varying(16_000);
        let mut first = speech_rig(
            vec![pair("en", 0.60, "fr")],
            Some(LanguageContext {
                language: "fr".to_owned(),
                end_sample: 16_000,
                evidence: None,
            }),
        );
        let mut shared = memo;
        let opts = TranscribeOptions {
            start_sample: 2 * 16_000,
            ..TranscribeOptions::default()
        };
        let left = first
            .router
            .transcribe_sharing(&pcm, 16_000, &speaker(), opts, &mut shared)
            .unwrap()
            .remove(0);
        let mut second = speech_rig(
            vec![pair("de", 0.99, "en")],
            Some(LanguageContext {
                language: "en".to_owned(),
                end_sample: 16_000,
                evidence: None,
            }),
        );
        let right = second
            .router
            .transcribe_sharing(&pcm, 16_000, &names(&["speaker_1"]), opts, &mut shared)
            .unwrap()
            .remove(0);
        assert_eq!(first.detects.load(Ordering::Relaxed), 1);
        assert_eq!(second.detects.load(Ordering::Relaxed), 0);
        assert_eq!(
            (left.language.as_deref(), right.language.as_deref()),
            (Some("fr"), Some("en"))
        );
        assert_eq!(
            left.language_detection.reason,
            DetectionReason::RecentContext
        );
        assert_eq!(
            right.language_detection.reason,
            DetectionReason::RecentContext
        );
        assert_eq!(left.cohere_raw_text.as_deref(), Some("  words fr  "));
        assert_eq!(right.text, "words en");
    }

    #[test]
    fn probe_edges_match_numpy_linspace_and_stay_inside_six_seconds() {
        assert_eq!(
            probe_edges(100_001, 16_000, true),
            vec![0, 33_333, 66_667, 100_001]
        );
        assert_eq!(probe_edges(144_000, 16_000, false), vec![0, 144_000]);
        let mut rig = speech_rig(
            vec![
                pair("fr", 0.97, "en"),
                pair("fr", 0.97, "en"),
                pair("fr", 0.97, "en"),
            ],
            None,
        );
        let rows = rig
            .router
            .transcribe(
                &varying(100_001),
                16_000,
                &speaker(),
                TranscribeOptions::default(),
            )
            .unwrap();
        assert_eq!(rows[0].start, 0.0);
        assert_eq!(rows.last().unwrap().end, 100_001.0 / 16_000.0);
        let seen = rig.pcm.lock().unwrap_or_else(|err| err.into_inner());
        assert!(seen.iter().all(|chunk| chunk.len() <= 96_000));
        assert_eq!(seen.iter().map(Vec::len).sum::<usize>(), 100_001);
    }

    #[test]
    fn invalid_audio_context_and_bad_receipts_fail_closed() {
        let mut rig = speech_rig(Vec::new(), None);
        let audio_error = rig
            .router
            .transcribe(&[], 16_000, &speaker(), TranscribeOptions::default())
            .unwrap_err();
        assert_eq!(
            audio_error.to_string(),
            "Language routing requires finite 16 kHz mono speech audio and its sample position."
        );
        assert!(
            rig.router
                .transcribe(
                    &varying(100),
                    48_000,
                    &speaker(),
                    TranscribeOptions::default()
                )
                .is_err()
        );
        assert!(
            rig.router
                .transcribe(
                    &varying(100),
                    16_000,
                    &speaker(),
                    TranscribeOptions {
                        start_sample: -1,
                        ..TranscribeOptions::default()
                    },
                )
                .is_err()
        );
        assert!(
            rig.router
                .transcribe(
                    &varying(100),
                    16_000,
                    &speaker(),
                    TranscribeOptions {
                        asr_padding: (1, 0),
                        ..TranscribeOptions::default()
                    },
                )
                .is_err()
        );
        match SpeechTranscriber::new(
            "auto",
            Box::new(FixedEvidence {
                decision: SpeechDecision::Speech,
                probability: 0.8,
            }),
            None,
            Box::new(ScriptAsr {
                calls: Arc::new(AtomicUsize::new(0)),
                pcm: Arc::new(Mutex::new(Vec::new())),
                results: VecDeque::new(),
            }),
            Some(LanguageContext {
                language: "ru".to_owned(),
                end_sample: 0,
                evidence: None,
            }),
        ) {
            Err(error) => assert_eq!(error.to_string(), "Invalid preceding language context."),
            Ok(_) => panic!("invalid context was accepted"),
        }

        let mut hot = fixed_admission(0, 100, SpeechDecision::NoSpeech, speech::ONSET_PROBABILITY);
        assert_eq!(
            accept_admission(hot.clone(), 0, 100).decision,
            SpeechDecision::Pending
        );
        hot.maximum_probability = Some(speech::ONSET_PROBABILITY - 0.01);
        assert_eq!(
            accept_admission(hot, 0, 100).decision,
            SpeechDecision::NoSpeech
        );
        let mut drifted = fixed_admission(0, 100, SpeechDecision::Speech, 0.8);
        drifted.model_revision = "drift".into();
        let rejected = accept_admission(drifted, 0, 100);
        assert_eq!(rejected.decision, SpeechDecision::Pending);
        assert!(!rejected.evidence_error);

        let mut kept = speech_rig(
            Vec::new(),
            Some(LanguageContext {
                language: "fr".to_owned(),
                end_sample: 0,
                evidence: None,
            }),
        );
        kept.router.set_language("auto").unwrap();
        assert_eq!(kept.router.context().unwrap().language, "fr");
        kept.router.set_language("en").unwrap();
        assert!(kept.router.context().is_none());
        kept.router.set_language("auto").unwrap();
        assert!(kept.router.context().is_none());
    }
}
