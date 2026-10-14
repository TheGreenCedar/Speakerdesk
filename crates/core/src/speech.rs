//! Speech admission ledger and acoustic evidence rules.
//!
//! Frame decisions are pure functions of injected probabilities and PCM.

use thiserror::Error;

use crate::ipc::GainPolicy;
use crate::limits;
use crate::messages;

/// 16 kHz Silero branch frame length. Alias of [`limits::VAD_FRAME_SAMPLES`].
pub const FRAME_SAMPLES: usize = limits::VAD_FRAME_SAMPLES;

/// In-memory hot tail from the Python ledger (60 s). Frames are not spilled to disk.
pub const HOT_FRAMES: usize = 60 * limits::SAMPLE_RATE as usize / FRAME_SAMPLES;

/// Onset hysteresis. A single positive frame is retained; there is no minimum duration.
pub const ONSET_PROBABILITY: f64 = 0.5;
/// Release hysteresis. Values in `[RELEASE, ONSET)` keep the previous speaking state.
pub const RELEASE_PROBABILITY: f64 = 0.35;

/// Quiet-speech normalized view: `min(256, max(1, 0.25 / peak))`, or 1 when peak is 0.
pub const GAIN_TARGET: f64 = 0.25;
pub const GAIN_CAP: f64 = 256.0;

pub const INPUT_POLICY: &str = "raw_and_peak025_gaincap256_per512_constant_guard_v3";
pub const SOURCE_UNITY_POLICY: &str = "apple_active_microphone_tap_raw_and_unity_view_v1";
pub const PRODUCTION_UNITY_POLICY: &str =
    "qualified_apple_processed_microphone_raw_and_unity_view_v1";

pub const SILERO_SOURCE: &str = "silero_v6";
pub const SILERO_REVISION: &str = "2ebf4a5e10726a2e78ddd4d70eedfb6f1c33eb06";

pub const NEGATIVE_SIGNAL_STATE: &str = "model_non_speech";
pub const CLASSIFICATION_SCOPE: &str =
    "exact_constant_pcm_or_observed_dual_view_hysteresis_not_acoustic_certainty";

const REASON_INSUFFICIENT: &str = "insufficient_speech";
const REASON_UNCERTAIN: &str = "speech_admission_uncertain";
const REASON_PENDING: &str = "speech_evidence_pending";
const STATE_DIGITAL_SILENCE: &str = "digital_silence";
const STATE_CONSTANT: &str = "constant_signal";
const STATE_MODEL_NON_SPEECH: &str = "model_non_speech";
const STATE_PENDING: &str = "speech_evidence_pending";
const PCM_SILENCE: &str = "exact_digital_silence";
const PCM_CONSTANT: &str = "exact_constant_signal";

#[derive(Debug, Error, Clone, PartialEq, Eq)]
pub enum SpeechError {
    #[error("Invalid physical frame peak.")]
    InvalidPeak,
    #[error("Invalid or noncontiguous Silero evidence.")]
    InvalidEvidence,
    #[error("Invalid neural speech probability.")]
    InvalidProbability,
    #[error("Invalid speech-admission range.")]
    InvalidRange,
    #[error("Invalid speech-evidence origin.")]
    InvalidOrigin,
    #[error("Speech evidence requires contiguous finite mono audio.")]
    InvalidPcm,
    #[error("Invalid source conditioning identity.")]
    InvalidConditioning,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SpeechDecision {
    Pending,
    Uncertain,
    Speech,
    NoSpeech,
}

impl SpeechDecision {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Pending => "pending",
            Self::Uncertain => "uncertain",
            Self::Speech => "speech",
            Self::NoSpeech => "no_speech",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GainMode {
    /// Default quiet-speech boost. Unknown, raw, import, and system peers use this.
    QuietSpeech,
    /// Qualified microphone conditioning. Gain is 1 for every finite nonnegative peak.
    Unity,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ExactPcm {
    DigitalSilence,
    Constant,
}

#[derive(Debug, Clone, PartialEq)]
pub struct PcmEvidence {
    pub kind: &'static str,
    pub start_sample: i64,
    pub end_sample: i64,
    pub sample_value: Option<f64>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct BlockedSpeech {
    pub reason: &'static str,
    pub audio_state: &'static str,
    pub pcm: Option<PcmEvidence>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SampleSpan {
    pub start_sample: i64,
    pub end_sample: i64,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Observation {
    pub input_policy: String,
    pub raw_probability: f64,
    pub normalized_probability: f64,
    pub gain: f64,
    pub constant_value: Option<f64>,
    pub constant_pcm_override: bool,
    pub conditioning_sha256: Option<String>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Frame {
    pub start: i64,
    pub end: i64,
    pub probability: f64,
    pub speaking: bool,
    pub observation: Option<Observation>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Admission {
    pub source: String,
    pub model_revision: String,
    pub start_sample: i64,
    pub end_sample: i64,
    pub observed_until_sample: i64,
    pub complete: bool,
    pub decision: SpeechDecision,
    pub speech_regions: Vec<SampleSpan>,
    pub input_policy: Option<String>,
    pub conditioning_sha256: Option<String>,
    pub maximum_probability: Option<f64>,
    pub maximum_model_probability: Option<f64>,
    pub uncertain_regions: Vec<SampleSpan>,
    pub model_negative_regions: Vec<SampleSpan>,
    pub negative_signal_state: Option<String>,
    pub classification_scope: Option<String>,
    pub evidence_error: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct InspectionCounts {
    pub speech_samples: i64,
    pub uncertain_samples: i64,
    pub negative_constant_samples: i64,
    pub model_negative_samples: i64,
}

#[derive(Debug, Clone)]
pub struct SpeechLedger {
    origin: i64,
    end_sample: i64,
    speaking: bool,
    previous_sample: Option<f64>,
    gain_mode: GainMode,
    input_policy: String,
    conditioning_sha256: Option<String>,
    frames: Vec<Frame>,
    discarded_counts: InspectionCounts,
}

pub fn speech_issues(verified: bool) -> &'static [&'static str] {
    if verified {
        &[]
    } else {
        const MISSING: [&str; 1] = [messages::SPEECH_MODEL_MISSING];
        &MISSING
    }
}

/// Quiet-speech gain. Zero peak stays 1. Nonfinite or negative peaks are rejected.
#[allow(clippy::float_cmp)]
pub fn quiet_speech_gain(peak: f64) -> Result<f64, SpeechError> {
    if !peak.is_finite() || peak < 0.0 {
        return Err(SpeechError::InvalidPeak);
    }
    if peak == 0.0 {
        return Ok(1.0);
    }
    Ok((GAIN_TARGET / peak).clamp(1.0, GAIN_CAP))
}

/// Qualified source conditioning never boosts, including quiet peaks.
pub fn unity_gain(peak: f64) -> Result<f64, SpeechError> {
    if !peak.is_finite() || peak < 0.0 {
        return Err(SpeechError::InvalidPeak);
    }
    Ok(1.0)
}

/// IPC gain policy. A set peak uses the quiet-speech formula; `None` is unity gain.
pub fn normalized_view_gain(policy: GainPolicy) -> Result<f32, SpeechError> {
    match policy.peak {
        None => Ok(1.0),
        Some(peak) => quiet_speech_gain(f64::from(peak)).map(|gain| gain as f32),
    }
}

pub fn next_speaking(speaking: bool, probability: f64) -> bool {
    if probability >= ONSET_PROBABILITY {
        true
    } else if probability < RELEASE_PROBABILITY {
        false
    } else {
        speaking
    }
}

/// Crop-level exact PCM. This is equality, not an energy threshold.
#[allow(clippy::float_cmp)]
pub fn exact_pcm(samples: &[f32]) -> Option<(ExactPcm, f64)> {
    let &first = samples.first()?;
    if samples.iter().any(|sample| *sample != first) {
        return None;
    }
    let value = f64::from(first);
    if first == 0.0 {
        Some((ExactPcm::DigitalSilence, value))
    } else {
        Some((ExactPcm::Constant, value))
    }
}

pub fn pcm_evidence(kind: ExactPcm, value: f64, start: i64, end: i64) -> PcmEvidence {
    match kind {
        ExactPcm::DigitalSilence => PcmEvidence {
            kind: PCM_SILENCE,
            start_sample: start,
            end_sample: end,
            sample_value: None,
        },
        ExactPcm::Constant => PcmEvidence {
            kind: PCM_CONSTANT,
            start_sample: start,
            end_sample: end,
            sample_value: Some(value),
        },
    }
}

/// Map a ledger decision and the physical crop onto the routing block, if any.
pub fn blocked_route(
    decision: SpeechDecision,
    pcm: &[f32],
    start: i64,
    end: i64,
) -> Option<BlockedSpeech> {
    let constant = exact_pcm(pcm);
    match decision {
        SpeechDecision::Speech => {
            let (kind, value) = constant?;
            Some(constant_block(kind, value, start, end))
        }
        SpeechDecision::NoSpeech => Some(BlockedSpeech {
            reason: REASON_INSUFFICIENT,
            audio_state: STATE_MODEL_NON_SPEECH,
            pcm: constant.map(|(kind, value)| pcm_evidence(kind, value, start, end)),
        }),
        SpeechDecision::Uncertain => Some(BlockedSpeech {
            reason: REASON_UNCERTAIN,
            audio_state: STATE_PENDING,
            pcm: None,
        }),
        SpeechDecision::Pending => Some(BlockedSpeech {
            reason: REASON_PENDING,
            audio_state: STATE_PENDING,
            pcm: None,
        }),
    }
}

fn constant_block(kind: ExactPcm, value: f64, start: i64, end: i64) -> BlockedSpeech {
    let audio_state = match kind {
        ExactPcm::DigitalSilence => STATE_DIGITAL_SILENCE,
        ExactPcm::Constant => STATE_CONSTANT,
    };
    BlockedSpeech {
        reason: REASON_INSUFFICIENT,
        audio_state,
        pcm: Some(pcm_evidence(kind, value, start, end)),
    }
}

pub fn inspection_decision(counts: &InspectionCounts) -> SpeechDecision {
    if counts.speech_samples > 0 {
        SpeechDecision::Speech
    } else if counts.uncertain_samples > 0 {
        SpeechDecision::Uncertain
    } else {
        SpeechDecision::NoSpeech
    }
}

impl SpeechLedger {
    pub fn new(origin: i64) -> Result<Self, SpeechError> {
        Self::build(origin, GainMode::QuietSpeech, INPUT_POLICY, None)
    }

    pub fn with_processed_microphone(
        origin: i64,
        contract_sha256: &str,
    ) -> Result<Self, SpeechError> {
        if !is_sha256_hex(contract_sha256) {
            return Err(SpeechError::InvalidConditioning);
        }
        Self::build(
            origin,
            GainMode::Unity,
            PRODUCTION_UNITY_POLICY,
            Some(contract_sha256.to_owned()),
        )
    }

    fn build(
        origin: i64,
        gain_mode: GainMode,
        input_policy: &str,
        conditioning_sha256: Option<String>,
    ) -> Result<Self, SpeechError> {
        if origin < 0 {
            return Err(SpeechError::InvalidOrigin);
        }
        Ok(Self {
            origin,
            end_sample: origin,
            speaking: false,
            previous_sample: None,
            gain_mode,
            input_policy: input_policy.to_owned(),
            conditioning_sha256,
            frames: Vec::new(),
            discarded_counts: InspectionCounts {
                speech_samples: 0,
                uncertain_samples: 0,
                negative_constant_samples: 0,
                model_negative_samples: 0,
            },
        })
    }

    pub fn end_sample(&self) -> i64 {
        self.end_sample
    }

    pub fn frames(&self) -> &[Frame] {
        &self.frames
    }

    #[cfg(test)]
    pub(crate) fn retained_frame_capacity(&self) -> usize {
        self.frames.capacity()
    }

    /// Release observations no decoder owns, without resetting hysteresis or
    /// the previous PCM value. Queries behind this boundary remain unverified.
    pub(crate) fn discard_before(&mut self, through: i64) {
        let count = self.frames.partition_point(|frame| frame.end <= through);
        for frame in &self.frames[..count] {
            count_frame(&mut self.discarded_counts, frame);
        }
        self.frames.drain(..count);
        self.origin = self
            .frames
            .first()
            .map_or(self.end_sample, |frame| frame.start);
        if self.frames.capacity() > self.frames.len().max(1024) * 2 {
            self.frames.shrink_to(self.frames.len().max(1024));
        }
    }

    pub fn input_policy(&self) -> &str {
        &self.input_policy
    }

    pub fn gain_mode(&self) -> GainMode {
        self.gain_mode
    }

    /// Append one neural frame. `probability` is the combined view before a constant override.
    pub fn append(
        &mut self,
        start: i64,
        end: i64,
        probability: f64,
        observation: Option<Observation>,
    ) -> Result<(), SpeechError> {
        if start != self.end_sample
            || end <= start
            || end > start + FRAME_SAMPLES as i64
            || !valid_probability(probability)
        {
            return Err(SpeechError::InvalidEvidence);
        }
        self.speaking = next_speaking(self.speaking, probability);
        let constant = observation
            .as_ref()
            .is_some_and(|item| item.constant_pcm_override);
        let frame = Frame {
            start,
            end,
            probability: if constant { 0.0 } else { probability },
            speaking: self.speaking && !constant,
            observation,
        };
        self.frames.push(frame);
        self.end_sample = end;
        Ok(())
    }

    /// Dual-view frame. Probabilities are injected; gain and constant PCM are decided here.
    pub fn append_pcm(
        &mut self,
        start: i64,
        samples: &[f32],
        raw_probability: f64,
        normalized_probability: f64,
    ) -> Result<(), SpeechError> {
        if samples.is_empty()
            || samples.len() > FRAME_SAMPLES
            || samples.iter().any(|sample| !sample.is_finite())
        {
            return Err(SpeechError::InvalidPcm);
        }
        if !valid_probability(raw_probability) || !valid_probability(normalized_probability) {
            return Err(SpeechError::InvalidProbability);
        }
        let peak = samples
            .iter()
            .fold(0.0f64, |max, sample| max.max(f64::from(sample.abs())));
        let gain = match self.gain_mode {
            GainMode::QuietSpeech => quiet_speech_gain(peak)?,
            GainMode::Unity => unity_gain(peak)?,
        };
        let constant_value = constant_sample(samples, self.previous_sample);
        let observation = Observation {
            input_policy: self.input_policy.clone(),
            raw_probability,
            normalized_probability,
            gain,
            constant_value,
            constant_pcm_override: constant_value.is_some(),
            conditioning_sha256: self.conditioning_sha256.clone(),
        };
        let end = start + samples.len() as i64;
        self.append(
            start,
            end,
            raw_probability.max(normalized_probability),
            Some(observation),
        )?;
        self.previous_sample = Some(f64::from(samples[samples.len() - 1]));
        Ok(())
    }

    pub fn admission(&self, start: i64, end: i64) -> Result<Admission, SpeechError> {
        if start < 0 || end <= start {
            return Err(SpeechError::InvalidRange);
        }
        let first = self.frames.partition_point(|frame| frame.end <= start);
        let last = self.frames.partition_point(|frame| frame.start < end);
        let selected: Vec<&Frame> = self.frames[first..last].iter().collect();
        let covered = self.range_covered(start, end, &selected);
        let mut result = pending_admission(start, end, self.end_sample);
        result.complete = covered;
        if !covered {
            return Ok(result);
        }
        let conditioned = self.frames_conditioned(&selected);
        if conditioned {
            result.input_policy = Some(self.input_policy.clone());
            result.conditioning_sha256 = self.conditioning_sha256.clone();
            result.maximum_model_probability = Some(maximum_model_probability(&selected));
            result.uncertain_regions = Vec::new();
            result.model_negative_regions = negative_regions(&selected, start, end);
        }
        result.maximum_probability = Some(
            selected
                .iter()
                .map(|frame| frame.probability)
                .fold(f64::NEG_INFINITY, f64::max),
        );
        result.speech_regions = speech_regions(&selected, start, end);
        result.decision = if result.speech_regions.is_empty() {
            SpeechDecision::NoSpeech
        } else {
            SpeechDecision::Speech
        };
        if conditioned && result.speech_regions.is_empty() {
            result.negative_signal_state = Some(NEGATIVE_SIGNAL_STATE.to_owned());
            result.classification_scope = Some(CLASSIFICATION_SCOPE.to_owned());
        }
        Ok(result)
    }

    pub fn inspection_counts(&self) -> InspectionCounts {
        let mut counts = self.discarded_counts;
        for frame in &self.frames {
            count_frame(&mut counts, frame);
        }
        counts
    }

    fn range_covered(&self, start: i64, end: i64, selected: &[&Frame]) -> bool {
        let mut cursor = start;
        for frame in selected {
            if frame.start > cursor {
                break;
            }
            cursor = cursor.max(frame.end);
        }
        start >= self.origin && end <= self.end_sample && cursor >= end
    }

    fn frames_conditioned(&self, selected: &[&Frame]) -> bool {
        selected.iter().all(|frame| {
            let Some(observation) = &frame.observation else {
                return false;
            };
            if observation.input_policy != self.input_policy {
                return false;
            }
            match &self.conditioning_sha256 {
                None => true,
                Some(sha) => observation.conditioning_sha256.as_ref() == Some(sha),
            }
        })
    }
}

fn count_frame(counts: &mut InspectionCounts, frame: &Frame) {
    let samples = frame.end - frame.start;
    if frame.speaking {
        counts.speech_samples += samples;
    } else if frame.observation.is_none() {
        counts.uncertain_samples += samples;
    } else if frame
        .observation
        .as_ref()
        .is_some_and(|o| o.constant_value.is_some())
    {
        counts.negative_constant_samples += samples;
    } else {
        counts.model_negative_samples += samples;
    }
}

/// A frozen canonical prefix keeps one aggregate, rather than all VAD frames.
pub(crate) fn join_admissions(
    mut prefix: Admission,
    tail: Admission,
) -> Result<Admission, SpeechError> {
    if prefix.end_sample != tail.start_sample
        || prefix.source != tail.source
        || prefix.model_revision != tail.model_revision
        || !prefix.complete
        || !tail.complete
    {
        return Err(SpeechError::InvalidEvidence);
    }
    let merge = |left: &mut Vec<SampleSpan>, right: Vec<SampleSpan>| {
        for span in right {
            if let Some(last) = left.last_mut()
                && last.end_sample == span.start_sample
            {
                last.end_sample = span.end_sample;
            } else {
                left.push(span);
            }
        }
    };
    prefix.end_sample = tail.end_sample;
    prefix.observed_until_sample = prefix.observed_until_sample.max(tail.observed_until_sample);
    prefix.maximum_probability = prefix
        .maximum_probability
        .zip(tail.maximum_probability)
        .map(|(a, b)| a.max(b));
    prefix.maximum_model_probability = prefix
        .maximum_model_probability
        .zip(tail.maximum_model_probability)
        .map(|(a, b)| a.max(b));
    if prefix.input_policy != tail.input_policy
        || prefix.conditioning_sha256 != tail.conditioning_sha256
    {
        prefix.input_policy = None;
        prefix.conditioning_sha256 = None;
        prefix.maximum_model_probability = None;
    }
    merge(&mut prefix.speech_regions, tail.speech_regions);
    merge(&mut prefix.uncertain_regions, tail.uncertain_regions);
    merge(
        &mut prefix.model_negative_regions,
        tail.model_negative_regions,
    );
    prefix.evidence_error |= tail.evidence_error;
    prefix.decision = if !prefix.speech_regions.is_empty() {
        SpeechDecision::Speech
    } else {
        SpeechDecision::NoSpeech
    };
    if prefix.decision == SpeechDecision::NoSpeech && prefix.input_policy.is_some() {
        prefix.negative_signal_state = Some(NEGATIVE_SIGNAL_STATE.into());
        prefix.classification_scope = Some(CLASSIFICATION_SCOPE.into());
    } else {
        prefix.negative_signal_state = None;
        prefix.classification_scope = None;
    }
    Ok(prefix)
}

fn pending_admission(start: i64, end: i64, observed_until: i64) -> Admission {
    Admission {
        source: SILERO_SOURCE.to_owned(),
        model_revision: SILERO_REVISION.to_owned(),
        start_sample: start,
        end_sample: end,
        observed_until_sample: observed_until,
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

fn speech_regions(frames: &[&Frame], start: i64, end: i64) -> Vec<SampleSpan> {
    let mut regions = Vec::new();
    for frame in frames {
        if !frame.speaking {
            continue;
        }
        push_clipped(&mut regions, frame.start, frame.end, start, end);
    }
    regions
}

fn negative_regions(frames: &[&Frame], start: i64, end: i64) -> Vec<SampleSpan> {
    let mut regions = Vec::new();
    for frame in frames {
        if frame.speaking {
            continue;
        }
        push_clipped(&mut regions, frame.start, frame.end, start, end);
    }
    regions
}

fn push_clipped(
    regions: &mut Vec<SampleSpan>,
    frame_start: i64,
    frame_end: i64,
    start: i64,
    end: i64,
) {
    let clipped_start = frame_start.max(start);
    let clipped_end = frame_end.min(end);
    if clipped_start >= clipped_end {
        return;
    }
    if let Some(last) = regions.last_mut()
        && last.end_sample == clipped_start
    {
        last.end_sample = clipped_end;
        return;
    }
    regions.push(SampleSpan {
        start_sample: clipped_start,
        end_sample: clipped_end,
    });
}

fn maximum_model_probability(frames: &[&Frame]) -> f64 {
    frames
        .iter()
        .map(|frame| {
            frame
                .observation
                .as_ref()
                .map_or(frame.probability, |observation| {
                    observation
                        .raw_probability
                        .max(observation.normalized_probability)
                })
        })
        .fold(f64::NEG_INFINITY, f64::max)
}

fn valid_probability(probability: f64) -> bool {
    probability.is_finite() && (0.0..=1.0).contains(&probability)
}

#[allow(clippy::float_cmp)]
fn constant_sample(samples: &[f32], previous: Option<f64>) -> Option<f64> {
    let first = samples[0];
    if samples.iter().any(|sample| *sample != first) {
        return None;
    }
    if previous.is_some_and(|value| value != f64::from(first)) {
        return None;
    }
    Some(f64::from(first))
}

fn is_sha256_hex(value: &str) -> bool {
    value.len() == 64
        && value
            .bytes()
            .all(|byte| byte.is_ascii_hexdigit() && !byte.is_ascii_uppercase())
}

pub(crate) fn admission_value(admission: &Admission) -> serde_json::Value {
    serde_json::json!({
        "source": admission.source,
        "model_revision": admission.model_revision,
        "start_sample": admission.start_sample,
        "end_sample": admission.end_sample,
        "observed_until_sample": admission.observed_until_sample,
        "complete": admission.complete,
        "decision": admission.decision.as_str(),
        "speech_regions": admission.speech_regions.iter().map(|region| serde_json::json!({
            "start_sample": region.start_sample,
            "end_sample": region.end_sample,
        })).collect::<Vec<_>>(),
        "maximum_probability": admission.maximum_probability,
        "input_policy": admission.input_policy,
        "conditioning_sha256": admission.conditioning_sha256,
        "maximum_model_probability": admission.maximum_model_probability,
        "uncertain_regions": admission.uncertain_regions.iter().map(|r| serde_json::json!({"start_sample":r.start_sample,"end_sample":r.end_sample})).collect::<Vec<_>>(),
        "evidence_error": admission.evidence_error,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ramp(len: usize, start: f32, end: f32) -> Vec<f32> {
        if len == 1 {
            return vec![start];
        }
        (0..len)
            .map(|index| start + (end - start) * (index as f32) / (len as f32 - 1.0))
            .collect()
    }

    #[test]
    fn contract_thresholds_and_speech_message_do_not_drift() {
        assert_eq!(limits::LANGUAGE_AUTO_THRESHOLD, 0.90);
        assert_eq!(limits::LANGUAGE_MARGIN_THRESHOLD, 0.20);
        assert_eq!(limits::SAMPLE_RATE, 16_000);
        assert_eq!(limits::VAD_FRAME_SAMPLES, 512);
        assert_eq!(FRAME_SAMPLES, 512);
        assert_eq!(HOT_FRAMES, 1_875);
        assert_eq!(ONSET_PROBABILITY, 0.5);
        assert_eq!(RELEASE_PROBABILITY, 0.35);
        assert_eq!(GAIN_TARGET, 0.25);
        assert_eq!(GAIN_CAP, 256.0);
        assert_eq!(
            INPUT_POLICY,
            "raw_and_peak025_gaincap256_per512_constant_guard_v3"
        );
        assert_eq!(
            SOURCE_UNITY_POLICY,
            "apple_active_microphone_tap_raw_and_unity_view_v1"
        );
        assert_eq!(
            PRODUCTION_UNITY_POLICY,
            "qualified_apple_processed_microphone_raw_and_unity_view_v1"
        );
        assert_eq!(SILERO_REVISION, "2ebf4a5e10726a2e78ddd4d70eedfb6f1c33eb06");
        assert_eq!(
            messages::SPEECH_MODEL_MISSING,
            "Speech detection needs its verified local model. Download models in Settings."
        );
        assert_eq!(speech_issues(true), &[] as &[&str]);
        assert_eq!(speech_issues(false), &[messages::SPEECH_MODEL_MISSING]);
        assert_eq!(
            speech_issues(false)[0],
            "Speech detection needs its verified local model. Download models in Settings."
        );
    }

    #[test]
    fn quiet_gain_matches_peak_formula_and_unity_stays_one() {
        assert_eq!(quiet_speech_gain(0.0).unwrap(), 1.0);
        assert_eq!(quiet_speech_gain(-0.0).unwrap(), 1.0);
        assert_eq!(quiet_speech_gain(0.25).unwrap(), 1.0);
        assert_eq!(quiet_speech_gain(0.125).unwrap(), 2.0);
        assert_eq!(quiet_speech_gain(0.25 / 256.0).unwrap(), 256.0);
        assert_eq!(quiet_speech_gain(1e-9).unwrap(), 256.0);
        let quiet = 0.001f32;
        let gain = quiet_speech_gain(f64::from(quiet)).unwrap();
        assert!((gain - 250.0).abs() < 1e-3);
        assert_eq!(gain, (0.25 / f64::from(quiet)).clamp(1.0, 256.0));
        assert_eq!(unity_gain(0.0003).unwrap(), 1.0);
        assert_eq!(unity_gain(0.0).unwrap(), 1.0);
        assert_eq!(
            quiet_speech_gain(f64::NAN).unwrap_err(),
            SpeechError::InvalidPeak
        );
        assert_eq!(
            quiet_speech_gain(-0.1).unwrap_err().to_string(),
            "Invalid physical frame peak."
        );
        assert_eq!(
            normalized_view_gain(GainPolicy { peak: None }).unwrap(),
            1.0
        );
        let boosted = normalized_view_gain(GainPolicy { peak: Some(0.125) }).unwrap();
        assert_eq!(boosted, 2.0);
    }

    #[test]
    fn missing_and_partial_evidence_stay_pending() {
        let ledger = SpeechLedger::new(0).unwrap();
        let pending = ledger.admission(0, FRAME_SAMPLES as i64).unwrap();
        assert_eq!(pending.decision, SpeechDecision::Pending);
        assert!(!pending.complete);
        assert_eq!(pending.decision.as_str(), "pending");
        assert!(pending.speech_regions.is_empty());

        let mut ledger = SpeechLedger::new(0).unwrap();
        ledger.append(0, FRAME_SAMPLES as i64, 0.01, None).unwrap();
        assert_eq!(
            ledger
                .admission(0, FRAME_SAMPLES as i64 + 1)
                .unwrap()
                .decision,
            SpeechDecision::Pending
        );
        let covered = ledger.admission(0, FRAME_SAMPLES as i64).unwrap();
        assert_eq!(covered.decision, SpeechDecision::NoSpeech);
        assert!(covered.complete);
        assert!(covered.input_policy.is_none());
        assert_eq!(
            ledger.admission(-1, 10).unwrap_err().to_string(),
            "Invalid speech-admission range."
        );
    }

    #[test]
    fn hysteresis_keeps_one_positive_frame_without_a_duration_floor() {
        let mut ledger = SpeechLedger::new(0).unwrap();
        for (index, score) in [0.01, 0.8, 0.4, 0.1].into_iter().enumerate() {
            let start = (index * FRAME_SAMPLES) as i64;
            ledger
                .append(start, start + FRAME_SAMPLES as i64, score, None)
                .unwrap();
        }
        let result = ledger.admission(0, 4 * FRAME_SAMPLES as i64).unwrap();
        assert_eq!(result.decision, SpeechDecision::Speech);
        assert_eq!(
            result.speech_regions,
            vec![SampleSpan {
                start_sample: FRAME_SAMPLES as i64,
                end_sample: 3 * FRAME_SAMPLES as i64,
            }]
        );
        assert!(next_speaking(true, 0.35));
        assert!(!next_speaking(true, 0.35 - f64::EPSILON));
        assert!(next_speaking(false, 0.5));
        assert!(!next_speaking(false, 0.5 - f64::EPSILON));
    }

    #[test]
    fn invalid_probability_and_gaps_fail_before_mutation() {
        let mut ledger = SpeechLedger::new(0).unwrap();
        for score in [f64::NAN, f64::INFINITY, -1.0, 2.0] {
            assert_eq!(
                ledger
                    .append(0, FRAME_SAMPLES as i64, score, None)
                    .unwrap_err(),
                SpeechError::InvalidEvidence
            );
        }
        assert_eq!(
            ledger
                .append(1, FRAME_SAMPLES as i64, 0.1, None)
                .unwrap_err()
                .to_string(),
            "Invalid or noncontiguous Silero evidence."
        );
        assert!(ledger.frames().is_empty());
        assert_eq!(
            SpeechLedger::new(-1).unwrap_err().to_string(),
            "Invalid speech-evidence origin."
        );
    }

    #[test]
    fn constant_and_zero_pcm_do_not_authorize_speech_but_hysteresis_survives() {
        let mut ledger = SpeechLedger::new(0).unwrap();
        let mut quiet = ramp(FRAME_SAMPLES, -1e-9, 1e-9);
        quiet[FRAME_SAMPLES - 1] = 0.0;
        ledger.append_pcm(0, &quiet, 0.99, 0.4).unwrap();
        ledger
            .append_pcm(FRAME_SAMPLES as i64, &vec![0.0; FRAME_SAMPLES], 0.99, 0.99)
            .unwrap();
        ledger
            .append_pcm(2 * FRAME_SAMPLES as i64, &quiet, 0.2, 0.4)
            .unwrap();

        let zero = ledger
            .admission(FRAME_SAMPLES as i64, 2 * FRAME_SAMPLES as i64)
            .unwrap();
        assert_eq!(zero.decision, SpeechDecision::NoSpeech);
        assert_eq!(zero.maximum_probability, Some(0.0));
        assert_eq!(zero.maximum_model_probability, Some(0.99));
        assert_eq!(
            zero.negative_signal_state.as_deref(),
            Some(NEGATIVE_SIGNAL_STATE)
        );
        assert!(
            ledger.frames()[1]
                .observation
                .as_ref()
                .unwrap()
                .constant_pcm_override
        );
        assert!((ledger.frames()[0].observation.as_ref().unwrap().gain - 250_000.0).abs() > 1.0);
        assert_eq!(ledger.frames()[0].observation.as_ref().unwrap().gain, 256.0);

        let held = ledger
            .admission(2 * FRAME_SAMPLES as i64, 3 * FRAME_SAMPLES as i64)
            .unwrap();
        assert_eq!(held.decision, SpeechDecision::Speech);
        assert_eq!(ledger.frames()[2].probability, 0.4);

        let mut stepped = SpeechLedger::new(0).unwrap();
        stepped
            .append_pcm(0, &vec![0.0; FRAME_SAMPLES], 0.99, 0.99)
            .unwrap();
        stepped
            .append_pcm(FRAME_SAMPLES as i64, &vec![0.04; FRAME_SAMPLES], 0.99, 0.99)
            .unwrap();
        assert_eq!(
            stepped.admission(0, FRAME_SAMPLES as i64).unwrap().decision,
            SpeechDecision::NoSpeech
        );
        assert_eq!(
            stepped
                .admission(FRAME_SAMPLES as i64, 2 * FRAME_SAMPLES as i64)
                .unwrap()
                .decision,
            SpeechDecision::Speech
        );
        assert!(
            stepped.frames()[1]
                .observation
                .as_ref()
                .unwrap()
                .constant_value
                .is_none()
        );
    }

    #[test]
    fn unity_conditioning_does_not_boost_and_negative_regions_clip() {
        let sha = "c".repeat(64);
        let mut ledger = SpeechLedger::with_processed_microphone(0, &sha).unwrap();
        assert_eq!(
            SpeechLedger::with_processed_microphone(0, "bad")
                .unwrap_err()
                .to_string(),
            "Invalid source conditioning identity."
        );
        let quiet = ramp(FRAME_SAMPLES, -0.001, 0.001);
        ledger.append_pcm(0, &quiet, 0.02, 0.02).unwrap();
        assert_eq!(ledger.frames()[0].observation.as_ref().unwrap().gain, 1.0);
        assert_eq!(ledger.input_policy(), PRODUCTION_UNITY_POLICY);
        let mut voiced = SpeechLedger::new(0).unwrap();
        voiced.append_pcm(0, &quiet, 0.01, 0.9).unwrap();
        assert!((voiced.frames()[0].observation.as_ref().unwrap().gain - 250.0).abs() < 1e-2);
        assert_eq!(
            voiced.admission(0, FRAME_SAMPLES as i64).unwrap().decision,
            SpeechDecision::Speech
        );

        let mut mixed = SpeechLedger::new(0).unwrap();
        let varying = ramp(FRAME_SAMPLES, -0.2, 0.2);
        mixed.append_pcm(0, &varying, 0.9, 0.9).unwrap();
        mixed
            .append_pcm(
                FRAME_SAMPLES as i64,
                &ramp(FRAME_SAMPLES, 0.2, -0.2),
                0.1,
                0.1,
            )
            .unwrap();
        assert_eq!(
            mixed.admission(700, 900).unwrap().model_negative_regions,
            vec![SampleSpan {
                start_sample: 700,
                end_sample: 900,
            }]
        );
        assert!(
            mixed
                .admission(0, 1024)
                .unwrap()
                .uncertain_regions
                .is_empty()
        );
    }

    #[test]
    fn blocked_routes_separate_pending_uncertain_zero_and_constant() {
        let zero = vec![0.0; 32];
        let constant = vec![0.04; 32];
        let varying = ramp(32, -0.1, 0.1);
        let speech_zero = blocked_route(SpeechDecision::Speech, &zero, 0, 32).unwrap();
        assert_eq!(speech_zero.reason, "insufficient_speech");
        assert_eq!(speech_zero.audio_state, "digital_silence");
        assert_eq!(
            speech_zero.pcm.as_ref().unwrap().kind,
            "exact_digital_silence"
        );
        assert!(speech_zero.pcm.as_ref().unwrap().sample_value.is_none());

        let speech_dc = blocked_route(SpeechDecision::Speech, &constant, 4, 36).unwrap();
        assert_eq!(speech_dc.audio_state, "constant_signal");
        assert_eq!(
            speech_dc.pcm.as_ref().unwrap().sample_value,
            Some(f64::from(0.04f32))
        );
        assert_eq!(
            speech_dc.pcm.as_ref().unwrap().kind,
            "exact_constant_signal"
        );

        assert!(blocked_route(SpeechDecision::Speech, &varying, 0, 32).is_none());

        let negative = blocked_route(SpeechDecision::NoSpeech, &constant, 0, 32).unwrap();
        assert_eq!(negative.audio_state, "model_non_speech");
        assert_eq!(negative.reason, "insufficient_speech");
        assert!(negative.pcm.is_some());

        let uncertain = blocked_route(SpeechDecision::Uncertain, &varying, 0, 32).unwrap();
        assert_eq!(uncertain.reason, "speech_admission_uncertain");
        assert_eq!(uncertain.audio_state, "speech_evidence_pending");
        let pending = blocked_route(SpeechDecision::Pending, &zero, 0, 32).unwrap();
        assert_eq!(pending.reason, "speech_evidence_pending");
        assert_eq!(pending.audio_state, "speech_evidence_pending");
    }

    #[test]
    fn inspection_counts_distinguish_uncertain_constant_and_model_negative() {
        let mut ledger = SpeechLedger::new(0).unwrap();
        ledger.append(0, 10, 0.1, None).unwrap();
        let counts = ledger.inspection_counts();
        assert_eq!(counts.uncertain_samples, 10);
        assert_eq!(inspection_decision(&counts), SpeechDecision::Uncertain);

        let mut voiced = SpeechLedger::new(0).unwrap();
        voiced.append_pcm(0, &ramp(8, -0.2, 0.0), 0.8, 0.8).unwrap();
        voiced.append_pcm(8, &[0.0; 5], 0.1, 0.1).unwrap();
        let counts = voiced.inspection_counts();
        assert_eq!(counts.speech_samples, 8);
        assert_eq!(counts.negative_constant_samples, 5);
        assert_eq!(inspection_decision(&counts), SpeechDecision::Speech);
    }
}
