//! Voice eligibility, enrollment, and one recognition decision per track.
//!
//! Embeddings come from a [`VoiceEmbedder`]. This module does not load or run a
//! model. A cosine at or above [`IDENTITY_REUSE_COSINE`] is only a signal for
//! the caller; it never mints a person.

use std::collections::{BTreeMap, BTreeSet};
use std::sync::LazyLock;

use regex::Regex;
use thiserror::Error;

use crate::limits;
use crate::messages;
use crate::people::{
    self, EXPLICIT_REMEMBER_VOICE, People, PeopleError, SpeakerAssignment, SuggestionDocument,
    SuggestionJob, SuggestionProvenance, SuggestionSegment, is_unassigned_speaker_label,
    reconcile_assignments, require_profile_consent, require_remember_consent,
};

/// Caller-owned reuse bar. Matching or enrollment does not create a person id.
pub const IDENTITY_REUSE_COSINE: f64 = 0.9999;

pub const SOURCE_AUTOMATIC_VOICE: &str = "automatic_voice";
pub const CHECK_CHECKING: &str = "checking";
pub const CHECK_UNKNOWN: &str = "unknown";
pub const CHECK_UNAVAILABLE: &str = "unavailable";
pub const CHECK_CANCELLED: &str = "cancelled";
pub const CHECK_MATCHED: &str = "matched";

const MODEL_IDENTITY: &str = "Supply a pinned, compatible voice model identity.";
const CALIBRATION_REQUIRED: &str =
    "Voice matching needs measured threshold and margin calibration.";
const INVALID_EMBEDDING: &str = "Invalid voice embedding.";

static NUMBERED_TRACK: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"^speaker_\d+$").expect("numbered track pattern"));

#[derive(Debug, Clone, PartialEq, Eq, Error)]
pub enum VoiceError {
    #[error("{}", messages::USE_FINALIZED_CLIPS)]
    FinalizedClips,
    #[error("{0}")]
    PassageCount(String),
    #[error("{}", messages::CLEAN_PASSAGE_LENGTH)]
    CleanPassage,
    #[error("{}", messages::NONOVERLAPPING_PASSAGES)]
    Nonoverlapping,
    #[error("{}", messages::PASSAGE_UNUSABLE)]
    Unusable,
    #[error("{}", messages::PROFILE_NEEDS_CLIPS)]
    ProfileNeedsClips,
    #[error("{}", messages::VOICE_PASSAGE_LENGTH)]
    PassageLength,
    #[error("{}", messages::REMEMBER_BEFORE_CONFIRM)]
    RememberBeforeConfirm,
    #[error("{}", MODEL_IDENTITY)]
    ModelIdentity,
    #[error("{}", CALIBRATION_REQUIRED)]
    Calibration,
    #[error("{}", INVALID_EMBEDDING)]
    Embedding,
    #[error(transparent)]
    People(#[from] PeopleError),
}

impl VoiceError {
    fn passage_count(minimum: usize) -> Self {
        Self::PassageCount(messages::clean_passage_count(minimum))
    }
}

/// `true` when the caller may reuse an existing identity. No person is created.
pub fn caller_may_reuse_identity(cosine: f64) -> bool {
    cosine.is_finite() && cosine >= IDENTITY_REUSE_COSINE
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct VoiceModel {
    pub model_id: String,
    pub revision: String,
    pub artifact_sha256: String,
    pub dimension: usize,
    pub sample_rate: u32,
}

impl VoiceModel {
    pub fn new(
        model_id: impl Into<String>,
        revision: impl Into<String>,
        artifact_sha256: impl Into<String>,
        dimension: usize,
        sample_rate: u32,
    ) -> Result<Self, VoiceError> {
        let model_id = model_id.into();
        let revision = revision.into();
        let artifact_sha256 = artifact_sha256.into();
        let sha_ok = artifact_sha256.len() == 64
            && artifact_sha256
                .bytes()
                .all(|byte| byte.is_ascii_hexdigit() && !byte.is_ascii_uppercase());
        if model_id.is_empty()
            || revision.is_empty()
            || !sha_ok
            || !(1..=4096).contains(&dimension)
            || sample_rate != limits::SAMPLE_RATE
        {
            return Err(VoiceError::ModelIdentity);
        }
        Ok(Self {
            model_id,
            revision,
            artifact_sha256,
            dimension,
            sample_rate,
        })
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct Calibration {
    pub model: VoiceModel,
    pub dataset_id: String,
    pub threshold: f64,
    pub margin: f64,
    pub genuine_trials: u64,
    pub impostor_trials: u64,
    pub false_accept_rate: f64,
    pub false_reject_rate: f64,
    pub minimum_clips: usize,
}

impl Calibration {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        model: VoiceModel,
        dataset_id: impl Into<String>,
        threshold: f64,
        margin: f64,
        genuine_trials: u64,
        impostor_trials: u64,
        false_accept_rate: f64,
        false_reject_rate: f64,
        minimum_clips: usize,
    ) -> Result<Self, VoiceError> {
        let dataset_id = dataset_id.into();
        let finite = [threshold, margin, false_accept_rate, false_reject_rate]
            .iter()
            .all(|value| value.is_finite());
        let open_unit = |value: f64| value > 0.0 && value <= 1.0;
        let closed_unit = |value: f64| (0.0..=1.0).contains(&value);
        if dataset_id.is_empty()
            || !finite
            || !open_unit(threshold)
            || !open_unit(margin)
            || !closed_unit(false_accept_rate)
            || !closed_unit(false_reject_rate)
            || genuine_trials < 1
            || impostor_trials < 1
            || !(2..=12).contains(&minimum_clips)
        {
            return Err(VoiceError::Calibration);
        }
        Ok(Self {
            model,
            dataset_id,
            threshold,
            margin,
            genuine_trials,
            impostor_trials,
            false_accept_rate,
            false_reject_rate,
            minimum_clips,
        })
    }
}

pub fn runtime_available(
    model: Option<&VoiceModel>,
    calibration: Option<&Calibration>,
    execution_allowed: bool,
) -> bool {
    match (model, calibration) {
        (Some(model), Some(calibration)) => {
            execution_allowed && same_model(model, &calibration.model)
        }
        _ => false,
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct VoiceClip {
    pub meeting_id: String,
    pub track_id: String,
    pub segment_id: String,
    pub start: f64,
    pub end: f64,
}

impl VoiceClip {
    pub fn try_new(
        meeting_id: impl Into<String>,
        track_id: impl Into<String>,
        segment_id: impl Into<String>,
        start: f64,
        end: f64,
    ) -> Result<Self, VoiceError> {
        let duration = end - start;
        if !start.is_finite()
            || !end.is_finite()
            || start < 0.0
            || start >= end
            || !(2.0..=10.0).contains(&duration)
        {
            return Err(VoiceError::PassageLength);
        }
        Ok(Self {
            meeting_id: meeting_id.into(),
            track_id: track_id.into(),
            segment_id: segment_id.into(),
            start,
            end,
        })
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct ClipEmbedding {
    pub vector: Vec<f64>,
    pub clean: bool,
}

pub trait VoiceEmbedder {
    fn embed_clip(&mut self, clip: &VoiceClip) -> Result<ClipEmbedding, VoiceError>;
}

impl<F> VoiceEmbedder for F
where
    F: FnMut(&VoiceClip) -> Result<ClipEmbedding, VoiceError>,
{
    fn embed_clip(&mut self, clip: &VoiceClip) -> Result<ClipEmbedding, VoiceError> {
        self(clip)
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AudioReason {
    DifferentSpeaker,
    UnverifiedAudio,
    Unfinished,
    Overlap,
}

impl AudioReason {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::DifferentSpeaker => "different_speaker",
            Self::UnverifiedAudio => "unverified_audio",
            Self::Unfinished => "unfinished",
            Self::Overlap => "overlap",
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct ActivityRegion {
    pub start_sample: i64,
    pub end_sample: i64,
    pub speakers: Vec<String>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct SpeakerActivity {
    pub audio_revision: i64,
    pub observed_end_sample: i64,
    pub regions: Vec<ActivityRegion>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct AudioSegment {
    pub id: String,
    pub speaker: String,
    pub start: f64,
    pub end: f64,
    pub text: String,
    pub review: Option<bool>,
    pub finalized: Option<bool>,
    pub speaker_candidates: Option<Vec<String>>,
    pub voice_eligible: Option<bool>,
    pub audio_state: Option<String>,
    pub canonical_unresolved: Option<String>,
    pub canonical_state: Option<String>,
    pub canonical_utterance_id: Option<String>,
    pub source_speaker_candidates: Option<Vec<String>>,
    pub protected_fields: Vec<String>,
    pub start_sample: Option<i64>,
    pub end_sample: Option<i64>,
    pub audio_revision: Option<i64>,
    pub speaker_activity: Option<SpeakerActivity>,
}

impl AudioSegment {
    pub fn passage(id: &str, start: f64, end: f64) -> Self {
        Self {
            id: id.to_string(),
            speaker: "speaker_0".to_string(),
            start,
            end,
            text: String::new(),
            review: None,
            finalized: None,
            speaker_candidates: None,
            voice_eligible: None,
            audio_state: None,
            canonical_unresolved: None,
            canonical_state: None,
            canonical_utterance_id: None,
            source_speaker_candidates: None,
            protected_fields: Vec::new(),
            start_sample: None,
            end_sample: None,
            audio_revision: None,
            speaker_activity: None,
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct RecognitionMeeting {
    pub id: String,
    pub provenance_kind: Option<String>,
    pub speakers: BTreeMap<String, String>,
    pub segments: Vec<AudioSegment>,
    pub assignments: BTreeMap<String, SpeakerAssignment>,
    pub checks: BTreeMap<String, VoiceCheck>,
    pub voice_suggestions: Vec<String>,
    pub dismissed_identity_suggestions: Vec<String>,
}

impl RecognitionMeeting {
    pub fn local(id: impl Into<String>, segments: Vec<AudioSegment>) -> Self {
        Self {
            id: id.into(),
            provenance_kind: Some("local_inference".to_string()),
            speakers: BTreeMap::from([("speaker_0".to_string(), "Speaker 1".to_string())]),
            segments,
            assignments: BTreeMap::new(),
            checks: BTreeMap::new(),
            voice_suggestions: Vec::new(),
            dismissed_identity_suggestions: Vec::new(),
        }
    }

    fn is_local(&self) -> bool {
        self.provenance_kind.as_deref() == Some("local_inference")
    }

    fn suggestion_job(&self) -> SuggestionJob {
        SuggestionJob {
            document: Some(SuggestionDocument {
                speakers: self.speakers.clone(),
                segments: self
                    .segments
                    .iter()
                    .map(|segment| SuggestionSegment {
                        id: segment.id.clone(),
                        speaker: segment.speaker.clone(),
                        start: segment.start,
                        end: segment.end,
                        text: segment.text.clone(),
                        review: segment.review,
                        finalized: segment.finalized,
                        speaker_candidates: segment.speaker_candidates.clone(),
                    })
                    .collect(),
                provenance: Some(SuggestionProvenance {
                    kind: self.provenance_kind.clone(),
                }),
            }),
            dismissed_identity_suggestions: self.dismissed_identity_suggestions.clone(),
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct VoiceCheck {
    pub id: String,
    pub status: String,
    pub checked_at: Option<f64>,
    pub model: VoiceModel,
    pub clips: Vec<VoiceClip>,
}

pub fn speaker_audio_reason(segment: &AudioSegment, track_id: &str) -> Option<AudioReason> {
    if segment.speaker != track_id {
        return Some(AudioReason::DifferentSpeaker);
    }
    if truthy(&segment.audio_state) {
        return Some(AudioReason::UnverifiedAudio);
    }
    if segment.finalized == Some(false) {
        return Some(AudioReason::Unfinished);
    }
    let count = segment.speaker_candidates.as_ref().map_or(1, Vec::len);
    if track_id.starts_with("overlap") || count > 1 {
        return Some(AudioReason::Overlap);
    }
    let eligible = segment
        .voice_eligible
        .unwrap_or(segment.review != Some(true));
    if count != 1 || !eligible {
        Some(AudioReason::UnverifiedAudio)
    } else {
        None
    }
}

pub fn eligible_segment(segment: &AudioSegment, track_id: &str) -> bool {
    speaker_audio_reason(segment, track_id).is_none()
        && duration_in_range(segment.end - segment.start)
}

pub fn speaker_audio_ranges(segment: &AudioSegment, track_id: &str) -> Vec<(f64, f64)> {
    if speaker_audio_reason(segment, track_id).is_none() {
        return vec![(segment.start, segment.end)];
    }
    if !crop_anchor_ok(segment, track_id) {
        return Vec::new();
    }
    let (Some(start_sample), Some(end_sample), Some(revision)) = (
        segment.start_sample,
        segment.end_sample,
        segment.audio_revision,
    ) else {
        return Vec::new();
    };
    if start_sample < 0 || start_sample >= end_sample || revision < 0 {
        return Vec::new();
    }
    let rate = f64::from(limits::SAMPLE_RATE);
    if !same_f64(segment.start, start_sample as f64 / rate)
        || !same_f64(segment.end, end_sample as f64 / rate)
    {
        return Vec::new();
    }
    let Some(activity) = segment.speaker_activity.as_ref() else {
        return Vec::new();
    };
    if activity.audio_revision != revision || activity.observed_end_sample != end_sample {
        return Vec::new();
    }
    let mut ranges: Vec<(i64, i64)> = Vec::new();
    let mut cursor = start_sample;
    for region in &activity.regions {
        if region.start_sample != cursor
            || region.start_sample >= region.end_sample
            || region.end_sample > end_sample
        {
            return Vec::new();
        }
        let owned = if region.speakers.is_empty() {
            false
        } else if region.speakers.len() == 1 && region.speakers[0] == track_id {
            true
        } else {
            return Vec::new();
        };
        if owned {
            if let Some(last) = ranges
                .last_mut()
                .filter(|last| last.1 == region.start_sample)
            {
                last.1 = region.end_sample;
            } else {
                ranges.push((region.start_sample, region.end_sample));
            }
        }
        cursor = region.end_sample;
    }
    if cursor != end_sample {
        return Vec::new();
    }
    ranges
        .into_iter()
        .map(|(left, right)| (left as f64 / rate, right as f64 / rate))
        .collect()
}

fn crop_anchor_ok(segment: &AudioSegment, track_id: &str) -> bool {
    segment.speaker == track_id
        && !truthy(&segment.audio_state)
        && NUMBERED_TRACK.is_match(track_id)
        && !truthy(&segment.canonical_unresolved)
        && segment.finalized == Some(true)
        && segment.canonical_state.as_deref() == Some("sealed")
        && segment.canonical_utterance_id.as_deref() == Some(segment.id.as_str())
        && sole_owner(&segment.speaker_candidates, track_id)
        && sole_owner(&segment.source_speaker_candidates, track_id)
        && !segment
            .protected_fields
            .iter()
            .any(|field| field == "speaker" || field == "start" || field == "end")
}

pub fn voice_clip_choices(job: &RecognitionMeeting, track_id: &str) -> Vec<(String, VoiceClip)> {
    let mut choices = Vec::new();
    for segment in &job.segments {
        for (left, right) in speaker_audio_ranges(segment, track_id) {
            let mut start = left;
            while right - start >= 2.0 {
                let end = if right - left > 10.0 {
                    (start + 6.0).min(right)
                } else {
                    right
                };
                let whole = same_f64(start, segment.start) && same_f64(end, segment.end);
                let sid = if whole {
                    segment.id.clone()
                } else {
                    format!(
                        "{}@{}:{}",
                        segment.id,
                        python_round(start * f64::from(limits::SAMPLE_RATE)),
                        python_round(end * f64::from(limits::SAMPLE_RATE))
                    )
                };
                choices.push((
                    sid,
                    VoiceClip {
                        meeting_id: job.id.clone(),
                        track_id: track_id.to_string(),
                        segment_id: segment.id.clone(),
                        start,
                        end,
                    },
                ));
                start = end;
            }
        }
    }
    choices
}

pub fn automatic_clips(job: &RecognitionMeeting, track_id: &str, minimum: usize) -> Vec<VoiceClip> {
    if !job.is_local() {
        return Vec::new();
    }
    let mut clips: Vec<VoiceClip> = voice_clip_choices(job, track_id)
        .into_iter()
        .map(|(_, clip)| clip)
        .collect();
    clips.sort_by(|left, right| {
        left.end
            .partial_cmp(&right.end)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then(
                left.start
                    .partial_cmp(&right.start)
                    .unwrap_or(std::cmp::Ordering::Equal),
            )
    });
    let mut chosen = Vec::new();
    let mut last_end = -1.0;
    for clip in clips {
        if clip.start >= last_end {
            last_end = clip.end;
            chosen.push(clip);
            if chosen.len() == minimum {
                return chosen;
            }
        }
    }
    Vec::new()
}

pub fn clips_current(job: &RecognitionMeeting, clips: &[VoiceClip]) -> bool {
    if !job.is_local() {
        return false;
    }
    let mut cache: BTreeMap<&str, Vec<VoiceClip>> = BTreeMap::new();
    for clip in clips {
        if !cache.contains_key(clip.track_id.as_str()) {
            let choices = voice_clip_choices(job, &clip.track_id)
                .into_iter()
                .map(|(_, choice)| choice)
                .collect();
            cache.insert(clip.track_id.as_str(), choices);
        }
        let choices = &cache[clip.track_id.as_str()];
        if !choices.iter().any(|choice| same_clip(choice, clip)) {
            return false;
        }
    }
    true
}

pub fn clean_clips(
    job: &RecognitionMeeting,
    track_id: &str,
    segment_ids: &[String],
    minimum: usize,
) -> Result<Vec<VoiceClip>, VoiceError> {
    if !job.is_local() {
        return Err(VoiceError::FinalizedClips);
    }
    if segment_ids.len() < minimum
        || segment_ids.len() > 12
        || segment_ids.iter().collect::<BTreeSet<_>>().len() != segment_ids.len()
    {
        return Err(VoiceError::passage_count(minimum));
    }
    let choices = voice_clip_choices(job, track_id);
    let mut clips = Vec::new();
    for sid in segment_ids {
        let Some((_, clip)) = choices.iter().find(|(id, _)| id == sid) else {
            return Err(VoiceError::CleanPassage);
        };
        clips.push(clip.clone());
    }
    clips.sort_by(|left, right| {
        left.start
            .partial_cmp(&right.start)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    if clips.windows(2).any(|pair| pair[0].end > pair[1].start) {
        return Err(VoiceError::Nonoverlapping);
    }
    Ok(clips)
}

pub fn normalized(vector: &[f64], dimension: usize) -> Result<Vec<f64>, VoiceError> {
    if vector.len() != dimension || vector.iter().any(|value| !value.is_finite()) {
        return Err(VoiceError::Embedding);
    }
    let norm = vector.iter().map(|value| value * value).sum::<f64>().sqrt();
    if !norm.is_finite() || norm < 1e-12 {
        return Err(VoiceError::Embedding);
    }
    Ok(vector.iter().map(|value| value / norm).collect())
}

pub fn extract_embeddings(
    dimension: usize,
    clips: &[VoiceClip],
    mut embedder: impl VoiceEmbedder,
) -> Result<Vec<Vec<f64>>, VoiceError> {
    let mut vectors = Vec::with_capacity(clips.len());
    for clip in clips {
        let embedded = embedder.embed_clip(clip)?;
        if !embedded.clean {
            return Err(VoiceError::Unusable);
        }
        vectors.push(normalized(&embedded.vector, dimension)?);
    }
    Ok(vectors)
}

#[derive(Debug, Clone, PartialEq)]
pub struct ProfileDraft {
    pub model: VoiceModel,
    pub centroid: Vec<f64>,
    pub clips: Vec<VoiceClip>,
    pub consent: String,
}

pub fn make_profile(
    model: &VoiceModel,
    vectors: &[Vec<f64>],
    clips: &[VoiceClip],
) -> Result<ProfileDraft, VoiceError> {
    if vectors.len() < 2 || vectors.len() != clips.len() {
        return Err(VoiceError::ProfileNeedsClips);
    }
    let vectors = vectors
        .iter()
        .map(|vector| normalized(vector, model.dimension))
        .collect::<Result<Vec<_>, _>>()?;
    let mut centroid = vec![0.0; model.dimension];
    for vector in &vectors {
        for (slot, value) in centroid.iter_mut().zip(vector) {
            *slot += *value;
        }
    }
    let count = vectors.len() as f64;
    for slot in &mut centroid {
        *slot /= count;
    }
    Ok(ProfileDraft {
        model: model.clone(),
        centroid: normalized(&centroid, model.dimension)?,
        clips: clips.to_vec(),
        consent: EXPLICIT_REMEMBER_VOICE.to_string(),
    })
}

#[derive(Debug, Clone, PartialEq)]
pub struct VoiceProfile {
    pub person_id: String,
    pub version: String,
    pub enrolled_at: f64,
    pub model: VoiceModel,
    pub centroid: Vec<f64>,
    pub clips: Vec<VoiceClip>,
    pub consent: String,
}

#[derive(Debug, Clone, PartialEq)]
pub struct MatchPolicy {
    pub dataset_id: String,
    pub threshold: f64,
    pub margin: f64,
    pub minimum_clips: usize,
}

#[derive(Debug, Clone, PartialEq)]
pub struct VoiceMatch {
    pub person_id: String,
    pub profile_version: String,
    pub score: f64,
    pub margin: f64,
    pub model: VoiceModel,
    pub calibration: MatchPolicy,
}

struct Ranked<'a> {
    score: f64,
    lowest: f64,
    profile: &'a VoiceProfile,
}

pub fn rank_matches<'a>(
    model: &VoiceModel,
    vectors: &[Vec<f64>],
    profiles: &'a [VoiceProfile],
) -> Result<Vec<(f64, f64, &'a VoiceProfile)>, VoiceError> {
    let vectors = vectors
        .iter()
        .map(|vector| normalized(vector, model.dimension))
        .collect::<Result<Vec<_>, _>>()?;
    let mut ranked = Vec::new();
    for profile in profiles {
        if !same_model(&profile.model, model) {
            continue;
        }
        let centroid = normalized(&profile.centroid, model.dimension)?;
        let scores: Vec<f64> = vectors
            .iter()
            .map(|vector| dot(vector, &centroid).clamp(-1.0, 1.0))
            .collect();
        let lowest = scores.iter().copied().reduce(f64::min).unwrap_or(0.0);
        let score = if scores.is_empty() {
            0.0
        } else {
            scores.iter().sum::<f64>() / scores.len() as f64
        };
        ranked.push(Ranked {
            score,
            lowest,
            profile,
        });
    }
    ranked.sort_by(|left, right| {
        right
            .score
            .partial_cmp(&left.score)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    Ok(ranked
        .into_iter()
        .map(|row| (row.score, row.lowest, row.profile))
        .collect())
}

pub fn propose_match(
    model: &VoiceModel,
    calibration: &Calibration,
    vectors: &[Vec<f64>],
    profiles: &[VoiceProfile],
) -> Result<Option<VoiceMatch>, VoiceError> {
    if !same_model(&calibration.model, model) || vectors.len() < calibration.minimum_clips {
        return Ok(None);
    }
    let ranked = rank_matches(model, vectors, profiles)?;
    Ok(match_from_ranked(model, calibration, &ranked))
}

fn match_from_ranked(
    model: &VoiceModel,
    calibration: &Calibration,
    ranked: &[(f64, f64, &VoiceProfile)],
) -> Option<VoiceMatch> {
    let (score, lowest, best) = ranked.first()?;
    let runner_up = ranked.get(1).map(|row| row.0).unwrap_or(0.0);
    let margin = score - runner_up;
    if *lowest < calibration.threshold || margin < calibration.margin {
        return None;
    }
    Some(VoiceMatch {
        person_id: best.person_id.clone(),
        profile_version: best.version.clone(),
        score: *score,
        margin,
        model: model.clone(),
        calibration: MatchPolicy {
            dataset_id: calibration.dataset_id.clone(),
            threshold: calibration.threshold,
            margin: calibration.margin,
            minimum_clips: calibration.minimum_clips,
        },
    })
}

pub struct Enrollment<'a> {
    pub people: &'a People,
    pub assignment: Option<&'a SpeakerAssignment>,
    pub person_id: &'a str,
    pub consent: bool,
    pub draft: ProfileDraft,
    pub version: String,
    pub enrolled_at: f64,
}

#[derive(Debug, Clone, Default, PartialEq)]
pub struct VoiceBook {
    profiles: Vec<VoiceProfile>,
}

impl VoiceBook {
    pub fn profiles(&self) -> &[VoiceProfile] {
        &self.profiles
    }

    /// Save a profile for an existing person. The supplied person id is kept
    /// even when another profile has cosine >= [`IDENTITY_REUSE_COSINE`].
    pub fn remember(&mut self, enrollment: Enrollment<'_>) -> Result<VoiceProfile, VoiceError> {
        enrollment.people.get(enrollment.person_id)?;
        require_remember_consent(enrollment.consent)?;
        if blocks_enrollment(enrollment.assignment, enrollment.person_id) {
            return Err(VoiceError::RememberBeforeConfirm);
        }
        if enrollment.draft.clips.len() < 2 {
            return Err(VoiceError::ProfileNeedsClips);
        }
        require_profile_consent(enrollment.consent, &enrollment.draft.consent)?;
        let profile = VoiceProfile {
            person_id: enrollment.person_id.to_string(),
            version: enrollment.version,
            enrolled_at: enrollment.enrolled_at,
            model: enrollment.draft.model,
            centroid: enrollment.draft.centroid,
            clips: enrollment.draft.clips,
            consent: enrollment.draft.consent,
        };
        self.profiles
            .retain(|saved| saved.person_id != profile.person_id);
        self.profiles.push(profile.clone());
        Ok(profile)
    }

    pub fn forget(&mut self, people: &People, person_id: &str) -> Result<(), VoiceError> {
        people.get(person_id)?;
        self.profiles
            .retain(|profile| profile.person_id != person_id);
        Ok(())
    }
}

fn blocks_enrollment(assignment: Option<&SpeakerAssignment>, person_id: &str) -> bool {
    let Some(assignment) = assignment else {
        return true;
    };
    assignment.person_id.as_deref() != Some(person_id)
        || assignment.source == SOURCE_AUTOMATIC_VOICE
        || assignment.confirmed == Some(false)
}

#[derive(Debug, Clone, PartialEq)]
pub struct ScheduledCheck {
    pub track_id: String,
    pub check_id: String,
    pub model: VoiceModel,
    pub calibration: Calibration,
    pub clips: Vec<VoiceClip>,
    pub profile_versions: Vec<(String, String)>,
}

#[derive(Clone, Copy)]
pub struct RecognitionContext<'a> {
    pub closed: bool,
    pub preference_enabled: bool,
    pub audio_available: bool,
    pub execution_allowed: bool,
    pub model: Option<&'a VoiceModel>,
    pub calibration: Option<&'a Calibration>,
    pub profiles: &'a [VoiceProfile],
    pub people: &'a People,
}

pub fn plan_checks(
    job: &RecognitionMeeting,
    ctx: &RecognitionContext<'_>,
    only_track: Option<&str>,
) -> Vec<ScheduledCheck> {
    if ctx.closed || !ctx.preference_enabled {
        return Vec::new();
    }
    if !runtime_available(ctx.model, ctx.calibration, ctx.execution_allowed) {
        return Vec::new();
    }
    let (Some(model), Some(calibration)) = (ctx.model, ctx.calibration) else {
        return Vec::new();
    };
    let profiles: Vec<&VoiceProfile> = ctx
        .profiles
        .iter()
        .filter(|profile| same_model(&profile.model, model))
        .collect();
    if profiles.is_empty() || !job.is_local() {
        return Vec::new();
    }
    let tracks: Vec<&str> = if let Some(track) = only_track {
        vec![track]
    } else {
        job.speakers.keys().map(String::as_str).collect()
    };
    let versions: Vec<(String, String)> = profiles
        .iter()
        .map(|profile| (profile.person_id.clone(), profile.version.clone()))
        .collect();
    let mut planned = Vec::new();
    for track in tracks {
        let Some(label) = job.speakers.get(track) else {
            continue;
        };
        if job.assignments.contains_key(track)
            || !is_unassigned_speaker_label(label)
            || job.checks.contains_key(track)
        {
            continue;
        }
        let clips = automatic_clips(job, track, calibration.minimum_clips);
        if clips.is_empty() || !ctx.audio_available {
            continue;
        }
        planned.push(ScheduledCheck {
            track_id: track.to_string(),
            check_id: String::new(),
            model: model.clone(),
            calibration: calibration.clone(),
            clips,
            profile_versions: versions.clone(),
        });
    }
    planned
}

pub fn commit_plan(
    job: &mut RecognitionMeeting,
    mut planned: Vec<ScheduledCheck>,
    mut ids: impl FnMut() -> String,
) -> Vec<ScheduledCheck> {
    for check in &mut planned {
        let id = ids();
        check.check_id.clone_from(&id);
        job.checks.insert(
            check.track_id.clone(),
            VoiceCheck {
                id,
                status: CHECK_CHECKING.to_string(),
                checked_at: None,
                model: check.model.clone(),
                clips: check.clips.clone(),
            },
        );
    }
    planned
}

#[derive(Debug, Clone, PartialEq)]
pub struct CheckOutcome {
    pub matched: Option<VoiceMatch>,
    pub failed: bool,
}

pub fn recognize_track(
    model: &VoiceModel,
    calibration: &Calibration,
    clips: &[VoiceClip],
    profiles: &[VoiceProfile],
    embedder: impl VoiceEmbedder,
) -> CheckOutcome {
    match extract_embeddings(model.dimension, clips, embedder) {
        Err(_) => CheckOutcome {
            matched: None,
            failed: true,
        },
        Ok(vectors) => match propose_match(model, calibration, &vectors, profiles) {
            Ok(matched) => CheckOutcome {
                matched,
                failed: false,
            },
            Err(_) => CheckOutcome {
                matched: None,
                failed: true,
            },
        },
    }
}

pub fn apply_check(
    job: &mut RecognitionMeeting,
    scheduled: &ScheduledCheck,
    ctx: &RecognitionContext<'_>,
    outcome: &CheckOutcome,
    checked_at: f64,
) {
    if ctx.closed {
        return;
    }
    let Some(check) = job.checks.get(&scheduled.track_id) else {
        return;
    };
    if check.id != scheduled.check_id || check.status != CHECK_CHECKING {
        return;
    }
    let unchanged = recognition_unchanged(job, scheduled, ctx);
    let assignment = if unchanged && !outcome.failed {
        matched_assignment(job, scheduled, ctx, outcome)
    } else {
        None
    };
    let status = if !unchanged {
        CHECK_CANCELLED
    } else if outcome.failed {
        CHECK_UNAVAILABLE
    } else if assignment.is_some() {
        CHECK_MATCHED
    } else {
        CHECK_UNKNOWN
    };
    if let Some(assignment) = assignment {
        job.speakers
            .insert(scheduled.track_id.clone(), assignment.name.clone());
        job.assignments
            .insert(scheduled.track_id.clone(), assignment);
    }
    if let Some(check) = job.checks.get_mut(&scheduled.track_id) {
        check.status = status.to_string();
        check.checked_at = Some(checked_at);
    }
}

/// Text, timing, and import edits drop suggestion evidence and cancel open checks.
pub fn reconcile_meeting(job: &mut RecognitionMeeting, imported: bool) {
    let mut checks = job
        .checks
        .iter()
        .map(|(track, check)| (track.clone(), check.status.clone()))
        .collect();
    job.assignments = reconcile_assignments(
        std::mem::take(&mut job.assignments),
        &job.speakers,
        imported,
        &mut checks,
        &mut job.voice_suggestions,
    );
    for (track, status) in checks {
        if let Some(check) = job.checks.get_mut(&track) {
            check.status = status;
        }
    }
}

fn matched_assignment(
    job: &RecognitionMeeting,
    scheduled: &ScheduledCheck,
    ctx: &RecognitionContext<'_>,
    outcome: &CheckOutcome,
) -> Option<SpeakerAssignment> {
    let matched = outcome.matched.as_ref()?;
    let person = ctx.people.get(&matched.person_id).ok()?;
    let intro = people::introduction_suggestions(&job.suggestion_job())
        .into_iter()
        .find(|suggestion| suggestion.track_id == scheduled.track_id);
    if let Some(intro) = intro
        && intro.name.to_lowercase() != person.name.to_lowercase()
    {
        return None;
    }
    Some(SpeakerAssignment {
        person_id: Some(person.id.clone()),
        meeting_id: job.id.clone(),
        track_id: scheduled.track_id.clone(),
        name: person.name.clone(),
        confirmed: Some(false),
        source: SOURCE_AUTOMATIC_VOICE.to_string(),
    })
}

fn recognition_unchanged(
    job: &RecognitionMeeting,
    scheduled: &ScheduledCheck,
    ctx: &RecognitionContext<'_>,
) -> bool {
    let Some(model) = ctx.model else {
        return false;
    };
    let Some(calibration) = ctx.calibration else {
        return false;
    };
    if !same_model(model, &scheduled.model)
        || !same_calibration(calibration, &scheduled.calibration)
    {
        return false;
    }
    let current: BTreeSet<(String, String)> = ctx
        .profiles
        .iter()
        .filter(|profile| same_model(&profile.model, model))
        .map(|profile| (profile.person_id.clone(), profile.version.clone()))
        .collect();
    let scheduled_versions: BTreeSet<(String, String)> =
        scheduled.profile_versions.iter().cloned().collect();
    let label_ok = job
        .speakers
        .get(&scheduled.track_id)
        .is_some_and(|label| is_unassigned_speaker_label(label));
    clips_current(job, &scheduled.clips)
        && current == scheduled_versions
        && ctx.preference_enabled
        && !job.assignments.contains_key(&scheduled.track_id)
        && label_ok
}

fn sole_owner(candidates: &Option<Vec<String>>, track_id: &str) -> bool {
    matches!(candidates, Some(owners) if owners.len() == 1 && owners[0] == track_id)
}

fn truthy(value: &Option<String>) -> bool {
    value.as_ref().is_some_and(|text| !text.is_empty())
}

fn duration_in_range(duration: f64) -> bool {
    duration.is_finite() && (2.0..=10.0).contains(&duration)
}

fn dot(left: &[f64], right: &[f64]) -> f64 {
    left.iter().zip(right).map(|(a, b)| a * b).sum()
}

fn python_round(value: f64) -> i64 {
    let rounded = value.round_ties_even();
    if rounded.is_finite() {
        rounded as i64
    } else {
        0
    }
}

#[allow(clippy::float_cmp)]
fn same_f64(left: f64, right: f64) -> bool {
    left == right
}

#[allow(clippy::float_cmp)]
fn same_clip(left: &VoiceClip, right: &VoiceClip) -> bool {
    left == right
}

fn same_model(left: &VoiceModel, right: &VoiceModel) -> bool {
    left == right
}

#[allow(clippy::float_cmp)]
fn same_calibration(left: &Calibration, right: &Calibration) -> bool {
    left == right
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::people::People;

    fn model() -> VoiceModel {
        VoiceModel::new(
            "synthetic-test-model",
            "test-revision",
            "a".repeat(64),
            2,
            limits::SAMPLE_RATE,
        )
        .unwrap()
    }

    fn policy(model: &VoiceModel) -> Calibration {
        Calibration::new(
            model.clone(),
            "synthetic-test-calibration",
            0.8,
            0.1,
            20,
            100,
            0.0,
            0.05,
            2,
        )
        .unwrap()
    }

    fn saved(model: &VoiceModel, person_id: &str, centroid: Vec<f64>) -> VoiceProfile {
        VoiceProfile {
            person_id: person_id.to_string(),
            version: "profile-v1".to_string(),
            enrolled_at: 1.0,
            model: model.clone(),
            centroid,
            clips: Vec::new(),
            consent: EXPLICIT_REMEMBER_VOICE.to_string(),
        }
    }

    fn two_turns(text: &str) -> RecognitionMeeting {
        let mut first = AudioSegment::passage("intro", 0.0, 3.0);
        first.text = text.to_string();
        let mut second = AudioSegment::passage("second", 4.0, 7.0);
        second.text = "We can start now.".to_string();
        RecognitionMeeting::local("meeting", vec![first, second])
    }

    fn confirmed(person_id: &str, name: &str) -> SpeakerAssignment {
        SpeakerAssignment {
            person_id: Some(person_id.to_string()),
            meeting_id: "meeting".to_string(),
            track_id: "speaker_0".to_string(),
            name: name.to_string(),
            confirmed: Some(true),
            source: "manual".to_string(),
        }
    }

    fn context<'a>(
        model: &'a VoiceModel,
        calibration: &'a Calibration,
        profiles: &'a [VoiceProfile],
        people: &'a People,
    ) -> RecognitionContext<'a> {
        RecognitionContext {
            closed: false,
            preference_enabled: true,
            audio_available: true,
            execution_allowed: true,
            model: Some(model),
            calibration: Some(calibration),
            profiles,
            people,
        }
    }

    #[test]
    fn model_and_calibration_reject_unmeasured_identities() {
        let model = model();
        assert_eq!(
            VoiceModel::new("", "rev", "a".repeat(64), 2, 16_000)
                .unwrap_err()
                .to_string(),
            MODEL_IDENTITY
        );
        assert!(VoiceModel::new("id", "rev", "A".repeat(64), 2, 16_000).is_err());
        assert!(VoiceModel::new("id", "rev", "a".repeat(64), 0, 16_000).is_err());
        assert!(VoiceModel::new("id", "rev", "a".repeat(64), 2, 8_000).is_err());
        for (dataset, genuine, impostor, minimum, margin, threshold) in [
            ("", 20, 100, 2, 0.1, 0.8),
            ("set", 0, 100, 2, 0.1, 0.8),
            ("set", 20, 0, 2, 0.1, 0.8),
            ("set", 20, 100, 1, 0.1, 0.8),
            ("set", 20, 100, 2, 0.0, 0.8),
            ("set", 20, 100, 2, 0.1, f64::NAN),
        ] {
            assert!(
                Calibration::new(
                    model.clone(),
                    dataset,
                    threshold,
                    margin,
                    genuine,
                    impostor,
                    0.0,
                    0.05,
                    minimum,
                )
                .is_err(),
                "{dataset} {minimum} {margin} {threshold}"
            );
        }
        assert!(!runtime_available(None, None, true));
        assert!(!runtime_available(
            Some(&model),
            Some(&policy(&model)),
            false
        ));
        let mut other = model.clone();
        other.revision = "new".to_string();
        let shifted = policy(&other);
        assert!(!runtime_available(Some(&model), Some(&shifted), true));
        assert!(runtime_available(Some(&model), Some(&policy(&model)), true));
    }

    #[test]
    fn eligibility_and_clip_windows_stay_between_two_and_ten_seconds() {
        let short = AudioSegment::passage("short", 0.0, 0.9);
        assert!(speaker_audio_reason(&short, "speaker_0").is_none());
        assert!(!eligible_segment(&short, "speaker_0"));
        assert!(
            voice_clip_choices(&RecognitionMeeting::local("m", vec![short]), "speaker_0")
                .is_empty()
        );

        let mut reviewed = AudioSegment::passage("phrase", 0.0, 6.0);
        reviewed.review = Some(true);
        reviewed.voice_eligible = Some(true);
        assert!(eligible_segment(&reviewed, "speaker_0"));

        let mut overlap = AudioSegment::passage("mix", 0.0, 4.0);
        overlap.speaker_candidates = Some(vec!["speaker_0".to_string(), "speaker_1".to_string()]);
        assert_eq!(
            speaker_audio_reason(&overlap, "speaker_0")
                .unwrap()
                .as_str(),
            "overlap"
        );
        let mut pending = AudioSegment::passage("pending", 0.0, 4.0);
        pending.finalized = Some(false);
        assert_eq!(
            speaker_audio_reason(&pending, "speaker_0")
                .unwrap()
                .as_str(),
            "unfinished"
        );

        let error = VoiceClip::try_new("m", "speaker_0", "s", 0.0, 1.5).unwrap_err();
        assert_eq!(error.to_string(), messages::VOICE_PASSAGE_LENGTH);

        let mut long = AudioSegment::passage("long", 0.0, 25.0);
        long.voice_eligible = Some(true);
        assert!(!eligible_segment(&long, "speaker_0"));
        let choices = voice_clip_choices(&RecognitionMeeting::local("m", vec![long]), "speaker_0");
        let windows: Vec<_> = choices
            .iter()
            .map(|(id, clip)| (id.as_str(), clip.start.to_bits(), clip.end.to_bits()))
            .collect();
        assert_eq!(
            windows,
            vec![
                ("long@0:96000", 0.0f64.to_bits(), 6.0f64.to_bits()),
                ("long@96000:192000", 6.0f64.to_bits(), 12.0f64.to_bits()),
                ("long@192000:288000", 12.0f64.to_bits(), 18.0f64.to_bits()),
                ("long@288000:384000", 18.0f64.to_bits(), 24.0f64.to_bits()),
            ]
        );
    }

    #[test]
    fn canonical_solo_crops_survive_only_while_ownership_matches() {
        let segment = canonical_segment();
        let job = RecognitionMeeting::local("m", vec![segment.clone()]);
        let choices = voice_clip_choices(&job, "speaker_0");
        assert_eq!(
            choices
                .iter()
                .map(|(id, clip)| (id.as_str(), clip.start.to_bits(), clip.end.to_bits()))
                .collect::<Vec<_>>(),
            vec![
                ("canonical@8000:56000", 0.5f64.to_bits(), 3.5f64.to_bits()),
                (
                    "canonical@112000:160000",
                    7.0f64.to_bits(),
                    10.0f64.to_bits()
                ),
            ]
        );
        assert!(!eligible_segment(&segment, "speaker_0"));

        let mut renamed = job.clone();
        renamed.segments[0].text = "Actual human words".to_string();
        assert!(clips_current(
            &renamed,
            &choices
                .iter()
                .map(|(_, clip)| clip.clone())
                .collect::<Vec<_>>()
        ));

        let breaks = [
            {
                let mut broken = segment.clone();
                broken.finalized = Some(false);
                broken
            },
            {
                let mut broken = segment.clone();
                broken.audio_state = Some("insufficient_speech".to_string());
                broken
            },
            {
                let mut broken = segment.clone();
                broken.canonical_unresolved = Some("unresolved_alignment".to_string());
                broken
            },
            {
                let mut broken = segment.clone();
                broken.protected_fields = vec!["speaker".to_string()];
                broken
            },
            {
                let mut broken = segment.clone();
                broken.speaker_candidates =
                    Some(vec!["speaker_0".to_string(), "speaker_1".to_string()]);
                broken
            },
            {
                let mut broken = segment.clone();
                broken.start = 0.1;
                broken
            },
            {
                let mut broken = segment.clone();
                broken.speaker_activity.as_mut().unwrap().audio_revision = 2;
                broken
            },
            {
                let mut broken = segment.clone();
                broken
                    .speaker_activity
                    .as_mut()
                    .unwrap()
                    .observed_end_sample = 160_000;
                broken
            },
            {
                let mut broken = segment.clone();
                broken.speaker_activity.as_mut().unwrap().regions = vec![ActivityRegion {
                    start_sample: 0,
                    end_sample: 192_000,
                    speakers: vec!["speaker_1".to_string()],
                }];
                broken
            },
        ];
        for broken in breaks {
            assert!(
                voice_clip_choices(&RecognitionMeeting::local("m", vec![broken]), "speaker_0")
                    .is_empty()
            );
        }
    }

    fn canonical_segment() -> AudioSegment {
        let mut segment = AudioSegment::passage("canonical", 0.0, 12.0);
        segment.finalized = Some(true);
        segment.canonical_state = Some("sealed".to_string());
        segment.canonical_utterance_id = Some("canonical".to_string());
        segment.start_sample = Some(0);
        segment.end_sample = Some(192_000);
        segment.audio_revision = Some(3);
        segment.speaker_candidates = Some(vec!["speaker_0".to_string()]);
        segment.source_speaker_candidates = Some(vec!["speaker_0".to_string()]);
        segment.voice_eligible = Some(false);
        segment.speaker_activity = Some(SpeakerActivity {
            audio_revision: 3,
            observed_end_sample: 192_000,
            regions: vec![
                ActivityRegion {
                    start_sample: 0,
                    end_sample: 8_000,
                    speakers: vec![],
                },
                ActivityRegion {
                    start_sample: 8_000,
                    end_sample: 56_000,
                    speakers: vec!["speaker_0".to_string()],
                },
                ActivityRegion {
                    start_sample: 56_000,
                    end_sample: 112_000,
                    speakers: vec![],
                },
                ActivityRegion {
                    start_sample: 112_000,
                    end_sample: 160_000,
                    speakers: vec!["speaker_0".to_string()],
                },
                ActivityRegion {
                    start_sample: 160_000,
                    end_sample: 192_000,
                    speakers: vec![],
                },
            ],
        });
        segment
    }

    #[test]
    fn clean_clips_require_local_nonoverlapping_passages() {
        let job = two_turns("I'm Priya.");
        let clips = clean_clips(
            &job,
            "speaker_0",
            &["intro".to_string(), "second".to_string()],
            2,
        )
        .unwrap();
        assert_eq!(clips.len(), 2);
        assert_eq!(
            clean_clips(&job, "speaker_0", &["intro".to_string()], 2)
                .unwrap_err()
                .to_string(),
            messages::clean_passage_count(2)
        );
        assert_eq!(
            clean_clips(
                &job,
                "speaker_0",
                &["intro".to_string(), "intro".to_string()],
                2
            )
            .unwrap_err()
            .to_string(),
            messages::clean_passage_count(2)
        );
        assert_eq!(
            clean_clips(
                &job,
                "speaker_0",
                &["intro".to_string(), "missing".to_string()],
                2
            )
            .unwrap_err()
            .to_string(),
            messages::CLEAN_PASSAGE_LENGTH
        );
        let mut overlap = job.clone();
        overlap.segments[1].speaker_candidates =
            Some(vec!["speaker_0".to_string(), "speaker_1".to_string()]);
        assert!(
            clean_clips(
                &overlap,
                "speaker_0",
                &["intro".to_string(), "second".to_string()],
                2
            )
            .is_err()
        );
        let mut imported = job.clone();
        imported.provenance_kind = Some("imported".to_string());
        assert_eq!(
            clean_clips(
                &imported,
                "speaker_0",
                &["intro".to_string(), "second".to_string()],
                2
            )
            .unwrap_err()
            .to_string(),
            messages::USE_FINALIZED_CLIPS
        );
        assert!(automatic_clips(&imported, "speaker_0", 2).is_empty());

        let mut touching = AudioSegment::passage("long", 0.0, 12.0);
        touching.voice_eligible = Some(true);
        let long_job = RecognitionMeeting::local("m", vec![touching]);
        let ids = ["long@0:96000".to_string(), "long@96000:192000".to_string()];
        assert!(clean_clips(&long_job, "speaker_0", &ids, 2).is_ok());
        let duplicate = two_turns("");
        let mut crowded = duplicate.clone();
        crowded
            .segments
            .push(AudioSegment::passage("duplicate-time", 0.0, 3.0));
        crowded.segments.push({
            let mut third = AudioSegment::passage("third", 8.0, 11.0);
            third.text = "Later".to_string();
            third
        });
        // `second` already occupies 4–7. `duplicate-time` overlaps `intro`.
        let overlapping = clean_clips(
            &crowded,
            "speaker_0",
            &["intro".to_string(), "duplicate-time".to_string()],
            2,
        )
        .unwrap_err();
        assert_eq!(overlapping.to_string(), messages::NONOVERLAPPING_PASSAGES);
        assert_eq!(
            automatic_clips(&crowded, "speaker_0", 3)
                .iter()
                .map(|clip| clip.segment_id.as_str())
                .collect::<Vec<_>>(),
            vec!["intro", "second", "third"]
        );
    }

    #[test]
    fn enrollment_needs_consent_confirmation_and_two_clips() {
        let model = model();
        let mut people = People::default();
        let person = people.create_with_id("Priya", 1.0, "p1").unwrap();
        let job = two_turns("hello");
        let clips = clean_clips(
            &job,
            "speaker_0",
            &["intro".to_string(), "second".to_string()],
            2,
        )
        .unwrap();
        let mut calls = 0;
        let vectors = extract_embeddings(model.dimension, &clips, |clip: &VoiceClip| {
            calls += 1;
            assert_eq!(clip.meeting_id, "meeting");
            Ok(ClipEmbedding {
                vector: vec![calls as f64, 0.0],
                clean: true,
            })
        })
        .unwrap();
        assert_eq!(calls, 2);
        assert_eq!(
            extract_embeddings(model.dimension, &clips, |_clip: &VoiceClip| {
                Ok(ClipEmbedding {
                    vector: vec![1.0, 0.0],
                    clean: false,
                })
            })
            .unwrap_err()
            .to_string(),
            messages::PASSAGE_UNUSABLE
        );
        assert_eq!(
            extract_embeddings(model.dimension, &clips, |_clip: &VoiceClip| {
                Ok(ClipEmbedding {
                    vector: vec![f64::NAN, 0.0],
                    clean: true,
                })
            })
            .unwrap_err()
            .to_string(),
            INVALID_EMBEDDING
        );
        assert_eq!(
            make_profile(&model, &vectors[..1], &clips[..1])
                .unwrap_err()
                .to_string(),
            messages::PROFILE_NEEDS_CLIPS
        );
        let draft = make_profile(&model, &vectors, &clips).unwrap();
        assert_eq!(draft.consent, EXPLICIT_REMEMBER_VOICE);
        assert_eq!(draft.centroid.len(), 2);

        let mut book = VoiceBook::default();
        let assignment = confirmed(&person.id, "Priya");
        let missing = book.remember(Enrollment {
            people: &people,
            assignment: Some(&assignment),
            person_id: "gone",
            consent: true,
            draft: draft.clone(),
            version: "v".to_string(),
            enrolled_at: 2.0,
        });
        assert_eq!(missing.unwrap_err().to_string(), messages::PERSON_MISSING);
        let refused = book.remember(Enrollment {
            people: &people,
            assignment: Some(&assignment),
            person_id: &person.id,
            consent: false,
            draft: draft.clone(),
            version: "v".to_string(),
            enrolled_at: 2.0,
        });
        assert_eq!(refused.unwrap_err().to_string(), messages::REMEMBER_CONSENT);
        let unconfirmed = book.remember(Enrollment {
            people: &people,
            assignment: None,
            person_id: &person.id,
            consent: true,
            draft: draft.clone(),
            version: "v".to_string(),
            enrolled_at: 2.0,
        });
        assert_eq!(
            unconfirmed.unwrap_err().to_string(),
            messages::REMEMBER_BEFORE_CONFIRM
        );
        let mut automatic = assignment.clone();
        automatic.source = SOURCE_AUTOMATIC_VOICE.to_string();
        automatic.confirmed = Some(false);
        assert_eq!(
            book.remember(Enrollment {
                people: &people,
                assignment: Some(&automatic),
                person_id: &person.id,
                consent: true,
                draft: draft.clone(),
                version: "v".to_string(),
                enrolled_at: 2.0,
            })
            .unwrap_err()
            .to_string(),
            messages::REMEMBER_BEFORE_CONFIRM
        );
        let mut unmarked = draft.clone();
        unmarked.consent = "inferred".to_string();
        assert_eq!(
            book.remember(Enrollment {
                people: &people,
                assignment: Some(&assignment),
                person_id: &person.id,
                consent: true,
                draft: unmarked,
                version: "v".to_string(),
                enrolled_at: 2.0,
            })
            .unwrap_err()
            .to_string(),
            messages::REMEMBER_PROFILE
        );
        let mut one = draft.clone();
        one.clips.pop();
        assert_eq!(
            book.remember(Enrollment {
                people: &people,
                assignment: Some(&assignment),
                person_id: &person.id,
                consent: true,
                draft: one,
                version: "v".to_string(),
                enrolled_at: 2.0,
            })
            .unwrap_err()
            .to_string(),
            messages::PROFILE_NEEDS_CLIPS
        );
        let stored = book
            .remember(Enrollment {
                people: &people,
                assignment: Some(&assignment),
                person_id: &person.id,
                consent: true,
                draft,
                version: "v1".to_string(),
                enrolled_at: 9.0,
            })
            .unwrap();
        assert_eq!(stored.person_id, "p1");
        assert_eq!(stored.consent, EXPLICIT_REMEMBER_VOICE);
        assert_eq!(people.len(), 1);
        assert_eq!(book.profiles().len(), 1);
    }

    #[test]
    fn matching_uses_threshold_margin_and_minimum_clips() {
        let model = model();
        let calibration = policy(&model);
        let good = saved(&model, "saved-person", vec![1.0, 0.0]);
        let runner = saved(&model, "runner-up", vec![0.99, 0.01]);
        let cases = [
            (vec![vec![1.0, 0.0]], vec![good.clone()]),
            (vec![vec![1.0, 0.0], vec![0.1, 1.0]], vec![good.clone()]),
            (vec![vec![0.0, 1.0], vec![0.0, 1.0]], vec![good.clone()]),
            (
                vec![vec![1.0, 0.0], vec![1.0, 0.0]],
                vec![good.clone(), runner],
            ),
            (vec![vec![1.0, 0.0], vec![1.0, 0.0]], vec![]),
        ];
        for (vectors, profiles) in cases {
            assert!(
                propose_match(&model, &calibration, &vectors, &profiles)
                    .unwrap()
                    .is_none()
            );
        }
        let mut revised = model.clone();
        revised.revision = "new".to_string();
        assert!(
            propose_match(
                &model,
                &calibration,
                &[vec![1.0, 0.0], vec![1.0, 0.0]],
                &[saved(&revised, "saved-person", vec![1.0, 0.0])]
            )
            .unwrap()
            .is_none()
        );
        let shifted = policy(&revised);
        assert!(
            propose_match(
                &model,
                &shifted,
                &[vec![1.0, 0.0], vec![1.0, 0.0]],
                std::slice::from_ref(&good)
            )
            .unwrap()
            .is_none()
        );
        let matched = propose_match(
            &model,
            &calibration,
            &[vec![2.0, 0.0], vec![3.0, 0.0]],
            &[good],
        )
        .unwrap()
        .unwrap();
        assert_eq!(matched.person_id, "saved-person");
        assert_eq!(matched.calibration.threshold.to_bits(), 0.8f64.to_bits());
        assert_eq!(matched.calibration.margin.to_bits(), 0.1f64.to_bits());
        assert_eq!(matched.score.to_bits(), 1.0f64.to_bits());
        assert_eq!(matched.calibration.minimum_clips, 2);
    }

    #[test]
    fn high_cosine_does_not_mint_a_person() {
        let model = model();
        let calibration = policy(&model);
        let mut people = People::default();
        let first = people.create_with_id("Priya", 1.0, "p1").unwrap();
        let second = people.create_with_id("Priya", 1.0, "p2").unwrap();
        assert!(caller_may_reuse_identity(IDENTITY_REUSE_COSINE));
        assert!(caller_may_reuse_identity(1.0));
        assert!(!caller_may_reuse_identity(0.999899999999));
        let clips = clean_clips(
            &two_turns(""),
            "speaker_0",
            &["intro".to_string(), "second".to_string()],
            2,
        )
        .unwrap();
        let vectors = vec![vec![1.0, 0.0], vec![1.0, 0.0]];
        let draft = make_profile(&model, &vectors, &clips).unwrap();
        let assignment = confirmed(&first.id, "Priya");
        let mut book = VoiceBook::default();
        book.remember(Enrollment {
            people: &people,
            assignment: Some(&assignment),
            person_id: &first.id,
            consent: true,
            draft: draft.clone(),
            version: "v1".to_string(),
            enrolled_at: 1.0,
        })
        .unwrap();
        let other = confirmed(&second.id, "Priya");
        book.remember(Enrollment {
            people: &people,
            assignment: Some(&other),
            person_id: &second.id,
            consent: true,
            draft,
            version: "v2".to_string(),
            enrolled_at: 2.0,
        })
        .unwrap();
        assert_eq!(people.len(), 2);
        assert_eq!(book.profiles().len(), 2);
        assert!(
            propose_match(&model, &calibration, &vectors, book.profiles())
                .unwrap()
                .is_none()
        );
        let only = &book.profiles()[..1];
        assert_eq!(
            propose_match(&model, &calibration, &vectors, only)
                .unwrap()
                .unwrap()
                .person_id,
            "p1"
        );
        assert_eq!(people.len(), 2);
    }

    #[test]
    fn one_check_per_track_and_invalidation_keep_the_decision() {
        let model = model();
        let calibration = policy(&model);
        let mut people = People::default();
        let person = people.create_with_id("Alex", 1.0, "alex").unwrap();
        let profile = saved(&model, &person.id, vec![1.0, 0.0]);
        let profiles = vec![profile];
        let ctx = context(&model, &calibration, &profiles, &people);
        let mut job = two_turns("Priya, what do you think?");
        assert!(plan_checks(&job, &context_off(&ctx, false), None).is_empty());
        assert!(plan_checks(&job, &ctx, None).len() == 1);
        let planned = schedule(&mut job, &ctx, "c1");
        assert!(plan_checks(&job, &ctx, None).is_empty());
        job.segments
            .push(AudioSegment::passage("later", 10.0, 13.0));
        assert!(plan_checks(&job, &ctx, None).is_empty());

        let mut calls = 0;
        let outcome = recognize_track(
            &model,
            &calibration,
            &planned.clips,
            &profiles,
            |_: &VoiceClip| {
                calls += 1;
                Ok(ClipEmbedding {
                    vector: vec![1.0, 0.0],
                    clean: true,
                })
            },
        );
        assert_eq!(calls, 2);
        assert!(outcome.matched.is_some());
        apply_check(&mut job, &planned, &ctx, &outcome, 5.0);
        assert_eq!(job.speakers["speaker_0"], "Alex");
        assert_eq!(
            job.assignments["speaker_0"].person_id.as_deref(),
            Some("alex")
        );
        assert_eq!(job.assignments["speaker_0"].confirmed, Some(false));
        assert_eq!(job.assignments["speaker_0"].source, SOURCE_AUTOMATIC_VOICE);
        assert_eq!(job.checks["speaker_0"].status, CHECK_MATCHED);
        assert_eq!(people.len(), 1);

        let mut conflict = two_turns("I'm Priya. Welcome.");
        let planned = schedule(&mut conflict, &ctx, "c2");
        apply_check(&mut conflict, &planned, &ctx, &outcome, 6.0);
        assert!(conflict.assignments.is_empty());
        assert_eq!(conflict.speakers["speaker_0"], "Speaker 1");
        assert_eq!(conflict.checks["speaker_0"].status, CHECK_UNKNOWN);

        let mut fresh = two_turns("Welcome.");
        let planned = schedule(&mut fresh, &ctx, "c3");
        fresh.segments[0].start = 1.0;
        apply_check(&mut fresh, &planned, &ctx, &outcome, 7.0);
        assert_eq!(fresh.checks["speaker_0"].status, CHECK_CANCELLED);
        assert!(fresh.assignments.is_empty());

        let mut pending = two_turns("Welcome.");
        let planned = schedule(&mut pending, &ctx, "c4");
        let mut disabled = context(&model, &calibration, &profiles, &people);
        disabled.preference_enabled = false;
        apply_check(&mut pending, &planned, &disabled, &outcome, 8.0);
        assert_eq!(pending.checks["speaker_0"].status, CHECK_CANCELLED);

        let mut forgotten = two_turns("Welcome.");
        let planned = schedule(&mut forgotten, &ctx, "c5");
        let empty: [VoiceProfile; 0] = [];
        let gone = context(&model, &calibration, &empty, &people);
        apply_check(&mut forgotten, &planned, &gone, &outcome, 9.0);
        assert_eq!(forgotten.checks["speaker_0"].status, CHECK_CANCELLED);

        let mut manual = two_turns("Welcome.");
        let planned = schedule(&mut manual, &ctx, "c6");
        manual.assignments.insert(
            "speaker_0".to_string(),
            confirmed("alex", "A different person"),
        );
        manual
            .speakers
            .insert("speaker_0".to_string(), "A different person".to_string());
        apply_check(&mut manual, &planned, &ctx, &outcome, 10.0);
        assert_eq!(manual.checks["speaker_0"].status, CHECK_CANCELLED);
        assert_eq!(manual.assignments["speaker_0"].source, "manual");
        assert_eq!(manual.speakers["speaker_0"], "A different person");

        let mut revised = model.clone();
        revised.revision = "other".to_string();
        let next_policy = policy(&revised);
        let mut runtime = two_turns("Welcome.");
        let planned = schedule(&mut runtime, &ctx, "c7");
        let changed = context(&revised, &next_policy, &profiles, &people);
        apply_check(&mut runtime, &planned, &changed, &outcome, 11.0);
        assert_eq!(runtime.checks["speaker_0"].status, CHECK_CANCELLED);

        let mut failed_job = two_turns("Welcome.");
        let planned = schedule(&mut failed_job, &ctx, "c8");
        let failed = recognize_track(
            &model,
            &calibration,
            &planned.clips,
            &profiles,
            |_: &VoiceClip| {
                Ok(ClipEmbedding {
                    vector: vec![1.0, 0.0],
                    clean: false,
                })
            },
        );
        assert!(failed.failed);
        apply_check(&mut failed_job, &planned, &ctx, &failed, 12.0);
        assert_eq!(failed_job.checks["speaker_0"].status, CHECK_UNAVAILABLE);
        assert!(failed_job.assignments.is_empty());

        let mut edited = two_turns("Welcome.");
        edited.voice_suggestions.push("guess".to_string());
        let planned = schedule(&mut edited, &ctx, "c9");
        edited.segments[0].start = 0.5;
        reconcile_meeting(&mut edited, false);
        assert!(edited.voice_suggestions.is_empty());
        assert_eq!(edited.checks["speaker_0"].status, CHECK_CANCELLED);
        apply_check(&mut edited, &planned, &ctx, &outcome, 13.0);
        assert!(edited.assignments.is_empty());
        assert_eq!(edited.checks["speaker_0"].status, CHECK_CANCELLED);

        let quiet = two_turns("Welcome.");
        let mut no_audio = context(&model, &calibration, &profiles, &people);
        no_audio.audio_available = false;
        assert!(plan_checks(&quiet, &no_audio, None).is_empty());
        let mut other_model = model.clone();
        other_model.model_id = "different".to_string();
        let foreign = vec![saved(&other_model, "alex", vec![1.0, 0.0])];
        assert!(
            plan_checks(
                &quiet,
                &context(&model, &calibration, &foreign, &people),
                None
            )
            .is_empty()
        );
        let named = {
            let mut named = quiet.clone();
            named
                .speakers
                .insert("speaker_0".to_string(), "Alex".to_string());
            named
        };
        assert!(plan_checks(&named, &ctx, None).is_empty());
    }

    fn schedule(
        job: &mut RecognitionMeeting,
        ctx: &RecognitionContext<'_>,
        id: &str,
    ) -> ScheduledCheck {
        let planned = plan_checks(job, ctx, None);
        commit_plan(job, planned, || id.to_string())
            .into_iter()
            .next()
            .expect("track was eligible")
    }

    fn context_off<'a>(ctx: &RecognitionContext<'a>, preference: bool) -> RecognitionContext<'a> {
        RecognitionContext {
            preference_enabled: preference,
            ..*ctx
        }
    }
}
