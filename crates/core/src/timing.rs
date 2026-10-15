//! CTC segmentation, text units, and reading turns.
//! Policy ids live in [`crate::timing_policy`].
//!
//! The segmentation dynamic program is the ctc-segmentation 1.7.4 trellis
//! (Apache-2.0, Technische Universität München). It runs on a caller-supplied
//! score matrix and does not load an acoustic model.

use std::sync::OnceLock;

use regex::Regex;
use serde::Deserialize;
use sha2::{Digest, Sha256};
use thiserror::Error;
use unicode_general_category::get_general_category;

use crate::timing_policy::{
    FUNCTIONAL_POLICY_ID, LEGACY_UNIT_POLICY_ID, NORMALIZATION_POLICY_ID, PROVIDER_ID,
    TIMING_UNIT_POLICY_ID,
};

/// Reading-turn sample rate from `reading_turns.py`.
pub const RATE: i64 = 16_000;
/// Speaker-transition margin, in samples, from `reading_turns.py`.
pub const MARGIN: i64 = 4_000;
/// Qualified acoustic model from `reading_turns.py`.
pub const MODEL: &str = "38ff6a225e75caa6e19c35a9b5823015e2550451bd0a394a10f6cda87f42058d";
/// Qualified coarse calibration paired with [`MODEL`].
pub const CALIBRATION: &str = "ami-english-coarse-mlx-f32-v1-5ab4e661e62f-38ff6a225e75";
/// Earlier qualified model identity from `reading_turns.py`.
pub const LEGACY_MODEL: &str = "e7c4e54ee4c4c47829cc6667d5d00ed8ea7bef1dcfeef0fce766f77752a2726c";
/// Earlier qualified calibration paired with [`LEGACY_MODEL`].
pub const LEGACY_CALIBRATION: &str = "ami-english-coarse-v1-5ab4e661e62f";

const FILL_PROB_MAX: f32 = -1_000_000_000.0;

#[derive(Debug, Clone, PartialEq, Error)]
pub enum TimingError {
    #[error("Unknown transcript timing unit policy.")]
    UnknownUnitPolicy,
    #[error("Audio is shorter than text!")]
    AudioShorterThanText,
    #[error("CTC backtracking was not successful.")]
    BacktrackFailed,
    #[error("Invalid CTC score matrix.")]
    InvalidScores,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum UnitKind {
    WhitespaceRun,
    CtcCharacter,
}

impl UnitKind {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::WhitespaceRun => "whitespace_run",
            Self::CtcCharacter => "ctc_character",
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RawUnit {
    pub text: String,
    pub start_char: usize,
    pub end_char: usize,
    pub start_utf16: usize,
    pub end_utf16: usize,
    pub unit_kind: UnitKind,
    pub status: &'static str,
    pub start_sample: Option<i64>,
    pub end_sample: Option<i64>,
}

#[derive(Debug, Clone, PartialEq, Eq, Deserialize)]
pub struct AudioAnchor {
    pub start_sample: i64,
    pub end_sample: i64,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TimedWord {
    pub text: String,
    pub start_char: usize,
    pub end_char: usize,
    pub start_sample: Option<i64>,
    pub end_sample: Option<i64>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ReadingWordEvidence {
    pub text_sha256: String,
    pub machine_revision: i64,
    pub audio_revision: i64,
    pub retained_audio_revision: Option<i64>,
    pub audio_anchor: AudioAnchor,
    pub model_sha256: String,
    pub calibration_id: Option<String>,
    pub frame_calibration_id: Option<String>,
    pub timing_unit_policy_id: Option<String>,
    pub provider_id: Option<String>,
    pub provider_identity_sha256: Option<String>,
    pub normalization_policy_id: Option<String>,
    pub words: Vec<TimedWord>,
}

#[derive(Debug, Clone, PartialEq, Eq, Deserialize)]
pub struct ActivityRegion {
    pub start_sample: i64,
    pub end_sample: i64,
    pub speakers: Vec<String>,
}

#[derive(Debug, Clone, PartialEq, Eq, Deserialize)]
pub struct SpeakerActivity {
    pub audio_revision: i64,
    pub observed_end_sample: Option<i64>,
    pub regions: Vec<ActivityRegion>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct ReadingRow {
    pub text: String,
    pub start_sample: i64,
    pub end_sample: i64,
    pub machine_revision: i64,
    pub audio_revision: i64,
    pub protected_fields: Vec<String>,
    pub text_audio_anchor: Option<AudioAnchor>,
    pub reading_word_evidence: Option<ReadingWordEvidence>,
    pub speaker_activity: Option<SpeakerActivity>,
    /// Stand-in for `alignment_provider.manifest()` so synthetic rows stay model-free.
    /// `None` rejects the mixed-script behavior policy, matching a missing manifest.
    pub installed_provider_identity: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Deserialize)]
pub struct ReadingTurn {
    pub text: String,
    pub start_offset: usize,
    pub end_offset: usize,
    pub speaker: String,
    pub attribution: String,
    pub start: Option<f64>,
    pub end: Option<f64>,
}

#[derive(Debug, Clone, PartialEq, Eq, Deserialize)]
pub struct TurnProvenance {
    pub utterance_id: String,
    pub machine_revision: i64,
    pub canonical_machine_revision: i64,
    pub audio_revision: i64,
    pub language_epoch: i64,
    pub capture_source: Option<serde_json::Value>,
    pub text_sha256: String,
    pub audio_anchor: AudioAnchor,
    pub activity_sha256: String,
    pub turns_sha256: String,
    pub model_sha256: String,
    pub calibration_id: String,
    pub frame_calibration_id: Option<String>,
    pub timing_unit_policy_id: Option<String>,
    pub provider_id: Option<String>,
    pub provider_identity_sha256: Option<String>,
    pub normalization_policy_id: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Deserialize)]
pub struct ReadingSegment {
    pub text: String,
    #[serde(default)]
    pub protected_fields: Vec<String>,
    pub canonical_utterance_id: String,
    pub machine_revision: i64,
    pub canonical_machine_revision: i64,
    pub audio_revision: i64,
    pub language_epoch: i64,
    pub capture_source: Option<serde_json::Value>,
    pub text_audio_anchor: AudioAnchor,
    pub speaker_activity: Option<SpeakerActivity>,
    pub start: f64,
    pub end: f64,
    pub reading_turns: Vec<ReadingTurn>,
    pub reading_turn_provenance: Option<TurnProvenance>,
    // Installed provider identity is trusted runtime state, never document input.
    #[serde(skip)]
    pub installed_provider_identity: Option<String>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct CtcSegmentationParameters {
    pub blank: i64,
    pub index_duration: f64,
    pub min_window_size: usize,
    pub max_window_size: usize,
    pub max_prob: f64,
    pub skip_prob: f64,
    pub blank_transition_cost_zero: bool,
    pub preamble_transition_cost_zero: bool,
    pub backtrack_from_max_t: bool,
    pub self_transition: String,
    pub char_list: Vec<String>,
}

impl CtcSegmentationParameters {
    /// Defaults used by `align_ctc_scores`, including `index_duration = 1.0`.
    pub fn for_vocabulary(char_list: Vec<String>) -> Self {
        Self {
            blank: 0,
            index_duration: 1.0,
            min_window_size: 8_000,
            max_window_size: 100_000,
            max_prob: -10_000_000_000.0,
            skip_prob: -10_000_000_000.0,
            blank_transition_cost_zero: false,
            preamble_transition_cost_zero: true,
            backtrack_from_max_t: false,
            self_transition: "<alignment-stay>".to_string(),
            char_list,
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct CtcSegmentation {
    pub timings: Vec<f64>,
    pub char_probs: Vec<f64>,
    pub states: Vec<String>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PreparedTokens {
    pub ground_truth: Vec<Vec<i64>>,
    pub utterance_begin_indices: Vec<usize>,
}

struct Behavior<'a> {
    calibration_id: Option<&'a str>,
    frame_calibration_id: Option<&'a str>,
    timing_unit_policy_id: Option<&'a str>,
    provider_id: Option<&'a str>,
    provider_identity_sha256: Option<&'a str>,
    normalization_policy_id: Option<&'a str>,
}

struct Projection<'a> {
    evidence: &'a ReadingWordEvidence,
    anchor: &'a AudioAnchor,
    units: Vec<RawUnit>,
    regions: Vec<ActivityRegion>,
    observed: Option<i64>,
}

struct WordWindow {
    row_start: i64,
    row_end: i64,
    anchor_start: i64,
    anchor_end: i64,
    observed: Option<i64>,
}

struct FilledTable {
    table: Vec<Vec<f32>>,
    offsets: Vec<i64>,
    time: i64,
    column: i64,
}

struct SwitchQuery<'a> {
    scores: &'a [Vec<f32>],
    ground_truth: &'a [Vec<i64>],
    table: &'a [Vec<f32>],
    cur_offset: &'a [i64],
    time: usize,
    column: usize,
    offset_sum: i64,
    n_rows: usize,
}

struct StayQuery<'a> {
    scores: &'a [Vec<f32>],
    table: &'a [Vec<f32>],
    blank: i64,
    time: usize,
    column: usize,
    offset_sum: i64,
    max_lpz: f32,
    stay_zero: bool,
    preamble_zero: bool,
}

/// CJK and kana scalar ranges from `alignment_text.cjk_character`.
pub fn cjk_character(character: char) -> bool {
    let value = character as u32;
    (0x3400..=0x9fff).contains(&value)
        || (0x20000..=0x3134f).contains(&value)
        || (0x3041..=0x3096).contains(&value)
        || (0x30a1..=0x30fa).contains(&value)
        || (0xff66..=0xff9d).contains(&value)
}

/// Kana scalars from `alignment_text.kana_character`. The prolonged mark is excluded.
pub fn kana_character(character: char) -> bool {
    let value = character as u32;
    ((0x3041..=0x3096).contains(&value)
        || (0x30a1..=0x30fa).contains(&value)
        || (0xff66..=0xff9d).contains(&value))
        && character != 'ー'
}

/// Lossless display units. Unknown policies fail closed.
pub fn raw_units(text: &str, policy: &str) -> Result<Vec<RawUnit>, TimingError> {
    if policy != LEGACY_UNIT_POLICY_ID && policy != TIMING_UNIT_POLICY_ID {
        return Err(TimingError::UnknownUnitPolicy);
    }
    let chars: Vec<char> = text.chars().collect();
    let mut units = Vec::new();
    let mut index = 0;
    while index < chars.len() {
        if chars[index].is_whitespace() {
            index += 1;
            continue;
        }
        let begin = index;
        while index < chars.len() && !chars[index].is_whitespace() {
            index += 1;
        }
        let end = index;
        for (start, stop, kind) in unit_starts(&chars, begin, end, policy) {
            units.push(RawUnit {
                text: chars[start..stop].iter().collect(),
                start_char: start,
                end_char: stop,
                start_utf16: utf16_len(&chars[..start]),
                end_utf16: utf16_len(&chars[..stop]),
                unit_kind: kind,
                status: "unresolved",
                start_sample: None,
                end_sample: None,
            });
        }
    }
    Ok(units)
}

/// Project reading turns for one synthetic row. Invalid evidence yields an empty list.
pub fn project_turns(row: &ReadingRow) -> Vec<ReadingTurn> {
    let Some(projection) = prepare_projection(row) else {
        return Vec::new();
    };
    assemble_turns(row, &projection)
}

/// Accept a saved projection whose provenance still matches the segment.
pub fn validated_turns(segment: &ReadingSegment) -> Vec<ReadingTurn> {
    let Some(receipt) = segment.reading_turn_provenance.as_ref() else {
        return Vec::new();
    };
    if !provenance_matches(segment, receipt) {
        return Vec::new();
    }
    if !turns_cover_text(segment) {
        return Vec::new();
    }
    segment.reading_turns.clone()
}

/// One-column ground truth used by `prepare_token_list` in ctc-segmentation 1.7.4.
pub fn prepare_token_list(blank: i64, utterances: &[Vec<i64>]) -> PreparedTokens {
    let mut ground_truth = vec![-1];
    let mut utterance_begin_indices = Vec::new();
    for utterance in utterances {
        if *ground_truth.last().unwrap_or(&-1) != blank {
            ground_truth.push(blank);
        }
        utterance_begin_indices.push(ground_truth.len() - 1);
        ground_truth.extend_from_slice(utterance);
        if *ground_truth.last().unwrap_or(&blank) != blank {
            ground_truth.push(blank);
        }
    }
    utterance_begin_indices.push(ground_truth.len() - 1);
    PreparedTokens {
        ground_truth: ground_truth.into_iter().map(|token| vec![token]).collect(),
        utterance_begin_indices,
    }
}

/// CTC segmentation dynamic program over `scores[frame][token]` log probabilities.
pub fn ctc_segmentation(
    config: &CtcSegmentationParameters,
    scores: &[Vec<f32>],
    ground_truth: &[Vec<i64>],
) -> Result<CtcSegmentation, TimingError> {
    let width = score_width(scores)?;
    if ground_truth.is_empty() || ground_truth.iter().any(|row| row.is_empty()) {
        return Err(TimingError::InvalidScores);
    }
    if config.blank < 0 || config.blank as usize >= width {
        return Err(TimingError::InvalidScores);
    }
    if ground_truth.len() > scores.len() && config.skip_prob <= config.max_prob {
        return Err(TimingError::AudioShorterThanText);
    }
    let mut window_size = config.min_window_size.max(1);
    loop {
        match segment_window(config, scores, ground_truth, window_size) {
            Ok(result) => return Ok(result),
            Err(WindowError::Backtrack) => {
                window_size = window_size.saturating_mul(2);
                if window_size < config.max_window_size {
                    continue;
                }
                return Err(TimingError::BacktrackFailed);
            }
            Err(WindowError::Invalid) => return Err(TimingError::InvalidScores),
        }
    }
}

fn unit_starts(
    chars: &[char],
    begin: usize,
    end: usize,
    policy: &str,
) -> Vec<(usize, usize, UnitKind)> {
    let mut starts = vec![(begin, UnitKind::WhitespaceRun)];
    let mixed = policy == TIMING_UNIT_POLICY_ID
        && chars[begin..end]
            .iter()
            .any(|character| cjk_character(*character) && *character != 'ー');
    if mixed {
        starts.clear();
        for (offset, character) in chars.iter().enumerate().take(end).skip(begin) {
            let character = *character;
            if decoration(character) {
                continue;
            }
            let kind = if cjk_character(character) {
                UnitKind::CtcCharacter
            } else {
                UnitKind::WhitespaceRun
            };
            let new_run = kind == UnitKind::CtcCharacter
                || starts.last().is_none_or(|(_, previous)| *previous != kind);
            if new_run {
                starts.push((offset, kind));
            }
        }
        if starts.is_empty() {
            starts.push((begin, UnitKind::WhitespaceRun));
        }
        starts[0].0 = begin;
    }
    starts
        .iter()
        .enumerate()
        .map(|(index, (start, kind))| {
            let stop = starts.get(index + 1).map(|(next, _)| *next).unwrap_or(end);
            (*start, stop, *kind)
        })
        .collect()
}

fn decoration(character: char) -> bool {
    let major = get_general_category(character).abbreviation().as_bytes()[0];
    major == b'P' || major == b'M' || character == 'ー'
}

fn utf16_len(chars: &[char]) -> usize {
    chars.iter().map(|character| character.len_utf16()).sum()
}

fn prepare_projection(row: &ReadingRow) -> Option<Projection<'_>> {
    let evidence = row.reading_word_evidence.as_ref()?;
    let anchor = row.text_audio_anchor.as_ref()?;
    if row_rejected(row, evidence, anchor) {
        return None;
    }
    let activity = row.speaker_activity.as_ref()?;
    if activity.audio_revision != row.audio_revision {
        return None;
    }
    if let Some(observed) = activity.observed_end_sample
        && !(row.start_sample..=row.end_sample).contains(&observed)
    {
        return None;
    }
    let policy = evidence
        .timing_unit_policy_id
        .as_deref()
        .unwrap_or(LEGACY_UNIT_POLICY_ID);
    let units = raw_units(&row.text, policy).ok()?;
    if units.is_empty() || units.len() != evidence.words.len() {
        return None;
    }
    let regions = merged_regions(
        &activity.regions,
        row.start_sample,
        activity.observed_end_sample.unwrap_or(row.end_sample),
    )?;
    Some(Projection {
        evidence,
        anchor,
        units,
        regions,
        observed: activity.observed_end_sample,
    })
}

fn row_rejected(row: &ReadingRow, evidence: &ReadingWordEvidence, anchor: &AudioAnchor) -> bool {
    if !row.protected_fields.is_empty() {
        return true;
    }
    let audio_revision = evidence
        .retained_audio_revision
        .unwrap_or(evidence.audio_revision);
    let calibration = evidence.calibration_id.as_deref();
    !(evidence.text_sha256 == sha256_hex(row.text.as_bytes())
        && evidence.machine_revision == row.machine_revision
        && audio_revision == row.audio_revision
        && evidence.audio_anchor == *anchor
        && anchor.start_sample == row.start_sample
        && row.start_sample < anchor.end_sample
        && anchor.end_sample <= row.end_sample
        && qualified_identity(Some(evidence.model_sha256.as_str()), calibration)
        && behavior_valid(
            &behavior_of_evidence(evidence),
            row.installed_provider_identity.as_deref(),
        ))
}

fn assemble_turns(row: &ReadingRow, projection: &Projection<'_>) -> Vec<ReadingTurn> {
    let chars: Vec<char> = row.text.chars().collect();
    let window = WordWindow {
        row_start: row.start_sample,
        row_end: row.end_sample,
        anchor_start: projection.anchor.start_sample,
        anchor_end: projection.anchor.end_sample,
        observed: projection.observed,
    };
    let mut turns = Vec::new();
    let mut position = 0;
    let mut previous_sample = row.start_sample;
    for (index, (word, unit)) in projection
        .evidence
        .words
        .iter()
        .zip(&projection.units)
        .enumerate()
    {
        if word.text != unit.text
            || word.start_char != unit.start_char
            || word.end_char != unit.end_char
        {
            return Vec::new();
        }
        let stop = projection
            .units
            .get(index + 1)
            .map(|next| next.start_char)
            .unwrap_or(chars.len());
        let (speaker, attribution, timed_end) =
            word_attribution(word, &projection.regions, &window, &mut previous_sample);
        let slice: String = chars[position..stop].iter().collect();
        push_turn(
            &mut turns,
            slice,
            position,
            stop,
            speaker,
            attribution,
            timed_end,
        );
        position = stop;
    }
    turns
}

fn word_attribution(
    word: &TimedWord,
    regions: &[ActivityRegion],
    window: &WordWindow,
    previous_sample: &mut i64,
) -> (String, String, Option<(f64, f64)>) {
    let Some((start, end)) = word.start_sample.zip(word.end_sample) else {
        return ("unassigned".to_string(), "unknown".to_string(), None);
    };
    if !(*previous_sample <= start && start < end && end <= window.anchor_end) {
        return ("unassigned".to_string(), "unknown".to_string(), None);
    }
    *previous_sample = end;
    let (speaker, attribution) = attribute_span(start, end, regions, window);
    let timed = (attribution != "unknown")
        .then_some((start as f64 / RATE as f64, end as f64 / RATE as f64));
    (speaker, attribution, timed)
}

fn push_turn(
    turns: &mut Vec<ReadingTurn>,
    slice: String,
    position: usize,
    stop: usize,
    speaker: String,
    attribution: String,
    timed: Option<(f64, f64)>,
) {
    let same = turns
        .last()
        .is_some_and(|turn| turn.speaker == speaker && turn.attribution == attribution);
    if same {
        let turn = turns.last_mut().expect("same turn exists");
        turn.end_offset = stop;
        turn.text.push_str(&slice);
        if attribution != "unknown" {
            turn.end = timed.map(|(_, end)| end);
        }
        return;
    }
    let (start, end) = match timed {
        Some((start, end)) if attribution != "unknown" => (Some(start), Some(end)),
        _ => (None, None),
    };
    turns.push(ReadingTurn {
        text: slice,
        start_offset: position,
        end_offset: stop,
        speaker,
        attribution,
        start,
        end,
    });
}

fn attribute_span(
    start: i64,
    end: i64,
    regions: &[ActivityRegion],
    window: &WordWindow,
) -> (String, String) {
    for (index, region) in regions.iter().enumerate() {
        if !span_owned(start, end, index, region, regions, window) {
            continue;
        }
        return speaker_label(&region.speakers);
    }
    ("unassigned".to_string(), "unknown".to_string())
}

fn span_owned(
    start: i64,
    end: i64,
    index: usize,
    region: &ActivityRegion,
    regions: &[ActivityRegion],
    window: &WordWindow,
) -> bool {
    let left = window.row_start.max(start - MARGIN);
    let right = window.row_end.min(end + MARGIN);
    let inside = region.start_sample <= start && start < end && end <= region.end_sample;
    if !inside
        || region.speakers.is_empty()
        || !safe_span(start, end, index, region, regions, window)
    {
        return false;
    }
    if !region.speakers.iter().all(|name| speaker_name(name)) {
        return false;
    }
    regions.iter().all(|other| {
        let overlaps = other.start_sample < right && left < other.end_sample;
        !overlaps || other.speakers.is_empty() || other.speakers == region.speakers
    })
}

fn safe_span(
    start: i64,
    end: i64,
    index: usize,
    region: &ActivityRegion,
    regions: &[ActivityRegion],
    window: &WordWindow,
) -> bool {
    let right = window.row_end.min(end + MARGIN);
    if let Some(observed) = window.observed {
        return right <= observed;
    }
    let lead = if index == 0 { 0 } else { MARGIN };
    let trail = if index + 1 < regions.len() { MARGIN } else { 0 };
    window.anchor_start + MARGIN <= start
        && end <= window.anchor_end - MARGIN
        && region.start_sample + lead <= start
        && end <= region.end_sample - trail
}

fn speaker_label(names: &[String]) -> (String, String) {
    if names.len() == 1 {
        (names[0].clone(), "single".to_string())
    } else {
        (
            format!("overlap_{}", names.join("_")),
            "overlap".to_string(),
        )
    }
}

fn merged_regions(regions: &[ActivityRegion], start: i64, end: i64) -> Option<Vec<ActivityRegion>> {
    let mut merged = Vec::new();
    let mut cursor = start;
    for region in regions {
        if cursor != region.start_sample
            || region.start_sample >= region.end_sample
            || region.end_sample > end
        {
            return None;
        }
        let mut names = region.speakers.clone();
        names.sort();
        if merged
            .last()
            .is_some_and(|last: &ActivityRegion| last.speakers == names)
        {
            merged.last_mut().expect("merged region").end_sample = region.end_sample;
        } else {
            merged.push(ActivityRegion {
                start_sample: region.start_sample,
                end_sample: region.end_sample,
                speakers: names,
            });
        }
        cursor = region.end_sample;
    }
    (cursor == end).then_some(merged)
}

fn provenance_matches(segment: &ReadingSegment, receipt: &TurnProvenance) -> bool {
    if segment.reading_turns.is_empty() || !segment.protected_fields.is_empty() {
        return false;
    }
    let behavior = Behavior {
        calibration_id: Some(receipt.calibration_id.as_str()),
        frame_calibration_id: receipt.frame_calibration_id.as_deref(),
        timing_unit_policy_id: receipt.timing_unit_policy_id.as_deref(),
        provider_id: receipt.provider_id.as_deref(),
        provider_identity_sha256: receipt.provider_identity_sha256.as_deref(),
        normalization_policy_id: receipt.normalization_policy_id.as_deref(),
    };
    receipt.utterance_id == segment.canonical_utterance_id
        && receipt.machine_revision == segment.machine_revision
        && receipt.canonical_machine_revision == segment.canonical_machine_revision
        && receipt.audio_revision == segment.audio_revision
        && receipt.language_epoch == segment.language_epoch
        && receipt.capture_source == segment.capture_source
        && receipt.text_sha256 == sha256_hex(segment.text.as_bytes())
        && receipt.audio_anchor == segment.text_audio_anchor
        && receipt.activity_sha256
            == sha256_hex(activity_json(segment.speaker_activity.as_ref()).as_bytes())
        && receipt.turns_sha256 == sha256_hex(turns_json(&segment.reading_turns).as_bytes())
        && qualified_identity(
            Some(receipt.model_sha256.as_str()),
            Some(receipt.calibration_id.as_str()),
        )
        && behavior_valid(&behavior, segment.installed_provider_identity.as_deref())
}

fn turns_cover_text(segment: &ReadingSegment) -> bool {
    let chars: Vec<char> = segment.text.chars().collect();
    let mut cursor = 0;
    for turn in &segment.reading_turns {
        if !turn_slice_valid(turn, &chars, cursor, segment.start, segment.end) {
            return false;
        }
        cursor = turn.end_offset;
    }
    cursor == chars.len()
}

fn turn_slice_valid(
    turn: &ReadingTurn,
    chars: &[char],
    cursor: usize,
    start: f64,
    end: f64,
) -> bool {
    if turn.start_offset != cursor || turn.end_offset <= cursor || turn.end_offset > chars.len() {
        return false;
    }
    let slice: String = chars[cursor..turn.end_offset].iter().collect();
    if turn.text != slice || !matches!(turn.attribution.as_str(), "single" | "overlap" | "unknown")
    {
        return false;
    }
    attribution_times_valid(turn, start, end)
}

fn attribution_times_valid(turn: &ReadingTurn, start: f64, end: f64) -> bool {
    if turn.attribution == "unknown" {
        return turn.speaker == "unassigned" && turn.start.is_none() && turn.end.is_none();
    }
    let pattern_ok = if turn.attribution == "single" {
        speaker_name(&turn.speaker)
    } else {
        overlap_name(&turn.speaker)
    };
    match (pattern_ok, turn.start, turn.end) {
        (true, Some(turn_start), Some(turn_end)) => {
            start <= turn_start && turn_start < turn_end && turn_end <= end
        }
        _ => false,
    }
}

fn behavior_of_evidence(evidence: &ReadingWordEvidence) -> Behavior<'_> {
    Behavior {
        calibration_id: evidence.calibration_id.as_deref(),
        frame_calibration_id: evidence.frame_calibration_id.as_deref(),
        timing_unit_policy_id: evidence.timing_unit_policy_id.as_deref(),
        provider_id: evidence.provider_id.as_deref(),
        provider_identity_sha256: evidence.provider_identity_sha256.as_deref(),
        normalization_policy_id: evidence.normalization_policy_id.as_deref(),
    }
}

fn behavior_valid(evidence: &Behavior<'_>, installed_provider_identity: Option<&str>) -> bool {
    let new = evidence.timing_unit_policy_id == Some(TIMING_UNIT_POLICY_ID);
    let calibration = evidence.calibration_id.or(evidence.frame_calibration_id);
    if !new {
        let fields_present = evidence.timing_unit_policy_id.is_some()
            || evidence.provider_id.is_some()
            || evidence.provider_identity_sha256.is_some()
            || evidence.normalization_policy_id.is_some();
        return calibration != Some(FUNCTIONAL_POLICY_ID) && !fields_present;
    }
    if evidence.provider_id != Some(PROVIDER_ID)
        || evidence.normalization_policy_id != Some(NORMALIZATION_POLICY_ID)
    {
        return false;
    }
    installed_provider_identity
        .is_some_and(|identity| evidence.provider_identity_sha256 == Some(identity))
}

fn qualified_identity(model: Option<&str>, calibration: Option<&str>) -> bool {
    matches!(
        (model, calibration),
        (Some(MODEL), Some(CALIBRATION))
            | (Some(MODEL), Some(FUNCTIONAL_POLICY_ID))
            | (Some(LEGACY_MODEL), Some(LEGACY_CALIBRATION))
    )
}

fn speaker_name(name: &str) -> bool {
    static PATTERN: OnceLock<Regex> = OnceLock::new();
    PATTERN
        .get_or_init(|| Regex::new(r"\Aspeaker_\d+\z").expect("speaker pattern"))
        .is_match(name)
}

fn overlap_name(name: &str) -> bool {
    static PATTERN: OnceLock<Regex> = OnceLock::new();
    PATTERN
        .get_or_init(|| {
            Regex::new(r"\Aoverlap_speaker_\d+(?:_speaker_\d+)+\z").expect("overlap pattern")
        })
        .is_match(name)
}

fn sha256_hex(bytes: &[u8]) -> String {
    hex::encode(Sha256::digest(bytes))
}

pub(crate) fn activity_json(activity: Option<&SpeakerActivity>) -> String {
    let Some(activity) = activity else {
        return "null".to_string();
    };
    let mut out = format!("{{\"audio_revision\":{}", activity.audio_revision);
    if let Some(observed) = activity.observed_end_sample {
        out.push_str(&format!(",\"observed_end_sample\":{observed}"));
    }
    out.push_str(",\"regions\":[");
    for (index, region) in activity.regions.iter().enumerate() {
        if index > 0 {
            out.push(',');
        }
        out.push_str(&format!(
            "{{\"end_sample\":{},\"speakers\":[",
            region.end_sample
        ));
        for (speaker_index, speaker) in region.speakers.iter().enumerate() {
            if speaker_index > 0 {
                out.push(',');
            }
            push_json_string(&mut out, speaker);
        }
        out.push_str(&format!("],\"start_sample\":{}}}", region.start_sample));
    }
    out.push_str("]}");
    out
}

pub(crate) fn turns_json(turns: &[ReadingTurn]) -> String {
    let mut out = String::from("[");
    for (index, turn) in turns.iter().enumerate() {
        if index > 0 {
            out.push(',');
        }
        out.push_str("{\"attribution\":");
        push_json_string(&mut out, &turn.attribution);
        out.push_str(",\"end\":");
        push_json_opt_f64(&mut out, turn.end);
        out.push_str(&format!(",\"end_offset\":{}", turn.end_offset));
        out.push_str(",\"speaker\":");
        push_json_string(&mut out, &turn.speaker);
        out.push_str(",\"start\":");
        push_json_opt_f64(&mut out, turn.start);
        out.push_str(&format!(
            ",\"start_offset\":{},\"text\":",
            turn.start_offset
        ));
        push_json_string(&mut out, &turn.text);
        out.push('}');
    }
    out.push(']');
    out
}

fn push_json_string(out: &mut String, value: &str) {
    out.push('"');
    for character in value.chars() {
        match character {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\u{0008}' => out.push_str("\\b"),
            '\u{000c}' => out.push_str("\\f"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            control if (control as u32) < 0x20 => {
                out.push_str(&format!("\\u{:04x}", control as u32));
            }
            other => out.push(other),
        }
    }
    out.push('"');
}

fn push_json_opt_f64(out: &mut String, value: Option<f64>) {
    match value {
        None => out.push_str("null"),
        Some(number) => out.push_str(&format_json_f64(number)),
    }
}

fn format_json_f64(value: f64) -> String {
    serde_json::Number::from_f64(value)
        .map(|number| number.to_string())
        .unwrap_or_else(|| "null".to_string())
}

enum WindowError {
    Backtrack,
    Invalid,
}

fn score_width(scores: &[Vec<f32>]) -> Result<usize, TimingError> {
    let Some(first) = scores.first() else {
        return Err(TimingError::InvalidScores);
    };
    if first.is_empty() || scores.iter().any(|row| row.len() != first.len()) {
        return Err(TimingError::InvalidScores);
    }
    Ok(first.len())
}

fn segment_window(
    config: &CtcSegmentationParameters,
    scores: &[Vec<f32>],
    ground_truth: &[Vec<i64>],
    window_size: usize,
) -> Result<CtcSegmentation, WindowError> {
    let n_rows = window_size.min(scores.len());
    let n_cols = ground_truth.len();
    if n_rows == 0 || n_cols == 0 {
        return Err(WindowError::Invalid);
    }
    let mut filled = fill_table(config, scores, ground_truth, n_rows)?;
    if config.backtrack_from_max_t {
        filled.time = n_rows as i64 - 1;
    }
    backtrack(
        config,
        scores,
        ground_truth,
        &filled.table,
        &filled.offsets,
        filled.time,
        filled.column,
    )
}

fn fill_table(
    config: &CtcSegmentationParameters,
    scores: &[Vec<f32>],
    ground_truth: &[Vec<i64>],
    n_rows: usize,
) -> Result<FilledTable, WindowError> {
    let n_cols = ground_truth.len();
    let n_frames = scores.len();
    let width = ground_truth.iter().map(Vec::len).max().unwrap_or(0);
    let max_prob = config.max_prob as f32;
    let mut table = vec![vec![max_prob; n_cols]; n_rows];
    table[0][0] = 0.0;
    let mut offsets = vec![0_i64; n_cols];
    let mut offset_sum = 0_i64;
    let mut last_arg_max = -1_i64;
    let mean_offset = (n_frames as f32 - n_rows as f32) / n_cols as f32;
    let lower_offset = mean_offset as i64;
    let higher_offset = lower_offset + 1;
    let mut cur_offset = vec![-1_i64; width];
    let flags = i32::from(config.blank_transition_cost_zero)
        + 2 * i32::from(config.preamble_transition_cost_zero);
    for column in 0..n_cols {
        if column > 0 {
            let step = next_offset(last_arg_max, n_rows, n_frames, higher_offset, offset_sum);
            for span in 0..width.saturating_sub(1) {
                cur_offset[span + 1] = cur_offset[span] + step;
            }
            if width > 0 {
                cur_offset[0] = step;
            }
            offset_sum += step;
        }
        offsets[column] = offset_sum;
        last_arg_max = -1;
        let mut last_max = 0.0_f32;
        let stay_zero = (flags & 1) != 0 && ground_truth[column].contains(&0);
        let preamble_zero = (flags & 2) != 0 && column == 0;
        let t_start = usize::from(column == 0);
        for time in t_start..n_rows {
            let (switch_prob, max_lpz) = switch_probability(SwitchQuery {
                scores,
                ground_truth,
                table: &table,
                cur_offset: &cur_offset,
                time,
                column,
                offset_sum,
                n_rows,
            })?;
            let stay_prob = stay_probability(StayQuery {
                scores,
                table: &table,
                blank: config.blank,
                time,
                column,
                offset_sum,
                max_lpz,
                stay_zero,
                preamble_zero,
            })?;
            table[time][column] = switch_prob.max(stay_prob);
            if last_arg_max == -1 || last_max < table[time][column] {
                last_max = table[time][column];
                last_arg_max = time as i64;
            }
        }
    }
    let column = n_cols - 1;
    let time = argmax_column(&table, column);
    Ok(FilledTable {
        table,
        offsets,
        time: time as i64,
        column: column as i64,
    })
}

fn next_offset(
    last_arg_max: i64,
    n_rows: usize,
    n_frames: usize,
    higher_offset: i64,
    offset_sum: i64,
) -> i64 {
    let centered = last_arg_max - n_rows as i64 / 2;
    let room = n_frames as i64 - n_rows as i64 - offset_sum;
    centered.max(0).min(higher_offset.min(room))
}

fn switch_probability(query: SwitchQuery<'_>) -> Result<(f32, f32), WindowError> {
    let mut switch_prob = FILL_PROB_MAX;
    let mut max_lpz = FILL_PROB_MAX;
    let frame = query.time as i64 + query.offset_sum;
    for (span, token) in query.ground_truth[query.column].iter().enumerate() {
        if *token == -1 {
            continue;
        }
        let lpz = frame_token(query.scores, frame, *token)?;
        let relative = query.cur_offset[span];
        let blocked = query.time as i64 >= query.n_rows as i64 - (relative - 1)
            || query.time as i64 - 1 + relative < 0
            || query.column == 0;
        let probability = if blocked {
            FILL_PROB_MAX
        } else if let Some(previous_c) = query.column.checked_sub(span + 1) {
            let previous_t = (query.time as i64 - 1 + relative) as usize;
            query
                .table
                .get(previous_t)
                .and_then(|row| row.get(previous_c))
                .map(|cell| *cell + lpz)
                .unwrap_or(FILL_PROB_MAX)
        } else {
            FILL_PROB_MAX
        };
        switch_prob = switch_prob.max(probability);
        max_lpz = max_lpz.max(lpz);
    }
    Ok((switch_prob, max_lpz))
}

fn stay_probability(query: StayQuery<'_>) -> Result<f32, WindowError> {
    if query.time == 0 {
        return Ok(FILL_PROB_MAX);
    }
    if query.preamble_zero {
        return Ok(0.0);
    }
    if query.stay_zero {
        return Ok(query.table[query.time - 1][query.column]);
    }
    let blank = frame_token(
        query.scores,
        query.time as i64 + query.offset_sum,
        query.blank,
    )?;
    Ok(query.table[query.time - 1][query.column] + blank.max(query.max_lpz))
}

fn frame_token(scores: &[Vec<f32>], frame: i64, token: i64) -> Result<f32, WindowError> {
    if frame < 0 || token < 0 {
        return Err(WindowError::Invalid);
    }
    scores
        .get(frame as usize)
        .and_then(|row| row.get(token as usize))
        .copied()
        .ok_or(WindowError::Invalid)
}

fn argmax_column(table: &[Vec<f32>], column: usize) -> usize {
    let mut best_index = 0;
    let mut best_value = f32::NEG_INFINITY;
    for (index, row) in table.iter().enumerate() {
        if index == 0 || row[column] > best_value {
            best_value = row[column];
            best_index = index;
        }
    }
    best_index
}

fn backtrack(
    config: &CtcSegmentationParameters,
    scores: &[Vec<f32>],
    ground_truth: &[Vec<i64>],
    table: &[Vec<f32>],
    offsets: &[i64],
    mut time: i64,
    mut column: i64,
) -> Result<CtcSegmentation, WindowError> {
    let mut path = PathBuffers {
        timings: vec![0.0; ground_truth.len()],
        char_probs: vec![0.0; scores.len()],
        states: vec![String::new(); scores.len()],
    };
    let mut steps = 0;
    let limit = scores
        .len()
        .saturating_mul(ground_truth.len())
        .saturating_add(5);
    while time != 0 || column != 0 {
        steps += 1;
        if steps > limit {
            return Err(WindowError::Backtrack);
        }
        let choice = transition_choice(config, scores, ground_truth, table, offsets, time, column)?;
        if choice.switch {
            apply_switch(
                config,
                ground_truth,
                &mut path,
                offsets,
                time,
                column,
                &choice,
            )?;
            column -= 1 + choice.min_span;
            time -= 1 - choice.offset;
        } else {
            let frame = frame_index(offsets, column, time, path.char_probs.len())?;
            path.char_probs[frame] = choice.stay_prob;
            path.states[frame].clone_from(&config.self_transition);
            time -= 1;
        }
    }
    Ok(CtcSegmentation {
        timings: path.timings,
        char_probs: path.char_probs,
        states: path.states,
    })
}

struct PathBuffers {
    timings: Vec<f64>,
    char_probs: Vec<f64>,
    states: Vec<String>,
}

struct TransitionChoice {
    switch: bool,
    min_span: i64,
    offset: i64,
    max_lpz: f64,
    stay_prob: f64,
}

fn transition_choice(
    config: &CtcSegmentationParameters,
    scores: &[Vec<f32>],
    ground_truth: &[Vec<i64>],
    table: &[Vec<f32>],
    offsets: &[i64],
    time: i64,
    column: i64,
) -> Result<TransitionChoice, WindowError> {
    if time <= 0
        || column < 0
        || time as usize >= table.len()
        || column as usize >= ground_truth.len()
    {
        return Err(WindowError::Backtrack);
    }
    let mut min_span = None;
    let mut min_delta = f64::INFINITY;
    let mut max_lpz = config.max_prob;
    let mut offset = 0_i64;
    for (span, token) in ground_truth[column as usize].iter().enumerate() {
        if *token == -1 {
            continue;
        }
        offset = column_offset(offsets, column, span)?;
        let switch_prob = if column > 0 {
            f64::from(frame_token(
                scores,
                time + offsets[column as usize],
                *token,
            )?)
        } else {
            config.max_prob
        };
        let estimated = estimated_switch(table, time, column, offset, span)?;
        let delta = (switch_prob - estimated).abs();
        if delta < min_delta {
            min_delta = delta;
            min_span = Some(span as i64);
        }
        max_lpz = max_lpz.max(switch_prob);
    }
    let blank = f64::from(frame_token(
        scores,
        time + offsets[column as usize],
        config.blank,
    )?);
    let stay_prob = blank.max(max_lpz);
    let estimated_stay = estimated_stay(table, time, column)?;
    let switch = (stay_prob - estimated_stay).abs() > min_delta;
    if switch && min_span.is_none() {
        return Err(WindowError::Backtrack);
    }
    Ok(TransitionChoice {
        switch,
        min_span: min_span.unwrap_or(0),
        offset,
        max_lpz,
        stay_prob,
    })
}

fn column_offset(offsets: &[i64], column: i64, span: usize) -> Result<i64, WindowError> {
    let current = *offsets.get(column as usize).ok_or(WindowError::Backtrack)?;
    if column as usize > span {
        let previous = column as usize - 1 - span;
        Ok(current - *offsets.get(previous).ok_or(WindowError::Backtrack)?)
    } else {
        Ok(current)
    }
}

fn estimated_switch(
    table: &[Vec<f32>],
    time: i64,
    column: i64,
    offset: i64,
    span: usize,
) -> Result<f64, WindowError> {
    let previous_t = time - 1 + offset;
    let previous_c = column - 1 - span as i64;
    let left = cell(table, previous_t, previous_c)?;
    let here = cell(table, time, column)?;
    Ok(f64::from(here - left))
}

fn estimated_stay(table: &[Vec<f32>], time: i64, column: i64) -> Result<f64, WindowError> {
    Ok(f64::from(
        cell(table, time, column)? - cell(table, time - 1, column)?,
    ))
}

fn cell(table: &[Vec<f32>], time: i64, column: i64) -> Result<f32, WindowError> {
    if time < 0 || column < 0 {
        return Err(WindowError::Backtrack);
    }
    table
        .get(time as usize)
        .and_then(|row| row.get(column as usize))
        .copied()
        .ok_or(WindowError::Backtrack)
}

fn apply_switch(
    config: &CtcSegmentationParameters,
    ground_truth: &[Vec<i64>],
    path: &mut PathBuffers,
    offsets: &[i64],
    time: i64,
    column: i64,
    choice: &TransitionChoice,
) -> Result<(), WindowError> {
    if column <= 0 {
        return Ok(());
    }
    let frame = frame_index(offsets, column, time, path.char_probs.len())?;
    for span in 0..=choice.min_span {
        let index = column - span;
        if index < 0 || index as usize >= path.timings.len() {
            return Err(WindowError::Backtrack);
        }
        path.timings[index as usize] =
            (offsets[column as usize] + time) as f64 * config.index_duration;
    }
    path.char_probs[frame] = choice.max_lpz;
    let token = *ground_truth
        .get(column as usize)
        .and_then(|row| row.get(choice.min_span as usize))
        .ok_or(WindowError::Backtrack)?;
    let label = config
        .char_list
        .get(token as usize)
        .ok_or(WindowError::Backtrack)?;
    path.states[frame].clone_from(label);
    Ok(())
}

fn frame_index(
    offsets: &[i64],
    column: i64,
    time: i64,
    frames: usize,
) -> Result<usize, WindowError> {
    let frame = *offsets.get(column as usize).ok_or(WindowError::Backtrack)? + time;
    if frame < 0 || frame as usize >= frames {
        return Err(WindowError::Backtrack);
    }
    Ok(frame as usize)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::timing_policy::{LEGACY_UNIT_POLICY_ID, TIMING_UNIT_POLICY_ID};

    fn fixture_text() -> String {
        "  go go café \u{1F469}\u{1F3FD}\u{200D}\u{1F4BB}  next end  ".to_string()
    }

    fn english_row() -> ReadingRow {
        let text = fixture_text();
        let words = raw_units(&text, LEGACY_UNIT_POLICY_ID)
            .unwrap()
            .into_iter()
            .enumerate()
            .map(|(index, unit)| {
                let start = (index as i64 + 1) * RATE;
                TimedWord {
                    text: unit.text,
                    start_char: unit.start_char,
                    end_char: unit.end_char,
                    start_sample: Some(start),
                    end_sample: Some(start + 3_200),
                }
            })
            .collect();
        text_sha256_row(text, words)
    }

    fn text_sha256_row(text: String, words: Vec<TimedWord>) -> ReadingRow {
        let anchor = AudioAnchor {
            start_sample: 0,
            end_sample: 160_000,
        };
        ReadingRow {
            reading_word_evidence: Some(ReadingWordEvidence {
                text_sha256: sha256_hex(text.as_bytes()),
                machine_revision: 2,
                audio_revision: 3,
                retained_audio_revision: None,
                audio_anchor: anchor.clone(),
                model_sha256: MODEL.to_string(),
                calibration_id: Some(CALIBRATION.to_string()),
                frame_calibration_id: None,
                timing_unit_policy_id: None,
                provider_id: None,
                provider_identity_sha256: None,
                normalization_policy_id: None,
                words,
            }),
            speaker_activity: Some(SpeakerActivity {
                audio_revision: 3,
                observed_end_sample: None,
                regions: vec![
                    ActivityRegion {
                        start_sample: 0,
                        end_sample: 80_000,
                        speakers: vec!["speaker_0".to_string()],
                    },
                    ActivityRegion {
                        start_sample: 80_000,
                        end_sample: 160_000,
                        speakers: vec!["speaker_1".to_string()],
                    },
                ],
            }),
            text,
            start_sample: 0,
            end_sample: 160_000,
            machine_revision: 2,
            audio_revision: 3,
            protected_fields: Vec::new(),
            text_audio_anchor: Some(anchor),
            installed_provider_identity: None,
        }
    }

    #[test]
    fn character_classes_follow_the_python_ranges() {
        assert!(cjk_character('日') && cjk_character('あ') && cjk_character('ア'));
        assert!(cjk_character('ｦ') && cjk_character('\u{2000B}'));
        assert!(!cjk_character('ー') && !cjk_character('한') && !cjk_character('A'));
        assert!(kana_character('あ') && kana_character('ア') && kana_character('ｦ'));
        assert!(!kana_character('ー') && !kana_character('日') && !kana_character('\u{2000B}'));
    }

    #[test]
    fn raw_units_keep_legacy_tokens_and_split_mixed_script_scalars() {
        let legacy = raw_units("「日本」、a ターネ。", LEGACY_UNIT_POLICY_ID).unwrap();
        assert_eq!(
            legacy
                .iter()
                .map(|unit| unit.text.as_str())
                .collect::<Vec<_>>(),
            ["「日本」、a", "ターネ。"]
        );
        let mixed = raw_units("「日本」、a ターネ。", TIMING_UNIT_POLICY_ID).unwrap();
        assert_eq!(
            mixed
                .iter()
                .map(|unit| (unit.text.as_str(), unit.unit_kind))
                .collect::<Vec<_>>(),
            [
                ("「日", UnitKind::CtcCharacter),
                ("本」、", UnitKind::CtcCharacter),
                ("a", UnitKind::WhitespaceRun),
                ("ター", UnitKind::CtcCharacter),
                ("ネ。", UnitKind::CtcCharacter),
            ]
        );
        let switched = raw_units("日ab本", TIMING_UNIT_POLICY_ID).unwrap();
        assert_eq!(
            switched
                .iter()
                .map(|unit| unit.text.as_str())
                .collect::<Vec<_>>(),
            ["日", "ab", "本"]
        );
        assert!(raw_units("   ", LEGACY_UNIT_POLICY_ID).unwrap().is_empty());
        assert_eq!(
            raw_units("...", TIMING_UNIT_POLICY_ID).unwrap()[0].unit_kind,
            UnitKind::WhitespaceRun
        );
        assert_eq!(
            raw_units("ー", TIMING_UNIT_POLICY_ID).unwrap()[0].text,
            "ー"
        );
        let error = raw_units("hello", "whitespace_runs_v2").unwrap_err();
        assert_eq!(error.to_string(), "Unknown transcript timing unit policy.");
    }

    #[test]
    fn raw_units_record_utf16_offsets_for_the_emoji_token() {
        let units = raw_units(&fixture_text(), LEGACY_UNIT_POLICY_ID).unwrap();
        let emoji = units.iter().find(|unit| unit.start_char == 13).unwrap();
        assert_eq!(
            (emoji.end_char, emoji.start_utf16, emoji.end_utf16),
            (17, 13, 20)
        );
        assert_eq!(emoji.status, "unresolved");
        assert_eq!((emoji.start_sample, emoji.end_sample), (None, None));
    }

    #[test]
    fn project_turns_assigns_sequential_owners_without_rewriting_text() {
        let row = english_row();
        let turns = project_turns(&row);
        assert_eq!(
            turns
                .iter()
                .map(|turn| turn.text.clone())
                .collect::<String>(),
            row.text
        );
        assert_eq!(
            turns
                .iter()
                .map(|turn| turn.attribution.as_str())
                .collect::<Vec<_>>(),
            ["single", "unknown", "single"]
        );
        assert_eq!(
            turns
                .iter()
                .map(|turn| turn.speaker.as_str())
                .collect::<Vec<_>>(),
            ["speaker_0", "unassigned", "speaker_1"]
        );
        assert_eq!((turns[0].start, turns[0].end), (Some(1.0), Some(4.2)));
        assert_eq!((turns[1].start, turns[1].end), (None, None));
        assert_eq!((turns[2].start, turns[2].end), (Some(6.0), Some(6.2)));
        assert_eq!((turns[0].start_offset, turns[0].end_offset), (0, 19));
        assert_eq!((turns[1].start_offset, turns[1].end_offset), (19, 24));
        assert_eq!((turns[2].start_offset, turns[2].end_offset), (24, 29));
    }

    #[test]
    fn project_turns_names_overlap_and_keeps_gaps_unknown() {
        let mut overlap = english_row();
        let activity = overlap.speaker_activity.as_mut().unwrap();
        activity.regions[0].speakers = vec!["speaker_0".to_string(), "speaker_1".to_string()];
        activity.regions[1].speakers = vec!["speaker_1".to_string(), "speaker_2".to_string()];
        let turns = project_turns(&overlap);
        assert_eq!(turns[0].speaker, "overlap_speaker_0_speaker_1");
        assert_eq!(turns[2].speaker, "overlap_speaker_1_speaker_2");
        assert_eq!(turns[1].attribution, "unknown");

        let mut gap = english_row();
        gap.speaker_activity.as_mut().unwrap().regions[1]
            .speakers
            .clear();
        let word = &mut gap.reading_word_evidence.as_mut().unwrap().words[0];
        word.start_sample = None;
        word.end_sample = None;
        let turns = project_turns(&gap);
        assert_eq!(turns[0].attribution, "unknown");
        assert_eq!(turns[1].attribution, "single");
        assert_eq!(turns[1].start, Some(2.0));
        assert_eq!(turns[2].attribution, "unknown");
        assert!(turns[2].start.is_none());
    }

    #[test]
    fn project_turns_merges_feed_cells_and_rejects_stale_rows() {
        let mut row = english_row();
        row.speaker_activity.as_mut().unwrap().regions = (0..10)
            .map(|index| ActivityRegion {
                start_sample: index * RATE,
                end_sample: (index + 1) * RATE,
                speakers: vec!["speaker_0".to_string()],
            })
            .collect();
        let turns = project_turns(&row);
        assert_eq!(turns.len(), 1);
        assert_eq!(turns[0].attribution, "single");
        assert_eq!(turns[0].text, row.text);

        row.machine_revision = 4;
        assert!(project_turns(&row).is_empty());
        let mut protected = english_row();
        protected.protected_fields.push("speaker".to_string());
        assert!(project_turns(&protected).is_empty());
        let mut revised = english_row();
        revised.speaker_activity.as_mut().unwrap().audio_revision = 4;
        assert!(project_turns(&revised).is_empty());
    }

    #[test]
    fn validated_turns_accepts_the_python_provenance_and_rejects_edits() {
        let row = english_row();
        let turns = project_turns(&row);
        let segment = segment_from(&row, turns);
        assert_eq!(
            sha256_hex(segment.text.as_bytes()),
            "950b1982d89316ad13d443612ed144f6537d73821501ff80d8ead53995bbd90b"
        );
        assert_eq!(
            sha256_hex(activity_json(segment.speaker_activity.as_ref()).as_bytes()),
            "49c290b9f3068521700f423689bb932a23492155cc3a7bf74fde79f730be2938"
        );
        assert_eq!(
            sha256_hex(turns_json(&segment.reading_turns).as_bytes()),
            "b1a8242664b37c5d5762de8ddb95a3fee76eea97f7cdee888da473104dd06b02"
        );
        assert_eq!(validated_turns(&segment), segment.reading_turns);

        let mut forged = segment.clone();
        forged.reading_turns[0].text = "replacement".to_string();
        assert!(validated_turns(&forged).is_empty());
        let mut renamed = segment.clone();
        renamed.reading_turns[0].speaker = "speaker_1".to_string();
        assert!(validated_turns(&renamed).is_empty());
        let mut revised = segment;
        revised.machine_revision = 3;
        assert!(validated_turns(&revised).is_empty());
    }

    #[test]
    fn mixed_script_turns_accept_a_synthetic_provider_identity() {
        let text = "日本".to_string();
        let units = raw_units(&text, TIMING_UNIT_POLICY_ID).unwrap();
        let words = units
            .iter()
            .enumerate()
            .map(|(index, unit)| TimedWord {
                text: unit.text.clone(),
                start_char: unit.start_char,
                end_char: unit.end_char,
                start_sample: Some(5_000 + index as i64 * 3_000),
                end_sample: Some(6_000 + index as i64 * 3_000),
            })
            .collect();
        let anchor = AudioAnchor {
            start_sample: 0,
            end_sample: 16_000,
        };
        let identity = "synthetic-provider-identity";
        let mut row = ReadingRow {
            text: text.clone(),
            start_sample: 0,
            end_sample: 16_000,
            machine_revision: 1,
            audio_revision: 1,
            protected_fields: Vec::new(),
            text_audio_anchor: Some(anchor.clone()),
            reading_word_evidence: Some(ReadingWordEvidence {
                text_sha256: sha256_hex(text.as_bytes()),
                machine_revision: 1,
                audio_revision: 1,
                retained_audio_revision: None,
                audio_anchor: anchor,
                model_sha256: MODEL.to_string(),
                calibration_id: Some(FUNCTIONAL_POLICY_ID.to_string()),
                frame_calibration_id: None,
                timing_unit_policy_id: Some(TIMING_UNIT_POLICY_ID.to_string()),
                provider_id: Some(PROVIDER_ID.to_string()),
                provider_identity_sha256: Some(identity.to_string()),
                normalization_policy_id: Some(NORMALIZATION_POLICY_ID.to_string()),
                words,
            }),
            speaker_activity: Some(SpeakerActivity {
                audio_revision: 1,
                observed_end_sample: Some(16_000),
                regions: vec![ActivityRegion {
                    start_sample: 0,
                    end_sample: 16_000,
                    speakers: vec!["speaker_0".to_string()],
                }],
            }),
            installed_provider_identity: None,
        };
        assert!(project_turns(&row).is_empty());
        row.installed_provider_identity = Some(identity.to_string());
        let turns = project_turns(&row);
        assert_eq!(turns.len(), 1);
        assert_eq!(turns[0].text, "日本");
        assert_eq!(turns[0].attribution, "single");
        assert_eq!(turns[0].speaker, "speaker_0");
    }

    fn segment_from(row: &ReadingRow, turns: Vec<ReadingTurn>) -> ReadingSegment {
        let activity = row.speaker_activity.clone();
        let provenance = TurnProvenance {
            utterance_id: "one".to_string(),
            machine_revision: row.machine_revision,
            canonical_machine_revision: row.machine_revision,
            audio_revision: row.audio_revision,
            language_epoch: 0,
            capture_source: None,
            text_sha256: "950b1982d89316ad13d443612ed144f6537d73821501ff80d8ead53995bbd90b"
                .to_string(),
            audio_anchor: row.text_audio_anchor.clone().unwrap(),
            activity_sha256: "49c290b9f3068521700f423689bb932a23492155cc3a7bf74fde79f730be2938"
                .to_string(),
            turns_sha256: "b1a8242664b37c5d5762de8ddb95a3fee76eea97f7cdee888da473104dd06b02"
                .to_string(),
            model_sha256: MODEL.to_string(),
            calibration_id: CALIBRATION.to_string(),
            frame_calibration_id: None,
            timing_unit_policy_id: None,
            provider_id: None,
            provider_identity_sha256: None,
            normalization_policy_id: None,
        };
        ReadingSegment {
            text: row.text.clone(),
            protected_fields: Vec::new(),
            canonical_utterance_id: "one".to_string(),
            machine_revision: row.machine_revision,
            canonical_machine_revision: row.machine_revision,
            audio_revision: row.audio_revision,
            language_epoch: 0,
            capture_source: None,
            text_audio_anchor: row.text_audio_anchor.clone().unwrap(),
            speaker_activity: activity,
            start: 0.0,
            end: 10.0,
            reading_turns: turns,
            reading_turn_provenance: Some(provenance),
            installed_provider_identity: None,
        }
    }

    #[test]
    fn ctc_segmentation_follows_the_library_path_on_a_synthetic_matrix() {
        let mut scores = vec![vec![-20.0_f32; 3]; 6];
        scores[0][0] = -0.5;
        scores[2][1] = -0.5;
        scores[4][2] = -0.5;
        scores[5][0] = -0.5;
        let prepared = prepare_token_list(0, &[vec![1, 2]]);
        assert_eq!(
            prepared.ground_truth,
            vec![vec![-1], vec![0], vec![1], vec![2], vec![0]]
        );
        assert_eq!(prepared.utterance_begin_indices, vec![1, 4]);
        let config = CtcSegmentationParameters::for_vocabulary(vec![
            "<s>".to_string(),
            "a".to_string(),
            "b".to_string(),
        ]);
        let path = ctc_segmentation(&config, &scores, &prepared.ground_truth).unwrap();
        assert_eq!(path.timings, vec![0.0, 2.0, 3.0, 4.0, 5.0]);
        assert_eq!(
            path.states,
            ["", "<alignment-stay>", "<s>", "a", "b", "<s>"]
        );
        assert_eq!(path.char_probs, vec![0.0, -20.0, -20.0, -20.0, -0.5, -0.5]);
        let short = ctc_segmentation(&config, &scores[..3], &prepared.ground_truth).unwrap_err();
        assert_eq!(short.to_string(), "Audio is shorter than text!");
    }
}
