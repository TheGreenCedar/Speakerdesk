//! Import pipeline. Owned by the import agent.
//!
//! Diarization, speech admission, and transcription go through [`Models`].
//! Word-level CTC timing is not attached: [`Models::ctc_scores`] does not carry
//! a vocabulary, and [`timing::ctc_segmentation`] needs a token score matrix.

use std::collections::BTreeMap;
use std::marker::PhantomData;

use serde_json::{Map, Value, json};

use crate::audio;
use crate::document::{Document, Segment};
use crate::error::ChildError;
use crate::export;
use crate::ipc::Turn;
use crate::language::{
    self, AsrOutput, LanguageChoice, LanguageDetection, LanguageDetector, LanguageError, Passage,
    Recognizer, RouteError, SpeechTranscriber, TranscribeOptions, TranscriptionReview,
};
use crate::languages;
use crate::limits;
use crate::messages;
use crate::models::Models;
use crate::review;
use crate::speech::{self, Admission, BlockedSpeech, SpeechLedger};
use crate::timing;
use crate::transcript::{self, TranscriptError};

/// Brief same-speaker pause that still belongs to one decode envelope.
pub const TURN_GAP_SECONDS: f64 = 0.65;
/// Default envelope split. Matches `pipeline.decode_regions`.
pub const DEFAULT_MAX_SECONDS: f64 = limits::PASSAGE_RETRY_MAX_SECONDS;
/// Import decode window from `pipeline.infer`.
pub const IMPORT_MAX_SECONDS: f64 = 18.0;
/// Manual-language padding into adjacent silence, in seconds.
pub const CROP_PADDING_SECONDS: f64 = 0.22;

const BOUNDARY_EPSILON: f64 = 1e-6;
const ENVELOPE_SECONDS: f64 = 100_000.0;
const MAX_CHUNKS: usize = 10_000;

const WARN_TIMES: &str =
    "Times are speech crop boundaries, not word timestamps. Review subtitle timing.";
const WARN_AUTO: &str = "Uncertain language uses recent established context or a marked supported-language guess when available. These attempts can be wrong; audio without a supported language remains available for review.";
const WARN_MANUAL: &str = "The selected language applies to every audio crop, including speech without a detected speaker.";
const WARN_OVERLAP: &str =
    "Overlapping speech is mixed audio; text cannot be reliably assigned to an individual speaker.";
const WARN_TIMING_AUTO: &str =
    "Retained audio with ≤3-second language probes and ≤6-second ASR crops; no word alignment";
const WARN_TIMING_MANUAL: &str = "Retained audio split into ≤25-second crops; no word alignment";

/// Failure that already carries a product string.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
#[error("{message}")]
pub struct ImportError {
    pub message: String,
}

impl ImportError {
    fn new(message: impl Into<String>) -> Self {
        Self {
            message: message.into(),
        }
    }
}

impl From<TranscriptError> for ImportError {
    fn from(error: TranscriptError) -> Self {
        Self::new(error.message)
    }
}

/// One slice of the activity ledger inside a bridged envelope.
#[derive(Debug, Clone, PartialEq)]
pub struct ActivitySlice {
    pub start: f64,
    pub end: f64,
    pub speakers: Vec<String>,
}

/// One continuous decode region, before or after [`crop_context`].
#[derive(Debug, Clone, PartialEq)]
pub struct DecodeRegion {
    pub start: f64,
    pub end: f64,
    pub speakers: Vec<String>,
    pub activity_regions: Option<Vec<ActivitySlice>>,
    pub audio_start: f64,
    pub audio_end: f64,
}

/// A passage retry is a candidate. Existing words stay until an editor applies it.
#[derive(Debug, Clone, PartialEq)]
pub struct PassageCandidate {
    pub start: f64,
    pub end: f64,
    pub text: String,
    pub language: String,
    pub review: bool,
    pub transcription_review: Option<Value>,
    pub retry_parts: Vec<Value>,
}

/// `pipeline.decode_regions`.
///
/// `max_seconds: None` keeps each bridged envelope intact. The Python default is
/// [`DEFAULT_MAX_SECONDS`].
pub fn decode_regions(
    turns: &[Turn],
    max_seconds: Option<f64>,
    coverage: Option<(f64, f64)>,
) -> Vec<DecodeRegion> {
    let regions = speech_crops(turns, ENVELOPE_SECONDS, coverage);
    let mut envelopes = Vec::new();
    let mut index = 0;
    while index < regions.len() {
        let mut region = region_from(&regions[index]);
        let owner = region.speakers.clone();
        index += 1;
        let mut ledger = vec![regions[index - 1].clone()];
        while index + 1 < regions.len() {
            let middle = &regions[index];
            let following = &regions[index + 1];
            let compatible = middle.speakers.is_empty()
                || (middle.speakers.len() > 1
                    && owner
                        .iter()
                        .all(|speaker| middle.speakers.contains(speaker)));
            let contiguous = (middle.start - region.end).abs() <= BOUNDARY_EPSILON
                && (following.start - middle.end).abs() <= BOUNDARY_EPSILON;
            if owner.len() != 1
                || !compatible
                || following.speakers != owner
                || middle.end - middle.start > TURN_GAP_SECONDS
                || !contiguous
            {
                break;
            }
            ledger.push(middle.clone());
            ledger.push(following.clone());
            region.end = following.end;
            region.audio_end = following.end;
            region.speakers = union_speakers(&region.speakers, &middle.speakers);
            index += 2;
        }
        if ledger.len() > 1 {
            region.activity_regions = Some(ledger);
        }
        envelopes.push(region);
    }
    let Some(max_seconds) = max_seconds else {
        return envelopes;
    };
    let mut chunks = Vec::new();
    for region in envelopes {
        let mut start = region.start;
        while start < region.end - BOUNDARY_EPSILON {
            let end = (start + max_seconds).min(region.end);
            if end <= start + BOUNDARY_EPSILON {
                break;
            }
            let mut chunk = region.clone();
            chunk.start = start;
            chunk.end = end;
            chunk.audio_start = start;
            chunk.audio_end = end;
            if let Some(activity) = &region.activity_regions {
                chunk.activity_regions = Some(activity_slice(activity, start, end));
            }
            chunks.push(chunk);
            start = end;
        }
    }
    chunks
}

/// `pipeline.crop_context`. Padding stays inside the gap to the neighboring region.
pub fn crop_context(chunks: &mut [DecodeRegion], duration: f64, padding: f64) {
    for index in 0..chunks.len() {
        let previous = if index == 0 {
            0.0
        } else {
            chunks[index - 1].end
        };
        let following = if index + 1 < chunks.len() {
            chunks[index + 1].start
        } else {
            duration
        };
        let start = chunks[index].start;
        let end = chunks[index].end;
        chunks[index].audio_start = 0.0_f64.max(start - padding).max((previous + start) / 2.0);
        chunks[index].audio_end = duration.min(end + padding).min((following + end) / 2.0);
    }
}

/// `pipeline.activity_slice`.
pub fn activity_slice(regions: &[ActivitySlice], start: f64, end: f64) -> Vec<ActivitySlice> {
    regions
        .iter()
        .filter(|part| part.start < end && part.end > start)
        .map(|part| ActivitySlice {
            start: start.max(part.start),
            end: end.min(part.end),
            speakers: part.speakers.clone(),
        })
        .collect()
}

/// Diarize, admit speech, transcribe, and validate one recording.
///
/// Manual language calls [`Models::transcribe`] for admitted speech only.
/// `auto` also calls [`Models::language_probe`] through [`SpeechTranscriber`].
pub fn import_recording<M: Models + ?Sized + 'static>(
    models: &M,
    pcm: &[f32],
    language: &str,
) -> Result<Document, ImportError> {
    let choice = LanguageChoice::parse(language).map_err(language_err)?;
    let duration = require_pcm(pcm)?;
    let limit = limits::SPEAKER_TRACK_LIMIT;
    let turns = normalize_turns(
        models.diarize_window(pcm).map_err(child_err)?,
        duration,
        limit,
    )?;
    let mut chunks = decode_regions(&turns, Some(IMPORT_MAX_SECONDS), Some((0.0, duration)));
    let padding = if matches!(choice, LanguageChoice::Auto) {
        0.0
    } else {
        CROP_PADDING_SECONDS
    };
    crop_context(&mut chunks, duration, padding);
    if chunks.len() > MAX_CHUNKS {
        return Err(ImportError::new(messages::TOO_MANY_TRANSITIONS));
    }
    let ledger = admit(models, pcm)?;
    let work = plan_work(&ledger, pcm, &chunks)?;
    let mut transcriber = if work.iter().any(|item| item.blocked.is_none()) {
        Some(open_transcriber(models, language, ledger)?)
    } else {
        None
    };
    let mut speakers = speaker_map(&turns);
    let mut segments = Vec::new();
    for item in work {
        let chunk = &chunks[item.chunk];
        let decoded = decode_window(transcriber.as_mut(), pcm, language, &chunk.speakers, &item)?;
        for row in decoded {
            let Some((start, end)) = clip_span(row.start, row.end, chunk.start, chunk.end) else {
                continue;
            };
            let Some((start, end)) = clip_span(start, end, 0.0, duration) else {
                continue;
            };
            let speaker = chunk_speaker(chunk);
            ensure_speaker(&mut speakers, &speaker);
            let review = row.review
                || speaker == "unassigned"
                || chunk.speakers.len() != 1
                || end - start < 0.5;
            segments.push(segment_from_row(
                format!("seg-{}", segments.len()),
                start,
                end,
                speaker,
                review,
                chunk,
                &row,
            ));
        }
    }
    let mut document = Document {
        schema_version: 1,
        speakers,
        segments,
        provenance: Some(provenance(language, limit)),
        warnings: warnings(language, limit),
        diarization: turns
            .iter()
            .map(|turn| json!({"start": turn.start, "end": turn.end, "speaker": turn.speaker}))
            .collect(),
        edited_at: None,
        rest: Map::new(),
    };
    review::retain_unassigned_audio(&mut document, duration);
    let value = serde_json::to_value(&document)
        .map_err(|_| ImportError::new(messages::PROCESSING_FAILED))?;
    let document = transcript::validate(&value, duration)?;
    export::export(&document, "txt").map_err(|_| ImportError::new(messages::EXPORT_TOO_LARGE))?;
    Ok(document)
}

/// Transcribe one explicit-language passage. The result is not written onto a document.
pub fn retry_passage<M: Models + ?Sized + 'static>(
    models: &M,
    pcm: &[f32],
    start: f64,
    end: f64,
    language: &str,
) -> Result<PassageCandidate, ImportError> {
    if language == languages::AUTOMATIC || !languages::is_supported(language) {
        return Err(ImportError::new(messages::RETRY_LANGUAGE));
    }
    let duration = require_pcm(pcm)?;
    if !start.is_finite()
        || !end.is_finite()
        || start < 0.0
        || end <= start
        || end - start > limits::PASSAGE_RETRY_MAX_SECONDS
        || end > duration + limits::TIME_EPSILON_SECONDS
    {
        return Err(ImportError::new(messages::RETRY_TOO_LONG));
    }
    let origin = seconds_to_sample(start, pcm.len() as i64);
    let stop = seconds_to_sample(end, pcm.len() as i64);
    if stop <= origin {
        return Err(ImportError::new(messages::RETRY_TOO_LONG));
    }
    let slice = &pcm[origin as usize..stop as usize];
    let ledger = admit(models, slice)?;
    let windows = examine_range(&ledger, slice, 0, slice.len() as i64)?;
    let mut transcriber = if windows.iter().any(|item| item.blocked.is_none()) {
        Some(open_transcriber(models, language, ledger)?)
    } else {
        None
    };
    let speakers = vec!["manual_retry".to_string()];
    let mut parts = Vec::new();
    for item in &windows {
        parts.extend(decode_window(
            transcriber.as_mut(),
            slice,
            language,
            &speakers,
            item,
        )?);
    }
    Ok(candidate_from_parts(start, end, language, origin, &parts))
}

fn region_from(slice: &ActivitySlice) -> DecodeRegion {
    DecodeRegion {
        start: slice.start,
        end: slice.end,
        speakers: slice.speakers.clone(),
        activity_regions: None,
        audio_start: slice.start,
        audio_end: slice.end,
    }
}

fn speech_crops(
    turns: &[Turn],
    max_seconds: f64,
    coverage: Option<(f64, f64)>,
) -> Vec<ActivitySlice> {
    let mut bounds = Vec::new();
    for turn in turns {
        bounds.push(turn.start);
        bounds.push(turn.end);
    }
    if let Some((first, last)) = coverage {
        bounds.retain(|bound| *bound > first && *bound < last);
        bounds.push(first);
        bounds.push(last);
    }
    bounds.sort_by(f64::total_cmp);
    dedup_bounds(&mut bounds);
    let mut regions: Vec<ActivitySlice> = Vec::new();
    for pair in bounds.windows(2) {
        let (start, end) = (pair[0], pair[1]);
        if end <= start {
            continue;
        }
        let speakers = active_speakers(turns, start, end);
        if speakers.is_empty() && coverage.is_none() {
            continue;
        }
        if let Some(last) = regions.last_mut()
            && last.speakers == speakers
            && (last.end - start).abs() < BOUNDARY_EPSILON
        {
            last.end = end;
        } else {
            regions.push(ActivitySlice {
                start,
                end,
                speakers,
            });
        }
    }
    let mut chunks = Vec::new();
    for region in regions {
        let mut start = region.start;
        while start < region.end - BOUNDARY_EPSILON {
            let end = (start + max_seconds).min(region.end);
            if end <= start + BOUNDARY_EPSILON {
                break;
            }
            chunks.push(ActivitySlice {
                start,
                end,
                speakers: region.speakers.clone(),
            });
            start = end;
        }
    }
    chunks
}

#[allow(clippy::float_cmp)]
fn dedup_bounds(bounds: &mut Vec<f64>) {
    bounds.dedup_by(|left, right| *left == *right);
}

fn active_speakers(turns: &[Turn], start: f64, end: f64) -> Vec<String> {
    let mut speakers: Vec<String> = turns
        .iter()
        .filter(|turn| turn.start < end && turn.end > start)
        .map(|turn| turn.speaker.clone())
        .collect();
    speakers.sort();
    speakers.dedup();
    speakers
}

fn union_speakers(left: &[String], right: &[String]) -> Vec<String> {
    let mut speakers = left.to_vec();
    for speaker in right {
        if !speakers.contains(speaker) {
            speakers.push(speaker.clone());
        }
    }
    speakers.sort();
    speakers
}

fn require_pcm(pcm: &[f32]) -> Result<f64, ImportError> {
    let duration = pcm.len() as f64 / f64::from(limits::SAMPLE_RATE);
    if pcm.is_empty() || duration > audio::MAX_DURATION_SECONDS {
        return Err(ImportError::new(messages::RECORDING_LENGTH));
    }
    if pcm.iter().any(|sample| !sample.is_finite()) {
        return Err(ImportError::new(messages::CAPTURE_FINITE_MONO));
    }
    Ok(duration)
}

fn normalize_turns(
    turns: Vec<Turn>,
    duration: f64,
    limit: usize,
) -> Result<Vec<Turn>, ImportError> {
    let mut normalized = Vec::new();
    for turn in turns {
        let Some(index) = speaker_index(&turn.speaker) else {
            return Err(ImportError::new("Invalid NVIDIA speaker or time output."));
        };
        if index >= limit || !turn.start.is_finite() || !turn.end.is_finite() {
            return Err(ImportError::new("Invalid NVIDIA speaker or time output."));
        }
        let start = turn.start.max(0.0);
        let end = turn.end.min(duration);
        if end > start {
            normalized.push(Turn {
                start,
                end,
                speaker: turn.speaker,
            });
        }
    }
    normalized.sort_by(|left, right| {
        left.start
            .total_cmp(&right.start)
            .then(left.end.total_cmp(&right.end))
    });
    Ok(normalized)
}

fn speaker_index(speaker: &str) -> Option<usize> {
    let rest = speaker.strip_prefix("speaker_")?;
    if rest.is_empty() || !rest.bytes().all(|byte| byte.is_ascii_digit()) {
        return None;
    }
    rest.parse().ok()
}

fn display_speaker(speaker: &str) -> String {
    match speaker {
        "overlap" => "Overlapping speakers".to_string(),
        "unassigned" => "Unknown speaker".to_string(),
        _ => speaker_index(speaker)
            .map(|index| format!("Speaker {}", index + 1))
            .unwrap_or_else(|| speaker.to_string()),
    }
}

fn speaker_map(turns: &[Turn]) -> BTreeMap<String, String> {
    let mut speakers = BTreeMap::new();
    for turn in turns {
        ensure_speaker(&mut speakers, &turn.speaker);
    }
    speakers
}

fn ensure_speaker(speakers: &mut BTreeMap<String, String>, speaker: &str) {
    speakers
        .entry(speaker.to_string())
        .or_insert_with(|| display_speaker(speaker));
}

fn chunk_speaker(chunk: &DecodeRegion) -> String {
    if chunk.speakers.len() > 1 {
        "overlap".to_string()
    } else if let Some(speaker) = chunk.speakers.first() {
        speaker.clone()
    } else {
        "unassigned".to_string()
    }
}

struct Window {
    start: i64,
    end: i64,
}

struct Examined {
    chunk: usize,
    start: i64,
    end: i64,
    blocked: Option<BlockedSpeech>,
    admission: Admission,
}

struct Decoded {
    start: f64,
    end: f64,
    text: String,
    raw: Option<String>,
    language: Option<String>,
    review: bool,
    audio_state: Option<String>,
    detection: Option<Value>,
    transcription_review: Option<Value>,
    acoustic: Option<Value>,
    pcm: Option<Value>,
    padding: Option<Value>,
}

fn plan_work(
    ledger: &SpeechLedger,
    pcm: &[f32],
    chunks: &[DecodeRegion],
) -> Result<Vec<Examined>, ImportError> {
    let mut work = Vec::new();
    let len = pcm.len() as i64;
    for (index, chunk) in chunks.iter().enumerate() {
        let start = seconds_to_sample(chunk.audio_start, len);
        let end = seconds_to_sample(chunk.audio_end, len);
        if end <= start {
            continue;
        }
        let windows = examine_range(ledger, pcm, start, end)?;
        let silent = windows.iter().all(|item| item.blocked.is_some());
        if chunk.speakers.is_empty() && silent {
            continue;
        }
        work.extend(windows.into_iter().map(|mut item| {
            item.chunk = index;
            item
        }));
    }
    Ok(work)
}

fn examine_range(
    ledger: &SpeechLedger,
    pcm: &[f32],
    start: i64,
    end: i64,
) -> Result<Vec<Examined>, ImportError> {
    let mut examined = Vec::new();
    for window in plan_windows(ledger, start, end)? {
        examined.push(examine(ledger, pcm, window)?);
    }
    Ok(examined)
}

fn plan_windows(ledger: &SpeechLedger, start: i64, end: i64) -> Result<Vec<Window>, ImportError> {
    if end <= start {
        return Ok(Vec::new());
    }
    let admission = ledger.admission(start, end).map_err(speech_err)?;
    if !admission.complete || admission.speech_regions.is_empty() {
        return Ok(vec![Window { start, end }]);
    }
    let pad = i64::from(language::CANONICAL_PAD_SAMPLES);
    let mut covered = Vec::new();
    for region in &admission.speech_regions {
        let lead = (region.start_sample - pad).max(start);
        let trail = (region.end_sample + pad).min(end);
        let region_start = lead.min(region.start_sample).max(start);
        let region_end = trail.max(region.end_sample).min(end);
        if region_start >= region_end {
            continue;
        }
        if let Some((_, last_end)) = covered.last_mut()
            && region_start <= *last_end
        {
            *last_end = (*last_end).max(region_end);
        } else {
            covered.push((region_start, region_end));
        }
    }
    if covered.is_empty() {
        return Ok(vec![Window { start, end }]);
    }
    let mut windows = Vec::new();
    let mut cursor = start;
    for (region_start, region_end) in covered {
        if region_start > cursor {
            windows.push(Window {
                start: cursor,
                end: region_start,
            });
        }
        windows.push(Window {
            start: region_start,
            end: region_end,
        });
        cursor = region_end;
    }
    if cursor < end {
        windows.push(Window { start: cursor, end });
    }
    Ok(windows)
}

fn examine(ledger: &SpeechLedger, pcm: &[f32], window: Window) -> Result<Examined, ImportError> {
    let admission = ledger
        .admission(window.start, window.end)
        .map_err(speech_err)?;
    let samples = &pcm[window.start as usize..window.end as usize];
    let blocked = speech::blocked_route(admission.decision, samples, window.start, window.end);
    Ok(Examined {
        chunk: 0,
        start: window.start,
        end: window.end,
        blocked,
        admission,
    })
}

fn decode_window(
    transcriber: Option<&mut SpeechTranscriber>,
    pcm: &[f32],
    language: &str,
    speakers: &[String],
    item: &Examined,
) -> Result<Vec<Decoded>, ImportError> {
    if let Some(blocked) = &item.blocked {
        return Ok(vec![blocked_row(language, item, blocked)]);
    }
    let Some(transcriber) = transcriber else {
        return Err(ImportError::new(messages::PROCESSING_FAILED));
    };
    let audio = &pcm[item.start as usize..item.end as usize];
    let passages = transcriber
        .transcribe(
            audio,
            limits::SAMPLE_RATE,
            speakers,
            TranscribeOptions {
                max_asr_seconds: IMPORT_MAX_SECONDS,
                start_sample: item.start,
                asr_padding: (0, 0),
            },
        )
        .map_err(language_err)?;
    Ok(passages
        .into_iter()
        .map(|passage| passage_row(item.start, passage))
        .collect())
}

fn blocked_row(language: &str, item: &Examined, blocked: &BlockedSpeech) -> Decoded {
    Decoded {
        start: samples_to_seconds(item.start),
        end: samples_to_seconds(item.end),
        text: String::new(),
        raw: None,
        language: manual_language(language),
        review: true,
        audio_state: Some(blocked.audio_state.to_string()),
        detection: None,
        transcription_review: None,
        acoustic: Some(admission_value(&item.admission)),
        pcm: blocked.pcm.as_ref().map(|pcm| {
            json!({
                "kind": pcm.kind,
                "start_sample": pcm.start_sample,
                "end_sample": pcm.end_sample,
                "sample_value": pcm.sample_value,
            })
        }),
        padding: None,
    }
}

fn passage_row(origin: i64, passage: Passage) -> Decoded {
    let rate = timing::RATE as f64;
    Decoded {
        start: origin as f64 / rate + passage.start,
        end: origin as f64 / rate + passage.end,
        text: passage.text,
        raw: passage.cohere_raw_text,
        language: passage.language,
        review: passage.review,
        audio_state: passage.audio_state.map(|state| state.as_str().to_string()),
        detection: Some(detection_value(&passage.language_detection)),
        transcription_review: passage
            .transcription_review
            .as_ref()
            .map(transcription_value),
        acoustic: Some(admission_value(&passage.acoustic_evidence)),
        pcm: passage.pcm_evidence.as_ref().map(|pcm| {
            json!({
                "kind": pcm.kind,
                "start_sample": pcm.start_sample,
                "end_sample": pcm.end_sample,
                "sample_value": pcm.sample_value,
            })
        }),
        padding: passage.cohere_input_padding.as_ref().map(|padding| {
            json!({
                "policy": padding.policy,
                "leading_samples": padding.leading_samples,
                "trailing_samples": padding.trailing_samples,
                "physical_start_sample": padding.physical_start_sample,
                "physical_end_sample": padding.physical_end_sample,
            })
        }),
    }
}

fn candidate_from_parts(
    start: f64,
    end: f64,
    language: &str,
    origin: i64,
    parts: &[Decoded],
) -> PassageCandidate {
    let _ = origin;
    let mut raws = Vec::new();
    let mut review = false;
    let mut reviews: Vec<&Value> = Vec::new();
    for part in parts {
        review |= part.review;
        if !part.text.trim().is_empty() {
            raws.push(part.raw.clone().unwrap_or_else(|| part.text.clone()));
        }
        if let Some(item) = &part.transcription_review {
            reviews.push(item);
        }
    }
    let partial = reviews
        .iter()
        .any(|item| item.get("partial_text").and_then(Value::as_bool) == Some(true))
        || parts.iter().any(|part| {
            part.text.trim().is_empty()
                && !matches!(
                    part.audio_state.as_deref(),
                    Some("model_non_speech" | "digital_silence" | "constant_signal")
                )
        });
    let transcription_review = reviews.first().map(|item| {
        let mut review = (*item).clone();
        if let Some(object) = review.as_object_mut() {
            object.insert("partial_text".to_string(), json!(partial));
        }
        review
    });
    PassageCandidate {
        start,
        end,
        text: raws.join(" "),
        language: language.to_string(),
        review,
        transcription_review,
        retry_parts: parts
            .iter()
            .map(|part| {
                json!({
                    "start": part.start,
                    "end": part.end,
                    "text": part.text,
                    "language": part.language,
                    "review": part.review,
                    "audio_state": part.audio_state,
                    "cohere_raw_text": part.raw,
                    "transcription_review": part.transcription_review,
                })
            })
            .collect(),
    }
}

fn segment_from_row(
    id: String,
    start: f64,
    end: f64,
    speaker: String,
    review: bool,
    chunk: &DecodeRegion,
    row: &Decoded,
) -> Segment {
    let mut rest = Map::new();
    rest.insert("speaker_candidates".into(), json!(chunk.speakers));
    rest.insert(
        "voice_eligible".into(),
        json!(
            chunk.speakers.len() == 1
                && row.audio_state.is_none()
                && chunk.activity_regions.is_none()
        ),
    );
    rest.insert("confidence".into(), Value::Null);
    rest.insert("timing".into(), json!("audio_crop"));
    if let Some(state) = &row.audio_state {
        rest.insert("audio_state".into(), json!(state));
    }
    if let Some(raw) = &row.raw {
        rest.insert("cohere_raw_text".into(), json!(raw));
    }
    if let Some(pcm) = &row.pcm {
        rest.insert("pcm_evidence".into(), pcm.clone());
    }
    if let Some(padding) = &row.padding {
        rest.insert("cohere_input_padding".into(), padding.clone());
    }
    if let Some(activity) = &chunk.activity_regions {
        rest.insert(
            "activity_regions".into(),
            json!(
                activity_slice(activity, start, end)
                    .into_iter()
                    .map(|part| json!({
                        "start": part.start,
                        "end": part.end,
                        "speakers": part.speakers,
                    }))
                    .collect::<Vec<_>>()
            ),
        );
    }
    Segment {
        id,
        start,
        end,
        speaker,
        text: row.text.clone(),
        language: row.language.clone(),
        language_detection: row.detection.clone(),
        transcription_review: row.transcription_review.clone(),
        reading_turns: None,
        reading_turn_provenance: None,
        alignment: None,
        speaker_activity: None,
        acoustic_evidence: row.acoustic.clone(),
        refinement_state: None,
        machine_revision: None,
        protected_fields: None,
        review_resolution: None,
        retry_candidate: None,
        canonical_utterance_id: None,
        audio_revision: None,
        review: Some(review),
        capture_source: None,
        rest,
    }
}

fn manual_language(language: &str) -> Option<String> {
    match LanguageChoice::parse(language) {
        Ok(LanguageChoice::Manual(code)) => Some(code),
        _ => None,
    }
}

fn detection_value(detection: &LanguageDetection) -> Value {
    json!({
        "mode": detection.mode.as_str(),
        "reason": detection.reason.as_str(),
        "probability": detection.probability,
        "margin": detection.margin,
        "detector_error": detection.detector_error,
        "context_end_sample": detection.context_end_sample,
        "candidates": detection.candidates.iter().map(|candidate| json!({
            "language": candidate.language,
            "probability": candidate.probability,
        })).collect::<Vec<_>>(),
    })
}

fn transcription_value(review: &TranscriptionReview) -> Value {
    let mut value = Map::new();
    value.insert("reason".into(), json!(review.reason));
    value.insert("partial_text".into(), json!(review.partial_text));
    if let Some(frames) = review.valid_feature_frames {
        value.insert("valid_feature_frames".into(), json!(frames));
    }
    if let Some(reason) = review.decoder_reason {
        value.insert("decoder_reason".into(), json!(reason));
    }
    Value::Object(value)
}

fn admission_value(admission: &Admission) -> Value {
    json!({
        "source": admission.source,
        "model_revision": admission.model_revision,
        "start_sample": admission.start_sample,
        "end_sample": admission.end_sample,
        "observed_until_sample": admission.observed_until_sample,
        "complete": admission.complete,
        "decision": admission.decision.as_str(),
        "speech_regions": admission.speech_regions.iter().map(|region| json!({
            "start_sample": region.start_sample,
            "end_sample": region.end_sample,
        })).collect::<Vec<_>>(),
        "maximum_probability": admission.maximum_probability,
    })
}

fn warnings(language: &str, limit: usize) -> Vec<String> {
    vec![
        WARN_TIMES.to_string(),
        if language == languages::AUTOMATIC {
            WARN_AUTO.to_string()
        } else {
            WARN_MANUAL.to_string()
        },
        WARN_OVERLAP.to_string(),
        format!(
            "This diarizer supports at most {limit} speakers; excess speakers cannot be reliably detected."
        ),
    ]
}

fn provenance(language: &str, limit: usize) -> Value {
    json!({
        "kind": "local_inference",
        "language": language,
        "speaker_limit": limit,
        "timing": if language == languages::AUTOMATIC {
            WARN_TIMING_AUTO
        } else {
            WARN_TIMING_MANUAL
        },
    })
}

fn clip_span(start: f64, end: f64, lo: f64, hi: f64) -> Option<(f64, f64)> {
    let start = start.max(lo);
    let end = end.min(hi);
    (end > start).then_some((start, end))
}

fn samples_to_seconds(sample: i64) -> f64 {
    sample as f64 / timing::RATE as f64
}

fn seconds_to_sample(seconds: f64, len: i64) -> i64 {
    let frames = (seconds * timing::RATE as f64).round_ties_even();
    if !frames.is_finite() || frames <= 0.0 {
        return 0;
    }
    if frames >= len as f64 {
        return len;
    }
    frames as i64
}

fn admit<M: Models + ?Sized + 'static>(
    models: &M,
    pcm: &[f32],
) -> Result<SpeechLedger, ImportError> {
    let handle = models.vad_open(None).map_err(child_err)?;
    let mut ledger = SpeechLedger::new(0).map_err(speech_err)?;
    let mut offset = 0usize;
    let mut failed: Option<ImportError> = None;
    while offset < pcm.len() && failed.is_none() {
        let end = (offset + speech::FRAME_SAMPLES).min(pcm.len());
        let last = end == pcm.len();
        match models.vad_feed(handle, &pcm[offset..end], last) {
            Ok(frames) => {
                let before = ledger.end_sample();
                for frame in frames {
                    let start = i64::try_from(frame.start).unwrap_or(-1);
                    let frame_end = i64::try_from(frame.end).unwrap_or(-1);
                    if start < 0 || frame_end <= start || frame_end as usize > pcm.len() {
                        failed = Some(ImportError::new(messages::PROCESSING_FAILED));
                        break;
                    }
                    if let Err(error) = ledger.append_pcm(
                        start,
                        &pcm[start as usize..frame_end as usize],
                        f64::from(frame.raw_p),
                        f64::from(frame.norm_p),
                    ) {
                        failed = Some(speech_err(error));
                        break;
                    }
                }
                if failed.is_none() && ledger.end_sample() != before + (end - offset) as i64 {
                    failed = Some(ImportError::new(messages::PROCESSING_FAILED));
                }
            }
            Err(error) => failed = Some(child_err(error)),
        }
        offset = end;
    }
    let closed = models.vad_close(handle).map_err(child_err);
    failed.map_or(Ok(()), Err)?;
    closed?;
    if ledger.end_sample() != pcm.len() as i64 {
        return Err(ImportError::new(messages::PROCESSING_FAILED));
    }
    Ok(ledger)
}

struct BorrowedModels<M: Models + ?Sized> {
    models: *const M,
    _marker: PhantomData<*const M>,
}

impl<M: Models + ?Sized> BorrowedModels<M> {
    fn new(models: &M) -> Self {
        Self {
            models: std::ptr::from_ref(models),
            _marker: PhantomData,
        }
    }

    fn models(&self) -> &M {
        // SAFETY: the pointer is the caller's `&M`, and every transcriber that
        // holds it returns before `import_recording` / `retry_passage` returns.
        unsafe { &*self.models }
    }
}

impl<M: Models + ?Sized> Recognizer for BorrowedModels<M> {
    fn transcribe(&mut self, pcm: &[f32], language: &str) -> Result<AsrOutput, RouteError> {
        if audio::pcm16_bytes(pcm).is_err() {
            return Err(RouteError::Recognition);
        }
        let decoded = self
            .models()
            .transcribe(pcm, language, (0, 0))
            .map_err(|_| RouteError::Recognition)?;
        let text = if decoded.raw_text.is_empty() && !decoded.text.is_empty() {
            decoded.text
        } else {
            decoded.raw_text
        };
        let token_count = text.split_whitespace().count();
        Ok(AsrOutput { text, token_count })
    }
}

impl<M: Models + ?Sized> LanguageDetector for BorrowedModels<M> {
    fn detect(&mut self, pcm: &[f32]) -> Result<Vec<(String, f64)>, RouteError> {
        let probe = self
            .models()
            .language_probe(pcm)
            .map_err(|_| RouteError::Detector)?;
        if probe.codes.len() != probe.probabilities.len() || probe.codes.is_empty() {
            return Err(RouteError::Detector);
        }
        Ok(probe
            .codes
            .into_iter()
            .zip(probe.probabilities.into_iter().map(f64::from))
            .collect())
    }
}

fn open_transcriber<M: Models + ?Sized + 'static>(
    models: &M,
    language: &str,
    ledger: SpeechLedger,
) -> Result<SpeechTranscriber, ImportError> {
    let detector = if language == languages::AUTOMATIC {
        Some(Box::new(BorrowedModels::new(models)) as Box<dyn LanguageDetector>)
    } else {
        None
    };
    SpeechTranscriber::new(
        language,
        Box::new(ledger),
        detector,
        Box::new(BorrowedModels::new(models)),
        None,
    )
    .map_err(language_err)
}

fn language_err(error: LanguageError) -> ImportError {
    ImportError::new(error.to_string())
}

fn speech_err(error: speech::SpeechError) -> ImportError {
    ImportError::new(error.to_string())
}

fn child_err(error: ChildError) -> ImportError {
    match error {
        ChildError::Failed { message, .. } => ImportError::new(message),
        _ => ImportError::new(messages::PROCESSING_FAILED),
    }
}

#[cfg(test)]
mod tests {
    use super::{
        ActivitySlice, CROP_PADDING_SECONDS, DEFAULT_MAX_SECONDS, IMPORT_MAX_SECONDS,
        TURN_GAP_SECONDS, crop_context, decode_regions, import_recording, retry_passage,
    };
    use crate::export;
    use crate::ipc::{
        self, AsrText, DiarFeedOut, GainPolicy, Handle, LanguageProbeOut, Turn, VadFrame,
    };
    use crate::language;
    use crate::languages;
    use crate::limits;
    use crate::messages;
    use crate::models::{Models, UnimplementedModels};
    use crate::timing;
    use serde_json::Value;
    use std::path::PathBuf;
    use std::sync::Mutex;

    #[derive(Clone, Debug)]
    struct AsrCall {
        samples: usize,
        language: String,
        padding: (u32, u32),
    }

    #[derive(Debug)]
    enum Event {
        Diarize,
        Vad,
        Probe,
        Transcribe(AsrCall),
    }

    struct Scripted {
        turns: Vec<Turn>,
        speech: Vec<(usize, usize)>,
        cursor: Mutex<usize>,
        events: Mutex<Vec<Event>>,
    }

    impl Scripted {
        fn new(turns: Vec<Turn>, speech: Vec<(usize, usize)>) -> Self {
            Self {
                turns,
                speech,
                cursor: Mutex::new(0),
                events: Mutex::new(Vec::new()),
            }
        }

        fn events(&self) -> Vec<String> {
            self.events
                .lock()
                .unwrap()
                .iter()
                .map(|event| match event {
                    Event::Diarize => "diarize".to_string(),
                    Event::Vad => "vad".to_string(),
                    Event::Probe => "probe".to_string(),
                    Event::Transcribe(call) => format!("asr:{}:{}", call.language, call.samples),
                })
                .collect()
        }

        fn calls(&self) -> Vec<AsrCall> {
            self.events
                .lock()
                .unwrap()
                .iter()
                .filter_map(|event| match event {
                    Event::Transcribe(call) => Some(call.clone()),
                    _ => None,
                })
                .collect()
        }
    }

    impl Models for Scripted {
        fn hello(&self, build_id: &str) -> Result<(String, String), crate::error::ChildError> {
            UnimplementedModels.hello(build_id)
        }
        fn vad_open(
            &self,
            conditioning: Option<GainPolicy>,
        ) -> Result<Handle, crate::error::ChildError> {
            let _ = conditioning;
            *self.cursor.lock().unwrap() = 0;
            self.events.lock().unwrap().push(Event::Vad);
            Ok(Handle(1))
        }
        fn vad_feed(
            &self,
            _handle: Handle,
            pcm: &[f32],
            _last: bool,
        ) -> Result<Vec<VadFrame>, crate::error::ChildError> {
            let mut cursor = self.cursor.lock().unwrap();
            let start = *cursor;
            let end = start + pcm.len();
            *cursor = end;
            let speech = self
                .speech
                .iter()
                .any(|(region_start, region_end)| start < *region_end && end > *region_start);
            let probability = if speech { 0.9 } else { 0.01 };
            Ok(vec![VadFrame {
                start: start as u64,
                end: end as u64,
                raw_p: probability,
                norm_p: probability,
                gain: 1.0,
                constant: false,
            }])
        }
        fn vad_close(&self, _handle: Handle) -> Result<(), crate::error::ChildError> {
            Ok(())
        }
        fn diar_open(&self, _preset: ipc::DiarPreset) -> Result<Handle, crate::error::ChildError> {
            UnimplementedModels.diar_open(_preset)
        }
        fn diar_feed(
            &self,
            handle: Handle,
            pcm: &[f32],
            last: bool,
        ) -> Result<DiarFeedOut, crate::error::ChildError> {
            UnimplementedModels.diar_feed(handle, pcm, last)
        }
        fn diar_close(&self, handle: Handle) -> Result<(), crate::error::ChildError> {
            UnimplementedModels.diar_close(handle)
        }
        fn diarize_file(
            &self,
            path: &std::path::Path,
        ) -> Result<Vec<Turn>, crate::error::ChildError> {
            UnimplementedModels.diarize_file(path)
        }
        fn diarize_window(&self, _pcm: &[f32]) -> Result<Vec<Turn>, crate::error::ChildError> {
            self.events.lock().unwrap().push(Event::Diarize);
            Ok(self.turns.clone())
        }
        fn language_probe(
            &self,
            _pcm: &[f32],
        ) -> Result<LanguageProbeOut, crate::error::ChildError> {
            self.events.lock().unwrap().push(Event::Probe);
            Ok(LanguageProbeOut {
                codes: vec!["en".to_string(), "fr".to_string()],
                probabilities: vec![0.99, 0.01],
            })
        }
        fn transcribe(
            &self,
            pcm: &[f32],
            language: &str,
            padding: (u32, u32),
        ) -> Result<AsrText, crate::error::ChildError> {
            self.events.lock().unwrap().push(Event::Transcribe(AsrCall {
                samples: pcm.len(),
                language: language.to_string(),
                padding,
            }));
            Ok(AsrText {
                text: format!("{language}:Hello"),
                raw_text: format!("  {language}:Hello  "),
            })
        }
        fn ctc_scores(
            &self,
            pcm: &[f32],
            token_ids: &[i32],
        ) -> Result<ipc::CtcScoresOut, crate::error::ChildError> {
            UnimplementedModels.ctc_scores(pcm, token_ids)
        }
        fn embed_voice(&self, pcm: &[f32]) -> Result<Vec<f32>, crate::error::ChildError> {
            UnimplementedModels.embed_voice(pcm)
        }
        fn setup_check(
            &self,
            models: &[String],
        ) -> Result<Vec<ipc::SetupModelReport>, crate::error::ChildError> {
            UnimplementedModels.setup_check(models)
        }
        fn metrics(&self) -> Result<ipc::MetricsOut, crate::error::ChildError> {
            UnimplementedModels.metrics()
        }
        fn cancel(&self, request_id: u64) -> Result<(), crate::error::ChildError> {
            UnimplementedModels.cancel(request_id)
        }
        fn shutdown(&self) -> Result<(), crate::error::ChildError> {
            UnimplementedModels.shutdown()
        }
    }

    fn tone(len: usize) -> Vec<f32> {
        (0..len)
            .map(|index| ((index % 17) as f32 - 8.0) / 1000.0)
            .collect()
    }

    fn seconds(samples: i64) -> f64 {
        samples as f64 / f64::from(limits::SAMPLE_RATE)
    }

    fn near(left: f64, right: f64) -> bool {
        (left - right).abs() <= 1e-6
    }

    fn golden(name: &str) -> Value {
        let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("../../tests/goldens")
            .join(name);
        let text = std::fs::read_to_string(&path)
            .unwrap_or_else(|error| panic!("read {}: {error}", path.display()));
        serde_json::from_str(&text).unwrap()
    }

    fn turns_of(value: &Value) -> Vec<Turn> {
        value
            .as_array()
            .unwrap()
            .iter()
            .map(|turn| Turn {
                start: turn["start"].as_f64().unwrap(),
                end: turn["end"].as_f64().unwrap(),
                speaker: turn["speaker"].as_str().unwrap().to_string(),
            })
            .collect()
    }

    fn slices_of(value: &Value) -> Vec<ActivitySlice> {
        value
            .as_array()
            .unwrap()
            .iter()
            .map(|part| ActivitySlice {
                start: part["start"].as_f64().unwrap(),
                end: part["end"].as_f64().unwrap(),
                speakers: part["speakers"]
                    .as_array()
                    .unwrap()
                    .iter()
                    .map(|speaker| speaker.as_str().unwrap().to_string())
                    .collect(),
            })
            .collect()
    }

    fn assert_slice(actual: &ActivitySlice, expected: &ActivitySlice) {
        assert!(
            near(actual.start, expected.start),
            "{actual:?} {expected:?}"
        );
        assert!(near(actual.end, expected.end), "{actual:?} {expected:?}");
        assert_eq!(actual.speakers, expected.speakers);
    }

    #[test]
    fn decode_regions_match_the_golden_file() {
        assert_eq!(TURN_GAP_SECONDS, 0.65);
        assert_eq!(DEFAULT_MAX_SECONDS, 24.5);
        assert_eq!(timing::RATE, i64::from(limits::SAMPLE_RATE));
        let golden = golden("decode_regions.json");
        assert_eq!(golden["turn_gap_seconds"].as_f64(), Some(0.65));
        assert_eq!(golden["default_max_seconds"].as_f64(), Some(24.5));
        for case in golden["cases"].as_array().unwrap() {
            let turns = turns_of(&case["turns"]);
            let arguments = &case["arguments"];
            let max_seconds = arguments
                .get("max_seconds")
                .and_then(Value::as_f64)
                .unwrap_or(DEFAULT_MAX_SECONDS);
            let coverage = arguments.get("coverage").map(|value| {
                let pair = value.as_array().unwrap();
                (pair[0].as_f64().unwrap(), pair[1].as_f64().unwrap())
            });
            let regions = decode_regions(&turns, Some(max_seconds), coverage);
            let expected = case["regions"].as_array().unwrap();
            assert_eq!(regions.len(), expected.len(), "{}", case["name"]);
            for (region, expected) in regions.iter().zip(expected) {
                assert!(near(region.start, expected["start"].as_f64().unwrap()));
                assert!(near(region.end, expected["end"].as_f64().unwrap()));
                assert_eq!(
                    region.speakers,
                    expected["speakers"]
                        .as_array()
                        .unwrap()
                        .iter()
                        .map(|speaker| speaker.as_str().unwrap().to_string())
                        .collect::<Vec<_>>()
                );
                match (&region.activity_regions, expected.get("activity_regions")) {
                    (None, None) => {}
                    (Some(actual), Some(expected)) => {
                        let expected = slices_of(expected);
                        assert_eq!(actual.len(), expected.len(), "{}", case["name"]);
                        for (actual, expected) in actual.iter().zip(&expected) {
                            assert_slice(actual, expected);
                        }
                    }
                    (actual, expected) => {
                        panic!("{} activity {actual:?} {expected:?}", case["name"])
                    }
                }
            }
        }
    }

    #[test]
    fn crop_context_matches_the_golden_file() {
        let golden = golden("crop_context.json");
        for case in golden["cases"].as_array().unwrap() {
            let mut chunks = slices_of(&case["chunks_before"])
                .into_iter()
                .map(|slice| super::DecodeRegion {
                    start: slice.start,
                    end: slice.end,
                    speakers: slice.speakers,
                    activity_regions: None,
                    audio_start: slice.start,
                    audio_end: slice.end,
                })
                .collect::<Vec<_>>();
            let padding = case["padding"].as_f64().unwrap();
            let duration = case["duration"].as_f64().unwrap();
            assert!(padding.abs() < 1e-9 || (padding - CROP_PADDING_SECONDS).abs() < 1e-9);
            crop_context(&mut chunks, duration, padding);
            let expected = case["chunks_after"].as_array().unwrap();
            assert_eq!(chunks.len(), expected.len());
            for (chunk, expected) in chunks.iter().zip(expected) {
                assert!(near(
                    chunk.audio_start,
                    expected["audio_start"].as_f64().unwrap()
                ));
                assert!(near(
                    chunk.audio_end,
                    expected["audio_end"].as_f64().unwrap()
                ));
                assert!(near(chunk.start, expected["start"].as_f64().unwrap()));
                assert!(near(chunk.end, expected["end"].as_f64().unwrap()));
            }
        }
    }

    #[test]
    fn manual_import_names_one_speaker_and_skips_language_id() {
        let samples = 8_192;
        let models = Scripted::new(
            vec![Turn {
                start: 0.0,
                end: seconds(samples),
                speaker: "speaker_0".into(),
            }],
            vec![(0, samples as usize)],
        );
        let document = import_recording(&models, &tone(samples as usize), "en").unwrap();
        assert_eq!(
            document.speakers.get("speaker_0").map(String::as_str),
            Some("Speaker 1")
        );
        let spoken = document
            .segments
            .iter()
            .find(|segment| !segment.text.is_empty())
            .unwrap();
        assert_eq!(spoken.speaker, "speaker_0");
        assert_eq!(spoken.text, "en:Hello");
        assert_eq!(spoken.review, Some(false));
        assert_eq!(spoken.language.as_deref(), Some("en"));
        assert_eq!(
            spoken.rest.get("voice_eligible").and_then(Value::as_bool),
            Some(true)
        );
        assert_eq!(
            spoken.rest.get("timing").and_then(Value::as_str),
            Some("audio_crop")
        );
        assert!(
            document
                .warnings
                .iter()
                .any(|warning| warning == super::WARN_MANUAL)
        );
        assert!(
            document
                .warnings
                .iter()
                .any(|warning| warning.contains("at most 8"))
        );
        assert_eq!(document.warnings.len(), 4);
        let calls = models.calls();
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].language, "en");
        assert_eq!(calls[0].padding, (0, 0));
        assert_eq!(calls[0].samples, samples as usize);
        assert!(!models.events().iter().any(|event| event == "probe"));
        let exported = export::export(&document, "txt").unwrap();
        assert_eq!(exported.body.matches("en:Hello").count(), 1);
        assert!(exported.body.contains("Speaker 1"));
    }

    #[test]
    fn speech_admission_blocks_the_silent_tail_before_asr() {
        let samples = 16_384;
        let speech = 8_192usize;
        let models = Scripted::new(
            vec![Turn {
                start: 0.0,
                end: seconds(samples),
                speaker: "speaker_0".into(),
            }],
            vec![(0, speech)],
        );
        let document = import_recording(&models, &tone(samples as usize), "en").unwrap();
        let calls = models.calls();
        assert_eq!(calls.len(), 1);
        let context = speech + language::CANONICAL_PAD_SAMPLES as usize;
        assert_eq!(calls[0].samples, context);
        assert!(calls[0].samples < samples as usize);
        let tail = document
            .segments
            .iter()
            .find(|segment| segment.text.is_empty())
            .unwrap();
        assert_eq!(
            tail.rest.get("audio_state").and_then(Value::as_str),
            Some("model_non_speech")
        );
        assert_eq!(tail.review, Some(true));
        assert_eq!(models.events()[0], "diarize");
        assert_eq!(models.events()[1], "vad");
    }

    #[test]
    fn silence_inside_a_speaker_crop_never_calls_asr() {
        let samples = 8_192usize;
        let models = Scripted::new(
            vec![Turn {
                start: 0.0,
                end: seconds(samples as i64),
                speaker: "speaker_0".into(),
            }],
            Vec::new(),
        );
        let document = import_recording(&models, &tone(samples), "en").unwrap();
        assert!(models.calls().is_empty());
        assert!(
            document
                .segments
                .iter()
                .all(|segment| segment.text.is_empty())
        );
        assert!(document.segments.iter().any(|segment| {
            segment.rest.get("audio_state").and_then(Value::as_str) == Some("model_non_speech")
        }));
    }

    #[test]
    fn overlap_is_mixed_and_reviewed() {
        let samples = 8_192usize;
        let end = seconds(samples as i64);
        let models = Scripted::new(
            vec![
                Turn {
                    start: 0.0,
                    end,
                    speaker: "speaker_0".into(),
                },
                Turn {
                    start: 0.0,
                    end,
                    speaker: "speaker_1".into(),
                },
            ],
            vec![(0, samples)],
        );
        let document = import_recording(&models, &tone(samples), "en").unwrap();
        assert_eq!(
            document.speakers.get("speaker_0").map(String::as_str),
            Some("Speaker 1")
        );
        assert_eq!(
            document.speakers.get("speaker_1").map(String::as_str),
            Some("Speaker 2")
        );
        assert_eq!(
            document.speakers.get("overlap").map(String::as_str),
            Some("Overlapping speakers")
        );
        let spoken = document
            .segments
            .iter()
            .find(|segment| segment.speaker == "overlap")
            .unwrap();
        assert_eq!(spoken.review, Some(true));
        assert!(!spoken.text.is_empty());
        assert!(
            document
                .warnings
                .iter()
                .any(|warning| warning == super::WARN_OVERLAP)
        );
    }

    #[test]
    fn retain_unassigned_audio_covers_silence_outside_speaker_turns() {
        let speech = 8_192i64;
        let samples = 16_384usize;
        let models = Scripted::new(
            vec![Turn {
                start: 0.0,
                end: seconds(speech),
                speaker: "speaker_0".into(),
            }],
            vec![(0, speech as usize)],
        );
        let document = import_recording(&models, &tone(samples), "en").unwrap();
        assert_eq!(
            document.speakers.get("unassigned").map(String::as_str),
            Some("Unassigned audio")
        );
        let gap = document
            .segments
            .iter()
            .find(|segment| segment.speaker == "unassigned")
            .unwrap();
        assert_eq!(gap.review, Some(true));
        assert_eq!(gap.text, "");
        assert_eq!(
            gap.rest.get("timing").and_then(Value::as_str),
            Some("audio_coverage")
        );
        assert_eq!(
            gap.transcription_review
                .as_ref()
                .and_then(|value| value.get("reason"))
                .and_then(Value::as_str),
            Some("unassigned_audio")
        );
        assert!(
            document
                .segments
                .iter()
                .any(|segment| { segment.speaker == "speaker_0" && segment.text == "en:Hello" })
        );
        let covered: f64 = document
            .segments
            .iter()
            .map(|segment| segment.end - segment.start)
            .sum();
        assert!(near(covered, seconds(samples as i64)));
    }

    #[test]
    fn automatic_language_probes_then_transcribes_the_winner() {
        let samples = 8_192usize;
        let models = Scripted::new(
            vec![Turn {
                start: 0.0,
                end: seconds(samples as i64),
                speaker: "speaker_0".into(),
            }],
            vec![(0, samples)],
        );
        let document = import_recording(&models, &tone(samples), languages::AUTOMATIC).unwrap();
        assert!(models.events().iter().any(|event| event == "probe"));
        let calls = models.calls();
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].language, "en");
        assert_eq!(
            document
                .segments
                .iter()
                .find(|segment| !segment.text.is_empty())
                .and_then(|segment| segment.language.as_deref()),
            Some("en")
        );
        assert!(
            document
                .warnings
                .iter()
                .any(|warning| warning == super::WARN_AUTO)
        );
    }

    #[test]
    fn retry_rejects_automatic_language_and_passages_over_24_5_seconds() {
        let models = Scripted::new(Vec::new(), Vec::new());
        let pcm = tone(1_600);
        let language = retry_passage(&models, &pcm, 0.0, 0.05, languages::AUTOMATIC).unwrap_err();
        assert_eq!(language.message, messages::RETRY_LANGUAGE);
        let unknown = retry_passage(&models, &pcm, 0.0, 0.05, "xx").unwrap_err();
        assert_eq!(unknown.message, messages::RETRY_LANGUAGE);
        let too_long = retry_passage(
            &models,
            &pcm,
            0.0,
            limits::PASSAGE_RETRY_MAX_SECONDS + 0.01,
            "en",
        )
        .unwrap_err();
        assert_eq!(too_long.message, messages::RETRY_TOO_LONG);
        assert!(models.calls().is_empty());
        assert!(!models.events().iter().any(|event| event == "diarize"));
    }

    #[test]
    fn retry_at_the_24_5_second_limit_returns_a_candidate_only() {
        let samples = (limits::PASSAGE_RETRY_MAX_SECONDS * f64::from(limits::SAMPLE_RATE)) as usize;
        assert_eq!(samples, 392_000);
        let pcm = tone(samples);
        let models = Scripted::new(
            vec![Turn {
                start: 0.0,
                end: limits::PASSAGE_RETRY_MAX_SECONDS,
                speaker: "speaker_0".into(),
            }],
            vec![(0, samples)],
        );
        let document = import_recording(&models, &pcm, "en").unwrap();
        let before = document.clone();
        let candidate =
            retry_passage(&models, &pcm, 0.0, limits::PASSAGE_RETRY_MAX_SECONDS, "de").unwrap();
        assert_eq!(document, before);
        assert_eq!(candidate.language, "de");
        assert_eq!(candidate.start, 0.0);
        assert_eq!(candidate.end, limits::PASSAGE_RETRY_MAX_SECONDS);
        assert_eq!(candidate.text, "  de:Hello  ");
        assert!(
            document
                .segments
                .iter()
                .any(|segment| segment.text == "en:Hello")
        );
        assert!(
            candidate
                .retry_parts
                .iter()
                .all(|part| part.get("speaker").is_none())
        );
        let calls = models.calls();
        assert!(calls.iter().any(|call| call.language == "de"));
        assert_eq!(
            models
                .events()
                .iter()
                .filter(|event| *event == "diarize")
                .count(),
            1
        );
    }

    #[test]
    fn retry_keeps_raw_text_from_each_admitted_island_and_skips_the_gap() {
        let samples = 16_384usize;
        let models = Scripted::new(Vec::new(), vec![(0, 512), (7_680, 8_192)]);
        let candidate =
            retry_passage(&models, &tone(samples), 0.0, seconds(samples as i64), "en").unwrap();
        let calls = models.calls();
        assert_eq!(calls.len(), 2);
        assert!(calls.iter().all(|call| call.language == "en"));
        assert_eq!(candidate.text, "  en:Hello     en:Hello  ");
        assert!(candidate.review);
        assert!(candidate.end - candidate.start <= limits::PASSAGE_RETRY_MAX_SECONDS);
        assert_eq!(IMPORT_MAX_SECONDS, 18.0);
    }

    #[test]
    fn empty_and_non_finite_audio_fail_before_diarization() {
        let models = Scripted::new(Vec::new(), Vec::new());
        let empty = import_recording(&models, &[], "en").unwrap_err();
        assert_eq!(empty.message, messages::RECORDING_LENGTH);
        let mut pcm = tone(512);
        pcm[0] = f32::NAN;
        let non_finite = import_recording(&models, &pcm, "en").unwrap_err();
        assert_eq!(non_finite.message, messages::CAPTURE_FINITE_MONO);
        assert!(models.events().is_empty());
        let unsupported = import_recording(&models, &tone(512), "xx").unwrap_err();
        assert_eq!(unsupported.message, messages::CHOOSE_LANGUAGE_MODE);
    }
}
