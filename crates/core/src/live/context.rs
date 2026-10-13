//! Bounded raw-text selection between adjacent, independently aligned contexts.
//! These cells come only from the local supplied-text adapter, which validates
//! model columns and raw-text normalization. Parse each result once; matching
//! uses acoustic cell identity and never writes normalized or matched spelling.
use super::*;
use crate::timing_policy::{
    NORMALIZATION_POLICY_ID, PROVIDER_ID, TIMING_UNIT_POLICY_ID, UPSTREAM_NORMALIZATION_REVISION,
};
use serde_json::json;
use std::ops::Range;

const MARGIN: i64 = 4_000;

#[derive(PartialEq)]
struct Cell {
    token: i64,
    start: i64,
    end: i64,
    chars: Range<usize>,
}
struct Word {
    timing: TimedWord,
    cells: Range<usize>,
}
struct Context {
    request: DecodeRequest,
    language: String,
    raw: Vec<char>,
    cells: Vec<Cell>,
    words: Vec<Word>,
}
pub(super) fn complete_alignment(part: TailPart) -> bool {
    Context::parse(part).is_some()
}

impl Context {
    fn parse(part: TailPart) -> Option<Self> {
        let result = part.alignment?;
        let request = part.request;
        let language = result["language"].as_str()?;
        let calibration = crate::alignment::calibration_for_language(language);
        if !part.complete
            || !languages::is_supported(language)
            || result["raw_text"] != part.text
            || result["text_sha256"] != sha256_hex(part.text.as_bytes())
            || result["audio_anchor"]
                != json!({"start_sample":request.start_sample,"end_sample":request.end_sample})
            || result["provider_identity_sha256"] != crate::alignment::NATIVE_PROVIDER
            || result["provider_id"] != PROVIDER_ID
            || result["model_sha256"] != timing::MODEL
            || result["normalization_policy_id"] != NORMALIZATION_POLICY_ID
            || result["timing_unit_policy_id"] != TIMING_UNIT_POLICY_ID
            || result["frame_calibration_id"] != calibration
            || result["score_calibration_id"] != calibration
            || result["complete"] != true
            || result["status"] != "aligned"
        {
            return None;
        }
        let raw: Vec<char> = part.text.chars().collect();
        let mut cells = Vec::new();
        let mut previous_frame = -1;
        let mut previous_start = 0;
        let mut previous_end = 0;
        for c in result["characters"].as_array()? {
            let a = usize::try_from(c["start_char"].as_u64()?).ok()?;
            let b = usize::try_from(c["end_char"].as_u64()?).ok()?;
            let frame = c["frame"].as_i64()?;
            let score = c["log_probability"].as_f64()?;
            let independent = c["independent_acoustic_timestamp"] != false;
            let (status, start, end) = if independent {
                (
                    c["status"].as_str()?,
                    c["start_sample"].as_i64()?,
                    c["end_sample"].as_i64()?,
                )
            } else if c["text"] == " " && c["status"] == "not_independently_timed" {
                // The adapter's Japanese prolonged-mark alias occupies a CTC
                // separator cell, but cannot supply a word boundary by itself.
                (
                    c["acoustic_target_status"].as_str()?,
                    c["target_cell_start_sample"].as_i64()?,
                    c["target_cell_end_sample"].as_i64()?,
                )
            } else {
                return None;
            };
            if a < previous_start
                || b < previous_end
                || b <= a
                || b > raw.len()
                || frame <= previous_frame
                || !score.is_finite()
                || !(-20. ..=0.).contains(&score)
                || status != "aligned"
                || c["raw_text"] != raw[a..b].iter().collect::<String>()
                || start != request.start_sample.checked_add(frame.checked_mul(320)?)?
                || end != start.checked_add(320)?
                || start < request.start_sample
                || end > request.end_sample
            {
                return None;
            }
            previous_frame = frame;
            previous_start = a;
            previous_end = b;
            if independent && c["text"] != " " {
                cells.push(Cell {
                    token: c["token_id"].as_i64()?,
                    start,
                    end,
                    chars: a..b,
                });
            }
        }
        let mut words = Vec::new();
        let mut prior_end = 0;
        for w in result["words"].as_array()? {
            let a = usize::try_from(w["start_char"].as_u64()?).ok()?;
            let b = usize::try_from(w["end_char"].as_u64()?).ok()?;
            if a < prior_end
                || b <= a
                || b > raw.len()
                || w["status"] != "aligned"
                || w["text"] != raw[a..b].iter().collect::<String>()
            {
                return None;
            }
            let first = cells.partition_point(|c| c.chars.end <= a);
            let last = cells.partition_point(|c| c.chars.start < b);
            if first == last
                || cells[first].chars.start < a
                || cells[last - 1].chars.end > b
                || w["start_sample"].as_i64()? != cells[first].start
                || w["end_sample"].as_i64()? != cells[last - 1].end
            {
                return None;
            }
            words.push(Word {
                timing: TimedWord {
                    text: raw[a..b].iter().collect(),
                    start_char: a,
                    end_char: b,
                    start_sample: Some(cells[first].start),
                    end_sample: Some(cells[last - 1].end),
                },
                cells: first..last,
            });
            prior_end = b;
        }
        if words.is_empty() || cells.is_empty() {
            return None;
        }
        Some(Self {
            request,
            language: language.into(),
            raw,
            cells,
            words,
        })
    }
}

fn temporal_match(a: &[Cell], b: &[Cell]) -> bool {
    a.len() == b.len()
        && a.iter().zip(b).all(|(x, y)| {
            x.token == y.token
                && (x.start - y.start).abs() <= MARGIN
                && (x.end - y.end).abs() <= MARGIN
        })
        && a.iter()
            .zip(b)
            .any(|(x, y)| x.start.max(y.start) < x.end.min(y.end))
}

fn boundary(left: &Context, right: &Context) -> Result<(usize, usize), &'static str> {
    let shared_start = left.request.start_sample.max(right.request.start_sample);
    let shared_end = left.request.end_sample.min(right.request.end_sample);
    let mut reason = "no_complete_boundary_anchor";
    for word in left.words.iter().rev().filter(|word| {
        let (start, end) = (
            word.timing.start_sample.unwrap(),
            word.timing.end_sample.unwrap(),
        );
        start < left.request.core_end_sample + MARGIN
            && shared_start + MARGIN <= start
            && end <= shared_end - MARGIN
    }) {
        let source = &left.cells[word.cells.clone()];
        let mut matches =
            right
                .cells
                .windows(source.len())
                .enumerate()
                .filter_map(|(first, group)| {
                    if !temporal_match(source, group) {
                        return None;
                    }
                    let last = first + group.len();
                    let a = right.words.partition_point(|w| w.cells.end <= first);
                    let b = right.words.partition_point(|w| w.cells.start < last);
                    (a < b
                        && right.words[a].cells.start == first
                        && right.words[b - 1].cells.end == last)
                        .then_some((first, right.words[b - 1].timing.end_char))
                });
        let Some((first, raw_end)) = matches.next() else {
            reason = "boundary_mapping_absent";
            continue;
        };
        if matches.next().is_some() {
            reason = "ambiguous_boundary_mapping";
            continue;
        }
        let group = &right.cells[first..first + source.len()];
        let mut reverse = left
            .cells
            .windows(group.len())
            .enumerate()
            .filter(|(_, c)| temporal_match(c, group));
        if reverse.next().map(|(first, _)| first) != Some(word.cells.start)
            || reverse.next().is_some()
        {
            reason = "ambiguous_bidirectional_boundary";
            continue;
        }
        return Ok((word.timing.end_char, raw_end));
    }
    Err(reason)
}

/// Keeps only adjacent full contexts. Accepted raw fragments and word envelopes
/// are the output, not an alternate document or a retained decode history.
#[derive(Default)]
pub(super) struct Composer {
    previous: Option<Context>,
    from: usize,
    text: String,
    characters: usize,
    words: Vec<TimedWord>,
}
impl Composer {
    pub(super) fn push(&mut self, part: TailPart) -> Result<(), &'static str> {
        let next = Context::parse(part).ok_or("unavailable_boundary_timing")?;
        if let Some(previous) = self.previous.take() {
            if previous.language != next.language {
                return Err("context_language_mismatch");
            }
            let (left_end, right_start) = boundary(&previous, &next)?;
            self.append(&previous, self.from, left_end)?;
            self.from = right_start;
        }
        self.previous = Some(next);
        Ok(())
    }
    fn append(&mut self, part: &Context, mut a: usize, mut b: usize) -> Result<(), &'static str> {
        if b < a || b > part.raw.len() {
            return Err("nonmonotonic_raw_boundary");
        }
        let boundary = a;
        while a < b && part.raw[a].is_whitespace() {
            a += 1;
        }
        while a < b && part.raw[b - 1].is_whitespace() {
            b -= 1;
        }
        if a == b {
            return Ok(());
        }
        // The right context owns the separator after the corroborated unit.
        // CJK character units can be adjacent without an intervening space.
        if !self.text.is_empty() && a > boundary {
            self.text.extend(&part.raw[boundary..a]);
            self.characters += a - boundary;
        }
        let offset = self.characters;
        self.text.extend(&part.raw[a..b]);
        self.characters += b - a;
        for word in part
            .words
            .iter()
            .filter(|w| a <= w.timing.start_char && w.timing.end_char <= b)
        {
            let mut timing = word.timing.clone();
            timing.start_char = offset + timing.start_char - a;
            timing.end_char = offset + timing.end_char - a;
            self.words.push(timing);
        }
        Ok(())
    }
    pub(super) fn finish(mut self) -> Result<TailAssembly, &'static str> {
        let last = self.previous.take().ok_or("missing_context")?;
        self.append(&last, self.from, last.raw.len())?;
        // Context estimators can disagree on neighbouring coarse envelopes. Keep
        // their raw words, but never publish overlapping word timing as certainty.
        let overlaps: Vec<usize> = self
            .words
            .windows(2)
            .enumerate()
            .filter_map(|(i, pair)| (pair[0].end_sample? > pair[1].start_sample?).then_some(i))
            .collect();
        for i in overlaps {
            for word in &mut self.words[i..=i + 1] {
                word.start_sample = None;
                word.end_sample = None;
            }
        }
        Ok(TailAssembly {
            policy: AUTHORITATIVE_POLICY,
            complete: true,
            recognition_complete: true,
            text: Some(self.text),
            reason: None,
            words: self.words,
        })
    }
}

/// Saved long work never holds a whole recording or a live model handle. Every
/// physical crop is bounded, with independent admission and language routing.
/// Incremental complete-anchor recognition. Each step consumes one bounded physical
/// context; callers can drain live audio and ordered controls between steps.
pub struct Refinement {
    original: Segment,
    start: i64,
    end: i64,
    requests: std::vec::IntoIter<DecodeRequest>,
    composer: Composer,
    detection: Option<Value>,
    selected_language: Option<String>,
    english_calibrated: bool,
    probes: Vec<Value>,
    audio_hash: Sha256,
    review: bool,
    transcription_review: Option<Value>,
    language_context: Option<crate::language::LanguageContext>,
    retained_probes: Vec<Value>,
    interpretation_generation: u64,
}
impl Refinement {
    pub fn new(original: &Segment) -> Result<Self, LiveError> {
        Self::with_language_context(original, None)
    }
    pub fn with_language_context(
        original: &Segment,
        language_context: Option<crate::language::LanguageContext>,
    ) -> Result<Self, LiveError> {
        let (start, end) = saved_bounds(original)?;
        let requests = if end - start <= MAX_DECODE_SAMPLES {
            Vec::new()
        } else {
            authoritative_requests(&AnchorRow {
                id: original
                    .canonical_utterance_id
                    .clone()
                    .expect("checked canonical"),
                start_sample: start,
                end_sample: end,
                machine_revision: original.machine_revision.unwrap_or(0),
                audio_revision: original.audio_revision.unwrap_or(0),
                language_epoch: original.rest["language_epoch"].as_i64().unwrap_or(0),
            })?
        };
        let prefix = original
            .rest
            .get("canonical_text_end_sample")
            .and_then(Value::as_i64)
            .is_some_and(|through| start < through && through < end);
        let mut retained_probes = if !original.text.trim().is_empty()
            && (prefix
                || (original
                    .transcription_review
                    .as_ref()
                    .is_none_or(|v| v["partial_text"] != true)
                    && original
                        .rest
                        .get("canonical_unresolved")
                        .is_none_or(Value::is_null)
                    && original.rest.get("audio_state").is_none_or(Value::is_null)))
        {
            original
                .language_detection
                .as_ref()
                .and_then(|d| d["probes"].as_array())
                .into_iter()
                .flatten()
                .filter(|p| {
                    p["decision"]["reason"] == "detected"
                        && p["decision"]["mode"] == "auto"
                        && p["end_sample"].as_i64().is_some_and(|b| {
                            b <= original
                                .rest
                                .get("canonical_text_end_sample")
                                .and_then(Value::as_i64)
                                .unwrap_or(end)
                        })
                })
                .cloned()
                .collect::<Vec<_>>()
        } else {
            Vec::new()
        };
        retained_probes.sort_by_key(|p| {
            (
                p["end_sample"].as_i64().unwrap_or(i64::MAX),
                p["start_sample"].as_i64().unwrap_or(i64::MAX),
            )
        });
        Ok(Self {
            original: original.clone(),
            start,
            end,
            requests: requests.into_iter(),
            composer: Composer::default(),
            detection: None,
            selected_language: None,
            english_calibrated: true,
            probes: Vec::new(),
            audio_hash: Sha256::new(),
            review: false,
            transcription_review: None,
            language_context,
            retained_probes,
            interpretation_generation: 0,
        })
    }
    pub fn set_interpretation_generation(&mut self, generation: u64) {
        self.interpretation_generation = generation;
        if let Some(evidence) = self
            .language_context
            .as_mut()
            .and_then(|context| context.evidence.as_mut())
        {
            evidence.interpretation_generation = generation;
        }
    }
    pub fn step<M: Models + ?Sized + 'static>(
        &mut self,
        models: &M,
        reader: &RecordingReader,
        vocabulary: Option<&crate::alignment::Vocabulary>,
    ) -> Result<Option<Segment>, LiveError> {
        let (start, end) = (self.start, self.end);
        let original = &self.original;
        let committed = u32::try_from(end)
            .map_err(|_| LiveError::Message("Saved audio range is too large."))?;
        let read = |a: i64, b: i64| {
            reader
                .read_samples(a as u32, b as u32, committed)
                .map_err(|e| LiveError::Detail(e.to_string()))
        };
        if end - start <= MAX_DECODE_SAMPLES {
            return refine_saved_passage_with_context(
                models,
                original,
                &read(start, end)?,
                vocabulary,
                self.language_context.clone(),
            )
            .map(Some);
        }
        let mode = original.rest["language_mode"]
            .as_str()
            .expect("checked mode");
        let processed = processed_catalog(original);
        let request = self.requests.next().ok_or(LiveError::Message(
            "Accuracy context was already completed.",
        ))?;
        let pcm = read(request.start_sample, request.end_sample)?;
        let padding = (
            if request.start_sample > start {
                3200
            } else {
                0
            },
            if request.end_sample < end { 3200 } else { 0 },
        );
        let source = crate::import::refine_context_source(
            models,
            &pcm,
            mode,
            request.start_sample,
            processed,
            Some(padding),
            if mode == languages::AUTOMATIC {
                same_anchor_window_context(original, &self.probes, request.start_sample, true)
                    .or_else(|| {
                        same_anchor_window_context(
                            original,
                            &self.retained_probes,
                            request.start_sample,
                            false,
                        )
                    })
                    .or_else(|| self.language_context.clone())
                    .map(|mut context| {
                        if let Some(evidence) = context.evidence.as_mut() {
                            evidence.interpretation_generation = self.interpretation_generation;
                        }
                        context
                    })
            } else {
                None
            },
        )
        .map_err(LiveError::from)?;
        let language = source.language.ok_or_else(|| LiveError::Language {reason:"mixed",message:
            "Long context accuracy needs unambiguous language ownership; previous words and audio are retained.".into(),
        })?;
        let (text, decision) = (source.text, source.detection);
        self.review |= source.review;
        if self.transcription_review.is_none() {
            self.transcription_review = source.transcription_review;
        }
        if self
            .selected_language
            .as_ref()
            .is_some_and(|prior| prior != &language)
        {
            return Err(LiveError::Language { reason:"mixed",message:
                "Long context accuracy requires consistent language ownership throughout; previous words and audio are retained.".into(),
            });
        }
        self.selected_language
            .get_or_insert_with(|| language.clone());
        if let Some(events) = decision["probes"].as_array() {
            for event in events {
                let key = (
                    event["end_sample"].as_i64().unwrap_or(i64::MAX),
                    event["start_sample"].as_i64().unwrap_or(i64::MAX),
                );
                let index = self.probes.partition_point(|p| {
                    (
                        p["end_sample"].as_i64().unwrap_or(i64::MAX),
                        p["start_sample"].as_i64().unwrap_or(i64::MAX),
                    ) <= key
                });
                self.probes.insert(index, event.clone());
            }
        }
        if self.detection.as_ref().is_none_or(|prior| {
            decision["probability"].as_f64().unwrap_or(1.)
                < prior["probability"].as_f64().unwrap_or(1.)
        }) {
            self.detection = Some(decision);
        }
        let vocabulary=vocabulary.ok_or(LiveError::Message("Long context accuracy needs supplied-text timing; previous words and audio are retained."))?;
        let alignment = vocabulary.align(models, &pcm, &text, request.start_sample, &language)?;
        self.english_calibrated &=
            alignment["qualified_scope"] == "bounded_ami_english_coarse_envelopes";
        for sample in &pcm[(request.core_start_sample - request.start_sample) as usize
            ..(request.core_end_sample - request.start_sample) as usize]
        {
            self.audio_hash.update(sample.to_le_bytes());
        }
        self.composer
            .push(TailPart {
                request,
                text,
                complete: true,
                alignment: Some(alignment),
            })
            .map_err(LiveError::Message)?;

        if self.requests.len() != 0 {
            return Ok(None);
        }
        let assembly = std::mem::take(&mut self.composer)
            .finish()
            .map_err(LiveError::Message)?;
        let language = self
            .selected_language
            .take()
            .expect("nonempty context plan");
        let mut detection = self.detection.take().expect("nonempty context plan");
        if mode == languages::AUTOMATIC {
            detection["probes"] = json!(std::mem::take(&mut self.probes));
        }
        let mut candidate = saved_candidate(
            original,
            assembly.text.expect("complete assembly"),
            Some(language.clone()),
            detection,
            end,
            self.review,
            self.transcription_review.take(),
        )?;
        let units = timing::raw_units(&candidate.text, TIMING_UNIT_POLICY_ID)
            .expect("known timing unit policy");
        if units.len() != assembly.words.len()
            || units.iter().zip(&assembly.words).any(|(u, w)| {
                u.text != w.text || u.start_char != w.start_char || u.end_char != w.end_char
            })
        {
            return Err(LiveError::Message(
                "Selected context timing differs from its raw text units.",
            ));
        }
        let words:Vec<Value>=assembly.words.iter().zip(&units).map(|(w,u)|json!({"text":w.text,"start_char":w.start_char,"end_char":w.end_char,
        "start_utf16":u.start_utf16,"end_utf16":u.end_utf16,"start_sample":w.start_sample,"end_sample":w.end_sample,
        "unit_kind":u.unit_kind.as_str(),
        "status":if w.start_sample.is_some(){"aligned"}else{"unresolved"}})).collect();
        let complete = assembly.words.iter().all(|w| w.start_sample.is_some());
        let calibration = crate::alignment::calibration_for_language(&language);
        let alignment = json!({"raw_text":candidate.text,"text_sha256":sha256_hex(candidate.text.as_bytes()),
        "audio_anchor":{"start_sample":start,"end_sample":end},"audio_float32_sha256":hex::encode(std::mem::take(&mut self.audio_hash).finalize()),
        "schema_version":1,"text_source":"cohere","offset_unit":"unicode_codepoint","timing_kind":"ctc_emission_cell_envelope",
        "model_sha256":timing::MODEL,"provider_id":PROVIDER_ID,"provider_identity_sha256":crate::alignment::NATIVE_PROVIDER,
        "language":language,"normalization_policy_id":NORMALIZATION_POLICY_ID,"timing_unit_policy_id":TIMING_UNIT_POLICY_ID,
        "upstream_normalization_revision":UPSTREAM_NORMALIZATION_REVISION,"linguistic_word_claim":false,
        "frame_calibration_id":calibration,"score_calibration_id":calibration,
        "qualified_scope":if self.english_calibrated{"bounded_ami_english_coarse_envelopes"}else{"multilingual_functional_ctc_units"},
        "timing_accuracy_calibrated_languages":if self.english_calibrated{vec!["en"]}else{vec![]},
        "status":if complete{"aligned"}else{"partial"},"complete":complete,"words":words,
        "alignment_backend":"speakerdesk-rust-selected-context-ctc","assembly_policy":AUTHORITATIVE_POLICY});
        crate::alignment::attach(&mut candidate, alignment);
        Ok(Some(candidate))
    }
}

pub(super) fn refine<M: Models + ?Sized + 'static>(
    models: &M,
    original: &Segment,
    reader: &RecordingReader,
    vocabulary: Option<&crate::alignment::Vocabulary>,
) -> Result<Segment, LiveError> {
    let mut refinement = Refinement::new(original)?;
    loop {
        if let Some(candidate) = refinement.step(models, reader, vocabulary)? {
            return Ok(candidate);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn functional_context_accepts_retained_normalization_replays() {
        for language in ["ja", "ar"] {
            let alignment = crate::alignment::tests::functional_fixture(language);
            let start = alignment["audio_anchor"]["start_sample"].as_i64().unwrap();
            let end = alignment["audio_anchor"]["end_sample"].as_i64().unwrap();
            let part = TailPart {
                request: DecodeRequest {
                    utterance_id: "functional-replay".into(),
                    start_sample: start,
                    end_sample: end,
                    core_start_sample: start,
                    core_end_sample: end,
                    machine_revision: 2,
                    audio_revision: 3,
                    language_epoch: 4,
                },
                text: alignment["raw_text"].as_str().unwrap().into(),
                complete: true,
                alignment: Some(alignment),
            };
            let parsed = Context::parse(part)
                .expect("the existing adapter's complete aliases are valid context evidence");
            let mut composer = Composer::default();
            composer.append(&parsed, 0, parsed.raw.len()).unwrap();
            assert_eq!(composer.text, parsed.raw.iter().collect::<String>());
            assert!(
                composer
                    .words
                    .iter()
                    .all(|word| word.start_sample.is_some())
            );
        }
    }
}
