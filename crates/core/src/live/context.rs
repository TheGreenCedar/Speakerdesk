//! Bounded raw-text selection between adjacent, independently aligned contexts.
//! These cells come only from the local supplied-text adapter, which validates
//! model columns and raw-text normalization. Parse each result once; matching
//! uses acoustic cell identity and never writes normalized or matched spelling.
use super::*;
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
    raw: Vec<char>,
    cells: Vec<Cell>,
    words: Vec<Word>,
}
impl Context {
    fn parse(part: TailPart) -> Option<Self> {
        let result = part.alignment?;
        let request = part.request;
        if !part.complete
            || result["raw_text"] != part.text
            || result["text_sha256"] != sha256_hex(part.text.as_bytes())
            || result["audio_anchor"]
                != json!({"start_sample":request.start_sample,"end_sample":request.end_sample})
            || result["provider_identity_sha256"] != crate::alignment::NATIVE_PROVIDER
            || result["model_sha256"] != timing::MODEL
            || result["frame_calibration_id"] != timing::CALIBRATION
            || result["score_calibration_id"] != timing::CALIBRATION
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
            let start = c["start_sample"].as_i64()?;
            let end = c["end_sample"].as_i64()?;
            if a < previous_start
                || b < previous_end
                || b <= a
                || b > raw.len()
                || frame <= previous_frame
                || !score.is_finite()
                || !(-20. ..=0.).contains(&score)
                || c["status"] != "aligned"
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
            if c["text"] != " " {
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
        while a < b && part.raw[a].is_whitespace() {
            a += 1;
        }
        while a < b && part.raw[b - 1].is_whitespace() {
            b -= 1;
        }
        if a == b {
            return Ok(());
        }
        if !self.text.is_empty() {
            self.text.push(' ');
            self.characters += 1;
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
pub(super) fn refine<M: Models + ?Sized + 'static>(
    models: &M,
    original: &Segment,
    reader: &RecordingReader,
    vocabulary: Option<&crate::alignment::Vocabulary>,
) -> Result<Segment, LiveError> {
    let (start, end) = saved_bounds(original)?;
    let committed =
        u32::try_from(end).map_err(|_| LiveError::Message("Saved audio range is too large."))?;
    reader
        .require_samples(committed)
        .map_err(|e| LiveError::Detail(e.to_string()))?;
    let read = |a: i64, b: i64| {
        reader
            .read_samples(a as u32, b as u32, committed)
            .map_err(|e| LiveError::Detail(e.to_string()))
    };
    if end - start <= MAX_DECODE_SAMPLES {
        return refine_saved_passage(models, original, &read(start, end)?, vocabulary);
    }
    let anchor = AnchorRow {
        id: original
            .canonical_utterance_id
            .clone()
            .expect("checked canonical"),
        start_sample: start,
        end_sample: end,
        machine_revision: original.machine_revision.unwrap_or(0),
        audio_revision: original.audio_revision.unwrap_or(0),
        language_epoch: original.rest["language_epoch"].as_i64().unwrap_or(0),
    };
    let mode = original.rest["language_mode"]
        .as_str()
        .expect("checked mode");
    let processed = processed_catalog(original);
    let mut composer = Composer::default();
    let mut detection: Option<Value> = None;
    let mut probes = Vec::new();
    let mut audio_hash = Sha256::new();
    for request in authoritative_requests(&anchor)? {
        let pcm = read(request.start_sample, request.end_sample)?;
        let padding = (
            if request.start_sample > start {
                3200
            } else {
                0
            },
            if request.end_sample < end { 3200 } else { 0 },
        );
        let (text, language, decision) = crate::import::refine_context_source(
            models,
            &pcm,
            mode,
            request.start_sample,
            processed,
            Some(padding),
        )
        .map_err(|e| LiveError::Detail(e.to_string()))?;
        if language != "en" {
            return Err(LiveError::Message(
                "Long context accuracy currently requires confident English in every crop; previous words and audio are retained.",
            ));
        }
        if let Some(events) = decision["probes"].as_array() {
            probes.extend(events.iter().cloned());
        }
        if detection.as_ref().is_none_or(|prior| {
            decision["probability"].as_f64().unwrap_or(1.)
                < prior["probability"].as_f64().unwrap_or(1.)
        }) {
            detection = Some(decision);
        }
        let vocabulary=vocabulary.ok_or(LiveError::Message("Long context accuracy needs supplied-text timing; previous words and audio are retained."))?;
        let alignment = vocabulary.align(models, &pcm, &text, request.start_sample, &language)?;
        for sample in &pcm[(request.core_start_sample - request.start_sample) as usize
            ..(request.core_end_sample - request.start_sample) as usize]
        {
            audio_hash.update(sample.to_le_bytes());
        }
        composer
            .push(TailPart {
                request,
                text,
                complete: true,
                alignment: Some(alignment),
            })
            .map_err(LiveError::Message)?;
    }
    let assembly = composer.finish().map_err(LiveError::Message)?;
    let mut detection = detection.expect("nonempty context plan");
    if mode == languages::AUTOMATIC {
        detection["probes"] = json!(probes);
    }
    let mut candidate = saved_candidate(
        original,
        assembly.text.expect("complete assembly"),
        "en".into(),
        detection,
        end,
    )?;
    let mut utf16 = vec![0];
    for c in candidate.text.chars() {
        utf16.push(utf16.last().unwrap() + c.len_utf16());
    }
    let words:Vec<Value>=assembly.words.iter().map(|w|json!({"text":w.text,"start_char":w.start_char,"end_char":w.end_char,
        "start_utf16":utf16[w.start_char],"end_utf16":utf16[w.end_char],"start_sample":w.start_sample,"end_sample":w.end_sample,
        "status":if w.start_sample.is_some(){"aligned"}else{"unresolved"}})).collect();
    let complete = assembly.words.iter().all(|w| w.start_sample.is_some());
    let alignment = json!({"raw_text":candidate.text,"text_sha256":sha256_hex(candidate.text.as_bytes()),
        "audio_anchor":{"start_sample":start,"end_sample":end},"audio_float32_sha256":hex::encode(audio_hash.finalize()),
        "model_sha256":timing::MODEL,"provider_identity_sha256":crate::alignment::NATIVE_PROVIDER,
        "frame_calibration_id":timing::CALIBRATION,"score_calibration_id":timing::CALIBRATION,
        "status":if complete{"aligned"}else{"partial"},"complete":complete,"words":words,
        "alignment_backend":"speakerdesk-rust-selected-context-ctc","assembly_policy":AUTHORITATIVE_POLICY});
    crate::alignment::attach(&mut candidate, alignment);
    Ok(candidate)
}
