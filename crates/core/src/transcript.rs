//! Transcript validation helpers beyond [`crate::document`]. Owned by the core-text agent.

use std::collections::{BTreeMap, HashSet};

use serde_json::{Map, Number, Value};

use crate::document::{Document, Segment};
use crate::{limits, messages};

/// Validation failure. `message` is a verbatim product string.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
#[error("{message}")]
pub struct TranscriptError {
    pub message: String,
}

impl TranscriptError {
    fn new(message: impl Into<String>) -> Self {
        Self {
            message: message.into(),
        }
    }
}

/// Download exceeded [`limits::MAX_EXPORT_BYTES`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, thiserror::Error)]
#[error("{}", messages::EXPORT_TOO_LARGE)]
pub struct ExportTooLarge;

/// `transcript.validate`: copy, check, normalize, and sort a transcript object.
pub fn validate(document: &Value, duration: f64) -> Result<Document, TranscriptError> {
    let Some(source) = document.as_object() else {
        return Err(TranscriptError::new(messages::TRANSCRIPT_OBJECT));
    };
    let speakers = match source.get("speakers").and_then(Value::as_object) {
        Some(speakers) => validated_speakers(speakers)?,
        None => return Err(TranscriptError::new(messages::SPEAKER_COUNT)),
    };
    let segments = match source.get("segments").and_then(Value::as_array) {
        Some(segments) => validated_segments(segments, &speakers, duration)?,
        None => return Err(TranscriptError::new(messages::SEGMENT_LIST)),
    };
    Ok(finish_document(source, speakers, segments))
}

/// `transcript.timestamp`. `separator` is `.` for txt/vtt and `,` for srt.
pub fn timestamp(seconds: f64, separator: &str) -> String {
    let mut millis = round_half_even(seconds * 1000.0);
    let (hours, rest) = divmod(millis, 3_600_000);
    millis = rest;
    let (minutes, rest) = divmod(millis, 60_000);
    millis = rest;
    let (secs, millis) = divmod(millis, 1_000);
    format!("{hours:02}:{minutes:02}:{secs:02}{separator}{millis:03}")
}

/// Join text parts, refusing a UTF-8 total above 16 MiB.
pub fn bounded_text<I, S>(parts: I) -> Result<String, ExportTooLarge>
where
    I: IntoIterator<Item = S>,
    S: AsRef<str>,
{
    let mut out = String::new();
    let mut size = 0usize;
    for part in parts {
        let part = part.as_ref();
        if part.len() > limits::MAX_EXPORT_BYTES.saturating_sub(size) {
            return Err(ExportTooLarge);
        }
        size += part.len();
        out.push_str(part);
    }
    Ok(out)
}

pub(crate) fn number_as_i64(number: &Number) -> Option<i64> {
    if let Some(value) = number.as_i64() {
        return Some(value);
    }
    let value = number.as_f64()?;
    if !value.is_finite() || value.fract() != 0.0 {
        return None;
    }
    if value < i64::MIN as f64 || value > i64::MAX as f64 {
        return None;
    }
    Some(value as i64)
}

pub(crate) fn json_truthy(value: Option<&Value>) -> bool {
    match value {
        None | Some(Value::Null) => false,
        Some(Value::Bool(flag)) => *flag,
        Some(Value::Number(number)) => number.as_f64().is_some_and(|number| number != 0.0),
        Some(Value::String(text)) => !text.is_empty(),
        Some(Value::Array(items)) => !items.is_empty(),
        Some(Value::Object(map)) => !map.is_empty(),
    }
}

fn validated_speakers(
    speakers: &Map<String, Value>,
) -> Result<BTreeMap<String, String>, TranscriptError> {
    if !(1..=limits::MAX_SPEAKERS).contains(&speakers.len()) {
        return Err(TranscriptError::new(messages::SPEAKER_COUNT));
    }
    let mut names = BTreeMap::new();
    for (key, value) in speakers {
        let Some(name) = value.as_str() else {
            return Err(TranscriptError::new(messages::SPEAKER_NAMES));
        };
        if key.chars().count() > limits::MAX_SPEAKER_KEY_CHARS
            || name.chars().count() > limits::MAX_SPEAKER_NAME_CHARS
            || name.trim().is_empty()
        {
            return Err(TranscriptError::new(messages::SPEAKER_NAMES));
        }
        names.insert(key.clone(), name.trim().to_string());
    }
    Ok(names)
}

fn validated_segments(
    segments: &[Value],
    speakers: &BTreeMap<String, String>,
    duration: f64,
) -> Result<Vec<Segment>, TranscriptError> {
    if segments.len() > limits::MAX_SEGMENTS {
        return Err(TranscriptError::new(messages::SEGMENT_LIST));
    }
    let mut built = Vec::with_capacity(segments.len());
    let mut seen = HashSet::new();
    for (index, value) in segments.iter().enumerate() {
        let Some(object) = value.as_object() else {
            return Err(TranscriptError::new(messages::SEGMENT_OBJECT));
        };
        let start = object
            .get("start")
            .and_then(value_as_f64)
            .ok_or_else(|| TranscriptError::new(messages::segment_times(index + 1)))?;
        let end = object
            .get("end")
            .and_then(value_as_f64)
            .ok_or_else(|| TranscriptError::new(messages::segment_times(index + 1)))?;
        if !start.is_finite()
            || !end.is_finite()
            || !(0.0 <= start && start < end && end <= duration + limits::TIME_EPSILON_SECONDS)
        {
            return Err(TranscriptError::new(messages::segment_bounds(index + 1)));
        }
        let speaker_ok = object
            .get("speaker")
            .and_then(Value::as_str)
            .is_some_and(|speaker| speakers.contains_key(speaker));
        let text_ok = object
            .get("text")
            .and_then(Value::as_str)
            .is_some_and(|text| text.chars().count() <= limits::MAX_SEGMENT_TEXT_CHARS);
        if !speaker_ok || !text_ok {
            return Err(TranscriptError::new(messages::segment_speaker_or_text(
                index + 1,
            )));
        }
        let sid = match object.get("id") {
            None => format!("seg-{index}"),
            Some(Value::String(id)) => id.clone(),
            Some(_) => return Err(TranscriptError::new(messages::SEGMENT_IDS)),
        };
        if sid.chars().count() > limits::MAX_SEGMENT_ID_CHARS || seen.contains(&sid) {
            return Err(TranscriptError::new(messages::SEGMENT_IDS));
        }
        seen.insert(sid.clone());
        built.push(segment_from_map(object.clone(), sid, start, end));
    }
    sort_segments(&mut built);
    Ok(built)
}

pub(crate) fn sort_segments(segments: &mut [Segment]) {
    segments.sort_by(|left, right| {
        left.start
            .total_cmp(&right.start)
            .then(left.end.total_cmp(&right.end))
            .then_with(|| capture_source_rank(left).cmp(&capture_source_rank(right)))
            .then_with(|| source_id_tie(left).cmp(source_id_tie(right)))
            .then_with(|| {
                left.separation
                    .as_ref()
                    .map(|v| (&v.episode_id, v.channel))
                    .cmp(
                        &right
                            .separation
                            .as_ref()
                            .map(|v| (&v.episode_id, v.channel)),
                    )
            })
    });
}

fn finish_document(
    source: &Map<String, Value>,
    speakers: BTreeMap<String, String>,
    segments: Vec<Segment>,
) -> Document {
    let mut rest = source.clone();
    rest.remove("speakers");
    rest.remove("segments");
    rest.remove("schema_version");
    let provenance = rest.remove("provenance");
    let warnings = take_string_array(&mut rest, "warnings");
    let diarization = take_value_array(&mut rest, "diarization");
    let edited_at = take_f64(&mut rest, "edited_at");
    Document {
        schema_version: 1,
        speakers,
        segments,
        provenance,
        warnings,
        diarization,
        edited_at,
        rest,
    }
}

fn segment_from_map(mut fields: Map<String, Value>, id: String, start: f64, end: f64) -> Segment {
    fields.remove("id");
    fields.remove("start");
    fields.remove("end");
    let speaker = take_string(&mut fields, "speaker").unwrap_or_default();
    let text = take_string(&mut fields, "text").unwrap_or_default();
    let language = take_string_opt(&mut fields, "language");
    let language_detection = fields.remove("language_detection");
    let transcription_review = fields.remove("transcription_review");
    let reading_turns = fields.remove("reading_turns");
    let reading_turn_provenance = fields.remove("reading_turn_provenance");
    let alignment = fields.remove("alignment");
    let speaker_activity = fields.remove("speaker_activity");
    let acoustic_evidence = fields.remove("acoustic_evidence");
    let refinement_state = take_string_opt(&mut fields, "refinement_state");
    let machine_revision = take_i64(&mut fields, "machine_revision");
    let protected_fields = take_string_list(&mut fields, "protected_fields");
    let review_resolution = take_string_opt(&mut fields, "review_resolution");
    let retry_candidate = fields.remove("retry_candidate");
    let canonical_utterance_id = take_string_opt(&mut fields, "canonical_utterance_id");
    let audio_revision = take_i64(&mut fields, "audio_revision");
    let review = take_bool(&mut fields, "review");
    let capture_source = fields.remove("capture_source");
    Segment {
        id,
        start,
        end,
        speaker,
        text,
        language,
        language_detection,
        transcription_review,
        reading_turns,
        reading_turn_provenance,
        alignment,
        speaker_activity,
        acoustic_evidence,
        refinement_state,
        machine_revision,
        protected_fields,
        review_resolution,
        retry_candidate,
        canonical_utterance_id,
        audio_revision,
        review,
        capture_source,
        separation_candidate: fields
            .remove("separation_candidate")
            .and_then(|v| serde_json::from_value(v).ok()),
        separation: fields
            .remove("separation")
            .and_then(|v| serde_json::from_value(v).ok()),
        rest: fields,
    }
}

fn value_as_f64(value: &Value) -> Option<f64> {
    match value {
        Value::Number(number) => number.as_f64(),
        Value::String(text) => text.trim().parse().ok(),
        _ => None,
    }
}

fn capture_source_rank(segment: &Segment) -> String {
    let Some(Value::Object(source)) = &segment.capture_source else {
        return String::new();
    };
    match source.get("source_id").and_then(Value::as_str) {
        Some(id @ ("microphone_clean" | "system")) => id.to_string(),
        _ => String::new(),
    }
}

fn source_id_tie(segment: &Segment) -> &str {
    if capture_source_rank(segment).is_empty() {
        ""
    } else {
        segment.id.as_str()
    }
}

fn take_string(fields: &mut Map<String, Value>, key: &str) -> Option<String> {
    match fields.remove(key) {
        Some(Value::String(text)) => Some(text),
        Some(other) => {
            fields.insert(key.to_string(), other);
            None
        }
        None => None,
    }
}

fn take_string_opt(fields: &mut Map<String, Value>, key: &str) -> Option<String> {
    take_string(fields, key)
}

fn take_i64(fields: &mut Map<String, Value>, key: &str) -> Option<i64> {
    match fields.remove(key) {
        Some(Value::Number(number)) => match number_as_i64(&number) {
            Some(value) => Some(value),
            None => {
                fields.insert(key.to_string(), Value::Number(number));
                None
            }
        },
        Some(other) => {
            fields.insert(key.to_string(), other);
            None
        }
        None => None,
    }
}

fn take_bool(fields: &mut Map<String, Value>, key: &str) -> Option<bool> {
    match fields.remove(key) {
        Some(Value::Bool(flag)) => Some(flag),
        Some(other) => {
            fields.insert(key.to_string(), other);
            None
        }
        None => None,
    }
}

fn take_string_list(fields: &mut Map<String, Value>, key: &str) -> Option<Vec<String>> {
    match fields.remove(key) {
        Some(Value::Array(items)) if items.iter().all(|item| item.as_str().is_some()) => Some(
            items
                .into_iter()
                .filter_map(|item| item.as_str().map(str::to_string))
                .collect(),
        ),
        Some(other) => {
            fields.insert(key.to_string(), other);
            None
        }
        None => None,
    }
}

fn take_string_array(fields: &mut Map<String, Value>, key: &str) -> Vec<String> {
    take_string_list(fields, key).unwrap_or_default()
}

fn take_value_array(fields: &mut Map<String, Value>, key: &str) -> Vec<Value> {
    match fields.remove(key) {
        Some(Value::Array(items)) => items,
        Some(other) => {
            fields.insert(key.to_string(), other);
            Vec::new()
        }
        None => Vec::new(),
    }
}

fn take_f64(fields: &mut Map<String, Value>, key: &str) -> Option<f64> {
    match fields.remove(key) {
        Some(Value::Number(number)) => match number.as_f64() {
            Some(value) => Some(value),
            None => {
                fields.insert(key.to_string(), Value::Number(number));
                None
            }
        },
        Some(other) => {
            fields.insert(key.to_string(), other);
            None
        }
        None => None,
    }
}

/// Python 3 `round` of a finite value, with nonfinite input mapped to zero.
fn round_half_even(value: f64) -> i64 {
    if !value.is_finite() {
        return 0;
    }
    value.round_ties_even() as i64
}

fn divmod(value: i64, divisor: i64) -> (i64, i64) {
    (value.div_euclid(divisor), value.rem_euclid(divisor))
}

#[cfg(test)]
pub(crate) fn bare_segment(id: &str, start: f64, end: f64, speaker: &str, text: &str) -> Segment {
    Segment {
        id: id.to_string(),
        start,
        end,
        speaker: speaker.to_string(),
        text: text.to_string(),
        language: None,
        language_detection: None,
        transcription_review: None,
        reading_turns: None,
        reading_turn_provenance: None,
        alignment: None,
        speaker_activity: None,
        acoustic_evidence: None,
        refinement_state: None,
        machine_revision: None,
        protected_fields: None,
        review_resolution: None,
        retry_candidate: None,
        canonical_utterance_id: None,
        audio_revision: None,
        review: None,
        capture_source: None,
        separation_candidate: None,
        separation: None,
        rest: Map::new(),
    }
}

#[cfg(test)]
pub(crate) fn bare_document(speakers: &[(&str, &str)], segments: Vec<Segment>) -> Document {
    Document {
        schema_version: 1,
        speakers: speakers
            .iter()
            .map(|(key, name)| ((*key).to_string(), (*name).to_string()))
            .collect(),
        segments,
        provenance: None,
        warnings: Vec::new(),
        diarization: Vec::new(),
        edited_at: None,
        rest: Map::new(),
    }
}

#[cfg(test)]
mod tests {
    use super::{bare_document, bare_segment, timestamp, validate};
    use crate::{limits, messages};
    use serde_json::{Map, Value, json};

    fn speaker_map(count: usize) -> Map<String, Value> {
        (0..count)
            .map(|index| (format!("s{index}"), Value::String(format!("Name {index}"))))
            .collect()
    }

    fn document(speakers: Map<String, Value>, segments: Vec<Value>) -> Value {
        json!({
            "speakers": speakers,
            "segments": segments,
        })
    }

    fn one_speaker() -> Map<String, Value> {
        Map::from_iter([("a".to_string(), Value::String("Ada".to_string()))])
    }

    fn segment(id: &str, start: f64, end: f64) -> Value {
        json!({
            "id": id,
            "start": start,
            "end": end,
            "speaker": "a",
            "text": "Hi",
        })
    }

    #[test]
    fn validate_rejects_speaker_count() {
        let err = validate(&json!({"segments": []}), 1.0).unwrap_err();
        assert_eq!(err.message, messages::SPEAKER_COUNT);
        let err = validate(&document(Map::new(), Vec::new()), 1.0).unwrap_err();
        assert_eq!(err.message, messages::SPEAKER_COUNT);
        let err = validate(&document(speaker_map(65), Vec::new()), 1.0).unwrap_err();
        assert_eq!(err.message, messages::SPEAKER_COUNT);
        let err = validate(&json!([]), 1.0).unwrap_err();
        assert_eq!(err.message, messages::TRANSCRIPT_OBJECT);
        assert!(validate(&document(speaker_map(1), Vec::new()), 1.0).is_ok());
        assert!(validate(&document(speaker_map(64), Vec::new()), 1.0).is_ok());
    }

    #[test]
    fn validate_rejects_speaker_name_length() {
        let mut speakers = one_speaker();
        speakers.insert("a".into(), Value::String("n".repeat(101)));
        let err = validate(&document(speakers, Vec::new()), 1.0).unwrap_err();
        assert_eq!(err.message, messages::SPEAKER_NAMES);

        let mut speakers = one_speaker();
        speakers.insert("a".into(), Value::String("   ".into()));
        let err = validate(&document(speakers, Vec::new()), 1.0).unwrap_err();
        assert_eq!(err.message, messages::SPEAKER_NAMES);

        let mut speakers = one_speaker();
        speakers.insert("a".into(), Value::String(String::new()));
        let err = validate(&document(speakers, Vec::new()), 1.0).unwrap_err();
        assert_eq!(err.message, messages::SPEAKER_NAMES);

        let speakers = Map::from_iter([(
            "k".repeat(limits::MAX_SPEAKER_KEY_CHARS + 1),
            Value::String("Ada".into()),
        )]);
        let err = validate(&document(speakers, Vec::new()), 1.0).unwrap_err();
        assert_eq!(err.message, messages::SPEAKER_NAMES);

        let mut speakers = one_speaker();
        speakers.insert("a".into(), json!(1));
        let err = validate(&document(speakers, Vec::new()), 1.0).unwrap_err();
        assert_eq!(err.message, messages::SPEAKER_NAMES);

        let mut speakers = one_speaker();
        speakers.insert("a".into(), Value::String("n".repeat(100)));
        assert!(validate(&document(speakers, Vec::new()), 1.0).is_ok());
    }

    #[test]
    fn validate_rejects_segment_bounds() {
        let err =
            validate(&document(one_speaker(), vec![segment("1", 1.0, 0.5)]), 10.0).unwrap_err();
        assert_eq!(err.message, messages::segment_bounds(1));

        let err = validate(
            &document(one_speaker(), vec![segment("1", 0.0, 10.002)]),
            10.0,
        )
        .unwrap_err();
        assert_eq!(err.message, messages::segment_bounds(1));

        let err = validate(
            &document(one_speaker(), vec![segment("1", -0.01, 1.0)]),
            10.0,
        )
        .unwrap_err();
        assert_eq!(err.message, messages::segment_bounds(1));

        let bad = json!({"id": "1", "start": "nope", "end": 1, "speaker": "a", "text": "Hi"});
        let err = validate(&document(one_speaker(), vec![bad]), 10.0).unwrap_err();
        assert_eq!(err.message, messages::segment_times(1));

        let second = json!({"id": "2", "start": 3, "end": 2, "speaker": "a", "text": "Hi"});
        let err = validate(
            &document(one_speaker(), vec![segment("1", 0.0, 1.0), second]),
            10.0,
        )
        .unwrap_err();
        assert_eq!(err.message, messages::segment_bounds(2));

        let edge = 10.0 + limits::TIME_EPSILON_SECONDS;
        let ok = validate(
            &document(one_speaker(), vec![segment("1", 0.0, edge)]),
            10.0,
        )
        .unwrap();
        assert_eq!(ok.segments[0].end, edge);

        let missing = validate(&json!({"speakers": {"a": "Ada"}}), 1.0).unwrap_err();
        assert_eq!(missing.message, messages::SEGMENT_LIST);
        let object = validate(&document(one_speaker(), vec![json!(1)]), 1.0).unwrap_err();
        assert_eq!(object.message, messages::SEGMENT_OBJECT);
    }

    #[test]
    fn validate_rejects_duplicate_and_non_string_ids() {
        let err = validate(
            &document(
                one_speaker(),
                vec![segment("same", 0.0, 1.0), segment("same", 1.0, 2.0)],
            ),
            10.0,
        )
        .unwrap_err();
        assert_eq!(err.message, messages::SEGMENT_IDS);

        let mut long = segment("x", 0.0, 1.0);
        long["id"] = Value::String("i".repeat(limits::MAX_SEGMENT_ID_CHARS + 1));
        let err = validate(&document(one_speaker(), vec![long]), 10.0).unwrap_err();
        assert_eq!(err.message, messages::SEGMENT_IDS);

        let mut numeric = segment("x", 0.0, 1.0);
        numeric["id"] = json!(7);
        let err = validate(&document(one_speaker(), vec![numeric]), 10.0).unwrap_err();
        assert_eq!(err.message, messages::SEGMENT_IDS);
    }

    #[test]
    fn validate_strips_names_sorts_segments_and_assigns_ids() {
        let speakers = Map::from_iter([
            ("b".to_string(), Value::String(" Bea ".into())),
            ("a".to_string(), Value::String("Ada".into())),
        ]);
        let segments = vec![
            json!({
                "id": "z",
                "start": 2,
                "end": 3,
                "speaker": "a",
                "text": "Second",
                "capture_source": {"source_id": "system"}
            }),
            json!({
                "id": "a",
                "start": 0,
                "end": 1,
                "speaker": "b",
                "text": "First"
            }),
            json!({
                "start": 0,
                "end": 1,
                "speaker": "a",
                "text": "Same",
                "capture_source": {"source_id": "microphone_clean", "x": 1}
            }),
        ];
        let result = validate(&document(speakers, segments), 10.0).unwrap();
        assert_eq!(result.schema_version, 1);
        assert_eq!(result.speakers.get("b").map(String::as_str), Some("Bea"));
        assert_eq!(result.speakers.get("a").map(String::as_str), Some("Ada"));
        assert_eq!(
            result
                .segments
                .iter()
                .map(|segment| segment.id.as_str())
                .collect::<Vec<_>>(),
            vec!["a", "seg-2", "z"]
        );
        assert_eq!(result.segments[0].text, "First");
        assert_eq!(result.segments[1].text, "Same");
        assert_eq!(result.segments[1].speaker, "a");
    }

    #[test]
    fn timestamp_matches_python_round_half_even() {
        for (seconds, separator, expected) in [
            (0.0, ".", "00:00:00.000"),
            (1.5, ",", "00:00:01,500"),
            (0.0005, ".", "00:00:00.000"),
            (0.0015, ".", "00:00:00.002"),
            (-0.0005, ".", "00:00:00.000"),
            (-0.0015, ".", "-1:59:59.998"),
            (-0.0025, ",", "-1:59:59,998"),
            (59.9995, ".", "00:01:00.000"),
            (3599.9995, ",", "01:00:00,000"),
            (3661.25, ".", "01:01:01.250"),
            (f64::NAN, ".", "00:00:00.000"),
            (f64::INFINITY, ",", "00:00:00,000"),
            (f64::NEG_INFINITY, ".", "00:00:00.000"),
            (f64::MAX, ".", "00:00:00.000"),
        ] {
            assert_eq!(timestamp(seconds, separator), expected);
        }
    }

    #[test]
    fn helpers_construct_a_document() {
        let document = bare_document(
            &[("speaker_0", "Ada")],
            vec![bare_segment("one", 0.0, 1.0, "speaker_0", "Hello")],
        );
        assert_eq!(document.segments[0].text, "Hello");
    }
}
