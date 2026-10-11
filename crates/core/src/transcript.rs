//! Port of `speakerdesk/transcript.py` — canonical transcript validation and
//! interoperable exports; no word-time inference.
//!
//! `validate` accepts the raw client `Value` (v0.6.6 coerces numeric
//! strings), validates it, and returns the typed `Document` — unknown keys
//! are ignored at that boundary. `export` runs on the typed document.
//! `validated_turns` still reads raw `Value`s: it is the v0.6.6 provenance
//! gate and runs in `store::legacy` only, before the ledger strip removes
//! `reading_turn_provenance`.

use serde_json::{Map, Value, json};
use sha2::{Digest, Sha256};

use crate::error::Error;
use crate::model::{Attribution, CaptureSourceId, Document, Segment};
use crate::review::export_text;

/// `transcript.LANGUAGES` — the supported meeting-language vocabulary.
pub const LANGUAGES: &[(&str, &str)] = &[
    ("en", "English"),
    ("de", "German"),
    ("fr", "French"),
    ("it", "Italian"),
    ("es", "Spanish"),
    ("pt", "Portuguese"),
    ("el", "Greek"),
    ("nl", "Dutch"),
    ("pl", "Polish"),
    ("vi", "Vietnamese"),
    ("zh", "Chinese"),
    ("ar", "Arabic"),
    ("ja", "Japanese"),
    ("ko", "Korean"),
];

/// 16 MiB export cap — downloads stay bounded even for long recordings.
pub const MAX_EXPORT_BYTES: usize = 16 * 1024 * 1024;

/// The serialized download exceeds the local export limit.
#[derive(Debug, thiserror::Error)]
#[error("Export exceeds the 16 MiB download limit. The saved meeting is unchanged.")]
pub struct ExportTooLarge;

/// Python `round()` — round half to even.
fn round_half_even(x: f64) -> i64 {
    let floor = x.floor();
    let frac = x - floor;
    let base = floor as i64;
    if frac < 0.5 {
        base
    } else if frac > 0.5 {
        base + 1
    } else if base % 2 == 0 {
        base
    } else {
        base + 1
    }
}

/// Python `float(value)` coercion: numbers, bools, numeric strings.
fn python_float(value: &Value) -> Option<f64> {
    match value {
        Value::Number(n) => n.as_f64(),
        Value::Bool(b) => Some(if *b { 1.0 } else { 0.0 }),
        Value::String(s) => s.trim().parse::<f64>().ok(),
        _ => None,
    }
}

/// Python `len()` on str — Unicode code points, not bytes.
fn str_len(s: &str) -> usize {
    s.chars().count()
}

/// Port of `transcript.validate`. Validates the raw client document —
/// coercing numeric strings exactly like Python's `float()` — and returns
/// the normalized typed `Document` (speakers trimmed, `schema_version` 1,
/// segments sorted). Unknown client keys are dropped at the typed
/// boundary, matching the product-field model.
pub fn validate(document: &Value, duration: f64) -> Result<Document, Error> {
    let Some(obj) = document.as_object() else {
        return Err(Error::bad_request("Transcript must be a JSON object."));
    };
    let speakers = obj.get("speakers").cloned().unwrap_or(Value::Null);
    let segments = obj.get("segments").cloned().unwrap_or(Value::Null);
    let speakers_map = match &speakers {
        Value::Object(m) if (1..=64).contains(&m.len()) => m.clone(),
        _ => return Err(Error::bad_request("Supply 1–64 speaker names.")),
    };
    for (k, v) in &speakers_map {
        let bad = str_len(k) > 80
            || !matches!(v, Value::String(s) if !s.trim().is_empty() && str_len(s) <= 100);
        if bad {
            return Err(Error::bad_request(
                "Speaker names must be nonempty text, up to 100 characters.",
            ));
        }
    }
    let segment_list = match &segments {
        Value::Array(list) if list.len() <= 20_000 => list.clone(),
        _ => {
            return Err(Error::bad_request(
                "Supply a segment list with at most 20,000 entries.",
            ));
        }
    };

    let mut identifiers: Vec<String> = Vec::new();
    let mut normalized: Vec<Value> = Vec::with_capacity(segment_list.len());
    for (i, row) in segment_list.iter().enumerate() {
        if !row.is_object() {
            return Err(Error::bad_request("Each segment must be an object."));
        }
        let mut s = row.clone();
        let (start, end) = match (s.get("start"), s.get("end")) {
            (Some(a), Some(b)) => match (python_float(a), python_float(b)) {
                (Some(a), Some(b)) => (a, b),
                _ => {
                    return Err(Error::bad_request(format!(
                        "Segment {}: supply numeric start and end times.",
                        i + 1,
                    )));
                }
            },
            _ => {
                return Err(Error::bad_request(format!(
                    "Segment {}: supply numeric start and end times.",
                    i + 1,
                )));
            }
        };
        if !start.is_finite()
            || !end.is_finite()
            || !(0.0 <= start && start < end && end <= duration + 0.001)
        {
            return Err(Error::bad_request(format!(
                "Segment {}: times must fall within the recording and end after start.",
                i + 1,
            )));
        }
        let speaker_ok = match s.get("speaker") {
            Some(Value::String(speaker)) => speakers_map.contains_key(speaker),
            _ => false,
        };
        let text_ok = matches!(s.get("text"), Some(Value::String(t)) if str_len(t) <= 20_000);
        if !speaker_ok || !text_ok {
            return Err(Error::bad_request(format!(
                "Segment {}: invalid speaker or text.",
                i + 1,
            )));
        }
        let sid_value = match s.get("id") {
            None => json!(format!("seg-{i}")),
            Some(v) => v.clone(),
        };
        let sid = match &sid_value {
            Value::String(s) if str_len(s) <= 100 => s.clone(),
            _ => {
                return Err(Error::bad_request(
                    "Segment IDs must be unique strings.".to_string(),
                ));
            }
        };
        if identifiers.iter().any(|id| id == &sid) {
            return Err(Error::bad_request(
                "Segment IDs must be unique strings.".to_string(),
            ));
        }
        identifiers.push(sid.clone());
        let s_obj = s.as_object_mut().unwrap();
        s_obj.insert("id".to_string(), Value::String(sid));
        s_obj.insert("start".to_string(), json!(start));
        s_obj.insert("end".to_string(), json!(end));
        normalized.push(s);
    }

    // `order` sort key — (start, end, source_id, id); source_id only
    // participates for recognized source-bound rows.
    normalized.sort_by(|a, b| {
        let (ka, kb) = (order_key(a), order_key(b));
        ka.0.partial_cmp(&kb.0)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| ka.1.partial_cmp(&kb.1).unwrap_or(std::cmp::Ordering::Equal))
            .then_with(|| ka.2.cmp(&kb.2))
            .then_with(|| ka.3.cmp(&kb.3))
    });

    let stripped: Map<String, Value> = speakers_map
        .iter()
        .map(|(k, v)| (k.clone(), json!(v.as_str().unwrap().trim())))
        .collect();
    let mut result = document.clone();
    let out = result.as_object_mut().unwrap();
    out.insert("segments".to_string(), Value::Array(normalized));
    out.insert("speakers".to_string(), Value::Object(stripped));
    out.insert("schema_version".to_string(), json!(1));
    // Unknown keys are ignored at the typed boundary; a deserialization
    // failure here means the product model is missing a kept field.
    let document: Document = serde_json::from_value(result)
        .map_err(|e| Error::bad_request(format!("Transcript could not be represented: {e}")))?;
    crate::model::check_opaque_bounds_document(&document).map_err(Error::bad_request)?;
    Ok(document)
}

/// `transcript`'s `order` sort key — (start, end, source_id, id). `source_id`
/// is the raw string (`''` sorts before `microphone_clean`/`system`); the id
/// participates only for source-bound rows.
fn order_key(segment: &Value) -> (f64, f64, String, String) {
    let source_id = segment
        .get("capture_source")
        .filter(|c| c.is_object())
        .and_then(|c| c.get("source_id"))
        .and_then(Value::as_str)
        .unwrap_or("");
    let recognized = matches!(source_id, "microphone_clean" | "system");
    (
        segment.get("start").and_then(Value::as_f64).unwrap_or(0.0),
        segment.get("end").and_then(Value::as_f64).unwrap_or(0.0),
        if recognized {
            source_id.to_string()
        } else {
            String::new()
        },
        if recognized {
            segment
                .get("id")
                .and_then(Value::as_str)
                .unwrap_or("")
                .to_string()
        } else {
            String::new()
        },
    )
}

/// Typed equivalent of `order_key` for already-validated segments.
fn typed_order_key(segment: &Segment) -> (f64, f64, String, String) {
    let source_id = segment
        .capture_source
        .as_ref()
        .map(|c| match c.source_id {
            CaptureSourceId::MicrophoneClean => "microphone_clean",
            CaptureSourceId::System => "system",
        })
        .unwrap_or("");
    (
        segment.start,
        segment.end,
        source_id.to_string(),
        if source_id.is_empty() {
            String::new()
        } else {
            segment.id.clone()
        },
    )
}

/// Sort segments in place with the `order` key — used by paths that build
/// documents without a raw `Value` round-trip.
pub fn sort_segments(segments: &mut [Segment]) {
    segments.sort_by(|a, b| {
        let (ka, kb) = (typed_order_key(a), typed_order_key(b));
        ka.0.partial_cmp(&kb.0)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| ka.1.partial_cmp(&kb.1).unwrap_or(std::cmp::Ordering::Equal))
            .then_with(|| ka.2.cmp(&kb.2))
            .then_with(|| ka.3.cmp(&kb.3))
    });
}

/// `transcript.timestamp` — `HH:MM:SS.mmm` (`,` for SRT).
pub fn timestamp(seconds: f64, separator: char) -> String {
    let mut ms = round_half_even(seconds * 1000.0);
    let hours = ms.div_euclid(3_600_000);
    ms %= 3_600_000;
    let minutes = ms.div_euclid(60_000);
    ms %= 60_000;
    let secs = ms.div_euclid(1000);
    ms %= 1000;
    format!("{hours:02}:{minutes:02}:{secs:02}{separator}{ms:03}")
}

/// `transcript.bounded_text` — join parts until the UTF-8 byte cap is hit.
fn bounded_text(parts: impl Iterator<Item = String>, cap: usize) -> Result<String, ExportTooLarge> {
    let mut out = String::new();
    let mut size = 0usize;
    for part in parts {
        size += part.len();
        if size > cap {
            return Err(ExportTooLarge);
        }
        out.push_str(&part);
    }
    Ok(out)
}

/// sha256 hex of UTF-8 text (Python `hashlib.sha256(text.encode()).hexdigest()`).
pub fn sha256_hex_text(text: &str) -> String {
    hex::encode(Sha256::digest(text.as_bytes()))
}

/// `json.dumps(value, sort_keys=True, separators=(',',':'))` — compact,
/// key-sorted JSON. serde_json's `Map` is BTreeMap-backed (sorted) and its
/// serializer is already compact and non-ASCII-raw.
fn json_sorted(value: &Value) -> String {
    serde_json::to_string(value).unwrap_or_default()
}

/// `json.dumps(..., ensure_ascii=True)` — Python's default escapes every
/// non-ASCII code point as `\uXXXX` (surrogate pairs beyond the BMP).
fn json_sorted_ascii(value: &Value) -> String {
    let mut out = String::new();
    fn write(out: &mut String, value: &Value) {
        match value {
            Value::Null => out.push_str("null"),
            Value::Bool(b) => out.push_str(if *b { "true" } else { "false" }),
            Value::Number(n) => out.push_str(&n.to_string()),
            Value::String(s) => {
                out.push('"');
                for ch in s.chars() {
                    match ch {
                        '"' => out.push_str("\\\""),
                        '\\' => out.push_str("\\\\"),
                        '\n' => out.push_str("\\n"),
                        '\r' => out.push_str("\\r"),
                        '\t' => out.push_str("\\t"),
                        '\u{08}' => out.push_str("\\b"),
                        '\u{0c}' => out.push_str("\\f"),
                        c if (c as u32) < 0x20 => out.push_str(&format!("\\u{:04x}", c as u32)),
                        c if (c as u32) < 0x80 => out.push(c),
                        c => {
                            // Python ensure_ascii: \uXXXX, surrogate pairs >BMP.
                            let mut buf = [0u16; 2];
                            for unit in c.encode_utf16(&mut buf) {
                                out.push_str(&format!("\\u{:04x}", unit));
                            }
                        }
                    }
                }
                out.push('"');
            }
            Value::Array(list) => {
                out.push('[');
                for (i, item) in list.iter().enumerate() {
                    if i > 0 {
                        out.push(',');
                    }
                    write(out, item);
                }
                out.push(']');
            }
            Value::Object(map) => {
                out.push('{');
                for (i, (k, v)) in map.iter().enumerate() {
                    if i > 0 {
                        out.push(',');
                    }
                    write(out, &Value::String(k.clone()));
                    out.push(':');
                    write(out, v);
                }
                out.push('}');
            }
        }
    }
    write(&mut out, value);
    out
}

/// sha256 hex of `json.dumps(value, sort_keys=True, separators=(',',':'))`
/// (Python's default `ensure_ascii=True`).
fn sha256_hex_json_ascii(value: &Value) -> String {
    hex::encode(Sha256::digest(json_sorted_ascii(value).as_bytes()))
}

/// sha256 hex of `json.dumps(..., sort_keys=True, separators=(',',':'),
/// ensure_ascii=False)`.
fn sha256_hex_json_utf8(value: &Value) -> String {
    hex::encode(Sha256::digest(json_sorted(value).as_bytes()))
}

fn get<'a>(value: &'a Value, key: &str) -> Option<&'a Value> {
    value.get(key)
}

/// Python truthiness for JSON values.
fn truthy(value: Option<&Value>) -> bool {
    match value {
        None | Some(Value::Null) => false,
        Some(Value::Bool(b)) => *b,
        Some(Value::Number(n)) => n.as_f64() != Some(0.0),
        Some(Value::String(s)) => !s.is_empty(),
        Some(Value::Array(a)) => !a.is_empty(),
        Some(Value::Object(o)) => !o.is_empty(),
    }
}

fn eq(a: Option<&Value>, b: Option<&Value>) -> bool {
    a.cloned().unwrap_or(Value::Null) == b.cloned().unwrap_or(Value::Null)
}

// `reading_turns` qualification constants.
const CALIBRATION: &str = "ami-english-coarse-mlx-f32-v1-5ab4e661e62f-38ff6a225e75";
const MODEL: &str = "38ff6a225e75caa6e19c35a9b5823015e2550451bd0a394a10f6cda87f42058d";
const FUNCTIONAL_POLICY_ID: &str = "omni-ctc-mlx-f32-functional-v2-38ff6a225e75";
const TIMING_UNIT_POLICY_ID: &str = "mixed_script_ctc_character_whitespace_v1";
const NORMALIZATION_POLICY_ID: &str = "omni_ctc_attached_ar_marks_ja_separator_v1";
const PROVIDER_ID: &str = "omnilingual-ctc-mlx-supplied-text-v2";
/// `alignment_artifact.ALIGNMENT_SPEC['provider_identity_sha256']` — the
/// digest `alignment_provider.manifest()` returns for an intact install.
const PROVIDER_IDENTITY_SHA256: &str =
    "3c5461697116190782716451d0896f345e90cb70161e3c2930a95d314fe3c3a1";
const LEGACY_MODEL: &str = "e7c4e54ee4c4c47829cc6667d5d00ed8ea7bef1dcfeef0fce766f77752a2726c";
const LEGACY_CALIBRATION: &str = "ami-english-coarse-v1-5ab4e661e62f";

const BEHAVIOR_FIELDS: &[&str] = &[
    "provider_id",
    "provider_identity_sha256",
    "normalization_policy_id",
    "timing_unit_policy_id",
];

/// `reading_turns.qualified_identity`.
fn qualified_identity(model: Option<&Value>, calibration: Option<&Value>) -> bool {
    let (Some(m), Some(c)) = (
        model.and_then(Value::as_str),
        calibration.and_then(Value::as_str),
    ) else {
        return false;
    };
    (m == MODEL && c == CALIBRATION)
        || (m == MODEL && c == FUNCTIONAL_POLICY_ID)
        || (m == LEGACY_MODEL && c == LEGACY_CALIBRATION)
}

/// `reading_turns.behavior_valid`. New-unit receipts must match the installed
/// provider identity; legacy receipts carry no behavior fields.
fn behavior_valid(evidence: &Value) -> bool {
    let new = evidence
        .get("timing_unit_policy_id")
        .and_then(Value::as_str)
        == Some(TIMING_UNIT_POLICY_ID);
    let calibration = evidence
        .get("calibration_id")
        .or_else(|| evidence.get("frame_calibration_id"));
    if !new {
        return calibration.and_then(Value::as_str) != Some(FUNCTIONAL_POLICY_ID)
            && !BEHAVIOR_FIELDS
                .iter()
                .any(|key| evidence.get(key).is_some());
    }
    evidence.get("provider_id").and_then(Value::as_str) == Some(PROVIDER_ID)
        && evidence
            .get("normalization_policy_id")
            .and_then(Value::as_str)
            == Some(NORMALIZATION_POLICY_ID)
        && evidence
            .get("provider_identity_sha256")
            .and_then(Value::as_str)
            == Some(PROVIDER_IDENTITY_SHA256)
}

/// `str.fullmatch(r'speaker_\d+')`.
fn is_speaker_track(name: &str) -> bool {
    name.strip_prefix("speaker_")
        .is_some_and(|rest| !rest.is_empty() && rest.bytes().all(|b| b.is_ascii_digit()))
}

/// `re.fullmatch(r'overlap_speaker_\d+(?:_speaker_\d+)+', name)`.
fn is_overlap_track(name: &str) -> bool {
    let Some(rest) = name.strip_prefix("overlap_") else {
        return false;
    };
    let mut parts = rest.split("_speaker_");
    let first = parts.next().unwrap_or("");
    if !is_speaker_track(first) {
        return false;
    }
    let mut count = 1usize;
    for part in parts {
        if part.is_empty() || !part.bytes().all(|b| b.is_ascii_digit()) {
            return false;
        }
        count += 1;
    }
    count >= 2
}

/// `reading_turns.validated_turns` — the full v0.6.6 export gate, ported for
/// the legacy importer (only). Returns the validated turns, else `None` (the
/// parent passage is emitted intact).
pub fn validated_turns(segment: &Value) -> Option<Vec<Value>> {
    let receipt = segment
        .get("reading_turn_provenance")
        .filter(|v| v.is_object())
        .cloned()
        .unwrap_or_else(|| json!({}));
    let turns = segment.get("reading_turns")?;
    let turn_list = match turns {
        Value::Array(list) if !list.is_empty() => list,
        _ => return None,
    };
    // Any truthy protected_fields fails the gate.
    if truthy(segment.get("protected_fields")) {
        return None;
    }
    let text = segment.get("text").and_then(Value::as_str)?;
    if !eq(
        get(&receipt, "utterance_id"),
        segment.get("canonical_utterance_id"),
    ) || !eq(
        get(&receipt, "machine_revision"),
        segment.get("machine_revision"),
    ) || !eq(
        get(&receipt, "canonical_machine_revision"),
        segment.get("canonical_machine_revision"),
    ) || !eq(
        get(&receipt, "audio_revision"),
        segment.get("audio_revision"),
    ) || !eq(
        get(&receipt, "language_epoch"),
        segment.get("language_epoch"),
    ) || !eq(
        get(&receipt, "capture_source"),
        segment.get("capture_source"),
    ) || get(&receipt, "text_sha256").and_then(Value::as_str)
        != Some(sha256_hex_text(text).as_str())
        || !eq(
            get(&receipt, "audio_anchor"),
            segment.get("text_audio_anchor"),
        )
        || get(&receipt, "activity_sha256").and_then(Value::as_str)
            != Some(
                sha256_hex_json_ascii(
                    &segment
                        .get("speaker_activity")
                        .cloned()
                        .unwrap_or(Value::Null),
                )
                .as_str(),
            )
        || get(&receipt, "turns_sha256").and_then(Value::as_str)
            != Some(sha256_hex_json_utf8(turns).as_str())
        || !qualified_identity(
            get(&receipt, "model_sha256"),
            get(&receipt, "calibration_id"),
        )
        || !behavior_valid(&receipt)
    {
        return None;
    }
    let text_chars: Vec<char> = text.chars().collect();
    let mut cursor = 0usize;
    for turn in turn_list {
        let turn = turn.as_object()?;
        let (Some(start), Some(end)) = (
            turn.get("start_offset").and_then(Value::as_i64),
            turn.get("end_offset").and_then(Value::as_i64),
        ) else {
            return None;
        };
        let (Ok(start), Ok(end)) = (usize::try_from(start), usize::try_from(end)) else {
            return None;
        };
        if !(cursor == start && start < end && end <= text_chars.len()) {
            return None;
        }
        let expected: String = text_chars[cursor..end].iter().collect();
        if turn.get("text").and_then(Value::as_str) != Some(expected.as_str()) {
            return None;
        }
        let attribution = turn.get("attribution").and_then(Value::as_str);
        let speaker = turn.get("speaker");
        match attribution {
            Some("unknown") => {
                if speaker.and_then(Value::as_str) != Some("unassigned")
                    || !turn.get("start").is_none_or(Value::is_null)
                    || !turn.get("end").is_none_or(Value::is_null)
                {
                    return None;
                }
            }
            Some("single") => {
                let name = speaker.and_then(Value::as_str)?;
                if !is_speaker_track(name) {
                    return None;
                }
            }
            Some("overlap") => {
                let name = speaker.and_then(Value::as_str)?;
                if !is_overlap_track(name) {
                    return None;
                }
            }
            _ => return None,
        }
        if attribution != Some("unknown") {
            let (Some(t_start), Some(t_end)) = (
                turn.get("start").and_then(Value::as_f64),
                turn.get("end").and_then(Value::as_f64),
            ) else {
                return None;
            };
            let seg_start = segment.get("start").and_then(Value::as_f64).unwrap_or(0.0);
            let seg_end = segment.get("end").and_then(Value::as_f64).unwrap_or(0.0);
            if !(seg_start <= t_start && t_start < t_end && t_end <= seg_end) {
                return None;
            }
        }
        cursor = end;
    }
    (cursor == text_chars.len()).then(|| turn_list.to_vec())
}

/// One export passage — either a stored segment or a projected reading-turn
/// slice. `coarse_timing`/`reading_slice` are v0.6.6's export-local flags;
/// they are never persisted.
struct Passage<'a> {
    segment: &'a Segment,
    /// The slice projection (cloned parent + turn overrides); `None` for a
    /// parent passage.
    slice: Option<Segment>,
    coarse_timing: bool,
    reading_slice: bool,
}

impl Passage<'_> {
    fn effective(&self) -> &Segment {
        self.slice.as_ref().unwrap_or(self.segment)
    }
}

/// Reading-turn slices for export under the product rule: `reading_turns`
/// present implies valid (the importer already applied the full gate;
/// runtime edits remove `reading_turns` whenever text/start/end change).
/// Malformed rows degrade to the parent passage rather than aborting export.
fn export_turns(segment: &Segment) -> Option<Vec<Passage<'_>>> {
    let turns = segment.reading_turns.as_ref()?;
    if turns.is_empty() {
        return None;
    }
    let mut projected = Vec::with_capacity(turns.len());
    for turn in turns {
        let (Some(text), Some(speaker)) = (&turn.text, &turn.speaker) else {
            return None;
        };
        let mut slice = segment.clone();
        slice.text = text.clone();
        slice.speaker = speaker.clone();
        slice.start = turn.start.as_option().copied().unwrap_or(segment.start);
        slice.end = turn.end.as_option().copied().unwrap_or(segment.end);
        slice.review = Some(false);
        slice.speaker_candidates = Some(Vec::new());
        projected.push(Passage {
            segment,
            slice: Some(slice),
            coarse_timing: turn.attribution != Some(Attribution::Unknown),
            reading_slice: true,
        });
    }
    Some(projected)
}

/// `transcript.export` — returns `(content, mimetype)`. `kind` is
/// `json|txt|srt|vtt`.
pub fn export(document: &Document, kind: &str) -> Result<(String, &'static str), ExportTooLarge> {
    export_with_limit(document, kind, MAX_EXPORT_BYTES)
}

/// `export` with an explicit byte cap — the Python tests patch
/// `MAX_EXPORT_BYTES`; same boundary behavior here.
pub fn export_with_limit(
    document: &Document,
    kind: &str,
    cap: usize,
) -> Result<(String, &'static str), ExportTooLarge> {
    if kind == "json" {
        // Python json.dumps(indent=2, ensure_ascii=False) + '\n'.
        let body = serde_json::to_string_pretty(document).unwrap_or_default();
        return Ok((
            bounded_text(std::iter::once(format!("{body}\n")), cap)?,
            "application/json",
        ));
    }

    let mut lines: Vec<String> = Vec::new();
    if kind == "vtt" {
        lines.push("WEBVTT".to_string());
        lines.push(String::new());
    }
    let mut emitted = kind == "vtt";
    let mut passages: Vec<Passage> = Vec::new();
    for s in &document.segments {
        if s.text.trim().is_empty() {
            continue;
        }
        let turns = export_turns(s);
        let sliceable = turns.is_some()
            && (kind == "txt"
                || s.reading_turns.as_ref().is_some_and(|list| {
                    list.iter()
                        .all(|t| t.attribution != Some(Attribution::Unknown))
                }));
        if sliceable {
            passages.extend(turns.unwrap());
        } else {
            passages.push(Passage {
                segment: s,
                slice: None,
                // `s.get('coarse_timing')`/`s.get('reading_slice')` — the
                // flags are honored verbatim from a stored row too.
                coarse_timing: s.coarse_timing == Some(true),
                reading_slice: s.reading_slice == Some(true),
            });
        }
    }
    let speakers = &document.speakers;
    for (n, s) in passages.iter().enumerate() {
        let n = n + 1;
        emitted = true;
        let seg = s.effective();
        let speaker_key = seg.speaker.as_str();
        let speaker = speakers.get(speaker_key).map(String::as_str).unwrap_or(
            if speaker_key == "unassigned" {
                "Unknown speaker"
            } else {
                "Overlapping speakers"
            },
        );
        let passage = export_text(seg);
        let text = collapse(sanitize(&passage));
        let name = collapse(sanitize(speaker));
        if kind == "txt" {
            let timing = if s.coarse_timing {
                "Coarse timing "
            } else if s.reading_slice {
                "Timing unknown; parent audio "
            } else {
                ""
            };
            lines.push(format!(
                "[{timing}{} – {}] {speaker}: {passage}",
                timestamp(seg.start, '.'),
                timestamp(seg.end, '.'),
            ));
        } else {
            let sep = if kind == "srt" { ',' } else { '.' };
            lines.push(n.to_string());
            lines.push(format!(
                "{} --> {}",
                timestamp(seg.start, sep),
                timestamp(seg.end, sep),
            ));
            lines.push(format!("{name}: {text}"));
            lines.push(String::new());
        }
    }
    if !emitted {
        lines.push(String::new());
    }
    let mimetype = if kind == "vtt" {
        "text/vtt; charset=utf-8"
    } else {
        "text/plain; charset=utf-8"
    };
    Ok((
        bounded_text(lines.into_iter().map(|line| format!("{line}\n")), cap)?,
        mimetype,
    ))
}

/// `' '.join(text.split())`.
fn collapse(text: String) -> String {
    text.split_whitespace().collect::<Vec<_>>().join(" ")
}

/// `.replace('-->', '→').replace('<', '‹').replace('>', '›')`.
fn sanitize(text: &str) -> String {
    text.replace("-->", "→").replace('<', "‹").replace('>', "›")
}
