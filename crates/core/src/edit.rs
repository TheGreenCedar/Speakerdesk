//! Transcript replacement and passage edit rules, including protected fields.

use std::collections::HashMap;

use serde_json::{Map, Value};

use crate::document::{Document, RECOVERY_ACTIVE, Segment};
use crate::error::ApiError;
use crate::messages::{
    self, ONLY_WORDS_SPEAKER_BOUNDS, PASSAGE_STALE, SUPPLY_PASSAGE_EDITS, TRANSCRIPT_STALE,
    WAIT_FOR_INFERENCE, WAIT_FOR_PASSAGES,
};
use crate::transcript::{json_truthy, number_as_i64, validate};

const PASSAGE_EDIT_BLOCKED: &[&str] = &["preparing", "queued", "processing"];
const NOT_FOUND: &str = "The requested URL was not found on the server. If you entered the URL manually please check your spelling and try again.";
const SAVED_LOCALLY: &str = "Saved locally.";
const IMPORTED_TIMING: &str = "User imported segment times; source unverified";
const IMPORTED_WARNING: &str =
    "Imported transcript. Text, speaker labels and timing have not been verified by this app.";
const MANUAL_TIMING: &str = "User supplied recording times";
const MANUAL_WARNING: &str =
    "This transcript contains manually entered or imported text and timing.";

const SERVER_REST_KEYS: &[&str] = &[
    "audio_anchor",
    "source_start",
    "source_end",
    "source_speaker_candidates",
    "activity_regions",
    "language_epoch",
    "finalized",
    "refinement_window",
    "fast_origin_sample",
    "language_generation",
    "language_mode",
    "canonical_machine_revision",
    "canonical_state",
    "start_sample",
    "end_sample",
    "capture_audio",
    "text_audio_anchor",
    "canonical_text_end_sample",
    "canonical_text_boundary_sample",
    "stream_generation",
    "reading_alignment",
    "decode_context",
    "speech_regions",
    "assembly_provenance",
    "bounded_decode_provenance",
    "audio_state",
    "canonical_unresolved",
    "refinement_issue",
    "deleted",
];

const TYPED_SERVER_KEYS: &[&str] = &[
    "canonical_utterance_id",
    "audio_revision",
    "capture_source",
    "alignment",
    "speaker_activity",
    "reading_turns",
    "reading_turn_provenance",
    "separation_candidate",
    "separation",
];

/// Inputs for `PATCH /api/jobs/<jid>/segments/<sid>`.
pub struct PassageEditInput<'a> {
    pub status: &'a str,
    pub duration: f64,
    pub revision: i64,
    pub document: Option<&'a Document>,
    pub segment_id: &'a str,
    pub body: &'a Value,
    pub now: f64,
}

/// Passage edit result. `revision` is the job revision after the write.
#[derive(Debug, Clone, PartialEq)]
pub struct PassageEditOutcome {
    pub document: Document,
    pub segment: Segment,
    pub revision: i64,
    pub language_context_changed: bool,
}

/// Inputs for `PUT /api/jobs/<jid>/transcript`.
pub struct SaveInput<'a> {
    pub status: &'a str,
    pub duration: Option<f64>,
    pub revision: i64,
    pub saved: Option<&'a Document>,
    pub body: &'a Value,
    pub now: f64,
    pub speaker_assignments: &'a Map<String, Value>,
    pub voice_checks: &'a Map<String, Value>,
}

/// Whole-transcript save result.
#[derive(Debug, Clone, PartialEq)]
pub struct SaveOutcome {
    pub document: Document,
    pub revision: i64,
    pub status: &'static str,
    pub message: &'static str,
    pub speaker_assignments: Map<String, Value>,
    pub voice_checks: Map<String, Value>,
    /// `save` always drops `voice_suggestions`.
    pub clear_voice_suggestions: bool,
    pub language_context_changed: bool,
}

/// `edit_passage` rules: protected fields, stale passage revision, server attachments.
pub fn edit_passage(input: PassageEditInput<'_>) -> Result<PassageEditOutcome, ApiError> {
    let Some(body) = input.body.as_object() else {
        return Err(ApiError::new(400, SUPPLY_PASSAGE_EDITS));
    };
    let Some(changes) = body.get("changes").and_then(Value::as_object) else {
        return Err(ApiError::new(400, SUPPLY_PASSAGE_EDITS));
    };
    if changes.is_empty()
        || changes.keys().any(|key| {
            !matches!(
                key.as_str(),
                "text" | "speaker" | "start" | "end" | "deleted"
            )
        })
    {
        return Err(ApiError::new(400, ONLY_WORDS_SPEAKER_BOUNDS));
    }
    if changes
        .get("deleted")
        .is_some_and(|value| !value.is_boolean())
    {
        return Err(ApiError::new(
            400,
            "Passage deletion must be true or false.",
        ));
    }
    let Some(document) = input.document else {
        return Err(ApiError::new(409, WAIT_FOR_PASSAGES));
    };
    if PASSAGE_EDIT_BLOCKED.contains(&input.status) {
        return Err(ApiError::new(409, WAIT_FOR_PASSAGES));
    }
    let Some(prior) = document
        .segments
        .iter()
        .find(|segment| segment.id == input.segment_id)
    else {
        return Err(ApiError::new(404, NOT_FOUND));
    };
    if !same_revision(
        body.get("segment_revision"),
        prior.machine_revision.unwrap_or(0),
    ) {
        return Err(ApiError::new(409, PASSAGE_STALE));
    }
    let prior = prior.clone();
    let mut value = serde_json::to_value(document).expect("transcript document serializes");
    apply_json_changes(&mut value, input.segment_id, changes);
    let mut validated =
        validate(&value, input.duration).map_err(|error| ApiError::new(400, error.message))?;
    let Some(row) = validated
        .segments
        .iter_mut()
        .find(|segment| segment.id == input.segment_id)
    else {
        return Err(ApiError::new(404, NOT_FOUND));
    };
    crate::separation::validate_bounds(row)?;
    finish_passage_edit(row, &prior, changes);
    let segment = row.clone();
    validated.edited_at = Some(input.now);
    Ok(PassageEditOutcome {
        language_context_changed: language_row_interpretation_changed(&prior, &segment),
        document: validated,
        segment,
        revision: input.revision + 1,
    })
}

/// User changes to acoustic ownership/interpretation invalidate one operation's
/// compact main-language summary. Spelling and display-name edits are neutral.
fn language_row_interpretation_changed(prior: &Segment, row: &Segment) -> bool {
    prior.start != row.start
        || prior.end != row.end
        || prior.capture_source != row.capture_source
        || prior.audio_revision != row.audio_revision
        || [
            "language_epoch",
            "language_mode",
            "stream_generation",
            "source_speaker_candidates",
        ]
        .iter()
        .any(|key| prior.rest.get(*key) != row.rest.get(*key))
        || prior
            .protected_fields
            .as_ref()
            .is_some_and(|fields| fields.iter().any(|field| field == "speaker"))
            != row
                .protected_fields
                .as_ref()
                .is_some_and(|fields| fields.iter().any(|field| field == "speaker"))
        || prior.language != row.language
        || (row
            .language_detection
            .as_ref()
            .is_some_and(|d| d["mode"] == "manual")
            && prior.language_detection != row.language_detection)
}
fn language_interpretation_changed(before: &Document, after: &Document) -> bool {
    if before.segments.len() != after.segments.len() {
        return true;
    }
    let priors: HashMap<&str, &Segment> = before
        .segments
        .iter()
        .map(|row| (row.id.as_str(), row))
        .collect();
    after.segments.iter().any(|row| {
        priors
            .get(row.id.as_str())
            .is_none_or(|prior| language_row_interpretation_changed(prior, row))
    })
}

/// `save` rules: stale transcript revision, protected fields, server-owned restoration.
pub fn save_transcript(input: SaveInput<'_>) -> Result<SaveOutcome, ApiError> {
    if RECOVERY_ACTIVE.contains(&input.status) || !duration_ready(input.duration) {
        return Err(ApiError::new(409, WAIT_FOR_INFERENCE));
    }
    let Some(body) = input.body.as_object() else {
        return Err(ApiError::new(400, messages::INVALID_REQUEST));
    };
    if !same_revision(body.get("revision"), input.revision) {
        return Err(ApiError::new(409, TRANSCRIPT_STALE));
    }
    let duration = input.duration.unwrap_or(0.0);
    let document_value = body.get("document").unwrap_or(&Value::Null);
    let mut incoming =
        validate(document_value, duration).map_err(|error| ApiError::new(400, error.message))?;
    let imported = json_truthy(body.get("imported"));
    let priors = prior_segments(input.saved);
    for segment in &mut incoming.segments {
        let prior = priors.get(&segment.id);
        apply_save_rules(segment, prior, imported, input.now);
        crate::separation::validate_bounds(segment)?;
    }
    crate::transcript::sort_segments(&mut incoming.segments);
    let mut document = assemble_document(input.saved, incoming, imported, input.now);
    document.edited_at = Some(input.now);
    let (speaker_assignments, voice_checks) = reconcile_assignments(
        &document.speakers,
        imported,
        input.speaker_assignments,
        input.voice_checks,
    );
    let language_context_changed = input
        .saved
        .is_some_and(|prior| language_interpretation_changed(prior, &document));
    Ok(SaveOutcome {
        language_context_changed,
        document,
        revision: input.revision + 1,
        status: "ready",
        message: SAVED_LOCALLY,
        speaker_assignments,
        voice_checks,
        clear_voice_suggestions: true,
    })
}

fn apply_json_changes(document: &mut Value, segment_id: &str, changes: &Map<String, Value>) {
    let Some(segments) = document.get_mut("segments").and_then(Value::as_array_mut) else {
        return;
    };
    for segment in segments {
        let matches = segment.get("id").and_then(Value::as_str) == Some(segment_id);
        if !matches {
            continue;
        }
        let Some(object) = segment.as_object_mut() else {
            return;
        };
        for (key, change) in changes {
            object.insert(key.clone(), change.clone());
        }
        return;
    }
}

fn finish_passage_edit(row: &mut Segment, prior: &Segment, changes: &Map<String, Value>) {
    let mut protected = prior.protected_fields.clone().unwrap_or_default();
    for key in changes.keys() {
        let key = if key == "deleted" {
            "text"
        } else {
            key.as_str()
        };
        if !protected.iter().any(|field| field == key) {
            protected.push(key.to_string());
        }
    }
    protected.sort();
    row.rest.remove("protected_fields");
    row.protected_fields = Some(protected);
    row.rest.remove("machine_revision");
    row.machine_revision = Some(prior.machine_revision.unwrap_or(0) + 1);
    row.rest.remove("refinement_state");
    row.refinement_state = Some("edited".to_string());
    clear_alignment_bundle(row);
    if (changes.contains_key("text") || changes.get("deleted") == Some(&Value::Bool(true)))
        && prior.rest.get("canonical_state").and_then(Value::as_str) == Some("open")
    {
        // A manual fill before first ASR covers the saved displayed audio scope. A
        // correction to decoded words covers only that decode's original endpoint.
        if let Some(boundary) = prior
            .rest
            .get("canonical_text_boundary_sample")
            .or_else(|| prior.rest.get("canonical_text_end_sample"))
            .or_else(|| prior.rest.get("end_sample"))
            .and_then(Value::as_i64)
        {
            row.rest.insert(
                "canonical_text_boundary_sample".into(),
                Value::from(boundary),
            );
        }
    }
    if row.start != prior.start || row.end != prior.end {
        row.rest.remove("text_audio_anchor");
    }
    if row.speaker != prior.speaker || row.start != prior.start || row.end != prior.end {
        row.rest
            .insert("voice_eligible".to_string(), Value::Bool(false));
        row.rest
            .insert("speaker_candidates".to_string(), Value::Array(Vec::new()));
        row.retry_candidate = None;
        row.rest.remove("retry_candidate");
    }
    if !row.text.trim().is_empty() && !row.speaker.starts_with("overlap") {
        row.rest.remove("review_resolution");
        row.review_resolution = Some("words_reviewed".to_string());
        row.rest.remove("review");
        row.review = Some(false);
    }
    clear_stale_refinement_issue(row);
}

fn apply_save_rules(segment: &mut Segment, prior: Option<&Segment>, imported: bool, now: f64) {
    restore_server_owned(segment, prior);
    let edited = edited_keys(segment, prior);
    apply_protection(segment, prior, &edited);
    if !edited.is_empty() || imported {
        clear_alignment_bundle(segment);
    }
    if edited.iter().any(|key| *key == "start" || *key == "end") || imported {
        segment.rest.remove("text_audio_anchor");
    }
    apply_voice(segment, prior, imported, now);
    if prior.is_some_and(|prior| language_row_interpretation_changed(prior, segment)) {
        segment.rest.remove("refinement_issue");
    }
    clear_stale_refinement_issue(segment);
}

fn clear_stale_refinement_issue(segment: &mut Segment) {
    if segment.rest.get("refinement_issue").is_some_and(|issue| {
        issue["segment_version"].as_str() != Some(crate::live::segment_version(segment).as_str())
    }) {
        segment.rest.remove("refinement_issue");
    }
}

fn restore_server_owned(segment: &mut Segment, prior: Option<&Segment>) {
    if let Some(detection) = segment
        .language_detection
        .as_mut()
        .and_then(Value::as_object_mut)
    {
        detection.remove("context_witnesses");
        if let Some(witnesses) = prior
            .and_then(|row| row.language_detection.as_ref())
            .and_then(|d| d.get("context_witnesses"))
        {
            detection.insert("context_witnesses".into(), witnesses.clone());
        }
    }
    segment.rest.remove("decode_core_plan");
    for key in SERVER_REST_KEYS {
        segment.rest.remove(*key);
    }
    for key in TYPED_SERVER_KEYS {
        segment.rest.remove(*key);
    }
    segment.separation_candidate = None;
    segment.separation = None;
    segment.canonical_utterance_id = None;
    segment.audio_revision = None;
    segment.capture_source = None;
    segment.alignment = None;
    segment.speaker_activity = None;
    segment.reading_turns = None;
    segment.reading_turn_provenance = None;
    let Some(prior) = prior else {
        return;
    };
    segment.separation_candidate = prior.separation_candidate.clone();
    segment.separation = prior.separation.clone();
    segment.canonical_utterance_id = prior.canonical_utterance_id.clone();
    segment.audio_revision = prior.audio_revision;
    segment.capture_source = prior.capture_source.clone();
    segment.alignment = prior.alignment.clone();
    segment.speaker_activity = prior.speaker_activity.clone();
    segment.reading_turns = prior.reading_turns.clone();
    segment.reading_turn_provenance = prior.reading_turn_provenance.clone();
    for key in SERVER_REST_KEYS {
        if let Some(value) = prior.rest.get(*key) {
            segment.rest.insert((*key).to_string(), value.clone());
        }
    }
}

fn edited_keys(segment: &Segment, prior: Option<&Segment>) -> Vec<&'static str> {
    let Some(prior) = prior else {
        return Vec::new();
    };
    let mut keys = Vec::new();
    if segment.text != prior.text {
        keys.push("text");
    }
    if segment.speaker != prior.speaker {
        keys.push("speaker");
    }
    if segment.start != prior.start {
        keys.push("start");
    }
    if segment.end != prior.end {
        keys.push("end");
    }
    keys
}

fn apply_protection(segment: &mut Segment, prior: Option<&Segment>, edited: &[&str]) {
    let Some(prior) = prior else {
        clear_bookkeeping(segment);
        return;
    };
    let protected_nonempty = prior
        .protected_fields
        .as_ref()
        .is_some_and(|fields| !fields.is_empty());
    if prior.machine_revision.is_none() && !protected_nonempty && edited.is_empty() {
        clear_bookkeeping(segment);
        return;
    }
    let mut fields = prior.protected_fields.clone().unwrap_or_default();
    for key in edited {
        if !fields.iter().any(|field| field == *key) {
            fields.push((*key).to_string());
        }
    }
    fields.sort();
    segment.rest.remove("protected_fields");
    segment.protected_fields = Some(fields);
    segment.rest.remove("machine_revision");
    let bump = i64::from(!edited.is_empty());
    segment.machine_revision = Some(prior.machine_revision.unwrap_or(0) + bump);
    segment.rest.remove("refinement_state");
    if protected_nonempty || !edited.is_empty() {
        segment.refinement_state = Some("edited".to_string());
    } else if let Some(state) = prior.refinement_state.clone() {
        segment.refinement_state = Some(state);
    } else {
        segment.refinement_state = None;
        segment
            .rest
            .insert("refinement_state".to_string(), Value::Null);
    }
}

fn clear_bookkeeping(segment: &mut Segment) {
    segment.protected_fields = None;
    segment.machine_revision = None;
    segment.refinement_state = None;
    segment.rest.remove("protected_fields");
    segment.rest.remove("machine_revision");
    segment.rest.remove("refinement_state");
}

pub(crate) fn clear_alignment_bundle(segment: &mut Segment) {
    segment.alignment = None;
    segment.reading_turns = None;
    segment.reading_turn_provenance = None;
    for key in [
        "alignment",
        "reading_alignment",
        "assembly_provenance",
        "bounded_decode_provenance",
        "reading_turns",
        "reading_turn_provenance",
    ] {
        segment.rest.remove(key);
    }
}

fn apply_voice(segment: &mut Segment, prior: Option<&Segment>, imported: bool, now: f64) {
    let unchanged = prior.is_some_and(|prior| {
        !imported
            && prior.speaker == segment.speaker
            && prior.start == segment.start
            && prior.end == segment.end
    });
    let eligible = unchanged
        && prior.is_some_and(|prior| speaker_audio_eligible(prior, prior.speaker.as_str()));
    segment
        .rest
        .insert("voice_eligible".to_string(), Value::Bool(eligible));
    let candidates = match prior {
        Some(prior) if unchanged => prior
            .rest
            .get("speaker_candidates")
            .cloned()
            .unwrap_or_else(|| Value::Array(vec![Value::String(prior.speaker.clone())])),
        _ => Value::Array(Vec::new()),
    };
    segment
        .rest
        .insert("speaker_candidates".to_string(), candidates);
    let prior_retry = prior.and_then(|prior| prior.retry_candidate.clone());
    segment.retry_candidate = None;
    segment.rest.remove("retry_candidate");
    if !unchanged {
        return;
    }
    let Some(mut candidate) = prior_retry else {
        return;
    };
    let explicit_language_repair = prior.is_some_and(|prior| {
        candidate["id"].is_string()
            && candidate["accepted_at"].is_null()
            && candidate["error"].is_null()
            && candidate["original_text"] == prior.text
            && candidate["text"]
                .as_str()
                .is_some_and(|text| !text.trim().is_empty() && text == segment.text)
            && candidate["language"].as_str().is_some_and(|language| {
                crate::languages::is_supported(language)
                    && segment.language.as_deref() == Some(language)
            })
            && segment
                .language_detection
                .as_ref()
                .is_some_and(|d| d["mode"] == "manual" && d["reason"] == "passage_retry")
            && (segment.language != prior.language
                || segment.language_detection != prior.language_detection)
    });
    if let Some(prior) = prior
        && let Some(object) = candidate.as_object_mut()
    {
        let candidate_text = object.get("text").and_then(Value::as_str);
        if candidate_text == Some(segment.text.as_str())
            && (segment.text != prior.text || explicit_language_repair)
        {
            object.insert(
                "replaced_text".to_string(),
                Value::String(prior.text.clone()),
            );
            object.insert("accepted_at".to_string(), f64_value(now));
            if explicit_language_repair && segment.text == prior.text {
                let mut protected = segment.protected_fields.clone().unwrap_or_default();
                if !protected.iter().any(|field| field == "text") {
                    protected.push("text".into());
                    protected.sort();
                }
                segment.protected_fields = Some(protected);
                segment.machine_revision = Some(prior.machine_revision.unwrap_or(0) + 1);
                segment.refinement_state = Some("edited".into());
                clear_alignment_bundle(segment);
            }
        }
    }
    segment.retry_candidate = Some(candidate);
}

fn speaker_audio_eligible(segment: &Segment, track_id: &str) -> bool {
    if segment.separation.is_some()
        || segment.speaker != track_id
        || json_truthy(segment.rest.get("audio_state"))
    {
        return false;
    }
    if segment.rest.get("finalized") == Some(&Value::Bool(false)) {
        return false;
    }
    let overlap = track_id.starts_with("overlap");
    let voice_ok = match segment.rest.get("voice_eligible") {
        Some(Value::Bool(true)) => true,
        Some(_) => false,
        None => segment.review != Some(true),
    };
    match segment.rest.get("speaker_candidates") {
        Some(Value::Array(items)) => !overlap && items.len() == 1 && voice_ok,
        Some(_) => false,
        None => !overlap && voice_ok,
    }
}

fn assemble_document(
    saved: Option<&Document>,
    mut incoming: Document,
    imported: bool,
    now: f64,
) -> Document {
    if imported {
        return Document {
            schema_version: 1,
            speakers: incoming.speakers,
            segments: incoming.segments,
            provenance: Some(serde_json::json!({
                "kind": "imported",
                "timing": IMPORTED_TIMING,
            })),
            warnings: vec![IMPORTED_WARNING.to_string()],
            diarization: Vec::new(),
            edited_at: Some(now),
            rest: Map::new(),
        };
    }
    if let Some(saved) = saved {
        for row in &saved.segments {
            if row.rest.get("deleted") == Some(&Value::Bool(true))
                && !incoming
                    .segments
                    .iter()
                    .any(|candidate| candidate.id == row.id)
            {
                incoming.segments.push(row.clone());
            }
        }
        crate::transcript::sort_segments(&mut incoming.segments);
        let mut document = saved.clone();
        document.speakers = incoming.speakers;
        document.segments = incoming.segments;
        document.edited_at = Some(now);
        return document;
    }
    Document {
        schema_version: 1,
        speakers: incoming.speakers,
        segments: incoming.segments,
        provenance: Some(serde_json::json!({
            "kind": "manual",
            "timing": MANUAL_TIMING,
        })),
        warnings: vec![MANUAL_WARNING.to_string()],
        diarization: Vec::new(),
        edited_at: Some(now),
        rest: Map::new(),
    }
}

fn reconcile_assignments(
    speakers: &std::collections::BTreeMap<String, String>,
    imported: bool,
    assignments: &Map<String, Value>,
    checks: &Map<String, Value>,
) -> (Map<String, Value>, Map<String, Value>) {
    let mut kept = Map::new();
    if !imported {
        for (track, value) in assignments {
            let Some(name) = speakers.get(track) else {
                continue;
            };
            let Some(object) = value.as_object() else {
                continue;
            };
            if object.get("name").and_then(Value::as_str) == Some(name.as_str()) {
                kept.insert(track.clone(), value.clone());
            }
        }
    }
    let mut voice_checks = checks.clone();
    for value in voice_checks.values_mut() {
        let Some(object) = value.as_object_mut() else {
            continue;
        };
        if object.get("status").and_then(Value::as_str) == Some("checking") {
            object.insert("status".to_string(), Value::String("cancelled".to_string()));
        }
    }
    (kept, voice_checks)
}

fn prior_segments(saved: Option<&Document>) -> HashMap<String, Segment> {
    let Some(saved) = saved else {
        return HashMap::new();
    };
    saved
        .segments
        .iter()
        .map(|segment| (segment.id.clone(), segment.clone()))
        .collect()
}

fn duration_ready(duration: Option<f64>) -> bool {
    match duration {
        Some(value) => value != 0.0,
        None => false,
    }
}

fn same_revision(provided: Option<&Value>, expected: i64) -> bool {
    provided.and_then(Value::as_number).and_then(number_as_i64) == Some(expected)
}

fn f64_value(value: f64) -> Value {
    serde_json::Number::from_f64(value)
        .map(Value::Number)
        .unwrap_or(Value::Null)
}

#[cfg(test)]
mod tests {
    use super::{PassageEditInput, SaveInput, edit_passage, save_transcript};
    use crate::messages::{self, PASSAGE_STALE, TRANSCRIPT_STALE};
    use crate::transcript::{bare_document, bare_segment};
    use serde_json::{Map, Value, json};

    fn prior_segment() -> crate::document::Segment {
        let mut segment = bare_segment("one", 0.0, 1.0, "speaker_0", "Hello");
        segment.machine_revision = Some(2);
        segment.refinement_state = Some("refined".to_string());
        segment.audio_revision = Some(4);
        segment.canonical_utterance_id = Some("canonical-one".to_string());
        segment.alignment = Some(json!({"old": true}));
        segment.reading_turns = Some(json!([{
            "text": "Hello",
            "speaker": "speaker_0",
            "attribution": "single",
            "start": 0.0,
            "end": 1.0
        }]));
        segment
            .rest
            .insert("language_generation".to_string(), json!(0));
        segment
            .rest
            .insert("text_audio_anchor".to_string(), json!({"start_sample": 0}));
        segment.retry_candidate = Some(json!({"text": "Hello"}));
        segment
    }

    fn saved_document(segment: crate::document::Segment) -> crate::document::Document {
        let mut document = bare_document(&[("speaker_0", "Ada")], vec![segment]);
        document.warnings = vec!["keep me".to_string()];
        document
    }

    fn empty_maps() -> (Map<String, Value>, Map<String, Value>) {
        (Map::new(), Map::new())
    }

    #[test]
    fn passage_stale_revision_returns_exact_409() {
        let segment = prior_segment();
        let document = saved_document(segment);
        let stale = json!({"segment_revision": 1, "changes": {"text": "Stale."}});
        let error = edit_passage(PassageEditInput {
            status: "ready",
            duration: 10.0,
            revision: 7,
            document: Some(&document),
            segment_id: "one",
            body: &stale,
            now: 12.0,
        })
        .unwrap_err();
        assert_eq!(error.status, 409);
        assert_eq!(error.message, PASSAGE_STALE);
        assert_eq!(
            error.message,
            "This passage changed. Review its latest words before saving."
        );

        let missing = json!({"changes": {"text": "Hello"}});
        let error = edit_passage(PassageEditInput {
            status: "ready",
            duration: 10.0,
            revision: 7,
            document: Some(&document),
            segment_id: "one",
            body: &missing,
            now: 12.0,
        })
        .unwrap_err();
        assert_eq!(error.status, 409);
        assert_eq!(error.message, messages::PASSAGE_STALE);
    }

    #[test]
    fn passage_edit_protects_text_and_keeps_audio_anchor() {
        let document = saved_document(prior_segment());
        let body = json!({"segment_revision": 2, "changes": {"text": "Hello"}});
        let outcome = edit_passage(PassageEditInput {
            status: "ready",
            duration: 10.0,
            revision: 7,
            document: Some(&document),
            segment_id: "one",
            body: &body,
            now: 12.0,
        })
        .unwrap();
        assert_eq!(outcome.revision, 8);
        assert_eq!(outcome.segment.text, "Hello");
        assert_eq!(
            outcome.segment.protected_fields,
            Some(vec!["text".to_string()])
        );
        assert_eq!(outcome.segment.machine_revision, Some(3));
        assert_eq!(outcome.segment.refinement_state.as_deref(), Some("edited"));
        assert!(outcome.segment.reading_turns.is_none());
        assert!(outcome.segment.alignment.is_none());
        assert_eq!(
            outcome.segment.review_resolution.as_deref(),
            Some("words_reviewed")
        );
        assert_eq!(outcome.segment.review, Some(false));
        assert_eq!(
            outcome.segment.retry_candidate,
            Some(json!({"text": "Hello"}))
        );
        assert_eq!(
            outcome.segment.rest.get("text_audio_anchor"),
            Some(&json!({"start_sample": 0}))
        );
        assert!(outcome.segment.rest.get("voice_eligible").is_none());
        assert_eq!(outcome.segment.audio_revision, Some(4));
        assert_eq!(
            outcome.segment.canonical_utterance_id.as_deref(),
            Some("canonical-one")
        );
        assert_eq!(outcome.document.edited_at, Some(12.0));
    }

    #[test]
    fn passage_deletion_is_protected_and_survives_stale_document_content() {
        let mut document = saved_document(prior_segment());
        document.segments[0]
            .rest
            .insert("canonical_state".into(), json!("open"));
        document.segments[0]
            .rest
            .insert("end_sample".into(), json!(16000));
        let deleted = edit_passage(PassageEditInput {
            status: "recording",
            duration: 10.,
            revision: 7,
            document: Some(&document),
            segment_id: "one",
            body: &json!({"segment_revision":2,"changes":{"deleted":true}}),
            now: 12.,
        })
        .unwrap();
        assert_eq!(deleted.segment.rest.get("deleted"), Some(&json!(true)));
        assert_eq!(deleted.segment.text, "Hello");
        assert_eq!(
            deleted.segment.rest.get("canonical_text_boundary_sample"),
            Some(&json!(16000))
        );
        let mut growing = document.segments[0].clone();
        growing.rest.insert("end_sample".into(), json!(32000));
        assert_eq!(
            crate::live::edited_boundary(&deleted.document, &growing)
                .unwrap()
                .end_sample,
            16000
        );
        assert_eq!(deleted.segment.protected_fields, Some(vec!["text".into()]));
        assert_eq!(
            deleted.segment.rest.get("text_audio_anchor"),
            document.segments[0].rest.get("text_audio_anchor")
        );
        let (assignments, checks) = empty_maps();
        let saved = save_transcript(SaveInput {
            status: "ready",
            duration: Some(10.),
            revision: 8,
            saved: Some(&deleted.document),
            body: &json!({"revision":8,"document":document}),
            now: 13.,
            speaker_assignments: &assignments,
            voice_checks: &checks,
        })
        .unwrap();
        assert_eq!(
            saved.document.segments[0].rest.get("deleted"),
            Some(&json!(true))
        );
        for kind in ["txt", "srt", "vtt", "json"] {
            let exported = crate::export::export(&saved.document, kind).unwrap();
            assert!(!exported.body.contains("Hello"), "{kind}");
        }
        let restored = edit_passage(PassageEditInput {
            status: "ready",
            duration: 10.,
            revision: 9,
            document: Some(&saved.document),
            segment_id: "one",
            body: &json!({"segment_revision":3,"changes":{"deleted":false}}),
            now: 14.,
        })
        .unwrap();
        assert_eq!(restored.segment.text, "Hello");
        assert_ne!(restored.segment.rest.get("deleted"), Some(&json!(true)));
        assert!(
            crate::export::export(&restored.document, "txt")
                .unwrap()
                .body
                .contains("Hello")
        );
    }

    #[test]
    fn transcript_stale_revision_returns_exact_409() {
        let document = saved_document(prior_segment());
        let (assignments, checks) = empty_maps();
        let body = json!({
            "revision": 6,
            "document": {
                "speakers": {"speaker_0": "Ada"},
                "segments": [{
                    "id": "one",
                    "start": 0.0,
                    "end": 1.0,
                    "speaker": "speaker_0",
                    "text": "Hello"
                }]
            }
        });
        let error = save_transcript(SaveInput {
            status: "ready",
            duration: Some(10.0),
            revision: 7,
            saved: Some(&document),
            body: &body,
            now: 20.0,
            speaker_assignments: &assignments,
            voice_checks: &checks,
        })
        .unwrap_err();
        assert_eq!(error.status, 409);
        assert_eq!(error.message, TRANSCRIPT_STALE);
        assert_eq!(
            error.message,
            "This transcript changed in another tab. Reload before saving."
        );

        let busy = save_transcript(SaveInput {
            status: "processing",
            duration: Some(10.0),
            revision: 7,
            saved: Some(&document),
            body: &body,
            now: 20.0,
            speaker_assignments: &assignments,
            voice_checks: &checks,
        })
        .unwrap_err();
        assert_eq!(busy.status, 409);
        assert_eq!(busy.message, messages::WAIT_FOR_INFERENCE);
    }

    #[test]
    fn saves_preserve_imported_voice_crop_ownership() {
        use crate::voice::{AudioSegment, speaker_audio_ranges};
        let segment = serde_json::from_value(json!({
            "id":"seg-0","speaker":"unassigned","start":0.03,"end":4.424,
            "text":"Words with a tail.","review":true,"voice_eligible":false,
            "canonical_utterance_id":"seg-0","audio_revision":0,
            "speaker_candidates":["speaker_0"],
            "decode_context":{"start_sample":480,"end_sample":70784,"boundary_sample":64800,"policy":"speech_utterance_tail_context_v1"},
            "speaker_activity":{"audio_revision":0,"observed_end_sample":70784,"regions":[
                {"start_sample":480,"end_sample":64800,"speakers":["speaker_0"]},
                {"start_sample":64800,"end_sample":70784,"speakers":[]}
            ]}
        })).unwrap();
        let mut document = saved_document(segment);
        document
            .speakers
            .insert("unassigned".into(), "Unassigned".into());
        let anchor = document.segments[0].rest["decode_context"].clone();
        let (assignments, checks) = empty_maps();
        for forged in [false, true] {
            let mut incoming = serde_json::to_value(&document).unwrap();
            let row = incoming["segments"][0].as_object_mut().unwrap();
            row.insert("text".into(), json!("Human text edit"));
            row.remove("decode_context");
            if forged {
                row.insert(
                    "decode_context".into(),
                    json!({"start_sample":0,"end_sample":80000,"policy":"forged"}),
                );
            }
            let outcome = save_transcript(SaveInput {
                status: "ready",
                duration: Some(10.),
                revision: 7,
                saved: Some(&document),
                body: &json!({"revision":7,"document":incoming}),
                now: 20.,
                speaker_assignments: &assignments,
                voice_checks: &checks,
            })
            .unwrap();
            let saved = &outcome.document.segments[0];
            assert_eq!(saved.rest.get("decode_context"), Some(&anchor));
            assert_eq!(saved.text, "Human text edit");
            let audio: AudioSegment =
                serde_json::from_value(serde_json::to_value(saved).unwrap()).unwrap();
            assert_eq!(
                speaker_audio_ranges(&audio, "speaker_0"),
                vec![(0.03, 4.05)]
            );
        }
    }

    #[test]
    fn save_restores_server_owned_fields_and_protects_edits() {
        let mut document = saved_document(prior_segment());
        document.segments[0].language_detection =
            Some(json!({"context_witnesses":[{"owner":"server"}]}));
        let (assignments, checks) = empty_maps();
        let unchanged = json!({
            "revision": 7,
            "document": {
                "speakers": {"speaker_0": "Ada"},
                "segments": [{
                    "id": "one",
                    "start": 0.0,
                    "end": 1.0,
                    "speaker": "speaker_0",
                    "text": "Hello",
                    "audio_revision": 0,
                    "canonical_utterance_id": "forged",
                    "alignment": {"forged": true},
                    "language_detection": {"context_witnesses":[{"owner":"uploaded"}]},
                    "language_generation": 99,
                    "machine_revision": 900,
                    "protected_fields": ["speaker"]
                }]
            }
        });
        let outcome = save_transcript(SaveInput {
            status: "ready",
            duration: Some(10.0),
            revision: 7,
            saved: Some(&document),
            body: &unchanged,
            now: 20.0,
            speaker_assignments: &assignments,
            voice_checks: &checks,
        })
        .unwrap();
        assert_eq!(outcome.revision, 8);
        assert_eq!(outcome.status, "ready");
        assert_eq!(outcome.message, "Saved locally.");
        let segment = &outcome.document.segments[0];
        assert_eq!(segment.alignment, Some(json!({"old": true})));
        assert_eq!(segment.audio_revision, Some(4));
        assert_eq!(
            segment.language_detection.as_ref().unwrap()["context_witnesses"],
            json!([{"owner":"server"}])
        );
        assert_eq!(
            segment.canonical_utterance_id.as_deref(),
            Some("canonical-one")
        );
        assert_eq!(segment.machine_revision, Some(2));
        assert_eq!(segment.protected_fields, Some(Vec::new()));
        assert_eq!(segment.refinement_state.as_deref(), Some("refined"));
        assert_eq!(segment.rest.get("language_generation"), Some(&json!(0)));
        assert!(segment.reading_turns.is_some());
        assert_eq!(outcome.document.warnings, vec!["keep me".to_string()]);

        let edited = json!({
            "revision": 7,
            "document": {
                "speakers": {"speaker_0": "Ada"},
                "segments": [{
                    "id": "one",
                    "start": 0.5,
                    "end": 1.0,
                    "speaker": "speaker_0",
                    "text": "Edited",
                    "audio_revision": 0,
                    "canonical_utterance_id": "forged",
                    "alignment": {"forged": true},
                    "language_generation": 99
                }]
            }
        });
        let outcome = save_transcript(SaveInput {
            status: "ready",
            duration: Some(10.0),
            revision: 7,
            saved: Some(&document),
            body: &edited,
            now: 21.0,
            speaker_assignments: &assignments,
            voice_checks: &checks,
        })
        .unwrap();
        let segment = &outcome.document.segments[0];
        assert_eq!(segment.text, "Edited");
        assert_eq!(segment.start, 0.5);
        assert!(segment.alignment.is_none());
        assert!(segment.reading_turns.is_none());
        assert!(segment.rest.get("text_audio_anchor").is_none());
        assert_eq!(segment.audio_revision, Some(4));
        assert_eq!(
            segment.canonical_utterance_id.as_deref(),
            Some("canonical-one")
        );
        assert_eq!(segment.rest.get("language_generation"), Some(&json!(0)));
        assert_eq!(
            segment.protected_fields,
            Some(vec!["start".to_string(), "text".to_string()])
        );
        assert_eq!(segment.machine_revision, Some(3));
        assert_eq!(segment.refinement_state.as_deref(), Some("edited"));
        assert_eq!(segment.rest.get("voice_eligible"), Some(&json!(false)));
        assert_eq!(segment.rest.get("speaker_candidates"), Some(&json!([])));
        assert_eq!(outcome.document.edited_at, Some(21.0));
        assert_eq!(outcome.message, "Saved locally.");
    }
}
