//! Port of the v0.6.6 PUT `/api/jobs/<jid>/transcript` and PATCH
//! `/api/jobs/<jid>/segments/<sid>` bodies (app.py). Pure functions over the
//! typed `Job`; the caller persists. The clock is injected.
//!
//! Client input is typed (`TranscriptPut`, `SegmentPatch`). PATCH inspects
//! the raw `changes` map first so the forbidden-field 400 keeps its exact
//! v0.6.6 message; PUT ignores unknown keys. Validation runs through
//! `transcript::validate`, which returns the typed `Document`.
//!
//! Product rule (field ledger): `reading_turns` present implies valid, and
//! an edit that changes a passage's text, start or end removes them — which
//! the server-owned-field handling below already does verbatim.

use std::collections::BTreeSet;

use serde_json::{Map, Value, json};

use crate::error::Error;
use crate::model::{
    Document, DocumentProvenance, Job, JobStatus, Nullable, ProvenanceKind, ReviewResolution,
    Segment,
};
use crate::transcript::validate;

/// Typed PUT `/transcript` input. `revision` stays a `Value` because v0.6.6
/// compares it to the stored revision with Python `!=` (numeric equality
/// across int/float); `document` stays raw until `validate` converts it
/// (Python coerces numeric strings); `imported` is Python-truthy.
#[derive(Debug, Clone, PartialEq)]
pub struct TranscriptPut {
    pub revision: Value,
    pub document: Value,
    pub imported: Value,
}

impl TranscriptPut {
    /// Deserialize a request body, ignoring unknown keys.
    pub fn from_value(body: &Value) -> Self {
        Self {
            revision: body.get("revision").cloned().unwrap_or(Value::Null),
            document: body.get("document").cloned().unwrap_or(Value::Null),
            imported: body.get("imported").cloned().unwrap_or(Value::Null),
        }
    }
}

/// Typed PATCH `/segments/<sid>` input. `changes` is the bounded raw edit
/// map (keys verified ⊆ {text, speaker, start, end} at construction);
/// `segment_revision` compares numerically like v0.6.6.
#[derive(Debug, Clone, PartialEq)]
pub struct SegmentPatch {
    pub segment_revision: Value,
    pub changes: Map<String, Value>,
}

impl SegmentPatch {
    /// Deserialize a request body with the v0.6.6 forbidden-field checks.
    /// The raw `changes` map is inspected before any typed conversion so
    /// disallowed keys keep the exact 400 message.
    pub fn from_value(body: &Value) -> Result<Self, Error> {
        if !body.is_object() || !body.get("changes").is_some_and(Value::is_object) {
            return Err(Error::bad_request("Supply passage edits."));
        }
        let changes = body["changes"].as_object().unwrap();
        let editable = ["text", "speaker", "start", "end"];
        if changes.is_empty() || changes.keys().any(|k| !editable.contains(&k.as_str())) {
            return Err(Error::bad_request(
                "Only words, speaker and passage bounds can be edited.",
            ));
        }
        Ok(Self {
            segment_revision: body.get("segment_revision").cloned().unwrap_or(Value::Null),
            changes: changes.clone(),
        })
    }
}

fn truthy(value: &Value) -> bool {
    match value {
        Value::Null => false,
        Value::Bool(b) => *b,
        Value::Number(n) => n.as_f64() != Some(0.0),
        Value::String(s) => !s.is_empty(),
        Value::Array(a) => !a.is_empty(),
        Value::Object(o) => !o.is_empty(),
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

/// `voice_profiles.speaker_audio_reason` — why saved source audio cannot
/// establish this track's identity, or `None` when eligible.
fn speaker_audio_reason<'a>(segment: &Segment, track_id: &str) -> Option<&'a str> {
    if segment.speaker != track_id {
        return Some("different_speaker");
    }
    if segment.audio_state.as_ref().is_some_and(|s| !s.is_empty()) {
        return Some("unverified_audio");
    }
    if segment.finalized == Some(false) {
        return Some("unfinished");
    }
    let candidates: Vec<&str> = segment
        .speaker_candidates
        .as_deref()
        .map(|list| list.iter().map(String::as_str).collect())
        .unwrap_or_else(|| vec![track_id]);
    if track_id.starts_with("overlap") || candidates.len() > 1 {
        return Some("overlap");
    }
    // `segment.get('voice_eligible', not segment.get('review')) is True`
    let eligible = segment
        .voice_eligible
        .unwrap_or(!segment.review.unwrap_or(false));
    if candidates.len() != 1 || !eligible {
        return Some("unverified_audio");
    }
    None
}

/// `voice_profiles.speaker_audio_eligible`.
fn speaker_audio_eligible(segment: &Segment, track_id: &str) -> bool {
    speaker_audio_reason(segment, track_id).is_none()
}

/// `people.reconcile_assignments` — ordinary corrections affect this
/// meeting; they never rename/enroll People.
fn reconcile_assignments(job: &mut Job, document: &Document, imported: bool) {
    let speakers = &document.speakers;
    let keep = job
        .speaker_assignments
        .take()
        .map(|assignments| {
            assignments
                .into_iter()
                .filter(|(track, value)| !imported && speakers.get(track) == Some(&value.name))
                .collect()
        })
        .unwrap_or_default();
    job.speaker_assignments = Some(keep);
    // Text/timing/track edits invalidate clip evidence.
    job.voice_suggestions = None;
    if let Some(checks) = job.voice_checks.as_mut() {
        for check in checks.values_mut() {
            if check.status == "checking" {
                check.status = "cancelled".to_string();
            }
        }
    }
}

/// PATCH `/api/jobs/<jid>/segments/<sid>` — merge `changes` into one
/// passage. `job` is mutated in place; the host reads the updated segment
/// (and `job.revision`) back for the response body.
pub fn patch_segment(job: &mut Job, sid: &str, patch: SegmentPatch, now: f64) -> Result<(), Error> {
    let duration = job.duration.unwrap_or(0.0);
    if job.document.is_none() || job.status.blocks_patch() {
        return Err(Error::conflict("Wait for transcript passages."));
    }
    let document = job.document.as_mut().unwrap();
    let Some(position) = document.segments.iter().position(|s| s.id == sid) else {
        // Flask `abort(404)` — the route renders the default description.
        return Err(Error::not_found(
            "The requested URL was not found on the server. If you entered the URL manually please check your spelling and try again.",
        ));
    };
    // Python `body.get('segment_revision') != row.get('machine_revision', 0)`;
    // serde_json Number equality is numeric like Python's.
    let expected_revision = json!(document.segments[position].machine_revision.unwrap_or(0));
    if patch.segment_revision != expected_revision {
        return Err(Error::conflict(
            "This passage changed. Review its latest words before saving.",
        ));
    }
    let prior = document.segments[position].clone();

    // Apply the raw change values with the same coercion `validate` would
    // perform, producing its messages. Validate's per-segment check order is
    // times first, then speaker/text — mirror that here. Only this row can
    // fail: the stored document was already valid.
    let index = position + 1;
    for key in ["start", "end"] {
        if let Some(v) = patch.changes.get(key) {
            let Some(number) = python_float(v) else {
                return Err(Error::bad_request(format!(
                    "Segment {index}: supply numeric start and end times.",
                )));
            };
            if key == "start" {
                document.segments[position].start = number;
            } else {
                document.segments[position].end = number;
            }
        }
    }
    for key in ["speaker", "text"] {
        if let Some(v) = patch.changes.get(key) {
            let Some(text) = v.as_str() else {
                return Err(Error::bad_request(format!(
                    "Segment {index}: invalid speaker or text.",
                )));
            };
            if key == "speaker" {
                document.segments[position].speaker = text.to_string();
            } else {
                document.segments[position].text = text.to_string();
            }
        }
    }

    // Re-validate the whole document (sort, speakers, bounds, schema
    // version) exactly like the route.
    let document_value = serde_json::to_value(&*document).unwrap_or_default();
    let mut validated = validate(&document_value, duration)?;
    let row = validated.segments.iter_mut().find(|s| s.id == sid).unwrap();

    let mut protected: BTreeSet<String> =
        prior.protected_fields.iter().flatten().cloned().collect();
    for key in patch.changes.keys() {
        protected.insert(key.clone());
    }
    row.protected_fields = Some(protected.into_iter().collect());
    row.machine_revision = Some(prior.machine_revision.unwrap_or(0) + 1);
    row.refinement_state = Nullable::of("edited".to_string());
    // Every attachment binds the old machine revision, including a
    // speaker-only edit. A client cannot reuse it for the new revision.
    row.reading_turns = None;
    // `row[k] != prior[k]` — a no-op change (same value) keeps anchors.
    if row.start != prior.start || row.end != prior.end {
        row.text_audio_anchor = None;
    }
    if row.speaker != prior.speaker || row.start != prior.start || row.end != prior.end {
        row.voice_eligible = Some(false);
        row.speaker_candidates = Some(Vec::new());
        row.retry_candidate = None;
    }
    if !row.text.trim().is_empty() && !row.speaker.starts_with("overlap") {
        row.review_resolution = Some(ReviewResolution::WordsReviewed);
        row.review = Some(false);
    }
    validated.edited_at = Some(now);
    job.document = Some(validated);
    job.revision += 1;
    Ok(())
}

/// Restore a server-owned segment field from the stored row: the client's
/// value is replaced by `prior`'s (or removed when `prior` lacks it).
macro_rules! restore {
    ($seg:expr, $prior:expr, $($field:ident),+ $(,)?) => {
        $(
            $seg.$field = $prior.and_then(|p| p.$field.clone());
        )+
    };
}

/// PUT `/api/jobs/<jid>/transcript` — whole-transcript save. `job` is
/// mutated in place; the host serializes `job` for the response body.
pub fn put_transcript(job: &mut Job, edit: TranscriptPut, now: f64) -> Result<(), Error> {
    // Python `status in ACTIVE or not job.get('duration')`.
    if job.status.is_active() || job.duration.unwrap_or(0.0) == 0.0 {
        return Err(Error::conflict(
            "Wait for audio preparation and inference to finish.",
        ));
    }
    // Python `body.get('revision') != job['revision']` — numeric equality.
    if edit.revision != json!(job.revision) {
        return Err(Error::conflict(
            "This transcript changed in another tab. Reload before saving.",
        ));
    }
    let duration = job.duration.unwrap_or(0.0);
    let mut incoming = validate(&edit.document, duration)?;
    let imported = truthy(&edit.imported);

    let original: std::collections::BTreeMap<String, Segment> = job
        .document
        .as_ref()
        .map(|d| {
            d.segments
                .iter()
                .map(|s| (s.id.clone(), s.clone()))
                .collect()
        })
        .unwrap_or_default();

    for segment in &mut incoming.segments {
        let prior = original.get(&segment.id);
        let protected: Vec<String> = prior
            .and_then(|p| p.protected_fields.clone())
            .unwrap_or_default();
        let edited: Vec<&'static str> = {
            let mut v = Vec::new();
            if let Some(prior) = prior {
                // Python `segment[k] != prior[k]`; both exist post-validate.
                if segment.text != prior.text {
                    v.push("text");
                }
                if segment.speaker != prior.speaker {
                    v.push("speaker");
                }
                if segment.start != prior.start {
                    v.push("start");
                }
                if segment.end != prior.end {
                    v.push("end");
                }
            }
            v
        };
        // Server-owned keys: the client's value is dropped and the stored
        // one restored (drop-class keys — `source_start`, `fast_origin_sample`,
        // `alignment`, `assembly_provenance`, `bounded_decode_provenance`,
        // `reading_turn_provenance`, `capture_audio`, `source_speaker_candidates`
        // — are gone from the model entirely).
        restore!(
            segment,
            prior,
            activity_regions,
            audio_anchor,
            audio_revision,
            audio_state,
            canonical_machine_revision,
            canonical_state,
            canonical_unresolved,
            canonical_utterance_id,
            capture_source,
            end_sample,
            finalized,
            language_epoch,
            language_generation,
            language_mode,
            reading_turns,
            refinement_window,
            speaker_activity,
            speech_regions,
            start_sample,
            text_audio_anchor,
        );
        if let Some(prior) = prior
            .filter(|p| p.machine_revision.is_some() || !protected.is_empty() || !edited.is_empty())
        {
            let mut merged: BTreeSet<String> = protected.iter().cloned().collect();
            for key in &edited {
                merged.insert(key.to_string());
            }
            segment.protected_fields = Some(merged.into_iter().collect());
            let bump: i64 = i64::from(!edited.is_empty());
            segment.machine_revision = Some(prior.machine_revision.unwrap_or(0) + bump);
            // `seg['refinement_state'] = prior.get('refinement_state')` —
            // v0.6.6 writes an explicit `null` when the prior row lacks it.
            segment.refinement_state = if !protected.is_empty() || !edited.is_empty() {
                Nullable::of("edited".to_string())
            } else {
                prior.refinement_state.clone()
            };
        } else {
            segment.protected_fields = None;
            segment.machine_revision = None;
            segment.refinement_state = Nullable::Absent;
        }
        if !edited.is_empty() || imported {
            segment.reading_turns = None;
        }
        if edited.iter().any(|k| ["start", "end"].contains(k)) || imported {
            segment.text_audio_anchor = None;
        }
        // Client edits/imports cannot manufacture server-owned clean-audio
        // evidence.
        let prior_speaker = prior.map(|p| p.speaker.as_str()).unwrap_or("");
        let unchanged = prior.is_some()
            && !imported
            && segment.speaker == prior_speaker
            && segment.start == prior.unwrap().start
            && segment.end == prior.unwrap().end;
        segment.voice_eligible =
            Some(unchanged && speaker_audio_eligible(prior.unwrap(), prior_speaker));
        segment.speaker_candidates = Some(if unchanged {
            prior
                .unwrap()
                .speaker_candidates
                .clone()
                .unwrap_or_else(|| vec![prior_speaker.to_string()])
        } else {
            Vec::new()
        });
        // A candidate belongs to its saved audio/track. Client edits cannot
        // retain or manufacture a retry for different source bounds.
        segment.retry_candidate = None;
        if unchanged && let Some(candidate) = prior.and_then(|p| p.retry_candidate.clone()) {
            let mut candidate = candidate;
            // `segment['text'] == candidate.get('text') and != prior['text']`:
            // absent candidate text compares unequal to a present string.
            if candidate.text.as_ref() == Some(&segment.text) && segment.text != prior.unwrap().text
            {
                candidate.replaced_text = Some(prior.unwrap().text.clone());
                candidate.accepted_at = Some(now);
            }
            segment.retry_candidate = Some(candidate);
        }
    }

    // Provenance is server-owned. Editing cannot pretend to be fresh
    // inference.
    let mut document = if imported {
        Document {
            schema_version: Some(1),
            speakers: incoming.speakers,
            segments: incoming.segments,
            provenance: Some(DocumentProvenance {
                kind: Some(ProvenanceKind::Imported),
                timing: Some(
                    "User imported segment times; source unverified".to_string(),
                ),
                ..Default::default()
            }),
            warnings: Some(vec![
                "Imported transcript. Text, speaker labels and timing have not been verified by this app."
                    .to_string(),
            ]),
            edited_at: None,
        }
    } else if job.document.is_some() {
        let mut document = job.document.take().unwrap();
        document.speakers = incoming.speakers;
        document.segments = incoming.segments;
        document
    } else {
        Document {
            schema_version: Some(1),
            speakers: incoming.speakers,
            segments: incoming.segments,
            provenance: Some(DocumentProvenance {
                kind: Some(ProvenanceKind::Manual),
                timing: Some("User supplied recording times".to_string()),
                ..Default::default()
            }),
            warnings: Some(vec![
                "This transcript contains manually entered or imported text and timing."
                    .to_string(),
            ]),
            edited_at: None,
        }
    };
    document.edited_at = Some(now);
    reconcile_assignments(job, &document, imported);
    job.document = Some(document);
    job.revision += 1;
    job.status = JobStatus::Ready;
    job.message = Some("Saved locally.".to_string());
    Ok(())
}
