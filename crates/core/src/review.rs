//! Review reasons and retained unassigned audio.

use std::collections::HashSet;

use serde_json::{Map, Value, json};

use crate::document::{Document, Segment};
use crate::limits::TIME_EPSILON_SECONDS;

const UNASSIGNED_AUDIO: &str = "Audio outside detected speech; may be silence or missed speech";
const OVERLAPPING_SPEECH: &str = "Overlapping speakers; voices are not separated";
const EMPTY_RESULT: &str = "No transcript text returned; audio retained";
const UNCERTAIN: &str = "Language uncertain; audio retained";
const MULTIPLE_SPEAKERS: &str = "Multiple speakers; see localized turns where timing is available";
const SPEAKER_UNCERTAIN: &str = "Speaker uncertain";

fn known_reason(code: &str) -> Option<&'static str> {
    Some(match code {
        "uncertain" => UNCERTAIN,
        "needs_language" => "Choose the meeting language; audio retained",
        "recent_context" => "Language inferred from recent speech",
        "best_effort" => "Language uncertain; best-effort transcript",
        "change_pending" => "Language change uncertain; audio retained",
        "unsupported" => "Detected language unsupported; audio retained",
        "insufficient_speech" => "Too little usable speech; audio retained",
        "insufficient_acoustic_context" => "Too little acoustic context to decode; audio retained",
        "short_acoustic_context" => {
            "Unconfirmed words from a short audio fragment; review the recording"
        }
        "possible_non_speech" => "Audio may contain no speech; model words need review",
        "overlapping_speech" => OVERLAPPING_SPEECH,
        "empty_result" => EMPTY_RESULT,
        "transcription_failed" => "Transcription failed for this passage; audio retained",
        "token_limit" => "Transcript may be incomplete; audio retained",
        "unassigned_audio" => UNASSIGNED_AUDIO,
        "refinement_incomplete" => "Refinement incomplete; previous words and audio retained",
        "canonical_ownership_unresolved" => {
            "Word ownership unresolved; previous words and original audio retained"
        }
        "speech_admission_uncertain" => {
            "Speech detection uncertain; original audio retained for review"
        }
        _ => return None,
    })
}

/// `review.review_reason`.
pub fn review_reason(segment: &Segment) -> &'static str {
    if segment.separation.is_some()
        && segment.review_resolution.as_deref() != Some("words_reviewed")
    {
        return "Separated voice estimate; check words and speaker";
    }
    let speaker = segment.speaker.as_str();
    let mixed = speaker.starts_with("overlap");
    let text = segment.text.as_str();
    if segment.review_resolution.as_deref() == Some("words_reviewed") && !text.trim().is_empty() {
        if mixed {
            return OVERLAPPING_SPEECH;
        }
        if speaker == "unassigned" {
            return SPEAKER_UNCERTAIN;
        }
        return "";
    }
    if let Some(reason) = object_code(segment.transcription_review.as_ref()).and_then(known_reason)
    {
        return reason;
    }
    if speaker == "multiple_speakers" {
        return MULTIPLE_SPEAKERS;
    }
    if mixed {
        return OVERLAPPING_SPEECH;
    }
    let detection = segment
        .language_detection
        .as_ref()
        .and_then(Value::as_object);
    let mode_auto = detection
        .and_then(|object| object.get("mode"))
        .and_then(Value::as_str)
        == Some("auto");
    let detection_reason = detection
        .and_then(|object| object.get("reason"))
        .and_then(Value::as_str);
    if mode_auto {
        if let Some(code @ ("recent_context" | "best_effort")) = detection_reason {
            return known_reason(code).unwrap_or(UNCERTAIN);
        }
        if segment.language.as_deref().unwrap_or("").is_empty() {
            return detection_reason.and_then(known_reason).unwrap_or(UNCERTAIN);
        }
    }
    if text.trim().is_empty() {
        return EMPTY_RESULT;
    }
    if speaker == "unassigned" {
        return SPEAKER_UNCERTAIN;
    }
    ""
}

/// `review.export_text`.
pub fn export_text(segment: &Segment) -> String {
    let reason = review_reason(segment);
    if reason.is_empty() {
        return segment.text.clone();
    }
    let text = segment.text.as_str();
    let spacer = if text.trim().is_empty() { "" } else { " " };
    format!("{text}{spacer}[Needs review: {reason}]")
}

/// `review.retain_unassigned_audio`. Gaps are coverage, not detected speech.
pub fn retain_unassigned_audio(document: &mut Document, duration: f64) {
    if duration <= 0.0 {
        return;
    }
    let mut ordered: Vec<(f64, f64)> = document
        .segments
        .iter()
        .map(|segment| (segment.start, segment.end))
        .collect();
    ordered.sort_by(|left, right| left.0.total_cmp(&right.0).then(left.1.total_cmp(&right.1)));
    let mut cursor = 0.0;
    let mut gaps = Vec::new();
    for (start, end) in ordered {
        if start - cursor > TIME_EPSILON_SECONDS {
            gaps.push((cursor, start));
        }
        cursor = cursor.max(end);
    }
    if duration - cursor > TIME_EPSILON_SECONDS {
        gaps.push((cursor, duration));
    }
    if gaps.is_empty() {
        return;
    }
    document
        .speakers
        .insert("unassigned".to_string(), "Unassigned audio".to_string());
    let mut identifiers: HashSet<String> = document
        .segments
        .iter()
        .map(|segment| segment.id.clone())
        .collect();
    for (mut start, end) in gaps {
        while end - start > TIME_EPSILON_SECONDS {
            let mut sid = format!("unassigned-{}", identifiers.len());
            while identifiers.contains(&sid) {
                sid.push_str("-gap");
            }
            identifiers.insert(sid.clone());
            let stop = end.min(start + 30.0);
            document.segments.push(unassigned_segment(sid, start, stop));
            start = stop;
        }
    }
    document.segments.sort_by(|left, right| {
        left.start
            .total_cmp(&right.start)
            .then(left.end.total_cmp(&right.end))
    });
}

fn object_code(value: Option<&Value>) -> Option<&str> {
    value
        .and_then(Value::as_object)
        .and_then(|object| object.get("reason"))
        .and_then(Value::as_str)
}

fn unassigned_segment(id: String, start: f64, end: f64) -> Segment {
    let mut rest = Map::new();
    rest.insert("language".into(), Value::Null);
    rest.insert("voice_eligible".into(), Value::Bool(false));
    rest.insert("speaker_candidates".into(), json!([]));
    rest.insert("confidence".into(), Value::Null);
    rest.insert("timing".into(), Value::String("audio_coverage".into()));
    Segment {
        id,
        start,
        end,
        speaker: "unassigned".to_string(),
        text: String::new(),
        language: None,
        language_detection: None,
        transcription_review: Some(json!({
            "reason": "unassigned_audio",
            "partial_text": false,
        })),
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
        review: Some(true),
        capture_source: None,
        separation_candidate: None,
        separation: None,
        rest,
    }
}

#[cfg(test)]
mod tests {
    use super::{export_text, retain_unassigned_audio, review_reason};
    use crate::transcript::{bare_document, bare_segment};
    use serde_json::json;

    #[test]
    fn retain_unassigned_audio_on_a_tiny_document() {
        let mut document = bare_document(
            &[("speaker_0", "Speaker 1")],
            vec![bare_segment("known", 1.0, 2.0, "speaker_0", "Hello")],
        );
        retain_unassigned_audio(&mut document, 0.0);
        assert_eq!(document.segments.len(), 1);

        retain_unassigned_audio(&mut document, 3.0);
        assert_eq!(
            document.speakers.get("unassigned").map(String::as_str),
            Some("Unassigned audio")
        );
        assert_eq!(document.segments.len(), 3);
        assert_eq!(document.segments[0].id, "unassigned-1");
        assert_eq!(
            (document.segments[0].start, document.segments[0].end),
            (0.0, 1.0)
        );
        assert_eq!(document.segments[0].speaker, "unassigned");
        assert_eq!(document.segments[0].text, "");
        assert_eq!(document.segments[0].review, Some(true));
        assert_eq!(document.segments[0].language, None);
        assert_eq!(
            document.segments[0].rest.get("voice_eligible"),
            Some(&json!(false))
        );
        assert_eq!(
            document.segments[0].rest.get("speaker_candidates"),
            Some(&json!([]))
        );
        assert_eq!(
            document.segments[0].rest.get("confidence"),
            Some(&json!(null))
        );
        assert_eq!(
            document.segments[0]
                .rest
                .get("timing")
                .and_then(|value| value.as_str()),
            Some("audio_coverage")
        );
        assert_eq!(
            document.segments[0].transcription_review,
            Some(json!({"reason": "unassigned_audio", "partial_text": false}))
        );
        assert_eq!(
            review_reason(&document.segments[0]),
            "Audio outside detected speech; may be silence or missed speech"
        );
        assert_eq!(document.segments[1].id, "known");
        assert_eq!(document.segments[1].text, "Hello");
        assert_eq!(document.segments[2].id, "unassigned-2");
        assert_eq!(
            (document.segments[2].start, document.segments[2].end),
            (2.0, 3.0)
        );

        let mut again = document.clone();
        retain_unassigned_audio(&mut again, 3.0);
        assert_eq!(again, document);

        let mut wide = bare_document(
            &[("speaker_0", "Speaker 1")],
            vec![bare_segment("known", 0.0, 0.5, "speaker_0", "Hi")],
        );
        retain_unassigned_audio(&mut wide, 31.0);
        assert_eq!(
            wide.segments
                .iter()
                .map(|segment| (segment.id.as_str(), segment.start, segment.end))
                .collect::<Vec<_>>(),
            vec![
                ("known", 0.0, 0.5),
                ("unassigned-1", 0.5, 30.5),
                ("unassigned-2", 30.5, 31.0),
            ]
        );

        let mut collision = bare_document(
            &[("speaker_0", "Speaker 1")],
            vec![bare_segment("unassigned-1", 0.0, 1.0, "speaker_0", "Hi")],
        );
        retain_unassigned_audio(&mut collision, 2.0);
        assert_eq!(collision.segments[1].id, "unassigned-1-gap");
    }

    #[test]
    fn review_reason_uses_product_copy() {
        let mut segment = bare_segment("one", 0.0, 1.0, "speaker_0", "Hello");
        assert_eq!(review_reason(&segment), "");
        assert_eq!(export_text(&segment), "Hello");

        segment.text.clear();
        assert_eq!(
            review_reason(&segment),
            "No transcript text returned; audio retained"
        );
        assert_eq!(
            export_text(&segment),
            "[Needs review: No transcript text returned; audio retained]"
        );

        segment.text = "Partial".into();
        segment.transcription_review = Some(json!({"reason": "token_limit"}));
        assert_eq!(
            review_reason(&segment),
            "Transcript may be incomplete; audio retained"
        );
        assert_eq!(
            export_text(&segment),
            "Partial [Needs review: Transcript may be incomplete; audio retained]"
        );

        segment.transcription_review = None;
        segment.review_resolution = Some("words_reviewed".into());
        assert_eq!(review_reason(&segment), "");

        segment.speaker = "unassigned".into();
        assert_eq!(review_reason(&segment), "Speaker uncertain");
        segment.speaker = "overlap_speaker_0_speaker_1".into();
        assert_eq!(
            review_reason(&segment),
            "Overlapping speakers; voices are not separated"
        );

        segment.review_resolution = None;
        segment.speaker = "multiple_speakers".into();
        segment.text = "Hello there".into();
        assert_eq!(
            review_reason(&segment),
            "Multiple speakers; see localized turns where timing is available"
        );

        segment.speaker = "speaker_0".into();
        segment.language = Some("en".into());
        segment.language_detection = Some(json!({"mode": "auto", "reason": "recent_context"}));
        assert_eq!(
            review_reason(&segment),
            "Language inferred from recent speech"
        );

        segment.language = None;
        segment.language_detection = Some(json!({"mode": "auto", "reason": "nope"}));
        assert_eq!(
            review_reason(&segment),
            "Language uncertain; audio retained"
        );
    }
}
