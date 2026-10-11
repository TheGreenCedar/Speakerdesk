//! Port of `speakerdesk/review.py` — review-flag semantics.
//!
//! Explain unresolved speech without inventing ASR confidence or transcript
//! words. Review values explain *why* a passage is flagged; explicit
//! `review=false` marks are preserved verbatim.

use crate::model::{Document, Nullable, ReviewResolution, Segment, TranscriptionReview};

/// `review.REASONS` — exact v0.6.6 strings.
fn reasons(key: &str) -> Option<&'static str> {
    Some(match key {
        "uncertain" => "Language uncertain; audio retained",
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
        "overlapping_speech" => "Overlapping speakers; voices are not separated",
        "empty_result" => "No transcript text returned; audio retained",
        "transcription_failed" => "Transcription failed for this passage; audio retained",
        "token_limit" => "Transcript may be incomplete; audio retained",
        "unassigned_audio" => "Audio outside detected speech; may be silence or missed speech",
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

/// Port of `review.review_reason`. Returns "" for no reason (Python returns
/// `''`, falsy like `None`).
pub fn review_reason(segment: &Segment) -> String {
    let speaker = segment.speaker.as_str();
    let mixed = speaker.starts_with("overlap");
    let multiple = speaker == "multiple_speakers";
    if segment.review_resolution == Some(ReviewResolution::WordsReviewed)
        && !segment.text.trim().is_empty()
    {
        return if mixed {
            reasons("overlapping_speech").unwrap().to_string()
        } else if speaker == "unassigned" {
            "Speaker uncertain".to_string()
        } else {
            String::new()
        };
    }
    if let Some(text) = segment
        .transcription_review
        .as_ref()
        .and_then(|t| t.reason.as_deref())
        .and_then(reasons)
    {
        return text.to_string();
    }
    if multiple {
        return "Multiple speakers; see localized turns where timing is available".to_string();
    }
    if mixed {
        return reasons("overlapping_speech").unwrap().to_string();
    }
    let detection = segment.language_detection.as_ref();
    let mode = detection.and_then(|d| d.mode.as_deref());
    let reason_key = detection.and_then(|d| d.reason.as_deref());
    if mode == Some("auto") && matches!(reason_key, Some("recent_context" | "best_effort")) {
        return reasons(reason_key.unwrap()).unwrap().to_string();
    }
    let language_present = matches!(&segment.language, Nullable::Value(l) if !l.is_empty());
    if mode == Some("auto") && !language_present {
        return reason_key
            .and_then(reasons)
            .unwrap_or_else(|| reasons("uncertain").unwrap())
            .to_string();
    }
    if segment.text.trim().is_empty() {
        return reasons("empty_result").unwrap().to_string();
    }
    if speaker == "unassigned" {
        return "Speaker uncertain".to_string();
    }
    String::new()
}

/// Port of `review.export_text` — text plus an optional review annotation.
pub fn export_text(segment: &Segment) -> String {
    let text = segment.text.as_str();
    let reason = review_reason(segment);
    if reason.is_empty() {
        text.to_string()
    } else {
        format!(
            "{}{}[Needs review: {reason}]",
            text,
            if text.trim().is_empty() { "" } else { " " },
        )
    }
}

/// Port of `review.retain_unassigned_audio` — account for audio outside
/// returned passages without calling it speech. Mutates `document` in place.
pub fn retain_unassigned_audio(document: &mut Document, duration: f64) {
    if duration <= 0.0 {
        return;
    }
    let mut cursor = 0.0f64;
    let mut gaps: Vec<(f64, f64)> = Vec::new();
    let mut ordered: Vec<(f64, f64)> = document.segments.iter().map(|s| (s.start, s.end)).collect();
    ordered.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
    for (start, end) in ordered {
        if start - cursor > 0.001 {
            gaps.push((cursor, start));
        }
        cursor = cursor.max(end);
    }
    if duration - cursor > 0.001 {
        gaps.push((cursor, duration));
    }
    if gaps.is_empty() {
        return;
    }
    // Python `{s['id'] for s in segments}` — a set, so dedupe.
    let mut identifiers: Vec<String> = Vec::new();
    for s in &document.segments {
        if !identifiers.iter().any(|id| id == &s.id) {
            identifiers.push(s.id.clone());
        }
    }
    document
        .speakers
        .insert("unassigned".to_string(), "Unassigned audio".to_string());
    for (mut start, end) in gaps {
        while end - start > 0.001 {
            let mut sid = format!("unassigned-{}", identifiers.len());
            while identifiers.iter().any(|id| id == &sid) {
                sid += "-gap";
            }
            identifiers.push(sid.clone());
            let stop = (start + 30.0).min(end);
            document.segments.push(Segment {
                id: sid,
                start,
                end: stop,
                speaker: "unassigned".to_string(),
                text: String::new(),
                language: Nullable::Null,
                review: Some(true),
                voice_eligible: Some(false),
                speaker_candidates: Some(Vec::new()),
                confidence: Nullable::Null,
                timing: Some("audio_coverage".to_string()),
                transcription_review: Some(TranscriptionReview {
                    reason: Some("unassigned_audio".to_string()),
                    partial_text: Some(false),
                }),
                ..Default::default()
            });
            start = stop;
        }
    }
    document.segments.sort_by(|a, b| {
        (a.start, a.end)
            .partial_cmp(&(b.start, b.end))
            .unwrap_or(std::cmp::Ordering::Equal)
    });
}
