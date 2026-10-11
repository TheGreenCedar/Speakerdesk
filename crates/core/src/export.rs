//! txt, srt, vtt, and json exports. Owned by the core-text agent.

use std::collections::BTreeMap;

use serde_json::Value;

use crate::document::{Document, Segment};
use crate::review::export_text;
use crate::transcript::{ExportTooLarge, bounded_text, json_truthy, timestamp};

/// One download body and its content type.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Export {
    pub body: String,
    pub content_type: &'static str,
}

/// Export current product text. Reading projections must still match their source.
pub fn export(document: &Document, kind: &str) -> Result<Export, ExportTooLarge> {
    if kind == "json" {
        let value = serde_json::to_value(document).expect("transcript document serializes");
        let json = to_pretty_json(&value);
        let body = bounded_text([json.as_str(), "\n"])?;
        return Ok(Export {
            body,
            content_type: "application/json",
        });
    }
    let lines = render_lines(document, kind);
    let rendered: Vec<String> = lines.into_iter().map(|line| format!("{line}\n")).collect();
    let body = bounded_text(rendered.iter().map(String::as_str))?;
    let content_type = if kind == "vtt" {
        "text/vtt; charset=utf-8"
    } else {
        "text/plain; charset=utf-8"
    };
    Ok(Export { body, content_type })
}

fn render_lines(document: &Document, kind: &str) -> Vec<String> {
    let mut lines = Vec::new();
    if kind == "vtt" {
        lines.push("WEBVTT".to_string());
        lines.push(String::new());
    }
    let mut emitted = kind == "vtt";
    let mut cue = 0usize;
    for segment in &document.segments {
        if segment.text.trim().is_empty() {
            continue;
        }
        let turns = local_turns(segment);
        let known = turns.iter().all(|turn| turn.attribution != "unknown");
        if !turns.is_empty() && (kind == "txt" || known) {
            for turn in turns {
                emitted = true;
                let slice = slice_segment(segment, &turn);
                push_passage(&mut lines, document, kind, &slice, &mut cue);
            }
        } else {
            emitted = true;
            push_passage(&mut lines, document, kind, segment, &mut cue);
        }
    }
    if !emitted {
        lines.push(String::new());
    }
    lines
}

fn local_turns(segment: &Segment) -> Vec<crate::timing::ReadingTurn> {
    let Ok(value) = serde_json::to_value(segment) else {
        return Vec::new();
    };
    let Ok(mut reading) = serde_json::from_value::<crate::timing::ReadingSegment>(value) else {
        return Vec::new();
    };
    reading.installed_provider_identity = Some(crate::alignment::NATIVE_PROVIDER.into());
    crate::timing::validated_turns(&reading)
}

fn slice_segment(parent: &Segment, turn: &crate::timing::ReadingTurn) -> Segment {
    let mut slice = parent.clone();
    slice.text = turn.text.clone();
    slice.speaker = turn.speaker.clone();
    slice.start = turn.start.unwrap_or(parent.start);
    slice.end = turn.end.unwrap_or(parent.end);
    slice.review = Some(false);
    slice
        .rest
        .insert("reading_slice".to_string(), Value::Bool(true));
    slice.rest.insert(
        "coarse_timing".to_string(),
        Value::Bool(turn.attribution != "unknown"),
    );
    slice
        .rest
        .insert("speaker_candidates".to_string(), Value::Array(Vec::new()));
    slice
}

fn push_passage(
    lines: &mut Vec<String>,
    document: &Document,
    kind: &str,
    passage: &Segment,
    cue: &mut usize,
) {
    let label = speaker_label(&document.speakers, &passage.speaker);
    let passage_text = export_text(passage);
    if kind == "txt" {
        let timing = timing_prefix(passage);
        lines.push(format!(
            "[{timing}{} \u{2013} {}] {label}: {passage_text}",
            timestamp(passage.start, "."),
            timestamp(passage.end, ".")
        ));
        return;
    }
    *cue += 1;
    let separator = if kind == "srt" { "," } else { "." };
    let text = sanitize_subtitle(&passage_text);
    let name = sanitize_subtitle(&label);
    lines.push(cue.to_string());
    lines.push(format!(
        "{} --> {}",
        timestamp(passage.start, separator),
        timestamp(passage.end, separator)
    ));
    lines.push(format!("{name}: {text}"));
    lines.push(String::new());
}

fn speaker_label(speakers: &BTreeMap<String, String>, speaker: &str) -> String {
    if let Some(name) = speakers.get(speaker) {
        return name.clone();
    }
    if speaker == "unassigned" {
        "Unknown speaker".to_string()
    } else {
        "Overlapping speakers".to_string()
    }
}

fn timing_prefix(segment: &Segment) -> &'static str {
    if json_truthy(segment.rest.get("coarse_timing")) {
        "Coarse timing "
    } else if json_truthy(segment.rest.get("reading_slice")) {
        "Timing unknown; parent audio "
    } else {
        ""
    }
}

fn sanitize_subtitle(text: &str) -> String {
    text.split_whitespace()
        .collect::<Vec<_>>()
        .join(" ")
        .replace("-->", "\u{2192}")
        .replace('<', "\u{2039}")
        .replace('>', "\u{203a}")
}

fn to_pretty_json(value: &Value) -> String {
    let mut out = String::new();
    write_json(&mut out, value, 0);
    out
}

fn write_json(out: &mut String, value: &Value, indent: usize) {
    match value {
        Value::Null => out.push_str("null"),
        Value::Bool(true) => out.push_str("true"),
        Value::Bool(false) => out.push_str("false"),
        Value::Number(number) => out.push_str(&number.to_string()),
        Value::String(text) => push_string(out, text),
        Value::Array(items) if items.is_empty() => out.push_str("[]"),
        Value::Array(items) => {
            out.push_str("[\n");
            for (index, item) in items.iter().enumerate() {
                if index > 0 {
                    out.push_str(",\n");
                }
                push_indent(out, indent + 1);
                write_json(out, item, indent + 1);
            }
            out.push('\n');
            push_indent(out, indent);
            out.push(']');
        }
        Value::Object(map) if map.is_empty() => out.push_str("{}"),
        Value::Object(map) => {
            out.push_str("{\n");
            for (index, (key, item)) in map.iter().enumerate() {
                if index > 0 {
                    out.push_str(",\n");
                }
                push_indent(out, indent + 1);
                push_string(out, key);
                out.push_str(": ");
                write_json(out, item, indent + 1);
            }
            out.push('\n');
            push_indent(out, indent);
            out.push('}');
        }
    }
}

fn push_indent(out: &mut String, level: usize) {
    for _ in 0..level {
        out.push_str("  ");
    }
}

fn push_string(out: &mut String, text: &str) {
    out.push('"');
    for ch in text.chars() {
        match ch {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\u{0008}' => out.push_str("\\b"),
            '\u{000c}' => out.push_str("\\f"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            control if (control as u32) < 0x20 => {
                out.push_str(&format!("\\u{code:04x}", code = control as u32));
            }
            other => out.push(other),
        }
    }
    out.push('"');
}

#[cfg(test)]
mod tests {
    use super::export;
    use crate::limits::MAX_EXPORT_BYTES;
    use crate::messages::EXPORT_TOO_LARGE;
    use crate::transcript::{bare_document, bare_segment};
    use serde_json::json;

    fn cafe_document() -> crate::document::Document {
        bare_document(
            &[("speaker_0", "Zoë")],
            vec![
                bare_segment("words", 0.0, 1.0, "speaker_0", "Checked café 日本語"),
                bare_segment("blank", 1.0, 2.0, "speaker_0", ""),
            ],
        )
    }

    #[test]
    fn export_txt_srt_vtt_json_for_one_document() {
        let document = cafe_document();
        let before = document.clone();
        let txt = export(&document, "txt").unwrap();
        let srt = export(&document, "srt").unwrap();
        let vtt = export(&document, "vtt").unwrap();
        let json_export = export(&document, "json").unwrap();
        assert_eq!(document, before);

        assert_eq!(txt.content_type, "text/plain; charset=utf-8");
        assert_eq!(
            txt.body,
            "[00:00:00.000 \u{2013} 00:00:01.000] Zoë: Checked café 日本語\n"
        );
        assert!(!txt.body.contains("blank"));

        assert_eq!(srt.content_type, "text/plain; charset=utf-8");
        assert_eq!(
            srt.body,
            "1\n00:00:00,000 --> 00:00:01,000\nZoë: Checked café 日本語\n\n"
        );

        assert_eq!(vtt.content_type, "text/vtt; charset=utf-8");
        assert_eq!(
            vtt.body,
            "WEBVTT\n\n1\n00:00:00.000 --> 00:00:01.000\nZoë: Checked café 日本語\n\n"
        );

        assert_eq!(json_export.content_type, "application/json");
        assert!(json_export.body.ends_with('\n'));
        assert!(json_export.body.contains("Checked café 日本語"));
        let parsed: serde_json::Value = serde_json::from_str(json_export.body.trim_end()).unwrap();
        assert_eq!(parsed["speakers"]["speaker_0"], "Zoë");
        assert_eq!(parsed["segments"][0]["text"], "Checked café 日本語");
        assert_eq!(parsed["segments"][1]["id"], "blank");
        assert_eq!(parsed["segments"][1]["text"], "");
        assert_eq!(parsed["schema_version"], 1);

        let empty = bare_document(&[("speaker_0", "Voice")], Vec::new());
        assert_eq!(export(&empty, "txt").unwrap().body, "\n");
        assert_eq!(export(&empty, "srt").unwrap().body, "\n");
        assert_eq!(export(&empty, "vtt").unwrap().body, "WEBVTT\n\n");
    }

    #[test]
    fn export_respects_inclusive_16_mib_cap() {
        let sample = export(&with_text("a"), "txt").unwrap();
        let overhead = sample.body.len() - 1;
        let room = MAX_EXPORT_BYTES - overhead;
        let exact = export(&with_text(&"a".repeat(room)), "txt").unwrap();
        assert_eq!(exact.body.len(), MAX_EXPORT_BYTES);

        let over = "a".repeat(room + 1);
        for kind in ["txt", "srt", "vtt", "json"] {
            let error = export(&with_text(&over), kind).unwrap_err();
            assert_eq!(error.to_string(), EXPORT_TOO_LARGE);
        }

        let json_sample = export(&with_text("q"), "json").unwrap();
        let json_overhead = json_sample.body.len() - 1;
        let json_room = MAX_EXPORT_BYTES - json_overhead;
        let json_exact = export(&with_text(&"q".repeat(json_room)), "json").unwrap();
        assert_eq!(json_exact.body.len(), MAX_EXPORT_BYTES);
        let json_over = export(&with_text(&"q".repeat(json_room + 1)), "json").unwrap_err();
        assert_eq!(json_over.to_string(), EXPORT_TOO_LARGE);
    }

    #[test]
    fn export_projects_known_reading_turns() {
        let cases: Vec<crate::document::Segment> =
            serde_json::from_str(include_str!("../tests/testdata/reading-turns.json")).unwrap();
        let segment = cases[0].clone();
        let document = bare_document(
            &[
                ("multiple_speakers", "Multiple speakers"),
                ("speaker_0", "Ada"),
                ("speaker_1", "Bea"),
            ],
            vec![segment],
        );
        let txt = export(&document, "txt").unwrap();
        assert_eq!(
            txt.body,
            "[Coarse timing 00:00:00.000 \u{2013} 00:00:00.400] Ada: Hello \n\
             [Coarse timing 00:00:00.400 \u{2013} 00:00:01.000] Bea: there\n"
        );
        let srt = export(&document, "srt").unwrap();
        assert_eq!(srt.body.matches("-->").count(), 2);
        assert!(srt.body.contains("Ada: Hello\n"));
        assert!(srt.body.contains("Bea: there\n"));

        let mixed = cases[1].clone();
        let mixed_document = bare_document(
            &[
                ("multiple_speakers", "Multiple speakers"),
                ("speaker_0", "Ada"),
                ("speaker_1", "Bea"),
            ],
            vec![mixed],
        );
        let mixed_txt = export(&mixed_document, "txt").unwrap();
        assert_eq!(
            mixed_txt.body,
            "[Coarse timing 00:00:00.000 \u{2013} 00:00:00.400] Ada: Hello \n\
             [Timing unknown; parent audio 00:00:00.000 \u{2013} 00:00:01.000] Unknown speaker: there [Needs review: Speaker uncertain]\n"
        );
        let mixed_srt = export(&mixed_document, "srt").unwrap();
        assert_eq!(mixed_srt.body.matches("-->").count(), 1);
        assert!(mixed_srt.body.contains(
            "Multiple speakers: Hello there [Needs review: Multiple speakers; see localized turns where timing is available]"
        ));

        let mut absent = bare_segment("one", 0.0, 1.0, "speaker_0", "Hello there");
        absent.reading_turns = None;
        let parent = bare_document(&[("speaker_0", "Ada")], vec![absent]);
        assert_eq!(
            export(&parent, "txt").unwrap().body,
            "[00:00:00.000 \u{2013} 00:00:01.000] Ada: Hello there\n"
        );
    }

    #[test]
    fn export_preserves_source_qualified_reading_turns() {
        let cases: Vec<crate::document::Segment> =
            serde_json::from_str(include_str!("../tests/testdata/reading-turns.json")).unwrap();
        let mut segment = cases[0].clone();
        let source =
            json!({"catalog_sha256": "abc", "source_id": "microphone", "source_revision": 1});
        segment.capture_source = Some(source.clone());
        segment.reading_turn_provenance.as_mut().unwrap()["capture_source"] = source;
        let document = bare_document(&[("speaker_0", "Ada"), ("speaker_1", "Bea")], vec![segment]);
        let body = export(&document, "srt").unwrap().body;
        assert!(body.contains("Ada: Hello\n"));
        assert!(body.contains("Bea: there\n"));
        assert_eq!(body.matches("-->").count(), 2);
    }

    fn with_text(text: &str) -> crate::document::Document {
        bare_document(
            &[("speaker_0", "Zoë")],
            vec![bare_segment("words", 0.0, 1.0, "speaker_0", text)],
        )
    }

    #[test]
    fn stale_reading_turns_export_current_text_and_bounds() {
        let cases: Vec<crate::document::Segment> =
            serde_json::from_str(include_str!("../tests/testdata/reading-turns.json")).unwrap();
        for mutation in [
            "text",
            "protected",
            "bounds",
            "revision",
            "speaker",
            "missing_receipt",
        ] {
            let mut segment = cases[0].clone();
            match mutation {
                "text" => segment.text = "Corrected café 日本語".into(),
                "protected" => segment.protected_fields = Some(vec!["text".into()]),
                "bounds" => {
                    segment.start = 3.0;
                    segment.end = 4.0;
                }
                "revision" => segment.audio_revision = Some(2),
                "speaker" => {
                    segment.reading_turns.as_mut().unwrap()[0]["speaker"] = json!("speaker_9")
                }
                "missing_receipt" => segment.reading_turn_provenance = None,
                _ => unreachable!(),
            }
            let expected_text = segment.text.clone();
            let expected_start = crate::transcript::timestamp(segment.start, ".");
            let document =
                bare_document(&[("multiple_speakers", "Current speaker")], vec![segment]);
            for kind in ["txt", "srt", "vtt"] {
                let output = export(&document, kind).unwrap().body;
                assert!(
                    output.contains(&expected_text),
                    "{mutation}/{kind}: {output}"
                );
                assert!(
                    output.replace(',', ".").contains(&expected_start),
                    "{mutation}/{kind}: {output}"
                );
                assert!(!output.contains("speaker_9"), "{mutation}/{kind}: {output}");
            }
        }
    }

    #[test]
    fn subtitle_text_is_sanitized() {
        let document = bare_document(
            &[("speaker_0", "A <B>")],
            vec![bare_segment("one", 0.0, 1.0, "speaker_0", "go\n-->\nnext")],
        );
        let srt = export(&document, "srt").unwrap();
        assert!(srt.body.contains("A \u{2039}B\u{203a}: go \u{2192} next\n"));
    }
}
