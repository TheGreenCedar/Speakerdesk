//! People records, introduction suggestions, and meeting-assignment reconciliation.
//!
//! Rename updates the People record only. Meeting name snapshots stay with the
//! assignment until [`reconcile_assignments`] drops them. Voice enrollment never
//! creates a person.

use std::collections::{BTreeMap, BTreeSet};
use std::sync::LazyLock;

use regex::Regex;
use serde::Deserialize;
use sha2::{Digest, Sha256};
use thiserror::Error;
use uuid::Uuid;

use crate::limits;
use crate::messages;

/// Stored on a profile only after an explicit Remember voice action.
pub const EXPLICIT_REMEMBER_VOICE: &str = "explicit_remember_voice";

const KIND_INTRODUCTION: &str = "explicit_introduction";

static SPEAKER_LABEL: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"^Speaker \d+$").expect("speaker label pattern"));

static INTRO: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(concat!(
        r"^(?:(?i:hi|hello|hey)[,!]?\s+)?",
        r"(?i:i['’]m|i am|my name is)\s+",
        r"(?P<name>[^\W\d_]+(?:[-’'][^\W\d_]+)*",
        r"(?:\s+[^\W\d_]+(?:[-’'][^\W\d_]+)*){0,2})"
    ))
    .expect("introduction pattern")
});

#[derive(Debug, Clone, PartialEq, Eq, Error)]
pub enum PeopleError {
    #[error("{}", messages::PERSON_NAME)]
    Name,
    #[error("{}", messages::PEOPLE_CAP)]
    Cap,
    #[error("{}", messages::PERSON_MISSING)]
    Missing,
    #[error("{}", messages::REMEMBER_CONSENT)]
    RememberConsent,
    #[error("{}", messages::REMEMBER_PROFILE)]
    RememberProfile,
}

/// Strip and accept a name of 1–100 characters, counted before trimming.
pub fn person_name(value: &str) -> Result<String, PeopleError> {
    if value.chars().count() > limits::PERSON_NAME_MAX_CHARS {
        return Err(PeopleError::Name);
    }
    let name = value.trim();
    if name.is_empty() {
        return Err(PeopleError::Name);
    }
    Ok(name.to_string())
}

/// Route gate: the request body must pass `consent: true`.
pub fn require_remember_consent(consent: bool) -> Result<(), PeopleError> {
    if consent {
        Ok(())
    } else {
        Err(PeopleError::RememberConsent)
    }
}

/// Store gate: consent is true and the profile carries the explicit marker.
pub fn require_profile_consent(consent: bool, profile_consent: &str) -> Result<(), PeopleError> {
    if consent && profile_consent == EXPLICIT_REMEMBER_VOICE {
        Ok(())
    } else {
        Err(PeopleError::RememberProfile)
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct Person {
    pub id: String,
    pub name: String,
    pub created: f64,
    pub updated: Option<f64>,
}

#[derive(Debug, Clone, Default, PartialEq)]
pub struct People {
    entries: Vec<Person>,
}

impl People {
    pub(crate) fn from_stored(entries: Vec<Person>) -> Self {
        Self { entries }
    }

    pub fn len(&self) -> usize {
        self.entries.len()
    }

    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    pub fn iter(&self) -> impl Iterator<Item = &Person> {
        self.entries.iter()
    }

    pub fn get(&self, id: &str) -> Result<&Person, PeopleError> {
        self.entries
            .iter()
            .find(|person| person.id == id)
            .ok_or(PeopleError::Missing)
    }

    pub fn create(&mut self, name: &str, now: f64) -> Result<Person, PeopleError> {
        self.create_with_id(name, now, Uuid::new_v4().simple().to_string())
    }

    pub fn create_with_id(
        &mut self,
        name: &str,
        now: f64,
        id: impl Into<String>,
    ) -> Result<Person, PeopleError> {
        let name = person_name(name)?;
        if self.entries.len() >= limits::MAX_PEOPLE {
            return Err(PeopleError::Cap);
        }
        let person = Person {
            id: id.into(),
            name,
            created: now,
            updated: None,
        };
        self.entries.push(person.clone());
        Ok(person)
    }

    /// Replace the People name. Existing meeting assignment snapshots are untouched.
    pub fn rename(&mut self, id: &str, name: &str, now: f64) -> Result<Person, PeopleError> {
        let index = self
            .entries
            .iter()
            .position(|person| person.id == id)
            .ok_or(PeopleError::Missing)?;
        let name = person_name(name)?;
        let person = &mut self.entries[index];
        person.name = name;
        person.updated = Some(now);
        Ok(person.clone())
    }
}

/// `Speaker N` is an unassigned diarizer label, not a confirmed person.
pub fn is_unassigned_speaker_label(label: &str) -> bool {
    SPEAKER_LABEL.is_match(label)
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SpeakerAssignment {
    pub person_id: Option<String>,
    pub meeting_id: String,
    pub track_id: String,
    pub name: String,
    pub confirmed: Option<bool>,
    pub source: String,
}

/// Ordinary corrections keep a meeting snapshot only while the document still
/// shows that name. Imports drop every assignment. Pending voice suggestions
/// are cleared, and checks still marked `checking` become `cancelled`.
pub fn reconcile_assignments<S>(
    assignments: BTreeMap<String, SpeakerAssignment>,
    speakers: &BTreeMap<String, String>,
    imported: bool,
    voice_checks: &mut BTreeMap<String, String>,
    voice_suggestions: &mut Vec<S>,
) -> BTreeMap<String, SpeakerAssignment> {
    let kept = if imported {
        BTreeMap::new()
    } else {
        assignments
            .into_iter()
            .filter(|(track, assignment)| speakers.get(track) == Some(&assignment.name))
            .collect()
    };
    voice_suggestions.clear();
    for status in voice_checks.values_mut() {
        if status == "checking" {
            *status = "cancelled".to_string();
        }
    }
    kept
}

#[derive(Debug, Clone, Deserialize, PartialEq)]
pub struct SuggestionJob {
    #[serde(default)]
    pub document: Option<SuggestionDocument>,
    #[serde(default)]
    pub dismissed_identity_suggestions: Vec<String>,
}

#[derive(Debug, Clone, Deserialize, PartialEq)]
pub struct SuggestionDocument {
    #[serde(default)]
    pub speakers: BTreeMap<String, String>,
    #[serde(default)]
    pub segments: Vec<SuggestionSegment>,
    #[serde(default)]
    pub provenance: Option<SuggestionProvenance>,
}

#[derive(Debug, Clone, Deserialize, PartialEq, Eq)]
pub struct SuggestionProvenance {
    #[serde(default)]
    pub kind: Option<String>,
}

#[derive(Debug, Clone, Deserialize, PartialEq)]
pub struct SuggestionSegment {
    pub id: String,
    pub speaker: String,
    pub start: f64,
    pub end: f64,
    pub text: String,
    #[serde(default)]
    pub review: Option<bool>,
    #[serde(default)]
    pub finalized: Option<bool>,
    #[serde(default)]
    pub speaker_candidates: Option<Vec<String>>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct IntroEvidence {
    pub segment_id: String,
    pub quote: String,
    pub start: f64,
    pub end: f64,
}

#[derive(Debug, Clone, PartialEq)]
pub struct IntroductionSuggestion {
    pub id: String,
    pub track_id: String,
    pub name: String,
    pub kind: String,
    pub evidence: IntroEvidence,
}

/// Conservative English introductions at the start of a finalized local turn.
/// Addressed names, quotes, overlap, pending text, and lowercase names are skipped.
pub fn introduction_suggestions(job: &SuggestionJob) -> Vec<IntroductionSuggestion> {
    let Some(document) = job.document.as_ref() else {
        return Vec::new();
    };
    if document
        .provenance
        .as_ref()
        .and_then(|provenance| provenance.kind.as_deref())
        != Some("local_inference")
    {
        return Vec::new();
    }
    let mut suggestions = Vec::new();
    let mut seen = BTreeSet::new();
    for segment in &document.segments {
        let track = segment.speaker.as_str();
        if seen.contains(track) {
            continue;
        }
        let Some(label) = document.speakers.get(track) else {
            continue;
        };
        if !is_unassigned_speaker_label(label) || track.starts_with("overlap") {
            continue;
        }
        if segment.review == Some(true) || segment.finalized == Some(false) {
            continue;
        }
        let candidates = segment
            .speaker_candidates
            .clone()
            .unwrap_or_else(|| vec![track.to_string()]);
        if candidates.len() != 1 {
            continue;
        }
        let Some(name) = introduced_name(&segment.text) else {
            continue;
        };
        let evidence = IntroEvidence {
            segment_id: segment.id.clone(),
            quote: segment.text.clone(),
            start: segment.start,
            end: segment.end,
        };
        let id = suggestion_id(track, &name, &evidence);
        if job
            .dismissed_identity_suggestions
            .iter()
            .any(|dismissed| dismissed == &id)
        {
            continue;
        }
        seen.insert(track.to_string());
        suggestions.push(IntroductionSuggestion {
            id,
            track_id: track.to_string(),
            name,
            kind: KIND_INTRODUCTION.to_string(),
            evidence,
        });
    }
    suggestions
}

fn introduced_name(text: &str) -> Option<String> {
    let captures = INTRO.captures(text.trim())?;
    let raw = captures.name("name")?.as_str();
    let mut words = Vec::new();
    for word in raw.split_whitespace() {
        let capital = word.chars().next().is_some_and(|ch| ch.is_uppercase());
        if !capital {
            break;
        }
        words.push(word);
    }
    if words.is_empty() {
        None
    } else {
        Some(words.join(" "))
    }
}

fn suggestion_id(track: &str, name: &str, evidence: &IntroEvidence) -> String {
    let payload = format!(
        "[{}, {}, {{\"end\": {}, \"quote\": {}, \"segment_id\": {}, \"start\": {}}}]",
        python_string(track),
        python_string(name),
        python_float(evidence.end),
        python_string(&evidence.quote),
        python_string(&evidence.segment_id),
        python_float(evidence.start),
    );
    hex::encode(Sha256::digest(payload.as_bytes()))[..32].to_string()
}

fn python_string(value: &str) -> String {
    let mut out = String::from("\"");
    for ch in value.chars() {
        match ch {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\u{0008}' => out.push_str("\\b"),
            '\u{000c}' => out.push_str("\\f"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            ch if (ch as u32) < 0x20 || !ch.is_ascii() => {
                let mut units = [0u16; 2];
                for unit in ch.encode_utf16(&mut units) {
                    out.push_str(&format!("\\u{unit:04x}"));
                }
            }
            ch => out.push(ch),
        }
    }
    out.push('"');
    out
}

fn python_float(value: f64) -> String {
    if let Ok(text) = serde_json::to_string(&value) {
        return text;
    }
    if value.is_nan() {
        "NaN".to_string()
    } else if value.is_sign_positive() {
        "Infinity".to_string()
    } else {
        "-Infinity".to_string()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn segment(text: &str) -> SuggestionSegment {
        SuggestionSegment {
            id: "intro".to_string(),
            speaker: "speaker_0".to_string(),
            start: 0.0,
            end: 3.0,
            text: text.to_string(),
            review: Some(false),
            finalized: Some(true),
            speaker_candidates: Some(vec!["speaker_0".to_string()]),
        }
    }

    fn job_with(text: &str) -> SuggestionJob {
        SuggestionJob {
            document: Some(SuggestionDocument {
                speakers: BTreeMap::from([("speaker_0".to_string(), "Speaker 1".to_string())]),
                segments: vec![segment(text)],
                provenance: Some(SuggestionProvenance {
                    kind: Some("local_inference".to_string()),
                }),
            }),
            dismissed_identity_suggestions: Vec::new(),
        }
    }

    #[test]
    fn names_are_stripped_and_capped_at_one_hundred_characters() {
        assert_eq!(person_name("  Priya  ").unwrap(), "Priya");
        assert_eq!(person_name(&"é".repeat(100)).unwrap(), "é".repeat(100));
        for bad in ["", "   ", &"a".repeat(101), &"é".repeat(101)] {
            let error = person_name(bad).unwrap_err();
            assert_eq!(error.to_string(), messages::PERSON_NAME);
        }
    }

    #[test]
    fn people_cap_is_500_and_rename_keeps_the_same_id() {
        let mut people = People::default();
        for index in 0..limits::MAX_PEOPLE {
            people
                .create_with_id("Ada", 1.0, format!("p{index}"))
                .unwrap();
        }
        assert_eq!(people.len(), 500);
        let overflow = people.create_with_id("Ada", 1.0, "overflow").unwrap_err();
        assert_eq!(overflow.to_string(), messages::PEOPLE_CAP);
        assert_eq!(
            people
                .create_with_id("   ", 1.0, "blank")
                .unwrap_err()
                .to_string(),
            messages::PERSON_NAME
        );
        assert_eq!(people.len(), 500);

        let mut people = People::default();
        let created = people.create_with_id("  Priya  ", 1.5, "pid").unwrap();
        assert_eq!(created.name, "Priya");
        assert_eq!(created.updated, None);
        let renamed = people.rename("pid", "Dr. Priya", 4.0).unwrap();
        assert_eq!(renamed.id, "pid");
        assert_eq!(renamed.name, "Dr. Priya");
        assert_eq!(renamed.created, 1.5);
        assert_eq!(renamed.updated, Some(4.0));
        let snapshot = "Priya".to_string();
        assert_eq!(snapshot, "Priya");
        assert_eq!(people.get("pid").unwrap().name, "Dr. Priya");
        assert_eq!(
            people
                .rename("missing", "Nope", 5.0)
                .unwrap_err()
                .to_string(),
            messages::PERSON_MISSING
        );
        assert_eq!(
            people.rename("missing", "", 5.0).unwrap_err().to_string(),
            messages::PERSON_MISSING
        );
        assert_eq!(
            people.rename("pid", "", 5.0).unwrap_err().to_string(),
            messages::PERSON_NAME
        );
        assert_eq!(people.get("pid").unwrap().name, "Dr. Priya");

        let first = people.create("Priya", 2.0).unwrap();
        let second = people.create("Priya", 3.0).unwrap();
        assert_ne!(first.id, second.id);
        assert_eq!(first.id.len(), 32);
        assert!(first.id.bytes().all(|byte| byte.is_ascii_hexdigit()));
    }

    #[test]
    fn remember_consent_uses_the_route_and_profile_messages() {
        assert_eq!(
            require_remember_consent(false).unwrap_err().to_string(),
            messages::REMEMBER_CONSENT
        );
        assert!(require_remember_consent(true).is_ok());
        assert_eq!(
            require_profile_consent(false, EXPLICIT_REMEMBER_VOICE)
                .unwrap_err()
                .to_string(),
            messages::REMEMBER_PROFILE
        );
        assert_eq!(
            require_profile_consent(true, "inferred")
                .unwrap_err()
                .to_string(),
            messages::REMEMBER_PROFILE
        );
        assert!(require_profile_consent(true, EXPLICIT_REMEMBER_VOICE).is_ok());
    }

    #[test]
    fn golden_introductions_match_the_reference_file() {
        let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../tests/goldens/introductions.json");
        let file: serde_json::Value =
            serde_json::from_str(&std::fs::read_to_string(path).unwrap()).unwrap();
        for case in file["cases"].as_array().unwrap() {
            let job: SuggestionJob = serde_json::from_value(case["job"].clone()).unwrap();
            let got = introduction_suggestions(&job);
            let expected = case["suggestions"].as_array().unwrap();
            assert_eq!(got.len(), expected.len(), "{}", case["name"]);
            for (suggestion, row) in got.iter().zip(expected) {
                assert_eq!(suggestion.id, row["id"].as_str().unwrap());
                assert_eq!(suggestion.track_id, row["track_id"].as_str().unwrap());
                assert_eq!(suggestion.name, row["name"].as_str().unwrap());
                assert_eq!(suggestion.kind, "explicit_introduction");
                assert_eq!(
                    suggestion.evidence.segment_id,
                    row["evidence"]["segment_id"].as_str().unwrap()
                );
                assert_eq!(
                    suggestion.evidence.quote,
                    row["evidence"]["quote"].as_str().unwrap()
                );
                assert_eq!(
                    suggestion.evidence.start.to_bits(),
                    row["evidence"]["start"].as_f64().unwrap().to_bits()
                );
                assert_eq!(
                    suggestion.evidence.end.to_bits(),
                    row["evidence"]["end"].as_f64().unwrap().to_bits()
                );
            }
        }
    }

    #[test]
    fn introductions_keep_conservative_english_names_only() {
        let rejected = [
            "Priya, what do you think?",
            "Alex said, I’m Priya.",
            "\"I’m Priya,\" she said.",
            "I'm happy to join.",
            "This is Priya’s proposal.",
            "Am I Priya?",
            "I'm priya.",
        ];
        for text in rejected {
            assert!(
                introduction_suggestions(&job_with(text)).is_empty(),
                "{text}"
            );
        }
        let accepted = [
            ("Hello, I'm Priya Patel.", "Priya Patel"),
            ("My name is José.", "José"),
            ("I’m Anne-Marie.", "Anne-Marie"),
            ("i'm Priya.", "Priya"),
            ("I'm Priya patel.", "Priya"),
        ];
        for (text, name) in accepted {
            let suggestions = introduction_suggestions(&job_with(text));
            assert_eq!(suggestions.len(), 1, "{text}");
            assert_eq!(suggestions[0].name, name);
            assert_eq!(suggestions[0].evidence.quote, text);
        }

        let mut reviewed = job_with("I'm Priya.");
        reviewed.document.as_mut().unwrap().segments[0].review = Some(true);
        assert!(introduction_suggestions(&reviewed).is_empty());
        reviewed.document.as_mut().unwrap().segments[0].review = Some(false);
        reviewed.document.as_mut().unwrap().segments[0].finalized = Some(false);
        assert!(introduction_suggestions(&reviewed).is_empty());
        reviewed.document.as_mut().unwrap().segments[0].finalized = Some(true);
        reviewed.document.as_mut().unwrap().segments[0].speaker_candidates =
            Some(vec!["speaker_0".to_string(), "speaker_1".to_string()]);
        assert!(introduction_suggestions(&reviewed).is_empty());
        for kind in ["manual", "imported", "pending_inference"] {
            let mut other = job_with("I'm Priya.");
            other.document.as_mut().unwrap().provenance = Some(SuggestionProvenance {
                kind: Some(kind.to_string()),
            });
            assert!(introduction_suggestions(&other).is_empty(), "{kind}");
        }

        let mut overlap = job_with("I'm Priya.");
        let document = overlap.document.as_mut().unwrap();
        document.segments[0].speaker = "overlap_0".to_string();
        document
            .speakers
            .insert("overlap_0".to_string(), "Speaker 1".to_string());
        assert!(introduction_suggestions(&overlap).is_empty());

        let mut named = job_with("I'm Priya.");
        named
            .document
            .as_mut()
            .unwrap()
            .speakers
            .insert("speaker_0".to_string(), "Priya".to_string());
        assert!(introduction_suggestions(&named).is_empty());
    }

    #[test]
    fn first_introduction_per_track_wins_unless_it_was_dismissed() {
        let mut job = job_with("I'm Priya. Welcome.");
        job.document
            .as_mut()
            .unwrap()
            .segments
            .push(SuggestionSegment {
                id: "later".to_string(),
                speaker: "speaker_0".to_string(),
                start: 4.0,
                end: 7.0,
                text: "I'm Alex.".to_string(),
                review: Some(false),
                finalized: Some(true),
                speaker_candidates: Some(vec!["speaker_0".to_string()]),
            });
        let first = introduction_suggestions(&job);
        assert_eq!(first.len(), 1);
        assert_eq!(first[0].name, "Priya");
        job.dismissed_identity_suggestions.push(first[0].id.clone());
        let next = introduction_suggestions(&job);
        assert_eq!(next.len(), 1);
        assert_eq!(next[0].name, "Alex");
        assert_eq!(next[0].evidence.segment_id, "later");
    }

    #[test]
    fn reconcile_keeps_matching_snapshots_and_cancels_open_checks() {
        let priya = SpeakerAssignment {
            person_id: Some("p1".to_string()),
            meeting_id: "m".to_string(),
            track_id: "speaker_0".to_string(),
            name: "Priya".to_string(),
            confirmed: Some(true),
            source: "manual".to_string(),
        };
        let guest = SpeakerAssignment {
            name: "Guest".to_string(),
            track_id: "speaker_1".to_string(),
            ..priya.clone()
        };
        let mut speakers = BTreeMap::from([
            ("speaker_0".to_string(), "Priya".to_string()),
            ("speaker_1".to_string(), "Speaker 2".to_string()),
        ]);
        let assignments = BTreeMap::from([
            ("speaker_0".to_string(), priya.clone()),
            ("speaker_1".to_string(), guest),
        ]);
        let mut checks = BTreeMap::from([
            ("speaker_0".to_string(), "checking".to_string()),
            ("speaker_1".to_string(), "unknown".to_string()),
        ]);
        let mut suggestions = vec!["voice-guess".to_string()];
        let kept = reconcile_assignments(
            assignments.clone(),
            &speakers,
            false,
            &mut checks,
            &mut suggestions,
        );
        assert_eq!(kept.len(), 1);
        assert_eq!(kept["speaker_0"].name, "Priya");
        assert!(suggestions.is_empty());
        assert_eq!(checks["speaker_0"], "cancelled");
        assert_eq!(checks["speaker_1"], "unknown");

        speakers.insert("speaker_0".to_string(), "Guest".to_string());
        let mut checks = BTreeMap::new();
        let mut suggestions = vec!["again".to_string()];
        let kept = reconcile_assignments(
            assignments.clone(),
            &speakers,
            false,
            &mut checks,
            &mut suggestions,
        );
        assert!(kept.is_empty());
        assert!(suggestions.is_empty());

        let mut checks = BTreeMap::from([("speaker_0".to_string(), "checking".to_string())]);
        let mut suggestions = vec!["imported".to_string()];
        let speakers = BTreeMap::from([("speaker_0".to_string(), "Priya".to_string())]);
        let kept =
            reconcile_assignments(assignments, &speakers, true, &mut checks, &mut suggestions);
        assert!(kept.is_empty());
        assert!(suggestions.is_empty());
        assert_eq!(checks["speaker_0"], "cancelled");
    }
}
