use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};

use crate::ids::JobId;

fn default_schema_version() -> i64 {
    1
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum JobKind {
    #[default]
    Import,
    Meeting,
}

impl JobKind {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Import => "import",
            Self::Meeting => "meeting",
        }
    }
}

/// Product fields the frontend, exports, edit/merge, and recovery read.
/// Unknown keys survive in `rest` so the legacy importer can round-trip them.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Segment {
    pub id: String,
    pub start: f64,
    pub end: f64,
    pub speaker: String,
    pub text: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub language: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub language_detection: Option<Value>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub transcription_review: Option<Value>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub reading_turns: Option<Value>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub reading_turn_provenance: Option<Value>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub alignment: Option<Value>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub speaker_activity: Option<Value>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub acoustic_evidence: Option<Value>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub refinement_state: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub machine_revision: Option<i64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub protected_fields: Option<Vec<String>>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub review_resolution: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub retry_candidate: Option<Value>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub canonical_utterance_id: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub audio_revision: Option<i64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub review: Option<bool>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub capture_source: Option<Value>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub separation_candidate: Option<Box<crate::separation::Candidate>>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub separation: Option<crate::separation::Voice>,
    #[serde(default, flatten)]
    pub rest: Map<String, Value>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Document {
    #[serde(default = "default_schema_version")]
    pub schema_version: i64,
    pub speakers: BTreeMap<String, String>,
    pub segments: Vec<Segment>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub provenance: Option<Value>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub warnings: Vec<String>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub diarization: Vec<Value>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub edited_at: Option<f64>,
    #[serde(default, flatten)]
    pub rest: Map<String, Value>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Job {
    pub id: JobId,
    pub name: String,
    #[serde(default)]
    pub kind: JobKind,
    pub status: String,
    #[serde(default)]
    pub message: String,
    #[serde(default)]
    pub language: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub source_file: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub duration: Option<f64>,
    #[serde(default)]
    pub revision: i64,
    pub created: f64,
    pub updated: f64,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub document: Option<Document>,
    /// Top-level job keys that are not columns: assignments, recovery, suggestions.
    #[serde(default, flatten)]
    pub extra: Map<String, Value>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct JobSummary {
    pub id: JobId,
    pub name: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub kind: Option<JobKind>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub source_file: Option<String>,
    pub created: f64,
    pub updated: f64,
    pub status: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub message: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub language: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub duration: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub revision: Option<i64>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Person {
    pub id: String,
    pub name: String,
    pub created: f64,
    pub updated: f64,
}

/// Statuses `JobStore::recover` rewrites to `failed` on startup.
pub const RECOVERY_ACTIVE: &[&str] = &[
    "preparing",
    "queued",
    "processing",
    "starting",
    "recording",
    "paused",
    "finishing",
];

#[cfg(test)]
mod tests {
    use super::{Document, Job, JobKind, Segment};
    use crate::ids::JobId;

    #[test]
    fn job_round_trip_keeps_extra_product_fields() {
        let job = Job {
            id: JobId::parse("0123456789abcdef0123456789abcdef").unwrap(),
            name: "Standup".into(),
            kind: JobKind::Meeting,
            status: "ready".into(),
            message: String::new(),
            language: "en".into(),
            source_file: None,
            duration: Some(1.5),
            revision: 2,
            created: 1.0,
            updated: 2.0,
            document: Some(Document {
                schema_version: 1,
                speakers: [("speaker_0".into(), "Speaker 1".into())].into(),
                segments: vec![Segment {
                    id: "seg-0".into(),
                    start: 0.0,
                    end: 1.0,
                    speaker: "speaker_0".into(),
                    text: "Hello".into(),
                    language: Some("en".into()),
                    language_detection: None,
                    transcription_review: None,
                    reading_turns: None,
                    reading_turn_provenance: None,
                    alignment: None,
                    speaker_activity: None,
                    acoustic_evidence: None,
                    refinement_state: Some("refined".into()),
                    machine_revision: Some(1),
                    protected_fields: None,
                    review_resolution: None,
                    retry_candidate: None,
                    canonical_utterance_id: None,
                    audio_revision: Some(0),
                    review: Some(false),
                    capture_source: None,
                    separation_candidate: None,
                    separation: None,
                    rest: serde_json::Map::new(),
                }],
                provenance: None,
                warnings: Vec::new(),
                diarization: Vec::new(),
                edited_at: None,
                rest: serde_json::Map::new(),
            }),
            extra: serde_json::Map::from_iter([(
                "speaker_assignments".into(),
                serde_json::json!({}),
            )]),
        };
        let json = serde_json::to_value(&job).unwrap();
        assert_eq!(json["kind"], "meeting");
        assert_eq!(json["speaker_assignments"], serde_json::json!({}));
        assert_eq!(
            json["document"]["segments"][0]["refinement_state"],
            "refined"
        );
        let parsed: Job = serde_json::from_value(json).unwrap();
        assert_eq!(parsed, job);
    }

    #[test]
    fn missing_kind_deserializes_as_import() {
        let value = serde_json::json!({
            "id": "0123456789abcdef0123456789abcdef",
            "name": "Clip",
            "status": "ready",
            "created": 1.0,
            "updated": 1.0
        });
        let job: Job = serde_json::from_value(value).unwrap();
        assert_eq!(job.kind, JobKind::Import);
    }
}
