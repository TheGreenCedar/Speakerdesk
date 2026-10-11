//! Rust ports of the pure-logic cases in the v0.6.6 Python tests
//! (`tests/test_export_bounds.py`, `tests/test_passage_review.py`,
//! `tests/test_job_polling.py`, `tests/test_imports.py` and the store/edit
//! parts of `tests/test_reading_turns.py`). Route-level assertions in the
//! same Python tests (HTTP status codes, executors, audio files) land with
//! the M1.3 host routes and are tracked in `xtask/port-ledger.toml`.
//!
//! Tests build fixture JSON, convert at the serde boundary into the typed
//! model (`Document`, `Job`, `Segment`), and compare through
//! `serde_json::to_value` — the same serialization boundary the store and
//! host use.

use std::fs;
use std::path::PathBuf;

use serde_json::{Value, json};
use speakerdesk_core::edit::{SegmentPatch, TranscriptPut, patch_segment, put_transcript};
use speakerdesk_core::error::ErrorKind;
use speakerdesk_core::model::{
    Document, Job, JobStatus, ProvenanceKind, RefinementStatus, ReviewResolution, Segment,
    TranscriptionReview,
};
use speakerdesk_core::review::{retain_unassigned_audio, review_reason};
use speakerdesk_core::store::Store;
use speakerdesk_core::transcript::{ExportTooLarge, export, export_with_limit, validate};

const NOW: f64 = 1_700_000_000.0;

struct TempDir(PathBuf);

impl TempDir {
    fn new(tag: &str) -> Self {
        let dir = std::env::temp_dir().join(format!(
            "speakerdesk-ported-{}-{}-{}",
            std::process::id(),
            tag,
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        fs::create_dir_all(&dir).unwrap();
        Self(dir)
    }
}

impl Drop for TempDir {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.0);
    }
}

fn document(value: &Value) -> Document {
    serde_json::from_value(value.clone()).unwrap()
}

fn segment(value: &Value) -> Segment {
    serde_json::from_value(value.clone()).unwrap()
}

// ---- tests/test_export_bounds.py ----

/// `test_export_bounds.py::test_utf8_limit_is_inclusive_and_all_formats_preserve_words`
/// (pure part; the GET/HEAD attachment half is route-level).
#[test]
fn utf8_limit_is_inclusive_and_all_formats_preserve_words() {
    let document = document(&json!({
        "speakers": {"speaker_0": "Zoë"},
        "segments": [
            {"id": "words", "start": 0.0, "end": 1.0, "speaker": "speaker_0",
             "text": "Checked café 日本語", "edited": true},
            {"id": "blank", "start": 1.0, "end": 2.0, "speaker": "speaker_0",
             "text": "", "audio_state": "digital_silence"}
        ]
    }));
    let before = document.clone();
    for kind in ["txt", "srt", "vtt", "json"] {
        let (content, _) = export(&document, kind).unwrap();
        let size = content.len();
        assert!(size > content.chars().count(), "{kind}");
        let (capped, _) = export_with_limit(&document, kind, size).unwrap();
        assert_eq!(capped, content, "{kind}");
        assert!(matches!(
            export_with_limit(&document, kind, size - 1),
            Err(ExportTooLarge)
        ));
        assert!(content.contains("Checked café 日本語"), "{kind}");
    }
    assert_eq!(document, before, "export must not mutate the document");
    let (json_out, _) = export(&document, "json").unwrap();
    // The typed boundary drops the non-model `edited` marker — compare
    // against the serialized typed document.
    assert_eq!(
        serde_json::from_str::<Value>(&json_out).unwrap(),
        serde_json::to_value(&document).unwrap()
    );
}

/// `test_export_bounds.py::test_empty_plain_exports_and_vtt_keep_existing_format`
#[test]
fn empty_plain_exports_and_vtt_keep_existing_format() {
    let document = document(&json!({"speakers": {"speaker_0": "Voice"}, "segments": []}));
    assert_eq!(export(&document, "txt").unwrap().0, "\n");
    assert_eq!(export(&document, "srt").unwrap().0, "\n");
    assert_eq!(export(&document, "vtt").unwrap().0, "WEBVTT\n\n");
}

// ---- tests/test_passage_review.py ----

/// `test_passage_review.py::test_text_exports_omit_blank_audio_but_json_retains_review_ranges`
#[test]
fn text_exports_omit_blank_audio_but_json_retains_review_ranges() {
    let mut raw = document(&json!({
        "speakers": {"speaker_0": "Speaker 1", "overlap": "Overlapping speakers"},
        "segments": [
            {"id": "known", "start": 0.0, "end": 2.0, "speaker": "speaker_0",
             "text": "Keep these words", "review": false},
            {"id": "unknown", "start": 2.0, "end": 4.0, "speaker": "speaker_0",
             "text": "", "language": null,
             "language_detection": {"mode": "auto", "reason": "uncertain"},
             "review": true},
            {"id": "partial", "start": 4.0, "end": 6.0, "speaker": "speaker_0",
             "text": "Partial words",
             "transcription_review": {"reason": "token_limit"}, "review": true},
            {"id": "mixed", "start": 7.0, "end": 9.0, "speaker": "overlap",
             "text": "", "speaker_candidates": ["speaker_0", "speaker_1"],
             "review": true}
        ]
    }));
    retain_unassigned_audio(&mut raw, 10.0);
    let document = validate(&serde_json::to_value(&raw).unwrap(), 10.0).unwrap();
    let before = document.clone();
    let mut again = document.clone();
    retain_unassigned_audio(&mut again, 10.0);
    assert_eq!(again, before, "retain_unassigned_audio is idempotent");
    for kind in ["txt", "srt", "vtt"] {
        let (text, _) = export(&document, kind).unwrap();
        for marker in [
            "Keep these words",
            "Partial words",
            "Transcript may be incomplete",
        ] {
            assert!(text.contains(marker), "{kind}: {marker}");
        }
        for marker in [
            "Language uncertain",
            "voices are not separated",
            "may be silence or missed speech",
        ] {
            assert!(!text.contains(marker), "{kind}: {marker}");
        }
        assert_eq!(text.matches("Speaker 1:").count(), 2, "{kind}");
    }
    let (json_out, _) = export(&document, "json").unwrap();
    assert_eq!(
        serde_json::from_str::<Value>(&json_out).unwrap(),
        serde_json::to_value(&document).unwrap(),
        "json export retains review ranges"
    );

    // `words_reviewed` on an unassigned row clears the reason.
    let mut checked = document.segments[1].clone();
    checked.text = "Manually checked words".to_string();
    checked.review_resolution = Some(ReviewResolution::WordsReviewed);
    assert_eq!(review_reason(&checked), "");

    let mut mixed = document
        .segments
        .iter()
        .find(|s| s.id == "mixed")
        .unwrap()
        .clone();
    mixed.text = "Heard words".to_string();
    mixed.review_resolution = Some(ReviewResolution::WordsReviewed);
    assert!(review_reason(&mixed).contains("not separated"));

    for reason in ["transcription_failed", "empty_result", "token_limit"] {
        // Python mutates doc['segments'][1] in place — the reviewed row.
        let mut repaired = checked.clone();
        repaired.transcription_review = Some(TranscriptionReview {
            reason: Some(reason.to_string()),
            partial_text: None,
        });
        assert_eq!(review_reason(&repaired), "");
        let single = Document {
            speakers: document.speakers.clone(),
            segments: vec![repaired],
            ..Default::default()
        };
        let (text, _) = export(&single, "txt").unwrap();
        assert!(!text.contains("Needs review"), "{reason}");
    }
}

/// `test_reading_turns.py::test_parent_review_does_not_infer_simultaneous_speech_from_union`
#[test]
fn parent_review_does_not_infer_simultaneous_speech_from_union() {
    let mut row = segment(&json!({
        "id": "one", "start": 0.0, "end": 10.0, "text": "words",
        "speaker": "multiple_speakers",
        "speaker_candidates": ["speaker_0", "speaker_1"],
        "speaker_activity": {"audio_revision": 3, "regions": [
            {"start_sample": 0, "end_sample": 80000, "speakers": ["speaker_0"]},
            {"start_sample": 80000, "end_sample": 160000, "speakers": ["speaker_1"]}
        ]}
    }));
    assert!(!review_reason(&row).contains("Overlapping"));
    row.speaker = "overlap_speaker_0_speaker_1".to_string();
    assert!(review_reason(&row).contains("Overlapping"));
}

/// `test_reading_turns.py::test_server_saved_edit_and_import_cannot_retain_or_forge_turns`
/// and `test_passage_review.py::test_candidate_is_removed_on_timing_track_edits_and_cannot_be_forged`
/// — the PUT/PATCH semantics, without the Flask shell.
#[test]
fn edits_cannot_retain_or_forge_turns_or_retry_candidates() {
    let turn = json!({
        "text": "words", "speaker": "speaker_0", "attribution": "single",
        "start": 0.0, "end": 10.0, "coarse_timing": false, "reading_slice": false
    });
    let document = json!({
        "schema_version": 1,
        "speakers": {"multiple_speakers": "Multiple speakers",
                     "speaker_0": "A", "speaker_1": "B"},
        "segments": [{
            "id": "one", "start": 0.0, "end": 10.0, "speaker": "multiple_speakers",
            "text": "words", "machine_revision": 2, "protected_fields": [],
            "reading_turns": [turn], "reading_turn_provenance": {"utterance_id": "one"},
            "speaker_candidates": ["speaker_0", "speaker_1"]
        }]
    });
    let mut job: Job = serde_json::from_value(json!({
        "id": "e".repeat(32), "kind": "meeting", "name": "CPU turn contract",
        "created": 1.0, "updated": 1.0, "status": "ready", "duration": 10.0,
        "language": "auto", "revision": 0, "document": document
    }))
    .unwrap();
    let saved_turns = job.document.as_ref().unwrap().segments[0]
        .reading_turns
        .clone();

    // A forged reading_turns payload is replaced by the server's copy.
    let mut forged = serde_json::to_value(&job.document).unwrap();
    forged["segments"][0]["reading_turns"][0]["speaker"] = json!("speaker_1");
    put_transcript(
        &mut job,
        TranscriptPut::from_value(&json!({"revision": 0, "document": forged})),
        NOW,
    )
    .unwrap();
    assert_eq!(
        job.document.as_ref().unwrap().segments[0].reading_turns,
        saved_turns,
        "forged turns are replaced by the stored projection"
    );

    // A text edit removes the stale turns (the model no longer carries the
    // receipt key at all).
    patch_segment(
        &mut job,
        "one",
        SegmentPatch::from_value(&json!({
            "segment_revision": 2, "changes": {"text": "Human correction"}
        }))
        .unwrap(),
        NOW,
    )
    .unwrap();
    let edited = &job.document.as_ref().unwrap().segments[0];
    assert!(edited.reading_turns.is_none());

    // Timing and speaker edits remove a stored retry_candidate…
    for change in [json!({"start": 0.5}), json!({"speaker": "speaker_1"})] {
        let mut job = job.clone();
        job.document.as_mut().unwrap().segments[0].retry_candidate =
            serde_json::from_value(json!({
                "id": "server-owned", "start": 0.0, "end": 10.0,
                "speaker": "speaker_0", "text": "Earlier audio words",
                "language": "fr", "original_text": "words"
            }))
            .unwrap();
        job.document.as_mut().unwrap().segments[0].speaker_candidates =
            Some(vec!["speaker_0".to_string()]);
        let mut incoming = serde_json::to_value(&job.document).unwrap();
        let obj = incoming["segments"][0].as_object_mut().unwrap();
        for (k, v) in change.as_object().unwrap() {
            obj.insert(k.clone(), v.clone());
        }
        let revision = job.revision;
        put_transcript(
            &mut job,
            TranscriptPut::from_value(&json!({"revision": revision, "document": incoming})),
            NOW,
        )
        .unwrap();
        assert!(
            job.document.as_ref().unwrap().segments[0]
                .retry_candidate
                .is_none(),
            "change {change} removes the candidate"
        );
    }

    // …and a client cannot forge one.
    let mut incoming = serde_json::to_value(&job.document).unwrap();
    incoming["segments"][0]["retry_candidate"] = json!({"text": "Forged model candidate"});
    let revision = job.revision;
    put_transcript(
        &mut job,
        TranscriptPut::from_value(&json!({"revision": revision, "document": incoming})),
        NOW,
    )
    .unwrap();
    assert!(
        job.document.as_ref().unwrap().segments[0]
            .retry_candidate
            .is_none()
    );

    // Imported text drops turns and stamps the imported provenance
    // (test_imports.py::test_pcm_batch_import_edit_export_and_restart_preserve_user_data).
    let mut imported = serde_json::to_value(&job.document).unwrap();
    imported["segments"][0]["id"] = json!("new-import");
    imported["segments"][0]["reading_turns"] = serde_json::to_value(&saved_turns).unwrap();
    let revision = job.revision;
    put_transcript(
        &mut job,
        TranscriptPut::from_value(&json!({
            "revision": revision, "document": imported, "imported": true
        })),
        NOW,
    )
    .unwrap();
    let row = &job.document.as_ref().unwrap().segments[0];
    assert!(row.reading_turns.is_none());
    let provenance = job.document.as_ref().unwrap().provenance.as_ref().unwrap();
    assert_eq!(provenance.kind, Some(ProvenanceKind::Imported));
    assert_eq!(
        provenance.timing.as_deref(),
        Some("User imported segment times; source unverified")
    );
    assert_eq!(
        job.document.as_ref().unwrap().warnings.as_ref().unwrap()[0],
        "Imported transcript. Text, speaker labels and timing have not been verified by this app."
    );
}

// ---- tests/test_job_polling.py (JobStore-level cases) ----

/// The Python `seed` job — meeting kind, protected fields, private blobs.
fn seed_job(jid: &str) -> Job {
    serde_json::from_value(json!({
        "id": jid, "name": "Synthetic meeting", "kind": "meeting",
        "created": 1.0, "updated": 1.0, "status": "ready", "message": "Saved.",
        "language": "en", "duration": 10.0, "revision": 7,
        "document": {
            "schema_version": 1,
            "speakers": {"speaker_0": "Speaker 1"},
            "segments": [{"id": "one", "start": 0.0, "end": 1.0,
                          "speaker": "speaker_0", "text": "Keep these words.",
                          "machine_revision": 2, "protected_fields": ["text"],
                          "canonical_utterance_id": "canonical-one",
                          "start_sample": 0, "end_sample": 16000,
                          "audio_revision": 4}],
            "warnings": [], "provenance": {"kind": "local_inference"}
        },
        "fast_history": {"old": {"text": "x".repeat(100)}},
        "refinement_history": {"old": {"text": "y".repeat(100)}}
    }))
    .unwrap()
}

/// `test_job_polling.py::test_list_is_compact_sorted_and_legacy_sql_updates_are_visible`
/// (store part; "legacy SQL updates" is exercised by the migration tests).
#[test]
fn list_is_compact_sorted_and_private_state_is_projected_out() {
    let home = TempDir::new("list");
    let store = Store::open(&home.0).unwrap();
    store.put(&mut seed_job(&"a".repeat(32)), NOW).unwrap();
    let mut second = seed_job(&"b".repeat(32));
    second.created = 2.0;
    second.name = "Second".to_string();
    store.put(&mut second, NOW).unwrap();

    let list = store.summaries().unwrap();
    assert_eq!(
        list.iter().map(|row| row.id.as_str()).collect::<Vec<_>>(),
        vec!["b".repeat(32), "a".repeat(32)]
    );
    let expected = [
        "id", "name", "kind", "created", "updated", "status", "message", "language", "duration",
        "revision",
    ]
    .into_iter()
    .collect::<std::collections::BTreeSet<_>>();
    assert_eq!(
        serde_json::to_value(&list[1])
            .unwrap()
            .as_object()
            .unwrap()
            .keys()
            .map(String::as_str)
            .collect::<std::collections::BTreeSet<_>>(),
        expected
    );
    // …but survives on the full read.
    let full = store.get(&"a".repeat(32), false).unwrap().unwrap();
    assert_eq!(
        full.private.fast_history.as_ref().unwrap()["old"]["text"],
        "x".repeat(100)
    );
}

/// `test_job_polling.py::test_duration_ticks_do_not_rewrite_job_or_revision_and_ignore_older_ticks`
#[test]
fn duration_ticks_do_not_rewrite_job_and_ignore_older_ticks() {
    let home = TempDir::new("ticks");
    let store = Store::open(&home.0).unwrap();
    let jid = "a".repeat(32);
    store.put(&mut seed_job(&jid), NOW).unwrap();

    assert!(store.checkpoint_duration(&jid, 12.5, NOW + 10.0).unwrap());
    let first = store.get(&jid, false).unwrap().unwrap();
    let first_updated = first.updated;
    assert_eq!(first.duration, Some(12.5));
    assert_eq!(first_updated, NOW + 10.0);

    // Older and equal ticks are dropped; the stored row is untouched.
    assert!(store.checkpoint_duration(&jid, 11.0, NOW + 20.0).unwrap());
    assert!(store.checkpoint_duration(&jid, 12.5, NOW + 30.0).unwrap());
    let current = store.get(&jid, false).unwrap().unwrap();
    assert_eq!(current.duration, Some(12.5));
    assert_eq!(current.revision, 7);
    assert_eq!(current.updated, first_updated);
    assert_eq!(
        store.summaries().unwrap()[0].duration,
        Some(12.5),
        "the list sees the checkpoint"
    );
    assert_eq!(current.document, seed_job(&jid).document);

    // Missing jobs report false; bad durations raise.
    assert!(
        !store
            .checkpoint_duration(&"c".repeat(32), 1.0, NOW)
            .unwrap()
    );
    assert!(store.checkpoint_duration(&jid, f64::NAN, NOW).is_err());
    assert!(store.checkpoint_duration(&jid, -1.0, NOW).is_err());
}

/// `test_job_polling.py::test_stale_full_write_folds_latest_checkpoint_and_delete_cleans_projections`
#[test]
fn stale_full_write_folds_latest_checkpoint_and_delete_cleans() {
    let home = TempDir::new("fold");
    let store = Store::open(&home.0).unwrap();
    let jid = "a".repeat(32);
    store.put(&mut seed_job(&jid), NOW).unwrap();

    let mut stale = store.get(&jid, false).unwrap().unwrap();
    store.checkpoint_duration(&jid, 13.0, NOW + 10.0).unwrap();
    stale.message = Some("Still saved.".to_string());
    store.put(&mut stale, NOW + 20.0).unwrap();

    let stored = store.get(&jid, false).unwrap().unwrap();
    assert_eq!(
        stored.duration,
        Some(13.0),
        "checkpoint folds into the payload"
    );
    assert_eq!(stored.document, seed_job(&jid).document);
    assert_eq!(
        stored.private.refinement_history.as_ref().unwrap()["old"]["text"],
        "y".repeat(100)
    );

    store.delete(&jid).unwrap();
    assert!(store.get(&jid, false).unwrap().is_none());
    assert!(store.summaries().unwrap().is_empty());
    // A stale checkpoint row must not survive the delete.
    assert!(!store.checkpoint_duration(&jid, 20.0, NOW).unwrap());
}

/// `test_job_polling.py::test_restart_recovers_latest_duration_and_interrupted_refinement`
#[test]
fn restart_recovers_latest_duration_and_interrupted_refinement() {
    let home = TempDir::new("recover");
    let store = Store::open(&home.0).unwrap();
    let jid = "a".repeat(32);
    let mut job = seed_job(&jid);
    job.status = JobStatus::Recording;
    job.refinement_status = Some(RefinementStatus::Refining);
    job.rolling_refinement = serde_json::from_value(json!({
        "version": 1, "completed_sample": 0, "pending": [],
        "last_scheduled_at": 0.0, "cancelled": false
    }))
    .unwrap();
    job.private.rolling_inflight = Some(json!({"operation_id": "interrupted"}));
    store.put(&mut job, NOW).unwrap();
    store.checkpoint_duration(&jid, 14.0, NOW + 5.0).unwrap();

    store.recover(NOW + 60.0).unwrap();
    let recovered = store.get(&jid, false).unwrap().unwrap();
    assert_eq!(recovered.status, JobStatus::Failed);
    assert_eq!(
        recovered.message.as_deref(),
        Some("The meeting was interrupted. Captured audio and transcript were preserved.")
    );
    assert_eq!(recovered.duration, Some(14.0), "crash checkpoint folds");
    assert_eq!(recovered.refinement_status, Some(RefinementStatus::Paused));
    assert!(recovered.private.rolling_inflight.is_none());
    assert_eq!(
        recovered.private.fast_history.as_ref().unwrap()["old"]["text"],
        "x".repeat(100)
    );
    assert_eq!(recovered.document, seed_job(&jid).document);
}

/// `test_job_polling.py::test_malformed_revision_and_unknown_job_are_rejected`
/// (store part; `?known_revision=` parsing is host-level).
#[test]
fn unknown_job_reads_none() {
    let home = TempDir::new("unknown");
    let store = Store::open(&home.0).unwrap();
    assert!(store.get(&"c".repeat(32), false).unwrap().is_none());
    assert!(store.get(&"c".repeat(32), true).unwrap().is_none());
}

/// Error kinds carry the v0.6.6 wire statuses (400/404/409).
#[test]
fn edit_errors_keep_the_verbatim_messages() {
    let mut job = seed_job(&"a".repeat(32));
    let err = patch_segment(
        &mut job,
        "missing",
        SegmentPatch::from_value(&json!({"changes": {"text": "x"}})).unwrap(),
        NOW,
    )
    .unwrap_err();
    assert_eq!(err.kind, ErrorKind::NotFound);
    assert_eq!(
        err.message,
        "The requested URL was not found on the server. If you entered the URL manually please check your spelling and try again."
    );
    let err = patch_segment(
        &mut job,
        "one",
        SegmentPatch::from_value(&json!({
            "segment_revision": 1, "changes": {"text": "x"}
        }))
        .unwrap(),
        NOW,
    )
    .unwrap_err();
    assert_eq!(err.kind, ErrorKind::Conflict);
    assert_eq!(
        err.message,
        "This passage changed. Review its latest words before saving."
    );
    let document = serde_json::to_value(&job.document).unwrap();
    let err = put_transcript(
        &mut job,
        TranscriptPut::from_value(&json!({"revision": 0, "document": document})),
        NOW,
    )
    .unwrap_err();
    assert_eq!(err.kind, ErrorKind::Conflict);
    assert_eq!(
        err.message,
        "This transcript changed in another tab. Reload before saving."
    );
}
