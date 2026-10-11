//! Golden-driven tests for `core::edit` — PUT `/transcript` and PATCH
//! `/segments/<sid>` through `tools/oracle/gen_edit.py` cases.
//!
//! Seed jobs are projected through the field ledger with
//! `legacy::map_payload` (reading-turn gate + drop-class removal) and then
//! strictly deserialized into the typed `Job` — the same mapping the
//! importer applies. Recovery is *not* applied: the seeds model rows of a
//! live store, not a migrated one. Comparisons stay `Value` equality at the
//! serialization boundary (`serde_json::to_value(&job)` vs the projected
//! golden).

use std::path::Path;

use serde_json::{Value, json};
use speakerdesk_core::edit::{SegmentPatch, TranscriptPut, patch_segment, put_transcript};
use speakerdesk_core::error::ErrorKind;
use speakerdesk_core::model::{Job, Segment};
use speakerdesk_core::store::legacy::map_payload;

const FIXED_TIME: f64 = 1_700_000_000.0;

fn goldens(dir: &str) -> Vec<(String, Value)> {
    let root = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../tests/goldens")
        .join(dir);
    let mut cases: Vec<(String, Value)> = std::fs::read_dir(&root)
        .unwrap_or_else(|e| panic!("{root:?}: {e}"))
        .map(|entry| {
            let path = entry.unwrap().path();
            let name = path.file_stem().unwrap().to_str().unwrap().to_string();
            (
                name,
                serde_json::from_str(&std::fs::read_to_string(&path).unwrap()).unwrap(),
            )
        })
        .collect();
    cases.sort_by(|a, b| a.0.cmp(&b.0));
    cases
}

/// Legacy payload → ledger mapping → typed `Job`.
fn mapped_job(payload: &Value) -> Job {
    let mut raw = payload.clone();
    map_payload(&mut raw);
    serde_json::from_value(raw).unwrap()
}

/// Project a golden job payload the same way the migration maps it — the
/// value is compared against `serde_json::to_value(&job)`.
fn project(payload: &Value) -> Value {
    let mut job = payload.clone();
    map_payload(&mut job);
    job
}

fn status_of(kind: ErrorKind) -> u16 {
    match kind {
        ErrorKind::BadRequest => 400,
        ErrorKind::NotFound => 404,
        ErrorKind::Conflict => 409,
    }
}

fn run_case(name: &str, case: &Value) {
    let input = &case["input"];
    let request = &input["request"];
    let method = request["method"].as_str().unwrap();
    let path = request["path"].as_str().unwrap();
    let body = &request["body"];

    let mut jobs: Vec<(String, Job)> = input["seed_jobs"]
        .as_object()
        .unwrap()
        .iter()
        .map(|(jid, payload)| (jid.clone(), mapped_job(payload)))
        .collect();

    // Route: PUT /api/jobs/<jid>/transcript · PATCH /api/jobs/<jid>/segments/<sid>
    let (result, jid) = if method == "PUT" {
        let jid = path
            .strip_prefix("/api/jobs/")
            .and_then(|rest| rest.strip_suffix("/transcript"))
            .unwrap()
            .to_string();
        let job = jobs
            .iter_mut()
            .find(|(id, _)| *id == jid)
            .map(|(_, j)| j)
            .unwrap_or_else(|| panic!("{name}: unknown job {jid}"));
        (
            put_transcript(job, TranscriptPut::from_value(body), FIXED_TIME),
            jid,
        )
    } else {
        let rest = path.strip_prefix("/api/jobs/").unwrap();
        let (jid, sid) = rest.split_once("/segments/").unwrap();
        let job = jobs
            .iter_mut()
            .find(|(id, _)| id == jid)
            .map(|(_, j)| j)
            .unwrap_or_else(|| panic!("{name}: unknown job {jid}"));
        (
            SegmentPatch::from_value(body)
                .and_then(|patch| patch_segment(job, sid, patch, FIXED_TIME)),
            jid.to_string(),
        )
    };

    let output = &case["output"];
    match result {
        Ok(()) => {
            assert_eq!(
                output["status"],
                json!(200),
                "{name}: expected success status"
            );
            // JobStore.put stamps updated=now on the successful write; the
            // PUT route then serves `get(jid)` — the post-write payload.
            {
                let job = jobs.iter_mut().find(|(id, _)| *id == jid).unwrap();
                job.1.updated = FIXED_TIME;
            }
            let served = jobs
                .iter()
                .find(|(id, _)| *id == jid)
                .map(|(_, j)| serde_json::to_value(j).unwrap())
                .unwrap();
            let expected = project(&output["body"]);
            if method == "PUT" {
                assert_eq!(served, expected, "{name}: PUT response body");
            } else {
                // PATCH serves {segment, revision} — the stored row value.
                assert_eq!(served["revision"], expected["revision"], "{name}");
                let sid = path.split_once("/segments/").unwrap().1;
                let job = &jobs.iter().find(|(id, _)| *id == jid).unwrap().1;
                let segment = job
                    .document
                    .as_ref()
                    .unwrap()
                    .segments
                    .iter()
                    .find(|s| s.id == sid)
                    .unwrap();
                // v0.6.6 returns the stored row verbatim — keys outside the
                // product model (segment-private engine state like
                // `decode_core_plan`) are projected out at the typed
                // boundary, so project the golden through it too.
                let expected_segment: Segment =
                    serde_json::from_value(expected["segment"].clone()).unwrap();
                assert_eq!(
                    serde_json::to_value(segment).unwrap(),
                    serde_json::to_value(&expected_segment).unwrap(),
                    "{name}"
                );
            }
        }
        Err(e) => {
            assert_eq!(
                output["status"],
                json!(status_of(e.kind)),
                "{name}: status kind",
            );
            let expected_body = output["body"].clone();
            assert_eq!(json!({"error": e.message}), expected_body, "{name}");
        }
    }

    // Stored jobs: compare every seeded row after the same projection.
    let stored = output["stored_jobs"].as_object().unwrap();
    for (jid, job) in &jobs {
        let expected = project(&stored[jid]);
        assert_eq!(
            serde_json::to_value(job).unwrap(),
            expected,
            "{name}: stored job {jid}"
        );
    }
}

#[test]
fn edit_goldens() {
    let cases = goldens("edit");
    assert!(!cases.is_empty());
    for (name, case) in &cases {
        run_case(name, case);
    }
}
