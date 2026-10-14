//! Import progress is committed with its resume cursor before it reaches the editor.
use super::*;
use sha2::{Digest, Sha256};
use speakerdesk_core::import::{ImportError, ImportProgress, ImportResume};

pub(super) fn infer(
    shared: &Arc<Shared>,
    operation: &OperationGuard,
    job: &Job,
    models: &dyn ApplicationModels,
) -> Result<Document, String> {
    let Some(pipeline) = models.import_identity() else {
        return models.infer(job, &operation.cancellation);
    };
    let path = crate::native::recording_path(&shared.data_dir, job).map_err(|e| e.to_string())?;
    let mut source = fs::File::open(path).map_err(|e| e.to_string())?;
    let mut hash = Sha256::new();
    let mut bytes = [0; 65536];
    loop {
        if operation.cancellation.is_cancelled() {
            return Err("Import cancelled.".into());
        }
        let count = source.read(&mut bytes).map_err(|e| e.to_string())?;
        if count == 0 {
            break;
        }
        hash.update(&bytes[..count]);
    }
    let source_id = format!("{:x}", hash.finalize());
    let resume = if let Some(checkpoint) = job.extra.get("import_checkpoint") {
        if checkpoint["source_sha256"] != source_id || checkpoint["language"] != job.language {
            return Err("The original import audio or language no longer matches its checkpoint. Accepted words are retained; restore the original source to resume.".into());
        }
        // Persist this invalidation: later saved-accuracy work must not resurrect
        // observations from a different model/pipeline generation.
        let document = shared
            .store
            .lock()
            .expect("store")
            .update_job(&operation.id, |current| {
                if !operation.current(current) || operation.cancellation.is_cancelled() {
                    return Err(ApiError::new(409, "Import resume was superseded."));
                }
                let document = current.document.as_mut().ok_or_else(|| {
                    ApiError::new(409, "Import checkpoint has no retained document.")
                })?;
                for row in &mut document.segments {
                    if row.rest.get("import_pipeline").and_then(Value::as_str) != Some(&pipeline) {
                        row.language_detection = None;
                    }
                }
                Ok::<_, ApiError>(document.clone())
            })
            .map_err(|e| e.message)?
            .ok_or("Import was removed before resume.")?
            .1;
        Some(ImportResume {
            next_work: checkpoint["next_work"]
                .as_u64()
                .ok_or("Invalid import cursor.")? as usize,
            plan_sha256: checkpoint["plan_sha256"]
                .as_str()
                .ok_or("Invalid import plan.")?
                .into(),
            plan: serde_json::from_value(checkpoint["plan"].clone()).map_err(|e| e.to_string())?,
            document,
        })
    } else {
        None
    };
    models.infer_progress(
        job,
        &operation.cancellation,
        resume.as_ref(),
        &mut |progress| publish(shared, operation, &source_id, &pipeline, progress),
    )
}

fn publish(
    shared: &Shared,
    operation: &OperationGuard,
    source_id: &str,
    pipeline: &str,
    progress: ImportProgress<'_>,
) -> Result<Vec<Segment>, ImportError> {
    let result = shared.store.lock().expect("store").update_job(&operation.id, |current| {
        if operation.cancellation.is_cancelled() || !operation.current(current) || current.status != "processing" {
            return Err(ApiError::new(409, "Import publication was superseded."));
        }
        if let Some(checkpoint) = current.extra.get("import_checkpoint") {
            if checkpoint["source_sha256"] != source_id || checkpoint["plan_sha256"] != progress.plan_sha256
                || checkpoint["next_work"].as_u64().is_some_and(|cursor| cursor > progress.next_work as u64) {
                return Err(ApiError::new(409, "Import checkpoint changed before publication."));
            }
        }
        let document = current.document.get_or_insert_with(|| Document {
            schema_version:1, speakers:progress.speakers.clone(), segments:Vec::new(),
            provenance:Some(progress.provenance.clone()), warnings:progress.warnings.to_vec(),
            diarization:progress.diarization.to_vec(), edited_at:None, rest:Map::new(),
        });
        for (id, name) in progress.speakers {document.speakers.entry(id.clone()).or_insert_with(|| name.clone());}
        let mut accepted = Vec::with_capacity(progress.rows.len());
        for proposed in progress.rows {
            if let Some(prior) = document.segments.iter().find(|row| row.id == proposed.id) {
                // A committed cursor never republishes a row, including user edits.
                accepted.push(prior.clone());
            } else {
                if !speakerdesk_core::live::borrowed_context_current(&document.segments, proposed) {
                    // Do not advance the cursor: retry re-reads current ownership.
                    return Err(ApiError::new(409, "A passage used for language context changed. Saved words are retained; resume transcription to use the correction."));
                }
                let mut row = proposed.clone();
                row.rest.insert("import_pipeline".into(), json!(pipeline));
                document.segments.push(row.clone());
                accepted.push(row);
            }
        }
        document.segments.sort_by(|a,b| a.start.total_cmp(&b.start).then(a.end.total_cmp(&b.end)).then(a.id.cmp(&b.id)));
        let validated = checked_document(document.clone(), current.duration.unwrap_or(0.)).map_err(|e| ApiError::new(422,e))?;
        let accepted = accepted.iter().filter_map(|row| validated.segments.iter().find(|stored| stored.id == row.id).cloned()).collect();
        current.document = Some(validated);
        current.extra.insert("import_checkpoint".into(), json!({"version":1,"source_sha256":source_id,"pipeline":pipeline,
            "language":current.language,"plan_sha256":progress.plan_sha256,"plan":progress.plan,"next_work":progress.next_work}));
        current.extra.insert("import_progress".into(), json!({"completed":progress.next_work,"total":progress.total_work}));
        current.extra.insert("import_resumable".into(), json!(true));
        current.message = format!("Transcribing: {} of {} passages saved.",progress.next_work,progress.total_work);
        Ok::<_,ApiError>(accepted)
    });
    result
        .map_err(|error| ImportError::new(error.message))?
        .map(|(_, rows)| rows)
        .ok_or_else(|| ImportError::new("The import was removed before publication."))
}

#[cfg(test)]
mod tests {
    use super::*;
    use speakerdesk_core::import::ImportPlan;

    #[test]
    fn accepted_import_rows_and_cursor_are_atomic_and_edits_survive_republication() {
        let app = AppState::new();
        let mut job: Job = serde_json::from_value(json!({"id":"aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa","name":"import","kind":"import","status":"processing","language":"en","duration":10.,"created":0.,"updated":0.,"revision":0})).unwrap();
        job.status = "uploaded".into();
        app.insert_job(job.clone());
        let operation = app
            .begin_operation(&mut app.store(), &mut job, Activity::Import)
            .unwrap();
        app.store()
            .update_job(&job.id, |job| {
                job.status = "processing".into();
                Ok::<_, ApiError>(())
            })
            .unwrap();
        let plan = ImportPlan {
            version: 1,
            samples: 160000,
            language: "en".into(),
            chunks: vec![],
            windows: vec![],
            diarization: vec![],
        };
        let speakers = std::collections::BTreeMap::from([("speaker_0".into(), "Speaker 1".into())]);
        let provenance = json!({"capture_mode":"import"});
        let mut rows: Vec<Segment> = serde_json::from_value(json!([{"id":"import-0-0","start":0.,"end":1.,"speaker":"speaker_0","text":"Initial words","canonical_utterance_id":"import-0-0","canonical_state":"sealed","machine_revision":1}])).unwrap();
        let publish_rows = |rows: &[Segment], cursor| {
            publish(
                &app.shared,
                &operation,
                "source",
                "pipeline",
                ImportProgress {
                    next_work: cursor,
                    total_work: 2,
                    plan_sha256: "plan",
                    plan: &plan,
                    rows,
                    speakers: &speakers,
                    provenance: &provenance,
                    warnings: &[],
                    diarization: &[],
                },
            )
        };
        publish_rows(&rows, 1).unwrap();
        app.store()
            .update_job(&job.id, |current| {
                let row = &mut current.document.as_mut().unwrap().segments[0];
                row.text = "Human correction".into();
                row.protected_fields = Some(vec!["text".into()]);
                Ok::<_, ApiError>(())
            })
            .unwrap();
        assert_eq!(publish_rows(&rows, 1).unwrap()[0].text, "Human correction");
        rows[0].id = "import-1-0".into();
        rows[0].start = 1.;
        rows[0].end = 2.;
        rows[0].language_detection = Some(
            json!({"context_evidence":{"anchors":[{"utterance_id":"missing","row_start_sample":0}]}}),
        );
        assert!(publish_rows(&rows, 2).is_err());
        let saved = app.store().get(&job.id, false).unwrap().unwrap();
        assert_eq!(saved.document.as_ref().unwrap().segments.len(), 1);
        assert_eq!(saved.extra["import_checkpoint"]["next_work"], 1);
        operation.cancellation.cancel();
        rows[0].language_detection = None;
        assert!(publish_rows(&rows, 2).is_err());
        let saved = app.store().get(&job.id, false).unwrap().unwrap();
        assert_eq!(saved.document.unwrap().segments[0].text, "Human correction");
        assert_eq!(saved.extra["import_checkpoint"]["next_work"], 1);
    }
}
