//! Explicit-language candidates share the import worker and saved recording owner.
use super::*;
use speakerdesk_core::document::Segment;

#[derive(Deserialize)]
struct RetryRequest {
    revision: i64,
    segment_id: String,
    language: String,
}

pub(super) async fn retry_passage(
    State(app): State<AppState>,
    AxumPath(jid): AxumPath<String>,
    body: Bytes,
) -> Response {
    blocking(move || match begin(&app, &jid, &body) {
        Ok(body) => (StatusCode::ACCEPTED, Json(body)).into_response(),
        Err(error) => from_api(error),
    })
    .await
}

fn begin(app: &AppState, jid: &str, body: &[u8]) -> Result<Value, ApiError> {
    let request: RetryRequest = serde_json::from_slice(body)
        .map_err(|_| ApiError::new(400, messages::CHOOSE_PASSAGE_AND_LANGUAGE))?;
    if request.language == languages::AUTOMATIC || !languages::is_supported(&request.language) {
        return Err(ApiError::new(400, messages::RETRY_LANGUAGE));
    }
    let id = require_job_id(jid)?;
    let lease = app
        .admission
        .lease(Activity::Refinement)
        .map_err(|error| ApiError::new(409, error.message()))?;
    let mut store = app.store();
    let mut job = load_job(&store, &id, false)?;
    if ACTIVE.contains(&job.status.as_str()) || meeting_id(&app.shared).is_some() {
        return Err(ApiError::new(409, messages::RETRY_BUSY));
    }
    if request.revision != job.revision {
        return Err(ApiError::new(409, messages::RETRY_STALE));
    }
    let segment = job
        .document
        .as_ref()
        .and_then(|doc| doc.segments.iter().find(|s| s.id == request.segment_id))
        .cloned()
        .ok_or_else(|| ApiError::new(404, NOT_FOUND))?;
    if !segment.start.is_finite()
        || !segment.end.is_finite()
        || segment.start < 0.
        || segment.end <= segment.start
        || segment.end - segment.start > limits::PASSAGE_RETRY_MAX_SECONDS
    {
        return Err(ApiError::new(400, messages::RETRY_TOO_LONG));
    }
    let path = crate::native::passage_recording_path(app.data_dir(), &job, &segment)
        .map_err(|_| ApiError::new(409, messages::RETRY_NEEDS_AUDIO))?;
    let retry_id = random_hex(16);
    job.extra.insert(
        "passage_retry".into(),
        json!({"id":retry_id,"segment_id":segment.id,"language":request.language}),
    );
    job.message = "Retrying this passage locally; existing words are retained.".into();
    job.status = "ready".into();
    job.revision += 1;
    job.updated = unix_now();
    let operation = app.begin_refinement_operation(&mut store, &mut job)?;
    drop(store);
    let body = stored_job_json(&job);
    let models = app.models();
    app.worker.enqueue(move || {
        let _lease = lease;
        execute(
            &operation,
            models.as_ref(),
            &job,
            &segment,
            &path,
            &retry_id,
            &request.language,
        );
    });
    Ok(body)
}

fn unchanged(current: &Job, original: &Job, prior: &Segment) -> bool {
    current
        .document
        .as_ref()
        .and_then(|doc| doc.segments.iter().find(|s| s.id == prior.id))
        .is_some_and(|s| {
            crate::native::passage_audio_unchanged(current, original, s, prior)
                && s.text == prior.text
                && s.start == prior.start
                && s.end == prior.end
                && s.speaker == prior.speaker
                && s.language == prior.language
                && s.language_detection == prior.language_detection
                && s.machine_revision == prior.machine_revision
                && s.protected_fields.as_deref().unwrap_or_default()
                    == prior.protected_fields.as_deref().unwrap_or_default()
        })
}

fn execute(
    operation: &OperationGuard,
    models: &dyn ApplicationModels,
    original: &Job,
    segment: &Segment,
    path: &Path,
    retry_id: &str,
    language: &str,
) {
    let result = (|| {
        {
            let store = operation.shared.store.lock().expect("jobs");
            let current = load_job(&store, &original.id, false).map_err(|e| e.message)?;
            if !operation.current(&current) || !unchanged(&current, original, segment) {
                return Err(messages::RETRY_STALE.to_string());
            }
        }
        let pcm =
            audio::read_pcm16_range(path, segment.start, segment.end).map_err(|e| e.to_string())?;
        models.retry_passage(&pcm, language, &operation.cancellation)
    })();
    let mut store = operation.shared.store.lock().expect("jobs");
    let publication = store.update_job(&original.id, |current| {
        if !operation.current(current) || current.extra.get("passage_retry").and_then(|v|v["id"].as_str()) != Some(retry_id) {
            return Err(ApiError::new(409,messages::RETRY_STALE));
        }
        current.extra.remove("passage_retry");
        if !unchanged(current, original, segment) {
            current.message = "Passage changed during retry. Existing words are retained; retry the current passage.".into();
            return Ok(());
        }
        // The model sees crop-local samples; only the host owns absolute passage coordinates.
        let mut candidate = json!({"id":retry_id,"start":segment.start,"end":segment.end,"speaker":segment.speaker,
            "original_text":segment.text,"language":language,"created":unix_now(),"text":""});
        match &result {
            Ok(result) => {
                candidate["text"]=json!(result.text);candidate["review"]=json!(result.review);
                candidate["transcription_review"]=json!(result.transcription_review);
                candidate["retry_parts"]=json!(result.retry_parts);
                current.message = "Passage retry ready. Existing text is retained until you choose the result.".into();
            }
            Err(error) => {
                tracing::warn!(%error,"passage retry failed");
                candidate["error"]=json!(messages::RETRY_RETAINED);
                current.message = messages::RETRY_RETAINED.into();
            }
        }
        current.document.as_mut().expect("unchanged document").segments.iter_mut()
            .find(|s|s.id==segment.id).expect("unchanged passage").retry_candidate=Some(candidate);
        Ok::<_,ApiError>(())
    });
    if let Err(error) = publication
        && error.status != 409
    {
        tracing::error!(?error, "could not publish passage retry");
    }
}
