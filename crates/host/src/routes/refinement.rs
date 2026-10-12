//! Canonical accuracy publication shared by saved work and the owned live worker.
mod rolling;
use super::*;
pub(crate) use rolling::{Rolling, Runner};
use speakerdesk_core::{document::Segment, live};

const RETAINED: &str = "Accuracy pass paused. Saved words and audio are retained.";

pub(crate) fn available(job: &Job) -> bool {
    job.status == "ready"
        && job.document.as_ref().is_some_and(|doc| {
            doc.segments.iter().any(|s| {
                s.canonical_utterance_id.is_some()
                    && s.rest.get("canonical_state").and_then(Value::as_str) == Some("sealed")
            })
        })
}

pub(crate) async fn action(
    State(app): State<AppState>,
    AxumPath((jid, action)): AxumPath<(String, String)>,
) -> Response {
    if let Some(response) = crate::meeting::refinement_action(&app, &jid, &action).await {
        return response;
    }
    blocking(move || {
        let result = match action.as_str() {
            "resume" => begin(&app, &jid),
            "pause" => pause(&app, &jid),
            _ => Err(ApiError::new(404, NOT_FOUND)),
        };
        match result {
            Ok(job) => (
                if action == "resume" {
                    StatusCode::ACCEPTED
                } else {
                    StatusCode::OK
                },
                Json(stored_job_json(&job)),
            )
                .into_response(),
            Err(error) => from_api(error),
        }
    })
    .await
}

pub(crate) async fn history(
    State(app): State<AppState>,
    AxumPath((jid, window)): AxumPath<(String, String)>,
) -> Response {
    blocking(move || {
        let result = (|| {
            let job = load_job(&app.store(), &require_job_id(&jid)?, false)?;
            job.extra
                .get("refinement_history")
                .and_then(|h| h.get(&window))
                .cloned()
                .ok_or_else(|| ApiError::new(404, NOT_FOUND))
        })();
        match result {
            Ok(value) => Json(value).into_response(),
            Err(error) => from_api(error),
        }
    })
    .await
}

fn begin(app: &AppState, jid: &str) -> Result<Job, ApiError> {
    let id = require_job_id(jid)?;
    let lease = app
        .admission
        .lease(Activity::Refinement)
        .map_err(|e| ApiError::new(409, e.message()))?;
    let models = app.models();
    if !app.model_status().0.is_empty() || models.live_models(&CancellationToken::new()).is_none() {
        return Err(ApiError::new(409, messages::SETUP_BEFORE_TRANSCRIPTION));
    }
    let mut store = app.store();
    let mut job = load_job(&store, &id, false)?;
    if !available(&job) {
        return Err(ApiError::new(
            409,
            "Accuracy work currently requires a saved Ready recording with canonical passages.",
        ));
    }
    if !job.extra.contains_key("rolling_refinement") {
        job.extra.insert(
            "rolling_refinement".into(),
            json!({"version":1,"mode":"saved","completed":{}}),
        );
    }
    job.extra.get_mut("rolling_refinement").expect("plan")["mode"] = json!("saved");
    job.extra.get_mut("rolling_refinement").expect("plan")["failed"] = json!({});
    job.extra
        .insert("refinement_status".into(), json!("waiting"));
    job.extra.remove("refinement_error");
    job.message = "Saved accuracy pass queued. Existing words remain editable.".into();
    job.revision = job
        .revision
        .checked_add(1)
        .ok_or_else(|| ApiError::new(409, "Transcript revision exhausted."))?;
    job.updated = unix_now();
    let operation = app.begin_refinement_operation(&mut store, &mut job)?;
    drop(store);
    app.worker.enqueue(move || {
        let _lease = lease;
        if let Err(error) = execute(&operation, models.as_ref()) {
            tracing::error!(?error, "saved accuracy pass could not publish its state");
        }
    });
    Ok(job)
}

pub(super) fn interrupt(job: &mut Job) {
    if job.extra.get("rolling_refinement").is_some()
        && job
            .extra
            .get("refinement_status")
            .and_then(Value::as_str)
            .is_some_and(|s| ["waiting", "refining"].contains(&s))
    {
        job.extra
            .insert("refinement_status".into(), json!("paused"));
        job.message = RETAINED.into();
    }
}

fn pause(app: &AppState, jid: &str) -> Result<Job, ApiError> {
    let id = require_job_id(jid)?;
    let result = app
        .store()
        .update_job(&id, |job| {
            if job
                .extra
                .get("rolling_refinement")
                .is_none_or(|p| p["mode"] != "saved")
            {
                return Err(ApiError::new(
                    409,
                    "There is no saved accuracy pass to pause.",
                ));
            }
            let running = app
                .shared
                .operations
                .lock()
                .expect("operations")
                .get(&id)
                .cloned();
            if let Some(active) = &running {
                if !job
                    .extra
                    .get("refinement_status")
                    .and_then(Value::as_str)
                    .is_some_and(|s| ["waiting", "refining"].contains(&s))
                    || active.activity != Activity::Refinement
                    || job
                        .extra
                        .get("operation_generation")
                        .and_then(Value::as_u64)
                        != Some(active.generation)
                {
                    return Err(ApiError::new(409, "Another operation owns this recording."));
                }
                job.extra.insert(
                    "operation_generation".into(),
                    json!(
                        active
                            .generation
                            .checked_add(1)
                            .ok_or_else(|| ApiError::new(409, "Operation generation exhausted"))?
                    ),
                );
            }
            job.extra
                .insert("refinement_status".into(), json!("paused"));
            job.message = RETAINED.into();
            Ok::<_, ApiError>(running.map(|active| active.cancellation))
        })?
        .ok_or_else(|| ApiError::new(404, NOT_FOUND))?;
    if let Some(cancel) = result.1 {
        cancel.cancel();
    }
    Ok(result.0)
}

fn checkpoint(job: &Job, segment: &Segment) -> bool {
    job.extra
        .get("rolling_refinement")
        .and_then(|p| p["completed"].get(&segment.id))
        .and_then(Value::as_str)
        == Some(live::segment_version(segment).as_str())
}

fn fresh(current: &Job, original: &Job, prior: &Segment) -> bool {
    let epoch = prior.rest.get("language_epoch");
    fn captured<'a>(job: &'a Job, epoch: Option<&Value>) -> Option<&'a Value> {
        job.extra
            .get("language_history")
            .and_then(Value::as_array)
            .and_then(|history| history.iter().find(|entry| Some(&entry["epoch"]) == epoch))
    }
    let covered = |job: &Job| {
        job.extra
            .get("language_history")
            .and_then(Value::as_array)
            .is_none_or(|history| {
                history
                    .iter()
                    .filter(|entry| entry["epoch"].as_i64() > epoch.and_then(Value::as_i64))
                    .all(|entry| {
                        entry["start_sample"].as_i64().unwrap_or(0)
                            >= prior.rest["end_sample"].as_i64().unwrap_or(0)
                    })
            })
    };
    captured(current, epoch) == captured(original, epoch)
        && covered(current)
        && current
            .document
            .as_ref()
            .and_then(|d| d.segments.iter().find(|s| s.id == prior.id))
            .is_some_and(|s| {
                crate::native::passage_audio_unchanged(current, original, s, prior)
                    && live::segment_version(s) == live::segment_version(prior)
            })
}

fn publish_candidate(
    current: &mut Job,
    original: &Job,
    prior: &Segment,
    candidate: &Segment,
) -> Result<bool, live::LiveError> {
    if !fresh(current, original, prior) {
        return Ok(false);
    }
    let document = current.document.as_mut().expect("fresh document");
    if !live::publish_saved_passage(document, prior, candidate)? {
        return Ok(false);
    }
    let version = live::segment_version(
        document
            .segments
            .iter()
            .find(|s| s.id == prior.id)
            .expect("accepted canonical row"),
    );
    current.extra.get_mut("rolling_refinement").expect("plan")["completed"][&prior.id] =
        json!(version);
    let history = current
        .extra
        .entry("refinement_history")
        .or_insert_with(|| json!({}));
    history[&prior.id] = json!({"window_id":prior.id,"created":unix_now(),"segments":[{
        "id":prior.id,"start":prior.start,"end":prior.end,"speaker":prior.speaker,"text":prior.text
    }]});
    Ok(true)
}

fn execute(
    operation: &OperationGuard,
    application: &dyn ApplicationModels,
) -> Result<(), ApiError> {
    let vocabulary = application.live_alignment();
    let Some(models) = application.live_models(&operation.cancellation) else {
        operation
            .shared
            .store
            .lock()
            .expect("store")
            .update_job(&operation.id, |job| {
                if !operation.current(job) {
                    return Err(ApiError::new(409, RETAINED));
                }
                job.extra
                    .insert("refinement_status".into(), json!("unresolved"));
                job.extra.insert(
                    "refinement_error".into(),
                    json!(messages::SETUP_BEFORE_TRANSCRIPTION),
                );
                Ok::<_, ApiError>(())
            })?;
        return Ok(());
    };
    let ids = {
        let mut store = operation.shared.store.lock().expect("store");
        let Some((job, ())) = store.update_job(&operation.id, |job| {
            if !operation.current(job) {
                return Err(ApiError::new(409, RETAINED));
            }
            job.extra
                .insert("refinement_status".into(), json!("refining"));
            Ok::<_, ApiError>(())
        })?
        else {
            return Ok(());
        };
        job.document
            .as_ref()
            .map(|d| d.segments.iter().map(|s| s.id.clone()).collect::<Vec<_>>())
            .unwrap_or_default()
    };
    let mut unresolved = false;
    for id in ids {
        let (original, prior) = {
            let store = operation.shared.store.lock().expect("store");
            let job = load_job(&store, &operation.id, false)?;
            if !operation.current(&job) {
                return Ok(());
            }
            let Some(segment) = job
                .document
                .as_ref()
                .and_then(|d| d.segments.iter().find(|s| s.id == id))
            else {
                continue;
            };
            if segment
                .protected_fields
                .as_ref()
                .is_some_and(|f| !f.is_empty())
                || checkpoint(&job, segment)
            {
                continue;
            }
            (job.clone(), segment.clone())
        };
        let result = (|| {
            let path = crate::native::passage_recording_path(
                &operation.shared.data_dir,
                &original,
                &prior,
            )
            .map_err(|e| e.to_string())?;
            let reader = audio::RecordingReader::open(path).map_err(|e| e.to_string())?;
            live::refine_saved_recording(&models, &prior, &reader, vocabulary.as_deref())
                .map_err(|e| e.to_string())
        })();
        let mut store = operation.shared.store.lock().expect("store");
        let publication = store.update_job(&operation.id, |current| {
            if !operation.current(current) {
                return Err(ApiError::new(409, RETAINED));
            }
            if !fresh(current, &original, &prior) {
                unresolved = true;
                return Ok(());
            }
            match &result {
                Ok(candidate) => match publish_candidate(current, &original, &prior, candidate) {
                    Ok(true) => {}
                    Ok(false) => unresolved = true,
                    Err(error) => {
                        unresolved = true;
                        current
                            .extra
                            .insert("refinement_error".into(), json!(error.to_string()));
                    }
                },
                Err(message) => {
                    unresolved = true;
                    current
                        .extra
                        .insert("refinement_error".into(), json!(message));
                }
            }
            Ok::<_, ApiError>(())
        });
        match publication {
            Err(e) if e.status == 409 => return Ok(()),
            Err(e) => return Err(e),
            _ => {}
        }
    }
    let mut store = operation.shared.store.lock().expect("store");
    store.update_job(&operation.id, |job| {
        if !operation.current(job) { return Err(ApiError::new(409,RETAINED)); }
        job.extra.insert("refinement_status".into(),json!(if unresolved {"unresolved"} else {"complete"}));
        job.message = if unresolved {"Some passages could not be refined. Previous words and audio are retained; retry when ready."} else {"Saved accuracy pass complete. Your corrections and recorded speaker identities are retained."}.into();
        Ok::<_,ApiError>(())
    })?;
    Ok(())
}
