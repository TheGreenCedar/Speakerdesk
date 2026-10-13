//! Canonical accuracy publication shared by saved work and the owned live worker.
mod rolling;
use super::*;
pub(crate) use rolling::{Rolling, Runner};
use speakerdesk_core::{document::Segment, live};

const RETAINED: &str = "Accuracy pass paused. Saved words and audio are retained.";

pub(crate) fn available(job: &Job) -> bool {
    job.status == "ready"
        && job.document.as_ref().is_some_and(|doc| {
            doc.segments
                .iter()
                .any(speakerdesk_core::import::refinable_segment)
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
    begin_guarded(app, jid, None)
}

/// Run after the accepting save and any retry worker release their leases.
/// A later edit/control/operation supersedes this one-shot continuation.
pub(super) fn resume_after_repair(
    app: &AppState,
    id: &str,
    repair: (String, String, String),
    generation: u64,
) {
    if let Err(error) = begin_guarded(app, id, Some((repair, generation))) {
        tracing::debug!(
            ?error,
            "accepted repair accuracy continuation was superseded or blocked"
        );
    }
}

fn begin_guarded(
    app: &AppState,
    jid: &str,
    expected: Option<((String, String, String), u64)>,
) -> Result<Job, ApiError> {
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
    let accepted_repair = expected.is_some();
    if let Some(((row_id, retry_id, authority), generation)) = expected {
        if !job.document.as_ref().is_some_and(|document| {
            document.segments.iter().any(|row| {
                row.id == row_id
                    && row.retry_candidate.as_ref().is_some_and(|receipt| {
                        receipt["id"] == retry_id && receipt["accepted_at"].is_number()
                    })
                    && live::segment_version(row) == authority
            })
        }) || job
            .extra
            .get("operation_generation")
            .and_then(Value::as_u64)
            .unwrap_or(0)
            != generation
            || job.extra.get("refinement_status").and_then(Value::as_str) != Some("unresolved")
        {
            return Err(ApiError::new(409, RETAINED));
        }
    }
    // Accepting the final blocked passage protects it from inference. The same
    // guarded continuation must still finalize the now-empty accuracy plan.
    if !available(&job) && !(accepted_repair && job.status == "ready") {
        return Err(ApiError::new(
            409,
            "Accuracy work currently requires a saved Ready recording with canonical passages.",
        ));
    }
    speakerdesk_core::import::repair_import_metadata(&mut job)
        .map_err(|error| ApiError::new(409, error.to_string()))?;
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
    let shared = app.shared.clone();
    let admission = app.admission.clone();
    let queue = app.worker.queue_handle();
    let generation = operation.generation;
    app.worker.enqueue(move || {
        if let Err(error) = execute(&operation, models.as_ref()) {
            tracing::error!(?error, "saved accuracy pass could not publish its state");
        }
        drop(operation);
        drop(lease);
        queue.enqueue(move || {
            separation::automatic(&shared, &admission, &id, generation);
        });
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
    captured(current, epoch)
        .is_none_or(|entry| entry.get("language") == prior.rest.get("language_mode"))
        && captured(current, epoch) == captured(original, epoch)
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

fn context_generation(job: &Job) -> u64 {
    job.extra
        .get("language_context_generation")
        .and_then(Value::as_u64)
        .unwrap_or(0)
}
pub(crate) fn context_superseded(job: &Job, candidate: &Segment) -> bool {
    let Some(detection) = &candidate.language_detection else {
        return false;
    };
    let decisions = std::iter::once(detection).chain(
        detection["probes"]
            .as_array()
            .into_iter()
            .flatten()
            .map(|probe| &probe["decision"]),
    );
    decisions
        .filter_map(|decision| decision.get("context_evidence"))
        .any(|evidence| {
            evidence["interpretation_generation"].as_u64() != Some(context_generation(job))
        })
        || job
            .document
            .as_ref()
            .is_some_and(|document| !live::borrowed_context_current(&document.segments, candidate))
}
fn selected_context(
    summary: &mut live::MainLanguageSummary,
    job: &Job,
    prior: &Segment,
) -> Option<speakerdesk_core::language::LanguageContext> {
    let rows = &job.document.as_ref()?.segments;
    if summary.has_context(prior) && summary.context(rows, prior).is_none() {
        *summary = live::MainLanguageSummary::from_segments(rows);
    }
    let mut context = summary.context(rows, prior)?;
    context.evidence.as_mut()?.interpretation_generation = context_generation(job);
    Some(context)
}

fn publish_candidate(
    current: &mut Job,
    original: &Job,
    prior: &Segment,
    candidate: &Segment,
) -> Result<bool, live::LiveError> {
    if prior.separation.is_some()
        || !fresh(current, original, prior)
        || context_superseded(current, candidate)
    {
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

fn publish_language_issue(current: &mut Job, prior: &Segment, error: &live::LiveError) {
    let Some(row) = current
        .document
        .as_mut()
        .and_then(|document| document.segments.iter_mut().find(|row| row.id == prior.id))
    else {
        return;
    };
    if live::segment_version(row) != live::segment_version(prior) {
        return;
    }
    row.rest.remove("refinement_issue");
    if let live::LiveError::Language { reason, message } = error {
        row.rest.insert("refinement_issue".into(),json!({"kind":"language","reason":reason,"message":message,"segment_version":live::segment_version(prior)}));
    }
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
    let (ids, mut summary, mut summary_generation) = {
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
        let rows = job
            .document
            .as_ref()
            .map(|d| d.segments.as_slice())
            .unwrap_or_default();
        (
            rows.iter().map(|s| s.id.clone()).collect::<Vec<_>>(),
            live::MainLanguageSummary::from_segments(rows),
            context_generation(&job),
        )
    };
    let mut unresolved = false;
    for id in ids {
        let mut refreshed = false;
        loop {
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
                    break;
                };
                if segment.separation.is_some()
                    || segment
                        .protected_fields
                        .as_ref()
                        .is_some_and(|f| !f.is_empty())
                    || checkpoint(&job, segment)
                {
                    break;
                }
                (job.clone(), segment.clone())
            };
            if summary_generation != context_generation(&original) {
                summary = live::MainLanguageSummary::from_segments(
                    &original.document.as_ref().expect("document").segments,
                );
                summary_generation = context_generation(&original);
            }
            let result = (|| {
                let path = crate::native::passage_recording_path(
                    &operation.shared.data_dir,
                    &original,
                    &prior,
                )
                .map_err(|e| live::LiveError::Detail(e.to_string()))?;
                let reader = audio::RecordingReader::open(path)
                    .map_err(|e| live::LiveError::Detail(e.to_string()))?;
                let context = selected_context(&mut summary, &original, &prior);
                live::refine_saved_recording_with_context(
                    &models,
                    &prior,
                    &reader,
                    vocabulary.as_deref(),
                    context,
                    context_generation(&original),
                )
            })();
            let mut store = operation.shared.store.lock().expect("store");
            let mut retry_context = false;
            let publication = store.update_job(&operation.id, |current| {
                if !operation.current(current) {
                    return Err(ApiError::new(409, RETAINED));
                }
                if !fresh(current, &original, &prior) {
                    unresolved = true;
                    return Ok(());
                }
                if result
                    .as_ref()
                    .is_ok_and(|candidate| context_superseded(current, candidate))
                {
                    if !refreshed {
                        retry_context = true;
                        return Ok(());
                    }
                    unresolved = true;
                    current.extra.insert(
                        "refinement_error".into(),
                        json!(
                            "Language context changed again; previous words and audio are retained."
                        ),
                    );
                    return Ok(());
                }
                match &result {
                    Ok(candidate) => match publish_candidate(current, &original, &prior, candidate)
                    {
                        Ok(true) => {
                            let row = current
                                .document
                                .as_ref()
                                .expect("document")
                                .segments
                                .iter()
                                .find(|row| row.id == prior.id)
                                .expect("accepted passage");
                            if summary_generation == context_generation(current) {
                                summary.accept_publication(Some(&prior), row);
                            } else {
                                summary = live::MainLanguageSummary::from_segments(
                                    &current.document.as_ref().expect("document").segments,
                                );
                                summary_generation = context_generation(current);
                            }
                        }
                        Ok(false) => unresolved = true,
                        Err(error) => {
                            unresolved = true;
                            current
                                .extra
                                .insert("refinement_error".into(), json!(error.to_string()));
                        }
                    },
                    Err(error) => {
                        unresolved = true;
                        publish_language_issue(current, &prior, error);
                        current
                            .extra
                            .insert("refinement_error".into(), json!(error.to_string()));
                    }
                }
                Ok::<_, ApiError>(())
            });
            match publication {
                Err(e) if e.status == 409 => return Ok(()),
                Err(e) => return Err(e),
                _ => {}
            }
            drop(store);
            if retry_context {
                refreshed = true;
                let job = load_job(
                    &operation.shared.store.lock().expect("store"),
                    &operation.id,
                    false,
                )?;
                summary = live::MainLanguageSummary::from_segments(
                    &job.document.as_ref().expect("document").segments,
                );
                summary_generation = context_generation(&job);
                continue;
            }
            break;
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
