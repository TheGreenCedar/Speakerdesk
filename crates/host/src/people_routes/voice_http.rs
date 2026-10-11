//! Saved-audio voice operations. The store is held only for snapshots and publication.
use super::{
    Catalog, HttpError, editing_job, from_people, invalid_job, json_object, not_found,
    readiness_fields, unix_now,
};
use crate::{
    admission::Activity,
    native::recording_path,
    routes::{AppState, blocking, stored_job_json},
};
use axum::{
    Json,
    body::{Body, Bytes},
    extract::{Path, Query, Request, State},
    http::StatusCode,
    response::{IntoResponse, Response},
};
use serde::Deserialize;
use serde_json::{Map, Value, json};
use speakerdesk_core::{
    audio,
    document::Job,
    ids::JobId,
    messages, people,
    store::JobStore,
    voice::{self, Calibration, RecognitionMeeting, VoiceClip, VoiceProfile},
};

pub(super) fn conflict(message: impl Into<String>) -> HttpError {
    HttpError::new(StatusCode::CONFLICT, message)
}
pub(super) fn voice_error(error: voice::VoiceError) -> HttpError {
    HttpError::new(StatusCode::BAD_REQUEST, error.to_string())
}
fn reply(result: Result<Value, HttpError>) -> Response {
    match result {
        Ok(value) => Json(value).into_response(),
        Err(error) => error.response(),
    }
}

pub(super) fn meeting(job: &Job) -> Result<RecognitionMeeting, HttpError> {
    let doc = job
        .document
        .as_ref()
        .ok_or_else(|| conflict(messages::NAME_BEFORE_FINAL))?;
    let segments = doc
        .segments
        .iter()
        .map(|segment| serde_json::from_value(serde_json::to_value(segment)?))
        .collect::<Result<Vec<_>, _>>()
        .map_err(invalid_job)?;
    let mut meeting = RecognitionMeeting::local(job.id.as_str(), segments);
    meeting.provenance_kind = doc
        .provenance
        .as_ref()
        .and_then(|p| p.get("kind"))
        .and_then(Value::as_str)
        .map(str::to_owned);
    meeting.speakers = doc.speakers.clone();
    Ok(meeting)
}

pub(super) async fn voice_clips(
    State(app): State<AppState>,
    Path((jid, track)): Path<(String, String)>,
) -> Response {
    blocking(move || reply((|| {
        let id = JobId::parse(jid).map_err(|_|not_found())?;
        let policy = app.models().voice_policy();
        let job = app.store().get(&id,false)?.ok_or_else(not_found)?;
        let meeting = meeting(&job)?;
        if !meeting.speakers.contains_key(&track) { return Err(not_found()); }
        let minimum = policy.as_ref().map_or(2,|p|p.minimum_clips);
        let choices = voice::voice_clip_choices(&meeting,&track);
        let recommended = voice::automatic_clips(&meeting,&track,minimum);
        let recommended: Vec<_> = choices.iter().filter(|(_,clip)|recommended.contains(clip)).map(|(id,_)|id).collect();
        let clips: Vec<_> = choices.iter().map(|(id,clip)| {
            let text = meeting.segments.iter().find(|s|s.id==clip.segment_id).map(|s|s.text.as_str()).unwrap_or("");
            let mut row = json!(clip); row["id"]=json!(id); row["text"]=json!(text); row
        }).collect();
        let audio_available = recording_path(app.data_dir(),&job).is_ok();
        let source_verified = meeting.provenance_kind.as_deref()==Some("local_inference");
        let excluded: Vec<_> = meeting.segments.iter().filter(|s|s.speaker==track && !choices.iter().any(|(_,c)|c.segment_id==s.id)).map(|s| {
            let message = if !source_verified { "This transcript has no verified local speaker audio." }
                else if s.end-s.start < 2. { "Shorter than 2 seconds. Use a longer solo passage." }
                else { "The speaker or timing is unverified, edited, or still being finalized. Use an unchanged solo passage." };
            json!({"segment_id":s.id,"start":s.start,"end":s.end,"message":message})
        }).collect();
        let readiness = readiness_fields(policy.is_some());
        let (status,message) = if policy.is_none() { ("unavailable",readiness["voice_message"].as_str().unwrap()) }
            else if app.admission.busy() { ("busy","Finish the active recording or transcription, then refresh passages.") }
            else if !source_verified { ("source_unverified","This transcript has no verified local speaker audio.") }
            else if !audio_available { ("no_audio","The original recording is unavailable.") }
            else if recommended.len()<minimum { ("insufficient_clips","Voice setup needs at least two separate solo passages of 2–10 seconds.") }
            else { ("ready","Preview the passages and choose which to use. No voice is saved until you give consent.") };
        Ok(json!({"revision":job.revision,"track_id":track,"status":status,"message":message,"minimum_clips":minimum,
            "audio_available":audio_available,"readiness_code":readiness["readiness_code"],"next_action":readiness["next_action"],"clips":clips,"recommended_ids":recommended,"excluded":excluded}))
    })())).await
}

#[derive(Deserialize)]
pub(super) struct Preview {
    revision: i64,
    clip: String,
}
pub(super) async fn voice_preview(
    State(app): State<AppState>,
    Path((jid, track)): Path<(String, String)>,
    Query(query): Query<Preview>,
    request: Request,
) -> Response {
    let path = tokio::task::spawn_blocking(move || {
        let id = JobId::parse(jid).map_err(|_| not_found())?;
        let job = app.store().get(&id, false)?.ok_or_else(not_found)?;
        if job.revision != query.revision {
            return Err(conflict("Refresh passages before previewing this audio."));
        }
        if !voice::voice_clip_choices(&meeting(&job)?, &track)
            .iter()
            .any(|(id, _)| *id == query.clip)
        {
            return Err(conflict("This voice passage is no longer current."));
        }
        recording_path(app.data_dir(), &job)
            .map_err(|_| conflict("Original speaker audio is unavailable."))
    })
    .await;
    match path {
        Ok(Ok(path)) => match tower_http::services::ServeFile::new(path)
            .try_call(request)
            .await
        {
            Ok(response) => response.map(Body::new),
            Err(_) => StatusCode::INTERNAL_SERVER_ERROR.into_response(),
        },
        Ok(Err(error)) => error.response(),
        Err(_) => StatusCode::INTERNAL_SERVER_ERROR.into_response(),
    }
}

pub(super) async fn remember_voice(
    State(app): State<AppState>,
    Path(pid): Path<String>,
    body: Bytes,
) -> Response {
    blocking(move || {
        reply((|| {
            let body = json_object(&body)?;
            people::require_remember_consent(
                body.get("consent").and_then(Value::as_bool) == Some(true),
            )
            .map_err(from_people)?;
            let jid = body
                .get("meeting_id")
                .and_then(Value::as_str)
                .ok_or_else(not_found)?
                .to_owned();
            let track = body
                .get("track_id")
                .and_then(Value::as_str)
                .ok_or_else(not_found)?
                .to_owned();
            run_voice(&app, &jid, &track, Some(&pid), &body)
        })())
    })
    .await
}

pub(super) async fn voice_suggestion(
    State(app): State<AppState>,
    Path((jid, track)): Path<(String, String)>,
    body: Bytes,
) -> Response {
    blocking(move || {
        reply(json_object(&body).and_then(|body| run_voice(&app, &jid, &track, None, &body)))
    })
    .await
}

pub(super) fn assignment<'a>(job: &'a Job, track: &str) -> Option<&'a Value> {
    job.extra
        .get("speaker_assignments")
        .and_then(|a| a.get(track))
        .filter(|a| !a.is_null())
}

fn confirmed(job: &Job, track: &str, pid: &str) -> bool {
    assignment(job, track).is_some_and(|a| {
        a["person_id"].as_str() == Some(pid)
            && a["confirmed"].as_bool() != Some(false)
            && a["source"] != "automatic_voice"
    })
}

pub(super) fn selected_audio_unchanged(
    current: &Job,
    original: &Job,
    clips: &[VoiceClip],
) -> Result<bool, HttpError> {
    let current = meeting(current)?;
    let original = meeting(original)?;
    Ok(voice::clips_current(&current,clips) && clips.iter().all(|clip| {
        let old = original.segments.iter().find(|s|s.id==clip.segment_id);
        let new = current.segments.iter().find(|s|s.id==clip.segment_id);
        matches!((old,new),(Some(old),Some(new)) if old.audio_revision==new.audio_revision && old.canonical_utterance_id==new.canonical_utterance_id)
    }))
}

fn run_voice(
    app: &AppState,
    jid: &str,
    track: &str,
    pid: Option<&str>,
    body: &Map<String, Value>,
) -> Result<Value, HttpError> {
    let _lease = app
        .admission
        .lease(Activity::Refinement)
        .map_err(|e| conflict(e.to_string()))?;
    let models = app.models();
    let policy = models
        .voice_policy()
        .ok_or_else(|| conflict("Voice recognition is not available in this build."))?;
    let id = JobId::parse(jid).map_err(|_| not_found())?;
    let check_id = pid.is_none().then(super::new_id);
    let (original, clips, generation, catalog, operation) = {
        let mut store = app.store();
        let mut job = store.get(&id, false)?.ok_or_else(not_found)?;
        editing_job(
            &serde_json::to_value(&job).map_err(invalid_job)?,
            body,
            Some(track),
        )?;
        let catalog = Catalog::load(&store, Some(policy.clone()))?;
        let generation = if let Some(pid) = pid {
            catalog.people.get(pid).map_err(from_people)?;
            if !confirmed(&job, track, pid) {
                return Err(conflict(
                    "Confirm this speaker as the person before remembering their voice.",
                ));
            }
            store.voice_generation(pid)?.ok_or_else(not_found)?
        } else {
            if assignment(&job, track).is_some() {
                return Err(conflict(
                    "This speaker already has a name. Choose a name correction to change it.",
                ));
            }
            0
        };
        let ids: Vec<String> =
            serde_json::from_value(body.get("segment_ids").cloned().unwrap_or(Value::Null))
                .map_err(|_| {
                    HttpError::new(StatusCode::BAD_REQUEST, "Choose separate voice passages.")
                })?;
        let clips = voice::clean_clips(&meeting(&job)?, track, &ids, policy.minimum_clips)
            .map_err(voice_error)?;
        if let Some(check_id) = &check_id {
            let checks = job
                .extra
                .entry("voice_checks")
                .or_insert_with(|| json!({}))
                .as_object_mut()
                .ok_or_else(|| conflict("Stored voice checks are invalid."))?;
            checks.insert(
                track.into(),
                json!({"id":check_id,"status":"checking","manual":true}),
            );
            job.revision = job
                .revision
                .checked_add(1)
                .ok_or_else(|| conflict("Revision exhausted"))?;
        }
        let operation = app
            .begin_refinement_operation(&mut store, &mut job)
            .map_err(|e| conflict(e.message))?;
        (job, clips, generation, catalog, operation)
    };
    let result = (|| {
        let path = recording_path(app.data_dir(), &original)
            .map_err(|_| conflict("Meeting audio is unavailable."))?;
        let vectors = embeddings(
            models.as_ref(),
            &path,
            &policy,
            &clips,
            &operation.cancellation,
        )?;
        let mut store = app.store();
        if app.admission.closed() || operation.cancellation.is_cancelled() {
            return Err(conflict("Voice operation cancelled."));
        }
        if let Some(pid) = pid {
            let draft =
                voice::make_profile(&policy.model, &vectors, &clips).map_err(voice_error)?;
            let profile = VoiceProfile {
                person_id: pid.into(),
                version: super::new_id(),
                enrolled_at: unix_now(),
                model: draft.model,
                centroid: draft.centroid,
                clips: draft.clips,
                consent: draft.consent,
            };
            let published = store.publish_voice(&id, generation, &profile, |current| {
                Ok::<_, HttpError>(
                    operation.current(current)
                        && confirmed(current, track, pid)
                        && assignment(current, track) == assignment(&original, track)
                        && selected_audio_unchanged(current, &original, &clips)?,
                )
            })?;
            if !published {
                return Err(conflict(
                    "The voice or its source passages changed. Refresh and try again.",
                ));
            }
            Ok(json!({"remembered":true}))
        } else {
            let profiles: Vec<VoiceProfile> = catalog
                .profiles
                .values()
                .filter_map(|value| serde_json::from_value(value.clone()).ok())
                .collect();
            let matched = voice::propose_match(&policy.model, &policy, &vectors, &profiles)
                .map_err(voice_error)?;
            let current_catalog = Catalog::load(&store, Some(policy.clone()))?;
            if current_catalog.versions != catalog.versions {
                return Err(conflict("Saved voices changed. Try again."));
            }
            let (job, ()) = store
                .update_job(&id, |current| {
                    if !operation.current(current)
                        || !pending_check(
                            current,
                            track,
                            check_id.as_deref().expect("manual check"),
                        )
                        || assignment(current, track).is_some()
                        || !selected_audio_unchanged(current, &original, &clips)?
                    {
                        return Err(conflict(
                            "The source passages changed. Refresh and try again.",
                        ));
                    }
                    let mut suggestions = current
                        .extra
                        .get("voice_suggestions")
                        .and_then(Value::as_array)
                        .cloned()
                        .unwrap_or_default();
                    suggestions.retain(|s| s["track_id"] != track);
                    if let Some(found) = &matched {
                        let person = current_catalog
                            .people
                            .get(&found.person_id)
                            .map_err(from_people)?;
                        suggestions.push(json!({
                        "id": super::new_id(), "kind": "voice_match", "track_id": track,
                        "person_id": found.person_id, "name": person.name,
                        "match": {
                            "person_id": found.person_id, "profile_version": found.profile_version,
                            "score": found.score, "margin": found.margin, "model": found.model,
                            "calibration": policy_json(&policy)
                        },
                        "evidence": {"clips": clips}
                    }));
                    }
                    current
                        .extra
                        .insert("voice_suggestions".into(), json!(suggestions));
                    current
                        .extra
                        .get_mut("voice_checks")
                        .expect("current check")[track]["status"] = json!(if matched.is_some() {
                        "matched"
                    } else {
                        "unknown"
                    });
                    Ok::<_, HttpError>(())
                })?
                .ok_or_else(not_found)?;
            Ok(json!({"job":stored_job_json(&job),"matched":matched.is_some()}))
        }
    })();
    if result.is_err()
        && let Some(check_id) = &check_id
    {
        let _ = app.store().update_job(&id, |current| {
            if !pending_check(current, track, check_id) {
                return Err(conflict("Check was superseded"));
            }
            current
                .extra
                .get_mut("voice_checks")
                .expect("current check")[track]["status"] =
                json!(if operation.cancellation.is_cancelled() {
                    "cancelled"
                } else {
                    "unavailable"
                });
            Ok::<_, HttpError>(())
        });
    }
    result
}

pub(super) fn policy_json(policy: &Calibration) -> Value {
    json!({"dataset_id":policy.dataset_id,"threshold":policy.threshold,"margin":policy.margin,"minimum_clips":policy.minimum_clips})
}

pub(super) fn pending_check(job: &Job, track: &str, id: &str) -> bool {
    job.extra
        .get("voice_checks")
        .and_then(|checks| checks.get(track))
        .is_some_and(|check| check["id"] == id && check["status"] == "checking")
}

pub(super) fn embeddings(
    models: &dyn crate::routes::ApplicationModels,
    path: &std::path::Path,
    policy: &Calibration,
    clips: &[VoiceClip],
    cancellation: &tokio_util::sync::CancellationToken,
) -> Result<Vec<Vec<f64>>, HttpError> {
    let mut vectors = Vec::with_capacity(clips.len());
    for clip in clips {
        if cancellation.is_cancelled() {
            return Err(conflict("Voice operation cancelled."));
        }
        let pcm = audio::read_pcm16_range(path, clip.start, clip.end)
            .map_err(|e| conflict(e.to_string()))?;
        let result = models.embed_voice(&pcm, cancellation).map_err(conflict)?;
        if result.model != policy.model {
            return Err(conflict(
                "The voice model changed. Refresh passages and try again.",
            ));
        }
        if !result.clip.clean {
            return Err(voice_error(voice::VoiceError::Unusable));
        }
        vectors.push(
            voice::normalized(&result.clip.vector, policy.model.dimension).map_err(voice_error)?,
        );
    }
    Ok(vectors)
}
