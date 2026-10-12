//! Optional checks run after transcript publication under the existing operation lease.
use std::{path::Path, sync::Mutex};

use serde_json::json;
use speakerdesk_core::{
    document::Job,
    store::{JobStore, SqliteJobStore},
    voice::{self, RecognitionContext, VoiceProfile},
};

use super::{Catalog, HttpError, unix_now, voice_http};
use crate::routes::{ApplicationModels, OperationGuard};

pub(crate) fn check_saved_voices(
    database: &Mutex<SqliteJobStore>,
    root: &Path,
    models: &dyn ApplicationModels,
    operation: &OperationGuard,
    published: &Job,
) {
    if let Err(error) = run(database, root, models, operation, published) {
        tracing::warn!(message = %error.message, "automatic voice checks did not complete");
    }
}

fn run(
    database: &Mutex<SqliteJobStore>,
    root: &Path,
    models: &dyn ApplicationModels,
    operation: &OperationGuard,
    published: &Job,
) -> Result<(), HttpError> {
    let Some(policy) = models.voice_policy() else {
        return Ok(());
    };
    let (original, scheduled, profiles) = {
        let mut store = database.lock().expect("store");
        let Some(job) = store.get(&published.id, false)? else {
            return Ok(());
        };
        if !operation.current(&job) {
            return Ok(());
        }
        let catalog = Catalog::load(&store, Some(policy.clone()))?;
        let profiles: Vec<VoiceProfile> = catalog
            .profiles
            .values()
            .filter_map(|value| serde_json::from_value(value.clone()).ok())
            .collect();
        let mut meeting = voice_http::meeting(&job)?;
        // Presence blocks rescheduling, including legacy/manual checks without model fields.
        meeting.speakers.retain(|track, _| {
            voice_http::assignment(&job, track).is_none()
                && job
                    .extra
                    .get("voice_checks")
                    .and_then(|checks| checks.get(track))
                    .is_none()
        });
        let context = RecognitionContext {
            closed: false,
            preference_enabled: store.preference("voice_recognition")?.as_deref() != Some("false"),
            audio_available: true,
            execution_allowed: true,
            model: Some(&policy.model),
            calibration: Some(&policy),
            profiles: &profiles,
            people: &catalog.people,
        };
        let planned = voice::plan_checks(&meeting, &context, None);
        if planned.is_empty() {
            return Ok(());
        }
        let scheduled = voice::commit_plan(&mut meeting, planned, super::new_id);
        let (job, ()) = store
            .update_job(&job.id, |current| {
                let checks = current
                    .extra
                    .entry("voice_checks")
                    .or_insert_with(|| json!({}))
                    .as_object_mut()
                    .ok_or_else(|| voice_http::conflict("Stored voice checks are invalid"))?;
                for check in &scheduled {
                    checks.insert(
                        check.track_id.clone(),
                        json!({"id":check.check_id,"status":"checking",
                    "model":check.model,"clips":check.clips,"manual":false}),
                    );
                }
                Ok::<_, HttpError>(())
            })?
            .ok_or_else(super::not_found)?;
        (job, scheduled, profiles)
    };
    for scheduled in scheduled {
        let track = &scheduled.track_id;
        let result = (|| {
            let eligible = {
                let store = database.lock().expect("store");
                let Some(current) = store.get(&original.id, false)? else {
                    return Ok(());
                };
                operation.current(&current)
                    && voice_http::pending_check(&current, track, &scheduled.check_id)
                    && voice_http::assignment(&current, track).is_none()
                    && store
                        .voice_profiles()?
                        .values()
                        .filter_map(|value| {
                            serde_json::from_value::<VoiceProfile>(value.clone()).ok()
                        })
                        .filter(|profile| profile.model == policy.model)
                        .map(|profile| (profile.person_id, profile.version))
                        .collect::<std::collections::BTreeSet<_>>()
                        == scheduled.profile_versions.iter().cloned().collect()
                    && store.preference("voice_recognition")?.as_deref() != Some("false")
                    && voice_http::selected_audio_unchanged(&current, &original, &scheduled.clips)?
            };
            let outcome = if eligible {
                let matched = voice_http::embeddings(
                    models,
                    root,
                    &original,
                    &policy,
                    &scheduled.clips,
                    &operation.cancellation,
                )
                .and_then(|vectors| {
                    voice::propose_match(&policy.model, &policy, &vectors, &profiles)
                        .map_err(voice_http::voice_error)
                });
                voice::CheckOutcome {
                    failed: matched.is_err(),
                    matched: matched.ok().flatten(),
                }
            } else {
                voice::CheckOutcome {
                    failed: false,
                    matched: None,
                }
            };
            let mut store = database.lock().expect("store");
            let catalog = Catalog::load(&store, Some(policy.clone()))?;
            let current_profiles: Vec<VoiceProfile> = catalog
                .profiles
                .values()
                .filter_map(|value| serde_json::from_value(value.clone()).ok())
                .collect();
            let enabled = store.preference("voice_recognition")?.as_deref() != Some("false");
            store.update_job(&original.id, |current| {
                if !voice_http::pending_check(current, track, &scheduled.check_id) {
                    return Err(voice_http::conflict("Voice check was superseded"));
                }
                let mut meeting = voice_http::meeting(current)?;
                meeting.checks.insert(track.clone(), voice::VoiceCheck {
                    id: scheduled.check_id.clone(), status: voice::CHECK_CHECKING.into(), checked_at: None,
                    model: scheduled.model.clone(), clips: scheduled.clips.clone(),
                });
                let fresh = eligible && operation.current(current) && voice_http::assignment(current, track).is_none()
                    && voice_http::selected_audio_unchanged(current, &original, &scheduled.clips)?;
                let context = RecognitionContext {
                    closed: false, preference_enabled: enabled && fresh, audio_available: true,
                    execution_allowed: true, model: Some(&policy.model), calibration: Some(&policy),
                    profiles: &current_profiles, people: &catalog.people,
                };
                let now = unix_now();
                voice::apply_check(&mut meeting, &scheduled, &context, &outcome, now);
                let check = &meeting.checks[track];
                let saved = &mut current.extra.get_mut("voice_checks").expect("pending check")[track];
                saved["status"] = json!(check.status); saved["checked_at"] = json!(now);
                if let Some(assignment) = meeting.assignments.get(track) {
                    let matched = outcome.matched.as_ref().expect("matched assignment");
                    let assignments = current.extra.entry("speaker_assignments").or_insert_with(|| json!({}))
                        .as_object_mut().ok_or_else(|| voice_http::conflict("Stored assignments are invalid"))?;
                    assignments.insert(track.clone(), json!({
                        "person_id":assignment.person_id,"meeting_id":assignment.meeting_id,"track_id":track,
                        "name":assignment.name,"source":assignment.source,"confirmed":false,"assigned_at":now,
                        "evidence":{"clips":scheduled.clips,"match":{
                            "person_id":matched.person_id,"profile_version":matched.profile_version,"score":matched.score,
                            "margin":matched.margin,"model":matched.model,"calibration":voice_http::policy_json(&policy)
                        }}
                    }));
                    current.document.as_mut().expect("local transcript").speakers.insert(track.clone(), assignment.name.clone());
                }
                Ok::<_, HttpError>(())
            })?;
            Ok::<_, HttpError>(())
        })();
        if let Err(error) = result {
            tracing::warn!(message = %error.message, %track, "automatic voice check was not published");
            let _ = database
                .lock()
                .expect("store")
                .update_job(&original.id, |current| {
                    if !voice_http::pending_check(current, track, &scheduled.check_id) {
                        return Err(voice_http::conflict("Voice check was superseded"));
                    }
                    let status = if operation.current(current) {
                        "unavailable"
                    } else {
                        "cancelled"
                    };
                    current
                        .extra
                        .get_mut("voice_checks")
                        .expect("pending check")[track]["status"] = json!(status);
                    Ok::<_, HttpError>(())
                });
        }
    }
    Ok(())
}
