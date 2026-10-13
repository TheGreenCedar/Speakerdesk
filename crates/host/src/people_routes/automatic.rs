//! Optional identity checks share the transcript owner's serial model boundary.
use std::{collections::VecDeque, path::Path, sync::Mutex};

use super::{Catalog, HttpError, unix_now, voice_http};
use crate::routes::{ApplicationModels, OperationGuard};
use serde_json::json;
use speakerdesk_core::{
    document::Job,
    store::{JobStore, SqliteJobStore},
    voice::{self, RecognitionContext, VoiceProfile},
};

pub(crate) fn check_saved_voices(
    database: &Mutex<SqliteJobStore>,
    root: &Path,
    models: &dyn ApplicationModels,
    operation: &OperationGuard,
    published: &Job,
) {
    let mut checks = Checks::new(false);
    loop {
        match checks.step(database, root, models, operation, published) {
            Ok(true) => {}
            Ok(false) => break,
            Err(error) => {
                tracing::warn!(message = %error, "automatic voice checks did not complete");
                checks.interrupt(&mut database.lock().expect("store"));
                break;
            }
        }
    }
}

struct Pending {
    original: Job,
    scheduled: voice::ScheduledCheck,
    profiles: Vec<VoiceProfile>,
    vectors: Vec<Vec<f64>>,
    failed: bool,
}
/// Saved operations select their existing check metadata once from one snapshot.
/// These IDs contain no audio or tasks; live discovery never uses this batch.
struct SavedPlan {
    original: Job,
    profiles: Vec<VoiceProfile>,
    checks: VecDeque<voice::ScheduledCheck>,
}
/// One selected track; each step embeds at most one stable solo crop. No audio queue.
#[derive(Default)]
pub(crate) struct Checks {
    live: bool,
    changed: bool,
    pending: Option<Pending>,
    child_failure: Option<speakerdesk_core::error::ChildError>,
    saved: Option<SavedPlan>,
}
impl Checks {
    pub(crate) fn new(live: bool) -> Self {
        Self {
            live,
            changed: true,
            pending: None,
            child_failure: None,
            saved: None,
        }
    }
    pub(crate) fn pending(&self) -> bool {
        self.pending.is_some()
    }
    pub(crate) fn changed(&mut self) {
        self.changed = true;
    }
    pub(crate) fn interrupt(&mut self, store: &mut SqliteJobStore) {
        if let Some(p) = self.pending.take() {
            interrupt_check(store, &p.original, &p.scheduled);
        }
        if let Some(plan) = &mut self.saved {
            for check in plan.checks.drain(..) {
                interrupt_check(store, &plan.original, &check);
            }
        }
    }
    pub(crate) fn step(
        &mut self,
        database: &Mutex<SqliteJobStore>,
        root: &Path,
        models: &dyn ApplicationModels,
        operation: &OperationGuard,
        published: &Job,
    ) -> Result<bool, speakerdesk_core::live::LiveError> {
        self.advance(database, root, models, operation, published)
            .map_err(|e| {
                self.child_failure.take().map_or(
                    speakerdesk_core::live::LiveError::Detail(e.message),
                    speakerdesk_core::live::LiveError::Child,
                )
            })
    }
    fn advance(
        &mut self,
        database: &Mutex<SqliteJobStore>,
        root: &Path,
        models: &dyn ApplicationModels,
        operation: &OperationGuard,
        published: &Job,
    ) -> Result<bool, HttpError> {
        let Some(policy) = models.voice_policy() else {
            self.interrupt(&mut database.lock().expect("store"));
            return Ok(false);
        };
        if self.pending.is_none()
            && let Some(plan) = &mut self.saved
        {
            let Some(scheduled) = plan.checks.pop_front() else {
                return Ok(false);
            };
            self.pending = Some(Pending {
                original: plan.original.clone(),
                scheduled,
                profiles: plan.profiles.clone(),
                vectors: Vec::new(),
                failed: false,
            });
        }
        if self.pending.is_none() {
            if self.live && !self.changed {
                return Ok(false);
            }
            self.changed = false;
            let mut store = database.lock().expect("store");
            let Some(job) = store.get(&published.id, false)? else {
                return Ok(false);
            };
            if !operation.current(&job) {
                return Ok(false);
            }
            let catalog = Catalog::load(&store, Some(policy.clone()))?;
            let profiles: Vec<VoiceProfile> = catalog
                .profiles
                .values()
                .filter_map(|value| serde_json::from_value(value.clone()).ok())
                .collect();
            let mut meeting = voice_http::meeting(&job)?;
            // Completed decisions and manual checks stay final. A failed/cancelled
            // live check may retry only when its selected evidence or child changes.
            meeting.speakers.retain(|track, _| {
                voice_http::assignment(&job, track).is_none()
                    && job
                        .extra
                        .get("voice_checks")
                        .and_then(|c| c.get(track))
                        .is_none_or(|check| {
                            self.live
                                && check["manual"] == false
                                && matches!(
                                    check["status"].as_str(),
                                    Some("cancelled" | "unavailable")
                                )
                        })
            });
            let context = RecognitionContext {
                closed: false,
                preference_enabled: store.preference("voice_recognition")?.as_deref()
                    != Some("false"),
                audio_available: true,
                execution_allowed: true,
                model: Some(&policy.model),
                calibration: Some(&policy),
                profiles: &profiles,
                people: &catalog.people,
            };
            let planned = if self.live {
                voice::plan_live_checks(&meeting, &context, None)
            } else {
                voice::plan_checks(&meeting, &context, None)
            };
            let planned = planned
                .into_iter()
                .filter(|check| {
                    job.extra
                        .get("voice_checks")
                        .and_then(|v| v.get(&check.track_id))
                        .is_none_or(|old| old["attempt_key"] != attempt_key(&job, check))
                })
                .take(if self.live { 1 } else { usize::MAX })
                .collect();
            let mut scheduled: VecDeque<_> =
                voice::commit_plan(&mut meeting, planned, super::new_id).into();
            if scheduled.is_empty() {
                return Ok(false);
            }
            let (original, ()) = store
                .update_job(&job.id, |current| {
                    let checks = current
                        .extra
                        .entry("voice_checks")
                        .or_insert_with(|| json!({}))
                        .as_object_mut()
                        .ok_or_else(|| voice_http::conflict("Stored voice checks are invalid"))?;
                    for check in &scheduled {
                        checks.insert(check.track_id.clone(), json!({"id":check.check_id,"status":"checking",
                            "model":check.model,"clips":check.clips,"manual":false,"attempt_key":attempt_key(&job, check)}));
                    }
                    Ok::<_, HttpError>(())
                })?
                .ok_or_else(super::not_found)?;
            let check = scheduled.pop_front().expect("selected check");
            if !self.live {
                self.saved = Some(SavedPlan {
                    original: original.clone(),
                    profiles: profiles.clone(),
                    checks: scheduled,
                });
            }
            self.pending = Some(Pending {
                original,
                scheduled: check,
                profiles,
                vectors: Vec::new(),
                failed: false,
            });
            self.changed = true; // Discover the next track after this one finishes.
        }
        let p = self.pending.as_mut().expect("selected track");
        let track = &p.scheduled.track_id;
        let fresh = {
            let store = database.lock().expect("store");
            let Some(current) = store.get(&p.original.id, false)? else {
                self.pending = None;
                return Ok(true);
            };
            operation.current(&current)
                && voice_http::pending_check(&current, track, &p.scheduled.check_id)
                && voice_http::assignment(&current, track).is_none()
                && store.preference("voice_recognition")?.as_deref() != Some("false")
                && store
                    .voice_profiles()?
                    .values()
                    .filter_map(|v| serde_json::from_value::<VoiceProfile>(v.clone()).ok())
                    .filter(|profile| profile.model == policy.model)
                    .map(|profile| (profile.person_id, profile.version))
                    .collect::<std::collections::BTreeSet<_>>()
                    == p.scheduled.profile_versions.iter().cloned().collect()
                && policy == p.scheduled.calibration
                && selected_audio_unchanged(self.live, &current, &p.original, &p.scheduled.clips)?
        };
        if fresh && !p.failed && p.vectors.len() < p.scheduled.clips.len() {
            // The one-clip helper retains model identity, cleanliness and dimension checks.
            let clip = &p.scheduled.clips[p.vectors.len()];
            let embedded = if self.live {
                let path = voice_http::clip_recording_path(root, &p.original, clip)?;
                let pcm = speakerdesk_core::audio::read_pcm16_range(path, clip.start, clip.end)
                    .map_err(|e| voice_http::conflict(e.to_string()))?;
                let embedding = match models.embed_live_voice(&pcm, &operation.cancellation) {
                    Ok(embedding) => Ok(embedding),
                    Err(error) => {
                        if matches!(
                            error,
                            speakerdesk_core::error::ChildError::ProcessLost { .. }
                                | speakerdesk_core::error::ChildError::Timeout
                                | speakerdesk_core::error::ChildError::Cancelled
                        ) {
                            self.child_failure = Some(error);
                            return Err(voice_http::conflict("Live voice child stopped."));
                        }
                        Err(voice_http::conflict(error.to_string()))
                    }
                };
                embedding.and_then(|embedding| {
                    if embedding.model != policy.model || !embedding.clip.clean {
                        return Err(voice_http::conflict(
                            "Voice embedding identity or solo audio changed.",
                        ));
                    }
                    voice::normalized(&embedding.clip.vector, policy.model.dimension)
                        .map(|v| vec![v])
                        .map_err(voice_http::voice_error)
                })
            } else {
                voice_http::embeddings(
                    models,
                    root,
                    &p.original,
                    &policy,
                    std::slice::from_ref(clip),
                    &operation.cancellation,
                )
            };
            match embedded {
                Ok(mut vectors) => p.vectors.push(vectors.pop().expect("one clip")),
                Err(_) => p.failed = true,
            }
            // Return to live audio even after the final embedding; publication rechecks
            // preferences, profile forget/version races and the current crop ownership.
            return Ok(true);
        }
        let p = self.pending.take().expect("selected track");
        let track = &p.scheduled.track_id;
        let matched = if fresh && !p.failed {
            voice::propose_match(&policy.model, &policy, &p.vectors, &p.profiles)
                .ok()
                .flatten()
        } else {
            None
        };
        let outcome = voice::CheckOutcome {
            failed: p.failed,
            matched,
        };
        let mut store = database.lock().expect("store");
        let catalog = Catalog::load(&store, Some(policy.clone()))?;
        let profiles: Vec<VoiceProfile> = catalog
            .profiles
            .values()
            .filter_map(|v| serde_json::from_value(v.clone()).ok())
            .collect();
        let enabled = store.preference("voice_recognition")?.as_deref() != Some("false");
        store.update_job(&p.original.id, |current| {
            if !voice_http::pending_check(current, track, &p.scheduled.check_id) { return Ok(()) }
            let mut meeting = voice_http::meeting(current)?;
            meeting.checks.insert(track.clone(), voice::VoiceCheck { id: p.scheduled.check_id.clone(),
                status: voice::CHECK_CHECKING.into(), checked_at: None, model: p.scheduled.model.clone(), clips: p.scheduled.clips.clone() });
            let still_fresh = fresh && operation.current(current) && voice_http::assignment(current, track).is_none()
                && selected_audio_unchanged(self.live, current, &p.original, &p.scheduled.clips)?;
            let context = RecognitionContext { closed: false, preference_enabled: enabled && still_fresh,
                audio_available: true, execution_allowed: true, model: Some(&policy.model), calibration: Some(&policy), profiles: &profiles, people: &catalog.people };
            let now = unix_now();
            if self.live {
                voice::apply_live_check(&mut meeting, &p.scheduled, &context, &outcome, now);
            } else {
                voice::apply_check(&mut meeting, &p.scheduled, &context, &outcome, now);
            }
            let check = &meeting.checks[track];
            let saved = &mut current.extra.get_mut("voice_checks").expect("pending check")[track];
            saved["status"] = json!(check.status); saved["checked_at"] = json!(now);
            if let Some(assignment) = meeting.assignments.get(track) {
                let matched = outcome.matched.as_ref().expect("matched assignment");
                let assignments = current.extra.entry("speaker_assignments").or_insert_with(|| json!({})).as_object_mut()
                    .ok_or_else(|| voice_http::conflict("Stored assignments are invalid"))?;
                assignments.insert(track.clone(), json!({"person_id":assignment.person_id,"meeting_id":assignment.meeting_id,"track_id":track,
                    "name":assignment.name,"source":assignment.source,"confirmed":false,"assigned_at":now,
                    "evidence":{"clips":p.scheduled.clips,"match":{"person_id":matched.person_id,"profile_version":matched.profile_version,
                        "score":matched.score,"margin":matched.margin,"model":matched.model,"calibration":voice_http::policy_json(&policy)}}}));
                current.document.as_mut().expect("local transcript").speakers.insert(track.clone(), assignment.name.clone());
            }
            Ok::<_, HttpError>(())
        })?;
        Ok(true)
    }
}

fn interrupt_check(store: &mut SqliteJobStore, original: &Job, check: &voice::ScheduledCheck) {
    let _ = store.update_job(&original.id, |job| {
        if voice_http::pending_check(job, &check.track_id, &check.check_id) {
            job.extra.get_mut("voice_checks").expect("pending check")[&check.track_id]["status"] =
                json!("cancelled");
        }
        Ok::<_, HttpError>(())
    });
}

fn selected_audio_unchanged(
    live: bool,
    current: &Job,
    original: &Job,
    clips: &[voice::VoiceClip],
) -> Result<bool, HttpError> {
    if !live {
        return voice_http::selected_audio_unchanged(current, original, clips);
    }
    // Capture may append audio to an open row. The selected solo prefix must still
    // exist with the same source binding and observed ownership; job duration and
    // the row's growing audio_revision are not identities of that immutable prefix.
    let current_meeting = voice_http::meeting(current)?;
    let original_meeting = voice_http::meeting(original)?;
    Ok(current.id == original.id && current.source_file == original.source_file
        && ["legacy_media_root", "audio_revision", "capture_source_catalog", "sources"].iter().all(|key| current.extra.get(*key) == original.extra.get(*key))
        && voice::live_clips_current(&current_meeting, clips) && voice::live_clips_current(&original_meeting, clips)
        && clips.iter().all(|clip| {
            let old = original.document.as_ref().and_then(|d| d.segments.iter().find(|s| s.id == clip.segment_id));
            let new = current.document.as_ref().and_then(|d| d.segments.iter().find(|s| s.id == clip.segment_id));
            matches!((old,new),(Some(old),Some(new)) if old.capture_source == new.capture_source
                && old.canonical_utterance_id == new.canonical_utterance_id && old.separation == new.separation)
        }))
}

fn attempt_key(job: &Job, check: &voice::ScheduledCheck) -> String {
    let source = json!({"source_file":job.source_file,"legacy_media_root":job.extra.get("legacy_media_root"),
        "audio_revision":job.extra.get("audio_revision"),"sources":job.extra.get("sources"),
        "catalog":job.extra.get("capture_source_catalog"),"child":job.extra.get("live_stream_generation"),
        "clips":check.clips,"profiles":check.profile_versions,"policy":voice_http::policy_json(&check.calibration)});
    speakerdesk_core::memo::sha256_hex(
        &serde_json::to_vec(&source).expect("check source serializes"),
    )
}
