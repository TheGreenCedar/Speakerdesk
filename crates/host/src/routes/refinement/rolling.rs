//! One accuracy continuation on the existing live worker. The capture actor owns
//! cancellation; SQLite owns checkpoints and the canonical document.
use super::*;
use crate::inference::ChildModels;
use speakerdesk_core::error::ChildError;

#[derive(Clone)]
pub(crate) struct Rolling(Arc<Mutex<Control>>);
struct Control {
    enabled: bool,
    generation: u64,
    active: Option<CancellationToken>,
    scan_from: Option<usize>,
    final_retry: bool,
}
impl Rolling {
    pub fn new() -> Self {
        Self(Arc::new(Mutex::new(Control {
            enabled: true,
            generation: 0,
            active: None,
            scan_from: None,
            final_retry: false,
        })))
    }
    /// Called with the store lock after an accepted sealed publication. Coalesce
    /// discovery into a cursor, never a queue of documents or audio.
    pub fn changed(&self, index: usize) {
        let mut control = self.0.lock().expect("rolling control");
        control.scan_from = Some(control.scan_from.map_or(index, |old| old.min(index)));
    }
    pub fn action(
        &self,
        app: &AppState,
        operation: &OperationGuard,
        resume: bool,
    ) -> Result<Job, ApiError> {
        let mut store = app.store();
        let mut control = self.0.lock().expect("rolling control");
        if resume && control.active.is_some() {
            return Err(ApiError::new(
                409,
                "Wait for the previous accuracy request to finish cancelling.",
            ));
        }
        let generation = control
            .generation
            .checked_add(1)
            .ok_or_else(|| ApiError::new(409, "Accuracy generation exhausted."))?;
        let result = store.update_job(&operation.id, |job| {
            if !operation.current(job) {
                return Err(ApiError::new(409, RETAINED));
            }
            job.extra
                .insert("refinement_generation".into(), json!(generation));
            job.extra.insert(
                "refinement_status".into(),
                json!(if resume { "waiting" } else { "paused" }),
            );
            if resume {
                job.extra.get_mut("rolling_refinement").expect("live plan")["failed"] = json!({});
                job.extra.remove("refinement_error");
            }
            Ok::<_, ApiError>(())
        });
        let job = match result {
            Ok(Some((job, ()))) => job,
            other => {
                if let Some(token) = &control.active {
                    token.cancel();
                }
                return Err(other.err().unwrap_or_else(|| ApiError::new(404, NOT_FOUND)));
            }
        };
        control.generation = generation;
        control.enabled = resume;
        if resume {
            control.scan_from = Some(0);
        }
        if let Some(token) = &control.active {
            token.cancel();
        }
        Ok(job)
    }
    /// Stop/recovery invalidates before cancellation; an already-invalidated
    /// recording needs only terminal cleanup, not another successful mutation.
    pub fn interrupt(&self, app: &AppState, operation: &OperationGuard) -> Result<(), ApiError> {
        let mut store = app.store();
        let mut control = self.0.lock().expect("rolling control");
        match store.get(&operation.id, false) {
            Ok(Some(job)) if operation.current(&job) => {}
            Ok(_) => {
                if let Some(token) = &control.active {
                    token.cancel();
                }
                return Ok(());
            }
            Err(error) => {
                if let Some(token) = &control.active {
                    token.cancel();
                }
                return Err(error.into());
            }
        }
        let generation = control
            .generation
            .checked_add(1)
            .ok_or_else(|| ApiError::new(409, "Accuracy generation exhausted."))?;
        let result = store.update_job(&operation.id, |job| {
            if !operation.current(job) {
                return Ok(false);
            }
            job.extra
                .insert("refinement_generation".into(), json!(generation));
            if control.enabled {
                job.extra
                    .insert("refinement_status".into(), json!("waiting"));
            }
            Ok::<_, ApiError>(true)
        });
        if let Some(token) = &control.active {
            token.cancel();
        }
        if result?.is_some_and(|(_, current)| current) {
            control.generation = generation;
            if control.enabled {
                control.scan_from = Some(0);
            }
        }
        Ok(())
    }
    fn release(&self) {
        // Only the serial runner creates active work. Resume cannot replace this slot
        // until terminal cleanup clears it, including after a cancelled generation.
        self.0.lock().expect("rolling control").active = None;
    }
}

struct Pending {
    original: Job,
    prior: Segment,
    reader: audio::RecordingReader,
    composition: live::SavedRefinement,
    models: ChildModels,
    generation: u64,
    cancellation: CancellationToken,
}
pub(crate) struct Runner {
    rolling: Rolling,
    pending: Option<Pending>,
}
impl Drop for Runner {
    fn drop(&mut self) {
        if self.pending.take().is_some() {
            self.rolling.release();
        }
    }
}
impl Runner {
    pub fn new(rolling: Rolling) -> Self {
        Self {
            rolling,
            pending: None,
        }
    }
    pub fn step(
        &mut self,
        app: &AppState,
        operation: &OperationGuard,
    ) -> Result<bool, live::LiveError> {
        if self
            .pending
            .as_ref()
            .is_some_and(|p| p.cancellation.is_cancelled())
        {
            drop(self.pending.take());
            self.rolling.release();
        }
        if self.pending.is_none() {
            let selection = {
                let mut store = app.store();
                let mut control = self.rolling.0.lock().expect("rolling control");
                if !control.enabled || operation.cancellation.is_cancelled() {
                    return Ok(false);
                }
                let Some(mut job) = store.get(&operation.id, false).map_err(|e| e.to_string())?
                else {
                    return Ok(false);
                };
                if !operation.current(&job) {
                    return Ok(false);
                }
                if job.status == "finishing" && !control.final_retry {
                    // Retry live failures once after Stop on the same worker and lease.
                    // Completed/protected anchors keep their existing selection guards.
                    let refreshed = store
                        .update_job(&operation.id, |job| {
                            job.extra.get_mut("rolling_refinement").expect("live plan")["failed"] =
                                json!({});
                            job.extra.remove("refinement_error");
                            Ok::<_, ApiError>(())
                        })
                        .map_err(|e| e.message)?;
                    let Some((refreshed, ())) = refreshed else {
                        return Ok(false);
                    };
                    job = refreshed;
                    control.final_retry = true;
                    control.scan_from = Some(0);
                }
                let Some(start) = control.scan_from.take() else {
                    return Ok(false);
                };
                let selected = job
                    .document
                    .as_ref()
                    .and_then(|doc| {
                        doc.segments
                            .iter()
                            .enumerate()
                            .skip(start)
                            .find(|(_, row)| {
                                row.separation.is_none()
                                    && row.canonical_utterance_id.is_some()
                                    && row.rest["canonical_state"] == "sealed"
                                    && row.protected_fields.as_ref().is_none_or(Vec::is_empty)
                                    && !checkpoint(&job, row)
                                    && job.extra["rolling_refinement"]["failed"][&row.id].as_str()
                                        != Some(live::segment_version(row).as_str())
                            })
                    })
                    .map(|(index, row)| (index, row.clone()));
                match selected {
                    Some((index, prior)) => {
                        control.scan_from = Some(index + 1);
                        let cancellation = operation.cancellation.child_token();
                        let generation = control.generation;
                        store
                            .update_job(&operation.id, |job| {
                                job.extra
                                    .insert("refinement_status".into(), json!("refining"));
                                Ok::<_, ApiError>(())
                            })
                            .map_err(|e| e.message)?;
                        control.active = Some(cancellation.clone());
                        Some((job, prior, generation, cancellation))
                    }
                    None => {
                        store
                            .update_job(&operation.id, |job| {
                                let failed = job.extra["rolling_refinement"]["failed"]
                                    .as_object()
                                    .is_some_and(|failed| !failed.is_empty());
                                job.extra.insert(
                                    "refinement_status".into(),
                                    json!(if failed {
                                        "unresolved"
                                    } else if job.status == "finishing" || job.status == "ready" {
                                        "complete"
                                    } else {
                                        "waiting"
                                    }),
                                );
                                Ok::<_, ApiError>(())
                            })
                            .map_err(|e| e.message)?;
                        None
                    }
                }
            };
            let Some((mut original, prior, generation, cancellation)) = selection else {
                return Ok(false);
            };
            // Keep the immutable audio/epoch snapshot, not another transcript history.
            original.document = None;
            original.extra.remove("refinement_history");
            original.extra.remove("rolling_refinement");
            let application = app.models();
            let preparation = (|| {
                let path = crate::native::passage_recording_path(app.data_dir(), &original, &prior)
                    .map_err(|e| e.to_string())?;
                let reader = audio::RecordingReader::open(path).map_err(|e| e.to_string())?;
                let composition = live::SavedRefinement::new(&prior)?;
                let models = application
                    .live_models(&cancellation)
                    .ok_or("Live accuracy models are unavailable")?;
                Ok::<_, live::LiveError>((reader, composition, models))
            })();
            match preparation {
                Ok((reader, composition, models)) => {
                    self.pending = Some(Pending {
                        original,
                        prior,
                        reader,
                        composition,
                        models,
                        generation,
                        cancellation,
                    })
                }
                Err(error) => {
                    self.finish(
                        app,
                        operation,
                        &original,
                        &prior,
                        generation,
                        &cancellation,
                        Err(error),
                    )?;
                    return Ok(true);
                }
            }
        }
        let pending = self.pending.as_mut().expect("selected work");
        let vocabulary = app.models().live_alignment();
        let result =
            pending
                .composition
                .step(&pending.models, &pending.reader, vocabulary.as_deref());
        if matches!(result, Ok(None)) {
            return Ok(true);
        }
        let pending = self.pending.take().expect("completed work");
        self.finish(
            app,
            operation,
            &pending.original,
            &pending.prior,
            pending.generation,
            &pending.cancellation,
            result.map(|candidate| candidate.expect("complete")),
        )?;
        Ok(true)
    }
    #[allow(clippy::too_many_arguments)]
    fn finish(
        &self,
        app: &AppState,
        operation: &OperationGuard,
        original: &Job,
        prior: &Segment,
        generation: u64,
        cancellation: &CancellationToken,
        result: Result<Segment, live::LiveError>,
    ) -> Result<(), live::LiveError> {
        let published = app.store().update_job(&operation.id, |current| {
            if !operation.current(current)
                || cancellation.is_cancelled()
                || current.extra["refinement_generation"].as_u64() != Some(generation)
            {
                return Ok(());
            }
            let accepted = match &result {
                Ok(candidate) => publish_candidate(current, original, prior, candidate)
                    .map_err(|e| ApiError::new(500, e.to_string()))?,
                Err(_) => false,
            };
            if !accepted
                && !matches!(
                    result,
                    Err(live::LiveError::Child(
                        ChildError::ProcessLost { .. }
                            | ChildError::Timeout
                            | ChildError::Cancelled
                    ))
                )
            {
                current.extra.get_mut("rolling_refinement").expect("plan")["failed"][&prior.id] =
                    json!(live::segment_version(prior));
                current.extra.insert(
                    "refinement_error".into(),
                    json!(result.as_ref().err().map_or(
                        "Passage changed; previous words retained.".to_string(),
                        ToString::to_string
                    )),
                );
            }
            current.extra.insert(
                "refinement_status".into(),
                json!(if accepted { "waiting" } else { "unresolved" }),
            );
            Ok::<_, ApiError>(())
        });
        self.rolling.release();
        published.map_err(|e| e.message)?;
        if let Err(
            error @ live::LiveError::Child(ChildError::ProcessLost { .. } | ChildError::Timeout),
        ) = result
        {
            return Err(error);
        }
        Ok(())
    }
}
