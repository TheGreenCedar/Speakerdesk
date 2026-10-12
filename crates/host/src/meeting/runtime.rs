use super::worker::{RecoveryPlan, Worker, worker};
use super::{Action, Control, Progress, conflict, number, update};
use crate::{
    admission::ActivityLease,
    capture::{CaptureConfig, CapturePhase, CaptureSnapshot, CaptureSupervisor, Command},
    routes::{AppState, OperationGuard, stored_job_json},
};
use serde_json::{Value, json};
use speakerdesk_core::{
    error::{ApiError, ChildError, StoreError},
    ids::JobId,
    live::{AudioNotice, LiveCommand, LiveError},
    store::JobStore,
};
use std::{path::PathBuf, sync::Arc};
use tokio::sync::{mpsc, watch};

struct Recording {
    app: AppState,
    id: JobId,
    operation: Arc<OperationGuard>,
    language: String,
    epoch: i64,
    received: i64,
    worker: Option<Worker>,
    progress: watch::Sender<Progress>,
    failure: Option<String>,
    phase: CapturePhase,
    pause_pending: bool,
    stopping: bool,
    recovery_attempted: bool,
    cleanup_error: Option<String>,
}
impl Recording {
    async fn mutate(
        &self,
        change: impl FnOnce(&mut speakerdesk_core::document::Job) -> Result<(), ApiError>
        + Send
        + 'static,
    ) -> Result<Value, ApiError> {
        let app = self.app.clone();
        let id = self.id.clone();
        let op = self.operation.clone();
        tokio::task::spawn_blocking(move || {
            update(&app, &id, &op, change).map(|j| stored_job_json(&j))
        })
        .await
        .map_err(|e| ApiError::new(500, e.to_string()))?
    }
    fn feed(&mut self, end: u32) -> Result<(), LiveError> {
        if let Some(worker) = &self.worker {
            if worker.result.borrow().is_some() {
                return Err(worker
                    .failed()
                    .unwrap_or_else(|| "Live worker ended before capture".into()));
            }
            while self.received < i64::from(end) {
                let next = (self.received + 16000).min(i64::from(end));
                worker.push(LiveCommand::Audio(AudioNotice {
                    start_sample: self.received,
                    end_sample: next,
                    language_epoch: self.epoch,
                    language: self.language.clone(),
                }))?;
                self.received = next;
            }
        }
        Ok(())
    }
    async fn fail(&mut self, message: String) {
        if self.failure.is_none() {
            self.failure = Some(message.clone());
            self.progress
                .send_modify(|p| p.error = Some(message.clone()));
            self.progress.send_modify(|p| p.recovering = false);
            let _=self.mutate(move|job|{job.message=format!("Live transcription stopped: {message}. Audio is still being recorded. Stop to retain the recording.");job.extra.insert("live_inference_error".into(),json!(message));if let Some(recovery)=job.extra.get_mut("live_recovery"){recovery["status"]=json!("failed");}Ok(())}).await;
        }
    }
    async fn recover(
        &mut self,
        error: LiveError,
        snapshot: &CaptureSnapshot,
        dual: bool,
    ) -> Result<(), ApiError> {
        if self.recovery_attempted
            || !matches!(
                error,
                LiveError::Child(ChildError::ProcessLost { .. } | ChildError::Timeout)
            )
            || self.operation.cancellation.is_cancelled()
        {
            self.fail(error.to_string()).await;
            return Ok(());
        }
        self.recovery_attempted = true;
        if let Some(worker) = self.worker.take()
            && let Err(error) = worker.join().await
        {
            self.cleanup_error = Some(error.to_string());
            self.fail(error.to_string()).await;
            return Ok(());
        }
        let app = self.app.clone();
        let id = self.id.clone();
        let op = self.operation.clone();
        let through = i64::from(snapshot.committed_sample);
        let plan = tokio::task::spawn_blocking(move || {
            app.store()
                .update_job(&id, |job| {
                    if !op.current(job) {
                        return Err(conflict("Recording was cancelled before recovery."));
                    }
                    RecoveryPlan::prepare(job, through)
                })?
                .map(|(_, plan)| plan)
                .ok_or_else(|| conflict("Recording disappeared before recovery."))
        })
        .await
        .map_err(|e| ApiError::new(500, e.to_string()))?;
        let plan = match plan {
            Ok(plan) => plan,
            Err(error) => {
                self.fail(error.message).await;
                return Ok(());
            }
        };
        self.received = through;
        self.progress.send_modify(|p| {
            p.recovering = true;
            p.error = None;
        });
        match worker(
            self.app.clone(),
            self.id.clone(),
            self.language.clone(),
            dual,
            snapshot.processing.clone(),
            self.operation.clone(),
            self.progress.clone(),
            Some(plan),
        )
        .await
        {
            Ok(worker) => self.worker = Some(worker),
            Err(error) => self.fail(error.to_string()).await,
        }
        Ok(())
    }
    async fn snapshot(&mut self, snapshot: &CaptureSnapshot, dual: bool) -> Result<(), ApiError> {
        self.progress
            .send_modify(|p| p.levels = snapshot.levels.clone());
        let app = self.app.clone();
        let id = self.id.clone();
        let duration = f64::from(snapshot.committed_sample) / 16000.;
        tokio::task::spawn_blocking(move || app.store().checkpoint_duration(&id, duration))
            .await
            .map_err(|e| ApiError::new(500, e.to_string()))??;
        if self.worker.is_none()
            && self.failure.is_none()
            && (!dual || snapshot.processing.is_some())
        {
            if let Some(receipt) = snapshot.processing.clone() {
                self.mutate(move |job| {
                    job.extra.insert(
                        "capture_processing".into(),
                        json!({"basename":"capture-processing.json","sha256":receipt.sha256}),
                    );
                    job.extra
                        .insert("capture_source_catalog".into(), receipt.catalog);
                    Ok(())
                })
                .await?;
            }
            match worker(
                self.app.clone(),
                self.id.clone(),
                self.language.clone(),
                dual,
                snapshot.processing.clone(),
                self.operation.clone(),
                self.progress.clone(),
                None,
            )
            .await
            {
                Ok(worker) => self.worker = Some(worker),
                Err(e) => self.fail(e.to_string()).await,
            }
        }
        if self.failure.is_none()
            && let Err(e) = self.feed(snapshot.committed_sample)
        {
            self.recover(e, snapshot, dual).await?;
        }
        if self.phase != snapshot.phase {
            self.phase = snapshot.phase;
            let phase = snapshot.phase;
            if matches!(
                phase,
                CapturePhase::Recording | CapturePhase::Paused | CapturePhase::Stopping
            ) {
                self.mutate(move |job| {
                    job.status = match phase {
                        CapturePhase::Recording => "recording",
                        CapturePhase::Paused => "paused",
                        _ => "finishing",
                    }
                    .into();
                    if !job.extra.contains_key("live_inference_error") {
                        job.message = match phase {
                            CapturePhase::Recording => "Recording…",
                            CapturePhase::Paused => "Paused",
                            _ => "Finishing saved audio…",
                        }
                        .into();
                    }
                    Ok(())
                })
                .await?;
            }
            if phase == CapturePhase::Paused && self.pause_pending {
                self.pause_pending = false;
                if self.failure.is_some() {
                    self.mutate(|job| {
                        job.extra
                            .insert("pause_flush".into(), json!({"status":"unavailable"}));
                        Ok(())
                    })
                    .await?;
                } else if let Some(worker) = &self.worker {
                    worker
                        .push(LiveCommand::Flush {
                            request_id: Some("pause".into()),
                            through_sample: Some(self.received),
                        })
                        .map_err(|e| conflict(e.to_string()))?;
                }
            }
        }
        Ok(())
    }
    async fn control(
        &mut self,
        action: Action,
        capture: &crate::capture::CaptureSession,
    ) -> Result<Value, ApiError> {
        if self.stopping {
            return Err(conflict("This meeting is finishing."));
        }
        match action {
            Action::Stop => {
                capture
                    .command(Command::Stop)
                    .map_err(|e| conflict(e.to_string()))?;
                self.stopping = true;
                self.mutate(|job| {
                    job.status = "finishing".into();
                    job.message = "Finishing saved audio…".into();
                    Ok(())
                })
                .await
            }
            Action::Pause => {
                if self.phase != CapturePhase::Recording || self.pause_pending {
                    return Err(conflict("This meeting is not recording."));
                }
                capture
                    .command(Command::Pause)
                    .map_err(|e| conflict(e.to_string()))?;
                self.pause_pending = true;
                self.mutate(|job| {
                    job.extra.insert(
                        "pause_flush".into(),
                        json!({"request_id":"pause","status":"pending"}),
                    );
                    Ok(())
                })
                .await
            }
            Action::Resume => {
                if self.phase != CapturePhase::Paused {
                    return Err(conflict("This meeting is not paused."));
                }
                let job = self
                    .mutate(|job| {
                        if job
                            .extra
                            .get("pause_flush")
                            .and_then(|p| p.get("status"))
                            .and_then(Value::as_str)
                            == Some("pending")
                        {
                            return Err(conflict("Wait for paused audio to finish processing."));
                        }
                        Ok(())
                    })
                    .await?;
                capture
                    .command(Command::Resume)
                    .map_err(|e| conflict(e.to_string()))?;
                Ok(job)
            }
            Action::Language { language, revision } => {
                if self.failure.is_some()
                    || self.progress.borrow().recovering
                    || self.worker.is_none()
                    || !matches!(self.phase, CapturePhase::Recording | CapturePhase::Paused)
                {
                    return Err(conflict(
                        "Live language changes require a running transcription worker.",
                    ));
                }
                let sample = self.received;
                let next = self.epoch + 1;
                let chosen = language.clone();
                let value=self.mutate(move|job|{
                    if number(job,"language_revision")!=revision{return Err(conflict("The meeting language changed. Reload and try again."));}
                    if number(job,"language_acknowledged_revision")!=revision{return Err(conflict("Wait for the previous language boundary to be processed."));}
                    job.language=chosen.clone();job.extra.insert("language_revision".into(),json!(next));job.extra.insert("language_epoch".into(),json!(next));
                    let history=job.extra.get_mut("language_history").and_then(Value::as_array_mut).ok_or_else(||conflict("Meeting language history is missing."))?;
                    history.push(json!({"epoch":next,"generation":next,"language":chosen,"start_sample":sample}));
                    let history=history.clone();
                    if let Some(provenance)=job.document.as_mut().and_then(|d|d.provenance.as_mut()).and_then(Value::as_object_mut){provenance.insert("language_history".into(),json!(history));}
                    Ok(())
                }).await?;
                self.worker
                    .as_ref()
                    .expect("checked worker")
                    .push(LiveCommand::Language {
                        language: language.clone(),
                        generation: next,
                        start_sample: sample,
                    })
                    .map_err(|e| conflict(e.to_string()))?;
                self.language = language;
                self.epoch = next;
                let app = self.app.clone();
                let chosen = self.language.clone();
                tokio::task::spawn_blocking(move || {
                    app.store().set_preference("default_language", &chosen)
                })
                .await
                .map_err(|e| ApiError::new(500, e.to_string()))??;
                let mut value = value;
                value
                    .as_object_mut()
                    .expect("stored job")
                    .insert("default_language".into(), json!(self.language));
                Ok(value)
            }
        }
    }
}

#[allow(clippy::too_many_arguments)]
pub(super) async fn record(
    app: AppState,
    id: JobId,
    language: String,
    sources: Vec<String>,
    helper: PathBuf,
    operation: Arc<OperationGuard>,
    _lease: ActivityLease,
    mut controls: mpsc::Receiver<Control>,
    progress: watch::Sender<Progress>,
) -> Result<(), StoreError> {
    let dual = sources.len() == 2;
    let mut state = Recording {
        app: app.clone(),
        id: id.clone(),
        operation: operation.clone(),
        language,
        epoch: 0,
        received: 0,
        worker: None,
        progress,
        failure: None,
        phase: CapturePhase::Starting,
        pause_pending: false,
        stopping: false,
        recovery_attempted: false,
        cleanup_error: None,
    };
    let config = CaptureConfig {
        job_id: id.as_str().into(),
        folder: app.data_dir().join(id.as_str()),
        microphone: sources.iter().any(|s| s == "microphone"),
        system: sources.iter().any(|s| s == "system"),
    };
    let capture = CaptureSupervisor::spawn(helper, config).await;
    let mut cleanup = Ok(());
    let mut retained_snapshot = None;
    let result = match capture {
        Err(error) => Err(error.to_string()),
        Ok(capture) => {
            let mut updates = capture.updates.clone();
            let running=async{
                loop {
                    let snapshot=updates.borrow_and_update().clone();
                    state.snapshot(&snapshot,dual).await.map_err(|e|e.message)?;
                    if matches!(snapshot.phase,CapturePhase::Stopped|CapturePhase::Failed){break Ok::<_,String>(());}
                    tokio::select!{
                        _=operation.cancellation.cancelled()=>break Err("Recording interrupted. Captured audio and edits are retained.".into()),
                        changed=updates.changed()=>{if changed.is_err(){break Err("Capture notices closed unexpectedly".into());}},
                        request=controls.recv()=>{
                            let Some(request)=request else{break Err("Meeting controls closed".into());};
                            // Observe every committed endpoint before freezing a language boundary.
                            let snapshot=updates.borrow_and_update().clone();
                            state.snapshot(&snapshot,dual).await.map_err(|e|e.message)?;
                            let response=state.control(request.action,&capture).await;
                            let _=request.reply.send(response);
                        }
                        _=async{match &mut state.worker {Some(w)=>{let _=w.result.changed().await;},None=>std::future::pending::<()>().await}}, if state.failure.is_none()=>{
                            if let Some(error)=state.worker.as_ref().and_then(Worker::failed){let snapshot=updates.borrow().clone();state.recover(error,&snapshot,dual).await.map_err(|e|e.message)?;}
                            else if state.worker.as_ref().is_some_and(|w|w.result.has_changed().is_err()){state.fail("Live worker ended unexpectedly".into()).await;}
                        }
                    }
                }
            }.await;
            // A capture failure is a product failure; a failed join is a cleanup failure.
            match capture.finish().await {
                Ok(final_snapshot) => {
                    retained_snapshot = Some(final_snapshot.clone());
                    if !operation.cancellation.is_cancelled() {
                        if let Err(e) = state.snapshot(&final_snapshot, dual).await {
                            state.failure.get_or_insert(e.message);
                        }
                    } else {
                        let app = app.clone();
                        let id = id.clone();
                        let duration = f64::from(final_snapshot.committed_sample) / 16000.;
                        let _ = tokio::task::spawn_blocking(move || {
                            app.store().checkpoint_duration(&id, duration)
                        })
                        .await;
                    }
                    running.and_then(|()| final_snapshot.error.map_or(Ok(()), Err))
                }
                Err(error) => {
                    let message = format!("Capture cleanup failed: {error}");
                    let saved = message.clone();
                    let _ = state
                        .mutate(move |job| {
                            job.status = "failed".into();
                            job.message = saved;
                            let generation =
                                job.extra["operation_generation"].as_u64().unwrap_or(0);
                            job.extra
                                .insert("operation_generation".into(), json!(generation + 1));
                            Ok(())
                        })
                        .await;
                    operation.cancellation.cancel();
                    cleanup = Err(StoreError::message(message.clone()));
                    Err(message)
                }
            }
        }
    };
    if state.worker.is_none() && result.is_ok() && state.failure.is_none() {
        state.failure = Some("Capture ended before live inference could initialize".into());
    }
    while let Some(worker) = state.worker.take() {
        if worker.result.borrow().is_none() {
            let command = if operation.cancellation.is_cancelled() {
                LiveCommand::Shutdown
            } else {
                LiveCommand::Stop
            };
            if let Err(e) = worker.push(command) {
                state.failure.get_or_insert(e.to_string());
            }
        }
        match worker.join().await {
            Ok(Err(error)) => {
                if state.failure.is_none()
                    && let Some(snapshot) = &retained_snapshot
                {
                    // EOF may be the first operation to observe a dead child. The capture
                    // writer is already drained; reconstruct its final saved prefix once.
                    if let Err(e) = state.recover(error, snapshot, dual).await {
                        state.failure.get_or_insert(e.message);
                    }
                } else {
                    state.failure.get_or_insert(error.to_string());
                }
            }
            Err(e) => cleanup = Err(e),
            Ok(Ok(())) => {}
        }
    }
    if let Err(e) = result {
        state.failure.get_or_insert(e);
    }
    if cleanup.is_err() {
        state
            .failure
            .get_or_insert("Live worker cleanup could not be confirmed".into());
    }
    if let Some(error) = state.cleanup_error.take() {
        cleanup = Err(StoreError::message(error));
    }
    if !operation.cancellation.is_cancelled() {
        let failure = state.failure.clone();
        if let Err(e)=state.mutate(move|job|{job.status=if failure.is_some(){"failed"}else{"ready"}.into();job.message=failure.map_or_else(||"Recording saved. Phrase text is editable; word timing and accuracy refinement are unavailable.".into(),|e|format!("{e}. Saved audio and edits are retained."));Ok(())}).await {cleanup=Err(StoreError::message(e.message));}
    }
    state.progress.send_modify(|p| p.error = state.failure);
    cleanup
}
