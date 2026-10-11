//! Update admission leases. The stdout `SPEAKERDESK_UPDATE=` frame is not used.
//!
//! Native commands travel on [`UpdateBridge`], an in-process channel installed
//! with `set_update_bridge`. Activity leases are RAII. Reservation refuses
//! recording (including pause), setup, import, refinement, and pending exports.

use std::collections::BTreeMap;
use std::sync::{Arc, Mutex, MutexGuard, OnceLock};

use serde_json::{Value, json};
use speakerdesk_core::messages;
use tokio::sync::mpsc::UnboundedSender;

const MAX_ATTEMPT: u64 = (1 << 53) - 1;

/// Work that holds the runtime. Every variant blocks an update reservation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Activity {
    /// Live capture. A paused meeting is still recording.
    Recording {
        paused: bool,
    },
    Setup,
    Import,
    Refinement,
    PendingExport,
}

impl Activity {
    pub const fn recording() -> Self {
        Self::Recording { paused: false }
    }

    pub const fn paused_recording() -> Self {
        Self::Recording { paused: true }
    }
}

/// Why admission refused a lease or an update.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AdmissionError {
    Reserved,
    Busy,
    Stale,
}

impl AdmissionError {
    pub const fn message(self) -> &'static str {
        match self {
            Self::Reserved => messages::UPDATE_RESERVED,
            Self::Busy => messages::UPDATE_BUSY,
            Self::Stale => messages::UPDATE_STALE,
        }
    }
}

impl std::fmt::Display for AdmissionError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str(self.message())
    }
}

impl std::error::Error for AdmissionError {}

/// Steps recorded for an editor acknowledgement and the shutdown handshake.
///
/// The handshake does not spawn a process. It records that quiesce was
/// signaled and that the inference child was asked to shut down. The caller
/// reports whether that child has exited.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Step {
    EditorReady { attempt: u64, preparation: u64 },
    QuiesceSignaled,
    InferenceChildShutdownRequested,
    ExitChecked { exited: bool },
}

#[derive(Debug, Default)]
struct State {
    next_lease: u64,
    leases: BTreeMap<u64, Activity>,
    attempt: u64,
    preparation: Option<u64>,
    last_preparation: u64,
    pending_install: bool,
    editor_ready: bool,
    reservation: Option<(u64, u64)>,
    shutting_down: bool,
    steps: Vec<Step>,
    replies: Option<UnboundedSender<Value>>,
    observed: u64,
}

/// Shared update admission. Clone shares the same leases and reservation.
#[derive(Clone, Debug)]
pub struct Admission {
    inner: Arc<Mutex<State>>,
}

impl Default for Admission {
    fn default() -> Self {
        Self::new()
    }
}

impl Admission {
    pub fn new() -> Self {
        Self {
            inner: Arc::new(Mutex::new(State::default())),
        }
    }

    /// Process-wide lock used by the desktop updater and later host routes.
    pub fn shared() -> Self {
        static SHARED: OnceLock<Admission> = OnceLock::new();
        SHARED.get_or_init(Admission::new).clone()
    }

    fn lock(&self) -> MutexGuard<'_, State> {
        self.inner
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
    }

    /// Hold `activity` until the lease is dropped.
    pub fn lease(&self, activity: Activity) -> Result<ActivityLease, AdmissionError> {
        let mut state = self.lock();
        if state.reservation.is_some() || state.shutting_down {
            return Err(AdmissionError::Reserved);
        }
        state.next_lease = state
            .next_lease
            .checked_add(1)
            .expect("activity lease id exhausted");
        let id = state.next_lease;
        state.leases.insert(id, activity);
        drop(state);
        Ok(ActivityLease {
            admission: self.clone(),
            id,
            activity,
        })
    }

    fn release_lease(&self, id: u64) {
        self.lock().leases.remove(&id);
    }

    /// True while any busy activity is leased.
    pub fn busy(&self) -> bool {
        !self.lock().leases.is_empty()
    }

    /// Download and install need an idle runtime.
    pub fn ensure_idle(&self) -> Result<(), AdmissionError> {
        let state = self.lock();
        if state.reservation.is_some() || state.shutting_down {
            return Err(AdmissionError::Reserved);
        }
        if !state.leases.is_empty() {
            return Err(AdmissionError::Busy);
        }
        Ok(())
    }

    pub fn reserved(&self) -> bool {
        self.lock().reservation.is_some()
    }

    pub fn steps(&self) -> Vec<Step> {
        self.lock().steps.clone()
    }

    /// `None` until the caller checks the inference child.
    pub fn inference_exit(&self) -> Option<bool> {
        self.steps().into_iter().rev().find_map(|step| match step {
            Step::ExitChecked { exited } => Some(exited),
            _ => None,
        })
    }

    pub fn observed_commands(&self) -> u64 {
        self.lock().observed
    }

    /// Attach the desktop reply channel. Later acknowledgements are sent there.
    pub fn bind_replies(&self, sender: UnboundedSender<Value>) {
        self.lock().replies = Some(sender);
    }

    fn emit(&self, reply: Value) {
        let sender = self.lock().replies.clone();
        if let Some(sender) = sender {
            let _ = sender.send(reply);
        }
    }

    /// Open the next preparation token for `attempt`.
    pub fn open_preparation(&self, attempt: u64) -> Result<u64, AdmissionError> {
        if attempt == 0 || attempt > MAX_ATTEMPT {
            return Err(AdmissionError::Stale);
        }
        let mut state = self.lock();
        if state.shutting_down {
            return Err(AdmissionError::Reserved);
        }
        let preparation = state
            .last_preparation
            .checked_add(1)
            .expect("preparation token exhausted");
        state.last_preparation = preparation;
        state.attempt = attempt;
        state.preparation = Some(preparation);
        state.pending_install = true;
        state.editor_ready = false;
        Ok(preparation)
    }

    /// Record that the editor saved and acknowledged this preparation.
    pub fn acknowledge_editor_ready(
        &self,
        attempt: u64,
        preparation: u64,
    ) -> Result<(), AdmissionError> {
        let reply = {
            let mut state = self.lock();
            if state.shutting_down
                || !state.pending_install
                || state.attempt != attempt
                || state.preparation != Some(preparation)
            {
                return Err(AdmissionError::Stale);
            }
            state.editor_ready = true;
            state.steps.push(Step::EditorReady {
                attempt,
                preparation,
            });
            json!({
                "op": "editor_ready",
                "id": attempt,
                "preparation": preparation,
            })
        };
        self.emit(reply);
        Ok(())
    }

    /// Reserve the runtime for this preparation.
    ///
    /// A host-opened preparation must already have an editor acknowledgement.
    /// A native `reserve` command for an idle process checks busy work only;
    /// the desktop state machine has already required the editor acknowledgement.
    pub fn reserve(&self, attempt: u64, preparation: u64) -> Result<(), AdmissionError> {
        if attempt == 0 || attempt > MAX_ATTEMPT || preparation == 0 || preparation > MAX_ATTEMPT {
            return Err(AdmissionError::Stale);
        }
        let mut state = self.lock();
        if state.shutting_down {
            return Err(AdmissionError::Reserved);
        }
        if state.pending_install
            && (state.attempt != attempt
                || state.preparation != Some(preparation)
                || !state.editor_ready)
        {
            return Err(AdmissionError::Stale);
        }
        if !state.leases.is_empty() {
            return Err(AdmissionError::Busy);
        }
        if let Some(owner) = state.reservation {
            if owner != (attempt, preparation) {
                return Err(AdmissionError::Stale);
            }
            return Ok(());
        }
        state.attempt = attempt;
        state.preparation = Some(preparation);
        state.reservation = Some((attempt, preparation));
        Ok(())
    }

    pub fn release(&self, attempt: u64, preparation: u64) -> Result<(), AdmissionError> {
        let mut state = self.lock();
        if state.shutting_down {
            return Err(AdmissionError::Reserved);
        }
        match state.reservation {
            Some(owner) if owner != (attempt, preparation) => Err(AdmissionError::Stale),
            Some(_) => {
                state.reservation = None;
                state.pending_install = false;
                state.editor_ready = false;
                Ok(())
            }
            None => Ok(()),
        }
    }

    /// Record quiesce and the inference-child shutdown request.
    ///
    /// Does not spawn a process and does not decide whether the child exited.
    pub fn shutdown_handshake(&self, attempt: u64, preparation: u64) -> Result<(), AdmissionError> {
        let mut state = self.lock();
        if state.reservation != Some((attempt, preparation)) || state.shutting_down {
            return Err(AdmissionError::Stale);
        }
        if !state.leases.is_empty() {
            return Err(AdmissionError::Busy);
        }
        state.shutting_down = true;
        state.steps.push(Step::QuiesceSignaled);
        state.steps.push(Step::InferenceChildShutdownRequested);
        Ok(())
    }

    /// Record the caller's check of whether the inference child has exited.
    ///
    /// Returns the `shutdown_ready` reply. `ok` is the caller's observation.
    pub fn confirm_exit(&self, exited: bool) -> Result<Value, AdmissionError> {
        let reply = {
            let mut state = self.lock();
            let Some((attempt, preparation)) = state.reservation else {
                return Err(AdmissionError::Stale);
            };
            if !state.shutting_down || !state.steps.contains(&Step::InferenceChildShutdownRequested)
            {
                return Err(AdmissionError::Stale);
            }
            state.steps.push(Step::ExitChecked { exited });
            json!({
                "op": "shutdown_ready",
                "id": attempt,
                "preparation": preparation,
                "ok": exited,
            })
        };
        self.emit(reply.clone());
        Ok(reply)
    }

    /// Apply one native command. Replies are returned to the desktop bridge.
    pub fn accept(&self, message: &Value) -> Option<Value> {
        let op = message.get("op").and_then(Value::as_str)?;
        let attempt = message.get("id").and_then(Value::as_u64)?;
        {
            let mut state = self.lock();
            state.observed = state.observed.saturating_add(1);
        }
        match op {
            "status" => None,
            "reserve" => {
                let preparation = message.get("preparation").and_then(Value::as_u64)?;
                Some(control_reply(
                    "reserved",
                    attempt,
                    preparation,
                    self.reserve(attempt, preparation),
                ))
            }
            "release" => {
                let preparation = message.get("preparation").and_then(Value::as_u64)?;
                Some(control_reply(
                    "released",
                    attempt,
                    preparation,
                    self.release(attempt, preparation),
                ))
            }
            "shutdown" => {
                let preparation = message.get("preparation").and_then(Value::as_u64)?;
                match self.shutdown_handshake(attempt, preparation) {
                    Ok(()) => None,
                    Err(error) => Some(control_reply(
                        "shutdown_ready",
                        attempt,
                        preparation,
                        Err(error),
                    )),
                }
            }
            _ => None,
        }
    }
}

fn control_reply(
    op: &str,
    attempt: u64,
    preparation: u64,
    result: Result<(), AdmissionError>,
) -> Value {
    match result {
        Ok(()) => json!({
            "op": op,
            "id": attempt,
            "preparation": preparation,
            "ok": true,
        }),
        Err(error) => json!({
            "op": op,
            "id": attempt,
            "preparation": preparation,
            "ok": false,
            "error": error.message(),
        }),
    }
}

/// RAII activity lease. Dropping it releases the busy hold.
#[must_use = "the activity stays busy until the lease is dropped"]
#[derive(Debug)]
pub struct ActivityLease {
    admission: Admission,
    id: u64,
    activity: Activity,
}

impl ActivityLease {
    pub fn activity(&self) -> Activity {
        self.activity
    }
}

impl Drop for ActivityLease {
    fn drop(&mut self) {
        self.admission.release_lease(self.id);
    }
}

struct BridgeInner {
    commands: std::sync::mpsc::Sender<Value>,
    inbox: Mutex<std::sync::mpsc::Receiver<Value>>,
    admission: Admission,
}

/// The in-process command channel is closed.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BridgeClosed;

impl std::fmt::Display for BridgeClosed {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str("The local service is unavailable")
    }
}

impl std::error::Error for BridgeClosed {}

/// In-process native → host channel. Store it with `set_update_bridge`.
#[derive(Clone)]
pub struct UpdateBridge {
    inner: Arc<BridgeInner>,
    replies: UnboundedSender<Value>,
}

impl UpdateBridge {
    /// Pair `admission` with a command channel and a desktop reply receiver.
    pub fn bind(admission: Admission) -> (Self, tokio::sync::mpsc::UnboundedReceiver<Value>) {
        let (commands, inbox) = std::sync::mpsc::channel();
        let (replies, incoming) = tokio::sync::mpsc::unbounded_channel();
        admission.bind_replies(replies.clone());
        let bridge = Self {
            inner: Arc::new(BridgeInner {
                commands,
                inbox: Mutex::new(inbox),
                admission,
            }),
            replies,
        };
        (bridge, incoming)
    }

    pub fn admission(&self) -> Admission {
        self.inner.admission.clone()
    }

    /// Queue one JSON command and apply whatever the inbox has received.
    pub fn dispatch(&self, command: Value) -> Result<(), BridgeClosed> {
        self.inner
            .commands
            .send(command)
            .map_err(|_| BridgeClosed)?;
        let pending = {
            let inbox = self
                .inner
                .inbox
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner());
            let mut pending = Vec::new();
            while let Ok(message) = inbox.try_recv() {
                pending.push(message);
            }
            pending
        };
        for message in pending {
            if let Some(reply) = self.inner.admission.accept(&message) {
                self.replies.send(reply).map_err(|_| BridgeClosed)?;
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn prepared() -> (Admission, u64) {
        let admission = Admission::new();
        let preparation = admission.open_preparation(4).unwrap();
        admission.acknowledge_editor_ready(4, preparation).unwrap();
        (admission, preparation)
    }

    #[test]
    fn busy_activities_use_the_update_busy_message() {
        let activities = [
            Activity::recording(),
            Activity::paused_recording(),
            Activity::Setup,
            Activity::Import,
            Activity::Refinement,
            Activity::PendingExport,
        ];
        for activity in activities {
            let admission = Admission::new();
            let lease = admission.lease(activity).unwrap();
            assert_eq!(lease.activity(), activity);
            assert!(admission.busy());
            let error = admission.ensure_idle().unwrap_err();
            assert_eq!(error, AdmissionError::Busy);
            assert_eq!(error.to_string(), messages::UPDATE_BUSY);
            let reserved = admission.reserve(4, 1).unwrap_err();
            assert_eq!(reserved.to_string(), messages::UPDATE_BUSY);
            drop(lease);
            assert!(!admission.busy());
            admission.ensure_idle().unwrap();
            admission.reserve(4, 1).unwrap();
        }
    }

    #[test]
    fn reservation_refuses_new_work_with_the_reserved_message() {
        let (admission, preparation) = prepared();
        admission.reserve(4, preparation).unwrap();
        let error = admission.lease(Activity::recording()).unwrap_err();
        assert_eq!(error, AdmissionError::Reserved);
        assert_eq!(error.to_string(), messages::UPDATE_RESERVED);
        assert!(admission.ensure_idle().is_err());
        admission.release(4, preparation).unwrap();
        let _lease = admission.lease(Activity::Setup).unwrap();
        assert!(admission.busy());
    }

    #[test]
    fn editor_acknowledgement_is_recorded_and_required() {
        let admission = Admission::new();
        let preparation = admission.open_preparation(4).unwrap();
        assert_eq!(
            admission
                .acknowledge_editor_ready(4, preparation - 1)
                .unwrap_err(),
            AdmissionError::Stale
        );
        assert_eq!(
            admission.reserve(4, preparation).unwrap_err(),
            AdmissionError::Stale
        );
        admission.acknowledge_editor_ready(4, preparation).unwrap();
        assert_eq!(
            admission.steps(),
            vec![Step::EditorReady {
                attempt: 4,
                preparation,
            }]
        );
        admission.reserve(4, preparation).unwrap();
        assert!(admission.reserved());
    }

    #[test]
    fn shutdown_handshake_records_quiesce_and_leaves_exit_to_the_caller() {
        let (admission, preparation) = prepared();
        let _recording = admission.lease(Activity::paused_recording()).unwrap();
        assert_eq!(
            admission
                .shutdown_handshake(4, preparation)
                .unwrap_err()
                .to_string(),
            messages::UPDATE_STALE
        );
        assert!(admission.steps().iter().all(|step| !matches!(
            step,
            Step::QuiesceSignaled | Step::InferenceChildShutdownRequested
        )));
        drop(_recording);
        admission.reserve(4, preparation).unwrap();
        admission.shutdown_handshake(4, preparation).unwrap();
        assert_eq!(
            admission.steps(),
            vec![
                Step::EditorReady {
                    attempt: 4,
                    preparation,
                },
                Step::QuiesceSignaled,
                Step::InferenceChildShutdownRequested,
            ]
        );
        assert_eq!(admission.inference_exit(), None);
        assert_eq!(
            admission.lease(Activity::Import).unwrap_err().to_string(),
            messages::UPDATE_RESERVED
        );

        let refused = admission.confirm_exit(false).unwrap();
        assert_eq!(refused["op"], "shutdown_ready");
        assert_eq!(refused["ok"], false);
        assert_eq!(admission.inference_exit(), Some(false));

        let ready = admission.confirm_exit(true).unwrap();
        assert_eq!(ready["ok"], true);
        assert_eq!(admission.inference_exit(), Some(true));
        assert!(
            admission
                .steps()
                .contains(&Step::ExitChecked { exited: false })
        );
        assert!(
            admission
                .steps()
                .contains(&Step::ExitChecked { exited: true })
        );
    }

    #[test]
    fn channel_reserve_and_shutdown_do_not_use_stdout_frames() {
        let admission = Admission::new();
        let (bridge, mut replies) = UpdateBridge::bind(admission);
        let host = bridge.admission();
        let lease = host.lease(Activity::PendingExport).unwrap();
        bridge
            .dispatch(json!({"op": "reserve", "id": 4, "preparation": 1}))
            .unwrap();
        let busy = replies.try_recv().unwrap();
        assert_eq!(busy["ok"], false);
        assert_eq!(busy["error"], messages::UPDATE_BUSY);
        assert!(
            !serde_json::to_string(&busy)
                .unwrap()
                .contains("SPEAKERDESK_UPDATE=")
        );
        drop(lease);

        bridge
            .dispatch(json!({"op": "status", "id": 4, "state": "preparing", "preparation": 1}))
            .unwrap();
        assert!(replies.try_recv().is_err());
        assert_eq!(host.observed_commands(), 2);

        bridge
            .dispatch(json!({"op": "reserve", "id": 4, "preparation": 1}))
            .unwrap();
        let reserved = replies.try_recv().unwrap();
        assert_eq!(reserved["op"], "reserved");
        assert_eq!(reserved["ok"], true);
        assert!(host.reserved());

        bridge
            .dispatch(json!({"op": "shutdown", "id": 4, "preparation": 1}))
            .unwrap();
        assert!(replies.try_recv().is_err());
        assert_eq!(
            host.steps(),
            vec![Step::QuiesceSignaled, Step::InferenceChildShutdownRequested,]
        );
        let reply = host.confirm_exit(true).unwrap();
        assert_eq!(replies.try_recv().unwrap(), reply);
        assert_eq!(reply["op"], "shutdown_ready");
        assert_eq!(reply["ok"], true);
    }

    #[test]
    fn stale_release_keeps_the_reservation() {
        let (admission, preparation) = prepared();
        admission.reserve(4, preparation).unwrap();
        assert_eq!(
            admission.release(3, preparation).unwrap_err(),
            AdmissionError::Stale
        );
        assert!(admission.reserved());
        admission.release(4, preparation).unwrap();
        assert!(!admission.reserved());
        admission.release(4, preparation).unwrap();
    }
}
