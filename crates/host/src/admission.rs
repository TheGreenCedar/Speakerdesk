//! Work leases and updater reservation belong to the same native application.
//! The desktop control protocol and editor acknowledgement live in native_updates.

use std::collections::BTreeMap;
use std::sync::{Arc, Mutex, MutexGuard};

use serde_json::{Value, json};
use speakerdesk_core::messages;

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

#[derive(Debug, Default)]
struct State {
    next_lease: u64,
    leases: BTreeMap<u64, Activity>,
    reservation: Option<(u64, u64)>,
    shutting_down: bool,
    update_shutdown: bool,
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

    /// Close ordinary work admission before native shutdown starts.
    pub fn close(&self) {
        self.lock().shutting_down = true;
    }

    pub fn closed(&self) -> bool {
        self.lock().shutting_down
    }

    /// Reserve idle work after native_updates verifies the current editor acknowledgement.
    pub fn reserve(&self, attempt: u64, preparation: u64) -> Result<(), AdmissionError> {
        if attempt == 0 || attempt > MAX_ATTEMPT || preparation == 0 || preparation > MAX_ATTEMPT {
            return Err(AdmissionError::Stale);
        }
        let mut state = self.lock();
        if state.shutting_down {
            return Err(AdmissionError::Reserved);
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
        state.update_shutdown = true;
        Ok(())
    }

    /// Format the observation made by the caller after actual cleanup and reaping.
    pub fn confirm_exit(&self, exited: bool) -> Result<Value, AdmissionError> {
        let state = self.lock();
        let Some((attempt, preparation)) = state.reservation else {
            return Err(AdmissionError::Stale);
        };
        if !state.update_shutdown {
            return Err(AdmissionError::Stale);
        }
        Ok(json!({"op":"shutdown_ready","id":attempt,"preparation":preparation,"ok":exited}))
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn busy_activities_block_reservation_until_their_lease_is_dropped() {
        for activity in [
            Activity::recording(),
            Activity::paused_recording(),
            Activity::Setup,
            Activity::Import,
            Activity::Refinement,
            Activity::PendingExport,
        ] {
            let admission = Admission::new();
            let lease = admission.lease(activity).unwrap();
            assert!(admission.busy());
            assert_eq!(admission.ensure_idle(), Err(AdmissionError::Busy));
            assert_eq!(admission.reserve(4, 1), Err(AdmissionError::Busy));
            drop(lease);
            admission.reserve(4, 1).unwrap();
            assert!(matches!(
                admission.lease(activity),
                Err(AdmissionError::Reserved)
            ));
            assert_eq!(admission.reserve(4, 2), Err(AdmissionError::Stale));
            assert_eq!(admission.release(4, 2), Err(AdmissionError::Stale));
            assert!(admission.reserved());
            admission.release(4, 1).unwrap();
            admission.release(4, 1).unwrap();
            admission.ensure_idle().unwrap();
        }
    }

    #[test]
    fn shutdown_closes_work_and_cannot_invent_an_exit_observation() {
        let admission = Admission::new();
        assert!(admission.confirm_exit(true).is_err());
        admission.reserve(4, 1).unwrap();
        assert!(admission.confirm_exit(true).is_err());
        assert!(admission.shutdown_handshake(4, 2).is_err());
        admission.shutdown_handshake(4, 1).unwrap();
        assert!(admission.closed());
        assert!(admission.release(4, 1).is_err());
        assert!(admission.lease(Activity::Import).is_err());
        assert_eq!(admission.confirm_exit(false).unwrap()["ok"], false);
        assert_eq!(admission.confirm_exit(true).unwrap()["ok"], true);
        let ordinary = Admission::new();
        ordinary.close();
        assert!(ordinary.ensure_idle().is_err());
        assert!(ordinary.confirm_exit(true).is_err());
    }
}
