//! One request owns cancellation, including an in-flight native graph call.
use speakerdesk_core::error::ChildError;
use std::{
    ops::Deref,
    sync::atomic::{AtomicBool, Ordering},
};

#[derive(Default)]
pub struct Operation {
    cancelled: AtomicBool,
    #[cfg(windows)]
    run: Option<ort::session::RunOptions>,
}

impl Operation {
    /// Protocol-only executors must work without loading a native model runtime.
    pub fn new() -> Self {
        Self::default()
    }

    pub fn native() -> Result<Self, ChildError> {
        Ok(Self {
            cancelled: AtomicBool::new(false),
            #[cfg(windows)]
            run: Some(crate::windows_runtime::run_options()?),
        })
    }

    pub fn cancel(&self) {
        self.cancelled.store(true, Ordering::Release);
        #[cfg(windows)]
        if let Some(run) = &self.run {
            // The parent still owns bounded termination/reaping if a provider cannot stop.
            if let Err(error) = run.terminate() {
                eprintln!("Failed to interrupt Windows inference: {error}");
            }
        }
    }

    pub fn check(&self) -> Result<(), ChildError> {
        if self.cancelled.load(Ordering::Acquire) {
            Err(ChildError::Cancelled)
        } else {
            Ok(())
        }
    }

    #[cfg(windows)]
    pub fn run_options(&self) -> Result<&ort::session::RunOptions, ChildError> {
        self.check()?;
        self.run.as_ref().ok_or_else(|| {
            ChildError::failed("Native inference requires an initialized operation", false)
        })
    }
}

impl Deref for Operation {
    type Target = AtomicBool;
    fn deref(&self) -> &Self::Target {
        &self.cancelled
    }
}
