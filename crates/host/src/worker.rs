//! Serial import job worker. One job at a time, matching the v0.6.6 executor.
#[derive(Debug, Default)]
pub struct JobWorker;

impl JobWorker {
    pub fn new() -> Self {
        Self
    }
}
