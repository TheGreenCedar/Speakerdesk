//! Swift helper supervisor. Protocol commands stay `start`, `pause`, `resume`, `stop`.
#[derive(Debug, Default)]
pub struct CaptureSupervisor;

impl CaptureSupervisor {
    pub fn new() -> Self {
        Self
    }
}
