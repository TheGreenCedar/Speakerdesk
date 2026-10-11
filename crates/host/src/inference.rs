//! `InferenceSupervisor` over the child stdio protocol. Filled by the host-inference agent.
#[derive(Debug, Default)]
pub struct InferenceSupervisor;

impl InferenceSupervisor {
    pub fn new() -> Self {
        Self
    }
}
