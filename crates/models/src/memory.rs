use speakerdesk_core::error::ChildError;
use speakerdesk_core::ipc::MetricsOut;

pub fn metrics() -> Result<MetricsOut, ChildError> {
    Err(ChildError::Unimplemented)
}
