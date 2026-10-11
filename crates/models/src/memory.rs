use speakerdesk_core::error::ChildError;
use speakerdesk_core::ipc::MetricsOut;
use speakerdesk_core::messages;

/// Refuses heavy work when the macOS memory-pressure free percentage is below 15.
pub fn ensure_memory_headroom() -> Result<(), ChildError> {
    match free_percent() {
        Ok(percent) if percent < 15 => {
            Err(ChildError::failed(messages::MEMORY_PRESSURE_HIGH, true))
        }
        Ok(_) => Ok(()),
        Err(error) => Err(error),
    }
}

pub fn free_percent() -> Result<u32, ChildError> {
    if cfg!(target_os = "macos") {
        Err(ChildError::failed(messages::MEMORY_PRESSURE_UNKNOWN, true))
    } else {
        Err(ChildError::failed(messages::MLX_APPLE_SILICON, false))
    }
}

pub fn metrics() -> Result<MetricsOut, ChildError> {
    Err(ChildError::Unimplemented)
}
