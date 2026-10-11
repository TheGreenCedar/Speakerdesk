//! AEC3 wrapper. The macOS agent replaces the stub bodies.
use speakerdesk_core::messages;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EchoError {
    Missing,
    IncompatibleVersion,
    FailedToStart,
}

impl std::fmt::Display for EchoError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let text = match self {
            Self::Missing => messages::ECHO_MISSING,
            Self::IncompatibleVersion => messages::ECHO_INCOMPATIBLE,
            Self::FailedToStart => messages::ECHO_FAILED,
        };
        formatter.write_str(text)
    }
}

impl std::error::Error for EchoError {}

pub struct EchoMixer;

impl EchoMixer {
    pub fn start() -> Result<Self, EchoError> {
        Err(EchoError::Missing)
    }
}

#[derive(Debug, Default)]
pub struct ReferenceClock;
