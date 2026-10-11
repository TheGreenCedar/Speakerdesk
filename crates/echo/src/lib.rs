//! Acoustic echo reference clock. The macOS agent replaces the mixer stub.
mod clock;
mod fft;

pub use clock::{
    CorrelationDecision, DELAY, LOOKAHEAD, RATE, ReferenceClock, SCORE_THRESHOLD,
    correlation_decision,
};

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
    /// Returns [`EchoError::Missing`] until the macOS static library is linked.
    pub fn start() -> Result<Self, EchoError> {
        Err(EchoError::Missing)
    }
}

#[cfg(all(test, not(target_os = "macos")))]
mod tests {
    use super::{EchoError, EchoMixer};

    #[test]
    fn mixer_stays_missing_without_the_aec_library() {
        assert!(matches!(EchoMixer::start(), Err(EchoError::Missing)));
    }
}
