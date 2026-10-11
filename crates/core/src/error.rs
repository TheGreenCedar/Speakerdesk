use serde::{Deserialize, Serialize};
use thiserror::Error;

/// HTTP error body is always `{"error": message}` with a verbatim v0.6.6 string.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ApiError {
    pub status: u16,
    pub message: String,
}

impl ApiError {
    pub fn new(status: u16, message: impl Into<String>) -> Self {
        Self {
            status,
            message: message.into(),
        }
    }

    pub fn body(&self) -> serde_json::Value {
        serde_json::json!({ "error": self.message })
    }
}

/// Failures that cross the inference-child frame boundary.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, Error)]
#[serde(rename_all = "snake_case")]
pub enum ChildError {
    #[error("{message}")]
    Failed { message: String, retryable: bool },
    #[error("unimplemented")]
    Unimplemented,
    #[error("cancelled")]
    Cancelled,
    #[error("timeout")]
    Timeout,
    #[error("metal unavailable")]
    MetalUnavailable,
}

impl ChildError {
    pub fn failed(message: impl Into<String>, retryable: bool) -> Self {
        Self::Failed {
            message: message.into(),
            retryable,
        }
    }
}

#[derive(Debug, Error)]
pub enum StoreError {
    #[error("{0}")]
    Message(String),
    #[error("unimplemented")]
    Unimplemented,
    #[error(transparent)]
    Sqlite(#[from] rusqlite::Error),
    #[error(transparent)]
    Migration(#[from] rusqlite_migration::Error),
}

impl StoreError {
    pub fn message(text: impl Into<String>) -> Self {
        Self::Message(text.into())
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Error)]
pub enum IdError {
    #[error("job id must be 32 lowercase hex characters")]
    JobId,
}
