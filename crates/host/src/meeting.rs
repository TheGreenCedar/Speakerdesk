//! Live meeting manager.
use axum::Router;

use crate::routes::AppState;

#[derive(Debug, Default)]
pub struct MeetingManager;

/// Meeting routes. Keep this signature; handlers are added by the meeting agent.
pub fn routes() -> Router<AppState> {
    Router::new()
}

impl MeetingManager {
    pub fn new() -> Self {
        Self
    }
}
