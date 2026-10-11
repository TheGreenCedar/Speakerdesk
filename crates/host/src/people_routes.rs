//! People and voice HTTP routes. The people-route agent fills these handlers.
use axum::Router;

use crate::routes::AppState;

pub fn routes() -> Router<AppState> {
    Router::new()
}
