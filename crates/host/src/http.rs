use axum::Router;

use speakerdesk_core::security::{
    CACHE_CONTROL, CONTENT_SECURITY_POLICY, REFERRER_POLICY, TOKEN_HEADER, X_CONTENT_TYPE_OPTIONS,
};

pub fn router() -> Router {
    crate::routes::router(crate::routes::AppState::new())
}

/// Headers every response keeps. The route agent applies these and must not change the values.
pub fn security_headers() -> [(&'static str, &'static str); 4] {
    [
        ("content-security-policy", CONTENT_SECURITY_POLICY),
        ("x-content-type-options", X_CONTENT_TYPE_OPTIONS),
        ("referrer-policy", REFERRER_POLICY),
        ("cache-control", CACHE_CONTROL),
    ]
}

pub fn token_header() -> &'static str {
    TOKEN_HEADER
}
