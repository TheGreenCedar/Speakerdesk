//! Local HTTP host, persistence, capture coordination and inference supervision.
//! [`routes::router`] composes the API; [`native::run`] owns the process lifecycle.
pub mod admission;
pub mod capture;
pub mod http;
pub mod inference;
pub mod meeting;
pub mod native;
mod native_updates;
pub mod people_routes;
pub mod recovery;
pub mod routes;
pub mod setup;
pub mod worker;

pub use http::router;
