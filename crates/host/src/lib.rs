//! Host process. Route and supervisor bodies are filled by later agents.
pub mod admission;
pub mod capture;
pub mod http;
pub mod inference;
pub mod meeting;
pub mod people_routes;
pub mod routes;
pub mod setup;
pub mod worker;

pub use http::router;
