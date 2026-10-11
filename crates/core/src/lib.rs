//! Speakerdesk domain crate. Contract modules are frozen by the coordinator.
//! Feature modules start as stubs and are filled by the owning port agent.
pub mod audio;
pub mod document;
pub mod edit;
pub mod error;
pub mod export;
pub mod ids;
pub mod import;
pub mod ipc;
pub mod language;
pub mod languages;
pub mod limits;
pub mod live;
pub mod memo;
pub mod messages;
pub mod models;
pub mod people;
pub mod review;
pub mod security;
pub mod speech;
pub mod store;
pub mod timing;
pub mod timing_policy;
pub mod transcript;
pub mod voice;
