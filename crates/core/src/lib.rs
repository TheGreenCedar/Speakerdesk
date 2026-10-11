//! Speakerdesk domain core: domain types, transcript rules, edit/merge,
//! exports, review, timing (CTC segmentation, reading turns), speech ledger,
//! language routing, import pipeline, live engine, refinement controller,
//! people/voice logic, the `Models` seam, IPC protocol types, audio file
//! I/O/decode/resample/mixer, the store and the legacy importer.
//!
//! Sync code with no tokio.

pub mod edit;
pub mod error;
pub mod model;
pub mod models;
pub mod review;
pub mod store;
pub mod transcript;
