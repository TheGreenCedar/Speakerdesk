//! Inference child. Dispatch and the stdio loop are coordinator-owned.
//! Each model directory is owned by its model agent.
pub mod cohere;
pub mod ctc;
pub mod dispatch;
pub mod memory;
pub mod metallib;
pub mod redimnet;
pub mod serve;
pub mod silero;
pub mod sortformer;
pub mod whisper_lid;

#[cfg(target_os = "macos")]
mod weights;

pub use serve::serve;
