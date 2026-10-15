//! Inference child. The control loop owns request cancellation;
//! model modules own preprocessing, decoding and native inference state.
pub mod cohere;
pub mod ctc;
pub mod dispatch;
pub mod memory;
pub mod metallib;
pub mod mossformer2;
pub mod operation;
pub mod redimnet;
pub mod serve;
pub mod silero;
pub mod sortformer;
pub mod whisper_lid;

#[cfg(windows)]
pub mod windows_runtime;

#[cfg(target_os = "macos")]
mod weights;

pub use serve::serve;
