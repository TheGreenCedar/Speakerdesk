//! Fixed two-source MossFormer2 SS 16 kHz graph from ClearerVoice-Studio
//! `6b3774dc79c46ae8bed2a4fa5f706f0ac8c75c61` (Apache-2.0).
//! Raw full-window forward; the upstream window/RMS wrapper is deliberately absent.
#[cfg(target_os = "macos")]
mod layers;
mod manifest;
#[cfg(target_os = "macos")]
pub mod network;
#[cfg(target_os = "macos")]
pub mod prepare;

pub const SOURCE_SHA256: &str = "00a3a48bda492db1e829b85dd443f8f43a43039a3e90f1a24962ea9caf14a11a";
pub const SOURCE_FILE: &str = "last_best_checkpoint.pt";
pub const DERIVED_FILE: &str = "model.safetensors";
pub const DERIVED_SHA256: &str = "f7fc8ec3cdb3200c17c3bf52fa5f836b8d6ec203bd58897f16ffe25e71e74473";

#[cfg(windows)]
#[path = "network_windows.rs"]
pub mod network;
