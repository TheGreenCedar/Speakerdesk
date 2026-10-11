# ADR-0001: Python-free Rust core with mlx-rs in a resident inference child

## Status

Accepted

## Context

Speakerdesk 0.6.6 runs product behavior in a Python sidecar beside the Tauri shell. Model work uses MLX on Apple Silicon. A Metal abort or an out-of-memory failure in that process can take down the recording writer and the saved job.

## Decision

The shipped app is a Rust core. axum serves the existing frontend and JSON API on `127.0.0.1` inside the Tauri process. MLX runs only in one resident child, the same executable started as `speakerdesk --inference`, using mlx-rs 0.32.x bound to the MLX version 0.6.6 ships. The parent owns import, speech admission, language routing, the live engine, refinement, and persistence. The child holds weights and streaming state.

## Alternatives

- In-process MLX inside Tauri. A Metal failure would kill the UI and the recording.
- ONNX Runtime. It would not run the pinned MLX graphs the 0.6.6 models use.
- candle. It is a different graph runtime and would not match the pinned mlx-audio and mlx-speech behavior.
- Raw mlx-c bindings. mlx-rs already wraps that surface at the required 0.32 pin.

## Consequences

Orchestration can be tested with model doubles. A child crash loses model state only. There is no CPU neural fallback when Metal is unavailable. Windows is a later port of the same `Models` trait.
