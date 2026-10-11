# Python-free Rust core with mlx-rs in a resident inference child

Speakerdesk 0.7 replaces the bundled Python sidecar with a Rust core. The core runs inside the Tauri process and serves the existing frontend over loopback HTTP. All model inference runs in one resident child process, `speakerdesk --inference`, built on mlx-rs (MLX 0.32). The parent talks to it through a small typed request protocol behind the `Models` trait. The child is the only process that touches MLX/Metal, so a Metal abort or out-of-memory kill loses model state but never the UI, the recording writer or the database. All orchestration lives in the parent, which means it can be tested with scripted models.

## Considered Options

- **In-process MLX.** Rejected. A GPU abort would take down live recording and persistence.
- **ONNX Runtime / CoreML.** Rejected. The pinned checkpoints are MLX safetensors (including 4-bit Cohere), CoreML can silently fall back to the CPU, and conversion would change numerics against 0.6.6.
- **candle.** Rejected. It has no Metal kernels for MLX's 4-bit quantized layout, and every graph would need a rewrite and requalification.
- **Raw mlx-c FFI.** Rejected. mlx-rs already provides safe bindings at the same MLX version, so raw FFI would add unsafe surface for no gain.
- **Keeping the stopped migration's architecture.** Rejected by the owner (#24). It ran recursive qualification in the product hot path and never produced an editable transcript.
