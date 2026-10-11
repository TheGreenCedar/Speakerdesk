# ADR-0001: Rust application with a resident inference child

## Status

Accepted design; application integration and model qualification are unfinished.

## Decision

Rust owns product behavior, persistence, publication and capture coordination.
Axum serves the existing frontend inside Tauri. One resident child isolates
native inference failures from recordings and the editor. Swift capture remains.

Speech model graphs and decoding use Rust with mlx-rs 0.32.0. Its locked native
runtime is MLX 0.32.2; the Python baseline uses 0.32.3. Qualify behavior and
performance rather than asserting identical runtimes. ReDimNet uses its existing
compiled artifact through objc2-core-ml; do not build a generic MIL interpreter.

Heavy inference requires measured acceleration. Ordinary CPU preprocessing and
mixed Core ML execution are allowed. Compare CPU_AND_GPU and ALL on equivalent
voice fixtures; ANE preference is not evidence of faster execution or compatible
embeddings. Saved profiles retain their original model identities.

The parent owns operation cancellation and child lifecycle. Control remains
serviceable during inference; termination is a bounded fallback followed by
reaping and generation invalidation. Model protocol types may change to carry
valid lengths, stream positions and operation identities correctly.

## Alternatives and limits

In-process inference would expose capture and the editor to runtime aborts.
MLX is the starting Mac backend, not a requirement to preserve Python APIs or
internal graphs. A runtime/operator or performance blocker requires a measured
reassessment. No project-owned C++ model implementation is planned.

Keep domain types independent of Metal and MLX. Windows implementation follows
Mac qualification; an ONNX Runtime/Windows ML route still needs graph/operator
qualification and separate AMD, NVIDIA and Intel hardware evidence.
