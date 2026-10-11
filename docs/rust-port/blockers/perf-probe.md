# Performance probe

This host is `Linux 6.12.94+ x86_64` with `rustc 1.99.0 (b940084d7 2026-09-28)`.

The Silero, Sortformer, and Cohere probe compares a Rust child against the packaged Speakerdesk 0.6.6 Mac app on Apple Silicon. It did not run here. No model agent is re-dispatched, because there is no measurement and no regression.

Run `cargo xtask probe` on the owner's Mac after those three model arms are real.
