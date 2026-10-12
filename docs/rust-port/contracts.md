# Integration contracts

Preserve product behavior and saved data. Change incomplete internal interfaces together with their callers and tests when the assembled application requires it.

## Toolchain

Rust **1.99.0** (`rust-toolchain.toml`), edition 2024. This is the stable 1.99 release from `rustup update stable` (rustc `1.99.0`, `b940084d7`, 2026-09-28).

## Crates

- `speakerdesk-core` — domain types, `Models`, IPC payloads, messages, schema SQL.
- `speakerdesk-models` — inference child. Linux builds the protocol stub. mlx-rs 0.32.0 is a macOS dependency.
- `speakerdesk-echo` — AEC3 stub off macOS.
- `speakerdesk-host` — axum process. `speakerdesk-dev` is behind the `fixtures` feature.
- `xtask` — `cargo xtask ledger --check` and `cargo xtask ownership --check`.

## Models and IPC

`speakerdesk_core::models::Models` is the sync trait. Payloads are `speakerdesk_core::ipc::Request` and `Response`. A frame is a big-endian `u32` length plus a postcard `Envelope` (`request_id` and `Result`). Timeouts are `request_timeout`. Offline diarization uses the same open/feed/close boundary as live sessions, with bounded parent feeds; there is no child file-access or whole-recording diarization request.

The Mac child initializes MLX and Metal resources before reporting its device.
Hello lists only loaded model identities. Silero, multi-window Nemotron,
Cohere, Whisper language scoring and supplied-text CTC run in Rust; ReDimNet uses
native Core ML. Nemotron Low and Offline state belongs to each recording, with
resident weights shared by the serial executor. Diarization turns use absolute
sample coordinates and separate received and observed endpoints. Its final
sub-160-sample suffix remains unobserved. Live capture integration remains
unfinished. Other platforms report
`"unavailable"` and explicit `ChildError::Unimplemented` results.

Numerical comparisons identify the native runtime and Metal library hashes,
not just the MLX package version. Apple's macOS 15 and macOS 26 wheels contain
different kernels under the same 0.32.2 version. This build pins the macOS 15
archive in `scripts/fetch_mlx_runtime.sh`.

CTC scores carry the original audio length, valid frame count, frame hop/origin,
explicit selected column IDs (including blank when requested), model identity
and vocabulary digest. There is no implicit blank column or padded-frame count.
Scores are normalized across all 9,812 tokens before selection. Core owns the
supplied-text adapter and the existing reading projection; model output cannot
replace transcript text. Vocabulary parsing and artifact checks occur once per
load, outside inference loops. The native provider has a distinct versioned
identity; saved documents cannot nominate their own trusted provider.

## Jobs

`Job` and `Document` round-trip the product fields named in `document.rs`. A missing `kind` deserializes as `import`. Job ids are 32 lowercase hex characters. `store::SCHEMA_V1` is the new database. `AppState::open` owns one file-backed store at an explicit root. Migration uses a consistent snapshot, private hot-journal recovery, and no-clobber publication; existing native state wins. Migrated audio remains legacy-owned and read-only. Native writes use copy-on-write media storage. Qualification has used disposable stores only.

Saved canonical accuracy passes use the existing parent operation and serial
worker. Each candidate owns one original audio anchor and captured language mode;
publication compares that passage with its current stored version and replaces
its slot without changing names, neighboring passages or source order. Pause
invalidates publication before cancellation, and admission remains held until
the worker exits. Completed passage versions survive restart; history retains
only the previous words, bounds, identity and speaker. This first saved path
supports Auto and explicit-language anchors up to 24.5 seconds. Auto reruns
language detection from the saved source audio, preserves captured mode and epoch,
and aligns with the newly selected language. Incomplete, uncertain or mixed-language
replacements retain the previous words. Longer anchors and scheduling during
capture remain unfinished.

## Messages and security

User-facing strings live in `speakerdesk_core::messages` and are copied from v0.6.6. The token header, CSP, and host allowlist live in `speakerdesk_core::security`. Do not change them without an owner decision.

## Ledger

`xtask/port-ledger.toml` lists every `tests/test_*.py`. Dropped rows carry a `D`-reference. `pending` rows are allowed until the owning agent lands tests. `cargo xtask ledger --check` fails if a test file is missing from the ledger.

Native model installation is owned by the parent, with one fs4 installer lock and
an exclusive setup admission lease. The child receives only `PrepareAlignment`
for its fixed private staging path, never an arbitrary installation destination.
Cancellation retains admission until terminal cleanup. Atomic model publication is
followed by parent-controlled child restart; old session generations cannot survive
an installation. Readiness comes from initialization of the activated identities.
The installer does not enroll voices or hold a SQLite transaction during I/O.
