# Architecture

Speakerdesk is one macOS executable that runs in three modes, plus a Swift capture helper. See [ADR-0001](adr/0001-rust-core-inference-child.md) for why inference lives in a separate process.

| Mode | Entry | Owns |
| --- | --- | --- |
| Tauri shell (default) | `desktop/src-tauri/src/main.rs` | Window, native menus, exports, updater. Starts the host. |
| `--native-host <storage> <models>` | `speakerdesk_host::native::run` | axum HTTP API and the embedded editor, persistence, capture coordination, setup, scheduling. |
| `--inference <models>` | `speakerdesk_models::serve` | All neural inference, in one resident child process. |
| `speakerdesk-capture` | `desktop/capture/*.swift` | Microphone (AVAudioEngine) and system audio (ScreenCaptureKit). |

Storage defaults to the app data directory (`SPEAKERDESK_HOME` overrides); `SPEAKERDESK_NATIVE_MODELS` overrides the model directory.

## Crates

- `speakerdesk-core`: domain types and product rules, with no I/O to models or Metal. Holds the document, edit, import, live, language, speech, timing/alignment, review, export, people and voice rules, plus the store, IPC payloads, `messages` and `security`.
- `speakerdesk-host`: the axum process. It runs routes, the inference supervisor (`inference.rs`), capture transport/writer/processing, admission, setup and workers. The `fixtures` feature builds `speakerdesk-dev` and `inference-fixture` for tests.
- `speakerdesk-models`: the inference child. On macOS it runs Silero, Nemotron (Sortformer), Cohere, Whisper language scoring and the CTC timing model with mlx-rs, and runs ReDimNet through Core ML. Linux builds a protocol stub that reports `unavailable`.
- `speakerdesk-echo`: buffering and reference clock for the pinned native WebRTC AEC3 library (`desktop/capture/echo`, built by `scripts/setup_echo.py`).
- `xtask`: development commands (`cargo xtask --help`).

The editor frontend lives in `speakerdesk/static/` and `speakerdesk/templates/` and is embedded into the host with `rust-embed`. The JSON files under `speakerdesk/` are model and voice policy metadata. There is no Python at runtime.

## Inference boundary

`speakerdesk_core::models::Models` is the sync model trait. The host implements it with `ChildModels`, which talks to the child over stdin/stdout. Each frame is a big-endian `u32` length followed by a postcard `Envelope` (`request_id` plus `Result`), with a 64 MiB frame cap (`ipc.rs`). Requests are Hello, VAD and diarization open/feed/close, language probe, transcribe, CTC scores, voice embedding and two-voice separation.

- The parent owns cancellation and the child's lifecycle. Control stays responsive while inference runs. Termination is a bounded fallback, followed by reaping and generation invalidation, so old session handles cannot survive a restart.
- Offline diarization uses the same open/feed/close sessions as live capture. The child never reads files and has no whole-recording request.
- Diarization turns use absolute sample coordinates.
- CTC scores carry audio length, valid frame count, frame hop/origin, explicit column IDs, model identity and vocabulary digest. Core owns the supplied-text alignment; model output never replaces Cohere's transcript text.
- Numerical comparisons must identify the native MLX runtime and Metal library, not just the MLX version. The build pins the macOS 15 MLX 0.32.2 archive in `scripts/fetch_mlx_runtime.sh`, and `vendor/mlx-sys` patches the build to link it (see `vendor/mlx-sys/SPEAKERDESK.md`).

### Inference performance

For sustained inference-performance claims, evaluate live meetings with primary transcription and auxiliary work active. Observe publication progress for both, accumulated backlog and Stop completion; measure memory, energy and thermal behavior as required by the accepted comparison. Compare completed equivalent work, including final outputs and failures; elapsed time alone can hide deferred or omitted work.

Reuse applicable measurements and keep new runs within the accepted scope and budget. For energy or battery claims, compare equivalent power-source and thermal conditions; report confounds rather than inferring a benefit from isolated model latency or a shorter Stop drain.

Distinguish requested model/device policy from observed runtime placement. The serial worker and child described below are the current design; they do not prove that serialization is a runtime constraint. Require runtime evidence before treating a model, backend or device limitation as a reason to keep or change that design.

Performance changes must preserve transcription and speaker accuracy, publication ownership, protected edits, continuation state, cancellation and recovery. Use the product rules in [Product behavior](product-behavior.md) as acceptance criteria for the same live interaction.

## Storage

The store is `<storage>/speakerdesk.sqlite` (`core/src/store/`). On first start the host discovers a legacy `jobs.sqlite` and migrates it once into the new file without modifying it ([ADR-0002](adr/0002-one-way-sqlite-migration.md)). An existing native database wins. Migrated audio stays read-only, and native writes use copy-on-write media storage. The schema stores product fields only ([ADR-0003](adr/0003-product-fields-only.md)).

`Job` and `Document` round-trip the fields in `document.rs`. A missing `kind` deserializes as `import`. Job IDs are 32 lowercase hex characters.

## Refinement and publication

Live text comes from the serial recording worker. Saved and live accuracy passes reuse the parent operation and the serial worker. Each candidate owns one original audio anchor and its captured language mode and epoch. Publication compares the passage with its current stored version and replaces only that slot. Names, neighbouring passages and source order stay as they are, and protected edits always win.

A failed language recheck preserves the existing words and language evidence and attaches a passage-specific issue only while its operation, source and passage version remain current. The editor uses that issue to expose the language repair beside retained words. Successful publication, an accepted repair or a change to the passage's authority clears it; ordinary naming changes do not.

Language context is derived from direct acoustic observations belonging to the same source, language epoch and stream generation. A qualified speaker main language takes precedence over the meeting main language; display names and saved-person assignments do not identify the acoustic track. Pauses do not erase the main language, and saved refinement can use later observed speech. Borrowed decisions carry their supporting observations and never become independent votes. The Engine and accuracy operation own compact physical-probe memberships and language counts, updated at accepted canonical and accuracy publication. Replacements remove former votes and preserve physical duplicate, conflict and overlap rules without scanning transcript history for each passage. Relevant interpretation edits invalidate them through one job generation; spelling and display-name edits do not force history rescans. Live decoding and saved refinement use the same selection rule, with crop clocks translated only at the import adapter boundary.

- Pause invalidates publication before it cancels. Admission stays held until the worker exits.
- Complete supported-language output may retain routing warnings. Short mixed-language output keeps per-language timing receipts; long context composition still requires consistent language ownership and unambiguous timing.
- Stop drains live EOF before running the remaining accuracy work and retries failed live anchors once. Shared recording control retains this limit across worker recovery.
- Child loss keeps its typed cause, rebuilds live sessions once from saved audio, and lets unfinished accuracy work retry under the new generation.
- Long rows plan only the unaccepted suffix. Accepted cores keep their text, language decisions and final sample boundary.
- GET, export and reopen never run models to rebuild metadata.

The recording worker also owns live voice recognition and separation under the same operation and cancellation token. It retains one auxiliary continuation and services pending live audio and controls between separation blocks and separated-transcription decoder steps, and between admission, language, timing and embedding requests. Models remain serial: encoder work, individual blocks and primary live ASR requests finish before yielding. Child session handles retain the full source window without chunk stitching; cancellation, close and generation invalidation release them. No additional inference worker runs concurrently.

Recognition checks observed solo crops without waiting for an ongoing passage to close; enrollment still requires finalized clips. Separated attribution uses its own measured model policy, one speech-admitted query crop per channel, unchanged natural source clips and compatible consented profiles. Generated embedding views receive quantization headroom; original recordings, saved estimates and ASR input stay unchanged. Store locks do not span inference. Publication rechecks source ownership, reference audio, names, assignments, profiles, recognition preference and the accepted rows. New saved-voice assignments remain unconfirmed and never enroll a voice. Protected edits and stale generations win; accepted separated rows prevent a later live snapshot from restoring the original mixed row.

## Capture

The helper streams PCM and events to the host (`host/src/capture/`). Apple voice-processing input is used when the route qualifies. Raw dual-source capture instead runs the bundled echo library on one writer thread. Original tracks flush before DSP, and only completed clean/mixed output advances the inference endpoint. With no detected echo, the microphone signal passes through unchanged. Resume requires a fresh producer observation that matches the original route and formats. Physical device restart is a hardware test gate; replay tests cover the protocol.

## Model installation

The parent owns installation (`host/src/setup.rs`, pins in `host/src/model-artifacts.json`). It holds an fs4 installer lock and an exclusive setup admission lease. Downloads are pinned and hash-checked. The timing model's MLX weights are prepared locally from the pinned source model. The child only prepares alignment in its fixed private staging path. Publication is atomic and is followed by a parent-controlled child restart. Readiness comes from initializing the activated model identities. The installer never enrolls voices and never holds a SQLite transaction during I/O.

## Messages and security

User-facing strings live in `speakerdesk_core::messages`. The token header, CSP and host allowlist live in `speakerdesk_core::security`. Do not change either without an owner decision.
