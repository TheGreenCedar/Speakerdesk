# Python-free Mac implementation

Accepted outcome: complete the native Mac application, preserving the shipped
0.6.6 behavior and saved data. Real import/capture must produce an editable
transcript that survives full application restart and export. Equivalent release
measurements must match or improve baseline latency, sustained live processing,
and peak memory. MossFormer2/episode work belongs to issue 25; Windows comes later.

Starting PR head: `01e757cd5f3db3f8fccfcfa34e94e91cadc0fe26`.
Starting base: `59e7b0c860f72ca94c24e6eab18ac820b144ee24`.
Local implementation branch: `codex/python-free-mac`.

## Sequence and current state

1. Source repairs verified: updater/export/bundle/parser/readiness and truthful
   tests. Mac CI added; prebuilt MLX linkage demonstrated (details below).
   Independent verifier architecture_capture found three issues, all corrected.
2. In progress: data slice implemented and independently reviewed. Explicit-root
   AppState uses SQLite for jobs, identities and preferences, immediate edit
   transactions, guarded import publication, consistent migration snapshots,
   private journal recovery and legacy audio copy-on-write. Process lifecycle and
   operation cancellation implemented; active meeting state integration remains pending.
3. Pending: Silero, offline Nemotron and Cohere through the real desktop;
   edit/save/quit/reopen/export with a short public fixture and measured Metal.
4. Pending: automatic language, CTC, native Core ML ReDimNet, identity and setup.
5. Pending: native capture coordination, streaming tails/recovery and bounded work.
6. Pending: Python-free packaged execution, full product/security/performance gates.

All validation uses disposable storage and synthetic/public audio. No installed
app replacement, production-data migration, cache mutation, merge, publication,
or GitHub comments. One implementer and one independent verifier per batch; one
owner of Mac builds/model runs. A component fix does not complete this objective.

The review already reproduced updater disconnection, stale export projection,
split identity storage, VAD partial-frame rejection and discarded EOF evidence,
cancel poisoning, hot-journal migration failure and bundle symlink bypass.
The review evidence is outside the checkout in the calling chat's review artifact.

## Environment constraints

Apple M5, macOS 27, Rust 1.99.0. Only Command Line Tools are installed; the Metal
compiler was absent at the initial build. The prebuilt MLX route below removes
that prerequisite. Xcode appeared and became selected during implementation;
its license is pending, so subsequent builds explicitly select the existing
Command Line Tools. No global toolchain settings or license acceptance changed.

## Foundation proof

- Core 123, host 47, xtask 14 tests passed; focused object-provenance export test
  added afterward passes. Core/host/xtask Clippy with warnings denied passed.
- New stale-export and bundle-link tests fail against the pre-fix production
  implementations. Object-provenance regression also failed before its fix.
- Disposable Cargo harness compiling the model component source without MLX:
  30 passed, six explicitly unimplemented acoustic parity tests ignored. This
  proves parser/protocol helpers only, not the normal models crate or inference.
- `cargo build --locked -p speakerdesk-models` reaches MLX native build and fails
  generating Metal JIT sources: xcrun Metal toolchain unavailable. No retry will
  resolve the source-build path without the toolchain. Prebuilt linkage below
  provides an alternative. Desktop, actual updater and package remain unverified.
- New synthetic FLAC fixture independently round-trips through afconvert. Saved
  reading-turn receipts originate from the baseline Python validator. Neither
  fixture contains user recordings.

## Data slice proof

Core 128 and host 51 tests pass; Clippy denies warnings. Identity through the
real router reaches ordinary GET/export, survives releasing and reopening the
store, and stale saves conflict. Tests cover protected late publication, private
response fields, exclusive native ownership and 0700 storage permissions.
Migration tests retain original database/journal/WAL bytes, preserve profile
version strings, recover a subprocess-created hot journal on a private copy,
and prove existing native data wins and publication cannot replace a destination.
Migrated audio normalization writes under native `derived/`; deleting its job
keeps legacy audio. All these fixtures are synthetic/replays, not inference proof.
The profile-version and hot-journal tests failed before their production fixes.
Independent verifier architecture_capture found private response leakage and
root permissions; both corrections are covered and verified in source.

fs4 1.1 no longer provides file locks; storage ownership uses its standard-library
successor, File::try_lock, available in the pinned Rust toolchain. Core ML, capture,
process lifecycle, actual editor/model integration and the release gates remain
open; data slice completion does not satisfy batch 2 or the replacement outcome.

## Process slice proof

One Tokio subprocess actor uses bounded portable pipes, an independent response
reader and a serial parent queue. Domain Models no longer owns process lifecycle.
Handles carry a child generation; protocol corruption, timeout and crash kill
and reap the owned child before lazy restart. Cancel waits for its targeted ack
and terminal result. A blocked partial write forces reaping; discarded successful
session creation/feed responses trigger bounded targeted cleanup. Explicit close
works after operation cancellation. The child keeps a control reader available
while a serial executor runs, and invalidates stateful work it cancels.

Job cancellation commits a new operation generation before signalling its token.
Retry and deletion stay blocked until the old worker relinquishes ownership.
The HTTP retry regression failed before the correction (202 rather than 409).
Core 128, host 46 and six real disposable subprocess tests pass; Clippy denies
warnings. Fixture-owned barriers exercise cancellation deterministically, including
ack-before-terminal, cancelled open/feed races, blocked stdin and unresponsive
termination. These subprocesses replay protocol responses, not acoustic inference.
The no-MLX component harness passes 33 tests with six unsupported acoustic tests
ignored. The production model forward passes and decoder cancellation boundaries
are still incomplete.

## Prebuilt MLX investigation

Official MLX 0.32.2 backend wheel, macOS 14 arm64, SHA256
`3825fff379dbc107dd3413e564a06caeaa24819910ec49c0439e454c06a1b9b8`, contains
libmlx.dylib, mlx.metallib, native headers and CMake configuration. The dylib links
only Apple/system libraries. A build-only mlx-sys patch uses mlx-c's existing
MLX_C_USE_SYSTEM_MLX option to compile unchanged upstream bindings with Command
Line Tools and link the prebuilt runtime. An optimized standalone Rust executable
evaluated GPU matmul with exact result [7,10,15,22]; no Python or Metal compiler ran.
This is linkage/initialization proof, not speech-model or performance acceptance.
Integration now pins the macOS 15 distribution (SHA256
`55a369250d220b2cf10213a87a2ac1b1a420608c5b35b1df4e7147ac8e32f121`) to match
Speakerdesk's minimum OS. The official mlx-sys source is vendored with only a
build.rs change; the C/C++ binding remains upstream code. The bootstrap downloads
and verifies the archive, then publishes under target/native. CI uses that path.

The actual models crate now builds and passes 34 tests (six acoustic placeholders
remain ignored). Its optimized metal_smoke example initializes Metal and evaluates
GPU math. A copy relocated to a disposable Contents/MacOS + Contents/Frameworks
layout runs with an empty environment; its loader trace resolves libmlx from that
copy and loads neither Python nor checkout libraries. Invalid Metal resources
fail initialization. This is a runtime probe, not the Speakerdesk app workflow.
The locked release desktop binary also builds with Command Line Tools; bundle
resources were disabled for this compile check, so it is not a qualified package.
The installed app was read only to establish version 0.6.6. PR/base remain the
starting SHAs above. Independent source review verified the ABI, artifact minimum
OS, unchanged bindings, and fetch boundary; its ignore-rule finding was fixed.
