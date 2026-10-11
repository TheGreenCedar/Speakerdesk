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
   tests. Mac CI added; local full build remains blocked by missing Metal.
   Independent verifier architecture_capture found three issues, all corrected.
2. In progress: data slice implemented and independently reviewed. Explicit-root
   AppState uses SQLite for jobs, identities and preferences, immediate edit
   transactions, guarded import publication, consistent migration snapshots,
   private journal recovery and legacy audio copy-on-write. Process lifecycle,
   operation cancellation and active meeting state integration remain pending.
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
compiler is absent. This blocks actual model/desktop build proof until the Metal
toolchain is available. Continue independent work; do not claim GPU acceptance.

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
  resolve this without installing/configuring the toolchain. Desktop, acceleration,
  actual updater, and package qualification remain unverified.
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
