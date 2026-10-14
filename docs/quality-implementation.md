# Reliability and workflow implementation

Accepted scope: implement the twelve-batch quality plan from the code review, starting
from `9e251a1e7760cc421a1d3f859ca300344e68b92b`. Branch:
`codex/reliability-and-workflow-quality`. No release or publication is part of this work.

| Batch | Scope | State |
| --- | --- | --- |
| 1 | Producer progress, retained late capture PCM, bounded capture queues | Complete; mounted host and native DSP checks pass |
| 2 | Async editor ownership, navigation, timing validation, export | Complete; frontend ownership checks pass |
| 3 | Pause acknowledgement recovery and passage-local live errors | Complete; core and mounted host checks pass |
| 4 | Import passage/language ownership, legacy compatibility, failure parity | Complete; core import and native migration checks pass |
| 5 | Durable retryable deletion | Complete; cleanup failure, restart and draft-copy suppression checks pass |
| 6 | Editing and save state during saved accuracy | Complete; revision-protected editing and final-repair completion checks pass |
| 7 | Import cancellation, stable controls, responsive actions and copy | Complete; frontend and rendered checks pass |
| 8 | Bootstrap/host recovery and host-backed theme | Complete; native callback, page identity, durable receipt and rendered checks pass |
| 9 | Metadata projection and selective startup recovery | Complete; store checks and local cost measurement pass |
| 10 | Durable progressive imports and bounded PCM reads | Complete; atomic publication, checkpoint resume and bounded audio checks pass |
| 11 | Live evidence retention | Complete; recent window plus canonical consumer lag, bounded witnesses and saved-crop recheck checks pass |
| 12 | Measured adaptive provisional scheduling | Complete; decode-cost sampling, cadence clamps and forced-flush checks pass |

Ownership: root integrates capture, lifecycle, store, host and documentation;
Sol 6.1 editor lane owns frontend and Node regressions; Sol 6.1 import lane owns
core import/audio and their regressions; Sol 6.1 live lane provides independent
review before taking an explicitly handed-off implementation seam.

Verification uses focused baseline-fail/fix-pass regressions, then required gates
on the integrated candidate. Independent goldens stay unchanged. Physical capture,
rendered UI and model-performance evidence are recorded separately from source and
fixture checks. No shared-source mutation during a proof run.

## Evidence

- Baseline import-to-saved-accuracy regression failed at the missing captured language
  setting, as expected (core import owner).
- Four frontend async ownership regressions failed on the original behavior and
  passed after repair and final integration.
- Import/audio owner checks: 31 import and 11 audio tests passed, including cancellation,
  stable checkpoint resume and bounded reader windows. Noncanonical decode spools
  float samples privately before bounded resampling instead of retaining whole PCM.
- Capture: 19 focused tests passed initially. The later full host library run passed
  84 tests, including combined late PCM/raw write failure and recovery receipt storage;
  three explicit opt-in tests were skipped in that run.
- Mounted native host/helper/model replay: delayed original PCM while ASR is held
  repairs saved source/playback, retains protected words/timing/names, invalidates
  stale acoustic claims and prevents the cancelled worker from publishing.
- Store: 10 focused tests passed, including metadata reads with deliberately invalid
  transcript/private payload and selective recovery, plus durable deletion intent.
- The metadata regression fails on an isolated copy of the starting commit because
  metadata GET parses the deliberately invalid transcript. The candidate store
  projection passes the same regression without reading that payload.
- The long-live retention regression failed on growing raw VAD evidence before the
  repair. The final candidate also passes continuous speech, delayed diarization,
  old protected continuations after incomplete private decoding, main-language
  retention across a long pause and ordinary model failure followed by healthy audio.
- Pinned MLX runtime prepared with the repository hash check. Pinned echo component
  built successfully from verified source; native raw/fault/queue checks pass.
- Desktop compilation uses the same resource override as CI. Initial native check
  found Rust 2021 let-chain syntax in host-disconnect callbacks; the callbacks were
  corrected without changing the desktop crate's edition.
- Initial rendered proof: shipped frontend with a disposable synthetic HTTP server
  and installed headless Chrome; nine behavioral checks pass. Wide/narrow light/dark
  text contrast measured 14.93–16.17, with no measured overlap/overflow or page errors.
  Screenshot inspection found a stale SVG recording indicator; corrected and
  rechecked on the final candidate. This does not qualify installed-app, physical
  capture or real models.

## Final qualification

- Integrated macOS Rust command: `cargo test --locked --features
  speakerdesk-host/fixtures -p speakerdesk-core -p speakerdesk-host -p
  speakerdesk-models -p speakerdesk-echo -p xtask -p speakerdesk`, with the CI
  `TAURI_CONFIG` resource override: **445 passed, 27 ignored, zero failures**.
  This includes 29 desktop tests, 216 core tests and 32 mounted live-meeting tests.
- Three additional opt-in native checks passed: raw dual-source partial pause tail,
  DSP failure endpoints across reopen, and native echo queue bounds/exact partial
  drain. The fault test used the repository's independently compiled fault library.
- Python 3.12 with pinned `pyyaml==6.0.3`: **63 tests, 62 passed, one skipped**.
  This includes executable Swift capture ordering and lifecycle checks.
- Full frontend suite: **180 tests, 179 passed, one skipped**. The final recovery
  subset also passed independently after the close-timeout correction.
- Rendered shipped frontend in disposable installed Chrome: **nine workflow checks**,
  four 1280/760-pixel light/dark layout measurements, contrast 14.93–16.17, no measured
  overflow or timing overlap, and zero page exceptions. Final report and inspected
  screenshots are in `/tmp/speakerdesk-reliability-final-rendered-proof`.
- Metadata measurement: 4,000 passages / 1,239,647 serialized transcript bytes;
  **100 full-document reads: 2.24468525 s; 100 metadata reads: 7.401542 ms**. This is
  one local debug-build comparison of the two read paths, not a production latency
  guarantee. The measurement is an explicit ignored test, run separately from load.
- Source boundary/version check and `git diff --check` pass. Independent golden
  reference files were not regenerated. This record accompanies the implementation
  checkpoint on `codex/reliability-and-workflow-quality`; no publication,
  installation or release was performed.

The integrated run also caught a real completion regression from sharing the
stricter refinement predicate: accepting the final blocked passage leaves no
machine-eligible rows. A verified accepted-repair continuation now finalizes that
empty plan without another ASR call; ordinary Resume still requires eligible work.
The existing mounted-host repair matrix passes unchanged.

## Evidence limits

The remaining opt-in tests require real model weights, independent model/DSP
references, special capture fixtures or physical devices. They were not executed.
No installed application workflow or physical microphone/system-audio session was
qualified. The adaptive scheduler was exercised with controlled model timing;
real-model throughput, end-to-end latency and process peak RSS remain unmeasured.

The raw live evidence window may extend to retain audio not yet consumed by delayed
diarization. Canonical transcript metadata still grows with meeting duration. Old
discarded crops receive fresh bounded VAD checks; this does not reproduce historical
recurrent-model probabilities. Abrupt app/OS loss recovers the last durable periodic
draft copy; normal close requires a final acknowledged copy. Recovery never applies
drafts to canonical text automatically.

## Overlap and first-attempt setup follow-up

The twelve-batch checkpoint is `c78f865`. The follow-up fixes the reported overlap
collapse and setup failure, and simplifies their user-facing controls.

- Saved evidence showed automatic separation replacing attributed original
  paragraphs with two generated whole-window rows, including an almost unchanged
  mixed channel and a much quieter sibling. Automatic processing now saves a
  candidate; only explicit acceptance replaces the original. Recognition of the
  generated voices runs after acceptance on the existing serial worker. This
  prevents silent publication of poor estimates; it does not improve the neural
  separator's accuracy. Original words, audio and restoration remain available.
- Growing live audio retains validated timing for unchanged words on their
  original prefix, then recalculates speaker attribution against current evidence.
  Changed words, protected edits and incompatible source ownership invalidate reuse.
- Passage controls say **Retry transcription** and **Edit full passage**. Comparison
  provides original/Voice A/Voice B playback; JSON diagnostics stay collapsed.
  Settings puts the separation model, its failure and its scoped retry before
  explanations. Start and transcript views expose separation readiness.
- The installed baseline reproduced separation setup activating its requested
  model, then failing because unrelated transcript timing was missing. Generic
  Retry started ordinary setup instead. The fix judges each request by its own
  target while reporting other capabilities separately, validates native separation
  initialization, and retains eight durable setup-attempt receipts. The update
  deleting weights was not established by this investigation.

### First-attempt native proof

Each qualified case used a fresh disposable root, one authenticated setup POST,
and no Retry. Installed application data and models were not modified.

| Case | Observed result |
| --- | --- |
| Installed baseline, other models present except timing, staged separation source | Failed after separation activation with `Transcript timing: No such file or directory` |
| Candidate, equivalent partially installed state | Separation activated and initialized; missing timing remained separately not ready |
| Candidate, empty model root | Downloaded the pinned 670,353,271-byte separation checkpoint over HTTP, prepared, activated and initialized it successfully |
| Candidate, ordinary setup with raw source files staged and no active model directories or prepared timing weights | All six models passed native initialization in 44.3 s; no ordinary network transfer was exercised |

The first ordinary debug probe reached timing preparation without a product error
but exceeded the harness's 300-second bound. A fresh probe passed after compiling
only the SHA-2 dependency with `profile.dev.package.sha2.opt-level=3`; application
logic and model pins were unchanged. This is a debug-build proof condition, not a
release performance claim. The failed baseline and bounded-out run remain in the
evidence record. All probe-owned hosts exited and their model trees were removed.

Evidence manifests are `/tmp/speakerdesk-firstattempt-proof-summary-20261013.json`
and `/tmp/speakerdesk-firstattempt-optimized-proof-summary-20261013.json`. They
record request/state receipts, durable SQLite history, logs, elapsed time and exact
binary/runtime hashes. The initial candidate binary SHA-256 is
`f84197a261c7e30c24832447d380f335d071086a687eca2c846962247578184a`;
the checksum-optimized candidate is
`280919a978f36ec0c3b8d37cfbb19a440e764955a9a81fd437ad5bd55cc922d5`.

### Follow-up qualification

- Integrated macOS Rust gate: **446 passed, 27 ignored, zero failures**, using the
  package/features/resource override above, four test threads and the same SHA-2
  optimization. Log: `/tmp/speakerdesk-overlap-setup-rust-final.log`. An earlier
  parallel fixture run timed out once in the existing rolling-accuracy matrix;
  its focused rerun and the final integrated gate passed without weakening it.
- Frontend: **184 passed, one existing skip**. Python: **62 passed, one existing
  skip**. Source boundary/version check passed; independent references unchanged.
- The overlap publication regression failed on the previous behavior. Disabling
  prefix-timing retention makes its independent CTC replay fail at missing speaker
  turns. The deferred-identity matrix covers cancellation, stale generations,
  edits, child loss, recognition changes and forgotten profiles.
- Rendered shipped frontend passed setup, paragraph preservation and playback
  checks at 1280 and 760 pixels with the full model catalog, no measured overflow
  and no page exceptions. Screenshots were inspected. Report:
  `/tmp/speakerdesk-overlap-setup-rendered/overlap-browser-report.json`.
- Sol 6.1's final adversarial review found no remaining blockers. The prior native
  DSP/capture evidence remains valid for unchanged code. No physical overlapping
  conversation, installed-app UI workflow, release package or publication was
  qualified by this follow-up; rendered checks used a disposable synthetic server.
