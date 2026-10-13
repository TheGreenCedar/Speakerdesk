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
