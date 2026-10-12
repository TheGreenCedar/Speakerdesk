# Python source removal

The current tree removes the legacy Python application/reference implementation,
its Flask/PyInstaller sidecar, dependency locks, setup launchers and dedicated
tests. Native application code, shared `speakerdesk/static/` and `templates/`,
model metadata, notices/licenses and all committed golden bytes remain intact.
The old filename port ledger and its CLI/CI check are removed; inventory counts
were never behavioral acceptance evidence.

## Test ownership

The removed Python suites exercised retired Python owners and their worker
protocols. The following native owners retain the corresponding product
contracts. These mappings identify continuing regression owners, not case-for-case
equivalence or complete native acceptance.

| Retired Python test family | Retained owner and representative proof |
| --- | --- |
| Flask imports, job polling, language preferences, People, transcript/review/export and loopback sidecar | `crates/host/src/routes/tests/`, `people_routes.rs`, `http.rs`; core `store/`, `people.rs`, `export.rs` and `review.rs` |
| Import speech bounds, tail context, language routing, silence/constant admission and source conditioning | Core `import.rs`, `language.rs`, `speech.rs`; `decode_regions_match_the_golden_file`, `crop_context_matches_the_golden_file`, `blocked_routes_separate_pending_uncertain_zero_and_constant` |
| Supplied-text timing, multilingual alignment, reading turns and speaker projection | Core `alignment.rs`, `timing.rs`; `independent_python_replays_preserve_text_offsets_aliases_and_cells`, `native_projection_survives_serialization_and_exports_reject_stale_or_forged_evidence`; retained alignment/export goldens and `crates/models/tests/ctc.rs` |
| Canonical/rolling/source runtime, live lifecycle, pause, retry, recovery and shutdown | Core `live.rs`; `crates/host/tests/live_meeting.rs` including exceptional recovery, editing open words, concurrent starts, saved candidates, live language and rolling accuracy matrices |
| Python voice workers, enrollment, readiness, profile policy and model setup | Core `voice.rs`, `people.rs`; host `people_routes/voice_http.rs`, `setup.rs` and `tests/process.rs`; `crates/models/tests/redimnet.rs` |
| Python capture mixer/echo/source processing | Host `capture/` and retained `capture-v066.json`; native echo and capture replay tests; Swift converter/route/startup/epoch tests remain in the Python tooling suite |
| Python-only receipts, reuse/process adapters, CoreML worker wrappers and policy implementation details | Retired with those Python owners. Historical D-references remain in `contracts.md`; no test-only implementation remains to satisfy them |
| Python-bundle acoustic acceptance and measurement supervisor | Retired with their runners. Native model tests retain separate timing/vector checks. They do not supply an automatic native release promotion gate |

`tests/test_voice_tools.py` keeps pinned-download integrity, bounded size/disk,
rollback and preservation checks. Its two `measure_voice.py` report/supervisor
tests retire with that obsolete Python model runner. Signer/updater trust tests,
artifact metadata, cache identities, workflow privilege checks and every
`*.test.cjs` frontend assertion remain. Native preparation regressions reject
mismatched QA and modified candidate bytes before any network call.

## Independent fixtures and retained Python

`tests/goldens/`, core/host testdata and model JSON references retain their exact
bytes, source hashes, schema evidence and historical function names. Native
tests read them directly. The oracle generator is removed; its reproduction
history is recorded in `tools/oracle/README.md`. Alignment, CTC and RedimNet
reference generators already take an explicit external source path; use a
separate checkout of their recorded baseline as described beside the fixtures.
Silero, Nemotron, Cohere and Whisper generators execute independently pinned
upstream code and remain unchanged.

Python remains for source/workflow guards, cache boundaries, artifact/notice
collection, pinned echo compilation and capture capability, independent fixture
generation/replay, the bounded standalone voice artifact downloader and numerical
reference tools. None is bundled as an application runtime. The voice corpus
generator now checks its own generated count/identity/hash bounds without
importing the removed application; native corpus tests continue to validate its
manifest before loading models. CI's only Python package dependency is PyYAML.

## Historical browser fixtures

All browser regression runners remain, including the unique autosave,
Save/Cancel/retry/export, icons, enrollment, timeline and draft-conflict checks.
Their Flask fixture servers and Python support workers are removed. Runners that
use those scenarios are labelled historical; they require the fixture server
from an external checkout of `7d94c8011a9501bd2357e07c73d61964c55a3886` and its
own documented environment. Run that checkout's paired server/browser commands
there. Their results qualify the tested historical fixture, not the current
native backend. No new native fixture server is introduced by this cleanup.

The current `frontend_dom.cjs` Node tests and the self-contained
`live_partition_browser.cjs` and `shortcut_hints_browser.cjs` remain executable
against the shared frontend without the old application.

## Release boundary

The deletion initially left `promote_release.py` failing closed because the old
gate admitted only Python bundles. The 0.7 release preparation subsequently
replaces that gate with final native-package QA and the retained authenticated
signer/updater checks; see [release acceptance](../core-acceptance.md). Candidate
metadata stays `public_ready: false` until the separate publication decision.
Deleting Python source alone neither qualifies nor publishes a native release.

## Cleanup verification

The removal was implemented by a third Sol 6.1 agent and independently checked
against `7d94c8011a9501bd2357e07c73d61964c55a3886`. Native runtime/desktop code,
shared UI assets, model metadata, notices/licenses and committed golden bytes are
unchanged by that deletion. The subsequent review-repair batch fixes migration
discovery and metadata/autosave, and consolidates packaged notices; see
[current delivery status](implementation.md).

On Apple M5/macOS 27 with Rust 1.99.0 and an isolated Python 3.13 environment:

- Locked core/host/xtask fixture tests: 287 passed, seven explicitly ignored.
- All 35 retained Python tooling/native-boundary tests passed, including Swift
  capture checks. Standalone Node tests: 112 passed, one existing optional skip.
- Source boundary, ownership, workflow guard, actionlint, Rust formatting and
  staged whitespace checks passed.
- The canonical optimized Mac build produced a Developer ID signed app/DMG.
  Strict signature verification and the bundle validator passed. The final app
  ZIP SHA-256 is `504e51ed94637b9a60d061c252fa64c19e795f417c6477cefef79616a7af850c`.
- Extracting that archive to disposable storage and running its native host with
  a sanitized environment served the embedded editor and empty job API, then
  shut down and was reaped normally. Its linked dependency inventory contains
  no Python runtime. This was service startup proof, without UI or model inference.

No acoustic/UI qualification was repeated for unchanged runtime code. Historical
browser fixture scripts were preserved and syntax-checked, not represented as
new native end-to-end tests. No installed app, production data or existing model
cache was changed by the cleanup.
