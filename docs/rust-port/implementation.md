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
3. In progress: Silero, the first offline Nemotron window and Cohere run through
   the actual native child and assembled import pipeline. An isolated Tauri
   preview now starts the Rust HTTP service and resident child. Actual HTTP
   import/edit/name/export/restart passes on synthetic audio (proof below).
   The actual editor import/edit/name/boundary/save/reopen/export/stale-save path
   passes, including Auto language and CTC. Short synthetic/public recordings have
   equivalent warm application measurements; long imports and full performance
   qualification remain pending.
4. In progress: automatic language, supplied-text CTC and native Core ML ReDimNet
   extraction work. Manual voice enrollment/checking now passes the actual editor
   and full restart (below). Automatic recognition and model setup remain pending.
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

Apple M5, macOS 27, Rust 1.99.0. Initially only Command Line Tools were available;
the Metal compiler was absent. The prebuilt MLX route below removes
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

## First model: Silero

The pinned 16 kHz graph now loads through mlx-rs and executes on Metal. The child
retains one network and independent raw/normalized recurrent states per session.
Feeds keep at most one partial 512-sample frame, report absolute received and
observed endpoints, and inspect EOF padding without extending physical time.
Cancellation checks occur between frames and invalidate partially advanced state.
The explicit child model root never falls back to the installed cache.

A generated 9.0903125-second two-voice fixture with a pause and 37-sample tail
matches the independent pinned Python graph on the same native backend within
1e-4. Against the development reference's MLX 0.32.3 macOS 26 backend, the largest probability
difference is 0.003350228, with no changed speech-hysteresis decisions. The macOS
15 and 26 wheel variants explain that numerical difference. On this M5, an
optimized warm native pass took about 164 ms versus the baseline component's
193 ms. These are component observations, not end-to-end acceptance measurements.
Reusing MLX's default GPU stream removed per-frame stream creation and its cost.

Actual Session tests cover packet sizes 1/511/512/513/16000, empty feeds, EOF,
interleaved sessions and targeted cancellation. Caller tests cover delayed
language boundaries, final-tail publication, partial-open cleanup, failed stop
and drop. Core 130, host 46, models 34 and six replay-process tests pass; five
other acoustic model placeholders remain ignored. Silero's ignored acoustic
test was explicitly run and passes. Clippy passes with warnings denied for these
crates; the vendored binding still emits its two upstream allocator-signature
warnings. An independent source verifier found no remaining Silero-slice defect.

The release desktop binary's actual `--inference` entry point passes the buffered
Silero pipe test and is reaped afterward. That test runs the built binary directly
with the pinned build library path; Cargo overrides DYLD_LIBRARY_PATH when running
the host-only harness, so the direct test executable was used. It is not packaged
app proof. Normal desktop startup still launches Python. Nemotron, Cohere, the
editor workflow, full model setup and all later acceptance gates remain open.

## First offline Nemotron window

The child now retains the pinned Nemotron network and serves the import caller's
`DiarizeWindow` request. Artifact hashes and all 363 tensor layouts are checked
once at load. Preprocessing, 31 BF16 transformer blocks and the subpixel speaker
head run through maintained MLX operations. Valid 10 ms frames remain separate
from padded feature and encoder shapes. Inputs beyond the first 27.2-second
offline window explicitly fail until AOSC/FIFO state is implemented.

The independent upstream reference generator is retained with source hashes.
On the nine-second fixture, all 909 by 8 probabilities match the same native
backend and both speaker turns match exactly. The independent verifier found
that mlx-rs's GELU promoted BF16 through float32 constants; a small compiled
expression with dtype-preserving constants fixes that. Removing unnecessary
per-layer evaluation reduced the optimized warm run from 54 ms to 12 ms;
the matching Python graph measured 16 ms. Model load including one SHA-256 pass
took about 349 ms. These measurements do not establish end-to-end performance.

The short/tail/pad-boundary, cancellation, nonfinite and over-window cases pass.
Host 46, models 34 and six subprocess replay tests pass; four other model
placeholders remain ignored. The release desktop child's actual pipe path passes
Silero plus this two-turn diarization and is reaped. Clippy passes for host/models
with project warnings denied. The same two upstream binding warnings remain.

Long diarization, streaming speaker identity, Cohere, the actual editor workflow
and later batches remain open. Cohere's pinned checkpoint has been downloaded
and checksum verified in disposable scratch storage for the next implementation.

## Cohere and the first assembled model import

The child retains the pinned 4-bit Cohere network. Rust composes its 48-layer
FastConformer encoder, 8-layer cached decoder and feature extraction through
maintained MLX and tokenizer operations. Weight layouts, valid frame masks,
mixed Float32/BF16 activations and decoder prompt match the independent upstream
reference. Decoder caches belong to one transcription, and cancellation is
checked between decoding steps. The 448-token cap retains partial text and sends
the actual token count to import/live review policy; a whitespace estimate lost
that warning. Its regression fails with the old estimate.

The independent generator and source/artifact hashes are retained. The nine-second
TTS fixture produces the same complete text and 31 tokens as the reference.
MLX's keyed dither has the same distribution but differs from NumPy's RNG;
bitwise feature equality is not claimed. Optimized warm native transcription
took about 272 ms versus 287 ms for the matching reference component. Native
load including artifact verification took 2.65 s. These are component observations,
not baseline end-to-end latency or memory qualification.

The actual desktop executable's inference child also runs Silero, Nemotron and
Cohere through core import and export. This assembled test exposed a missing
0.6.6 behavior: short VAD continuations past a diarization boundary decoded
separately and hallucinated “you”. Import now reuses the live canonical utterance
policy and extends only eligible single-speaker crops into adjacent unassigned
tails. Activity and candidate identities remain intact; without alignment the
straddling text is unassigned, reviewed and ineligible for voice enrollment.
Import/retry also preserve canonical pauses between 400 and 500 ms. Both failures
have regressions that fail before the correction. Planning reuses one utterance
book and scans only spans intersecting each crop.

Core 134, host 46 and six replay-process tests pass. The actual child test passes
in 4.07 s including direct model checks and the assembled import, yielding two
complete passages without standalone tails; this is not time-to-editor evidence.
The child is reaped. Cohere's explicit acoustic test was separately run and
passes; unavailable language/timing/voice paths remain explicit. Current input
limits are the first 27.2-second Nemotron window and bounded Cohere crops.
Normal desktop startup still uses Python; the existing editor is the next join.

## Native host and desktop join

An explicit `SPEAKERDESK_NATIVE_MODELS` directory selects the development native
host in the existing Tauri shell. It also requires an explicit `SPEAKERDESK_HOME`;
ordinary startup remains Python until the native product gates pass. The same
executable runs the Rust host and one resident inference child. The host embeds
the existing editor assets, serves ranged audio, persists language preferences,
and checks the three pinned models during actual initialization. Manual-language
readiness no longer depends on unfinished Auto or CTC. Preview setup reports
the 27.2-second import limit and unavailable live/language/alignment/voice work.

The existing bounded desktop updater control protocol now shares the host's real
admission owner. Request, preparation and inference leases cover their actual
lifetimes. Shutdown closes new admission, invalidates active operation generations,
cancels/reaps the child independently of slow HTTP clients, then drains workers.
A late completed-operation entry cannot reset a published job. Normal native
Tauri quit waits for the matching service generation's termination. The unused in-process bridge, global admission owner,
duplicate editor/preparation state and step receipts were deleted.

Independent review found that the name dialog always requested a missing
voice-clips route. Its 404 reproduced in the real router regression. The route
now returns the current revision and explicit unavailable voice state while
keeping naming usable. The reviewer also found shutdown ordering and test gaps;
the corrected process test requires the inference PID to be gone when a positive
`shutdown_ready` acknowledgement arrives, without a later grace period.

On this M5, the locked optimized desktop build passes with Command Line Tools.
Core 134, host 45, six inference-process replays, one native-service lifecycle
matrix and 17 editor-controller tests pass. The lifecycle matrix uses the real
host with a clearly labelled inference replay and covers stdin EOF, oversized
control, an upload stalled after HTTP 100 Continue, and updater reservation and
shutdown. Clippy passes with project warnings denied. Two unchanged upstream
mlx-sys allocator-signature warnings remain in desktop builds.

The disposable preview at `/private/tmp/speakerdesk-native-desktop.WJKAde/`
starts three native processes: Tauri, Rust host and the MLX inference child.
Launching through `/tmp` failed because Tauri rejects executable-path symlinks;
the canonical `/private/tmp` path works. The preview also needs an existing
Contents/Resources directory. It currently loads the pinned runtime from the
build directory and is not a qualified standalone package.

Actual HTTP calls to that app's service uploaded the synthetic two-voice fixture,
ran the models, changed words/speaker/boundary, advanced revisions 3→4→5, exported
TXT/SRT/VTT/JSON, and rejected the stale revision with 409. After terminating the
owned preview process, its host and inference child exited; relaunch preserved
the complete edited document and byte-identical exports. This is real service
and restart evidence, not UI interaction or a normal Quit-menu test. Results and
exports are retained under the disposable preview directory. The final preview
executable SHA-256 at that service-only checkpoint was
`8e16aacc29ea643b22bbe467996247b1507a63b0047ee7b7a5250cebe2e78402`.

## First native editor workflow

After the Mac was unlocked, the actual WebKit editor exposed a setup-polling
failure: initialized native models report zero download bytes and zero total
bytes. Assigning their quotient to HTMLProgressElement.value throws in WebKit,
preventing the subsequent config refresh and leaving Transcribe disabled.
The shared controller now uses zero progress when there is no download total.
The regression drives refreshSetup with WebKit's finite-value setter behavior;
it fails before the change and passes after it. All 18 focused controller tests
pass. The locked optimized desktop rebuild passed in 18.77 s.

The rebuilt disposable Tauri preview completed import of `two-voice.wav`, real
model transcription, playback, word editing, speaker assignment/naming and a
start-boundary edit to 0.10 s. Job `2b4ae227979644b28df177e667aca9ab` reached
revision 7. Normal Cmd-Q exited successfully and all three owned processes were
gone. Relaunching the whole app preserved the words, `UI Speaker One` and edited
boundary. Native Text export saved a 273-byte file identical to the service's
TXT export. A revision-6 PUT against that same job returned 409 and left the
document unchanged; this conflict check was HTTP, not a stale-save UI action.
The other three export formats retain their earlier HTTP evidence, not native
save-dialog coverage.

The preview executable SHA-256 for this UI run is
`d6d0d5b223f544b2f9e62f8431808a8b41accd8085a315d6c99d4b07365c534a`.
Its source is `fc324c0` plus the finite-progress fix; edited data, text export and
receipt remain in `/private/tmp/speakerdesk-native-desktop.WJKAde/`.
The nine-second fixture is synthetic speech: this establishes product integration,
not broad acoustic quality. The preview still loads MLX from the build directory.

Public-audio quality, equivalent optimized 0.6.6 latency/backlog/memory
measurements, the remaining models, native capture and final packaging remain
open. The copied installed 0.6.6 runtime reports MLX 0.32.2, matching its packaging
lock; the separate development lockfiles specify 0.32.3. Baseline explicit-English
imports still require alignment, so omitting alignment permits only a labelled
shared-stage diagnostic, not full product performance acceptance. Small numerical
drift is acceptable; behavioral and measured performance gates are unchanged.

## Shared import-stage measurements

On the M5/24 GiB/macOS 27 machine, release builds and the copied installed 0.6.6
worker binary ran sequentially on disposable model/data copies. The baseline
runtime SHA-256 is `a2159e95ed48dbf9f54b76dc07c25a02856115946f8e9f0efeabe8e917f64c61`.
Its actual capability output confirms MLX 0.32.2, mlx-audio 0.5.7 and mlx-speech
0.5.3. Exact-base Python crop planning supplied the frozen workers; alignment
was omitted. Native measurements used the actual release HTTP host, preparation,
resident models and SQLite publication. Neither timing includes Tauri rendering.
Baseline memory covers worker trees, not its absent application/planning host.
These boundaries prevent treating the results as complete product parity.

The second fixture is the 7.06-second LibriSpeech sample 1688-142285-0007 from
[the TorchAudio tutorial](https://docs.pytorch.org/audio/2.8.0/tutorials/asr_inference_with_ctc_decoder_tutorial.html),
converted once with afconvert to canonical PCM16 and supplied identically to
both implementations. Download SHA-256:
`67dfc27041e572bea27936e75eedc542ccf6f3af5cc251a0ab4e1e947d787fc8`;
PCM WAV SHA-256: `0ef932371d181b185f01b2ede213ebe649650457e5781d8e428f831dfbbe5343`.
Both implementations returned the reference words exactly after case and
punctuation normalization. This one public clip is an acoustic sanity check,
not a quality benchmark or overlap/separation evidence.

| Observation | Two synthetic voices, 9.09 s | Public speech, 7.06 s |
|---|---:|---:|
| 0.6.6 shared workers, including child startup/load | 9.73–10.45 s | 11.10 s |
| Native process startup to initialized models | 3.08 s | 3.53 s |
| Native first import to persisted document | 0.81 s | 0.65 s |
| Native second import to persisted document | 0.61 s | 0.49 s |
| Native observed physical-footprint upper bound | 2.071 GB | 2.085 GB |
| Baseline observed ASR-worker footprint upper bound | 1.897 GB | 1.932 GB |

Footprint uses macOS libproc v4 physical footprint, which includes Metal
allocations. The upper bounds sum each owned process's last observed lifetime
peak; they are not necessarily simultaneous. Sampling is process traversal plus
a nominal 50 ms sleep and can miss the final interval before exit. RSS alone
varied substantially and is not used to claim memory parity. The synthetic
baseline timings precede a harness bookkeeping correction of at most a sampling
interval per worker; the public result excludes those joins. One public baseline
run overlapped compilation and was discarded from latency reporting.

The original native allocator cache retained intermediates between resident
requests. Matching the baseline's 256 MiB cache cap reduced public-clip footprint
from about 2.41 GB to 2.16 GB; disabling only the released-buffer cache reduced
it further to about 2.08 GB. Weights and model sessions remain resident. This
policy now runs once at child initialization and propagates errors. All four
post-change imports returned identical words, times and speakers to the uncapped
candidate. The final optimized binary SHA-256 is
`c924bd730760d0c2cb25f7efa31fc39e8b791cadf02619b3e8be2ec9307ef682`.
Build and acoustic-service checks pass; no live-throughput claim follows.

The measurement script, raw outputs and stores are retained under
`/private/tmp/speakerdesk-shared-stages-aqno7nex/`. Independent review corrected
its startup deadline, exceptional process-group cleanup and asymmetric timing
bookkeeping. The remaining native-versus-worker memory difference is unresolved;
full application memory acceptance must include equivalent alignment, UI and
live work. This measurement is sufficient to proceed from the working first
import to language and timing integration, not to declare the replacement done.

## Automatic language through the native editor

Whisper-tiny now runs in Rust through the resident MLX child. The pinned FP16
encoder and one-SOT decoder step score all 99 language tokens. Only the SOT and
language embedding rows stay resident; transcript generation remains Cohere's
job. Each probe has local activations and cancellation checks between layers.
Weights and support files are verified once per load. Raw input is padded or
trimmed to 30 seconds before the centered Hann400/hop160 Slaney-mel frontend.

On the M5, four acoustic cases and two independent calls matched every pinned
reference language score within `1.2e-7`; 50 mel points per case passed `1e-4`.
The reference is unchanged upstream Python graph code executed independently
with the same MLX 0.32.2 native backend. These are numerical fixtures, not proof
of multilingual acoustic quality. The reference generator and sampled results
are under `crates/models/tests/data/`; logs are
`/tmp/speakerdesk-whisper-numerical.log` and
`/tmp/speakerdesk-whisper-features.log`.

The Auto join exposed a publication defect: merged language windows lost all
their original probe decisions when converted to the saved document. Publication
now retains a flat list of absolute spans, selected languages, decisions and
review flags. A two-probe replay with distinct confident/uncertain decisions
failed before the fix and passes after import, SQLite close and reopen. It is an
orchestration replay, not an acoustic assertion.

The language model is optional for manual imports. Both readiness endpoints and
the server-side Auto run guard require its successful initialized checkpoint
identity. The HTTP matrix covers an unknown identity, success and failure;
manual admission still succeeds after Auto failure. A separate real native host
started with the three-model directory and no Whisper: core readiness remained
true, Auto remained false, and shutdown reaped the child. Its disposable output
is `/var/folders/8s/9n6rxkrx6dv46zhjfnhg69rw0000gn/T/speakerdesk-optional-whisper-rmil3yvl/`.

The optimized desktop was built with the same packaging-resource override as
CI; its binary SHA-256 is
`4602bd9c4411b2edf51a5421bf058997bbe41ad4ec18f951a95b3e49190ffce6`.
In the existing disposable native app, the actual UI workflow passed:

1. Select Automatic in Settings; observe the initialized language model.
2. Import `two-voice.wav` through the native picker and transcribe.
3. Observe two editable passages, both detected as English; edit and save words.
4. Quit the entire application and verify all three owned PIDs exited.
5. Reopen the same store, observe the edited words, and export a JSON project
   through the native Save dialog.

Job `9a478a9463224475881bf95a9e8ac8a0`, revision 4, retained four original language
probes and its complete document across restart. Exported segments equal the
saved document's segments. The Auto default also persisted. The app used PIDs
16552/16567/16568, then 18026/18040/18041; both complete process trees were reaped
after normal quit. Raw edited/reopened JSON and native export are retained beside
the earlier UI evidence in `/private/tmp/speakerdesk-native-desktop.WJKAde/`.
JSON export SHA-256:
`cb3fd6d51c78b7f2470296c95f363e35571c1606f86991c467ecda2021a3fc01`.

Core 135, host 46 and tooling 14 tests pass, as do six disposable child lifecycle
replays and the four-case native-host lifecycle matrix. Clippy passes with
warnings denied for the application crates; upstream mlx-sys retains its two
allocator-signature warnings. Independent review found no further defect in
this slice. This extends the editable import milestone; alignment, voice,
language repair/retry integration, live capture, packaged independence and
equivalent full-application performance remain unfinished.

## Supplied-text alignment through native publication

The pinned Omnilingual CTC graph now runs in the resident MLX child. Rust converts
the pinned ONNX tensors once into safetensors and loads them through maintained
loaders. The fixed graph retains weights; input crops, activations and cancellation
checks belong to each call. Full-vocabulary log-softmax precedes column selection.
The result carries actual valid frame count, source sample count, frame hop/origin,
token IDs and model/vocabulary identities. Padding never becomes observed audio.

The parent owns character-to-token mapping, conservative Unicode spans, supplied-text
alignment and publication. It preserves authoritative text and its source crop.
Weak or unsupported characters remain unresolved. Projected reading turns pass the
existing freshness validator, including trusted native provider identity at export.
The unwired live scaffold no longer invents word-ordinal token IDs or calls CTC and
discards its result. Live timing is explicitly unavailable until the live join lands.

Independent reference generation executes the unchanged Python CTC graph from the
original NPZ weights with the same MLX 0.32.2 backend. Two calls on six inputs cover
the full two-voice fixture, minimum context, partial final frames and silence;
permuted vocabulary columns match at F32 precision. Separately, ctc-segmentation
1.7.4 and the unchanged Python text adapter generate hostile supplied-text replays.
Review caught excessive selected probability mass and reordered combining-mark span
drift; both now have boundary tests. Fixtures are numerical/replay evidence, not a
multilingual acoustic benchmark.

The optimized native Tauri preview loaded all five models and published two aligned
passages from Auto import through its real HTTP/store path. TXT/SRT/VTT/JSON exports
passed; JSON segments equal the persisted document. In the actual editor, job
`56151c37d75b47349dd6029ab7134a56` was edited to change words, name a speaker,
assign the passage and move its start to 100 ms. A full quit reaped the Tauri,
host and inference PIDs (42917/42932/42933). Reopening preserved the entire
document at revision 7. The native Save dialog's `ctc-ui-reopened.json` matches
SQLite; TXT/SRT/VTT use the protected words and boundary instead of stale CTC
projections. A revision-6 PUT returns 409 and leaves revision 7 unchanged.
Edited/reopened JSON and exports are retained in
`/private/tmp/speakerdesk-native-desktop.WJKAde/`. The tested preview executable
SHA-256 is `849233493374cdd1f4b20bdc7cda260bd25bcd2a4d023430ac97e9619d569f0c`;
it contains the CTC path but not the subsequent native Core ML adapter.

For a closer comparison, an APFS copy of installed 0.6.6 runs under its own bundle
identifier and disposable storage. Both previews use the same pinned artifacts,
Auto selection and alignment. This is ready-app upload to persisted HTTP result;
it includes polling latency and excludes cold startup and editor rendering.

| Observation | Two synthetic voices, 9.09 s | Public speech, 7.06 s |
|---|---:|---:|
| Native import to persisted result | 1.25 s | 1.07 s |
| Copied 0.6.6 import to persisted result | 3.42 s | 3.05 s |
| Native sampled aggregate physical footprint | 3.44 GB | 3.33 GB |
| Copied 0.6.6 sampled aggregate physical footprint | 3.94 GB | 3.94 GB |

Both clips have identical words, source PCM hashes/anchors, activity and reading
turns across implementations. Sampling includes the Tauri root, descendants and
positive WebKit service PIDs from that exact app's launchd domain, refreshed and
deduplicated each sample. Maximum intervals were 0.11 s native and 0.22 s baseline;
these are simultaneous sampled sums, not sums of process lifetime peaks. Startup
and between-sample peaks remain unmeasured. The public run's sampled CPU lower
bounds are 0.77 s native and 2.57 s baseline; exited workers can lose their final
interval. Mach ticks use the observed 125/3 ns timebase, cross-checked against
getrusage. The earlier two-voice raw CPU fields predate this conversion correction
and must not be read as seconds. Retained scripts/results:
`/private/tmp/speakerdesk-baseline-ctc-wc5si6wc/`.

A separate ignored test captured a complete 720-sample CTC forward pass to a real Metal
GPU trace using the already-pinned MLX C capture bindings. The system capture
metadata records one frame; traced/untraced scores match. No Metal compiler was
required. Trace: `/private/tmp/speakerdesk-ctc-metal-20261011.gputrace` (1.47 GB).
It has not been opened/replayed in Xcode, and trace overhead is excluded above.

Core 140, host 46, tooling 14 and five CTC unit tests pass. Six child lifecycle
replays and the four-case native-host cleanup matrix pass; the separate Metal
capture test passes. Application Clippy passes with warnings denied, apart from
the previously recorded upstream mlx-sys allocator signatures. This slice does
not complete voice, model setup, retry/language repair, long imports, live capture,
packaging or full cold/warm/live performance acceptance.

## Native Core ML voice extraction

ReDimNet now loads the pinned compiled B6 artifact through `objc2-core-ml`.
Waveform preparation, quality screening, normalization and model identity remain
Rust-owned. The model stays on its construction thread; the child control reader
runs separately so cancellation remains serviceable without an unsafe `Send`
implementation. The result carries both the embedding and its complete model
identity over ordinary postcard IPC. Setup reports the actual initialized policy.

Artifact validation makes one load-time pass over the eight pinned files and the
compiled tree. Tests reject linked roots/directories/files, extra compiled files
or directories, and same-size byte corruption. No per-prediction artifact rereads
or qualification receipts were added. Cancellation is checked around the bounded
synchronous Core ML call; the existing parent timeout remains the escape path.

An initial direct `MLMultiArray.dataPointer` copy passed ordinary predictions but
failed under Metal capture (cosine -0.0199 against the ordinary output). The adapter
now writes through Core ML's scoped mutable-buffer API and validates that callback's
capacity and strides. Ordinary, traced and subsequent predictions then agree
(cosine 1.0). The passing trace is
`/private/tmp/speakerdesk-redimnet-scoped-metal-20261011.gputrace`, one captured
frame, 3.18 GB. Xcode replay remains unperformed; capture overhead is excluded below.

Optimized policy tests ran separately on the M5/macOS 27 host, on the same two
synthetic speaker crops, three calls per crop. Neither test includes application
startup, publication or a live capture workload. Peak physical footprint is the
process value from `/usr/bin/time -l`, not an aggregate system or service peak.

| Core ML policy | Model load | Warm extraction | Peak process footprint | Minimum cosine vs legacy MLX |
|---|---:|---:|---:|---:|
| CPU_AND_GPU | 0.345 s | 29.8–30.0 ms | 375 MB | 0.9999986 |
| ALL | 5.713 s | 119.8–122.8 ms | 144 MB | 0.9999931 |

Both policies reproduce the independent baseline MLX vectors and reject invalid,
silent or saturated inputs as expected. GPU remains the integration default based
on its lower latency; ALL's lower memory footprint remains relevant to the final
whole-application memory gate. No ANE placement claim follows from requesting ALL.
Outputs/logs are retained as `native-coreml-scoped-{gpu,all}.json` under
`/private/tmp/speakerdesk-model-qualification/` and
`/tmp/speakerdesk-coreml-scoped-{gpu,all}.log`.

The actual optimized desktop child loads the model, reports its initialized
identity, returns a reference-matching embedding over IPC, survives invalid input
and silence, and is reaped on shutdown. This is extraction and process-boundary
evidence. Native enrollment/recognition, calibration and broader saved-profile
compatibility remain unfinished; HTTP voice readiness stays unavailable and the
native embedding identity remains distinct from legacy MLX/ALL identities.

Core 140, host 46, model 36, six child lifecycle replays and the native-host
cleanup matrix pass. The separate real-child voice and artifact-boundary tests
pass. Clippy with warnings denied includes the desktop this time; upstream
mlx-sys still emits its two previously documented allocator-signature warnings.
Independent review found no remaining blocker in this bounded extraction slice.

The original compact-voice synthetic corpus was regenerated in disposable storage:
72 clips in 36 groups, manifest SHA256
`4a6eb0102d420587fba74fa0047db89f0ec17dcaeb75da79eb1c6144562d44a7`.
The independent test asserts that digest before model loading and uses the existing
`voice_calibration.json` threshold (0.7264125328398843) and margin
(0.10003429964182987) without fitting. Both policies pass all 28 recognition
decisions across the calibration and held-out groups, including unseen synthetic
voices. Median extraction is 30.4 ms for CPU_AND_GPU and 130.1 ms for ALL. Results:
`/private/tmp/speakerdesk-model-qualification/native-frozen-voice-policy.json`.
This is synthetic-policy evidence, not human population accuracy or permission to
relabel legacy profile identities.

Voice crop selection now understands native import activity and excludes its
unassigned tail without discarding separately verified solo audio. Whole-document
saves preserve the server-owned decode anchor; client omission/forgery cannot
replace it. Text-only edits keep the crop while stale activity, competing owners,
changed speaker/bounds or incomplete observations reject it. The save regression
failed before the fix; all 142 core tests pass afterward. HTTP enrollment remains
the next slice.

## Voice editor integration

The production router uses bounded saved-audio crops and the resident native child
for consented enrollment and manual voice suggestions. SQLite owns profiles and
monotonic person generations: forget/replacement invalidates in-flight enrollment,
including the no-existing-profile case. Publication reloads the job and checks its
operation, assignment and selected audio; unrelated text edits survive. Inference
and crop I/O hold no database lock. Cancellation/shutdown retain admission until
the worker terminates. Restart cancels orphaned checking states once.

The first actual UI attempt caught a gap missed by component tests: imported solo
speech bridged across a short pause retained activity but could not enroll unless
word alignment had produced reading turns. Import now publishes speaker observations
independently of optional alignment. The import-to-voice regression failed before
the correction; Auto/English replay plus actual passage-edit handlers prove crop
preservation for text edits and invalidation for speaker/boundary edits.

On M5/macOS 27, the optimized isolated Tauri app imported and transcribed the frozen
synthetic Daniel enrollment file, saved a new person, previewed the second crop at
5.14 seconds, and enrolled its two solo crops. Whole-app quit reaped the three owned
processes (34418/34453/34454). After restart, People showed the same saved compatible
voice. Importing different held-out words produced a manual "Possible Daniel fixture"
suggestion (score 0.9643743364); confirming it updated the editor and its actual text
export. No person was assigned before confirmation. This proves synthetic product
integration, not human-population recognition accuracy or automatic checks.

Core 144 and host 48 tests pass; strict all-target core/host Clippy and locked release
desktop build pass. The blocked-inference HTTP matrix covers text/boundary edits,
forget/replacement, cancellation/shutdown, failures and preference changes. Tests use
explicit model doubles; the desktop path above used real MLX/Core ML inference.
Independent read-only verification found no remaining voice-slice source blocker.
Release executable SHA256: `d93e2474069c47889ee70fb5dbd2a5b1f9096d0d4f67bb9eae667cb1f54903f4`.
The ad-hoc signed preview executable hash is
`4e77503943ee84dfad8dd19377a81956eb5c6b860f0ca355852b7da485ed104c`.
Evidence under `/private/tmp/speakerdesk-native-desktop.WJKAde/`:
`voice-people-{before-restart,reopened}.json`, `voice-enrolled-job.json`,
`voice-heldout-{suggested,confirmed}.json`, and `voice-ui-recognized.txt`.

The same recordings exposed another assembled-path defect: 21–26 ms final tails
after a passage with an internal pause were decoded separately and hallucinated
short text. The earlier tail correction only handled unbridged diarization regions.
This remains a blocking import-correctness task for the next slice. Ordinary
non-tail passages also still lack supplied-text alignment. Live capture, long
imports, automatic voice checks, packaged independence and the full performance
gate remain unfinished; this slice does not complete the replacement.

## Bridged tails and ordinary passage timing

Import now joins a final unassigned tail to a solo passage whose complete activity
ledger contains only that speaker and silence. It retains internal pauses and
appends the tail as unassigned; incomplete, competing or nonadjacent activity still
rejects the join. Ordinary passages publish their observed audio independently of
alignment and run supplied-text CTC on their displayed bounds. Timing provenance
describes the available projections and canonical fallback accurately.

The two regressions fail before the change: the bridged Daniel replay produces two
passages, and an ordinary supplied-text replay has no reading turns. Core 145 and
host 48 tests, strict core/host Clippy and the locked optimized desktop build pass.
Independent review found no blocker. In the actual isolated editor, importing the
same Daniel enrollment WAV now produces one aligned passage through the final
11.0315625-second sample, without the previous hallucinated "Oh." Its two verified
voice crops and internal pause remain intact. This is real MLX inference; the CTC
boundary test separately uses an independently generated numerical replay.

The current ready app imports and persists the unchanged two-voice fixture in
1.386 seconds with 3.546 GB sampled aggregate physical footprint, versus the
retained equivalent 0.6.6 run's 3.422 seconds and 3.935 GB. Words and reading turns
are unchanged. The same sampler includes owned descendants and WebKit; maximum
interval is 0.144 seconds. This checks warm HTTP publication, not cold startup,
UI rendering latency or sustained live performance. Evidence:
`/private/tmp/speakerdesk-native-voice-performance/native-two-voice-1791756542556898000/`
and `tail-timing-{fixed,clips}.json` in the disposable desktop directory.
Release executable SHA256:
`d07486e83c078c35549cc4ebe6b668220e2b9cda6405261fb6bd365afc616ff2`;
signed preview SHA256:
`f048ff64952774129eb7ce43ea7394e1dbe058da675e6cb2a9b2038c18ef0406`.
The remaining replacement gates listed above are still open.

## Automatic saved-voice checks after import

Successful imports now publish an editable transcript, then check eligible tracks
under the same serial worker, operation generation and admission lease. Existing
core planning/matching rules retain one decision per track, compatible profile
identity, opt-out and conflicting-introduction behavior. Publication reloads the
document, preserves unrelated edits/metadata and merges only the target decision.
Names remain tentative (`automatic_voice`, `confirmed: false`); recognition never
enrolls or updates a voice. Ready jobs report ongoing inference ownership until
cleanup, keeping deletion and updater admission consistent.

The assembled import replay failed before integration with zero embedding calls.
Five eligibility cases and an eleven-case blocked, two-track matrix now pass,
including edits, names, forget/replacement, opt-out toggling, cancellation,
shutdown, malformed model identity and truncated audio. Review caught an overly
broad profile guard: forgetting an unrelated incompatible profile cancelled the
second track. The regression reproduces two calls instead of four before the
compatible-only fix, then checks both matched identities afterward. These tests
use explicit model doubles, not acoustic evidence.

Real MLX/Core ML inference in the isolated optimized Tauri app recognized the
held-out Daniel fixture at score 0.9643743364. Its two verified crops were
0–5.99 s and 6.24–12.12 s. A local name correction persisted at revision 6 through
whole-app quit/reopen; the reopened API document equals its saved predecessor.
Quit reaped owned PIDs 57333/57366/57367. The profile remained unchanged.

The real editor also exposed a presentation gap: a canonical unassigned passage
could have an attributed reading turn, hiding the existing recognition marker.
Reading turns now show their own recognized status and remove it upon confirmation.
The final preview's fresh import shows `Daniel fixture Recognized`; applying the
saved person removes the marker without changing the transcript or saved profile.
The renderer regression fails before this change and passes afterward. Its cache
key tracks assignment sources, avoiding serialization of profile evidence per row.

Core 145, host 50 and 34 renderer/performance checks pass; one optional external
reading fixture is skipped. Strict core/host Clippy and the locked release build
pass, with the previously recorded upstream mlx-sys warnings. Independent review
found no remaining slice blocker. Final release SHA256:
`a7fa550f5a2088e34356a3f0ff00a28750a201ab925262e0007d86288f696306`;
signed preview SHA256:
`78aa479b8f30a404fe205db45d953600a39e60738dcdb540e5fa0ffbf0b77277`.
Evidence is `automatic-voice-*.json` under the disposable desktop directory.

Remaining acceptance: native passage retry/language repair, long-file diarization,
model download/setup, live capture/recovery, Python-free default packaging/updater,
and full cold/warm/live performance gates. Synthetic voice evidence does not
establish human-population accuracy or compatibility with legacy embedding spaces.

## Passage retry in the native editor

The editor's Repair action now reaches `POST /api/jobs/{id}/retry` through the
existing authenticated router. A ready transcript stays editable while the same
serial worker reads only the selected WAV range and runs resident Silero/Cohere
with an explicit language. The result is a candidate; only “Use this text” changes
the words. Publication checks the operation generation and target passage/audio
snapshot, then merges into the fresh document. Unrelated edits survive. Cancellation
invalidates publication before notifying inference and retains admission through
cleanup. Pending retries recover once; completed legacy `passage_retry: null`
records remain unchanged.

Before this change the actual editor displayed “The string did not match the
expected pattern” because the route was missing. The HTTP regression reproduces
404 before implementation. File-backed router replays cover exact nonzero/fractional
crop samples, unrelated saves, target text/bounds/speaker/language/audio changes,
queued and active cancellation, shutdown, model failure, explicit acceptance,
stale saves, and restart. These are retained-output tests, not acoustic proof.
CTC metadata is now restored from the server and cleared when words or bounds
change. The renderer includes retry ownership in its cache key and offers Cancel.

Independent review found two further failures and verified their corrections:
legacy null retry records were misclassified as interrupted, and candidate
publication could advance revision before dirty text autosaved. A controller test
reproduces the latter 409 before the fix. A single guarded retry compares the last
server document with the current one, excluding only candidate metadata; actual
word, speaker, timing or name changes still conflict. Newer local typing survives.

The optimized isolated Tauri app ran a real retry on saved synthetic Daniel audio.
It retained a deliberate placeholder until the user action selected the model
candidate. Revision 10 preserved the placeholder as `replaced_text`, recorded
manual passage language, and removed stale alignment. Whole-app quit reaped
PIDs 82205/82222/82223. Reopen returned the identical saved document and revision;
the native Save dialog exported the accepted words, and a revision-9 PUT returned
409 without changing them. Evidence: `retry-ui-{before,candidate,accepted,reopened,
stale}.json` and `retry-ui-export.txt` under
`/private/tmp/speakerdesk-native-desktop.WJKAde/`.

Core 145 and host 52 tests pass. Renderer checks pass 45 with one optional external
fixture skipped. Strict core/host Clippy and the locked release build pass (the
previously recorded upstream mlx-sys warnings remain). Release SHA256:
`20f949a712067cee240079654cf3cf2f5c60f71a14cf97f9b4e0e95c3b114ef9`;
ad-hoc signed preview SHA256:
`39e9d324f70d3356536dc2cf177acf4323c63c08d8e6987bde2d44ca36f7f392`.

The complete replacement remains unfinished. Next work is long-file diarization,
model download/setup, live capture and saved-audio recovery, Python-free default
packaging/updater, and the full cold/warm/live performance gates. The installed
application, production data and existing model cache remain untouched.

## Stateful long-recording diarization

PR state was refreshed during this slice: head
`db9da655b3eeeeef598ba21a21cb54505e6bce21`, base
`59e7b0c860f72ca94c24e6eab18ac820b144ee24`. The fetched PR head matches the local
starting commit. This slice has not been pushed or published by this run.

Nemotron now retains per-recording AOSC/FIFO evidence and bounded waveform history
across windows. Low and Offline presets are session-owned; weights remain resident
in the serial child. Offline import uses bounded open/feed/final/close calls.
Unused whole-recording and child-file diarization requests were removed. Absolute
sample turns carry separate received/observed endpoints; failed feeds invalidate
only their own session. The 27.2-second host guard is removed.

The independent 82.0099375-second fixture matches all 8,200 native probability
frames exactly and retains identical speaker decisions under irregular packet
splits. It repeats the existing synthetic two-voice recording and ends with a
159-sample partial frame (WAV SHA256
`3297950fe04a7a7310436f2afaa2f6a0f49163c6c5a176871a3337a90d70f6a8`). This is window,
cache and tail evidence, not returning-speaker acoustic quality: the upstream
long-context model groups this particular repeated fixture into one track.

The first comparison exposed two numerical differences. Apple's macOS 26 and
macOS 15 MLX wheels both report 0.32.2 but contain different native kernels.
The reference now records the actual loaded native-library and Metal-resource
hashes. With the same pinned macOS 15 runtime, the first two windows matched.
Replacing `square` with the pinned power operation removed the remaining
rounding differences and later cache-selection drift. No probability tolerance
was relaxed. Separate module globals in the reference loader also fix a latent
Sortformer helper collision reached only by cache compression.

A separate pinned Low reference passes with interleaved Low/Offline sessions and
cancellation of one session, followed by a clean recording. The actual optimized
desktop child passes the long fixture through the bounded pipe protocol while a
Low session remains independent; invalid feeds remove their handles, close is
idempotent, and shutdown reaps the child. The direct test runner is required here:
Cargo replaces `DYLD_LIBRARY_PATH`, so the first attempt failed in dyld before
handshake and ran no inference. The actual-child pass used the explicit pinned
native-library directory. This is not packaged-runtime independence proof.

Live replays exposed sealing ahead of delayed speaker observations and discarded
EOF results. Both are corrected. A non-hop-aligned language boundary reproduced
a panic (`16037 > 16000`) in an empty activity range; it now publishes unknown
ownership and cleans up. Import and live metadata preserve the final unobserved
suffix. Timing and voice validators retain usable evidence inside the observed
prefix while rejecting gaps, competing owners, edited bounds and evidence past
that endpoint; incomplete speaker observations no longer discard earlier valid
word timing or enrollment crops.

Validation: 148 core tests, 52 host tests, one native-host lifecycle replay and
seven process replays pass, including a recording larger than the 64 MiB IPC frame,
final-only output, cancellation races, failed-feed cleanup, crash/restart and
malformed framing. Three ignored Metal/Nemotron qualification tests were run and
pass. Strict core/host/models Clippy and locked release build pass, with the
previous upstream mlx-sys malloc/realloc warnings. Release executable SHA256:
`0e85c1db40f9c85cf52c73341f536121b0591ba71c5ecf3e88581cf178f46bc5`.
Independent read-only review found no remaining source-slice blocker.

Evidence: `/private/tmp/speakerdesk-nemotron-long/` contains reference/runtime
identities, numerical diagnostics, partition/Low tests, actual-child evidence and
before/after live regressions. Suite/lint/build logs are `/tmp/speakerdesk-diar-*`.
The desktop UI check is pending Mac unlock; the existing preview still runs the
previous retry build. Long-import memory/performance parity is unqualified: the
parent still owns the complete imported PCM. No installed app, production store,
recordings or existing model cache were changed.

The accepted complete replacement remains unfinished. Continue with native model
setup/downloads, finish this candidate's editor check, then live capture/drain and
saved-audio recovery, Python-free default packaging/updater, security qualification
and full cold/warm/live performance comparisons against 0.6.6. Windows and issue
#25 remain follow-up scope. A source checkpoint or successful model slice does
not replace those acceptance criteria.

## Native model setup and repair

The native settings path now downloads the pinned artifact catalog, resumes partial
HTTP downloads, verifies every consumed file, prepares CTC weights in the resident
child, and activates complete model directories. Core ML is fetched as individual
pinned files, so setup has no archive extraction path. A private fs4 owner prevents
competing installers; links and multiply linked writable files are rejected.
The Mac uses atomic directory exchange for repair. Unknown inventory marks an
installation incomplete; interrupted swapped staging is rebuilt safely on retry.
Disk admission uses missing artifact and conversion bytes instead of the old fixed
2.4 GB reserve. No database transaction spans setup work.

Setup owns the existing admission lease through download, conversion, initialization
and cancellation cleanup. Short data mutations remain available and still block an
updater reservation. Review caught the old middleware taking an Import lease for
every write: it would reject setup itself and block cancellation and editor saves.
The real-router regression now covers those joins. The parent restarts and reaps
the inference child before final initialization, invalidating old model handles.
Capability cells and reports are refreshed from checked model identities. The
settings screen exposes cancellation and refreshes timing capability after setup.
The native alignment-only action is hidden; the main control sets up all models.

Validation on this M5 Mac: 148 core tests, 57 host tests, one native-service lifecycle
replay and eight process replays pass. The renderer suite passes 101 tests with one
optional external fixture skipped. Strict core/host/models Clippy and the locked
optimized desktop build pass; the previously recorded upstream mlx-sys warnings
remain. New failure tests cover bad/missing ranges, HTTP restart, interrupted bodies,
wrong sizes and hashes, cancellation, competing installers, symlinks/hard links,
partial activation, unexpected inventory, stale staged directories, and process reset.
Independent read-only verification found no remaining source-slice blocker.

Real pinned Silero download (1,247,300 bytes) passed in
`/private/tmp/speakerdesk-native-setup.cLJP4C/models`. To avoid downloading the large
unchanged weights again, five other models' pinned source files were APFS-cloned
from this task's qualification copies into private staging. The production HTTP
setup route verified/activated those files, ran real native CTC conversion, restarted
the child, and initialized all six capabilities. A real import of `two-voice.wav`
produced the expected two passages and timed reading turns for two speakers.
The first harness assertion incorrectly required silence-gap rows to contain text;
retained output proves two spoken passages plus two legitimate blank gap rows.
No inference rerun was needed to correct that assertion. People remained empty:
setup did not enroll any voices.

A subsequent whole-host restart preserved that output. A saved edit, TXT export and
409 stale-save rejection passed while exercising the same file-backed store. All six
models initialized again, and the host exited successfully. This is native service
and actual-model proof, not Tauri/editor acceptance or a performance comparison.
Logs, setup snapshots, retained model output, edited document and export are in the
same disposable directory. Test/build logs are `/tmp/speakerdesk-setup-*`.
Optimized executable SHA256:
`f5ddb1e4cccef83484d5e4a1d797f0cf985c8d5f73d579ae0be80ec79a27e1bb`.
Only this worktree's regenerable debug incremental compiler output was reclaimed
when free space fell below 1 GB; binaries, test evidence, installed app, recordings,
production databases and the existing model cache were preserved.

The UI tool still reports a locked Mac despite the user's unlock replies. The old
isolated desktop preview remains at the earlier retry build. Both the long-recording
editor check and setup controls in Tauri remain pending. Continue the accepted full
outcome: native live capture coordination/drain and saved-audio recovery, bounded
memory/history work, Python-free default package and updater, security qualification,
and equivalent cold/warm/live performance gates. Do not treat this checkpoint as
replacement completion. Windows and issue #25 remain follow-up scope.
