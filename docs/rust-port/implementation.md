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
   Desktop interaction is blocked by the locked Mac; the UI workflow, public
   audio and end-to-end performance qualification remain pending.
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
