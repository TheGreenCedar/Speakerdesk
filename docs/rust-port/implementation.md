# Python-free Mac implementation

Accepted outcome: complete the native Mac application, preserving the shipped
0.6.6 behavior and saved data. Real import/capture must produce an editable
transcript that survives full application restart and export. Equivalent release
measurements must match or improve baseline latency, sustained live processing,
and peak memory, except for the explicitly accepted extra Stop time described
below. MossFormer2/episode work belongs to issue 25; Windows comes later.

Starting PR head: `01e757cd5f3db3f8fccfcfa34e94e91cadc0fe26`.
Starting base: `59e7b0c860f72ca94c24e6eab18ac820b144ee24`.
Local implementation branch: `codex/python-free-mac`.

## Delivery status after migration testing

The main native Mac workflows are implemented and have passed the retained actual-app
tests: import and live capture, editing and revision conflicts, save/reopen and
export, model setup, language/alignment/voice paths, child/helper failure and
disposable updater replacement. The canonical signed package runs without a
Python application runtime. The optimization work did not replace these broader
migration acceptance criteria.

The accepted Mac implementation and local qualification are complete at code
revision `0201c8063c0a56bd597f721ddedce3946bcbeafe`, including the repaired
Ogg/Opus editor path and forced-shutdown process cleanup. This is a local
qualification decision within the recorded fixtures and hardware, not a release
or installation. The bounded long-import comparison and
dispatched-accuracy cancellation check below establish no observed regression on
those fixtures; baseline lacks an automatic child-recovery comparator. Existing legacy voice profiles are preserved
but incompatible model identities do not participate in native recognition;
new native enrollment/recognition does not qualify historical profile reuse.
That exclusion implements the accepted preservation policy rather than a missing
embedding-space bridge: the existing People flow asks the user to save the voice
again, and guarded replacement preserves the old profile until consented success.
Windows and issue 25 remain deliberate follow-ups.

On October 12 the owner explicitly accepted this PR's extra Stop time for final
refinement. The retained 2.570-second native versus 1.952–1.959-second baseline
result remains disclosed, with native final-passage timing absent from that
baseline. It is no longer a completion blocker. This acceptance does not waive
unrelated latency, sustained backlog or peak-memory regressions. Neural Engine
execution/overlap for the speech models is separate follow-up work, not part of
this rewrite. Existing ReDimNet compute policy stays measured CPU_AND_GPU.

Code revision `0201c8063c0a56bd597f721ddedce3946bcbeafe` passed all three GitHub
CI jobs: source, Linux Rust and Mac Rust (run `38277886287`). The final status
update changes documentation only; results below remain scoped to their recorded
artifacts. Earlier entries are a historical work log, not the current gate list.

Legacy Python application sources remain test/oracle references, not package
inputs. Current CI, reference generators and source-boundary checks still depend
on them; removal requires a separate coherent cleanup of those callers. The
`speakerdesk` directory also contains the editor assets and model metadata that
the Rust build still embeds. No legacy implementation or shared assets are
deleted in this delivery. Existing build-time Python scripts are distinct from
the application's runtime-free requirement.

## Sequence and current state

1. Source repairs verified: updater/export/bundle/parser/readiness and truthful
   tests. Mac CI added; prebuilt MLX linkage demonstrated (details below).
   Independent verifier architecture_capture found three issues, all corrected.
2. Implemented and independently reviewed through mounted replay tests: explicit-root
   AppState uses SQLite for jobs, identities and preferences, immediate edit
   transactions, guarded import publication, consistent migration snapshots,
   private journal recovery and legacy audio copy-on-write. Process lifecycle and
   operation cancellation and active meeting state share that store and owner.
3. Qualified: Silero, multi-window offline Nemotron and Cohere run through
   the actual native child and assembled import pipeline. An isolated Tauri
   preview now starts the Rust HTTP service and resident child. Actual HTTP
   import/edit/name/export/restart passes on synthetic audio (proof below).
   The actual editor import/edit/name/boundary/save/reopen/export/stale-save path
   passes, including Auto language and CTC. Short synthetic/public recordings have
   equivalent warm application measurements. Long Nemotron execution has an
   independent numerical reference. An 82-second import now passes the actual
   editor, tail edit, save and full restart. Short/long import and nine-minute live
   comparisons are recorded below, with output differences and sampling limits.
4. Qualified: automatic language, supplied-text CTC and native Core ML ReDimNet
   extraction work. Manual voice enrollment/checking now passes the actual editor
   and full restart (below). Automatic import recognition and native model setup
   pass service/model tests; actual setup cancel/resume and post-setup import now
   pass in Tauri. Live recognition also passes
   a real-model saved-PCM test, described below.
5. Qualified: native capture coordination, streaming tails, child-loss recovery
   and live-edit continuation pass mounted replays and saved-PCM real-model tests.
   Sealed live timing, identity and saved Auto/explicit-language accuracy passes pass
   real-model saved-PCM checks, including long English context assembly and live
   language-decision persistence across protected continuation and restart. Long
   growing rows retain accepted cores instead of re-decoding completed audio.
   Qualified Pause/Resume passes native controls and mounted process replays.
   Raw dual-source echo passes independent baseline DSP and mounted failure
   replays. Saved long non-English accuracy passes functional timing/publication
   tests and a real-model German recording. Rolling live refinement passes mounted
   lifecycle replays and real-model retained-audio checks. Actual Tauri live
   controls, editing, playback, four exports and restart pass with a synthetic
   PCM helper. Real microphone/Mac-audio Start/Pause/Resume/Stop also reaches the
   editor on the current Mac route. Bundled-helper failure preserves data through
   restart and permits a new capture. Nine-minute saved-PCM comparison preserves
   words/audio with lower observed backlog and memory; Stop is slower. Equivalent
   performance observations meet the amended acceptance criteria on these fixtures;
   the stated acoustic limits remain qualifications on the result.
6. Qualified: native-default startup and canonical Python-free packaging are
   implemented. The relocated Developer ID signed package runs real inference
   with an isolated store and sanitized environment, reaches the editor and
   saves word/name/boundary edits through full restart and native export. The
   packaged security/data matrix passes. Disposable outgoing updater cancellation,
   retry, shutdown/replacement, network failure and preservation pass. Equivalent
   performance observations and their limits are recorded below. The refreshed
   signed package also passes the final timing-inspector and failed-meeting
   accuracy-control checks.

Planned validation uses disposable storage and synthetic/public audio. An
unintended baseline relaunch touched default storage directory metadata; the
exception and limits of its inspection are recorded below. No installed app
replacement, production-data migration, merge, publication or GitHub comments.
One implementer and one independent verifier per batch; one
owner of Mac builds/model runs. A component fix does not complete this objective.

Execution preference: new subagents use Sol 6.1 with explicit reasoning effort:
high for bounded implementation/verification, xhigh for model correctness or
cross-boundary architecture. The coordinator remains the sole Mac build/model/UI
owner. The latest product corrections distinguish installed-model loading from
downloads and add honest activity feedback during uploaded-recording transcription.

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

## Native startup and packaging candidate (qualification in progress)

The canonical Mac build now bundles the Rust desktop/host/child executable,
Swift capture helper, echo library, MLX 0.32.2 dylib and Metal kernels, with
native dependency notices. Ordinary startup uses the native store and its
`models` directory by default. The Python sidecar is no longer a startup or
packaging branch. Recovery checks the native resources; ordinary startup still
opens the editor when capture resources are unavailable.

Installed-model initialization reports `loading`; download work reports
`downloading`. Loading hides the pre-meeting setup card and displays “Loading
models…” in its compact status hint. Settings retains the descriptive load or
cancellation phase without a download bar or download action. A cached presence
check distinguishes missing download files from an initialization failure, so
first setup still explicitly requests downloading models.

Coordinator verification on the current Mac: host 65 unit tests, 18 live replay
tests, one native-service lifecycle test, nine process tests and 15 xtask tests
pass. Three host, one live and three process tests remain explicitly ignored;
their separately retained hardware/model proof is unchanged. Twelve build-tool
tests, source/workflow checks, shell syntax and strict host/models/desktop/xtask
Clippy pass. The actual license collector stages 470 dependency notice entries;
23 packages require retained version-specific license text because their crates
omit it. Logs are `/tmp/speakerdesk-native-package-final-host-xtask.log` and
`/tmp/speakerdesk-native-package-clippy.log`. No new acoustic acceptance is implied.

The uploaded-transcription activity bar is joined. It covers local submission,
queued/preparing/processing and passage retries, stops on completion/failure and
does not claim a percentage unsupported by the model worker. Four focused tests
fail against the old frontend and pass against this candidate. The complete Node
suite has 110 passes and one existing optional performance skip. A late response
cannot replace a newer selected transcript revision.

Both canonical Mac builds completed: the ad-hoc build passes bundle verification
but fails actual hardened-runtime launch when loading the separately ad-hoc signed
MLX library. It is not a runnable qualification artifact. The same source built
with the existing Developer ID identity runs without relaxing library validation
or changing system settings. Neither artifact is notarized or published.

The signed ZIP was extracted into
`/private/tmp/speakerdesk-native-package.waaqwdz8/signed/`. Its actual app launches
with a system-only PATH, no Python/DYLD/model override, an explicit disposable
storage root and the normal storage/models directory. Observed application
processes are the desktop, native host and resident native model child. Loaded
files show the bundled MLX dylib; no Python runtime, site-packages or developer
checkout supplies application behavior. `candidate-source.json` in the evidence
root records HEAD plus dirty-source hashes for this build.

An 82-second synthetic recording was imported and transcribed through the
packaged editor. The indeterminate loading bar was visible during real model
execution. Its ten spoken passages became editable, and word, speaker-name and
boundary edits persisted at revision 6. Full quit reaped the desktop/host/model
processes. Reopening displayed the same words, name and boundary; the complete
document/revision and both WAV hashes were identical. Native project export and
text export after restart contain those edits. The actual startup screen also
showed “Loading models…” with no download button while initialized files loaded.

The packaged HTTP/data matrix passed 72 checks: mounted Host/Origin/token guards,
malformed uploads/saves, protected and stale writes, private response fields,
SQLite/detail/export agreement, all four export formats, full/ranged audio bytes,
and audio-file/parent-directory symlink confinement. Temporary file swaps were
restored and independently verified. Evidence is under
`/private/tmp/speakerdesk-package-security.diCepT/results-final/`. A first harness
comparison was too strict about stored null versus omitted optional fields; it
stopped before mutations and was corrected without weakening raw row equality.

Actual bundled Swift-helper failure was injected into a short synthetic system
audio capture. The failed helper was reaped; the corrected phrase, people and
133,546 samples in all three WAVs survived. Full application restart preserved
them, then a fresh real capture produced text and stopped Ready. This proves
retained-data recovery and new capture, not automatic repair of a failed device
session. The terminal failed meeting snapshot is intentionally retained until
another meeting starts; an initial harness expectation of idle was corrected.
Evidence is `speakerdesk-native-package.waaqwdz8/helper-loss/`.

These actual checks exposed two small editor defects, now fixed and independently
reviewed: timing-input changes refresh the row's visible/accessibility labels;
failed recordings hide accuracy controls after their worker has stopped. The
focused regressions fail before each change. The final Node suite passes 112
tests with one existing optional performance skip. The refreshed Developer ID
package now passes both actual UI checks: changing the start to 0.06 immediately
updates all passage labels with the inspector still open and saves revision 7;
the failed capture keeps its error, words and playback with no accuracy controls.
Quit reaped desktop/host/model processes. Evidence is
`/private/tmp/speakerdesk-final-package.1cj7mpxh/ui-delta-check.json`; its source
and archive hashes are in `candidate-source.json`. The packaged executable SHA is
`8728f1d839839f20d2bbcbfb4f772f1d498a3233217554f24b30f72ad36fde6a`.
The full package workflow above precedes these two fixes. None of this implies
performance acceptance.

The exact-version native package's actual updater check returned current. A
disposable copy compiled as 0.6.5 (same source, approved HTTPS feed/public key,
signature/version policy and Developer ID signing) exercised nonzero-byte
download cancellation, a fresh retry, verified download, real editor preparation,
native child shutdown and replacement with the published 0.6.6 bundle. The
replacement restarted in its own disposable home. All 11 prior file hashes,
the complete native SQLite logical dump and three native jobs were preserved;
codesign verification and the installed 0.6.6 version passed. Both the original
native children and the subsequently closed legacy runtime were reaped.

This qualifies the outgoing native updater to the existing published legacy
release. It does not qualify a future incoming native release or legacy access
to the native database. The observation driver stopped early on the expected
connection-refused transition; an independent post-install check verified the
completed replacement and preservation without repeating it. The external driver
now recognizes explicit nested connection-shutdown errors while retaining TLS,
timeout and unknown-error failures. Evidence is under
`/private/tmp/speakerdesk-updater-qualification.XSxn2Q0F/`, especially
`post-install-verification.json`.

A separate disposable copy also passes actual updater network failure and retry.
A process-local loopback proxy made the approved HTTPS feed unreachable; the
check reported an error, download/install returned 409, admission was released,
and the host, executable and native storage remained intact. Relaunching that
same copy without the proxy completed a real HTTPS check as current. Both process
trees were reaped. Evidence is `failure-driver.log`, `retry-driver.log` and
`failure-retry-launch.json` in the same updater evidence directory. No global
network settings or installed application changed.

A single temporary, optimized Stop diagnostic reproduced 2.902 seconds on the
retained 54.456-second synthetic PCM fixture. Interruption/database/lease work
took milliseconds; EOF work took 1.018 seconds, including 0.338 seconds of sealed
CTC, followed by 1.403 seconds of saved accuracy work (1.055 seconds decoding and
0.341 seconds CTC). These nested phases are not additive across their parents.
The run preserved exact PCM and final timing, then reaped its process tree.
Sparse instrumentation was removed and all six touched Rust files verified
byte-identical to their pre-diagnostic hashes. Evidence is under
`/private/tmp/speakerdesk-stop-profile.k_0ivl9_/`. Attribution is established;
equivalent uninstrumented performance acceptance is still open.

The bounded saved-passage accuracy path now reuses completed CTC only after fresh
ASR returns the same text/language and current PCM, vocabulary, target/word
coverage, frame bounds and provider/policy identities match. Existing context
validation checks acoustic cells; the ordinary attachment rebuilds current
revision/source/activity provenance. No cache, worker contract or persistent
receipt was added. Mismatches take the existing fresh-CTC path, and cancellation
and protected/stale publication remain guarded.

The original regression failed with two CTC calls instead of one. The extended
existing EOF test covers reuse, input/identity/cell mismatches, current provenance
and protected/stale rejection. Final validation: 177 core tests, 18 live-route
tests (one existing ignored) and strict core/host Clippy pass. Two intermediate
test-fixture mistakes (reference cloning and a moved single-cell word boundary)
were repaired before the green run; production was not weakened to fit them.

The normal optimized binary then consumed the unchanged 54.456-second fixture in
`/private/tmp/speakerdesk-live-comparison.tBmzLh/native-1791812182239152000/`.
Stop took 2.570 seconds versus the prior 2.894 seconds; first persisted words
took 7.864 seconds and Start-to-Ready took 56.925 seconds. Sampled peak owned
physical footprint was 3.520 GB and sampled CPU was 18.997 seconds. Words and
boundaries match the prior result, all spoken rows have complete timing, every
PCM sample is preserved, and all owned processes were reaped. The binary SHA is
`70ee2d866a1e85cb68241311e1af143547d679a337d8c8519e6925afed78b0f1`.
This confirms the expected roughly 0.34-second reduction, not full parity:
retained baseline Stop is still 1.952–1.959 seconds and its final timing is absent.

During the next external comparison, the first collector exited on the baseline
long recording's missing timing and its attached process group also stopped the
preview. A subsequent UI lookup unexpectedly relaunched that copied baseline
without the explicit disposable-storage environment. It was immediately quit;
no import, capture or edit was performed in that launch. All original and
relaunched processes were reaped. The exact relaunched storage path was not
observed live; its compiled default identifier is `com.speakerdesk.desktop`.
Read-only metadata checks found that the default root and recordings directory
change times match that launch, consistent with legacy startup's chmod 0700.
The database and model settings still have modification times predating this
run, with no WAL/SHM/journal present. No prior hashes or permission snapshot exist
to establish byte identity or prior mode. These files were not opened through
SQLite during inspection; the unintended application startup did open its store.
The collector now starts an independent process session and saves each import's
clocks before checking later quality; the coordinator checks recorded PID
liveness before any UI binding. This unintended launch is excluded from all
comparison evidence. Details are
`/private/tmp/speakerdesk-application-cold.ryyiBjrt/launch-incident.json`.

The repaired collector retains a fresh-process short Auto import on the final
signed native package and the copied 0.6.6 application. Both produced identical
words, passage bounds, language, PCM anchor and complete timing on the retained
7.06-second public fixture. Process launch to HTTP Ready was 9.051 seconds native
and 11.534 seconds baseline. Baseline SQLite verification completed at 11.542
seconds; the original native persistence check rejected stored-null versus
public-omitted language on two empty gaps. A separate strict postcheck confirms
that serialization-only difference and persistence, without fabricating an
earlier persistence timestamp. Sampled peak owned physical footprint was 3.478
versus 3.930 GB, sampled CPU lower bounds 8.207 versus 7.742 seconds. These are one
pair with existing OS caches and explicit disposable storage, not machine-cold
or editor-render clocks. Both actual editors displayed the saved transcript.

The same native process then imported the unchanged 82.010-second fixture in
7.974 seconds to HTTP Ready and 7.993 seconds to verified persisted aligned
output: all ten spoken rows are timed. The phase sampled 3.597 GB peak owned
physical footprint and 4.441 CPU seconds. A bounded baseline repeat with the
repaired collector measured 8.932 seconds from import to HTTP Ready and 8.937
seconds to verified persistence. It reproduced the earlier baseline output:
only one aligned spoken row out of eighteen, including eight standalone “you”
rows. The baseline short-plus-long process tree sampled 4.029 GB peak owned
physical footprint and 12.755 CPU seconds; those cumulative CPU observations
are not the native warm-phase CPU scope. The native result is faster on this
fixture while producing more complete timing, not an equal-output acoustic
benchmark. The repeat's preceding short import took 4.978 seconds (11.481 from
process launch), consistent with the earlier 5.044-second baseline import.
No hardware caches were flushed. All measured native/baseline root, service and
model processes were reaped through actual UI Quit, including the repeat in
`baseline-long-closeout/shutdown.json`. Evidence:
`/private/tmp/speakerdesk-application-cold.ryyiBjrt/`.

## Additional performance candidates (bounded changes joined)

The accepted implementation now includes four ordered opportunities, without
assuming defects or speedups: Cohere decoder cache/mask/evaluation efficiency;
live work priority at existing refinement boundaries; bounded preparation ahead
of import inference; and truthful stopped/saved/final-accuracy presentation.
Sol 6.1 high and xhigh agents investigated independent source seams in parallel,
with a separate implementer and verifier for the selected changes. The coordinator
alone owned Mac builds/model measurements. Decoder investigation came first;
the two selected changes below are joined without expanding the scheduler.

Pinned mlx-rs and MLX 0.32.2 source must establish API support; newer KV-cache
guidance and benchmark numbers do not. Any selected change retains exact input
alignment reuse and publication guards, fresh saved-accuracy decoding, valid
lengths, cancellation, model/session ownership and existing latency/memory gates.
Concurrent heavy inference remains a later evidence-dependent experiment. The
2.570-second native Stop versus 1.952–1.959-second baseline remains a distinct
interaction cost, qualified by baseline's absent final-passage timing. The owner
explicitly accepted that cost for this PR; faster Start-to-Ready does not erase
it. The measurements below do not
establish a performance gain from this additional batch.

Candidate decisions:

- Keep only the request-local cross mask: move the exact valid-prefix expression
  from each decoder step to the caller where encoded frames and valid length
  become fixed. This is a small invariant-allocation cleanup with no demonstrated
  speedup. The experimental scalar causal-mask branch and its library-only test
  were removed after measurements failed to establish a benefit. Prompt and
  single-token causal construction, KV layout, evaluation, cancellation and
  fresh-request ownership remain unchanged.
- Defer chunked/preallocated KV storage. Pinned MLX 0.32.2 concatenation copies
  the prefix, but slice updates can also copy the whole input unless ownership
  permits donation. A naive preallocation is not sufficient evidence. Per-cache
  eval calls after token evaluation are not proof of repeated GPU synchronization
  or model work; retain the current graph-lifetime boundary.
- Reject a blanket “no accuracy while any audio is queued” guard. Synchronous
  live inference accrues new notices, so that guard could starve rolling accuracy
  and increase Stop's drain. Existing six-second live batches and pending-control
  priority already bound/interleave work. Retain scheduling until a measured
  backlog case supports a small change with demonstrated fairness.
- Defer concurrent import preparation. It shares the serial queue with inference;
  source inspection confirms head-of-line delay is possible, but measured WAV
  preparation is about 56 ms. The inference path can itself prepare audio, so a
  second worker would need same-job writer ownership; normalization currently
  retains source and resampled vectors, with an unmeasured overlap-memory cost.
  No batch-throughput or memory improvement is claimed.
- Implement one Stop presentation change: after healthy helper/writer drain,
  atomically checkpoint coverage and say “Audio saved. Finishing transcript…”.
  Existing refinement controls distinguish active, paused and unresolved work;
  a second header would add synchronization without a useful new state. Keep
  finishing/Ready, admission and playback/export behavior unchanged. The existing
  held-final-model regression fails before the change with “Finishing saved
  audio…”. The guarded message uses the existing coverage transaction, without
  another write, revision, task or public state. The regression now also verifies
  that Pause leaves the audio-saved message while cancellation still owns
  inference. This is clarity, not an inference speedup.

The discarded combined mask experiment preserved the independent 31-token
two-voice reference across both repeated requests and both process pairs.
Reference first/second decode times were 316.491/282.735 and 622.535/279.276 ms;
candidate times were 392.831/272.151 and 336.668/413.246 ms. This variation supports
neither a speedup nor a causal regression claim. Its one unchanged 54.456-second
live run (`native-1791814798166761000` under the retained live-comparison root)
also preserves exact PCM, words, bounds and complete final timing: first words
7.889 seconds, Stop 2.668 seconds, total 57.008 seconds, peak owned physical
footprint 3.518 GB and sampled CPU lower bound 19.645 seconds. The roughly
0.098-second Stop difference from the prior 2.570-second result is near the
100-ms polling interval; the sampler's largest gap was 0.313 seconds. All owned
processes were reaped. No further long variant run is justified by this evidence.
Neither discarded-variant nor earlier signed-package results establish latency
for the final hoist-only package. Logs are `/tmp/speakerdesk-decoder-*.log`.

The final hoist-only source passes the independent real Cohere reference twice
with identical text and 31 tokens (312.508 and 273.562 ms observed decode times,
not a comparative speedup). The existing live suite passes 18 tests with one
previously ignored; strict all-target host/models Clippy passes. Locked canonical
optimized packaging, bundle verification and strict Developer ID signature
verification pass. The upstream allocator warnings and duplicate rpath warning
remain. Logs are `/tmp/speakerdesk-decoder-hoist-final.log`,
`/tmp/speakerdesk-stop-status-live-tests.log`,
`/tmp/speakerdesk-stop-hoist-final-clippy.log` and
`/tmp/speakerdesk-stop-hoist-package-build.log`.

The final signed package was extracted to
`/private/tmp/speakerdesk-stop-status-package.6rhhcgy2/Speakerdesk.app` and launched
with explicit disposable storage, private model clones, sanitized PATH and no
development-library or capture-helper override. Executable SHA-256 is
`dd66f06f9badc8b601efd68f8d38057f91df374a6838d6a67226fdb4fa5c177c`;
the ZIP SHA-256 is
`42fd84a622798a746f585f0e5927f7e0270f6b11cde91058c894ef33671ccf7a`.
The retained launch receipt binds the decoder/network/runtime source hashes.

Actual public playback through the bundled Mac-audio helper reached the editor
and Ready with complete final timing. That first recording contains 1,383,129
mono 16 kHz samples including startup/trailing silence; independently hashed
spoken crops match their row and alignment anchors. On a second recording,
stopping during speech visibly showed “Audio saved. Finishing transcript…” and
“Rechecking sealed passages…” with live text retained. The actual editor then
showed Saved, Accuracy pass complete, the editable passage and 42-second playback.
These physical captures prove UI/capture boundaries, not equivalent PCM or a
latency comparison. `ui-stop-observation.json`, `capture-final.json` and
`short-capture-final.json` retain the observations. Actual UI Quit reaped the
desktop, host and resident model child; the public player also exited, recorded
in `shutdown.json`. No package was installed over the working app or published.

Independent verification after Quit confirms that the second job's SQLite
document equals the API receipt exactly at Ready revision 55. Its audio/system
WAVs contain identical 686,806 mono samples, and the reference WAV has the same
frame count. Coverage, canonical observation and independently hashed spoken
crop match. Reading turns partition the current text with current provenance;
the cut-off final “last.” fragment remains unassigned with null reading-turn
times. Complete supplied-text alignment does not establish full speaker/time
attribution or full-fixture transcription accuracy for this interrupted capture.

The final format and lifecycle checks are now joined, as recorded below.
The known Stop cost is accepted, and the long-import timing and
output differences are recorded above. No additional cache, scheduler, concurrent preparation
or heavyweight GPU-overlap experiment is justified by the current measurements.

## Push closeout and format compatibility check

The reviewed implementation was pushed to PR 27 at
`bff6074d987aa6deef892ab268332cbb436b6473`, without rewriting the existing branch.
Its source CI ran 797 tests and found one stale workflow-mutation fixture that
still targeted the removed `.cache/uv` entry. The test now targets `.cache/npm`
while retaining the same forbidden Hugging Face cache substitution and exact
checker rejection. The focused two-test suite passes; production policy is unchanged.

The follow-up read-only compatibility review found that the pinned Symphonia
registry has no Opus decoder, while import routes accept `.ogg` and `.webm`.
A public Opus sample from the codec project's examples, renamed to `.ogg`, was
accepted by the actual signed native host and persisted as failed with
“Could not decode this recording. Supply a supported audio file.” The unchanged
legacy `audio.normalize` function succeeds on the same bytes using this Mac's
system `afconvert`, producing 228.107 seconds of canonical PCM. This reproduces
an import-preparation compatibility defect; no models or private data were used.
The file is the public `ehren-paper_lights-96.opus` example, SHA-256
`997f5aaa2528de8392bac3f59140e1ab24b491a70048697c8d3dc91cb4300df3`.
It remains outside Git with the two boundary results under
`/private/tmp/speakerdesk-opus-import-wbllqn5f/`.

The first native driver uploaded during initialization and received 409; a later
driver mistakenly expected 200 instead of the correct 201 creation response.
The accepted job's persisted decode failure was then read through the native
service after restart, without another upload. Every owned host exited normally.
The synthetic Ogg encoder attempt failed before producing a qualifying fixture;
only the valid public sample supports the decoder finding. The legacy result is
a direct normalization-function test, not a new full baseline desktop run.

## Final compatibility and lifecycle corrections

The Opus repair uses the existing macOS system decoder only when Symphonia's
probe or decoder construction reports `Unsupported`. Recognized decoding errors,
nonfinite samples and duration errors retain their existing failure path. The
fixed `/usr/bin/afconvert` command receives absolute input/output paths, a cleared
environment, null standard streams and a private sibling directory. Its inherited
three-minute conversion deadline and two-hour PCM limit are retained; the output
ceiling permits bounded WAV header overhead and is checked both during conversion
and after exit. Full canonical WAV validation precedes atomic publication.
Ordinary success, errors and timeouts reap the child and remove staging. No codec,
process framework, Python runtime or Cargo dependency was added.

An authored 8,680-byte Ogg/Opus tone fixture, with a partial final frame, reproduces
the native decode error before the change and yields the independently checked
16,006 mono PCM16 frames afterward. Negative cases preserve the prior destination
for malformed input and nonfinite known-codec samples. Real disposable child
tests cover conversion timeout and both running/completed oversized output.
Independent review caught and corrected the completed-child size-check gap.

The same review traced the desktop's 45-second forced-Quit path. The shell
plugin's default killed only the host, which could leave a longer-running native
converter alive. The existing `set_process_group(true)` API now makes forced
termination cover the owned host group and descendants. A real fake-host/sleep
regression through `RuntimeSettings.command` fails before that change and passes
afterward. Forced SIGKILL can still leave private per-job staging on disk because
destructors cannot run; deleting that native job removes its complete media
directory. Normal conversion failures clean staging immediately. This does not
affect legacy-owned rollback recordings.

The joined source passes 180 core and 27 desktop tests, strict all-target
core/host/desktop Clippy and the source-boundary check. Logs and the original
public-file reproduction remain in
`/private/tmp/speakerdesk-opus-import-wbllqn5f/`.

Canonical optimized packaging, strict Developer ID verification and bundle
inspection pass. The relocated executable is SHA-256
`7b8ddc1c51107df501184d226359ca47b277c83b457c004f8a3228b7e58a4918`.
With sanitized environment and disposable storage, the original failing public
Opus file now reaches persisted `uploaded` state at 228.1066875 seconds. All
3,649,707 normalized PCM samples exactly match the retained legacy result; no
model inference was needed for this boundary test. Actual process observations
show the native host and model child in the host's owned process group, with no
Python application process. The cloned incompatible voice profile, including
centroid, version and enrollment payload, and all saved names remain unchanged.
The People API exposes saved/incompatible status and re-enrollment guidance.
After unlock, the same signed package imported the 7.06-second public speech
fixture encoded as Ogg/Opus through the actual Tauri file picker. The UI showed
preparation and transcription activity, then the expected model-produced words
and timed reading turn. A word edit saved at revision 4, survived a full app
Quit/relaunch, and appeared unchanged in the reopened editor and native text
export. API documents and revisions match exactly across restart. The actual
People panel showed “Save this voice again to use recognition” and the preserved
Daniel fixture name before and after restart; the second observation used a
screenshot because the accessibility tree omitted the reopened modal controls.
The profile centroid, identity, version, consent and enrollment payload, and all
person rows, match the pre-run copied fixture exactly after both quits. All six
attributed desktop/host/inference/WebKit processes from each run were absent
after Quit. This is a synthetic incompatible-identity fixture, not proof that
historical embeddings can participate in native recognition. Receipts:
`/private/tmp/speakerdesk-import-closeout-0ckgcmrz/`.

One short real-model service pair uses the unchanged 288,647-sample synthetic
tail (18.040 seconds). After Stop and observable dispatched refinement, native
Pause acknowledged in 2.124 ms and reached terminal ownership release in
6.945 ms. Current words and exact saved PCM survived; a subsequent protected
edit survived accepted Resume. Baseline acknowledged Pause but retained
`finishing` and inference ownership throughout the 30-second observation window.
Its eventual drain time was not measured. This is a censored observation, not
a numeric speedup or a claim about interrupting an active GPU kernel. Both owned
process trees were reaped. A preceding driver attempt used the wrong capture
packet encoding and was excluded; float32 conversion was independently checked
before the recorded pair. Evidence:
`/private/tmp/speakerdesk-cancel-closeout-yi7lo2c_/comparison.json`.

There is no baseline generic import-cancellation endpoint or automatic live-child
rebuild to compare with native's added behavior. Existing deterministic held-model
tests establish acknowledgement-before-release and stale-publication boundaries;
the retained actual native recovery exercise establishes preserved edits/audio,
restart/export and a 4,105 ms worker rebuild on that fixture. It does not establish
an unavailable baseline recovery speedup. Completed baseline/native refinement
checkpoints were not deleted merely to manufacture a cancellation benchmark.

## Earlier foundation proof

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

## Live engine reads retained audio

The live engine now opens its canonical recording once and reads bounded PCM16
crops from the writer's committed sample range. It no longer retains whole-session
PCM or a second mutable transcript document. Both source endpoints are checked
before either model session advances, so an incomplete system write cannot leave
the microphone session ahead. Replay tests use real growing WAV files, including
arbitrary splits around 512 samples, final partial frames, and reconstruction by a
fresh engine from the retained file. These remain model replays, not capture proof.

Only unfinished utterances are revisited. Speaker activity older than the active
utterance or next onset context is discarded; speech evidence lookup uses binary
range selection. Activity projection uses a boundary sweep instead of scanning
every turn for every region. Sealed candidate and decode caches are released. The
utterance book and speech metadata still grow with the transcript; long-session
memory and performance gates remain outstanding.

A reproduced two-source bug assigned the same utterance ID to simultaneous
microphone and system passages. Native IDs now include the source namespace.
Qualified microphone configuration selects unity gain and the production input
policy; the system source retains quiet-speech conditioning. Verification of the
Swift processing receipt belongs to the upcoming host capture join.

Validation: 153 core tests, 57 host tests, one native-service lifecycle replay and
eight disposable-process replays pass. The network setup test and three real-model
qualification tests were not repeated for this engine-only change. Strict core/host
Clippy passes. The independent verifier closed source review after the dual-source
preflight and production conditioning corrections. Logs: `/tmp/speakerdesk-live-reader-*`
and `/tmp/speakerdesk-live-source-id-red.log` (the reproduced identity collision).
The optimized desktop executable remains the prior setup build; capture is not yet
wired to these engines. No real UI check was possible: CUA still reports the Mac
locked despite the user's unlock replies.

The accepted replacement outcome remains active and unfinished. Continue the Swift
capture process/writer/store join, recovery and stop/drain handling; replace the
bespoke echo FFT and measure correlation choices; then qualify live hardware,
native default packaging/updater, security and equivalent 0.6.6 performance. The
actual Tauri setup/long-import checks remain pending. Installed app, production data
and existing model cache stay untouched; Windows and issue #25 remain follow-ups.

## Maintained FFT and measured reference correlation

Removed the project-owned Bluestein/complex FFT implementation. The echo clock now
uses the workspace's already pinned `realfft` 3.5.0 for power spectra and linear
correlation. Independent DFT equations cover even, odd and prime lengths; direct
dot products check every valid correlation lag, including the production 9,999 by
1,999 sample shape. The existing polarity, stereo-channel, tonal rejection, delay
and drift tests pass through the new implementation.

On this M5 Mac in an optimized build, seven alternating comparisons included
planning and allocation. Direct correlation took 8.259–12.206 ms per channel
(median 9.327 ms); FFT took 0.167–0.637 ms (median 0.210 ms). This supports choosing
FFT for the production correlation shape. It is a DSP-kernel comparison, not an
end-to-end capture or 0.6.6 performance acceptance result.

Independent review found a new nonfinite-input panic at the inverse FFT. A release
regression reproduced it, then passed after the public correlation boundary began
rejecting NaN and infinities from every source. Final optimized echo tests: 16 pass;
the explicit comparison also passed. Strict echo Clippy passes. Source review is
closed. Evidence: `/tmp/speakerdesk-echo-{correlation-performance,nonfinite-red,final-tests,clippy}.log`.
Capture orchestration, native default packaging and the remaining full-app gates
are still unfinished; this library is not yet wired into active native capture.

## Owned capture transport and retained source writer

The host now owns the unchanged Swift helper through bounded Tokio pipes and
channels. Partial JSON lines survive control selection. Stop retains its original
15-second deadline; failure kills and reaps the helper. Terminal snapshots follow
both process reaping and the blocking writer's drain. Independent verification
reproduced a cleanup hang from an inherited stderr descriptor, then verified the
bounded diagnostic drain fix. Unused deadline/classification scaffolding was removed.

One blocking writer aligns microphone and system packets on the recording clock,
retains the original stereo reference, and commits a sample endpoint only after
all track headers are flushed. Hound's reusable block writer avoids individual
sample syscalls. Qualified Apple microphone output is retained unchanged, with one
validated helper/route receipt and no second echo pass. Raw single-source capture
is supported; unqualified dual-source capture still needs the echo integration.
Qualified Pause/format changes retain the tail and end that capture epoch, matching
the existing helper boundary. A permanently stuck recording filesystem can still
delay writer completion after the helper is reaped; no terminal success is invented.

The four retained-PCM fixtures were independently generated with the exact 0.6.6
SourceMixer from `91f7593fc9c890c69befd6663cdcdeb0072d5f70` and NumPy 2.3.5.
Rust matches every retained byte for packet splits, delayed/inverted stereo,
missing/late sources, half-even timestamps and final tails. This qualifies the
identity mix and retention, not physical devices, AEC or voice-processing quality.
A sparse recording regression also removes the accidental two-hour import limit
from live WAV reads/writes; import policy remains unchanged, and RIFF size overflow
is rejected.

Validation: 154 core tests, 65 host tests, one native-service replay and eight
process replays pass. After style corrections all 13 capture tests pass again;
strict core/host Clippy passes. One network setup test and three real-model tests
were not repeated for this transport-only change. Evidence:
`/tmp/speakerdesk-capture-{stderr-red,tests,final-tests,clippy}.log` and
`/private/tmp/speakerdesk-capture-reference.6D7BHy/`.

The accepted complete replacement remains active and unfinished. Next join this
transport to the real meeting routes, persistent job authority, live model engine,
publication, language controls, final drain and shutdown. Then complete recovery,
live timing/refinement/identity, hardware capture and long-session performance,
native default packaging/updater, security and full equivalent 0.6.6 measurements.
Tauri long-import/setup acceptance remains pending because CUA still reports a
locked Mac after the user's unlock replies. The existing isolated preview and
optimized desktop executable remain the earlier builds recorded above. No working
app, production data, recordings or existing model cache were changed. Windows and
issue #25 remain follow-up work.

## Native capture, live models and persistent publication joined

The mounted meeting routes now own the capture helper and one serial live model
worker. The old simulated meeting manager and duplicate current-meeting flag are
removed. Runtime state contains handles and progress; the operation map owns
admission, SQLite owns jobs/documents, and the writer owns retained PCM. Inference
receives coalesced committed sample endpoints and reads bounded saved crops.
Qualified dual-source receipts/catalogs are persisted before inference starts, and
published passages retain their source binding independently of editable speakers.

Publication reloads the current document in the existing transaction and preserves
protected passage edits and saved names. Language changes queue the old-language
audio boundary before the new epoch; Stop consumes final VAD/diarization output
before Ready and releases admission only after helper, writer and worker cleanup.
Independent review closed start/terminal replacement and start/shutdown races with
one owned Tokio start gate. A database invalidation failure now cancels every
affected token under the store lock, returns the persistence error and still allows
owned-process cleanup. Duplicate session closes match the real child's idempotent
behavior. No placeholder live refinement capability is advertised.

Validation on the M5 Mac: 154 core tests, 56 host tests, three mounted live-service
replays, one native-service lifecycle test and nine process tests passed. The live
replays cover concurrent starts, live passage/name/boundary edits, later candidates,
language boundaries, raw Pause/Resume, EOF tails, source identities, helper/child
failure, whole-host shutdown, export, restart and stale saves. Strict core/host
Clippy passed; the three live tests passed again after style corrections. Existing
network and separate real-model qualification tests were not repeated. Logs:
`/tmp/speakerdesk-meeting-{full-tests,final-tests,clippy,release}.log`.

The optimized desktop executable built successfully (known upstream mlx-sys
malloc/realloc binding warnings remain), SHA-256
`e3149fb98fa5365a987bd4da98672bb36d6736e7c09c47a88bd7fa6258052be9`.
The real native host then processed the existing synthetic two-voice fixture through
an owned shell capture-protocol replay and the real resident models. All 145,445
PCM16 samples were retained exactly, both expected utterances reached SQLite, and
the final observation covered the partial last frame. Text, speaker-name and
boundary edits survived whole-host restart; TXT export and stale-save rejection
passed. Evidence: `/private/tmp/speakerdesk-live-model-join.1u4zhv/`.
The single warm, unpaced replay reached terminal state in 1.069 seconds; this is a
diagnostic observation, not device, UI, sustained-live or 0.6.6 performance acceptance.
The qualification hosts and their children exited; the older isolated Tauri preview
remains unchanged. CUA still reports the Mac locked after the user's unlock replies.

The complete replacement remains active and unfinished. Next implement recovery
from retained audio after child loss, then live timing/refinement/identity and the
remaining capture boundaries. Qualified Pause currently ends its epoch with saved
audio; qualified Resume and raw dual-source echo still need work. Real device and
Tauri live/setup/long-import checks, long-session memory/backlog, native default
startup and packaging, updater/security qualification and equivalent 0.6.6
measurements remain required. The installed app, production data and existing model
cache remain untouched; Windows and issue #25 remain follow-ups.

UI follow-up: the old isolated preview process and both children were stopped and
their exits observed. The same disposable preview bundle now runs the optimized
728b3c3 executable with the saved-PCM replay helper in its Resources directory.
This is deliberately a UI qualification fixture, not a distributable capture bundle.
Native service `http://127.0.0.1:56552/` reports models ready and capture available;
process IDs at launch were 97307/97339/97340. Log and state:
`/private/tmp/speakerdesk-native-desktop.WJKAde/app-live-join.log` and
`live-join-preview.json`. The editor check awaits the latest unlock request. A
retained-output check also confirms the live result contains exactly the two
expected utterances on distinct speaker tracks, with both passages sealed.

## Exceptional live inference recovery

Live child loss now has a typed error through VAD, diarization, ASR and language
adapters. The failed worker is joined before SQLite advances the stream generation.
Capture keeps writing while a replacement worker rebuilds acoustic state from
bounded reads of committed WAV data. Normal processing does not replay history.
Each source has its own publication cutoff, independent of edited speaker IDs;
new utterance/track IDs use the new stream generation without changing the capture
catalog revision or earlier names. Publication checks operation and stream
generations. Recovery also covers a child lost during final EOF processing.

Independent review reproduced a skipped-tail defect: a passage published through
seven seconds could still contain only words decoded through six. The failing
regression observed a 112,000-sample restart instead of 96,000. Publication now
persists the represented text endpoint separately from growing audio and stale
timing evidence; protected text and blank-result fallback retain that endpoint.
Recovery preserves exact saved words/bounds and marks the interrupted boundary
for review. It makes one automatic attempt per recording. A failed replacement
leaves audio capture running and records an explicit inference failure.

Validation: 157 core and 59 host tests pass, plus five mounted live-service tests,
one native-service lifecycle test and nine process tests. Coverage includes
asymmetric source cutoffs, language history, protected edits, stale publication,
pause/resume during held recovery, child loss in ASR and EOF, replacement failure,
shutdown, restart/export and process reaping. Strict all-target core/host Clippy
passes. The setup network test and three separate model qualification tests were
not repeated. Retained red/green evidence is in
`/tmp/speakerdesk-recovery-coverage-{core-red,host-red,host-green}.log`; other logs
are `/tmp/speakerdesk-recovery-{final-tests,integrations,clippy,release}.log`.
The final unit log contains the earlier incomplete fixture failure; its corrected
host case passed in the focused green log. Reusing the other unchanged passing
tests gives the totals above.

The optimized desktop build passed with the existing upstream mlx-sys warnings;
binary SHA-256 is
`29716bc3b27e5b6dd1922139267364a12b54034c8322e3eb11f1ce95de8f385a`.
The real-model replay in `/private/tmp/speakerdesk-real-recovery.3iHE4V/` records
job `1b646ebf0c7c48c7a86dd39c92626a91`: pause, edit words/boundary/name, kill owned
child 15175, resume saved synthetic PCM, replacement child 15293, Stop, whole-host
restart and TXT export all passed. Both expected post-recovery utterances reached
the store; all 290,890 retained PCM samples match the two source clips exactly.
The final observation reaches sample 290,890 and earlier edits remain unchanged.
Both qualification hosts and their children exited. The shell helper reads only
the retained synthetic fixture. Python is used by the external qualification
harness, not by the application or helper.

One unpaced run took 4.753 seconds from resume after child loss to terminal state;
the recorded 4,105 ms recovery interval covers prefix replay/lazy initialization
and uncovered-tail decoding, excluding the old-worker join and SQLite transition.
These are diagnostics, not sustained-live, device, desktop or 0.6.6 performance
acceptance. Recovery deliberately uses new speaker tracks and flags uncertainty.

The accepted complete replacement remains active and unfinished. Next address
ordinary live publication after editing an open passage: protecting its text can
currently suppress later words while its audio grows. A single restart frontier
cannot repair older coverage gaps once later passages exist; this recovery slice
does not claim general gap repair. Then complete live timing/refinement/identity,
qualified Pause/Resume and raw dual-source echo, physical capture/long-session
measurements, the pending Tauri live/setup/long-import workflow, default native
startup/packaging, updater/security qualification and equivalent 0.6.6 gates.
The installed app, production data and existing model cache remain untouched;
Windows and issue #25 remain follow-ups.

## Live edits preserve subsequent speech

An independent real-model replay reproduced lost content after editing an open
passage: the recording reached Ready with all 293,127 PCM samples retained, but
none of the later words published. Editing live text now records the endpoint
represented by those words. Publication feeds that boundary back to the serial
live engine, which seals the protected prefix and decodes the remaining audio as
a separate passage. Manual text entered before the first decode uses the displayed
saved-audio boundary. Protected bounds and speaker names survive. No recording
replay, extra document authority or new control queue is involved.

Both halves retain only their own speech and speaker activity. This matters when
the final decode races an edit across a speaker change after the engine has pruned
its global turn history. Continuation IDs are stable across packet splits and
language boundaries; EOF and repeated feedback do not duplicate them. Removed the
unused per-utterance machine-version copies.

On the M5 Mac, 160 core, 59 host, six mounted live-service, one native-service,
nine process and 14 xtask tests pass. The mounted live-edit test groups text-only,
protected bounds, held ASR, empty-before-ASR, language-change and child-recovery
cases; it also checks restart, export and stale saves. Strict all-target core,
host and xtask Clippy passes. Four external/network/model qualification tests were
explicitly ignored; unchanged model proofs were reused. Logs:
`/tmp/speakerdesk-live-edit-{tests,clippy,release}.log` and the focused
`matrix-core` / `matrix-host-fixtures` logs. The initial host matrix invocation
without the fixture feature ran zero tests; only the corrected fixtures log is
evidence for that matrix.

The optimized executable built successfully with the known upstream mlx-sys
binding warnings. Its SHA-256 is
`736426c07b856bb14dcaadc0028e6ba08b9b2c887d5b2698e2a9f3d2bce79783`.
The same synthetic Samantha speech replay now publishes the later afternoon
meeting, delivery schedule, after-lunch and project-team text. Exact edited words,
start bound and Casey speaker name survive Stop, whole-host restart and TXT export.
All retained PCM bytes match the source. The qualification hosts and children
exited. Before/after input, documents, logs and export are retained in
`/private/tmp/speakerdesk-live-edit.deNKoi/`; the successful job is
`dab840a2f01f44bfb9bebcd021d55748`. This is real model execution through the native
capture/service path with a saved-PCM helper, not physical capture, Tauri interaction
or a performance comparison. Python runs only the external qualification harness.

The isolated Tauri preview was refreshed earlier to 3eb78e7, with the same
disposable data and replay helper; its service at `http://127.0.0.1:59923/` reports
ready models and capture availability. State/logs are `recovery-preview.json` and
`app-recovery.log` in `/private/tmp/speakerdesk-native-desktop.WJKAde/`. CUA still
reports the Mac locked after the latest unlock reply, so no new editor claim is
made. GitHub PR #27 remains open at head
`db9da655b3eeeeef598ba21a21cb54505e6bce21`, base
`59e7b0c860f72ca94c24e6eab18ac820b144ee24`; these local changes are not pushed.

The complete replacement remains active and unfinished. Next complete live
timing/refinement/identity and source-specific saved-audio consumers, then qualified
Pause/Resume and raw dual-source echo, physical capture and long-session behavior,
the pending Tauri live/setup/long-import workflow, default native startup and actual
Python-free packaging, updater/security qualification and equivalent 0.6.6
performance gates. The installed app, production data and existing model cache
remain untouched. Windows and issue #25 remain deferred.

## Actual setup and physical capture on October 12

The native Tauri setup flow now passes with an independent disposable SQLite
store and APFS clones of this task's installed test models. Silero was absent at
startup, so Start meeting was disabled. Resume download entered verification;
Cancel setup reached its terminal cancelled state. Resuming downloaded the
pinned Silero files (1247300 bytes total), activated them and initialized all six
capabilities. The settings UI showed Installed/Ready. Importing `two-voice.wav`
through the native Open dialog and Transcribe recording produced both expected
passages, two speakers and timed reading turns in the editor. People stayed
unchanged. The complete app and its model child exited on Cmd-Q.

Evidence is in
`/var/folders/8s/9n6rxkrx6dv46zhjfnhg69rw0000gn/T/speakerdesk-setup-ui.fvat1u4i/`:
`cancelled.json`, `ready.json`, `transcript.json`, `people-before.json` and
`result.json`. This uses the pre-progress-fix `e995d34` executable. Only Silero
was freshly downloaded; the other five models were cloned and reverified.

The joint-progress release build completed in 27.38 seconds with the two existing
upstream mlx-sys warnings, SHA-256
`2f7cf4b6cd27c2ac75135d6408c2eeb4ae4ec30bd7268afe793d57acd7522e30`.
The isolated preview was refreshed with that executable, the already-qualified
Swift capture helper, the WebRTC echo library and microphone/audio privacy
descriptions. The signed preview executable SHA-256 is
`32e09eef9f2ee80e5ba4fd29d3bb2c0969c3b948e184101fa04531561ff9a42f`;
signed helper SHA-256 is
`3cca87f83380d6b186711861de09dc0e3bb3c8562b75aedac2d3bdb554c89ef2`.
The capability receipt is bound to those helper bytes.

After Albert confirmed a quiet room and no unrelated app audio, the actual
Tauri Start meeting used both microphone and Mac audio. The Swift helper reports
active Apple voice processing, AGC disabled, a 48 kHz nine-discrete-channel input
with exact replicas, and a 48 kHz stereo output route. The model pipeline produced
the two expected phrases from synthetic playback. Pause acknowledged 560176
received samples, 560128 speech-observed samples and 552960 jointly available
samples. Resume retained the qualified graph and captured the repeated fixture;
Stop drained all five WAV files to exactly 1022452 samples (63.90325 seconds).
The four expected system passages reached the editor. A correction to the first
passage saved through the actual passage editor.

One additional 464 ms microphone fragment produced `you`, with `review=true`
and unresolved refinement. Its evidence is retained; this is not a perfect
recognition claim. The stopped editor incorrectly labelled this provisional row
Live/awaiting more context. Shared passage grouping now refreshes the existing
heading to Needs review, with saved-audio review help and matching accessibility
text, when capture is no longer active. Active capture wording is unchanged.
The regression failed before the fix and now preserves the focused card, words,
caret, document and Retry control across the stopped-state transition. Root ran
the full frontend suite: 105 passed, one existing optional performance fixture
skipped. Logs are `/tmp/speakerdesk-stopped-review-{frontend,release}.log`.
No real-device acoustic-quality, echo calibration, long-session or performance
parity claim follows from this functional check.

The device job is `0ad018d9920545b2b581388502d392bc`. Original PCM, processing
receipt, Pause snapshot, stopped/saved documents, uncertainty evidence and
candidate hashes are in `/private/tmp/speakerdesk-device-capture.vhdapugw/`.
The capture helper exited after Stop; Cmd-Q then reaped shell/host/model PIDs
23305/23310/23311. The updated preview then reopened this exact job: the saved
correction was visible and the uncertain fragment appeared under Needs review.
The canonical document, people catalog and every WAV hash were unchanged.
The renderer-only release build completed in 18.98 seconds, SHA-256
`def12c199385c501cf18f707d250120ad68b36a943aded342fbb0ef16e8d1fc1`;
its signed preview executable is
`24e4f1b1c32c218603e1bf1fda83faea3b4b9c85acf851caa73c87f83a553ba4`.
All audio and data belong to the disposable test store; the installed app and
its data/model cache remain untouched.

## Nine-minute live comparison

The optimized `2e98b6a` native service and copied 0.6.6 service each consumed the
same 8713010 samples (544.563125 seconds), ten successive copies of the retained
synthetic fixture. Runs were sequential, native then baseline, with fresh
disposable stores, pinned checkpoints, explicit English, microphone-only replay
and the accuracy pass enabled. No task build or other model run overlapped them.
This uses the paced saved-PCM helper, not physical capture or Tauri rendering.

| Diagnostic | Native | 0.6.6 |
| --- | --- | --- |
| Start to first persisted words | 8.378 s | 9.799 s |
| Start to Ready | 547.691 s | 548.746 s |
| Stop to Ready | 3.091 s | 2.259 s |
| Pending audio, median / maximum | 0.880 / 2.960 s | 0.950 / 5.590 s |
| Sampled peak owned physical footprint | 3.585 GB | 4.963 GB |
| Sampled process CPU lower bound | 133.032 s | 197.211 s |

Both outputs contain the same 1691-word sequence. Against the supplied 1700-word
script, both replace one article in each repetition and omit the same conjunction
at the nine joined boundaries. Both preserve every PCM sample and complete their
accuracy pass. Native retains aligned timing for every spoken passage, including
the last sample; baseline again lacks complete timing evidence on the final row.
This output difference remains part of interpreting Stop latency. Native's later
two-minute memory-window medians remain around 3.15–3.16 GB; baseline's rise from
4.00 GB in the first window to 4.20 GB in the final window.

This is one run per implementation at this duration, not evidence for arbitrary
session lengths or the entire performance gate. The Stop increase is retained;
the roughly one-second total-latency difference is not a repeatable speed claim.
Maximum observed sampling intervals were 0.130 s native and 0.181 s baseline.
All owned hosts, workers and replay helpers exited and were reaped. Inputs,
documents, process samples, word differences and `comparison.json` are retained
under `/private/tmp/speakerdesk-live-duration.09lje54v/`, in runs
`native-1791806141029214000` and `baseline-1791806741780652000`.

Albert also reported that ordinary startup presents downloaded models as a
disabled Download models action while they initialize. The existing DOM test
reproduces the visible action during loading; its red result is
`/tmp/speakerdesk-loading-state-red.log`. The host and both startup/settings views
now distinguish loading from downloading. The signed package's subsequent actual
startup displayed “Loading models…” without the download card or action, as
recorded in the current qualification section above.

## Joint live progress and the first equivalent live diagnostic

The first equivalent saved-PCM service comparison exposed a progress defect:
native audio updates reported the Silero endpoint while publication also waited
for diarization. Pause/Stop reported all received samples. The unchanged editor
uses this value for its behind/caught-up display, so it could hide pending speaker
observations. All three progress events now report the minimum of received,
Silero-observed and diarization-observed samples. Received and canonical EOF
endpoints, model calls, publication and accuracy work are unchanged.

The existing delayed-diarization test was extended before the fix. It failed with
4.992 seconds processed although diarization had observed zero. The repaired
audio and Pause events retain zero processed while receiving 80037 samples;
Stop reports 80000 jointly observed samples, and canonical publication still
preserves all 80037 samples. Coordinator verification passes 177 core tests,
18 mounted live tests (one ordinary-run native-echo exclusion) and strict
core/host all-target Clippy. Logs are `/tmp/speakerdesk-joint-progress-{red,core,live,clippy}.log`.

The pre-fix optimized candidate `e995d34` was measured against the copied 0.6.6
runtime with the same 54.4563125-second fixture, pinned model files, explicit
English, microphone-only source and enabled accuracy pass. Each run starts with
a fresh disposable store. The external helper emits synthetic PCM on its audio
clock; Stop follows a producer EOF marker. Tauri rendering and physical capture
are excluded. The first order was baseline/native/native on October 11; after an
interruption the second baseline ran on October 12. No app/build workload from
this task overlapped these measurements.

| Diagnostic | Native, two runs | 0.6.6, two runs |
| --- | --- | --- |
| Start to first persisted words | 7.850 / 8.452 s | 10.008 / 10.081 s |
| Start to Ready | 57.199 / 58.065 s | 58.291 / 58.583 s |
| Stop to Ready | 2.890 / 3.750 s | 1.952 / 1.959 s |
| Sampled peak owned physical footprint | 3.562 / 3.541 GB | 4.698 / 4.689 GB |
| Sampled process CPU lower bound | 19.968 / 20.794 s | 16.870 / 16.496 s |

Every run completed refinement, preserved all 871301 PCM samples and returned the
same 170-word sequence (one article differs from the source text). Native has
complete aligned timing with current text, audio anchor and revision on both
passages. Baseline has reading-turn evidence only on the first passage and no
reading alignment field; its final passage has no reading turns. These outputs
are not equivalent timing work. The native Stop/CPU increase remains visible;
neither it nor the improved first-text/memory values qualify the full performance
gate. Pre-fix native pending-seconds values are invalid for backlog comparison.
Post-fix backlog, sustained long sessions, package/UI latency, cold/warm loads,
physical capture, cancellation and recovery performance remain to qualify.

One post-fix native run (`native-1791805861056236000`) verifies the actual joint
progress path: median pending 0.840 seconds, maximum 2.500 seconds, versus
baseline medians around 0.890 seconds in the earlier retained sample. It retained
identical PCM and words, completed both timing results and reaped all owned
processes. First text was 7.889 seconds, Stop-to-Ready 2.894 seconds and sampled
peak physical footprint 3.528 GB. This reuses the same short diagnostic; it does
not establish sustained long-session or package performance acceptance.

Inputs, logs, process samples, final documents and `comparison.json` are retained
in `/private/tmp/speakerdesk-live-comparison.tBmzLh/`. The process sampler includes
the helper and all observed application descendants; every owned process was
reaped. Peak memory is sampled and CPU a lower bound; maximum observed sampling
intervals reached 0.653 seconds. An initial derived producer-to-ready number used
two Python 3.9 monotonic clocks, which are not comparable across Mac processes.
It was removed from interpreted measurements and the harness, with an explicit
note beside retained raw stdout. Driver-local latencies and the marker-existence
Stop trigger are unaffected. No expensive rerun was used to repair that metric.

## Incremental activity and the live/long desktop workflows

The growing-utterance activity path previously rebuilt its complete history on
every observation, even before another ASR request was due. A retained failing
test records six scans beginning at sample zero and visiting 90 through 105
turns while a recording grows from 30 to 35 seconds. The implementation now
reuses each utterance's observed activity prefix, recomputes its last interval
and new suffix, and selects overlapping turns from end-sorted validated batches.
Growing an utterance retains activity; shrinking clips it. Observations delayed
past a new EOF row retain their truthful watermark. Publication and ASR cadence
are unchanged. Whole growing-document serialization is outside this claim.

The regression compares every prefix and the partial EOF with the unchanged
full-sweep implementation using overlapping and duplicate speakers, delayed
observations and original emitted turns. It also checks bounded historical
visits before the next ASR deadline. Independent coordinator checks pass:
177 core tests, 18 mounted live tests (one existing native-echo test excluded
from the ordinary run), and strict all-target core/host Clippy. The native-echo
gate and unchanged host/process/frontend proof from the preceding slice are
reused. Logs are `/tmp/speakerdesk-growing-activity-{red,core-tests,live-tests,clippy}.log`.

The optimized Mac build completed in 35.61 seconds with the existing upstream
mlx-sys warnings. Executable SHA-256 is
`5bd0a2f1185998678366402419d461fde236161d3312986366c720c64d10560c`;
the isolated, ad-hoc signed preview executable is
`b6677b0559fae1b32665255ed5431a66acda8d72f5da5129466424257781ac17`.
The following actual Tauri checks use this candidate:

- Live job `b376906fe0524060bcb55bfb2208f722`: Start, independent capture and
  accuracy Pause/Resume, edit passage words, apply the name Taylor Native without
  enrollment, Stop, adjust the first boundary to 0.010 seconds, save and play.
  All four native export dialogs wrote the corrected words/name. A stale HTTP
  save returns 409. Full Cmd-Q reaped the shell, host and inference child;
  reopening the meeting visibly restores the correction, name and boundary.
  Its canonical document and people catalog are exactly unchanged after restart.
  The retained recording contains all 145445 input samples byte-for-byte.
- Long import `bd0a9e4c9aee4c4dbca4b9b2595e87ca`: the native Open dialog imported
  `three-windows.wav` (82.0099375 seconds, 1312159 samples). Real models produced
  ten visible passages, including the final partial frame. Editing the last
  passage, saving, quitting the complete app and reopening preserves the exact
  canonical document and displays the correction. Saved PCM is exact. There are
  twenty canonical rows including silence; the UI correctly counts ten passages.

Evidence is retained in `/private/tmp/speakerdesk-native-desktop.WJKAde/`:
`live-ui-result.json`, `live-ui-saved.json`, `long-ui-result.json`,
`long-ui-saved.json`, the four `Native_live_editor_qualification` exports, and
the corresponding app logs. The long source WAV SHA-256 is
`3297950fe04a7a7310436f2afaa2f6a0f49163c6c5a176871a3337a90d70f6a8`.

These are real model and editor checks with synthetic audio. The live helper
replays saved PCM and supplies its entire nine-second input before Pause; it
does not prove physical capture, continued input during Resume or realtime
performance. The repeated long fixture does not establish diarization quality.
The preview still uses a development MLX library path and resources-disabled
build; default startup/packaging still uses Python. The accepted replacement
outcome remains active: physical capture, setup UI, long-session and equivalent
performance gates, native default packaging and disposable updater/security
qualification remain. No installed app, production data or existing model cache
was changed; nothing was pushed or published.

## Rolling accuracy during capture

The existing serial live worker now advances one bounded accuracy context between
live polls. Capture and its single writer continue saving audio. Saved and rolling
accuracy share the same context composition and publication rules; rolling work
uses fresh VAD sessions and never feeds a live model handle. Discovery coalesces
changed sealed rows into a cursor instead of repeatedly loading every job on each
audio notice. Pending work retains the audio/epoch snapshot, not a second document
history. The existing frontend accuracy controls now work during recording.

Pause persists a generation before cancellation, and Resume waits for the terminal
result. Stop invalidates active accuracy, drains live EOF, then handles remaining
eligible anchors under the same lease. Accuracy Pause remains available during
final draining. Child loss retains typed lifecycle errors and reconstructs from
saved audio through the existing bounded recovery path. Protected edits, source
changes, language epochs and earlier accepted accuracy results survive later live
publication. Unrelated future language switches do not invalidate an earlier
anchor. Completed and failed versions are checkpointed through restart.

The real-model run exposed an existing saved-refinement bug: positively classified
non-speech rows carry an editor review marker, which rejected otherwise complete
speech separated by silence. Refinement now accepts only empty spans explicitly
classified as model non-speech, digital silence or a constant signal. Unknown or
unobserved coverage, uncertain language, short acoustic context and incomplete
decodes still reject publication. Import/retry review behavior is unchanged. A
test-only red reproduced the failure. An older test was passing because of its
silent gap: the alleged short tail actually had 3,744 physical context samples.
Its assertions now distinguish complete speech plus silence, a truncated second
decode, and a genuine 160-sample speech crop.

Final M5 checks: 176 core, 64 host, 18 mounted live, one native-service lifecycle
and nine process tests pass; strict all-target core/host Clippy passes. The earlier
unchanged frontend candidate passed 32 Node assertions with one existing long
performance skip. The native raw-dual mounted gate also passed; other unchanged
native/model proofs are reused. Logs are
`/tmp/speakerdesk-rolling-accuracy-{final-tests,final-clippy,frontend-all,raw-mounted}.log`
and `/tmp/speakerdesk-rolling-silence-red.log`. A first invocation used a nonexistent
`test-fixtures` feature and ran no tests; the corrected `fixtures` run is the result
reported here. The optimized build took 27.15 seconds with the two known upstream
mlx-sys warnings; executable SHA-256:
`0dafb7e6aedf3cceab5dacc96f71693c2754b088b0da8666fd4a4346b679b23c`.

Real MLX/CTC qualification used 54.4563125 seconds of synthetic Samantha speech,
a pause and a 37-sample final tail: 871,301 samples, WAV SHA-256
`54c4f45c915fa8d5517f6487abc021449ee743ee86cf4fb46a5f4b05b3dcae59`.
Job `9247280953bf40ae9ec7d81eedd89055` published the first 114-word accuracy result
while recording continued. Its edited words, start boundary and speaker name
survived Stop and full-host restart. The silence correction then successfully
retried only the retained failed tail; the protected first row stayed identical.
The final result passes whole-transcript save, stale-save 409, full-host restart,
all four exports, unchanged people and exact PCM comparison. No recapture was
needed. All qualification hosts and children exited. Inputs, scripts, rejected and
accepted documents are retained in `/private/tmp/speakerdesk-rolling-accuracy.62j29J/`,
especially `after-corrected/result.json` and `tail-result.json`.

The first external driver incorrectly waited for the writer's final buffered tile
before sending Stop; the corrected driver sends Stop and verifies draining. A
second driver asserted perfect reference wording although its one article change
also occurred before this implementation. Those failed assertions are retained;
neither is represented as a successful complete workflow. The final continuation
and tail retry above verify the accepted behavior. This is real inference through
the mounted native service with an external saved-PCM helper, not physical capture,
Tauri interaction, a packaged artifact or an equivalent performance measurement.

## Saved long multilingual accuracy

Long saved passages now use the same bounded adjacent-context selection for every
supported language. Each context must resolve confidently to the same language;
uncertain or mixed routing retains the prior transcript. Native CTC results keep
their language's existing calibration identity. Non-English results explicitly
retain functional timing scope and an empty calibrated-language list, with no
English accuracy claim. Raw Unicode and the selected right-context separator are
preserved, including adjoining CJK units and non-ASCII whitespace. Existing
ambiguous acoustic-boundary rejection remains unchanged. The adapter's Japanese
separator aliases are validated but never become independent boundary anchors.

Independent red controls reproduced both the Composer's English calibration
assumption and the mounted saved route's English-only guard. Tests now cover
German Auto, Chinese explicit selection, cross-context language changes,
uncertainty, wrong provider/calibration/policies, protected late edits, pause,
reopen and idempotent resume without more model calls. Retained independent
Japanese/Arabic normalization scores are replayed through the context parser.
Exports and code-point/UTF16 offsets match the selected raw text. The existing
raw-unit helper supplies UTF16 offsets; no duplicate whole-text offset buffer or
new scheduling/cache layer was added. Setup and editor wording match this scope.

All 174 core, 64 host, 14 mounted live, one service and nine process tests pass,
with the existing explicit native/network ignores. The retained raw AEC proof
is reused. Eight relevant Node checks and strict all-target core/host Clippy
pass. Logs are `/tmp/speakerdesk-long-multilingual-{core-red,mounted-red,core-tests,host-tests,frontend,clippy}.log`.
The locked optimized desktop build passes in 27.00 seconds with the two unchanged
upstream mlx-sys warnings. Binary SHA256:
`5a09574ad6805b0b6b10de4b914c86d245f213ca98252b7dd8c805d9bc6b8727`.

The independent real-model input is 54.953 seconds of generated German speech,
879,248 samples including a partial tail, with 136 authored expected words. Its
WAV SHA256 is `f06c41ff8831a1311d5a3f3c70192c6875cca5b2b7d5ee00a4c3fcce048af85a`.
The pre-change binary produced a canonical passage through paced saved-PCM capture
but rejected saved refinement and preserved its words. The corrected binary
refines that same store and audio: all 136 expected words remain in order, and
functional German timing reaches reading turns. Speaker-name, word and boundary
edits, whole-transcript save, history, stale-save HTTP 409, full service restart,
protected resume and all four exports pass. PCM and voice profiles stay unchanged.
The external harness initially retried refinement before model readiness after
restart and correctly received 409. Fixing only that harness race completed the
restart checks without repeating inference. Evidence is under
`/private/tmp/speakerdesk-long-german.jtRhdA/`; job
`92799e77ff984d5680b21f623f8a313f`. Qualification hosts and children exited.

This is real MLX/CTC through the mounted service using synthetic saved audio,
not physical capture, UI proof, non-English population accuracy or an equivalent
performance comparison. The complete replacement remains active and unfinished:
rolling accuracy, long-session and physical capture gates, actual Tauri live/setup
workflows, default native startup and packaging, disposable updater/security
qualification and equivalent 0.6.6 performance remain. The Mac UI tool continues
to report a locked screen. No installed app, production data, existing model cache
or GitHub state was changed. Windows and issue #25 stay deferred.

## Raw dual-source echo and recoverable capture failures

Raw dual-source capture now loads the existing pinned ABI-3 WebRTC component from
an explicit canonical path. Rust owns its bounded frame queues, reference clock
and routing policy. Native state is created, called and destroyed on the recording
writer's blocking thread; no unsafe Send implementation or C++ model code was
added. Qualified Apple voice processing bypasses this component. Its receipt stays
distinct from the raw capture source catalog.

Original microphone, system and stereo reference tracks commit before DSP. Only
completed clean audio advances inference and playback. Process/reset failures
terminate that DSP instance while draining accepted original packets; persisted
raw and clean sample endpoints survive reopening. Missing, incompatible or
untrusted runtime files fail before PCM creation. Pause and format changes drain
partial frames and reset the existing native state. Tauri supplies its bundled
dylib path; the Mac CI job builds it through the existing script and runs the
focused native boundary tests.

The pre-change raw 513-sample test failed at writer creation. Independent replay
now compares eight cases with the actual v0.6.6 SourceMixer and echo wrapper:
echo, latency, drift, absent echo, near-only speech, stereo paths, inverted
uncorrelated paths and a partial Pause/Resume tail. Original track bytes match
exactly in all eight; clean/mixed bytes match exactly in the four non-echo cases.
Echo outputs are numerically close, not bit-identical: maximum PCM16 differences
are 1, 2, 1 and 170 units respectively for echo, latency, drift and stereo paths.
All unchanged 0.6.6 acoustic quality gates pass. Stereo clean RMS difference is
3.0474 PCM16 units; its native ERLE is 20.51355 dB and near-speech SNR 21.94628 dB.

The native queue test feeds 500 arbitrary splits and verifies bounded held input,
exact final draining and a fresh 513-sample resume. A mounted router/SQLite/owned
process test covers raw Pause/Resume/Stop, missing library, ABI mismatch, process
failure and reset failure. Raw sources, clean-only playback, retained failure
endpoints, restart and owned-process cleanup pass. Model output in these mounted
tests is explicitly replayed. The small fault-injection C library is test-only.

The full gate passes 172 core, 16 echo, 64 host, 14 mounted live, one service and
nine process tests. Four new native gates were run explicitly; the unchanged
network and three separately qualified model tests remain ignored. Strict
all-target core/host/echo Clippy, workflow lint and diff whitespace checks pass.
Logs are `/tmp/speakerdesk-raw-echo-{red,green,artifacts,reference,queues,mounted,tests,clippy}.log`.
The independent inputs, baseline extraction, outputs and metric comparisons are
retained under `/private/tmp/speakerdesk-raw-echo-reference.PYKutc/`. These are
correctness and acoustic quality replays, not application performance measurements.

The existing DSP build uses WebRTC revision
`852150686748f5cce0c1c96f4fde50e6be43ef73`; its source lock and C++ implementation
are unchanged from v0.6.6. Dylib SHA256:
`09f3673e48c2256d25f671118922c078341035c7f5a69e66c2c2a77121e2a815`.
Its loaded dependencies are Apple/system libraries, with no Python runtime.
Building this component does not establish the complete Python-free package.
The locked optimized desktop build passes in 25.02 seconds with the two unchanged
upstream mlx-sys warnings. Binary SHA256:
`0d468bb1efc1f7fa8dcedd0329c674d7a8dd622020e1e15bab982eb886d60308`.
The compile check disables bundle resources; packaging remains a separate gate.

The UI tool still reports the Mac locked after the latest unlock reply. Physical
capture, actual editor checks for these changes and sustained performance remain
unverified. The accepted full replacement remains active: finish non-English and
rolling accuracy, long-session behavior, actual desktop/capture gates, native
default startup and packaging, disposable updater/security checks and equivalent
0.6.6 performance. The installed app, production data and existing model cache
remain untouched. Windows and issue #25 remain deferred.

## Qualified capture Pause/Resume

The original protocol replay failed because qualified Pause terminated capture
with “restart recording.” The retained Swift graph now drains at Pause and resumes
only after a fresh microphone callback confirms the original formats and
voice-processing readbacks. Device checks run before and after engine start on
the control path. The first accepted callback advances the shared clock from the
saved pause endpoint, excluding pause/negotiation time and rejecting older queued
callbacks. The same stream, recording files, source catalog and receipt survive.

The Rust writer requires an internal parent-owned Resume before accepting fresh
qualification. Helper JSON cannot arm that transition. Repeated Resume conflicts;
Stop retains admission until drain/reap, even if Recording arrives afterward.
Qualified microphone audio continues draining while system startup is pending.

Independent verification reproduced the old failure, then passed two cycles with
511-, 513- and 17-sample blocks: microphone, clean microphone, stereo reference and
the 0.6.6 anti-phase downmix retain exact expected bytes, with no late samples.
Hostile cases reject missing, duplicate and changed receipts without replacing
the original receipt or losing accepted audio. The mounted router/SQLite/process
matrix covers two resumes, device failure, Stop before acknowledgement and late
Recording after Stop; owned processes are reaped. The replay helper now models
real paused Stop without inventing an extra 513 samples. Its error packets use
the actual Swift field name; the initial mounted failure exposed that fixture bug.

The full Rust gate passes 172 core, 63 host, 14 mounted live, one service and nine
process tests (the same four explicit network/previous hardware test ignores).
Strict all-target core/host Clippy passes. The locked optimized desktop build
passes with the two unchanged upstream mlx-sys binding warnings; binary SHA256
is `9b3d3c37d96904d68db9ad2b211d5752c0fcd1259b7e64a3b753ed969ba7320b`.
Four Swift harnesses pass startup, route-format, converter and epoch controls;
these execute native predicates/converters without opening devices. The optimized
production Swift helper compiles cleanly; metadata-only `--check` reports capture
false. SHA256: `99601766788b8d339ff4bc38a94a0171096163c1546be43dc1b57090588eb983`.
Evidence is in `/private/tmp/speakerdesk-qualified-resume.p2kB7G/` and
`/tmp/speakerdesk-qualified-resume-{red,green,mounted,mounted-green,tests}.log`.

The UI tool still reports the Mac locked. Physical graph/device Pause/Resume and
the actual editor workflow are unverified for this slice. Replay and native
control success do not establish capture quality, acceleration or performance.

## Bounded repeated live decoding

The existing decode-progress owner now retains accepted disjoint-core text,
language decisions and the completed sample endpoint. Growing audio replans only
the unaccepted suffix. The last core remains replaceable, incomplete results stay
retryable without erasing published words, and protected continuation and sealed
cleanup release the associated progress. No PCM history or second document store
was added. Whole-open activity/projection scans are not bounded by this change.

The counted-model replay failed before the change: at 36 seconds it decoded two
18-second crops, including the already accepted origin. Three focused tests now
pass for 30/36/42-second growth, unique acoustic ownership markers, incomplete
prefix/tail retry, split EOF, retained mixed-language decisions, source identity,
protected edits and language boundaries. Fixtures distinguish received audio from
Silero's complete-frame observed endpoint. The mounted mixed-language replay also
checks that retained English probes are exactly the original values while only
the final tail receives new German decisions. The recovery matrix was corrected
to distinguish an already sealed protected prefix from an interrupted open tail;
its former unconditional review flag assumption failed deterministically.

The final gate passes 172 core, 61 host, 13 mounted live-service, one service
lifecycle and nine process tests, plus strict all-target core/host Clippy. Four
existing explicit Rust ignores remain. Logs are
`/tmp/speakerdesk-live-growth-{red,green,recovery-red,recovery-green,tests,clippy,release}.log`.
The unused language-probe sequence fixture branch was removed.

The locked optimized executable built in 23.46 seconds with the existing upstream
mlx-sys warnings. SHA-256:
`123b0fbd465cdaeeb7e316dacdaed97c46c1c97c72c5facae4fc253d1c293f33`.
The retained 34.9135625-second synthetic speech passed paced capture through real
MLX models with all 114 independently authored words in exact order. All 558,617
PCM samples remain byte-exact. Manual-language evidence, the single canonical
identity, full host restart and TXT/SRT/VTT/JSON exports pass. The successful job
is `cf11d9e804e547d9ae287fcf8ccaeae8`, canonical row
`utterance-6ed1f93802b38185db42b85f`. Evidence is in
`/private/tmp/speakerdesk-long-refinement.USlLQY/growth-after/`, the external
qualification harness is `growth.py`, and its log is `growth-after.log` in the
parent directory. All qualification hosts, children and replay helpers exited.
This proves real-model service behavior with saved synthetic audio, not physical
capture, Tauri interaction or an equivalent performance measurement.

The complete replacement remains active and unfinished. The isolated Tauri
preview remains the preceding long-context candidate; no new UI claim is made
while CUA reports the Mac locked. Rolling accuracy, affected-history work,
qualified Pause/Resume and raw dual-source echo, physical capture and long-session
gates, pending Tauri workflows, native startup/package, disposable updater/security
qualification and equivalent 0.6.6 measurements remain. Installed app, production
data, model cache and GitHub are untouched; Windows and issue #25 are deferred.

## Ordinary live language publication

Ordinary short, routed and disjoint-core live decodes now retain typed language
decisions in the existing utterance state and serialize them through the shared
document adapter. The captured Auto/manual mode remains separate from the actual
selected language. Long Auto rows no longer publish the mode string `auto` as a
detected language. Mixed results retain per-probe decisions without an invented
single-language probability; the editor describes them as multiple languages.
Known digital silence, constant signal and model non-speech do not erase a spoken
language, while uncertain speech retains its review reason. Protected prefixes
keep their original evidence, and new continuation starts with fresh metadata.

The mounted red test reproduced a completed Auto transcript with null detection
metadata. The corrected seven-case matrix covers short/long Auto, manual mode,
mixed decisions, uncertainty, edit/epoch change, child recovery and reopen. Core
tests additionally cover verified non-speech versus unknown speech, and a routed
recent-context decision at a seven-second crop origin whose nested context endpoint
remains exactly ten seconds. Read/reopen checks assert no new probe calls.
Final results: 169 core, 61 host, 13 mounted live-service, one service-lifecycle
and nine process tests pass; strict all-target core/host Clippy passes. Four
existing explicit Rust ignores remain. Reading-turn frontend checks pass 16 tests
with one existing source-fixture-dependent skip, including the mixed-language
label. Logs are `/tmp/speakerdesk-live-language-{red,core-green,silence,host-green,tests,clippy,frontend,release}.log`.

The locked optimized build completed in 25.24 seconds with the existing upstream
mlx-sys warnings. Binary SHA-256:
`6c4676bf7d672122659c8ebcc8c8705d33017e4e35c149b380f30a27749a0388`.
Real MLX execution on the existing synthetic 18.3204375-second recording passes
capture replay, Pause/edit/name, Resume/Stop, whole-host restart and export.
Job `474aad8710ef423bbf40984671af76a3` keeps three original probes with the edited
prefix, and publishes four continuation probes covering samples 115,200–293,127.
Both rows retain Auto mode and actual English language. All 293,127 PCM samples
remain byte-exact, and all expected continuation markers remain. Evidence and the
external Python test harness are in `/private/tmp/speakerdesk-live-language.wy1io308/`.
All qualification hosts and children exited. This is real-model service evidence,
not physical capture, UI, population-quality or performance qualification.

The running isolated Tauri preview remains the preceding long-context candidate
at `http://127.0.0.1:57677/`; no new UI claim is made while CUA reports the Mac
locked. The accepted full replacement remains active: growing live work and
rolling accuracy, long non-English accuracy, qualified Pause/Resume and raw
dual-source echo, actual capture and long-session gates, pending Tauri workflows,
native default startup/package, disposable updater/security checks and equivalent
0.6.6 measurements. Installed app, production data, existing model cache and GitHub
remain untouched. Windows and issue #25 remain deferred.

## Saved long-context accuracy

Saved English canonical anchors now pass through bounded physical contexts rather
than the former 24.5-second rejection. The existing recording reader supplies at
most 20 seconds per crop; Cohere receives the baseline's 3,200-sample padding only
at internal physical edges. VAD, language probes and supplied-text CTC retain the
original unpadded coordinates. One composer keeps adjacent parsed contexts and
the selected output. It does not retain PCM history, recursively validate earlier
packets or add an authority store. Bidirectionally unique acoustic cells select
raw-text boundaries without deduplicating words by spelling. Final word evidence
uses the existing timing attachment under the final passage identity/revision.

The retained red checks reproduced both the unavailable overlap assembler and
the real route's duration rejection. The actual live path uses disjoint decode
cores: its dormant authoritative assembler was unfinished integration, not proof
of lost live words. The initial unpaced qualification helper overflowed its
bounded capture queue; a paced replay then retained all 558,617 samples. That
harness failure is not a demonstrated sustained-capture regression.

Independent source review caught valid NFC targets sharing raw character spans;
the final normalization regression uses the real alignment materializer. Focused
checks cover punctuation, a non-grid audio origin, a partial final frame, distinct
repetitions, missing and genuinely ambiguous timing. Mounted child replays cover
complete publication, an edit during the last context, cancellation, bounded
crops, timing, export and idempotent restart. Final validation: 167 core, 61 host,
12 mounted live-service, one service-lifecycle and nine process tests pass, plus
strict all-target core/host Clippy and four capture-presentation Node tests. The
four existing explicit ignores remain unchanged. Logs are
`/tmp/speakerdesk-long-refinement-{core-red,host-red,core-green,normalization,host-green,tests,clippy,controls,release}.log`.

The locked optimized build completed in 24.39 seconds with the existing upstream
mlx-sys warnings. Executable SHA-256:
`ba961b39a6e00a7cebf43a2e3c9b7080286f50b7b629ca6e0b32bad3684d58e3`.
Real MLX and CTC execution on the retained 34.9135625-second synthetic recording
produces all 114 independently authored expected words in order. Selected timing
reaches the saved document. Naming, editing words and bounds, whole-document save,
whole-host restart, protected Resume, TXT/SRT/VTT/JSON exports and stale-save HTTP
409 pass. PCM and saved people remain unchanged. The 4.163-second observation is
diagnostic, not an equivalent 0.6.6 comparison or general accuracy claim. Evidence
and the external test harness are in
`/private/tmp/speakerdesk-long-refinement.USlLQY/`; the tested native host and model
child exited. The paced helper is external Python test code, not artifact proof.

The isolated Tauri preview was refreshed after its previous idle parent and both
children exited. It serves initialized models at `http://127.0.0.1:57677/` and
describes the English context limit. Signed executable SHA-256:
`5c452742a8200dbca686ac83bf1013640a51bdb9726f0f1e7063a6dc1d715712`.
`long-refinement-preview.json` and `app-long-refinement.log` are retained under
`/private/tmp/speakerdesk-native-desktop.WJKAde/`. CUA still reports the Mac locked,
so this adds no actual editor, physical capture or distributable-package claim.

PR #27 remains open at `db9da655b3eeeeef598ba21a21cb54505e6bce21`, base
`59e7b0c860f72ca94c24e6eab18ac820b144ee24`. The accepted full replacement remains
active and unfinished: long non-English and rolling live accuracy, ordinary live
language-decision metadata, qualified Pause/Resume and raw dual-source echo,
physical capture and long-session gates, pending Tauri workflows, native default
startup and packaging, disposable updater/security checks and equivalent 0.6.6
performance. Windows and issue #25 remain deferred. Installed app, production
data, existing model cache and GitHub state remain untouched.

## Saved Auto-language accuracy and stable restart checkpoints

Saved canonical passages up to 24.5 seconds now use their captured Auto setting
to run fresh Whisper routing before Cohere. They keep the captured mode and epoch,
carry actual language decisions and absolute probe coordinates, and align against
the freshly selected language. Manual Retry remains explicitly manual. The shared
decode path retains source conditioning and rejects incomplete, uncertain,
unsupported or mixed-language replacements without discarding existing words.

The core and mounted red tests first reproduced the former explicit-language
restriction. A retained real-model run then exposed a separate persistence defect:
JSON parsing changed model probabilities by one floating-point bit, invalidating
an unchanged completed-passage checkpoint after restart. The SQLite regression
reproduces that exact hash change. Enabling the existing serde_json dependency's
`float_roundtrip` feature preserves the evidence; no rounding or weakened
freshness comparison was added. The mounted restart test now asserts that Resume
neither republishes the passage nor makes new probe/transcription calls.

Final validation on the same M5 Mac: 164 core, 61 host, ten mounted live-service,
one service-lifecycle and nine process tests pass, as does strict all-target
core/host Clippy. Four existing tests remain explicitly ignored (one network and
three separately qualified acoustic tests). Four capture-presentation Node checks
pass; unchanged editor/updater proof is reused. Logs are
`/tmp/speakerdesk-auto-refinement-{core-red,host-red,float-red,float-green,final-tests,final-clippy,controls,final-release}.log`.
The corrected locked optimized build completed in 34.98 seconds with the existing
upstream mlx-sys warnings. Its SHA-256 is
`c82ae8d583cc3e12b47c63ee32b891676b6a24430ccb7acacceff755f9418b89`.

Real MLX/CTC execution on synthetic Auto job
`726c942e55de4fb1b6df70eb8ded04af` preserves one protected prefix and refines its
unedited tail with four fresh English probes and 36 aligned words. Names, profile
catalog and PCM bytes remain unchanged. History, stale-save HTTP 409, whole-host
restart, export and idempotent Resume pass. The 1.177-second measurement is
diagnostic, not equivalent performance or improved-accuracy proof. The saved-PCM
capture, failed first refinement and corrected fresh-copy run are retained under
`/private/tmp/speakerdesk-auto-refinement.1uvO3t/`; the successful run is in
`after-float-fix/`. The Python harness is external test code; the application and
model child are native. All qualification hosts and children exited.

The isolated Tauri preview now serves ready models at `http://127.0.0.1:54035/` and
reports saved Auto support accurately. Its ad-hoc signed executable SHA-256 is
`088c177dca54b88d2b27ea04d5791133de6fd5d048a8aecf1522bf82c162e565`.
`auto-refinement-preview.json` and `app-auto-refinement.log` live under the existing
temporary preview directory. CUA still reports the Mac locked despite the unlock
reply. This adds no editor, physical capture or distributable-package claim; the
preview still uses a development library path and saved-PCM capture helper.

PR #27 remains open at `db9da655b3eeeeef598ba21a21cb54505e6bce21`, base
`59e7b0c860f72ca94c24e6eab18ac820b144ee24`. The full accepted replacement remains
active and unfinished: long-passage/context and rolling live refinement, ordinary
live language-decision metadata, qualified Pause/Resume and raw dual-source echo,
physical capture and long-session gates, pending Tauri workflows, native default
startup and packaging, disposable updater/security checks and equivalent 0.6.6
performance measurements. Windows and issue #25 remain deferred. Installed app,
production data, existing model cache and GitHub state remain untouched.

## Saved canonical accuracy pass

The real refinement routes now recheck saved, sealed, explicit-language passages
through the resident child and existing serial worker. This first slice supports
anchors up to 24.5 seconds. Automatic-language and longer passages stay unresolved;
the UI and readiness describe that limit. Captured language mode must be present:
a detected language alone cannot silently turn an Auto recording into a manual
decode. Qualified Apple-processed microphone audio retains unity gain through the
shared admission/retry path, including ordinary passage retries.

Candidates preserve their original source, speaker, audio anchor and epoch. An
incomplete decode keeps previous words. Publication compares the current passage
with its inference snapshot, then replaces exactly that vector slot. It preserves
current names, protected corrections, unrelated passages, display order and final
CTC identity. Empty and absent protection lists have the same meaning. History
stores only the previous ID, bounds, speaker and words. Unused generic window
reconciliation, ID remapping and no-op live refinement commands were removed.

Pause commits generation invalidation before signalling cancellation. A held
cancellation acknowledgement does not release admission. Completed passage
versions survive interruption and restart; resume skips unchanged completed work.
Model-child failure leaves the Ready transcript and audio intact for retry.

The initial real route test reproduced HTTP 409 from the former unavailable
handler. Independent checks then reproduced an unrelated whole-document save
incorrectly blocking publication, and overlapping source rows changing order.
The corrected candidate passes 162 core, 61 host, nine mounted live-service, one
service-lifecycle and nine process tests, plus strict all-target core/host Clippy.
Four tests remain explicitly ignored: the network download and three acoustic
tests with separately retained proof. The 34 existing editor/updater Node checks
and four capture-presentation checks pass, including saved Start/Pause/Retry
controls. Logs are `/tmp/speakerdesk-saved-refinement-{red,unrelated-red,final-tests,clippy,frontend,controls}.log`.

The locked optimized build completed in 23.53 seconds, with the existing upstream
mlx-sys warnings. Executable SHA-256:
`11b0e76d21836a5093f5693070aa843f6e417e3d528958c81171b634182796a8`.
Real MLX/CTC execution on a disposable copy of synthetic job
`90367b1ffb2e4418b04984619eec5bc7` refines one passage, retains the protected prefix,
and supplies 36 aligned words. Names, profiles and PCM bytes remain unchanged.
History, full service restart, export, idempotent resume and stale-save HTTP 409
pass. The measured 1.306 seconds is diagnostic only, not an equivalent performance
comparison or proof of improved acoustic accuracy. Evidence and the external
Python harness are in `/private/tmp/speakerdesk-saved-refinement.eGKZR0/`;
the application and its inference child do not invoke Python.

The isolated Tauri preview was refreshed only after its previous idle parent and
children exited. It serves the native setup at `http://127.0.0.1:51180/` and
correctly reports the saved explicit-language limit. Its signed executable SHA-256
is `8288b91036e44afa45adca71cb99c3dd5dcb6c48c6568b352d88f83b554dc3a0`;
`saved-refinement-preview.json` and `app-saved-refinement.log` are retained under
`/private/tmp/speakerdesk-native-desktop.WJKAde/`. It still uses the development
library path and synthetic saved-PCM capture helper, not a distributable package.

CUA still reports the Mac locked after the latest unlock reply, so no new actual
editor claim is made. PR #27 remains open at
`db9da655b3eeeeef598ba21a21cb54505e6bce21`, base
`59e7b0c860f72ca94c24e6eab18ac820b144ee24`. The complete replacement remains active:
finish automatic/long-passage and rolling live refinement, qualified Pause/Resume
and raw dual-source echo, physical capture and long-session measurements, pending
Tauri workflows, native default startup and packaging, disposable updater/security
checks and equivalent 0.6.6 performance gates. No installed app, production data,
existing model cache or GitHub state was changed. Windows and issue #25 stay deferred.

## Sealed live timing, voice evidence and original source audio

Live publication now retains actual speaker activity regions, owner candidates and
sealed acoustic admission. Open publication does not rescan the full utterance's
speech ledger. Supported sealed, unprotected crops use the existing resident CTC
model and cached vocabulary. Timing retains passage/source identity, nonzero
machine/audio revisions and language epoch. A late model candidate cannot reattach
stale timing after an edit. Ordinary CTC errors retain transcribed words with an
unavailable timing result; cancellation, timeout and child loss keep their lifecycle
semantics. Accuracy refinement remains unavailable.

Retry, voice preview, enrollment and recognition resolve each passage's original
capture source. They reject stale catalogs and escaping files, preserve migrated
v1 source audio read-only, and prefer a native derived source only when that source
exists. A derived mixed file alone does not supersede a legacy source. Review found
and reproduced a symlinked `derived` parent escaping the native root; confinement
now uses the selected storage authority, including parent components. Filesystem
errors other than absence do not silently fall back to legacy audio.

Automatic voice checks now run after both import and live transcript publication
under the existing operation and lease. Live capture and inference have joined
before optional recognition begins; the result is already Ready and editable.
Cancellation invalidates the persisted generation before notifying the child.
Matching proposes an unconfirmed name and never enrolls or replaces a profile.

Validation on the M5 Mac: 163 core, 61 host, seven mounted live-service, one service
lifecycle and nine process tests pass. Strict all-target core/host Clippy passes.
One network and three separately qualified model tests remain explicitly ignored;
unchanged acoustic proofs are reused. New red controls reproduced missing activity,
incorrect timing revisions, discarded words on CTC failure, absent post-live voice
checks and the derived-parent symlink escape. The broader run also exposed a test
race: continuation publication can correctly cause a speaker-name save to return
409. The recovery test now refreshes only that exact conflict, with a bounded
deadline; production revision checks are unchanged. Logs are
`/tmp/speakerdesk-live-evidence-{projection-red,voice-red,optional-red,optional-green,automatic-red,source-red,final-tests,host-final,clippy}.log`.
The corrected full host run is the final host/integration result.

An optimized build completed in 23.51 seconds with the existing upstream mlx-sys
warnings, SHA-256
`648accf0184e9c27daa2158587dd80b0698a54bfa5b94802a4c44e16e4bb15b0`.
Real-model evidence uses the same 18.32-second synthetic Samantha recording as
the live-edit regression. Job `90367b1ffb2e4418b04984619eec5bc7` preserves the
edited prefix, name, boundary, continued words and all PCM bytes through Stop and
restart. Its unedited tail has 36 aligned words and two eligible voice clips;
preview bytes match saved audio. Explicit consent enrollment saves a compatible
native profile that survives restart/export. A second live job,
`1d522699072043a285161f07b006f7f7`, matches that synthetic voice at 0.9608983951
without changing the profile catalog. It remains an unconfirmed assignment.
This tests real MLX/Core ML execution through the mounted application path using
a saved-PCM helper, not physical capture, UI, population accuracy or performance.
The external Python harness is not application code.

The copied historical job that previously returned HTTP 500 from voice clips now
returns 200 with `insufficient_clips`; its document remains exactly unchanged and
no activity is invented. All qualification hosts and children exited. Inputs,
scripts, documents and results are retained in
`/private/tmp/speakerdesk-live-evidence.ILB58D/` and
`/private/tmp/speakerdesk-live-edit.deNKoi/evidence-after/`.

Runtime inspection found one stale setup sentence. A wording-only correction and
19.76-second incremental build produce SHA-256
`6068166783c2e16be891621d1b7026dd4b9494947a17334a1bd3aa18fe3d7f6a`;
unchanged acoustic proof above is reused. The refreshed temporary Tauri bundle's
ad-hoc signed executable is
`703ff4d836f8cfabae2c5e1ac1da1d0d8d70d9cfa4814499440c72989d641fc2`.
Its actual setup response reports supported sealed live timing and unavailable
accuracy refinement. It is idle with ready models at `http://127.0.0.1:64790/`;
`live-evidence-preview.json` and `app-live-evidence-final.log` record its state in
`/private/tmp/speakerdesk-native-desktop.WJKAde/`. The old previews and their
children were reaped before replacement. CUA still reports the Mac locked, so
there is no new editor interaction claim. The preview still uses a saved-PCM helper
and a development library path; this is not Python-free package acceptance.

PR #27 is still open at `db9da655b3eeeeef598ba21a21cb54505e6bce21`, base
`59e7b0c860f72ca94c24e6eab18ac820b144ee24`. Local implementation is not pushed.
The full replacement remains active and unfinished: next complete the accuracy
pass, qualified Pause/Resume and raw dual-source echo, physical capture and
long-session gates, pending Tauri live/setup/long-import checks, native default
startup and packaging, updater/security qualification and equivalent 0.6.6
performance gates. The installed app, production data and existing model cache
remain untouched. Windows and issue #25 remain deferred.
