# Meeting and import reliability handoff

The isolated branch `codex/meeting-import-reliability` repairs reproduced lifecycle
and storage failures without changing the transcript schema or export formats.
Its base is `88b311dea8c1bc606c06ffc05b84fdd5c6ed4fc1`, confirmed as private
Speakerdesk main at 17:38 UTC on 2026-10-05. The canonical checkout was not edited.

## Verified behavior

All 25 CPU tests pass. The test suite uses actual subprocess pipes, Flask routes,
SQLite persistence and PCM WAV files, with synthetic capture and inference peers.
Those peers never import models, use capture APIs, request permissions or download
anything. Their text is an explicit transport fixture, not an inference result.

The tests cover microphone-only, system-only and mixed selection; pause/resume
with a partial audio block; exact mixed and separate track samples; stop;
playback/export access; persistence after reopening; import batches; JSON import
validation and provenance; revision conflicts; TXT/SRT/VTT/JSON exports; active
job edit/delete guards; and preservation of edited transcripts.

Ten regressions fail on unchanged main in the 24-test baseline run. A subsequent
targeted check reproduces one additional storage-startup regression on that same
base. The repaired 25-test suite passes all of them:

| Contract | Observed on the baseline | Repair |
| --- | --- | --- |
| Stop while models start | Still active after the 2-second cancellation bound | Cancel readiness polling and prevent capture startup after stop |
| Late recording acknowledgement | Changes `finishing` back to `recording` | Keep stop ownership until cleanup finishes |
| Capture error tail | Saves 8,000 of 17,600 valid received samples | Flush the mixer's valid pending tail on failure |
| Backlog recovery | Saves 500,000 of 545,600 received samples | Stop inference admission while continuing to save received capture data |
| Error following `stopped` | Marks a failed capture `ready` | Read through helper EOF; failed Swift startup emits no success acknowledgement |
| Worker failure with an unresponsive helper | Still active after 18 seconds | Apply the real 15-second stop watchdog to failure paths |
| Runtime shutdown | Returns with an active owner ID | Wait for WAV closure and terminal persistence during close |
| Over-range source samples | Positive samples wrap to negative PCM values | Saturate each saved source track before PCM conversion |
| Import/live inference ownership | Accepts import inference during live capture | Share admission locking; reject competing inference in either direction |
| Failed batch file save | Leaves an earlier upload and an orphaned folder | Stage files, commit one SQLite transaction, then submit preparation |
| Failed recording-directory creation | Leaves the manager's status endpoint returning 404 | Publish the meeting owner only after directory and job persistence succeed |

Terminal meeting status is published after subprocess, transport and recording
cleanup, so editing/deleting cannot race those owners. An error preserves the
first diagnostic and previously persisted transcript segments. Audio recovery
is limited to data received by the app and storage that remains writable.

## Validation and resources

The final CPU verification took 20.709 seconds, including the real 15-second
watchdog. The runner's peak RSS was 148,193,280 bytes (141.3 MiB); the largest
child peak was 49,676,288 bytes (47.4 MiB). These are per-process measurements,
not aggregate machine memory. Dependencies were reused read-only from the
existing packaging environment with bytecode writes disabled.

Source checks passed for 752 tracked files at version 0.2.0, along with Swift
syntax parsing, JavaScript syntax checks, shell syntax checks and diff whitespace
checks. Swift parsing does not prove native compilation or runtime behavior.
Local logs are in the ignored `evidence/final-cpu-tests.log`,
`evidence/final-cpu-verification.json`, `evidence/baseline-main.log` and
`evidence/baseline-startup-storage.log`.

No model downloads, inference, dependency installs or packaging builds were run.
The isolated checkout measured approximately 10.5 MiB. Free disk at the final
verification was 40.96 GB (38.15 GiB); heavy work remains deferred to the parent
coordinator and must preserve its disk floor.

## Native observations and remaining handoff

Supported native UI control opened the existing `/Applications/Speakerdesk.app`
and observed its idle meeting screen and selected source controls. Choosing
**Speakerdesk → Quit Speakerdesk** removed the app and both runtime processes,
verified with a process snapshot. No capture helper process was present. These
observations apply to the installed, pre-patch binary; it was not rebuilt here.

The coordinator still needs a resource-approved build containing this patch.
Albert must own OS permission decisions and real microphone/system capture
checks. Test each source mode, pause/resume, stop during startup/recording,
denied or revoked access, playback of saved audio, persisted text after reopen,
native import/export dialogs and Quit during an active meeting. CPU fixtures do
not establish native capture quality, real transcription accuracy or packaged
parity. No quarantine removal, Gatekeeper bypass, credentials or public release
are part of this handoff.
