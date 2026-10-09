# Mac speaker echo repair

Dual-source meeting capture now subtracts acoustic speaker leakage from the
microphone before adding the direct Mac audio. It preserves local speech,
remote speech and simultaneous talking, with cancellation enabled automatically
for the existing dual-source path. There is no microphone ducking or text
postprocessing. File-backed source controls pass. The retained physical-room noise test measured
15.61 dB settled echo reduction, below its recorded 20 dB target. It does not
establish duplicated-word removal or independent local-speech gain during overlap.
The later separate-source public speech controls establish word preservation and
zero microphone echo words only for the tested fixtures and route transition.

## Root cause and reproduction

Frozen integrator base: `6e6e9c2cae25c0e7620cc26f2be26f79fc10bde1`, including
the required pipeline and integrated updates/shortcuts. Work occurred only in
this lane's isolated `fix/capture-acoustic-echo` checkout.

`MeetingCapture.swift` independently taps raw AVAudioEngine microphone input
and ScreenCaptureKit system audio. Their converter packets use host-clock
sample timestamps. Before this change there was no playback reference in the
microphone processing path. `SourceMixer` aligns packets with a 500 ms
watermark, overwrites overlapping callbacks within a source, then sums sources.
Consequently the direct system signal and its delayed acoustic copy in the
microphone both reached `audio.wav` and transcription. This establishes missing
acoustic echo cancellation rather than duplicate digital callback summation.
It is source evidence and a deterministic reproduction, not a recording of
Albert's v0.6.3 binary or proof of a particular room/device response.

`scripts/echo_fixture.py` synthesizes independent voice-like excitation and
persists PCM16 WAVs before replaying them through the real mixer. The frozen
cases have 63/237 ms latency, +180 ppm reference drift, room taps at
0/13/47/91 ms, far-only speech, double-talk, near-only speech, and independent
no-echo controls. The original mixer retained the entire echo: 0.0 dB ERLE in
all three cases, with double-talk residual SNR 5.75/5.77/6.04 dB. Each recording
contained exactly 192000 samples. The uncorrected implementation fails the
predeclared clean-audio assertions.

## Implemented path

The helper retains two system-reference channels at 16 kHz Float32 instead of
collapsing independent speaker paths before cancellation. It reports source
format changes before new-format audio. The mixer aligns those channels with
mono microphone samples and retains the original microphone, mono system
compatibility track, and stereo `system_reference.wav` before DSP processing.
Stereo mono downmix normally averages channels; if phase inversion would erase
the remote signal, it retains the stronger channel instead. The AEC always
receives both original reference channels.

`speakerdesk/capture_echo.py` buffers 500 ms of additional lookahead and feeds
10 ms frames into the project-local native component. Bounded broadband
correlation validates an acoustic match, accepts either transducer polarity,
estimates latency up to 500 ms, and compensates measured clock skew up to
500 ppm. Pure tones and silence cannot establish an echo match. Independent
no-echo input preserves the microphone exactly. Delay jumps need three
consistent matches, preventing one double-talk outlier from replacing the
clock estimate. Ring history is bounded to two seconds.

The native boundary calls upstream WebRTC `EchoCanceller3::AnalyzeRender` with
the explicit reference and `ProcessCapture` with the microphone. Both mono and
stereo adaptive histories remain warm, with a short crossfade when reference
dimensionality changes. Each contribution contains the local voice once;
outputs are interpolated rather than summed. Refined and coarse filters retain
a 128 ms room tail, using shorter startup filters and a one-second initial
phase. The null neural-estimator pointer excludes neural residual inference.
This is CPU DSP; no model/GPU/ANE job is required.

Only the exported **linear prediction error** contributes to the canonical
mix. In qualification the full nonlinear suppressor attenuated quieter local
speech; that version was rejected. No AGC, microphone gating or suppressor gain
is applied to the microphone contribution. Upstream source is unmodified.
The measured 64-sample linear-output framing delay is compensated before adding
the direct system signal. Partial frames and delayed tails drain at Pause,
Stop and input-format boundaries before reference/adaptive state resets.
Format reset points split the shared sample timeline, preserving pending audio.
Existing late-packet handling and explicit timing-jump failures remain in force.

`audio.wav` and the existing live/refinement workers receive the same canonical
clean-microphone-plus-system signal. A missing/incompatible native component
fails dual-source Start before model startup or recording creation. Single-source
capture needs no echo component. DSP failure preserves the raw source tracks
and reports failure rather than silently claiming clean capture.

The existing 500 ms watermark plus DSP lookahead adds approximately one second
of live audio latency. Final saved timing and sample counts are unchanged.
Startup and a changed acoustic path require adaptation; strong nonlinear
speaker distortion or a room tail beyond the configured filter can leave a
residual. These limits need measurement on Albert's actual Mac.

## Linked-backend measurements

Final file-backed fixture results (ERLE at 2–4 s; gain/SNR during 5–8 s):

| Case | Echo reduction | Local-speech gain | Local residual SNR |
| --- | ---: | ---: | ---: |
| 63 ms latency | 23.24 dB | 0.99936 | 24.83 dB |
| 237 ms latency | 24.58 dB | 1.00467 | 17.08 dB |
| +180 ppm drift | 22.78 dB | 0.99903 | 24.05 dB |
| No echo | exact expected mix | 1.00000 | zero residual |
| Near only | exact expected mix | 1.00000 | zero residual |

All five cases retain 192000 samples. The frozen assertions remain ERLE >=20 dB,
normal double-talk gain .95–1.05, residual SNR >=15 dB, >=30 dB control SNR,
and exact sample count. The independent stereo test also passes >=20 dB ERLE
and the same gain bounds. Additional tests cover double-talk from the first
frame without a far-only training period, local gains at .5/.25/.1 amplitude,
inverted acoustic polarity, phase-inverted stereo, other voice seeds,
17/312 ms delays with -220 ppm drift and a 109 ms room tail, independent no-echo
voices, correlated tones, format boundaries, partial Pause/Resume/Stop frames,
DSP failure retention, invalid frames, missing-component preflight, and bounded
streaming state. The longer-tail/other-voice cases use a separate >=15 dB
settled ERLE/SNR contract; they do not weaken the original frozen assertions.

65 targeted tests pass: 17 capture-DSP, 11 live meeting, 14 real-pipe/SQLite/WAV
lifecycle, 9 live-language, 4 progress, and 10 cache tests. The lifecycle peers
are synthetic and model modules are mocked; no devices or models run. Linux
Source CI uses an explicit native-platform peer for lifecycle tests; macOS
package tests execute the real linked component. An additional simulated Linux lifecycle run timed out during import on two
bounded attempts and is unclaimed; the integrator must run Source CI.
The Swift helper compiles, and its metadata-only `--check`
returns `capture_started:false` before constructing an audio engine. Source,
shell syntax, diff and library linkage checks pass. A combined app build,
Developer ID signing and the integrator's full pipeline checks remain with
the integrator; this lane did not install or publish an app.

Private synthetic evidence is in `/private/tmp/speakerdesk-echo-task10-local`,
a verified ordinary local directory with mode 0700. Final PCM/WAV proof is
`final-linked-contract-2/metrics.json`; originals and earlier experiments remain
intact. No fixture audio or binary is tracked in Git.

## Dependency, build and integration

Albert explicitly approved adding/building the project-local dependency.
`desktop/capture/echo/source-lock.json` authenticates official source subsets
using Git tree hashes and individual file SHA256 hashes:

- WebRTC: `852150686748f5cce0c1c96f4fde50e6be43ef73`.
- Its pinned Chromium third_party/Abseil tree:
  `8a83e63525b495aa7f35648a3b4b676279deb864`, upstream Abseil
  `7f008af1930f59d869a816795ac5d445e3628e4d`.
- BSD3/PATENTS/AUTHORS, Apache2 and Ooura notices ship under
  `packaging/licenses/capture-echo` and in the runtime.

`scripts/setup_echo.py` uses existing Apple Clang, CMake and Ninja. Fetches are
limited to approved official origins, with redirects refused, 30 s wall bounds,
32 MiB received /128 MiB extracted budgets and safe file-only archive extraction.
Only the explicit AEC/audio/support closure is compiled: no browser checkout,
upstream hooks, gclient, global tool installation or neural estimator. Two
compiler slots, 900 MiB per compiler, 600 s build wall / per-process CPU bounds and a 512 MiB
output cap are enforced. Owned children are killed/joined on interruption.
Verified local source caches avoid repeated rehashes; fresh source is fully
verified before publication. Build receipts remain project-local.

The packaging script builds/signs the component before tests and bundles the
library and licenses. Exact runtime cache identities include DSP source,
source lock, build helpers and native toolchain versions. The linked library's
only dynamic dependencies are Apple system frameworks/libc++/libSystem.
The tested native binary SHA256 is
`6bf686f5ddd0e32dfa49d2127cf41c4bce293e015d2fa9621c54dc951e9addc6`;
it is an isolated build artifact, not an installer or release.

Reproduce in an authorized Apple Silicon checkout:

```sh
python scripts/setup_echo.py
python -m unittest discover -s tests -p test_capture_echo.py -v
python scripts/echo_fixture.py /private/tmp/new-local-evidence --assert-clean
```

The fixture destination must be new; the script refuses to overwrite evidence.
Use an environment with the repository's existing NumPy dependency installed.

The actual API supports this explicit render reference:
[WebRTC AEC3 header at the pinned revision](https://webrtc.googlesource.com/src/+/852150686748f5cce0c1c96f4fde50e6be43ef73/modules/audio_processing/aec3/echo_canceller3.h).
The inspected Apple [voice-processing presentation](https://developer.apple.com/videos/play/wwdc2019/510/)
requires live device rendering; it did not establish a silent external-reference
route for ScreenCaptureKit. This patch does not claim an OS echo canceller
processes that reference.

## Capture clock and converter follow-up

An authorized retained capture exposed a second source defect. AVAudioConverter
buffers output between callbacks, but the helper stamped each output buffer
with the latest input callback timestamp. At 48 kHz, the emitted microphone
buffers included 1360, 1664 and 1365 frames; assigning those variable lengths
successive 100 ms timestamps produced 400 overlaps of 64 samples and 109 gaps.
SourceMixer overwrote overlapping samples and inserted silence in those gaps.
A within-source overwrite is distinct from the missing acoustic echo cancellation
that originally allowed both direct system audio and microphone echo into the mix.

CaptureAudioConverter now anchors each source to its first host timestamp and
advances output time by the actual count of 16 kHz samples. It drains available
output without re-supplying an input buffer, signals EOF on Pause/Stop and format
or genuine input-time discontinuities, then starts a fresh converter on resume.
Small timestamp jitter and real clock drift preserve the continuous output clock.
The explicit-reference estimator handles relative sample-clock drift; its fit
now removes each speaker channel's fixed acoustic delay before estimating the
common slope. Changing the active stationary speaker formerly invented 75 ppm
of drift in a deterministic regression. Real +180/-220 ppm drift remains detected.
A strong broadband match still validates echo before delay selection. When a
significant earlier local correlation peak lies within 16 ms of that match,
the estimator uses that earlier path, excluding the adjacent autocorrelation
lobe. This keeps the causal adaptive filter able to model a weaker direct path
followed by a stronger reflection. A deterministic 40 ms direct path/45 ms
stronger reflection case improves from 12.42 to 25.99 dB ERLE, with quiet
near-end gain 1.0098. Existing no-echo and speech-preservation gates remain intact.
No native AEC tuning, upstream fork, neural inference or user option was added.

The device-free Swift regression uses the actual production converter. Mono
44.1/48 kHz and interleaved stereo 16/48 kHz inputs each produce 32000 output
frames from two seconds of input, contiguous timestamps, an idempotent EOF and
zero sample error against independently chunked one-shot conversion. The former
converter protocol reproduces the varying callback lengths and missing EOF tail;
its concatenated PCM matches the reference prefix exactly. This isolates timing
and tail loss rather than corrupted input memory.

Fresh deterministic DSP fixtures retain 22.78–24.58 dB echo reduction, near-end
relative gain 0.9990–1.0047 during double-talk, exact no-echo/near-only controls
and 192000 frames per case. Quiet-near, first-frame double-talk, stereo,
polarity, room-tail, positive/negative drift and lifecycle regressions pass.

## Retained test validity and hardware gap

Albert reported that he heard no noise during the earlier coordinated test.
Its hardware validity is therefore unresolved. The pinned 51-second stimulus
contained nonverbal synthetic broadband audio at approximately -31.7 dBFS RMS
in active regions. The saved playback command exited successfully, and retained
system PCM corresponds to that stimulus. Retimed microphone PCM contains
correlated energy roughly 45 ms later. These establish recorded correspondence,
not listener audibility or verified human speech. The saved controller timing
places playback approximately 53 seconds after the parent's cue, so scheduled
local-only and double-talk intervals cannot qualify participant speech.

The original retained-file DSP result was 0.245 dB ERLE. Reconstructing every
emitted microphone sample once improved a replay to 6.264 dB; reconstructing
both source clocks improved the fixed-block replay to 10.442 dB. These are
retained-file diagnostics, not confirmed physical hardware acceptance or failure.
They do not establish 20 dB hardware reduction or quiet human speech preservation.
The original device-rate microphone input and converter tail were not retained;
unobserved end intervals cannot be reconstructed. Original evidence is unchanged,
and candidate WAVs/reports live in fresh verified local-only private folders.

Required coordinated follow-up: the integrator must build/sign the combined
candidate with both Swift files, identify its exact helper path/hash, and obtain
an explicit timely cue for one bounded speaker/microphone test. Confirm audible
playback before interpreting scheduled human speech segments, and preserve raw
reference/mic/canonical tracks. Assess settled echo reduction, local/remote
intelligibility, quiet and normal double-talk, sample counts, latency and route
changes. No repeat, device/volume change, private meeting, installation, release,
upload or model job is authorized by this source handoff.
