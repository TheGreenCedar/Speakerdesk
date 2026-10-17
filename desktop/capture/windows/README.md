# Windows capture helper

Standalone native `speakerdesk-capture` executable using the maintained
`windows` 0.62.2 Rust bindings and WASAPI shared mode. This crate has its own
workspace until the parent integrates build and packaging.

```powershell
cargo test --locked --manifest-path desktop/capture/windows/Cargo.toml
cargo build --release --locked --manifest-path desktop/capture/windows/Cargo.toml
target/release/speakerdesk-capture.exe --describe
```

The executable is idle until an explicit JSONL Start command. `--describe` and
`--check` are metadata-only: they do not initialize COM, enumerate devices,
request permissions or start capture. They report `windows_wasapi_shared_v1`,
the `raw_dual_mono_stereo16k_v1` export contract and `audio_queue_fence_v1` progress.
These declarations do not establish installation ownership or device proof.

Stdin commands match the existing host protocol:

```json
{"type":"start","microphone":true,"system":true}
{"type":"pause"}
{"type":"resume"}
{"type":"stop"}
```

The helper opens the default console-role microphone and render endpoint. Render
capture uses
[shared-mode WASAPI loopback](https://learn.microsoft.com/en-us/windows/win32/coreaudio/loopback-recording),
including the complete endpoint mix. It requests Float32 16 kHz mono microphone
and stereo system audio using Windows'
[automatic channel and sample-rate conversion](https://learn.microsoft.com/en-us/windows/win32/coreaudio/audclnt-streamflags-xxx-constants).
It never substitutes another device or format after a rejection. Errors include
the HRESULT, requested format and observed endpoint mix rate/channels.
Microphone capture requests RAW mode to avoid stacking driver speech processing
with the included WebRTC AEC3 and noise suppressor. If the endpoint rejects RAW
properties or initialization, it opens a fresh default stream on the same device
and format; the host processor still runs. The runtime observation records the
RAW result and the actual stream effects (or the query failure), without assuming
that a communications category supplies AEC. Start and Resume recheck the route
and effects. Hardware effects availability is device-specific.

Windows microphone-only recording also opens loopback as a private echo reference.
The host keeps it in an anonymous temporary file for alignment and late-packet
repair. With system audio unselected, it never enters the saved tracks, transcript,
mix, meters or source catalog. Selected original microphone/system tracks remain
retained; only completed cleaned microphone output advances inference.

Stdout events retain the existing types and fields:

- `audio`: `source`, sample-clock `time`, `rate:16000`, `channels` (1 microphone,
  2 system) and base64 little-endian Float32 `pcm`. Packets contain at most 4,000
  frames. Interleaved stereo channel order is preserved.
- `capture_runtime`: actual endpoint IDs, complete original mix format bytes,
  microphone RAW/effects observations,
  requested capture formats, producer QPC/device observations and explicit
  `packet_and_clock` or `clock_and_empty_buffer` observation kinds.
- `recording`, `paused`, `stopped`: recording-clock `time`; initial/resumed
  recording also reports selected sources.
- `clock`: recording `time`, `paused:false`, monotonically increasing `sequence`
  and `producer_fence:"audio_queue_fence_v1"`.
- `error`: actionable cause and startup phase. A failure does not emit a
  successful stopped acknowledgement.

All WASAPI interfaces stay on one STA owner. Packet timestamps come from
[GetBuffer's QPC position in 100 ns units](https://learn.microsoft.com/en-us/windows/win32/api/audioclient/nf-audioclient-iaudiocaptureclient-getbuffer).
Each source keeps a continuous 16 kHz output sample clock; ordinary callback
jitter does not shift already accepted samples. Invalid timestamps, nonfinite
PCM, device discontinuities after the initial packet and timing gaps beyond
20 ms stop capture explicitly. Pre-Start/Resume samples are not admitted.

Start/Resume requires fresh valid microphone PCM when selected. A silent render
endpoint may have no packets, so system readiness instead permits a successful
producer-clock read plus capture polling drained to empty and an active unchanged
endpoint/mix format. The receipt records which was observed. Readiness has a
five-second bound. Resume rechecks the original endpoint IDs and complete mix
formats before restarting; it resets source clocks and requires new observations.
Pause stops both producers, drains accepted PCM and then acknowledges the paused
recording boundary. Paused wall time is excluded from the recording clock.

Clock fences are emitted only after successful fresh WASAPI observations and
draining both capture clients to empty. Their time is bounded by the producer
observations captured before draining; a timer alone cannot advance progress.
Pause and Stop drain PCM before their acknowledgements. Stream/route failures
preserve previously accepted output and end with an error.

The stdout queue holds at most 32 records, with one bounded packet batch retained
if admission fills the queue. Startup retention is also bounded at 32 records.
Overflow stops the producers before waiting for accepted output to drain. A
disappearing stdout consumer unblocks the queue and fails the process. Drain and
writer completion each have five-second bounds; the helper never joins a blocked
stdout or stdin thread indefinitely. A dead/stalled consumer cannot receive the
remaining packet; that path exits unsuccessfully and makes no coverage claim.

Portable tests cover exact original stereo bytes, packet/frame limits, pause
clock exclusion, callback jitter, refusal of timer-only/stale progress, rejection
of unsupported Apple contracts and retention of the record that overflows the
output queue. Actual Windows compilation and microphone/loopback execution,
permission denial, silent loopback, Pause/Resume, device loss and host persistence
still require the parent Windows qualification lane.
