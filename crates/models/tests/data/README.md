# Silero acoustic fixture

`two-voice.wav` is 16 kHz mono PCM16 generated on macOS with the installed
Samantha (US) and Daniel (UK) `say` voices at 175 words/minute. It contains:

- “I checked the recording this morning. The transcript needs a speaker name.”
- 0.8 seconds of silence.
- “I can review it after lunch. Please save the final version.”

There are 1600 leading zero samples and a silent final tail. The complete file
has 145445 samples (9.0903125 seconds), leaving a 37-sample partial Silero frame.
It contains no user recordings. It is a speech detector fixture, not evidence of
diarization quality or a substitute for public multi-speaker acoustic acceptance.

The two JSON files were generated independently from the pinned Python
`SileroVADBranch` at `Blaizzy/mlx-audio@feb25a37b07923bae556e59111995071d66afa0d`,
using the pinned `mlx-community/silero-vad-v6@2ebf4a5e10726a2e78ddd4d70eedfb6f1c33eb06`
weights. The source hash and WAV hash are recorded in each file. Both raw and
per-frame peak-normalized views maintain separate recurrent states.

- `silero-reference.json`: baseline MLX 0.32.3, macOS 26 backend wheel.
- `silero-reference-macos15.json`: MLX 0.32.2 with the exact macOS 15 native
  backend linked by Rust. This isolates graph correctness from backend kernels.

The small backend-dependent probability drift did not change any speech
hysteresis decisions on this clip. The test checks those decisions against the
baseline and numerical agreement against the matching backend. Recorded timings
are component measurements on an M5; they do not qualify the whole application.

`generate_silero_reference.py` reproduces the reference from the pinned upstream
`silero_vad.py`, model directory and this WAV. Run it in a disposable environment
with the stated MLX version and backend wheel, providing those three paths, a new
output path, and `--backend macosx_26_0_arm64` (or `macosx_15_0_arm64`). It never
invokes the Rust implementation. Python is used only for independent validation.

Run with the explicit, disposable model directory:

```sh
SPEAKERDESK_TEST_SILERO=/path/to/models/silero-speech \
  cargo test --release --locked -p speakerdesk-models --test silero -- --ignored --nocapture
```
