# Automatic meeting language

New meetings and imports default to `auto`. The start form no longer requires a
language choice. Settings offers an optional, persisted override for Cohere's
14 supported languages. Existing recordings keep their saved language; changing
Settings does not change an active worker or overwrite transcript edits.

## Evidence and architecture

[Superwhisper's official changelog](https://superwhisper.com/changelog) records
Cohere automatic language detection in 2.17.1 (August 4, 2026) and improved
detection accuracy in 2.17.3 (August 17). Its
[voice model documentation](https://superwhisper.com/docs/models/voice) describes
local, offline Cohere transcription in 14 languages without translation.
Superwhisper does not document its detector implementation; we do not claim to
reproduce that implementation.

The [Cohere model card](https://huggingface.co/CohereLabs/cohere-transcribe-03-2026)
requires a pre-specified language and describes inconsistent code-switched
audio. Its [API](https://docs.cohere.com/reference/create-audio-transcription)
also requires a language. The pinned MLX Speech adapter defaults a missing
language to English; its result language echoes the requested language. Passing
`None` is therefore not automatic acoustic detection.

Speakerdesk performs acoustic language identification before invoking Cohere.
It reuses the already-pinned MLX Audio Whisper implementation, loading only a
local multilingual tiny model. Whisper's dedicated
[language-token operation](https://github.com/Blaizzy/mlx-audio/blob/feb25a37b07923bae556e59111995071d66afa0d/mlx_audio/stt/models/whisper/decoding.py)
runs the encoder and one decoder step with SOT alone. It scores all 99 language
tokens. No Whisper text decoder, transcription task, translation task,
Transformers processor, or implicit checkpoint download is invoked. The small
token adapter is deliberately coupled to this pinned dependency and artifact.
Waveforms are converted to MLX and padded with silence to 30 seconds **before**
log-Mel extraction; zero padding the feature matrix is incorrect.

Import/live workers share `SpeechTranscriber`. Auto divides retained audio into
balanced probes of at most three seconds. Adjacent confident probes can share a
bounded ASR crop. NVIDIA speaker labels do not decide whether words are attempted:
unknown speakers and overlapping audio still reach Cohere. Their text remains
separate from eligibility for clean voice enrollment.

A supported winner with probability at least .90 and margin at least .20 applies
immediately. These closed-set scores are not calibrated accuracy guarantees.
Uncertain probes use the most recent successfully transcribed, confidently
detected supported language from the same meeting configuration epoch, provided
it ended no more than 60 audio seconds earlier. This fallback does not extend
its own expiry. Clear contrary evidence invalidates contradicted context even
if the new ASR attempt fails. Historical refinement receives a frozen preceding
snapshot and cannot borrow language from future resident-worker audio.

Without fresh context, an uncertain supported winner produces a marked
best-effort transcript. Unsupported evidence without usable context or a failed
detector leaves retained audio available for an explicit meeting-language choice.
Manual language bypasses detection. Short, quiet and overlapping speech do not
abstain solely because of duration, amplitude or speaker uncertainty. Only exact
zero PCM skips detection and ASR, marked `audio_state=digital_silence`; nonzero
room noise cannot safely be classified as silence by these outputs.

The meeting header has one current-language selector. Settings controls the
persisted default independently. Blank machine passages are hidden from the
ordinary transcript; unresolved audio is available through one collapsed review
surface. TXT/SRT/VTT omit empty passages. Canonical JSON, WAV, segment identities
and review metadata retain all internal ranges. User edits remain protected
from asynchronous blank or failed results. Returned best-effort words are shown
with an understated uncertainty note. No word alignment or reliable separation
of overlapping voices is claimed.

## Artifact and packaging

- [Converted artifact](https://huggingface.co/mlx-community/whisper-tiny-asr-fp16/tree/77fa3f52b482ec80d086df55f893065a9baab172),
  revision `77fa3f52b482ec80d086df55f893065a9baab172`.
- `model.safetensors`: **74,385,959 bytes**, SHA-256
  `1267601753d2996d68dc065d1de786895a7ad9a874d1d3548a5274c1bc067444`.
- Checksum-pinned config and token metadata: 2,440 and 34,604 bytes. Setup also
  retains and checks the converted model card. Detector startup strictly loads
  the reviewed dimensions and verifies the weight digest.
- Distribution card declares Apache-2.0. Canonical
  [OpenAI code and weights](https://github.com/openai/whisper#license) are MIT;
  Apple Whisper implementation and MLX Audio are MIT. Both distribution and
  upstream notices are retained. Installer contains no model weights.
- Setup adds the language model. Manual overrides remain usable with the two
  original models while the detector is absent. AUTO reports missing setup
  clearly. `LID_MODEL_PATH` overrides the local detector directory.
- Build/setup narrow both VAD and STT registries to avoid eager imports of
  unrelated architectures. The pinned Whisper implementation remains unchanged.
  No new Python dependency is added; Transformers stays excluded from the bundle.

## Verification and handoff

The full **64-test CPU suite passed** (47 existing, 17 new) in 20.804 seconds.
Peak process RSS was **164,233,216 bytes** (156.6 MiB); maximum child RSS was
19,578,880 bytes. Source boundaries/version checks passed for 772 tracked files
at 0.2.0, along with JS and shell syntax and `git diff --check`. These values
describe tests with substituted models, not detector or live inference memory.

CPU tests execute the real import and live-worker routing with synthetic PCM and
substituted model outputs. They exercise mixed languages, uncertainty, overlap,
silence/short speech, hysteresis, per-speaker state, original audio, timing,
manual override, defaults, editor persistence, and JSON export. A verbatim excerpt of the pinned
language-decoding function is executed with NumPy operations and toy logits,
with AST equality checked against the installed dependency when available.
This checks SOT-only input, all 99 classes, and non-language masking without loading
MLX or any trained model. Deliberate waveform-padding and stale-continuity
mutations fail their intended assertions in disposable copies.

The local source UI was checked through supported in-app-browser controls:
Automatic initially selected in Settings, French override persisted after
reload, and Automatic restored. No mandatory start-form language selector and
no browser console errors were observed. No recording or setup button was used.
This was the source web UI, not a rebuilt native app.

During the combined 0.3.0 integration, the user separately approved the exact
pinned community conversion and a coordinated native resource window. The
74,385,959-byte weight was downloaded and checksum verified. Real Whisper
detection completed on M5 Metal `Device(gpu, 0)`, with peak MLX allocation
276,688,318 bytes and peak process RSS 163,233,792 bytes in the first detector
run. The complete NVIDIA/Whisper/Cohere/B6 job returned four original English
and French sentences and both synthetic identities. See
[the integration measurements](0.3.0-integration.md) for timings, memory,
abstention results and limits.

The first development fixture exposed moderate-confidence wrong-language
routing on accented synthetic English. Raising admission to .90 made that
case abstain. That development-case rerun does not establish generalization.
Fourteen fresh synthetic cases were then evaluated against the frozen policy,
using new sentences and recordings from the same compact voices. Clean English
and French routed correctly; unsupported Russian and Swedish, silence, short
audio and annotated overlap abstained. One accented English passage was still
confidently admitted as French. Thresholds were not retuned. The detector can
therefore choose the wrong language even above .90; use a manual language retry
when the words or language look wrong. The user waived this additional test as
a release gate; the failure remains a documented limitation. Evidence includes
synthetic unsupported speech,
silence, short probes, noise, and annotated overlap. It does not establish
human accuracy, NVIDIA overlap detection accuracy, or exact switches inside a
probe. No private audio was uploaded and no real device capture or permission
prompt was exercised.

Unresolved/partial passages retain audio and time boundaries, explain the
reason in the UI and exports, and can be retried with an explicitly chosen
language. A retry candidate never replaces existing text automatically. Audio
outside returned speech passages remains visible as possibly silence or missed
speech. Packaged local loading and signing remain separate acceptance checks.

Overlapping integration files: `model_setup.py`, `live_worker.py`, `app.py`,
`pipeline.py`, `live_meeting.py`, Settings HTML/JS, and packaging imports.
Keep the voice worker's `voice_eligible` separate from generic boundary or
language review flags. AUTO split offsets need to reach that eligibility logic.
