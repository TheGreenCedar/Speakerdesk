# Speakerdesk third-party attribution

The native Mac app uses Rust, Apple MLX and Core ML. Transcription model weights are downloaded separately and remain subject to their upstream terms. The small LocalVQE capture model is bundled with the app. The build generates the complete app notice at `target/native/THIRD_PARTY_NOTICES.txt` from the locked Cargo dependency sources, the fetched MLX and echo sources, and hash-pinned upstream texts in `native-license-sources.json`. It is packaged as `Contents/Resources/THIRD_PARTY_NOTICES.txt` and can be exported from Settings. Build dependencies can appear in the notice even when they are not linked into the app.

- NVIDIA Nemotron 3 Diarization: [OpenMDW 1.1](https://openmdw.ai/license/1-1/), [upstream model](https://huggingface.co/nvidia/Nemotron-3-Diarization). The MLX conversion `mlx-community/Nemotron-3-Diarization` is pinned to `59ed2dbfc1346dcea9d423c71306a3a2499c568f`.
- Cohere Transcribe03-2026: Apache 2.0, [upstream model](https://huggingface.co/CohereLabs/cohere-transcribe-03-2026). The 4-bit conversion `spokedotso/cohere-transcribe-03-2026-mlx-4bit` is pinned to `064e51eab6db47066cbeaa85e2894b5691bd8d12`.
- Whisper tiny language detector: `mlx-community/whisper-tiny-asr-fp16`, pinned to `77fa3f52b482ec80d086df55f893065a9baab172`. Its conversion declares Apache 2.0. Canonical [OpenAI Whisper](https://github.com/openai/whisper) code and weights are MIT. Speakerdesk uses local language identification only.
- Silero VAD v6: MIT, [upstream model](https://github.com/snakers4/silero-vad). The `mlx-community/silero-vad-v6` conversion is pinned to `2ebf4a5e10726a2e78ddd4d70eedfb6f1c33eb06`.
- Supplied-text timing: Apache 2.0 model `csukuangfj2/sherpa-onnx-omnilingual-asr-1600-languages-300M-ctc-int8-2025-11-12`, pinned to `6fc542a3b0661c8278cca1230c34deb989f31202`. The native MLX conversion follows the MIT MLX Audio source.
- ReDimNet2-B6 voice recognition: MIT, [Palabra.ai source](https://github.com/PalabraAI/redimnet2), pinned to `2a8d15f65b1dfb5d73fede2f11ee42bcccca3035`. The Core ML artifact `aufklarer/ReDimNet2-B6-CoreML` is pinned to `112dd8f4f836abdf8a420e66a5e2885cf8ec64ab`.
- Native speech graph references: MIT, [MLX Audio](https://github.com/Blaizzy/mlx-audio) at `feb25a37b07923bae556e59111995071d66afa0d`, and [MLX Speech](https://github.com/appautomaton/mlx-speech) at `4577ad28e6bef2ae2a11b8fa76a6cdb23bb2a9cc`. Their Python packages are not shipped.
- Apple MLX 0.32.2: MIT, [upstream](https://github.com/ml-explore/mlx). The official arm64 backend artifact supplies `libmlx.dylib` and Metal kernels; mlx-rs 0.32.0 and mlx-sys 0.6.0 supply the Rust and C bindings. MLX C retains its upstream license in the vendored source.
- Capture echo cancellation: WebRTC AEC3, BSD 3-Clause, revision `852150686748f5cce0c1c96f4fde50e6be43ef73`, with its patent grant and authors. Abseil support is Apache 2.0. The Ooura FFT permission notice is retained from the pinned WebRTC source. The source lock is `desktop/capture/echo/source-lock.json`.
- Lucide 0.577.0: ISC and the MIT terms for portions derived from Feather. The app icon and interface icons use [Lucide](https://lucide.dev). Fonts come from the operating system.

Speakerdesk is not affiliated with the upstream model, framework or icon authors. Synthetic test fixtures use macOS speech voices and are identified as scripted material.


## Windows converted model materials

The prepared Windows graphs are derived model materials, separate from their
original checkpoints, conversion code and native runtime libraries. Their exact
source revisions and output file hashes are recorded in
`crates/models/src/windows-model-artifacts.json`. Cohere's ten assets target
`TheGreenCedar/Speakerdesk` tag `windows-models-20261016`; all other model prefixes
remain on `windows-models-20261015`. This notice does not assert publication.
Each asset name starts with its model ID and
`--`. Model terms continue to apply when these converted files are downloaded
separately from the application.

| Asset prefix | Weight origin and declared terms | Speakerdesk alterations |
| --- | --- | --- |
| `silero-speech--` | Silero Team, through `mlx-community/silero-vad-v6@2ebf4a5e10726a2e78ddd4d70eedfb6f1c33eb06`; MIT. | The pinned Float32 checkpoint is represented as an ONNX inference graph; recurrent state and frame preparation are explicit. |
| `nemotron--` | NVIDIA Nemotron 3 Diarization, through `mlx-community/Nemotron-3-Diarization@59ed2dbfc1346dcea9d423c71306a3a2499c568f`; OpenMDW 1.1. MLX conversion by Lucas Newman / MLX Community. | BF16 parameters are represented as Float32; inference omits training heads and exposes cache/state. The frontend uses standard ONNX operators. |
| `cohere-speech--` | CohereLabs `cohere-transcribe-03-2026`, quantized by Spoke in `spokedotso/cohere-transcribe-03-2026-mlx-4bit@064e51eab6db47066cbeaa85e2894b5691bd8d12`, following the MLX Speech conversion; Apache 2.0. | Original packed 4-bit weights, affine scales and biases are preserved. DirectML reconstructs parameters for computation; encoder and request-local decoder/KV graphs are separate. External stores are sharded, and constant integer shape operations are folded without changing weight values. The encoder uses finite FLOAT minimum for 48 attention-mask literals to prevent invalid fully masked rows; parameter values remain unchanged. |
| `whisper-language--` | OpenAI Whisper code and weights: MIT. The MLX Community `whisper-tiny-asr-fp16@77fa3f52b482ec80d086df55f893065a9baab172` conversion declares Apache 2.0; both notices are retained. | Language-identification-only encoder/first-decoder-step graph, explicit frontend and precision/materialization lowering. |
| `coarse-alignment--` | Meta Omnilingual ASR, distributed by `csukuangfj2/sherpa-onnx-omnilingual-asr-1600-languages-300M-ctc-int8-2025-11-12@6fc542a3b0661c8278cca1230c34deb989f31202`; Apache 2.0. Copyright 2025 Meta Platforms, Inc. and affiliates. | The original INT8 source is downloaded and its parameters are dequantized during setup into the exact qualified Float32 store. The separate graph lowers attention to the maintained Microsoft operator. Output is selected CTC scores for supplied-text alignment. |
| `redimnet2-b6--` | Palabra.ai, `PalabraAI/redimnet2@2a8d15f65b1dfb5d73fede2f11ee42bcccca3035`, release checkpoint `b6-vb2+vox2_v0-lm.pt`; MIT. | Maintained graph converted to ONNX, with a half-precision backbone and Float32 waveform/frontend/output boundaries. The Mac Core ML artifact is a different conversion. |
| `mossformer2-16k--` | Alibaba / Shengkui Zhao, `alibabasglab/MossFormer2_SS_16K@407cb030cd66340918ebb6c8cc63b18f8592cdbe`; Apache 2.0. ClearerVoice source at `6b3774dc79c46ae8bed2a4fa5f706f0ac8c75c61`; Apache 2.0. | Raw full-window inference split into 50 ONNX stages; dynamic padding, bounded centered-normalization reductions and decomposed attention. The upstream sliding-window/RMS wrapper is omitted. |

Graph implementation references retain their separate terms: MLX Audio and MLX
Speech are MIT; Hugging Face Transformers 5.7.0 is Apache 2.0. The ClearerVoice
source acknowledges SpeechBrain and its MossFormer2 block acknowledges
FLASH-pytorch by Phil Wang; the latter's MIT copyright and terms are included.
The existing OpenAI Whisper, Palabra.ai and ClearerVoice source notices above
remain applicable. Python conversion packages are build tools, not application
runtime libraries. Model assets do not contain the evaluation WAVs or corpora.

The complete generated `THIRD_PARTY_NOTICES.txt`, including the full OpenMDW,
Apache and MIT texts and applicable copyright notices, must accompany the model
release as well as the application. Keep this origin/change notice available
with that distribution. These notices identify modifications and retain upstream
terms; they do not relicense third-party materials under Speakerdesk's license
or suggest endorsement by the original authors. Immutable declaration URLs and
hashes are recorded in `experiments/windows/model-distribution.md`.

- Capture enhancement: LocalVQE v1.2 source `f53063c9eb2a85f96479867d1dd911dc3bf6319b` and the bundled `localvqe-v1.2-1.3M-f32.gguf` weights are Apache 2.0. GGML `c044a8eeae2591faa0950c8b5e514cbc4bbfc4ca` is MIT. Exact source/archive/model pins are in `desktop/capture/echo/localvqe-lock.json`; the annotated `localvqe-portability.patch` uses UTF-8 file opening, disables backend discovery and restricts the upstream model allowlist to this one model without an environment bypass. The owned CMake composition builds only the v1.2 graph/API and static CPU backend; upstream download/build hooks do not run.
