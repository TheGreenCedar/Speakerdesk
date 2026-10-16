# Pinned Nemotron ONNX preparation

This build-only tool converts the exact Speakerdesk Nemotron checkpoint into
standard ONNX graphs for native Windows ML/ONNX Runtime execution. Python is
used for conversion and comparison; the graphs require no Python or custom
runtime operators. The shared Rust runner lives in `../ort-probe`.

The weights remain pinned to
[`mlx-community/Nemotron-3-Diarization@59ed2db`](https://huggingface.co/mlx-community/Nemotron-3-Diarization/tree/59ed2dbfc1346dcea9d423c71306a3a2499c568f).
Both the config and checkpoint size and SHA-256 must match the production
manifest before conversion. Every inference parameter and its shape is consumed;
only the known auxiliary training heads are omitted. Source BF16 weights are
represented exactly as Float32 values, and inference uses Float32 activations.
This precision policy changes the macOS BF16 execution and is qualified explicitly
below. It does not change the speaker threshold or speaker identities.

The neural topology follows the pinned
[`mlx-audio@feb25a37` Nemotron implementation](https://github.com/Blaizzy/mlx-audio/blob/feb25a37b07923bae556e59111995071d66afa0d/mlx_audio/vad/models/nemotron_diarization/nemotron_diarization.py),
also implemented by `crates/models/src/sortformer/network.rs`. The exporter uses
the official ONNX package, opset 17, and standard operators. In particular,
RoPE uses Sin/Cos/Mul/Add; attention uses MatMul/Softmax; the frontend uses a
fixed real DFT matrix. This avoids runtime-specific attention and FFT extensions.
The same ONNX API is available through
[Windows ML](https://learn.microsoft.com/en-us/windows/ai/new-windows-ml/overview).
Successful export does not establish provider placement; inspect executed
events in the native runner's saved profiles.

Use an existing disposable Python environment with `onnx==1.19.1`,
`numpy==2.2.6`, and `onnxruntime==1.24.4` for preparation:

```sh
python experiments/windows/nemotron/export.py /read-only/models/nemotron /private/artifacts/nemotron
python experiments/windows/nemotron/qualify.py /private/artifacts/nemotron /private/artifacts/nemotron/cpu.json \
  --low-reference /retained/independent/nemotron-reference/low.json
python experiments/windows/nemotron/prepare_cases.py /private/artifacts/nemotron
python -m unittest discover -s experiments/windows/nemotron -p 'test_*.py' -v
```

`qualify.py` verifies the existing WAV and independent upstream source revision.
It never modifies reference JSON or audio. The committed offline reference and
the optional retained Low reference remain independent MLX outputs. Comparison
requires all valid frames, maximum probability drift at most 0.025, and zero
changed `p > 0.5` activity decisions. This is a fixture conversion bound, not a
new acoustic calibration. Hostile packetization and fresh-session repetition
must reproduce the conversion output exactly.

The two graphs have these interfaces:

| Graph | Inputs | Outputs |
| --- | --- | --- |
| `frontend.onnx` | `audio_windows`: Float32 `[1,T,512]`; `valid_frames`: Int64 `[1]` | `features`: Float32 `[1,T,128]` |
| `nemotron.onnx` | `features`: Float32 `[1,T,128]`; `cache`: Float32 `[1,C,512]`; `fifo`: Float32 `[1,F,512]`; `valid_frames`: Int64 `[1]` | `chunk`: Float32 `[1,T/8,512]`; `high`: Float32 `[1,(C+F+T/8)*8,8]`; `low`: Float32 `[1,C+F+T/8,8]` |

`T` is positive and divisible by 8. `C` and `F` can be zero. All graph inputs
must be finite. The host retains original PCM, gathers each centered 512 sample
window at `frame * 160 - 256`, and preemphasizes its valid samples using the
original preceding sample (`x[t] - 0.97*x[t-1]`). Samples outside the source are
zero. Keeping the 257 samples preceding the next frame preserves this boundary.
The graph applies the pinned 400 sample Hann window centered in those 512 samples,
then power/mel/log. `valid_frames` excludes the partial 160 sample tail; padded
features are zero. Native frame counts and padded tensor sizes stay distinct.

The host owns each recording's cache, FIFO, predictions, compression flag, PCM
suffix, received sample count, processed frame count, and finished state.
`qualify.py::Session` documents the same transition as the native Rust session.
It crops `central` high-resolution frames after `(C+F)*8`, appends only
`ceil(central/8)` chunk embeddings, and scores cache candidates from `low`.
The AOSC selector preserves speaker-major order, repeated source frames, the
learned silence embedding, and the existing 264 frame bound. The graph does not
carry hidden session state. Cancellation must invalidate the caller's partially
advanced session; a new recording must begin with fresh empty cache/FIFO.

`prepare_cases.py` emits all 13 calls of the Low fixture, one offline call, and
paired frontend calls. Each `case.json` matches the shared native Rust runner's
named tensor schema and repeats twice. Its expected raw tensors come from this
ONNX graph on macOS CPU, and are explicitly separate from independent acoustic
goldens. Compare returned native files with:

```sh
python experiments/windows/nemotron/compare_probe.py /private/artifacts/nemotron/cases/low-12 /returned/native/low-12 --provider dml
```

That comparator first binds the runtime, provider, executed profile, model,
case and inputs, then checks tensor names, shape, hashes, finiteness, every output of
both repetitions, maximum error at most 0.001, and zero changed probability
activity decisions. Per-call cases have precomputed CPU state; they do not prove
that GPU-produced state was fed into later calls. Full application continuation
must execute that chain through native recording sessions.

The initial macOS ORT 1.24.4 CPU qualification produced 909 valid frames for each
preset. Offline maximum drift was `0.0195279121` with the NumPy frontend and
`0.0195278525` with the ONNX frontend. Low maximum drift was `0.0095938444` and
`0.0095927715`, respectively. All four comparisons preserved every activity
decision; packetization and reset errors were zero. These synthetic voices
establish conversion consistency on this fixture, not human acoustic accuracy.

The short Low fixture reaches FIFO continuation but does not reach AOSC
compression. The small behavior checks protect compression's silent selection,
latest-frame retention, speaker-major ordering, and STFT source coordinates;
there is no retained long acoustic reference in this lane. Cancellation,
cross-source isolation, installed-app execution, and Windows placement require
their corresponding native evidence. Model binaries, prepared tensors, and
machine profiles stay outside the repository.

The Windows native implementation is prepared separately in
`crates/models/src/sortformer/network_windows.rs` and `session_windows.rs`.
`Network::load(directory, backend)` first calls the owning runtime's
`verify_model(directory, "nemotron")`, then loads both graphs. Each graph call
uses the request's `Operation::run_options()` and checks cancellation before
and after execution. Tensor shapes, Float32 extraction, finite embeddings,
and `[0,1]` probabilities are checked before state can advance. Per-recording
PCM/cache state stays outside the shared model sessions. An errored or cancelled
stateful feed remains finished, matching the existing macOS session rule.
Module wiring, artifact admission, and application dispatch belong to the
integration owner and were not activated by this lane.

The exporter now also emits `silence.f32`, exactly 512 little-endian Float32
values, and records `silence_sha256` in provenance. It preserves the existing
NumPy export and both graph outputs. The native loader checks the raw file's
length and finiteness after shared artifact admission; it needs no NumPy
runtime reader. The retained embedding's raw payload hash is
`d4417b3c0eabdf7c47032fac2b5b5a7ee83d819a6ddda8fd8eaf74e2b5cc4ac7`.

Six focused Rust checks cover centered preemphasis, retained left context,
hostile packetization and exact observed frame counts, final tails, fresh
source state, cancellation before cache commit, malformed output poisoning,
FIFO overflow, retained compressed predictions, and speaker-major repeated
frames/silence. They passed in an isolated macOS harness using the actual two
source files and ORT rc.14 API 27, with protocol/backend admission stubbed and
no native graph execution. Three Python contract checks also pass, including
rejection of a corrupted actual loaded-runtime hash. These checks do not
replace Windows compilation or native model execution.

Cache rank ties use deterministic source-index order in the Rust implementation.
MLX and NumPy partition kernels do not promise identical selection among equal
scores. The existing short acoustic reference does not exercise compression;
an independent long reference remains necessary to qualify that selection and
GPU-produced cache continuation. The implementation does not waive that gap.
