# Cohere Windows graph qualification

The draft Windows inventory pins the original-affine UINT4 encoder and the
shape-folded decoder graphs. The loader disables ORT optimization for all three;
the encoder disables DML pooling/fusion to bound reconstruction memory, while
the smaller decoders enable both. Before returning a session, actual API27
assignments must match that exact graph's reviewed placement policy.

The actual Rust resident path preserves all 31 reference tokens and exact text
in two complete requests. The 9.09-second clip takes **3.78 / 2.87 seconds**
(first after load / second request), including preprocessing, encoder, all 32
decoder calls and text construction. Native lifetime private peak is **4.29 GiB**;
maximum sampled GPU commitment is **3.89 GiB**. Load is separately measured at
10.96 seconds. These are retained-clip observations, not sustained live-app proof.
[compact-weights.md](compact-weights.md) records the original weights, failed
proxy gates and separate complete-request decision. Combined model residency,
assembled application qualification and distribution remain open.

## FP32 diagnostic exporter

Build-only conversion of the installed, pinned MLX 4-bit checkpoint into ONNX.
The application remains Rust. No upstream ONNX checkpoint is substituted and no
independent reference is regenerated.

The graph owners are the maintained
[Transformers CohereAsr](https://huggingface.co/docs/transformers/v5.7.0/model_doc/cohere_asr)
decoder and Parakeet encoder in `transformers==5.7.0`. `export.py` converts the
MLX tensor layout, reconstructs each group of 64 affine weights as
`q_u4 * scale_bf16 + bias_bf16`, then loads every tensor strictly into those
graphs. The low nibble is first within each packed U32. It rejects unknown or
missing tensors and verifies the source model/config/tokenizer hashes against
the repository's existing installation pins.

The initial candidate uses FP32 weights and activations. This retains the
checkpoint's quantization values, but its arithmetic differs from the original
mixed FP32/BF16 MLX kernels. Artifact size and memory increase to about 8 GB.
It must match every token and the text in the retained independent
`cohere-reference.json` before exporting. That gate qualifies one synthetic
English clip, not general transcription accuracy or all 14 languages.

The maintained feature extractor requires both PyTorch and librosa, using
`librosa.filters.mel` for Slaney filters and PyTorch for dither/STFT/normalization.
The build requirements pin librosa 0.11.0 and its tested numerical dependencies.
The fixture WAV is decoded by Python's standard `wave` module, so this path
does not need torchaudio, torchcodec or an external media-decoding backend.

Run in an isolated Python 3.12 build environment on a machine with at least
20 GB free disk and enough RAM for the dequantized graph and tracing:

```powershell
python -m pip install -r experiments/windows/cohere/requirements.txt
python experiments/windows/cohere/test_export.py -v
python experiments/windows/cohere/export.py C:/private/models/cohere-speech C:/private/cohere-onnx
```

An output directory must be new. Source files and independent fixtures remain
read-only. The export uses ONNX opset 17 and standard operators. Weight files
are external data, with one decoder weight store shared by its two graphs.
This avoids ONNX protobuf's 2 GB limit and does not require a custom runtime
kernel. The pinned legacy TorchScript exporter is deliberate: the maintained
layers are traced, with dynamic lengths checked by the small executable test.

The exported interface is:

| Graph | Inputs | Outputs |
| --- | --- | --- |
| `encoder.onnx` | `features` FP32 `[1,frames,128]`, `attention_mask` I64 `[1,frames]` | `encoded` FP32 `[1,ceil(frames/8),1280]`, `encoder_mask` I64 `[1,ceil(frames/8)]` |
| `decoder-init.onnx` | `tokens`, `positions` I64 `[1,tokens]`; `encoded`; `self_mask` FP32 `[1,1,tokens,tokens]`; `cross_mask` FP32 `[1,1,tokens,encoded_frames]` | Last-position `logits` FP32 `[1,16384]`; 8 sets of self/cross K/V |
| `decoder-step.onnx` | `tokens`, `positions`; self/cross masks; `past_self_k_N`, `past_self_v_N`, `past_cross_k_N`, `past_cross_v_N` | `logits`; updated `present_*` K/V |

All K/V are FP32 `[1,8,length,128]`. Self length includes the prompt and emitted
tokens; cross length is the physical encoded shape. `encoded` can be removed
from the step graph by dead-code elimination because cross K/V already exist.
The emitted case JSON lists the actual graph inputs. Masks use `0` for allowed
entries and finite FP32 minimum for blocked entries. `positions` are absolute
decoder offsets. Token generation is greedy, with EOS 3 and a 448-token cap.
Each request owns fresh caches. Keep cross K/V unchanged during continuation.

`prompts.json` contains verified tokenizer IDs for all 14 existing languages.
Caller validation still owns finite PCM, the existing 35-second limit including
permitted boundary padding, explicit supported language, sub-160-sample empty
results, cancellation between encoder/decode calls, and session lifetime. The
encoder graph accepts already normalized features; build-only feature extraction
uses the maintained feature extractor and checks the retained physical frame
count (909). A portable application feature frontend needs separate qualification.

The exporter emits primed numerical cases for the generic native Rust probe:

```powershell
ort-probe.exe C:/runtime/onnxruntime.dll dml C:/private/cohere-onnx/encoder-case/case.json C:/private/cohere-encoder-dml
python experiments/windows/cohere/compare.py C:/private/cohere-onnx/encoder-case C:/private/cohere-encoder-dml --provider dml
```

Repeat for `decoder-init-case` and `decoder-step-case`. `compare.py` checks every
output, exact integer masks, finite floating values and greedy-logit winners.
Before numerical comparison, the shared `verify_probe.py` admits the pinned
Windows ML 2.4.89 runtime, case, graph, external weights and every input by hash,
checks the requested provider, and reads actual placement from the raw profile.
Default absolute and relative tolerances are each 0.002; report measured drift
before changing these bounds. Cases include two identical executions to expose
unexpected session mutation. The native probe retains runtime hashes and actual
executed provider events; provider registration alone is insufficient.

For complete autoregression through the native probe, run:

```powershell
python experiments/windows/cohere/run_native_decode.py C:/probe/ort-probe.exe C:/runtime/onnxruntime.dll dml C:/private/cohere-onnx C:/private/cohere-full-native
```

This build-only orchestrator launches the Rust probe for the encoder and each
decoder step, passes the actual resulting K/V to the next native invocation,
and fails at the first token divergence. `decode-result.json` is written only
after all retained tokens/text and EOS match. Per-step cases, outputs, logs and
profiles remain available. Process/session creation overhead makes this a
correctness probe rather than an application latency measurement.

The small test independently guards packed nibble/group reconstruction,
variable encoder lengths with masked valid tails, and cached decoder continuation
against full-sequence decoding for multiple prompt/continuation lengths. It does
not replace the full checkpoint, Windows runtime, hardware-placement or product
interaction gates.

## Release-sized encoder weight shards

GitHub release assets must each be [under 2 GiB](https://docs.github.com/en/repositories/releasing-projects-on-github/about-releases#storage-and-bandwidth-quotas).
`shard_encoder.py` repacks the qualified `cohere-fp32-v2` encoder into ordinary
ONNX external-data files capped at **1 GiB**, without model computation or new
dependencies. Its original source pins are frozen in the helper, independently
of subsequent runtime-registry changes:

| Source file | Bytes | SHA-256 |
| --- | ---: | --- |
| `encoder.onnx` | 2,022,217 | `0c243acb5f78842b7573d9a1c46227b9436b7fb21986d03cc45a84bc9d57d9ad` |
| `encoder.weights` | 7,581,848,576 | `6e8d004d81a1d870de6725665ce381ff4b36df207909ed228bddbd8bb0246100` |

Using the existing ONNX build environment:

```powershell
python -m unittest discover -s experiments/windows/cohere -p test_shard_encoder.py -v
python experiments/windows/cohere/shard_encoder.py C:/private/cohere-fp32-v2 C:/private/cohere-sharded
```

The destination must be new and outside the source tree. Output contains only
`encoder.onnx`, `encoder.weights-NNN` and `encoder-shards.json`; unchanged decoder
and tokenizer assets are staged separately. Each tensor remains whole, exact
source interval aliases share one copied interval, and copying uses 1 MiB chunks.
Both input hashes, numeric byte ranges and single-tensor size are checked before
creating the output. Unsupported external metadata or paths fail closed.
The release command requires Windows: one source weight handle denies writes
and deletion throughout hashing and copying. The graph is hashed and parsed from
the same byte string. Portable synthetic tests instead recheck the same opened
weight file before publishing a manifest. No whole weight file is loaded into
memory.

The manifest records source/tool pins, actual output sizes/hashes, each copied
range hash and aliases, and the hash of all unchanged non-external graph fields.
A copy failure can leave an incomplete destination without this manifest; retry
in a new directory. Before admitting the output, update artifact/case pins from
the verified manifest and rerun native encoder/decode qualification. The helper
does not change the exporter, independent references, numerical bounds or runtime
format.

The historical FP32 repack preserves all 1,884 tensor ranges in eight shards.
Its encoder graph is `c777228a…`; it has been superseded in the draft artifact
inventory by the compact candidate above. Five focused tests passed on
Windows, including an attempted same-length write blocked by the retained handle.
The native Rust resident transcription test also passes with the sharded layout:
all 31 retained tokens, text and EOS match across both requests, with the existing
empty-input and pre-cancel checks. On the RX 7900 XT, loading took 22.11 s and the
two requests took 1.258 s and 0.535 s. This qualifies the artifact layout and Rust
frontend on the retained clip; multilingual quality and published-download setup
remain separate gates.

The previous FP32 `windows_residency` integration test kept Cohere, Silero, Nemotron,
ReDimNet and MossFormer loaded together. Cohere decoder state survives VAD,
diarization, embedding and the maximum 392,000-sample separation window; a decoder
step also runs between separation phases. Resumed and subsequent transcription
retain the reference text and 31-token count, and both separated waveforms remain
inside their original bounds (maximum error `0.0003259182`). All admitted files
deny write access while the networks are loaded. A separate native regression
failed before the loaders retained admission handles and passes after the fix,
including restored write access after teardown; it never changes model bytes.

On the RX 7900 XT this combined check passed in 66.83 s, including 28.90 s loading.
Sampling the owned process about every 0.5 s observed peak GPU committed memory
17.65 GiB and process peak working set 6.34 GiB. GPU committed memory includes
shared allocations and is not a precise VRAM peak. CTC, Whisper, the host and
capture are absent: this does not establish seven-model memory headroom or live
meeting throughput. Raw evidence is `windows-residency-native.log` in the owning
task's evidence directory.
