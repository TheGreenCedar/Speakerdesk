# Windows CTC qualification

The owner approved a separate, prospective Windows qualification after the
original Mac scalar tolerance failed. `windows-qualification-v1.json` freezes
the new functional contract; all diagnostic experiments and failed original
comparisons below remain historical evidence. No new graph candidate is planned.

`prepare_windows_qualification.py` seals 19 raw-PCM cases and source transcripts
without inference. It uses complete retained public utterances, rejects cropped
audio paired with full text, and preserves the original six cases and column order.
`ctc_windows_qualification.rs` calls the actual Mac and Windows adapters, including
their normalization, model admission, repeated requests and cancellation boundary.
Seal the Mac capture before running Windows; retain both complete receipts.

The core replay checks text/offset/acceptance compatibility and reading-turn
attribution under shared speaker activity. The selected 20 ms envelope budget
for new natural cases is not a claim of acoustic accuracy. Windows metadata has
its own provider identity and no calibrated timing languages, including English.
The fixed graph inventory permits qualification loading; production readiness
still depends on the native gate and assembled application checks.

The frozen functional run passed on 2026-10-15: the actual Mac and Windows
adapters produced identical character and word boundaries across all 19 cases
(1,544 character cells and 283 words), and all 2,660 actual-core activity
projections matched. Each backend completed 38 ordered probes plus short-input
and post-cancellation controls. Same-backend repeats were bit-exact. The native
captures took 48.56 s on Mac and 12.15 s on the AMD PC; these are harness totals,
not application throughput measurements.

| Retained record | SHA-256 |
| --- | --- |
| Prospective protocol | `f45fdd309f979a3d491a00732aeb82a68e9e0c0e5f28b628d0cd5530ee67dac2` |
| Frozen input manifest | `132a9c5e47f7740b8a28423926bcbea62b3d2c554271987c1be82b455fcb8020` |
| Mac capture receipt | `81aad3d054be365345fbcd264068e3c855e40cdbf2733cfb0b57cf112dc82a24` |
| Windows capture receipt | `bf0b8252a74ae93a7872c05b761e2ccb5cf37e0e945036967a204f3dc940a67e` |
| Core replay report | `e3ec8f33fd0918b900aef827b38870b30be6b1c68e4f1044ea7ebea91a8d45c1` |

The independent original whole-clip scalar comparison still fails: maximum
Windows error is `0.0020656585693359375` against the unchanged `<1e-4` bound.
The other five original control comparisons pass, and Mac reproduces all six
original Float32 references exactly. Functional compatibility does not establish
human-labelled acoustic boundary accuracy: Windows retains no calibrated timing
languages. Pre-cancellation checks do not establish in-flight interruption.

## Retained maintained-attention experiment

`fuse_attention.py` prepares one **unqualified** full-model candidate. It accepts
only the original Float32 graph (`4e9c26ad…`) and unchanged external weights
(`ebb4aa54…`), preserving the original exporter and prior diagnostic graphs.

The original ONNX export decomposes attention and splits its scale across Q/K
using two square-root factors. The maintained MLX source instead applies
`q_proj(x) * 0.125` including the projection bias, then fast SDPA at scale 1.
The earlier query-scaling-only adaptation did not resolve the full score error.

This transform replaces each of the 24 attention decompositions with
`com.microsoft::MultiHeadAttention(num_heads=16, scale=1.0)`, fed by the complete
biased Q projection multiplied by exactly 0.125 and the unchanged biased K/V
projections. Its 3D output feeds the existing output projection. It changes no
weights, normalization, GELU, casts, full-vocabulary log-softmax, or column gather.
All original initializer bytes/external references stay intact; the candidate
graph is a fresh sibling that shares `ctc.weights` without copying it.

This is a bounded causal control. Same-QKV MLX fast/decomposed attention differs;
replacing only attention in the otherwise exact MLX model produces whole-score
error 0.00056934. The maintained MHA kernel executes on the actual AMD DirectML
runtime, but its first-layer error (`3.2783e-6`) does not establish an improvement
over decomposed attention. The complete native DML candidate fails the original
score gate: maximum absolute error is `0.00198364`, with exact repetition.
Supplying the exact retained MLX encoder input still leaves `0.000454426` error,
so correcting preprocessing alone cannot qualify it. Existing CPU and DML
profiles already fuse all 48 encoder layer normalizations; replacing those
expressions with the standard fused operator would repeat the same execution.
The candidate remains unqualified and is not admitted by the production child.

The maintained implementation and contract are available in the exact ORT
[v1.27.1 DML kernel](https://github.com/microsoft/onnxruntime/blob/v1.27.1/onnxruntime/core/providers/dml/DmlExecutionProvider/src/Operators/DmlOperatorMultiHeadAttention.cpp)
and [contrib schema](https://github.com/microsoft/onnxruntime/blob/v1.27.1/docs/ContribOperators.md#commicrosoftmultiheadattention).

Build-only preparation, using the existing original artifact and cases:

```sh
python experiments/windows/ctc/fuse_attention.py /private/ctc-fp32-v1/ctc.onnx \
  /private/ctc-fp32-v1/ctc-mha.onnx --cases /private/ctc-fp32-v1/mha-cases
python -m unittest discover -s experiments/windows/ctc -p 'test_fuse_attention.py' -v
```

Preparation seals source/candidate/transform hashes and copies the six original
input/reference tensors unchanged. It does not execute the model or regenerate
an independent reference. Run each candidate case with the existing native
probe on CPU and DML, then compare:

```sh
python experiments/windows/ctc/compare_mha_probe.py /private/ctc-fp32-v1/mha-cases/whole \
  /fresh/native-result --provider dml
```

Comparison retains shared DLL/graph/input/provider admission, exact repeat, and
the original independent `max_absolute_error < 1e-4` gate. It additionally rejects
CPU Attention/MHA fallback, even when unrelated graph work executed on DML.
The focused tests exercise dynamic sequence lengths against an independent
source-formula calculation, full bias/scaling/layout, parameter retention,
rejection before mutation when topology changes, and mixed-provider fallback.
They do not establish full-model numerical qualification.

## Same-input Float32 GELU control

`gelu_control.py` prepares one small diagnostic using the retained, unchanged
`conv0_norm` → `conv0_gelu` capture (`740869f4…` → `c89feb24…`, shape
`[1,96,512]`). The capture run returned the independent whole scores bit-exactly.
Existing prefix errors include normalization drift; they do not isolate GELU.
The prior Whisper GELU control tested Half materialization and cannot decide this
Float32 question.

The Rust GELU closure uses `x * (1 + erf(x / sqrt(2f32))) / 2`. MLX 0.32.2
compiles Float32 temporaries through `Divide`, `Erf`, `Add` and `Multiply` with
the default safe Metal math mode. No explicit reciprocal or tanh approximation
appears in that source. Its [Metal erf kernel](https://github.com/ml-explore/mlx/blob/v0.32.2/mlx/backend/metal/kernels/erf.h)
uses a piecewise polynomial with explicit FMA and `expm1f`. The native CPU
[MLAS erf](https://github.com/microsoft/onnxruntime/blob/v1.27.1/onnxruntime/core/mlas/lib/erf.cpp)
has different coefficients and split point; its [fused GELU](https://github.com/microsoft/onnxruntime/blob/v1.27.1/onnxruntime/core/mlas/lib/gelu.cpp)
uses a rounded reciprocal multiply. Existing native full-graph profiles contain
`Gelu` and `BiasGelu`, without `FastGelu`; [DML BiasGelu](https://github.com/microsoft/onnxruntime/blob/v1.27.1/onnxruntime/core/providers/dml/DmlExecutionProvider/src/Operators/DmlOperatorBiasGelu.cpp)
uses an add with fused [erf-based GELU](https://learn.microsoft.com/en-us/windows/ai/directml/api/ns-directml-dml_activation_gelu_operator_desc).
This establishes a distinct implementation, not the cause of the failed score gate.

The control executes three branches on exactly the same input: explicit native
`Gelu`, `BiasGelu` with zero bias, and source-order `Div`/`Erf`/`Add`/`Mul`/`Div`.
Every explicit intermediate is an output, and ORT graph optimizations are disabled.
Comparison reuses shared actual DLL/graph/input/provider admission, rejects any
CPU branch in a DML run, checks every output hash/shape and exact repetition, and
reports bit/ULP errors without a new acceptance threshold.

```powershell
python experiments/windows/ctc/gelu_control.py prepare C:/private/ctc-fp32-v1/ctc.onnx C:/private/ctc-prefix-captures C:/private/ctc-gelu-control
ort-probe.exe C:/runtime/onnxruntime.dll dml C:/private/ctc-gelu-control/case.json C:/private/ctc-gelu-dml
python experiments/windows/ctc/gelu_control.py compare C:/private/ctc-gelu-control C:/private/ctc-gelu-dml --provider dml
python -m unittest discover -s experiments/windows/ctc -p test_gelu_control.py -v
```

Run the same case on CPU in a fresh result directory. The capture directory needs
only the original `capture.json`, `conv0_norm.f32` and `conv0_gelu.f32`; their fixed
identities are checked before creating output. No external model weights are read.

The native Windows CPU and AMD DirectML control completed with exact repetition;
all DML branches were accelerated. Retained `ctc-gelu-control.log` reports:

| Provider / branch | Maximum source error | Mean source error |
| --- | ---: | ---: |
| CPU fused / zero-bias fused | `4.246830940e-7` | `3.155582817e-8` |
| CPU explicit chain | `4.246830940e-7` | `3.121022019e-8` |
| DML fused / zero-bias fused | `5.066394806e-7` | `2.724048187e-8` |
| DML explicit chain | `5.066394806e-7` | `2.724048187e-8` |

Zero-bias fused GELU equals fused GELU bit-exactly on each provider. Both explicit
chains equal Float32 reassembly of their native-Erf outputs exactly. CPU input
scaling matches IEEE Float32 division; DML scaling differs by at most one ULP
(`4.768371582e-7`, 19,872 of 49,152 values). Removing ORT GELU fusion therefore
does not close the source discrepancy on this retained input and supplies no
demonstrated whole-model correction.

There is no retained source-Erf capture, so the remaining source difference cannot
uniquely distinguish MLX's polynomial from compiled arithmetic. The standard graph
does not force MLX's FMA/`expm1f` implementation. This control authorizes no full
candidate or generic polynomial rewrite; `1e-4` remains the independent whole-score
gate, and the complete CTC graph remains unqualified.

## First frontend normalization control

The earliest observed prefix divergence is `conv0_norm`: the captured first
convolution is exact, while native CPU normalization differs by `1.430511475e-6`
and DML by `4.768371582e-7`. The recorded whole-graph profiles execute this norm
as separate `ReduceMean/Sub/Pow/ReduceMean/Add/Sqrt/Div/Mul/Add` operators on both
providers. This differs from the already-fused 48 encoder norms investigated
earlier; the frontend control does not repeat that rejected candidate family.

The pinned [MLX host dispatch](https://github.com/ml-explore/mlx/blob/v0.32.2/mlx/backend/metal/normalization.cpp)
selects 64 threads, eight values per thread, for this 512-channel row. Its
[Metal kernel](https://github.com/ml-explore/mlx/blob/v0.32.2/mlx/backend/metal/kernels/layer_norm.metal)
accumulates each thread's eight Float32 values, performs 32-lane SIMD sums and a
threadgroup sum, then repeats a centered sum of squares. It multiplies centered
values by `precise::rsqrt(variance + epsilon)` and applies affine parameters in
the same kernel. The export also uses centered variance; epsilon is exactly the
same Float32 `1e-5`, and both affine tensor hashes match the prepared source.
There is no demonstrated epsilon, axis, dtype, parameter or `E[x²]-mean²` defect.
Reduction order, reciprocal square root versus materialized square root/division,
and affine contraction remain distinct implementation details. No source mean,
inverse-standard-deviation or pre-affine capture exists to separate them yet.

`norm_control.py` prepares one small same-input comparison of the original
explicit chain against standard ONNX `LayerNormalization`, with the original
two affine tensors embedded. The maintained DML operator maps the latter to
[MVN2](https://github.com/microsoft/onnxruntime/blob/v1.27.1/onnxruntime/core/providers/dml/DmlExecutionProvider/src/Operators/DmlOperatorLayerNormalization.cpp).
Explicit intermediates remain observable and ORT optimizations are disabled.
The comparator retains shared runtime/graph/input/provider checks, output hashes,
complete branches and bit-exact repeats. It compares both branches with the
source output and checks whether isolation reproduces the original native output.

```powershell
python experiments/windows/ctc/norm_control.py prepare C:/private/ctc-fp32-v1/ctc.onnx C:/private/coarse-alignment/model.safetensors C:/private/ctc-prefix-captures C:/private/ctc-norm-control
ort-probe.exe C:/runtime/onnxruntime.dll dml C:/private/ctc-norm-control/case.json C:/private/ctc-norm-dml
python experiments/windows/ctc/norm_control.py compare C:/private/ctc-norm-control C:/private/ctc-norm-dml --provider dml
python -m unittest discover -s experiments/windows/ctc -p test_norm_control.py -v
```

Run CPU in a fresh result directory too. Preparation reads only the safetensors
header and two 2,048-byte parameter ranges, checking their fixed tensor hashes;
it does not copy/load full weights or execute a model. The capture directory needs
the original `capture.json`, `conv0_raw.f32`, `conv0_norm.f32`, and retained
`native-cpu-conv0_norm.f32` / `native-dml-conv0_norm.f32`; all identities are fixed.

Coverage is narrow: all 96 sampled rows are the same single vector, with centered
variance about `0.007031899`. There is no near-zero-variance evidence here. If the
explicit branch reproduces retained native error while the fused operator matches
source, the supported operator can correct this earliest boundary on that vector.
If it does not, no source-faithful substitution is demonstrated. A changed isolated
explicit result exposes a kernel/isolation dependency instead. None of those
outcomes qualifies the frontend across other vectors or fixes the known failing
encoder suffix; no full-model candidate is authorized by this control. The
independent whole-score gate and references remain unchanged.

Native CPU/DML execution completed with exact repetition (`ctc-norm-control.log`).
CPU source error is `4.768371582e-7` for explicit and `1.430511475e-6` for fused;
DML source error is `9.536743164e-7` for explicit and `4.768371582e-7` for fused.
On both providers the isolated explicit output differs from the original native
output by up to `1.430511475e-6`. The standard fused operator does not reproduce
source exactly, and isolation changes the baseline kernel result. These results
support no causal full-model candidate; no further normalization rewrite follows.

## Supplemental current-core timing replay

The frozen `ctc-timing-v1-replay.json` compares the historical MLX source and
the native AMD MHA candidate using the original supplied text and WAV, 30 selected
columns, and current `Vocabulary::align` at sample origin 16,000. Source capture
first reproduced the original ten independent score columns bit-exactly. Native
admission checked the physical graph/weights, loaded runtime DLLs, inputs,
execution profile and two bit-exact repeats before creating replay metadata.

The ignored `alignment::timing_replay` test passes: all 130 acoustic character
frames and 23 word sample envelopes, raw text, Unicode/UTF-16 offsets, aliases,
status and reason are identical. Both results are complete/aligned. Full outputs
and an empty `behavior_differences` list are retained in the report. Score maximum
absolute error is `0.001983642578125`, mean `0.00002273408851169`; the unchanged
original `< 1e-4` numerical gate still fails. Accepted-character confidence differs
by at most `0.00003266334534`; its nearest source margin above the actual `-20`
acceptance threshold is `13.06341075897`. No threshold crossing was observed.

This is replay of captured scores through the current core, not an actual Windows
child/user-flow test or an independent timing-accuracy reference. The synthetic
two-voice clip does not establish path stability on near ties, low-confidence or
multilingual speech. The current core also emits Mac-specific runtime/provider
and English calibration tags; those identical replay tags are not Windows
provenance or Windows calibration evidence. Behavioral equality on this clip
does not admit the artifact, waive score qualification or establish readiness.
