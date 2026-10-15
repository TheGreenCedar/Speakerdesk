# Whisper language scoring — Windows functional qualification

The frozen Windows graph passed the separate, prospectively reviewed v2
functional contract: 56 cases through the actual native adapters and frontends,
with zero changed core routing decisions or comparison errors. This establishes
the component result described below. Three original scalar-fidelity cases
still fail the unchanged strict `<1e-4` bound; no source-equivalence or general
language-calibration claim follows from the functional pass.

## Retained scalar comparison

The final native candidate is `compiled-gelu-v1`, graph SHA-256
`1c47965522f7495fc85814c70c6b17dc4952436774c43225d55665e4ecbc7f14`.
The earlier standalone neural probes used Windows ML 2.4.89 / ORT 1.27.1 on the
RX 7900 XT, with graph optimizations disabled and all model work on DirectML:

| Original case | Maximum probability error | Strict `<1e-4` gate |
| --- | ---: | --- |
| Whole 145,445 samples | `0.000016629695892333984` | Pass |
| First 48,000 samples | `0.00020307302474975586` | **Fail** |
| Nonzero 549 sample crop | `0.0030588507652282715` | **Fail** |
| 512 sample silence | `0.0009639859199523926` | **Fail** |

All four cases retained the reference language/confidence routing decisions and
had bitexact repeats. That does not waive scalar failures or establish human
language accuracy. Earlier CPU results below describe earlier graph revisions.

## Separate Windows qualification

The native v2 comparison **passed**, with `errors: []`. Both adapters completed
113 successful probes: two passes over all 56 inputs and one valid post-failure
isolation probe. All 99 columns met the finite/range/normalization rules;
same-backend repeats and the context-boundary metamorphic checks were bitexact.

All 56 projected core outcomes matched: 32 `detected`, 20 `best_effort`, one
`unsupported` and three `needs_language`. On the 32 natural utterances, both
backends matched all 32 human top-language labels. The 28 supported-language
utterances produced 24 confident detections and four best-effort routes. All
four Russian/Hindi utterances returned no supported-language route: one
confident unsupported result and three needing a language choice. There were
no confident human-label mistakes or unsupported-to-supported leaks in this
fixed sample. Two utterances per language do not establish general accuracy
or probability calibration.

The maximum observed Mac/Windows probability drift was `0.005328834056854248`
on `natural-hi_in-1`; one case had a different third-ranked candidate without
changing the winner or core decision. These remain diagnostics, with no new
scalar tolerance. The complete adapter replay against the independent legacy
reference again passed only `whole` (`9.69757093116641e-6`), while `first_probe`
(`0.00028502941131591797`), `partial_frame` (`0.0016963481903076172`) and
`silence` (`0.0009639859199523926`) failed the original strict `<1e-4` test.
The earlier standalone results above and their failed receipts remain intact.

The evidence binds the source checkpoint
`mlx-community/whisper-tiny-asr-fp16@77fa3f52b482ec80d086df55f893065a9baab172`,
including `model.safetensors` SHA-256
`1267601753d2996d68dc065d1de786895a7ad9a874d1d3548a5274c1bc067444`,
to the frozen Windows graph above. Exact config/token/frontend/runtime pins
are in the contract and verified by the sealed captures.

| Evidence file | SHA-256 |
| --- | --- |
| `cases.json` | `733e23ad9c6bb270104dc2cae07dfbf4d927bba5c46353c77443a4b03b67b579` |
| Native Mac `mac.json` | `8d526419c5d374200a1639276cb4bfaf2e7ace837e271697f880e790874bed96` |
| Native Windows `windows.json` | `5e1eb8977e1a80ebff1a381aa5d7cbfd287dc7d471284ecdda808f1c61899e4d` |
| `comparison.json` | `52a04d2e0376da9332bac00511482024fca5b6cba14760956d3baa92e1d1289c` |

The comparator always writes `activation_recommended: false`: it makes no
application activation decision. `passed: true` certifies this component
contract; assembled lifecycle qualification and activation remain separate.

The active, independently reviewed amendment is
[`windows-qualification-v2.json`](windows-qualification-v2.json), SHA-256
`c49f415cc4d68e429d803ee9f793cdd9c24a26e267d88cadf4304d7a459d1598`.
The attempted [`v1`](windows-qualification-v1.json) is retained unchanged.
Before any inference under either contract, v2 replaced the blocked viewer
row transport/selector with fixed first-two archive-member selection. Its
acceptance rules, case count and model-call budget are unchanged.
It does not replace the failed scalar checks above. No new graph, arithmetic
candidate, language threshold or independent expectation is introduced.

The frozen matrix contains the original four cases, the first two regular WAV
members in each FLEURS validation/dev archive for the 14 supported languages
plus Russian and Hindi, and 20 deterministic stress cases. Every natural input
is bound to FLEURS revision `70bb2e84b976b7e960aa89f1c648e09c59f894dd`;
the noisy French case reuses the retained quality pilot. Quiet, clipped,
constructed mixed/overlapping speech, nine nonzero short crops and three
30-second context boundaries have fixed recipes selected before inference.
The short/silent cases test numerical behavior; they do not bypass Silero's
speech-admission role or establish acoustic language accuracy.

Each backend calls its actual native `Network::probe`, including its frontend,
twice for all 56 inputs. All 99 columns must retain their code order, finite
probabilities in `[0,1]`, sum error at most `1e-5`, and bitexact same-backend
repeats. A fresh actual `LanguagePolicy` supplies each case's winner, routed
language, review flag, reason and remembered context; all five fields must
match the frozen Mac capture exactly. Top-three candidates, all-column scalar
drift and threshold distances are reported, without a new scalar tolerance.
Actual-core sequence tests separately cover unsupported speech, speaker
isolation, ties and adjacent confidence values.

Natural human labels are also checked. A new confident label mistake, or an
unsupported utterance routed to a supported language, fails. A confident
mistake shared with Mac remains an acoustic failure that blocks an activation
recommendation; it is never relabeled correct. Shared uncertain mistakes stay
visible. This small fixed sample is functional qualification, not a claim of
general calibration or accuracy in all 14 languages.

Prepare inputs once, without model execution, in a fresh private directory:

```sh
python experiments/windows/whisper/windows_qualification.py prepare \
  --retained /private/retained/quality-pilot --output /private/whisper-windows-v2
python -m unittest discover -s experiments/windows/whisper \
  -p test_windows_qualification.py -v
```

Preparation uses `numpy==2.3.3` and `soundfile==0.14.0` with its declared
dependencies. It bounds new corpus downloads to 60 MiB and prepared storage
to 150 MiB, and records source/normalized/derived byte hashes. Freeze `cases.json` before
the Mac capture; do not replace selected rows after any model result.

The Italian viewer request exceeds Hugging Face's server-side 300 MB Parquet
scan limit. `fleurs_archive.py` instead streams `dev.tar.gz` with stdlib
`tarfile` mode `r|gz`, closes after two complete regular WAV members, and
matches their basenames to the same revision's `dev.tsv` for ID, transcript
and sample count. Archive member order is authoritative; neither TSV nor
filename sorting changes selection. The Italian pilot obtained both complete
utterances after 1,198,080 compressed bytes, versus the 288,887,332-byte archive.
V2 permits up to 1 MiB for a complete per-config TSV; the aggregate cap remains
60 MiB. Compressed bytes consumed and the pinned tree's LFS identity are
recorded, without claiming a hash of the unread complete archive. No Parquet
reader is needed by the preparation or runtime path.
If transport preparation failed earlier, pass its cumulative byte cost with
`--prior-downloaded-bytes`; the same 60 MiB budget is reduced by that amount.

The ignored integration test is the same native adapter boundary on each
platform. Set `SPEAKERDESK_WHISPER_CASES` to the frozen manifest,
`SPEAKERDESK_WHISPER_RESULT` to a fresh result filename, and
`SPEAKERDESK_TEST_WHISPER` to the pinned platform model directory. On Mac, also
set `SPEAKERDESK_TEST_MLX` and `DYLD_LIBRARY_PATH` to the pinned runtime's `lib`
directory. The shared runtime guard checks the actual loaded dylib and
explicit Metal library before capture. On Windows, set
`SPEAKERDESK_TEST_WINDOWS_RUNTIME` to the private pinned runtime and
`SPEAKERDESK_WHISPER_MAC_REFERENCE` to the completed Mac capture; its SHA-256 is
bound before inference. Model loading still requires normal `ModelAdmission`.

```sh
cargo test --locked -p speakerdesk-models --test whisper_windows_qualification \
  native_adapter_captures_frozen_windows_contract -- --ignored --exact --nocapture
python experiments/windows/whisper/windows_qualification.py compare \
  --cases /private/whisper-windows-v2/cases.json --mac /returned/mac.json \
  --windows /returned/windows.json --output /returned/comparison.json
```

The harness writes every returned vector, exact Float32 bits and its hash to
a fresh failed-by-default receipt before checking validity or repeats. Errors
and unwinds retain those observations. The comparator preserves raw capture
identities and partial comparisons in a fresh failure report, then exits
nonzero. Complete comparable captures report the old strict results separately.
Neither tool activates inference or rewrites a reference.

The completed matrix used 112 complete probes per backend plus one valid
post-failure isolation probe, within the frozen maximum of 128. Empty and
nonfinite input rejection and pre-cancellation occur before graph entry.
Pre-cancellation does not prove in-flight ORT termination or reuse; that stays
an assembled lifecycle gate. Loaded DLL hashes bind the runtime, while the
existing exact-graph profile receipts establish component DML placement;
this adapter capture adds no per-node placement or application-latency claim.
Windows x64 dispatcher/setup is now enabled for assembled application
qualification. Release still requires the assembled application and
distribution gates; this component report does not decide activation.

## Retained scalar diagnosis and export tools

The observer graph reproduced the failed full graph's final probability bits
exactly. Conv1, Conv2 and added positions matched retained source values; the
first observed difference was encoder block 0. A paired encoder-0 diagnostic
then reproduced the full native block output bit for bit before substituting
the exact retained first-norm output at one consumer. The intervention removed
all 97 first-norm differences, but block-0 differences only fell from 156,990 to
93,132 of 576,000 values; maximum absolute error remained `0.00390625`.
Both modes repeated exactly and used only DML model work. First-norm fidelity
is one contributor, not a complete explanation. The remaining computation has
not been localized further; it may include the block's later normalization.

The 105,848,793-byte graph contains 7,669 nodes, including explicit source
rounding/materialization rules and the 256 KiB compiled GELU table. Standalone
native probes measured session loading at 3.07–3.13 seconds, first execution at
115–120 ms, and the second execution at 72.83–72.90 ms. These are traced neural
probe measurements, excluding frontend work and application scheduling; they
do not establish application latency, memory use or cancellation behavior.

Standard ONNX reductions do not specify the pinned Metal SIMD reduction tree,
and the current lowering does not reproduce every source normalization value.
The isolated Cast, convolution, GELU and materialization controls establish
their stated operation boundaries, not complete-model fidelity. Matching the
remaining scalar target would require more source arithmetic compatibility
work inside the block and renewed full-case qualification. Its cost is not
bounded by a demonstrated single remaining fix; it also adds source/runtime
pin and cross-device verification obligations. Further candidate guesses are
not authorized by this closed arithmetic-matching batch. The owner instead
authorized the separate prospective functional contract above, which has now
passed. That decision preserves the original scalar gate and independent
references; it does not establish matching source reductions or normalization.

The checkpoint is
[`mlx-community/whisper-tiny-asr-fp16@77fa3f5`](https://huggingface.co/mlx-community/whisper-tiny-asr-fp16/tree/77fa3f52b482ec80d086df55f893065a9baab172).
The model, config, and token metadata must match the production manifest's sizes
and SHA-256 values before conversion. The exporter consumes every neural weight
with its expected shape. It retains only the SOT embedding, first decoder
position, and 99 language embedding rows; it never exports a transcription
decoder or the unused vocabulary. Language codes preserve token metadata order.

The topology follows `crates/models/src/whisper_lid/network.rs` and the pinned
[`mlx-audio` Whisper graph](https://github.com/Blaizzy/mlx-audio/blob/feb25a37b07923bae556e59111995071d66afa0d/mlx_audio/stt/models/whisper/whisper.py).
It uses standard ONNX opset 17. The initial neural export uses Float16 operations
and constants, with Float32 input/output and final language softmax. Convolution
and bias addition remain separate because the source rounds convolution before
adding bias. The frontend uses Float32 fixed real DFT matrices and the exact
400 sample periodic Hann/80 band Slaney mel rules, then rounds mel features to
Float16. The graph builder is reused read-only from `../nemotron/export.py` and
its hash is recorded. There is no Python process at runtime.

Use an existing disposable environment with `onnx==1.19.1`, `numpy==2.2.6`, and
`onnxruntime==1.24.4`:

```sh
python experiments/windows/whisper/export.py /read-only/models/whisper-language /private/whisper \
  --positions /private/retained/positions.npy
python experiments/windows/whisper/qualify.py /private/whisper /private/whisper/cpu.json
python -m unittest discover -s experiments/windows/whisper -p 'test_*.py' -v
```

The second command writes the complete results and eight native probe cases,
then **exits with failure** if any original independent score, sampled mel, or
repeat check fails. It never regenerates a golden. `cases.json` enumerates four
neural and four frontend cases. Each named tensor case matches the shared Rust
runner in `../ort-probe`; it requires Float32 I/O only.

| Graph | Input | Output |
| --- | --- | --- |
| `frontend.onnx` | `audio_windows`: Float32 `[1,3000,400]` | `mel`: Float32 `[1,3000,80]`, with Float16-rounded values |
| `whisper.onnx` | `mel`: Float32 `[1,3000,80]` | `probabilities`: Float32 `[99]`, in `languages.json` code order |

Native framing validates finite, nonempty 16 kHz mono PCM, truncates or pads to
480,000 samples, applies 200 sample reflection padding on each edge, and gathers
3,000 windows at a 160 sample hop. The extra centered frame is excluded. Each
probe owns fresh input padding; the model has no persistent acoustic or decoder
state. The small framing test checks reflection, silence padding, truncation,
invalid input, and isolation from a later source.

Every neural `expected.json` contains both a macOS ONNX CPU tensor and the
**original independent 99 probabilities**, their code order, reference hash,
and case name. Frontend cases carry the existing 50 independent mel sample
points. Compare Windows CPU or DirectML results with:

```sh
python experiments/windows/whisper/compare_probe.py \
  /private/whisper/cases/neural-partial_frame /returned/native/partial-frame --provider dml
```

The shared verifier first admits the requested provider, pinned Windows ML runtime,
model/case/input hashes, and executed profile events. The comparator then verifies shape, hash, finiteness, normalized probability range,
both repetitions, and exact repeatability. It prints differences from macOS ORT
CPU separately from the independent reference gate. It also reports the top
probability, top-two margin, their distances from `0.90` and `0.20`, and whether
either routing decision changed. It exits nonzero when an original reference
bound fails. Matching the top language alone cannot pass; a focused negative
control verifies that a changed 99 score vector is rejected.

On macOS ORT 1.24.4 CPU, the assembled ONNX frontend and neural graph produced:

| Existing case | Max error across all 99 scores | Top probability | Top-two margin | Probability minus 0.90 | Margin minus 0.20 | Original `1e-4` gate |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| Whole 145,445 samples | `0.00000969745` | `0.9985323` | `0.9979162` | `+0.0985323` | `+0.7979162` | Pass |
| First 48,000 samples | `0.0000944138` | `0.9757239` | `0.9705232` | `+0.0757239` | `+0.7705232` | Pass |
| Nonzero 549 sample crop | `0.000588693` | `0.6208009` | `0.5347923` | `-0.2791991` | `+0.3347923` | **Fail** |
| 512 sample silence | `0.000414461` | `0.3838373` | `0.1220830` | `-0.5161627` | `-0.0779170` | **Fail** |

All 200 sampled mel points matched their independent values exactly, and repeat
errors were zero. Every case retained its top language and confidence outcome.
The NumPy frontend is also reported as a diagnostic; the first-probe vector
has additional drift through that alternate frontend. These observations do
not waive the original score failures or qualify human language accuracy.

The existing application owns routing and admission. Explicit language choices
bypass Whisper; the existing test
`core::language::tests::manual_override_skips_detection_and_marks_short_or_overlap`
owns that interaction. Silence here is a numerical graph case; it does not
establish speech admission or select a language for a silent passage. This lane
does not change the existing context fallback, supported-language decisions,
passage repair, or manual override rules.

Windows execution, actual provider placement, cancellation, and application
session ownership require native evidence from the owning lane. The original
low confidence score drift remains unresolved until that evidence and the
precision policy are reviewed. Source weights, prepared tensors, private
diagnostic intermediates, and machine profiles stay outside the repository.

The bounded precision investigation supplies one separate candidate through
`precision_candidate.py`; it does not overwrite `export.py` or the original
graph. The pinned Whisper attention calls `mx.softmax(..., precise=True)`.
MLX 0.32.2's [precise softmax kernel instantiation](https://github.com/ml-explore/mlx/blob/v0.32.2/mlx/backend/metal/kernels/softmax.metal#L14-L23)
uses a Float32 accumulator for Half inputs and rounds its output back to Half.
The original graph used Half input/output Softmax, which does not explicitly
require that accumulation rule. The candidate inserts Float32 casts around
exactly the 12 attention Softmax nodes and retains the Half output boundary.
The final 99-language Float32 softmax, constants, contractions, normalization,
frontend, and prepared case inputs remain unchanged.

Layer normalization has no established precision defect here. The
[ONNX specification](https://onnx.ai/onnx/operators/onnx__LayerNormalization.html)
defaults `stash_type` to Float32 for normalization and then casts back before
the affine stage. MLX's [layer normalization kernel](https://github.com/ml-explore/mlx/blob/v0.32.2/mlx/backend/metal/kernels/layer_norm.metal#L59-L111)
also reduces and normalizes in Float32, then casts to the input type before
scale/bias. The source [GEMM instantiation](https://github.com/ml-explore/mlx/blob/v0.32.2/mlx/backend/metal/kernels/steel/gemm/kernels/steel_gemm_fused.metal#L7-L25),
[GEMV accumulator](https://github.com/ml-explore/mlx/blob/v0.32.2/mlx/backend/metal/kernels/gemv.h#L18-L37),
and [implicit convolution](https://github.com/ml-explore/mlx/blob/v0.32.2/mlx/backend/metal/kernels/steel/conv/kernels/steel_conv.h#L93-L104)
use Float32 accumulation before Half storage. GEMV's bias epilogue itself
casts the accumulator to Half before bias; the selected kernel and provider
lowering matter. This investigation establishes no cause for changing those
contractions or their affine boundaries.

`precision_oracle.py` observes a small synthetic operator input, independent of
the audio fixtures. With MLX 0.32.2 GPU, default Half softmax differs from
`precise=True` in 16,808 of 27,000 values (maximum `0.0001220703125`). Explicit
Float32 softmax followed by Half conversion matches `precise=True` exactly.
The observation records runtime/library hashes and never writes an existing
model reference. It is an operator diagnostic, not a replacement golden.

```sh
# Run only the small source operator in the retained MLX 0.32.2 environment.
python experiments/windows/whisper/precision_oracle.py /private/softmax-observation
# Run graph preparation in the existing ONNX environment; output must be new.
python experiments/windows/whisper/precision_candidate.py prepare \
  /private/original-whisper /private/softmax-observation /private/precise-softmax
```

`precision.json` lists the four neural cases and `operator/case.json`. All use
Float32 I/O. Each neural case preserves its original independent expected
probabilities and macOS baseline, so the existing comparator still applies
the unchanged `1e-4` gate. The operator graph returns `native` and `precise`
Float32 tensors of shape `[1,6,3,1500]`; both are compared with the observed
MLX precise result after the shared native runtime/provider verifier:

```sh
python experiments/windows/whisper/precision_candidate.py compare-operator \
  /private/precise-softmax/operator /returned/native/operator --provider dml
python experiments/windows/whisper/compare_probe.py \
  /private/precise-softmax/cases/neural-silence /returned/native/silence --provider dml
```

The small macOS ORT 1.24.4 CPU operator check observed maximum errors of
`0.0000280067` for native Half Softmax and `0.0000152588` for the explicit
Float32 boundary; the latter differs in six values, each by one Half ulp.
The retained optimized graph shows the native Half Softmax inlined into
Float32 reduction/exponential operations without rounding its final result
to Half. The explicit Float32 operator keeps its output Half conversion.
This is direct evidence that the original precision boundary is insufficient
for that lowering. It does not establish the cause of every model score
difference, or predict DirectML's implementation.

The separate candidate graph SHA-256 is
`41027f0ee2ca1b6c63476dbc7f484b1506382170fa4a2a37cc3bed57efd0dc7c`.
Only the small operator and five focused tests were run in this correction
lane; the complete candidate has no new CPU or DirectML qualification here.
Native neural cases consume the prepared macOS mel tensor, so the separately
observed DirectML frontend drift cannot explain their existing neural drift.
Further precision changes require a new causal finding and owner decision.

The next isolated control, `gelu_candidate.py`, preserves the observed compiled
MLX Half GELU boundary. With the retained silence convolution output as its exact
input, multiplication by the Half reciprocal `0.70703125`, followed by Half
rounding after every primitive, reproduces all 1,152,000 retained MLX GELU values
exactly in the small ORT CPU operator check. Ordinary division by Half
`sqrt(2)` does not. The transform makes each rounding explicit with Float32
arithmetic followed by Half conversion. Division by two retains its exact
power-of-two scale and final Half conversion.

This control requires the frozen precise-softmax graph above. It changes only
the ten GELU subgraphs, retains every existing constant (including positions),
and sets `disable_optimizations` on copied neural cases to preserve the separately
observed convolution-before-bias Half boundary. The independent expectations and
their `1e-4` acceptance remain unchanged. No complete native qualification is
implied by the operator result.

```sh
python experiments/windows/whisper/gelu_candidate.py check-capture \
  /private/retained/conv1-pre.npy /private/retained/conv1.npy /private/gelu-observation.json
python experiments/windows/whisper/gelu_candidate.py prepare \
  /private/precise-softmax-v1 /private/compiled-half-gelu-v1
python experiments/windows/whisper/compare_probe.py \
  /private/compiled-half-gelu-v1/cases/neural-silence /returned/native/silence --provider dml
```

The sequential positional control accepts the complete maintained MLX positional
parameter, not acoustic outputs. `position-constant.json` binds its source,
generator, capture and raw Half bytes; its recorded runtime-identity limitation
is retained. The exporter now requires this pinned model parameter through
`--positions`, rather than recomputing it with another backend's transcendental
functions. The table is validated before graph construction.

```sh
python experiments/windows/whisper/position_candidate.py \
  /private/compiled-half-gelu-v1 /private/retained/positions.npy /private/source-positions-v1
```

This replaces exactly one initializer. All GELU corrections, disabled graph
optimizations, graph nodes, other constants, probe inputs and independent score
expectations remain unchanged. Use the existing `compare_probe.py` on the copied
neural cases. This control does not change the separate DirectML Float32-to-Half
rounding behavior.

The standalone `rounding_control.py` does not alter any model. It tests the
conversion boundary using the production-runtime source capture and a hostile
numeric matrix. ORT [v1.27.1 Cast lowering](https://github.com/microsoft/onnxruntime/blob/v1.27.1/onnxruntime/core/providers/dml/DmlExecutionProvider/src/Operators/DmlOperatorCast.cpp)
uses DirectML Cast without a rounding-mode option. Microsoft's
[floating conversion rules](https://microsoft.github.io/DirectX-Specs/d3d/archive/D3D11_3_FunctionalSpec.htm)
specify truncation for conversion to a lower floating format. DirectML's
[Round operator](https://learn.microsoft.com/en-us/windows/win32/api/directml/ns-directml-dml_element_wise_round_operator_desc)
has nearest-even integer rounding; ORT
[v1.27.1 selects that mode](https://github.com/microsoft/onnxruntime/blob/v1.27.1/onnxruntime/core/providers/dml/DmlExecutionProvider/src/Operators/DmlOperatorElementWise.cpp).

The control estimates a binary16 exponent, repairs a neighboring-bin estimate
with exact comparisons, looks up exact powers of two, rounds the scaled Float32
value, and rescales it to an exactly Half-representable Float32 value. It returns
native Cast, the grid result, and grid followed by Cast as separate outputs.
The comparison checks exact finite values, signed zero and infinity signs;
NaN payloads may canonicalize. Half subnormals and finite overflow are included.
The source convolution/GELU-product controls themselves stay within `[-16,16]`.

```sh
python experiments/windows/whisper/rounding_control.py prepare \
  /private/debug-pinned-runtime-v1 /private/rounding-control-v1
# Run the three case.json files listed in rounding.json through the native probe.
python experiments/windows/whisper/rounding_control.py compare \
  /private/rounding-control-v1/gelu-first-product /returned/native/gelu-product --provider dml
```

The local CPU 386,711-value hostile matrix includes every finite Half value and every
adjacent Half midpoint with Float32 neighbors, both signs, signed zero,
Float32/Half subnormals, power-of-two boundaries, overflow ties, infinities and
NaNs. Native `hostile-finite` filters that matrix to finite inputs with finite
nearest-even Half outcomes (`abs(x)<65520`), preserving both zero signs and
subnormal/tie/bin boundaries. Nonfinite values and finite overflow remain
local-only evidence; the native probe's finite input/output rules are unchanged.
The macOS ORT 1.24.4 CPU control matches NumPy's IEEE nearest-even conversion
exactly over that matrix and both 1,152,000-value source cases. This does not
qualify DirectML, or establish a complete-model correction.

After native qualification of that operator, `rounding_candidate.py` adds the
same grid before the frozen graph's 63 explicitly typed Float32-to-Half Casts:
the mel input, 50 GELU rounds and 12 attention Softmax rounds. It preserves the
original Cast operations, all existing constants and every other source node.
Its provenance separately inventories the implicit Half arithmetic results
(including Conv, MatMul, LayerNormalization, Add and Mul), whose internal
accumulation/rounding has not been established by the Cast control. Those
operators are unchanged. This candidate has no complete-model qualification.

```sh
python experiments/windows/whisper/rounding_candidate.py \
  /private/source-positions-v1 /private/nearest-half-v1
```

Run the original silence case and comparator first, retaining the existing
independent `1e-4` gate, before deciding whether any further model run is useful.

The residual `implicit_control.py` uses the frozen `nearest-half-v1` graph and
captures reproduced with the production MLX 0.32.2 library and Metal kernels.
Its first control isolates the second convolution's bias addition using all 384
channels at the two padded edges and one repeated silence interior position.
The exact contraction rounded to Half, followed by Half bias addition, matches
the maintained MLX operator for all 1,152 sampled results. In the local ORT 1.24.4
CPU control, Half Add followed by a Float32 consumer loses that rounding at 606
values; applying the already qualified nearest-Half grid fixes every value.
The [ORT 1.27.1 fallback transformer](https://github.com/microsoft/onnxruntime/blob/v1.27.1/onnxruntime/core/optimizer/insert_cast_transformer.cc)
promotes unsupported Half operators and allows cleanup of its inserted casts.
This is a concrete additional CPU boundary, beyond explicit source Casts.

The second control prunes the frozen graph after its first encoder LayerNorm,
retains original nodes/constants and adds six Float32 output observers. It
compares those outputs with the pinned source captures. Output observers can
change CPU promotion: the prefix CPU results cannot be treated as values from
the unobserved complete graph. The native DML prefix can locate an earlier
remaining discrepancy, with that observer limitation retained. Neither control
changes a complete candidate or its original strict probability gate.

```sh
python experiments/windows/whisper/implicit_control.py prepare \
  /private/nearest-half-v1 /private/debug-pinned-runtime-v1 \
  /private/conv2-pinned-v1 /private/implicit-control-v3
# Run only the two small case.json files listed in implicit.json.
python experiments/windows/whisper/implicit_control.py compare \
  /private/implicit-control-v3/conv2-bias /returned/native/bias --provider dml
python experiments/windows/whisper/implicit_control.py compare \
  /private/implicit-control-v3/frozen-prefix /returned/native/prefix --provider dml
```

The comparator verifies shared runtime/model/input/case identities and requires
all model work on DML when DML is requested. It reports exact stage differences
and rejects nondeterministic repeats. A zero stage difference remains diagnostic
evidence; complete-model qualification still requires the independent language
probabilities and routing checks.

After the native prefix found its first DML difference at Conv1, the isolated
`conv_control.py` compares native Half convolution with Float32 convolution over
exactly the same Half-valued mel and checkpoint weights. The Float32 branch
rounds to the qualified nearest-Half grid before bias addition, then rounds the
bias result separately. It exposes the raw Float32 accumulator, both rounded
convolutions and both bias results; the comparator requires the corrected bias
result to be bitexact to the actual production-MLX Conv1 capture.

The capture uses the maintained Conv1 call's `mx.conv1d` and separate bias with
the pinned production MLX library/kernel. Its bias result reproduces the
retained capture exactly. Raw source convolution has one edge value different
from Float64 exact contraction rounded to Half; the following bias masks that
raw difference. The exact contraction is therefore a diagnostic accumulator
reference, not a replacement for the source capture. Local CPU execution still
differs at one corrected bias value (`6.103515625e-5`); this control has not
passed native qualification and does not justify a whole-model candidate yet.

```sh
python experiments/windows/whisper/conv_control.py prepare \
  /private/nearest-half-v1 /private/conv1-pinned-v1 /private/conv-control-v1
python experiments/windows/whisper/conv_control.py compare \
  /private/conv-control-v1/conv1 /returned/native/conv1 --provider dml
```

The native Conv1 control's corrected bias output subsequently passed all
1,152,000 retained source values exactly on DML, with deterministic repeats and
no CPU model work. `conv_candidate.py` applies that same arithmetic class to both
convolution paths of the frozen `nearest-half-v1` graph. Each convolution now
uses Float32 copies of the exact Half checkpoint values, rounds its contraction
through the qualified grid before the original Half Cast, adds the bias
separately in Float32, then rounds through the same grid before the original
bias Half Cast. Conv2's matching operator class is evaluated by the complete
independent gate; it has no separate native operator qualification.

```sh
python experiments/windows/whisper/conv_candidate.py \
  /private/nearest-half-v1 /private/float-conv-v1
python experiments/windows/whisper/compare_probe.py \
  /private/float-conv-v1/cases/neural-silence /returned/native/silence --provider dml
```

The transformation checks that every other original node and all original
initializers are unchanged. The four original neural inputs, CPU comparisons,
independent expectations and strict `<1e-4` gate are copied byte for byte; graph
optimizations remain disabled. The complete candidate remains unqualified until
those gates run on the intended backend.

Source materialization determines the next conversion, rather than blindly
rounding every exported Half node. In MLX 0.32.2,
[LayerNorm](https://github.com/ml-explore/mlx/blob/v0.32.2/mlx/backend/metal/kernels/layer_norm.metal)
uses centered Float32 statistics, rounds the normalized value to Half, then
applies one fused Half affine operation. That affine rule matches every retained
first-norm value; a separately rounded product does not.
[Encoder GEMM](https://github.com/ml-explore/mlx/blob/v0.32.2/mlx/backend/metal/kernels/steel/gemm/kernels/steel_gemm_fused.h)
adds bias to its Float32 accumulator before the Half store, while the
[one-row decoder GEMV](https://github.com/ml-explore/mlx/blob/v0.32.2/mlx/backend/metal/kernels/gemv.h)
rounds the dot product before bias. Residual Add, scaling Mul and unbiased
MatMul each materialize Half outputs.

`materialization_control.py` tests those operation classes together against
fresh outputs from the pinned production library and kernels. Exact-lattice
matrix inputs separate rounding placement from reduction order. Cancellation,
subnormal and deliberately constructed affine double-round cases are included.
The fused-affine lowering recovers the Float32 sum's low part and resolves Half
midpoints; its finite Half-operand contract requires `abs(product+bias)<65504`.
The matrix also retains the complete captured first LayerNorm, so rare reduction
edge differences remain visible. Stage exactness is diagnostic; the original
independent `<1e-4` language probability gate remains the product decision.

```sh
DYLD_LIBRARY_PATH=/private/pinned-mlx/lib python \
  experiments/windows/whisper/materialization_control.py prepare \
  /private/nearest-half-v1 /private/debug-pinned-runtime-v1 \
  /private/conv2-pinned-v1 /private/source-materialization-v3
python experiments/windows/whisper/materialization_control.py compare \
  /private/source-materialization-v3/operators /returned/native/operators --provider dml
```

The case binds graph, inputs and source expectation metadata. The comparison
verifies runtime/provider placement and deterministic repeats, reports every
source difference, and does not promote operator differences into a new product
gate. This diagnostic creates no complete-model candidate.

`materialization_candidate.py` composes those reviewed source rules in the frozen
FloatConv graph: all 89 remaining MatMul operations, 20 encoder biased Linear
groups, and 22 LayerNorm operations. Decoder bias additions, residual additions,
scaling multiplications, convolution, GELU, positions and Softmax retain their
reviewed paths. Original parameters remain byte-identical; new Float32 parameter
copies contain exactly the same Half values.

The synthetic native norm differences were all the constant-4 row. A Float32
truncated reciprocal mean (`3.999999761581421`) predicts every differing output.
The candidate therefore uses the exact condition `min(row)==max(row)` to make
centered values zero for constant rows. Nonconstant rows keep the reviewed
centered-variance formula, without an epsilon or tolerance adjustment.

```sh
python experiments/windows/whisper/materialization_candidate.py \
  /private/float-conv-v1 /private/source-rules-v1
python experiments/windows/whisper/compare_probe.py \
  /private/source-rules-v1/cases/neural-silence /returned/native/silence --provider dml
```

This is one composition under the original independent score/routing/repeat
gates. All four inputs and references are copied unchanged. Rare intermediate
reduction-edge differences remain diagnostics, rather than an additional product
acceptance bar.

`boundary_diagnostic.py` observes one frozen source-rule candidate at 16 retained
source boundaries, from Conv1 through final probabilities. It preserves every
original node/initializer and adds only Float32 observers. Source captures,
input and independent gate metadata remain pinned. The comparison reports the
first observed difference and Half-ULP distances; optional baseline arguments
compare final probabilities with the existing single-output native run, since
adding observers can affect provider scheduling/fusion. This produces one graph
and case under the 150 MB artifact cap, and changes no acceptance gate.
The compiled-GELU graph is selected with `--candidate-sha256
1c47965522f7495fc85814c70c6b17dc4952436774c43225d55665e4ecbc7f14`
on both commands. Its existing native baseline is required, and any final-output
bit difference makes the diagnosis fail.

```sh
python experiments/windows/whisper/boundary_diagnostic.py prepare \
  /private/source-rules-v1 /private/debug-pinned-runtime-v1 /private/source-boundaries-v1
python experiments/windows/whisper/boundary_diagnostic.py compare \
  /private/source-boundaries-v1/silence /returned/native/boundaries --provider dml \
  --baseline-case /private/source-rules-v1/cases/neural-silence \
  --baseline-result /returned/native/original-single-output
```

`conv2_layout_control.py` separates the second convolution from its following
GELU on the same retained Half input and checkpoint parameters. It observes
pre-GELU results from Float32 Conv and explicit NLC windows plus Float32 MatMul,
then the scaled input, Erf and final GELU result. This distinction matters
because the retained `conv2` stage is after GELU. Both paths match the pinned
source locally; the native comparison determines the first differing primitive.

MLX's [implicit convolution loader](https://github.com/ml-explore/mlx/blob/v0.32.2/mlx/backend/metal/kernels/steel/conv/loaders/loader_channel_l.h)
advances kernel positions inside each 16-channel block. Natural NLC windows
therefore test a separate contraction implementation, without claiming to
reproduce the source kernel's literal accumulation order. This control creates
one small graph and changes no full-model candidate or independent reference.

```sh
DYLD_LIBRARY_PATH=/private/pinned-mlx/lib python \
  experiments/windows/whisper/conv2_layout_control.py prepare \
  /private/source-rules-v1 /private/debug-pinned-runtime-v1 \
  /private/conv2-pinned-v1 /private/conv2-control-v1
python experiments/windows/whisper/conv2_layout_control.py compare \
  /private/conv2-control-v1/conv2 /returned/native/conv2 --provider dml
```

The native Conv2 control found exact contraction and scaled inputs, followed by
4,495 differing Erf values and 2,997 differing GELU results. `gelu_table.py`
therefore compiles the maintained MLX GELU over all 65,536 binary16 patterns
using the pinned production library and kernels. The 256 KiB Float32 table
contains the complete function domain; it is independent of acoustic fixtures.
Source intermediate overflow and signed zeros are retained, and NaN inputs use
one canonical NaN index without promising payload or sign identity.

The native control tests every finite Half value and appends nonfinite patterns
inside the graph. Its finite diagnostic outputs report exact values, indices,
and explicit infinity/NaN/zero classes; the underlying table stays unchanged.
One shared graph also tests the retained Conv2 input. The comparator requires
bitexact outputs, deterministic repeats and DML execution for all model work,
including the dynamic Gather. No full candidate is generated by this control.

```sh
DYLD_LIBRARY_PATH=/private/pinned-mlx/lib python \
  experiments/windows/whisper/gelu_table.py prepare \
  /private/conv2-pinned-v1 /private/gelu-table-control-v1
python experiments/windows/whisper/gelu_table.py compare \
  /private/gelu-table-control-v1/complete-domain /returned/native/domain --provider dml
python experiments/windows/whisper/gelu_table.py compare \
  /private/gelu-table-control-v1/captured-conv2 /returned/native/conv2 --provider dml
```

Both native lookup controls passed every index, finite-value and classification
bit, with exact repeats and all model work on DML. `gelu_table_candidate.py`
replaces the ten complete GELU groups in the frozen source-rule graph with that
single admitted table. Original Half inputs and final Half outputs remain;
all other nodes, original constants, model I/O and four independent cases stay
unchanged. The complete model still requires the original strict score gate.

```sh
python experiments/windows/whisper/gelu_table_candidate.py \
  /private/source-rules-v1 /private/gelu-table-control-v1 /private/compiled-gelu-v1
python experiments/windows/whisper/compare_probe.py \
  /private/compiled-gelu-v1/cases/neural-silence /returned/native/silence --provider dml
```

`encoder0_norm_bypass.py` retains the frozen encoder-0 operations and parameters,
using identical exact positions in both modes. Only the first norm's consumer
selects computed versus retained source state. The comparator admits a causal
interpretation only after the baseline matches the existing native full-graph
observer bit for bit, the injected norm is exact, and both modes have identical
repeats and the requested provider. The result above closes this diagnostic;
the bypass is not a runtime model path.

```sh
python experiments/windows/whisper/encoder0_norm_bypass.py prepare \
  /private/compiled-gelu-v1 /private/debug-pinned-runtime-v1 \
  /private/conv2-pinned-v1 /private/encoder0-norm-bypass-v1
python experiments/windows/whisper/encoder0_norm_bypass.py compare \
  /private/encoder0-norm-bypass-v1 /returned/native/baseline /returned/native/bypass \
  --provider dml --frozen-case /private/compiled-gelu-boundaries-v1/silence \
  --frozen-result /returned/native/compiled-gelu-boundaries
```
