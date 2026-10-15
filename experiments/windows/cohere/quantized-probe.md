# Exact-affine UINT4 feasibility gate

This is one bounded synthetic family, not a Cohere conversion or shipping
qualification. It keeps each original 4-bit nibble and affine scale/offset value;
it never approximates an offset with an integer zero point. Nontrivial BF16
values are stored exactly as FP32 metadata. No full FP32 weight matrix is saved.

The opset-21 graph uses UINT4 `DequantizeLinear` with scalar scale 1 and explicit
zero 0, groups the decoded nibbles into 64 values, multiplies the per-group
scales, adds the per-group offsets, and calls the existing FP32 MatMul shape.
The tiny case exposes reconstructed weights for a bit-exact scalar comparison;
all cases expose products for an independent NumPy control, under the existing
0.002 absolute/relative bounds. Repeats must be bit-exact. An unchanged weight
representation does not promise identical fused execution or model quality.

Prepare in the existing Windows export environment, with NumPy and ONNX:

```powershell
$base = 'C:\Users\alber\source\repos\Speakerdesk-worktrees'
$python = "$base\issue20-models\export-env\Scripts\python.exe"
$family = "$base\issue20-models\cohere-q4-probe-v1"
& $python experiments/windows/cohere/quantized_probe.py prepare $family
```

The destination must be new. There are no shape/count options or checkpoint
inputs. Preparation produces at most 128 MiB across three cases:

| Case | Input | Weight shape | Distinct matrices | Purpose |
| --- | --- | --- | ---: | --- |
| `correctness` | `[3,128]` | `[4,128]` | 1 | Nibbles, group boundaries, signed scales, fractional offsets, products |
| `one` | `[114,4096]` | `[4096,4096]` | 1 | Cold/warm execution and idle allocations |
| `eight` | `[114,4096]` | `[4096,4096]` | 8 | Distinguish packed persistent storage from full expansion |

One large matrix has 8 MiB of UINT4 data plus 2 MiB of exact FP32 scale/offset
metadata, versus 64 MiB for its expanded weights. Eight have 80 MiB of packed
data/metadata versus 512 MiB expanded. Each matrix has its own seeded packed
bytes and metadata; they are not aliases. The graph has independent matrix
branches. Idle samples distinguish persistent storage; sampled execution peaks
cannot exclude temporary allocations shorter than the sampling interval.
Reference generation decodes only 128 rows at a time.

Build the existing native probe; it has opt-in case fields for this gate:

```powershell
cargo build --release --locked --manifest-path experiments/windows/ort-probe/Cargo.toml --target-dir "$base\issue20-target"
```

The owning lane must run a timestamped memory/timeout wrapper around each command
below. Use one owned process at a time, a **120-second limit for the entire native
family**, and a **2 GiB GPU committed allocation stop**. The wrapper must sample
the owned PID's GPUProcessMemory committed/dedicated/shared allocations and
private/working memory at about 500 ms, retain missing-counter failures, and reap
each process before the next case. These are diagnostic budgets, not a new
product performance target. Build/preparation are outside the native-run timer.

```powershell
$probe = "$base\issue20-target\release\speakerdesk-windows-ort-probe.exe"
$runtime = "$base\issue20-models\runtime\onnxruntime.dll"
$results = "$base\cohere-q4-probe-v1-results"
New-Item -ItemType Directory $results
# Run each invocation through the owning lane's bounded memory wrapper.
& $probe $runtime dml "$family\correctness\case.json" "$results\correctness"
& $python experiments/windows/cohere/quantized_probe.py compare $family $results --case correctness
# Proceed only after the tiny numerical/provider gate passes.
& $probe $runtime dml "$family\one\case.json" "$results\one"
& $python experiments/windows/cohere/quantized_probe.py compare $family $results --case one
& $probe $runtime dml "$family\eight\case.json" "$results\eight"
& $python experiments/windows/cohere/quantized_probe.py compare $family $results
```

Every case disables ORT graph optimization and the DML memory arena, rejects CPU
fallback at session creation, and requires exclusively executed DML provider
events. The comparator reuses the existing verifier for the pinned Windows ML
2.4.89 runtime/DLLs, graph, case, input, external-file and profile identities.
It checks all output sizes/hashes/shapes, finite values, complete repetitions,
exact small reconstructed weights and bounded product errors. It also verifies
the complete prepared family file inventory, including independent references.

The native process writes and flushes three phase receipts, each with `pid`,
`unix_ms` and `idle_ms=2000`, then waits for two seconds:

1. `phase-baseline.json`: before initializing ORT.
2. `phase-loaded.json`: after session load, before either request.
3. `phase-warm.json`: after both requests and output release, before session drop.

Use the last stable samples within each phase; preserve raw samples and process
exit status. Compare one versus eight directly. GPU counters can be absent before
ORT initialization; retain those nulls rather than treating them as measured zero.
Persistent growth near full FP32 inventory fails
the compact-residency requirement. Record execution peaks separately: a temporary
matrix is permitted only when bounded and measured; a whole-inventory temporary
peak is not evidence of bounded per-operation reconstruction. Driver/compiler
allocations make simple payload arithmetic a reference, not an exact GPU-memory
prediction. Missing/ambiguous memory samples leave the residency gate unresolved.

The Python comparator deliberately reports residency as **UNDECIDED** even if
all numerical/provider checks pass. It does not consume memory-counter receipts
or claim source-quality equivalence, model throughput or live-meeting behavior.
Stop on the first failed gate; do not export the whole checkpoint or generate
another candidate automatically.

## RX 7900 XT result, 2026-10-15

The complete native family finished in 23.55 seconds. All outputs matched their
independent controls exactly, both repetitions matched, and executed profile
events were exclusively DirectML with CPU fallback disabled.

| Stable allocation | One matrix | Eight matrices | Per additional matrix |
| --- | ---: | ---: | ---: |
| Loaded dedicated GPU | 24.176 MiB | 98.145 MiB | 10.567 MiB |
| Loaded total GPU committed | 37.652 MiB | 188.676 MiB | 21.575 MiB |
| Warm total GPU committed | 45.684 MiB | 225.215 MiB | 25.647 MiB |

Six stable readings per loaded/warm phase, bound to the correct PID, support
compact persistent storage on this device. The loaded dedicated slope closely
tracks the 10 MiB packed payload and metadata per matrix, rather than 64 MiB
expanded weights. Total GPU committed includes shared allocations.

The highest observed allocation was 225.215 MiB. This is not a true peak bound:
the two eight-matrix requests took 63.7 and 66.4 ms, shorter than the roughly
330 ms counter interval. Full-model temporary allocation, throughput and
transcription quality remain unqualified. This result permits preparing a
compact candidate; it does not admit that candidate to the application.

Raw evidence is retained in the owning task's `cohere-q4-probe-v1-results`
directory: per-case inputs/outputs hashes, runtime/provider profiles, phase
receipts, numerical comparisons and `samples.jsonl`.

Local controls (Python 3.11+; no NumPy/ONNX needed for these controls):

```sh
python -m unittest discover -s experiments/windows/cohere -p test_quantized_probe.py -v
cargo test --locked --manifest-path experiments/windows/ort-probe/Cargo.toml
```

## Bounded scheduling contrast

The full compact encoder passed load-only placement inspection but exceeded its
resource guard on the first request. The original synthetic branches were
independent. This additional **single pair** tests the dependent weight lifetimes
that the original family did not cover; it is not another full-model attempt.

Both graphs contain the same eight distinct `[3072,3072]` UINT4 affine matrices
and a dependent MatMul chain, with input `[frames,3072]` and representative
`frames=114`. Each intermediate product is multiplied by the exact power-of-two
constant `1/256` before becoming the next input, keeping the scalar/NumPy controls
finite. All eight products are compared under the unchanged 0.002 bounds; both
repetitions must match exactly. The reference is computed once in bounded row
blocks and shared by both cases. No checkpoint, retained-model oracle or original
FP32 model is read. The earlier tiny correctness graph and its reference remain
the fail-fast nibble/affine control.

`interleaved` preserves generation order. `hoisted` stable-partitions the complete
input-independent node closure, including the existing weight Transposes. Every
node, initializer, input/output and operation remains identical; only node order
changes. Preparation checks this identity and replays the pinned ORT default
reverse-DFS traversal. Its modeled live transposed-weight inventory must be
**288 MiB interleaved versus 36 MiB hoisted**. Hoisting only reconstruction nodes
does not produce the intended contrast.

Both cases set `disable_graph_fusion=true`, which sets the maintained
`ep.dml.disable_graph_fusion=1` option. This prevents later DML fusion from
appending new high-index weight islands and undoing the controlled ordering.
The option defaults to false for all existing cases. Both cases also disable ORT
optimization and the DML arena and retain strict DML-only admission. These
controls do not guarantee a particular physical allocation or GPU lifetime.

```powershell
$family = "$base\issue20-models\cohere-scheduling-v1"
& $python experiments/windows/cohere/quantized_probe.py prepare-scheduling $family
```

The new destination contains `family.json`, `interleaved.onnx`, `hoisted.onnx`,
one shared `weights.bin`, one input, one reference array set, and per-case
`interleaved/case.json` and `hoisted/case.json`. The graph-local ONNX external data
location is simply `weights.bin`; it does not escape the graph directory. Packed
weights and affine metadata occupy 45 MiB once. Preparation rejects a complete
family larger than 128 MiB. There are no shape/count knobs or expansion variants.

After adversarial review and the existing tiny correctness control, root owns
these three invocations under one wrapper. The tiny control may use a fresh case
copy with `disable_graph_fusion=true`, preserving its graph, inputs and reference.
Use fresh output directories and reap each owned process before proceeding:

```powershell
$results = "$base\cohere-scheduling-v1-results"
# Each native command runs through the owning lane's bounded wrapper.
& $probe $runtime dml "$family\interleaved\case.json" "$results\interleaved"
& $python experiments/windows/cohere/quantized_probe.py compare-scheduling $family $results --case interleaved
& $probe $runtime dml "$family\hoisted\case.json" "$results\hoisted"
& $python experiments/windows/cohere/quantized_probe.py compare-scheduling $family $results
```

The **120-second whole-family limit**, **2 GiB sampled GPU committed guard** and
**2 GiB process private-commit guard** include the tiny control and this pair.
Stop on the first failed gate. Never increase a guard, retry the full model or
start native decoding automatically. Preparation/build are outside the native
timer. Timestamped GPUProcessMemory samples remain required; missing counters
fail the sampled GPU policy and cannot be replaced by a process-memory number.

The native probe writes `phase-run-{0,1}-start.json` immediately before each Run,
after input loading, and `phase-run-{0,1}-complete.json` after Run returns, before
output serialization. These receipts carry run index, PID, UTC Unix milliseconds,
`model_run_returned` and `private_commit`. Baseline/loaded/warm idle phases last
1500 ms for this pair. A start marker identifies entry to a request; it does not
claim that execution or device work completed.

On Windows, `private_commit` contains `current_bytes`, `lifetime_peak_bytes`, and
source `GetProcessMemoryInfo/PROCESS_MEMORY_COUNTERS_EX`. `PrivateUsage` measures
current process private commit; `PeakPagefileUsage` measures its lifetime peak.
This is neither resident physical RAM nor GPU peak. The result records the same
process-lifetime measurement after both requests and profile collection. It
includes loader, compiler, driver, inputs and output handling. No counter is reset
between requests; use one fresh process per graph.

The fixed decision requires both numerical/provider/repeat gates, no resource
guard violation, pre-Run current **and lifetime-peak** private-commit baselines
within **36 MiB**, and an inference-complete peak above the pre-Run lifetime peak
for each case. Hoisted lifetime peak must be at least **108 MiB (three expanded
matrices)** below interleaved. Loader-dominated peaks or a smaller effect leave
the scheduling contrast unproven; the comparator exits nonzero for that outcome.
These thresholds are frozen before measurement. A pass establishes a bounded
process-commit scheduling contrast, not allocation-site attribution, GPU peak,
packed persistent residency of the full model, or a shipping claim.

The focused optional ONNX control runs only in the existing generation
environment. It builds a tiny dynamic dependent chain, checks both graph orders
and external storage, and compares the single NumPy control against scalar affine
matrix products. It skips locally when ONNX is unavailable; it never runs ORT.

The single native scheduling family passed on the AMD PC on 2026-10-15. Its
59,815,706-byte input family is pinned by manifest
`5aa8c988faf9145a06caab4016eb865f1cc1b6a56583fcafbd5d26d8eff8b64f`.
All outputs match the unchanged reference bounds (maximum absolute difference
`3.6508e-7`), repeats are exact, and both graph orders return identical bytes.
Profiles contain only DirectML events. Lifetime peak private commit fell from
594,743,296 to 335,712,256 bytes, a 247.03 MiB reduction; loaded current and peak
baselines differ by less than 9 MiB. Both have distinct inference peaks. All
three owned processes exited normally and were independently observed absent.

The sampled GPU observations miss the short requests and do not establish GPU
peak. This result supports one full-model ordering/fusion-disabled candidate
under the unchanged full-model guards; it does not qualify that candidate or
prove its allocation cause. Receipts remain under `cohere-scheduling-v1-results`.

The resulting full-model family also passed once on the same PC. Ordered manifest
`3b4b81dd69360dc62e245645aeef2a6cf7d0f5d18ea778cb0abd1afe7543767f`
preserves all graph operations and weights; only node order changes. Encoder,
decoder-init and decoder-step match the retained references within `5.7221e-6`,
with byte-identical repeats. The family took 38.50 seconds under the original
8 GiB sampled GPU / 6 GiB private-commit / 180-second guards. Encoder lifetime
peak private commit was 3,064,528,896 bytes; maximum sampled GPU commitment was
2,461,843,456 bytes. The exact placement policy still permits only proven integer
shape/index work on CPU. These are retained tensor checks; full transcription,
combined model residency and the shipping application remain separate gates.
