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
