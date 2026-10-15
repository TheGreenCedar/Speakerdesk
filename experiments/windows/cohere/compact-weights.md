# Compact original-weight Cohere candidate

`compact_weights.py` creates one unqualified candidate from the frozen
**unsharded `cohere-fp32-v2`** graphs and the already pinned MLX checkpoint. It
does not trace PyTorch, run inference, generate references, change a checkpoint,
or modify the diagnostic source directory. `export.py` is imported only to reuse
its exact `source_name` mapping; the existing export environment supplies those
imports. The converter itself uses ONNX, NumPy and bounded file reads.

The five graph/weight pins are frozen in the tool, including the original
7,581,848,576-byte encoder parameter file. The checkpoint, configuration,
tokenizer and independent token/text reference also have frozen pins. The 119
retained case descriptors, inputs, expected arrays, prompts and provenance files
are separately frozen in `compact-retained-pins.json`, whose exact SHA-256 is
`fe7882c8eed28d87c40b348f16ecab42bba0335edc52d593356ac56aa22433c7`.
It uses canonical LF line endings for Git checkout stability. The original
Windows CRLF receipt had SHA-256
`03672c204f198738210a5ee401d73b4947e1c4e86eca09020e254d839bb2876e`;
all 119 semantic file pins are unchanged.
That inventory was captured from the existing FP32-v2 diagnostics before compact
conversion; it is not a new MLX oracle or a regenerated reference. The converter
verifies its hardcoded inventory hash and opens/verifies all 119 files before
graph processing. `provenance.json` remains supporting origin evidence, not a
self-authenticating admission authority. Every source is hashed before use; the
sharder supplies Windows read handles that deny
writes/deletion throughout graph rewriting and copying. Graphs are parsed from
the hashed byte snapshots, with external weights kept as metadata. No default
external-data loading or full checkpoint loading is used.

For each external initializer, the converter:

1. Resolves its exact MLX name using the existing export mapping, checks its
   source shape/type, and rejects missing names or unsupported external metadata.
2. For an original U32-packed parameter, reconstructs at most 1 MiB of FP32 rows
   at a time and compares **every byte** with that parameter's frozen FP32 range.
   A mismatch stops conversion; it cannot be repaired by new quantization.
3. Copies the original packed bytes unchanged into UINT4 external data. Original
   BF16 scales/offsets are widened exactly to FP32 metadata. The tested scalar
   UINT4 dequantization, group multiply/add and reshape produce the original
   initializer name immediately before its first consumer. Original MatMul,
   Gather, transpose and other consumers remain intact.
4. Copies ordinary FP32 parameters from their original ranges with bounded I/O.
   Exact source intervals share a stored range; decoder-init and decoder-step
   use one shared decoder store. Weight files are capped at 1 GiB.

Both graph uses and unique output ranges appear in `compact-manifest.json`.
Every source tensor must be accounted for. The sole permitted unused source
alias is the known `decoder.embed_tokens`/`proj_out` tie, and only if packed
weights, scales and offsets each have identical shape, type and bytes. Its exact
alias/hash record is retained. Unknown leftovers or a nonidentical tie fail.

The tool calls ONNX's version converter for opset **17 → 21**, rather than
changing the version number. For example, ReduceMean axes become an input
Constant under the maintained adapter. It explicitly changes **IR 8 → 10**,
because UINT4 requires IR 10. All other model/graph metadata, original initializer
and graph I/O fields must survive that conversion unchanged before rebinding. ONNX's checker
validates the saved graphs; it does not execute the graphs or expand their
external parameter files into model arrays.

Retained encoder/init/step inputs, `expected.json`, and expected output arrays
are copied byte for byte from their already verified, locked handles. Their
immutable source pins and destination pins are recorded; changed originals
cannot be admitted by calculating new hashes during conversion.
Only `case.json` graph/weight admission and session policy change. The independent
token/text reference retains its frozen hash. These cases compare against the
retained FP32 eager arrays; they do not establish original MLX arithmetic
equivalence or general transcription quality.

Use Python 3.11 or newer in the existing export environment. The macOS system
Python 3.9 lacks the hashing API used by these diagnostics.

Run the focused controls in that environment **before** any full
conversion:

```powershell
& $python -m unittest discover -s experiments/windows/cohere -p test_compact_weights.py -v
```

One control rewrites and checks a tiny external-data graph containing a
ReduceMean axes attribute, verifies decoder range reuse and, when ONNX Runtime
is installed, compares its output against independently calculated arithmetic.
Other controls cover same-length tampering of each retained source class and its
inventory, exact affine bytes, rejected mismatches, hostile checkpoint
headers, source ties, bounded shard copies and incomplete writes. On Mac without
ONNX, that graph control is explicitly skipped; the available controls do not
substitute for it.

After the owning lane reviews this implementation, the intended single command
is:

```powershell
& $python experiments/windows/cohere/compact_weights.py `
  <retained-cohere-fp32-v2-directory> <existing-pinned-mlx-cohere-directory> `
  <fresh-cohere-q4-v1-directory>
```

Use the existing Python environment and source files. No downloads, environment
changes or checkpoint export are required. Budget one conversion, at most
**10 minutes**, **4 GiB private process memory**, and **3 GiB candidate output**;
an owning-process wrapper should enforce those diagnostic limits. The time is
mostly source hashing, bounded exact-byte checks and copying, not inference.
No automatic alternative candidate or retry is authorized by this tool. A failed
conversion may leave a partial fresh output directory without a manifest.

The single conversion completed on the AMD Windows machine in **45.60 seconds**
with exit 0. All six controls passed in its existing export environment. It
accounted for all 3,136 source tensors, including 517 packed parameters and 582
quantized graph uses; all reconstructed FP32 bytes matched their frozen source.
The complete diagnostic directory is 2,118,944,990 bytes, including retained
cases and references. This is an unqualified candidate, not a model release.

Native qualification is separate. The first strict encoder initialization
failed before inference because ORT assigned some graph nodes to CPU. One verbose
initialization diagnostic identified CPU-preferred shape/index operations and
was stopped at its 60-second wall-time guard; no inference ran. Both failures
remain retained.

The candidate case policy
disables ORT optimization and the DML memory arena and requests strict DML with
CPU fallback disabled. The production Cohere loader used Level3 optimization
at this diagnostic stage. A strict full-graph placement failure is evidence
to review, not permission to add numerical CPU fallback.

The next native gate must retain expected outputs/tokens/EOS, repeated requests,
decoder cache continuation and existing numerical bounds, then measure full-model
persistent/temporary GPU allocations and latency. The synthetic packed-matrix
result does not establish those full-model properties, particularly if DML fuses
a larger graph. The later ordered candidate and draft registry integration are
recorded below.

## Placement admitted; execution memory gate failed

Load-only native API inspection now accounts for every non-Constant graph node.
Independent graph review matched each name/domain/operator to its assignment and
proved every CPU output is INT64 shape/index work, with ancestry terminating only
at Shape metadata or INT64 literals. Omitted nodes are embedded Constant literals;
none performs numerical computation. Every complete five-node UINT4
reconstruction chain and its numerical consumers is assigned to DirectML.

| Graph | DirectML nodes | CPU shape/index nodes | Omitted Constant literals |
| --- | ---: | ---: | ---: |
| Encoder | 8,322 | 1,228 | 2,884 |
| Decoder init | 1,221 | 161 | 292 |
| Decoder step | 1,004 | 128 | 227 |

The frozen placement group has SHA-256
`b1df4f33abb3ff27b101ee1def18fc417fe30f90012220e203813f0152a0e56a`.
The probe admits those exact graph-specific policies before Run. This permits
the proven integer work, without admitting arbitrary numerical CPU fallback.

One retained-output family then attempted the unchanged `.002` absolute/relative
bounds and two repeats, with optimization and the DML memory arena disabled.
The encoder loaded in 9.332 seconds and passed placement admission, but its first
execution attempt exceeded the **6 GiB private-memory guard**. The last sample
recorded 8,355,348,480 bytes private memory and 7,885,697,024 bytes GPU commitment.
The owned process was killed/reaped at 16.290 seconds and independently observed
absent afterward. No output tensors or completed profile were returned; neither
decoder started. The 8 GiB GPU guard was not exceeded in sampled observations,
which do not establish the true peak.

This is a resource failure, not a numerical comparison result. The pre-Run
receipt's `model_run_called: false` describes that earlier snapshot, not the
interrupted process's final execution state. The allocation cause, numerical
correctness, full decoding, residency and throughput remain unqualified. Keep
the failed candidate and receipts; raising the guard does not resolve compact
execution. Private evidence is retained under
`cohere-compact-qualified-placement-v1-results` in the owning work record.

## Ordered candidate and draft Rust integration

The subsequent candidate stable-partitions the complete input-independent node
closure before the remaining nodes and disables DML graph fusion. Only node
order changes: each NodeProto, all other model fields, external weights and
retained references remain byte-identical. The three retained-output graph gates
passed in 38.5 seconds with the original `.002` bounds, two repeats, 8 GiB sampled
GPU commitment, 6 GiB process private commit and 180-second family limits.
Independent profile review matched every CPU event to the reviewed INT64
shape/index nodes and found no numerical CPU execution. These are retained
graph-case results; full resident transcription and throughput remain separate.

The ordered manifest SHA-256 is
`3b4b81dd69360dc62e245645aeef2a6cf7d0f5d18ea778cb0abd1afe7543767f`.
The draft artifact inventory pins its three graphs, three external weight stores,
tokenizer and three graph-specific placement policies: **2,094,730,822 bytes**
total. Exact artifact admission retains all verified file handles. The loader
parses each policy and hashes its graph through those admitted handles, enables
API27 assignment recording before session construction, and checks every actual
assignment before returning the session. Unknown providers, numerical CPU work,
missing/extra/duplicate identities or a graph-hash mismatch fail closed. This
enforces placement on the product-selected HighPerformance adapter instead of
assuming it is the probe's adapter0. The shared admission source is also used by
the native probe; its hostile placement tests move with that implementation.
The dedicated Cohere loader uses disabled ORT
optimization, serial execution, no memory patterns, four intra-op threads,
disabled DML memory arena, disabled DML graph fusion and the probe's enabled CPU
arena. Other model loaders retain their existing settings.

The ignored Rust resident test passes both exact 31-token/text requests,
decoder-cache continuation, tiny-crop and pre-cancel checks under the original
guards. Lifetime peak private commit was 3,717,390,336 bytes; maximum sampled GPU
commitment was 3,284,135,936 bytes. Its `COHERE_MEMORY` receipt captures brief
process-commit peaks between external samples, not a precise GPU peak or resident
RAM. The 9.09-second clip took 12.95/11.12 seconds, so live throughput failed.
Publication, seven-model residency and assembled compact-app qualification remain
open. Earlier failed candidates and receipts remain retained.

## Decoder throughput diagnostics

The converter now makes stable input-independent ordering reproducible. For the
decoders it also folds only embedded INT64 scalar `Unsqueeze` operations with
literal axis 0 or -1 into equivalent one-element initializers. It preserves each
surviving NodeProto, every original initializer, unrelated protobuf fields and
all weight/reference bytes; controls include signed INT64 extremes, dynamic and
floating inputs, invalid axes, duplicate identities and idempotence. These graphs
are experimental: changing their hashes requires fresh reviewed placement
policies and native qualification before changing the draft artifact registry.

The same retained inputs and `.002` absolute/relative comparisons were used in
these consecutive diagnostics. Each speed gate requires **every** completed Run
to be at most 150 ms, with two repeats, 6 GiB lifetime private commit, 8 GiB sampled
GPU commitment and a 180-second family budget. Each failed at init, so none ran
its subsequent decoder-step case or changed production settings.

| Decoder-init configuration | Cold / repeated Run | Result |
| --- | ---: | --- |
| DML fusion enabled, original ordered graph | 457.09 / 328.57 ms | Speed failed |
| Exact INT64 shape fold | 434.43 / 310.14 ms | Speed failed |
| Shape fold with GPU-bound KV outputs | 438.16 / 301.21 ms | Speed failed |
| Shape fold with decoder-only memory pooling, ordinary outputs | 163.47 / 23.76 ms | First-call speed failed |

Offline comparisons for each failed gate preserve all 66 output tensors,
repeated bytes and greedy-logit identity; maximum reference error is 5.72205e-6.
The pooling run's lifetime private peak was 980,033,536 bytes and maximum sampled
GPU commitment 880,242,688 bytes. All owned processes exited and were reaped.

The probe's optional `dml_device_outputs` is limited to physical adapter 0. It
records actual output allocation, waits for input/output synchronization inside
the Run timer, and records oracle copies separately. A tiny exact-output native
gate verified that API before the full diagnostic. A retained `DynValue` alone
does not prove GPU storage, and this diagnostic does not prove cross-session GPU
input ownership for the production HighPerformance adapter.

The [pinned DirectML provider](https://github.com/microsoft/onnxruntime/blob/df2ba1cf81/onnxruntime/core/providers/dml/DmlExecutionProvider/src/ExecutionProvider.cpp#L1187-L1202)
disables post-load bucket rounding when its memory arena is disabled; the
[nonbucket allocator path](https://github.com/microsoft/onnxruntime/blob/df2ba1cf81/onnxruntime/core/providers/dml/DmlExecutionProvider/src/BucketizedBufferAllocator.cpp#L123-L130)
creates new committed resources.
Enabling decoder pooling materially reduces repeated-shape cost. The remaining
cold overhead spans many DML nodes; host profile spans cannot distinguish resource
creation, first-use compilation, GPU execution and waits. Do not omit first-use
work from timing or reinterpret the failed cold-call gate as a family pass.
