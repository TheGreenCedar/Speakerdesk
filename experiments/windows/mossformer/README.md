# MossFormer2 dynamic staged export

This build-only exporter preserves the pinned raw full-window separation graph
and its existing execution seams. It produces an encoder graph, 24 alternating
FLASH/FSMN graph pairs, and a finish graph: 50 graphs, matching the current
encoder/open work followed by 49 yieldable steps. Every stage retains the full
window. There is no RMS normalization, acoustic padding to a fixed duration,
sliding window, source permutation, or Python runtime component.

The checkpoint is
[`alibabasglab/MossFormer2_SS_16K@407cb030`](https://huggingface.co/alibabasglab/MossFormer2_SS_16K/tree/407cb030cd66340918ebb6c8cc63b18f8592cdbe),
SHA-256 `00a3a48bda492db1e829b85dd443f8f43a43039a3e90f1a24962ea9caf14a11a`.
The exporter verifies its production size/hash, then all 1,099 tensor names,
shapes, Float32 types, and per-tensor bytes against the existing Rust manifest.
It loads the maintained
[`ClearerVoice` source at `6b3774dc`](https://github.com/modelscope/ClearerVoice-Studio/tree/6b3774dc79c46ae8bed2a4fa5f706f0ac8c75c61/clearvoice/clearvoice/models/mossformer2_ss)
after verifying all five source hashes recorded by the independent reference
generator. The source package initializer must be empty.

`stages.py` calls the maintained encoder, normalization, position, attention,
memory, mask, and decoder modules. It only exposes their existing boundaries.
The export adapter disables rotary memoization and expresses padding as
`(-frames) % 256`. Tracing from 257 frames then includes a tensor padding/mask
path whose padding can also be zero at runtime. Decoder padding is always the
original sample count minus its kernel-16/stride-8 output length, preserving the
0–7 sample zero tail. These adaptations are recorded in `program.json`; no
upstream source file is edited.

The ONNX exporter also applies `normalization.py` to the two source
`GroupNorm(1,512,eps=1e-8)` sites: encoder `/norm` and finish `/global_norm`.
The original CPU InstanceNormalization lowering accumulated a biased variance
over about 25 million Float32 elements on the long reference. Saved-input F64
oracles isolated that reduction; a finish-only replacement reduced the
same-state Torch waveform discrepancy from about `0.03` to below `3e-5`.
The corrected lowering keeps centered variance, epsilon, and learned affine
bytes, but sums through bounded Float32 blocks of 512, 256, and at most 192
values. Padding applies only to the small partial-sum vector. FSMN's learned
per-channel InstanceNorm2d operations remain unchanged. The finish-only result
does not qualify the complete chain or its independent raw references.

An existing frozen export can receive the identical correction without tracing
the model again:

```powershell
python experiments/windows/mossformer/normalize_artifact.py inventory C:\private\moss-original
python experiments/windows/mossformer/normalize_artifact.py transform `
  C:\private\moss-original C:\private\moss-normalized
```

The inventory verifies all source graph hashes and rejects an unexpected global
normalization class. The transform preserves 48 graph files and the shared
parameter file exactly, changes only encoder/finish normalization, and retains
`source-program.json`, source program/helper hashes, and exact changed-subgraph
descriptors. Run the original whole-chain waveform gates on that one candidate;
no reference or threshold is rewritten by this transform.

`attention.py` validates the twelve source attention equations before lowering
them to supported matrix products and broadcasts. Batch is fixed at one.
The two `bgnd,bgne->bde` contractions become matrix multiplication over grouped frames.
DirectML otherwise lowers this two-axis reduction through a broadcast product:
5.125 GiB at the shorter reference and 24 GiB at the maximum window. The
standalone first FLASH graph crashed in that operation; preceding graph cuts
passed. Flattening the contracted axes preserves the source equation and
parameter bytes without that intermediate. Eight ellipsis equations also need
explicit MatMul/Mul forms: DirectML otherwise leaves them on CPU. The two
supported explicit-label equations stay unchanged. The exporter applies this
lowering directly; `attention.py SOURCE OUTPUT` applies it to the pinned
normalized export. Tests compare original ONNX and lowered operations against
independent Float64 formulas across the 256-frame grouping boundary.

Stage isolation matters separately from graph correctness: keeping all 50
DirectML sessions alive retained over 16 GiB of GPU committed allocations
halfway through even the shorter window. A diagnostic stopped that owned
process before the maximum-window driver reset could recur. The native network
now validates all graphs during loading, then creates and drops one ORT session
per stage with the DirectML memory arena disabled. Owned CPU outputs carry the
continuation state. Verified read handles keep graph and weight files immutable
until the network and its active recording state are dropped.

The retained native AMD run of program `3c82696f94670d288fd2f7a9e795ec6131e7f1417fb224b77842cd3252169246`
passed both complete independent waveform references. Maximum channel error was
`0.000326`, below the unchanged `0.002` maximum and normalized RMS bounds.
After setup, the 83,120- and 392,000-sample windows took 21.64 s and 37.10 s;
sampled GPU committed memory peaked at 8.48 GiB on the RX 7900 XT. These are
component measurements, not live-meeting backlog or installed-app evidence.
Maximum FLASH, FSMN and finish profiles place heavy matrix products,
convolutions and normalization on DirectML; small shape operations remain on CPU.

FLASH stages expose a one-tensor wrapper that calls the maintained module with
`mask=None` by keyword. Its original `forward(x, *, mask=None)` signature is
not directly compatible with the legacy ONNX exporter's argument normalization:
that helper appends the default mask as a positional argument and raises
`TypeError`. The focused check reproduces that failure before the wrapper and
verifies the wrapped call matches the original keyword invocation exactly.
The wrapper changes no tensor operation, mask rule, or parameter bytes.

The scripts require Python 3.11 or later. Use the owning Windows build-only
environment's existing CPU PyTorch. The exporter records its version and never
installs, downgrades, or pins that shared environment. The retained oracle used
PyTorch 2.8.0, NumPy 2.3.3, einops 0.8.1, rotary-embedding-torch 0.8.9, and
torchinfo 1.8.0; the Windows lane currently owns a newer PyTorch environment.
ONNX and ONNX Runtime are also needed for export and numerical checks. Version
differences require comparison with the retained outputs, not regeneration.

Prepare a copy of existing independent references without running a model:

```sh
python experiments/windows/mossformer/prepare_references.py /private/moss-references \
  /retained/reference-solo /retained/reference-max
```

This checks source/checkpoint identity, sample bounds, lengths, and finite raw
Float32 values, then seals copied inputs/outputs and original record hashes.
It never invokes the reference generator or changes the retained files.
The prepared two-case pack is about 5.5 MiB: 83,120 and 392,000 samples.

Run the heavy export and waveform gate on the owning Windows machine:

```powershell
python experiments/windows/mossformer/export.py `
  C:\private\models\mossformer2-16k\last_best_checkpoint.pt `
  C:\private\moss-source C:\private\moss-artifacts
python experiments/windows/mossformer/qualify.py `
  C:\private\moss-artifacts C:\private\moss-references C:\private\moss-artifacts\cpu.json
```

The source parent must contain `upstream/` with the five pinned files and empty
`__init__.py`. The artifact directory must be fresh. Export uses maintained
`torch.onnx.export(..., dynamo=False)` with standard opset 17 operators. A
content-addressed external-data writer deduplicates parameter bytes across
graphs into one `mossformer.weights` file, about 223 MB before deduplication.
Graph hashes, the shared parameter hash, source pins, exporter/stage hashes,
actual tool versions, and graph interfaces are recorded in `program.json`.

| Stage | Inputs | Outputs |
| --- | --- | --- |
| `encoder` | `audio`: Float32 `[1,N]` | `encoded`, `residual`: `[1,512,T]`; `x`: `[1,T,512]` |
| Each `flash-NN` / `fsmn-NN` | `x`: Float32 `[1,T,512]` | `x_out`: Float32 `[1,T,512]` |
| `finish` | Original `audio`, `encoded`, `residual`, final `x` | `source0`, `source1`: Float32 `[1,N]` |

`T = floor((N-16)/8)+1`. Native admission retains finite mono 16 kHz audio
with `16 <= N <= 392000`; graph tensor shapes do not replace that validation.
The caller owns the original source, encoded features, residual, evolving full
latent window, next stage, cancellation, and cleanup. It can check cancellation
before/after each graph and yield to primary work between stages. The separate
native cancellation test interrupts a maximum-size FLASH call and then verifies
that a fresh request on the same graph runs successfully. The measured return
was 54 ms, not a guaranteed deadline; the parent retains its bounded termination
and reaping fallback.

`qualify.py` executes the complete 50 graph chain through CPU ORT for both
independent references. It preserves source channel order and the original
maximum absolute and normalized RMS error bounds, both `2e-3`, independently
for each channel. It writes complete results before reporting a numerical gate
failure. It also prepares encoder, first FLASH, first FSMN, and finish calls
from the smaller reference for native CPU/DirectML probes. Alongside shared
weights, this selected case set is expected to remain below about 500 MB.

Each probe case declares `model_files` for the shared parameter file. Compare
returned native runs with:

```sh
python experiments/windows/mossformer/compare_probe.py \
  /private/moss-artifacts/cases/ref-0-mix_whole-finish /returned/native/finish --provider dml
```

The comparator first uses the shared verifier to admit the requested provider,
pinned Windows ML runtime, graph/case/input/external-weight identities, and
executed profile events. It then checks every output of both repetitions,
finite values, shapes, numeric CPU controls, and exact repetition. Finish cases
also carry the original independent raw outputs and use the original waveform
bounds. Per-call state comes from the complete CPU chain; those calls do not
establish a chain that feeds GPU-produced state through every later block.

The local small checks passed for external-data deduplication and reload,
absolute/relative error rejection, and exact maintained-source versus staged
composition at 24/25/31/2056/2063/2064 samples. Traced FLASH and FSMN stages also
matched eager source exactly at 2/255/256/257/258 frames. These use a small
one-block model to check composition and trace branches, not checkpoint quality.

One observed boundary remains explicit: maintained eager PyTorch rejects
16–23 sample inputs because its InstanceNorm2d wrapper rejects a single spatial
element, while the current Rust contract accepts 16 samples. ONNX normalization
may evaluate that mathematical case, but this lane has no independent raw
forward reference for it. The native full-window test also rejects reuse after
completion or cancellation between stages. Application scheduling and
installed-app behavior remain separate qualification gaps.
