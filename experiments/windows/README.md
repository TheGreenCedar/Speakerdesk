# Windows qualification

The Windows production inference dispatcher remains disabled. Native component
implementations and their tests are available, but they do not yet constitute a
working Windows application. [Issue #20](https://github.com/TheGreenCedar/Speakerdesk/issues/20)
owns the complete application acceptance; [PR #49](https://github.com/TheGreenCedar/Speakerdesk/pull/49)
remains a draft.

Start at the model's adjacent Windows implementation and the linked experiment
record when investigating a numerical or lifecycle failure. Python is used for
export and qualification only; it is not part of the application runtime.

| Component | Current boundary | Implementation / evidence |
| --- | --- | --- |
| Silero | Native recurrent/reference, reset, framing and cancellation checks pass. | `crates/models/src/silero/network_windows.rs`; original experiment below |
| Cohere | Native text, tokens, decoder continuation and sharded artifact checks pass. | `crates/models/src/cohere/network_windows.rs`; [Cohere](cohere/README.md) |
| Nemotron | Native offline/live sessions and cache compression reproduce the qualified long run. | `crates/models/src/sortformer/{network,session}_windows.rs`; [Nemotron](nemotron/README.md) |
| ReDimNet | Component vectors and frozen separated-voice decisions pass. Ordinary recognition matches Mac, but the strict new pilot has one abstention; Windows policy approval is pending. | `crates/models/src/redimnet/network_windows.rs`; [ReDimNet](redimnet/README.md) |
| MossFormer | Both independent waveform windows, bounded session lifetime and cancellation pass on the AMD machine. | `crates/models/src/mossformer2/network_windows.rs`; [MossFormer](mossformer/README.md) |
| Whisper | Three of four original numerical checks fail. All four observed language-routing outcomes match. The execution adapter is compiled but artifact admission remains blocked. | `crates/models/src/whisper_lid/network_windows.rs`; [Whisper](whisper/README.md) |
| CTC timing | Original numerical check fails. One supplemental replay preserves 130 character frames and 23 word envelopes; Windows provenance and calibration remain unresolved. The compiled adapter reports its actual Windows graph identity, which core rejects today. | `crates/models/src/ctc/network_windows.rs`; [CTC](ctc/README.md) |

`crates/models/tests/windows_residency.rs` exercises all five implemented
networks together, including Cohere continuation across maximum-window
separation and denial of model-file writes while loaded. It does not include
Whisper, CTC, capture or the host. Its memory observation is not seven-model
headroom or sustained meeting throughput.

`crates/host/tests/windows_capture.rs` exercises actual WASAPI capture through
the host. `echo_quality.rs` retains a frozen simulated-room diagnostic. Both
Windows and Mac initially failed its far-only 6 dB minimum (5.087/5.090 dB).
The shared clock had replaced a validated correlation peak with a different,
earlier lobe, inventing drift and shifting native alignment. Keeping delay and
confidence on the validated peak, with 16 ms causal headroom at the native AEC
boundary, passes the unchanged test: 46.145/46.111 dB far-only reduction and
11.053/11.052 dB double-talk target-error reduction. Local-only bypass and system
audio remain bit-exact. All four runs retain the same recipe, input hashes and
native libraries. Double-talk proves aggregate error improvement, not exact
near-voice amplitude or active-DSP alignment; physical-room qualification remains
required. The original failed receipts are retained beside the corrected runs.

Runtime pins and immutable admission live in
`crates/models/src/windows_runtime.rs`; download/catalog activation remains a
separate host boundary. `scripts/build_windows.ps1` builds an unsigned candidate.
Signing, standard-user install/update, model distribution and the assembled
import/live/edit/save/reopen interactions remain required before release.

## Original Silero feasibility probe

This is a disposable experiment for issue #20. It exports the exact pinned
Silero-v6 weights and runs them through native ONNX Runtime CPU and DirectML.
It does not implement a production Windows backend, change `Models`, alter
installation manifests, or qualify the complete application. The C++ file is
an experiment runner; all model operations are in the exported ONNX graph.

The source inspection used baseline `c038fd81a905e00bbb3d41e461950a5e5ecd055b`.
The seven-model table below describes those artifacts, not universal backend
compatibility. No large model was downloaded for this experiment.

## Reproduce

Use a disposable Python 3.12 environment **only to prepare the experiment**:

```sh
python3.12 -m venv /private/scratch/silero-export-env
/private/scratch/silero-export-env/bin/pip install onnx==1.19.1 numpy==2.2.6
/private/scratch/silero-export-env/bin/python experiments/windows/export_silero.py \
  /private/scratch/model.safetensors /private/scratch/artifacts
```

Obtain `model.safetensors` from
[`mlx-community/silero-vad-v6@2ebf4a5`](https://huggingface.co/mlx-community/silero-vad-v6/resolve/2ebf4a5e10726a2e78ddd4d70eedfb6f1c33eb06/model.safetensors).
The exporter rejects a size or SHA-256 mismatch with the production pin. It
checks the complete parameter inventory and shapes, runs the ONNX checker, and
prepares both raw and normalized views of the existing 16 kHz synthetic WAV.
It records source, reference, fixture, exported artifact, and tool identities.
The original WAV and independent reference JSON remain unchanged.

Download and extract these official NuGet packages into private `ort` and
`dml` directories, preserving their internal paths:

| Package | Version | Archive SHA-256 |
| --- | --- | --- |
| [Microsoft.ML.OnnxRuntime.DirectML](https://www.nuget.org/packages/Microsoft.ML.OnnxRuntime.DirectML/1.24.4) | 1.24.4 | `57e9f11b73437bef7a309496135d4c1f96b1a8e9ddba60013fa27bfc1d788681` |
| [Microsoft.AI.DirectML](https://www.nuget.org/packages/Microsoft.AI.DirectML/1.15.4) | 1.15.4 | `4e7cb7ddce8cf837a7a75dc029209b520ca0101470fcdf275c1f49736a3615b9` |

ONNX Runtime's package records source commit
`2d924974ef147392ced8409d36bd6d2e7fcc8a74` and depends on DirectML 1.15.4.
The runner script additionally checks the exact three runtime DLL hashes. Only
the x64 native runtime files, `build/native/include`, and DirectML's
`include` and `bin/x64-win/DirectML.dll` are needed from the packages.

On Windows, load the **x64 MSVC Native Tools environment** with the Windows SDK,
then invoke PowerShell from that environment:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\experiments\windows\run_silero_probe.ps1 `
  -Artifacts C:\private\silero-probe\artifacts `
  -OrtRoot C:\private\silero-probe\ort `
  -DirectMLRoot C:\private\silero-probe\dml -Adapter 0
```

The adapter is a DXGI enumeration index. The executable records the selected
adapter's name and IDs. The script writes only inside the supplied artifact
directory and requires no Python, .NET, or model download on the Windows
runtime path.

## Acceptance and evidence boundaries

The native runner first executes CPU, then DirectML. Each session runs all
285 frames through two independent recurrent states (raw and peak normalized),
resets both states, and repeats the entire clip. It checks:

- Finite probabilities in `[0, 1]` and finite hidden/cell outputs.
- Maximum probability error of 0.01 against the independent MLX reference and
  **zero changed hysteresis decisions**. The repository already documents MLX
  backend probability drift; this numeric bound does not qualify new acoustics.
- Repeatability after reset within `1e-5` for probability and recurrent state.
- DirectML versus CPU probability error at most `0.001`, and hidden/cell absolute
  error at most `0.01` at every step.
- Executed `DmlExecutionProvider` kernel events in the saved profile. The script
  reports every observed provider and its operations, including any CPU nodes.
  Provider registration alone does not pass.

`run.log`, `placement.json`, the original ORT profiles,
`runtime-provenance.json`, and `provenance.json` are the evidence. Load time and
both full-clip run times include profiling, host-side state handling, and
CPU/GPU transfers. They are component costs on this fixture, with one session
per provider; they do not establish sustained application performance, energy,
or a preferred device policy. The probe intentionally permits mixed placement
and reports it; it rejects a run with no executed DirectML work.

The exporter has been checked with ONNX 1.19.1 and NumPy 2.2.6. Before any Windows
claim, the exported graph ran locally with ONNX Runtime 1.24.4 CPU on macOS:
570 recurrent steps, maximum probability error `0.0033502280712127686` versus
the independent reference, and zero changed speech decisions.

## Recorded AMD Windows result

The owning Windows lane compiled and ran the probe on 2026-10-14 using Visual
Studio 2026 Developer Tools 18.10.4, Windows 11 x64 build 26300, Ryzen 7 5700X3D,
and DXGI adapter 0: **AMD Radeon RX 7900 XT** (vendor 4098, device 29772).
The owning lane's machine inventory records driver `32.0.31041.1004` and 64 GB
RAM. Runtime DLL hashes match the pins above, and the returned provenance
matches the locally checked graph and fixture.

Both CPU and DirectML passed all 285 frames, both views and the reset repeat:
1,140 model calls per provider. Maximum probability drift from independent MLX
was `0.00335038` on CPU and `0.00335044` on DirectML, with zero changed speech
decisions in every pass. CPU/DirectML maximum disagreement was `1.07288e-6` for
probability and `6.48499e-5` for recurrent state; reset disagreement was zero.

The raw DirectML profile contains **1,140 executed `DmlFusedNode_0_0` kernels and
zero CPU kernel events**. That fused node consumes audio plus both recurrent
states and returns probability plus both updated states. This establishes
execution of the complete exported model graph on DirectML for these inputs,
rather than merely provider registration or a trivial GPU operation. Host
framing, state copies, and runtime orchestration still run on CPU.

| Recorded component cost | CPU | DirectML |
| --- | ---: | ---: |
| Session load | 9.306 ms | 223.051 ms |
| First full clip, both views | 144.010 ms | 140.594 ms |
| Repeat after recurrent reset | 147.153 ms | 113.559 ms |

These single-session profiled costs include transfers and do not establish a
production device policy or sustained speedup. The result qualifies this
fixed-shape Silero export on this AMD machine; the other six models, Windows
ML distribution, application lifecycle, capture, cancellation and save/reopen
path remain unqualified here.

The raw profile hashes are CPU
`ece0b08d57e4bf1147046703035ba1a7b9ef840ac343e20ef7301d6324dbec6e`
and DirectML
`580c66f753265f20a519f58987bd71ff4b4f7f4d13ad9fb5e3fbe53bab269297`.
The owning lane retained `run.log`, profiles, placement and provenance under
its private `silero-evidence` directory; they are not committed model or audio
artifacts.

## Seven pinned models

| Model and current artifact | Windows inference work still needed | Decisive next evidence |
| --- | --- | --- |
| Silero-v6: Float32 safetensors, 1.24 MB; revision `2ebf4a5e10726a2e78ddd4d70eedfb6f1c33eb06` | This exact graph has an experimental fixed-shape ONNX export. Production still needs native session ownership, framing, cancellation, setup/readiness, and pinned derived artifact handling. | AMD numerical/reference/reset and executed-placement checks passed above. Next preserve production session packet boundaries and lifecycle behavior. |
| Nemotron-3 diarization: BF16 MLX safetensors, 198.6 MB; revision `59ed2dbfc1346dcea9d423c71306a3a2499c568f` | No ONNX graph in the installed artifact. Export or implement the existing encoder with explicit audio features, masks, dtype conversion policy, and streaming cache state. Keep current diarization identities and 10 ms coordinates. | Existing independent probability/turn reference, then chunked continuation/cache behavior and observed placement for actual shapes. BF16 or attention compatibility has not been established. |
| Cohere Transcribe: MLX 4-bit safetensors, 1.506 GB; revision `064e51eab6db47066cbeaa85e2894b5691bd8d12` | No portable graph. The current layers use packed Uint32 weights, group size 64, and BF16 scales/biases. An ONNX route needs a qualified equivalent representation and decoder/KV-cache export; weight dequantization changes memory requirements. | Same tokens/text against the independent reference, mixed-language decoder behavior, cache continuation, peak memory, and per-operator placement. A generic ONNX loader cannot consume this artifact directly. |
| Whisper tiny language probe: Float16 MLX safetensors, 74.4 MB; revision `77fa3f52b482ec80d086df55f893065a9baab172` | Export the existing log-mel frontend, encoder and language scoring step. Speakerdesk needs language probabilities rather than an entire new ASR decoder. | Existing 99-language probability references, including short and silent inputs; repeat probes and placement. |
| Omnilingual CTC timing: pinned INT8 ONNX, 365.4 MB; revision `6fc542a3b0661c8278cca1230c34deb989f31202` | This is already a portable graph format, but its quantized operators and actual input/output metadata still need execution qualification. The current macOS path removes dynamic activation quantization and dequantizes cached UINT8 weights to Float32; running the source ONNX may produce different scores. | Run the exact graph, inspect unsupported/CPU operators, compare existing timing envelopes and token-column metadata; do not equate successful loading with compatible scores or GPU coverage. No download/run was done here. |
| ReDimNet2-B6 voice embeddings: compiled Core ML bundle, 30.0 MB; revision `112dd8f4f836abdf8a420e66a5e2885cf8ec64ab` | The shipped `.mlmodelc` is platform-specific. Recover the pinned source checkpoint (`e0a7d340…`) and export its graph with the exact waveform/frontend rules; do not treat compiled Core ML weights as an ONNX checkpoint. | Existing independent embedding vectors and held-out calibrated recognition decisions, then profile migration/compatibility and execution placement. Human voice accuracy remains a separate qualification. |
| MossFormer2 separation: source PyTorch checkpoint, 670.4 MB, prepared safetensors 223.1 MB; revision `407cb030cd66340918ebb6c8cc63b18f8592cdbe` | No ONNX graph in the installation pins. Export the pinned raw two-source graph, preserving source order, full-window state, and existing admission/cancellation behavior. Avoid silently adding upstream RMS/window wrappers that current native inference omits. | Existing independent separated-audio reference on fixed windows, channel correspondence, overlap quality, memory, cancellation and placement for real window sizes. |

These are source-based work estimates. Only Silero has been exported and run in
this lane; the table does not claim the other graphs are supported or blocked
by an accelerator.

## Backend direction

[Windows ML](https://learn.microsoft.com/en-us/windows/ai/new-windows-ml/overview)
is Windows' maintained ONNX Runtime distribution and preserves the ONNX API.
Its current docs identify included CPU and legacy DirectML providers, plus
on-demand hardware providers. The
[provider requirements](https://learn.microsoft.com/en-us/windows/ai/new-windows-ml/supported-execution-providers)
list AMD MIGraphX for RDNA 3 or newer and driver 25.10.13.09 or newer, with
GenAI currently unsupported. The test PC's Windows driver build string alone
does not establish compatibility with that version notation or with a model.
This experiment uses pinned ORT DirectML to bound the first AMD execution
question; it does not choose the production Windows ML distribution or policy.

The maintained Rust [`ort` binding](https://docs.rs/ort/latest/ort/) currently
documents release `2.0.0-rc.14` and dynamic runtime loading. Production should
evaluate that binding against the chosen Windows ML/ORT API version, rather
than promoting this disposable C++ executable into the application. No
production dependency is introduced here.

DirectML's official
[configuration requirements](https://onnxruntime.ai/docs/execution-providers/DirectML-ExecutionProvider.html#configuration-options)
require sequential execution and disabled memory patterns. The probe applies
both. Per-node evidence comes from ORT's supported
[profiling interface](https://onnxruntime.ai/docs/performance/tune-performance/profiling-tools.html).
