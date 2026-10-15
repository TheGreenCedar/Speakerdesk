# ReDimNet2 B6 Windows graph qualification

This build-only exporter uses the original
[PalabraAI/redimnet2](https://github.com/PalabraAI/redimnet2/tree/2a8d15f65b1dfb5d73fede2f11ee42bcccca3035)
graph at commit `2a8d15f65b1dfb5d73fede2f11ee42bcccca3035` and the exact
`b6-vb2+vox2_v0-lm.pt` checkpoint. Its SHA-256 is
`e0a7d340a92f798720d1208949aa6a6bd0cddcb0ba7d4cec33596a17a484e6a2`.
`source-pins.json` records the hashes of every upstream Python file and license.
The checkpoint's TensorFlow-style convolutional DFT/mel frontend already uses
portable operations; no spectral graph is rewritten for ONNX.

The exported contract matches the existing prepared Core ML artifact:

- `audio`: Float32 `[1,96000]`, six seconds of mono 16 kHz PCM.
- `embedding`: Float32 `[1,192]`, L2 normalized.
- Waveform normalization, pre-emphasis, spectral/mel/log/feature processing,
  pooling and output head stay FP32. Only the maintained backbone runs FP16.
- Two-to-six-second clips repeat to fill the input; six-to-ten-second clips
  center-crop, using floor division for an odd surplus. The caller retains the
  existing finite-input, duration, silence, clipping, consent and cancellation
  checks. No peak normalization or extra padding is introduced.

The precision wrapper casts at the backbone's input and output while leaving
upstream's complete forward method unchanged. The strict checkpoint load must
consume every state entry. The exporter checks both retained independent MLX
voice vectors against the existing identity-reuse cosine bound, **0.9999**,
before exporting. Independent references are copied unchanged, never regenerated.

Use the existing isolated Python 3.12 build environment, installing the matching
torchaudio dependency as well as the pinned torch version:

```powershell
python -m pip install -r experiments/windows/redimnet/requirements.txt
Invoke-WebRequest 'https://codeload.github.com/PalabraAI/redimnet2/tar.gz/2a8d15f65b1dfb5d73fede2f11ee42bcccca3035' -OutFile source.tar.gz
tar -xzf source.tar.gz
Invoke-WebRequest 'https://github.com/PalabraAI/redimnet2/releases/download/v1.0.0/b6-vb2+vox2_v0-lm.pt' -OutFile b6-vb2+vox2_v0-lm.pt
python experiments/windows/redimnet/test_export.py ./redimnet2-2a8d15f65b1dfb5d73fede2f11ee42bcccca3035
python experiments/windows/redimnet/export.py ./redimnet2-2a8d15f65b1dfb5d73fede2f11ee42bcccca3035 ./b6-vb2+vox2_v0-lm.pt C:/private/redimnet-onnx
```

The output directory must be new. Downloads are build inputs and are verified
before importing source or loading the checkpoint. The checkpoint is about 50 MB;
the exported graph and weights are about 30 MB. The output retains the exact
upstream MIT license, original Python files with their individual copyright and
license headers, source pins, checkpoint identity, package versions and hashes
of every graph, parameter, reference and case file. Python source is retained
for provenance; application inference uses Rust and ONNX.

Run the generic native Rust probe for both voice cases, first CPU and then DML:

```powershell
ort-probe.exe C:/runtime/onnxruntime.dll cpu C:/private/redimnet-onnx/voice-0-case/case.json C:/private/redimnet-voice0-cpu
python experiments/windows/redimnet/compare.py C:/private/redimnet-onnx/voice-0-case C:/private/redimnet-voice0-cpu --provider cpu
ort-probe.exe C:/runtime/onnxruntime.dll dml C:/private/redimnet-onnx/voice-0-case/case.json C:/private/redimnet-voice0-dml
python experiments/windows/redimnet/compare.py C:/private/redimnet-onnx/voice-0-case C:/private/redimnet-voice0-dml --provider dml
```

Repeat for `voice-1-case`. Each case runs three times in one resident native
session. The shared `verify_probe.py` requires the pinned Windows ML 2.4.89
runtime, matching case/graph/external-weight/input hashes, the explicitly
requested provider, and placement read from the raw execution profile.
The model uses ONNX opset 17 standard operators and an external parameter file;
each case's `model_files` binds that parameter file by hash.

`compare.py` requires finite 192-dimensional normalized outputs, cosine at least
0.9999 against the independent legacy vector, the same bound across repeated
calls, and every requested repetition. The output is normalized again in Float64
for comparison, matching the existing profile normalization boundary. This does
not relax the identity-reuse threshold or authorize a new saved-model identity.

The small executable check covers duration and nonfinite waveform boundaries,
partial repetition and odd center crops, and the maintained graph's complete
mixed-precision export on ordinary and quiet input. On Mac ORT CPU, its ONNX/eager
cosines were greater than 0.999999998. The actual pinned checkpoint's two eager
cosines against retained MLX vectors were 0.99999859 and 0.99999831. These are
conversion checks on synthetic voices. Native DirectML with ORT graph
optimizations disabled passes the independent vectors at cosine `0.9999842` and
`0.9999855` across three repetitions. The default optimizer changes those results
substantially, so the native loader retains this model-specific exception.
The Rust boundary also passes repeat, waveform-policy and cancellation checks.
Windows has a distinct model identity; existing profiles are preserved but do
not participate under that identity. Ordinary saved-voice calibration and broader
profile compatibility remain separate gates; no threshold or fixture is fitted
by these conversion checks.

## Frozen natural separated-query qualification

The native Windows runner and the existing Rust `voice::separation::propose`
decision path pass the retained public LibriSpeech pilot with the prospective
policy frozen before Windows measurements. All reference centroids and query
vectors come from the Windows model space: `PalabraAI/redimnet2`, revision
`2a8d15f65b1dfb5d73fede2f11ee42bcccca3035:b6-vb2+vox2_v0-lm:sd-pcm-v1:windowsml-2.4.89-directml-v1:ort-opt-disabled`,
artifact SHA-256 `89c532e48539bebddc168e773cdcbf014b1941043c50d0a4ff7ccd1139efa5f1`.
The policy retains threshold `0.7264125328398843`, margin
`0.10003429964182987`, and two independent reference clips per person.

Development uses six identities, twelve references, six generated queries and
three controls (21 native embedding calls). All six queries match correctly and
all controls abstain. The true holdout adds four disjoint identities, eight
references and four queries (12 calls), scored against all ten identities. Three
holdout queries match correctly; `5639-40744-0034_4507-16021-0058-ch1`
abstains as required by the retained Mac receipt, before Windows scores were
observed. All 13 decisions meet their frozen expectations with no wrong accepted
match. No held-out threshold fitting or profile relabeling is involved.

The staged fixed-crop manifests preserve the original WAV hashes and the exact
Mac query ranges. Generated queries alone pass through the shared `query_pcm`
headroom rule; ordinary references and controls retain their raw PCM. Results
record the prepared Float32 PCM hashes. The scorer checks exact manifest/result
joins, crop and WAV hashes, model identity, and two distinct reference WAVs per
person before using the core centroid and paired-channel decision rules.

| Receipt or input | SHA-256 |
| --- | --- |
| `prospective-policy-fixed-crops.json` | `8a146574442c1040ad9717c6ff9018fca10952d5d6366ddefda90293422b4dd1` |
| `development-fixed-crops.json` | `bc0fb551da2d4f363dcacf4c603ff5ac3132ab7d2b31250e58d8383061bed16e` |
| `holdout-fixed-crops.json` | `5196dcf60928753215d9e69bba04ce040ea5b76574608219ccce29d096823870` |
| Source development `prepared-query-cases.json` | `2394e4299200f637477a1b652fb2adc7c40a414b1976a15399702df8ea33b6ed` |
| Source holdout `labelled-query-cases.json` | `36af866bd13d41fd21c9b6bfeb2257967b52a918f751e57c0d2a3e8cd192123c` |
| Mac development crop receipt `prepared-query-vectors-2.json` | `9812f0e0924e6d79498ac02c7b01ec5cfa21209a19fbcbcb13bbee64139e5c65` |
| Mac holdout crop receipt `query-vectors.json` | `9604bd9cadd8c171edaf7ec2308844142a1f1fbbf55568b0639e21f3593b259b` |
| Mac expected-abstention receipt `holdout/scores.json` | `9b1d2ae2f0c26724173408051e8fc5631c940b248a389e6a4161bbb68223d67c` |
| `development-windows-v1.json` | `9cafba251f4a64a1e68c605cbfb9871dc2641e66f862f5a8e5d4939f41e47701` |
| `holdout-windows-v1.json` | `36e18883356947c4be34cfde74097f3a2fe786324dd6dedfb5c55f963b7bbbb7` |
| `decisions-windows-v1.json` | `dd7cc2bc7e43906725abf12f517167a511d2b21953afaf0c5986f0624d074bb9` |

The source policy and crop receipt pins are also retained inside the prospective
policy. The native tests require the staged manifests and their `audio/` files;
every output must be fresh. Run from the repository root, setting model and
runtime directories to the admitted artifacts:

```powershell
$env:SPEAKERDESK_TEST_VOICE = 'C:/private/redimnet-onnx'
$env:SPEAKERDESK_TEST_WINDOWS_RUNTIME = 'C:/private/windows-runtime'
$env:SPEAKERDESK_VOICE_QUALIFICATION = 'C:/private/voice-natural-v1'
foreach ($split in @('development', 'holdout')) {
  $env:SPEAKERDESK_VOICE_CASES = "$env:SPEAKERDESK_VOICE_QUALIFICATION/$split-fixed-crops.json"
  $env:SPEAKERDESK_VOICE_RESULT = "$env:SPEAKERDESK_VOICE_QUALIFICATION/$split-windows-v1.json"
  cargo test --release --locked -p speakerdesk-models --test redimnet_windows public_crop_voice_embeddings -- --ignored --exact --nocapture
  if ($LASTEXITCODE -ne 0) { throw "Native embeddings failed: $split" }
}
cargo test --release --locked -p speakerdesk-models --test redimnet_windows frozen_natural_separated_voice_decisions -- --ignored --exact --nocapture
```

This qualifies prepared Mac separated-query PCM with Windows native embeddings
and Windows references. It does not establish Windows separation, speech-range
selection or complete product query preparation, ordinary saved-voice
calibration, population false-accept rates, or compatibility with old saved
vectors. Production policy activation remains a separate integration gate.

## Prospective ordinary saved-voice pilot

`prepare_ordinary_voice.py` freezes a separate natural-reading pilot before any
Windows embedding run. It reuses the six development speakers' two independent
clean references, then takes four additional original utterances per speaker in
dataset row order, excluding every prior reference/query and clips under two
seconds. Clips over ten seconds use the first six seconds. The retained row
range supplies all selections except speaker 4077's fourth unused utterance:
an adjacent source receipt confirms `4077-13754-0006` at row 1247. Source URLs
retain LibriSpeech revision `71cacbfb7e2354c4226d01e70d77d5fca3d04ba1`.
Acquisition used 3,928,487 bytes including metadata, within the 20 MiB cap.

The frozen `ordinary-cases.json` SHA-256 is
`85df77b04cfb1330a5859c70cd24d27a86466beddfd74b0d28fd97df8d57f635`;
`ordinary-policy.json` is
`0c21f21805f2988297bfd128abb2935767308c875dc60442346e26455be9239d`.
The manifest pins all 44 WAVs, crops, source row receipts and 16 query pairs.
Twelve known query pairs must match correctly against six Windows profiles.
The four disjoint holdout speakers' existing reference pairs become unknown
queries, with no holdout enrollment; all four must abstain. Every clip must
produce a clean embedding, including unknowns. The policy freezes the same
threshold `0.7264125328398843`, margin `0.10003429964182987` and minimum two
query clips. Results cannot trigger fitting, reselection or a relaxed gate.

With the admitted model/runtime environment variables above, run:

```powershell
$env:SPEAKERDESK_ORDINARY_VOICE_QUALIFICATION = 'C:/private/voice-ordinary-v1'
$env:SPEAKERDESK_VOICE_CASES = "$env:SPEAKERDESK_ORDINARY_VOICE_QUALIFICATION/ordinary-cases.json"
$env:SPEAKERDESK_VOICE_RESULT = "$env:SPEAKERDESK_ORDINARY_VOICE_QUALIFICATION/ordinary-windows-v1.json"
cargo test --release --locked -p speakerdesk-models --test redimnet_windows public_crop_voice_embeddings -- --ignored --exact --nocapture
if ($LASTEXITCODE -ne 0) { throw 'Ordinary native embeddings failed' }
cargo test --release --locked -p speakerdesk-models --test redimnet_windows frozen_natural_ordinary_voice_decisions -- --ignored --exact --nocapture
```

The scorer uses core `make_profile`, `rank_matches` and `propose_match`, checks
the frozen manifest/model/input hashes and exact native result join, and writes
a fresh `ordinary-decisions-windows-v1.json` before checking acceptance. Ordinary
PCM receives no generated-query headroom adjustment.

The native Windows pilot **fails its prospective acceptance gate**: all 44
inputs produce clean embeddings, eleven known pairs match correctly and all
four unknown pairs reject, but `known-5142-1` abstains. Its two queries are
`5142-33396-0066` and `5142-33396-0067`; the correct reference has mean cosine
`0.7458300803983144` and lowest cosine `0.7168574570256199`, below the frozen
threshold `0.7264125328398843`. The source selection, thresholds and strict test
remain unchanged. The embedding receipt SHA-256 is
`d5436c4038240ef1f9ac459fdac4f8ab9e82927f2a1177e9fd3fa7025a3f4cb6`;
the failed decision receipt is
`f76225f74442dcb38836c16600c39a62775010aa07b15f4328947f6d91128c75`.
This small pilot does not estimate population accuracy or establish old profile
compatibility. Ordinary Windows policy activation remains blocked.

The completed Mac comparison uses the same frozen 44 inputs and sixteen ordinary
trials through the existing Core ML `CPU_AND_GPU` runner and the shared actual
core scorer. It keeps the truthful Mac identity and Mac reference centroids;
the retained prospective policy still identifies the original Windows gate.
All 44 prepared PCM hashes, WAV hashes, crop bounds, headroom flags and clean
results agree exactly. Cross-backend vector cosines range from
`0.9999667948084634` to `0.9999857759915454` (median `0.999980537534273`, mean
`0.9999797459403108`); all exceed the existing `0.9999` identity-reuse bound.
All sixteen decisions agree, including the sole known-speaker abstention:
Mac's lowest cosine for `known-5142-1` is `0.7165316121014398`, also below the
unchanged threshold. This reproduces the current Mac policy's abstention on
these exact inputs; it does not make the strict Windows pilot pass.

Mac embedding receipt SHA-256:
`2f8d220e1fc578c3ef9f6fc6960074993e9821d4a6f4454b866f690ad043369b`.
Mac decision receipt SHA-256:
`72eb3d4d090f02332047e2c85106fad09e1c54faa6f41d08ccc7246503088dba`.
Observed false accepts are `0/4` unknown trials and false rejects are `1/12`
known trials on both backends. The historical Mac synthetic approval in
`speakerdesk/voice_calibration.json` records maximum FAR `0` and maximum FRR
`0.25`; those numeric limits are less strict than this pilot's frozen all-correct
requirement. That approval binds its own Mac identity and synthetic corpus.
The parity result does not transfer approval to the Windows identity or change
this prospective gate. Any different admission basis needs an explicit owner
decision; the failed gate and all receipts remain retained.

To reproduce in a fresh evidence directory, with `SPEAKERDESK_TEST_VOICE` set to
the admitted Mac Core ML model:

```sh
export SPEAKERDESK_ORDINARY_VOICE_QUALIFICATION=/tmp/speakerdesk-windows-20261014/voice-ordinary-v1
export SPEAKERDESK_VOICE_CASES="$SPEAKERDESK_ORDINARY_VOICE_QUALIFICATION/ordinary-cases.json"
export SPEAKERDESK_VOICE_RESULT="$SPEAKERDESK_ORDINARY_VOICE_QUALIFICATION/ordinary-mac-v1.json"
export SPEAKERDESK_TEST_COMPUTE_POLICY=CPU_AND_GPU
cargo test --release --locked -p speakerdesk-models --test redimnet public_crop_voice_embeddings -- --ignored --exact --nocapture
cargo test --release --locked -p speakerdesk-models --test redimnet frozen_natural_ordinary_voice_comparison -- --ignored --exact --nocapture
```

Both Mac result paths must be fresh. Passing the comparison diagnostic does not
activate a policy or relax the ordinary Windows qualification test.
