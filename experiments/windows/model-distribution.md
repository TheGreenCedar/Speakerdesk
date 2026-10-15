# Windows model release notices

The reserved `windows-models-20261015` GitHub release is not published by this
change. `crates/models/src/windows-model-artifacts.json` identifies its seven
converted model inputs, reserved asset prefixes and exact output files. Runtime
qualification, installer admission and publication remain separate gates.

`packaging/THIRD_PARTY_NOTICES.md` records each origin, applicable declared
weight/conversion terms and Speakerdesk's alterations. The existing notice
generator now also fetches hash-pinned full OpenMDW 1.1, Apache 2.0, Silero MIT,
Meta CTC copyright, Transformers and FLASH-pytorch notices. Existing Mac, MLX,
Whisper, Palabra.ai and ClearerVoice source entries remain unchanged.

Generate the complete notice on the prepared Windows packaging machine:

```powershell
python scripts/native_licenses.py --platform windows --runtime-directory C:/private/windows-runtime --destination C:/private/release-notices/THIRD_PARTY_NOTICES.txt
```

Publish that complete text and the origin/change notice alongside the model
assets, and include the generated notice in the application. Do not publish a
source-only preview as the complete app notice: the real generator also verifies
and collects the locked native dependency, capture, Microsoft runtime and echo
notices. Retain the conversion manifests and source/tool versions with the
release record. The final notice and graph inventory should be reviewed before
publication. No evaluation audio, Python environment or source checkpoint is an
asset in this model release.

## Pinned model declarations

These exact source repositories were checked on 2026-10-15. The cards are
license/origin evidence, not model files to download at application runtime.
Their hashes describe the original UTF-8 bytes.

| Model | Pinned declaration | SHA-256 | Declared weight/conversion terms |
| --- | --- | --- | --- |
| Silero | [MLX card](https://huggingface.co/mlx-community/silero-vad-v6/resolve/2ebf4a5e10726a2e78ddd4d70eedfb6f1c33eb06/README.md) | `34990444751e0f8340ea3189ddffb3ef6300c8ba346c458d1d811690d76350ce` | MIT, matching upstream Silero |
| Nemotron | [MLX card](https://huggingface.co/mlx-community/Nemotron-3-Diarization/resolve/59ed2dbfc1346dcea9d423c71306a3a2499c568f/README.md) | `d1b1fbd0f419306e2de500e1e1884aff0a90d8cc0b9422152c18929d903e44e9` | OpenMDW 1.1 |
| Cohere | [Spoke MLX card](https://huggingface.co/spokedotso/cohere-transcribe-03-2026-mlx-4bit/resolve/064e51eab6db47066cbeaa85e2894b5691bd8d12/README.md) | `11b92505082f5707e9601f24141d480960919388d610f427ab2d6f07f5c85e45` | Apache 2.0 |
| Whisper | [MLX card](https://huggingface.co/mlx-community/whisper-tiny-asr-fp16/resolve/77fa3f52b482ec80d086df55f893065a9baab172/README.md) | `d68f867f75885b5fb22073dee9befb9bcb5586a4ba1cdfe39153f0335b2ca9e4` | Conversion declares Apache 2.0; original OpenAI code/weights remain MIT |
| Omnilingual CTC | [Meta notice in the pinned model repository](https://huggingface.co/csukuangfj2/sherpa-onnx-omnilingual-asr-1600-languages-300M-ctc-int8-2025-11-12/resolve/6fc542a3b0661c8278cca1230c34deb989f31202/LICENSE) | `a70a523bafbb595c2844104feb313d204904dac91c3d186c05f22a10a71c7a94` | Apache 2.0; Copyright 2025 Meta Platforms, Inc. and affiliates |
| ReDimNet2 | [Palabra.ai license](https://raw.githubusercontent.com/PalabraAI/redimnet2/2a8d15f65b1dfb5d73fede2f11ee42bcccca3035/LICENSE) | `3d5a4bf9936a7d09dd770588c0d7cb116fc16916a0310d71fb1fff5d64562101` | MIT; source/checkpoint origin also frozen in `redimnet/source-pins.json` |
| MossFormer2 | [Weight card](https://huggingface.co/alibabasglab/MossFormer2_SS_16K/resolve/407cb030cd66340918ebb6c8cc63b18f8592cdbe/README.md) | `f43ca6b52db24028a287e44f38525929ec97bce7c0036af9ef9e836d9c9b90a1` | Apache 2.0; maintained ClearerVoice source is separately attributed |

The upstream [NVIDIA card snapshot](https://huggingface.co/nvidia/Nemotron-3-Diarization/resolve/f667ed73aee57d40cc39428eb768b4fd87a0a29e/README.md)
(`5fe88ca4020d2ee134f806a0a02c52c19855b73047bc2e6f25b9171e1abeebfd`)
also specifies OpenMDW 1.1. The upstream [Cohere card snapshot](https://huggingface.co/CohereLabs/cohere-transcribe-03-2026/resolve/d552dadc2922ba6f6c1f994f8db904d52dcaf942/README.md)
(`28a65565bfebe4fbc39c894d6339d26521ee20a606e83add4e32ff02bd6808fd`)
states Apache 2.0. These snapshots support the terms; they do not invent an
unrecorded original-weight revision for the pinned community conversions.

The reviewed declarations identify no additional redistribution restriction or
separate permission requirement. They require retaining their applicable terms,
copyright/origin notices and, under Apache 2.0, modification notices. OpenMDW 1.1
also requires its agreement and applicable origin notices to travel with the
model materials. This review records the publishers' declarations and source
texts; it does not independently establish rights in their training corpora or
transfer any dataset license to model users. Inspect any additional NOTICE or
origin file added to the final release inputs rather than dropping it silently.
