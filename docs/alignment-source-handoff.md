> Historical source-only handoff. The adapter is now wired into the app and approved cached acoustic tests have run. The former protobuf6 proposal is superseded by the existing protobuf7.36.2 runtime. Read [current integration and bounded English scope](alignment-integration-status.md) for operative behavior and remaining signed-package gates.

# Cohere alignment source prototype

This branch is based on `fdc27e67760093a97feef6ea3c2e7012b512813f`. It adds an isolated supplied-text alignment adapter and proposal files. It does not wire the adapter into a running app, load an acoustic model, install dependencies, build native code, capture audio or replace Cohere. Source ownership was coordinated with integrator `01a10a26-9958-70a6-b686-61dd425cf0cf`; its canonical contract was inspected at `1599e815ba9c56239d316fb9b48e0c1de0dccd04`.

## Shared interface

`AlignmentRequest.from_utterance(row)` binds utterance ID, `machine_revision`, `audio_revision`, unchanged Cohere text, and exact original 16 kHz `text_audio_anchor`. Text decoded against an older audio endpoint cannot attach to a growing utterance. `align_ctc_scores` accepts precomputed normalized scores; there is no acoustic-model loader or alternate transcript argument.

`attachment_for(result, request, current_row)` rejects a stale request, different text/hash/audio, incomplete display-unit coverage, reordered ranges and unresolved positions. A successful attachment supplies `id`, `machine_revision`, `audio_revision`, `text_sha256` and nonempty `words`, matching the integrator's `UtteranceBook.attach_alignment(id, revision, text_sha256, words, audio_revision=...)` contract. Per-word provenance survives that existing attachment API. The app must still apply its current-row compare-and-swap protection after asynchronous work.

The source-only flow is:

```python
request = AlignmentRequest.from_utterance(utterance_snapshot)
# Future approved acoustic stage supplies normalized CTC scores and vocabulary.
result = align_ctc_scores(
    request.raw_text, scores, vocabulary,
    clock=validated_frame_clock, policy=validated_score_policy,
    audio_start_sample=request.start_sample,
    audio_num_samples=request.end_sample-request.start_sample,
)
attachment = attachment_for(result, request, current_utterance)
# Integrator owns scheduling, persistence and the actual attachment call.
```

No production frame clock or score policy has been validated. Passing neither produces unresolved text with null sample spans. Tests use explicitly invented grid/policy values, not a claimed acoustic calibration.

## Raw text and timing semantics

Raw Cohere text and its SHA-256 stay unchanged. Targets use NFC with an explicit map back to raw Unicode code point ranges; results also carry UTF-16 offsets for the JavaScript editor. Whitespace target runs collapse to a literal space token; outer whitespace has no acoustic target. Punctuation remains in the raw display units and is explicitly excluded from acoustic targets. Digits, decimal text, repetitions and scripts are never romanized, number-normalized or deduplicated. Unknown non-punctuation characters make the alignment unresolved rather than silently disappearing.

Repeated adjacent target tokens require an observed blank separator. Greek epsilon is a normal target, with distinct internal alignment markers. Word-like display units preserve raw whitespace runs; Chinese/Japanese without spaces remain coarse display units, while character evidence stays available. This is not a new linguistic word tokenizer.

Sample spans are envelopes of observed CTC emission cells. They are not phonetic onset/offset measurements, sentence boundaries, speech-admission proof or word correctness certificates. The frontend origin and acceptance policy require measured evidence for the exact model digest. No interpolation, extrapolation or duration scaling fabricates a position. Weak, nonfinite, missing, nonmonotonic or out-of-audio anchors produce unresolved/null timing. Empty or failed alignment never establishes silence. Speaker annotation retains unknown ownership for gaps or multiple speakers; it does not replace Nemotron slot reconciliation or ReDimNet identity.

## Independent artifact and license review

Exact proposed conversion: `csukuangfj2/sherpa-onnx-omnilingual-asr-1600-languages-300M-ctc-int8-2025-11-12@6fc542a3b0661c8278cca1230c34deb989f31202`. Model SHA-256 is `e7c4e54ee4c4c47829cc6667d5d00ed8ea7bef1dcfeef0fce766f77752a2726c`. The four required files total **365,453,052 bytes**. Published binary size/hash were checked through repository API/LFS metadata; the binary was not fetched. Vocabulary, model card and license at the exact revision were inspected and hashed. [Publisher metadata](https://huggingface.co/api/models/csukuangfj2/sherpa-onnx-omnilingual-asr-1600-languages-300M-ctc-int8-2025-11-12?blobs=true), [pinned artifact](https://huggingface.co/csukuangfj2/sherpa-onnx-omnilingual-asr-1600-languages-300M-ctc-int8-2025-11-12/tree/6fc542a3b0661c8278cca1230c34deb989f31202).

The conversion's license bytes exactly match the independently inspected Meta license, SHA-256 `a70a523bafbb595c2844104feb313d204904dac91c3d186c05f22a10a71c7a94`. Meta explicitly releases both code and weights under Apache 2.0. Commercial redistribution must preserve the applicable license/notices. [Pinned primary license](https://github.com/facebookresearch/omnilingual-asr/blob/81f51e224ce9e74b02cc2a3eaf21b2d91d743455/LICENSE), [primary model terms](https://github.com/facebookresearch/omnilingual-asr/blob/81f51e224ce9e74b02cc2a3eaf21b2d91d743455/README.md).

Maintained sherpa-onnx sources name this artifact, export a shared 9,812-token CTC vocabulary and waveform-to-logit graph, and copy Meta's license. Quantization uses dynamic QUInt8 MatMul weights. This supports the declared lineage, but does not independently attest or reproduce the binary: conversion dependencies are incompletely pinned and the inspected recipe commit follows its publication timestamp. That distinction remains part of artifact approval. [Pinned exporter](https://github.com/k2-fsa/sherpa-onnx/blob/46d24f181fe3ad55e2b1ac8561e4cf01ea11d2c8/scripts/omnilingual-asr/export-onnx.py), [publishing workflow](https://github.com/k2-fsa/sherpa-onnx/blob/46d24f181fe3ad55e2b1ac8561e4cf01ea11d2c8/.github/workflows/export-omnilingual-asr-to-onnx.yaml).

The exact PyPI `ctc-segmentation` 1.7.4 source archive was inspected in memory and SHA-256 verified; its included license is Apache 2.0. The adapter uses explicit token sequences, validates returned state/frame correspondence and preloads only a matching compiled extension. It refuses a missing native extension before package import can trigger upstream's runtime Cython compilation fallback. Native code and packaging have not been exercised. [Official registry source metadata](https://pypi.org/pypi/ctc-segmentation/1.7.4/json).

The 14-language mapping in the proposal is **metadata coverage only**. Shared Latin/French/Arabic vocabulary is verified; accuracy for mixed English/French/Arabic is untested. Mandarin/Standard Arabic metadata must not be generalized to every dialect.

## Dependency and resource proposal

`alignment-runtime-proposal.json` pins official PyPI artifacts for ONNX Runtime 1.30.0, flatbuffers 25.12.19, protobuf 6.33.5, Cython 3.1.4 and CTC segmentation 1.7.4, with hashes and byte counts. Reuse bundled NumPy/packaging/setuptools. ONNX Runtime is MIT, flatbuffers and Cython are Apache 2.0, protobuf is BSD 3-Clause. Cython is build-only; the finished app must not compile on the user's machine. No Torch, Transformers, fairseq2, alternate ASR or per-language model fleet is proposed. Native ABI, bundled notices, signature and frozen-runtime compatibility remain pending. [Runtime distribution](https://pypi.org/project/onnxruntime/1.30.0/), [runtime license](https://github.com/microsoft/onnxruntime/blob/v1.30.0/LICENSE), [flatbuffers license](https://github.com/google/flatbuffers/blob/v25.12.19/LICENSE), [protobuf license](https://github.com/protocolbuffers/protobuf/blob/v33.5/LICENSE), [Cython license](https://github.com/cython/cython/blob/3.1.4/LICENSE.txt).

Proposed combined download is **390,484,915 bytes**, including 25,031,863 bytes of runtime/build artifacts. Conservative additional disk reserve is **1,133,425,327 bytes** above the fresh parent floor and other reservations. It includes staging plus a retained model copy, expanded runtime, compilation scratch, fixtures and 64 MiB outputs. This is a proposal, not an admitted reservation.

Native compilation proposes a **180-second / 1 GiB RSS** ceiling. The separate alignment-only window proposes **900 seconds / 3 GiB RSS**, one CPU thread, at most 90 seconds per case, 23 approved fixtures totaling at most 254 audio seconds, each bounded by the app's 24.5-second maximum. The 2 GiB RSS planning estimate is unmeasured; the 3 GiB ceiling requires a supervisor. Cohere snapshots are held fixed and its model is not loaded in that window. Aggregate resident app memory with Cohere/Nemotron/ReDimNet needs later separate measurement.

`alignment-test-manifest.json` has no ready audio paths and cannot authorize execution. Exact-artifact approval, fresh coordination and approved public/controlled fixture inputs are required. Measure frame origin, word/character coverage, unresolved behavior, mixed-script/repetition/number cases and independently labeled boundary errors before declaring a production policy calibrated. No acoustic pass or 14-language accuracy claim has been made.

## Source verification and integration boundary

Eleven pure-data unit tests passed in 0.002 seconds using the existing Python runtime. They exercise source contracts with invented anchors, not acoustic accuracy or optional runtime compatibility. A disposable-source negative control removing the repeated-token blank guard failed the intended assertion. Proposal JSON totals/case bounds were checked; staged diff whitespace validation is recorded in the source check receipt. No capture, private audio, weights, model inference, native build, application UI, release or website changes occurred.

The integrator can cherry-pick the isolated source commit without conflict with its owned files. Runtime activation, model setup/download plumbing, refinement scheduling, canonical storage/export schema and reading projection remain integrator work after the approval gates above. Retain unresolved candidates and existing text; do not retire numeric/prefix ownership or speaker reconciliation merely because this source adapter exists.
