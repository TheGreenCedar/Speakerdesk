# Local ReDimNet2 integration and next measured step

The source adapter is implemented. Recognition remains unavailable until the pinned local artifact, optional runtime and independently measured calibration are installed and reviewed. No model was downloaded or run for this change. Name-only People and confirmed introduction suggestions remain usable.

## Artifact and provenance

The community [Core ML artifact](https://huggingface.co/aufklarer/ReDimNet2-B6-CoreML/tree/112dd8f4f836abdf8a420e66a5e2885cf8ec64ab) is pinned at `112dd8f4f836abdf8a420e66a5e2885cf8ec64ab`, revision `frontend-fp32-v1`: eight allowlisted files totaling **30,036,479 bytes**. Compiled files total 30,024,704 bytes; weights alone are 29,519,872 bytes. The older model-card 24.7 MiB/fp16 description does not describe this newer artifact. Its config specifies a float32 frontend/head with float16 backbone.

[PalabraAI upstream](https://github.com/PalabraAI/redimnet2/tree/2a8d15f65b1dfb5d73fede2f11ee42bcccca3035) is pinned at `2a8d15f65b1dfb5d73fede2f11ee42bcccca3035`; checkpoint `b6-vb2+vox2_v0-lm.pt` SHA256 is `e0a7d340a92f798720d1208949aa6a6bd0cddcb0ba7d4cec33596a17a484e6a2`. The artifact and upstream MIT licenses have identical bytes: SHA256 `3d5a4bf9936a7d09dd770588c0d7cb116fc16916a0310d71fb1fff5d64562101`, copyright 2026 Palabra.ai, preserved in `packaging/licenses/redimnet2/LICENSE`.

`speakerdesk/redimnet2_artifact.json` pins every file size/hash. Hashes for large graph/weight files are publisher/Hugging Face metadata, awaiting local download verification. The aggregate file-manifest digest is `cf8cc300e240efaed56cb931bd7715ded41b66bf55af7e9a19d0b17e7fe986c9`. The model identity additionally pins preprocessing revision and compute policy, preventing profile reuse across incompatible conversions or execution policies.

The [Swift reference](https://github.com/soniqo/speech-swift/blob/1f54e56cf137078ed681a03e0955e777f7314610/Sources/SpeechVAD/ReDimNet2Speaker.swift) was inspected for input/output and crop/repeat behavior. Its SDK is not used or copied. A complete conversion-script audit and independent PyTorch/Core ML parity test remain outstanding. Publisher numeric/performance figures are not independently verified results.

## Runtime behavior

The independently written Python adapter uses Apple's [CompiledMLModel API](https://apple.github.io/coremltools/source/coremltools.models.html), lazy-loaded only for an explicit remembered-voice or suggestion action. It consumes saved mono PCM16 16 kHz audio, reads only selected 2–10 second clips, repeats shorter clips or center-crops to six seconds without gain normalization, and validates finite 192-dimensional output. It screens silence/clipping; it does **not** classify acoustic noise, overlap, or speaker changes. Select clean finalized single-speaker passages and include difficult negatives in deployment evaluation.

Recognition requires Apple Silicon, macOS 15+, Core ML Tools 9.0 and a reviewed model-compatible calibration. Host metadata here is arm64/macOS 27; native compatibility has not been exercised. Voice work waits until active recording/transcription ends. Names remain editable during meetings. No Nemotron slot/cache becomes cross-meeting identity; explicit enrollment/confirmation and forget semantics from People remain intact.

`requirements-voice.lock.txt` pins seven official PyPI wheels totaling **9,949,582 download bytes**, including the 2,764,608-byte cp313/arm64 Core ML Tools wheel, SHA256 `9f2f858beec7f5d486cd1a59aefb452d59347e236670b67db325795bf692f480`. It reuses the existing pinned NumPy/tqdm/packaging/PyYAML/typing-extensions. mpmath is pinned to 1.3.0 to satisfy SymPy's `<1.4` bound. No wheel has been downloaded/installed; imports, ABI compatibility and installed footprint remain unverified. Optional packaging collection is prepared; no native/frozen-app build or license collection for newly installed packages has run.

## Smallest approval-gated action

After CodeStory releases the resource window, obtain explicit approval to download this community artifact and the pinned optional wheels, install the wheels into a **new isolated target** using the existing Python 3.13 environment (`--no-deps --require-hashes`), and run the extraction smoke below. Reserve at most 250 MB of new disk use and retain 40 GB free; recheck disk before starting. Do not alter the shared existing environment.

1. `python scripts/download_voice_model.py NEW_MODEL_DIRECTORY --approved-community-artifact` downloads only the eight pinned files over HTTPS, bounds/checks every file, stages atomically and never loads them. Existing destinations are refused.
2. Use the existing synthetic `fixtures/conversation.wav` from the canonical fixture folder, whose reference declares installed macOS Samantha/Daniel voices. Create a smoke manifest with two speaker groups, two separate reviewed passages each: Samantha `[0,4.0910625]` and `[9.7034375,14.124875]`; Daniel `[4.5910625,9.2034375]` and `[14.624875,19.75425]`. This uses synthetic audio only and does not generate new audio or enroll People.
3. Run `python scripts/measure_voice.py --model-dir NEW_MODEL_DIRECTORY --fixtures SMOKE_MANIFEST --output NEW_REPORT --smoke --compute-units ALL`. Four clip extractions, two warmups and two repeats: **eight predictions total**, maximum 60 seconds and 1024 MiB RSS. A supervisor samples RSS/disk every 0.5 seconds and kills the worker on observed breach/timeout; the worker also checks peak RSS before writing its report. Native transient allocation can occur between samples. Stop on unexpected behavior. Report raw output norms, load/predict timing, repeated-output cosines, same/different-label diagnostics and peak RSS. Recognition stays unavailable.

The approval is required by the delegated task's explicit community-model download/run restriction. No approval or resource-window release is inferred from the passage of time.

## Calibration before enabling recognition

The two-voice recording proves extraction only. Collect separately recorded, labelled synthetic/consented/licensed pilot meetings, with several recurring speakers, unknown/similar voices, varied microphones, levels and difficult passages. Each reference/query uses at least two distinct clean clips. Fixture manifests declare `schema_version`, `dataset_id`, `fixture_source`, `fixture_root` and `groups`; each group declares unique `id`, `recording_id`, `speaker_id` (null for unknown), `split` (`enrollment`, `calibration`, `held_out`) and `file`, plus `clips` of `{start,end,clean:true}`. Bound files to 64 MiB each/256 MiB total and a run to 200 clips; split IDs, paths and identical file contents cannot overlap. Curators must also catch re-encoded/cropped derivatives and speaker-label mistakes.

Run measurement with explicit reviewed `--maximum-far` and `--maximum-frr`, without `--smoke`. No guessed production threshold is provided. The harness derives threshold/margin candidates from calibration scores using the production scorer, rejects wrong-known identities, freezes the policy and evaluates separate held-out queries once. Failed held-out results remain failed; do not retune against them. Rates are observed pilot rates, not broad accuracy guarantees or confidence bounds. Small trial counts need more coverage before deployment.

Reports always set `approved_for_recognition:false`. After reviewing measured results, trial counts, independent split provenance and deployment coverage, create a local reviewed configuration with approval true and point `SPEAKERDESK_VOICE_CONFIG` at it. Startup checks model/hash/runtime/calibration and held-out verdict without making a prediction. Every actual identity suggestion still needs confirmation, and unconfirmed matches never update profiles. A smoke report lacks calibration and cannot enable the app.

## Verification limits

CPU tests exercise PCM preprocessing, malformed output, checksum/symlink/extra-file rejection, disabled configuration fallback, active-meeting isolation, measured-policy behavior, unknown fallback, split leakage and installer rollback. Mocked predictor tests validate the adapter boundary only. Actual compiled-model loading/output/latency, cross-meeting identity quality, error calibration, dependency imports and packaged desktop behavior remain unverified until the approved measured phases above.
