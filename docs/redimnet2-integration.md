# Local ReDimNet2 integration

B6 is part of standard local model setup, and Recognize saved voices defaults on. End users use Settings; the desktop bundles its pinned Python runtime. The common download action includes transcription, speaker separation and the 30 MB voice model, with progress, interrupted-download resumption, per-file checksums, retained MIT notices, disk checks and recoverable errors. Successful setup activates the compatible shipped policy without restarting; startup restores it from the managed cache. Downloads never enroll a person.

## Artifact and provenance

The community [Core ML artifact](https://huggingface.co/aufklarer/ReDimNet2-B6-CoreML/tree/112dd8f4f836abdf8a420e66a5e2885cf8ec64ab) is pinned at `112dd8f4f836abdf8a420e66a5e2885cf8ec64ab`, revision `frontend-fp32-v1`: eight allowlisted files totaling **30,036,479 bytes**. Compiled files total 30,024,704 bytes; weights alone are 29,519,872 bytes. The older model-card 24.7 MiB/fp16 description is stale; this artifact specifies an FP32 frontend/head and FP16 backbone.

[PalabraAI upstream](https://github.com/PalabraAI/redimnet2/tree/2a8d15f65b1dfb5d73fede2f11ee42bcccca3035) is pinned at `2a8d15f65b1dfb5d73fede2f11ee42bcccca3035`; checkpoint `b6-vb2+vox2_v0-lm.pt` SHA256 is `e0a7d340a92f798720d1208949aa6a6bd0cddcb0ba7d4cec33596a17a484e6a2`. Upstream and artifact MIT license bytes match SHA256 `3d5a4bf9936a7d09dd770588c0d7cb116fc16916a0310d71fb1fff5d64562101`, copyright 2026 Palabra.ai, preserved in `packaging/licenses/redimnet2/LICENSE`.

`speakerdesk/redimnet2_artifact.json` pins every file size/hash. All downloaded bytes were independently verified locally. The aggregate file-manifest digest is `cf8cc300e240efaed56cb931bd7715ded41b66bf55af7e9a19d0b17e7fe986c9`. The model identity also pins preprocessing and compute policy, preventing profiles from crossing incompatible embedding spaces.

The [Swift reference](https://github.com/soniqo/speech-swift/blob/1f54e56cf137078ed681a03e0955e777f7314610/Sources/SpeechVAD/ReDimNet2Speaker.swift) was inspected for input/output and repeat/crop behavior; its SDK is not used or copied. Complete converter-script audit and independent PyTorch/Core ML parity remain outstanding. Publisher performance numbers are not our measured results.

## Runtime and track behavior

The Python adapter uses Apple's [CompiledMLModel API](https://apple.github.io/coremltools/source/coremltools.models.html). It loads lazily at the first actual identity/enrollment check, consumes saved mono PCM16 16 kHz audio, reads selected 2–10 second clips, repeats short clips or center-crops to six seconds without gain normalization, and validates finite 192-dimensional output. It screens silence/saturation, rather than independently classifying acoustic noise or overlapping speakers.

Apple Silicon, macOS 15+ and Core ML Tools 9.0 are required. Native loading/output passed on this arm64/macOS 27 host. Automatic recognition runs in a separate single-worker executor outside the recording lock. A new track is checked once after multiple eligible clips, then matching or unknown is cached for the meeting. There is no per-phrase embedding loop. User corrections and incompatible/changed evidence invalidate pending work. Automatically recognized names do not enroll or update profiles. Explicit enrollment still waits for active recording/transcription to finish.

`requirements-voice.lock.txt` pins seven official PyPI wheels totaling **9,949,582 download bytes**. Core ML Tools 9.0 cp313/arm64 is 2,764,608 bytes, SHA256 `9f2f858beec7f5d486cd1a59aefb452d59347e236670b67db325795bf692f480`. Existing pinned NumPy/tqdm/packaging/PyYAML/typing-extensions are reused; mpmath 1.3.0 satisfies SymPy's `<1.4` bound. The isolated installed runtime occupies 42,564,886 logical bytes. Exact wheel licenses/versions are retained in `packaging/licenses/voice-runtime` and exported notices. Packaging source collects the runtime, native library, policy and notices; no frozen/native build was run in this lane.

## Completed extraction smoke

After explicit approval and a coordinated resource release, eight predictions ran on the existing synthetic Samantha/Daniel fixture. Supervised wall time was 10.145 s, peak RSS 255,016,960 bytes (243.2 MiB), initial load 6,691 ms. Seven warm predictions ranged 119.524–122.443 ms, median 120.045 ms. These timings are slower than the publisher's figure on this host. ALL is the requested compute policy, not evidence of actual device placement.

Raw norms were 0.999999971–1.000000008; repeated-clip cosine was 1.0 twice. Same-labelled pairs were 0.866/0.853, different-labelled pairs 0.272–0.286. Smoke demonstrates extraction only, rather than a calibrated identity policy. Its report remains `approved_for_recognition:false`. The worker exited and resources were released immediately; no user recording or enrollment was involved.

## Independent synthetic pilot

`scripts/generate_voice_fixtures.py` verifies eight already installed compact macOS voices and generates a bounded development fixture outside Git. It downloads no voice asset, captures no user audio and runs no speaker model. The completed fixture has 36 unique recording hashes and 72 distinct scripts: 16 enrollment clips, 24 calibration clips, 32 held-out clips. Samantha, Daniel, Karen and Rishi recur as known speakers; Moira/Tessa are unknown and Alice/Anna appear only as unseen held-out unknowns. Each passage is 4.024–7.723 s, mono PCM16/16 kHz. Audio totals 12,971,276 bytes; the tree is 13,050,110 bytes. Manifest SHA256 is `4a6eb0102d420587fba74fa0047db89f0ec17dcaeb75da79eb1c6144562d44a7`.

The calibration harness scores reference centroids and two clean query clips using the production scorer. It derives candidates and interior score-gap thresholds from calibration only, excludes wrong-known identities, selects a feasible policy with separation from observed boundaries, freezes it and evaluates held-out queries once. Requested pilot limits are observed FAR 0, FRR ≤0.25 and zero wrong-known identities. Absolute pilot-unknown rejection also protects the single-profile case, where a runner-up margin alone is insufficient. A failed held-out result stays failed; it is not a tuning set.

Native pilot execution is pending its separately coordinated resource window. The shipped policy asset remains pending until that result exists. A successful held-out verdict enables the compact policy automatically; no separate calibration-approval ritual is imposed. Synthetic development voices and a small trial count do not establish human-population accuracy or coverage across microphones, noise and rooms.

## Bounded verification and remaining limits

Developer tools are separate from end-user setup. `scripts/download_voice_model.py NEW_DIRECTORY --approved-community-artifact` allows only pinned HTTPS files and verifies bytes before atomic promotion. `scripts/measure_voice.py --model-dir EXISTING_MODEL --fixtures NEW_MANIFEST --output NEW_REPORT --compute-units ALL --maximum-far 0 --maximum-frr .25` supervises native work for at most 60 s/1024 MiB RSS while retaining 40 GB free. RSS/disk are sampled every 0.5 s, the worker checks peak RSS before report creation, and failures preserve an existing report. Transient allocations can occur between samples. Every native run needs an actual coordinated resource release; no inference or heavy build runs concurrently with CodeStory's window.

CPU checks cover waveform contracts, malformed output, exact hashes/symlinks/extra-file rejection, model-version policy, split leakage, held-out failure, downloader resume/corruption/storage/auth/restart, persistent ON/OFF preference, once-per-track behavior and correction/forget races. Existing import/live lifecycle checks are retained. Real-human identity quality, conversion parity, packaged dependencies and real capture/transcription concurrency remain unverified. Nemotron cache is never cross-meeting identity, and no unconfirmed match changes a saved voice.
