# Speakerdesk 0.1.0 — third-party notices

Speakerdesk uses local NVIDIA speaker diarization and Cohere transcription models, through independently maintained Apple MLX implementations. It is not affiliated with NVIDIA, Cohere, Apple, Tauri or the MLX conversion authors.

- NVIDIA Nemotron 3 Diarization: OpenMDW License Agreement 1.1. Converted public weights: mlx-community/Nemotron-3-Diarization, pinned 59ed2dbfc1346dcea9d423c71306a3a2499c568f. Upstream origin: https://huggingface.co/nvidia/Nemotron-3-Diarization . License: https://openmdw.ai/license/1-1/ . No model weights are redistributed inside this installer; the first-run setup downloads these public pinned files.
- Cohere Transcribe03-2026: Apache License 2.0, upstream https://huggingface.co/CohereLabs/cohere-transcribe-03-2026 . Compatible 4-bit MLX conversion: spokedotso/cohere-transcribe-03-2026-mlx-4bit, pinned 064e51eab6db47066cbeaa85e2894b5691bd8d12. Community conversion model card is retained.
- MLX Audio 0.5.7: MIT, https://github.com/Blaizzy/mlx-audio , commit feb25a37b07923bae556e59111995071d66afa0d. The packaged vad/models/__init__.py is changed to avoid eager imports of unrelated speech models. NVIDIA model code, strict weight loading and generation remain unchanged.
- MLX Speech 0.5.3: MIT, https://github.com/appautomaton/mlx-speech , commit 4577ad28e6bef2ae2a11b8fa76a6cdb23bb2a9cc. Its Cohere architecture and generation implementation are reused.
- Apple MLX 0.32.2: MIT, https://github.com/ml-explore/mlx . Official macOS 14 arm64 wheels are used explicitly.
- Flask 3.1.3: BSD-3-Clause. Python 3.13.14: PSF license. NumPy, SciPy, SoundFile/libsndfile, Hugging Face Hub, tokenizers and supporting Python dependencies are included under their accompanying terms.
- Tauri 2.12.1 and Rust dependencies: applicable MIT/Apache/BSD and other terms are retained below. Full lockfile versions are in the accompanying source project.
- Lucide icons 0.577: ISC, https://lucide.dev . The app icon and interface icons use this library.

The synthetic fixture was made with macOS built-in speech voices for this project. Its transcript labels the original supplied sample as scripted; real-model acceptance independently transcribes that sample audio.

The following collected license files preserve their upstream copyright and license notices. Package inventories include build-resolved transitive dependencies and may include components that are not linked into the final executable.
