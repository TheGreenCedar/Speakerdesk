# Speakerdesk

Private meeting transcription and speaker labeling for Apple Silicon Macs. Speakerdesk records or imports audio and works out who spoke. It transcribes locally with NVIDIA Nemotron diarization and Cohere Transcribe on MLX, and you can name, edit and export the transcript. Audio never leaves the Mac, and no paid transcription API is involved.

Requires macOS 15 or newer on Apple Silicon. Model setup downloads about 2.2 GB of pinned public checkpoints and prepares about 3.5 GB on disk, plus space for recordings. This repository contains no recordings, model weights, credentials or prebuilt app.

**Download:** <https://rootandruntime.com/downloads/speakerdesk/latest/macos-arm64>

## Interface previews

The 0.7.1 interface with sample names, text and silent audio. [Capture details](docs/screenshots/README.md).

### Meeting

Review speaker names, edit passages and export the transcript.

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="docs/screenshots/meeting-dark-0.7.1.jpg">
  <img alt="Speakerdesk 0.7.1 sample meeting with Priya and Alex, editable passages and playback controls" src="docs/screenshots/meeting-light-0.7.1.jpg">
</picture>

### People

Save names for future meetings.

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="docs/screenshots/people-dark-0.7.1.jpg">
  <img alt="Speakerdesk 0.7.1 People dialog with sample saved names and voice profiles" src="docs/screenshots/people-light-0.7.1.jpg">
</picture>

### Name picker

Choose a saved person or apply a name to this meeting.

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="docs/screenshots/name-picker-dark-0.7.1.jpg">
  <img alt="Speakerdesk 0.7.1 name picker showing sample saved people and voice profile controls" src="docs/screenshots/name-picker-light-0.7.1.jpg">
</picture>

## Development

Install Node.js 24, Rust 1.99.0 (pinned by `rust-toolchain.toml`), Python 3.13 and the Apple command line tools. Python is only used for build and test tooling.

These checks need no models or audio devices:

```sh
cargo test --locked --features speakerdesk-host/fixtures -p speakerdesk-core -p speakerdesk-host -p xtask
python3 -m pip install -r requirements-ci.lock.txt
python3 scripts/check_source.py
python3 -m unittest discover -s tests -v
node --test tests/*.test.cjs
```

On macOS, run `bash scripts/fetch_mlx_runtime.sh` once, then add `-p speakerdesk-models -p speakerdesk-echo` to the cargo command. The Python suite also runs the Swift capture checks on macOS. To build an app, run `APPLE_SIGNING_IDENTITY=- ./scripts/build_macos.sh`, then use the in-app setup to download models.

Windows support is in progress in [#20](https://github.com/TheGreenCedar/Speakerdesk/issues/20).
With the MSVC C++ build tools, Windows SDK and Rust installed, run the same Cargo
command with `-p speakerdesk-models` added. Windows CI covers storage, WAV I/O,
HTTP routes and child-process cleanup; model tests cover the portable protocol
and unavailable backend. Native capture, production inference and a Windows
installer are still unfinished. The [AMD Silero experiment](experiments/windows/README.md)
records actual GPU execution separately from application support.

## Documentation

- [Architecture](docs/architecture.md): processes, crates, inference boundary, storage
- [Product behavior](docs/product-behavior.md): the user-visible rules
- [Releasing](docs/releasing.md): build, signing, promotion and delivery
- [CI build reuse](docs/ci-build-reuse.md): cache keys, trust and uncached runs
- [Glossary](GLOSSARY.md) and [decision records](docs/adr/)

## License

Speakerdesk's own code and documentation use the [MIT license](LICENSE). Third-party code and models keep their own licenses, recorded in [packaging/THIRD_PARTY_NOTICES.md](packaging/THIRD_PARTY_NOTICES.md). The Mac build collects every dependency notice into one app resource, which you can view from Settings. NVIDIA's checkpoint uses OpenMDW 1.1 and Cohere's uses Apache 2.0. Those licenses still apply if you download the weights separately.
