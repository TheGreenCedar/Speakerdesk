# Speakerdesk

Private meeting transcription and speaker labeling for Apple Silicon Macs, using NVIDIA Nemotron 3 diarization and Cohere Transcribe through MLX. Audio stays on the Mac; the app downloads pinned public model checkpoints during explicit setup. No paid transcription API is required.

The desktop shell uses Tauri, a bundled Python runtime and a Swift microphone/system-audio capture helper. macOS 14 or newer and an Apple Silicon processor are required. Allow roughly 2.3 GB for model setup and additional space for recordings. The models themselves occupy about 1.7 GB. This source repository contains no recordings, model weights, credentials or prebuilt application.

## Interface previews

These previews use invented names and transcript text with silent fixture audio. Native capture and model inference were not run for these screenshots. Meeting and name-picker images follow your GitHub light or dark appearance.

### Meeting

Review speaker names, edit passages and export the transcript.

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="docs/screenshots/meeting-dark.jpg">
  <img alt="Speakerdesk meeting transcript with Priya and an unnamed second speaker, editable passages and playback controls" src="docs/screenshots/meeting-light.jpg">
</picture>

### People

Save names for future meetings.

![People dialog with a saved Priya name and controls to add or rename a person](docs/screenshots/people-light.jpg)

### Name picker

Choose a saved person or apply a name to this meeting.

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="docs/screenshots/name-picker-dark.jpg">
  <img alt="Name picker with Priya selected from saved people and an Apply name button" src="docs/screenshots/name-picker-light.jpg">
</picture>

## Development

Install Node.js 24, Rust 1.97.1, `uv`, and Apple command line tools. Then run:

```sh
./scripts/setup.sh
./scripts/run_local.sh
```

The local web app opens at http://127.0.0.1:8790. Use its setup control to download models. Importing audio works in this mode; native capture requires the desktop helper built by the Apple packaging script. CPU lifecycle tests do not download models or record audio:

```sh
.venv-package/bin/python -m unittest discover -s tests -v
```

## Apple builds

`Apple package` is a manual GitHub Actions workflow. Its default ad-hoc mode produces private test artifacts. Ad-hoc signing does not establish trusted distribution or notarization. Its notarized mode requires existing Apple credentials using BatCave's secret names; see [credential reuse](docs/credential-reuse.md). No secrets are provisioned by the workflow source and no release is published.

```sh
APPLE_SIGNING_IDENTITY="existing Developer ID identity" ./scripts/build_macos.sh
```

Local real-model streaming has been exercised on synthetic two-voice speech. CPU CI checks timing, lifecycle and access controls; it does not prove noisy-meeting accuracy, native capture quality or packaged-model parity. Live text is committed in short crops; stopping saves pending results without a full-recording refinement pass. Diarization provides region boundaries; Cohere supplies text without word timestamps, so exported times are speech-region boundaries.

The current build remains a release candidate until notarization, normal first-launch checks and native meeting capture checks succeed. [Download contract](docs/download-contract.md) describes the future stable download handoff.

People saves reusable local names; speaker corrections and confirmed introduction suggestions apply to the current meeting. Voice-profile interfaces are present, but recognition stays unavailable until a local model adapter and measured calibration are approved. No voice is saved by adding a name. See [People and voice profiles](docs/people-and-voice-profiles.md) for storage behavior and the proposed measured plan.

Third-party model and dependency terms are included in [packaging/THIRD_PARTY_NOTICES.md](packaging/THIRD_PARTY_NOTICES.md) and the accompanying license files. NVIDIA's checkpoint uses OpenMDW 1.1; Cohere's checkpoint uses Apache 2.0. Model licenses remain applicable when downloading weights separately.
