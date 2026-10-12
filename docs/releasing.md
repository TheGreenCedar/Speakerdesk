# Releasing

A release moves one exact build of a commit through four stages. Speakerdesk builds the candidate, AppleRelease signs it, promotion publishes it, and the website serves it. Nothing gets rebuilt for signing or release. Model weights and recordings never ship in an installer.

## 1. Build the candidate

Run `Apple package` (`.github/workflows/apple-build.yml`) by manual dispatch on `main`. It produces an ad-hoc signed candidate artifact. The workflow has read-only permissions and no Apple credentials. PR artifacts are not promotable.

To build locally, run `APPLE_SIGNING_IDENTITY="<Developer ID identity>" ./scripts/build_macos.sh` (or `-` for ad-hoc). Local signing is fine for disposable checks. It does not count as notarization or release provenance.

## 2. Sign and notarize (AppleRelease)

The [`TheGreenCedar/AppleRelease`](https://github.com/TheGreenCedar/AppleRelease/blob/main/.github/workflows/sign.yml) signer checks the exact producer run, source commit, artifact ID and archive digest against its central policy. It then signs the native code, refreshes the capture-helper receipt, and notarizes and staples the app and the DMG. It also signs a version-bound Tauri updater archive. It uses its existing Developer ID, notarization and updater-signing credentials, and needs no new grants. Private keys stay in AppleRelease and must never appear in logs, chat or source.

The signed artifact contains exactly:

- `Speakerdesk_<version>_AppleSilicon.dmg`
- `Speakerdesk_<version>_AppleSilicon.app.zip`
- `Speakerdesk_<version>_AppleSilicon.app.tar.gz` and `.app.tar.gz.sig`
- `artifact-manifest.json`, `SHA256SUMS`

The updater public key is compiled in from `desktop/src-tauri/tauri.conf.json` and must match AppleRelease's trusted app policy.

## 3. QA and promote

Test the final signed app: normal launch, relocated bundle/runtime checks, owned-process cleanup, and any product checks the change needs. Use synthetic data only, and never install over the working app or touch production data. Write the report and a `native-qa.json` with `passed`, `source_commit`, `version`, `app_archive_sha256`, `verification` (the report path) and `verification_sha256`.

```sh
python3 scripts/promote_release.py --manifest artifact-manifest.json --native-qa native-qa.json \
  --notes notes.md --signer-archive signer.zip            # prepare and verify only
# add --publish only for an authorized release
```

Preparation checks the payload inventory and hashes, notarization, the compiled updater key, and the QA binding to the signed app ZIP. It then authenticates the producer run, the signer run and the original signer artifact against GitHub. The notes must name the exact source commit. `--publish` creates a draft release, verifies its assets and tag, publishes it, and rehashes the downloads. Producer manifests stay `public_ready: false`, and promotion never rewrites provenance artifacts.

## 4. Website and updater delivery

`rootandruntime.com` serves releases from `TheGreenCedar/the-green-cedar`. Its [operations doc](https://github.com/TheGreenCedar/the-green-cedar/blob/main/docs/operations/speakerdesk-release.md) owns the storage and deploy details.

| Route | Content |
| --- | --- |
| `/downloads/speakerdesk/latest.json` | Current installer metadata |
| `/downloads/speakerdesk/latest/macos-arm64` | Redirect to the immutable DMG |
| `/downloads/speakerdesk/updater/latest.json` | Tauri updater metadata and signature (HTTP 204 when none) |
| `/downloads/speakerdesk/releases/<version>/<filename>` | Immutable DMG and updater archive |

1. Upload only the final DMG and signed updater archive to new versioned keys. Never overwrite an existing object. Read them back and check the hashes, and keep content types, filenames, SHA-256 metadata and the private/no-store policy.
2. Update `src/data/speakerdesk-latest.json` and move the previous entry into `speakerdesk-previous-releases.json`. Copy `updater.signature` from the signer manifest, and the tar's `bytes`/`sha256` from its `files`. Point `download_url` at the website's versioned tar route, never at a producer artifact.
3. Deploy, then check the installer and archive bytes anonymously, along with the latest metadata, the updater version and signature, and the previous version's routes.

The app ZIP, signature, signer manifest and `SHA256SUMS` stay attached to the GitHub release.
