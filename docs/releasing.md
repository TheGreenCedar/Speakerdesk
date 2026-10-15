# Releasing

A release moves one exact build of a commit through four stages. Speakerdesk builds the candidate, AppleRelease signs it, promotion publishes it, and the website serves it. Nothing gets rebuilt for signing or release. Model weights and recordings never ship in an installer.

## 1. Build the candidate

Run `Apple package` (`.github/workflows/apple-build.yml`) by manual dispatch on `main`. It produces an ad-hoc signed candidate artifact. Caching is described in [CI build reuse](ci-build-reuse.md). The workflow has read-only permissions and no Apple credentials. PR artifacts are not promotable.

To build locally, run `APPLE_SIGNING_IDENTITY="<Developer ID identity>" ./scripts/build_macos.sh` (or `-` for ad-hoc). Local signing is fine for disposable checks. It does not count as notarization or release provenance.

DMG creation retries only `hdiutil`'s `Resource busy` failure, up to three attempts
with 2- and 4-second delays, using the same verified staged app. Other errors
and exhausted retries stop the build before the DMG is signed or an artifact
manifest is produced.

## 2. Sign and notarize (AppleRelease)

Before dispatching the signer, register the successful producer's exact run,
artifact, source commit, archive digest/size and payload hashes in AppleRelease's
`policy/apps.json`. Retain the native runtime, capture capability and updater
opt-ins. Validate and merge that one-build registration through its normal PR
checks; the signer accepts only registered artifacts. Do not dispatch it against
an older registration or rebuild an already qualified producer artifact.

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

Preparation checks the payload inventory and hashes, notarization, the compiled updater key, and the QA binding to the signed app ZIP. It then authenticates the producer run, the signer run and the original signer artifact against GitHub. The notes must name the exact source commit. `--publish` uploads a draft release and resolves it by identity, then verifies its assets. If the `v<version>` tag doesn't exist, it creates the tag at the source commit; an existing tag must already point there. It then publishes the release and rehashes the downloads. Producer manifests stay `public_ready: false`, and promotion never rewrites provenance artifacts.

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

## Windows preparation and promotion

Windows uses the [two-phase signing path](windows-signing.md), with the updater key
remaining on the Mac. Freeze the qualified source/model candidate before packaging.
After final NSIS signing and actual standard-user installation QA, prepare locally:

```sh
python3 scripts/promote_release.py --manifest <candidate>/artifact-manifest.json \
  --native-qa <proof>/native-qa.json --notes <proof>/notes.md
```

This branch binds local operator reports, signed payload snapshots, final installer,
signature and build-time configuration pins. It supplies no Apple producer/signer
provenance. Native installed observations and actual updater
verification must name these exact bytes; portable contract tests cannot supply them.
The candidate manifest stays `public_ready: false`. Model distribution/readiness and
owner-controlled public promotion remain separate gates. Never relabel the existing
Mac `v0.7.6` tag or artifacts with a different Windows source commit.

The website's independent `speakerdesk-windows-releases.json` catalog starts empty.
Its `/downloads/speakerdesk/updater/windows-x86_64/latest.json` endpoint returns 204
until an approved Windows entry is selected; the Mac feed/catalog are unchanged.
Approved Windows installers use the existing immutable versioned download route
with `application/octet-stream`. Real A-to-B update proof requires an older installed
Windows build with this Windows-specific endpoint and a newer approved signed target;
direct installer replacement and detached-signature verification do not prove that flow.

Keep the application PR draft while qualifying the update. First freeze a clean newer
Windows candidate, sign it and qualify its standard-user installation. Under release
authorization, stage those exact bytes at a new immutable website key, verify the
download, and select that approved target in the Windows feed. Then use the older
installed app's update UI through its compiled production endpoint to download,
install and restart into the target, preserving synthetic saved data. Record
`app_driven_update_observed: true` in the existing hashed installation report only
after that interaction succeeds; refresh its `verification_sha256` in `native-qa.json`.
Preparation is allowed before this observation, but `--publish` requires it explicitly.

After acceptance, merge the PR preserving the original candidate commit and exactly
its Git tree. Windows publication requires that commit to be an ancestor of current
reviewed `main` with the same tree. A changed merge tree blocks promotion; never
rewrite the signed candidate's source to the merge SHA. With an authorized release,
rerun preparation with `--publish`. The shared draft/upload/tag/download checks keep
the new `v<version>` tag at the original qualified source and refuse existing releases
or a differently bound tag. The release title identifies Windows and `make_latest`
is false, preserving the existing Mac latest selection. Mac publication retains its
exact-main and AppleRelease provenance requirements.
