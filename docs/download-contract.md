# Public download and updater delivery

The existing website at `rootandruntime.com` serves approved Speakerdesk releases:

- `/downloads/speakerdesk/latest.json`: current installer metadata.
- `/downloads/speakerdesk/latest/macos-arm64`: redirect to the immutable installer.
- `/downloads/speakerdesk/updater/latest.json`: Tauri updater metadata and signature.
- `/downloads/speakerdesk/releases/<version>/<filename>`: immutable DMG and updater archive.

The website source is `TheGreenCedar/the-green-cedar`. Its
`src/data/speakerdesk-latest.json` owns the current release and
`src/data/speakerdesk-previous-releases.json` preserves approved older releases.
The route implementation and
[operations document](https://github.com/TheGreenCedar/the-green-cedar/blob/main/docs/operations/speakerdesk-release.md)
own storage and deployment details. Historical provisioning instructions are not
new release prerequisites.

After authenticating and qualifying the exact AppleRelease package, publish its
GitHub release, upload only the final DMG and signed updater archive to new
versioned keys in the existing storage, and read them back to verify exact hashes.
Uploads must not overwrite an existing object. Preserve content types, attachment
filenames, SHA-256 metadata and the existing private/no-store storage policy.

Only then advance the website catalog through its normal deployment, retaining
the previous catalog entry and objects. Verify anonymous installer/archive bytes,
latest metadata, updater version/signature and the previous version's routes.
The app ZIP, signature file, authenticated signer manifest and SHA256SUMS remain
available in the GitHub release. No new storage, credentials, visibility changes,
CMS post or newsletter is required for an ordinary release.

Producer and signer manifests describe immutable candidate bytes; qualification
and public catalog promotion do not rewrite those provenance artifacts. Model
weights and recordings are excluded from installers.
