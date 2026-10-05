# Future rootandruntime download handoff

Keep application source private. A private GitHub release asset needs authentication, so it cannot serve as the anonymous website download link.

Suggested stable page: `https://rootandruntime.com/software/speakerdesk/`.
Suggested stable download redirect: `/downloads/speakerdesk/latest/macos-arm64`.
Suggested metadata endpoint: `/downloads/speakerdesk/latest.json`.
These are proposed routes, not deployed endpoints.

The private Apple workflow produces:

- `Speakerdesk_<version>_AppleSilicon.dmg` as the primary installer.
- `Speakerdesk_<version>_AppleSilicon.app.zip` as a secondary app archive.
- `SHA256SUMS`, calculated after any DMG stapling.
- `artifact-manifest.json`, containing source commit, version, architecture, minimum macOS version, file sizes, SHA-256 digests and actual signing/notarization state.

The manifest always says `channel: candidate`, `public_ready: false` and `native_meeting_qa: pending`. A successful signing job does not automatically promote a candidate to stable. Model weights and recordings are excluded from the repository and installer.

Before website publication, verify the exact downloaded artifact's checksum, Developer ID signature and notarization ticket, normal first launch, model setup/resumption and native microphone/system-audio meeting capture. The public manifest must refer to the tested bytes and explicitly record that readiness decision. Do not relabel the current unnotarized candidate as stable.

After readiness, publish only approved binaries, checksums, release notes and a public manifest. Choose an existing website/static storage destination if it supports roughly 100 MB downloads; otherwise use an existing approved object store/CDN or a separate public binary-only release destination. Creating new paid storage, credentials or changing source-repository visibility is not part of this workflow. No website or public asset has been published by this source preparation.

The public metadata should contain `version`, `source_commit`, `released_at`, `platform`, `architecture`, `minimum_os`, `download_url`, `bytes`, `sha256`, `signing`, `notarized` and `release_notes_url`. Public URLs should resolve anonymously. Publish immutable versioned files first, verify their hashes, then atomically update the stable metadata and redirect. Preserve previous approved versions for rollback. The software page can read this metadata to display the latest version, macOS requirement, size and download link.
