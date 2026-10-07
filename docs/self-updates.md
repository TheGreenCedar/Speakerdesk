# Desktop self-updates

Speakerdesk uses the official Tauri updater from Rust. The local web page can
request Check, Download, Install and Cancel; it cannot supply update URLs, keys,
signatures or file paths. Users of 0.6.1 install the first updater-enabled version
through the existing DMG. That version can update subsequent versions in-app.

The endpoint is
`https://rootandruntime.com/downloads/speakerdesk/updater/latest.json`.
It returns HTTP 204 when no qualified updater release is published. An available
release identifies a version-specific Apple Silicon `.app.tar.gz` on the same
website. The client verifies its Tauri signature and authenticated version before
making installation available. Downgrades are disabled.

The plugin is locked to official revision
[`65ce65bdbb4390f73076c8182fbf06a52036f241`](https://github.com/tauri-apps/plugins-workspace/tree/65ce65bdbb4390f73076c8182fbf06a52036f241/plugins/updater).
It includes the macOS replacement repair that stages on the installation volume,
uses atomic exchange, and preserves/restores the prior app on fallback failure.
Move to a corrected stable release after validating the same replacement and
signed-version behavior. See the [official updater documentation](https://v2.tauri.app/plugin/updater/).

## Installation admission

Settings → Updates separates checking, downloading and installation. Interrupted
downloads can be retried after another check; the plugin does not resume partial
downloads. Cancel rejects late download and preparation callbacks. Installation
is unavailable during starting, recording, paused or finishing capture.

Before installing, Speakerdesk rejects active passage drafts, recoverable edits,
Undo windows, exports and open identity/dialog changes. These refusals keep local
controls usable. Otherwise it freezes editing, waits for existing mutations and
saves the transcript. A matching attempt and preparation token must acknowledge
the completed save. Cancel remains usable while preparing.

Native exports and the Python runtime then reserve their actual shared admission
locks. Queued/running inference, capture finalization, voice recognition, model
setup and writes keep their leases until cleanup. Installation requires both the
runtime's safe-cleanup acknowledgement and observed clean process termination.
A timeout or unknown process state preserves protection and never force-kills a
recording or starts a second service. An install failure recovers one service
only after confirmed safe termination and usable executable paths. Successful
installation restarts Speakerdesk. Read-only/translocated/development app
locations are refused before shutdown.

## Release contract

AppleRelease produces the archive from the final signed, notarized and stapled
app, verifies its extracted bytes/modes/links and native signature/ticket, then
signs with an authenticated version. The signed artifact contains exactly:

- `Speakerdesk_<version>_AppleSilicon.dmg`
- `Speakerdesk_<version>_AppleSilicon.app.zip`
- `Speakerdesk_<version>_AppleSilicon.app.tar.gz`
- `Speakerdesk_<version>_AppleSilicon.app.tar.gz.sig`
- `artifact-manifest.json`
- `SHA256SUMS`

The existing producer/signer provenance, acoustic validation and native export
gates still apply. Promotion binds the updater public key to the compiled client
and requires the exact signed payload inventory. Releases 0.6.2 and later fail
promotion without configured updater artifacts. Legacy 0.6.1 retains its original
four-member artifact contract.

The public website release entry copies `updater.signature` from the authenticated
signer manifest, and the tar payload's `bytes` and `sha256` from `files`. Its
`download_url` is the website's version-specific tar route; it must not use an
unsigned producer artifact. Publish the qualified archive and manifest only with
the corresponding validated release. The existing R2 delivery account is enough.

## One-time key configuration and remaining verification

The committed native configuration deliberately has an empty `pubkey`; updater
operations fail closed until the operator supplies the application-specific
public key. AppleRelease needs `TAURI_SIGNING_PRIVATE_KEY` and, for an encrypted
key, `TAURI_SIGNING_PRIVATE_KEY_PASSWORD`. The private key stays in AppleRelease
and the operator's backup. Speakerdesk and the website receive only public data.
No additional Apple, GitHub release, website or R2 credential is required.

The separately prepared AppleRelease `scripts/setup_updater_key.sh` is a
user-run one-time workflow; agents have not executed it. Its instructions are in
AppleRelease `setup/UPDATER.md`. After operator provisioning, set the same public
key in `desktop/src-tauri/tauri.conf.json` and AppleRelease's trusted app policy,
then register the immutable combined candidate with updater support.

Focused source tests cover real editor save/Undo behavior, HTTP/runtime admission,
sidecar shutdown, native export files, preparation ownership and filesystem
preflight. Full Tauri compilation, real plugin HTTP failure/signature cases and
disposable signed native installation/recovery remain release gates. Source tests
do not establish that an installed upgrade has succeeded. Test installed updates
with synthetic fixture data and preserve recordings and model caches.
