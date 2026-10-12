# Existing Apple release credentials

Speakerdesk's `Apple package` workflow builds ad-hoc signing inputs without Apple
credentials. The separate `TheGreenCedar/AppleRelease` main workflow validates an
exact registered producer run, source commit, artifact ID and archive digest,
then uses its existing central Developer ID and notarization credentials.

The signer uses its existing GitHub App installation with Actions read access to
Speakerdesk. It signs native code, refreshes the capture-helper receipt, notarizes
and staples both app and DMG, and produces a version-bound updater archive with
the existing app-specific updater signing key. It does not execute the uploaded
application or publish a release. The same workflow retains its registered older
PyInstaller candidates through an explicit per-build runtime mode.

No credential copying or new repository/environment grants are needed for this
path. Private keys and passwords must never enter logs, chat or source control.
The signing job removes its temporary keychain and credential files on exit.
Local builds can use an existing Developer ID identity for disposable checks;
local signing alone does not establish notarization or release provenance.

Canonical owners:

- [Speakerdesk producer](https://github.com/TheGreenCedar/Speakerdesk/blob/main/.github/workflows/apple-build.yml)
- [AppleRelease signer](https://github.com/TheGreenCedar/AppleRelease/blob/main/.github/workflows/sign.yml)
- [Native release acceptance](core-acceptance.md)
