# Windows candidate signing

`scripts/build_windows.ps1 -Sign` adds Azure Artifact Signing and a version-bound
Tauri updater signature to the existing Windows candidate build. It does not
publish a release, install the application, configure cloud access or make the
candidate `public_ready`. Without `-Sign`, the existing unsigned build and its
unsigned NSIS payload derivation remain available.

Use the shared publisher account and public-trust certificate profile. Basic
supports one profile of each type; separate repositories use separate federated
identities and grants against that shared profile. Subscription, identity
validation, billing, profile creation, repository federation and release policy
remain separate owner-controlled setup. This adapter performs no login or
credential registration. Its metadata permits only `AzureCliCredential`.

## Prepare on the Windows build machine

Use x64 Windows, the existing MSVC/Windows SDK build environment, Python 3.9+,
the .NET 8 runtime required by Microsoft's signing client, Azure CLI and the
normal Speakerdesk Node/Cargo dependencies. Authenticate Azure CLI through the
already approved identity. Do not put secrets in metadata or command arguments.
For single-machine `-Sign`, the updater key/password must be available through protected
`TAURI_SIGNING_PRIVATE_KEY` / `TAURI_SIGNING_PRIVATE_KEY_PASSWORD` environment
variables; the compiled updater public key must match it.

```powershell
# Fresh local directories/files; no generated tools or metadata are committed.
.\scripts\windows_signing.ps1 -Action PrepareTools -Tools C:\build\artifact-signing-tools
.\scripts\windows_signing.ps1 -Action PrepareMetadata -Path C:\build\speakerdesk-signing.json `
  -Endpoint https://eus.codesigning.azure.net -Account thegreencedar-signing `
  -Profile public-release -Correlation speakerdesk-<source-commit>
.\scripts\windows_signing.ps1 -Action Check -Tools C:\build\artifact-signing-tools `
  -Metadata C:\build\speakerdesk-signing.json
.\scripts\build_windows.ps1 -Sign -SigningTools C:\build\artifact-signing-tools `
  -SigningMetadata C:\build\speakerdesk-signing.json -OutputDirectory C:\build\speakerdesk-signed-candidate
```

The approved shared account is `thegreencedar-signing`, regional endpoint
`https://eus.codesigning.azure.net`, with the active `public-release` profile.
Publisher identity validation is complete. An owned disposable copy of the
capture helper has passed real Azure signing, RFC3161 timestamping,
`Get-AuthenticodeSignature` (`Valid`) and `signtool verify /pa /all`; its original
bytes were preserved. This proves the helper signing path. A complete signed
installer and its installed behavior remain to be qualified.

Metadata is create-new and excludes all non-CLI credentials, including environment,
workload identity, managed identity, PowerShell and interactive browser
credentials. A CI federated identity can establish the Azure CLI session
separately; this adapter does not edit workflows or service permissions.

`scripts/windows-signing.json` reuses BatCave's exact NuGet pins:
Microsoft.ArtifactSigning.Client 1.0.128 and
Microsoft.Windows.SDK.BuildTools 10.0.26100.8249. Preparation verifies each package
SHA-256 before extraction. Every signing invocation verifies the extracted
inventory and all bytes against those pinned archives, validates the restricted
metadata, and uses SHA-256 Authenticode with Microsoft's RFC3161 timestamp service.
The required verified publisher is `CN=Albert Najjar`; an unexpected existing
signature is rejected rather than replaced.

## Payload and updater evidence

The build signs only the owned capture helper, echo DLL, post-NSIS-patch app,
generated uninstaller and final NSIS installer. The capture capability receipt
is regenerated after signing so its admission hash names the signed helper.
The Tauri 2.12.1 custom hook records hashes, verified publisher/timestamp and a
copy of each actual signed PE under `signing/payloads`. The sole generated-path
exception is NSIS's temporary uninstaller PE in a fresh build-owned TEMP/TMP
directory; all other hook paths must match the exact role allowlist.

The five upstream NSIS plugins are preserved by exact SHA-256 from the pinned
NSIS 3.11 and nsis_tauri_utils 0.5.3 sources. Their receipt does **not** claim a
publisher signature. The three Microsoft runtime DLLs retain their committed
hashes and verified Microsoft signatures. They are never re-signed. Unknown
paths, plugins and changed runtime bytes stop the signed build.

The generated NSIS script is checked for the exact app/resource sources and
destinations. Its embedded WebView2 bootstrapper and Visual C++ CRT inputs bypass
Tauri's hook, so the adapter separately records their hashes and verifies their
Microsoft signatures with Authenticode and SignTool. CRT inputs retain
`signtool verify /pa /all`; only the exact pinned WebView2 bootstrapper uses
`verify /pa /ds 0 /tw /v`, requiring exit 0, Microsoft Corporation and a timestamp.
Its package receipt must record `signature_index: 0` and `timestamped: true`;
primary-only receipt markers on other file classes are rejected.
New packaged input classes fail closed. These upstream files remain unchanged.
The bootstrapper is embedded for installation, rather than left in the app's
installed file inventory. Bundled CRT DLLs are included in that inventory.

`windows-signing.json` pins the independently downloaded [official bootstrapper](https://go.microsoft.com/fwlink/p/?LinkId=2124703)
at **1,860,304 bytes**, SHA-256
`baa78c71c5615533a1ee0c4d2ca2ea3fd24dae01965f9033ac530a61b5aa93a0`,
and records its resolved source URL. Its primary signature passed native trust
and timestamp verification with zero warnings. `/all` fails on its supplementary
self-signed EdgeBuild signature; that signature is preserved without trusting its
root. A different Tauri cache file stops the build and must be replaced with the
reviewed pinned bytes. There is no automatic download, trust-store change,
signature removal or re-signing. All other signed inputs retain `/pa /all`.

Tauri restores the loose app to unsigned `UNK` bytes after bundling. Accordingly,
`package.json.post_build_inputs` describes the files observed **after** bundling;
the loose app entry is not the installed app hash. The adapter validates that
patching its sole `UNK` marker to `NSS` reproduces the exact pre-signing hash
recorded by the app hook. `app_packaged_sha256` and
`artifact-manifest.json.installed_payloads["speakerdesk.exe"]` instead come from
the actual signed hook snapshot. No signed hash is derived from an unsigned PE.

After final Authenticode verification, the locked Tauri CLI signs the final
installer bytes with `--app-version`. A build-only Rust verifier checks the
Minisign payload/global signature, the configured updater public key and exactly
one authenticated `version:` field matching the app version. It rejects a
different key, tampered bytes or a forged trusted comment. Its dependencies
already exist in the application lockfile; it adds no runtime component. Cargo
uses its warmed cache offline for this small helper. The installer must remain
byte-identical through updater signing.

The output contains the installer, `.exe.sig`, actual signed payload snapshots,
hook/package receipts and manifest. `inputs` describes loose/staged build
inputs; `installed_payloads` is the expected installed inventory, including the
generated signed `uninstall.exe`. `payloads_observed_at_hook: true` does not mean
installation was observed: `installed_payloads_observed` and `public_ready` stay
false. A final signed install/launch/capture/runtime/uninstall exercise and the
owner's distribution/promotion checks are still required before publication.

## Keep the updater key on the Mac

For two-phase signing with the existing updater key kept on the Mac:

```powershell
.\scripts\build_windows.ps1 -Sign -DeferUpdaterSignature `
  -SigningTools C:\build\artifact-signing-tools -SigningMetadata C:\build\speakerdesk-signing.json `
  -OutputDirectory C:\build\speakerdesk-signed-candidate
```

`-DeferUpdaterSignature` requires `-Sign` and preserves Authenticode/payload checks.
The manifest records updater `pending: true`, `verified: false` and SHA-256 pins
of the base and Windows configurations captured before and checked after compilation.
The Windows overlay must preserve the base version, updater key and version policy.
The candidate has no `.sig`; plain `-Sign` retains its existing complete path.

Freeze the manifest and transfer the final installer bytes to the Mac, checking
their exact size/SHA-256. Use AppleRelease's independently locked Tauri CLI 2.12.1:
`signer sign --private-key-path <existing-encrypted-key> --app-version <version> <installer>`.
Keep the password only in the signer's child environment. Do not rerun key setup
or copy/rotate the key. Verify the signature/version against the compiled public
key and unchanged installer hash; return only `.sig` and public provenance.
AppleRelease's existing workflow accepts `.app` bundles; reuse its generic
signer/verifier tooling for this local Windows-byte signing step.

Finalize from the same reviewed Speakerdesk source/configuration used to build
the candidate:

```powershell
.\scripts\windows_signing.ps1 -Action FinalizeUpdater `
  -Path C:\build\speakerdesk-signed-candidate -Signature C:\build\returned-installer.exe.sig `
  -Tools C:\build\artifact-signing-tools
```

Finalization checks installer name/size/hash, the build-time configuration pin,
app version, pinned tools, native publisher/timestamp and SignTool trust, then the
existing Rust verifier's payload/key/signed-version contract. It rechecks all
inputs before committing. No credentials/login are required. Different existing
signatures or completed metadata are rejected; identical verified inputs are
idempotent. Verification failure leaves pending metadata untouched.

The candidate lock serializes finalizers. No-replacement `.sig` publication
precedes the atomic manifest commit; interruption can leave an identical `.sig`
beside pending metadata, verified again on retry. Remove an abandoned
`.updater-finalize.lock` only after confirming no finalizer is active. Finalization
keeps installed evidence and `public_ready` false; signed installed QA and
promotion remain separate gates.

Local Windows release preparation requires `native-qa.json` with `passed`,
`platform: windows-x86_64`, exact `version`, full `source_commit`, and
`artifact_manifest_sha256`. Its `verification`/`verification_sha256` bind a JSON
operator report with kind `signed-standard-user-install`, `passed`, `standard_user`,
`installer_sha256`, and the exact observed `installed_payloads` map. The report
confirms `authenticode_verified`, `save_reopen`, `clean_shutdown`,
`uninstall_reinstall` and `data_preserved` for the signed installed package.

The QA file also binds `updater_verification`/`updater_verification_sha256` to the
actual Mac verifier receipt: kind `nsis-installer-updater-verification`, exact
`payload_sha256`, `signature_sha256`, `version`, compiled `public_key`,
`signature_key_version_verified: true` and `payload_unchanged: true`. Helper fixtures,
dirty source, pending signatures, changed reports/configurations or missing signed
snapshots fail preparation. These are explicit local operator assertions bound to
files; preparation does not rerun native trust or cryptography, authenticate CI, prove
an app-driven update, or itself publish a release. For publication, this same hashed
installation report must explicitly record `app_driven_update_observed: true` after
an older installed app downloads, verifies, installs and restarts into this exact
target through its compiled production feed. Direct overinstallation is not that
observation. Missing or non-true values still permit preparation and block publication.
See [releasing](releasing.md#windows-preparation-and-promotion) for staging and merge order.

## Focused local checks

```sh
python3 -m unittest discover -s tests -p test_windows_signing.py -v
cargo test --offline --manifest-path scripts/windows-updater-verifier/Cargo.toml
```

The portable suite exercises exact path/pin rejection, metadata credential
restrictions, signer extraction tampering, generated package source/destination
changes, signed snapshot/capture receipt binding, and the restored app distinction.
The verifier's public-only disposable-key fixture checks real cryptographic
success and payload/key/version/comment failures. These checks do not establish
Windows trust or installed behavior. Those require owner-controlled Windows
execution; the real helper signing proof above is separate from this suite.
