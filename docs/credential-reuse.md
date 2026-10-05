# Reuse existing Apple credentials

The Apple packaging workflow follows BatCave's existing Apple Silicon build, ephemeral keychain import and Apple API key notarization pattern. It does not publish releases or configure credentials.

Read-only metadata inspection on 2026-10-05 confirmed these existing scopes:

| Purpose | BatCave repository secret | CodeStory `macos-release-signing` environment secret |
| --- | --- | --- |
| Developer ID certificate/private key, base64 P12 | `APPLE_CERTIFICATE` | `APPLE_DEVELOPER_ID_P12_BASE64` |
| P12 password | `APPLE_CERTIFICATE_PASSWORD` | `APPLE_DEVELOPER_ID_P12_PASSWORD` |
| Existing signing identity | `APPLE_SIGNING_IDENTITY` | `APPLE_SIGNING_IDENTITY` |
| Notarization API key ID | `APPLE_API_KEY` | `APPLE_NOTARY_KEY_ID` |
| Notarization issuer ID | `APPLE_API_ISSUER` | `APPLE_NOTARY_ISSUER_ID` |
| Notarization private key | `APPLE_API_KEY_CONTENT` (raw P8) | `APPLE_NOTARY_KEY_P8_BASE64` (base64 P8) |

The Speakerdesk workflow references the six BatCave names. These are references only. Its notarized mode fails before building if any required credential is unavailable. No new Apple keys or certificates are needed.

`TheGreenCedar` is a personal account. Its existing repository and environment secrets do not automatically become available to another repository, and GitHub's secret API does not return decrypted values. Passing `secrets: inherit` to a reusable workflow would use caller-accessible secrets; it does not import the called repository's secret store.

The concrete outstanding access action is to authorize the existing BatCave signing certificate, P12 password, signing identity, notarization key ID, issuer ID and raw P8 key for `TheGreenCedar/Speakerdesk`, under the six BatCave names above. This would give this repository's authorized workflows access to the existing Developer ID private key and notarization authority. It requires action-specific approval and a private provisioning path using the user's existing credential source; no decrypted keys should be sent through chat or committed. This action has not been applied. Existing BatCave and CodeStory secret scopes and workflow access settings remain as inspected.

Alternatively, an authorized local build can use the existing Mac Developer ID identity and an existing notarytool credential profile without copying the private key into another repository. A local build must still be notarized and assessed normally before public distribution.

Workflow references:

- https://github.com/TheGreenCedar/BatCave/blob/main/.github/workflows/bundles.yml
- https://github.com/TheGreenCedar/BatCave/blob/main/.github/workflows/release.yml
- https://github.com/TheGreenCedar/CodeStory/blob/main/.github/workflows/release.yml

The workflow reuses the standard `macos-15` arm64 label documented in [GitHub's runner reference](https://docs.github.com/en/actions/reference/runners/github-hosted-runners). Apple packaging is manually dispatched. Standard private-repository runners use the account's existing Actions minute allowance and billing policy; no billing settings are changed by this source.
