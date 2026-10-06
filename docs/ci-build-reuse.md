# CI build reuse

Speakerdesk builds unsigned, exact-source candidates in `apple-build.yml`.
AppleRelease remains the trusted signer and publisher: approve an existing
successful main/dispatch producer run, artifact ID, source commit, archive digest
and payload hashes in its central policy, then sign that artifact. Do not build
the same commit again merely to notarize or release it. PR artifacts cannot be
promoted by that policy. This change does not edit AppleRelease or its active
release approvals.

## Baseline and expected savings

Read-only observations from actual hosted macOS runs on 2026-10-05:

| Run | Source | Complete build step | PyInstaller | npm cache observation |
| --- | --- | --- | --- | --- |
| [37363635410](https://github.com/TheGreenCedar/Speakerdesk/actions/runs/37363635410) | `8201f195` | 345 seconds | 49.7 seconds | Hit restored only 1,793 bytes |
| [37386407723](https://github.com/TheGreenCedar/Speakerdesk/actions/runs/37386407723) | `05abcc3d` | 391 seconds | 80.9 seconds | Miss |

These different source heads are baseline observations, not a cold/warm
comparison. Both rebuilt Python dependencies, PyInstaller analysis/archive and
Rust dependencies. The npm action restored `~/.npm`; the script actually used
`.cache/npm`. Rust had no persisted objects and explicitly cleared its compiler
wrapper. Swift module state stayed on the ephemeral runner. PyInstaller used
`--clean` with no persisted component output.

The new workflow is expected to avoid Python downloads, reuse compatible Rust
compiler objects, and skip the complete PyInstaller/Swift component compilation
when exact inputs are unchanged. Bundle creation, source tests and archive
hashing still run. No new native build has been run for this draft, and no
percentage or end-to-end speedup has been measured. The observed 49.7–80.9
seconds of PyInstaller work identifies a potential avoidable cost, not a promised
net saving after cache transfer. A cold cache will still compile.

## Cache boundaries

Every native key binds macOS version, runner image, architecture, deployment
target, exact Python/uv/Rust/Node/sccache versions, Swift/Xcode and SDK identity.
This conservatively invalidates all caches on toolchain drift.

| Cache | Inputs and reuse rule | Persisted paths |
| --- | --- | --- |
| uv dependencies | All requirements lock files; exact key | `.cache/uv` |
| Cargo downloads | Cargo.lock; exact key | Cargo registry and git downloads |
| npm downloads | package-lock.json; exact key | `.cache/npm` via npm configuration |
| Rust compiler | Compatible toolchain/build contract prefix; immutable source key | Bounded local sccache objects |
| Python component | All tracked app, packaging, requirements and builder inputs; exact key only | Built ad-hoc sidecar and SHA-256 receipt |
| Swift component | Swift source, plist, entitlements and builder inputs; exact key only | Capture binary before signing and SHA-256 receipt |

Whole components reject mismatched receipts and changed bytes before copying.
PyInstaller pickle analysis state is never persisted; `--clean` still applies
when a component misses. Rust source edits preserve Python and Swift reuse.
Developer ID local builds bypass component caching entirely. The total save
budget is 2 GiB and sccache itself is bounded to 512 MiB. Oversized caches are
reported and not uploaded; build/artifact production still completes.

Only successful `main` runs save dependency/compiler/component caches. Branch
and PR jobs can restore caches within GitHub's branch scope but never save
trusted caches. The producer has read-only permissions, no signing credentials,
and no automatic PR/tag build trigger. Cache paths never include keychains,
credentials, user recordings, model downloads or whole home/temp directories.
Pin action implementations, separate cache restore/save, and never use a cached
executable or a successful PR job as release authorization.

## Source routing and measurement

Feature pushes do not run CI; pull requests do. Only superseded PR source checks
cancel automatically. Main pushes and manual producer/signing runs are never
cancelled by PR concurrency. Documentation-only changes skip dependency installs
and application tests while retaining workflow validation, cache tests and the
tracked-payload guard, keeping the required Source checks job present.

Each manual candidate run reports requested/matched cache keys, exact component
hits, component elapsed time, complete build elapsed time, compiler statistics
and cache bytes in its summary/log. After this draft is reviewed and the current
release owner clears the lane, use the next already-needed trusted main build
as the cold population run. A later already-needed same-toolchain build supplies
warm evidence; compare changed inputs and component hit types before reporting
savings. Reuse an existing exact-source candidate for release instead of
dispatching a timing-only rebuild. No paid runner or repeated experiment is
required.

## Applying this pattern to future repositories

Use [CodeStory's cache contract](https://github.com/TheGreenCedar/CodeStory/blob/424d56b1b5452555460d528f40d08a231ac72a92/.github/scripts/cargo-cache-contract.mjs)
as the reference for compatible compiler objects, exact-source evidence and
hit/miss reporting. Its Rust/native workspace cache cannot simply replace
Speakerdesk's Python/MLX and Tauri component inputs. For each future repository:

1. Measure current job/step cost and prove cache directories match actual writes.
2. Separate downloads, compiler objects, exact component outputs and release artifacts.
3. Key actual toolchains, OS/architecture, locks and all component inputs; bound bytes.
4. Save trusted caches only, keep PR privileges read-only, and filter unaffected work.
5. Promote a verified artifact of the exact commit through the trusted signer.
6. Validate workflow syntax and hostile identity/privilege cases before enabling it.
7. Report expected savings separately until actual compatible cold/warm runs exist.

This is a reusable review pattern; it does not install workflows in other repos.
