# CI build reuse

The Linux/macOS native checks and the manual Apple producer reuse Cargo
downloads and compiler objects through the existing pinned sccache 0.16.0
action. Cargo target directories stay fresh. Every candidate still runs the
locked build, native dependency preparation, bundle checks and exact-source
artifact manifest generation.

AppleRelease signs and notarizes an authenticated producer artifact selected by
its immutable run, artifact, source and payload identities. Speakerdesk's
promotion tooling verifies the signed package and native QA before publishing.
A cache hit supplies build inputs; it does not authorize a release or replace
package verification.

## Persisted paths and invalidation

| Cache | Reuse rule | Persisted paths |
| --- | --- | --- |
| Cargo downloads | Exact Cargo.lock key, compatible tool/environment fallback | `~/.cargo/registry`, `~/.cargo/git` |
| Rust compiler objects | Exact source/profile key, compatible native contract fallback; sccache validates each compilation | `$RUNNER_TEMP/speakerdesk-sccache` |
| npm downloads (producer) | Exact package-lock.json key, compatible tool/environment fallback | `.cache/npm` |
| Swift capture helper (producer) | Exact source/tool/environment key and verified receipt; no fallback | `.cache/components/capture` |
| Source-check Python downloads | Exact CI requirements key | `~/.cache/pip` |

Compiler compatibility binds OS, runner image, architecture/host target, Rust,
sccache, native C compiler, CMake/Ninja and Rust flags. On macOS it also binds
Xcode, Swift, SDK and deployment target (15.0 in the producer; Rust's default in
native CI unless explicitly set). The tracked contract covers
Cargo configuration/toolchain, workflow/build scripts, vendored MLX adapter,
MLX runtime fetch pins, echo source lock/build guard, Tauri configuration and
embedded web/model metadata. Changing these inputs blocks compiler fallback.
The pinned MLX 0.32.2 runtime and echo sources are still prepared and checked by
their existing scripts; their downloads and build trees are not persisted.

Rust source, Cargo manifest/lock/version and job profile changes select a new
exact compiler key. If the native contract is unchanged, the compatible prefix
can retain unchanged dependency objects;
sccache checks the actual rustc flags, source, dependencies and environment.
The two native CI jobs identify their test profiles separately from the Apple
producer's release profile. Python/Node versions do not split the Rust object
store. A different Rust version, target or native toolchain does.

Rust incremental compilation is disabled. sccache does not cache Rust crates
that invoke the linker, so final executables, linking and packaging remain real
work on every run. See the [pinned sccache Rust documentation](https://github.com/mozilla/sccache/blob/v0.16.0/docs/Rust.md).
No Cargo target directory, model weights, recording, credential, keychain,
whole home/temp directory or signed artifact is saved. The existing limits are
512 MiB for sccache and 2 GiB for total native cache uploads; an oversized cache
skips upload without invalidating an otherwise successful build.

The exact Swift cache remains confined to the ad-hoc producer. Its receipt must
match both identity and payload digest before restoration. Developer ID builds
bypass component caching, and the trusted signer signs the freshly staged
package after authenticating its producer provenance.

## Trust and uncached builds

Restore/save use separate pinned Actions. Only successful `main` push or manual
runs save native CI caches, and only successful `main` manual runs save producer
caches. The job byte budget must pass before native saves. PR and branch runs
can restore within GitHub's cache scope but cannot write trusted main caches.
The sccache GitHub backend is disabled, so explicit guarded save steps own
persistence. All source/producer jobs retain read-only permissions and no
signing credentials.

Both workflows accept the boolean manual input `cache`, defaulting to `true`.
Dispatch with `cache=false` for an uncached comparison: all cache restores/saves
are skipped, `RUSTC_WRAPPER` is empty, and the producer passes no Swift component
key. Locked commands, integrity checks, packaging and provenance remain the
same. The expression checks the raw event input against the string `'false'`;
absent inputs on push/PR keep normal caching enabled.

## Measurements

Job summaries/logs report requested and restored keys, native test/desktop or
complete producer build time, sccache requests/hits/misses, and cache bytes.
Restore/save step durations are available in the Actions run. Compatible-prefix
restores and exact-key hits are distinguishable from the matched key.

No cold/warm speedup is claimed by this source change. Record comparable hosted
runs of the same revision, runner image, toolchain, target and profile with
`cache=false` and with a populated cache; report transfer costs as well as build
time. A new version/contract key may be cold, and sccache's library-only coverage
limits the possible gain. Reuse an already required exact-source producer
artifact for signing instead of rebuilding it merely for release.

The implementation follows [CodeStory's pinned sccache and explicit trusted
save pattern](https://github.com/TheGreenCedar/CodeStory/blob/424d56b1b5452555460d528f40d08a231ac72a92/.github/workflows/rust-ci.yml).
[BatCave's maintained Rust-cache action](https://github.com/TheGreenCedar/BatCave/blob/2355d82b08a097259a67ccd789da032945768027/.github/workflows/validation.yml)
is another local reference; Speakerdesk retains its existing mechanism to avoid
persisting Cargo targets alongside bundle output. Neither reference's timings
establish Speakerdesk's performance. Superseded main jobs can also occupy macOS
runner capacity; that scheduling issue is separate from cache effectiveness.
