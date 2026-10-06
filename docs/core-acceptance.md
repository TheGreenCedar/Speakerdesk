# Core audio acceptance and release promotion

A green source/browser/export suite is insufficient to promote a Speakerdesk
release. The supported promotion command requires a completed actual-model
replay against the exact signed candidate and its pinned checkpoints, plus the
separate native Save/Cancel/retry/replacement gate for that same app. Existing
published releases remain immutable; this policy applies to future promotion.
Publication still requires the user's approval of the new exact candidate.

## What executes

`scripts/packaged_replay.py` executes the unchanged signed frozen
`Speakerdesk.app/Contents/MacOS/speakerdesk-runtime` in a new disposable home.
It does not import the source server, replace models, alter probabilities or
launch hardware capture. Its synthetic helper implements the existing capture
PCM protocol. Real live APIs start meetings, pause/flush provisional output,
apply protected corrections with CAS, resume the production accuracy pass and
stop/save. Real NVIDIA, Cohere and Whisper run from approved read-only cached
checkpoints. Each case records primary provisional and refined documents,
language probes, raw activity, source ownership, unchanged saved PCM and
synthesis metadata. This proves the packaged backend's production audio route;
native microphone/system capture and natural-room quality require separate QA.

Tracked recipe JSON includes silence, seeded stationary/colored noise, quiet and
brief speech, genuinely repeated “thank you”, continuous sentences, mixed
voices, real preceding-language context and protected corrections. Independent
holdouts cover fresh English/French content and short/long pauses. Holdout
failures remain failures; do not tune on or silently replace those cases.
These are small synthetic structural checks, not a benchmark of human identity,
accent robustness, separated overlap transcription or general meeting accuracy.

The helper uses only verified preinstalled Apple super-compact voices. It
does not download or substitute a missing voice. Only digital-zero padding is
trimmed before assembly; every nonzero original speech sample is retained.
Timing assertions refer to audio regions, never claimed word alignment.
Protected-edit controls use a natural long-pause boundary so a legitimate
indivisible candidate conflict is not “fixed” by weakening edit protection.

## Resource and identity requirements

Freeze a clean source commit before signing/testing. The runner compares the
entire app inventory with the signed ZIP, verifies the hardened Developer ID
signature, and hashes all loaded checkpoints and frozen configuration metadata.
It requires a fresh coordinated resource lease with `admitted: true`,
`checked_utc`, `floor_plus_other_reserve_bytes`, `seconds` (60–1200),
`output_reserve_bytes` (64–128 MiB), and
`other_model_build_operations_joined: true`. Coordination must establish the
other owners have stopped their model/build work. No denied shared-process
inventory is retried. Metrics cover only this harness's owned process groups;
if those metrics are denied, acceptance stops and remains unqualified.

The independent guard samples disk/output/memory/owned RSS during blocking
operations, caps owned RSS at 4 GiB, and terminates only recorded owned groups
on deadline or resource failure. Every started group is cleaned even if server
construction or teardown fails. Missing/unverified cleanup cannot pass.
Use existing cached model folders under the requested `--models` directory;
disposable links may point at approved snapshots without copying weights.

```sh
python scripts/packaged_replay.py \
  --app /path/to/exact/Speakerdesk.app \
  --manifest /path/to/signed/artifact-manifest.json \
  --models /path/to/approved/cached-models \
  --lease /path/to/new-coordinated-lease.json \
  --output /path/to/new/disposable-acceptance-attempt
```

Use a distinct output for every attempt. A failed or partial receipt is retained
and blocks promotion. Do not run this command under the expired/released baseline
window; obtain a new coordinated window for the corrective candidate.

## Fail-closed promotion

`core_acceptance.py` recomputes all mandatory case assertions from the hashed
raw documents and saved/input WAVs. It rejects missing/skipped/partial/source/
mock/browser evidence, source/package/runtime/bundle/model/config/suite/harness
mismatches, changed source, failed cases, missing stages, forged assertion maps
that contradict traces, altered saved PCM, and failed resource/cleanup evidence.
Checkpoint, recipe and verifier changes require a fresh exact-candidate run.
It authenticates local receipt bindings; it is not hardware attestation against
a malicious operator who forges every local trace. Independent review remains
part of the release process.

`scripts/promote_release.py` first validates core and native export gates, then
independently checks trusted successful main producer/signer runs and their
artifact provenance through read-only authenticated GitHub requests. Supply
the retained original signer artifact ZIP: its service digest and exact embedded
manifest/package bytes prevent relabelling an older signed app with a new source.
Default preparation prints a reviewable plan and makes no release writes.

```sh
python scripts/promote_release.py \
  --report /path/to/core-acceptance.json \
  --manifest /path/to/signed/artifact-manifest.json \
  --native-qa /path/to/verified-native-gate.json \
  --signer-archive /path/to/original-signer-artifact.zip \
  --notes /path/to/final-reviewed-notes.md
```

Only after approval use the same command with `--publish`. It preserves an
existing draft/release, verifies current main and peeled Git tag targets, uploads
the same four signed release assets as before, checks their actual service
digests before publication, and streams/re-hashes authenticated published
downloads with bounded waits. Partial drafts/download failures need inspection;
do not auto-overwrite or auto-resume. Administrators can bypass a CLI by using
GitHub directly; this is the mandatory supported path, not a claim of external
permission enforcement. Do not use historical one-off publishers for new releases.

## Hosted CI and evidence scope

Normal source CI executes the CPU policy/fixture/ownership tests. Its fabricated
contract traces are explicitly labelled and never actual-model QA. The existing
package build contains no model weights and remains an unsigned producer with
read-only permissions; central signing controls are unchanged. Package manifests
mark core acceptance required/not-run and remain non-public candidates.

A Developer ID runtime can report actual Metal/library capability with
`--model-capability`; that report always says `models_executed: false`.
Hardened ad-hoc Python is not executed to manufacture a green capability check:
that build records capability unavailable in `.cache/model-capability.json`
and CI logs. The fixed four-file signing artifact envelope is unchanged.
Metal/library capability or packaged startup does not satisfy acoustic acceptance.
Actual acceptance is a bounded coordinated local test with the approved cache;
hosted CI alone cannot promote this candidate.
