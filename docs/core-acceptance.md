# Release acceptance status

`scripts/promote_release.py` promotes an authenticated native package after
recorded operator QA. Preparation checks the exact payload inventory and hashes,
app/DMG notarization, the compiled updater key, and the final signed app archive
named by that QA. It then authenticates the producer and signer runs and the
original signer artifact against GitHub. Publication creates a draft, verifies
its assets and tag, publishes it, and rehashes the actual downloads.

Use `--manifest artifact-manifest.json --native-qa native-qa.json --notes notes.md
--signer-archive signer.zip` to prepare; add `--publish` only for an authorized
release. Native QA contains `passed`, `source_commit`, `version`,
`app_archive_sha256`, `verification` (the operator's report path), and
`verification_sha256`. The report distinguishes fresh signed-package checks from
retained runtime/model evidence and its limitations. It is an operator acceptance
record, not an automatic acoustic benchmark. The final signed package must pass
normal launch, relocated bundle/runtime checks and owned-process cleanup; reuse
unchanged broader proof when the source/runtime comparison supports it.

The former `core_acceptance.py`, `packaged_replay.py` and
`acceptance_contract_v2.py` runners are removed with the legacy application.
Their historical behavior and invocation are retained at commit
`7d94c8011a9501bd2357e07c73d61964c55a3886`. Historical reports do not qualify a
new native candidate.

The synthetic recipe and holdout JSON, model metadata, independent fixture
generation, and capture replay helper remain. Native regression tests and
actual-package evidence are described in [the implementation record](rust-port/implementation.md);
[source-removal scope](rust-port/python-source-removal.md) maps retained owners.
Source checks, library capability and synthetic fixtures alone do not establish
acoustic quality or authorize publication. Producer manifests remain candidates
with `public_ready: false` and require signed-package QA. Public download/updater
promotion is separate from GitHub publication; see [download delivery](download-contract.md).
