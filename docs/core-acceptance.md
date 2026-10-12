# Release acceptance status

Native release promotion automation is unsupported. `scripts/promote_release.py`
refuses preparation before network access because the retired acoustic verifier
accepted only Python `speakerdesk-runtime` bundles. Its signer provenance,
artifact inventory, updater key and version checks remain independently tested.
No native promotion framework or release-readiness claim replaces that gate.

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
acoustic quality or authorize publication. Candidate manifests keep
`public_ready: false` and explicitly record unsupported native promotion.
