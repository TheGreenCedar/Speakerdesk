# Speakerdesk

## Build and test

These mirror `.github/workflows/ci.yml`:

- `cargo test --locked --features speakerdesk-host/fixtures -p speakerdesk-core -p speakerdesk-host -p xtask`
- macOS only: run `bash scripts/fetch_mlx_runtime.sh` once, then add `-p speakerdesk-models -p speakerdesk-echo`. Desktop shell: `cargo test --locked -p speakerdesk`.
- `python3 scripts/check_source.py` and `python3 -m unittest discover -s tests -v` (the Python suite runs Swift capture checks on macOS)
- `node --test tests/*.test.cjs`

## Gotchas

- Linux builds the inference child as a protocol stub. MLX, Core ML and the echo library only build and run on macOS.
- `speakerdesk/` holds the editor frontend and model metadata that Rust embeds. It is not a Python package, and there is no Python at runtime.
- `tests/goldens/`, `crates/*/tests/testdata/` and the model reference JSON are independent references with recorded provenance (see the README beside each one and `tools/oracle/README.md`). Fix the code; regenerating a fixture to turn a test green is never the fix.
- `speakerdesk_core::messages` and `speakerdesk_core::security` change only with an owner decision.

## Docs

- `docs/architecture.md`: read before changing process, IPC, storage, refinement, capture or setup boundaries.
- `docs/product-behavior.md`: update it in the same change when you change a user-visible rule.
- `docs/releasing.md`: build, signing, promotion and website delivery.

## Agent skills

### Issue tracker

Issues are tracked in GitHub Issues for `TheGreenCedar/Speakerdesk` (via the `gh` CLI). See `docs/agents/issue-tracker.md`.

### Triage labels

Default vocabulary: `needs-triage`, `needs-info`, `ready-for-agent`, `ready-for-human`, `wontfix`. See `docs/agents/triage-labels.md`.

### Domain docs

Single-context: `GLOSSARY.md` and `docs/adr/` at the repo root. See `docs/agents/domain.md`.
