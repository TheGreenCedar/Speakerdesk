# Speakerdesk

## Build and test

These mirror `.github/workflows/ci.yml`:

- `cargo test --locked --features speakerdesk-host/fixtures -p speakerdesk-core -p speakerdesk-host -p xtask`
- Windows: add `-p speakerdesk-models -p speakerdesk-echo`; also run `cargo test --locked --manifest-path desktop/capture/windows/Cargo.toml`. These checks need no devices or downloaded models; native hardware checks are opt-in.
- macOS only: run `bash scripts/fetch_mlx_runtime.sh` once, then add `-p speakerdesk-models -p speakerdesk-echo`. Desktop shell: `cargo test --locked -p speakerdesk`.
- `python3 scripts/check_source.py` and `python3 -m unittest discover -s tests -v` (the Python suite runs Swift capture checks on macOS)
- `node --test tests/*.test.cjs`

## Gotchas

- Linux builds the inference child as a protocol stub. MLX and Core ML only build and run on macOS; native echo also builds on Windows. Windows inference remains under qualification (see `experiments/windows/README.md`).
- `speakerdesk/` holds the editor frontend and model metadata that Rust embeds. It is not a Python package, and there is no Python at runtime.
- `tests/goldens/`, `crates/*/tests/testdata/` and the model reference JSON are independent references with recorded provenance (see the README beside each one and `tools/oracle/README.md`). Fix the code; regenerating a fixture to turn a test green is never the fix.
- `speakerdesk_core::messages` and `speakerdesk_core::security` change only with an owner decision.

## Docs

- `docs/architecture.md`: read before changing process, IPC, storage, refinement, capture or setup boundaries.
- `docs/product-behavior.md`: read before designing user-facing changes; update it in the same change when you change a user-visible rule.
- `docs/releasing.md`: build, signing, promotion and website delivery.
- `docs/ci-build-reuse.md`: read before changing CI caches or workflow trust boundaries.
- `docs/architecture.md#inference-performance`: read before changing inference scheduling, placement or performance.

Derive user-facing acceptance from the complete interaction as it evolves. Include recording progress, actionable repair, continuation after an accepted repair, and save/reopen persistence where the flow reaches them. Revisit acceptance when the interaction changes; resolving the reported symptom alone does not establish that the flow works.

Separate background work, model uncertainty and failures requiring user action. Show a repairable problem beside the affected passage, explain its consequence and provide the repair there. Continue dependent work after a successful repair. Ordinary uncertainty must not become mandatory review; preserve explicit voice-enrollment consent and protected edits.

## Agent skills

### Issue tracker

Issues are tracked in GitHub Issues for `TheGreenCedar/Speakerdesk` (via the `gh` CLI). See `docs/agents/issue-tracker.md`.

### Triage labels

Default vocabulary: `needs-triage`, `needs-info`, `ready-for-agent`, `ready-for-human`, `wontfix`. See `docs/agents/triage-labels.md`.

### Domain docs

Single-context: `GLOSSARY.md` and `docs/adr/` at the repo root. See `docs/agents/domain.md`.
