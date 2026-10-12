# Speakerdesk

## Build and test

The Rust workspace uses toolchain **1.99.0** (`rust-toolchain.toml`), edition 2024.

- `cargo test -p speakerdesk-core -p speakerdesk-host -p xtask`
- `cargo xtask ownership --check`

Linux builds the inference child as a protocol stub. MLX and the echo library compile on macOS.

## Agent skills

### Issue tracker

Issues are tracked in GitHub Issues for `TheGreenCedar/Speakerdesk` (via the `gh` CLI). See `docs/agents/issue-tracker.md`.

### Triage labels

Default vocabulary: `needs-triage`, `needs-info`, `ready-for-agent`, `ready-for-human`, `wontfix`. See `docs/agents/triage-labels.md`.

### Domain docs

Single-context: `GLOSSARY.md` and `docs/adr/` at the repo root. See `docs/agents/domain.md`.
