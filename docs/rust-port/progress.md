# Port progress

- Wave 0 foundation: Rust 1.99.0 workspace, frozen contracts, ledger (87 test files), ownership check. `cargo test` and clippy are green on Linux. Draft PR #27.
- Wave 0 oracle: in progress on `cursor/rust-oracle-c96f`.
- Wave 1: agents are filling disjoint worktrees.
- Perf probe: Linux host recorded in `docs/rust-port/blockers/perf-probe.md`. No regression to re-dispatch.
