use anyhow::bail;

/// Real-model comparison is separate from component and replay tests.
pub fn run() -> anyhow::Result<()> {
    bail!(
        "Not implemented: native model execution and the packaged comparison runner are unfinished. No baseline search or performance measurement was performed."
    );
}
