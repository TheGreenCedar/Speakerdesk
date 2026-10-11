use anyhow::bail;

pub fn run(name: &str) -> anyhow::Result<()> {
    bail!("{name} is registered and not implemented in this wave")
}
