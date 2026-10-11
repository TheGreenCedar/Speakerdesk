use std::collections::BTreeSet;
use std::fs;
use std::path::Path;

use anyhow::{Context, bail};
use serde::Deserialize;

use super::repo_root;

#[derive(Debug, Deserialize)]
struct Ledger {
    #[serde(default)]
    row: Vec<Row>,
}

#[derive(Debug, Deserialize)]
struct Row {
    python: String,
    disposition: String,
    status: String,
    #[serde(default)]
    d_ref: String,
    #[serde(default)]
    rust: String,
}

pub fn run(check: bool) -> anyhow::Result<()> {
    if !check {
        bail!("pass --check");
    }
    let root = repo_root();
    let ledger_path = root.join("xtask/port-ledger.toml");
    let text = fs::read_to_string(&ledger_path)
        .with_context(|| format!("read {}", ledger_path.display()))?;
    let ledger: Ledger = toml::from_str(&text).context("parse port ledger")?;
    let mut listed = BTreeSet::new();
    for row in &ledger.row {
        if !listed.insert(row.python.clone()) {
            bail!("duplicate ledger row {}", row.python);
        }
        if row.disposition == "drop" && !row.d_ref.starts_with('D') {
            bail!("{} is dropped without a D-reference", row.python);
        }
        if row.status == "ported" {
            let rust_path = root.join(&row.rust);
            if row.rust.is_empty() || !rust_path.exists() {
                bail!("{} is ported but {} is missing", row.python, row.rust);
            }
        }
        if row.status == "dropped" && row.disposition != "drop" {
            bail!(
                "{} is marked dropped but disposition is {}",
                row.python,
                row.disposition
            );
        }
    }
    let mut missing = Vec::new();
    for entry in fs::read_dir(root.join("tests")).context("read tests/")? {
        let entry = entry?;
        let name = entry.file_name();
        let name = name.to_string_lossy();
        if name.starts_with("test_") && name.ends_with(".py") {
            let relative = format!("tests/{name}");
            if !listed.contains(&relative) {
                missing.push(relative);
            }
        }
    }
    if !missing.is_empty() {
        missing.sort();
        bail!(
            "Python test files missing from the ledger:\n{}",
            missing.join("\n")
        );
    }
    let test_rows = ledger
        .row
        .iter()
        .filter(|row| row.python.starts_with("tests/test_"))
        .count();
    println!("ledger ok: {test_rows} python test files");
    let _ = Path::new(".");
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::run;

    #[test]
    fn check_passes_for_the_committed_ledger() {
        run(true).unwrap();
    }
}
