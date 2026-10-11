use std::fs;
use std::path::Path;
use std::process::Command;

use anyhow::{Context, bail};
use serde::Deserialize;

use super::repo_root;

#[derive(Debug, Deserialize)]
struct Ownership {
    agent: Vec<Agent>,
}

#[derive(Debug, Deserialize)]
struct Agent {
    id: String,
    paths: Vec<String>,
}

pub fn run(check: bool, agent: Option<String>, base: Option<String>) -> anyhow::Result<()> {
    if !check {
        bail!("pass --check");
    }
    let root = repo_root();
    let path = root.join("docs/rust-port/ownership.toml");
    let text = fs::read_to_string(&path).with_context(|| format!("read {}", path.display()))?;
    let ownership: Ownership = toml::from_str(&text).context("parse ownership")?;
    check_overlaps(&ownership)?;
    if let Some(agent_id) = agent {
        let base = base.context("--base is required with --agent")?;
        check_diff(&root, &ownership, &agent_id, &base)?;
    }
    println!("ownership ok: {} agents", ownership.agent.len());
    Ok(())
}

fn check_overlaps(ownership: &Ownership) -> anyhow::Result<()> {
    for (left_index, left) in ownership.agent.iter().enumerate() {
        for right in ownership.agent.iter().skip(left_index + 1) {
            for left_path in &left.paths {
                for right_path in &right.paths {
                    if patterns_overlap(left_path, right_path) {
                        bail!(
                            "ownership overlap: {} ({left_path}) and {} ({right_path})",
                            left.id,
                            right.id
                        );
                    }
                }
            }
        }
    }
    Ok(())
}

fn check_diff(
    root: &Path,
    ownership: &Ownership,
    agent_id: &str,
    base: &str,
) -> anyhow::Result<()> {
    let agent = ownership
        .agent
        .iter()
        .find(|agent| agent.id == agent_id)
        .with_context(|| format!("unknown agent {agent_id}"))?;
    let output = Command::new("git")
        .arg("diff")
        .arg("--name-only")
        .arg(format!("{base}...HEAD"))
        .current_dir(root)
        .output()
        .context("git diff")?;
    if !output.status.success() {
        bail!("{}", String::from_utf8_lossy(&output.stderr));
    }
    let text = String::from_utf8(output.stdout).context("diff output")?;
    for line in text.lines().filter(|line| !line.is_empty()) {
        if !agent
            .paths
            .iter()
            .any(|pattern| path_matches(pattern, line))
        {
            bail!("{agent_id} does not own {line}");
        }
    }
    Ok(())
}

fn patterns_overlap(left: &str, right: &str) -> bool {
    if left == right {
        return true;
    }
    match (tree_prefix(left), tree_prefix(right)) {
        (Some(left_prefix), Some(right_prefix)) => {
            left_prefix.starts_with(right_prefix.as_str())
                || right_prefix.starts_with(left_prefix.as_str())
        }
        (Some(prefix), None) => right.starts_with(prefix.as_str()),
        (None, Some(prefix)) => left.starts_with(prefix.as_str()),
        (None, None) => false,
    }
}

fn path_matches(pattern: &str, path: &str) -> bool {
    if let Some(prefix) = tree_prefix(pattern) {
        path.starts_with(prefix.as_str())
    } else {
        path == pattern
    }
}

fn tree_prefix(pattern: &str) -> Option<String> {
    pattern
        .strip_suffix("/**")
        .map(|prefix| format!("{prefix}/"))
}

#[cfg(test)]
mod tests {
    use super::{path_matches, patterns_overlap, run};

    #[test]
    fn trees_overlap_when_one_contains_the_other() {
        assert!(patterns_overlap(
            "crates/core/src/**",
            "crates/core/src/edit.rs"
        ));
        assert!(!patterns_overlap(
            "crates/core/src/edit.rs",
            "crates/core/src/review.rs"
        ));
        assert!(path_matches(
            "crates/models/src/silero/**",
            "crates/models/src/silero/mod.rs"
        ));
    }

    #[test]
    fn committed_ownership_does_not_overlap() {
        run(true, None, None).unwrap();
    }
}
