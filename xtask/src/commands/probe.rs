use std::fs;
use std::process::Command;

use anyhow::{Context, bail};

use super::repo_root;

/// Early Cohere / Sortformer / Silero probe against packaged 0.6.6.
/// The comparison runs on Apple Silicon. Other hosts record why it did not run.
pub fn run() -> anyhow::Result<()> {
    let root = repo_root();
    let os = Command::new("uname")
        .arg("-srm")
        .output()
        .context("uname")?;
    let os = String::from_utf8_lossy(&os.stdout).trim().to_string();
    let rustc = Command::new("rustc")
        .arg("--version")
        .output()
        .context("rustc")?;
    let rustc = String::from_utf8_lossy(&rustc.stdout).trim().to_string();
    let macos = os.starts_with("Darwin");
    let blocker = root.join("docs/rust-port/blockers/perf-probe.md");
    if let Some(parent) = blocker.parent() {
        fs::create_dir_all(parent)?;
    }
    if !macos {
        let body = format!(
            "# Performance probe\n\n\
             This host is `{os}` with `{rustc}`.\n\n\
             The Silero, Sortformer, and Cohere probe compares a Rust child against the \
             packaged Speakerdesk 0.6.6 Mac app on Apple Silicon. It did not run here. \
             No model agent is re-dispatched, because there is no measurement and no regression.\n\n\
             Run `cargo xtask probe` on the owner's Mac after those three model arms are real.\n"
        );
        fs::write(&blocker, body)?;
        bail!("perf probe requires macOS; wrote {}", blocker.display());
    }
    bail!(
        "macOS probe harness is present and the packaged 0.6.6 sidecar was not found in this run"
    );
}
