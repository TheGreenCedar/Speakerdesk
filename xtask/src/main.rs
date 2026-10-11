mod ledger;
mod upstream;

use std::path::{Path, PathBuf};

use anyhow::{Context, bail};
use clap::{Parser, Subcommand};

#[derive(Parser)]
#[command(name = "xtask", about = "Speakerdesk dev tooling")]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    /// Validate xtask/port-ledger.toml against the Python tree.
    Ledger {
        /// Fail on any ledger inconsistency.
        #[arg(long)]
        check: bool,
    },
    /// Fetch the pinned upstream model sources into .cache/upstream and
    /// record their licenses and mx.compile/mx.fast usage inventory.
    FetchUpstream,
}

fn main() -> anyhow::Result<()> {
    let cli = Cli::parse();
    let root = repo_root()?;
    match cli.command {
        Command::Ledger { check } => {
            if !check {
                bail!("usage: cargo xtask ledger --check");
            }
            ledger::check(&root)
        }
        Command::FetchUpstream => upstream::fetch(&root),
    }
}

fn repo_root() -> anyhow::Result<PathBuf> {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .map(Path::to_path_buf)
        .context("xtask must live at <repo>/xtask")
}
