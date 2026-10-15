use std::path::PathBuf;
use std::process::ExitCode;

use clap::{Parser, Subcommand};

mod commands;

#[derive(Debug, Parser)]
#[command(name = "xtask")]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Debug, Subcommand)]
enum Command {
    /// Check a native app bundle for required resources and forbidden Python artifacts.
    VerifyBundle { path: Option<PathBuf> },
    /// Check that the desktop shell does not start a Python sidecar.
    CheckPythonFree,
    /// Compare Silero, Sortformer, and Cohere with packaged 0.6.6. macOS only.
    Probe,
}

fn main() -> ExitCode {
    let cli = Cli::parse();
    let result = match cli.command {
        Command::VerifyBundle { path } => commands::verify_bundle::run(path),
        Command::CheckPythonFree => commands::python_free::run(),
        Command::Probe => commands::probe::run(),
    };
    match result {
        Ok(()) => ExitCode::SUCCESS,
        Err(error) => {
            eprintln!("{error:#}");
            ExitCode::from(1)
        }
    }
}
