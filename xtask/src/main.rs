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
    /// Fail if agent ownership paths overlap, or if a diff leaves an agent's set.
    Ownership {
        #[arg(long)]
        check: bool,
        #[arg(long)]
        agent: Option<String>,
        #[arg(long)]
        base: Option<String>,
    },
    Parity,
    FakeCapture,
    Acceptance,
    Bench,
    Corpora,
    VerifyBundle {
        path: Option<PathBuf>,
    },
    CheckPythonFree,
    Notices,
    CiChanges,
    VoiceCalibrate,
    E2e {
        kind: Option<String>,
    },
    FetchUpstream,
    /// Compare Silero, Sortformer, and Cohere with packaged 0.6.6. macOS only.
    Probe,
}

fn main() -> ExitCode {
    let cli = Cli::parse();
    let result = match cli.command {
        Command::Ownership { check, agent, base } => commands::ownership::run(check, agent, base),
        Command::FakeCapture => commands::fake_capture::run(),
        Command::Parity => commands::pending::run("parity"),
        Command::Acceptance => commands::pending::run("acceptance"),
        Command::Bench => commands::pending::run("bench"),
        Command::Corpora => commands::pending::run("corpora"),
        Command::VerifyBundle { path } => commands::verify_bundle::run(path),
        Command::CheckPythonFree => commands::python_free::run(),
        Command::Notices => commands::pending::run("notices"),
        Command::CiChanges => commands::ci_changes::run(),
        Command::VoiceCalibrate => commands::pending::run("voice-calibrate"),
        Command::E2e { kind } => {
            let _ = kind;
            commands::pending::run("e2e")
        }
        Command::FetchUpstream => commands::pending::run("fetch-upstream"),
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
