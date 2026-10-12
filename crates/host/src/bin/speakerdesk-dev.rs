//! Local fixture modes for the Speakerdesk host.
//!
//! This process does not download models. `--fixture people` would build
//! [`speakerdesk_host::router`] with a tiny in-memory people document if that
//! function accepted one. It does not: people catalogs stay inside the route
//! module, so binding a port needs state this binary cannot see.

use std::process::ExitCode;

const HELP: &str = "\
speakerdesk-dev

Local fixture modes for the Speakerdesk host. This program does not download models.

Usage:
  speakerdesk-dev --help
  speakerdesk-dev --fixture <mode>

Modes:
  people
      Build speakerdesk_host::router() with a tiny in-memory people document
      when the public API accepts one. The router builds an empty host, and
      people catalogs are not part of that API, so binding a port needs more
      state than this process can see. This mode prints the fixture name and
      exits.
";

fn main() -> ExitCode {
    match command(std::env::args().skip(1)) {
        Ok(Command::Help) => {
            print!("{HELP}");
            ExitCode::SUCCESS
        }
        Ok(Command::People) => {
            println!("people");
            ExitCode::SUCCESS
        }
        Err(message) => {
            eprintln!("{message}\n\n{HELP}");
            ExitCode::from(2)
        }
    }
}

enum Command {
    Help,
    People,
}

fn command(args: impl Iterator<Item = String>) -> Result<Command, String> {
    let args: Vec<String> = args.collect();
    if args.iter().any(|arg| arg == "--help" || arg == "-h") {
        return Ok(Command::Help);
    }
    if args.is_empty() {
        return Err("missing fixture mode".to_string());
    }

    let mut fixture = None;
    let mut index = 0;
    while index < args.len() {
        let arg = &args[index];
        if let Some(value) = arg.strip_prefix("--fixture=") {
            set_fixture(&mut fixture, value)?;
        } else if arg == "--fixture" {
            let Some(value) = args.get(index + 1) else {
                return Err("missing fixture name after --fixture".to_string());
            };
            if value.starts_with('-') {
                return Err("missing fixture name after --fixture".to_string());
            }
            set_fixture(&mut fixture, value)?;
            index += 1;
        } else {
            return Err(format!("unexpected argument `{arg}`"));
        }
        index += 1;
    }

    match fixture {
        Some("people") => Ok(Command::People),
        Some(other) => Err(format!("unknown fixture mode `{other}`")),
        None => Err("missing fixture mode".to_string()),
    }
}

fn set_fixture<'a>(slot: &mut Option<&'a str>, value: &'a str) -> Result<(), String> {
    if value.is_empty() {
        return Err("missing fixture name after --fixture".to_string());
    }
    if slot.is_some() {
        return Err("fixture mode given more than once".to_string());
    }
    *slot = Some(value);
    Ok(())
}
