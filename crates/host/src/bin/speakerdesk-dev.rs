//! Local fixture modes for the Speakerdesk host.
//!
//! `--fixture people` serves the embedded editor with a fictional transcript in
//! disposable storage with synthetic silent audio. No models are loaded or downloaded.

use std::io::Write;
use std::process::ExitCode;
use std::time::{SystemTime, UNIX_EPOCH};

use serde_json::json;
use speakerdesk_core::document::Job;
use speakerdesk_host::routes::AppState;

const PEOPLE_JOB_ID: &str = "0123456789abcdef0123456789abcdef";

const HELP: &str = "\
speakerdesk-dev

Local fixture modes for the Speakerdesk host. This program does not download models.

Usage:
  speakerdesk-dev --help
  speakerdesk-dev --fixture <mode>

Modes:
  people
      Open the editor with a fictional transcript in a fresh temporary library.
      Name speakers, edit, save, reopen and export without models.
      Playback uses 18 seconds of synthetic silence, not recorded voices.
      The loopback URL and library path are printed. Ctrl-C stops the server
      and removes its temporary library, including any edits.
";

#[tokio::main]
async fn main() -> ExitCode {
    match command(std::env::args().skip(1)) {
        Ok(Command::Help) => {
            print!("{HELP}");
            ExitCode::SUCCESS
        }
        Ok(Command::People) => match serve_people().await {
            Ok(()) => ExitCode::SUCCESS,
            Err(error) => {
                eprintln!("Could not run people fixture: {error}");
                ExitCode::FAILURE
            }
        },
        Err(message) => {
            eprintln!("{message}\n\n{HELP}");
            ExitCode::from(2)
        }
    }
}

async fn serve_people() -> Result<(), Box<dyn std::error::Error>> {
    let directory = tempfile::Builder::new()
        .prefix("speakerdesk-dev-")
        .tempdir()?;
    let app = AppState::open(directory.path())?;
    let now = SystemTime::now().duration_since(UNIX_EPOCH)?.as_secs_f64();
    let job: Job = serde_json::from_value(json!({
        "id": PEOPLE_JOB_ID,
        "name": "Fictional planning meeting (silent fixture)",
        "kind": "import",
        "status": "ready",
        "language": "en",
        "duration": 18.0,
        "revision": 0,
        "created": now,
        "updated": now,
        "document": {
            "speakers": {"speaker_0": "Speaker 1", "speaker_1": "Speaker 2"},
            "segments": [
                {"id": "intro", "start": 0.0, "end": 6.0, "speaker": "speaker_0",
                 "language": "en", "text": "I'm Mira from the fictional Lantern team."},
                {"id": "reply", "start": 6.0, "end": 12.0, "speaker": "speaker_1",
                 "language": "en", "text": "I'm Rowan. Let's name the speakers and correct these sample words."},
                {"id": "closing", "start": 12.0, "end": 18.0, "speaker": "speaker_0",
                 "language": "en", "text": "Then save, reopen the meeting, and export the transcript."}
            ]
        }
    }))?;
    let media = app.data_dir().join(PEOPLE_JOB_ID);
    std::fs::create_dir(&media)?;
    {
        let mut audio = speakerdesk_core::audio::RecordingWriter::create(media.join("audio.wav"))?;
        audio.write_block(&vec![
            0;
            18 * speakerdesk_core::limits::SAMPLE_RATE as usize * 2
        ])?;
    }
    app.insert_job(job);
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await?;
    println!("SPEAKERDESK_URL=http://{}/", listener.local_addr()?);
    println!("SPEAKERDESK_DATA_DIR={}", app.data_dir().display());
    println!("Fictional transcript with 18 seconds of synthetic silence; models are unavailable.");
    println!("Press Ctrl-C to stop and remove this temporary library.");
    std::io::stdout().flush()?;

    // Drop the server, store and worker before removing their storage.
    let result = {
        let (shutdown, stopped) = tokio::sync::oneshot::channel();
        let server = async move {
            axum::serve(listener, speakerdesk_host::router(app))
                .with_graceful_shutdown(async {
                    let _ = stopped.await;
                })
                .await
        };
        tokio::pin!(server);
        tokio::select! {
            result = &mut server => result,
            signal = tokio::signal::ctrl_c() => {
                signal?;
                let _ = shutdown.send(());
                server.await
            }
        }
    };
    result?;
    directory.close()?;
    Ok(())
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
