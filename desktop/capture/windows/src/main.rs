#[cfg(any(windows, test))]
#[cfg_attr(not(windows), allow(dead_code))]
mod output;
#[cfg(any(windows, test))]
#[cfg_attr(not(windows), allow(dead_code))]
mod protocol;
#[cfg(windows)]
mod runtime;
#[cfg(windows)]
mod wasapi;

#[cfg(windows)]
fn main() {
    use std::io::{BufRead, Read};
    if std::env::args()
        .skip(1)
        .any(|value| matches!(value.as_str(), "--describe" | "--check"))
    {
        println!("{}", protocol::description());
        return;
    }
    let (sender, receiver) = std::sync::mpsc::sync_channel(8);
    std::thread::spawn(move || {
        let mut reader = std::io::stdin().lock();
        loop {
            let mut line = Vec::new();
            let result = (&mut reader).take(4097).read_until(b'\n', &mut line);
            let input = match result {
                Ok(0) => runtime::Input::Eof,
                Ok(_) if line.len() > 4096 || line.last() != Some(&b'\n') => runtime::Input::Error(
                    "Capture command exceeded its bound or ended mid-line.".into(),
                ),
                Ok(_) => match serde_json::from_slice(&line) {
                    Ok(command) => runtime::Input::Command(command),
                    Err(error) => {
                        runtime::Input::Error(format!("Invalid capture command: {error}"))
                    }
                },
                Err(error) => {
                    runtime::Input::Error(format!("Capture command input failed: {error}"))
                }
            };
            let terminal = matches!(input, runtime::Input::Eof | runtime::Input::Error(_));
            if sender.send(input).is_err() || terminal {
                return;
            }
        }
    });
    if let Err(error) = runtime::run(receiver) {
        eprintln!("{error}");
        std::process::exit(1);
    }
}

#[cfg(not(windows))]
fn main() {
    eprintln!("speakerdesk-capture requires Windows WASAPI.");
    std::process::exit(1);
}
