//! Native installation conversion, exercised only on an explicit disposable directory.
#[cfg(target_os = "macos")]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut args = std::env::args_os().skip(1);
    let directory = std::path::PathBuf::from(args.next().ok_or("model directory required")?);
    let destination = std::path::PathBuf::from(args.next().ok_or("staging file required")?);
    if args.next().is_some() || destination.exists() {
        return Err("expected model directory and unused staging file".into());
    }
    speakerdesk_models::ctc::prepare::convert(&directory, &destination)?;
    Ok(())
}
#[cfg(not(target_os = "macos"))]
fn main() {
    eprintln!("This model backend requires macOS.");
    std::process::exit(1);
}
