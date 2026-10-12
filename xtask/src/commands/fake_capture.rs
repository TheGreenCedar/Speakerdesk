use anyhow::Context;

/// Stdin command names from `desktop/capture/MeetingCapture.swift`.
pub const COMMANDS: &[&str] = &["start", "pause", "resume", "stop"];

/// JSON `type` values the desktop capture helper emits.
pub const EVENTS: &[&str] = &[
    "capture_processing",
    "audio",
    "capture_runtime",
    "clock",
    "format_changed",
    "recording",
    "paused",
    "stopped",
    "error",
    "helper_check",
];

const SAMPLE_EVENT_LINE: &str = r#"{"type":"recording","microphone":true,"system":true}"#;

pub fn run() -> anyhow::Result<()> {
    if COMMANDS.is_empty() || EVENTS.is_empty() {
        return Err(anyhow::anyhow!("fake-capture protocol table is empty"));
    }
    println!("commands: {}", COMMANDS.join(", "));
    println!("events: {}", EVENTS.join(", "));
    check_protocol()
}

/// Parses a sample capture JSON line and requires its event name to be known.
pub fn check_protocol() -> anyhow::Result<()> {
    let value: serde_json::Value =
        serde_json::from_str(SAMPLE_EVENT_LINE).context("parse sample event line")?;
    let name = value
        .get("type")
        .and_then(serde_json::Value::as_str)
        .context("sample event line has no type")?;
    if !EVENTS.contains(&name) {
        return Err(anyhow::anyhow!("unknown capture event {name}"));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::check_protocol;

    #[test]
    fn sample_json_line_names_a_known_event() {
        check_protocol().expect("sample event is in the protocol table");
    }
}
