// Native runtime initialization proof, not acoustic model acceptance.
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let path = std::env::args_os().nth(1).map(std::path::PathBuf::from);
    speakerdesk_models::metallib::prepare(path.as_deref())?;
    println!("Metal initialization and GPU matrix evaluation passed.");
    Ok(())
}
