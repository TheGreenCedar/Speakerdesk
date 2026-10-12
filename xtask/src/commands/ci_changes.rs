pub const NOTE: &str = "change detection is the Rust xtask, not ci_cache.py";

pub fn run() -> anyhow::Result<()> {
    println!("{NOTE}");
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::NOTE;

    #[test]
    fn note_says_change_detection_is_the_rust_xtask() {
        assert!(
            NOTE.contains("Rust xtask"),
            "note should name the Rust xtask, got {NOTE:?}"
        );
        assert!(
            NOTE.contains("ci_cache.py"),
            "note should name ci_cache.py, got {NOTE:?}"
        );
    }
}
