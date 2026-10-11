/// The 14 product languages from `transcript.LANGUAGES` / `alignment_text.SUPPORTED_LANGUAGES`.
pub const LANGUAGES: &[(&str, &str)] = &[
    ("en", "English"),
    ("de", "German"),
    ("fr", "French"),
    ("it", "Italian"),
    ("es", "Spanish"),
    ("pt", "Portuguese"),
    ("el", "Greek"),
    ("nl", "Dutch"),
    ("pl", "Polish"),
    ("vi", "Vietnamese"),
    ("zh", "Chinese"),
    ("ar", "Arabic"),
    ("ja", "Japanese"),
    ("ko", "Korean"),
];

pub const CALIBRATED_LANGUAGES: &[&str] = &["en"];
pub const AUTOMATIC: &str = "auto";

pub fn is_supported(code: &str) -> bool {
    LANGUAGES.iter().any(|(id, _)| *id == code)
}

pub fn language_name(code: &str) -> Option<&'static str> {
    LANGUAGES
        .iter()
        .find(|(id, _)| *id == code)
        .map(|(_, name)| *name)
}

#[cfg(test)]
mod tests {
    use super::{LANGUAGES, is_supported, language_name};

    #[test]
    fn fourteen_languages() {
        assert_eq!(LANGUAGES.len(), 14);
        assert!(is_supported("en"));
        assert!(!is_supported("auto"));
        assert_eq!(language_name("ja"), Some("Japanese"));
    }
}
