//! Pure checks for Whisper `added_tokens.json` language ids.
//!
//! Mirrors `language_tokens` in `speakerdesk/language_detection.py`: 99
//! `<|xx|>` / `<|xxx|>` codes, `en` and `fr` present, ids a contiguous
//! in-vocab block, and `<|startoftranscript|>` exactly one below that block.

use thiserror::Error;

pub const LANGUAGE_COUNT: usize = 99;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LanguageTokens {
    /// Capability guard from the Python adapter. Always `en`.
    pub language: &'static str,
    pub language_token: u32,
    pub sot: i64,
    /// Language codes in `added_tokens` insertion order.
    pub codes: Vec<String>,
    /// Token ids parallel to `codes`.
    pub token_ids: Vec<u32>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Error)]
#[error("Incompatible multilingual language-token metadata.")]
pub struct LanguageTokenError;

/// Accept the pinned multilingual language-token map.
///
/// `tokens` is the flattened `added_tokens.json` object in file order.
/// Non-language special tokens are ignored. A repeated language code keeps
/// its first position and the later id, matching Python dict update.
pub fn parse_language_tokens(
    tokens: &[(&str, i64)],
    vocab_size: u32,
) -> Result<LanguageTokens, LanguageTokenError> {
    let mut codes = Vec::new();
    let mut ids = Vec::new();
    let mut sot = None;
    for &(token, value) in tokens {
        if token == "<|startoftranscript|>" {
            sot = Some(value);
            continue;
        }
        let Some(code) = language_code(token) else {
            continue;
        };
        if let Some(position) = codes.iter().position(|existing| existing == code) {
            ids[position] = value;
        } else {
            codes.push(code.to_string());
            ids.push(value);
        }
    }

    if codes.len() != LANGUAGE_COUNT
        || !codes.iter().any(|code| code == "en")
        || !codes.iter().any(|code| code == "fr")
    {
        return Err(LanguageTokenError);
    }

    let mut token_ids = Vec::with_capacity(LANGUAGE_COUNT);
    for value in ids {
        token_ids.push(vocab_id(value, vocab_size)?);
    }

    let mut ordered = token_ids.clone();
    ordered.sort_unstable();
    let min = ordered[0];
    for (offset, id) in ordered.iter().enumerate() {
        let Some(expected) =
            min.checked_add(u32::try_from(offset).map_err(|_| LanguageTokenError)?)
        else {
            return Err(LanguageTokenError);
        };
        if *id != expected {
            return Err(LanguageTokenError);
        }
    }

    let Some(sot) = sot else {
        return Err(LanguageTokenError);
    };
    if sot != i64::from(min) - 1 {
        return Err(LanguageTokenError);
    }

    let language_token = token_ids[codes
        .iter()
        .position(|code| code == "en")
        .ok_or(LanguageTokenError)?];
    Ok(LanguageTokens {
        language: "en",
        language_token,
        sot,
        codes,
        token_ids,
    })
}

fn language_code(token: &str) -> Option<&str> {
    let code = token.strip_prefix("<|")?.strip_suffix("|>")?;
    if (2..=3).contains(&code.len()) && code.bytes().all(|byte| byte.is_ascii_lowercase()) {
        Some(code)
    } else {
        None
    }
}

fn vocab_id(value: i64, vocab_size: u32) -> Result<u32, LanguageTokenError> {
    let id = u32::try_from(value).map_err(|_| LanguageTokenError)?;
    if id >= vocab_size {
        return Err(LanguageTokenError);
    }
    Ok(id)
}

#[cfg(test)]
mod tests {
    use super::{LanguageTokenError, parse_language_tokens};

    fn synthetic(start_id: i64, sot: i64) -> Vec<(String, i64)> {
        let mut codes = vec!["en".to_string(), "fr".to_string()];
        for first in b'a'..=b'z' {
            for second in b'a'..=b'z' {
                let code = format!("{}{}", first as char, second as char);
                if code == "en" || code == "fr" {
                    continue;
                }
                codes.push(code);
                if codes.len() == 99 {
                    break;
                }
            }
            if codes.len() == 99 {
                break;
            }
        }
        let mut pairs = vec![
            ("<|startoftranscript|>".to_string(), sot),
            ("<|transcribe|>".to_string(), 1),
            ("<|0.00|>".to_string(), 2),
            ("<|EN|>".to_string(), 3),
            ("<|e|>".to_string(), 4),
        ];
        for (offset, code) in codes.into_iter().enumerate() {
            pairs.push((
                format!("<|{code}|>"),
                start_id + i64::try_from(offset).unwrap(),
            ));
        }
        pairs
    }

    fn refs(pairs: &[(String, i64)]) -> Vec<(&str, i64)> {
        pairs
            .iter()
            .map(|(token, id)| (token.as_str(), *id))
            .collect()
    }

    #[test]
    fn accepts_contiguous_language_block() {
        let pairs = synthetic(200, 199);
        let tokens = parse_language_tokens(&refs(&pairs), 51865).unwrap();
        assert_eq!(tokens.codes.len(), 99);
        assert_eq!(tokens.language, "en");
        assert_eq!(tokens.language_token, 200);
        assert_eq!(tokens.sot, 199);
        assert_eq!(tokens.codes[0], "en");
        assert_eq!(tokens.codes[1], "fr");
        assert_eq!(tokens.token_ids[1], 201);
        assert!(tokens.codes.iter().all(|code| code != "transcribe"));
    }

    #[test]
    fn rejects_incomplete_or_shifted_metadata() {
        let vocab = 51865u32;
        let mut missing = synthetic(200, 199);
        missing.pop();
        assert_eq!(
            parse_language_tokens(&refs(&missing), vocab).unwrap_err(),
            LanguageTokenError
        );

        let mut no_french = synthetic(200, 199);
        let french = no_french
            .iter_mut()
            .find(|pair| pair.0 == "<|fr|>")
            .unwrap();
        french.0 = "<|zz|>".to_string();
        assert!(parse_language_tokens(&refs(&no_french), vocab).is_err());

        let mut gap = synthetic(200, 199);
        gap.last_mut().unwrap().1 += 10;
        assert!(parse_language_tokens(&refs(&gap), vocab).is_err());

        let mut duplicate = synthetic(200, 199);
        duplicate.last_mut().unwrap().1 = 200;
        assert!(parse_language_tokens(&refs(&duplicate), vocab).is_err());

        let mut shifted_sot = synthetic(200, 198);
        assert!(parse_language_tokens(&refs(&shifted_sot), vocab).is_err());
        shifted_sot.retain(|pair| pair.0 != "<|startoftranscript|>");
        assert!(parse_language_tokens(&refs(&shifted_sot), vocab).is_err());

        let out_of_vocab = synthetic(200, 199);
        assert!(parse_language_tokens(&refs(&out_of_vocab), 200).is_err());

        let mut negative = synthetic(200, 199);
        negative.last_mut().unwrap().1 = -1;
        assert!(parse_language_tokens(&refs(&negative), vocab).is_err());
    }

    #[test]
    fn repeated_code_keeps_first_position_and_later_id() {
        let mut pairs = synthetic(200, 199);
        pairs.push(("<|en|>".to_string(), 201));
        pairs.push(("<|fr|>".to_string(), 200));
        let tokens = parse_language_tokens(&refs(&pairs), 51865).unwrap();
        assert_eq!(tokens.codes[0], "en");
        assert_eq!(tokens.codes[1], "fr");
        assert_eq!(tokens.token_ids[0], 201);
        assert_eq!(tokens.token_ids[1], 200);
        assert_eq!(tokens.language_token, 201);
        assert_eq!(tokens.codes.iter().filter(|code| *code == "en").count(), 1);
    }

    #[test]
    fn pinned_added_tokens_json_has_ninety_nine_languages() {
        let text = include_str!("../../../../tests/metadata/whisper-tiny/added_tokens.json");
        let pairs = added_token_pairs(text);
        let view = refs(&pairs);
        let tokens = parse_language_tokens(&view, 51_865).unwrap();
        assert_eq!(tokens.codes.len(), 99);
        assert!(tokens.codes.iter().any(|code| code == "ru"));
        assert_eq!(tokens.sot, 50_258);
        assert_eq!(tokens.language_token, 50_259);
        let french = tokens.codes.iter().position(|code| code == "fr").unwrap();
        assert_eq!(tokens.token_ids[french], 50_265);
        let mut ordered = tokens.token_ids.clone();
        ordered.sort_unstable();
        assert_eq!(ordered.first().copied(), Some(50_259));
        assert_eq!(ordered.last().copied(), Some(50_357));
    }

    fn added_token_pairs(text: &str) -> Vec<(String, i64)> {
        let bytes = text.as_bytes();
        let mut pairs = Vec::new();
        let mut index = 0;
        while index < bytes.len() {
            if bytes[index] != b'"' {
                index += 1;
                continue;
            }
            let start = index + 1;
            let mut end = start;
            while end < bytes.len() && bytes[end] != b'"' {
                end += 1;
            }
            if end >= bytes.len() {
                break;
            }
            let key = text[start..end].to_string();
            end += 1;
            while end < bytes.len() && bytes[end].is_ascii_whitespace() {
                end += 1;
            }
            if end >= bytes.len() || bytes[end] != b':' {
                index = end;
                continue;
            }
            end += 1;
            while end < bytes.len() && bytes[end].is_ascii_whitespace() {
                end += 1;
            }
            let number_start = end;
            if end < bytes.len() && bytes[end] == b'-' {
                end += 1;
            }
            let digits = end;
            while end < bytes.len() && bytes[end].is_ascii_digit() {
                end += 1;
            }
            if end > digits
                && let Ok(value) = text[number_start..end].parse::<i64>()
            {
                pairs.push((key, value));
            }
            index = end;
        }
        pairs
    }
}
