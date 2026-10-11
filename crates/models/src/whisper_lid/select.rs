//! Language-token softmax selection.
//!
//! Mirrors the pinned `detect_language` excerpt: non-language logits are
//! excluded, `argmax` breaks ties toward the lowest token id, and the top
//! margin is the best probability minus the next one. Equal probabilities keep
//! the earlier code, matching `LanguagePolicy.decide`'s stable sort.

use thiserror::Error;

use super::tokens::LanguageTokens;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Error)]
pub enum LanguageSelectionError {
    #[error("language token id is outside the logit vector")]
    LogitsTooShort,
    #[error("language scores must be finite")]
    NonFinite,
    #[error("language selection needs at least one score")]
    Empty,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TopMargin {
    /// First maximum, in input order.
    pub index: usize,
    pub probability: f64,
    pub margin: f64,
}

#[derive(Debug, Clone, PartialEq)]
pub struct LanguageSelection {
    /// `argmax` over language-token logits. Ties use the lowest token id.
    pub token_id: u32,
    pub code: String,
    pub probability: f64,
    /// Best probability minus the second best. A single score has margin zero.
    pub margin: f64,
    /// Softmax in `LanguageTokens::codes` order. Non-language logits contribute nothing.
    pub probabilities: Vec<f64>,
}

/// Index into `token_ids` of the maximum finite logit.
///
/// Ties resolve to the lowest token id, which is what `argmax` returns after
/// every other vocab position has been masked to negative infinity.
pub fn argmax_language_token(
    logits: &[f32],
    token_ids: &[u32],
) -> Result<usize, LanguageSelectionError> {
    if token_ids.is_empty() {
        return Err(LanguageSelectionError::Empty);
    }
    let mut best = 0usize;
    let mut best_logit = language_logit(logits, token_ids[0])?;
    let mut best_id = token_ids[0];
    for (index, &token_id) in token_ids.iter().enumerate().skip(1) {
        let logit = language_logit(logits, token_id)?;
        let prefer = match logit.total_cmp(&best_logit) {
            std::cmp::Ordering::Greater => true,
            std::cmp::Ordering::Equal => token_id < best_id,
            std::cmp::Ordering::Less => false,
        };
        if prefer {
            best = index;
            best_logit = logit;
            best_id = token_id;
        }
    }
    Ok(best)
}

/// Softmax over `token_ids` only. Other logit positions are ignored, including
/// a larger non-language score.
pub fn softmax_language_tokens(
    logits: &[f32],
    token_ids: &[u32],
) -> Result<Vec<f64>, LanguageSelectionError> {
    if token_ids.is_empty() {
        return Err(LanguageSelectionError::Empty);
    }
    let mut values = Vec::with_capacity(token_ids.len());
    for &token_id in token_ids {
        values.push(f64::from(language_logit(logits, token_id)?));
    }
    let max = values.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    if !max.is_finite() {
        return Err(LanguageSelectionError::NonFinite);
    }
    let mut sum = 0.0;
    for value in &mut values {
        *value = (*value - max).exp();
        sum += *value;
    }
    if !sum.is_finite() {
        return Err(LanguageSelectionError::NonFinite);
    }
    for value in &mut values {
        *value /= sum;
    }
    Ok(values)
}

/// Highest probability minus the next highest.
///
/// On a tie the earlier entry wins, matching Python's stable descending sort.
/// One probability yields margin `0`.
pub fn top_margin(probabilities: &[f64]) -> Result<TopMargin, LanguageSelectionError> {
    if probabilities.is_empty() {
        return Err(LanguageSelectionError::Empty);
    }
    if probabilities
        .iter()
        .any(|probability| !probability.is_finite())
    {
        return Err(LanguageSelectionError::NonFinite);
    }
    let mut best = 0usize;
    let mut second = None;
    for (index, &probability) in probabilities.iter().enumerate().skip(1) {
        if probability > probabilities[best] {
            second = Some(probabilities[best]);
            best = index;
        } else {
            second = Some(second.map_or(probability, |current| current.max(probability)));
        }
    }
    let margin = match second {
        Some(second) => probabilities[best] - second,
        None => 0.0,
    };
    Ok(TopMargin {
        index: best,
        probability: probabilities[best],
        margin,
    })
}

/// Argmax language token plus the softmax top margin.
pub fn select_language(
    logits: &[f32],
    tokens: &LanguageTokens,
) -> Result<LanguageSelection, LanguageSelectionError> {
    let probabilities = softmax_language_tokens(logits, &tokens.token_ids)?;
    let argmax = argmax_language_token(logits, &tokens.token_ids)?;
    let margin = top_margin(&probabilities)?;
    Ok(LanguageSelection {
        token_id: tokens.token_ids[argmax],
        code: tokens.codes[argmax].clone(),
        probability: probabilities[argmax],
        margin: margin.margin,
        probabilities,
    })
}

fn language_logit(logits: &[f32], token_id: u32) -> Result<f32, LanguageSelectionError> {
    let logit = logits
        .get(usize::try_from(token_id).map_err(|_| LanguageSelectionError::LogitsTooShort)?)
        .copied()
        .ok_or(LanguageSelectionError::LogitsTooShort)?;
    if !logit.is_finite() {
        return Err(LanguageSelectionError::NonFinite);
    }
    Ok(logit)
}

#[cfg(test)]
mod tests {
    use super::{argmax_language_token, select_language, softmax_language_tokens, top_margin};
    use crate::whisper_lid::tokens::parse_language_tokens;

    fn tokens() -> crate::whisper_lid::tokens::LanguageTokens {
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
        let mut pairs = vec![("<|startoftranscript|>".to_string(), 199)];
        for (offset, code) in codes.into_iter().enumerate() {
            pairs.push((format!("<|{code}|>"), 200 + i64::try_from(offset).unwrap()));
        }
        let view: Vec<(&str, i64)> = pairs
            .iter()
            .map(|(token, id)| (token.as_str(), *id))
            .collect();
        parse_language_tokens(&view, 400).unwrap()
    }

    #[test]
    fn non_language_logit_cannot_win_the_ninety_nine_way_softmax() {
        let tokens = tokens();
        let mut logits = vec![0.0f32; 400];
        logits[123] = 1000.0;
        let french = tokens.codes.iter().position(|code| code == "fr").unwrap();
        logits[usize::try_from(tokens.token_ids[french]).unwrap()] = 12.0;

        let selected = select_language(&logits, &tokens).unwrap();
        assert_eq!(selected.code, "fr");
        assert_eq!(selected.token_id, tokens.token_ids[french]);
        assert_eq!(selected.probabilities.len(), 99);
        let sum: f64 = selected.probabilities.iter().sum();
        assert!((sum - 1.0).abs() < 1e-5);

        let expected = 1.0 / (1.0 + 98.0 * (-12.0f64).exp());
        assert!((selected.probability - expected).abs() < 1e-6);
        let second = (-12.0f64).exp() / (1.0 + 98.0 * (-12.0f64).exp());
        assert!((selected.margin - (expected - second)).abs() < 1e-6);
        assert!(
            selected
                .probabilities
                .iter()
                .all(|probability| *probability > 0.0)
        );
    }

    #[test]
    fn argmax_tie_uses_lowest_token_id_and_margin_is_zero() {
        let mut logits = [0.0f32; 40];
        logits[30] = 5.0;
        logits[10] = 5.0;
        logits[20] = 1.0;
        logits[3] = 1000.0;
        let token_ids = [30u32, 20, 10];
        assert_eq!(argmax_language_token(&logits, &token_ids).unwrap(), 2);
        let probabilities = softmax_language_tokens(&logits, &token_ids).unwrap();
        assert!((probabilities.iter().sum::<f64>() - 1.0).abs() < 1e-6);
        assert!((probabilities[0] - probabilities[2]).abs() < 1e-12);
        assert!(probabilities[0] > probabilities[1]);
        let margin = top_margin(&probabilities).unwrap();
        assert_eq!(margin.index, 0);
        assert!(margin.margin.abs() < 1e-12);
    }

    #[test]
    fn top_margin_keeps_the_earlier_code_on_equal_probabilities() {
        let tied = top_margin(&[0.4, 0.4, 0.2]).unwrap();
        assert_eq!(tied.index, 0);
        assert!((tied.probability - 0.4).abs() < 1e-12);
        assert!(tied.margin.abs() < 1e-12);

        let ranked = top_margin(&[0.2, 0.5, 0.3]).unwrap();
        assert_eq!(ranked.index, 1);
        assert!((ranked.margin - 0.2).abs() < 1e-12);

        let only = top_margin(&[1.0]).unwrap();
        assert_eq!(only.index, 0);
        assert!(only.margin.abs() < 1e-12);
    }

    #[test]
    fn selection_rejects_short_or_non_finite_logits() {
        let tokens = tokens();
        assert!(select_language(&[0.0; 10], &tokens).is_err());
        let mut logits = vec![0.0f32; 400];
        logits[200] = f32::NAN;
        assert!(softmax_language_tokens(&logits, &tokens.token_ids).is_err());
        assert!(top_margin(&[0.5, f64::INFINITY]).is_err());
        assert!(top_margin(&[]).is_err());
        assert!(argmax_language_token(&[], &[]).is_err());
    }
}
