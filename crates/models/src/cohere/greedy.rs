//! EOS token pin from mlx-speech `generation/cohere_asr.py`.
//!
//! The autoregressive loop breaks on `decoder.eos_token_id` and then drops
//! that trailing id. The id is the pinned provider state in
//! `speakerdesk/asr-source-reuse-profile.json`.

/// `decoder.eos_token_id` from the pinned Cohere provider state.
pub const EOS_TOKEN_ID: u32 = 3;

#[cfg(test)]
mod tests {
    use super::EOS_TOKEN_ID;

    #[test]
    fn stop_id_matches_the_pinned_decoder_eos() {
        let profile = include_str!("../../../../speakerdesk/asr-source-reuse-profile.json");
        let key = "\"eos_token_id\": ";
        let start = profile.find(key).expect("decoder.eos_token_id") + key.len();
        let digits = profile[start..]
            .chars()
            .take_while(|c| c.is_ascii_digit())
            .collect::<String>();
        let pinned: u32 = digits.parse().expect("eos token id");
        assert_eq!(EOS_TOKEN_ID, pinned);
        assert_eq!(pinned, 3);
    }
}
