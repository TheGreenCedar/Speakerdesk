//! Greedy stop from mlx-speech `generation/cohere_asr.py`.
//!
//! The autoregressive loop breaks on `decoder.eos_token_id` and then drops
//! that trailing id. The id is the pinned provider state in
//! `speakerdesk/asr-source-reuse-profile.json`.

/// `decoder.eos_token_id` from the pinned Cohere provider state.
pub const EOS_TOKEN_ID: u32 = 3;

/// Ids kept once greedy decode has emitted `stop_id`.
///
/// The stop id is excluded, matching the trailing-EOS strip. Later ids are
/// excluded because the loop does not append past the stop.
pub fn greedy_stop(token_ids: &[u32], stop_id: u32) -> &[u32] {
    match token_ids.iter().position(|&id| id == stop_id) {
        Some(index) => &token_ids[..index],
        None => token_ids,
    }
}

#[cfg(test)]
mod tests {
    use super::{EOS_TOKEN_ID, greedy_stop};

    #[test]
    fn drops_the_stop_id_and_anything_after_it() {
        assert_eq!(greedy_stop(&[4, 9, EOS_TOKEN_ID, 8], EOS_TOKEN_ID), &[4, 9]);
        assert_eq!(greedy_stop(&[EOS_TOKEN_ID], EOS_TOKEN_ID), &[] as &[u32]);
        assert_eq!(greedy_stop(&[4, 9], EOS_TOKEN_ID), &[4, 9]);
        assert_eq!(greedy_stop(&[], EOS_TOKEN_ID), &[] as &[u32]);
    }

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
