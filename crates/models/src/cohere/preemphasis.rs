//! First-order preemphasis from mlx-speech `feature_extraction.py`.
//!
//! The coefficient is the extractor default and the value stored on the
//! loaded provider as `feature_config.preemphasis` in
//! `speakerdesk/asr-source-reuse-profile.json`.

/// `y[n] = x[n] - 0.97 * x[n - 1]`, with `y[0] = x[0]`.
pub const PREEMPHASIS: f32 = 0.97;

#[cfg(test)]
mod tests {
    #[test]
    fn coefficient_matches_the_pinned_feature_config() {
        assert_eq!(super::PREEMPHASIS, 0.97);
        let profile = include_str!("../../../../speakerdesk/asr-source-reuse-profile.json");
        assert!(
            profile.contains("\"preemphasis\": 0.97"),
            "mlx-speech feature_config.preemphasis drifted from 0.97"
        );
    }
}
