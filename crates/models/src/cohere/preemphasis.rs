//! First-order preemphasis from mlx-speech `feature_extraction.py`.
//!
//! The coefficient is the extractor default and the value stored on the
//! loaded provider as `feature_config.preemphasis` in
//! `speakerdesk/asr-source-reuse-profile.json`.

/// `y[n] = x[n] - 0.97 * x[n - 1]`, with `y[0] = x[0]`.
pub const PREEMPHASIS: f32 = 0.97;

/// High-pass filter used before dithered log-mel extraction.
///
/// An empty input stays empty. mlx-speech indexes `waveform[0]` and therefore
/// assumes at least one sample.
pub fn apply_preemphasis(samples: &[f32]) -> Vec<f32> {
    let mut filtered = Vec::with_capacity(samples.len());
    for (index, sample) in samples.iter().copied().enumerate() {
        let value = if index == 0 {
            sample
        } else {
            sample - PREEMPHASIS * samples[index - 1]
        };
        filtered.push(value);
    }
    filtered
}

#[cfg(test)]
mod tests {
    use super::apply_preemphasis;

    #[test]
    fn copies_the_first_sample_and_filters_the_rest() {
        let input = [0.5_f32, -0.25, 0.0, 1.0];
        let output = apply_preemphasis(&input);
        assert_eq!(output.len(), input.len());
        assert_eq!(output[0], input[0]);
        assert_eq!(output[1], input[1] - 0.97 * input[0]);
        assert_eq!(output[2], input[2] - 0.97 * input[1]);
        assert_eq!(output[3], input[3] - 0.97 * input[2]);
    }

    #[test]
    fn leaves_an_empty_or_single_sample_unchanged() {
        assert!(apply_preemphasis(&[]).is_empty());
        assert_eq!(apply_preemphasis(&[0.25]), vec![0.25]);
    }

    #[test]
    fn coefficient_matches_the_pinned_feature_config() {
        let profile = include_str!("../../../../speakerdesk/asr-source-reuse-profile.json");
        assert!(
            profile.contains("\"preemphasis\": 0.97"),
            "mlx-speech feature_config.preemphasis drifted from 0.97"
        );
    }
}
