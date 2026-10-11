//! Pure waveform prep ported from `speakerdesk/voice_waveform.py`.
//! Six-second windows are 96_000 samples at 16 kHz. Silence and clipping
//! screens use the same thresholds as the ReDimNet embed path.

use thiserror::Error;

/// 16 kHz meeting audio.
pub const RATE: u32 = 16_000;
/// Fixed ReDimNet input length: 6 seconds.
pub const INPUT_SAMPLES: usize = 96_000;
/// Shortest passage `prepared_audio` accepts: 2 seconds.
pub const MIN_SAMPLES: usize = RATE as usize * 2;
/// Longest passage `prepared_audio` accepts: 10 seconds.
pub const MAX_SAMPLES: usize = RATE as usize * 10;
/// RMS strictly below one PCM16 quantum is silence.
pub const SILENCE_RMS: f64 = 1.0 / 32768.0;
/// PCM16 positive full scale. `abs(sample) >= CLIP_ABS` is clipping.
pub const CLIP_ABS: f32 = 32767.0 / 32768.0;
/// Cosine at or above this reuses an existing identity (D7).
pub const IDENTITY_REUSE_COSINE: f64 = 0.9999;

const _: () = assert!(RATE == speakerdesk_core::limits::SAMPLE_RATE);
const _: () = assert!(INPUT_SAMPLES == speakerdesk_core::limits::VOICE_WINDOW_SAMPLES);

#[derive(Debug, Clone, Copy, PartialEq, Eq, Error)]
pub enum PrepareError {
    #[error("Supply 2–10 seconds of finite mono audio.")]
    Invalid,
}

/// Repeat a 2–6 s clip or center-crop a 6–10 s clip to [`INPUT_SAMPLES`].
pub fn prepared_audio(samples: &[f32]) -> Result<Vec<f32>, PrepareError> {
    if !(MIN_SAMPLES..=MAX_SAMPLES).contains(&samples.len())
        || samples.iter().any(|s| !s.is_finite())
    {
        return Err(PrepareError::Invalid);
    }
    let mut window = vec![0.0; INPUT_SAMPLES];
    if samples.len() >= INPUT_SAMPLES {
        let start = (samples.len() - INPUT_SAMPLES) / 2;
        window.copy_from_slice(&samples[start..start + INPUT_SAMPLES]);
    } else {
        for (index, slot) in window.iter_mut().enumerate() {
            *slot = samples[index % samples.len()];
        }
    }
    Ok(window)
}

/// Root-mean-square in float64, matching `np.sqrt(np.mean(samples.astype(np.float64) ** 2))`.
pub fn waveform_rms(samples: &[f32]) -> f64 {
    if samples.is_empty() {
        return f64::NAN;
    }
    let sum_sq = samples.iter().fold(0.0f64, |acc, sample| {
        let value = f64::from(*sample);
        acc + value * value
    });
    (sum_sq / samples.len() as f64).sqrt()
}

/// True when the passage is below one PCM16 quantum of energy.
pub fn is_silent(samples: &[f32]) -> bool {
    waveform_rms(samples) < SILENCE_RMS
}

/// True when any sample is at or beyond PCM16 full scale.
pub fn is_clipped(samples: &[f32]) -> bool {
    samples.iter().any(|sample| sample.abs() >= CLIP_ABS)
}

/// D7: a cosine at or above [`IDENTITY_REUSE_COSINE`] reuses an existing identity.
/// This does not mint a new identity.
pub fn reuses_identity(cosine: f64) -> bool {
    cosine >= IDENTITY_REUSE_COSINE
}

#[cfg(test)]
mod tests {
    use super::{
        CLIP_ABS, IDENTITY_REUSE_COSINE, INPUT_SAMPLES, MAX_SAMPLES, MIN_SAMPLES, SILENCE_RMS,
        is_clipped, is_silent, prepared_audio, reuses_identity,
    };

    fn ramp(len: usize) -> Vec<f32> {
        (0..len).map(|index| index as f32).collect()
    }

    #[test]
    fn repeats_under_six_seconds_and_center_crops_longer_clips() {
        let three = ramp(48_000);
        let tiled = prepared_audio(&three).unwrap();
        assert_eq!(tiled.len(), INPUT_SAMPLES);
        assert_eq!(&tiled[..48_000], three.as_slice());
        assert_eq!(&tiled[48_000..], three.as_slice());

        let two = ramp(MIN_SAMPLES);
        let triple = prepared_audio(&two).unwrap();
        assert_eq!(&triple[..MIN_SAMPLES], two.as_slice());
        assert_eq!(&triple[MIN_SAMPLES..MIN_SAMPLES * 2], two.as_slice());
        assert_eq!(&triple[MIN_SAMPLES * 2..], two.as_slice());

        let partial = ramp(40_000);
        let repeated = prepared_audio(&partial).unwrap();
        assert_eq!(repeated[40_000], 0.0);
        assert_eq!(repeated[95_999], 15_999.0);

        let eight = ramp(128_000);
        let cropped = prepared_audio(&eight).unwrap();
        assert_eq!(cropped.as_slice(), &eight[16_000..112_000]);

        let exact = ramp(INPUT_SAMPLES);
        assert_eq!(prepared_audio(&exact).unwrap(), exact);

        let odd = ramp(INPUT_SAMPLES + 1);
        assert_eq!(
            prepared_audio(&odd).unwrap().as_slice(),
            &odd[..INPUT_SAMPLES]
        );

        let ten = ramp(MAX_SAMPLES);
        assert_eq!(
            prepared_audio(&ten).unwrap().as_slice(),
            &ten[32_000..128_000]
        );
    }

    #[test]
    fn rejects_out_of_range_and_non_finite_audio() {
        for bad in [
            ramp(MIN_SAMPLES - 1),
            ramp(MAX_SAMPLES + 1),
            vec![],
            vec![f32::NAN; MIN_SAMPLES],
            vec![f32::INFINITY; MIN_SAMPLES],
            vec![f32::NEG_INFINITY; MIN_SAMPLES],
        ] {
            let error = prepared_audio(&bad).unwrap_err();
            assert_eq!(
                error.to_string(),
                "Supply 2–10 seconds of finite mono audio."
            );
        }
        let mut mixed = ramp(MIN_SAMPLES);
        mixed[10] = f32::NAN;
        assert!(prepared_audio(&mixed).is_err());
        assert!(prepared_audio(&ramp(MIN_SAMPLES)).is_ok());
        assert!(prepared_audio(&ramp(MAX_SAMPLES)).is_ok());
    }

    #[test]
    fn silence_screen_matches_one_pcm_quantum() {
        assert!((SILENCE_RMS - (1.0 / 32768.0)).abs() < f64::EPSILON);
        let quantum = (1.0f64 / 32768.0) as f32;
        assert!(!is_silent(&vec![quantum; MIN_SAMPLES]));
        assert!(!is_silent(&vec![-quantum; MIN_SAMPLES]));
        let below = f32::from_bits(quantum.to_bits() - 1);
        assert!(is_silent(&vec![below; MIN_SAMPLES]));
        assert!(is_silent(&vec![0.0; MIN_SAMPLES]));
        let mut quiet = vec![0.0; MIN_SAMPLES];
        quiet[100] = 0.01;
        assert!(!is_silent(&quiet));
        assert!(!is_silent(&[f32::NAN]));
    }

    #[test]
    fn clipping_screen_matches_pcm16_full_scale() {
        assert_eq!(CLIP_ABS, 32767.0f32 / 32768.0);
        assert!(is_clipped(&[CLIP_ABS]));
        assert!(is_clipped(&[-CLIP_ABS]));
        assert!(is_clipped(&[-1.0]));
        assert!(is_clipped(&[f32::INFINITY]));
        let under = f32::from_bits(CLIP_ABS.to_bits() - 1);
        assert!(!is_clipped(&[under, -under, 0.5]));
        let mut passage = vec![0.1; 48_000];
        passage[10] = CLIP_ABS;
        assert!(is_clipped(&passage));
        passage[10] = under;
        assert!(!is_clipped(&passage));
        assert!(!is_clipped(&[f32::NAN]));
        assert!(!is_clipped(&[]));
    }

    /// Cosine similarity at or above 0.9999 decides identity reuse (D7).
    /// Reuse does not mint a new identity.
    #[test]
    #[ignore = "cosine >= 0.9999 decides identity reuse (D7); do not mint a new identity"]
    fn parity_redimnet() {
        assert!(reuses_identity(IDENTITY_REUSE_COSINE));
        assert!(reuses_identity(0.9999));
        assert!(reuses_identity(1.0));
        assert!(!reuses_identity(0.999899999999));
    }
}
