//! Silero VAD v6. Upstream: `Blaizzy/mlx-audio@feb25a37`.
//! Weights: `mlx-community/silero-vad-v6@2ebf4a5`
//! (`2ebf4a5e10726a2e78ddd4d70eedfb6f1c33eb06`).
use speakerdesk_core::limits::{VAD_CONTEXT_SAMPLES, VAD_FRAME_SAMPLES};

#[cfg(target_os = "macos")]
pub mod network;
#[cfg(windows)]
#[path = "network_windows.rs"]
pub mod network;
#[cfg(any(target_os = "macos", windows))]
pub mod session;

/// New samples in one Silero 16 kHz frame (`chunk_size` in the pinned branch).
pub const FRAME_SAMPLES: usize = VAD_FRAME_SAMPLES;
/// Samples carried from the previous frame and concatenated in front (`context_size`).
pub const CONTEXT_SAMPLES: usize = VAD_CONTEXT_SAMPLES;
/// Model input length: 64 context samples followed by one 512-sample frame.
pub const MODEL_SAMPLES: usize = CONTEXT_SAMPLES + FRAME_SAMPLES;

/// Onset threshold from `SpeechFrames.append` (`probability >= 0.5`).
pub const ONSET_PROBABILITY: f32 = 0.5;
/// Release threshold from `SpeechFrames.append` (`probability < 0.35` clears the latch).
pub const RELEASE_PROBABILITY: f32 = 0.35;

const GAIN_TARGET_PEAK: f32 = 0.25;
const GAIN_MIN: f32 = 1.0;
const GAIN_MAX: f32 = 256.0;

/// Short weight pin. Full revision is `2ebf4a5e10726a2e78ddd4d70eedfb6f1c33eb06`.
pub const WEIGHT_PIN: &str = "mlx-community/silero-vad-v6@2ebf4a5";

/// Speaking flag for this frame and the latch carried forward.
///
/// Matches `SpeechFrames.append`: onset at >= 0.5, release below 0.35, and the
/// band in between holds `was_speaking`. A constant-PCM frame reports
/// `speaking == false` and still updates the latch from the probability.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct HysteresisFlag {
    pub speaking: bool,
    pub latch: bool,
}

/// Quiet-speech gain `min(256, max(1, 0.25 / peak))` for a positive finite peak.
/// A zero peak has nothing to scale, so the gain is 1, matching
/// `min(256, max(1, 0.25 / peak)) if peak else 1` in `SpeechSession.feed`.
/// A non-finite or negative peak also yields 1.
pub fn peak_gain(peak: f32) -> f32 {
    if peak.is_finite() && peak > 0.0 {
        (GAIN_TARGET_PEAK / peak).clamp(GAIN_MIN, GAIN_MAX)
    } else {
        GAIN_MIN
    }
}

pub fn hysteresis_flag(was_speaking: bool, probability: f32, constant_pcm: bool) -> HysteresisFlag {
    let latch = if probability >= ONSET_PROBABILITY {
        true
    } else if probability < RELEASE_PROBABILITY {
        false
    } else {
        was_speaking
    };
    HysteresisFlag {
        speaking: latch && !constant_pcm,
        latch,
    }
}

#[cfg(test)]
mod tests {
    use super::{CONTEXT_SAMPLES, FRAME_SAMPLES, MODEL_SAMPLES, hysteresis_flag, peak_gain};

    #[test]
    fn frames_are_512_and_context_is_64() {
        assert_eq!(FRAME_SAMPLES, 512);
        assert_eq!(CONTEXT_SAMPLES, 64);
        assert_eq!(MODEL_SAMPLES, 576);
    }

    #[test]
    fn gain_caps_quiet_peaks_and_is_unity_at_peak_zero() {
        assert_eq!(peak_gain(0.0), 1.0);
        assert_eq!(peak_gain(-0.2), 1.0);
        assert_eq!(peak_gain(f32::NAN), 1.0);
        assert_eq!(peak_gain(0.25), 1.0);
        assert_eq!(peak_gain(1.0), 1.0);
        assert_eq!(peak_gain(0.125), 2.0);
        assert_eq!(peak_gain(0.0625), 4.0);
        assert_eq!(peak_gain(0.25 / 256.0), 256.0);
        assert_eq!(peak_gain(0.25 / 512.0), 256.0);
    }

    #[test]
    fn hysteresis_holds_the_band_and_constant_pcm_hides_speech() {
        let onset = hysteresis_flag(false, 0.5, false);
        assert!(onset.speaking && onset.latch);

        let hold = hysteresis_flag(true, 0.4, false);
        assert!(hold.speaking && hold.latch);
        let still = hysteresis_flag(false, 0.35, false);
        assert!(!still.speaking && !still.latch);

        let release = hysteresis_flag(true, 0.25, false);
        assert!(!release.speaking && !release.latch);

        let constant = hysteresis_flag(true, 0.4, true);
        assert!(!constant.speaking);
        assert!(constant.latch);
        let resumed = hysteresis_flag(constant.latch, 0.4, false);
        assert!(resumed.speaking && resumed.latch);
    }
}
