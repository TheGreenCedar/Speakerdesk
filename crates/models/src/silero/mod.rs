//! Silero VAD v6. Upstream: `Blaizzy/mlx-audio@feb25a37`.
//! Weights: `mlx-community/silero-vad-v6@2ebf4a5`
//! (`2ebf4a5e10726a2e78ddd4d70eedfb6f1c33eb06`).
use speakerdesk_core::limits::{VAD_CONTEXT_SAMPLES, VAD_FRAME_SAMPLES};

#[cfg(target_os = "macos")]
pub mod network;
#[cfg(target_os = "macos")]
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

/// One 512-sample step, plus the 64-sample context that precedes it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FrameSpan {
    pub index: u64,
    /// Absolute start of the new PCM samples.
    pub start: u64,
    /// Physical samples in this frame. At most [`FRAME_SAMPLES`]. A short count
    /// is only produced for a final tail and is zero-padded to 512 for the model.
    pub samples: usize,
    /// Absolute start of the context copied from earlier PCM.
    /// `None` on the first frame: the model context is 64 zeros.
    pub context_start: Option<u64>,
}

/// Where the 64 context samples, the physical frame, and the zero pad sit
/// inside the 576-sample vector `concat(context, padded_frame)`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ModelInput {
    pub context: std::ops::Range<usize>,
    pub frame: std::ops::Range<usize>,
    pub pad: std::ops::Range<usize>,
}

/// The last 64 samples of the padded 512-sample frame, which become the next context.
/// Padding zeros, when a final frame is short, sit at the end of that tail.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ContextTail {
    /// Offset of the first physical sample inside the padded frame that is kept.
    /// Equal to `physical_samples` when the tail is entirely padding.
    pub pcm_offset: usize,
    pub pcm_len: usize,
    pub trailing_zeros: usize,
}

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

/// Next frame available in `pending` samples that begin at absolute `start`.
/// Incomplete frames stay pending until `last`.
pub fn next_frame(index: u64, start: u64, pending: usize, last: bool) -> Option<FrameSpan> {
    let samples = if pending >= FRAME_SAMPLES {
        FRAME_SAMPLES
    } else if last && pending > 0 {
        pending
    } else {
        return None;
    };
    let context_start = if index == 0 {
        None
    } else {
        start.checked_sub(CONTEXT_SAMPLES as u64)
    };
    Some(FrameSpan {
        index,
        start,
        samples,
        context_start,
    })
}

/// Index every complete frame in `total_samples`, plus a short final frame when `last`.
/// `pending` is the unconsumed tail.
pub fn index_frames(total_samples: u64, last: bool) -> (Vec<FrameSpan>, u64) {
    let mut frames = Vec::new();
    let mut start = 0u64;
    while let Some(frame) = next_frame(
        frames.len() as u64,
        start,
        samples_pending(total_samples, start),
        last,
    ) {
        start += frame.samples as u64;
        frames.push(frame);
    }
    (frames, total_samples.saturating_sub(start))
}

fn samples_pending(total_samples: u64, start: u64) -> usize {
    usize::try_from(total_samples.saturating_sub(start)).unwrap_or(usize::MAX)
}

/// Layout of one model step. `physical_samples` is in `1..=512`.
pub fn model_input(physical_samples: usize) -> Option<ModelInput> {
    if physical_samples == 0 || physical_samples > FRAME_SAMPLES {
        return None;
    }
    let frame_end = CONTEXT_SAMPLES + physical_samples;
    Some(ModelInput {
        context: 0..CONTEXT_SAMPLES,
        frame: CONTEXT_SAMPLES..frame_end,
        pad: frame_end..MODEL_SAMPLES,
    })
}

/// Index the 64-sample context the model keeps after this frame (`audio[:, -64:]`).
pub fn context_tail(physical_samples: usize) -> Option<ContextTail> {
    if physical_samples == 0 || physical_samples > FRAME_SAMPLES {
        return None;
    }
    let tail_start = FRAME_SAMPLES - CONTEXT_SAMPLES;
    let pcm_len = physical_samples.saturating_sub(tail_start);
    Some(ContextTail {
        pcm_offset: physical_samples - pcm_len,
        pcm_len,
        trailing_zeros: CONTEXT_SAMPLES - pcm_len,
    })
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
    use super::{
        CONTEXT_SAMPLES, FRAME_SAMPLES, MODEL_SAMPLES, context_tail, hysteresis_flag, index_frames,
        model_input, next_frame, peak_gain,
    };

    #[test]
    fn frames_are_512_and_context_is_64() {
        assert_eq!(FRAME_SAMPLES, 512);
        assert_eq!(CONTEXT_SAMPLES, 64);
        assert_eq!(MODEL_SAMPLES, 576);
    }

    #[test]
    fn indexes_complete_frames_and_a_final_short_tail() {
        let held = next_frame(0, 0, 100, false);
        assert!(held.is_none());

        let first = next_frame(0, 0, 600, false).unwrap();
        assert_eq!(first.index, 0);
        assert_eq!(first.start, 0);
        assert_eq!(first.samples, 512);
        assert_eq!(first.context_start, None);

        let (open, pending) = index_frames(512 + 88, false);
        assert_eq!(open.len(), 1);
        assert_eq!(open[0].samples, 512);
        assert_eq!(pending, 88);

        let (closed, pending) = index_frames(512 + 88, true);
        assert_eq!(pending, 0);
        assert_eq!(closed.len(), 2);
        assert_eq!(closed[1].start, 512);
        assert_eq!(closed[1].samples, 88);
        assert_eq!(closed[1].context_start, Some(512 - 64));
        assert_eq!(closed[1].start - closed[1].context_start.unwrap(), 64);
    }

    #[test]
    fn context_is_prepended_and_the_outgoing_tail_is_the_last_64() {
        let full = model_input(512).unwrap();
        assert_eq!(full.context, 0..64);
        assert_eq!(full.frame, 64..576);
        assert!(full.pad.is_empty());

        let short = model_input(100).unwrap();
        assert_eq!(short.context, 0..64);
        assert_eq!(short.frame, 64..164);
        assert_eq!(short.pad, 164..576);
        assert!(model_input(0).is_none());
        assert!(model_input(513).is_none());

        let full_tail = context_tail(512).unwrap();
        assert_eq!(full_tail.pcm_offset, 448);
        assert_eq!(full_tail.pcm_len, 64);
        assert_eq!(full_tail.trailing_zeros, 0);

        let mid = context_tail(500).unwrap();
        assert_eq!(mid.pcm_offset, 448);
        assert_eq!(mid.pcm_len, 52);
        assert_eq!(mid.trailing_zeros, 12);
        assert_eq!(mid.pcm_offset + mid.pcm_len, 500);

        let tiny = context_tail(100).unwrap();
        assert_eq!(tiny.pcm_len, 0);
        assert_eq!(tiny.trailing_zeros, 64);
        assert_eq!(tiny.pcm_offset, 100);
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
