//! Broadband reference clock from `speakerdesk/capture_echo.py`.
//!
//! First differences emphasize broadband excitation. A pure tone has an
//! ambiguous delay and is not evidence of acoustic echo. Polarity is ignored
//! because a transducer can invert it. The adaptive filter learns the sign.

use crate::fft::power_spectrum;

/// Capture rate used by the reference clock.
pub const RATE: usize = 16_000;
/// Pinned AEC3 linear output has one 64-sample block-framing delay.
pub const DELAY: usize = 64;
/// Extra reference history available while resampling for clock skew.
pub const LOOKAHEAD: usize = RATE / 2;
/// Normalized correlation required before a delay updates the clock.
pub const SCORE_THRESHOLD: f64 = 0.3;

const WINDOW: usize = RATE / 8;
const SEARCH: usize = RATE / 2;
const TONAL_FRACTION: f64 = 0.1;
const ENERGY_FLOOR: f64 = 1e-4;
const ROUTE_JUMP: f64 = RATE as f64 * 0.003;
const SKEW_SPAN: f64 = RATE as f64 * 0.75;
const MAX_SKEW_SLOPE: f64 = 0.0005;
const MAX_SKEW_ERROR: f64 = 0.6;

/// A broadband delay match between the stereo reference and the microphone.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CorrelationDecision {
    /// Microphone delay in samples relative to the far-end reference.
    pub delay: f64,
    /// Reference channel that produced the strongest match.
    pub channel: usize,
    /// Signed normalized correlation at the validating peak.
    pub score: f64,
}

/// Correlate one analysis window.
///
/// `search` is the maximum delay in samples. `far` holds
/// `microphone.len() + search` frames and the microphone window is aligned
/// with the end of that reference. Matches below [`SCORE_THRESHOLD`], silence,
/// and fully tonal references return `None`.
pub fn correlation_decision(
    far: &[[f32; 2]],
    microphone: &[f32],
    search: usize,
) -> Option<CorrelationDecision> {
    if microphone.len() < 2 || far.len() != microphone.len() + search {
        return None;
    }
    let reference = [
        differences(far.iter().map(|frame| frame[0])),
        differences(far.iter().map(|frame| frame[1])),
    ];
    let near = differences(microphone.iter().copied());
    let near_energy = dot(&near, &near);
    if near_energy < ENERGY_FLOOR {
        return None;
    }
    let tonal = [is_tonal(&reference[0]), is_tonal(&reference[1])];
    if tonal[0] && tonal[1] {
        return None;
    }

    let mut correlation = [Vec::new(), Vec::new()];
    let mut score = [Vec::new(), Vec::new()];
    for channel in 0..2 {
        if tonal[channel] {
            let lags = search + 1;
            correlation[channel] = vec![0.0; lags];
            score[channel] = vec![0.0; lags];
            continue;
        }
        let raw = linear_correlation(&reference[channel], &near);
        let energy = window_energy(&reference[channel], near.len());
        let normalized = raw
            .iter()
            .zip(&energy)
            .map(|(value, bin_energy)| {
                if *bin_energy < ENERGY_FLOOR {
                    0.0
                } else {
                    value / (bin_energy * near_energy).max(1e-20).sqrt()
                }
            })
            .collect();
        correlation[channel] = raw;
        score[channel] = normalized;
    }

    let (peak, channel, peak_score) = peak_score(&score)?;
    if peak_score.abs() < SCORE_THRESHOLD {
        return None;
    }
    let peak = earlier_arrival(&score[channel], peak);
    let fraction = parabolic_fraction(&correlation[channel], peak);
    Some(CorrelationDecision {
        delay: search as f64 - peak as f64 - fraction,
        channel,
        score: peak_score,
    })
}

/// Bounded broadband correlation validates echo and estimates clock skew.
///
/// A 125 ms window searches speaker latency up to half a second. Only strong,
/// consistent matches update the drift estimate. Weak double-talk matches
/// retain the last reliable rate.
#[derive(Debug, Clone, Default)]
pub struct ReferenceClock {
    far: Vec<[f32; 2]>,
    microphone: Vec<f32>,
    start: usize,
    end: usize,
    matches: Vec<(f64, f64)>,
    match_channels: Vec<usize>,
    route_matches: Vec<(f64, f64, usize)>,
    echo: bool,
    skew: f64,
    phase: f64,
}

impl ReferenceClock {
    pub fn echo(&self) -> bool {
        self.echo
    }

    pub fn skew(&self) -> f64 {
        self.skew
    }

    pub fn phase(&self) -> f64 {
        self.phase
    }

    /// Accepted `(center, delay)` matches, oldest first.
    pub fn matches(&self) -> &[(f64, f64)] {
        &self.matches
    }

    /// Append aligned far-end frames and microphone samples.
    pub fn append(&mut self, far: &[[f32; 2]], microphone: &[f32]) {
        let count = far.len().min(microphone.len());
        if count == 0 {
            return;
        }
        self.far.extend_from_slice(&far[..count]);
        self.microphone.extend_from_slice(&microphone[..count]);
        self.end += count;
        if self.far.len() > RATE * 2 {
            let trim = self.far.len() - RATE * 2;
            self.far.drain(..trim);
            self.microphone.drain(..trim);
            self.start += trim;
        }
        if self.far.len() < WINDOW + SEARCH {
            return;
        }
        let far_window = &self.far[self.far.len() - WINDOW - SEARCH..];
        let mic_window = &self.microphone[self.microphone.len() - WINDOW..];
        let Some(decision) = correlation_decision(far_window, mic_window, SEARCH) else {
            return;
        };
        self.record(self.end as f64 - WINDOW as f64 / 2.0, decision);
    }

    /// Interleaved stereo reference for `[start, start + count)`, resampled
    /// by the current skew estimate.
    pub fn reference(&mut self, start: usize, count: usize) -> Vec<f32> {
        if self.far.is_empty() || count == 0 {
            return vec![0.0; count * 2];
        }
        let phase = self.phase;
        let skew = self.skew;
        let origin = self.start;
        let mut rendered = Vec::with_capacity(count * 2);
        for index in 0..count {
            let position = start as f64 + index as f64 + phase + index as f64 * skew;
            rendered.push(interpolate(&self.far, 0, position, origin));
            rendered.push(interpolate(&self.far, 1, position, origin));
        }
        let advanced = self.phase + count as f64 * self.skew;
        self.phase = advanced.clamp(-(LOOKAHEAD as f64), LOOKAHEAD as f64);
        rendered
    }

    fn record(&mut self, center: f64, decision: CorrelationDecision) {
        if let Some(&(_, previous)) = self.matches.last()
            && (decision.delay - previous).abs() > ROUTE_JUMP
        {
            if let Some(&(_, previous_route, _)) = self.route_matches.last()
                && (decision.delay - previous_route).abs() > ROUTE_JUMP
            {
                self.route_matches.clear();
            }
            self.route_matches
                .push((center, decision.delay, decision.channel));
            if self.route_matches.len() < 3 {
                return;
            }
            let accepted = &self.route_matches[self.route_matches.len() - 3..];
            self.matches = accepted
                .iter()
                .map(|(match_center, delay, _)| (*match_center, *delay))
                .collect();
            self.match_channels = accepted.iter().map(|matched| matched.2).collect();
            self.route_matches.clear();
            self.skew = 0.0;
            return;
        }
        self.route_matches.clear();
        self.matches.push((center, decision.delay));
        self.match_channels.push(decision.channel);
        if self.matches.len() > 12 {
            let trim = self.matches.len() - 12;
            self.matches.drain(..trim);
            self.match_channels.drain(..trim);
        }
        if self.matches.len() >= 3 {
            self.echo = true;
        }
        self.update_skew();
    }

    fn update_skew(&mut self) {
        if self.matches.len() < 4 {
            return;
        }
        let span = self.matches.last().map(|matched| matched.0).unwrap_or(0.0)
            - self.matches.first().map(|matched| matched.0).unwrap_or(0.0);
        if span < SKEW_SPAN {
            return;
        }
        let times: Vec<f64> = self.matches.iter().map(|matched| matched.0).collect();
        let delays: Vec<f64> = self.matches.iter().map(|matched| matched.1).collect();
        if let Some(skew) = fit_skew(&times, &delays, &self.match_channels) {
            self.skew = skew;
        }
    }
}

fn differences(samples: impl Iterator<Item = f32>) -> Vec<f64> {
    let samples: Vec<f64> = samples.map(f64::from).collect();
    samples.windows(2).map(|pair| pair[1] - pair[0]).collect()
}

fn dot(left: &[f64], right: &[f64]) -> f64 {
    left.iter().zip(right).map(|(a, b)| a * b).sum()
}

fn is_tonal(samples: &[f64]) -> bool {
    let spectrum = power_spectrum(samples);
    let total = spectrum.iter().sum::<f64>().max(1e-20);
    let peak = spectrum.iter().copied().fold(0.0, f64::max);
    peak > TONAL_FRACTION * total
}

fn linear_correlation(reference: &[f64], near: &[f64]) -> Vec<f64> {
    let lags = reference.len() - near.len() + 1;
    (0..lags)
        .map(|lag| dot(&reference[lag..lag + near.len()], near))
        .collect()
}

fn window_energy(samples: &[f64], window: usize) -> Vec<f64> {
    let mut prefix = Vec::with_capacity(samples.len() + 1);
    prefix.push(0.0);
    let mut sum = 0.0;
    for sample in samples {
        sum += sample * sample;
        prefix.push(sum);
    }
    (0..=samples.len() - window)
        .map(|start| prefix[start + window] - prefix[start])
        .collect()
}

fn peak_score(score: &[Vec<f64>; 2]) -> Option<(usize, usize, f64)> {
    let lags = score[0].len();
    if lags == 0 || score[1].len() != lags {
        return None;
    }
    let mut best_lag = 0;
    let mut best_channel = 0;
    let mut best = -1.0;
    for (lag, (left, right)) in score[0].iter().zip(&score[1]).enumerate() {
        for (channel, value) in [left, right].into_iter().enumerate() {
            let magnitude = value.abs();
            if magnitude > best {
                best = magnitude;
                best_lag = lag;
                best_channel = channel;
            }
        }
    }
    Some((best_lag, best_channel, score[best_channel][best_lag]))
}

/// A reflected path can outrank the direct path. After the strong match
/// validates echo, align to a significant earlier local peak so the causal
/// filter can model both. Skip the adjacent autocorrelation lobe and search
/// only 16 ms of pre-echo.
fn earlier_arrival(score: &[f64], peak: usize) -> usize {
    let magnitude: Vec<f64> = score.iter().map(|value| value.abs()).collect();
    let threshold = 0.12_f64.max(0.4 * magnitude[peak]);
    let first = peak + 8;
    let last = (score.len() - 1).min(peak + RATE * 16 / 1000);
    let mut chosen = peak;
    if first < last {
        for index in first..last {
            let local = magnitude[index] >= threshold
                && magnitude[index] > magnitude[index - 1]
                && magnitude[index] >= magnitude[index + 1];
            if local {
                chosen = index;
            }
        }
    }
    chosen
}

fn parabolic_fraction(correlation: &[f64], peak: usize) -> f64 {
    if peak == 0 || peak + 1 >= correlation.len() {
        return 0.0;
    }
    let (left, mid, right) = (
        correlation[peak - 1],
        correlation[peak],
        correlation[peak + 1],
    );
    let denominator = left - 2.0 * mid + right;
    if denominator == 0.0 {
        return 0.0;
    }
    (0.5 * (left - right) / denominator).clamp(-0.5, 0.5)
}

/// Fit one drift shared by every channel that has enough matches.
///
/// Path delay may differ between speakers. Demean within each channel so
/// changing the strongest speaker does not look like clock drift.
fn fit_skew(times: &[f64], delays: &[f64], channels: &[usize]) -> Option<f64> {
    if times.len() != delays.len() || times.len() != channels.len() || times.is_empty() {
        return None;
    }
    let mut centered_time = vec![0.0; times.len()];
    let mut centered_delay = vec![0.0; delays.len()];
    let mut reliable = vec![false; times.len()];
    let mut seen = Vec::new();
    for &channel in channels {
        if seen.contains(&channel) {
            continue;
        }
        seen.push(channel);
        let indexes: Vec<usize> = channels
            .iter()
            .enumerate()
            .filter_map(|(index, candidate)| (*candidate == channel).then_some(index))
            .collect();
        if indexes.len() < 4 {
            continue;
        }
        let count = indexes.len() as f64;
        let mean_time = indexes.iter().map(|&index| times[index]).sum::<f64>() / count;
        let mean_delay = indexes.iter().map(|&index| delays[index]).sum::<f64>() / count;
        for index in indexes {
            centered_time[index] = times[index] - mean_time;
            centered_delay[index] = delays[index] - mean_delay;
            reliable[index] = true;
        }
    }
    let mut energy = 0.0;
    let mut covariance = 0.0;
    let mut count = 0.0;
    let mut residual = 0.0;
    for index in 0..times.len() {
        if !reliable[index] {
            continue;
        }
        energy += centered_time[index] * centered_time[index];
        covariance += centered_time[index] * centered_delay[index];
    }
    if energy == 0.0 {
        return None;
    }
    let slope = covariance / energy;
    for index in 0..times.len() {
        if !reliable[index] {
            continue;
        }
        let error = centered_delay[index] - slope * centered_time[index];
        residual += error * error;
        count += 1.0;
    }
    let error = (residual / count).sqrt();
    if slope.abs() <= MAX_SKEW_SLOPE && error < MAX_SKEW_ERROR {
        Some(-slope)
    } else {
        None
    }
}

fn interpolate(frames: &[[f32; 2]], channel: usize, position: f64, start: usize) -> f32 {
    if frames.is_empty() {
        return 0.0;
    }
    let first = start as f64;
    let last = (start + frames.len() - 1) as f64;
    if position < first {
        return 0.0;
    }
    if position >= last {
        return frames[frames.len() - 1][channel];
    }
    let offset = position - first;
    let index = offset.floor() as usize;
    let fraction = offset - index as f64;
    let left = f64::from(frames[index][channel]);
    let right = f64::from(frames[index + 1][channel]);
    (left + (right - left) * fraction) as f32
}

#[cfg(test)]
mod tests {
    use super::{
        CorrelationDecision, DELAY, RATE, ReferenceClock, SCORE_THRESHOLD, SEARCH, WINDOW,
        correlation_decision, fit_skew,
    };

    fn lcg(count: usize, seed: u32) -> Vec<f32> {
        let mut state = seed;
        (0..count)
            .map(|_| {
                state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
                ((state >> 8) as f32 / 16_777_216.0 - 0.5) * 0.4
            })
            .collect()
    }

    fn stereo(left: &[f32], right: &[f32]) -> Vec<[f32; 2]> {
        left.iter()
            .zip(right)
            .map(|(&left, &right)| [left, right])
            .collect()
    }

    fn tone(count: usize) -> Vec<f32> {
        (0..count)
            .map(|index| {
                (2.0 * std::f64::consts::PI * 8.0 * index as f64 / count as f64).sin() as f32 * 0.25
            })
            .collect()
    }

    fn assert_delay(decision: &CorrelationDecision, channel: usize, delay: f64) {
        assert_eq!(decision.channel, channel);
        assert!(
            decision.score.abs() >= SCORE_THRESHOLD,
            "score {}",
            decision.score
        );
        assert!(
            (decision.delay - delay).abs() < 0.05,
            "delay {} expected {delay}",
            decision.delay
        );
    }

    #[test]
    fn clock_constants_match_capture_echo() {
        assert_eq!(DELAY, 64);
        assert_eq!(RATE, 16_000);
        assert_eq!(SCORE_THRESHOLD, 0.3);
    }

    #[test]
    fn delayed_noise_clears_the_score_threshold() {
        let search = 48;
        let window = 400;
        let delay = 20;
        let left = lcg(window + search, 1);
        let right = vec![0.0; window + search];
        let microphone = left[search - delay..search - delay + window].to_vec();
        let decision = correlation_decision(&stereo(&left, &right), &microphone, search)
            .expect("broadband delay");
        assert_delay(&decision, 0, delay as f64);
        assert!(decision.score > 0.0);
    }

    #[test]
    fn inverted_noise_still_matches_the_same_delay() {
        let search = 48;
        let window = 400;
        let delay = 20;
        let left = lcg(window + search, 1);
        let right = lcg(window + search, 2);
        let microphone: Vec<f32> = left[search - delay..search - delay + window]
            .iter()
            .map(|sample| -sample)
            .collect();
        let decision = correlation_decision(&stereo(&left, &right), &microphone, search)
            .expect("inverted delay");
        assert_delay(&decision, 0, delay as f64);
        assert!(decision.score < 0.0);
    }

    #[test]
    fn the_stronger_channel_supplies_the_delay() {
        let search = 48;
        let window = 400;
        let left = lcg(window + search, 1);
        let right = lcg(window + search, 99);
        let delay = 16;
        let mut microphone = vec![0.0; window];
        for index in 0..window {
            microphone[index] = 0.5 * left[search - 11 + index] + right[search - delay + index];
        }
        let decision = correlation_decision(&stereo(&left, &right), &microphone, search)
            .expect("channel delay");
        assert_delay(&decision, 1, delay as f64);
    }

    #[test]
    fn a_tone_is_not_an_echo_match() {
        let search = 48;
        let window = 400;
        let shaped = tone(window + search);
        let microphone = shaped[search - 20..search - 20 + window].to_vec();
        assert!(correlation_decision(&stereo(&shaped, &shaped), &microphone, search).is_none());
    }

    #[test]
    fn a_tonal_channel_does_not_hide_broadband_delay() {
        let search = 48;
        let window = 400;
        let delay = 20;
        let shaped = tone(window + search);
        let noise = lcg(window + search, 1);
        let microphone = noise[search - delay..search - delay + window].to_vec();
        let decision = correlation_decision(&stereo(&shaped, &noise), &microphone, search)
            .expect("noise channel");
        assert_delay(&decision, 1, delay as f64);
    }

    #[test]
    fn silence_and_uncorrelated_noise_are_rejected() {
        let search = 48;
        let window = 400;
        let far = stereo(&lcg(window + search, 1), &lcg(window + search, 2));
        assert!(correlation_decision(&far, &vec![0.0; window], search).is_none());
        assert!(correlation_decision(&far, &lcg(window, 99), search).is_none());
    }

    #[test]
    fn shared_drift_survives_a_fixed_path_offset() {
        let drift = 0.0002;
        let mut times = Vec::new();
        let mut delays = Vec::new();
        let mut channels = Vec::new();
        for index in 0..8 {
            let time = index as f64 * 2_000.0;
            let channel = index % 2;
            let path = if channel == 0 { 715.0 } else { 717.0 };
            times.push(time);
            delays.push(path - drift * time);
            channels.push(channel);
        }
        let skew = fit_skew(&times, &delays, &channels).expect("skew");
        assert!((skew - drift).abs() < 1e-12, "{skew}");
    }

    #[test]
    fn fixed_path_offsets_do_not_invent_skew() {
        let mut times = Vec::new();
        let mut delays = Vec::new();
        let mut channels = Vec::new();
        for index in 0..8 {
            let channel = index % 2;
            times.push(index as f64 * 2_000.0);
            delays.push(if channel == 0 { 715.0 } else { 717.0 });
            channels.push(channel);
        }
        assert_eq!(fit_skew(&times, &delays, &channels), Some(0.0));
        assert!(fit_skew(&[0.0, 1.0, 2.0], &[0.0, 1.0, 2.0], &[0, 0, 0]).is_none());
    }

    #[test]
    fn steep_or_scattered_delay_does_not_update_skew() {
        let times = [0.0, 4_000.0, 8_000.0, 12_000.0];
        assert!(fit_skew(&times, &[0.0, 40.0, 80.0, 120.0], &[0, 0, 0, 0]).is_none());
        assert!(fit_skew(&times, &[0.0, 0.0, 0.0, 5.0], &[1, 1, 1, 1]).is_none());
    }

    #[test]
    fn reference_clock_accepts_three_delayed_noise_blocks() {
        let delay = DELAY;
        let count = WINDOW + SEARCH;
        let noise = lcg(count, 7);
        let mut far = vec![[0.0; 2]; count];
        let mut microphone = vec![0.0; count];
        for index in 0..count {
            far[index][0] = noise[index];
            if index >= delay {
                microphone[index] = noise[index - delay];
            }
        }
        let mut clock = ReferenceClock::default();
        for _ in 0..3 {
            clock.append(&far, &microphone);
        }
        assert!(clock.echo());
        assert_eq!(clock.skew(), 0.0);
        assert_eq!(clock.matches().len(), 3);
        for matched in clock.matches() {
            assert!((matched.1 - delay as f64).abs() < 0.05, "{}", matched.1);
        }
    }

    #[test]
    fn reference_clock_rejects_a_tone_and_silence() {
        let count = WINDOW + SEARCH;
        let shaped = tone(count);
        let mut far = vec![[0.0; 2]; count];
        let mut microphone = vec![0.0; count];
        for index in 0..count {
            far[index] = [shaped[index], shaped[index]];
            if index >= DELAY {
                microphone[index] = shaped[index - DELAY];
            }
        }
        let mut clock = ReferenceClock::default();
        clock.append(&far, &microphone);
        clock.append(&vec![[0.0; 2]; count], &vec![0.0; count]);
        assert!(!clock.echo());
        assert!(clock.matches().is_empty());
        assert_eq!(clock.skew(), 0.0);
    }

    #[test]
    fn reference_resamples_stored_frames() {
        let mut clock = ReferenceClock::default();
        assert_eq!(clock.reference(0, 2), vec![0.0; 4]);
        clock.append(&[[0.0, 1.0], [10.0, 2.0], [30.0, 4.0]], &[0.0, 0.0, 0.0]);
        assert_eq!(clock.reference(0, 2), vec![0.0, 1.0, 10.0, 2.0]);
        assert_eq!(clock.reference(5, 1), vec![30.0, 4.0]);
    }
}
