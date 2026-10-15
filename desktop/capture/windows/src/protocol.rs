use base64::{Engine, engine::general_purpose::STANDARD};
use serde::Deserialize;
use serde_json::{Value, json};
use std::collections::VecDeque;

pub const CAPABILITY: &str = "windows_wasapi_shared_v1";
pub const EXPORT: &str = "raw_dual_mono_stereo16k_v1";
pub const PROGRESS: &str = "audio_queue_fence_v1";
pub const RATE: u32 = 16000;
pub const TICKS_PER_SECOND: u64 = 10_000_000;

#[derive(Debug, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case", deny_unknown_fields)]
pub enum Command {
    Start {
        microphone: bool,
        system: bool,
        job_id: Option<String>,
        capture_export_contract: Option<String>,
    },
    Pause,
    Resume,
    Stop,
}

pub fn validate_start(
    microphone: bool,
    system: bool,
    job: Option<&str>,
    contract: Option<&str>,
) -> Result<(), String> {
    if !microphone && !system {
        return Err("Select a microphone or system audio.".into());
    }
    if contract.is_some_and(|value| value != EXPORT) {
        return Err("Windows capture does not provide the requested processing contract.".into());
    }
    if job.is_some_and(|value| {
        value.len() != 32
            || !value
                .bytes()
                .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
    }) {
        return Err("Invalid recording identity.".into());
    }
    Ok(())
}

pub fn description() -> Value {
    json!({"type":"helper_check", "capture_started":false,
        "capability_version":CAPABILITY,"export_contract":EXPORT,"progress_contract":PROGRESS,
        "microphone":{"rate":RATE,"channels":1,"format":"float32_le"},
        "system":{"rate":RATE,"channels":2,"format":"float32_le"},
        "processing":"original_sources_no_apple_processing","clock":"qpc_100ns"})
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Source {
    Microphone,
    System,
}
impl Source {
    pub fn name(self) -> &'static str {
        match self {
            Self::Microphone => "microphone",
            Self::System => "system",
        }
    }
    pub fn channels(self) -> usize {
        if self == Self::Microphone { 1 } else { 2 }
    }
}

#[derive(Clone, Copy, Debug, Default)]
pub struct ProducerClock {
    pub hresult: i32,
    pub position: u64,
    pub ticks: u64,
}
impl ProducerClock {
    pub fn pending(self) -> bool {
        self.hresult >= 0 && self.ticks == 0
    }
    pub fn validate(self) -> Result<(), String> {
        if self.hresult < 0 || (self.ticks == 0 && self.position != 0) {
            return Err(format!(
                "WASAPI returned an invalid producer clock: HRESULT 0x{:08x}, device_position={}, qpc_100ns={}.",
                self.hresult as u32, self.position, self.ticks
            ));
        }
        // A successful zero/zero read immediately after Start/Reset is pending,
        // not evidence of progress. The existing readiness deadline bounds it.
        Ok(())
    }
    pub fn ready(self, source: Source, floor: u64, packet_seen: bool) -> bool {
        self.hresult >= 0
            && self.ticks != 0
            && self.ticks >= floor
            && (source == Source::System || packet_seen)
    }
}

pub fn render_refill_frames(capacity: u32, padding: u32, running: bool) -> Result<u32, String> {
    if capacity == 0 || capacity > RATE || padding > capacity {
        return Err(format!(
            "WASAPI silent render returned invalid buffer bounds: capacity={capacity}, padding={padding}."
        ));
    }
    if running && padding == 0 {
        return Err(format!(
            "WASAPI silent render underflowed its {capacity}-frame buffer; recording stopped and accepted audio is retained."
        ));
    }
    Ok(capacity - padding)
}

pub fn audio_events(source: Source, time: f64, samples: &[f32]) -> Result<VecDeque<Value>, String> {
    let channels = source.channels();
    if !time.is_finite()
        || time < 0.0
        || samples.is_empty()
        || !samples.len().is_multiple_of(channels)
        || samples.len() / channels > RATE as usize
        || samples.iter().any(|value| !value.is_finite())
    {
        return Err("Invalid captured Float32 packet.".into());
    }
    Ok(samples
        .chunks(4000 * channels)
        .enumerate()
        .map(|(chunk, pcm)| {
            let bytes: Vec<_> = pcm.iter().flat_map(|value| value.to_le_bytes()).collect();
            json!({"type":"audio","source":source.name(),"time":time + chunk as f64 * 0.25,
            "rate":RATE,"channels":channels,"pcm":STANDARD.encode(bytes)})
        })
        .collect())
}

pub struct RecordingClock {
    origin: u64,
    paused: u64,
    pause_at: Option<u64>,
}
impl RecordingClock {
    pub fn new(origin: u64) -> Self {
        Self {
            origin,
            paused: 0,
            pause_at: None,
        }
    }
    pub fn at(&self, ticks: u64) -> f64 {
        self.pause_at
            .unwrap_or(ticks)
            .saturating_sub(self.origin)
            .saturating_sub(self.paused) as f64
            / TICKS_PER_SECOND as f64
    }
    pub fn pause(&mut self, ticks: u64) {
        self.pause_at = Some(ticks);
    }
    pub fn resume(&mut self, ticks: u64) -> Result<(), String> {
        let paused = self.pause_at.take().ok_or("Capture is not paused.")?;
        self.paused = self
            .paused
            .checked_add(
                ticks
                    .checked_sub(paused)
                    .ok_or("Capture clock moved backwards.")?,
            )
            .ok_or("Capture clock overflow.")?;
        Ok(())
    }
}

#[derive(Default)]
pub struct AudioTimeline {
    origin: Option<f64>,
    frames: u64,
    expected_ticks: Option<u64>,
}
impl AudioTimeline {
    pub fn accept(
        &mut self,
        ticks: u64,
        frames: usize,
        clock: &RecordingClock,
    ) -> Result<f64, String> {
        if let Some(expected) = self.expected_ticks
            && ticks.abs_diff(expected) > TICKS_PER_SECOND / 50
        {
            return Err("Capture timing changed by more than 20 ms; restart recording. Original audio is retained.".into());
        }
        let position =
            *self.origin.get_or_insert_with(|| clock.at(ticks)) + self.frames as f64 / RATE as f64;
        self.frames = self
            .frames
            .checked_add(frames as u64)
            .ok_or("Capture frame count overflow.")?;
        self.expected_ticks = Some(
            ticks
                .checked_add(frames as u64 * 625)
                .ok_or("Capture timestamp overflow.")?,
        );
        Ok(position)
    }
}

#[derive(Default)]
pub struct Progress {
    sequence: u64,
    time: f64,
}
impl Progress {
    pub fn fence(&mut self, time: f64, observed_and_drained: bool) -> Result<Value, String> {
        if !observed_and_drained
            || !time.is_finite()
            || time < self.time
            || (self.sequence > 0 && time == self.time)
        {
            return Err("Capture progress requires a fresh drained producer observation.".into());
        }
        self.sequence = self
            .sequence
            .checked_add(1)
            .ok_or("Capture fence sequence overflow.")?;
        self.time = time;
        Ok(
            json!({"type":"clock","time":time,"paused":false,"producer_fence":PROGRESS,"sequence":self.sequence}),
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn serializes_original_stereo_without_downmixing_and_splits_on_frame_boundaries() {
        let samples: Vec<_> = (0..8002).map(|index| index as f32 / 10000.0).collect();
        let events = audio_events(Source::System, 1.25, &samples).unwrap();
        assert_eq!(events.len(), 2);
        assert_eq!(events[1]["time"], 1.5);
        let recovered: Vec<_> = events
            .iter()
            .flat_map(|value| STANDARD.decode(value["pcm"].as_str().unwrap()).unwrap())
            .collect();
        assert_eq!(
            recovered,
            samples
                .iter()
                .flat_map(|sample| sample.to_le_bytes())
                .collect::<Vec<_>>()
        );
        assert!(audio_events(Source::System, 0.0, &[1.0]).is_err());
        assert!(audio_events(Source::Microphone, 0.0, &[f32::NAN]).is_err());
    }
    #[test]
    fn pause_removes_only_paused_wall_time_and_sample_clock_ignores_callback_jitter() {
        let mut clock = RecordingClock::new(10_000_000);
        clock.pause(30_000_000);
        assert_eq!(clock.at(90_000_000), 2.0);
        clock.resume(80_000_000).unwrap();
        assert_eq!(clock.at(90_000_000), 3.0);
        let mut audio = AudioTimeline::default();
        assert_eq!(audio.accept(90_000_000, 160, &clock).unwrap(), 3.0);
        assert_eq!(audio.accept(90_100_500, 160, &clock).unwrap(), 3.01);
        assert!(audio.accept(91_000_000, 160, &clock).is_err());
    }
    #[test]
    fn a_timer_or_stale_observation_cannot_mint_a_progress_fence() {
        let mut progress = Progress::default();
        assert!(progress.fence(1.0, false).is_err());
        assert_eq!(progress.fence(1.0, true).unwrap()["sequence"], 1);
        assert!(progress.fence(1.0, true).is_err());
        assert!(progress.fence(0.5, true).is_err());
        assert_eq!(progress.fence(1.5, true).unwrap()["sequence"], 2);
    }
    #[test]
    fn zero_startup_clock_waits_for_fresh_producer_evidence_after_start_and_reset() {
        let pending = ProducerClock::default();
        pending.validate().unwrap();
        assert!(pending.pending());
        for source in [Source::Microphone, Source::System] {
            assert!(!pending.ready(source, 0, true));
            assert!(!pending.ready(source, 100, false));
        }
        let first = ProducerClock {
            hresult: 0,
            position: 0,
            ticks: 110,
        };
        first.validate().unwrap();
        assert!(!first.pending());
        assert!(first.ready(Source::System, 100, false));
        assert!(!first.ready(Source::Microphone, 100, false));
        assert!(first.ready(Source::Microphone, 100, true));
        // A fresh resume floor rejects the pre-pause observation and old packet.
        assert!(!first.ready(Source::System, 200, true));
        let resumed = ProducerClock {
            ticks: 210,
            ..first
        };
        assert!(!resumed.ready(Source::Microphone, 200, false));
        assert!(resumed.ready(Source::Microphone, 200, true));
        for invalid in [
            ProducerClock {
                position: 1,
                ..pending
            },
            ProducerClock {
                hresult: 0x88890004_u32 as i32,
                ..first
            },
        ] {
            let error = invalid.validate().unwrap_err();
            assert!(
                error.contains("HRESULT")
                    && error.contains("device_position=")
                    && error.contains("qpc_100ns=")
            );
            assert!(!invalid.ready(Source::System, 100, true));
        }
        // A running client can expose a position before a correlated clock is
        // available. Startup may wait for it, but it still cannot establish readiness.
        let uncorrelated = ProducerClock {
            position: 1280,
            ..pending
        };
        assert!(uncorrelated.pending());
        assert!(!uncorrelated.ready(Source::System, 100, true));
    }
    #[test]
    fn silent_render_primes_once_refills_only_free_frames_and_rejects_underflow() {
        for capacity in [1, 1600, RATE] {
            assert_eq!(render_refill_frames(capacity, 0, false).unwrap(), capacity);
            assert_eq!(render_refill_frames(capacity, capacity, true).unwrap(), 0);
            assert_eq!(
                render_refill_frames(capacity, 1, true).unwrap(),
                capacity - 1
            );
            assert!(
                render_refill_frames(capacity, 0, true)
                    .unwrap_err()
                    .contains("underflow")
            );
            assert!(render_refill_frames(capacity, capacity + 1, false).is_err());
        }
        assert_eq!(render_refill_frames(1600, 1440, true).unwrap(), 160);
        for capacity in [0, RATE + 1, u32::MAX] {
            assert!(render_refill_frames(capacity, 0, false).is_err());
        }
        // Reset on resume permits a fresh prime, not a running refill from empty.
        assert_eq!(render_refill_frames(1600, 0, false).unwrap(), 1600);
    }
    #[test]
    fn rejects_empty_sources_and_apple_processing_claims() {
        assert!(validate_start(false, false, None, None).is_err());
        assert!(validate_start(true, true, None, Some("apple_processed_mono16k_v1")).is_err());
        assert!(validate_start(true, true, Some("bad"), Some(EXPORT)).is_err());
        assert!(validate_start(true, false, None, None).is_ok());
        assert!(!description().to_string().contains("qualified_apple"));
    }
}
