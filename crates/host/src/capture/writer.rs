use std::{
    collections::BTreeMap,
    fs::{self, File},
    path::PathBuf,
};

use base64::{Engine, engine::general_purpose::STANDARD};
use serde::Deserialize;
use serde_json::Value;
use speakerdesk_core::audio::pcm16_bytes;
use tokio::sync::{mpsc, watch};

use super::processing::{Helper, invalid, private_create};
use super::{CaptureError, CaptureEvent, EventKind, ProcessingReceipt, disk_guard_message};

const RATE: u32 = 16000;
const TILE: u32 = 4000;

#[derive(Debug, Clone)]
pub struct CaptureConfig {
    pub job_id: String,
    /// An existing private job directory. Capture creates fixed files exclusively.
    pub folder: PathBuf,
    pub microphone: bool,
    pub system: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CapturePhase {
    Starting,
    Recording,
    Paused,
    Stopping,
    Stopped,
    Failed,
}

/// Coalescible notice: every retained source covers `[0, committed_sample)`.
/// The terminal phases mean both the child and writer have been joined.
#[derive(Debug, Clone, PartialEq)]
pub struct CaptureSnapshot {
    pub phase: CapturePhase,
    pub committed_sample: u32,
    pub late_samples: u64,
    pub levels: BTreeMap<String, f32>,
    pub processing: Option<ProcessingReceipt>,
    pub error: Option<String>,
}
impl Default for CaptureSnapshot {
    fn default() -> Self {
        Self {
            phase: CapturePhase::Starting,
            committed_sample: 0,
            late_samples: 0,
            levels: BTreeMap::new(),
            processing: None,
            error: None,
        }
    }
}

pub(super) struct SourceWriter {
    config: CaptureConfig,
    helper: Helper,
    files: BTreeMap<&'static str, hound::WavWriter<File>>,
    tiles: BTreeMap<u32, Tile>,
    last_packet: [f64; 2],
    pcm_seen: bool,
    stopped: bool,
    write_failed: bool,
    pub snapshot: CaptureSnapshot,
}

#[derive(Default)]
struct Tile {
    microphone: Option<Vec<f32>>,
    system: Option<Vec<f32>>,
    reference: Option<Vec<[f32; 2]>>,
}

impl SourceWriter {
    pub fn create(config: CaptureConfig, helper: Helper) -> Result<Self, CaptureError> {
        if config.job_id.len() != 32
            || !config
                .job_id
                .bytes()
                .all(|b| b.is_ascii_hexdigit() && !b.is_ascii_uppercase())
            || (!config.microphone && !config.system)
        {
            return Err(invalid("Invalid capture recording or source selection."));
        }
        if config.microphone && config.system && !helper.qualified() {
            return Err(invalid(
                "Raw dual-source capture requires the echo canceller; this helper is not qualified.",
            ));
        }
        let metadata = fs::symlink_metadata(&config.folder)?;
        if !metadata.is_dir() || metadata.file_type().is_symlink() {
            return Err(invalid("Capture requires a private recording directory."));
        }
        #[cfg(unix)]
        {
            use std::os::unix::fs::MetadataExt;
            if metadata.uid() != unsafe { libc::getuid() } || metadata.mode() & 0o077 != 0 {
                return Err(invalid(
                    "Capture requires an owned private recording directory.",
                ));
            }
        }
        let mut names = vec![("audio", 1)];
        if config.microphone {
            names.push(("microphone", 1));
        }
        if config.system {
            names.extend([("system", 1), ("system_reference", 2)]);
        }
        if config.microphone && config.system {
            names.push(("microphone_clean", 1));
        }
        // Reject collisions before creating any recording. No previous WAV is overwritten.
        for (name, _) in &names {
            match fs::symlink_metadata(config.folder.join(format!("{name}.wav"))) {
                Err(e) if e.kind() == std::io::ErrorKind::NotFound => (),
                Err(e) => return Err(e.into()),
                Ok(_) => return Err(invalid("Capture recording already exists.")),
            }
        }
        let mut files = BTreeMap::new();
        for (name, channels) in names {
            let file = private_create(&config.folder.join(format!("{name}.wav")))?;
            let mut wav = hound::WavWriter::new(
                file,
                hound::WavSpec {
                    channels,
                    sample_rate: RATE,
                    bits_per_sample: 16,
                    sample_format: hound::SampleFormat::Int,
                },
            )?;
            wav.flush()?;
            files.insert(name, wav);
        }
        let mut snapshot = CaptureSnapshot::default();
        if config.microphone {
            snapshot.levels.insert("microphone".into(), 0.);
        }
        if config.system {
            snapshot.levels.insert("system".into(), 0.);
        }
        Ok(Self {
            config,
            helper,
            files,
            tiles: BTreeMap::new(),
            last_packet: [0.; 2],
            pcm_seen: false,
            stopped: false,
            write_failed: false,
            snapshot,
        })
    }

    pub fn accept(&mut self, event: CaptureEvent) -> Result<(), CaptureError> {
        if self.stopped {
            return Err(invalid("Capture emitted data after its stopped boundary."));
        }
        if self.config.microphone
            && self.config.system
            && self.snapshot.processing.is_none()
            && !matches!(
                event.kind(),
                EventKind::CaptureProcessing
                    | EventKind::CaptureRuntime
                    | EventKind::HelperCheck
                    | EventKind::Error
            )
        {
            return Err(invalid(
                "Capture advanced before the qualified microphone receipt.",
            ));
        }
        let value = event.body();
        match event.kind() {
            EventKind::CaptureProcessing => {
                if !self.config.microphone
                    || !self.config.system
                    || self.pcm_seen
                    || self.snapshot.committed_sample != 0
                    || self.snapshot.processing.is_some()
                {
                    return Err(invalid(
                        "Capture processing crossed its recording boundary.",
                    ));
                }
                self.snapshot.processing = Some(self.helper.receipt(
                    &self.config.job_id,
                    value,
                    &self.config.folder,
                )?);
            }
            EventKind::Audio => {
                if self.config.microphone
                    && self.config.system
                    && self.snapshot.processing.is_none()
                {
                    return Err(invalid(
                        "Capture PCM arrived before the qualified microphone receipt.",
                    ));
                }
                self.add(value)?;
                self.pcm_seen = true;
            }
            EventKind::Clock => {
                let seconds = seconds(value)?;
                self.flush(seconds, false)?;
                if let Some(message) =
                    disk_guard_message(fs4::available_space(&self.config.folder)?)
                {
                    return Err(invalid(message));
                }
            }
            EventKind::FormatChanged => {
                if self.snapshot.processing.is_some() {
                    return Err(invalid(
                        "Qualified microphone processing changed; restart recording. Audio is preserved.",
                    ));
                }
                self.source(value.get("source").and_then(Value::as_str).unwrap_or(""))?;
                self.position(seconds(value)?)?;
                // Raw single-source has no AEC state to reset; Swift already drained its converter.
            }
            EventKind::Recording => {
                if self.config.microphone
                    && self.config.system
                    && self.snapshot.processing.is_none()
                {
                    return Err(invalid(
                        "Capture started without a qualified microphone receipt.",
                    ));
                }
                self.snapshot.phase = CapturePhase::Recording;
            }
            EventKind::Paused => {
                self.flush(seconds(value)?, true)?;
                if self.snapshot.processing.is_some() {
                    return Err(invalid(
                        "Qualified microphone processing ended at Pause; restart recording. Audio is preserved.",
                    ));
                }
                self.snapshot.phase = CapturePhase::Paused;
            }
            EventKind::Stopped => {
                self.flush(seconds(value)?, true)?;
                self.stopped = true;
                self.snapshot.phase = CapturePhase::Stopping;
            }
            EventKind::Error => {
                return Err(invalid(
                    value
                        .get("error")
                        .and_then(Value::as_str)
                        .unwrap_or("Audio capture failed."),
                ));
            }
            EventKind::CaptureRuntime | EventKind::HelperCheck => (),
        }
        Ok(())
    }

    fn source(&self, source: &str) -> Result<usize, CaptureError> {
        match source {
            "microphone" if self.config.microphone => Ok(0),
            "system" if self.config.system => Ok(1),
            _ => Err(invalid("Invalid capture source.")),
        }
    }

    fn position(&self, seconds: f64) -> Result<u32, CaptureError> {
        if !seconds.is_finite()
            || seconds < 0.
            || seconds * f64::from(RATE) > f64::from(self.max_frames())
        {
            return Err(invalid("Invalid capture timestamp."));
        }
        let position = (seconds * f64::from(RATE)).round_ties_even() as u32;
        if position > self.snapshot.committed_sample.saturating_add(RATE * 5) {
            return Err(invalid(
                "Audio capture timing jumped; recording stopped to preserve its timeline.",
            ));
        }
        Ok(position)
    }

    fn add(&mut self, value: &Value) -> Result<(), CaptureError> {
        #[derive(Deserialize)]
        struct Audio {
            source: String,
            time: f64,
            rate: u32,
            #[serde(default = "mono")]
            channels: usize,
            pcm: String,
        }
        fn mono() -> usize {
            1
        }
        let packet: Audio = serde_json::from_value(value.clone())
            .map_err(|_| invalid("Malformed capture audio packet."))?;
        let source = self.source(&packet.source)?;
        if packet.rate != RATE
            || ![1, 2].contains(&packet.channels)
            || (source == 0 && packet.channels != 1)
            || packet.pcm.len() > (RATE as usize * 2 * 4).div_ceil(3) * 4
        {
            return Err(invalid("Invalid capture audio format or packet size."));
        }
        let bytes = STANDARD
            .decode(packet.pcm)
            .map_err(|_| invalid("Invalid capture PCM encoding."))?;
        if !bytes.len().is_multiple_of(4 * packet.channels)
            || bytes.len() / (4 * packet.channels) > RATE as usize
        {
            return Err(invalid("Incomplete or oversized capture audio frame."));
        }
        let samples: Vec<f32> = bytes
            .as_chunks::<4>()
            .0
            .iter()
            .map(|b| f32::from_le_bytes(*b))
            .collect();
        if samples.iter().any(|sample| !sample.is_finite()) {
            return Err(invalid("Nonfinite capture audio."));
        }
        let frames = samples.len() / packet.channels;
        let mut position = self.position(packet.time)?;
        if position.saturating_add(frames as u32) > self.max_frames() {
            return Err(invalid("Recording exceeds the PCM16 RIFF size limit."));
        }
        let reference: Vec<[f32; 2]> = samples
            .chunks_exact(packet.channels)
            .map(|s| [s[0], s[packet.channels - 1]])
            .collect();
        let mut mono: Vec<f32> = reference
            .iter()
            .map(|v| {
                if packet.channels == 1 {
                    v[0]
                } else {
                    (v[0] + v[1]) * 0.5
                }
            })
            .collect();
        if packet.channels == 2 && frames > 0 {
            let power = reference.iter().fold([0.; 2], |mut p, v| {
                p[0] += f64::from(v[0]).powi(2);
                p[1] += f64::from(v[1]).powi(2);
                p
            });
            let mono_power: f64 = mono.iter().map(|v| f64::from(*v).powi(2)).sum();
            if mono_power < 0.05 * power[0].max(power[1]) {
                let channel = usize::from(power[1] > power[0]);
                mono = reference.iter().map(|v| v[channel]).collect();
            }
        }
        if mono.iter().any(|v| !v.is_finite()) {
            return Err(invalid("Nonfinite downmixed capture audio."));
        }
        let rms = if frames == 0 {
            0.
        } else {
            (mono.iter().map(|v| f64::from(*v).powi(2)).sum::<f64>() / frames as f64)
                .sqrt()
                .min(1.) as f32
        };
        self.snapshot.levels.insert(packet.source, rms);
        self.last_packet[source] =
            self.last_packet[source].max(packet.time + frames as f64 / f64::from(RATE));
        let mut offset = self
            .snapshot
            .committed_sample
            .saturating_sub(position)
            .min(frames as u32) as usize;
        self.snapshot.late_samples += offset as u64;
        position += offset as u32;
        while offset < frames {
            let within = (position % TILE) as usize;
            let count = (TILE as usize - within).min(frames - offset);
            let tile = self.tiles.entry(position / TILE).or_default();
            let buffer = if source == 0 {
                &mut tile.microphone
            } else {
                &mut tile.system
            };
            buffer.get_or_insert_with(|| vec![0.; TILE as usize])[within..within + count]
                .copy_from_slice(&mono[offset..offset + count]);
            if source == 1 {
                tile.reference
                    .get_or_insert_with(|| vec![[0.; 2]; TILE as usize])[within..within + count]
                    .copy_from_slice(&reference[offset..offset + count]);
            }
            position += count as u32;
            offset += count;
        }
        Ok(())
    }

    fn flush(&mut self, seconds: f64, final_tail: bool) -> Result<(), CaptureError> {
        if !seconds.is_finite() || seconds < 0. {
            return Err(invalid("Invalid capture clock."));
        }
        let mut target = self.position(if final_tail {
            seconds
        } else {
            (seconds - 0.5).max(0.)
        })?;
        if !final_tail {
            target = target / TILE * TILE;
        }
        for (index, name) in ["microphone", "system"].into_iter().enumerate() {
            if seconds - self.last_packet[index] > 0.5
                && let Some(level) = self.snapshot.levels.get_mut(name)
            {
                *level = 0.;
            }
        }
        while self.snapshot.committed_sample < target {
            let cursor = self.snapshot.committed_sample;
            let within = (cursor % TILE) as usize;
            let count = (TILE as usize - within).min((target - cursor) as usize);
            let tile = self.tiles.entry(cursor / TILE).or_default();
            let silence = vec![0.; count];
            let microphone = tile
                .microphone
                .as_ref()
                .map(|v| &v[within..within + count])
                .unwrap_or(&silence);
            let system = tile
                .system
                .as_ref()
                .map(|v| &v[within..within + count])
                .unwrap_or(&silence);
            let mixed: Vec<_> = microphone
                .iter()
                .zip(system)
                .map(|(m, s)| (m + s).clamp(-1., 1.))
                .collect();
            let stereo: Vec<_> = tile
                .reference
                .as_ref()
                .map(|v| {
                    v[within..within + count]
                        .iter()
                        .flatten()
                        .copied()
                        .collect()
                })
                .unwrap_or_else(|| vec![0.; count * 2]);
            let result = (|| {
                for (name, wav) in &mut self.files {
                    let samples = match *name {
                        "audio" => &mixed,
                        "microphone" | "microphone_clean" => microphone,
                        "system" => system,
                        "system_reference" => &stereo,
                        _ => unreachable!(),
                    };
                    let bytes = pcm16_bytes(samples).map_err(|e| invalid(e.to_string()))?;
                    let mut block = wav.get_i16_writer((bytes.len() / 2) as u32);
                    for sample in bytes.as_chunks::<2>().0 {
                        block.write_sample(i16::from_le_bytes(*sample));
                    }
                    block.flush()?;
                    wav.flush()?;
                }
                Ok::<_, CaptureError>(())
            })();
            if result.is_err() {
                self.write_failed = true;
            }
            result?;
            self.snapshot.committed_sample += count as u32;
            if within + count == TILE as usize {
                self.tiles.remove(&(cursor / TILE));
            }
        }
        Ok(())
    }

    pub fn finish(mut self) -> CaptureSnapshot {
        // Preserve valid packets still behind the watermark on every failure path.
        if !self.stopped
            && !self.write_failed
            && let Err(error) = self.flush(self.last_packet[0].max(self.last_packet[1]), true)
        {
            self.snapshot.error.get_or_insert(error.to_string());
        }
        for (_, wav) in self.files {
            if let Err(error) = wav.finalize() {
                self.snapshot.error.get_or_insert(error.to_string());
            }
        }
        self.snapshot
    }

    fn max_frames(&self) -> u32 {
        // A system source also retains an interleaved stereo PCM16 reference.
        (u32::MAX - 36) / if self.config.system { 4 } else { 2 }
    }

    #[cfg(test)]
    pub(super) fn make_audio_read_only(&mut self) {
        self.files.remove("audio").unwrap().finalize().unwrap();
        let read_only = File::open(self.config.folder.join("audio.wav")).unwrap();
        self.files
            .insert("audio", hound::WavWriter::new_append(read_only).unwrap());
    }
}

fn seconds(value: &Value) -> Result<f64, CaptureError> {
    value
        .get("time")
        .and_then(Value::as_f64)
        .ok_or_else(|| invalid("Capture event has no timestamp."))
}

pub(super) fn run_writer(
    mut writer: SourceWriter,
    mut input: mpsc::Receiver<CaptureEvent>,
    output: watch::Sender<CaptureSnapshot>,
) -> CaptureSnapshot {
    while let Some(event) = input.blocking_recv() {
        if let Err(error) = writer.accept(event) {
            writer.snapshot.error = Some(error.to_string());
            writer.snapshot.phase = CapturePhase::Stopping;
            output.send_replace(writer.snapshot.clone());
            break;
        }
        output.send_replace(writer.snapshot.clone());
    }
    let snapshot = writer.finish();
    output.send_replace(snapshot.clone());
    snapshot
}
