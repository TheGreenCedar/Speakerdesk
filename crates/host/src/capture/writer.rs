use std::{
    collections::BTreeMap,
    fs::{self, File},
    io::{BufRead, BufReader, Seek, SeekFrom, Write},
    path::PathBuf,
};

use base64::{Engine, engine::general_purpose::STANDARD};
use serde::Deserialize;
use serde_json::Value;
use speakerdesk_core::audio::pcm16_bytes;
use speakerdesk_echo::EchoMixer;
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
    pub echo_library: Option<PathBuf>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CapturePhase {
    Starting,
    Recording,
    Paused,
    Resuming,
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
    /// Original aligned tracks may extend beyond failed or buffered DSP output.
    pub raw_sample: u32,
    pub late_samples: u64,
    /// Formerly committed audio changed. Transcript authority after this point is invalid.
    pub repaired_from_sample: Option<u32>,
    /// Corrected originals were used to rebuild all playable derived media at EOF.
    pub derived_rebuilt: bool,
    pub levels: BTreeMap<String, f32>,
    pub processing: Option<ProcessingReceipt>,
    pub source_catalog: Option<Value>,
    pub error: Option<String>,
}
impl Default for CaptureSnapshot {
    fn default() -> Self {
        Self {
            phase: CapturePhase::Starting,
            committed_sample: 0,
            raw_sample: 0,
            late_samples: 0,
            repaired_from_sample: None,
            derived_rebuilt: false,
            levels: BTreeMap::new(),
            processing: None,
            source_catalog: None,
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
    resume_qualified: bool,
    echo: Option<EchoMixer>,
    clean_failed: bool,
    reset_points: Vec<u32>,
    reset_history: Vec<u32>,
    late_pcm: Option<File>,
    fence_sequence: u64,
    pub snapshot: CaptureSnapshot,
}

#[derive(Default)]
struct Tile {
    microphone: Option<Vec<f32>>,
    system: Option<Vec<f32>>,
    reference: Option<Vec<[f32; 2]>>,
}

impl SourceWriter {
    #[cfg(test)]
    pub(super) fn next_fence_sequence(&self) -> u64 {
        self.fence_sequence + 1
    }
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
        let echo = if config.microphone && config.system && !helper.qualified() {
            let path = config
                .echo_library
                .as_deref()
                .ok_or_else(|| invalid(speakerdesk_core::messages::ECHO_MISSING))?;
            Some(EchoMixer::start(path).map_err(|e| invalid(e.to_string()))?)
        } else {
            None
        };
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
        if echo.is_some() {
            snapshot.source_catalog = Some(super::processing::source_catalog(&config.job_id));
        }
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
            resume_qualified: false,
            echo,
            clean_failed: false,
            reset_points: Vec::new(),
            reset_history: Vec::new(),
            late_pcm: None,
            fence_sequence: 0,
            snapshot,
        })
    }

    pub fn accept(&mut self, event: CaptureEvent) -> Result<(), CaptureError> {
        if self.stopped {
            return Err(invalid("Capture emitted data after its stopped boundary."));
        }
        if self.config.microphone
            && self.config.system
            && self.echo.is_none()
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
                if self.snapshot.phase == CapturePhase::Resuming {
                    let initial = self
                        .snapshot
                        .processing
                        .as_ref()
                        .ok_or_else(|| invalid("Unexpected resumed processing receipt."))?;
                    if self.resume_qualified || initial.metadata.get("producer") != Some(value) {
                        return Err(invalid(
                            "Resumed microphone qualification differs from the retained capture graph.",
                        ));
                    }
                    // Exact equality includes the already validated job, helper, route,
                    // formats and readbacks. Keep the original on-disk receipt/catalog.
                    self.resume_qualified = true;
                    return Ok(());
                }
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
                self.snapshot.source_catalog = self
                    .snapshot
                    .processing
                    .as_ref()
                    .map(|receipt| receipt.catalog.clone());
            }
            EventKind::Audio => {
                self.require_active_input()?;
                if self.config.microphone
                    && self.config.system
                    && self.echo.is_none()
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
                if self.snapshot.phase == CapturePhase::Paused
                    || (self.snapshot.phase == CapturePhase::Resuming && !self.resume_qualified)
                {
                    return Ok(());
                }
                let seconds = seconds(value)?;
                if self.helper.qualified() {
                    if value.get("producer_fence").and_then(Value::as_str)
                        != Some(super::processing::PROGRESS)
                    {
                        return Err(invalid("Capture clock is not a drained producer fence."));
                    }
                    let sequence = value
                        .get("sequence")
                        .and_then(Value::as_u64)
                        .filter(|sequence| *sequence > self.fence_sequence)
                        .ok_or_else(|| invalid("Capture producer fence is stale or unordered."))?;
                    self.fence_sequence = sequence;
                }
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
                if self.echo.is_some() {
                    let point = self
                        .position(seconds(value)?)?
                        .max(self.snapshot.raw_sample);
                    if !self.reset_points.contains(&point) {
                        if self.reset_points.len() >= 8 {
                            return Err(invalid(
                                "Audio format changed too frequently. Original audio is preserved.",
                            ));
                        }
                        self.reset_points.push(point);
                        self.reset_history.push(point);
                        self.reset_points.sort_unstable();
                    }
                }
            }
            EventKind::Recording => {
                self.require_active_input()?;
                if self.config.microphone
                    && self.config.system
                    && self.echo.is_none()
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

    pub(super) fn resume(&mut self) -> Result<(), CaptureError> {
        if self.snapshot.phase != CapturePhase::Paused {
            return Err(invalid(
                "Capture can resume only from an acknowledged Pause.",
            ));
        }
        self.resume_qualified = self.snapshot.processing.is_none();
        self.snapshot.phase = CapturePhase::Resuming;
        Ok(())
    }

    fn require_active_input(&self) -> Result<(), CaptureError> {
        if self.snapshot.phase == CapturePhase::Paused
            || (self.snapshot.phase == CapturePhase::Resuming && !self.resume_qualified)
        {
            return Err(invalid(
                "Capture advanced without an owned Resume and fresh microphone qualification.",
            ));
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
        if position > self.snapshot.raw_sample.saturating_add(RATE * 5) {
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
            .raw_sample
            .saturating_sub(position)
            .min(frames as u32) as usize;
        self.snapshot.late_samples += offset as u64;
        if offset > 0 {
            // Journal the validated original before doing anything to a committed prefix.
            // Replaying this journal at EOF repairs source WAVs, then rebuilds derived audio.
            if self.late_pcm.is_none() {
                self.late_pcm = Some(private_create(
                    &self.config.folder.join("capture-late-pcm.jsonl"),
                )?);
            }
            let original = if source == 0 {
                mono.clone()
            } else {
                reference.iter().flatten().copied().collect()
            };
            let original_bytes =
                pcm16_bytes(&original).map_err(|error| invalid(error.to_string()))?;
            let mono_bytes = pcm16_bytes(&mono).map_err(|error| invalid(error.to_string()))?;
            let entry = serde_json::json!({"source":source,"start_sample":position,"channels":if source==0 {1}else{2},
                "pcm16":STANDARD.encode(original_bytes),"mono_pcm16":STANDARD.encode(mono_bytes)});
            let journal = self.late_pcm.as_mut().expect("late PCM journal");
            serde_json::to_writer(&mut *journal, &entry).map_err(|e| invalid(e.to_string()))?;
            journal.write_all(b"\n")?;
            journal.sync_data()?;
            let affected = position / TILE * TILE;
            let cutoff = self
                .snapshot
                .repaired_from_sample
                .map_or(affected, |old| old.min(affected));
            self.snapshot.repaired_from_sample = Some(cutoff);
            self.snapshot.committed_sample = self.snapshot.committed_sample.min(cutoff);
            self.clean_failed = true;
            self.snapshot.error.get_or_insert_with(|| "Delayed capture PCM changed saved audio. Recording stopped; originals are retained and playable audio is being rebuilt. Transcript after the affected boundary requires review.".into());
        }
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
        while self.snapshot.raw_sample < target {
            self.reset_at_cursor()?;
            let cursor = self.snapshot.raw_sample;
            let within = (cursor % TILE) as usize;
            let mut count = (TILE as usize - within).min((target - cursor) as usize);
            if let Some(point) = self.reset_points.first() {
                count = count.min((*point - cursor) as usize);
            }
            let tile = self.tiles.entry(cursor / TILE).or_default();
            let microphone = tile
                .microphone
                .as_ref()
                .map(|v| v[within..within + count].to_vec())
                .unwrap_or_else(|| vec![0.; count]);
            let system = tile
                .system
                .as_ref()
                .map(|v| v[within..within + count].to_vec())
                .unwrap_or_else(|| vec![0.; count]);
            let reference = tile
                .reference
                .as_ref()
                .map(|v| v[within..within + count].to_vec())
                .unwrap_or_else(|| vec![[0.; 2]; count]);
            let stereo: Vec<_> = reference.iter().flatten().copied().collect();
            // Raw tracks commit first, including when DSP or its clean output fails.
            let raw_result = (|| {
                for (name, samples) in [
                    ("microphone", microphone.as_slice()),
                    ("system", system.as_slice()),
                    ("system_reference", stereo.as_slice()),
                ] {
                    if let Some(wav) = self.files.get_mut(name) {
                        write_samples(wav, samples)?;
                    }
                }
                Ok::<_, CaptureError>(())
            })();
            if raw_result.is_err() {
                self.write_failed = true;
            }
            raw_result?;
            self.snapshot.raw_sample += count as u32;
            if within + count == TILE as usize {
                self.tiles.remove(&(cursor / TILE));
            }
            if !self.clean_failed {
                let clean_result = if let Some(echo) = &mut self.echo {
                    echo.add(&microphone, &system, &reference, false)
                        .map_err(|e| invalid(e.to_string()))
                        .and_then(|out| self.write_clean(&out.clean, &out.system))
                } else {
                    self.write_clean(&microphone, &system)
                };
                if clean_result.is_err() {
                    self.clean_failed = true;
                }
                clean_result?;
            }
        }
        self.reset_at_cursor()?;
        if final_tail {
            self.drain_echo()?;
        }
        Ok(())
    }

    fn write_clean(&mut self, clean: &[f32], system: &[f32]) -> Result<(), CaptureError> {
        let mixed: Vec<_> = clean
            .iter()
            .zip(system)
            .map(|(m, s)| (m + s).clamp(-1., 1.))
            .collect();
        if let Some(wav) = self.files.get_mut("microphone_clean") {
            write_samples(wav, clean)?;
        }
        write_samples(self.files.get_mut("audio").expect("mix"), &mixed)?;
        self.snapshot.committed_sample += clean.len() as u32;
        Ok(())
    }

    fn drain_echo(&mut self) -> Result<(), CaptureError> {
        if self.clean_failed {
            return Ok(());
        }
        if let Some(echo) = &mut self.echo {
            let result = echo
                .add(&[], &[], &[], true)
                .map_err(|e| invalid(e.to_string()))
                .and_then(|out| self.write_clean(&out.clean, &out.system));
            if result.is_err() {
                self.clean_failed = true;
            }
            result?;
        }
        Ok(())
    }

    fn reset_at_cursor(&mut self) -> Result<(), CaptureError> {
        while self
            .reset_points
            .first()
            .is_some_and(|point| *point <= self.snapshot.raw_sample)
        {
            self.reset_points.remove(0);
            self.drain_echo()?;
        }
        Ok(())
    }

    pub fn finish(mut self) -> CaptureSnapshot {
        // Preserve valid packets still behind the watermark on every failure path.
        if (!self.stopped || self.clean_failed)
            && !self.write_failed
            && let Err(error) = self.flush(self.last_packet[0].max(self.last_packet[1]), true)
        {
            self.snapshot.error.get_or_insert(error.to_string());
        }
        for (_, wav) in std::mem::take(&mut self.files) {
            if let Err(error) = wav.finalize() {
                self.snapshot.error.get_or_insert(error.to_string());
            }
        }
        if self.snapshot.repaired_from_sample.is_some() {
            let repaired = if self.write_failed {
                Err(invalid("Original capture WAV writes failed."))
            } else {
                self.repair_originals()
                    .and_then(|()| self.rebuild_derived())
            };
            match repaired {
                Ok(()) => self.snapshot.derived_rebuilt = true,
                Err(error) => {
                    self.snapshot.error = Some(format!(
                        "Capture audio repair did not finish: {error}. Original packets are retained in capture-late-pcm.jsonl."
                    ));
                    // Never leave stale processed audio looking complete.
                    let cutoff = self.snapshot.repaired_from_sample.unwrap();
                    for name in ["audio", "microphone_clean"] {
                        if self.config.folder.join(format!("{name}.wav")).exists() {
                            if let Err(error) = truncate_wav(
                                &self.config.folder.join(format!("{name}.wav")),
                                cutoff,
                            ) {
                                self.snapshot.error = Some(format!(
                                    "Capture audio repair and invalidation failed: {error}. Original packets are retained."
                                ));
                            }
                        }
                    }
                }
            }
        }
        self.snapshot
    }

    fn repair_originals(&mut self) -> Result<(), CaptureError> {
        if let Some(journal) = self.late_pcm.take() {
            journal.sync_all()?;
        }
        let journal = BufReader::new(File::open(
            self.config.folder.join("capture-late-pcm.jsonl"),
        )?);
        for line in journal.lines() {
            let entry: Value = serde_json::from_str(&line?).map_err(|e| invalid(e.to_string()))?;
            let source = entry["source"]
                .as_u64()
                .ok_or_else(|| invalid("Invalid late PCM source."))?;
            let start = entry["start_sample"]
                .as_u64()
                .ok_or_else(|| invalid("Invalid late PCM clock."))? as u32;
            let bytes = STANDARD
                .decode(
                    entry["pcm16"]
                        .as_str()
                        .ok_or_else(|| invalid("Invalid late PCM journal."))?,
                )
                .map_err(|e| invalid(e.to_string()))?;
            if source == 0 {
                patch_wav(&self.config.folder.join("microphone.wav"), start, &bytes)?;
            } else {
                patch_wav(
                    &self.config.folder.join("system_reference.wav"),
                    start,
                    &bytes,
                )?;
                // Preserve exactly the original float-domain downmix, not an i16-domain remix.
                let mono = STANDARD
                    .decode(
                        entry["mono_pcm16"]
                            .as_str()
                            .ok_or_else(|| invalid("Missing late system downmix."))?,
                    )
                    .map_err(|e| invalid(e.to_string()))?;
                patch_wav(&self.config.folder.join("system.wav"), start, &mono)?;
            }
        }
        Ok(())
    }

    fn rebuild_derived(&mut self) -> Result<(), CaptureError> {
        let mut mic = self
            .config
            .microphone
            .then(|| hound::WavReader::open(self.config.folder.join("microphone.wav")))
            .transpose()?;
        let mut system = self
            .config
            .system
            .then(|| hound::WavReader::open(self.config.folder.join("system.wav")))
            .transpose()?;
        let mut reference = self
            .config
            .system
            .then(|| hound::WavReader::open(self.config.folder.join("system_reference.wav")))
            .transpose()?;
        let mut echo = if self.echo.is_some() {
            Some(
                EchoMixer::start(
                    self.config
                        .echo_library
                        .as_deref()
                        .ok_or_else(|| invalid("Missing echo library during repair."))?,
                )
                .map_err(|e| invalid(e.to_string()))?,
            )
        } else {
            None
        };
        let spec = hound::WavSpec {
            channels: 1,
            sample_rate: RATE,
            bits_per_sample: 16,
            sample_format: hound::SampleFormat::Int,
        };
        let audio_path = self.config.folder.join("audio.repair.wav");
        let clean_path = self.config.folder.join("microphone_clean.repair.wav");
        let mut audio = hound::WavWriter::new(private_create(&audio_path)?, spec)?;
        let mut clean = (self.config.microphone && self.config.system)
            .then(|| {
                private_create(&clean_path)
                    .and_then(|file| hound::WavWriter::new(file, spec).map_err(CaptureError::from))
            })
            .transpose()?;
        let mut committed = 0;
        let mut cursor = 0;
        let mut resets = self.reset_history.iter().copied().peekable();
        while cursor < self.snapshot.raw_sample {
            if resets.peek().is_some_and(|point| *point == cursor) {
                if let Some(echo) = &mut echo {
                    let out = echo
                        .add(&[], &[], &[], true)
                        .map_err(|e| invalid(e.to_string()))?;
                    write_rebuilt(&mut audio, clean.as_mut(), &out.clean, &out.system)?;
                    committed += out.clean.len() as u32;
                }
                resets.next();
            }
            let end = (cursor + TILE)
                .min(self.snapshot.raw_sample)
                .min(resets.peek().copied().unwrap_or(u32::MAX));
            let count = (end - cursor) as usize;
            let m = read_pcm16(mic.as_mut(), count)?;
            let s = read_pcm16(system.as_mut(), count)?;
            let r = read_pcm16(reference.as_mut(), count * 2)?;
            let stereo: Vec<[f32; 2]> = r.chunks_exact(2).map(|v| [v[0], v[1]]).collect();
            if let Some(echo) = &mut echo {
                let out = echo
                    .add(&m, &s, &stereo, false)
                    .map_err(|e| invalid(e.to_string()))?;
                write_rebuilt(&mut audio, clean.as_mut(), &out.clean, &out.system)?;
                committed += out.clean.len() as u32;
            } else {
                write_rebuilt(&mut audio, clean.as_mut(), &m, &s)?;
                committed += count as u32;
            }
            cursor = end;
        }
        if let Some(echo) = &mut echo {
            let out = echo
                .add(&[], &[], &[], true)
                .map_err(|e| invalid(e.to_string()))?;
            write_rebuilt(&mut audio, clean.as_mut(), &out.clean, &out.system)?;
            committed += out.clean.len() as u32;
        }
        if committed != self.snapshot.raw_sample {
            return Err(invalid("Rebuilt capture audio has incomplete coverage."));
        }
        audio.finalize()?;
        if let Some(clean) = clean {
            clean.finalize()?;
        }
        fs::rename(audio_path, self.config.folder.join("audio.wav"))?;
        if self.config.microphone && self.config.system {
            fs::rename(clean_path, self.config.folder.join("microphone_clean.wav"))?;
        }
        self.snapshot.committed_sample = committed;
        Ok(())
    }

    fn max_frames(&self) -> u32 {
        // A system source also retains an interleaved stereo PCM16 reference.
        (u32::MAX - 36) / if self.config.system { 4 } else { 2 }
    }

    #[cfg(test)]
    pub(super) fn make_audio_read_only(&mut self) {
        self.make_file_read_only("audio");
    }
    #[cfg(test)]
    pub(super) fn make_file_read_only(&mut self, name: &'static str) {
        self.files.remove(name).unwrap().finalize().unwrap();
        let read_only = File::open(self.config.folder.join(format!("{name}.wav"))).unwrap();
        self.files
            .insert(name, hound::WavWriter::new_append(read_only).unwrap());
    }
}

fn read_pcm16(
    reader: Option<&mut hound::WavReader<BufReader<File>>>,
    count: usize,
) -> Result<Vec<f32>, CaptureError> {
    let Some(reader) = reader else {
        return Ok(vec![0.; count]);
    };
    let values: Vec<f32> = reader
        .samples::<i16>()
        .take(count)
        .map(|sample| sample.map(|v| f32::from(v) / 32768.))
        .collect::<Result<_, _>>()?;
    if values.len() != count {
        return Err(invalid("Original capture source is incomplete."));
    }
    Ok(values)
}

fn write_rebuilt(
    audio: &mut hound::WavWriter<File>,
    clean: Option<&mut hound::WavWriter<File>>,
    m: &[f32],
    s: &[f32],
) -> Result<(), CaptureError> {
    if m.len() != s.len() {
        return Err(invalid("Rebuilt source clocks differ."));
    }
    if let Some(clean) = clean {
        write_samples(clean, m)?;
    }
    let mixed: Vec<f32> = m
        .iter()
        .zip(s)
        .map(|(m, s)| (m + s).clamp(-1., 1.))
        .collect();
    write_samples(audio, &mixed)
}

fn wav_data(path: &std::path::Path) -> Result<(File, u64, u16, u32), CaptureError> {
    let file = std::fs::OpenOptions::new()
        .read(true)
        .write(true)
        .open(path)?;
    let reader = hound::WavReader::new(file)?;
    let spec = reader.spec();
    if spec.bits_per_sample != 16 || spec.sample_format != hound::SampleFormat::Int {
        return Err(invalid("Unexpected capture WAV format."));
    }
    let frames = reader.duration();
    let mut file = reader.into_inner();
    let offset = file.stream_position()?;
    Ok((file, offset, spec.channels, frames))
}

fn patch_wav(path: &std::path::Path, start: u32, bytes: &[u8]) -> Result<(), CaptureError> {
    let (mut file, offset, channels, frames) = wav_data(path)?;
    let frame_bytes = u64::from(channels) * 2;
    if !bytes.len().is_multiple_of(frame_bytes as usize)
        || u64::from(start) + bytes.len() as u64 / frame_bytes > u64::from(frames)
    {
        return Err(invalid("Late PCM exceeds retained original audio."));
    }
    file.seek(SeekFrom::Start(offset + u64::from(start) * frame_bytes))?;
    file.write_all(bytes)?;
    file.sync_all()?;
    Ok(())
}

fn truncate_wav(path: &std::path::Path, through: u32) -> Result<(), CaptureError> {
    let (mut file, offset, channels, frames) = wav_data(path)?;
    let size = through.min(frames) * u32::from(channels) * 2;
    let total = offset + u64::from(size);
    file.set_len(total)?;
    file.seek(SeekFrom::Start(offset - 4))?;
    file.write_all(&size.to_le_bytes())?;
    file.seek(SeekFrom::Start(4))?;
    file.write_all(
        &u32::try_from(total - 8)
            .map_err(|_| invalid("Invalid RIFF repair size."))?
            .to_le_bytes(),
    )?;
    file.sync_all()?;
    Ok(())
}

fn write_samples(wav: &mut hound::WavWriter<File>, samples: &[f32]) -> Result<(), CaptureError> {
    let bytes = pcm16_bytes(samples).map_err(|e| invalid(e.to_string()))?;
    let mut block = wav.get_i16_writer((bytes.len() / 2) as u32);
    for sample in bytes.as_chunks::<2>().0 {
        block.write_sample(i16::from_le_bytes(*sample));
    }
    block.flush()?;
    wav.flush()?;
    Ok(())
}

fn seconds(value: &Value) -> Result<f64, CaptureError> {
    value
        .get("time")
        .and_then(Value::as_f64)
        .ok_or_else(|| invalid("Capture event has no timestamp."))
}

pub(super) enum WriterInput {
    Event(CaptureEvent),
    Resume,
}

pub(super) fn run_writer(
    mut writer: SourceWriter,
    mut input: mpsc::Receiver<WriterInput>,
    output: watch::Sender<CaptureSnapshot>,
) -> CaptureSnapshot {
    while let Some(event) = input.blocking_recv() {
        let result = match event {
            WriterInput::Event(event) => writer.accept(event),
            WriterInput::Resume => writer.resume(),
        };
        if let Err(error) = result {
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
