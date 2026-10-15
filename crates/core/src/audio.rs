//! Decode, resample, mix, and recording writes.
//!
//! Canonical input is 16 kHz mono PCM16. Anything else is decoded with symphonia,
//! mixed to mono, and resampled with rubato. macOS uses its system decoder only
//! when symphonia reports an unsupported format or codec.

use std::fs::{self, File, OpenOptions};
use std::io::{BufReader, BufWriter, Read, Seek, SeekFrom, Write};
use std::path::Path;
use std::sync::Mutex;

#[cfg(test)]
use hound::WavSpec;
use hound::{SampleFormat, WavReader, WavWriter};
use rubato::audioadapter_buffers::direct::InterleavedSlice;
use rubato::{Async, FixedAsync, Resampler, SincInterpolationParameters, WindowFunction};
use symphonia::core::audio::GenericAudioBufferRef;
use symphonia::core::codecs::CodecParameters;
use symphonia::core::codecs::audio::AudioDecoderOptions;
use symphonia::core::errors::Error as SymphoniaError;
use symphonia::core::formats::probe::Hint;
use symphonia::core::formats::{FormatOptions, TrackType};
use symphonia::core::io::MediaSourceStream;
use symphonia::core::meta::MetadataOptions;
use thiserror::Error;

use crate::limits::SAMPLE_RATE;

/// Python `audio.MAX_DURATION`: imported audio is limited to two hours.
pub const MAX_DURATION_SECONDS: f64 = 7200.0;

const MAX_FRAMES: u32 = (MAX_DURATION_SECONDS as u32) * SAMPLE_RATE;
// afconvert uses a 4096-byte WAV header. Allow bounded ancillary chunks.
#[cfg(target_os = "macos")]
const MAX_APPLE_WAV_BYTES: u64 = MAX_FRAMES as u64 * 2 + 65536;
/// Mono PCM16 frames whose RIFF size (36 + two bytes per frame) fits u32.
/// Live capture is independent of the two-hour import normalization policy.
pub const MAX_RECORDING_FRAMES: u32 = (u32::MAX - 36) / 2;
const HEADER_LEN: u64 = 44;

const EXPECTED_PCM: &str = "Expected 16 kHz mono 16-bit PCM WAV.";
const DURATION_LIMIT: &str = "Recordings must be nonempty and no longer than two hours.";
const RECORDING_LIMIT: &str = "Recording exceeds the PCM16 RIFF size limit.";
const TRUNCATED: &str = "The PCM recording is truncated.";
const INCOMPLETE_SAMPLE: &str = "The PCM recording has an incomplete sample.";
const DECODE_PCM: &str = "The file could not be decoded as PCM audio.";
const DECODE_RECORDING: &str = "Could not decode this recording. Supply a supported audio file.";
const FINITE_MONO: &str = "Capture requires finite mono audio.";

/// PCM and recording failures. Validation uses the Python `audio.py` messages.
#[derive(Debug, Error)]
pub enum AudioError {
    #[error("{0}")]
    Message(&'static str),
    #[error(transparent)]
    Io(#[from] std::io::Error),
}

impl AudioError {
    fn message(text: &'static str) -> Self {
        Self::Message(text)
    }
}

/// Quantize finite mono samples the way `pcm16_bytes` does.
///
/// PCM16 decoding divides by 32768. NumPy `rint` rounds half to even, then the
/// value is clipped to the signed 16-bit range and stored little-endian.
pub fn pcm16_bytes(audio: &[f32]) -> Result<Vec<u8>, AudioError> {
    if audio.iter().any(|sample| !sample.is_finite()) {
        return Err(AudioError::message(FINITE_MONO));
    }
    let mut bytes = Vec::with_capacity(audio.len().saturating_mul(2));
    for sample in audio {
        bytes.extend_from_slice(&quantize_pcm16(*sample).to_le_bytes());
    }
    Ok(bytes)
}

fn quantize_pcm16(sample: f32) -> i16 {
    let rounded = (f64::from(sample) * 32768.0).round_ties_even();
    let clipped = rounded.clamp(-32768.0, 32767.0);
    clipped as i16
}

/// Duration of a canonical 16 kHz mono PCM16 WAV, after reading every sample.
pub fn wav_duration(path: impl AsRef<Path>) -> Result<f64, AudioError> {
    wav_duration_cancellable(path.as_ref(), &|| false)
}

fn wav_duration_cancellable(path: &Path, cancelled: &dyn Fn() -> bool) -> Result<f64, AudioError> {
    check_cancelled(cancelled)?;
    let mut reader = WavReader::open(path).map_err(map_open)?;
    let spec = reader.spec();
    if spec.channels != 1
        || spec.bits_per_sample != 16
        || spec.sample_rate != SAMPLE_RATE
        || spec.sample_format != SampleFormat::Int
    {
        return Err(AudioError::message(EXPECTED_PCM));
    }
    let frames = reader.duration();
    let duration = f64::from(frames) / f64::from(SAMPLE_RATE);
    if !(duration > 0.0 && duration <= MAX_DURATION_SECONDS) {
        return Err(AudioError::message(DURATION_LIMIT));
    }
    for (index, sample) in reader.samples::<i16>().enumerate() {
        if index % 8192 == 0 {
            check_cancelled(cancelled)?;
        }
        sample.map_err(map_sample)?;
    }
    check_cancelled(cancelled)?;
    Ok(duration)
}

/// Read a normalized recording for offline inference; callers own crop limits.
pub fn read_pcm16(path: impl AsRef<Path>) -> Result<Vec<f32>, AudioError> {
    let mut reader = WavReader::open(path).map_err(map_open)?;
    let spec = reader.spec();
    if spec.channels != 1
        || spec.bits_per_sample != 16
        || spec.sample_rate != SAMPLE_RATE
        || spec.sample_format != SampleFormat::Int
    {
        return Err(AudioError::message(EXPECTED_PCM));
    }
    if reader.duration() == 0 || reader.duration() > MAX_FRAMES {
        return Err(AudioError::message(DURATION_LIMIT));
    }
    reader
        .samples::<i16>()
        .map(|sample| sample.map(|s| f32::from(s) / 32768.0).map_err(map_sample))
        .collect()
}

/// Read only the requested saved-audio range, rejecting incomplete or noncanonical PCM.
pub fn read_pcm16_range(
    path: impl AsRef<Path>,
    start: f64,
    end: f64,
) -> Result<Vec<f32>, AudioError> {
    let mut reader = WavReader::open(path).map_err(map_open)?;
    let spec = reader.spec();
    if spec.channels != 1
        || spec.bits_per_sample != 16
        || spec.sample_rate != SAMPLE_RATE
        || spec.sample_format != SampleFormat::Int
    {
        return Err(AudioError::message(EXPECTED_PCM));
    }
    let start = round_frames(start, SAMPLE_RATE)?;
    let end = round_frames(end, SAMPLE_RATE)?;
    if start >= end || end > reader.duration() {
        return Err(AudioError::message(TRUNCATED));
    }
    reader.seek(start)?;
    let pcm = reader
        .samples::<i16>()
        .take((end - start) as usize)
        .map(|sample| sample.map(|s| f32::from(s) / 32768.).map_err(map_sample))
        .collect::<Result<Vec<_>, _>>()?;
    if pcm.len() != (end - start) as usize {
        return Err(AudioError::message(TRUNCATED));
    }
    Ok(pcm)
}

/// Copy a canonical WAV, or decode and resample anything else to 16 kHz mono PCM16.
pub fn normalize(
    source: impl AsRef<Path>,
    destination: impl AsRef<Path>,
) -> Result<f64, AudioError> {
    normalize_cancellable(source, destination, &|| false)
}

/// A cancelled preparation never installs an incomplete recording or touches its source.
pub fn normalize_cancellable(
    source: impl AsRef<Path>,
    destination: impl AsRef<Path>,
    cancelled: &dyn Fn() -> bool,
) -> Result<f64, AudioError> {
    let source = source.as_ref();
    let destination = destination.as_ref();
    check_cancelled(cancelled)?;
    let parent = destination
        .parent()
        .filter(|p| !p.as_os_str().is_empty())
        .unwrap_or_else(|| Path::new("."));
    let staged = parent.join(format!(".speakerdesk-audio-{}.part", uuid::Uuid::new_v4()));
    let decoded_path = parent.join(format!(
        ".speakerdesk-audio-{}.decode",
        uuid::Uuid::new_v4()
    ));
    let result = (|| {
        match wav_duration_cancellable(source, cancelled) {
            Ok(duration) => {
                if source == destination {
                    return Ok(duration);
                }
                let mut input = File::open(source)?;
                let mut output = OpenOptions::new()
                    .write(true)
                    .create_new(true)
                    .open(&staged)?;
                let mut buffer = [0_u8; 65536];
                loop {
                    check_cancelled(cancelled)?;
                    let count = input.read(&mut buffer)?;
                    if count == 0 {
                        break;
                    }
                    output.write_all(&buffer[..count])?;
                }
                output.sync_all()?;
                check_cancelled(cancelled)?;
                fs::rename(&staged, destination)?;
                return Ok(duration);
            }
            Err(AudioError::Message(DURATION_LIMIT)) => {
                return Err(AudioError::message(DURATION_LIMIT));
            }
            Err(AudioError::Message(CANCELLED)) => return Err(AudioError::message(CANCELLED)),
            Err(_) => {}
        }
        let decoded_file = OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&decoded_path)?;
        let mut decoded = BufWriter::new(decoded_file);
        let outcome = decode_mono_cancellable(source, cancelled, &mut |chunk| {
            for part in chunk.chunks(8192) {
                check_cancelled(cancelled)?;
                let bytes: Vec<u8> = part
                    .iter()
                    .flat_map(|sample| sample.to_le_bytes())
                    .collect();
                decoded.write_all(&bytes)?;
            }
            Ok(())
        })?;
        decoded.flush()?;
        drop(decoded);
        let Some((rate, samples)) = outcome else {
            #[cfg(target_os = "macos")]
            return normalize_apple_cancellable(source, destination, cancelled);
            #[cfg(not(target_os = "macos"))]
            return Err(AudioError::message(DECODE_RECORDING));
        };
        let mut decoded = BufReader::new(File::open(&decoded_path)?);
        resample_to_wav(samples, rate, &staged, cancelled, &mut |count| {
            let mut bytes = vec![0; count * 4];
            decoded.read_exact(&mut bytes)?;
            Ok(bytes
                .as_chunks::<4>()
                .0
                .iter()
                .map(|sample| f32::from_le_bytes(*sample))
                .collect())
        })?;
        let duration = wav_duration_cancellable(&staged, cancelled)?;
        File::open(&staged)?.sync_all()?;
        check_cancelled(cancelled)?;
        fs::rename(&staged, destination)?;
        Ok(duration)
    })();
    let _ = fs::remove_file(staged);
    let _ = fs::remove_file(decoded_path);
    result
}

const CANCELLED: &str = "Audio preparation cancelled.";
fn check_cancelled(cancelled: &dyn Fn() -> bool) -> Result<(), AudioError> {
    if cancelled() {
        Err(AudioError::message(CANCELLED))
    } else {
        Ok(())
    }
}

/// Copy a sample range. `start` and `end` are seconds, rounded half-to-even like Python `round`.
pub fn crop(
    source: impl AsRef<Path>,
    destination: impl AsRef<Path>,
    start: f64,
    end: f64,
) -> Result<(), AudioError> {
    let mut reader = WavReader::open(source).map_err(map_open)?;
    let spec = reader.spec();
    if spec.bits_per_sample != 16 || spec.sample_format != SampleFormat::Int {
        return Err(AudioError::message(DECODE_PCM));
    }
    let start_frame = round_frames(start, spec.sample_rate)?;
    let count = round_frames(end - start, spec.sample_rate)?;
    if start_frame > reader.duration() {
        return Err(AudioError::message(DECODE_PCM));
    }
    reader.seek(start_frame)?;
    let mut writer = WavWriter::create(destination, spec).map_err(map_open)?;
    let want = (count as usize).saturating_mul(usize::from(spec.channels));
    for sample in reader.samples::<i16>().take(want) {
        writer
            .write_sample(sample.map_err(map_sample)?)
            .map_err(map_open)?;
    }
    writer.finalize().map_err(map_open)?;
    Ok(())
}

fn round_frames(seconds: f64, rate: u32) -> Result<u32, AudioError> {
    if !seconds.is_finite() {
        return Err(AudioError::message(DECODE_PCM));
    }
    let frames = (seconds * f64::from(rate)).round_ties_even();
    if !(0.0..=f64::from(u32::MAX)).contains(&frames) {
        return Err(AudioError::message(DECODE_PCM));
    }
    Ok(frames as u32)
}

fn advance_frames(current: u32, add: u32) -> Result<u32, AudioError> {
    let next = current
        .checked_add(add)
        .ok_or(AudioError::message(RECORDING_LIMIT))?;
    if next > MAX_RECORDING_FRAMES {
        Err(AudioError::message(RECORDING_LIMIT))
    } else {
        Ok(next)
    }
}

/// A recording opened once; the writer's notices authorize reads of committed samples.
pub struct RecordingReader {
    file: Mutex<File>,
    data_offset: u64,
    declared_samples: u32,
}

impl RecordingReader {
    pub fn open(path: impl AsRef<Path>) -> Result<Self, AudioError> {
        let reader = WavReader::new(File::open(path)?).map_err(map_open)?;
        let spec = reader.spec();
        if spec.channels != 1
            || spec.bits_per_sample != 16
            || spec.sample_rate != SAMPLE_RATE
            || spec.sample_format != SampleFormat::Int
        {
            return Err(AudioError::message(EXPECTED_PCM));
        }
        let declared_samples = reader.duration();
        let mut file = reader.into_inner();
        let data_offset = file.stream_position()?;
        Ok(Self {
            file: Mutex::new(file),
            data_offset,
            declared_samples,
        })
    }

    /// Immutable imported audio is bounded by its WAV header and physical data.
    pub fn import_samples(&self) -> Result<u32, AudioError> {
        if self.declared_samples == 0 || self.declared_samples > MAX_FRAMES {
            return Err(AudioError::message(DURATION_LIMIT));
        }
        self.require_samples(self.declared_samples)?;
        Ok(self.declared_samples)
    }

    pub fn read_samples(
        &self,
        start: u32,
        end: u32,
        committed: u32,
    ) -> Result<Vec<f32>, AudioError> {
        if start >= end || end > committed || committed > MAX_RECORDING_FRAMES {
            return Err(AudioError::message(TRUNCATED));
        }
        let mut bytes = vec![0; (end - start) as usize * 2];
        let mut file = self
            .file
            .lock()
            .map_err(|_| AudioError::message(DECODE_PCM))?;
        file.seek(SeekFrom::Start(self.data_offset + u64::from(start) * 2))?;
        file.read_exact(&mut bytes)?;
        Ok(bytes
            .as_chunks::<2>()
            .0
            .iter()
            .map(|pair| f32::from(i16::from_le_bytes(*pair)) / 32768.)
            .collect())
    }

    /// Check the writer's announced endpoint before advancing any model session.
    pub fn require_samples(&self, committed: u32) -> Result<(), AudioError> {
        if committed > MAX_RECORDING_FRAMES
            || self
                .file
                .lock()
                .map_err(|_| AudioError::message(DECODE_PCM))?
                .metadata()?
                .len()
                < self.data_offset + u64::from(committed) * 2
        {
            return Err(AudioError::message(TRUNCATED));
        }
        Ok(())
    }
}

/// Growing 16 kHz mono PCM16 WAV. The header is rewritten and flushed after every block.
pub struct RecordingWriter {
    file: File,
    frames: u32,
}

impl RecordingWriter {
    pub fn create(path: impl AsRef<Path>) -> Result<Self, AudioError> {
        let mut file = OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .truncate(true)
            .open(path)?;
        file.write_all(&wav_header(0))?;
        file.flush()?;
        Ok(Self { file, frames: 0 })
    }

    pub fn frames(&self) -> u32 {
        self.frames
    }

    /// Append little-endian PCM16 and flush the RIFF and data sizes.
    pub fn write_block(&mut self, pcm_le: &[u8]) -> Result<(), AudioError> {
        if !pcm_le.len().is_multiple_of(2) {
            return Err(AudioError::message(INCOMPLETE_SAMPLE));
        }
        let frames =
            u32::try_from(pcm_le.len() / 2).map_err(|_| AudioError::message(DURATION_LIMIT))?;
        let next = advance_frames(self.frames, frames)?;
        let offset = HEADER_LEN + u64::from(self.frames) * 2;
        write_at(&self.file, pcm_le, offset)?;
        self.frames = next;
        self.flush_header()?;
        Ok(())
    }

    /// Quantize float samples and append them as one block.
    pub fn write_samples(&mut self, samples: &[f32]) -> Result<(), AudioError> {
        self.write_block(&pcm16_bytes(samples)?)
    }

    /// Read `length` bytes at `offset` without moving the file cursor.
    pub fn read_range(&self, offset: u64, length: usize) -> Result<Vec<u8>, AudioError> {
        let mut buf = vec![0_u8; length];
        read_exact_at(&self.file, &mut buf, offset)?;
        Ok(buf)
    }

    fn flush_header(&mut self) -> Result<(), AudioError> {
        let data_bytes = self.frames * 2;
        write_at(&self.file, &(36 + data_bytes).to_le_bytes(), 4)?;
        write_at(&self.file, &data_bytes.to_le_bytes(), 40)?;
        self.file.flush()?;
        Ok(())
    }
}

fn wav_header(data_bytes: u32) -> [u8; 44] {
    let mut header = [0_u8; 44];
    header[0..4].copy_from_slice(b"RIFF");
    header[4..8].copy_from_slice(&(36 + data_bytes).to_le_bytes());
    header[8..12].copy_from_slice(b"WAVE");
    header[12..16].copy_from_slice(b"fmt ");
    header[16..20].copy_from_slice(&16_u32.to_le_bytes());
    header[20..22].copy_from_slice(&1_u16.to_le_bytes());
    header[22..24].copy_from_slice(&1_u16.to_le_bytes());
    header[24..28].copy_from_slice(&SAMPLE_RATE.to_le_bytes());
    header[28..32].copy_from_slice(&(SAMPLE_RATE * 2).to_le_bytes());
    header[32..34].copy_from_slice(&2_u16.to_le_bytes());
    header[34..36].copy_from_slice(&16_u16.to_le_bytes());
    header[36..40].copy_from_slice(b"data");
    header[40..44].copy_from_slice(&data_bytes.to_le_bytes());
    header
}

fn write_at(file: &File, bytes: &[u8], offset: u64) -> Result<(), AudioError> {
    #[cfg(unix)]
    {
        use std::os::unix::fs::FileExt;
        file.write_all_at(bytes, offset)?;
        Ok(())
    }
    #[cfg(not(unix))]
    {
        let _ = (file, bytes, offset);
        Err(AudioError::Io(std::io::Error::new(
            std::io::ErrorKind::Unsupported,
            "positioned WAV writes require a Unix file",
        )))
    }
}

fn read_exact_at(file: &File, buf: &mut [u8], offset: u64) -> Result<(), AudioError> {
    #[cfg(unix)]
    {
        use std::os::unix::fs::FileExt;
        file.read_exact_at(buf, offset)?;
        Ok(())
    }
    #[cfg(not(unix))]
    {
        let _ = (file, buf, offset);
        Err(AudioError::Io(std::io::Error::new(
            std::io::ErrorKind::Unsupported,
            "positioned WAV reads require a Unix file",
        )))
    }
}

fn decode_mono_cancellable(
    path: &Path,
    cancelled: &dyn Fn() -> bool,
    output: &mut dyn FnMut(&[f32]) -> Result<(), AudioError>,
) -> Result<Option<(u32, usize)>, AudioError> {
    let file = File::open(path)?;
    let source = MediaSourceStream::new(Box::new(file), Default::default());
    let mut hint = Hint::new();
    if let Some(extension) = path.extension().and_then(|extension| extension.to_str()) {
        hint.with_extension(extension);
    }
    let mut format = match symphonia::default::get_probe().probe(
        &hint,
        source,
        FormatOptions::default(),
        MetadataOptions::default(),
    ) {
        Ok(format) => format,
        Err(SymphoniaError::Unsupported(_)) => return Ok(None),
        Err(_) => return Err(AudioError::message(DECODE_RECORDING)),
    };
    let track = format
        .default_track(TrackType::Audio)
        .or_else(|| {
            format
                .tracks()
                .iter()
                .find(|track| matches!(track.codec_params, Some(CodecParameters::Audio(_))))
        })
        .ok_or(AudioError::message(DECODE_RECORDING))?;
    let track_id = track.id;
    let params = match &track.codec_params {
        Some(CodecParameters::Audio(params)) => params.clone(),
        _ => return Err(AudioError::message(DECODE_RECORDING)),
    };
    if let (Some(frames), Some(rate)) = (track.num_frames, params.sample_rate)
        && (rate == 0 || frames as f64 / f64::from(rate) > MAX_DURATION_SECONDS)
    {
        return Err(AudioError::message(DURATION_LIMIT));
    }
    let mut decoder = match symphonia::default::get_codecs()
        .make_audio_decoder(&params, &AudioDecoderOptions::default())
    {
        Ok(decoder) => decoder,
        Err(SymphoniaError::Unsupported(_)) => return Ok(None),
        Err(_) => return Err(AudioError::message(DECODE_RECORDING)),
    };
    let mut rate = params.sample_rate.filter(|rate| *rate > 0);
    let mut mono = Vec::new();
    let mut samples = 0;
    loop {
        check_cancelled(cancelled)?;
        let packet = match format.next_packet() {
            Ok(Some(packet)) => packet,
            Ok(None) => break,
            Err(SymphoniaError::IoError(err))
                if err.kind() == std::io::ErrorKind::UnexpectedEof =>
            {
                break;
            }
            Err(_) => return Err(AudioError::message(DECODE_RECORDING)),
        };
        if packet.track_id != track_id {
            continue;
        }
        let decoded = decoder
            .decode(&packet)
            .map_err(|_| AudioError::message(DECODE_RECORDING))?;
        let spec_rate = decoded.spec().rate();
        if spec_rate == 0 {
            return Err(AudioError::message(DECODE_RECORDING));
        }
        match rate {
            None => rate = Some(spec_rate),
            Some(existing) if existing != spec_rate => {
                return Err(AudioError::message(DECODE_RECORDING));
            }
            Some(_) => {}
        }
        mono.clear();
        append_mono(&mut mono, &decoded)?;
        samples += mono.len();
        if samples as f64 / f64::from(spec_rate) > MAX_DURATION_SECONDS {
            return Err(AudioError::message(DURATION_LIMIT));
        }
        output(&mono)?;
    }
    let rate = rate.ok_or(AudioError::message(DECODE_RECORDING))?;
    Ok(Some((rate, samples)))
}

#[cfg(target_os = "macos")]
struct AppleConversion {
    directory: std::path::PathBuf,
    child: Option<std::process::Child>,
    deadline: std::time::Instant,
}

#[cfg(target_os = "macos")]
impl Drop for AppleConversion {
    fn drop(&mut self) {
        if let Some(child) = &mut self.child {
            let _ = child.kill();
            let _ = child.wait();
        }
        let _ = fs::remove_dir_all(&self.directory);
    }
}

#[cfg(target_os = "macos")]
impl AppleConversion {
    fn wait_cancellable(&mut self, cancelled: &dyn Fn() -> bool) -> Result<(), AudioError> {
        let output = self.directory.join("audio.wav");
        let child = self.child.as_mut().expect("conversion child");
        loop {
            check_cancelled(cancelled)?;
            if fs::metadata(&output).is_ok_and(|meta| meta.len() > MAX_APPLE_WAV_BYTES) {
                return Err(AudioError::message(DURATION_LIMIT));
            }
            if let Some(status) = child.try_wait()? {
                if !status.success() {
                    return Err(AudioError::message(DECODE_RECORDING));
                }
                if fs::metadata(&output)?.len() > MAX_APPLE_WAV_BYTES {
                    return Err(AudioError::message(DURATION_LIMIT));
                }
                return Ok(());
            }
            if std::time::Instant::now() >= self.deadline {
                return Err(AudioError::message(
                    "Audio conversion exceeded three minutes.",
                ));
            }
            std::thread::sleep(std::time::Duration::from_millis(20));
        }
    }
}

#[cfg(target_os = "macos")]
fn normalize_apple_cancellable(
    source: &Path,
    destination: &Path,
    cancelled: &dyn Fn() -> bool,
) -> Result<f64, AudioError> {
    use std::os::unix::fs::DirBuilderExt;
    use std::process::{Command, Stdio};
    let source = fs::canonicalize(source)?;
    let parent = destination
        .parent()
        .filter(|parent| !parent.as_os_str().is_empty())
        .unwrap_or_else(|| Path::new("."));
    let directory =
        fs::canonicalize(parent)?.join(format!(".speakerdesk-audio-{}", uuid::Uuid::new_v4()));
    fs::DirBuilder::new().mode(0o700).create(&directory)?;
    let mut conversion = AppleConversion {
        directory,
        child: None,
        deadline: std::time::Instant::now() + std::time::Duration::from_secs(180),
    };
    let output = conversion.directory.join("audio.wav");
    conversion.child = Some(
        Command::new("/usr/bin/afconvert")
            .arg(source)
            .arg(&output)
            .args(["-f", "WAVE", "-d", "LEI16@16000", "-c", "1"])
            .env_clear()
            .stdin(Stdio::null())
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .spawn()?,
    );
    conversion.wait_cancellable(cancelled)?;
    let duration = wav_duration_cancellable(&output, cancelled)?;
    File::open(&output)?.sync_all()?;
    check_cancelled(cancelled)?;
    fs::rename(&output, destination)?;
    Ok(duration)
}

fn append_mono(
    destination: &mut Vec<f32>,
    decoded: &GenericAudioBufferRef<'_>,
) -> Result<(), AudioError> {
    let channels = decoded.num_planes();
    let frames = decoded.frames();
    if frames == 0 {
        return Ok(());
    }
    if channels == 0 {
        return Err(AudioError::message(DECODE_RECORDING));
    }
    let mut interleaved = vec![0.0_f32; channels * frames];
    decoded.copy_to_slice_interleaved(&mut interleaved);
    if interleaved.iter().any(|sample| !sample.is_finite()) {
        return Err(AudioError::message(FINITE_MONO));
    }
    if channels == 1 {
        destination.extend_from_slice(&interleaved);
        return Ok(());
    }
    let scale = 1.0 / channels as f32;
    for frame in interleaved.chunks_exact(channels) {
        let sum: f32 = frame.iter().copied().sum();
        destination.push(sum * scale);
    }
    Ok(())
}

fn resample_to_wav(
    input_len: usize,
    from_rate: u32,
    destination: &Path,
    cancelled: &dyn Fn() -> bool,
    read: &mut dyn FnMut(usize) -> Result<Vec<f32>, AudioError>,
) -> Result<(), AudioError> {
    if from_rate == 0 || input_len == 0 {
        return Err(AudioError::message(DECODE_RECORDING));
    }
    let ratio = f64::from(SAMPLE_RATE) / f64::from(from_rate);
    let mut resampler = Async::<f32>::new_sinc(
        ratio,
        1.0,
        &SincInterpolationParameters::new(256, WindowFunction::BlackmanHarris2),
        1024,
        1,
        FixedAsync::Input,
    )
    .map_err(|_| AudioError::message(DECODE_RECORDING))?;
    let expected = (ratio * input_len as f64).ceil() as usize;
    if expected == 0 || expected > MAX_FRAMES as usize {
        return Err(AudioError::message(DURATION_LIMIT));
    }
    let mut writer = RecordingWriter::create(destination)?;
    let mut offset = 0;
    let mut trim = resampler.output_delay();
    let mut written = 0;
    while offset < input_len || written < expected {
        check_cancelled(cancelled)?;
        let count = resampler.input_frames_next().min(input_len - offset);
        let input = read(count)?;
        if input.len() != count {
            return Err(AudioError::message(TRUNCATED));
        }
        let adapter = InterleavedSlice::new(&input, 1, input.len())
            .map_err(|_| AudioError::message(DECODE_RECORDING))?;
        let indexing = rubato::Indexing {
            input_offset: 0,
            output_offset: 0,
            partial_len: Some(count),
            active_channels_mask: None,
        };
        let output = resampler
            .process(&adapter, Some(&indexing))
            .map_err(|_| AudioError::message(DECODE_RECORDING))?
            .take_data();
        offset += count;
        let skipped = trim.min(output.len());
        trim -= skipped;
        let stop = (skipped + expected - written).min(output.len());
        if stop > skipped {
            writer.write_samples(&output[skipped..stop])?;
            written += stop - skipped;
        }
    }
    check_cancelled(cancelled)?;
    Ok(())
}

fn map_open(err: hound::Error) -> AudioError {
    match err {
        hound::Error::FormatError("data chunk length is not a multiple of sample size")
        | hound::Error::FormatError("invalid data chunk length") => {
            AudioError::message(INCOMPLETE_SAMPLE)
        }
        hound::Error::IoError(err) if err.kind() == std::io::ErrorKind::UnexpectedEof => {
            AudioError::message(DECODE_PCM)
        }
        hound::Error::IoError(err) => AudioError::Io(err),
        _ => AudioError::message(DECODE_PCM),
    }
}

fn map_sample(err: hound::Error) -> AudioError {
    match err {
        hound::Error::IoError(_) => AudioError::message(TRUNCATED),
        _ => AudioError::message(DECODE_PCM),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::{Seek, SeekFrom};

    fn message(error: AudioError) -> &'static str {
        match error {
            AudioError::Message(text) => text,
            AudioError::Io(err) => panic!("unexpected io error: {err}"),
        }
    }

    #[test]
    fn cancelled_normalization_preserves_source_destination_and_removes_staging() {
        use std::cell::Cell;
        for canonical in [true, false] {
            let dir = std::env::temp_dir()
                .join(format!("speakerdesk-cancel-audio-{}", uuid::Uuid::new_v4()));
            fs::create_dir_all(&dir).unwrap();
            let source = dir.join(if canonical {
                "source.wav"
            } else {
                "source.flac"
            });
            if canonical {
                let mut writer = RecordingWriter::create(&source).unwrap();
                writer.write_samples(&vec![0.1; 100_000]).unwrap();
            } else {
                fs::write(&source, include_bytes!("../tests/testdata/stereo-48k.flac")).unwrap();
            }
            let original = fs::read(&source).unwrap();
            let destination = dir.join("existing.wav");
            fs::write(&destination, b"previous output").unwrap();
            let checks = Cell::new(0);
            let error = normalize_cancellable(&source, &destination, &|| {
                let next = checks.get() + 1;
                checks.set(next);
                next >= if canonical { 17 } else { 3 }
            })
            .unwrap_err();
            assert_eq!(message(error), CANCELLED);
            assert_eq!(fs::read(&source).unwrap(), original);
            assert_eq!(fs::read(&destination).unwrap(), b"previous output");
            assert_eq!(fs::read_dir(&dir).unwrap().count(), 2);
            fs::remove_dir_all(dir).unwrap();
        }
    }

    #[test]
    fn resampling_checks_cancellation_between_bounded_chunks() {
        use std::cell::Cell;
        let path = std::env::temp_dir().join(format!(
            "speakerdesk-cancel-resample-{}.wav",
            uuid::Uuid::new_v4()
        ));
        let checks = Cell::new(0);
        let input = vec![0.1; 48_000];
        let mut offset = 0;
        let error = resample_to_wav(
            input.len(),
            48_000,
            &path,
            &|| {
                let next = checks.get() + 1;
                checks.set(next);
                next >= 4
            },
            &mut |count| {
                let chunk = input[offset..offset + count].to_vec();
                offset += count;
                Ok(chunk)
            },
        )
        .unwrap_err();
        assert_eq!(message(error), CANCELLED);
        let reader = RecordingReader::open(&path).unwrap();
        assert!(reader.import_samples().unwrap() < 16_000);
        fs::remove_file(path).unwrap();
    }

    #[test]
    fn pcm16_matches_python_rounding_and_saturation() {
        let samples: Vec<i16> = (-32768..=32767).map(|sample| sample as i16).collect();
        let decoded: Vec<f32> = samples
            .iter()
            .map(|sample| *sample as f32 / 32768.0)
            .collect();
        let bytes = pcm16_bytes(&decoded).unwrap();
        assert_eq!(i16_le(&bytes), samples);

        let saturated = pcm16_bytes(&[-2.0, -1.0, 0.0, 1.0, 2.0]).unwrap();
        assert_eq!(i16_le(&saturated), vec![-32768, -32768, 0, 32767, 32767]);

        let ties = [
            (0.5_f64 / 32768.0) as f32,
            (1.5 / 32768.0) as f32,
            (2.5 / 32768.0) as f32,
            (-0.5 / 32768.0) as f32,
            (-1.5 / 32768.0) as f32,
            (-2.5 / 32768.0) as f32,
        ];
        let ties = pcm16_bytes(&ties).unwrap();
        assert_eq!(i16_le(&ties), vec![0, 2, 2, 0, -2, -2]);

        assert_eq!(message(pcm16_bytes(&[f32::NAN]).unwrap_err()), FINITE_MONO);
        assert_eq!(
            message(pcm16_bytes(&[f32::INFINITY]).unwrap_err()),
            FINITE_MONO
        );
    }

    #[test]
    fn resident_reader_observes_only_committed_crops_of_a_growing_recording() {
        let dir =
            std::env::temp_dir().join(format!("speakerdesk-growing-{}", uuid::Uuid::new_v4()));
        fs::create_dir_all(&dir).unwrap();
        let path = dir.join("recording.wav");
        let mut writer = super::RecordingWriter::create(&path).unwrap();
        let reader = super::RecordingReader::open(&path).unwrap();
        assert!(reader.read_samples(0, 1, 0).is_err());
        writer.write_samples(&[0.25, -0.5, 0.75]).unwrap();
        assert_eq!(reader.read_samples(0, 2, 2).unwrap(), [0.25, -0.5]);
        assert!(
            reader.read_samples(0, 3, 2).is_err(),
            "unannounced samples are unavailable"
        );
        writer.write_samples(&[-1., 0.]).unwrap();
        assert_eq!(reader.read_samples(2, 5, 5).unwrap(), [0.75, -1., 0.]);
        assert_eq!(
            reader.read_samples(0, 1, 5).unwrap(),
            [0.25],
            "old crops remain available"
        );
        fs::OpenOptions::new()
            .write(true)
            .open(&path)
            .unwrap()
            .set_len(48)
            .unwrap();
        assert!(
            reader.read_samples(2, 5, 5).is_err(),
            "truncation must surface"
        );
        let stereo = dir.join("stereo.wav");
        write_i16(&stereo, 2, SAMPLE_RATE, &[1, 2]);
        assert!(super::RecordingReader::open(stereo).is_err());
        fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn canonical_wav_is_copied_and_checked() {
        let dir =
            std::env::temp_dir().join(format!("speakerdesk-audio-fast-{}", std::process::id()));
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(&dir).unwrap();
        let source = dir.join("source.wav");
        let destination = dir.join("copy.wav");
        let samples = [-32768_i16, -1, 0, 1, 32767];
        write_i16(&source, 1, SAMPLE_RATE, &samples);
        let duration = normalize(&source, &destination).unwrap();
        assert_eq!(duration, samples.len() as f64 / f64::from(SAMPLE_RATE));
        assert_eq!(fs::read(&source).unwrap(), fs::read(&destination).unwrap());
        assert_eq!(
            super::read_pcm16_range(&source, 1. / 16000., 4. / 16000.).unwrap(),
            vec![-1. / 32768., 0., 1. / 32768.]
        );
        for (start, end) in [
            (-1., 1.),
            (0., f64::NAN),
            (4. / 16000., 6. / 16000.),
            (1., 0.),
        ] {
            assert!(super::read_pcm16_range(&source, start, end).is_err());
        }
        fs::OpenOptions::new()
            .write(true)
            .open(&destination)
            .unwrap()
            .set_len(48)
            .unwrap();
        assert!(super::read_pcm16_range(&destination, 0., 5. / 16000.).is_err());

        let stereo = dir.join("stereo.wav");
        write_i16(&stereo, 2, SAMPLE_RATE, &[1000, -1000]);
        assert!(super::read_pcm16_range(&stereo, 0., 1. / 16000.).is_err());
        assert_eq!(message(wav_duration(&stereo).unwrap_err()), EXPECTED_PCM);

        let empty = dir.join("empty.wav");
        write_i16(&empty, 1, SAMPLE_RATE, &[]);
        assert_eq!(message(wav_duration(&empty).unwrap_err()), DURATION_LIMIT);

        let mut header = wav_header(4);
        header[40..44].copy_from_slice(&4_u32.to_le_bytes());
        let truncated = dir.join("truncated.wav");
        fs::write(&truncated, [&header[..], &1_i16.to_le_bytes()].concat()).unwrap();
        assert_eq!(message(wav_duration(&truncated).unwrap_err()), TRUNCATED);

        let mut odd = wav_header(1);
        odd[4..8].copy_from_slice(&(36_u32 + 1).to_le_bytes());
        let incomplete = dir.join("odd.wav");
        fs::write(&incomplete, [&odd[..], &[0_u8]].concat()).unwrap();
        assert_eq!(
            message(wav_duration(&incomplete).unwrap_err()),
            INCOMPLETE_SAMPLE
        );

        let mut too_long = wav_header(0);
        let frames = MAX_FRAMES + SAMPLE_RATE;
        let data_bytes = frames.saturating_mul(2);
        too_long[4..8].copy_from_slice(&(36 + data_bytes).to_le_bytes());
        too_long[40..44].copy_from_slice(&data_bytes.to_le_bytes());
        let long = dir.join("long.wav");
        fs::write(&long, too_long).unwrap();
        assert_eq!(message(wav_duration(&long).unwrap_err()), DURATION_LIMIT);

        fs::write(dir.join("notes.txt"), b"not audio").unwrap();
        assert_eq!(
            message(normalize(dir.join("notes.txt"), dir.join("out.wav")).unwrap_err()),
            DECODE_RECORDING
        );
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    #[cfg(target_os = "macos")]
    fn macos_normalizes_ogg_opus_with_os_decoder() {
        let dir =
            std::env::temp_dir().join(format!("speakerdesk-audio-opus-{}", uuid::Uuid::new_v4()));
        fs::create_dir_all(&dir).unwrap();
        let source = dir.join("source.ogg");
        fs::write(
            &source,
            include_bytes!("../tests/testdata/authored-tone.ogg"),
        )
        .unwrap();
        let destination = dir.join("audio.wav");
        let duration = normalize(&source, &destination).unwrap();
        assert_eq!(duration, 16006. / f64::from(SAMPLE_RATE));
        let pcm = read_pcm16(&destination).unwrap();
        assert_eq!(pcm.len(), 16006);
        let rms = (pcm.iter().map(|sample| sample * sample).sum::<f32>() / pcm.len() as f32).sqrt();
        assert!((0.14..0.18).contains(&rms), "{rms}");
        assert!(pcm.iter().any(|sample| *sample > 0.1));
        assert!(pcm.iter().any(|sample| *sample < -0.1));
        assert_eq!(fs::read_dir(&dir).unwrap().count(), 2);
        fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    #[cfg(target_os = "macos")]
    fn apple_conversion_limits_kill_and_reap_the_child_and_remove_partial_output() {
        use std::process::{Command, Stdio};
        for (oversized, completed, cancelled) in [
            (false, false, false),
            (true, false, false),
            (true, true, false),
            (false, false, true),
        ] {
            let directory = std::env::temp_dir()
                .join(format!("speakerdesk-audio-bound-{}", uuid::Uuid::new_v4()));
            fs::create_dir_all(&directory).unwrap();
            let partial = File::create(directory.join("audio.wav")).unwrap();
            partial
                .set_len(if oversized {
                    MAX_APPLE_WAV_BYTES + 1
                } else {
                    1
                })
                .unwrap();
            drop(partial);
            let mut child = Command::new("/bin/sleep")
                .arg(if completed { "0" } else { "30" })
                .env_clear()
                .stdin(Stdio::null())
                .stdout(Stdio::null())
                .stderr(Stdio::null())
                .spawn()
                .unwrap();
            if completed {
                assert!(child.wait().unwrap().success());
            }
            let pid = child.id();
            let mut conversion = AppleConversion {
                directory: directory.clone(),
                child: Some(child),
                deadline: std::time::Instant::now()
                    + std::time::Duration::from_secs(if oversized { 180 } else { 0 }),
            };
            assert_eq!(
                message(conversion.wait_cancellable(&|| cancelled).unwrap_err()),
                if cancelled {
                    CANCELLED
                } else if oversized {
                    DURATION_LIMIT
                } else {
                    "Audio conversion exceeded three minutes."
                }
            );
            drop(conversion);
            assert!(!directory.exists());
            let mut status = 0;
            assert_eq!(
                unsafe { libc::waitpid(pid as i32, &mut status, libc::WNOHANG) },
                -1
            );
            assert_eq!(
                std::io::Error::last_os_error().raw_os_error(),
                Some(libc::ECHILD)
            );
        }
    }

    #[test]
    fn normalization_errors_preserve_the_existing_destination() {
        let dir =
            std::env::temp_dir().join(format!("speakerdesk-audio-errors-{}", uuid::Uuid::new_v4()));
        fs::create_dir_all(&dir).unwrap();
        let destination = dir.join("audio.wav");
        write_i16(&destination, 1, SAMPLE_RATE, &[1000, -1000]);
        let original = fs::read(&destination).unwrap();

        let nonfinite = dir.join("nonfinite.wav");
        let mut writer = WavWriter::create(
            &nonfinite,
            WavSpec {
                channels: 1,
                sample_rate: SAMPLE_RATE,
                bits_per_sample: 32,
                sample_format: SampleFormat::Float,
            },
        )
        .unwrap();
        writer.write_sample(f32::NAN).unwrap();
        writer.finalize().unwrap();
        assert_eq!(
            message(normalize(&nonfinite, &destination).unwrap_err()),
            FINITE_MONO
        );
        assert_eq!(fs::read(&destination).unwrap(), original);

        let invalid = dir.join("invalid.ogg");
        fs::write(&invalid, b"not an audio container").unwrap();
        assert_eq!(
            message(normalize(&invalid, &destination).unwrap_err()),
            DECODE_RECORDING
        );
        assert_eq!(fs::read(&destination).unwrap(), original);
        assert_eq!(fs::read_dir(&dir).unwrap().count(), 3);
        fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn symphonia_and_rubato_normalize_wav_and_flac() {
        let dir =
            std::env::temp_dir().join(format!("speakerdesk-audio-decode-{}", std::process::id()));
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(&dir).unwrap();
        let frames = 4800_usize;
        let wav = dir.join("tone.wav");
        let mut samples = Vec::with_capacity(frames * 2);
        for _ in 0..frames {
            samples.push(16_384);
            samples.push(0);
        }
        write_i16(&wav, 2, 48_000, &samples);
        let normalized = dir.join("tone-16k.wav");
        let duration = normalize(&wav, &normalized).unwrap();
        assert!((duration - 0.1).abs() < 1.0 / f64::from(SAMPLE_RATE));
        let reader = WavReader::open(&normalized).unwrap();
        assert_eq!(reader.spec().channels, 1);
        assert_eq!(reader.spec().sample_rate, SAMPLE_RATE);
        assert_eq!(reader.spec().bits_per_sample, 16);
        let pcm: Vec<i16> = reader
            .into_samples::<i16>()
            .map(|sample| sample.unwrap())
            .collect();
        let middle = &pcm[pcm.len() / 4..pcm.len() * 3 / 4];
        assert!(middle.iter().all(|sample| (*sample - 8192).abs() <= 8));

        let flac = dir.join("tone.flac");
        fs::write(&flac, include_bytes!("../tests/testdata/stereo-48k.flac")).unwrap();
        let flac_out = dir.join("flac-16k.wav");
        normalize(&flac, &flac_out).unwrap();
        let flac_pcm: Vec<i16> = WavReader::open(&flac_out)
            .unwrap()
            .into_samples::<i16>()
            .map(|sample| sample.unwrap())
            .collect();
        assert_eq!(flac_pcm, pcm);
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn recording_writer_flushes_header_and_reads_a_byte_range() {
        let dir =
            std::env::temp_dir().join(format!("speakerdesk-audio-writer-{}", std::process::id()));
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(&dir).unwrap();
        let path = dir.join("live.wav");
        let mut writer = RecordingWriter::create(&path).unwrap();
        writer.file.seek(SeekFrom::Start(0)).unwrap();
        writer.write_block(&1_i16.to_le_bytes()).unwrap();
        assert_eq!(writer.file.stream_position().unwrap(), 0);
        let size = writer.read_range(40, 4).unwrap();
        assert_eq!(u32::from_le_bytes(size.try_into().unwrap()), 2);
        assert_eq!(writer.file.stream_position().unwrap(), 0);
        let seen = fs::read(&path).unwrap();
        assert_eq!(&seen[36..40], b"data");
        assert_eq!(u32::from_le_bytes(seen[40..44].try_into().unwrap()), 2);

        writer
            .write_samples(&[0.5 / 32768.0, 1.5 / 32768.0])
            .unwrap();
        assert_eq!(writer.frames(), 3);
        let samples = writer.read_range(HEADER_LEN, 6).unwrap();
        assert_eq!(samples, [1, 0, 0, 0, 2, 0]);
        assert_eq!(wav_duration(&path).unwrap(), 3.0 / f64::from(SAMPLE_RATE));
        assert_eq!(
            message(writer.write_block(&[0]).unwrap_err()),
            INCOMPLETE_SAMPLE
        );
        assert_eq!(
            message(advance_frames(MAX_RECORDING_FRAMES, 1).unwrap_err()),
            RECORDING_LIMIT
        );

        let cropped = dir.join("crop.wav");
        crop(
            &path,
            &cropped,
            1.5 / f64::from(SAMPLE_RATE),
            2.5 / f64::from(SAMPLE_RATE),
        )
        .unwrap();
        let cropped: Vec<i16> = WavReader::open(&cropped)
            .unwrap()
            .into_samples::<i16>()
            .map(|sample| sample.unwrap())
            .collect();
        assert_eq!(cropped, vec![2]);
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn live_recording_reads_beyond_import_limit_without_retaining_pcm() {
        let dir =
            std::env::temp_dir().join(format!("speakerdesk-sparse-live-{}", std::process::id()));
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(&dir).unwrap();
        let path = dir.join("audio.wav");
        let mut writer = RecordingWriter::create(&path).unwrap();
        writer.frames = MAX_FRAMES + 17;
        writer
            .file
            .set_len(HEADER_LEN + u64::from(writer.frames) * 2)
            .unwrap();
        writer.write_samples(&[0.25]).unwrap();
        let reader = RecordingReader::open(&path).unwrap();
        reader.require_samples(writer.frames()).unwrap();
        assert_eq!(
            reader
                .read_samples(MAX_FRAMES + 17, MAX_FRAMES + 18, writer.frames())
                .unwrap(),
            vec![0.25]
        );
        assert!(
            wav_duration(&path).is_err(),
            "import duration policy remains separate"
        );
        assert!(reader.require_samples(MAX_RECORDING_FRAMES + 1).is_err());
        assert_eq!(
            advance_frames(MAX_RECORDING_FRAMES - 1, 1).unwrap(),
            MAX_RECORDING_FRAMES
        );
        assert!(advance_frames(MAX_RECORDING_FRAMES, 1).is_err());
        assert!(advance_frames(u32::MAX, 1).is_err());
        drop(reader);
        drop(writer);
        fs::remove_dir_all(dir).unwrap();
    }

    fn i16_le(bytes: &[u8]) -> Vec<i16> {
        bytes
            .as_chunks::<2>()
            .0
            .iter()
            .map(|chunk| i16::from_le_bytes(*chunk))
            .collect()
    }

    fn write_i16(path: &Path, channels: u16, sample_rate: u32, samples: &[i16]) {
        let spec = WavSpec {
            channels,
            sample_rate,
            bits_per_sample: 16,
            sample_format: SampleFormat::Int,
        };
        let mut writer = WavWriter::create(path, spec).unwrap();
        for sample in samples {
            writer.write_sample(*sample).unwrap();
        }
        writer.finalize().unwrap();
    }
}
