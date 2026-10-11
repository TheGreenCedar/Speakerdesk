//! Decode, resample, mix, and recording writes. Owned by the core-audio agent.
//!
//! Canonical input is 16 kHz mono PCM16. Anything else is decoded with symphonia,
//! mixed to mono, and resampled with rubato. macOS `afconvert` is not available here.

use std::fs::{self, File, OpenOptions};
use std::io::{Read, Seek, SeekFrom, Write};
use std::path::Path;
use std::sync::Mutex;

use hound::{SampleFormat, WavReader, WavSpec, WavWriter};
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

/// Python `audio.MAX_DURATION`: recordings must be nonempty and at most two hours.
pub const MAX_DURATION_SECONDS: f64 = 7200.0;

const MAX_FRAMES: u32 = (MAX_DURATION_SECONDS as u32) * SAMPLE_RATE;
const HEADER_LEN: u64 = 44;

const EXPECTED_PCM: &str = "Expected 16 kHz mono 16-bit PCM WAV.";
const DURATION_LIMIT: &str = "Recordings must be nonempty and no longer than two hours.";
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
    for sample in reader.samples::<i16>() {
        sample.map_err(map_sample)?;
    }
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
    let source = source.as_ref();
    let destination = destination.as_ref();
    match wav_duration(source) {
        Ok(_) => {
            if source != destination {
                fs::copy(source, destination)?;
            }
            return wav_duration(destination);
        }
        Err(AudioError::Message(DURATION_LIMIT)) => {
            return Err(AudioError::message(DURATION_LIMIT));
        }
        Err(_) => {}
    }
    let (rate, mono) = decode_mono(source)?;
    let resampled = if mono.is_empty() {
        Vec::new()
    } else {
        resample_mono(&mono, rate)?
    };
    write_pcm16_wav(destination, &resampled)?;
    wav_duration(destination)
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
        .ok_or(AudioError::message(DURATION_LIMIT))?;
    if next > MAX_FRAMES {
        Err(AudioError::message(DURATION_LIMIT))
    } else {
        Ok(next)
    }
}

/// A recording opened once; the writer's notices authorize reads of committed samples.
pub struct RecordingReader {
    file: Mutex<File>,
    data_offset: u64,
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
        let mut file = reader.into_inner();
        let data_offset = file.stream_position()?;
        Ok(Self {
            file: Mutex::new(file),
            data_offset,
        })
    }

    pub fn read_samples(
        &self,
        start: u32,
        end: u32,
        committed: u32,
    ) -> Result<Vec<f32>, AudioError> {
        if start >= end || end > committed || committed > MAX_FRAMES {
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
        if committed > MAX_FRAMES
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

fn write_pcm16_wav(path: &Path, samples: &[f32]) -> Result<(), AudioError> {
    let spec = WavSpec {
        channels: 1,
        sample_rate: SAMPLE_RATE,
        bits_per_sample: 16,
        sample_format: SampleFormat::Int,
    };
    let mut writer = WavWriter::create(path, spec).map_err(map_open)?;
    for sample in samples {
        if !sample.is_finite() {
            return Err(AudioError::message(FINITE_MONO));
        }
        writer
            .write_sample(quantize_pcm16(*sample))
            .map_err(map_open)?;
    }
    writer.finalize().map_err(map_open)?;
    Ok(())
}

fn decode_mono(path: &Path) -> Result<(u32, Vec<f32>), AudioError> {
    let file = File::open(path)?;
    let source = MediaSourceStream::new(Box::new(file), Default::default());
    let mut hint = Hint::new();
    if let Some(extension) = path.extension().and_then(|extension| extension.to_str()) {
        hint.with_extension(extension);
    }
    let mut format = symphonia::default::get_probe()
        .probe(
            &hint,
            source,
            FormatOptions::default(),
            MetadataOptions::default(),
        )
        .map_err(|_| AudioError::message(DECODE_RECORDING))?;
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
    let mut decoder = symphonia::default::get_codecs()
        .make_audio_decoder(&params, &AudioDecoderOptions::default())
        .map_err(|_| AudioError::message(DECODE_RECORDING))?;
    let mut rate = params.sample_rate.filter(|rate| *rate > 0);
    let mut mono = Vec::new();
    loop {
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
        append_mono(&mut mono, &decoded)?;
    }
    let rate = rate.ok_or(AudioError::message(DECODE_RECORDING))?;
    Ok((rate, mono))
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

fn resample_mono(input: &[f32], from_rate: u32) -> Result<Vec<f32>, AudioError> {
    if from_rate == 0 || input.is_empty() {
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
    let adapter = InterleavedSlice::new(input, 1, input.len())
        .map_err(|_| AudioError::message(DECODE_RECORDING))?;
    let output = resampler
        .process_all(&adapter, input.len(), None)
        .map_err(|_| AudioError::message(DECODE_RECORDING))?;
    Ok(output.take_data())
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
            message(advance_frames(MAX_FRAMES, 1).unwrap_err()),
            DURATION_LIMIT
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
