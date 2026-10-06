"""Use OS/FFmpeg decoders, with a dependency-free fast path for canonical PCM WAV."""
import shutil
import subprocess
import wave
from pathlib import Path

MAX_DURATION = 7200


def pcm16_bytes(audio):
    """Quantize float capture to PCM16 without attenuating quiet integer samples."""
    import numpy as np
    values = np.asarray(audio)
    if values.ndim != 1 or not np.isfinite(values).all():
        raise ValueError('Capture requires finite mono audio.')
    # PCM16 decoding divides by32768. Round the inverse before saturation;
    # truncating x*32767 changes almost every nonzero quiet integer sample.
    return np.clip(np.rint(values.astype(np.float64)*32768), -32768, 32767).astype('<i2').tobytes()


def wav_duration(path):
    try:
        with wave.open(str(path), 'rb') as audio:
            if (audio.getnchannels(), audio.getsampwidth(), audio.getframerate()) != (1, 2, 16000):
                raise ValueError('Expected 16 kHz mono 16-bit PCM WAV.')
            duration = audio.getnframes() / 16000
            if not 0 < duration <= MAX_DURATION:
                raise ValueError('Recordings must be nonempty and no longer than two hours.')
            remaining=audio.getnframes()*2
            while remaining:
                block=audio.readframes(min(65536,(remaining+1)//2))
                if not block:raise ValueError('The PCM recording is truncated.')
                remaining-=len(block)
            if remaining!=0:raise ValueError('The PCM recording has an incomplete sample.')
            return duration
    except (wave.Error, EOFError):
        raise ValueError('The file could not be decoded as PCM audio.') from None


def normalize(source, destination):
    try:
        wav_duration(source)
        shutil.copyfile(source, destination)
    except ValueError:
        ffmpeg = shutil.which('ffmpeg')
        afconvert = shutil.which('afconvert')
        if ffmpeg:
            cmd = [ffmpeg, '-nostdin', '-v', 'error', '-y', '-protocol_whitelist', 'file,pipe',
                   '-format_whitelist','wav,mp3,mov,flac,ogg,aiff,aac,matroska,webm',
                   '-i', str(source), '-vn', '-ac', '1', '-ar', '16000', '-c:a', 'pcm_s16le', str(destination)]
        elif afconvert:
            cmd = [afconvert, str(source), str(destination), '-f', 'WAVE', '-d', 'LEI16@16000', '-c', '1']
        else:
            raise ValueError('Install FFmpeg to decode this format, or supply mono 16 kHz 16-bit PCM WAV.') from None
        try:
            result = subprocess.run(cmd, capture_output=True, timeout=180)
        except subprocess.TimeoutExpired:
            raise ValueError('Audio conversion exceeded three minutes.') from None
        if result.returncode:
            raise ValueError('Could not decode this recording. Supply a supported audio file.')
    return wav_duration(destination)


def crop(source, destination, start, end):
    with wave.open(str(source), 'rb') as reader:
        rate = reader.getframerate()
        reader.setpos(round(start * rate))
        frames = reader.readframes(round((end-start) * rate))
        with wave.open(str(destination), 'wb') as writer:
            writer.setparams(reader.getparams())
            writer.writeframes(frames)
