# Independent synthetic fixtures

`reading-turns.json` contains known and unknown speaker projections. Their hashes
use Python's standard JSON/SHA-256 encoding; the shipped Python `validated_turns`
accepted both before they were used as Rust export fixtures.

`stereo-48k.flac` contains 4,800 stereo PCM16 frames at 48 kHz. Every left sample
is 16,384; every right sample is zero. Apple's `afconvert -f flac -d flac` encoded
a standard-library generated WAV. Independent decoding back to WAV verified all
samples and format fields. The fixture avoids an ffmpeg installation at test time.

These files contain no recordings or user data and do not exercise neural models.

`authored-tone.ogg` is an authored 48,017-frame mono signal at 48 kHz: 440 Hz and
997 Hz sine waves with amplitudes 0.2 and 0.1. SoundFile 0.14 / libsndfile 1.2.2
encoded it as Ogg/Opus (pre-skip 312, output gain zero). Independent macOS
`afconvert -f WAVE -d LEI16@16000 -c 1` normalization produced 16,006 mono PCM16
frames, RMS 0.157768. SHA-256: `38ac7dc3f49aea6a6f7ac68f36e883b1eab043e9fa4070281412670f1467a912`.

`alignment-reference.json` contains independent synthetic CTC score replays.
`generate_alignment_reference.py` takes an explicit historical Python `source`
directory and token file; use `speakerdesk/` from a separate checkout of baseline
`59e7b0c860f72ca94c24e6eab18ac820b144ee24`. The current tree no longer supplies
those Python definitions. The retained JSON and its source hashes are unchanged.
