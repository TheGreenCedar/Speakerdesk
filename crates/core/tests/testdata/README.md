# Independent synthetic fixtures

`reading-turns.json` contains known and unknown speaker projections. Their hashes
use Python's standard JSON/SHA-256 encoding; the shipped Python `validated_turns`
accepted both before they were used as Rust export fixtures.

`stereo-48k.flac` contains 4,800 stereo PCM16 frames at 48 kHz. Every left sample
is 16,384; every right sample is zero. Apple's `afconvert -f flac -d flac` encoded
a standard-library generated WAV. Independent decoding back to WAV verified all
samples and format fields. The fixture avoids an ffmpeg installation at test time.

These files contain no recordings or user data and do not exercise neural models.
