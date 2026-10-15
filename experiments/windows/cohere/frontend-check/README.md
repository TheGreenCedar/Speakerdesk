# Portable Cohere frontend check

This isolated Rust crate compiles the production candidate directly from
`crates/models/src/cohere/features_windows.rs`. It does not modify the workspace
manifest or platform module wiring.

```sh
CARGO_TARGET_DIR=/tmp/speakerdesk-cohere-frontend-target cargo test --release --locked --offline --manifest-path experiments/windows/cohere/frontend-check/Cargo.toml
CARGO_TARGET_DIR=/tmp/speakerdesk-cohere-frontend-target cargo run --release --locked --offline --manifest-path experiments/windows/cohere/frontend-check/Cargo.toml -- mono-pcm.f32 new-features.f32
```

Both files use little-endian Float32. Input is finite mono PCM, 160–560,000
samples at 16 kHz. The output is flattened time-major `[1,frames,128]`, with
`valid = samples / 160` and `frames = valid + 1`. The binary prints this metadata
as JSON and refuses to overwrite an output file. In the application, the existing
less-than-160-sample empty result belongs to the decoder caller.

`Features::new()` prepares the standard realfft 3.5.0/rustfft 6.4.1 plan, symmetric
400-point Hann window centered in 512 samples and existing Slaney mel filters.
`extract(&mut self, pcm)` returns flattened features and the separate valid frame
count. It uses the existing 0.97 pre-emphasis, zero-centered STFT, power spectrum,
`2^-24` log guard, per-feature physical-frame mean and sample standard deviation,
and `1e-5` normalization epsilon. The extra centered frame is always zero and
never votes in normalization. One valid frame uses standard deviation zero,
matching the explicit Mac guard and keeping all outputs finite.

Every crop owns a fresh `StdRng::seed_from_u64(samples)` and StandardNormal Float32
samples scaled by `1e-5`. Pin rand **0.9.5** and rand_distr **0.5.1** when wiring
the production dependency. The maintained Transformers extractor similarly
creates a local Torch generator seeded by crop length; the Mac frontend uses an
explicit MLX length key. These generators produce different noise samples.
The native code preserves the distribution and isolation, not backend RNG bits.
RustFFT, platform transcendental functions and Float32 reductions also differ
from Torch/MLX kernels, so bit-for-bit feature equality is not promised.

The tests cover finite silence/quiet inputs across supported lengths, the
physical/tail frame distinction, valid=1 behavior, normalization over only
physical frames, partial sample tails, crop determinism across unrelated calls,
invalid lengths/nonfinite input and arithmetic overflow. They exercise the actual
production source rather than copying its formulas into a test implementation.

No sampled Cohere feature tensor is retained in the independent repository
fixture; that fixture owns the 31 expected tokens and decoded text. To inspect
feature drift against the maintained extractor using the existing build-only
Python environment:

```sh
python experiments/windows/cohere/frontend-check/compare_reference.py /path/to/pinned/cohere-speech mono-pcm.f32 new-features.f32 new-comparison.json
```

The diagnostic records source/input/output hashes and numerical differences. It
does not turn a shape match or high feature cosine into a transcription pass.
On the retained 145,445-sample WAV, the initial Mac native output had 909 valid
frames and shape `[1,910,128]`. Compared with maintained Transformers 5.7.0 on the
same PCM, cosine was 0.999978758, RMSE 0.00651079 and maximum absolute difference
0.16491866. Noise and backend math both differ; this measurement does not isolate
their contributions. The complete native frontend → encoder → cached decoder
must still reproduce all 31 independent tokens and text, including execution on
the intended Windows runtime.
