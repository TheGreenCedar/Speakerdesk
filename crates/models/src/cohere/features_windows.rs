//! Portable Cohere log-mel frontend; the encoder consumes [1, valid + 1, 128].
use rand::{SeedableRng, rngs::StdRng};
use rand_distr::{Distribution, StandardNormal};
use realfft::{RealFftPlanner, RealToComplex, num_complex::Complex32};
use std::sync::Arc;

const FFT: usize = 512;
const HOP: usize = 160;
const MELS: usize = 128;

pub struct Features {
    fft: Arc<dyn RealToComplex<f32>>,
    input: Vec<f32>,
    spectrum: Vec<Complex32>,
    scratch: Vec<Complex32>,
    window: [f32; FFT],
    filters: Vec<f32>,
}

impl Features {
    pub fn new() -> Self {
        let fft = RealFftPlanner::<f32>::new().plan_fft_forward(FFT);
        let mut window = [0.0; FFT];
        for (i, value) in window[56..456].iter_mut().enumerate() {
            *value = (0.5 - 0.5 * (std::f64::consts::TAU * i as f64 / 399.0).cos()) as f32;
        }
        let log_step = 6.4f64.ln() / 27.0;
        let mel_max = 15.0 + 8.0f64.ln() / log_step;
        let edges: Vec<_> = (0..MELS + 2)
            .map(|i| {
                let mel = mel_max * i as f64 / (MELS + 1) as f64;
                if mel < 15.0 {
                    mel * (200.0 / 3.0)
                } else {
                    1000.0 * (log_step * (mel - 15.0)).exp()
                }
            })
            .collect();
        let mut filters = vec![0.0; MELS * (FFT / 2 + 1)];
        for (mel, row) in filters
            .as_chunks_mut::<{ FFT / 2 + 1 }>()
            .0
            .iter_mut()
            .enumerate()
        {
            for (bin, value) in row.iter_mut().enumerate() {
                let hz = bin as f64 * 16000.0 / FFT as f64;
                let lower = (hz - edges[mel]) / (edges[mel + 1] - edges[mel]);
                let upper = (edges[mel + 2] - hz) / (edges[mel + 2] - edges[mel + 1]);
                *value = (lower.min(upper).max(0.0) * 2.0 / (edges[mel + 2] - edges[mel])) as f32;
            }
        }
        Self {
            input: fft.make_input_vec(),
            spectrum: fft.make_output_vec(),
            scratch: fft.make_scratch_vec(),
            fft,
            window,
            filters,
        }
    }

    /// Flattened time-major features and the physical 10 ms frame count.
    pub fn extract(&mut self, pcm: &[f32]) -> Result<(Vec<f32>, usize), String> {
        if !(HOP..=35 * 16000).contains(&pcm.len()) || pcm.iter().any(|value| !value.is_finite()) {
            return Err("Cohere frontend requires 160–560000 finite samples".into());
        }
        // Like MLX and Transformers, each crop owns a length-seeded normal generator.
        // StdRng's samples differ from both backends; preserve the 1e-5 distribution and isolation.
        let mut rng = StdRng::seed_from_u64(pcm.len() as u64);
        let mut previous = 0.0;
        let filtered: Vec<_> = pcm
            .iter()
            .enumerate()
            .map(|(index, &sample)| {
                let noise: f32 = StandardNormal.sample(&mut rng);
                let current = sample + noise * 1e-5;
                let value = if index == 0 {
                    current
                } else {
                    current - super::PREEMPHASIS * previous
                };
                previous = current;
                value
            })
            .collect();
        let valid = pcm.len() / HOP;
        let mut features = vec![0.0; (valid + 1) * MELS];
        for (frame, row) in features
            .as_chunks_mut::<MELS>()
            .0
            .iter_mut()
            .take(valid)
            .enumerate()
        {
            for (index, value) in self.input.iter_mut().enumerate() {
                let position = (frame * HOP + index).checked_sub(FFT / 2);
                *value = position
                    .and_then(|position| filtered.get(position))
                    .copied()
                    .unwrap_or(0.0)
                    * self.window[index];
            }
            self.fft
                .process_with_scratch(&mut self.input, &mut self.spectrum, &mut self.scratch)
                .map_err(|error| error.to_string())?;
            let mut power = [0.0; FFT / 2 + 1];
            for (value, bin) in power.iter_mut().zip(&self.spectrum) {
                let magnitude = bin.norm();
                *value = magnitude * magnitude;
            }
            for (value, filter) in row
                .iter_mut()
                .zip(self.filters.as_chunks::<{ FFT / 2 + 1 }>().0)
            {
                let energy: f32 = power
                    .iter()
                    .zip(filter)
                    .map(|(power, weight)| power * weight)
                    .sum();
                *value = (energy + 2.0f32.powi(-24)).ln();
            }
        }
        for mel in 0..MELS {
            let mean = features[..valid * MELS]
                .as_chunks::<MELS>()
                .0
                .iter()
                .map(|row| row[mel])
                .sum::<f32>()
                / valid as f32;
            let std = if valid > 1 {
                (features[..valid * MELS]
                    .as_chunks::<MELS>()
                    .0
                    .iter()
                    .map(|row| (row[mel] - mean).powi(2))
                    .sum::<f32>()
                    / (valid - 1) as f32)
                    .sqrt()
            } else {
                0.0
            };
            for row in features[..valid * MELS].as_chunks_mut::<MELS>().0 {
                row[mel] = (row[mel] - mean) / (std + 1e-5);
            }
        }
        if features.iter().any(|value| !value.is_finite()) {
            return Err("Cohere frontend produced nonfinite features".into());
        }
        Ok((features, valid))
    }
}
