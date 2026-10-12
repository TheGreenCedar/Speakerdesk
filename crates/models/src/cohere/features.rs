use mlx_rs::{
    Array,
    error::Exception,
    ops::{self, indexing::IndexOp},
};

pub struct Features {
    window: Array,
    filters: Array,
}

impl Features {
    pub fn new() -> Self {
        let mut window = [0.0f32; 512];
        for (i, sample) in window[56..456].iter_mut().enumerate() {
            *sample = (0.5 - 0.5 * (std::f64::consts::TAU * i as f64 / 399.0).cos()) as f32;
        }
        let log_step = 6.4f64.ln() / 27.0;
        let mel_max = 15.0 + 8.0f64.ln() / log_step;
        let edges: Vec<f64> = (0..130)
            .map(|i| {
                let mel = mel_max * i as f64 / 129.0;
                if mel < 15.0 {
                    mel * (200.0 / 3.0)
                } else {
                    1000.0 * (log_step * (mel - 15.0)).exp()
                }
            })
            .collect();
        let mut filters = vec![0.0; 128 * 257];
        for (i, row) in filters.as_chunks_mut::<257>().0.iter_mut().enumerate() {
            for (bin, value) in row.iter_mut().enumerate() {
                let hz = bin as f64 * 16000.0 / 512.0;
                let lower = (hz - edges[i]) / (edges[i + 1] - edges[i]);
                let upper = (edges[i + 2] - hz) / (edges[i + 2] - edges[i + 1]);
                *value = (lower.min(upper).max(0.0) * 2.0 / (edges[i + 2] - edges[i])) as f32;
            }
        }
        Self {
            window: Array::from_slice(&window, &[512]),
            filters: Array::from_slice(&filters, &[128, 257]),
        }
    }

    /// Returns the centered STFT shape and its separate physical frame count.
    pub fn extract(&self, pcm: &[f32]) -> Result<(Array, usize), Exception> {
        let length = pcm.len() as i32;
        let valid = pcm.len() / 160;
        let frames = valid + 1;
        let waveform = Array::from_slice(pcm, &[length]);
        // Explicit per-crop key keeps dither deterministic without shared RNG state.
        // MLX's normal generator replaces NumPy's incidental generator, not its noise scale.
        let key = mlx_rs::random::key(pcm.len() as u64)?;
        let noise = mlx_rs::random::normal::<f32>(&[length][..], None, Some(1e-5), &key)?;
        let waveform = waveform.add(noise)?;
        let filtered = ops::concatenate(
            &[
                waveform.index(..1),
                waveform.index(1..).subtract(
                    waveform
                        .index(..length - 1)
                        .multiply(Array::from(super::PREEMPHASIS))?,
                )?,
            ],
            0,
        )?;
        let zero = ops::zeros::<f32>(&[256])?;
        let padded = ops::concatenate(&[&zero, &filtered, &zero], 0)?;
        let framed = padded.as_strided(&[frames as i32, 512][..], &[160, 1][..], 0)?;
        let power = mlx_rs::fft::rfft(framed.multiply(&self.window)?, None, None)?
            .abs()?
            .square()?;
        let mel = power
            .matmul(self.filters.t())?
            .add(Array::from(2.0f32.powi(-24)))?
            .log()?;
        let physical = mel.index(..valid as i32);
        let mean = physical.mean_axis(0, true)?;
        let std = if valid > 1 {
            physical
                .subtract(&mean)?
                .square()?
                .sum_axis(0, true)?
                .divide(Array::from((valid - 1) as f32))?
                .sqrt()?
        } else {
            ops::zeros::<f32>(&[1, 128])?
        };
        let normalized = mel.subtract(mean)?.divide(std.add(Array::from(1e-5f32))?)?;
        let mask = Array::from_slice(
            &(0..frames).map(|i| i < valid).collect::<Vec<_>>(),
            &[frames as i32, 1],
        );
        Ok((
            ops::select(mask, normalized, Array::from(0.0f32))?.reshape(&[
                1,
                frames as i32,
                128,
            ])?,
            valid,
        ))
    }
}
