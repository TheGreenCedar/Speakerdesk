//! Length-n real DFTs for tonal rejection.
//!
//! `realfft` lives in `speakerdesk-core`. This crate keeps a small Bluestein
//! transform so Linux builds do not link the C++ canceller or add FFT crates.

use std::f64::consts::PI;

#[derive(Clone, Copy)]
struct Complex {
    re: f64,
    im: f64,
}

impl Complex {
    const fn new(re: f64, im: f64) -> Self {
        Self { re, im }
    }

    fn conj(self) -> Self {
        Self {
            re: self.re,
            im: -self.im,
        }
    }

    fn mul(self, other: Self) -> Self {
        Self {
            re: self.re * other.re - self.im * other.im,
            im: self.re * other.im + self.im * other.re,
        }
    }
}

fn fft(values: &mut [Complex], inverse: bool) {
    let n = values.len();
    debug_assert!(n.is_power_of_two());
    if n <= 1 {
        return;
    }
    let mut reversed = 0;
    for index in 1..n {
        let mut bit = n >> 1;
        while reversed & bit != 0 {
            reversed ^= bit;
            bit >>= 1;
        }
        reversed ^= bit;
        if index < reversed {
            values.swap(index, reversed);
        }
    }
    let mut length = 2;
    while length <= n {
        let angle = if inverse { 2.0 } else { -2.0 } * PI / length as f64;
        let step = Complex::new(angle.cos(), angle.sin());
        let mut start = 0;
        while start < n {
            let mut twiddle = Complex::new(1.0, 0.0);
            for offset in 0..length / 2 {
                let even = values[start + offset];
                let odd = values[start + offset + length / 2].mul(twiddle);
                values[start + offset] = Complex::new(even.re + odd.re, even.im + odd.im);
                values[start + offset + length / 2] =
                    Complex::new(even.re - odd.re, even.im - odd.im);
                twiddle = twiddle.mul(step);
            }
            start += length;
        }
        length <<= 1;
    }
    if inverse {
        let scale = 1.0 / n as f64;
        for value in values.iter_mut() {
            *value = Complex::new(value.re * scale, value.im * scale);
        }
    }
}

/// One-sided power spectrum `|rfft(samples)|^2`, unnormalized like NumPy.
pub(crate) fn power_spectrum(samples: &[f64]) -> Vec<f64> {
    real_dft(samples)
        .into_iter()
        .map(|bin| bin.re * bin.re + bin.im * bin.im)
        .collect()
}

fn real_dft(samples: &[f64]) -> Vec<Complex> {
    let bins = samples.len() / 2 + 1;
    bluestein(samples).into_iter().take(bins).collect()
}

fn bluestein(samples: &[f64]) -> Vec<Complex> {
    let n = samples.len();
    if n == 0 {
        return Vec::new();
    }
    if n == 1 {
        return vec![Complex::new(samples[0], 0.0)];
    }
    let chirp: Vec<Complex> = (0..n)
        .map(|bin| {
            let angle = -PI * (bin as f64) * (bin as f64) / n as f64;
            Complex::new(angle.cos(), angle.sin())
        })
        .collect();
    let padded = (2 * n - 1).next_power_of_two();
    let mut left = vec![Complex::new(0.0, 0.0); padded];
    let mut right = vec![Complex::new(0.0, 0.0); padded];
    right[0] = Complex::new(1.0, 0.0);
    for index in 0..n {
        left[index] = Complex::new(samples[index], 0.0).mul(chirp[index]);
        if index > 0 {
            let conj = chirp[index].conj();
            right[index] = conj;
            right[padded - index] = conj;
        }
    }
    fft(&mut left, false);
    fft(&mut right, false);
    for index in 0..padded {
        left[index] = left[index].mul(right[index]);
    }
    fft(&mut left, true);
    left.into_iter()
        .take(n)
        .enumerate()
        .map(|(index, value)| value.mul(chirp[index]))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::real_dft;

    fn close(actual: (f64, f64), expected: (f64, f64)) {
        let error = (actual.0 - expected.0).hypot(actual.1 - expected.1);
        assert!(error < 1e-9, "{actual:?} != {expected:?} (error {error})");
    }

    #[test]
    fn real_dft_matches_numpy_rfft() {
        let four = real_dft(&[1.0, 2.0, 3.0, 4.0]);
        close((four[0].re, four[0].im), (10.0, 0.0));
        close((four[1].re, four[1].im), (-2.0, 2.0));
        close((four[2].re, four[2].im), (-2.0, 0.0));

        let five = real_dft(&[1.0, 2.0, 3.0, 4.0, 5.0]);
        close((five[0].re, five[0].im), (15.0, 0.0));
        close((five[1].re, five[1].im), (-2.5, 3.440_954_801_177_934));
        close((five[2].re, five[2].im), (-2.5, 0.812_299_240_582_266));
    }
}
