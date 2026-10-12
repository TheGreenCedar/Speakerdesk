//! Real FFTs for tonal rejection and reference correlation.

use realfft::{RealFftPlanner, num_complex::Complex};

/// One-sided power spectrum `|rfft(samples)|^2`, unnormalized like NumPy.
pub(crate) fn power_spectrum(samples: &[f64]) -> Vec<f64> {
    real_dft(samples)
        .into_iter()
        .map(|bin| bin.norm_sqr())
        .collect()
}

fn real_dft(samples: &[f64]) -> Vec<Complex<f64>> {
    if samples.is_empty() {
        return Vec::new();
    }
    let transform = RealFftPlanner::<f64>::new().plan_fft_forward(samples.len());
    let mut output = transform.make_output_vec();
    transform
        .process(&mut samples.to_vec(), &mut output)
        .expect("matching FFT lengths");
    output
}

/// Valid dot products against every full reference window, with zero-padding to avoid wraparound.
pub(crate) fn correlation(reference: &[f64], near: &[f64]) -> Vec<f64> {
    assert!(!near.is_empty() && reference.len() >= near.len());
    let size = (reference.len() + near.len() - 1).next_power_of_two();
    let mut planner = RealFftPlanner::<f64>::new();
    let forward = planner.plan_fft_forward(size);
    let inverse = planner.plan_fft_inverse(size);
    let mut input = forward.make_input_vec();
    input[..reference.len()].copy_from_slice(reference);
    let mut far = forward.make_output_vec();
    forward
        .process(&mut input, &mut far)
        .expect("matching FFT lengths");
    input.fill(0.);
    input[..near.len()].copy_from_slice(near);
    let mut microphone = forward.make_output_vec();
    forward
        .process(&mut input, &mut microphone)
        .expect("matching FFT lengths");
    for (far, microphone) in far.iter_mut().zip(microphone) {
        *far *= microphone.conj();
    }
    inverse
        .process(&mut far, &mut input)
        .expect("real correlation spectrum");
    input.truncate(reference.len() - near.len() + 1);
    input.iter_mut().for_each(|sample| *sample /= size as f64);
    input
}

#[cfg(test)]
mod tests {
    use super::*;

    fn direct(reference: &[f64], near: &[f64]) -> Vec<f64> {
        reference
            .windows(near.len())
            .map(|window| window.iter().zip(near).map(|(a, b)| a * b).sum())
            .collect()
    }

    #[test]
    fn real_dft_matches_independent_definition_for_even_odd_and_prime_lengths() {
        for length in [1, 2, 3, 4, 5, 16, 31, 127] {
            let samples: Vec<_> = (0..length)
                .map(|i| ((i * 37 % 101) as f64 - 50.) / 100.)
                .collect();
            for (k, actual) in real_dft(&samples).iter().enumerate() {
                let expected: Complex<f64> = samples
                    .iter()
                    .enumerate()
                    .map(|(n, sample)| {
                        Complex::from_polar(
                            *sample,
                            -std::f64::consts::TAU * (k * n) as f64 / length as f64,
                        )
                    })
                    .sum();
                assert!(
                    (*actual - expected).norm() < 1e-10,
                    "length {length}, bin {k}"
                );
            }
        }
        assert!(power_spectrum(&[]).is_empty());
        assert_eq!(power_spectrum(&[1., 2., 3., 4.]), vec![100., 8., 4.]);
    }

    #[test]
    fn fft_correlation_matches_direct_dot_products_including_tail_lags() {
        for (length, near_length) in [(1, 1), (5, 3), (448, 400), (9999, 1999)] {
            let reference: Vec<_> = (0..length)
                .map(|i| ((i * 37 % 101) as f64 - 50.) / 100.)
                .collect();
            let near: Vec<_> = (0..near_length)
                .map(|i| ((i * 19 % 97) as f64 - 48.) / 100.)
                .collect();
            let actual = correlation(&reference, &near);
            let expected = direct(&reference, &near);
            assert_eq!(actual.len(), expected.len());
            for (actual, expected) in actual.iter().zip(expected) {
                assert!((actual - expected).abs() < 1e-9, "{actual} != {expected}");
            }
        }
    }

    #[test]
    #[cfg(not(debug_assertions))]
    #[ignore = "optimized production-size comparison; run explicitly with --release --ignored --nocapture"]
    fn compare_production_correlation() {
        let reference: Vec<_> = (0..9999)
            .map(|i| ((i * 37 % 101) as f64 - 50.) / 100.)
            .collect();
        let near = &reference[715..2714];
        let mut direct_times = Vec::new();
        let mut fft_times = Vec::new();
        for round in 0..7 {
            for fft in if round % 2 == 0 {
                [false, true]
            } else {
                [true, false]
            } {
                let start = std::time::Instant::now();
                let result = if fft {
                    correlation(std::hint::black_box(&reference), std::hint::black_box(near))
                } else {
                    direct(std::hint::black_box(&reference), std::hint::black_box(near))
                };
                std::hint::black_box(result);
                let elapsed = start.elapsed().as_secs_f64() * 1000.;
                if fft {
                    fft_times.push(elapsed);
                } else {
                    direct_times.push(elapsed);
                }
            }
        }
        println!(
            "9999 reference samples, 1999 microphone samples; milliseconds per channel, allocation and planning included\ndirect={direct_times:?}\nfft={fft_times:?}"
        );
    }
}
