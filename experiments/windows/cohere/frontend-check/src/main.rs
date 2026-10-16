#[path = "../../../../../crates/models/src/cohere/features_windows.rs"]
mod features;
#[path = "../../../../../crates/models/src/cohere/preemphasis.rs"]
mod preemphasis;
use preemphasis::PREEMPHASIS;
use std::{error::Error, fs, io::Write, path::Path};

fn main() -> Result<(), Box<dyn Error>> {
    let args: Vec<_> = std::env::args_os().collect();
    if args.len() != 3 {
        return Err("usage: cohere-frontend-check <mono-pcm.f32> <new-features.f32>".into());
    }
    let bytes = fs::read(Path::new(&args[1]))?;
    if bytes.len() % 4 != 0 {
        return Err("input length is not a whole Float32 sample".into());
    }
    let pcm: Vec<_> = bytes
        .as_chunks::<4>()
        .0
        .iter()
        .map(|sample| f32::from_le_bytes(*sample))
        .collect();
    let (values, valid) = features::Features::new().extract(&pcm)?;
    let mut output = std::io::BufWriter::new(
        fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(Path::new(&args[2]))?,
    );
    for value in values {
        output.write_all(&value.to_le_bytes())?;
    }
    output.flush()?;
    println!(
        "{{\"samples\":{},\"valid_frames\":{valid},\"shape\":[1,{},128],\"dtype\":\"f32\",\"dither\":\"StdRng0.9.5/StandardNormal0.5.1,len-seed,1e-5\"}}",
        pcm.len(),
        valid + 1
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::features::Features;

    #[test]
    fn physical_frames_exclude_the_centered_tail_and_single_frame_stays_finite() {
        let mut frontend = Features::new();
        for length in [160, 161, 319, 320, 549, 16000, 145445, 560000] {
            let (features, valid) = frontend.extract(&vec![0.0; length]).unwrap();
            assert_eq!(valid, length / 160);
            assert_eq!(features.len(), (valid + 1) * 128);
            assert!(features.iter().all(|value| value.is_finite()));
            assert!(features[valid * 128..].iter().all(|value| *value == 0.0));
            if valid == 1 {
                assert!(features.iter().all(|value| *value == 0.0));
            }
        }
    }

    #[test]
    fn normalizes_only_physical_frames_and_keeps_partial_sample_tails() {
        let mut frontend = Features::new();
        let audio: Vec<_> = (0..20480)
            .map(|index| ((index as f32 * 0.06).sin() + (index as f32 * 0.17).cos()) * 0.1)
            .collect();
        let (features, valid) = frontend.extract(&audio).unwrap();
        for mel in 0..128 {
            let values: Vec<_> = features[..valid * 128]
                .as_chunks::<128>()
                .0
                .iter()
                .map(|row| row[mel])
                .collect();
            let mean = values.iter().sum::<f32>() / valid as f32;
            let variance = values
                .iter()
                .map(|value| (value - mean).powi(2))
                .sum::<f32>()
                / (valid - 1) as f32;
            assert!(mean.abs() < 0.0002, "mel {mel}, mean {mean}");
            assert!(
                (0.99..=1.0001).contains(&variance),
                "mel {mel}, variance {variance}"
            );
        }
        let mut original = audio[..16079].to_vec();
        original[16000..].fill(0.0);
        let first = frontend.extract(&original).unwrap();
        original[16000..].fill(0.5);
        let changed = frontend.extract(&original).unwrap();
        assert_eq!(first.1, changed.1);
        assert_ne!(first.0, changed.0);
    }

    #[test]
    fn per_crop_dither_is_repeatable_and_independent_of_other_requests() {
        let mut reused = Features::new();
        let audio: Vec<_> = (0..16000).map(|i| (i as f32 * 0.1).sin() * 1e-4).collect();
        let first = reused.extract(&audio).unwrap();
        reused.extract(&vec![0.0; 549]).unwrap();
        assert_eq!(first, reused.extract(&audio).unwrap());
        assert_eq!(first, Features::new().extract(&audio).unwrap());
    }

    #[test]
    fn rejects_invalid_audio_and_arithmetic_overflow() {
        let mut frontend = Features::new();
        for audio in [
            vec![],
            vec![0.0; 159],
            vec![0.0; 560001],
            vec![f32::NAN; 160],
            vec![f32::INFINITY; 160],
            vec![f32::MAX; 160],
        ] {
            assert!(frontend.extract(&audio).is_err());
        }
    }
}
