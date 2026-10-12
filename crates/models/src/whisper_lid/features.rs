use mlx_rs::{
    Array, Dtype,
    error::Exception,
    ops::{self, indexing::IndexOp},
};

pub struct Features {
    window: Array,
    filters: Array,
}
impl Features {
    pub fn new() -> Result<Self, Exception> {
        let window: Vec<f32> = (0..400)
            .map(|i| (0.5 * (1.0 - (std::f64::consts::TAU * i as f64 / 400.0).cos())) as f32)
            .collect();
        let options = |count| ops::LinspaceOptions {
            count,
            endpoint: true,
        };
        let frequencies = ops::linspace::<_, f32>(0., 8000., options(201))?;
        let log_step = 6.4_f64.ln() / 27.0;
        let mel = ops::linspace::<_, f32>(0., 15.0 + 8_f64.ln() / log_step, options(82))?;
        let edges = ops::select(
            mel.ge(Array::from(15f32))?,
            mel.subtract(Array::from(15f32))?
                .multiply(Array::from(log_step as f32))?
                .exp()?
                .multiply(Array::from(1000f32))?,
            mel.multiply(Array::from((200. / 3.) as f32))?,
        )?;
        let differences = edges.index(1..).subtract(edges.index(..81))?;
        let slopes = edges
            .reshape(&[1, 82])?
            .subtract(frequencies.reshape(&[201, 1])?)?;
        let down = slopes
            .index((.., ..80))
            .negative()?
            .divide(differences.index(..80))?;
        let up = slopes.index((.., 2..)).divide(differences.index(1..))?;
        let filters = ops::maximum(ops::minimum(down, up)?, Array::from(0f32))?
            .multiply(Array::from(2f32).divide(edges.index(2..).subtract(edges.index(..80))?)?)?
            .t();
        Ok(Self {
            window: Array::from_slice(&window, &[400]),
            filters,
        })
    }

    pub fn extract(&self, pcm: &[f32]) -> Result<Array, Exception> {
        // Whisper observes a fixed 30-second acoustic context, including silence padding.
        let mut audio = vec![0f32; 480000];
        let count = pcm.len().min(audio.len());
        audio[..count].copy_from_slice(&pcm[..count]);
        let mut centered = Vec::with_capacity(480400);
        centered.extend(audio[1..201].iter().rev());
        centered.extend_from_slice(&audio);
        centered.extend(audio[479799..479999].iter().rev());
        let frames = Array::from_slice(&centered, &[480400]).as_strided(
            &[3000, 400][..],
            &[160, 1][..],
            0,
        )?;
        // The extra centered STFT frame is excluded, as in the pinned checkpoint frontend.
        let power = mlx_rs::fft::rfft(frames.multiply(&self.window)?, None, None)?
            .abs()?
            .square()?;
        let mel = power.matmul(self.filters.t())?;
        let log = ops::maximum(mel, Array::from(1e-10f32))?.log10()?;
        ops::maximum(&log, log.max(None)?.subtract(Array::from(8f32))?)?
            .add(Array::from(4f32))?
            .divide(Array::from(4f32))?
            .as_dtype(Dtype::Float16)
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn centered_padding_and_mel_match_independent_reference() {
        crate::metallib::prepare(None).unwrap();
        let mut reader = hound::WavReader::open(
            std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/data/two-voice.wav"),
        )
        .unwrap();
        let pcm: Vec<f32> = reader
            .samples::<i16>()
            .map(|s| f32::from(s.unwrap()) / 32768.)
            .collect();
        let reference: serde_json::Value =
            serde_json::from_str(include_str!("../../tests/data/whisper-reference.json")).unwrap();
        let features = super::Features::new().unwrap();
        for (audio, expected) in [
            &pcm[..],
            &pcm[..48000],
            &pcm[16000..16549],
            &[0f32; 512][..],
        ]
        .into_iter()
        .zip(reference["cases"].as_array().unwrap())
        {
            let mel = features
                .extract(audio)
                .unwrap()
                .as_dtype(mlx_rs::Dtype::Float32)
                .unwrap();
            assert_eq!(mel.shape(), [3000, 80]);
            mel.eval().unwrap();
            let values = mel.as_slice::<f32>();
            for point in expected["mel"].as_array().unwrap() {
                let index = point["frame"].as_u64().unwrap() as usize * 80
                    + point["band"].as_u64().unwrap() as usize;
                assert!(
                    (f64::from(values[index]) - point["value"].as_f64().unwrap()).abs() < 1e-4,
                    "{} {point} actual {}",
                    expected["name"],
                    values[index]
                );
            }
        }
    }
}
