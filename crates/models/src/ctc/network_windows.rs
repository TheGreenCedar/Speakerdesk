//! Execution boundary for the frozen Windows CTC graph.
use crate::{
    operation::Operation,
    windows_runtime::{Backend, ModelAdmission, error},
};
use ort::{session::Session, value::Tensor};
use speakerdesk_core::{error::ChildError, ipc::CtcScoresOut};
use std::path::Path;

pub const ARTIFACT_SHA256: &str =
    "233ac8980dab9c142ef9fe91ab1d3b5d6309a9549f59a284e50199f7b1081699";
pub const VOCABULARY_SIZE: i32 = 9812;
pub const MAX_SAMPLES: usize = 392000;

pub struct Network {
    graph: Session,
    _admission: ModelAdmission,
}

impl Network {
    pub fn load(directory: &Path, backend: &Backend) -> Result<Self, ChildError> {
        // Qualification and production use the same immutable artifact admission.
        let admission = crate::windows_runtime::admit_model(directory, "coarse-alignment")?;
        let directory = admission.directory();
        Ok(Self {
            graph: backend.load(&directory.join("ctc-mha.onnx"))?,
            _admission: admission,
        })
    }

    pub fn scores(
        &mut self,
        pcm: &[f32],
        token_ids: &[i32],
        operation: &Operation,
    ) -> Result<CtcScoresOut, ChildError> {
        operation.check()?;
        validate_request(pcm, token_ids)?;
        let frames = valid_frames(pcm.len());
        let gathered = if frames == 0 {
            Vec::new()
        } else {
            let normalized = normalize(pcm, operation)?;
            let audio = Tensor::from_array(([1, pcm.len()], normalized)).map_err(error)?;
            let ids = Tensor::from_array((
                [token_ids.len()],
                token_ids
                    .iter()
                    .map(|&id| i64::from(id))
                    .collect::<Vec<_>>(),
            ))
            .map_err(error)?;
            operation.check()?;
            let result = self.graph.run_with_options(
                ort::inputs!["x" => audio, "token_ids" => ids],
                operation.run_options()?,
            );
            // Cancellation takes precedence over a provider termination error.
            operation.check()?;
            let outputs = result.map_err(error)?;
            let output = outputs
                .get("scores")
                .ok_or_else(|| error("CTC graph did not return scores"))?;
            let (shape, values) = output.try_extract_tensor::<f32>().map_err(error)?;
            if &shape[..] != [1, frames as i64, token_ids.len() as i64]
                || values.len() != frames * token_ids.len()
                || values.iter().any(|value| !value.is_finite())
            {
                return Err(error("CTC returned an incompatible score tensor"));
            }
            values.to_vec()
        };
        operation.check()?;
        Ok(CtcScoresOut {
            valid_frames: frames as u32,
            audio_num_samples: pcm.len() as u32,
            frame_hop_samples: 320,
            frame_origin_samples: 0,
            token_ids: token_ids.to_vec(),
            // Core resolves this exact Windows identity without borrowing Mac calibration.
            model: ARTIFACT_SHA256.into(),
            vocabulary_sha256: super::TOKENS_SHA256.into(),
            gathered,
        })
    }
}

fn validate_request(pcm: &[f32], token_ids: &[i32]) -> Result<(), ChildError> {
    if pcm.is_empty() || pcm.len() > MAX_SAMPLES || pcm.iter().any(|value| !value.is_finite()) {
        return Err(error(
            "CTC requires finite 16 kHz audio of at most 24.5 seconds",
        ));
    }
    if token_ids.is_empty()
        || token_ids.len() > VOCABULARY_SIZE as usize
        || token_ids
            .iter()
            .any(|id| !(0..VOCABULARY_SIZE).contains(id))
    {
        return Err(error("invalid CTC token selection"));
    }
    Ok(())
}

fn normalize(pcm: &[f32], operation: &Operation) -> Result<Vec<f32>, ChildError> {
    operation.check()?;
    // Same population-variance rule as the Mac adapter; wider host reductions
    // avoid a crop-length-dependent sequential F32 sum. Joined source/native
    // scalar fidelity remains unqualified, including this frontend boundary.
    let count = pcm.len() as f64;
    let mean = pcm.iter().map(|&value| f64::from(value)).sum::<f64>() / count;
    let variance = pcm
        .iter()
        .map(|&value| (f64::from(value) - mean).powi(2))
        .sum::<f64>()
        / count;
    let scale = (variance + f64::from(1e-5_f32)).sqrt();
    operation.check()?;
    let normalized = pcm
        .iter()
        .map(|&value| ((f64::from(value) - mean) / scale) as f32)
        .collect::<Vec<_>>();
    if normalized.iter().any(|value| !value.is_finite()) {
        return Err(error("CTC normalization returned nonfinite audio"));
    }
    operation.check()?;
    Ok(normalized)
}

pub fn valid_frames(samples: usize) -> usize {
    if samples < 400 {
        0
    } else {
        (samples - 400) / 320 + 1
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn crop_and_column_boundaries_keep_partial_frames_and_duplicate_order() {
        for (samples, frames) in [
            (1, 0),
            (399, 0),
            (400, 1),
            (719, 1),
            (720, 2),
            (721, 2),
            (MAX_SAMPLES, 1224),
        ] {
            assert_eq!(valid_frames(samples), frames);
            validate_request(&vec![0.; samples], &[9811, 0, 9811]).unwrap();
        }
        for (audio, ids) in [
            (vec![], vec![0]),
            (vec![0.; MAX_SAMPLES + 1], vec![0]),
            (vec![f32::NAN], vec![0]),
            (vec![0.; 400], vec![]),
            (vec![0.; 400], vec![-1]),
            (vec![0.; 400], vec![VOCABULARY_SIZE]),
        ] {
            assert!(validate_request(&audio, &ids).is_err());
        }
    }

    #[test]
    fn population_normalization_is_request_local_and_cancellation_is_not_a_score() {
        let running = Operation::new();
        assert!(
            normalize(&vec![0.1; MAX_SAMPLES], &running)
                .unwrap()
                .iter()
                .all(|&x| x == 0.)
        );
        let signal = [1., -1., 1., -1.];
        let actual = normalize(&signal, &running).unwrap();
        assert_eq!(actual[0], -actual[1]);
        assert!(actual[0] > 0.999 && actual[0] < 1.);
        normalize(&[0.; 720], &running).unwrap();
        assert_eq!(normalize(&signal, &running).unwrap(), actual);
        let cancelled = Operation::new();
        cancelled.cancel();
        assert!(matches!(
            normalize(&signal, &cancelled),
            Err(ChildError::Cancelled)
        ));
    }
}
