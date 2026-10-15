//! Pinned ReDimNet2 B6 through Windows ML with the qualified optimizer policy.
use super::{INPUT_SAMPLES, RATE, is_clipped, is_silent, prepared_audio};
use crate::{
    operation::Operation,
    windows_runtime::{Backend, ModelAdmission, error},
};
use ort::{session::Session, value::Tensor};
use speakerdesk_core::{
    error::ChildError,
    voice::{ClipEmbedding, VoiceModel, normalized},
};
use std::path::Path;

pub const MODEL_ID: &str = "PalabraAI/redimnet2";
pub const REVISION: &str = "2a8d15f65b1dfb5d73fede2f11ee42bcccca3035:b6-vb2+vox2_v0-lm:sd-pcm-v1:windowsml-2.4.89-directml-v1:ort-opt-disabled";
/// Exact ONNX graph. Its external weights are separately admitted by verify_model.
pub const ARTIFACT_SHA256: &str =
    "89c532e48539bebddc168e773cdcbf014b1941043c50d0a4ff7ccd1139efa5f1";
pub const DIMENSION: usize = 192;

/// This backend has its own identity; legacy/Core ML profile labels are never rewritten.
pub fn identity() -> VoiceModel {
    VoiceModel::new(MODEL_ID, REVISION, ARTIFACT_SHA256, DIMENSION, RATE)
        .expect("pinned Windows voice identity")
}

pub struct Network {
    graph: Session,
    _admission: ModelAdmission,
}

impl Network {
    pub fn load(directory: &Path, backend: &Backend) -> Result<Self, ChildError> {
        let admission = crate::windows_runtime::admit_model(directory, "redimnet2-b6")?;
        let directory = admission.directory();
        Ok(Self {
            // The original graph passes both independent identity fixtures only
            // with ORT optimization disabled. Optimized DML attenuates stage norms.
            graph: backend.load_unoptimized(&directory.join("redimnet.onnx"))?,
            _admission: admission,
        })
    }

    pub fn identity(&self) -> VoiceModel {
        identity()
    }

    pub fn embed(
        &mut self,
        pcm: &[f32],
        operation: &Operation,
    ) -> Result<ClipEmbedding, ChildError> {
        operation.check()?;
        let input = prepared_audio(pcm).map_err(error)?;
        let unclean = is_silent(pcm) || is_clipped(pcm);
        operation.check()?;
        if unclean {
            return Ok(ClipEmbedding {
                vector: vec![],
                clean: false,
            });
        }
        let audio = Tensor::from_array(([1, INPUT_SAMPLES], input)).map_err(error)?;
        let result = self
            .graph
            .run_with_options(ort::inputs! { "audio" => audio }, operation.run_options()?);
        // Cancellation wins over an ORT termination error and over output commit.
        operation.check()?;
        let outputs = result.map_err(error)?;
        let output = outputs
            .get("embedding")
            .ok_or_else(|| error("ReDimNet did not return an embedding"))?;
        let (shape, data) = output.try_extract_tensor::<f32>().map_err(error)?;
        let vector = validated_embedding(&shape[..], data)?;
        operation.check()?;
        Ok(ClipEmbedding {
            vector,
            clean: true,
        })
    }
}

fn validated_embedding(shape: &[i64], data: &[f32]) -> Result<Vec<f64>, ChildError> {
    if shape != [1, DIMENSION as i64] || data.len() != DIMENSION {
        return Err(error("ReDimNet returned an incompatible embedding layout"));
    }
    normalized(
        &data
            .iter()
            .map(|value| f64::from(*value))
            .collect::<Vec<_>>(),
        DIMENSION,
    )
    .map_err(error)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn output_commit_requires_exact_layout_and_finite_nonzero_normalized_vector() {
        let mut data = [0.; DIMENSION];
        data[0] = 3.;
        data[1] = 4.;
        let vector = validated_embedding(&[1, DIMENSION as i64], &data).unwrap();
        assert_eq!(&vector[..2], &[0.6, 0.8]);
        for (shape, values) in [
            (vec![DIMENSION as i64], data.to_vec()),
            (vec![1, DIMENSION as i64], data[..DIMENSION - 1].to_vec()),
            (vec![1, DIMENSION as i64], vec![0.; DIMENSION]),
            (vec![1, DIMENSION as i64], vec![f32::NAN; DIMENSION]),
            (vec![1, DIMENSION as i64], vec![f32::INFINITY; DIMENSION]),
        ] {
            assert!(validated_embedding(&shape, &values).is_err());
        }
    }
}
