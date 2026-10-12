//! Native installation conversion. Candle interprets the pinned PyTorch container;
//! model inference itself uses MLX and never calls Python or Candle.
use crate::weights::{error, verify};
use candle_core::{DType, pickle::PthTensors};
use safetensors::tensor::{TensorView, serialize_to_file};
use sha2::{Digest, Sha256};
use speakerdesk_core::error::ChildError;
use std::{
    path::Path,
    sync::atomic::{AtomicBool, Ordering},
};

pub fn convert(directory: &Path, destination: &Path) -> Result<(), ChildError> {
    convert_cancellable(directory, destination, &AtomicBool::new(false))
}

pub fn convert_cancellable(
    directory: &Path,
    destination: &Path,
    cancel: &AtomicBool,
) -> Result<(), ChildError> {
    if cancel.load(Ordering::Acquire) {
        return Err(ChildError::Cancelled);
    }
    verify(directory, &[(super::SOURCE_FILE, super::SOURCE_SHA256)])?;
    let parameters = super::manifest::parameters().map_err(error)?;
    let source =
        PthTensors::new(directory.join(super::SOURCE_FILE), Some("model")).map_err(error)?;
    if source.tensor_infos().len() != parameters.len() {
        return Err(error("unexpected MossFormer2 checkpoint inventory"));
    }
    let mut tensors = Vec::with_capacity(parameters.len());
    for parameter in parameters {
        if cancel.load(Ordering::Acquire) {
            return Err(ChildError::Cancelled);
        }
        let tensor = source
            .get(&parameter.name)
            .map_err(error)?
            .ok_or_else(|| error(format!("missing MossFormer2 tensor {}", parameter.name)))?;
        if tensor.dtype() != DType::F32 || tensor.dims() != parameter.shape {
            return Err(error(format!(
                "invalid MossFormer2 tensor {}",
                parameter.name
            )));
        }
        let data = tensor
            .flatten_all()
            .map_err(error)?
            .to_vec1::<f32>()
            .map_err(error)?;
        if data.iter().any(|x| !x.is_finite()) {
            return Err(error(format!(
                "nonfinite MossFormer2 tensor {}",
                parameter.name
            )));
        }
        let bytes: Vec<u8> = data.into_iter().flat_map(f32::to_le_bytes).collect();
        if format!("{:x}", Sha256::digest(&bytes)) != parameter.sha256 {
            return Err(error(format!(
                "MossFormer2 tensor digest differs: {}",
                parameter.name
            )));
        }
        tensors.push((parameter.name, parameter.shape, bytes));
    }
    let views = tensors
        .iter()
        .map(|(name, shape, bytes)| {
            TensorView::new(safetensors::Dtype::F32, shape.clone(), bytes).map(|view| (name, view))
        })
        .collect::<Result<Vec<_>, _>>()
        .map_err(error)?;
    if cancel.load(Ordering::Acquire) {
        return Err(ChildError::Cancelled);
    }
    serialize_to_file(views, None, destination).map_err(error)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    #[ignore = "requires the frozen 670 MB checkpoint; no inference"]
    fn native_conversion_preserves_all_pinned_tensor_bytes_and_rejects_corruption() {
        let directory =
            std::env::var_os("SPEAKERDESK_MOSSFORMER_CHECKPOINT").expect("checkpoint directory");
        let temporary = tempfile::tempdir().unwrap();
        let destination = temporary.path().join(super::super::DERIVED_FILE);
        convert(Path::new(&directory), &destination).unwrap();
        let mut bytes = std::fs::read(destination).unwrap();
        super::super::manifest::validate(&bytes).unwrap();
        *bytes.last_mut().unwrap() ^= 1;
        assert!(super::super::manifest::validate(&bytes).is_err());
        assert!(matches!(
            convert_cancellable(
                Path::new(&directory),
                &temporary.path().join("cancelled"),
                &AtomicBool::new(true)
            ),
            Err(ChildError::Cancelled)
        ));
        assert!(!temporary.path().join("cancelled").exists());
    }
}
