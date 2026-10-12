//! Installation-only conversion. Callers own the private staging destination.
use super::manifest::{convert_parameter, parse_manifest};
use crate::weights::{error, verify};
use safetensors::{
    Dtype,
    tensor::{TensorView, serialize_to_file},
};
use sha2::{Digest, Sha256};
use speakerdesk_core::error::ChildError;
use std::{
    collections::HashSet,
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
    verify(
        directory,
        &[
            ("model.int8.onnx", super::SOURCE_SHA256),
            ("tokens.txt", super::TOKENS_SHA256),
        ],
    )?;
    let manifest = parse_manifest(include_str!(
        "../../../../speakerdesk/alignment-mlx/manifest.json"
    ))
    .map_err(error)?;
    let source = std::fs::read(directory.join("model.int8.onnx")).map_err(error)?;
    let mut tensors = Vec::new();
    let mut used = HashSet::new();
    for parameter in &manifest.parameters {
        if cancel.load(Ordering::Acquire) {
            return Err(ChildError::Cancelled);
        }
        let tensor = convert_parameter(&source, &manifest, parameter).map_err(error)?;
        if tensor.data.iter().any(|value| !value.is_finite()) {
            return Err(error(format!("nonfinite CTC weight {}", parameter.target)));
        }
        let bytes: Vec<u8> = tensor.data.into_iter().flat_map(f32::to_le_bytes).collect();
        let digest = format!("{:x}", Sha256::digest(&bytes));
        if parameter.converted_tensor_sha256.as_deref() != Some(digest.as_str()) {
            return Err(error(format!(
                "CTC tensor digest differs: {}",
                parameter.target
            )));
        }
        used.extend(parameter.sources.iter().map(String::as_str));
        tensors.push((&parameter.target, tensor.shape, bytes));
    }
    if used
        != manifest
            .initializers
            .iter()
            .map(|item| item.name.as_str())
            .collect()
    {
        return Err(error("unmapped CTC initializer"));
    }
    drop(source);
    let views = tensors
        .iter()
        .map(|(name, shape, bytes)| {
            TensorView::new(Dtype::F32, shape.clone(), bytes).map(|view| (*name, view))
        })
        .collect::<Result<Vec<_>, _>>()
        .map_err(error)?;
    if cancel.load(Ordering::Acquire) {
        return Err(ChildError::Cancelled);
    }
    serialize_to_file(views, None, destination).map_err(error)
}
