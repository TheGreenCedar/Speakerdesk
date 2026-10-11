//! Check pinned artifacts once at load, then transfer tensor ownership into the graph.
use mlx_rs::{
    Array, Dtype,
    module::Param,
    nn::{LayerNorm, Linear},
};
use sha2::{Digest, Sha256};
use speakerdesk_core::error::ChildError;
use std::{collections::HashMap, fs::File, io::Read, path::Path};

pub fn verify(directory: &Path, files: &[(&str, &str)]) -> Result<(), ChildError> {
    let mut buffer = [0; 64 * 1024];
    for (name, digest) in files {
        let mut file = File::open(directory.join(name)).map_err(error)?;
        let mut hash = Sha256::new();
        loop {
            let count = file.read(&mut buffer).map_err(error)?;
            if count == 0 {
                break;
            }
            hash.update(&buffer[..count]);
        }
        if format!("{:x}", hash.finalize()) != *digest {
            return Err(error(format!("{name} does not match the pinned artifact")));
        }
    }
    Ok(())
}

pub fn take(
    weights: &mut HashMap<String, Array>,
    name: &str,
    shape: &[i32],
    dtype: Dtype,
) -> Result<Array, ChildError> {
    let tensor = weights
        .remove(name)
        .ok_or_else(|| error(format!("missing weight {name}")))?;
    if tensor.shape() != shape || tensor.dtype() != dtype {
        return Err(error(format!("invalid weight layout {name}")));
    }
    Ok(tensor)
}

pub fn linear(
    weights: &mut HashMap<String, Array>,
    name: &str,
    input: i32,
    output: i32,
    bias: bool,
    dtype: Dtype,
) -> Result<Linear, ChildError> {
    Ok(Linear {
        weight: Param::new(take(
            weights,
            &format!("{name}.weight"),
            &[output, input],
            dtype,
        )?),
        bias: Param::new(if bias {
            Some(take(weights, &format!("{name}.bias"), &[output], dtype)?)
        } else {
            None
        }),
    })
}

pub fn norm(
    weights: &mut HashMap<String, Array>,
    name: &str,
    dimensions: i32,
    dtype: Dtype,
) -> Result<LayerNorm, ChildError> {
    Ok(LayerNorm {
        dimensions,
        eps: 1e-5,
        weight: Param::new(Some(take(
            weights,
            &format!("{name}.weight"),
            &[dimensions],
            dtype,
        )?)),
        bias: Param::new(Some(take(
            weights,
            &format!("{name}.bias"),
            &[dimensions],
            dtype,
        )?)),
    })
}

pub fn error(error: impl std::fmt::Display) -> ChildError {
    ChildError::failed(error.to_string(), false)
}
