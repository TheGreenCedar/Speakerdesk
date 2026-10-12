use serde::Deserialize;

#[derive(Deserialize)]
pub struct Parameter {
    pub name: String,
    pub shape: Vec<usize>,
    pub sha256: String,
}

pub fn parameters() -> Result<Vec<Parameter>, serde_json::Error> {
    serde_json::from_str(include_str!("tensor_manifest.json"))
}

#[cfg(all(target_os = "macos", test))]
pub fn validate(bytes: &[u8]) -> Result<(), speakerdesk_core::error::ChildError> {
    use crate::weights::error;
    use sha2::{Digest, Sha256};
    let tensors = safetensors::SafeTensors::deserialize(bytes).map_err(error)?;
    let parameters = parameters().map_err(error)?;
    if tensors.len() != parameters.len() {
        return Err(error("unexpected MossFormer2 tensor inventory"));
    }
    for parameter in parameters {
        let tensor = tensors.tensor(&parameter.name).map_err(error)?;
        if tensor.shape() != parameter.shape || tensor.dtype() != safetensors::Dtype::F32 {
            return Err(error(format!(
                "invalid MossFormer2 tensor {}",
                parameter.name
            )));
        }
        if format!("{:x}", Sha256::digest(tensor.data())) != parameter.sha256 {
            return Err(error(format!(
                "MossFormer2 tensor digest differs: {}",
                parameter.name
            )));
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn manifest_matches_the_pinned_complete_state_dictionary() {
        let parameters = parameters().unwrap();
        let names: std::collections::HashSet<_> = parameters.iter().map(|p| &p.name).collect();
        assert_eq!(parameters.len(), 1099);
        assert_eq!(names.len(), parameters.len());
        assert_eq!(
            parameters
                .iter()
                .map(|p| p.shape.iter().product::<usize>())
                .sum::<usize>(),
            55_736_034
        );
        assert!(
            parameters
                .iter()
                .all(|p| p.sha256.len() == 64 && p.sha256.bytes().all(|c| c.is_ascii_hexdigit()))
        );
    }
}
