//! Setup-only expansion into the byte-identical, qualified DirectML external store.
use super::manifest::{
    FloatTensor, Manifest, Parameter, Transform, apply_transform, read_initializer,
};
use serde::Deserialize;
use sha2::{Digest, Sha256};
use speakerdesk_core::error::ChildError;
use std::{
    collections::HashSet,
    fs::{self, File},
    io::{Read, Seek, SeekFrom, Write},
    path::Path,
    sync::atomic::{AtomicBool, Ordering},
};

const MANIFEST: &str = include_str!("../../../../speakerdesk/alignment-mlx/manifest.json");
const RECIPE: &str = include_str!("windows-reconstruction.json");

#[derive(Clone, Deserialize)]
#[serde(deny_unknown_fields)]
struct Recipe {
    source_sha256: String,
    source_bytes: u64,
    graph_sha256: String,
    manifest_sha256: String,
    weights_sha256: String,
    weights_bytes: u64,
    tensors: Vec<Tensor>,
}
#[derive(Clone, Deserialize)]
#[serde(deny_unknown_fields)]
struct Tensor {
    initializer: String,
    parameter: String,
    shape: Vec<usize>,
    offset: u64,
    bytes: u64,
}

fn error(message: impl std::fmt::Display) -> ChildError {
    ChildError::failed(message.to_string(), false)
}
fn checked(cancel: &AtomicBool) -> Result<(), ChildError> {
    if cancel.load(Ordering::Acquire) {
        Err(ChildError::Cancelled)
    } else {
        Ok(())
    }
}

#[cfg(windows)]
pub fn convert_cancellable(directory: &Path, cancel: &AtomicBool) -> Result<(), ChildError> {
    checked(cancel)?;
    let recipe: Recipe = serde_json::from_str(RECIPE).map_err(error)?;
    if recipe.source_sha256 != super::SOURCE_SHA256
        || recipe.graph_sha256 != super::network::ARTIFACT_SHA256
        || recipe.manifest_sha256 != format!("{:x}", Sha256::digest(MANIFEST.as_bytes()))
    {
        return Err(error(
            "CTC reconstruction recipe does not match its pinned sources",
        ));
    }
    let manifest = super::manifest::parse_manifest(MANIFEST).map_err(error)?;
    let admission = crate::windows_runtime::admit_source_files(directory, "coarse-alignment")?;
    let mut source = admission.file("model.int8.onnx")?;
    let part = directory.join("ctc.weights.part");
    match fs::symlink_metadata(&part) {
        Ok(meta) if !meta.is_file() || meta.is_symlink() => {
            return Err(error("CTC staging output is not a regular file"));
        }
        Ok(_) => fs::remove_file(&part).map_err(error)?,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => {}
        Err(e) => return Err(error(e)),
    }
    write_prepared(&mut source, &part, &manifest, &recipe, cancel)?;
    checked(cancel)?;
    speakerdesk_core::filesystem::rename(&part, &directory.join("ctc.weights")).map_err(error)
}

fn validate<'a>(manifest: &'a Manifest, recipe: &Recipe) -> Result<Vec<&'a Parameter>, ChildError> {
    let mut seen = HashSet::new();
    let mut offset = 0_u64;
    let mut parameters = Vec::new();
    for tensor in &recipe.tensors {
        let parameter = manifest
            .parameters
            .iter()
            .find(|p| p.target == tensor.parameter)
            .ok_or_else(|| error("Unknown CTC reconstruction parameter"))?;
        let name = match parameter.sources.as_slice() {
            [name] if parameter.transform != Transform::Uint8DequantizeThenIoToOi => name.clone(),
            [name, _, _] if parameter.transform == Transform::Uint8DequantizeThenIoToOi => {
                format!("{name}_dequantized")
            }
            _ => return Err(error("Invalid CTC reconstruction sources")),
        };
        let mut shape = parameter.shape.clone();
        match parameter.transform {
            Transform::Identity => {}
            Transform::OiKernelToOKernelI if shape.len() == 3 => shape.swap(1, 2),
            Transform::Uint8DequantizeThenIoToOi if shape.len() == 2 => shape.swap(0, 1),
            _ => return Err(error("Invalid CTC reconstruction layout")),
        }
        let bytes = shape
            .iter()
            .try_fold(4_u64, |n, &d| n.checked_mul(d as u64))
            .ok_or_else(|| error("CTC reconstruction size overflow"))?;
        if !seen.insert(&tensor.parameter)
            || tensor.initializer != name
            || tensor.shape != shape
            || tensor.bytes != bytes
            || bytes == 0
            || tensor.offset != offset
        {
            return Err(error(
                "CTC reconstruction has duplicate, missing or incompatible spans",
            ));
        }
        offset = offset
            .checked_add(bytes)
            .ok_or_else(|| error("CTC store size overflow"))?;
        parameters.push(parameter);
    }
    if seen.len() != manifest.parameters.len() || offset != recipe.weights_bytes {
        return Err(error(
            "CTC reconstruction does not cover the complete weight store",
        ));
    }
    Ok(parameters)
}

fn hash_reader(reader: &mut impl Read, cancel: &AtomicBool) -> Result<String, ChildError> {
    let mut digest = Sha256::new();
    let mut bytes = [0_u8; 65536];
    loop {
        checked(cancel)?;
        let count = reader.read(&mut bytes).map_err(error)?;
        if count == 0 {
            break;
        }
        digest.update(&bytes[..count]);
    }
    Ok(format!("{:x}", digest.finalize()))
}

fn parameter(
    source: &mut File,
    manifest: &Manifest,
    p: &Parameter,
) -> Result<FloatTensor, ChildError> {
    let mut inputs = Vec::new();
    for name in &p.sources {
        let mut initializer = manifest
            .initializers
            .iter()
            .find(|v| &v.name == name)
            .ok_or_else(|| error("Missing CTC source initializer"))?
            .clone();
        let mut bytes = Vec::new();
        if initializer.scalar_hex.is_none() {
            let offset = initializer
                .offset
                .ok_or_else(|| error("Missing source offset"))?;
            let count = initializer
                .nbytes
                .ok_or_else(|| error("Missing source length"))?;
            bytes.resize(usize::try_from(count).map_err(error)?, 0);
            source.seek(SeekFrom::Start(offset)).map_err(error)?;
            source.read_exact(&mut bytes).map_err(error)?;
            initializer.offset = Some(0);
        }
        inputs.push(read_initializer(&bytes, &initializer).map_err(error)?);
    }
    let tensor = apply_transform(p.transform, &inputs).map_err(error)?;
    let mut digest = Sha256::new();
    for value in &tensor.data {
        if !value.is_finite() {
            return Err(error("Nonfinite CTC weight"));
        }
        digest.update(value.to_le_bytes());
    }
    if tensor.shape != p.shape
        || p.converted_tensor_sha256.as_deref() != Some(format!("{:x}", digest.finalize()).as_str())
    {
        return Err(error(format!("CTC converted tensor differs: {}", p.target)));
    }
    Ok(tensor)
}

fn write_prepared(
    source: &mut File,
    destination: &Path,
    manifest: &Manifest,
    recipe: &Recipe,
    cancel: &AtomicBool,
) -> Result<(), ChildError> {
    let parameters = validate(manifest, recipe)?;
    if source.metadata().map_err(error)?.len() != recipe.source_bytes {
        return Err(error("CTC compact source size differs"));
    }
    source.rewind().map_err(error)?;
    if hash_reader(source, cancel)? != recipe.source_sha256 {
        return Err(error("CTC compact source digest differs"));
    }
    let mut options = File::options();
    options.write(true).create_new(true);
    #[cfg(windows)]
    {
        use std::os::windows::fs::OpenOptionsExt;
        options.share_mode(0);
    }
    let mut output = options.open(destination).map_err(error)?;
    let result = (|| {
        let mut digest = Sha256::new();
        let mut buffer = Vec::with_capacity(65536);
        for (span, p) in recipe.tensors.iter().zip(parameters) {
            checked(cancel)?;
            let tensor = parameter(source, manifest, p)?;
            // Undo only the Mac layout transform; preserve every Float32 bit.
            for start in (0..tensor.data.len()).step_by(16384) {
                checked(cancel)?;
                buffer.clear();
                for index in start..(start + 16384).min(tensor.data.len()) {
                    let src = match p.transform {
                        Transform::Identity => index,
                        Transform::Uint8DequantizeThenIoToOi => {
                            (index % span.shape[1]) * span.shape[0] + index / span.shape[1]
                        }
                        Transform::OiKernelToOKernelI => {
                            let [_, channels, kernel] = span.shape.as_slice() else {
                                unreachable!()
                            };
                            let o = index / (channels * kernel);
                            let i = (index / kernel) % channels;
                            (o * kernel + index % kernel) * channels + i
                        }
                    };
                    buffer.extend_from_slice(&tensor.data[src].to_le_bytes());
                }
                digest.update(&buffer);
                output.write_all(&buffer).map_err(error)?;
            }
        }
        checked(cancel)?;
        if output.metadata().map_err(error)?.len() != recipe.weights_bytes
            || format!("{:x}", digest.finalize()) != recipe.weights_sha256
        {
            return Err(error(
                "Reconstructed CTC weight store differs from the qualified artifact",
            ));
        }
        output.sync_all().map_err(error)
    })();
    drop(output);
    if result.is_err() {
        let _ = fs::remove_file(destination);
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;

    fn bytes(values: &[f32]) -> Vec<u8> {
        values.iter().flat_map(|v| v.to_le_bytes()).collect()
    }
    fn digest(data: &[u8]) -> String {
        format!("{:x}", Sha256::digest(data))
    }

    fn fixture() -> (Vec<u8>, Manifest, Recipe, Vec<u8>) {
        let mut source = vec![2, 3, 4, 5, 6, 7];
        source.extend(bytes(&[1.0, -2.0]));
        source.extend(bytes(&[0.0, 1.0, 2.0, 3.0, 4.0, 5.0]));
        let expected = bytes(&[
            1.0, -2.0, 0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0,
        ]);
        let manifest = serde_json::json!({
            "parameters": [
                {"target":"bias","shape":[2],"sources":["b"],"transform":"identity",
                 "converted_tensor_sha256":digest(&bytes(&[1.0,-2.0]))},
                {"target":"conv","shape":[1,3,2],"sources":["c"],"transform":"OI-kernel_to_O-kernel-I",
                 "converted_tensor_sha256":digest(&bytes(&[0.0,3.0,1.0,4.0,2.0,5.0]))},
                {"target":"linear","shape":[3,2],"sources":["q","s","z"],"transform":"UINT8_dequantize_then_I-O_to_O-I",
                 "converted_tensor_sha256":digest(&bytes(&[0.5,2.0,1.0,2.5,1.5,3.0]))}
            ],
            "initializers":{
                "q":{"shape":[2,3],"dtype":"u1","offset":0,"bytes":6},
                "s":{"shape":[],"dtype":"<f4","scalar_hex":"0000003f"},
                "z":{"shape":[],"dtype":"u1","scalar_hex":"01"},
                "b":{"shape":[2],"dtype":"<f4","offset":6,"bytes":8},
                "c":{"shape":[1,2,3],"dtype":"<f4","offset":14,"bytes":24}
            }
        });
        let manifest = super::super::manifest::parse_manifest(&manifest.to_string()).unwrap();
        let recipe = Recipe {
            source_sha256: digest(&source),
            source_bytes: source.len() as u64,
            graph_sha256: String::new(),
            manifest_sha256: String::new(),
            weights_sha256: digest(&expected),
            weights_bytes: expected.len() as u64,
            tensors: vec![
                Tensor {
                    initializer: "b".into(),
                    parameter: "bias".into(),
                    shape: vec![2],
                    offset: 0,
                    bytes: 8,
                },
                Tensor {
                    initializer: "c".into(),
                    parameter: "conv".into(),
                    shape: vec![1, 2, 3],
                    offset: 8,
                    bytes: 24,
                },
                Tensor {
                    initializer: "q_dequantized".into(),
                    parameter: "linear".into(),
                    shape: vec![2, 3],
                    offset: 32,
                    bytes: 24,
                },
            ],
        };
        (source, manifest, recipe, expected)
    }

    #[test]
    fn reconstructs_original_windows_layout_and_rejects_hostile_recipes() {
        let temp = tempfile::tempdir().unwrap();
        let input = temp.path().join("source");
        let output = temp.path().join("weights.part");
        let (source, manifest, recipe, expected) = fixture();
        fs::write(&input, &source).unwrap();
        write_prepared(
            &mut File::open(&input).unwrap(),
            &output,
            &manifest,
            &recipe,
            &AtomicBool::new(false),
        )
        .unwrap();
        assert_eq!(fs::read(&output).unwrap(), expected);
        fs::remove_file(&output).unwrap();
        for bad in [
            "duplicate",
            "missing",
            "unknown",
            "name",
            "shape",
            "span",
            "gap",
            "source_size",
            "source_hash",
            "parameter_hash",
            "output_hash",
            "cancel",
            "tamper",
        ] {
            let (mut source, mut manifest, mut recipe, _) = fixture();
            match bad {
                "duplicate" => recipe.tensors.push(recipe.tensors[0].clone()),
                "missing" => {
                    recipe.tensors.pop();
                }
                "unknown" => recipe.tensors[0].parameter = "unknown".into(),
                "name" => recipe.tensors[0].initializer = "unknown".into(),
                "shape" => recipe.tensors[2].shape.reverse(),
                "span" => recipe.tensors[0].bytes += 4,
                "gap" => recipe.tensors[1].offset += 4,
                "source_size" => recipe.source_bytes += 1,
                "source_hash" => recipe.source_sha256 = "wrong".into(),
                "parameter_hash" => {
                    manifest.parameters[1].converted_tensor_sha256 = Some("wrong".into())
                }
                "output_hash" => recipe.weights_sha256 = "wrong".into(),
                "tamper" => source[0] ^= 1,
                "cancel" => {}
                _ => unreachable!(),
            }
            fs::write(&input, source).unwrap();
            assert!(
                write_prepared(
                    &mut File::open(&input).unwrap(),
                    &output,
                    &manifest,
                    &recipe,
                    &AtomicBool::new(bad == "cancel")
                )
                .is_err(),
                "{bad}"
            );
            assert!(!output.exists(), "failed preparation left an output: {bad}");
        }
        fs::write(&output, b"occupied").unwrap();
        fs::write(&input, &source).unwrap();
        assert!(
            write_prepared(
                &mut File::open(&input).unwrap(),
                &output,
                &manifest,
                &recipe,
                &AtomicBool::new(false)
            )
            .is_err()
        );
        assert_eq!(fs::read(&output).unwrap(), b"occupied");
    }

    #[test]
    fn frozen_recipe_covers_the_qualified_external_store() {
        let manifest = super::super::manifest::parse_manifest(MANIFEST).unwrap();
        let recipe: Recipe = serde_json::from_str(RECIPE).unwrap();
        assert_eq!(validate(&manifest, &recipe).unwrap().len(), 422);
        assert_eq!(recipe.manifest_sha256, digest(MANIFEST.as_bytes()));
        assert_eq!(recipe.source_sha256, super::super::SOURCE_SHA256);
        assert_eq!(recipe.source_bytes, 365352120);
        assert_eq!(
            recipe.graph_sha256,
            "233ac8980dab9c142ef9fe91ab1d3b5d6309a9549f59a284e50199f7b1081699"
        );
        assert_eq!(recipe.weights_bytes, 1301979472);
        assert_eq!(
            recipe.weights_sha256,
            "ebb4aa54f64240fe95c079c0b3ec326498f647ebe8d9bef50d316d47a59a53e5"
        );
    }

    #[cfg(windows)]
    #[test]
    #[ignore = "writes CTC staging beneath the explicitly supplied private model root; no inference"]
    fn pinned_source_reconstructs_qualified_store() {
        use crate::{dispatch::Runtime, operation::Operation};
        use speakerdesk_core::ipc::{Request, Response};
        use windows_sys::Win32::System::{
            ProcessStatus::{GetProcessMemoryInfo, PROCESS_MEMORY_COUNTERS_EX},
            Threading::GetCurrentProcess,
        };
        let root = std::path::PathBuf::from(
            std::env::var_os("SPEAKERDESK_TEST_CTC_PREPARE")
                .expect("explicit private model root with staged CTC graph, source and tokens"),
        );
        let directory = root.join(".staging/coarse-alignment");
        assert!(
            !directory.join("ctc.weights").exists(),
            "prepare a fresh private copy"
        );
        let started = std::time::Instant::now();
        let mut runtime = Runtime::new(root).unwrap();
        assert_eq!(
            runtime
                .execute(Request::PrepareAlignment, &Operation::native().unwrap())
                .unwrap(),
            Response::Ack
        );
        // Independent readback admits and locks both downloaded and prepared files.
        let admission =
            crate::windows_runtime::admit_model(&directory, "coarse-alignment").unwrap();
        assert!(
            File::options()
                .write(true)
                .open(directory.join("ctc.weights"))
                .is_err(),
            "prepared store must retain its admission handle"
        );
        drop(admission);
        assert!(!directory.join("ctc.weights.part").exists());
        let mut memory: PROCESS_MEMORY_COUNTERS_EX = unsafe { std::mem::zeroed() };
        let size = std::mem::size_of_val(&memory) as u32;
        memory.cb = size;
        assert_ne!(
            unsafe {
                GetProcessMemoryInfo(
                    GetCurrentProcess(),
                    (&mut memory as *mut PROCESS_MEMORY_COUNTERS_EX).cast(),
                    size,
                )
            },
            0,
            "GetProcessMemoryInfo: {}",
            std::io::Error::last_os_error()
        );
        assert!(memory.PrivateUsage > 0 && memory.PeakPagefileUsage >= memory.PrivateUsage);
        eprintln!(
            "CTC_PREPARED {}",
            serde_json::json!({
                "pid":std::process::id(), "seconds":started.elapsed().as_secs_f64(),
                "source":"GetProcessMemoryInfo/PROCESS_MEMORY_COUNTERS_EX",
                "current_bytes":memory.PrivateUsage, "lifetime_peak_bytes":memory.PeakPagefileUsage,
                "weights_bytes":1_301_979_472_u64,
                "weights_sha256":"ebb4aa54f64240fe95c079c0b3ec326498f647ebe8d9bef50d316d47a59a53e5",
            })
        );
    }
}
