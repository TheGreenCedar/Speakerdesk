use super::{INPUT_SAMPLES, is_clipped, is_silent, prepared_audio};
use crate::weights::{error, verify};
use objc2::{
    AnyThread,
    rc::{Retained, autoreleasepool},
    runtime::ProtocolObject,
};
use objc2_core_ml::{
    MLComputeUnits, MLDictionaryFeatureProvider, MLFeatureProvider, MLModel, MLModelConfiguration,
    MLMultiArray, MLMultiArrayDataType,
};
use objc2_foundation::{NSArray, NSDictionary, NSNumber, NSString, NSURL, ns_string};
use serde::Deserialize;
use speakerdesk_core::{
    error::ChildError,
    voice::{ClipEmbedding, VoiceModel, normalized},
};
use std::{
    cell::Cell,
    collections::{BTreeMap, BTreeSet},
    ffi::c_void,
    path::{Path, PathBuf},
    ptr::NonNull,
    sync::atomic::{AtomicBool, Ordering},
};

pub const MODEL_ID: &str = "aufklarer/ReDimNet2-B6-CoreML";
pub const REVISION: &str = "112dd8f4f836abdf8a420e66a5e2885cf8ec64ab";
pub const ARTIFACT_SHA256: &str =
    "cf8cc300e240efaed56cb931bd7715ded41b66bf55af7e9a19d0b17e7fe986c9";
pub const DIMENSION: usize = 192;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ComputePolicy {
    CpuAndGpu,
    CpuAndNeuralEngine,
    All,
}
impl ComputePolicy {
    pub fn name(self) -> &'static str {
        match self {
            Self::CpuAndGpu => "CPU_AND_GPU",
            Self::CpuAndNeuralEngine => "CPU_AND_NEURAL_ENGINE",
            Self::All => "ALL",
        }
    }
    pub fn identity(self) -> VoiceModel {
        VoiceModel::new(
            MODEL_ID,
            format!("{REVISION}:sd-pcm-v1:native-coreml-v1:{}", self.name()),
            ARTIFACT_SHA256,
            DIMENSION,
            super::RATE,
        )
        .expect("pinned voice identity")
    }
}

/// Created, used and dropped by the serial inference executor; deliberately !Send.
pub struct Network {
    model: Retained<MLModel>,
    policy: ComputePolicy,
}
impl Network {
    pub fn load(directory: &Path, policy: ComputePolicy) -> Result<Self, ChildError> {
        verify_artifact(directory)?;
        autoreleasepool(|_| {
            let path = directory.join("ReDimNet2B6.mlmodelc");
            let path = path
                .to_str()
                .ok_or_else(|| error("Voice model path is not UTF-8"))?;
            let url = NSURL::fileURLWithPath(&NSString::from_str(path));
            // SAFETY: configuration and URL are correctly typed, live Objective-C values.
            unsafe {
                let configuration = MLModelConfiguration::new();
                configuration.setComputeUnits(match policy {
                    ComputePolicy::CpuAndGpu => MLComputeUnits::CPUAndGPU,
                    ComputePolicy::CpuAndNeuralEngine => MLComputeUnits::CPUAndNeuralEngine,
                    ComputePolicy::All => MLComputeUnits::All,
                });
                let model =
                    MLModel::modelWithContentsOfURL_configuration_error(&url, &configuration)
                        .map_err(error)?;
                Ok(Self { model, policy })
            }
        })
    }

    pub fn identity(&self) -> VoiceModel {
        self.policy.identity()
    }

    pub fn embed(&self, pcm: &[f32], cancel: &AtomicBool) -> Result<ClipEmbedding, ChildError> {
        check_cancel(cancel)?;
        let input = prepared_audio(pcm).map_err(error)?;
        if is_silent(pcm) || is_clipped(pcm) {
            return Ok(ClipEmbedding {
                vector: vec![],
                clean: false,
            });
        }
        let vector = autoreleasepool(|_| {
            // SAFETY: all Objective-C objects are owned on this executor and retained
            // through the synchronous prediction. Only our fresh input array is mutated.
            unsafe {
                let shape = NSArray::from_retained_slice(&[
                    NSNumber::new_i32(1),
                    NSNumber::new_usize(INPUT_SAMPLES),
                ]);
                let audio = MLMultiArray::initWithShape_dataType_error(
                    MLMultiArray::alloc(),
                    &shape,
                    MLMultiArrayDataType::Float32,
                )
                .map_err(error)?;
                let copied = Cell::new(false);
                // Core ML may replace backing storage; use the buffer and strides
                // supplied together by its scoped mutable-access API.
                audio.getMutableBytesWithHandler(&block2::StackBlock::new(
                    |bytes: NonNull<c_void>, size: isize, strides: NonNull<NSArray<NSNumber>>| {
                        let strides = strides.as_ref();
                        if size >= std::mem::size_of_val(input.as_slice()) as isize
                            && strides.len() == 2
                            && strides.objectAtIndex(0).unsignedIntegerValue() == INPUT_SAMPLES
                            && strides.objectAtIndex(1).unsignedIntegerValue() == 1
                        {
                            std::ptr::copy_nonoverlapping(
                                input.as_ptr(),
                                bytes.cast::<f32>().as_ptr(),
                                input.len(),
                            );
                            copied.set(true);
                        }
                    },
                ));
                if !copied.get() {
                    return Err(error("Core ML returned an unexpected input layout"));
                }
                let values = NSDictionary::from_slices(&[ns_string!("audio")], &[audio.as_ref()]);
                let features = MLDictionaryFeatureProvider::initWithDictionary_error(
                    MLDictionaryFeatureProvider::alloc(),
                    &values,
                )
                .map_err(error)?;
                check_cancel(cancel)?;
                let result = self
                    .model
                    .predictionFromFeatures_error(ProtocolObject::from_ref(&*features))
                    .map_err(error)?;
                check_cancel(cancel)?;
                let output = result
                    .featureValueForName(ns_string!("embedding"))
                    .and_then(|value| value.multiArrayValue())
                    .ok_or_else(|| error("Voice model did not return an embedding"))?;
                let shape: Vec<usize> = output
                    .shape()
                    .iter()
                    .map(|n| n.unsignedIntegerValue())
                    .collect();
                if !matches!(shape.as_slice(), [DIMENSION] | [1, DIMENSION])
                    || ![
                        MLMultiArrayDataType::Float16,
                        MLMultiArrayDataType::Float32,
                        MLMultiArrayDataType::Double,
                    ]
                    .contains(&output.dataType())
                {
                    return Err(error(
                        "Voice model returned an incompatible embedding layout",
                    ));
                }
                let vector: Vec<f64> = (0..DIMENSION)
                    .map(|i| output.objectAtIndexedSubscript(i as isize).doubleValue())
                    .collect();
                normalized(&vector, DIMENSION).map_err(error)
            }
        })?;
        Ok(ClipEmbedding {
            vector,
            clean: true,
        })
    }
}

fn check_cancel(cancel: &AtomicBool) -> Result<(), ChildError> {
    if cancel.load(Ordering::Acquire) {
        Err(ChildError::Cancelled)
    } else {
        Ok(())
    }
}

#[derive(Deserialize)]
struct FilePin {
    sha256: String,
    size: u64,
}
#[derive(Deserialize)]
struct Artifact {
    files: BTreeMap<String, FilePin>,
}

/// One load-time walk/hash pass. Core ML must not consume unpinned extra files.
pub fn verify_artifact(directory: &Path) -> Result<(), ChildError> {
    let artifact: Artifact = serde_json::from_str(include_str!(
        "../../../../speakerdesk/redimnet2_artifact.json"
    ))
    .map_err(error)?;
    if !directory.is_dir() || directory.is_symlink() {
        return Err(error("Voice model directory is missing or linked"));
    }
    for (name, pin) in &artifact.files {
        let mut path = directory.to_path_buf();
        for part in Path::new(name).components() {
            path.push(part);
            if path.is_symlink() {
                return Err(error("Voice model files must not be linked"));
            }
        }
        let meta = std::fs::metadata(path).map_err(error)?;
        if !meta.is_file() || meta.len() != pin.size {
            return Err(error(format!("Unexpected voice artifact size: {name}")));
        }
    }
    let mut expected: BTreeSet<PathBuf> = artifact
        .files
        .keys()
        .filter(|s| s.starts_with("ReDimNet2B6.mlmodelc/"))
        .map(PathBuf::from)
        .collect();
    let mut pending = vec![PathBuf::from("ReDimNet2B6.mlmodelc")];
    while let Some(relative) = pending.pop() {
        for entry in std::fs::read_dir(directory.join(relative)).map_err(error)? {
            let entry = entry.map_err(error)?;
            let path = entry.path();
            let relative = path.strip_prefix(directory).map_err(error)?.to_path_buf();
            let kind = entry.file_type().map_err(error)?;
            if kind.is_dir()
                && artifact
                    .files
                    .keys()
                    .any(|name| Path::new(name).starts_with(&relative))
            {
                pending.push(relative);
            } else if !kind.is_file() || !expected.remove(&relative) {
                return Err(error("Unexpected compiled voice artifact"));
            }
        }
    }
    if !expected.is_empty() {
        return Err(error("Missing compiled voice artifact"));
    }
    let files: Vec<_> = artifact
        .files
        .iter()
        .map(|(name, pin)| (name.as_str(), pin.sha256.as_str()))
        .collect();
    verify(directory, &files)
}
