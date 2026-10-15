//! Qualification comes from the owned executable and its application manifest.
use std::{
    fs::{self, File, OpenOptions},
    io::{Read, Write},
    path::{Path, PathBuf},
};

use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use sha2::{Digest, Sha256};

use super::{CaptureError, PROCESSED};

pub(super) const EXPORT: &str = "apple_processed_mono16k_v1";
pub(super) const PROGRESS: &str = "audio_queue_fence_v1";
const CAPABILITY: &str = "apple_input_node_output_route_v1";
const WINDOWS_EXPORT: &str = "raw_dual_mono_stereo16k_v1";
const WINDOWS_CAPABILITY: &str = "windows_wasapi_shared_v1";

#[derive(Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct Qualification {
    helper_sha256: String,
    export_contract: String,
    capability_version: String,
    progress_contract: String,
}

#[derive(Clone)]
pub(super) struct Helper {
    pub path: PathBuf,
    sha256: String,
    qualification: Option<Qualification>,
}

impl Helper {
    pub fn inspect(path: &Path) -> Result<Self, CaptureError> {
        let metadata = fs::symlink_metadata(path)?;
        if !metadata.is_file() || metadata.file_type().is_symlink() {
            return Err(invalid(
                "Capture helper must be a regular owned executable.",
            ));
        }
        let path = path.canonicalize()?;
        let mut hasher = Sha256::new();
        let mut file = File::open(&path)?;
        let mut block = [0; 65536];
        loop {
            let count = file.read(&mut block)?;
            if count == 0 {
                break;
            }
            hasher.update(&block[..count]);
        }
        let sha256 = format!("{:x}", hasher.finalize());
        let manifest = PathBuf::from(format!("{}.capability.json", path.display()));
        let qualification = match fs::symlink_metadata(&manifest) {
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => None,
            Err(error) => return Err(error.into()),
            Ok(metadata) => {
                if !metadata.is_file() || metadata.file_type().is_symlink() || metadata.len() > 4096
                {
                    return Err(invalid("Unsafe capture helper capability manifest."));
                }
                #[cfg(unix)]
                {
                    use std::os::unix::fs::MetadataExt;
                    if (metadata.uid() != 0 && metadata.uid() != unsafe { libc::getuid() })
                        || metadata.mode() & 0o022 != 0
                    {
                        return Err(invalid("Capture capability is not application owned."));
                    }
                }
                let value: Qualification = serde_json::from_slice(&fs::read(manifest)?)
                    .map_err(|_| invalid("Invalid capture helper capability manifest."))?;
                let supported = matches!(
                    (
                        value.export_contract.as_str(),
                        value.capability_version.as_str()
                    ),
                    (EXPORT, CAPABILITY) | (WINDOWS_EXPORT, WINDOWS_CAPABILITY)
                );
                if value.helper_sha256 != sha256
                    || !supported
                    || value.progress_contract != PROGRESS
                {
                    return Err(invalid("Capture capability differs from its built helper."));
                }
                Some(value)
            }
        };
        Ok(Self {
            path,
            sha256,
            qualification,
        })
    }

    pub fn progress_qualified(&self) -> bool {
        self.qualification.is_some()
    }

    pub fn processed_microphone(&self) -> bool {
        self.qualification
            .as_ref()
            .is_some_and(|value| value.export_contract == EXPORT)
    }

    pub fn start_contract(&self, dual_source: bool) -> Option<&str> {
        self.qualification
            .as_ref()
            .filter(|value| dual_source || value.export_contract == WINDOWS_EXPORT)
            .map(|value| value.export_contract.as_str())
    }

    pub fn receipt(
        &self,
        job_id: &str,
        packet: &Value,
        folder: &Path,
    ) -> Result<ProcessingReceipt, CaptureError> {
        let qualification = self
            .qualification
            .as_ref()
            .filter(|value| value.export_contract == EXPORT)
            .ok_or_else(|| invalid("Capture producer is not qualified by the application."))?;
        let parsed: Packet = serde_json::from_value(packet.clone())
            .map_err(|_| invalid("Malformed capture processing receipt."))?;
        parsed.validate(job_id, &self.sha256)?;
        let catalog = source_catalog(job_id);
        let catalog_sha256 = digest(&serde_json::to_vec(&catalog).expect("JSON value"));
        let metadata = json!({"schema_version":1,"job_id":job_id,
            "capture_source":{"catalog_sha256":catalog_sha256,"source_id":"microphone_clean","source_revision":1},
            "producer":packet,"classification":PROCESSED,"qualification":qualification});
        let bytes = serde_json::to_vec(&metadata).expect("JSON value");
        if bytes.len() > 8192 {
            return Err(invalid("Capture processing receipt is too large."));
        }
        let mut file = private_create(&folder.join("capture-processing.json"))?;
        file.write_all(&bytes)?;
        file.sync_all()?;
        let sha256 = digest(&bytes);
        Ok(ProcessingReceipt {
            metadata,
            catalog,
            sha256,
        })
    }
}

pub(super) fn source_catalog(job_id: &str) -> Value {
    json!({"schema_version":1,"job_id":job_id,"sample_rate":16000,
        "clock":"recording_samples","policy":"aligned_aec_microphone_and_original_system_v1",
        "sources":[{"source_id":"microphone_clean","basename":"microphone_clean.wav","revision":1},
                   {"source_id":"system","basename":"system.wav","revision":1}]})
}

/// Immutable server receipt. Its digest is the production unity input contract.
#[derive(Debug, Clone, PartialEq)]
pub struct ProcessingReceipt {
    pub metadata: Value,
    pub catalog: Value,
    pub sha256: String,
}

pub(super) fn private_create(path: &Path) -> Result<File, CaptureError> {
    let mut options = OpenOptions::new();
    options.read(true).write(true).create_new(true);
    #[cfg(unix)]
    {
        use std::os::unix::fs::OpenOptionsExt;
        options.mode(0o600).custom_flags(libc::O_NOFOLLOW);
    }
    Ok(options.open(path)?)
}

pub(super) fn invalid(message: impl Into<String>) -> CaptureError {
    CaptureError::Protocol(message.into())
}
fn digest(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}

#[derive(Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
struct Format {
    sample_rate: u32,
    channels: u32,
    layout: String,
}
impl Format {
    fn valid(&self) -> bool {
        self.sample_rate > 0 && self.channels > 0
    }
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Formats {
    input_output: Format,
    output_input: Format,
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Route {
    node: String,
    bus: u32,
    tap_scope: String,
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Readbacks {
    input_enabled: bool,
    output_enabled: bool,
    engine_running: bool,
    bypassed: bool,
    manual_rendering: bool,
    input_muted: bool,
    agc_enabled: bool,
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Packet {
    #[serde(rename = "type")]
    kind: String,
    schema_version: u32,
    job_id: String,
    helper_sha256: String,
    processor: String,
    export_contract: String,
    output: String,
    stage: String,
    postprocessing: String,
    stream_epoch: u32,
    format: Format,
    readbacks: Readbacks,
    capability_version: String,
    route: Route,
    state_phase: String,
    channel_mapping: String,
    callback_format: Format,
    io_formats: Formats,
}
impl Packet {
    fn validate(&self, job_id: &str, sha256: &str) -> Result<(), CaptureError> {
        let callback = &self.callback_format;
        let render = &self.io_formats.output_input;
        let mapping = (callback.channels == 1
            && callback.layout == "mono"
            && self.channel_mapping == "mono_direct")
            || (callback.channels == 9
                && callback.layout == "discrete_float32_noninterleaved"
                && self.channel_mapping == "nine_discrete_exact_replicas");
        let render_supported = render == callback
            || (render.sample_rate == callback.sample_rate
                && render.channels == 2
                && render.layout == "stereo_float32_noninterleaved");
        let read = &self.readbacks;
        // AGC is observed, but it did not confer qualification in 0.6.6.
        let _agc_observation = read.agc_enabled;
        if self.kind != "capture_processing"
            || self.schema_version != 1
            || self.job_id != job_id
            || self.helper_sha256 != sha256
            || self.stream_epoch != 0
            || self.processor != "apple_voice_processing_io"
            || self.capability_version != CAPABILITY
            || self.export_contract != EXPORT
            || self.output != "processed_microphone"
            || self.route.node != "inputNode"
            || self.route.bus != 0
            || self.route.tap_scope != "output"
            || self.state_phase != "after_start"
            || self.stage != "retained_source_pcm"
            || self.postprocessing != "identity"
            || self.format
                != (Format {
                    sample_rate: 16000,
                    channels: 1,
                    layout: "mono".into(),
                })
            || !callback.valid()
            || callback != &self.io_formats.input_output
            || !render_supported
            || !mapping
            || !read.input_enabled
            || !read.output_enabled
            || !read.engine_running
            || read.bypassed
            || read.manual_rendering
            || read.input_muted
        {
            return Err(invalid(
                "Capture processing is not the qualified owned microphone route.",
            ));
        }
        Ok(())
    }
}
