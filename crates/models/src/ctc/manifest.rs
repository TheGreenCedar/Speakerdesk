//! Byte-offset reader for `speakerdesk/alignment-mlx/manifest.json`.
//!
//! Each initializer is either a file span (`offset` + `bytes`) or an inline
//! `scalar_hex`. Parameters apply one of the three pinned transforms from
//! `alignment_model._convert`: `identity`, `OI-kernel_to_O-kernel-I`, and
//! `UINT8_dequantize_then_I-O_to_O-I`. Reading stops at those bytes. This
//! module does not open `weights.npz`.

use super::json::{self, Json};

/// Failure while reading the alignment manifest or applying a transform.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
#[error("{message}")]
pub struct ManifestError {
    message: String,
}

impl ManifestError {
    pub(super) fn new(message: impl Into<String>) -> Self {
        Self {
            message: message.into(),
        }
    }
}

/// How a source initializer is stored in the ONNX blob.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Dtype {
    /// Little-endian IEEE-754 binary32 (`<f4`).
    F32Le,
    /// Unsigned byte (`u1`).
    U8,
}

/// Layout change applied after the source bytes are decoded.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Transform {
    /// Copy a float32 tensor unchanged.
    Identity,
    /// `ndarray.transpose(0, 2, 1)` on an `[O, I, kernel]` weight.
    OiKernelToOKernelI,
    /// `(uint8 - zero) * scale`, then transpose `[I, O]` to `[O, I]`.
    Uint8DequantizeThenIoToOi,
}

/// One ONNX initializer named by the manifest.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Initializer {
    pub name: String,
    pub shape: Vec<usize>,
    pub dtype: Dtype,
    /// Byte offset into the source model. Absent for `scalar_hex` values.
    pub offset: Option<u64>,
    /// Byte length of the file span. Absent for `scalar_hex` values.
    pub nbytes: Option<u64>,
    pub scalar_hex: Option<String>,
}

/// One converted parameter and the sources it reads.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Parameter {
    pub target: String,
    pub shape: Vec<usize>,
    pub sources: Vec<String>,
    pub transform: Transform,
}

/// File span for an initializer that lives in the source model bytes.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ByteSpan {
    pub name: String,
    pub offset: u64,
    pub nbytes: u64,
}

/// Parsed `parameters` and `initializers` from the alignment manifest.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Manifest {
    pub parameters: Vec<Parameter>,
    pub initializers: Vec<Initializer>,
}

/// Decoded initializer, still in source dtype.
#[derive(Debug, Clone, PartialEq)]
pub enum SourceTensor {
    F32 { shape: Vec<usize>, data: Vec<f32> },
    U8 { shape: Vec<usize>, data: Vec<u8> },
}

/// Contiguous float32 tensor after a transform.
#[derive(Debug, Clone, PartialEq)]
pub struct FloatTensor {
    pub shape: Vec<usize>,
    pub data: Vec<f32>,
}

impl Manifest {
    /// File spans in manifest order. Inline scalars are omitted.
    pub fn byte_spans(&self) -> Vec<ByteSpan> {
        self.initializers
            .iter()
            .filter_map(|item| {
                Some(ByteSpan {
                    name: item.name.clone(),
                    offset: item.offset?,
                    nbytes: item.nbytes?,
                })
            })
            .collect()
    }

    fn initializer(&self, name: &str) -> Result<&Initializer, ManifestError> {
        self.initializers
            .iter()
            .find(|item| item.name == name)
            .ok_or_else(|| ManifestError::new(format!("Missing source initializer: {name}")))
    }
}

/// Parse the alignment manifest. Unknown keys are ignored.
pub fn parse_manifest(text: &str) -> Result<Manifest, ManifestError> {
    let root = json::parse(text)?;
    let fields = object(&root)?;
    let parameters = array(required(fields, "parameters")?)?
        .iter()
        .map(parse_parameter)
        .collect::<Result<Vec<_>, _>>()?;
    let mut initializers = Vec::new();
    for (name, value) in object(required(fields, "initializers")?)? {
        if initializers
            .iter()
            .any(|item: &Initializer| item.name == *name)
        {
            return Err(ManifestError::new(format!("duplicate initializer {name}")));
        }
        initializers.push(parse_initializer(name, value)?);
    }
    Ok(Manifest {
        parameters,
        initializers,
    })
}

/// Read one initializer from `source` or from its inline scalar.
pub fn read_initializer(
    source: &[u8],
    initializer: &Initializer,
) -> Result<SourceTensor, ManifestError> {
    let expected = usize::try_from(expected_nbytes(initializer)?)
        .map_err(|_| ManifestError::new("initializer shape overflows"))?;
    let raw = if let Some(hex) = &initializer.scalar_hex {
        let raw = decode_hex(hex)?;
        if raw.len() != expected {
            return Err(ManifestError::new("Source tensor data is truncated."));
        }
        raw
    } else {
        let offset = initializer
            .offset
            .ok_or_else(|| ManifestError::new("initializer has no offset or scalar"))?;
        let nbytes = initializer
            .nbytes
            .ok_or_else(|| ManifestError::new("initializer has no byte length"))?;
        if nbytes != expected as u64 {
            return Err(ManifestError::new("Source tensor byte length differs."));
        }
        let start = usize::try_from(offset)
            .map_err(|_| ManifestError::new("Source tensor data is truncated."))?;
        let end = start
            .checked_add(expected)
            .ok_or_else(|| ManifestError::new("Source tensor data is truncated."))?;
        source
            .get(start..end)
            .ok_or_else(|| ManifestError::new("Source tensor data is truncated."))?
            .to_vec()
    };
    match initializer.dtype {
        Dtype::U8 => Ok(SourceTensor::U8 {
            shape: initializer.shape.clone(),
            data: raw,
        }),
        Dtype::F32Le => {
            let mut data = Vec::with_capacity(raw.len() / 4);
            for chunk in raw.as_chunks::<4>().0 {
                data.push(f32::from_le_bytes(*chunk));
            }
            Ok(SourceTensor::F32 {
                shape: initializer.shape.clone(),
                data,
            })
        }
    }
}

/// Apply one of the three pinned transforms. The result is contiguous float32.
pub fn apply_transform(
    transform: Transform,
    inputs: &[SourceTensor],
) -> Result<FloatTensor, ManifestError> {
    match transform {
        Transform::Identity => identity(inputs),
        Transform::OiKernelToOKernelI => oi_kernel_to_o_kernel_i(inputs),
        Transform::Uint8DequantizeThenIoToOi => uint8_dequantize_then_transpose(inputs),
    }
}

/// Read sources for `parameter` and apply its transform.
pub fn convert_parameter(
    source: &[u8],
    manifest: &Manifest,
    parameter: &Parameter,
) -> Result<FloatTensor, ManifestError> {
    let mut inputs = Vec::with_capacity(parameter.sources.len());
    for name in &parameter.sources {
        inputs.push(read_initializer(source, manifest.initializer(name)?)?);
    }
    let tensor = apply_transform(parameter.transform, &inputs)?;
    if tensor.shape != parameter.shape {
        return Err(ManifestError::new(format!(
            "Converted tensor identity differs: {}",
            parameter.target
        )));
    }
    Ok(tensor)
}

fn identity(inputs: &[SourceTensor]) -> Result<FloatTensor, ManifestError> {
    let [SourceTensor::F32 { shape, data }] = inputs else {
        return Err(ManifestError::new("identity requires one float32 tensor"));
    };
    Ok(FloatTensor {
        shape: shape.clone(),
        data: data.clone(),
    })
}

fn oi_kernel_to_o_kernel_i(inputs: &[SourceTensor]) -> Result<FloatTensor, ManifestError> {
    let [SourceTensor::F32 { shape, data }] = inputs else {
        return Err(ManifestError::new(
            "OI-kernel transform requires one float32 tensor",
        ));
    };
    let [out_channels, in_channels, kernel] = shape.as_slice() else {
        return Err(ManifestError::new(
            "OI-kernel transform requires a rank-3 tensor",
        ));
    };
    let mut transposed = vec![0.0; data.len()];
    for o in 0..*out_channels {
        for i in 0..*in_channels {
            for k in 0..*kernel {
                let src = (o * in_channels + i) * kernel + k;
                let dst = (o * kernel + k) * in_channels + i;
                transposed[dst] = data[src];
            }
        }
    }
    Ok(FloatTensor {
        shape: vec![*out_channels, *kernel, *in_channels],
        data: transposed,
    })
}

fn uint8_dequantize_then_transpose(inputs: &[SourceTensor]) -> Result<FloatTensor, ManifestError> {
    let [quantized, scale, zero] = inputs else {
        return Err(ManifestError::new(
            "UINT8 dequantize requires quantized weights, scale, and zero point",
        ));
    };
    let SourceTensor::U8 {
        shape: q_shape,
        data: q,
    } = quantized
    else {
        return Err(ManifestError::new("quantized weights must be uint8"));
    };
    let [rows, cols] = q_shape.as_slice() else {
        return Err(ManifestError::new("quantized weights must be rank 2"));
    };
    let SourceTensor::F32 {
        shape: scale_shape,
        data: scale_data,
    } = scale
    else {
        return Err(ManifestError::new("scale must be float32"));
    };
    if !scale_shape.is_empty() || scale_data.len() != 1 {
        return Err(ManifestError::new("scale must be a scalar"));
    }
    let SourceTensor::U8 {
        shape: zero_shape,
        data: zero_data,
    } = zero
    else {
        return Err(ManifestError::new("zero point must be uint8"));
    };
    if !zero_shape.is_empty() || zero_data.len() != 1 {
        return Err(ManifestError::new("zero point must be a scalar"));
    }
    let scale = scale_data[0];
    let zero = f32::from(zero_data[0]);
    let mut data = vec![0.0; q.len()];
    for row in 0..*rows {
        for col in 0..*cols {
            let value = (f32::from(q[row * cols + col]) - zero) * scale;
            data[col * rows + row] = value;
        }
    }
    Ok(FloatTensor {
        shape: vec![*cols, *rows],
        data,
    })
}

fn parse_parameter(value: &Json) -> Result<Parameter, ManifestError> {
    let fields = object(value)?;
    Ok(Parameter {
        target: string(required(fields, "target")?)?,
        shape: shape(required(fields, "shape")?)?,
        sources: array(required(fields, "sources")?)?
            .iter()
            .map(string)
            .collect::<Result<Vec<_>, _>>()?,
        transform: parse_transform(&string(required(fields, "transform")?)?)?,
    })
}

fn parse_initializer(name: &str, value: &Json) -> Result<Initializer, ManifestError> {
    let fields = object(value)?;
    let scalar_hex = optional(fields, "scalar_hex").map(string).transpose()?;
    let offset = optional(fields, "offset").map(u64_field).transpose()?;
    let nbytes = optional(fields, "bytes").map(u64_field).transpose()?;
    if scalar_hex.is_none() && (offset.is_none() || nbytes.is_none()) {
        return Err(ManifestError::new(format!(
            "{name} needs offset and bytes, or scalar_hex"
        )));
    }
    Ok(Initializer {
        name: name.to_string(),
        shape: shape(required(fields, "shape")?)?,
        dtype: parse_dtype(&string(required(fields, "dtype")?)?)?,
        offset,
        nbytes,
        scalar_hex,
    })
}

fn parse_transform(name: &str) -> Result<Transform, ManifestError> {
    match name {
        "identity" => Ok(Transform::Identity),
        "OI-kernel_to_O-kernel-I" => Ok(Transform::OiKernelToOKernelI),
        "UINT8_dequantize_then_I-O_to_O-I" => Ok(Transform::Uint8DequantizeThenIoToOi),
        _ => Err(ManifestError::new("Unknown pinned tensor transform.")),
    }
}

fn parse_dtype(name: &str) -> Result<Dtype, ManifestError> {
    match name {
        "<f4" => Ok(Dtype::F32Le),
        "u1" => Ok(Dtype::U8),
        _ => Err(ManifestError::new(format!(
            "unknown initializer dtype {name}"
        ))),
    }
}

fn expected_nbytes(initializer: &Initializer) -> Result<u64, ManifestError> {
    let count = element_count(&initializer.shape)?;
    let width: u64 = match initializer.dtype {
        Dtype::U8 => 1,
        Dtype::F32Le => 4,
    };
    u64::try_from(count)
        .ok()
        .and_then(|count| count.checked_mul(width))
        .ok_or_else(|| ManifestError::new("initializer shape overflows"))
}

fn element_count(shape: &[usize]) -> Result<usize, ManifestError> {
    shape
        .iter()
        .try_fold(1usize, |acc, dim| acc.checked_mul(*dim))
        .ok_or_else(|| ManifestError::new("initializer shape overflows"))
}

fn decode_hex(hex: &str) -> Result<Vec<u8>, ManifestError> {
    if !hex.len().is_multiple_of(2) || !hex.bytes().all(|byte| byte.is_ascii_hexdigit()) {
        return Err(ManifestError::new("invalid scalar hex"));
    }
    let mut raw = Vec::with_capacity(hex.len() / 2);
    for index in (0..hex.len()).step_by(2) {
        let byte = u8::from_str_radix(&hex[index..index + 2], 16)
            .map_err(|_| ManifestError::new("invalid scalar hex"))?;
        raw.push(byte);
    }
    Ok(raw)
}

fn object(value: &Json) -> Result<&[(String, Json)], ManifestError> {
    match value {
        Json::Object(fields) => Ok(fields),
        _ => Err(ManifestError::new("expected JSON object")),
    }
}

fn array(value: &Json) -> Result<&[Json], ManifestError> {
    match value {
        Json::Array(items) => Ok(items),
        _ => Err(ManifestError::new("expected JSON array")),
    }
}

fn required<'a>(fields: &'a [(String, Json)], key: &str) -> Result<&'a Json, ManifestError> {
    optional(fields, key).ok_or_else(|| ManifestError::new(format!("missing field {key}")))
}

fn optional<'a>(fields: &'a [(String, Json)], key: &str) -> Option<&'a Json> {
    fields
        .iter()
        .find(|(name, _)| name == key)
        .map(|(_, value)| value)
}

fn string(value: &Json) -> Result<String, ManifestError> {
    match value {
        Json::String(text) => Ok(text.clone()),
        _ => Err(ManifestError::new("expected JSON string")),
    }
}

fn shape(value: &Json) -> Result<Vec<usize>, ManifestError> {
    array(value)?
        .iter()
        .map(|item| {
            let number = u64_field(item)?;
            usize::try_from(number).map_err(|_| ManifestError::new("shape dimension overflows"))
        })
        .collect()
}

fn u64_field(value: &Json) -> Result<u64, ManifestError> {
    match value {
        Json::Number(text) if !text.is_empty() && !text.contains(['-', '.', 'e', 'E']) => text
            .parse()
            .map_err(|_| ManifestError::new("expected unsigned integer")),
        _ => Err(ManifestError::new("expected unsigned integer")),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Tiny stand-in for `speakerdesk/alignment-mlx/manifest.json`.
    /// Offsets point into the buffer built below. Scalars stay inline.
    const SYNTHETIC_MANIFEST: &str = r#"{
      "parameters": [
        {
          "target": "identity.bias",
          "shape": [2],
          "sources": ["bias"],
          "transform": "identity"
        },
        {
          "target": "conv.weight",
          "shape": [1, 3, 2],
          "sources": ["conv"],
          "transform": "OI-kernel_to_O-kernel-I"
        },
        {
          "target": "layer.weight",
          "shape": [2, 2],
          "sources": ["q", "scale", "zero"],
          "transform": "UINT8_dequantize_then_I-O_to_O-I"
        }
      ],
      "initializers": {
        "q": {"shape": [2, 2], "dtype": "u1", "offset": 0, "bytes": 4},
        "bias": {"shape": [2], "dtype": "<f4", "offset": 4, "bytes": 8},
        "conv": {"shape": [1, 2, 3], "dtype": "<f4", "offset": 12, "bytes": 24},
        "scale": {"shape": [], "dtype": "<f4", "scalar_hex": "0000003f"},
        "zero": {"shape": [], "dtype": "u1", "scalar_hex": "01"}
      }
    }"#;

    fn source_bytes() -> Vec<u8> {
        let mut source = Vec::new();
        source.extend_from_slice(&[2, 3, 4, 5]);
        source.extend_from_slice(&1.0f32.to_le_bytes());
        source.extend_from_slice(&(-2.0f32).to_le_bytes());
        for value in [0.0f32, 1.0, 2.0, 3.0, 4.0, 5.0] {
            source.extend_from_slice(&value.to_le_bytes());
        }
        source
    }

    #[test]
    fn reads_offsets_and_applies_the_three_transforms() {
        let manifest = parse_manifest(SYNTHETIC_MANIFEST).unwrap();
        assert_eq!(
            manifest.byte_spans(),
            vec![
                ByteSpan {
                    name: "q".to_string(),
                    offset: 0,
                    nbytes: 4
                },
                ByteSpan {
                    name: "bias".to_string(),
                    offset: 4,
                    nbytes: 8
                },
                ByteSpan {
                    name: "conv".to_string(),
                    offset: 12,
                    nbytes: 24
                },
            ]
        );

        let source = source_bytes();
        let identity = convert_parameter(&source, &manifest, &manifest.parameters[0]).unwrap();
        assert_eq!(identity.shape, vec![2]);
        assert_eq!(identity.data, vec![1.0, -2.0]);

        let kernel = convert_parameter(&source, &manifest, &manifest.parameters[1]).unwrap();
        assert_eq!(kernel.shape, vec![1, 3, 2]);
        assert_eq!(kernel.data, vec![0.0, 3.0, 1.0, 4.0, 2.0, 5.0]);

        // (q - 1) * 0.5, then transpose, matching alignment_model._convert.
        let weight = convert_parameter(&source, &manifest, &manifest.parameters[2]).unwrap();
        assert_eq!(weight.shape, vec![2, 2]);
        assert_eq!(weight.data, vec![0.5, 1.5, 1.0, 2.0]);
    }

    #[test]
    fn rejects_a_byte_length_that_disagrees_with_shape() {
        let text = r#"{
          "parameters": [],
          "initializers": {
            "bias": {"shape": [2], "dtype": "<f4", "offset": 0, "bytes": 4}
          }
        }"#;
        let manifest = parse_manifest(text).unwrap();
        let error = read_initializer(&[0; 8], &manifest.initializers[0]).unwrap_err();
        assert_eq!(error.to_string(), "Source tensor byte length differs.");
    }

    #[test]
    fn rejects_an_unknown_transform() {
        let text = r#"{
          "parameters": [{
            "target": "x",
            "shape": [1],
            "sources": ["x"],
            "transform": "transpose_guess"
          }],
          "initializers": {
            "x": {"shape": [1], "dtype": "<f4", "offset": 0, "bytes": 4}
          }
        }"#;
        let error = parse_manifest(text).unwrap_err();
        assert_eq!(error.to_string(), "Unknown pinned tensor transform.");
    }
}
