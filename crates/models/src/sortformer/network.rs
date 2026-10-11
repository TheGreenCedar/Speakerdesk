//! Pinned Nemotron 3 graph. Native 10 ms frames remain distinct from padded tensors.
use crate::weights::{error, linear, norm, take, verify};
use mlx_rs::{
    Array, Dtype,
    module::{Module, Param},
    nn::{Conv1d, LayerNorm, Linear},
    ops::indexing::IndexOp,
};
use speakerdesk_core::error::ChildError;
use std::{
    path::Path,
    sync::atomic::{AtomicBool, Ordering},
};

const DTYPE: Dtype = Dtype::Bfloat16;
const HOP: usize = 160;
const FFT: usize = 512;
const DIM: i32 = 512;
const HEADS: i32 = 8;
const SPEAKERS: i32 = 8;
/// The first offline window emits 340 * 8 native frames before cache updates.
pub const OFFLINE_WINDOW_SAMPLES: usize = 340 * 8 * HOP;

struct Block {
    norm1: LayerNorm,
    qkv: Linear,
    attention_output: Linear,
    norm2: LayerNorm,
    expand: Linear,
    contract: Linear,
}

impl Block {
    fn forward(&mut self, input: &Array, mask: &Array) -> Result<Array, mlx_rs::error::Exception> {
        let length = input.dim(1);
        let qkv = self.qkv.forward(&self.norm1.forward(input)?)?.reshape(&[
            1,
            length,
            3,
            HEADS,
            DIM / HEADS,
        ])?;
        let q = qkv
            .index((.., .., 0, .., ..))
            .transpose_axes(&[0, 2, 1, 3])?;
        let k = qkv
            .index((.., .., 1, .., ..))
            .transpose_axes(&[0, 2, 1, 3])?;
        let v = qkv
            .index((.., .., 2, .., ..))
            .transpose_axes(&[0, 2, 1, 3])?;
        let q = mlx_rs::fast::rope(q, 64, false, 10000.0, 1.0, 0, None)?;
        let k = mlx_rs::fast::rope(k, 64, false, 10000.0, 1.0, 0, None)?;
        let attention = mlx_rs::fast::scaled_dot_product_attention(q, k, v, 0.125, mask, None)?
            .transpose_axes(&[0, 2, 1, 3])?
            .reshape(&[1, length, DIM])?;
        let x = input.add(self.attention_output.forward(&attention)?)?;
        let expanded = self.expand.forward(&self.norm2.forward(&x)?)?;
        let activated = gelu(&expanded)?;
        debug_assert_eq!(activated.dtype(), DTYPE);
        x.add(self.contract.forward(&activated)?)
    }
}

pub struct Network {
    window: Array,
    filterbank: Array,
    projection: Linear,
    embed_norm: LayerNorm,
    layers: Vec<Block>,
    final_norm: LayerNorm,
    speaker_projection: Linear,
    upsample: Conv1d,
    speaker_hidden: Linear,
    speaker_output: Linear,
}

impl Network {
    pub fn load(directory: &Path) -> Result<Self, ChildError> {
        verify(
            directory,
            &[
                (
                    "config.json",
                    "f215091252bc54f8dcc7d682962a659e03a6bcb26d1c86e4728b8c6730130eb4",
                ),
                (
                    "model.safetensors",
                    "21e8427d1795c9c46c5800f56b16061734ffd0dcadd71d9bcf0b4d6ef7261da5",
                ),
            ],
        )?;
        let mut weights =
            Array::load_safetensors(directory.join("model.safetensors")).map_err(error)?;
        let window = take(&mut weights, "preprocessor.window", &[400], Dtype::Float32)?;
        let filterbank = take(
            &mut weights,
            "preprocessor.fb",
            &[1, 128, 257],
            Dtype::Float32,
        )?;
        let projection = linear(
            &mut weights,
            "encoder.pre_encode.proj",
            1024,
            DIM,
            false,
            DTYPE,
        )?;
        let embed_norm = norm(&mut weights, "encoder.embed_norm", DIM, DTYPE)?;
        let mut layers = Vec::with_capacity(31);
        for index in 0..31 {
            let prefix = format!("encoder.layers.{index}");
            layers.push(Block {
                norm1: norm(&mut weights, &format!("{prefix}.norm1"), DIM, DTYPE)?,
                qkv: linear(
                    &mut weights,
                    &format!("{prefix}.attn.w_qkv"),
                    DIM,
                    3 * DIM,
                    false,
                    DTYPE,
                )?,
                attention_output: linear(
                    &mut weights,
                    &format!("{prefix}.attn.out_proj"),
                    DIM,
                    DIM,
                    true,
                    DTYPE,
                )?,
                norm2: norm(&mut weights, &format!("{prefix}.norm2"), DIM, DTYPE)?,
                expand: linear(
                    &mut weights,
                    &format!("{prefix}.ffn.linear1"),
                    DIM,
                    2048,
                    true,
                    DTYPE,
                )?,
                contract: linear(
                    &mut weights,
                    &format!("{prefix}.ffn.linear2"),
                    2048,
                    DIM,
                    true,
                    DTYPE,
                )?,
            });
        }
        let prefix = "sortformer_modules";
        let network = Self {
            window,
            filterbank,
            projection,
            embed_norm,
            layers,
            final_norm: norm(&mut weights, "encoder.final_norm", DIM, DTYPE)?,
            speaker_projection: linear(
                &mut weights,
                &format!("{prefix}.encoder_proj"),
                DIM,
                192,
                true,
                DTYPE,
            )?,
            upsample: Conv1d {
                weight: Param::new(take(
                    &mut weights,
                    &format!("{prefix}.subpixel_upsample.weight"),
                    &[1536, 3, 192],
                    DTYPE,
                )?),
                bias: Param::new(Some(take(
                    &mut weights,
                    &format!("{prefix}.subpixel_upsample.bias"),
                    &[1536],
                    DTYPE,
                )?)),
                stride: 1,
                padding: 1,
                dilation: 1,
                groups: 1,
            },
            speaker_hidden: linear(
                &mut weights,
                &format!("{prefix}.first_hidden_to_hidden"),
                192,
                192,
                true,
                DTYPE,
            )?,
            speaker_output: linear(
                &mut weights,
                &format!("{prefix}.single_hidden_to_spks"),
                192,
                SPEAKERS,
                true,
                DTYPE,
            )?,
        };
        // These training/stream-cache tensors are not used by the first offline window.
        for (name, shape) in [
            ("activity_head.layers.0.weight", vec![192]),
            ("activity_head.layers.0.bias", vec![192]),
            ("activity_head.layers.1.weight", vec![3, 192]),
            ("activity_head.layers.1.bias", vec![3]),
            ("hidden_to_spks.weight", vec![8, 384]),
            ("hidden_to_spks.bias", vec![8]),
            ("learnable_sil_emb", vec![512]),
        ] {
            take(&mut weights, &format!("{prefix}.{name}"), &shape, DTYPE)?;
        }
        if !weights.is_empty() {
            return Err(error("unexpected Nemotron weight inventory"));
        }
        Ok(network)
    }

    /// shortcut: one offline window; add AOSC/FIFO before accepting longer recordings.
    pub fn probabilities(
        &mut self,
        pcm: &[f32],
        cancel: &AtomicBool,
    ) -> Result<Vec<[f32; 8]>, ChildError> {
        if pcm.len() > OFFLINE_WINDOW_SAMPLES {
            return Err(error("Nemotron multi-window inference is not implemented"));
        }
        if pcm.iter().any(|s| !s.is_finite()) {
            return Err(error("Nemotron requires finite mono PCM"));
        }
        if cancel.load(Ordering::Acquire) {
            return Err(ChildError::Cancelled);
        }
        let valid = pcm.len() / HOP;
        if valid == 0 {
            return Ok(Vec::new());
        }
        let padded = (valid + 1).div_ceil(16) * 16;
        let features = self.features(pcm, padded).map_err(error)?;
        let probs =
            mlx_rs::with_stream(&mlx_rs::Stream::gpu(), || -> Result<Array, ChildError> {
                let stacked = features
                    .as_dtype(DTYPE)
                    .map_err(error)?
                    .reshape(&[1, (padded / 8) as i32, 1024])
                    .map_err(error)?;
                let mut x = self
                    .embed_norm
                    .forward(&self.projection.forward(&stacked).map_err(error)?)
                    .map_err(error)?;
                let valid_encoded = valid.div_ceil(8);
                let mask = Array::from_slice(
                    &(0..padded / 8)
                        .map(|i| i < valid_encoded)
                        .collect::<Vec<_>>(),
                    &[1, 1, 1, (padded / 8) as i32],
                );
                for layer in &mut self.layers {
                    if cancel.load(Ordering::Acquire) {
                        return Err(ChildError::Cancelled);
                    }
                    x = layer.forward(&x, &mask).map_err(error)?;
                    debug_assert_eq!(x.dtype(), DTYPE);
                }
                x = self.final_norm.forward(&x).map_err(error)?;
                x = self.speaker_projection.forward(&x).map_err(error)?;
                x = self
                    .upsample
                    .forward(&x)
                    .map_err(error)?
                    .reshape(&[1, padded as i32, 192])
                    .map_err(error)?;
                x = self
                    .speaker_hidden
                    .forward(&mlx_rs::nn::relu(&x).map_err(error)?)
                    .map_err(error)?;
                x = self
                    .speaker_output
                    .forward(&mlx_rs::nn::relu(&x).map_err(error)?)
                    .map_err(error)?;
                let probs = mlx_rs::ops::sigmoid(x)
                    .map_err(error)?
                    .index((0, ..valid as i32, ..))
                    .as_dtype(Dtype::Float32)
                    .map_err(error)?;
                if cancel.load(Ordering::Acquire) {
                    return Err(ChildError::Cancelled);
                }
                probs.eval().map_err(error)?;
                Ok(probs)
            })?;
        if cancel.load(Ordering::Acquire) {
            return Err(ChildError::Cancelled);
        }
        let values = probs.as_slice::<f32>();
        if values
            .iter()
            .any(|p| !p.is_finite() || !(0.0..=1.0).contains(p))
        {
            return Err(error("invalid Nemotron probabilities"));
        }
        Ok(values.as_chunks::<8>().0.to_vec())
    }

    fn features(&self, pcm: &[f32], count: usize) -> Result<Array, mlx_rs::error::Exception> {
        let valid = pcm.len() / HOP;
        let mut windows = vec![0.0f32; count * FFT];
        for (frame, window) in windows.as_chunks_mut::<FFT>().0.iter_mut().enumerate() {
            for (offset, sample) in window.iter_mut().enumerate() {
                let position = (frame * HOP + offset) as isize - 256;
                if let Some(current) = pcm.get(position as usize) {
                    let previous = pcm.get((position - 1) as usize).copied().unwrap_or(0.0);
                    *sample = *current - 0.97 * previous;
                }
            }
        }
        mlx_rs::with_stream(&mlx_rs::Stream::gpu(), || {
            let mut window = [0.0; FFT];
            window[56..456].copy_from_slice(self.window.as_slice::<f32>());
            let framed = Array::from_slice(&windows, &[count as i32, FFT as i32]);
            let window = Array::from_slice(&window, &[FFT as i32]);
            let spectrum = mlx_rs::fft::rfft(framed.multiply(window)?, None, None)?;
            let power = spectrum.abs()?.square()?;
            let features = power
                .matmul(self.filterbank.index(0).t())?
                .add(Array::from(2.0f32.powi(-24)))?
                .log()?;
            let valid = Array::from_slice(
                &(0..count).map(|i| i < valid).collect::<Vec<_>>(),
                &[count as i32, 1],
            );
            mlx_rs::ops::select(valid, features, Array::from(0.0f32))
        })
    }
}

// mlx-rs 0.32's GELU promotes BF16 through float32 scalar constants.
fn gelu(x: &Array) -> Result<Array, mlx_rs::error::Exception> {
    let mut compiled = mlx_rs::transforms::compile::compile(
        |x: &Array| {
            let sqrt_two = Array::from(2f32.sqrt()).as_dtype(x.dtype())?;
            let one = Array::from(1.0f32).as_dtype(x.dtype())?;
            let two = Array::from(2.0f32).as_dtype(x.dtype())?;
            x.multiply(one.add(mlx_rs::ops::erf(x.divide(sqrt_two)?)?)?)?
                .divide(two)
        },
        true,
    );
    compiled(x)
}
