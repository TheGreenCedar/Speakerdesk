//! Pinned dequantized Wav2Vec2 graph. Calls have no cache or padded audio.
use crate::weights::{error, linear, norm, take, verify};
use mlx_rs::{
    Array, Dtype,
    error::Exception,
    module::{Module, Param},
    nn::{Conv1d, LayerNorm, Linear},
    ops::indexing::IndexOp,
};
use speakerdesk_core::{error::ChildError, ipc::CtcScoresOut};
use std::{
    collections::HashMap,
    path::Path,
    sync::atomic::{AtomicBool, Ordering},
};

type Weights = HashMap<String, Array>;
const F32: Dtype = Dtype::Float32;
pub const WEIGHTS_SHA256: &str = "3dd733ec85933e07abfb6e363fcc1a37e693cbf9dda0e088db1f7c700252aa3d";
pub const VOCABULARY_SIZE: i32 = 9812;
pub const MAX_SAMPLES: usize = 392000;

struct Block {
    norm: LayerNorm,
    q: Linear,
    k: Linear,
    v: Linear,
    out: Linear,
    final_norm: LayerNorm,
    up: Linear,
    down: Linear,
}
impl Block {
    fn load(w: &mut Weights, index: usize) -> Result<Self, ChildError> {
        let n = format!("wav2vec2.encoder.layers.{index}");
        Ok(Self {
            norm: norm(w, &format!("{n}.layer_norm"), 1024, F32)?,
            q: linear(w, &format!("{n}.attention.q_proj"), 1024, 1024, true, F32)?,
            k: linear(w, &format!("{n}.attention.k_proj"), 1024, 1024, true, F32)?,
            v: linear(w, &format!("{n}.attention.v_proj"), 1024, 1024, true, F32)?,
            out: linear(w, &format!("{n}.attention.out_proj"), 1024, 1024, true, F32)?,
            final_norm: norm(w, &format!("{n}.final_layer_norm"), 1024, F32)?,
            up: linear(
                w,
                &format!("{n}.feed_forward.intermediate_dense"),
                1024,
                4096,
                true,
                F32,
            )?,
            down: linear(
                w,
                &format!("{n}.feed_forward.output_dense"),
                4096,
                1024,
                true,
                F32,
            )?,
        })
    }
    fn forward(&mut self, input: &Array) -> Result<Array, Exception> {
        let x = self.norm.forward(input)?;
        let heads = |x: Array| x.reshape(&[1, -1, 16, 64])?.transpose_axes(&[0, 2, 1, 3]);
        let q = heads(self.q.forward(&x)?.multiply(Array::from(0.125f32))?)?;
        let k = heads(self.k.forward(&x)?)?;
        let v = heads(self.v.forward(&x)?)?;
        let attention = mlx_rs::fast::scaled_dot_product_attention(&q, &k, &v, 1.0, None, None)?
            .transpose_axes(&[0, 2, 1, 3])?
            .reshape(&[1, -1, 1024])?;
        let x = input.add(self.out.forward(&attention)?)?;
        let y = self.final_norm.forward(&x)?;
        let y = mlx_rs::nn::gelu(self.up.forward(&y)?)?;
        x.add(self.down.forward(&y)?)
    }
}

pub struct Network {
    features: Vec<(Conv1d, LayerNorm)>,
    feature_norm: LayerNorm,
    projection: Linear,
    position: Conv1d,
    blocks: Vec<Block>,
    norm: LayerNorm,
    output: Linear,
}
impl Network {
    pub fn load(directory: &Path) -> Result<Self, ChildError> {
        verify(
            directory,
            &[
                ("model.safetensors", WEIGHTS_SHA256),
                ("tokens.txt", super::TOKENS_SHA256),
            ],
        )?;
        let mut w = Array::load_safetensors(directory.join("model.safetensors")).map_err(error)?;
        // Evaluate once so lazy weight I/O does not recur during the first crop.
        mlx_rs::transforms::eval(w.values()).map_err(error)?;
        let mut features = Vec::new();
        for (i, (kernel, stride)) in [10, 3, 3, 3, 3, 2, 2]
            .into_iter()
            .zip([5, 2, 2, 2, 2, 2, 2])
            .enumerate()
        {
            let name = format!("wav2vec2.feature_extractor.conv_layers.{i}");
            features.push((
                conv(
                    &mut w,
                    &format!("{name}.conv"),
                    [512, kernel, if i == 0 { 1 } else { 512 }],
                    stride,
                    0,
                    1,
                )?,
                norm(&mut w, &format!("{name}.layer_norm"), 512, F32)?,
            ));
        }
        let network = Self {
            features,
            feature_norm: norm(&mut w, "wav2vec2.feature_projection.layer_norm", 512, F32)?,
            projection: linear(
                &mut w,
                "wav2vec2.feature_projection.projection",
                512,
                1024,
                true,
                F32,
            )?,
            position: conv(
                &mut w,
                "wav2vec2.encoder.pos_conv_embed.conv",
                [1024, 128, 64],
                1,
                64,
                16,
            )?,
            blocks: (0..24)
                .map(|i| Block::load(&mut w, i))
                .collect::<Result<_, _>>()?,
            norm: norm(&mut w, "wav2vec2.encoder.layer_norm", 1024, F32)?,
            output: linear(&mut w, "lm_head", 1024, VOCABULARY_SIZE, true, F32)?,
        };
        if !w.is_empty() {
            return Err(error("unexpected CTC weight inventory"));
        }
        Ok(network)
    }

    pub fn scores(
        &mut self,
        pcm: &[f32],
        token_ids: &[i32],
        cancel: &AtomicBool,
    ) -> Result<CtcScoresOut, ChildError> {
        if pcm.is_empty() || pcm.len() > MAX_SAMPLES || pcm.iter().any(|v| !v.is_finite()) {
            return Err(error(
                "CTC requires finite 16 kHz audio of at most 24.5 seconds",
            ));
        }
        if token_ids.is_empty()
            || token_ids.len() > VOCABULARY_SIZE as usize
            || token_ids
                .iter()
                .any(|id| !(0..VOCABULARY_SIZE).contains(id))
        {
            return Err(error("invalid CTC token selection"));
        }
        cancelled(cancel)?;
        let valid_frames = valid_frames(pcm.len());
        let gathered = if valid_frames == 0 {
            Vec::new()
        } else {
            mlx_rs::with_stream(
                &mlx_rs::Stream::gpu(),
                || -> Result<Vec<f32>, ChildError> {
                    let audio = Array::from_slice(pcm, &[pcm.len() as i32]);
                    let mean = audio.mean(None).map_err(error)?;
                    let scale = audio
                        .var(None, None)
                        .map_err(error)?
                        .add(Array::from(1e-5f32))
                        .map_err(error)?
                        .sqrt()
                        .map_err(error)?;
                    let mut x = audio
                        .subtract(mean)
                        .map_err(error)?
                        .divide(scale)
                        .map_err(error)?
                        .reshape(&[1, -1, 1])
                        .map_err(error)?;
                    for (conv, norm) in &mut self.features {
                        x = mlx_rs::nn::gelu(
                            norm.forward(&conv.forward(&x).map_err(error)?)
                                .map_err(error)?,
                        )
                        .map_err(error)?;
                    }
                    if x.dim(1) != valid_frames as i32 {
                        return Err(error("CTC valid frame count differs"));
                    }
                    x = self
                        .projection
                        .forward(&self.feature_norm.forward(&x).map_err(error)?)
                        .map_err(error)?;
                    let position = self.position.forward(&x).map_err(error)?.index((
                        ..,
                        ..valid_frames as i32,
                        ..,
                    ));
                    x = x
                        .add(mlx_rs::nn::gelu(position).map_err(error)?)
                        .map_err(error)?;
                    for block in &mut self.blocks {
                        cancelled(cancel)?;
                        x = block.forward(&x).map_err(error)?;
                        x.eval().map_err(error)?;
                    }
                    let logits = self
                        .output
                        .forward(&self.norm.forward(&x).map_err(error)?)
                        .map_err(error)?
                        .reshape(&[valid_frames as i32, VOCABULARY_SIZE])
                        .map_err(error)?;
                    // Normalize across the complete vocabulary before gathering requested columns.
                    let scores = logits
                        .subtract(logits.logsumexp_axis(-1, true).map_err(error)?)
                        .map_err(error)?;
                    let selected = scores
                        .take_axis(Array::from_slice(token_ids, &[token_ids.len() as i32]), -1)
                        .map_err(error)?
                        .flatten(None, None)
                        .map_err(error)?;
                    selected.eval().map_err(error)?;
                    Ok(selected.as_slice::<f32>().to_vec())
                },
            )?
        };
        cancelled(cancel)?;
        if gathered.iter().any(|v| !v.is_finite()) {
            return Err(error("nonfinite CTC scores"));
        }
        Ok(CtcScoresOut {
            valid_frames: valid_frames as u32,
            audio_num_samples: pcm.len() as u32,
            frame_hop_samples: 320,
            frame_origin_samples: 0,
            token_ids: token_ids.to_vec(),
            model: speakerdesk_core::timing::MODEL.into(),
            vocabulary_sha256: super::TOKENS_SHA256.into(),
            gathered,
        })
    }
}

fn cancelled(cancel: &AtomicBool) -> Result<(), ChildError> {
    if cancel.load(Ordering::Acquire) {
        Err(ChildError::Cancelled)
    } else {
        Ok(())
    }
}
fn conv(
    w: &mut Weights,
    name: &str,
    shape: [i32; 3],
    stride: i32,
    padding: i32,
    groups: i32,
) -> Result<Conv1d, ChildError> {
    Ok(Conv1d {
        weight: Param::new(take(w, &format!("{name}.weight"), &shape, F32)?),
        bias: Param::new(Some(take(w, &format!("{name}.bias"), &[shape[0]], F32)?)),
        stride,
        padding,
        dilation: 1,
        groups,
    })
}
pub fn valid_frames(samples: usize) -> usize {
    if samples < 400 {
        0
    } else {
        (samples - 400) / 320 + 1
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn receptive_field_and_partial_tail() {
        for (samples, frames) in [
            (0, 0),
            (399, 0),
            (400, 1),
            (719, 1),
            (720, 2),
            (721, 2),
            (392000, 1224),
        ] {
            assert_eq!(super::valid_frames(samples), frames);
        }
    }
}
