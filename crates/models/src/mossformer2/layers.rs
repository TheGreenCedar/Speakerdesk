use crate::weights::{linear, norm, take};
use mlx_rs::{
    Array, Dtype,
    error::Exception,
    module::{Module, Param},
    nn::{Conv1d, LayerNorm, Linear},
    ops::{self, indexing::IndexOp},
};
use speakerdesk_core::error::ChildError;
use std::collections::HashMap;

type Weights = HashMap<String, Array>;
const F32: Dtype = Dtype::Float32;

#[allow(clippy::too_many_arguments)]
pub(super) fn conv(
    w: &mut Weights,
    name: &str,
    input: i32,
    output: i32,
    kernel: i32,
    stride: i32,
    padding: i32,
    dilation: i32,
    groups: i32,
    bias: bool,
) -> Result<Conv1d, ChildError> {
    Ok(Conv1d {
        weight: Param::new(
            take(
                w,
                &format!("{name}.weight"),
                &[output, input / groups, kernel],
                F32,
            )?
            .transpose_axes(&[0, 2, 1])
            .map_err(crate::weights::error)?,
        ),
        bias: Param::new(if bias {
            Some(take(w, &format!("{name}.bias"), &[output], F32)?)
        } else {
            None
        }),
        stride,
        padding,
        dilation,
        groups,
    })
}

pub(super) fn layer_norm(
    w: &mut Weights,
    name: &str,
    channels: i32,
    eps: f32,
) -> Result<LayerNorm, ChildError> {
    let mut norm = norm(w, name, channels, F32)?;
    norm.eps = eps;
    Ok(norm)
}

pub(super) struct AffineNorm {
    weight: Array,
    bias: Array,
}
impl AffineNorm {
    pub(super) fn load(w: &mut Weights, name: &str, channels: i32) -> Result<Self, ChildError> {
        Ok(Self {
            weight: take(w, &format!("{name}.weight"), &[channels], F32)?,
            bias: take(w, &format!("{name}.bias"), &[channels], F32)?,
        })
    }
    pub(super) fn forward(&self, x: &Array, axes: &[i32], eps: f32) -> Result<Array, Exception> {
        let centered = x.subtract(x.mean_axes(axes, true)?)?;
        let variance = centered.square()?.mean_axes(axes, true)?;
        centered
            .divide(variance.add(Array::from(eps))?.sqrt()?)?
            .multiply(&self.weight)?
            .add(&self.bias)
    }
}

pub(super) fn prelu(x: &Array, weight: &Array) -> Result<Array, Exception> {
    ops::maximum(x, Array::from(0f32))?.add(ops::minimum(x, Array::from(0f32))?.multiply(weight)?)
}

enum Normalization {
    Scale(Array),
    Layer(LayerNorm),
}
pub(super) struct FeedForward {
    norm: Normalization,
    linear: Linear,
    conv: Conv1d,
}
impl FeedForward {
    pub(super) fn load(
        w: &mut Weights,
        name: &str,
        input: i32,
        output: i32,
        scale: bool,
    ) -> Result<Self, ChildError> {
        Ok(Self {
            norm: if scale {
                Normalization::Scale(take(w, &format!("{name}.mdl.0.g"), &[1], F32)?)
            } else {
                Normalization::Layer(layer_norm(w, &format!("{name}.mdl.0"), input, 1e-5)?)
            },
            linear: linear(w, &format!("{name}.mdl.1"), input, output, true, F32)?,
            conv: conv(
                w,
                &format!("{name}.mdl.3.sequential.1.conv"),
                output,
                output,
                17,
                1,
                8,
                1,
                output,
                false,
            )?,
        })
    }
    pub(super) fn forward(&mut self, x: &Array) -> Result<Array, Exception> {
        let normalized = match &mut self.norm {
            Normalization::Layer(norm) => norm.forward(x)?,
            Normalization::Scale(g) => {
                let rms = x
                    .square()?
                    .sum_axis(-1, true)?
                    .sqrt()?
                    .multiply(Array::from((x.dim(-1) as f32).powf(-0.5)))?;
                x.divide(ops::maximum(rms, Array::from(1e-5f32))?)?
                    .multiply(g)?
            }
        };
        let projected = mlx_rs::nn::silu(self.linear.forward(&normalized)?)?;
        projected.add(self.conv.forward(&projected)?)
    }
}

pub(super) struct Flash {
    hidden: FeedForward,
    qk: FeedForward,
    output: FeedForward,
    gamma: Array,
    beta: Array,
    frequencies: Array,
}
impl Flash {
    pub(super) fn load(w: &mut Weights, name: &str) -> Result<Self, ChildError> {
        Ok(Self {
            hidden: FeedForward::load(w, &format!("{name}.to_hidden"), 512, 2048, true)?,
            qk: FeedForward::load(w, &format!("{name}.to_qk"), 512, 128, true)?,
            output: FeedForward::load(w, &format!("{name}.to_out"), 1024, 512, true)?,
            gamma: take(w, &format!("{name}.qk_offset_scale.gamma"), &[4, 128], F32)?,
            beta: take(w, &format!("{name}.qk_offset_scale.beta"), &[4, 128], F32)?,
            frequencies: take(w, &format!("{name}.rotary_pos_emb.freqs"), &[16], F32)?,
        })
    }

    pub(super) fn forward(&mut self, input: &Array) -> Result<Array, Exception> {
        const GROUP: i32 = 256;
        let n = input.dim(1);
        let shifted = ops::concatenate(
            &[
                ops::zeros::<f32>(&[1, 1, 256])?,
                input.index((.., ..n - 1, ..256)),
            ],
            1,
        )?;
        let x = ops::concatenate(&[shifted, input.index((.., .., 256..))], -1)?;
        let hidden = self.hidden.forward(&x)?;
        let v = hidden.index((.., .., ..1024));
        let u = hidden.index((.., .., 1024..));
        let qk = self.qk.forward(&x)?;
        let heads = qk.expand_dims(-2)?.multiply(&self.gamma)?.add(&self.beta)?;
        let padded = ((n + GROUP - 1) / GROUP) * GROUP;
        let rotate = |index| -> Result<Array, Exception> {
            pad_time(
                &rotary(&heads.index((.., .., index, ..)), &self.frequencies)?,
                padded,
            )?
            .reshape(&[1, -1, GROUP, 128])
        };
        let quad_q = rotate(0)?;
        let lin_q = rotate(1)?;
        let quad_k = rotate(2)?;
        let lin_k = rotate(3)?;
        let grouped_v = pad_time(&v, padded)?.reshape(&[1, -1, GROUP, 1024])?;
        let grouped_u = pad_time(&u, padded)?.reshape(&[1, -1, GROUP, 1024])?;
        let similarity = quad_q
            .matmul(quad_k.swap_axes(-1, -2)?)?
            .divide(Array::from(GROUP as f32))?;
        let attention = mlx_rs::nn::relu(similarity)?.square()?;
        let mask = Array::from_slice(
            &(0..padded).map(|i| i < n).collect::<Vec<_>>(),
            &[1, padded / GROUP, 1, GROUP],
        );
        let attention = ops::select(mask, attention, Array::from(0f32))?;
        // The noncausal linear branch divides by the original unpadded length.
        let linear_key = lin_k.reshape(&[1, padded, 128])?.swap_axes(1, 2)?;
        let attend = |values: &Array| -> Result<Array, Exception> {
            let kv = linear_key
                .matmul(values.reshape(&[1, padded, 1024])?)?
                .divide(Array::from(n as f32))?;
            let linear = lin_q.reshape(&[1, padded, 128])?.matmul(kv)?;
            attention
                .matmul(values)?
                .reshape(&[1, padded, 1024])?
                .add(linear)?
                .index((.., ..n, ..))
                .contiguous()
        };
        let att_v = attend(&grouped_v)?;
        let att_u = attend(&grouped_u)?;
        let gated = att_u
            .multiply(&v)?
            .multiply(ops::sigmoid(att_v.multiply(&u)?)?)?;
        input.add(self.output.forward(&gated)?)
    }
}

fn rotary(x: &Array, frequencies: &Array) -> Result<Array, Exception> {
    let n = x.dim(1);
    let positions = Array::from_slice(&(0..n).map(|i| i as f32).collect::<Vec<_>>(), &[n, 1]);
    let angles = positions.multiply(frequencies)?.reshape(&[1, n, 16, 1])?;
    let pairs = x.index((.., .., ..32)).reshape(&[1, n, 16, 2])?;
    let rotated = ops::stack(
        &[
            pairs.index((.., .., .., 1)).negative()?,
            pairs.index((.., .., .., 0)),
        ],
        -1,
    )?;
    let first = pairs
        .multiply(angles.cos()?)?
        .add(rotated.multiply(angles.sin()?)?)?
        .reshape(&[1, n, 32])?;
    ops::concatenate(&[first, x.index((.., .., 32..))], -1)
}

pub(super) fn pad_time(x: &Array, length: i32) -> Result<Array, Exception> {
    if length <= x.dim(1) {
        return Ok(x.clone());
    }
    ops::concatenate(
        &[
            x,
            &ops::zeros::<f32>(&[x.dim(0), length - x.dim(1), x.dim(2)])?,
        ],
        1,
    )
}

pub(super) struct Fsmn {
    input: Conv1d,
    input_prelu: Array,
    norm1: LayerNorm,
    norm2: LayerNorm,
    output: Conv1d,
    u: FeedForward,
    v: FeedForward,
    linear: Linear,
    project: Linear,
    dense: [Conv1d; 2],
    dense_norm: [AffineNorm; 2],
    dense_prelu: [Array; 2],
}
impl Fsmn {
    pub(super) fn load(w: &mut Weights, name: &str) -> Result<Self, ChildError> {
        let prefix = format!("{name}.gated_fsmn.fsmn");
        let mut dense = Vec::new();
        let mut dense_norm = Vec::new();
        let mut dense_prelu = Vec::new();
        for i in 0..2 {
            let n = i + 1;
            let weight = take(
                w,
                &format!("{prefix}.conv.conv{n}.weight"),
                &[256, n, 39, 1],
                F32,
            )?;
            dense.push(Conv1d {
                weight: Param::new(
                    weight
                        .reshape(&[256, n, 39])
                        .map_err(crate::weights::error)?
                        .transpose_axes(&[0, 2, 1])
                        .map_err(crate::weights::error)?,
                ),
                bias: Param::new(None),
                stride: 1,
                padding: 19 * (1 << i),
                dilation: 1 << i,
                groups: 256,
            });
            dense_norm.push(AffineNorm::load(w, &format!("{prefix}.conv.norm{n}"), 256)?);
            dense_prelu.push(take(
                w,
                &format!("{prefix}.conv.prelu{n}.weight"),
                &[256],
                F32,
            )?);
        }
        Ok(Self {
            input: conv(w, &format!("{name}.conv1.0"), 512, 256, 1, 1, 0, 1, 1, true)?,
            input_prelu: take(w, &format!("{name}.conv1.1.weight"), &[1], F32)?,
            norm1: layer_norm(w, &format!("{name}.norm1"), 256, 1e-5)?,
            norm2: layer_norm(w, &format!("{name}.norm2"), 256, 1e-5)?,
            output: conv(w, &format!("{name}.conv2"), 256, 512, 1, 1, 0, 1, 1, true)?,
            u: FeedForward::load(w, &format!("{name}.gated_fsmn.to_u"), 256, 256, false)?,
            v: FeedForward::load(w, &format!("{name}.gated_fsmn.to_v"), 256, 256, false)?,
            linear: linear(w, &format!("{prefix}.linear"), 256, 256, true, F32)?,
            project: linear(w, &format!("{prefix}.project"), 256, 256, false, F32)?,
            dense: dense.try_into().ok().unwrap(),
            dense_norm: dense_norm.try_into().ok().unwrap(),
            dense_prelu: dense_prelu.try_into().ok().unwrap(),
        })
    }
    pub(super) fn forward(&mut self, input: &Array) -> Result<Array, Exception> {
        let projected = prelu(&self.input.forward(input)?, &self.input_prelu)?;
        let x = self.norm1.forward(&projected)?;
        let u = self.u.forward(&x)?;
        let v = self.v.forward(&x)?;
        let projected = self
            .project
            .forward(&mlx_rs::nn::relu(self.linear.forward(&u)?)?)?;
        let first = prelu(
            &self.dense_norm[0].forward(&self.dense[0].forward(&projected)?, &[1], 1e-5)?,
            &self.dense_prelu[0],
        )?;
        // Preserve PyTorch's concatenated channel order before its grouped convolution.
        let dense_input = ops::concatenate(&[first, projected], -1)?;
        let second = prelu(
            &self.dense_norm[1].forward(&self.dense[1].forward(&dense_input)?, &[1], 1e-5)?,
            &self.dense_prelu[1],
        )?;
        let gated = u.add(second)?.multiply(v)?.add(x)?;
        input.add(self.output.forward(&self.norm2.forward(&gated)?)?)
    }
}
