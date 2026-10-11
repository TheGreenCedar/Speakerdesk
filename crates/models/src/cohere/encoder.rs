use super::layers::{BF16, Weights, conv1d, conv2d, heads, quantized, scalar};
use crate::weights::{error, norm, take};
use mlx_rs::{
    Array,
    error::Exception,
    module::{Module, Param},
    nn::{BatchNorm, Conv1d, Conv2d, LayerNorm, QuantizedLinear},
    ops::{self, indexing::IndexOp},
};
use speakerdesk_core::error::ChildError;
use std::sync::atomic::{AtomicBool, Ordering};

const DIM: i32 = 1280;

struct FeedForward {
    expand: QuantizedLinear,
    contract: QuantizedLinear,
}
impl FeedForward {
    fn load(w: &mut Weights, name: &str) -> Result<Self, ChildError> {
        Ok(Self {
            expand: quantized(w, &format!("{name}.linear1"), DIM, 5120, true)?,
            contract: quantized(w, &format!("{name}.linear2"), 5120, DIM, true)?,
        })
    }
    fn forward(&mut self, x: &Array) -> Result<Array, Exception> {
        self.contract
            .forward(&mlx_rs::nn::silu(&self.expand.forward(x)?)?)
    }
}

struct Attention {
    q: QuantizedLinear,
    k: QuantizedLinear,
    v: QuantizedLinear,
    out: QuantizedLinear,
    relative: QuantizedLinear,
    bias_u: Array,
    bias_v: Array,
}
impl Attention {
    fn load(w: &mut Weights, name: &str) -> Result<Self, ChildError> {
        Ok(Self {
            q: quantized(w, &format!("{name}.q_proj"), DIM, DIM, true)?,
            k: quantized(w, &format!("{name}.k_proj"), DIM, DIM, true)?,
            v: quantized(w, &format!("{name}.v_proj"), DIM, DIM, true)?,
            out: quantized(w, &format!("{name}.o_proj"), DIM, DIM, true)?,
            relative: quantized(w, &format!("{name}.relative_k_proj"), DIM, DIM, false)?,
            bias_u: take(w, &format!("{name}.bias_u"), &[8, 160], BF16)?,
            bias_v: take(w, &format!("{name}.bias_v"), &[8, 160], BF16)?,
        })
    }
    fn forward(&mut self, x: &Array, positions: &Array, mask: &Array) -> Result<Array, Exception> {
        let length = x.dim(1);
        let q = heads(self.q.forward(x)?, 8)?;
        let k = heads(self.k.forward(x)?, 8)?;
        let v = heads(self.v.forward(x)?, 8)?;
        let relative = heads(self.relative.forward(positions)?, 8)?;
        let bd = q
            .add(self.bias_v.reshape(&[1, 8, 1, 160])?)?
            .matmul(relative.transpose_axes(&[0, 1, 3, 2])?)?;
        let zero = ops::zeros_dtype(&[1, 8, length, 1], bd.dtype())?;
        let shifted = ops::concatenate(&[zero, bd], -1)?
            .reshape(&[1, 8, 2 * length, length])?
            .index((.., .., 1.., ..))
            .reshape(&[1, 8, length, 2 * length - 1])?
            .index((.., .., .., ..length));
        let scale = scalar(1.0 / 160f32.sqrt(), &shifted)?;
        let shifted = shifted.multiply(&scale)?;
        let shifted = ops::select(mask, &shifted, scalar(f32::MIN, &shifted)?)?;
        let scores = q
            .add(self.bias_u.reshape(&[1, 8, 1, 160])?)?
            .matmul(k.transpose_axes(&[0, 1, 3, 2])?)?
            .multiply(scale)?
            .add(shifted)?;
        let weights = ops::softmax_axis(scores.as_dtype(mlx_rs::Dtype::Float32)?, -1, None)?
            .as_dtype(x.dtype())?;
        self.out.forward(
            &weights
                .matmul(v)?
                .transpose_axes(&[0, 2, 1, 3])?
                .reshape(&[1, length, DIM])?,
        )
    }
}

struct Convolution {
    expand: Conv1d,
    depthwise: Conv1d,
    norm: BatchNorm,
    contract: Conv1d,
}
impl Convolution {
    fn load(w: &mut Weights, name: &str) -> Result<Self, ChildError> {
        Ok(Self {
            expand: conv1d(w, &format!("{name}.pointwise_conv1"), DIM, 2 * DIM, 1)?,
            depthwise: Conv1d {
                weight: Param::new(take(w, &format!("{name}.dw_weight"), &[DIM, 9, 1], BF16)?),
                bias: Param::new(Some(take(w, &format!("{name}.dw_bias"), &[DIM], BF16)?)),
                stride: 1,
                padding: 4,
                dilation: 1,
                groups: DIM,
            },
            norm: BatchNorm {
                feature_count: DIM,
                eps: Array::from(1e-5f32).as_dtype(BF16).map_err(error)?,
                momentum: Array::from(0.1f32),
                training: false,
                weight: Param::new(Some(take(w, &format!("{name}.norm.weight"), &[DIM], BF16)?)),
                bias: Param::new(Some(take(w, &format!("{name}.norm.bias"), &[DIM], BF16)?)),
                running_mean: Param::new(Some(take(
                    w,
                    &format!("{name}.norm.running_mean"),
                    &[DIM],
                    BF16,
                )?)),
                running_var: Param::new(Some(take(
                    w,
                    &format!("{name}.norm.running_var"),
                    &[DIM],
                    BF16,
                )?)),
            },
            contract: conv1d(w, &format!("{name}.pointwise_conv2"), DIM, DIM, 1)?,
        })
    }
    fn forward(&mut self, x: &Array, valid: &Array) -> Result<Array, Exception> {
        let expanded = self.expand.forward(x)?;
        let gated = expanded
            .index((.., .., ..DIM))
            .multiply(ops::sigmoid(expanded.index((.., .., DIM..)))?)?
            .multiply(valid.as_dtype(x.dtype())?)?;
        let convolved = self.depthwise.forward(&gated)?;
        self.contract
            .forward(&mlx_rs::nn::silu(&self.norm.forward(&convolved)?)?)
    }
}

struct Block {
    ff1: FeedForward,
    attention: Attention,
    conv: Convolution,
    ff2: FeedForward,
    norms: [LayerNorm; 5],
}
impl Block {
    fn load(w: &mut Weights, prefix: &str) -> Result<Self, ChildError> {
        Ok(Self {
            ff1: FeedForward::load(w, &format!("{prefix}.feed_forward1"))?,
            attention: Attention::load(w, &format!("{prefix}.self_attn"))?,
            conv: Convolution::load(w, &format!("{prefix}.conv"))?,
            ff2: FeedForward::load(w, &format!("{prefix}.feed_forward2"))?,
            norms: [
                norm(w, &format!("{prefix}.norm_feed_forward1"), DIM, BF16)?,
                norm(w, &format!("{prefix}.norm_self_att"), DIM, BF16)?,
                norm(w, &format!("{prefix}.norm_conv"), DIM, BF16)?,
                norm(w, &format!("{prefix}.norm_feed_forward2"), DIM, BF16)?,
                norm(w, &format!("{prefix}.norm_out"), DIM, BF16)?,
            ],
        })
    }
    fn forward(
        &mut self,
        input: &Array,
        positions: &Array,
        mask: &Array,
        valid: &Array,
    ) -> Result<Array, Exception> {
        let ff = self.ff1.forward(&self.norms[0].forward(input)?)?;
        let x = input.add(ff.multiply(scalar(0.5, &ff)?)?)?;
        let x = x.add(
            self.attention
                .forward(&self.norms[1].forward(&x)?, positions, mask)?,
        )?;
        let x = x.add(self.conv.forward(&self.norms[2].forward(&x)?, valid)?)?;
        let ff = self.ff2.forward(&self.norms[3].forward(&x)?)?;
        self.norms[4].forward(&x.add(ff.multiply(scalar(0.5, &ff)?)?)?)
    }
}

pub struct Encoder {
    conv: Conv2d,
    depthwise: [Conv2d; 2],
    pointwise: [Conv2d; 2],
    projection: QuantizedLinear,
    blocks: Vec<Block>,
}
impl Encoder {
    pub fn load(w: &mut Weights) -> Result<Self, ChildError> {
        let prefix = "encoder.subsampling";
        let depthwise = |w: &mut Weights, index| -> Result<Conv2d, ChildError> {
            Ok(Conv2d {
                weight: Param::new(take(
                    w,
                    &format!("{prefix}.dw_weight_{index}"),
                    &[256, 3, 3, 1],
                    BF16,
                )?),
                bias: Param::new(Some(take(
                    w,
                    &format!("{prefix}.dw_bias_{index}"),
                    &[256],
                    BF16,
                )?)),
                stride: (2, 2),
                padding: (1, 1),
                dilation: (1, 1),
                groups: 256,
            })
        };
        Ok(Self {
            conv: conv2d(w, &format!("{prefix}.conv0"), 1, 3, 2)?,
            depthwise: [depthwise(w, 0)?, depthwise(w, 1)?],
            pointwise: [
                conv2d(w, &format!("{prefix}.pw_convs.0"), 256, 1, 1)?,
                conv2d(w, &format!("{prefix}.pw_convs.1"), 256, 1, 1)?,
            ],
            projection: quantized(w, &format!("{prefix}.linear"), 4096, DIM, true)?,
            blocks: (0..48)
                .map(|i| Block::load(w, &format!("encoder.layers.{i}")))
                .collect::<Result<_, _>>()?,
        })
    }
    pub fn forward(
        &mut self,
        features: &Array,
        mut valid: usize,
        cancel: &AtomicBool,
    ) -> Result<(Array, usize), ChildError> {
        let mut subsample = || -> Result<Array, Exception> {
            let mut x = mlx_rs::nn::relu(&self.conv.forward(&features.reshape(&[
                1,
                features.dim(1),
                128,
                1,
            ])?)?)?;
            valid = valid.div_ceil(2);
            x = time_mask(x, valid)?;
            for (depthwise, pointwise) in self.depthwise.iter_mut().zip(&mut self.pointwise) {
                x = mlx_rs::nn::relu(&pointwise.forward(&depthwise.forward(&x)?)?)?;
                valid = valid.div_ceil(2);
                x = time_mask(x, valid)?;
            }
            let length = x.dim(1);
            self.projection.forward(
                &x.transpose_axes(&[0, 1, 3, 2])?
                    .reshape(&[1, length, 4096])?,
            )
        };
        let mut x = subsample().map_err(error)?;
        let length = x.dim(1);
        let positions = positions(length, x.dtype()).map_err(error)?;
        let row = Array::from_slice(
            &(0..length as usize).map(|i| i < valid).collect::<Vec<_>>(),
            &[1, length, 1],
        );
        let mask = row
            .reshape(&[1, 1, length, 1])
            .map_err(error)?
            .logical_and(row.reshape(&[1, 1, 1, length]).map_err(error)?)
            .map_err(error)?;
        for block in &mut self.blocks {
            if cancel.load(Ordering::Acquire) {
                return Err(ChildError::Cancelled);
            }
            x = block.forward(&x, &positions, &mask, &row).map_err(error)?;
        }
        if cancel.load(Ordering::Acquire) {
            return Err(ChildError::Cancelled);
        }
        x.eval().map_err(error)?;
        Ok((x, valid))
    }
}

fn time_mask(x: Array, valid: usize) -> Result<Array, Exception> {
    let mask = Array::from_slice(
        &(0..x.dim(1) as usize)
            .map(|i| i < valid)
            .collect::<Vec<_>>(),
        &[1, x.dim(1), 1, 1],
    );
    x.multiply(mask.as_dtype(x.dtype())?)
}

fn positions(length: i32, dtype: mlx_rs::Dtype) -> Result<Array, Exception> {
    let frequencies = Array::from_slice(
        &(0..DIM / 2)
            .map(|i| i as f32 * 2.0 / DIM as f32)
            .collect::<Vec<_>>(),
        &[1, 1, DIM / 2],
    );
    let inverse = Array::from(1.0f32).divide(Array::from(10000.0f32).power(frequencies)?)?;
    let offsets = Array::from_slice(
        &(-length + 1..length)
            .rev()
            .map(|i| i as f32)
            .collect::<Vec<_>>(),
        &[1, 2 * length - 1, 1],
    );
    let angles = inverse.multiply(offsets)?;
    ops::stack(&[angles.sin()?, angles.cos()?], -1)?
        .reshape(&[1, 2 * length - 1, DIM])?
        .as_dtype(dtype)
}
