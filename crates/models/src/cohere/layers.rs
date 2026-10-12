use crate::weights::take;
use mlx_rs::{
    Array, Dtype,
    error::Exception,
    module::Param,
    nn::{Conv1d, Conv2d, Embedding, Linear, QuantizedEmbedding, QuantizedLinear},
};
use speakerdesk_core::error::ChildError;
use std::collections::HashMap;

pub type Weights = HashMap<String, Array>;
pub const BF16: Dtype = Dtype::Bfloat16;

pub fn quantized(
    w: &mut Weights,
    name: &str,
    input: i32,
    output: i32,
    bias: bool,
) -> Result<QuantizedLinear, ChildError> {
    Ok(QuantizedLinear {
        group_size: 64,
        bits: 4,
        scales: Param::new(take(
            w,
            &format!("{name}.scales"),
            &[output, input / 64],
            BF16,
        )?),
        biases: Param::new(take(
            w,
            &format!("{name}.biases"),
            &[output, input / 64],
            BF16,
        )?),
        inner: Linear {
            weight: Param::new(take(
                w,
                &format!("{name}.weight"),
                &[output, input / 8],
                Dtype::Uint32,
            )?),
            bias: Param::new(if bias {
                Some(take(w, &format!("{name}.bias"), &[output], BF16)?)
            } else {
                None
            }),
        },
    })
}

pub fn embedding(
    w: &mut Weights,
    name: &str,
    count: i32,
) -> Result<QuantizedEmbedding, ChildError> {
    let linear = quantized(w, name, 1024, count, false)?;
    Ok(QuantizedEmbedding {
        group_size: 64,
        bits: 4,
        scales: linear.scales,
        biases: linear.biases,
        inner: Embedding {
            weight: linear.inner.weight,
        },
    })
}

pub fn conv1d(
    w: &mut Weights,
    name: &str,
    input: i32,
    output: i32,
    kernel: i32,
) -> Result<Conv1d, ChildError> {
    Ok(Conv1d {
        weight: Param::new(take(
            w,
            &format!("{name}.weight"),
            &[output, kernel, input],
            BF16,
        )?),
        bias: Param::new(Some(take(w, &format!("{name}.bias"), &[output], BF16)?)),
        stride: 1,
        padding: 0,
        dilation: 1,
        groups: 1,
    })
}

pub fn conv2d(
    w: &mut Weights,
    name: &str,
    input: i32,
    kernel: i32,
    stride: i32,
) -> Result<Conv2d, ChildError> {
    Ok(Conv2d {
        weight: Param::new(take(
            w,
            &format!("{name}.weight"),
            &[256, kernel, kernel, input],
            BF16,
        )?),
        bias: Param::new(Some(take(w, &format!("{name}.bias"), &[256], BF16)?)),
        stride: (stride, stride),
        padding: (kernel / 2, kernel / 2),
        dilation: (1, 1),
        groups: 1,
    })
}

pub fn scalar(value: f32, like: &Array) -> Result<Array, Exception> {
    Array::from(value).as_dtype(like.dtype())
}

pub fn heads(x: Array, head_count: i32) -> Result<Array, Exception> {
    let length = x.dim(1);
    x.reshape(&[1, length, head_count, -1])?
        .transpose_axes(&[0, 2, 1, 3])
}
