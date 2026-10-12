//! The pinned v6 16 kHz graph; immutable weights are shared by independent recurrent states.
use std::collections::HashMap;
use std::path::Path;

use mlx_rs::module::{Module, Param};
use mlx_rs::nn::{Conv1d, Lstm};
use mlx_rs::ops::indexing::IndexOp;
use mlx_rs::{Array, Dtype};
use speakerdesk_core::error::ChildError;

use super::{CONTEXT_SAMPLES, FRAME_SAMPLES, MODEL_SAMPLES};

pub struct Network {
    stft: Conv1d,
    encoder: [Conv1d; 4],
    lstm: Lstm,
    output: Conv1d,
}

#[derive(Default)]
pub struct State {
    context: Vec<f32>,
    hidden: Option<Array>,
    cell: Option<Array>,
}

impl Network {
    pub fn load(directory: &Path) -> Result<Self, ChildError> {
        for (name, digest) in [
            (
                "config.json",
                "9fe1befb9692a0d4135adadc33f8075ef6d350bd2391b88d750f2c233f97fa0b",
            ),
            (
                "model.safetensors",
                "65b6c5f0293cbc44d109e58bef78b474d9c65dedbee814cf0b90ef5f0d9150ff",
            ),
        ] {
            let bytes = std::fs::read(directory.join(name)).map_err(model_error)?;
            if speakerdesk_core::memo::sha256_hex(&bytes) != digest {
                return Err(model_error(format!(
                    "Silero {name} does not match the pinned artifact"
                )));
            }
        }
        let mut weights =
            Array::load_safetensors(directory.join("model.safetensors")).map_err(model_error)?;
        let network = Self {
            stft: conv(&mut weights, "stft_conv", [258, 256, 1], 128, 0, false)?,
            encoder: [
                conv(&mut weights, "conv1", [128, 3, 129], 1, 1, true)?,
                conv(&mut weights, "conv2", [64, 3, 128], 2, 1, true)?,
                conv(&mut weights, "conv3", [64, 3, 64], 2, 1, true)?,
                conv(&mut weights, "conv4", [128, 3, 64], 1, 1, true)?,
            ],
            lstm: Lstm {
                wx: Param::new(take(&mut weights, "lstm.Wx", &[512, 128])?),
                wh: Param::new(take(&mut weights, "lstm.Wh", &[512, 128])?),
                bias: Param::new(Some(take(&mut weights, "lstm.bias", &[512])?)),
            },
            output: conv(&mut weights, "final_conv", [1, 1, 128], 1, 0, true)?,
        };
        if !weights.is_empty() {
            return Err(model_error("unexpected Silero weight inventory"));
        }
        Ok(network)
    }

    pub fn step(
        &mut self,
        frame: &[f32; FRAME_SAMPLES],
        state: &mut State,
    ) -> Result<f32, ChildError> {
        let mut input = [0.0; MODEL_SAMPLES + CONTEXT_SAMPLES];
        if !state.context.is_empty() {
            input[..CONTEXT_SAMPLES].copy_from_slice(&state.context);
        }
        input[CONTEXT_SAMPLES..MODEL_SAMPLES].copy_from_slice(frame);
        // Upstream STFT uses right reflection, excluding the final sample itself.
        for index in 0..CONTEXT_SAMPLES {
            input[MODEL_SAMPLES + index] = input[MODEL_SAMPLES - 2 - index];
        }
        let (probability, hidden, cell) = mlx_rs::with_stream(
            &mlx_rs::Stream::gpu(),
            || -> Result<_, mlx_rs::error::Exception> {
                let audio = Array::from_slice(&input, &[1, input.len() as i32, 1]);
                let spectrum = self.stft.forward(&audio)?;
                let real = spectrum.index((.., .., ..129));
                let imaginary = spectrum.index((.., .., 129..));
                let mut x = real.square()?.add(imaginary.square()?)?.sqrt()?;
                for layer in &mut self.encoder {
                    x = mlx_rs::nn::relu(layer.forward(&x)?)?;
                }
                // This graph has exactly one recurrent timestep per 512 new samples.
                if x.shape() != [1, 1, 128] {
                    return Err(mlx_rs::error::Exception::custom(
                        "unexpected Silero encoder shape",
                    ));
                }
                let (hidden, cell) =
                    self.lstm
                        .step(&x, state.hidden.as_ref(), state.cell.as_ref())?;
                let logits = self.output.forward(&mlx_rs::nn::relu(&hidden)?)?;
                let probability = mlx_rs::ops::sigmoid(logits)?.mean(None)?;
                mlx_rs::transforms::eval([&probability, &hidden, &cell])?;
                Ok((
                    probability.item_exact::<f32>(),
                    hidden.index((.., -1, ..)),
                    cell.index((.., -1, ..)),
                ))
            },
        )
        .map_err(model_error)?;
        if !probability.is_finite() || !(0.0..=1.0).contains(&probability) {
            return Err(model_error("invalid Silero speech probability"));
        }
        state.hidden = Some(hidden);
        state.cell = Some(cell);
        state.context.clear();
        state
            .context
            .extend_from_slice(&frame[FRAME_SAMPLES - CONTEXT_SAMPLES..]);
        Ok(probability)
    }
}

fn conv(
    weights: &mut HashMap<String, Array>,
    name: &str,
    shape: [i32; 3],
    stride: i32,
    padding: i32,
    bias: bool,
) -> Result<Conv1d, ChildError> {
    Ok(Conv1d {
        weight: Param::new(take(weights, &format!("{name}.weight"), &shape)?),
        bias: Param::new(if bias {
            Some(take(weights, &format!("{name}.bias"), &[shape[0]])?)
        } else {
            None
        }),
        stride,
        padding,
        dilation: 1,
        groups: 1,
    })
}

fn take(
    weights: &mut HashMap<String, Array>,
    name: &str,
    shape: &[i32],
) -> Result<Array, ChildError> {
    let weight = weights
        .remove(&format!("vad_16k.{name}"))
        .ok_or_else(|| model_error(format!("missing Silero weight {name}")))?;
    if weight.shape() != shape || weight.dtype() != Dtype::Float32 {
        return Err(model_error(format!("invalid Silero weight layout {name}")));
    }
    weight.eval().map_err(model_error)?;
    Ok(weight)
}

fn model_error(error: impl std::fmt::Display) -> ChildError {
    ChildError::failed(format!("Silero: {error}"), false)
}
