use super::layers::{self, AffineNorm, Flash, Fsmn};
use crate::weights::{error, take, verify};
use mlx_rs::{
    Array, Dtype,
    module::{Module, Param},
    nn::{Conv1d, ConvTranspose1d, LayerNorm},
    ops::{self, indexing::IndexOp},
};
use speakerdesk_core::error::ChildError;
use std::{
    path::Path,
    sync::atomic::{AtomicBool, Ordering},
};

pub struct Network {
    encoder: Conv1d,
    decoder: ConvTranspose1d,
    input_norm: AffineNorm,
    input_projection: Conv1d,
    position_scale: Array,
    position_frequencies: Array,
    flash: Vec<Flash>,
    fsmn: Vec<Fsmn>,
    intra_norm: LayerNorm,
    output_norm: AffineNorm,
    prelu: Array,
    output_projection: Conv1d,
    output: Conv1d,
    output_gate: Conv1d,
    mask_projection: Conv1d,
}

impl Network {
    pub fn load(directory: &Path) -> Result<Self, ChildError> {
        let path = directory.join(super::DERIVED_FILE);
        verify(directory, &[(super::DERIVED_FILE, super::DERIVED_SHA256)])?;
        let mut w = Array::load_safetensors(path).map_err(error)?;
        let mut flash = Vec::with_capacity(24);
        let mut fsmn = Vec::with_capacity(24);
        let base = "mask_net.mdl.intra_mdl.mossformerM";
        for i in 0..24 {
            flash.push(Flash::load(&mut w, &format!("{base}.layers.{i}"))?);
            fsmn.push(Fsmn::load(&mut w, &format!("{base}.fsmn.{i}"))?);
        }
        let network = Self {
            encoder: layers::conv(&mut w, "enc.conv1d", 1, 512, 16, 8, 0, 1, 1, false)?,
            // PyTorch transposed convolution is [input, output, kernel].
            decoder: ConvTranspose1d {
                weight: Param::new(
                    take(&mut w, "dec.weight", &[512, 1, 16], Dtype::Float32)?
                        .transpose_axes(&[1, 2, 0])
                        .map_err(error)?,
                ),
                bias: Param::new(None),
                padding: 0,
                output_padding: 0,
                stride: 8,
            },
            input_norm: AffineNorm::load(&mut w, "mask_net.norm", 512)?,
            input_projection: layers::conv(
                &mut w,
                "mask_net.conv1d_encoder",
                512,
                512,
                1,
                1,
                0,
                1,
                1,
                false,
            )?,
            position_scale: take(&mut w, "mask_net.pos_enc.scale", &[1], Dtype::Float32)?,
            position_frequencies: take(
                &mut w,
                "mask_net.pos_enc.inv_freq",
                &[256],
                Dtype::Float32,
            )?,
            flash,
            fsmn,
            intra_norm: layers::layer_norm(&mut w, "mask_net.mdl.intra_mdl.norm", 512, 1e-6)?,
            output_norm: AffineNorm::load(&mut w, "mask_net.mdl.intra_norm", 512)?,
            prelu: take(&mut w, "mask_net.prelu.weight", &[1], Dtype::Float32)?,
            output_projection: layers::conv(
                &mut w,
                "mask_net.conv1d_out",
                512,
                1024,
                1,
                1,
                0,
                1,
                1,
                true,
            )?,
            output: layers::conv(&mut w, "mask_net.output.0", 512, 512, 1, 1, 0, 1, 1, true)?,
            output_gate: layers::conv(
                &mut w,
                "mask_net.output_gate.0",
                512,
                512,
                1,
                1,
                0,
                1,
                1,
                true,
            )?,
            mask_projection: layers::conv(
                &mut w,
                "mask_net.conv1_decoder",
                512,
                512,
                1,
                1,
                0,
                1,
                1,
                false,
            )?,
        };
        if !w.is_empty() {
            return Err(error("unmapped MossFormer2 weight"));
        }
        Ok(network)
    }

    pub fn separate(
        &mut self,
        pcm: &[f32],
        cancel: &AtomicBool,
    ) -> Result<[Vec<f32>; 2], ChildError> {
        if cancel.load(Ordering::Acquire) {
            return Err(ChildError::Cancelled);
        }
        if pcm.len() < 16 || pcm.len() > 392_000 || pcm.iter().any(|x| !x.is_finite()) {
            return Err(error(
                "MossFormer2 requires finite mono 16 kHz PCM from 16 to 392000 samples",
            ));
        }
        let encoded = mlx_rs::nn::relu(
            self.encoder
                .forward(&Array::from_slice(pcm, &[1, pcm.len() as i32, 1]))
                .map_err(error)?,
        )
        .map_err(error)?;
        let n = encoded.dim(1);
        let x = self
            .input_projection
            .forward(
                &self
                    .input_norm
                    .forward(&encoded, &[1, 2], 1e-8)
                    .map_err(error)?,
            )
            .map_err(error)?;
        let positions = Array::from_slice(&(0..n).map(|i| i as f32).collect::<Vec<_>>(), &[n, 1]);
        let angles = positions
            .multiply(&self.position_frequencies)
            .map_err(error)?;
        let embedding = ops::concatenate(
            &[angles.sin().map_err(error)?, angles.cos().map_err(error)?],
            -1,
        )
        .map_err(error)?
        .multiply(&self.position_scale)
        .map_err(error)?;
        let residual = x.add(embedding).map_err(error)?;
        let mut x = residual.clone();
        for (flash, fsmn) in self.flash.iter_mut().zip(&mut self.fsmn) {
            if cancel.load(Ordering::Acquire) {
                return Err(ChildError::Cancelled);
            }
            x = flash.forward(&x).map_err(error)?;
            x.eval().map_err(error)?;
            if cancel.load(Ordering::Acquire) {
                return Err(ChildError::Cancelled);
            }
            x = fsmn.forward(&x).map_err(error)?;
            x.eval().map_err(error)?;
        }
        x = self.intra_norm.forward(&x).map_err(error)?;
        x = self
            .output_norm
            .forward(&x, &[1, 2], 1e-8)
            .map_err(error)?
            .add(residual)
            .map_err(error)?;
        x = layers::prelu(&x, &self.prelu).map_err(error)?;
        x = self.output_projection.forward(&x).map_err(error)?;
        // Reshape in channel-first order, exactly as the source's [B*2,C,T] view.
        x = x
            .transpose_axes(&[0, 2, 1])
            .map_err(error)?
            .reshape(&[2, 512, n])
            .map_err(error)?
            .transpose_axes(&[0, 2, 1])
            .map_err(error)?;
        let output = ops::tanh(self.output.forward(&x).map_err(error)?).map_err(error)?;
        let gate = ops::sigmoid(self.output_gate.forward(&x).map_err(error)?).map_err(error)?;
        let masks = mlx_rs::nn::relu(
            self.mask_projection
                .forward(&output.multiply(gate).map_err(error)?)
                .map_err(error)?,
        )
        .map_err(error)?;
        let masked = masks.multiply(encoded).map_err(error)?;
        if cancel.load(Ordering::Acquire) {
            return Err(ChildError::Cancelled);
        }
        let decoded = self.decoder.forward(&masked).map_err(error)?;
        let decoded = layers::pad_time(&decoded, pcm.len() as i32).map_err(error)?;
        decoded.eval().map_err(error)?;
        if cancel.load(Ordering::Acquire) {
            return Err(ChildError::Cancelled);
        }
        let mut result = [Vec::new(), Vec::new()];
        for (channel, samples) in result.iter_mut().enumerate() {
            let output = decoded
                .index((channel as i32, ..pcm.len() as i32, 0))
                .contiguous()
                .map_err(error)?;
            *samples = output.try_as_slice::<f32>().map_err(error)?.to_vec();
        }
        if result
            .iter()
            .any(|channel| channel.len() != pcm.len() || channel.iter().any(|x| !x.is_finite()))
        {
            return Err(error("MossFormer2 produced an invalid waveform"));
        }
        Ok(result)
    }
}
