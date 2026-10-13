use super::layers::{self, AffineNorm, Flash, Fsmn};
use crate::weights::{error, take, verify};
use mlx_rs::{
    Array, Dtype,
    error::Exception,
    module::Module,
    nn::{Conv1d, LayerNorm},
    ops::{self, indexing::IndexOp},
};
use speakerdesk_core::error::ChildError;
use std::{
    path::Path,
    sync::atomic::{AtomicBool, Ordering},
};

pub struct Network {
    encoder: Conv1d,
    decoder: Array,
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

/// Full-window state; only execution is split, never the acoustic context.
pub struct Session {
    encoded: Array,
    residual: Array,
    x: Array,
    phase: usize,
    samples: usize,
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
            decoder: take(&mut w, "dec.weight", &[512, 1, 16], Dtype::Float32)?
                .reshape(&[512, 16])
                .map_err(error)?,
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

    pub fn begin(&mut self, pcm: &[f32], cancel: &AtomicBool) -> Result<Session, ChildError> {
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
        Ok(Session {
            x: residual.clone(),
            encoded,
            residual,
            phase: 0,
            samples: pcm.len(),
        })
    }

    pub fn step(
        &mut self,
        session: &mut Session,
        cancel: &AtomicBool,
    ) -> Result<Option<[Vec<f32>; 2]>, ChildError> {
        if cancel.load(Ordering::Acquire) {
            return Err(ChildError::Cancelled);
        }
        if session.phase < 48 {
            let index = session.phase / 2;
            let x = if session.phase % 2 == 0 {
                self.flash[index].forward(&session.x)
            } else {
                self.fsmn[index].forward(&session.x)
            }
            .map_err(error)?;
            x.eval().map_err(error)?;
            if cancel.load(Ordering::Acquire) {
                return Err(ChildError::Cancelled);
            }
            session.x = x;
            session.phase += 1;
            return Ok(None);
        }
        if session.phase != 48 {
            return Err(error("completed MossFormer2 session"));
        }
        let n = session.encoded.dim(1);
        let encoded = &session.encoded;
        let residual = &session.residual;
        let mut x = session.x.clone();
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
        let decoded = decode(&masked, &self.decoder).map_err(error)?;
        let decoded = layers::pad_time(&decoded, session.samples as i32).map_err(error)?;
        decoded.eval().map_err(error)?;
        if cancel.load(Ordering::Acquire) {
            return Err(ChildError::Cancelled);
        }
        let mut result = [Vec::new(), Vec::new()];
        for (channel, samples) in result.iter_mut().enumerate() {
            let output = decoded
                .index((channel as i32, ..session.samples as i32, 0))
                .contiguous()
                .map_err(error)?;
            *samples = output.try_as_slice::<f32>().map_err(error)?.to_vec();
        }
        if result.iter().any(|channel| {
            channel.len() != session.samples || channel.iter().any(|x| !x.is_finite())
        }) {
            return Err(error("MossFormer2 produced an invalid waveform"));
        }
        session.phase += 1;
        Ok(Some(result))
    }

    pub fn separate(
        &mut self,
        pcm: &[f32],
        cancel: &AtomicBool,
    ) -> Result<[Vec<f32>; 2], ChildError> {
        let mut session = self.begin(pcm, cancel)?;
        loop {
            if let Some(audio) = self.step(&mut session, cancel)? {
                return Ok(audio);
            }
        }
    }
}

/// Kernel 16 / stride 8 transpose convolution without MLX's dilated-input unfold buffer.
fn decode(masked: &Array, weight: &Array) -> Result<Array, Exception> {
    let frames = masked.matmul(weight)?;
    let shape = [frames.dim(0), frames.dim(1) * 8, 1];
    let head = frames.index((.., .., ..8)).reshape(&shape)?;
    let tail = frames.index((.., .., 8..)).reshape(&shape)?;
    let zero = ops::zeros::<f32>(&[frames.dim(0), 8, 1])?;
    ops::concatenate(&[head, zero.clone()], 1)?.add(ops::concatenate(&[zero, tail], 1)?)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn decoder_matches_transpose_convolution_sample_coordinates_and_zero_tails() {
        mlx_rs::with_device(mlx_rs::Device::cpu(), || {
            let mut weights = vec![0f32; 512 * 16];
            for tap in 0..16 {
                weights[tap] = (tap + 1) as f32;
                weights[16 + tap] = (16 - tap) as f32 * 0.25;
                weights[511 * 16 + tap] =
                    (tap + 1) as f32 * if tap % 2 == 0 { 0.125 } else { -0.125 };
            }
            let weight = Array::from_slice(&weights, &[512, 16]);
            for samples in 16..40 {
                let n = (samples - 16) / 8 + 1;
                let mut input = vec![0f32; 2 * n * 512];
                let mut expected = vec![0f32; 2 * samples];
                for batch in 0..2 {
                    for frame in 0..n {
                        let start = (batch * n + frame) * 512;
                        input[start] = (batch + frame + 1) as f32;
                        input[start + 1] = -(batch as f32 + 1.) * (frame as f32 + 1.) * 0.5;
                        input[start + 511] = frame as f32 - batch as f32 + 0.25;
                        for tap in 0..16 {
                            for channel in [0, 1, 511] {
                                expected[batch * samples + frame * 8 + tap] +=
                                    input[start + channel] * weights[channel * 16 + tap];
                            }
                        }
                    }
                }
                let masked = Array::from_slice(&input, &[2, n as i32, 512]);
                let decoded = decode(&masked, &weight).unwrap();
                assert_eq!(decoded.shape(), [2, (8 * n + 8) as i32, 1]);
                let padded = layers::pad_time(&decoded, samples as i32).unwrap();
                let actual = padded.try_as_slice::<f32>().unwrap();
                assert_eq!(actual.len(), expected.len());
                for (actual, expected) in actual.iter().zip(&expected) {
                    assert!(
                        (actual - expected).abs() <= 1e-5,
                        "{samples} samples: {actual} != {expected}"
                    );
                }
            }
        });
    }
}
