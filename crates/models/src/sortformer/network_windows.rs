//! Explicit-state Nemotron graphs. Artifact admission belongs to the Windows backend owner.
use super::session::{Request, Runner, Session, Step};
use crate::{
    operation::Operation,
    windows_runtime::{Backend, ModelAdmission, error},
};
use ort::{session::SessionOutputs, value::Tensor};
use speakerdesk_core::{error::ChildError, ipc::DiarPreset};
use std::{fs, path::Path};

pub struct Network {
    frontend: ort::session::Session,
    neural: ort::session::Session,
    silence: [f32; 512],
    _admission: ModelAdmission,
}

impl Network {
    pub fn load(directory: &Path, backend: &Backend) -> Result<Self, ChildError> {
        let admission = crate::windows_runtime::admit_model(directory, "nemotron")?;
        let directory = admission.directory();
        let bytes = fs::read(directory.join("silence.f32")).map_err(error)?;
        if bytes.len() != 512 * 4 {
            return Err(error(
                "Nemotron silence embedding must contain 512 Float32 values",
            ));
        }
        let mut silence = [0.; 512];
        for (value, bytes) in silence.iter_mut().zip(bytes.as_chunks::<4>().0) {
            *value = f32::from_le_bytes(*bytes);
        }
        if silence.iter().any(|value| !value.is_finite()) {
            return Err(error(
                "Nemotron silence embedding contains nonfinite values",
            ));
        }
        Ok(Self {
            frontend: backend.load(&directory.join("frontend.onnx"))?,
            neural: backend.load(&directory.join("nemotron.onnx"))?,
            silence,
            _admission: admission,
        })
    }

    pub fn probabilities(
        &mut self,
        pcm: &[f32],
        cancel: &Operation,
    ) -> Result<Vec<[f32; 8]>, ChildError> {
        Session::new(DiarPreset::Offline)?.probabilities(self, pcm, true, cancel)
    }
}

impl Runner for Network {
    fn silence(&self) -> &[f32; 512] {
        &self.silence
    }

    fn run(&mut self, request: Request<'_>, cancel: &Operation) -> Result<Step, ChildError> {
        cancel.check()?;
        if request.count == 0 || request.count % 8 != 0 || request.valid > request.count {
            return Err(error("invalid Nemotron native frame count"));
        }
        let frames = frame_windows(
            request.pcm,
            request.sample_offset,
            request.total_samples,
            request.start,
            request.count,
            cancel,
        )?;
        let frames = Tensor::from_array(([1, request.count, 512], frames)).map_err(error)?;
        let valid = Tensor::from_array(([1usize], vec![request.valid as i64])).map_err(error)?;
        let result = self.frontend.run_with_options(
            ort::inputs!["audio_windows" => frames, "valid_frames" => valid],
            cancel.run_options()?,
        );
        // Provider termination errors must retain the operation's cancellation classification.
        cancel.check()?;
        let result = result.map_err(error)?;
        let features = output::<128>(&result, "features", request.count)?;
        drop(result);
        let features = Tensor::from_array((
            [1, request.count, 128],
            features.into_iter().flatten().collect::<Vec<_>>(),
        ))
        .map_err(error)?;
        let cache = tensor(request.cache)?;
        let fifo = tensor(request.fifo)?;
        let valid = Tensor::from_array(([1usize], vec![request.valid as i64])).map_err(error)?;
        cancel.check()?;
        let result = self.neural.run_with_options(
            ort::inputs!["features" => features, "cache" => cache, "fifo" => fifo, "valid_frames" => valid],
            cancel.run_options()?,
        );
        cancel.check()?;
        let result = result.map_err(error)?;
        let chunk_frames = request.count / 8;
        let total = request.cache.len() + request.fifo.len() + chunk_frames;
        let step = Step {
            chunk: output::<512>(&result, "chunk", chunk_frames)?,
            high: output::<8>(&result, "high", total * 8)?,
            low: output::<8>(&result, "low", total)?,
        };
        if step
            .high
            .iter()
            .chain(&step.low)
            .flatten()
            .any(|p| !(0.0..=1.0).contains(p))
        {
            return Err(error("invalid Nemotron probabilities"));
        }
        Ok(step)
    }
}

fn tensor<const WIDTH: usize>(rows: &[[f32; WIDTH]]) -> Result<Tensor<f32>, ChildError> {
    Tensor::from_array((
        [1, rows.len(), WIDTH],
        rows.iter().flatten().copied().collect::<Vec<_>>(),
    ))
    .map_err(error)
}

fn output<const WIDTH: usize>(
    outputs: &SessionOutputs<'_>,
    name: &str,
    frames: usize,
) -> Result<Vec<[f32; WIDTH]>, ChildError> {
    let value = outputs
        .get(name)
        .ok_or_else(|| error(format!("missing Nemotron {name}")))?;
    let (shape, values) = value.try_extract_tensor::<f32>().map_err(error)?;
    if **shape != [1, frames as i64, WIDTH as i64]
        || values.len() != frames * WIDTH
        || values.iter().any(|value| !value.is_finite())
    {
        return Err(error(format!("invalid Nemotron {name} tensor: {shape:?}")));
    }
    Ok(values.as_chunks::<WIDTH>().0.to_vec())
}

fn frame_windows(
    pcm: &[f32],
    sample_offset: usize,
    total_samples: usize,
    start: usize,
    count: usize,
    cancel: &Operation,
) -> Result<Vec<f32>, ChildError> {
    if sample_offset.checked_add(pcm.len()) != Some(total_samples) {
        return Err(error("invalid Nemotron PCM history coordinates"));
    }
    let length = count
        .checked_mul(512)
        .ok_or_else(|| error("Nemotron framing overflow"))?;
    let mut output = vec![0.; length];
    for (frame, window) in output.as_chunks_mut::<512>().0.iter_mut().enumerate() {
        cancel.check()?;
        for (offset, value) in window.iter_mut().enumerate() {
            let position = (start as i128 + frame as i128) * 160 + offset as i128 - 256;
            let local = position - sample_offset as i128;
            if let Ok(local) = usize::try_from(local)
                && let Some(current) = pcm.get(local)
            {
                let previous = local
                    .checked_sub(1)
                    .and_then(|p| pcm.get(p))
                    .copied()
                    .unwrap_or(0.);
                *value = *current - 0.97 * previous;
            }
        }
    }
    Ok(output)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn centered_preemphasis_uses_global_samples_and_retained_left_context() {
        let pcm: Vec<_> = (0..1200).map(|sample| sample as f32 / 1200.).collect();
        let operation = Operation::new();
        let whole = frame_windows(&pcm, 0, pcm.len(), 0, 8, &operation).unwrap();
        assert!(whole[..256].iter().all(|x| *x == 0.));
        assert_eq!(whole[257], pcm[1] - 0.97 * pcm[0]);
        let retained = frame_windows(&pcm[223..], 223, pcm.len(), 3, 5, &operation).unwrap();
        assert_eq!(retained, whole[3 * 512..]);
        assert!(frame_windows(&pcm, 1, pcm.len(), 0, 8, &operation).is_err());
        operation.cancel();
        assert!(matches!(
            frame_windows(&pcm, 0, pcm.len(), 0, 8, &operation),
            Err(ChildError::Cancelled)
        ));
    }
}
