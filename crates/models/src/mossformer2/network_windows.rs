//! Raw full-window MossFormer2 execution, yielding between the 50 admitted graph stages.
//! Native waveform qualification remains an artifact admission prerequisite.
use crate::{
    operation::Operation,
    windows_runtime::{Backend, ModelAdmission, error},
};
use ort::{
    session::SessionOutputs,
    value::{DynValue, Tensor},
};
use serde::Deserialize;
use speakerdesk_core::error::ChildError;
use std::{fs, path::Path, rc::Rc};

const BLOCK_STEPS: usize = 48;

pub struct Network {
    backend: Backend,
    admission: Rc<ModelAdmission>,
    graphs: Vec<Graph>,
}

/// Dropping or invalidating a session releases only this recording's native tensors.
pub struct Session {
    state: Option<State>,
}

struct State {
    audio: DynValue,
    encoded: DynValue,
    residual: DynValue,
    x: DynValue,
    phase: usize,
    samples: usize,
    frames: usize,
    // Keep the admitted files locked even if the network is dropped before this state.
    _admission: Rc<ModelAdmission>,
}

#[derive(Deserialize)]
struct Program {
    dtype: String,
    sample_bounds: [usize; 2],
    latent_channels: usize,
    graphs: Vec<Graph>,
}

#[derive(Deserialize)]
struct Graph {
    name: String,
    model: String,
    inputs: Vec<String>,
    outputs: Vec<String>,
}

impl Program {
    fn validate(&self) -> Result<(), ChildError> {
        if self.dtype != "float32"
            || self.sample_bounds != [16, 392000]
            || self.latent_channels != 512
            || self.graphs.len() != BLOCK_STEPS + 2
        {
            return Err(error("incompatible MossFormer2 staged program"));
        }
        for (index, graph) in self.graphs.iter().enumerate() {
            let (name, inputs, outputs) = if index == 0 {
                (
                    "encoder".to_owned(),
                    vec!["audio"],
                    vec!["encoded", "residual", "x"],
                )
            } else if index == BLOCK_STEPS + 1 {
                (
                    "finish".to_owned(),
                    vec!["audio", "encoded", "residual", "x"],
                    vec!["source0", "source1"],
                )
            } else {
                let phase = index - 1;
                (
                    format!(
                        "{}-{:02}",
                        if phase % 2 == 0 { "flash" } else { "fsmn" },
                        phase / 2
                    ),
                    vec!["x"],
                    vec!["x_out"],
                )
            };
            if graph.name != name
                || graph.model != format!("{name}.onnx")
                || graph.inputs != inputs
                || graph.outputs != outputs
            {
                return Err(error(format!("incompatible MossFormer2 stage {index}")));
            }
        }
        Ok(())
    }
}

impl Network {
    pub fn load(directory: &Path, backend: &Backend) -> Result<Self, ChildError> {
        let admission = Rc::new(crate::windows_runtime::admit_model(
            directory,
            "mossformer2-16k",
        )?);
        let directory = admission.directory();
        let program: Program =
            serde_json::from_slice(&fs::read(directory.join("program.json")).map_err(error)?)
                .map_err(error)?;
        program.validate()?;
        // Validate every graph at setup; retaining all 50 DML sessions exhausts device memory.
        for graph in &program.graphs {
            backend.load_without_memory_arena(&directory.join(&graph.model))?;
        }
        Ok(Self {
            backend: backend.clone(),
            admission,
            graphs: program.graphs,
        })
    }

    pub fn begin(&mut self, pcm: &[f32], cancel: &Operation) -> Result<Session, ChildError> {
        cancel.check()?;
        let frames = window_frames(pcm)?;
        let audio = Tensor::from_array(([1, pcm.len()], pcm.to_vec()))
            .map_err(error)?
            .into_dyn();
        let mut graph = self
            .backend
            .load_without_memory_arena(&self.admission.directory().join(&self.graphs[0].model))?;
        let result = graph.run_with_options(ort::inputs!["audio" => &audio], cancel.run_options()?);
        cancel.check()?;
        let mut result = result.map_err(error)?;
        if result.len() != 3 {
            return Err(error("invalid MossFormer2 encoder output inventory"));
        }
        let encoded = output(&mut result, "encoded", &[1, 512, frames])?;
        let residual = output(&mut result, "residual", &[1, 512, frames])?;
        let x = output(&mut result, "x", &[1, frames, 512])?;
        cancel.check()?;
        Ok(Session {
            state: Some(State {
                audio,
                encoded,
                residual,
                x,
                phase: 0,
                samples: pcm.len(),
                frames,
                _admission: self.admission.clone(),
            }),
        })
    }

    /// One FLASH/FSMN block per call, followed by finish: exactly 49 calls after begin.
    pub fn step(
        &mut self,
        session: &mut Session,
        cancel: &Operation,
    ) -> Result<Option<[Vec<f32>; 2]>, ChildError> {
        // Take ownership before checking cancellation. Any failure releases this state
        // rather than preserving native buffers at an uncertain execution boundary.
        let state = session.state.take();
        cancel.check()?;
        let mut state =
            state.ok_or_else(|| error("completed or invalidated MossFormer2 session"))?;
        if state.phase < BLOCK_STEPS {
            let mut graph = self.backend.load_without_memory_arena(
                &self
                    .admission
                    .directory()
                    .join(&self.graphs[state.phase + 1].model),
            )?;
            let result =
                graph.run_with_options(ort::inputs!["x" => &state.x], cancel.run_options()?);
            cancel.check()?;
            let mut result = result.map_err(error)?;
            if result.len() != 1 {
                return Err(error("invalid MossFormer2 block output inventory"));
            }
            let x = output(&mut result, "x_out", &[1, state.frames, 512])?;
            cancel.check()?;
            state.x = x;
            state.phase += 1;
            session.state = Some(state);
            return Ok(None);
        }
        if state.phase != BLOCK_STEPS {
            return Err(error("invalid MossFormer2 session phase"));
        }
        let mut graph = self.backend.load_without_memory_arena(
            &self
                .admission
                .directory()
                .join(&self.graphs[BLOCK_STEPS + 1].model),
        )?;
        let result = graph.run_with_options(
            ort::inputs!["audio" => &state.audio, "encoded" => &state.encoded,
                         "residual" => &state.residual, "x" => &state.x],
            cancel.run_options()?,
        );
        cancel.check()?;
        let mut result = result.map_err(error)?;
        if result.len() != 2 {
            return Err(error("invalid MossFormer2 finish output inventory"));
        }
        let mut channels = [Vec::new(), Vec::new()];
        for (name, channel) in ["source0", "source1"].into_iter().zip(&mut channels) {
            let value = output(&mut result, name, &[1, state.samples])?;
            *channel = value.try_extract_tensor::<f32>().map_err(error)?.1.to_vec();
        }
        cancel.check()?;
        // Successful completion also leaves no state to reuse or native tensors to retain.
        Ok(Some(channels))
    }

    pub fn separate(
        &mut self,
        pcm: &[f32],
        cancel: &Operation,
    ) -> Result<[Vec<f32>; 2], ChildError> {
        let mut session = self.begin(pcm, cancel)?;
        loop {
            if let Some(channels) = self.step(&mut session, cancel)? {
                return Ok(channels);
            }
        }
    }
}

fn window_frames(pcm: &[f32]) -> Result<usize, ChildError> {
    if !(16..=392000).contains(&pcm.len()) || pcm.iter().any(|x| !x.is_finite()) {
        return Err(error(
            "MossFormer2 requires finite mono 16 kHz PCM from 16 to 392000 samples",
        ));
    }
    Ok((pcm.len() - 16) / 8 + 1)
}

fn output(
    outputs: &mut SessionOutputs<'_>,
    name: &str,
    expected: &[usize],
) -> Result<DynValue, ChildError> {
    // remove() retains an owned native handle; it does not copy tensor storage into Rust.
    let value = outputs
        .remove(name)
        .ok_or_else(|| error(format!("missing MossFormer2 {name}")))?;
    let (shape, data) = value.try_extract_tensor::<f32>().map_err(error)?;
    if shape.len() != expected.len()
        || shape
            .iter()
            .zip(expected)
            .any(|(&got, &want)| got != want as i64)
        || data.iter().any(|sample| !sample.is_finite())
    {
        return Err(error(format!(
            "invalid MossFormer2 {name} tensor: {shape:?}"
        )));
    }
    Ok(value)
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn program() -> serde_json::Value {
        let mut graphs = vec![json!({"name":"encoder","model":"encoder.onnx",
            "inputs":["audio"],"outputs":["encoded","residual","x"]})];
        for index in 0..24 {
            for kind in ["flash", "fsmn"] {
                let name = format!("{kind}-{index:02}");
                graphs.push(json!({"model":format!("{name}.onnx"),"name":name,"inputs":["x"],"outputs":["x_out"]}));
            }
        }
        graphs.push(json!({"name":"finish","model":"finish.onnx",
            "inputs":["audio","encoded","residual","x"],"outputs":["source0","source1"]}));
        json!({"dtype":"float32","sample_bounds":[16,392000],"latent_channels":512,"graphs":graphs})
    }

    #[test]
    fn staged_program_requires_full_context_order_and_exactly_49_post_open_steps() {
        let value = program();
        let parsed: Program = serde_json::from_value(value.clone()).unwrap();
        parsed.validate().unwrap();
        assert_eq!(parsed.graphs[1..].len(), 49);
        for index in 0..24 {
            assert_eq!(
                parsed.graphs[index * 2 + 1].name,
                format!("flash-{index:02}")
            );
            assert_eq!(
                parsed.graphs[index * 2 + 2].name,
                format!("fsmn-{index:02}")
            );
        }
        let mut changed = value.clone();
        changed["graphs"].as_array_mut().unwrap().swap(1, 2);
        assert!(
            serde_json::from_value::<Program>(changed)
                .unwrap()
                .validate()
                .is_err()
        );
        let mut changed = value.clone();
        changed["graphs"][49]["inputs"] = json!(["x"]);
        assert!(
            serde_json::from_value::<Program>(changed)
                .unwrap()
                .validate()
                .is_err()
        );
        let mut changed = value;
        changed["graphs"][3]["model"] = json!("../unadmitted.onnx");
        assert!(
            serde_json::from_value::<Program>(changed)
                .unwrap()
                .validate()
                .is_err()
        );
    }

    #[test]
    fn raw_window_bounds_and_stride_tail_preserve_original_sample_count() {
        for samples in 16..40 {
            let frames = window_frames(&vec![0.; samples]).unwrap();
            assert_eq!(frames * 8 + 8 + (samples - 16) % 8, samples);
        }
        assert_eq!(window_frames(&vec![0.; 392000]).unwrap(), 48999);
        for invalid in [
            vec![],
            vec![0.; 15],
            vec![0.; 392001],
            vec![f32::NAN; 16],
            vec![f32::INFINITY; 16],
        ] {
            assert!(window_frames(&invalid).is_err());
        }
    }
}
