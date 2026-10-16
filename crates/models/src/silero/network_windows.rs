//! The pinned Float32 Silero v6 graph; only context and recurrent state are stream-local.
use super::{CONTEXT_SAMPLES, FRAME_SAMPLES, MODEL_SAMPLES};
use crate::{
    operation::Operation,
    windows_runtime::{Backend, ModelAdmission, error},
};
use ort::{
    session::{Session as Graph, SessionOutputs},
    value::Tensor,
};
use speakerdesk_core::error::ChildError;
use std::path::Path;

pub struct Network {
    graph: Graph,
    _admission: ModelAdmission,
}

pub struct State {
    context: [f32; CONTEXT_SAMPLES],
    hidden: [f32; 128],
    cell: [f32; 128],
}

impl Default for State {
    fn default() -> Self {
        Self {
            context: [0.0; CONTEXT_SAMPLES],
            hidden: [0.0; 128],
            cell: [0.0; 128],
        }
    }
}

impl Network {
    pub fn load(directory: &Path, backend: &Backend) -> Result<Self, ChildError> {
        let admission = crate::windows_runtime::admit_model(directory, "silero-speech")?;
        let directory = admission.directory();
        Ok(Self {
            graph: backend.load(&directory.join("silero.onnx"))?,
            _admission: admission,
        })
    }

    pub fn step(
        &mut self,
        frame: &[f32; FRAME_SAMPLES],
        state: &mut State,
        operation: &Operation,
    ) -> Result<f32, ChildError> {
        operation.check()?;
        if frame.iter().any(|sample| !sample.is_finite()) {
            return Err(error("Silero requires finite mono PCM"));
        }
        let audio = Tensor::from_array((
            [1, 1, MODEL_SAMPLES + CONTEXT_SAMPLES],
            model_input(frame, &state.context).to_vec(),
        ))
        .map_err(error)?;
        let hidden = Tensor::from_array(([1, 128], state.hidden.to_vec())).map_err(error)?;
        let cell = Tensor::from_array(([1, 128], state.cell.to_vec())).map_err(error)?;
        let result = self.graph.run_with_options(
            ort::inputs! { "audio" => audio, "hidden" => hidden, "cell" => cell },
            operation.run_options()?,
        );
        operation.check()?;
        let outputs = result.map_err(error)?;
        let [probability] = output(&outputs, "probability", &[1, 1, 1])?;
        let hidden = output(&outputs, "hidden_out", &[1, 128])?;
        let cell = output(&outputs, "cell_out", &[1, 128])?;
        if !(0.0..=1.0).contains(&probability) {
            return Err(error("Silero returned an invalid speech probability"));
        }
        operation.check()?;
        // Commit only after all outputs and cancellation have been checked.
        state.hidden = hidden;
        state.cell = cell;
        state
            .context
            .copy_from_slice(&frame[FRAME_SAMPLES - CONTEXT_SAMPLES..]);
        Ok(probability)
    }
}

fn model_input(
    frame: &[f32; FRAME_SAMPLES],
    context: &[f32; CONTEXT_SAMPLES],
) -> [f32; MODEL_SAMPLES + CONTEXT_SAMPLES] {
    let mut input = [0.0; MODEL_SAMPLES + CONTEXT_SAMPLES];
    input[..CONTEXT_SAMPLES].copy_from_slice(context);
    input[CONTEXT_SAMPLES..MODEL_SAMPLES].copy_from_slice(frame);
    // The qualified graph contains STFT convolution; its host input includes right reflection.
    // Exclude the final sample itself, exactly as the pinned MLX graph and exporter do.
    for index in 0..CONTEXT_SAMPLES {
        input[MODEL_SAMPLES + index] = input[MODEL_SAMPLES - 2 - index];
    }
    input
}

fn output<const N: usize>(
    outputs: &SessionOutputs<'_>,
    name: &str,
    expected: &[i64],
) -> Result<[f32; N], ChildError> {
    let value = outputs
        .get(name)
        .ok_or_else(|| error(format!("Missing Silero output {name}")))?;
    let (shape, data) = value.try_extract_tensor::<f32>().map_err(error)?;
    if &shape[..] != expected || data.len() != N || data.iter().any(|value| !value.is_finite()) {
        return Err(error(format!("Invalid Silero output {name}")));
    }
    data.try_into().map_err(error)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn context_precedes_new_audio_and_reflection_excludes_the_final_sample() {
        let context = std::array::from_fn(|index| -((index + 1) as f32));
        let frame = std::array::from_fn(|index| index as f32);
        let input = model_input(&frame, &context);
        assert_eq!(&input[..CONTEXT_SAMPLES], &context);
        assert_eq!(&input[CONTEXT_SAMPLES..MODEL_SAMPLES], &frame);
        assert_eq!(input[MODEL_SAMPLES], 510.0);
        assert_eq!(input[input.len() - 1], 447.0);
        assert!(
            input[MODEL_SAMPLES..]
                .windows(2)
                .all(|pair| pair[1] == pair[0] - 1.0)
        );
    }
}

#[cfg(all(test, windows))]
mod native_tests {
    use super::*;
    use crate::silero::{hysteresis_flag, peak_gain, session::Session};
    use speakerdesk_core::ipc::VadFrame;
    use std::path::PathBuf;

    #[test]
    #[ignore = "requires pinned SPEAKERDESK_TEST_SILERO, SPEAKERDESK_TEST_WINDOWS_RUNTIME and DirectML"]
    fn pinned_windows_graph_preserves_reference_reset_and_packet_boundaries() {
        let directory =
            PathBuf::from(std::env::var_os("SPEAKERDESK_TEST_SILERO").expect("model directory"));
        let runtime = PathBuf::from(
            std::env::var_os("SPEAKERDESK_TEST_WINDOWS_RUNTIME")
                .expect("private runtime directory"),
        );
        let backend = Backend::from_directory(&runtime).unwrap();
        let mut network = Network::load(&directory, &backend).unwrap();
        let running = Operation::native().unwrap();
        let wav = include_bytes!("../../tests/data/two-voice.wav");
        let reference: serde_json::Value =
            serde_json::from_str(include_str!("../../tests/data/silero-reference.json")).unwrap();
        assert_eq!(
            speakerdesk_core::memo::sha256_hex(wav),
            reference["wav_sha256"].as_str().unwrap()
        );
        let mut reader = hound::WavReader::new(std::io::Cursor::new(wav)).unwrap();
        let pcm: Vec<_> = reader
            .samples::<i16>()
            .map(|sample| f32::from(sample.unwrap()) / 32768.0)
            .collect();
        assert_eq!(pcm.len() as u64, reference["samples"].as_u64().unwrap());
        assert_eq!(
            pcm.len().div_ceil(FRAME_SAMPLES),
            reference["frames"].as_array().unwrap().len()
        );
        let mut previous = Vec::new();
        for run in 0..2 {
            let started = std::time::Instant::now();
            let (mut raw, mut normalized) = (State::default(), State::default());
            let (mut actual_latch, mut expected_latch) = (false, false);
            let mut max_error = 0.0_f32;
            for (index, (chunk, expected)) in pcm
                .chunks(FRAME_SAMPLES)
                .zip(reference["frames"].as_array().unwrap())
                .enumerate()
            {
                let mut frame = [0.0; FRAME_SAMPLES];
                frame[..chunk.len()].copy_from_slice(chunk);
                let gain = peak_gain(chunk.iter().map(|sample| sample.abs()).fold(0.0, f32::max));
                let raw_p = network.step(&frame, &mut raw, &running).unwrap();
                let norm_p = network
                    .step(
                        &frame.map(|sample| sample * gain),
                        &mut normalized,
                        &running,
                    )
                    .unwrap();
                for (key, actual) in [("raw_p", raw_p), ("norm_p", norm_p)] {
                    max_error =
                        max_error.max((actual - expected[key].as_f64().unwrap() as f32).abs());
                }
                actual_latch = hysteresis_flag(actual_latch, raw_p.max(norm_p), false).latch;
                expected_latch = hysteresis_flag(
                    expected_latch,
                    expected["raw_p"]
                        .as_f64()
                        .unwrap()
                        .max(expected["norm_p"].as_f64().unwrap()) as f32,
                    false,
                )
                .latch;
                assert_eq!(
                    actual_latch, expected_latch,
                    "speech decision at frame {index}"
                );
                let states = [raw.hidden, raw.cell, normalized.hidden, normalized.cell];
                if run == 0 {
                    previous.push((raw_p, norm_p, states));
                } else {
                    let (old_raw, old_norm, old_states) = &previous[index];
                    assert!((raw_p - old_raw).abs() <= 1e-5 && (norm_p - old_norm).abs() <= 1e-5);
                    for (actual, expected) in
                        states.iter().flatten().zip(old_states.iter().flatten())
                    {
                        assert!(
                            (actual - expected).abs() <= 1e-5,
                            "recurrent reset at frame {index}"
                        );
                    }
                }
            }
            eprintln!(
                "Silero Windows pass {run}: {:?}, max reference probability drift {max_error}",
                started.elapsed()
            );
            assert!(max_error <= 0.01, "independent MLX probability bound");
        }
        packet_boundaries(&mut network, &pcm, &running);
    }

    fn frames_equal(actual: &[VadFrame], expected: &[VadFrame]) {
        assert_eq!(actual.len(), expected.len());
        for (actual, expected) in actual.iter().zip(expected) {
            assert_eq!(
                (actual.start, actual.end, actual.constant, actual.gain),
                (
                    expected.start,
                    expected.end,
                    expected.constant,
                    expected.gain
                )
            );
            assert!((actual.raw_p - expected.raw_p).abs() <= 1e-5);
            assert!((actual.norm_p - expected.norm_p).abs() <= 1e-5);
        }
    }

    fn packet_boundaries(network: &mut Network, pcm: &[f32], running: &Operation) {
        let expected = Session::new(None)
            .unwrap()
            .feed(network, pcm, true, running)
            .unwrap();
        assert_eq!(expected.observed, pcm.len() as u64);
        assert_eq!(
            expected.frames.last().unwrap().end - expected.frames.last().unwrap().start,
            37
        );
        for size in [1, 511, 512, 513, 16000] {
            let mut session = Session::new(None).unwrap();
            let mut frames = Vec::new();
            let mut received = 0;
            for chunk in pcm.chunks(size) {
                received += chunk.len() as u64;
                let output = session.feed(network, chunk, false, running).unwrap();
                assert_eq!(output.received, received);
                assert_eq!(output.observed, received / 512 * 512);
                frames.extend(output.frames);
            }
            assert!(
                session
                    .feed(network, &[], false, running)
                    .unwrap()
                    .frames
                    .is_empty()
            );
            let tail = session.feed(network, &[], true, running).unwrap();
            assert_eq!(tail.received, tail.observed);
            frames.extend(tail.frames);
            frames_equal(&frames, &expected.frames);
            assert!(session.feed(network, &[0.0], false, running).is_err());
        }
        let mut cancelled = Session::new(None).unwrap();
        let mut other = Session::new(None).unwrap();
        cancelled
            .feed(network, &pcm[..512], false, running)
            .unwrap();
        assert_eq!(
            other
                .feed(network, &pcm[..511], false, running)
                .unwrap()
                .observed,
            0
        );
        let cancellation = Operation::native().unwrap();
        cancellation.cancel();
        assert!(matches!(
            cancelled.feed(network, &pcm[512..], false, &cancellation),
            Err(ChildError::Cancelled)
        ));
        assert!(cancelled.feed(network, &[], true, running).is_err());
        frames_equal(
            &other
                .feed(network, &pcm[511..], true, running)
                .unwrap()
                .frames,
            &expected.frames,
        );
    }
}
