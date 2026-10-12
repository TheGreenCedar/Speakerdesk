//! Per-recording Nemotron state. Weights are shared; PCM history and speaker evidence are not.
use super::{cache::State, network::Network};
use crate::weights::error;
use speakerdesk_core::{
    error::ChildError,
    ipc::{DiarFeedOut, DiarPreset, DiarTurn},
};
use std::sync::atomic::{AtomicBool, Ordering};

pub struct Session {
    state: State,
    central: usize,
    right: usize,
    pcm: Vec<f32>,
    sample_offset: usize,
    received: usize,
    frames: usize,
    finished: bool,
}

impl Session {
    pub fn new(preset: DiarPreset) -> Result<Self, ChildError> {
        let (central, right, fifo, period) = match preset {
            DiarPreset::Offline => (340 * 8, 40 * 8, 40, 300),
            DiarPreset::Low => (9 * 8, 4 * 8, 264, 222),
        };
        Ok(Self {
            state: State::new(fifo, period).map_err(error)?,
            central,
            right,
            pcm: Vec::new(),
            sample_offset: 0,
            received: 0,
            frames: 0,
            finished: false,
        })
    }

    pub fn feed(
        &mut self,
        network: &mut Network,
        pcm: &[f32],
        last: bool,
        cancel: &AtomicBool,
    ) -> Result<DiarFeedOut, ChildError> {
        let start = self.frames * 160;
        let probabilities = self.probabilities(network, pcm, last, cancel)?;
        let turns = super::frames_to_turns(&probabilities)
            .into_iter()
            .map(|turn| DiarTurn {
                start: start as u64 + (turn.start * 16000.).round() as u64,
                end: start as u64 + (turn.end * 16000.).round() as u64,
                speaker: turn.speaker,
            })
            .collect();
        Ok(DiarFeedOut {
            turns,
            received: self.received as u64,
            observed: (self.frames * 160) as u64,
        })
    }

    pub(super) fn probabilities(
        &mut self,
        network: &mut Network,
        pcm: &[f32],
        last: bool,
        cancel: &AtomicBool,
    ) -> Result<Vec<[f32; 8]>, ChildError> {
        if self.finished || pcm.iter().any(|x| !x.is_finite()) {
            return Err(error("invalid or finished Nemotron session"));
        }
        // An interrupted stateful feed cannot be reused at an unknown observation boundary.
        self.finished = true;
        let mut output = Vec::new();
        for chunk in pcm.chunks(16000) {
            if cancel.load(Ordering::Acquire) {
                return Err(ChildError::Cancelled);
            }
            self.pcm.extend_from_slice(chunk);
            self.received += chunk.len();
            self.process(network, false, cancel, &mut output)?;
        }
        if last {
            self.process(network, true, cancel, &mut output)?;
        }
        if cancel.load(Ordering::Acquire) {
            return Err(ChildError::Cancelled);
        }
        self.finished = last;
        if last {
            self.pcm.clear();
        }
        Ok(output)
    }

    fn process(
        &mut self,
        network: &mut Network,
        last: bool,
        cancel: &AtomicBool,
        output: &mut Vec<[f32; 8]>,
    ) -> Result<(), ChildError> {
        loop {
            let available = self.received / 160 - self.frames;
            let needed = (self.frames + self.central + self.right - 1) * 160 + 256;
            if available == 0 || (!last && self.received < needed) {
                break;
            }
            let central = available.min(self.central);
            let count = if last {
                ((self.received / 160 + 1).div_ceil(16) * 16 - self.frames)
                    .min(self.central + self.right)
            } else {
                self.central + self.right
            };
            let features = network
                .features(
                    &self.pcm,
                    self.sample_offset,
                    self.received,
                    self.frames,
                    count,
                )
                .map_err(error)?;
            let result = mlx_rs::with_stream(&mlx_rs::Stream::gpu(), || {
                network.step(
                    &features,
                    available.min(count),
                    central,
                    &mut self.state,
                    cancel,
                )
            })?;
            let values = result.as_slice::<f32>();
            if values
                .iter()
                .any(|p| !p.is_finite() || !(0.0..=1.0).contains(p))
            {
                return Err(error("invalid Nemotron probabilities"));
            }
            output.extend_from_slice(values.as_chunks::<8>().0);
            self.frames += central;
            let keep = (self.frames * 160).saturating_sub(257);
            self.pcm.drain(..keep - self.sample_offset);
            self.sample_offset = keep;
        }
        Ok(())
    }
}
