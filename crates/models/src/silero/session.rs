use std::sync::atomic::{AtomicBool, Ordering};

use speakerdesk_core::{
    error::ChildError,
    ipc::{GainPolicy, VadFeedOut, VadFrame},
};

use super::{
    FRAME_SAMPLES,
    network::{Network, State},
    peak_gain,
};

/// Only a partial frame and recurrent state survive a feed; weights belong to the child.
pub struct Session {
    raw: State,
    normalized: State,
    pending: [f32; FRAME_SAMPLES],
    pending_len: usize,
    received: u64,
    observed: u64,
    previous: Option<f32>,
    fixed_gain: Option<f32>,
    ended: bool,
}

impl Session {
    pub fn new(conditioning: Option<GainPolicy>) -> Result<Self, ChildError> {
        let fixed_gain = conditioning
            .map(speakerdesk_core::speech::normalized_view_gain)
            .transpose()
            .map_err(|e| ChildError::failed(e.to_string(), false))?;
        Ok(Self {
            raw: State::default(),
            normalized: State::default(),
            pending: [0.0; FRAME_SAMPLES],
            pending_len: 0,
            received: 0,
            observed: 0,
            previous: None,
            fixed_gain,
            ended: false,
        })
    }

    pub fn feed(
        &mut self,
        network: &mut Network,
        pcm: &[f32],
        last: bool,
        cancel: &AtomicBool,
    ) -> Result<VadFeedOut, ChildError> {
        if self.ended || pcm.iter().any(|s| !s.is_finite()) {
            return Err(ChildError::failed(
                "Silero requires an open session and finite mono PCM",
                false,
            ));
        }
        self.received = self
            .received
            .checked_add(pcm.len() as u64)
            .ok_or_else(|| ChildError::failed("Silero sample endpoint overflow", false))?;
        let result = self.consume(network, pcm, last, cancel);
        // Partial recurrent work cannot be replayed safely after an error or cancellation.
        self.ended = last || result.is_err();
        result
    }

    fn consume(
        &mut self,
        network: &mut Network,
        mut pcm: &[f32],
        last: bool,
        cancel: &AtomicBool,
    ) -> Result<VadFeedOut, ChildError> {
        let mut frames = Vec::new();
        loop {
            if cancel.load(Ordering::Acquire) {
                return Err(ChildError::Cancelled);
            }
            let count = pcm.len().min(FRAME_SAMPLES - self.pending_len);
            self.pending[self.pending_len..self.pending_len + count].copy_from_slice(&pcm[..count]);
            self.pending_len += count;
            pcm = &pcm[count..];
            if self.pending_len == 0 || (self.pending_len < FRAME_SAMPLES && !last) {
                break;
            }
            let valid = self.pending_len;
            let original = &self.pending[..valid];
            let first = original[0];
            let constant =
                self.previous.is_none_or(|s| s == first) && original.iter().all(|s| *s == first);
            let previous = original[valid - 1];
            let peak = original.iter().map(|s| s.abs()).fold(0.0, f32::max);
            let gain = self.fixed_gain.unwrap_or_else(|| peak_gain(peak));
            self.pending[valid..].fill(0.0);
            let raw_p = network.step(&self.pending, &mut self.raw)?;
            let normalized = self.pending.map(|s| s * gain);
            let norm_p = network.step(&normalized, &mut self.normalized)?;
            frames.push(VadFrame {
                start: self.observed,
                end: self.observed + valid as u64,
                raw_p,
                norm_p,
                gain,
                constant,
            });
            self.observed += valid as u64;
            self.previous = Some(previous);
            self.pending_len = 0;
            if pcm.is_empty() {
                break;
            }
        }
        Ok(VadFeedOut {
            frames,
            received: self.received,
            observed: self.observed,
        })
    }
}
