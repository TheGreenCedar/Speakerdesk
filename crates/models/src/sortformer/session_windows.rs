//! Source-local Nemotron history and cache policy, with explicit native graph state.
use super::network::Network;
use crate::operation::Operation;
use speakerdesk_core::{
    error::ChildError,
    ipc::{DiarFeedOut, DiarPreset, DiarTurn},
};

pub(super) struct Request<'a> {
    pub pcm: &'a [f32],
    pub sample_offset: usize,
    pub total_samples: usize,
    pub start: usize,
    pub count: usize,
    pub valid: usize,
    pub cache: &'a [[f32; 512]],
    pub fifo: &'a [[f32; 512]],
}

pub(super) struct Step {
    pub chunk: Vec<[f32; 512]>,
    pub high: Vec<[f32; 8]>,
    pub low: Vec<[f32; 8]>,
}

pub(super) trait Runner {
    fn run(&mut self, request: Request<'_>, cancel: &Operation) -> Result<Step, ChildError>;
    fn silence(&self) -> &[f32; 512];
}

fn invalid(message: &str) -> ChildError {
    ChildError::failed(message, false)
}

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
            state: State::new(fifo, period),
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
        cancel: &Operation,
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
        cancel: &Operation,
    ) -> Result<Vec<[f32; 8]>, ChildError> {
        self.advance(network, pcm, last, cancel)
    }

    fn advance(
        &mut self,
        network: &mut impl Runner,
        pcm: &[f32],
        last: bool,
        cancel: &Operation,
    ) -> Result<Vec<[f32; 8]>, ChildError> {
        if self.finished || pcm.iter().any(|x| !x.is_finite()) {
            return Err(invalid("invalid or finished Nemotron session"));
        }
        // Any interrupted feed is unusable at an unknown observation boundary.
        self.finished = true;
        let mut output = Vec::new();
        for chunk in pcm.chunks(16000) {
            cancel.check()?;
            self.received = self
                .received
                .checked_add(chunk.len())
                .ok_or_else(|| invalid("Nemotron sample count overflow"))?;
            self.pcm.extend_from_slice(chunk);
            self.process(network, false, cancel, &mut output)?;
        }
        if last {
            self.process(network, true, cancel, &mut output)?;
        }
        cancel.check()?;
        self.finished = last;
        if last {
            self.pcm.clear();
        }
        Ok(output)
    }

    fn process(
        &mut self,
        network: &mut impl Runner,
        last: bool,
        cancel: &Operation,
        output: &mut Vec<[f32; 8]>,
    ) -> Result<(), ChildError> {
        loop {
            cancel.check()?;
            let available = self.received / 160 - self.frames;
            let needed = (self.frames + self.central + self.right - 1)
                .checked_mul(160)
                .and_then(|n| n.checked_add(256))
                .ok_or_else(|| invalid("Nemotron sample count overflow"))?;
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
            let prefix = self.state.cache.len() + self.state.fifo.len();
            let step = network.run(
                Request {
                    pcm: &self.pcm,
                    sample_offset: self.sample_offset,
                    total_samples: self.received,
                    start: self.frames,
                    count,
                    valid: available.min(count),
                    cache: &self.state.cache,
                    fifo: &self.state.fifo,
                },
                cancel,
            )?;
            cancel.check()?;
            let total = prefix + count / 8;
            if step.chunk.len() != count / 8
                || step.high.len() != total * 8
                || step.low.len() != total
                || step.chunk.iter().flatten().any(|v| !v.is_finite())
                || step
                    .high
                    .iter()
                    .chain(&step.low)
                    .flatten()
                    .any(|p| !p.is_finite() || !(0.0..=1.0).contains(p))
            {
                return Err(invalid("invalid Nemotron graph output"));
            }
            self.state
                .update(&step, central.div_ceil(8), network.silence());
            output.extend_from_slice(&step.high[prefix * 8..prefix * 8 + central]);
            self.frames += central;
            let keep = (self.frames * 160).saturating_sub(257);
            self.pcm.drain(..keep - self.sample_offset);
            self.sample_offset = keep;
        }
        Ok(())
    }
}

struct State {
    cache: Vec<[f32; 512]>,
    fifo: Vec<[f32; 512]>,
    cache_predictions: Vec<[f32; 8]>,
    compressed: bool,
    fifo_limit: usize,
    update_period: usize,
}

impl State {
    fn new(fifo_limit: usize, update_period: usize) -> Self {
        Self {
            cache: Vec::new(),
            fifo: Vec::new(),
            cache_predictions: Vec::new(),
            compressed: false,
            fifo_limit,
            update_period,
        }
    }

    fn update(&mut self, step: &Step, central: usize, silence: &[f32; 512]) {
        let cache_len = self.cache.len();
        let prefix = cache_len + self.fifo.len();
        self.fifo.extend_from_slice(&step.chunk[..central]);
        let predictions = &step.low[cache_len..prefix + central];
        if self.fifo.len() > self.fifo_limit {
            let pop = self
                .fifo
                .len()
                .min(self.update_period.max(self.fifo.len() - self.fifo_limit));
            if !self.compressed {
                self.cache_predictions = step.low[..cache_len].to_vec();
            }
            self.cache.extend(self.fifo.drain(..pop));
            self.cache_predictions
                .extend_from_slice(&predictions[..pop]);
            if self.cache.len() > 264 {
                (self.cache, self.cache_predictions) =
                    compress(&self.cache, &self.cache_predictions, silence);
                self.compressed = true;
            }
        }
    }
}

/// Retain speaker-major arrival order, including repeated source frames and silence slots.
/// Equal scores use source-index order; backend partition ties still need long-chain qualification.
fn compress(
    embeddings: &[[f32; 512]],
    predictions: &[[f32; 8]],
    silence: &[f32; 512],
) -> (Vec<[f32; 512]>, Vec<[f32; 8]>) {
    let mut scores = Vec::with_capacity(predictions.len());
    for row in predictions {
        let other = row.map(|p| (1. - p).max(0.25).ln());
        let sum: f32 = other.iter().sum();
        scores.push(std::array::from_fn::<_, 8, _>(|s| {
            if row[s] > 0.5 {
                row[s].max(0.25).ln() - other[s] + sum - 0.5_f32.ln()
            } else {
                f32::NEG_INFINITY
            }
        }));
    }
    for speaker in 0..8 {
        let enough = scores.iter().filter(|row| row[speaker] > 0.).count() >= 16;
        for (frame, row) in scores.iter_mut().enumerate() {
            if enough && row[speaker] <= 0. && predictions[frame][speaker] > 0.5 {
                row[speaker] = f32::NEG_INFINITY;
            }
            if frame >= 264 {
                row[speaker] += 0.05;
            }
        }
    }
    for (count, scale) in [(24, 2.), (48, 1.)] {
        for speaker in 0..8 {
            let mut indices: Vec<_> = (0..scores.len()).collect();
            indices.sort_unstable_by(|a, b| {
                scores[*b][speaker]
                    .total_cmp(&scores[*a][speaker])
                    .then(a.cmp(b))
            });
            for index in indices.into_iter().take(count) {
                if scores[index][speaker].is_finite() {
                    scores[index][speaker] += -scale * 0.5_f32.ln();
                }
            }
        }
    }
    let frames = scores.len();
    let mut ranked: Vec<_> = (0..8)
        .flat_map(|speaker| (0..=frames).map(move |frame| (speaker, frame)))
        .collect();
    ranked.sort_unstable_by(|a, b| {
        let score = |&(s, f): &(usize, usize)| {
            if f == frames {
                f32::INFINITY
            } else {
                scores[f][s]
            }
        };
        score(b).total_cmp(&score(a)).then(a.cmp(b))
    });
    ranked.truncate(264);
    // Disabled selections sort after every valid speaker-major frame, as in the source's 99999 sentinel.
    let mut selected: Vec<_> = ranked
        .into_iter()
        .map(|(s, f)| {
            if f == frames || scores[f][s] > f32::NEG_INFINITY {
                s * (frames + 1) + f
            } else {
                99999
            }
        })
        .collect();
    selected.sort_unstable();
    selected
        .into_iter()
        .map(|index| {
            let frame = index % (frames + 1);
            if index == 99999 || frame >= frames {
                (*silence, [0.; 8])
            } else {
                (embeddings[frame], predictions[frame])
            }
        })
        .unzip()
}

#[cfg(test)]
mod tests {
    use super::*;

    struct Fake {
        silence: [f32; 512],
        calls: Vec<(usize, usize, usize, usize)>,
        cancel_after_run: bool,
    }
    impl Fake {
        fn new() -> Self {
            Self {
                silence: [-1.; 512],
                calls: Vec::new(),
                cancel_after_run: false,
            }
        }
    }
    impl Runner for Fake {
        fn silence(&self) -> &[f32; 512] {
            &self.silence
        }
        fn run(&mut self, r: Request<'_>, operation: &Operation) -> Result<Step, ChildError> {
            assert_eq!(r.sample_offset + r.pcm.len(), r.total_samples);
            self.calls
                .push((r.start, r.count, r.cache.len(), r.fifo.len()));
            let chunk: Vec<_> = (0..r.count / 8)
                .map(|frame| [(r.start / 8 + frame) as f32; 512])
                .collect();
            let embeddings: Vec<_> = r.cache.iter().chain(r.fifo).chain(&chunk).collect();
            let mut high = Vec::new();
            for embedding in embeddings {
                for sub in 0..8 {
                    let frame = embedding[0] as usize * 8 + sub;
                    high.push([if frame % 7 < 3 { 0.75 } else { 0.25 }; 8]);
                }
            }
            let low = high
                .chunks_exact(8)
                .map(|rows| std::array::from_fn(|s| rows.iter().map(|r| r[s]).sum::<f32>() / 8.))
                .collect();
            if self.cancel_after_run {
                operation.cancel();
            }
            Ok(Step { chunk, high, low })
        }
    }

    #[test]
    fn fragmented_low_feed_preserves_frames_tail_history_and_new_source_state() {
        let pcm = vec![0.; 48037];
        let operation = Operation::new();
        let mut whole = Session::new(DiarPreset::Low).unwrap();
        let expected = whole
            .advance(&mut Fake::new(), &pcm, true, &operation)
            .unwrap();
        let mut fragmented = Session::new(DiarPreset::Low).unwrap();
        let mut network = Fake::new();
        let mut actual = Vec::new();
        let mut cursor = 0;
        for size in [1, 159, 160, 511, 512, 16000, 17, 21000] {
            actual.extend(
                fragmented
                    .advance(&mut network, &pcm[cursor..cursor + size], false, &operation)
                    .unwrap(),
            );
            cursor += size;
        }
        actual.extend(
            fragmented
                .advance(&mut network, &pcm[cursor..], true, &operation)
                .unwrap(),
        );
        assert_eq!(actual, expected);
        assert_eq!(actual.len(), pcm.len() / 160);
        assert!(fragmented.pcm.is_empty());
        assert_eq!(fragmented.received, pcm.len());
        assert!(
            fragmented
                .advance(&mut network, &[], true, &operation)
                .is_err()
        );
        let mut fresh = Session::new(DiarPreset::Low).unwrap();
        let mut network = Fake::new();
        fresh
            .advance(&mut network, &pcm[..549], true, &operation)
            .unwrap();
        assert_eq!(network.calls[0], (0, 16, 0, 0));
        let mut empty = Session::new(DiarPreset::Offline).unwrap();
        assert!(
            empty
                .advance(&mut network, &[0.; 159], true, &operation)
                .unwrap()
                .is_empty()
        );
    }

    #[test]
    fn cancellation_poisoning_precedes_cache_commit_and_cannot_be_reused() {
        let operation = Operation::new();
        let mut network = Fake::new();
        network.cancel_after_run = true;
        let mut session = Session::new(DiarPreset::Low).unwrap();
        assert!(matches!(
            session.advance(&mut network, &[0.; 16000], true, &operation),
            Err(ChildError::Cancelled)
        ));
        assert!(session.state.cache.is_empty() && session.state.fifo.is_empty());
        assert_eq!(session.frames, 0);
        assert!(
            session
                .advance(&mut Fake::new(), &[], true, &Operation::new())
                .is_err()
        );
        let mut valid = Session::new(DiarPreset::Low).unwrap();
        assert!(
            valid
                .advance(&mut Fake::new(), &[f32::NAN], false, &Operation::new())
                .is_err()
        );
        assert!(!valid.finished);
    }

    #[test]
    fn malformed_native_state_cannot_advance_or_reuse_a_session() {
        struct Bad(usize);
        impl Runner for Bad {
            fn silence(&self) -> &[f32; 512] {
                &[-1.; 512]
            }
            fn run(&mut self, r: Request<'_>, operation: &Operation) -> Result<Step, ChildError> {
                let mut step = Fake::new().run(r, operation)?;
                match self.0 {
                    0 => {
                        step.low.pop();
                    }
                    1 => {
                        step.high[0][0] = f32::NAN;
                    }
                    _ => {
                        step.chunk[0][0] = f32::INFINITY;
                    }
                }
                Ok(step)
            }
        }
        for kind in 0..3 {
            let mut session = Session::new(DiarPreset::Low).unwrap();
            assert!(
                session
                    .advance(&mut Bad(kind), &[0.; 16000], true, &Operation::new())
                    .is_err()
            );
            assert_eq!(session.frames, 0);
            assert!(session.state.cache.is_empty() && session.state.fifo.is_empty());
            assert!(
                session
                    .advance(&mut Fake::new(), &[], true, &Operation::new())
                    .is_err()
            );
        }
    }

    #[test]
    fn cache_overflow_moves_arrival_order_and_preserves_compressed_predictions() {
        let mut state = State::new(2, 2);
        let mut step = Step {
            chunk: vec![[0.; 512], [1.; 512], [2.; 512]],
            high: Vec::new(),
            low: vec![[0.1; 8]; 3],
        };
        state.update(&step, 3, &[-1.; 512]);
        assert_eq!(
            state.cache.iter().map(|r| r[0]).collect::<Vec<_>>(),
            [0., 1.]
        );
        assert_eq!(state.fifo.iter().map(|r| r[0]).collect::<Vec<_>>(), [2.]);
        // A real compressed prefix has 264 rows. Fresh neural predictions for that
        // prefix must not replace the evidence that selected its cached embeddings.
        state.cache = (0..264).map(|frame| [frame as f32; 512]).collect();
        state.cache_predictions = vec![[0.1; 8]; 264];
        state.fifo = vec![[264.; 512]];
        state.compressed = true;
        step.chunk = vec![[300.; 512], [301.; 512], [302.; 512]];
        step.low = vec![[0.9; 8]; 268];
        state.update(&step, 3, &[-1.; 512]);
        assert_eq!(state.cache.len(), 264);
        assert!(
            state
                .cache
                .iter()
                .all(|row| [-1., 264., 300.].contains(&row[0]))
        );
        assert!(state.cache.iter().any(|row| row[0] == 264.));
        assert!(state.cache.iter().any(|row| row[0] == 300.));
    }

    #[test]
    fn compression_retains_speaker_major_duplicates_and_explicit_silence_slots() {
        let embeddings: Vec<_> = (0..300).map(|n| [n as f32; 512]).collect();
        let mut predictions = vec![[0.; 8]; 300];
        for row in &mut predictions[..64] {
            row[0] = 0.99;
            row[1] = 0.99;
        }
        let (cache, scores) = compress(&embeddings, &predictions, &[-1.; 512]);
        assert_eq!(cache.len(), 264);
        for speaker in 0..2 {
            assert_eq!(
                cache[speaker * 65..speaker * 65 + 64]
                    .iter()
                    .map(|r| r[0])
                    .collect::<Vec<_>>(),
                (0..64).map(|n| n as f32).collect::<Vec<_>>()
            );
            assert_eq!(cache[speaker * 65 + 64], [-1.; 512]);
        }
        assert!(cache[130..].iter().all(|r| *r == [-1.; 512]));
        assert!(scores[130..].iter().all(|r| *r == [0.; 8]));
        let (cache, scores) = compress(&embeddings, &vec![[0.; 8]; 300], &[-1.; 512]);
        assert!(cache.iter().all(|r| *r == [-1.; 512]));
        assert!(scores.iter().all(|r| *r == [0.; 8]));
    }
}

#[cfg(all(test, windows))]
mod native_tests {
    use super::*;
    use crate::windows_runtime::Backend;
    use std::{fs, path::PathBuf};

    // These are qualified native-driver captures, not an independent acoustic oracle.
    #[test]
    #[ignore = "requires pinned SPEAKERDESK_TEST_NEMOTRON, SPEAKERDESK_TEST_WINDOWS_RUNTIME and SPEAKERDESK_TEST_NEMOTRON_LONG"]
    fn resident_native_sessions_preserve_long_cache_continuation() {
        let path = |name| PathBuf::from(std::env::var_os(name).expect(name));
        let backend = Backend::from_directory(&path("SPEAKERDESK_TEST_WINDOWS_RUNTIME")).unwrap();
        let mut network = Network::load(&path("SPEAKERDESK_TEST_NEMOTRON"), &backend).unwrap();
        let reference = path("SPEAKERDESK_TEST_NEMOTRON_LONG");
        let metadata: serde_json::Value =
            serde_json::from_slice(&fs::read(reference.join("result.json")).unwrap()).unwrap();
        let wav = include_bytes!("../../tests/data/two-voice.wav");
        assert_eq!(
            metadata["source_sha256"],
            speakerdesk_core::memo::sha256_hex(wav)
        );
        assert_eq!(metadata["repetitions"], 6);
        assert_eq!(metadata["passed"], true);
        let mut reader = hound::WavReader::new(std::io::Cursor::new(wav)).unwrap();
        let source: Vec<f32> = reader
            .samples::<i16>()
            .map(|v| f32::from(v.unwrap()) / 32768.)
            .collect();
        let pcm = source.repeat(6);
        let mut sessions = [
            Session::new(DiarPreset::Offline).unwrap(),
            Session::new(DiarPreset::Low).unwrap(),
        ];
        let mut actual = [Vec::new(), Vec::new()];
        let started = std::time::Instant::now();
        let sizes = [1, 159, 160, 511, 512, 16000, 17];
        let mut offset = 0;
        for size in sizes.into_iter().cycle() {
            if offset == pcm.len() {
                break;
            }
            let end = (offset + size).min(pcm.len());
            for (session, output) in sessions.iter_mut().zip(&mut actual) {
                output.extend(
                    session
                        .advance(
                            &mut network,
                            &pcm[offset..end],
                            false,
                            &Operation::native().unwrap(),
                        )
                        .unwrap(),
                );
                assert_eq!(session.received, end);
                assert!(session.state.cache.len() <= 264);
                assert!(session.state.fifo.len() <= session.state.fifo_limit);
            }
            offset = end;
        }
        for (session, output) in sessions.iter_mut().zip(&mut actual) {
            output.extend(
                session
                    .advance(&mut network, &[], true, &Operation::native().unwrap())
                    .unwrap(),
            );
            assert_eq!(session.frames, pcm.len() / 160);
            assert_eq!(session.frames, 5454);
            assert_eq!(session.received, pcm.len());
            assert!(session.state.compressed && !session.state.cache.is_empty());
            assert!(session.pcm.is_empty());
            assert!(
                session
                    .advance(&mut network, &[], true, &Operation::native().unwrap())
                    .is_err()
            );
        }
        for ((mode, hash), output) in [
            (
                "offline",
                "047fbeab38a1c329ed580c06f329e90d05b5fb71215a1f021d8f1b2ee94aa73e",
            ),
            (
                "low",
                "a64eeb533600541f460dacfde3e01e267991a7e7280109a2bd463a6cd8238e98",
            ),
        ]
        .into_iter()
        .zip(actual)
        {
            let bytes = fs::read(reference.join(format!("{mode}-dml/probabilities.f32"))).unwrap();
            assert_eq!(
                speakerdesk_core::memo::sha256_hex(&bytes),
                hash,
                "qualified driver capture changed"
            );
            let expected: Vec<_> = bytes
                .chunks_exact(4)
                .map(|v| f32::from_le_bytes(v.try_into().unwrap()))
                .collect();
            assert_eq!(expected.len(), output.len() * 8);
            let mut maximum = 0_f32;
            let mut decisions = 0;
            for (actual, expected) in output.into_iter().flatten().zip(expected) {
                maximum = maximum.max((actual - expected).abs());
                decisions += usize::from((actual > 0.5) != (expected > 0.5));
            }
            eprintln!(
                "Nemotron resident {mode}: max drift {maximum}, changed decisions {decisions}, interleaved elapsed {:?}",
                started.elapsed()
            );
            assert!(
                maximum <= 0.001,
                "qualified native-driver probability bound"
            );
            assert_eq!(decisions, 0);
        }
    }
}
