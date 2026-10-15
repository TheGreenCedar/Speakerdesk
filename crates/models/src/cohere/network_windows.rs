//! Request-local greedy decoding through the qualified Windows Cohere graphs.
use super::features::Features;
use crate::{
    operation::Operation,
    windows_runtime::{Backend, ModelAdmission, error},
};
use ort::{
    session::{Session as Graph, SessionInputValue, SessionOutputs},
    value::{DynValue, Tensor, TensorElementType, ValueType},
};
use speakerdesk_core::{error::ChildError, ipc::AsrText};
use std::path::Path;

const LANGUAGES: [&str; 14] = [
    "ar", "de", "el", "en", "es", "fr", "it", "ja", "ko", "nl", "pl", "pt", "vi", "zh",
];
const MAX_SAMPLES: usize = 35 * 16000;
const MAX_TOKENS: usize = 448;
const VOCABULARY: usize = 16384;

pub struct Network {
    encoder: Graph,
    decoder_init: Graph,
    decoder_step: Graph,
    features: Features,
    tokenizer: tokenizers::Tokenizer,
    _admission: ModelAdmission,
}

/// Encoded crop and native, owned KV tensors live only for this request.
pub struct Session {
    encoded: Option<DynValue>,
    encoded_frames: usize,
    valid_frames: usize,
    caches: Vec<DynValue>,
    tokens: Vec<u32>,
    input: Vec<u32>,
    offset: usize,
    done: bool,
}

impl Network {
    pub fn load(directory: &Path, backend: &Backend) -> Result<Self, ChildError> {
        let admission = crate::windows_runtime::admit_model(directory, "cohere-speech")?;
        let directory = admission.directory();
        let tokenizer =
            tokenizers::Tokenizer::from_file(directory.join("tokenizer.json")).map_err(error)?;
        for language in LANGUAGES {
            prompt(&tokenizer, language)?;
        }
        Ok(Self {
            encoder: backend.load(&directory.join("encoder.onnx"))?,
            decoder_init: backend.load(&directory.join("decoder-init.onnx"))?,
            decoder_step: backend.load(&directory.join("decoder-step.onnx"))?,
            features: Features::new(),
            tokenizer,
            _admission: admission,
        })
    }

    pub fn begin(
        &mut self,
        pcm: &[f32],
        language: &str,
        padding: (u32, u32),
        operation: &Operation,
    ) -> Result<Session, ChildError> {
        validate_crop(pcm, language, padding)?;
        operation.check()?;
        let mut session = Session {
            encoded: None,
            encoded_frames: 0,
            valid_frames: 0,
            caches: Vec::new(),
            tokens: Vec::new(),
            input: Vec::new(),
            offset: 0,
            done: false,
        };
        // As on macOS, boundary padding does not turn a sub-frame crop into speech.
        if pcm.len() < 160 {
            return Ok(session);
        }
        session.input = prompt(&self.tokenizer, language)?;
        let mut padded = vec![0.0; padding.0 as usize];
        padded.extend_from_slice(pcm);
        padded.resize(padded.len() + padding.1 as usize, 0.0);
        let (features, valid) = self.features.extract(&padded).map_err(error)?;
        operation.check()?;
        let frames = valid + 1;
        let features = Tensor::from_array(([1, frames, 128], features)).map_err(error)?;
        let mask = Tensor::from_array((
            [1, frames],
            (0..frames)
                .map(|frame| i64::from(frame < valid))
                .collect::<Vec<_>>(),
        ))
        .map_err(error)?;
        let mut outputs = run(
            &mut self.encoder,
            vec![
                ("features".into(), features.into()),
                ("attention_mask".into(), mask.into()),
            ],
            operation,
        )?;
        let encoded = take(&mut outputs, "encoded")?;
        let mask = take(&mut outputs, "encoder_mask")?;
        session.encoded_frames = frames.div_ceil(8);
        session.valid_frames = valid.div_ceil(8);
        expect_tensor(
            &encoded,
            TensorElementType::Float32,
            &[1, session.encoded_frames as i64, 1280],
            "encoded",
        )?;
        expect_tensor(
            &mask,
            TensorElementType::Int64,
            &[1, session.encoded_frames as i64],
            "encoder_mask",
        )?;
        let (_, mask) = mask.try_extract_tensor::<i64>().map_err(error)?;
        if mask
            .iter()
            .enumerate()
            .any(|(frame, &value)| value != i64::from(frame < session.valid_frames))
        {
            return Err(error("Cohere encoder returned an invalid frame mask"));
        }
        operation.check()?;
        session.encoded = Some(encoded);
        Ok(session)
    }

    pub fn step(
        &mut self,
        session: &mut Session,
        operation: &Operation,
    ) -> Result<Option<AsrText>, ChildError> {
        operation.check()?;
        if session.done {
            return Err(error("completed Cohere session"));
        }
        if session.tokens.len() < MAX_TOKENS
            && let Some(encoded) = &session.encoded
        {
            let length = session.input.len();
            let total = session.offset + length;
            let tokens = Tensor::from_array((
                [1, length],
                session
                    .input
                    .iter()
                    .map(|&token| i64::from(token))
                    .collect::<Vec<_>>(),
            ))
            .map_err(error)?;
            let positions = Tensor::from_array((
                [1, length],
                (session.offset..total)
                    .map(|position| position as i64)
                    .collect::<Vec<_>>(),
            ))
            .map_err(error)?;
            let self_mask =
                Tensor::from_array(([1, 1, length, total], causal_mask(session.offset, length)))
                    .map_err(error)?;
            let cross_mask = Tensor::from_array((
                [1, 1, length, session.encoded_frames],
                (0..length)
                    .flat_map(|_| {
                        (0..session.encoded_frames).map(|frame| {
                            if frame < session.valid_frames {
                                0.0
                            } else {
                                f32::MIN
                            }
                        })
                    })
                    .collect::<Vec<_>>(),
            ))
            .map_err(error)?;
            let mut inputs = vec![
                ("tokens".into(), SessionInputValue::from(tokens)),
                ("positions".into(), SessionInputValue::from(positions)),
                ("self_mask".into(), SessionInputValue::from(self_mask)),
                ("cross_mask".into(), SessionInputValue::from(cross_mask)),
            ];
            let graph = if session.offset == 0 {
                inputs.push(("encoded".into(), encoded.into()));
                &mut self.decoder_init
            } else {
                for (name, value) in cache_names().zip(&session.caches) {
                    inputs.push((format!("past_{name}"), value.into()));
                }
                &mut self.decoder_step
            };
            let mut outputs = run(graph, inputs, operation)?;
            let logits = take(&mut outputs, "logits")?;
            expect_tensor(
                &logits,
                TensorElementType::Float32,
                &[1, VOCABULARY as i64],
                "logits",
            )?;
            let (_, scores) = logits.try_extract_tensor::<f32>().map_err(error)?;
            let token = greedy_token(scores)?;
            // remove() retains the native OrtValue. Never round-trip the KV tensors through Vec.
            let mut caches = Vec::with_capacity(32);
            for name in cache_names() {
                let name = format!("present_{name}");
                let value = take(&mut outputs, &name)?;
                let frames = if name.starts_with("present_self_") {
                    total
                } else {
                    session.encoded_frames
                };
                expect_tensor(
                    &value,
                    TensorElementType::Float32,
                    &[1, 8, frames as i64, 128],
                    &name,
                )?;
                caches.push(value);
            }
            operation.check()?;
            if token != super::EOS_TOKEN_ID {
                session.caches = caches;
                session.tokens.push(token);
                session.offset = total;
                session.input.clear();
                session.input.push(token);
                return Ok(None);
            }
        }
        operation.check()?;
        let text = self
            .tokenizer
            .decode(&session.tokens, true)
            .map_err(error)?;
        session.done = true;
        Ok(Some(AsrText {
            raw_text: text.clone(),
            text,
            token_count: session.tokens.len() as u32,
        }))
    }

    pub fn transcribe(
        &mut self,
        pcm: &[f32],
        language: &str,
        padding: (u32, u32),
        operation: &Operation,
    ) -> Result<AsrText, ChildError> {
        let mut session = self.begin(pcm, language, padding, operation)?;
        loop {
            if let Some(text) = self.step(&mut session, operation)? {
                return Ok(text);
            }
        }
    }
}

fn validate_crop(pcm: &[f32], language: &str, padding: (u32, u32)) -> Result<(), ChildError> {
    if !LANGUAGES.contains(&language) {
        return Err(error("Cohere requires an explicit supported language"));
    }
    if ![0, 3200].contains(&padding.0)
        || ![0, 3200].contains(&padding.1)
        || pcm.iter().any(|value| !value.is_finite())
    {
        return Err(error("invalid Cohere audio or boundary padding"));
    }
    if pcm.len() > MAX_SAMPLES - padding.0 as usize - padding.1 as usize {
        return Err(error("Cohere crops must not exceed 35 seconds"));
    }
    Ok(())
}

fn prompt(tokenizer: &tokenizers::Tokenizer, language: &str) -> Result<Vec<u32>, ChildError> {
    let language = format!("<|{language}|>");
    [
        "▁",
        "<|startofcontext|>",
        "<|startoftranscript|>",
        "<|emo:undefined|>",
        &language,
        &language,
        "<|pnc|>",
        "<|noitn|>",
        "<|notimestamp|>",
        "<|nodiarize|>",
    ]
    .iter()
    .map(|token| {
        tokenizer
            .token_to_id(token)
            .ok_or_else(|| error(format!("missing Cohere prompt token {token}")))
    })
    .collect()
}

fn causal_mask(offset: usize, length: usize) -> Vec<f32> {
    (0..length)
        .flat_map(|row| {
            (0..offset + length).map(move |column| {
                if column <= offset + row {
                    0.0
                } else {
                    f32::MIN
                }
            })
        })
        .collect()
}

fn cache_names() -> impl Iterator<Item = String> {
    (0..8).flat_map(|layer| {
        ["self_k", "self_v", "cross_k", "cross_v"].map(move |kind| format!("{kind}_{layer}"))
    })
}

fn greedy_token(scores: &[f32]) -> Result<u32, ChildError> {
    if scores.len() != VOCABULARY || scores.iter().any(|score| !score.is_finite()) {
        return Err(error("Cohere decoder returned invalid logits"));
    }
    let mut token = 0;
    for (index, &score) in scores.iter().enumerate().skip(1) {
        if score > scores[token] {
            token = index;
        }
    }
    Ok(token as u32)
}

fn take(outputs: &mut SessionOutputs<'_>, name: &str) -> Result<DynValue, ChildError> {
    outputs
        .remove(name)
        .ok_or_else(|| error(format!("missing Cohere graph output {name}")))
}

fn expect_tensor(
    value: &DynValue,
    element: TensorElementType,
    expected: &[i64],
    name: &str,
) -> Result<(), ChildError> {
    if matches!(value.dtype(), ValueType::Tensor { ty, shape, .. }
        if *ty == element && &shape[..] == expected)
    {
        Ok(())
    } else {
        Err(error(format!(
            "invalid Cohere graph tensor {name}: {:?}",
            value.dtype()
        )))
    }
}

fn run<'a>(
    graph: &'a mut Graph,
    inputs: Vec<(String, SessionInputValue<'_>)>,
    operation: &'a Operation,
) -> Result<SessionOutputs<'a>, ChildError> {
    operation.check()?;
    let result = graph.run_with_options(inputs, operation.run_options()?);
    // An interrupted native run is cancellation, rather than a model failure receipt.
    operation.check()?;
    result.map_err(error)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn crop_budget_includes_both_boundaries_and_requires_an_explicit_language() {
        let pcm = vec![0.0; MAX_SAMPLES - 6400];
        for language in LANGUAGES {
            validate_crop(&pcm, language, (3200, 3200)).unwrap();
        }
        assert!(validate_crop(&vec![0.0; pcm.len() + 1], "en", (3200, 3200)).is_err());
        assert!(validate_crop(&[], "auto", (0, 0)).is_err());
        assert!(validate_crop(&[], "EN", (0, 0)).is_err());
        assert!(validate_crop(&[], "en", (0, 1)).is_err());
        for sample in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
            assert!(validate_crop(&[sample], "en", (0, 0)).is_err());
        }
    }

    #[test]
    fn prompt_prefill_is_causal_and_incremental_tokens_see_all_previous_positions() {
        assert_eq!(
            causal_mask(0, 3),
            [0.0, f32::MIN, f32::MIN, 0.0, 0.0, f32::MIN, 0.0, 0.0, 0.0]
        );
        assert_eq!(causal_mask(10, 1), [0.0; 11]);
    }

    #[test]
    fn greedy_decoding_keeps_the_first_tie_and_rejects_nonfinite_scores() {
        let mut scores = vec![-100.0; VOCABULARY];
        scores[3] = 5.0;
        scores[4] = 5.0;
        assert_eq!(greedy_token(&scores).unwrap(), super::super::EOS_TOKEN_ID);
        scores[4] = 5.001;
        assert_eq!(greedy_token(&scores).unwrap(), 4);
        scores[0] = f32::NAN;
        assert!(greedy_token(&scores).is_err());
        assert!(greedy_token(&[]).is_err());
    }

    #[test]
    #[ignore = "requires pinned SPEAKERDESK_TEST_COHERE, SPEAKERDESK_TEST_WINDOWS_RUNTIME and DirectML"]
    fn pinned_windows_reference_preserves_tokens_across_resident_requests() {
        let model = std::env::var_os("SPEAKERDESK_TEST_COHERE").expect("model directory");
        let runtime = std::env::var_os("SPEAKERDESK_TEST_WINDOWS_RUNTIME")
            .expect("private runtime directory");
        let backend = Backend::from_directory(Path::new(&runtime)).unwrap();
        let started = std::time::Instant::now();
        let mut network = Network::load(Path::new(&model), &backend).unwrap();
        eprintln!("Cohere Windows load {:?}", started.elapsed());
        let reference: serde_json::Value =
            serde_json::from_str(include_str!("../../tests/data/cohere-reference.json")).unwrap();
        let expected: Vec<u32> = reference["tokens"]
            .as_array()
            .unwrap()
            .iter()
            .map(|value| value.as_u64().unwrap() as u32)
            .collect();
        let mut reader = hound::WavReader::new(std::io::Cursor::new(include_bytes!(
            "../../tests/data/two-voice.wav"
        )))
        .unwrap();
        let pcm: Vec<_> = reader
            .samples::<i16>()
            .map(|sample| f32::from(sample.unwrap()) / 32768.0)
            .collect();
        for run in 0..2 {
            let operation = Operation::native().unwrap();
            let started = std::time::Instant::now();
            let mut session = network.begin(&pcm, "en", (0, 0), &operation).unwrap();
            let result = loop {
                if let Some(result) = network.step(&mut session, &operation).unwrap() {
                    break result;
                }
            };
            eprintln!("Cohere Windows resident run {run}: {:?}", started.elapsed());
            assert_eq!(session.tokens, expected);
            assert_eq!(result.text, reference["text"].as_str().unwrap());
            assert_eq!(result.raw_text, result.text);
            assert_eq!(result.token_count as usize, expected.len());
            assert!(network.step(&mut session, &operation).is_err());
        }
        let operation = Operation::native().unwrap();
        let empty = network
            .transcribe(&[0.0; 159], "en", (3200, 3200), &operation)
            .unwrap();
        assert!(empty.text.is_empty());
        assert!(empty.raw_text.is_empty());
        assert_eq!(empty.token_count, 0);
        let operation = Operation::native().unwrap();
        operation.cancel();
        assert!(matches!(
            network.begin(&pcm, "en", (0, 0), &operation),
            Err(ChildError::Cancelled)
        ));
    }
}
