use super::{
    decoder::{Cache, Decoder},
    encoder::Encoder,
    features::Features,
};
use crate::weights::{error, verify};
use mlx_rs::Array;
use speakerdesk_core::{error::ChildError, ipc::AsrText};
use std::{
    path::Path,
    sync::atomic::{AtomicBool, Ordering},
};

pub struct Network {
    encoder: Encoder,
    decoder: Decoder,
    features: Features,
    tokenizer: tokenizers::Tokenizer,
}
/// Full encoded crop and request-local KV caches for cooperative decoding.
pub struct Session {
    encoded: Option<Array>,
    cross_mask: Option<Array>,
    caches: [Cache; 8],
    tokens: Vec<u32>,
    input: Vec<u32>,
    offset: usize,
    done: bool,
}

impl Network {
    pub fn load(directory: &Path) -> Result<Self, ChildError> {
        verify(
            directory,
            &[
                (
                    "model.safetensors",
                    "ae947c13ba1cb8ce24c8bf72ded2520dcda1894ebf5a54fbb803c03bc28ef7fb",
                ),
                (
                    "config.json",
                    "c9f2934c33c22014a339d2451493c72f2fddff289ca83c62cd07c6502fcc092d",
                ),
                (
                    "preprocessor_config.json",
                    "9f297d330646ecc8ebb9dc5784f48b7c35b118c913e306a1ccd0192f2c976332",
                ),
                (
                    "tokenizer.json",
                    "780ccca2de2ccd289971b1fb7d4f0b5ec2dc908872f6e350181c7eba9db3fa9f",
                ),
            ],
        )?;
        let mut weights =
            Array::load_safetensors(directory.join("model.safetensors")).map_err(error)?;
        let network = Self {
            encoder: Encoder::load(&mut weights)?,
            decoder: Decoder::load(&mut weights)?,
            features: Features::new(),
            tokenizer: tokenizers::Tokenizer::from_file(directory.join("tokenizer.json"))
                .map_err(error)?,
        };
        if !weights.is_empty() {
            return Err(error("unexpected Cohere weight inventory"));
        }
        Ok(network)
    }

    pub fn begin(
        &mut self,
        pcm: &[f32],
        language: &str,
        padding: (u32, u32),
        cancel: &AtomicBool,
    ) -> Result<Session, ChildError> {
        if ![
            "ar", "de", "el", "en", "es", "fr", "it", "ja", "ko", "nl", "pl", "pt", "vi", "zh",
        ]
        .contains(&language)
        {
            return Err(error("Cohere requires an explicit supported language"));
        }
        if ![0, 3200].contains(&padding.0)
            || ![0, 3200].contains(&padding.1)
            || pcm.iter().any(|v| !v.is_finite())
        {
            return Err(error("invalid Cohere audio or boundary padding"));
        }
        // shortcut: one bounded crop; add energy-based splitting before accepting longer input.
        if pcm.len() + padding.0 as usize + padding.1 as usize > 35 * 16000 {
            return Err(error("Cohere crops must not exceed 35 seconds"));
        }
        if cancel.load(Ordering::Acquire) {
            return Err(ChildError::Cancelled);
        }
        if pcm.len() < 160 {
            return Ok(Session {
                encoded: None,
                cross_mask: None,
                caches: std::array::from_fn(|_| Default::default()),
                tokens: Vec::new(),
                input: Vec::new(),
                offset: 0,
                done: false,
            });
        }
        let mut padded = vec![0.0; padding.0 as usize];
        padded.extend_from_slice(pcm);
        padded.resize(padded.len() + padding.1 as usize, 0.0);
        let prompt: Vec<u32> = [
            "▁",
            "<|startofcontext|>",
            "<|startoftranscript|>",
            "<|emo:undefined|>",
            &format!("<|{language}|>"),
            &format!("<|{language}|>"),
            "<|pnc|>",
            "<|noitn|>",
            "<|notimestamp|>",
            "<|nodiarize|>",
        ]
        .iter()
        .map(|token| {
            self.tokenizer
                .token_to_id(token)
                .ok_or_else(|| error(format!("missing Cohere prompt token {token}")))
        })
        .collect::<Result<_, _>>()?;
        let (encoded, cross_mask) = mlx_rs::with_stream(
            &mlx_rs::Stream::gpu(),
            || -> Result<(Array, Array), ChildError> {
                let (features, valid) = self.features.extract(&padded).map_err(error)?;
                let (encoded, valid) = self.encoder.forward(&features, valid, cancel)?;
                let encoded = self.decoder.project(&encoded).map_err(error)?;
                let cross_mask = Array::from_slice(
                    &(0..encoded.dim(1) as usize)
                        .map(|i| i < valid)
                        .collect::<Vec<_>>(),
                    &[1, 1, 1, encoded.dim(1)],
                );
                Ok((encoded, cross_mask))
            },
        )?;
        Ok(Session {
            encoded: Some(encoded),
            cross_mask: Some(cross_mask),
            caches: std::array::from_fn(|_| Default::default()),
            tokens: Vec::new(),
            input: prompt,
            offset: 0,
            done: false,
        })
    }

    pub fn step(
        &mut self,
        session: &mut Session,
        cancel: &AtomicBool,
    ) -> Result<Option<AsrText>, ChildError> {
        if cancel.load(Ordering::Acquire) {
            return Err(ChildError::Cancelled);
        }
        if session.done {
            return Err(error("completed Cohere session"));
        }
        let finish = if let Some(encoded) = &session.encoded {
            if session.tokens.len() >= 448 {
                true
            } else {
                let token = mlx_rs::with_stream(&mlx_rs::Stream::gpu(), || {
                    self.decoder.step(
                        &session.input,
                        session.offset,
                        encoded,
                        session.cross_mask.as_ref().expect("encoded mask"),
                        &mut session.caches,
                    )
                })?;
                if token == super::EOS_TOKEN_ID {
                    true
                } else {
                    session.tokens.push(token);
                    session.offset += session.input.len();
                    session.input.clear();
                    session.input.push(token);
                    false
                }
            }
        } else {
            true
        };
        if cancel.load(Ordering::Acquire) {
            return Err(ChildError::Cancelled);
        }
        if !finish {
            return Ok(None);
        }
        session.done = true;
        let text = self
            .tokenizer
            .decode(&session.tokens, true)
            .map_err(error)?;
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
        cancel: &AtomicBool,
    ) -> Result<AsrText, ChildError> {
        let mut session = self.begin(pcm, language, padding, cancel)?;
        loop {
            if let Some(text) = self.step(&mut session, cancel)? {
                return Ok(text);
            }
        }
    }
}
