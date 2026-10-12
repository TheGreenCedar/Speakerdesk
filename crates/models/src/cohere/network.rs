use super::{decoder::Decoder, encoder::Encoder, features::Features};
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

    pub fn transcribe(
        &mut self,
        pcm: &[f32],
        language: &str,
        padding: (u32, u32),
        cancel: &AtomicBool,
    ) -> Result<AsrText, ChildError> {
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
            return Ok(AsrText {
                text: String::new(),
                raw_text: String::new(),
                token_count: 0,
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
        let tokens = mlx_rs::with_stream(
            &mlx_rs::Stream::gpu(),
            || -> Result<Vec<u32>, ChildError> {
                let (features, valid) = self.features.extract(&padded).map_err(error)?;
                let (encoded, valid) = self.encoder.forward(&features, valid, cancel)?;
                let encoded = self.decoder.project(&encoded).map_err(error)?;
                let cross_mask = Array::from_slice(
                    &(0..encoded.dim(1) as usize)
                        .map(|i| i < valid)
                        .collect::<Vec<_>>(),
                    &[1, 1, 1, encoded.dim(1)],
                );
                let mut caches = std::array::from_fn(|_| Default::default());
                let mut tokens = Vec::new();
                let mut input = prompt.clone();
                let mut offset = 0;
                for _ in 0..448 {
                    if cancel.load(Ordering::Acquire) {
                        return Err(ChildError::Cancelled);
                    }
                    let token =
                        self.decoder
                            .step(&input, offset, &encoded, &cross_mask, &mut caches)?;
                    if token == super::EOS_TOKEN_ID {
                        return Ok(tokens);
                    }
                    tokens.push(token);
                    offset += input.len();
                    input.clear();
                    input.push(token);
                }
                Ok(tokens)
            },
        )?;
        if cancel.load(Ordering::Acquire) {
            return Err(ChildError::Cancelled);
        }
        let text = self.tokenizer.decode(&tokens, true).map_err(error)?;
        Ok(AsrText {
            raw_text: text.clone(),
            text,
            token_count: tokens.len() as u32,
        })
    }
}
