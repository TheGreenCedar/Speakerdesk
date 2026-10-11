//! Pinned Whisper-tiny encoder and one SOT decoder step. No transcript generation.
use super::{LanguageTokens, features::Features, parse_language_tokens};
use crate::weights::{error, linear, norm, take, verify};
use mlx_rs::{
    Array, Dtype,
    error::Exception,
    module::{Module, Param},
    nn::{Conv1d, LayerNorm, Linear},
    ops::{self, indexing::IndexOp},
};
use speakerdesk_core::{error::ChildError, ipc::LanguageProbeOut};
use std::{
    collections::HashMap,
    path::Path,
    sync::atomic::{AtomicBool, Ordering},
};

const F16: Dtype = Dtype::Float16;
type Weights = HashMap<String, Array>;

struct Attention {
    query: Linear,
    key: Linear,
    value: Linear,
    out: Linear,
}
impl Attention {
    fn load(w: &mut Weights, name: &str) -> Result<Self, ChildError> {
        Ok(Self {
            query: linear(w, &format!("{name}.query"), 384, 384, true, F16)?,
            key: linear(w, &format!("{name}.key"), 384, 384, false, F16)?,
            value: linear(w, &format!("{name}.value"), 384, 384, true, F16)?,
            out: linear(w, &format!("{name}.out"), 384, 384, true, F16)?,
        })
    }
    fn forward(&mut self, x: &Array, source: &Array) -> Result<Array, Exception> {
        let scale = Array::from(64f32.powf(-0.25)).as_dtype(F16)?;
        let heads = |x: Array| x.reshape(&[1, -1, 6, 64])?.transpose_axes(&[0, 2, 1, 3]);
        let q = heads(self.query.forward(x)?)?.multiply(&scale)?;
        let k = heads(self.key.forward(source)?)?.multiply(&scale)?;
        let v = heads(self.value.forward(source)?)?;
        let scores = q.matmul(k.transpose_axes(&[0, 1, 3, 2])?)?;
        let probabilities = ops::softmax_axis(scores, -1, Some(true))?;
        self.out.forward(
            &probabilities
                .matmul(v)?
                .transpose_axes(&[0, 2, 1, 3])?
                .reshape(&[1, x.dim(1), 384])?,
        )
    }
}

struct Block {
    attention: Attention,
    attention_norm: LayerNorm,
    cross: Option<(Attention, LayerNorm)>,
    expand: Linear,
    contract: Linear,
    ff_norm: LayerNorm,
}
impl Block {
    fn load(w: &mut Weights, name: &str, decoder: bool) -> Result<Self, ChildError> {
        Ok(Self {
            attention: Attention::load(w, &format!("{name}.attn"))?,
            attention_norm: norm(w, &format!("{name}.attn_ln"), 384, F16)?,
            cross: if decoder {
                Some((
                    Attention::load(w, &format!("{name}.cross_attn"))?,
                    norm(w, &format!("{name}.cross_attn_ln"), 384, F16)?,
                ))
            } else {
                None
            },
            expand: linear(w, &format!("{name}.mlp1"), 384, 1536, true, F16)?,
            contract: linear(w, &format!("{name}.mlp2"), 1536, 384, true, F16)?,
            ff_norm: norm(w, &format!("{name}.mlp_ln"), 384, F16)?,
        })
    }
    fn forward(&mut self, input: &Array, encoded: Option<&Array>) -> Result<Array, Exception> {
        let normalized = self.attention_norm.forward(input)?;
        let mut x = input.add(self.attention.forward(&normalized, &normalized)?)?;
        if let Some((cross, norm)) = &mut self.cross {
            x = x.add(cross.forward(&norm.forward(&x)?, encoded.expect("decoder audio"))?)?;
        }
        let hidden = gelu(&self.expand.forward(&self.ff_norm.forward(&x)?)?)?;
        x.add(self.contract.forward(&hidden)?)
    }
}

pub struct Network {
    features: Features,
    conv: [Conv1d; 2],
    positions: Array,
    encoder: Vec<Block>,
    encoder_norm: LayerNorm,
    decoder: Vec<Block>,
    decoder_norm: LayerNorm,
    sot: Array,
    language_embeddings: Array,
    tokens: LanguageTokens,
}
impl Network {
    pub fn load(directory: &Path) -> Result<Self, ChildError> {
        verify(
            directory,
            &[
                (
                    "model.safetensors",
                    "1267601753d2996d68dc065d1de786895a7ad9a874d1d3548a5274c1bc067444",
                ),
                (
                    "config.json",
                    "cf7def105e7cc5ba77a0c8fae7e97aac1ca896b70b288dec3d64fad6d9c08f7a",
                ),
                (
                    "added_tokens.json",
                    "9715fd2243b6f06a5858b5e32950d2853f73dd5bc201aafcf76f5082a2d8acd1",
                ),
            ],
        )?;
        // serde's map visitor preserves file order without implementing a JSON parser.
        let metadata: TokenEntries = serde_json::from_slice(
            &std::fs::read(directory.join("added_tokens.json")).map_err(error)?,
        )
        .map_err(error)?;
        let entries: Vec<_> = metadata
            .0
            .iter()
            .map(|(name, id)| (name.as_str(), *id))
            .collect();
        let tokens = parse_language_tokens(&entries, 51865).map_err(error)?;
        let mut w = Array::load_safetensors(directory.join("model.safetensors")).map_err(error)?;
        let embedding = take(&mut w, "decoder.token_embedding.weight", &[51865, 384], F16)?;
        let learned = take(&mut w, "decoder.positional_embedding", &[448, 384], F16)?;
        let sot = embedding
            .index(tokens.sot as i32)
            .add(learned.index(0))
            .map_err(error)?
            .reshape(&[1, 1, 384])
            .map_err(error)?;
        let language_embeddings = embedding
            .take_axis(Array::from_slice(&tokens.token_ids, &[99]), 0)
            .map_err(error)?;
        let network = Self {
            features: Features::new().map_err(error)?,
            conv: [
                conv(&mut w, "encoder.conv1", 80, 1)?,
                conv(&mut w, "encoder.conv2", 384, 2)?,
            ],
            positions: positions().map_err(error)?,
            encoder: (0..4)
                .map(|i| Block::load(&mut w, &format!("encoder.blocks.{i}"), false))
                .collect::<Result<_, _>>()?,
            encoder_norm: norm(&mut w, "encoder.ln_post", 384, F16)?,
            decoder: (0..4)
                .map(|i| Block::load(&mut w, &format!("decoder.blocks.{i}"), true))
                .collect::<Result<_, _>>()?,
            decoder_norm: norm(&mut w, "decoder.ln", 384, F16)?,
            sot,
            language_embeddings,
            tokens,
        };
        if !w.is_empty() {
            return Err(error("unexpected Whisper weight inventory"));
        }
        // Materialize the two small slices so the unused vocabulary is not retained.
        network.sot.eval().map_err(error)?;
        network.language_embeddings.eval().map_err(error)?;
        Ok(network)
    }

    pub fn probe(
        &mut self,
        pcm: &[f32],
        cancel: &AtomicBool,
    ) -> Result<LanguageProbeOut, ChildError> {
        if pcm.is_empty() || pcm.iter().any(|x| !x.is_finite()) {
            return Err(error("Whisper requires finite nonempty 16 kHz audio"));
        }
        cancelled(cancel)?;
        let probabilities = mlx_rs::with_stream(
            &mlx_rs::Stream::gpu(),
            || -> Result<Vec<f32>, ChildError> {
                let mut x = self
                    .features
                    .extract(pcm)
                    .map_err(error)?
                    .reshape(&[1, 3000, 80])
                    .map_err(error)?;
                for conv in &mut self.conv {
                    x = gelu(&conv.forward(&x).map_err(error)?).map_err(error)?;
                }
                x = x.add(&self.positions).map_err(error)?;
                for block in &mut self.encoder {
                    cancelled(cancel)?;
                    x = block.forward(&x, None).map_err(error)?;
                    x.eval().map_err(error)?;
                }
                let encoded = self.encoder_norm.forward(&x).map_err(error)?;
                let mut x = self.sot.clone();
                for block in &mut self.decoder {
                    cancelled(cancel)?;
                    x = block.forward(&x, Some(&encoded)).map_err(error)?;
                    x.eval().map_err(error)?;
                }
                let logits = self
                    .decoder_norm
                    .forward(&x)
                    .map_err(error)?
                    .matmul(self.language_embeddings.t())
                    .map_err(error)?
                    .as_dtype(Dtype::Float32)
                    .map_err(error)?;
                let probabilities = ops::softmax_axis(logits, -1, None)
                    .map_err(error)?
                    .reshape(&[99])
                    .map_err(error)?;
                probabilities.eval().map_err(error)?;
                Ok(probabilities.as_slice::<f32>().to_vec())
            },
        )?;
        cancelled(cancel)?;
        Ok(LanguageProbeOut {
            probabilities,
            codes: self.tokens.codes.clone(),
        })
    }
}

fn cancelled(flag: &AtomicBool) -> Result<(), ChildError> {
    if flag.load(Ordering::Acquire) {
        Err(ChildError::Cancelled)
    } else {
        Ok(())
    }
}
fn conv(w: &mut Weights, name: &str, input: i32, stride: i32) -> Result<Conv1d, ChildError> {
    Ok(Conv1d {
        weight: Param::new(take(w, &format!("{name}.weight"), &[384, 3, input], F16)?),
        bias: Param::new(Some(take(w, &format!("{name}.bias"), &[384], F16)?)),
        stride,
        padding: 1,
        dilation: 1,
        groups: 1,
    })
}
fn positions() -> Result<Array, Exception> {
    let frequencies = ops::arange::<_, f32>(0, 192, None)?
        .multiply(Array::from((-10000_f64.ln() / 191.) as f32))?
        .exp()?;
    let phases = ops::arange::<_, f32>(0, 1500, None)?
        .reshape(&[1500, 1])?
        .multiply(frequencies.reshape(&[1, 192])?)?;
    ops::concatenate(&[phases.sin()?, phases.cos()?], 1)?.as_dtype(F16)
}
// mlx-rs GELU's f32 constants promote half activations; preserve checkpoint arithmetic.
fn gelu(x: &Array) -> Result<Array, Exception> {
    let mut compiled = mlx_rs::transforms::compile::compile(
        |x: &Array| {
            let sqrt_two = Array::from(2f32.sqrt()).as_dtype(x.dtype())?;
            let one = Array::from(1f32).as_dtype(x.dtype())?;
            let two = Array::from(2f32).as_dtype(x.dtype())?;
            x.multiply(one.add(ops::erf(x.divide(sqrt_two)?)?)?)?
                .divide(two)
        },
        true,
    );
    compiled(x)
}

struct TokenEntries(Vec<(String, i64)>);
impl<'de> serde::Deserialize<'de> for TokenEntries {
    fn deserialize<D: serde::Deserializer<'de>>(d: D) -> Result<Self, D::Error> {
        struct Visitor;
        impl<'de> serde::de::Visitor<'de> for Visitor {
            type Value = TokenEntries;
            fn expecting(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
                f.write_str("Whisper token map")
            }
            fn visit_map<M: serde::de::MapAccess<'de>>(
                self,
                mut map: M,
            ) -> Result<TokenEntries, M::Error> {
                let mut entries = Vec::new();
                while let Some(pair) = map.next_entry()? {
                    entries.push(pair);
                }
                Ok(TokenEntries(entries))
            }
        }
        d.deserialize_map(Visitor)
    }
}
