use super::layers::{BF16, Weights, embedding, heads, quantized, scalar};
use crate::weights::{error, norm};
use mlx_rs::{
    Array, Dtype,
    error::Exception,
    module::Module,
    nn::{LayerNorm, QuantizedEmbedding, QuantizedLinear},
    ops::{self, indexing::IndexOp},
};
use speakerdesk_core::error::ChildError;

type Kv = (Array, Array);
#[derive(Default)]
pub struct Cache {
    self_kv: Option<Kv>,
    cross_kv: Option<Kv>,
}

struct Attention {
    q: QuantizedLinear,
    k: QuantizedLinear,
    v: QuantizedLinear,
    out: QuantizedLinear,
}
impl Attention {
    fn load(w: &mut Weights, prefix: &str) -> Result<Self, ChildError> {
        Ok(Self {
            q: quantized(w, &format!("{prefix}.q_proj"), 1024, 1024, true)?,
            k: quantized(w, &format!("{prefix}.k_proj"), 1024, 1024, true)?,
            v: quantized(w, &format!("{prefix}.v_proj"), 1024, 1024, true)?,
            out: quantized(w, &format!("{prefix}.o_proj"), 1024, 1024, true)?,
        })
    }
    fn kv(&mut self, x: &Array) -> Result<Kv, Exception> {
        Ok((heads(self.k.forward(x)?, 8)?, heads(self.v.forward(x)?, 8)?))
    }
    fn forward(&mut self, x: &Array, (k, v): &Kv, mask: &Array) -> Result<Array, Exception> {
        let q = heads(self.q.forward(x)?, 8)?;
        let scores = q.matmul(k.transpose_axes(&[0, 1, 3, 2])?)?;
        let scores = scores.multiply(scalar(1.0 / 128f32.sqrt(), &scores)?)?;
        let scores = ops::select(mask, &scores, scalar(f32::MIN, &scores)?)?;
        let weights =
            ops::softmax_axis(scores.as_dtype(Dtype::Float32)?, -1, None)?.as_dtype(x.dtype())?;
        let result = weights
            .matmul(v)?
            .transpose_axes(&[0, 2, 1, 3])?
            .reshape(&[1, x.dim(1), 1024])?;
        self.out.forward(&result)
    }
}

struct Layer {
    self_attention: Attention,
    cross_attention: Attention,
    expand: QuantizedLinear,
    contract: QuantizedLinear,
    norm1: LayerNorm,
    norm2: LayerNorm,
    norm3: LayerNorm,
}
impl Layer {
    fn load(w: &mut Weights, prefix: &str) -> Result<Self, ChildError> {
        Ok(Self {
            self_attention: Attention::load(w, &format!("{prefix}.self_attn"))?,
            cross_attention: Attention::load(w, &format!("{prefix}.encoder_attn"))?,
            expand: quantized(w, &format!("{prefix}.mlp.fc1"), 1024, 4096, true)?,
            contract: quantized(w, &format!("{prefix}.mlp.fc2"), 4096, 1024, true)?,
            norm1: norm(w, &format!("{prefix}.input_layernorm"), 1024, BF16)?,
            norm2: norm(w, &format!("{prefix}.post_attention_layernorm"), 1024, BF16)?,
            norm3: norm(w, &format!("{prefix}.final_layernorm"), 1024, BF16)?,
        })
    }
    fn forward(
        &mut self,
        x: &Array,
        encoded: &Array,
        self_mask: &Array,
        cross_mask: &Array,
        cache: &mut Cache,
    ) -> Result<Array, Exception> {
        let norm = self.norm1.forward(x)?;
        let (mut k, mut v) = self.self_attention.kv(&norm)?;
        if let Some((past_k, past_v)) = &cache.self_kv {
            k = ops::concatenate(&[past_k, &k], 2)?;
            v = ops::concatenate(&[past_v, &v], 2)?;
        }
        let kv = (k, v);
        let x = x.add(self.self_attention.forward(&norm, &kv, self_mask)?)?;
        cache.self_kv = Some(kv);
        if cache.cross_kv.is_none() {
            cache.cross_kv = Some(self.cross_attention.kv(encoded)?);
        }
        let x = x.add(self.cross_attention.forward(
            &self.norm2.forward(&x)?,
            cache.cross_kv.as_ref().expect("created above"),
            cross_mask,
        )?)?;
        x.add(self.contract.forward(&mlx_rs::nn::relu(
            &self.expand.forward(&self.norm3.forward(&x)?)?,
        )?)?)
    }
}

pub struct Decoder {
    tokens: QuantizedEmbedding,
    positions: QuantizedEmbedding,
    embedding_norm: LayerNorm,
    projection: QuantizedLinear,
    layers: Vec<Layer>,
    norm: LayerNorm,
    output: QuantizedLinear,
}
impl Decoder {
    pub fn load(w: &mut Weights) -> Result<Self, ChildError> {
        Ok(Self {
            tokens: embedding(w, "decoder.embed_tokens", 16384)?,
            positions: embedding(w, "decoder.pos_emb", 1024)?,
            embedding_norm: norm(w, "decoder.embedding_layernorm", 1024, BF16)?,
            projection: quantized(w, "decoder.proj", 1280, 1024, true)?,
            layers: (0..8)
                .map(|i| Layer::load(w, &format!("decoder.layers.{i}")))
                .collect::<Result<_, _>>()?,
            norm: norm(w, "decoder.norm", 1024, BF16)?,
            output: quantized(w, "proj_out", 1024, 16384, true)?,
        })
    }
    pub fn project(&mut self, encoded: &Array) -> Result<Array, Exception> {
        self.projection.forward(encoded)
    }

    pub fn step(
        &mut self,
        ids: &[u32],
        offset: usize,
        encoded: &Array,
        valid: usize,
        caches: &mut [Cache; 8],
    ) -> Result<u32, ChildError> {
        let run = || -> Result<u32, Exception> {
            let length = ids.len() as i32;
            let tokens = Array::from_slice(ids, &[1, length]);
            let positions = Array::from_slice(
                &(offset as u32..(offset + ids.len()) as u32).collect::<Vec<_>>(),
                &[length],
            );
            let mut x = self.embedding_norm.forward(
                &self
                    .tokens
                    .forward(&tokens)?
                    .add(self.positions.forward(&positions)?)?,
            )?;
            let total = offset + ids.len();
            let causal = Array::from_slice(
                &(0..ids.len())
                    .flat_map(|row| (0..total).map(move |column| column <= offset + row))
                    .collect::<Vec<_>>(),
                &[1, 1, length, total as i32],
            );
            let cross = Array::from_slice(
                &(0..encoded.dim(1) as usize)
                    .map(|i| i < valid)
                    .collect::<Vec<_>>(),
                &[1, 1, 1, encoded.dim(1)],
            );
            for (layer, cache) in self.layers.iter_mut().zip(caches.iter_mut()) {
                x = layer.forward(&x, encoded, &causal, &cross, cache)?;
            }
            let logits = self
                .output
                .forward(&self.norm.forward(&x)?)?
                .index((0, length - 1, ..));
            let token = ops::indexing::argmax(logits, None)?;
            token.eval()?;
            // Materialized caches cannot retain a chain of prior decoder graphs.
            for cache in caches {
                for (k, v) in [&cache.self_kv, &cache.cross_kv].into_iter().flatten() {
                    k.eval()?;
                    v.eval()?;
                }
            }
            Ok(token.item_exact::<u32>())
        };
        run().map_err(error)
    }
}
