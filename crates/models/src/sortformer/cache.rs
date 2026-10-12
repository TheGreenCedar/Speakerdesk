//! Arrival-order speaker cache, using the pinned Nemotron checkpoint policy.
use mlx_rs::{
    Array, Dtype,
    error::Exception,
    ops::{self, indexing::IndexOp},
};

const CACHE: i32 = 264;

pub(super) struct State {
    pub cache: Array,
    pub fifo: Array,
    cache_predictions: Array,
    fifo_predictions: Array,
    compressed: bool,
    fifo_limit: i32,
    update_period: i32,
}

impl State {
    pub fn new(fifo_limit: i32, update_period: i32) -> Result<Self, Exception> {
        Ok(Self {
            cache: ops::zeros_dtype(&[1, 0, 512], Dtype::Bfloat16)?,
            fifo: ops::zeros_dtype(&[1, 0, 512], Dtype::Bfloat16)?,
            cache_predictions: ops::zeros::<f32>(&[1, 0, 8])?,
            fifo_predictions: ops::zeros::<f32>(&[1, 0, 8])?,
            compressed: false,
            fifo_limit,
            update_period,
        })
    }

    pub fn update(
        &mut self,
        chunk: &Array,
        predictions: &Array,
        central: i32,
        silence: &Array,
    ) -> Result<(), Exception> {
        let cache_len = self.cache.dim(1);
        let start = cache_len + self.fifo.dim(1);
        self.fifo = ops::concatenate(&[&self.fifo, &chunk.index((.., ..central, ..))], 1)?;
        self.fifo_predictions = ops::concatenate(
            &[
                predictions.index((.., cache_len..start, ..)),
                predictions.index((.., start..start + central, ..)),
            ],
            1,
        )?;
        if self.fifo.dim(1) > self.fifo_limit {
            let pop = self
                .fifo
                .dim(1)
                .min(self.update_period.max(self.fifo.dim(1) - self.fifo_limit));
            self.cache = ops::concatenate(&[&self.cache, &self.fifo.index((.., ..pop, ..))], 1)?;
            let previous = if self.compressed {
                self.cache_predictions.clone()
            } else {
                predictions.index((.., ..cache_len, ..))
            };
            self.cache_predictions =
                ops::concatenate(&[previous, self.fifo_predictions.index((.., ..pop, ..))], 1)?;
            self.fifo = self.fifo.index((.., pop.., ..));
            self.fifo_predictions = self.fifo_predictions.index((.., pop.., ..));
            if self.cache.dim(1) > CACHE {
                (self.cache, self.cache_predictions) =
                    compress(&self.cache, &self.cache_predictions, silence)?;
                self.compressed = true;
            }
        }
        Ok(())
    }

    pub fn eval(&self, output: &Array) -> Result<(), Exception> {
        mlx_rs::transforms::eval([
            output,
            &self.cache,
            &self.cache_predictions,
            &self.fifo,
            &self.fifo_predictions,
        ])
    }
}

fn boost(scores: &Array, count: i32, scale: f32) -> Result<Array, Exception> {
    let mut columns = Vec::with_capacity(8);
    for speaker in 0..8 {
        let column = scores.index((.., .., speaker));
        let indices =
            ops::argpartition_axis(column.negative()?, count - 1, 1)?.index((.., ..count));
        let mask = ops::zeros_like(&column)?.put_along_axis(
            &indices,
            ops::ones::<f32>(indices.shape())?,
            1,
        )?;
        let finite = column
            .gt(Array::from(f32::NEG_INFINITY))?
            .as_dtype(Dtype::Float32)?;
        columns.push(
            column.add(
                mask.multiply(Array::from(-scale * 0.5f32.ln()))?
                    .multiply(finite)?,
            )?,
        );
    }
    ops::stack(&columns, -1)
}

/// Selection preserves speaker-major order and intentionally permits repeated source frames.
pub(super) fn compress(
    embeddings: &Array,
    predictions: &Array,
    silence: &Array,
) -> Result<(Array, Array), Exception> {
    let complement = Array::from(1f32).subtract(predictions)?;
    let log = ops::maximum(predictions, Array::from(0.25f32))?.log()?;
    let log_complement = ops::maximum(complement, Array::from(0.25f32))?.log()?;
    let mut scores = log
        .subtract(&log_complement)?
        .add(log_complement.sum_axis(2, true)?)?
        .subtract(Array::from(0.5f32.ln()))?;
    let speech = predictions.gt(Array::from(0.5f32))?;
    scores = ops::select(&speech, scores, Array::from(f32::NEG_INFINITY))?;
    let positive = scores.gt(Array::from(0f32))?;
    let enough = positive
        .as_dtype(Dtype::Float32)?
        .sum_axis(1, true)?
        .ge(Array::from(16f32))?;
    let disable = positive
        .logical_not()?
        .logical_and(&speech)?
        .logical_and(enough)?;
    scores = ops::select(disable, Array::from(f32::NEG_INFINITY), scores)?;
    let frames = scores.dim(1);
    let latest = Array::from_slice(
        &(0..frames)
            .map(|i| if i >= CACHE { 0.05f32 } else { 0. })
            .collect::<Vec<_>>(),
        &[1, frames, 1],
    );
    scores = scores.add(latest)?;
    scores = boost(&scores, 24, 2.)?;
    scores = boost(&scores, 48, 1.)?;
    scores = ops::concatenate(
        &[
            scores,
            ops::full::<f32>(&[1, 1, 8], Array::from(f32::INFINITY))?,
        ],
        1,
    )?;
    let flat = scores.transpose_axes(&[0, 2, 1])?.reshape(&[1, -1])?;
    let chosen = ops::argpartition_axis(flat.negative()?, CACHE - 1, 1)?
        .index((.., ..CACHE))
        .as_dtype(Dtype::Int32)?;
    let valid = flat
        .take_along_axis(&chosen, 1)?
        .gt(Array::from(f32::NEG_INFINITY))?;
    let ordered = ops::sort_axis(ops::select(valid, chosen, Array::from(99999i32))?, 1)?;
    let indices = ordered.remainder(Array::from(frames + 1))?;
    let disabled = ordered
        .eq(Array::from(99999i32))?
        .logical_or(indices.ge(Array::from(frames))?)?;
    let indices = ops::select(&disabled, Array::from(0i32), indices)?.reshape(&[CACHE])?;
    let disabled = disabled.reshape(&[1, CACHE, 1])?;
    let result = (
        ops::select(
            &disabled,
            silence.reshape(&[1, 1, 512])?,
            embeddings.take_axis(&indices, 1)?,
        )?,
        ops::select(
            &disabled,
            Array::from(0f32),
            predictions.take_axis(&indices, 1)?,
        )?,
    );
    mlx_rs::transforms::eval([&result.0, &result.1])?;
    Ok(result)
}
