//! Exact-input memo for transcription, language detection and transcript timing.
//!
//! This is the resident exact-input LRU from `language_probe_cache.py`.
//! Dropped ASR reuse profiles (qualification receipts, authority files) are not stored.

use std::collections::VecDeque;

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use thiserror::Error;

pub const DEFAULT_MAX_ENTRIES: usize = 1024;
pub const DEFAULT_MAX_BYTES: usize = 4 * 1024 * 1024;
pub const F32_DTYPE: &str = "<f4";
pub const F64_DTYPE: &str = "<f8";

#[derive(Debug, Error, Clone, PartialEq, Eq)]
pub enum MemoError {
    #[error("Invalid language probe cache bound.")]
    InvalidBound,
    #[error("Invalid language probe scope.")]
    InvalidScope,
    #[error("Invalid language detection probabilities.")]
    InvalidProbabilities,
    #[error("detector failed")]
    Detector,
    #[error("asr failed")]
    Asr,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MemoScope {
    pub owner: String,
    pub model_identity: String,
}

impl MemoScope {
    pub fn new(owner: impl Into<String>, model_identity: impl Into<String>) -> Self {
        Self {
            owner: owner.into(),
            model_identity: model_identity.into(),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct ProbeKey {
    pub language_epoch: u64,
    pub start_sample: u64,
    pub dtype: String,
    pub shape: Vec<usize>,
    pub pcm_sha256: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct AsrKey {
    pub pcm_sha256: String,
    pub dtype: String,
    pub shape: Vec<usize>,
    pub language: String,
    pub sample_rate: u32,
    pub padding: (u32, u32),
    pub start_sample: u64,
    pub end_sample: u64,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
enum MemoKey {
    Probe(ProbeKey),
    Asr(AsrKey),
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AsrRecord {
    pub text: String,
    pub raw_text: String,
    pub token_count: u32,
}

#[derive(Debug, Clone)]
pub struct ExactMemo {
    max_entries: usize,
    max_bytes: usize,
    bytes: usize,
    scope: Option<MemoScope>,
    entries: VecDeque<(MemoKey, Vec<u8>)>,
}

pub fn sha256_hex(bytes: &[u8]) -> String {
    hex::encode(Sha256::digest(bytes))
}

pub fn probe_key_f32(language_epoch: u64, start_sample: u64, samples: &[f32]) -> ProbeKey {
    sample_probe_key(
        language_epoch,
        start_sample,
        F32_DTYPE,
        samples.len(),
        &f32_bytes(samples),
    )
}

pub fn probe_key_f64(language_epoch: u64, start_sample: u64, samples: &[f64]) -> ProbeKey {
    sample_probe_key(
        language_epoch,
        start_sample,
        F64_DTYPE,
        samples.len(),
        &f64_bytes(samples),
    )
}

pub fn try_probe_key_f32(
    language_epoch: i64,
    start_sample: i64,
    samples: &[f32],
) -> Result<ProbeKey, MemoError> {
    if language_epoch < 0 || start_sample < 0 {
        return Err(MemoError::InvalidScope);
    }
    Ok(probe_key_f32(
        language_epoch as u64,
        start_sample as u64,
        samples,
    ))
}

pub fn asr_key_f32(
    samples: &[f32],
    language: &str,
    sample_rate: u32,
    padding: (u32, u32),
    start_sample: u64,
    end_sample: u64,
) -> AsrKey {
    AsrKey {
        pcm_sha256: sha256_hex(&f32_bytes(samples)),
        dtype: F32_DTYPE.to_owned(),
        shape: vec![samples.len()],
        language: language.to_owned(),
        sample_rate,
        padding,
        start_sample,
        end_sample,
    }
}

/// Probabilities must be finite, inside `[0, 1]`, and sum to about 1.
pub fn validate_language_scores(probabilities: &[(String, f64)]) -> Result<(), MemoError> {
    if probabilities.is_empty() {
        return Err(MemoError::InvalidProbabilities);
    }
    let mut sum = 0.0;
    for (code, probability) in probabilities {
        if code.is_empty() || !probability.is_finite() || !(0.0..=1.0).contains(probability) {
            return Err(MemoError::InvalidProbabilities);
        }
        sum += probability;
    }
    if !(0.98..=1.02).contains(&sum) {
        return Err(MemoError::InvalidProbabilities);
    }
    Ok(())
}

impl ExactMemo {
    pub fn new(max_entries: usize, max_bytes: usize) -> Result<Self, MemoError> {
        if max_entries < 1 || max_bytes < 1 {
            return Err(MemoError::InvalidBound);
        }
        Ok(Self {
            max_entries,
            max_bytes,
            bytes: 0,
            scope: None,
            entries: VecDeque::new(),
        })
    }

    pub fn max_entries(&self) -> usize {
        self.max_entries
    }

    pub fn max_bytes(&self) -> usize {
        self.max_bytes
    }

    pub fn len(&self) -> usize {
        self.entries.len()
    }

    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    pub fn stored_bytes(&self) -> usize {
        self.bytes
    }

    pub fn probe<F>(
        &mut self,
        scope: &MemoScope,
        key: ProbeKey,
        detect: F,
    ) -> Result<Vec<(String, f64)>, MemoError>
    where
        F: FnOnce() -> Result<Vec<(String, f64)>, MemoError>,
    {
        self.align_scope(scope);
        if let Some(hit) = self.recall(&MemoKey::Probe(key.clone()))
            && let Some(scores) = decode_scores(&hit)
        {
            return Ok(scores);
        }
        let scores = detect()?;
        validate_language_scores(&scores)?;
        if let Some(bytes) = encode_scores(&scores) {
            self.insert(MemoKey::Probe(key), bytes);
        }
        Ok(scores)
    }

    pub fn asr<F>(
        &mut self,
        scope: &MemoScope,
        key: AsrKey,
        compute: F,
    ) -> Result<AsrRecord, MemoError>
    where
        F: FnOnce() -> Result<AsrRecord, MemoError>,
    {
        self.align_scope(scope);
        if let Some(hit) = self.recall(&MemoKey::Asr(key.clone()))
            && let Ok(record) = serde_json::from_slice::<AsrRecord>(&hit)
        {
            return Ok(record);
        }
        let record = compute()?;
        if let Ok(bytes) = serde_json::to_vec(&record) {
            self.insert(MemoKey::Asr(key), bytes);
        }
        Ok(record)
    }

    fn align_scope(&mut self, scope: &MemoScope) {
        if self.scope.as_ref() != Some(scope) {
            self.entries.clear();
            self.bytes = 0;
            self.scope = Some(scope.clone());
        }
    }

    fn recall(&mut self, key: &MemoKey) -> Option<Vec<u8>> {
        let position = self.entries.iter().position(|(stored, _)| stored == key)?;
        let (_, bytes) = self.entries.remove(position).expect("index exists");
        let cloned = bytes.clone();
        self.entries.push_back((key.clone(), bytes));
        Some(cloned)
    }

    fn insert(&mut self, key: MemoKey, bytes: Vec<u8>) {
        if bytes.len() > self.max_bytes {
            return;
        }
        if let Some(position) = self.entries.iter().position(|(stored, _)| stored == &key) {
            let (_, previous) = self.entries.remove(position).expect("index exists");
            self.bytes -= previous.len();
        }
        self.bytes += bytes.len();
        self.entries.push_back((key, bytes));
        while self.entries.len() > self.max_entries || self.bytes > self.max_bytes {
            let Some((_, old)) = self.entries.pop_front() else {
                break;
            };
            self.bytes -= old.len();
        }
    }
}

impl Default for ExactMemo {
    fn default() -> Self {
        Self::new(DEFAULT_MAX_ENTRIES, DEFAULT_MAX_BYTES).expect("default memo bounds are valid")
    }
}

fn sample_probe_key(
    language_epoch: u64,
    start_sample: u64,
    dtype: &str,
    len: usize,
    bytes: &[u8],
) -> ProbeKey {
    ProbeKey {
        language_epoch,
        start_sample,
        dtype: dtype.to_owned(),
        shape: vec![len],
        pcm_sha256: sha256_hex(bytes),
    }
}

fn f32_bytes(samples: &[f32]) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(samples.len() * 4);
    for sample in samples {
        bytes.extend_from_slice(&sample.to_le_bytes());
    }
    bytes
}

fn f64_bytes(samples: &[f64]) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(samples.len() * 8);
    for sample in samples {
        bytes.extend_from_slice(&sample.to_le_bytes());
    }
    bytes
}

#[derive(Serialize, Deserialize)]
struct StoredScore {
    code: String,
    bits: u64,
}

fn encode_scores(scores: &[(String, f64)]) -> Option<Vec<u8>> {
    let stored: Vec<StoredScore> = scores
        .iter()
        .map(|(code, probability)| StoredScore {
            code: code.clone(),
            bits: probability.to_bits(),
        })
        .collect();
    serde_json::to_vec(&stored).ok()
}

fn decode_scores(bytes: &[u8]) -> Option<Vec<(String, f64)>> {
    let stored: Vec<StoredScore> = serde_json::from_slice(bytes).ok()?;
    Some(
        stored
            .into_iter()
            .map(|score| (score.code, f64::from_bits(score.bits)))
            .collect(),
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    fn scope(owner: &str, model: &str) -> MemoScope {
        MemoScope::new(owner, model)
    }

    fn en_fr() -> Vec<(String, f64)> {
        vec![("en".to_owned(), 0.97), ("fr".to_owned(), 0.03)]
    }

    #[test]
    fn sha256_and_bounds_match_the_exact_input_contract() {
        assert_eq!(
            sha256_hex(b"abc"),
            "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
        );
        assert_eq!(DEFAULT_MAX_ENTRIES, 1024);
        assert_eq!(DEFAULT_MAX_BYTES, 4 * 1024 * 1024);
        assert_eq!(
            ExactMemo::new(0, 10).unwrap_err().to_string(),
            "Invalid language probe cache bound."
        );
        assert_eq!(
            try_probe_key_f32(-1, 0, &[0.0]).unwrap_err().to_string(),
            "Invalid language probe scope."
        );
        assert_eq!(
            validate_language_scores(&[("en".to_owned(), f64::NAN)])
                .unwrap_err()
                .to_string(),
            "Invalid language detection probabilities."
        );
        assert!(
            validate_language_scores(&[("en".to_owned(), 0.8), ("fr".to_owned(), 0.8)]).is_err()
        );
        assert!(validate_language_scores(&[]).is_err());
        assert!(validate_language_scores(&[("en".to_owned(), 0.98)]).is_ok());
        assert!(validate_language_scores(&[("en".to_owned(), 1.0)]).is_ok());
        assert!(
            validate_language_scores(&[("en".to_owned(), 0.51), ("fr".to_owned(), 0.51)]).is_ok()
        );
        assert!(validate_language_scores(&[("en".to_owned(), 1.02)]).is_err());
    }

    #[test]
    fn probe_hit_is_exact_and_returned_scores_are_detached() {
        let mut memo = ExactMemo::new(4, 4096).unwrap();
        let key = probe_key_f32(0, 0, &[0.25, -0.25]);
        let mut calls = 0;
        let first = memo
            .probe(&scope("detector", "model-a"), key.clone(), || {
                calls += 1;
                Ok(en_fr())
            })
            .unwrap();
        let mut mutated = first;
        mutated[0].1 = 0.0;
        let second = memo
            .probe(&scope("detector", "model-a"), key, || {
                calls += 1;
                Ok(vec![("ru".to_owned(), 1.0)])
            })
            .unwrap();
        assert_eq!(calls, 1);
        assert_eq!(second, en_fr());
        assert_eq!(mutated[0].1, 0.0);
    }

    #[test]
    fn epoch_pcm_dtype_start_and_scope_never_alias() {
        let mut memo = ExactMemo::new(8, 4096).unwrap();
        let pcm = [0.2f32, -0.2, 0.2];
        let mut calls = 0;
        let load = |calls: &mut i32| {
            *calls += 1;
            en_fr()
        };
        let base = probe_key_f32(0, 0, &pcm);
        memo.probe(&scope("detector", "model-a"), base, || Ok(load(&mut calls)))
            .unwrap();
        let variants = [
            probe_key_f32(1, 0, &pcm),
            probe_key_f32(0, 1, &pcm),
            probe_key_f32(0, 0, &[0.2, -0.2]),
            probe_key_f32(0, 0, &[0.2, -0.2, 0.21]),
            probe_key_f64(0, 0, &[0.2, -0.2, 0.2]),
        ];
        for key in variants {
            memo.probe(&scope("detector", "model-a"), key, || Ok(load(&mut calls)))
                .unwrap();
        }
        assert_eq!(calls, 6);
        memo.probe(
            &scope("other", "model-a"),
            probe_key_f32(0, 0, &pcm),
            || Ok(load(&mut calls)),
        )
        .unwrap();
        memo.probe(
            &scope("other", "model-b"),
            probe_key_f32(0, 0, &pcm),
            || Ok(vec![("fr".to_owned(), 0.97), ("en".to_owned(), 0.03)]),
        )
        .unwrap();
        assert_eq!(calls, 7);
        assert_eq!(memo.len(), 1);
    }

    #[test]
    fn eviction_drops_the_oldest_exact_entry_and_oversize_is_not_stored() {
        let mut memo = ExactMemo::new(1, 4096).unwrap();
        let first = probe_key_f32(0, 0, &[0.1]);
        let second = probe_key_f32(0, 1, &[0.1]);
        memo.probe(&scope("d", "m"), first.clone(), || Ok(en_fr()))
            .unwrap();
        memo.probe(&scope("d", "m"), second.clone(), || Ok(en_fr()))
            .unwrap();
        let mut calls = 0;
        memo.probe(&scope("d", "m"), first, || {
            calls += 1;
            Ok(en_fr())
        })
        .unwrap();
        assert_eq!(calls, 1);
        assert!(memo.len() <= 1);

        let mut tiny = ExactMemo::new(4, 1).unwrap();
        let mut calls = 0;
        tiny.probe(&scope("d", "m"), second, || {
            calls += 1;
            Ok(en_fr())
        })
        .unwrap();
        tiny.probe(&scope("d", "m"), probe_key_f32(0, 1, &[0.1]), || {
            calls += 1;
            Ok(en_fr())
        })
        .unwrap();
        assert_eq!(calls, 2);
        assert_eq!(tiny.stored_bytes(), 0);
        assert!(tiny.is_empty());
    }

    #[test]
    fn invalid_or_failed_probes_are_not_remembered() {
        let mut memo = ExactMemo::new(4, 4096).unwrap();
        let key = probe_key_f32(0, 0, &[1.0]);
        let err = memo
            .probe(&scope("d", "m"), key.clone(), || Err(MemoError::Detector))
            .unwrap_err();
        assert_eq!(err, MemoError::Detector);
        let invalid = memo.probe(&scope("d", "m"), key.clone(), || {
            Ok(vec![("en".to_owned(), f64::NAN)])
        });
        assert_eq!(invalid.unwrap_err(), MemoError::InvalidProbabilities);
        assert!(memo.is_empty());
        memo.probe(&scope("d", "m"), key, || Ok(en_fr())).unwrap();
        assert_eq!(memo.len(), 1);
    }

    #[test]
    fn asr_keys_are_exact_and_do_not_reuse_probe_entries() {
        let mut memo = ExactMemo::new(4, 4096).unwrap();
        let pcm = [0.5f32, -0.25, 0.125];
        let key = asr_key_f32(&pcm, "en", 16_000, (0, 0), 0, 3);
        let mut calls = 0;
        let record = AsrRecord {
            text: "hello".to_owned(),
            raw_text: " hello ".to_owned(),
            token_count: 2,
        };
        let first = memo
            .asr(&scope("asr", "cohere"), key.clone(), || {
                calls += 1;
                Ok(record.clone())
            })
            .unwrap();
        let second = memo
            .asr(&scope("asr", "cohere"), key, || {
                calls += 1;
                Ok(AsrRecord {
                    text: "other".to_owned(),
                    raw_text: "other".to_owned(),
                    token_count: 9,
                })
            })
            .unwrap();
        assert_eq!(calls, 1);
        assert_eq!(first, second);
        assert_eq!(second.raw_text, " hello ");

        let other_language = asr_key_f32(&pcm, "fr", 16_000, (0, 0), 0, 3);
        memo.asr(&scope("asr", "cohere"), other_language, || {
            calls += 1;
            Ok(record.clone())
        })
        .unwrap();
        let padded = asr_key_f32(&pcm, "en", 16_000, (3200, 0), 0, 3);
        memo.asr(&scope("asr", "cohere"), padded, || {
            calls += 1;
            Ok(record.clone())
        })
        .unwrap();
        assert_eq!(calls, 3);
        let failing = asr_key_f32(&[0.0], "en", 16_000, (0, 0), 0, 1);
        let missed = memo
            .asr(&scope("asr", "cohere"), failing.clone(), || {
                Err(MemoError::Asr)
            })
            .unwrap_err();
        assert_eq!(missed, MemoError::Asr);
        memo.asr(&scope("asr", "cohere"), failing, || {
            calls += 1;
            Ok(record)
        })
        .unwrap();
        memo.probe(&scope("asr", "cohere"), probe_key_f32(0, 0, &pcm), || {
            calls += 1;
            Ok(en_fr())
        })
        .unwrap();
        assert_eq!(calls, 5);
    }
}
