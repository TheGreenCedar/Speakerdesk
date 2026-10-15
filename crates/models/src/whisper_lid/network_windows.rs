//! Request-local Windows Whisper execution; numerical admission remains blocked.
use super::{LANGUAGE_COUNT, parse_language_tokens};
use crate::{
    operation::Operation,
    windows_runtime::{Backend, ModelAdmission, error},
};
use ort::{session::Session, value::Tensor};
use serde::Deserialize;
use speakerdesk_core::{error::ChildError, ipc::LanguageProbeOut};
use std::{fs, path::Path};

pub const ARTIFACT_SHA256: &str =
    "1c47965522f7495fc85814c70c6b17dc4952436774c43225d55665e4ecbc7f14";
const CONTEXT_SAMPLES: usize = 480000;
const FRAMES: usize = 3000;
const WINDOW: usize = 400;
const BANDS: usize = 80;

pub struct Network {
    frontend: Session,
    graph: Session,
    codes: Vec<String>,
    _admission: ModelAdmission,
}

impl Network {
    pub fn load(directory: &Path, backend: &Backend) -> Result<Self, ChildError> {
        // Admission precedes metadata reads and session creation. The current
        // catalog deliberately has no admitted Windows Whisper artifact.
        let admission = crate::windows_runtime::admit_model(directory, "whisper-language")?;
        let directory = admission.directory();
        let codes = language_codes(&fs::read(directory.join("languages.json")).map_err(error)?)?;
        Ok(Self {
            frontend: backend.load(&directory.join("frontend.onnx"))?,
            graph: backend.load_unoptimized(&directory.join("whisper.onnx"))?,
            codes,
            _admission: admission,
        })
    }

    pub fn probe(
        &mut self,
        pcm: &[f32],
        operation: &Operation,
    ) -> Result<LanguageProbeOut, ChildError> {
        let windows = frame_audio(pcm, operation)?;
        let windows = Tensor::from_array(([1, FRAMES, WINDOW], windows)).map_err(error)?;
        operation.check()?;
        let result = self.frontend.run_with_options(
            ort::inputs!["audio_windows" => windows],
            operation.run_options()?,
        );
        operation.check()?;
        let mut outputs = result.map_err(error)?;
        let mel = outputs
            .remove("mel")
            .ok_or_else(|| error("Whisper frontend did not return mel features"))?;
        let (shape, values) = mel.try_extract_tensor::<f32>().map_err(error)?;
        if &shape[..] != [1, FRAMES as i64, BANDS as i64]
            || values.len() != FRAMES * BANDS
            || values.iter().any(|value| !value.is_finite())
        {
            return Err(error("Whisper frontend returned incompatible mel features"));
        }
        // Keep the owned OrtValue between graphs; no acoustic feature copy or
        // state survives this request.
        drop(outputs);
        operation.check()?;
        let result = self
            .graph
            .run_with_options(ort::inputs!["mel" => mel], operation.run_options()?);
        operation.check()?;
        let outputs = result.map_err(error)?;
        let probabilities = outputs
            .get("probabilities")
            .ok_or_else(|| error("Whisper graph did not return probabilities"))?;
        let (shape, data) = probabilities.try_extract_tensor::<f32>().map_err(error)?;
        let probabilities = validated_probabilities(&shape[..], data)?;
        operation.check()?;
        Ok(LanguageProbeOut {
            probabilities,
            codes: self.codes.clone(),
        })
    }
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct LanguageMetadata {
    codes: Vec<String>,
    token_ids: Vec<u32>,
    sot: i64,
}

fn language_codes(bytes: &[u8]) -> Result<Vec<String>, ChildError> {
    let metadata: LanguageMetadata = serde_json::from_slice(bytes).map_err(error)?;
    if metadata.codes.len() != LANGUAGE_COUNT || metadata.token_ids.len() != LANGUAGE_COUNT {
        return Err(error("Whisper requires exactly 99 language columns"));
    }
    let mut entries = metadata
        .codes
        .iter()
        .zip(&metadata.token_ids)
        .map(|(code, &id)| (format!("<|{code}|>"), i64::from(id)))
        .collect::<Vec<_>>();
    entries.push(("<|startoftranscript|>".into(), metadata.sot));
    let pairs = entries
        .iter()
        .map(|(token, id)| (token.as_str(), *id))
        .collect::<Vec<_>>();
    // languages.json is part of the admitted graph inventory. Its order is
    // the exported graph's column order, not token-id order or an extra sort.
    Ok(parse_language_tokens(&pairs, 51865).map_err(error)?.codes)
}

fn frame_audio(pcm: &[f32], operation: &Operation) -> Result<Vec<f32>, ChildError> {
    operation.check()?;
    if pcm.is_empty() || pcm.iter().any(|value| !value.is_finite()) {
        return Err(error("Whisper requires finite nonempty 16 kHz mono audio"));
    }
    let mut windows = Vec::with_capacity(FRAMES * WINDOW);
    for frame in 0..FRAMES {
        operation.check()?;
        for offset in 0..WINDOW {
            let position = (frame * 160 + offset) as isize - 200;
            let position = if position < 0 {
                -position
            } else if position >= CONTEXT_SAMPLES as isize {
                2 * CONTEXT_SAMPLES as isize - 2 - position
            } else {
                position
            } as usize;
            windows.push(pcm.get(position).copied().unwrap_or(0.));
        }
    }
    operation.check()?;
    Ok(windows)
}

fn validated_probabilities(shape: &[i64], data: &[f32]) -> Result<Vec<f32>, ChildError> {
    if shape != [LANGUAGE_COUNT as i64]
        || data.len() != LANGUAGE_COUNT
        || data
            .iter()
            .any(|value| !value.is_finite() || !(0.0..=1.0).contains(value))
        || (data.iter().map(|&value| f64::from(value)).sum::<f64>() - 1.).abs() > 1e-5
    {
        return Err(error(
            "Whisper returned an incompatible language probability vector",
        ));
    }
    Ok(data.to_vec())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn graph_metadata_preserves_column_order_instead_of_sorting_token_ids() {
        let mut codes = vec!["fr".to_string(), "en".to_string()];
        for first in b'a'..=b'z' {
            for second in b'a'..=b'z' {
                let code = format!("{}{}", first as char, second as char);
                if code != "en" && code != "fr" && codes.len() < LANGUAGE_COUNT {
                    codes.push(code);
                }
            }
        }
        let mut metadata = serde_json::json!({
            "codes": codes.clone(),
            "token_ids": (50259..50358_u32).rev().collect::<Vec<_>>(),
            "sot": 50258,
        });
        assert_eq!(
            language_codes(&serde_json::to_vec(&metadata).unwrap()).unwrap(),
            codes
        );
        metadata["token_ids"].as_array_mut().unwrap().pop();
        assert!(language_codes(&serde_json::to_vec(&metadata).unwrap()).is_err());
    }

    #[test]
    fn framing_preserves_centered_reflection_context_limit_and_final_frame() {
        let operation = Operation::new();
        let pcm = (0..CONTEXT_SAMPLES + 400)
            .map(|index| index as f32)
            .collect::<Vec<_>>();
        let windows = frame_audio(&pcm, &operation).unwrap();
        assert_eq!(windows.len(), FRAMES * WINDOW);
        assert_eq!(&windows[..3], &[200., 199., 198.]);
        assert_eq!(&windows[198..203], &[2., 1., 0., 1., 2.]);
        let last = &windows[(FRAMES - 1) * WINDOW..];
        assert_eq!(last[0], 479640.);
        assert_eq!(&last[359..362], &[479999., 479998., 479997.]);
        assert_eq!(last[399], 479959.);
        assert_eq!(
            windows,
            frame_audio(&pcm[..CONTEXT_SAMPLES], &operation).unwrap()
        );
    }

    #[test]
    fn short_silence_and_quiet_crops_are_isolated_and_invalid_or_cancelled_input_fails() {
        let operation = Operation::new();
        let short = frame_audio(&[1., 2., 3.], &operation).unwrap();
        assert_eq!(&short[198..205], &[3., 2., 1., 2., 3., 0., 0.]);
        assert!(short[2 * WINDOW..].iter().all(|&value| value == 0.));
        assert!(
            frame_audio(&[0.], &operation)
                .unwrap()
                .iter()
                .all(|&value| value == 0.)
        );
        assert!(
            frame_audio(&[1e-30, -1e-30], &operation)
                .unwrap()
                .iter()
                .all(|value| value.is_finite())
        );
        assert_eq!(frame_audio(&[1., 2., 3.], &operation).unwrap(), short);
        assert!(frame_audio(&[], &operation).is_err());
        assert!(frame_audio(&[f32::NAN], &operation).is_err());
        let mut beyond_context = vec![0.; CONTEXT_SAMPLES + 1];
        beyond_context[CONTEXT_SAMPLES] = f32::INFINITY;
        assert!(frame_audio(&beyond_context, &operation).is_err());
        let cancelled = Operation::new();
        cancelled.cancel();
        assert!(matches!(
            frame_audio(&[0.], &cancelled),
            Err(ChildError::Cancelled)
        ));
    }

    #[test]
    fn malformed_score_vectors_do_not_commit_a_language_observation() {
        let valid = vec![1. / LANGUAGE_COUNT as f32; LANGUAGE_COUNT];
        assert_eq!(validated_probabilities(&[99], &valid).unwrap(), valid);
        for (shape, data) in [
            (vec![1, 99], valid.clone()),
            (vec![99], vec![0.; 99]),
            (vec![99], vec![f32::NAN; 99]),
            (vec![99], vec![-0.01; 99]),
            (vec![99], vec![1.01; 99]),
            (vec![99], vec![1. / 99.; 98]),
        ] {
            assert!(validated_probabilities(&shape, &data).is_err());
        }
    }
}
