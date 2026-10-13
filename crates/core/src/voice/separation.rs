//! Identity evidence for generated estimates uses its own measured policy.
use super::{Calibration, VoiceError, VoiceModel, normalized};

#[derive(Debug, Clone, PartialEq)]
pub struct Policy {
    pub separator_model: String,
    pub calibration: Calibration,
}

/// Existing speech admission owns these sample ranges. No peak/energy heuristic
/// turns padded tails or separators' output order into speaker evidence.
pub fn query_range(samples: usize, speech: &[(usize, usize)]) -> Option<(usize, usize)> {
    let mut spans: Vec<(usize, usize)> = Vec::new();
    for &(start, end) in speech {
        if start >= end || end > samples || spans.last().is_some_and(|span| start < span.1) {
            return None;
        }
        if let Some(last) = spans.last_mut().filter(|span| span.1 == start) {
            last.1 = end;
        } else {
            spans.push((start, end));
        }
    }
    let &(start, end) = spans
        .iter()
        .filter(|(start, end)| end - start >= 32000)
        .max_by(|a, b| (a.1 - a.0).cmp(&(b.1 - b.0)).then_with(|| b.0.cmp(&a.0)))?;
    Some((start, end.min(start.saturating_add(96000))))
}

/// Generated PCM was peak-normalized by the separator. Leave one integer PCM
/// level below clipping for its embedding view; ordinary recording checks stay.
pub fn query_pcm(pcm: &[f32]) -> Result<Vec<f32>, VoiceError> {
    if pcm.is_empty() || pcm.iter().any(|s| !s.is_finite()) {
        return Err(VoiceError::Embedding);
    }
    let peak = pcm.iter().map(|s| s.abs()).fold(0_f32, f32::max);
    let headroom = 32766_f32 / 32768.;
    let gain = if peak > headroom { headroom / peak } else { 1. };
    Ok(pcm.iter().map(|sample| sample * gain).collect())
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Target {
    Track(String),
    Person {
        person_id: String,
        profile_version: String,
    },
}

#[derive(Debug, Clone, PartialEq)]
pub struct Reference {
    pub target: Target,
    /// An already assigned track and its saved profile compete as one identity.
    pub person_id: Option<String>,
    pub centroid: Vec<f64>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Match {
    pub target: Target,
    pub score: f64,
    pub margin: f64,
}

fn identity(reference: &Reference) -> (&str, &str) {
    if let Some(person) = &reference.person_id {
        return ("person", person);
    }
    match &reference.target {
        Target::Track(track) => ("track", track),
        Target::Person { person_id, .. } => ("person", person_id),
    }
}

pub fn centroid(model: &VoiceModel, vectors: &[Vec<f64>]) -> Result<Vec<f64>, VoiceError> {
    if vectors.is_empty() {
        return Err(VoiceError::Embedding);
    }
    let mut mean = vec![0.; model.dimension];
    for vector in vectors {
        for (sum, value) in mean.iter_mut().zip(normalized(vector, model.dimension)?) {
            *sum += value / vectors.len() as f64;
        }
    }
    normalized(&mean, model.dimension)
}

/// Unknown or ambiguous channels stay independent; two estimates cannot acquire
/// the same identity merely because both contain leaked speech from that person.
pub fn propose(
    policy: &Policy,
    separator_model: &str,
    model: &VoiceModel,
    channels: &[Vec<Vec<f64>>; 2],
    references: &[Reference],
) -> Result<[Option<Match>; 2], VoiceError> {
    let mut matches = [None, None];
    if policy.separator_model != separator_model || policy.calibration.model != *model {
        return Ok(matches);
    }
    let references: Vec<_> = references
        .iter()
        .map(|reference| Ok((reference, normalized(&reference.centroid, model.dimension)?)))
        .collect::<Result<_, VoiceError>>()?;
    let mut winners = [None, None];
    for (channel, vectors) in channels.iter().enumerate() {
        // The measured generated query is one admitted speech crop. The profile
        // and source-reference centroids still require independent natural clips.
        if vectors.is_empty() {
            continue;
        }
        let vectors = vectors
            .iter()
            .map(|vector| normalized(vector, model.dimension))
            .collect::<Result<Vec<_>, _>>()?;
        let mut ranked = Vec::new();
        for (index, (_, centroid)) in references.iter().enumerate() {
            let scores: Vec<_> = vectors
                .iter()
                .map(|vector| super::dot(vector, centroid).clamp(-1., 1.))
                .collect();
            let score = scores.iter().sum::<f64>() / scores.len() as f64;
            let lowest = scores.iter().copied().reduce(f64::min).unwrap_or(0.);
            ranked.push((score, lowest, index));
        }
        ranked.sort_by(|a, b| b.0.total_cmp(&a.0));
        let Some(&(score, lowest, index)) = ranked.first() else {
            continue;
        };
        let best = references[index].0;
        let runner = ranked
            .iter()
            .find(|row| identity(references[row.2].0) != identity(best))
            .map_or(0., |row| row.0);
        let margin = score - runner;
        if lowest >= policy.calibration.threshold && margin >= policy.calibration.margin {
            winners[channel] = Some(index);
            matches[channel] = Some(Match {
                target: best.target.clone(),
                score,
                margin,
            });
        }
    }
    if let [Some(a), Some(b)] = winners
        && identity(references[a].0) == identity(references[b].0)
    {
        return Ok([None, None]);
    }
    Ok(matches)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn policy() -> Policy {
        let model = VoiceModel::new("fixture", "r1", "a".repeat(64), 3, 16000).unwrap();
        Policy {
            separator_model: "separator-fixture".into(),
            calibration: Calibration::new(
                model,
                "independent-fixture",
                0.9,
                0.1,
                12,
                12,
                0.,
                0.,
                2,
            )
            .unwrap(),
        }
    }
    fn twice(vector: &[f64]) -> Vec<Vec<f64>> {
        vec![vector.to_vec(), vector.to_vec()]
    }

    #[test]
    fn query_uses_supported_speech_not_equal_halves_or_padded_tails() {
        assert_eq!(query_range(160000, &[(16000, 80000)]), Some((16000, 80000)));
        assert_eq!(
            query_range(320000, &[(0, 32000), (64000, 224000)]),
            Some((64000, 160000))
        );
        assert_eq!(
            query_range(160000, &[(0, 20000), (20000, 64000)]),
            Some((0, 64000))
        );
        assert_eq!(query_range(160000, &[(0, 16000), (144000, 160000)]), None);
        assert_eq!(query_range(160000, &[(0, 64000), (32000, 96000)]), None);
        assert_eq!(query_range(160000, &[(0, 160001)]), None);
        assert_eq!(
            query_range(usize::MAX, &[(usize::MAX - 40000, usize::MAX)]),
            Some((usize::MAX - 40000, usize::MAX))
        );
        let samples = [0., 32767_f32 / 32768., -1., 0.125];
        let adjusted = query_pcm(&samples).unwrap();
        assert!(adjusted.iter().all(|s| s.abs() <= 32766_f32 / 32768.));
        assert_eq!(samples[1], 32767_f32 / 32768.);
        assert_eq!(
            query_pcm(&[0., 0.125, -0.125]).unwrap(),
            vec![0., 0.125, -0.125]
        );
        assert!(query_pcm(&[f32::NAN]).is_err());
    }

    #[test]
    fn evidence_resolves_swapped_channels_without_needing_two_known_people() {
        let policy = policy();
        let refs = vec![
            Reference {
                target: Target::Track("speaker_64".into()),
                person_id: None,
                centroid: vec![1., 0., 0.],
            },
            Reference {
                target: Target::Person {
                    person_id: "p".into(),
                    profile_version: "2".into(),
                },
                person_id: Some("p".into()),
                centroid: vec![0., 1., 0.],
            },
        ];
        let channels = [vec![vec![0., 1., 0.]], vec![vec![1., 0., 0.]]];
        let result = propose(
            &policy,
            &policy.separator_model,
            &policy.calibration.model,
            &channels,
            &refs,
        )
        .unwrap();
        assert_eq!(result[0].as_ref().unwrap().target, refs[1].target);
        assert_eq!(result[1].as_ref().unwrap().target, refs[0].target);
        let result = propose(
            &policy,
            &policy.separator_model,
            &policy.calibration.model,
            &[channels[0].clone(), twice(&[0., 0., 1.])],
            &refs,
        )
        .unwrap();
        assert!(result[0].is_some());
        assert!(result[1].is_none());
    }

    #[test]
    fn leakage_competitors_bad_clip_and_wrong_models_abstain() {
        let policy = policy();
        let known = Reference {
            target: Target::Track("speaker_0".into()),
            person_id: Some("p".into()),
            centroid: vec![1., 0., 0.],
        };
        let duplicate = Reference {
            target: Target::Person {
                person_id: "p".into(),
                profile_version: "1".into(),
            },
            person_id: Some("p".into()),
            centroid: vec![1., 0., 0.],
        };
        let collision = [twice(&[1., 0., 0.]), twice(&[1., 0., 0.])];
        assert_eq!(
            propose(
                &policy,
                &policy.separator_model,
                &policy.calibration.model,
                &collision,
                &[known.clone(), duplicate]
            )
            .unwrap(),
            [None, None]
        );
        let competing = Reference {
            target: Target::Track("speaker_1".into()),
            person_id: None,
            centroid: vec![0.995, 0.1, 0.],
        };
        let unknown = twice(&[0., 0., 1.]);
        assert_eq!(
            propose(
                &policy,
                &policy.separator_model,
                &policy.calibration.model,
                &[collision[0].clone(), unknown.clone()],
                &[known.clone(), competing]
            )
            .unwrap(),
            [None, None]
        );
        assert_eq!(
            propose(
                &policy,
                &policy.separator_model,
                &policy.calibration.model,
                &[vec![vec![1., 0., 0.], vec![0., 1., 0.]], unknown.clone()],
                &[known.clone()]
            )
            .unwrap(),
            [None, None]
        );
        assert_eq!(
            propose(
                &policy,
                "other-separator",
                &policy.calibration.model,
                &[collision[0].clone(), unknown],
                &[known]
            )
            .unwrap(),
            [None, None]
        );
    }
}
