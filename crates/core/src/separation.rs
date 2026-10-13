//! Saved-passage candidates. Separation estimates audio; it never identifies a person.
use std::collections::BTreeSet;

use serde::{Deserialize, Serialize};
use serde_json::{Value, json};

use crate::{
    document::{Job, Segment},
    error::ApiError,
    live,
    memo::sha256_hex,
};

pub const MODEL: &str = "mossformer2-16k@407cb030cd66340918ebb6c8cc63b18f8592cdbe";
pub const MAX_SAMPLES: usize = (crate::limits::PASSAGE_RETRY_MAX_SECONDS * 16000.) as usize;

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Voice {
    pub episode_id: String,
    pub original_id: String,
    pub channel: u8,
    pub origin_sample: u64,
    pub length_samples: u64,
    pub model: String,
}

impl Voice {
    pub fn label(&self) -> &'static str {
        if self.channel == 0 {
            "Voice A"
        } else {
            "Voice B"
        }
    }
    pub fn contains(&self, segment: &Segment) -> bool {
        let Some(end_sample) = self.origin_sample.checked_add(self.length_samples) else {
            return false;
        };
        let start = (segment.start * 16000.).round_ties_even();
        let end = (segment.end * 16000.).round_ties_even();
        start >= self.origin_sample as f64 && end <= end_sample as f64 && end > start
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CandidateVoice {
    pub text: String,
    pub language: String,
    pub review: bool,
    pub transcription_review: Option<Value>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Candidate {
    pub id: String,
    pub source_key: String,
    pub original: Box<Segment>,
    pub voices: [CandidateVoice; 2],
    pub model: String,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct History {
    pub original: Box<Segment>,
    pub accepted: [Segment; 2],
}

pub fn sample_bounds(segment: &Segment) -> Result<(u64, u64), ApiError> {
    if !segment.start.is_finite()
        || !segment.end.is_finite()
        || segment.start < 0.
        || segment.end <= segment.start
        || segment.end - segment.start > crate::limits::PASSAGE_RETRY_MAX_SECONDS
    {
        return Err(ApiError::new(
            400,
            "Separate a passage within the supported recording window.",
        ));
    }
    let start = (segment.start * 16000.).round_ties_even() as u64;
    let end = (segment.end * 16000.).round_ties_even() as u64;
    if end.saturating_sub(start) < 16 || end - start > MAX_SAMPLES as u64 {
        return Err(ApiError::new(
            400,
            "The selected audio window is unavailable.",
        ));
    }
    Ok((start, end))
}

/// A union of track labels cannot establish simultaneous speech.
pub fn eligible(job: &Job, segment: &Segment) -> Result<(), ApiError> {
    if job.status != "ready" {
        return Err(ApiError::new(
            409,
            "Separate an original passage after the recording is ready.",
        ));
    }
    eligible_source(job, segment)
}

/// Capture's serial owner may process a completed original passage under its
/// existing operation. Manual saved-passage routes keep their ready-only guard.
pub fn eligible_live(job: &Job, segment: &Segment) -> Result<(), ApiError> {
    if !matches!(
        job.status.as_str(),
        "recording" | "paused" | "finishing" | "ready"
    ) || segment.canonical_utterance_id.is_none()
        || segment.rest.get("canonical_state").and_then(Value::as_str) != Some("sealed")
        || segment
            .protected_fields
            .as_ref()
            .is_some_and(|fields| !fields.is_empty())
    {
        return Err(ApiError::new(
            409,
            "Live separation needs a completed, unedited passage.",
        ));
    }
    eligible_source(job, segment)
}

fn eligible_source(job: &Job, segment: &Segment) -> Result<(), ApiError> {
    if segment.separation.is_some() {
        return Err(ApiError::new(409, "Separate an original passage."));
    }
    let (start, end) = sample_bounds(segment)?;
    let activity = segment
        .speaker_activity
        .as_ref()
        .and_then(|v| serde_json::from_value::<crate::voice::SpeakerActivity>(v.clone()).ok())
        .ok_or_else(|| ApiError::new(409, "This passage has no observed two-voice overlap."))?;
    if Some(activity.audio_revision) != segment.audio_revision
        || activity.observed_end_sample < end as i64
    {
        return Err(ApiError::new(
            409,
            "The passage's audio observations are stale.",
        ));
    }
    let mut tracks = BTreeSet::new();
    let mut overlap = false;
    for region in activity.regions {
        if region.end_sample <= start as i64 || region.start_sample >= end as i64 {
            continue;
        }
        if region.start_sample < 0 || region.end_sample <= region.start_sample {
            return Err(ApiError::new(
                409,
                "The passage's audio observations are invalid.",
            ));
        }
        let region_tracks: BTreeSet<_> = region.speakers.iter().collect();
        overlap |= region_tracks.len() == 2;
        tracks.extend(region.speakers);
    }
    if tracks.len() != 2 || !overlap {
        return Err(ApiError::new(
            409,
            "Separation needs an observed overlap between exactly two voices.",
        ));
    }
    if job
        .extra
        .get("language_history")
        .and_then(Value::as_array)
        .is_some_and(|history| {
            history.iter().any(|epoch| {
                epoch["start_sample"]
                    .as_u64()
                    .is_some_and(|at| at > start && at < end)
            })
        })
    {
        return Err(ApiError::new(
            409,
            "Choose a passage inside one language setting.",
        ));
    }
    Ok(())
}

/// Only source and language epochs covering this passage affect its result.
pub fn source_key(job: &Job, segment: &Segment) -> String {
    let end = (segment.end * 16000.).round_ties_even() as u64;
    let epochs: Vec<_> = job
        .extra
        .get("language_history")
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
        .filter(|epoch| epoch["start_sample"].as_u64().unwrap_or(0) < end)
        .collect();
    let source = json!({"job":job.id,"source_file":job.source_file,
        "legacy_media_root":job.extra.get("legacy_media_root"),"audio_revision":job.extra.get("audio_revision"),
        "capture_source_catalog":job.extra.get("capture_source_catalog"),"sources":job.extra.get("sources"),
        "language":if epochs.is_empty(){Some(job.language.as_str())}else{None},"epochs":epochs,
        "passage":live::segment_version(segment),"activity":segment.speaker_activity,"separation":segment.separation});
    sha256_hex(&serde_json::to_vec(&source).expect("source serializes"))
}

pub fn fresh(job: &Job, candidate: &Candidate) -> bool {
    job.status == "ready"
        && job
            .document
            .as_ref()
            .and_then(|d| d.segments.iter().find(|s| s.id == candidate.original.id))
            .is_some_and(|segment| source_key(job, segment) == candidate.source_key)
}

pub fn accept(
    job: &mut Job,
    passage_id: &str,
    candidate_id: &str,
) -> Result<[Segment; 2], ApiError> {
    accept_current(job, passage_id, candidate_id, false)
}

pub fn accept_live(
    job: &mut Job,
    passage_id: &str,
    candidate_id: &str,
) -> Result<[Segment; 2], ApiError> {
    accept_current(job, passage_id, candidate_id, true)
}

fn accept_current(
    job: &mut Job,
    passage_id: &str,
    candidate_id: &str,
    live: bool,
) -> Result<[Segment; 2], ApiError> {
    let candidate = job
        .document
        .as_ref()
        .and_then(|d| d.segments.iter().find(|s| s.id == passage_id))
        .and_then(|s| s.separation_candidate.as_deref())
        .filter(|c| c.id == candidate_id)
        .cloned()
        .ok_or_else(|| ApiError::new(409, "This separation candidate is no longer available."))?;
    let current = if live {
        job.document
            .as_ref()
            .and_then(|d| d.segments.iter().find(|s| s.id == passage_id))
            .is_some_and(|s| {
                eligible_live(job, s).is_ok() && source_key(job, s) == candidate.source_key
            })
    } else {
        fresh(job, &candidate)
    };
    if !current {
        return Err(ApiError::new(
            409,
            "The passage changed. Separate its current words and audio again.",
        ));
    }
    if candidate.voices.iter().any(|voice| {
        voice.text.trim().is_empty() || !crate::languages::is_supported(&voice.language)
    }) {
        return Err(ApiError::new(
            409,
            "Both voice estimates need supported, nonempty words before replacing the original.",
        ));
    }
    let (origin, end) = sample_bounds(&candidate.original)?;
    let pair: [Segment; 2] = std::array::from_fn(|channel| {
        let mut row = (*candidate.original).clone();
        row.id = format!("{}-{}", candidate.id, channel);
        row.start = origin as f64 / 16000.;
        row.end = end as f64 / 16000.;
        row.text = candidate.voices[channel].text.clone();
        row.speaker = "unassigned".into();
        row.language = Some(candidate.voices[channel].language.clone());
        row.language_detection = None;
        row.transcription_review = candidate.voices[channel].transcription_review.clone();
        row.review = Some(true);
        row.review_resolution = Some("separated_estimate".into());
        row.protected_fields = None;
        row.machine_revision = Some(0);
        row.refinement_state = Some("separated".into());
        row.canonical_utterance_id = None;
        row.speaker_activity = None;
        row.capture_source = None;
        row.acoustic_evidence = None;
        row.retry_candidate = None;
        row.separation_candidate = None;
        crate::edit::clear_alignment_bundle(&mut row);
        row.rest.clear();
        row.rest.insert("voice_eligible".into(), json!(false));
        row.separation = Some(Voice {
            episode_id: candidate.id.clone(),
            original_id: passage_id.into(),
            channel: channel as u8,
            origin_sample: origin,
            length_samples: end - origin,
            model: candidate.model.clone(),
        });
        row
    });
    let document = job.document.as_mut().expect("candidate document");
    document
        .speakers
        .entry("unassigned".into())
        .or_insert_with(|| "Unassigned".into());
    let index = document
        .segments
        .iter()
        .position(|s| s.id == passage_id)
        .expect("candidate passage");
    document.segments.splice(index..=index, pair.clone());
    let history = History {
        original: candidate.original,
        accepted: pair.clone(),
    };
    job.extra
        .entry("separation_history")
        .or_insert_with(|| json!({}))[&candidate.id] =
        serde_json::to_value(history).expect("history serializes");
    Ok(pair)
}

pub fn restore(job: &mut Job, episode_id: &str) -> Result<(), ApiError> {
    let history: History = job
        .extra
        .get("separation_history")
        .and_then(|h| h.get(episode_id))
        .and_then(|h| serde_json::from_value(h.clone()).ok())
        .ok_or_else(|| ApiError::new(404, "The original passage is unavailable."))?;
    let document = job
        .document
        .as_mut()
        .ok_or_else(|| ApiError::new(404, "Transcript unavailable."))?;
    let positions: Vec<_> = document
        .segments
        .iter()
        .enumerate()
        .filter(|(_, s)| {
            s.separation
                .as_ref()
                .is_some_and(|v| v.episode_id == episode_id)
        })
        .map(|(i, _)| i)
        .collect();
    if positions.len() != 2
        || positions[1] != positions[0] + 1
        || history
            .accepted
            .iter()
            .zip(&positions)
            .any(|(prior, index)| {
                live::segment_version(prior) != live::segment_version(&document.segments[*index])
            })
    {
        return Err(ApiError::new(
            409,
            "A separated passage was edited. Copy from the original instead of replacing those edits.",
        ));
    }
    document
        .segments
        .splice(positions[0]..=positions[1], [*history.original]);
    Ok(())
}

pub fn validate_bounds(segment: &Segment) -> Result<(), ApiError> {
    if segment
        .separation
        .as_ref()
        .is_some_and(|voice| !voice.contains(segment))
    {
        return Err(ApiError::new(
            400,
            "A separated voice must stay inside its original passage.",
        ));
    }
    Ok(())
}

/// Quantize once before speech admission so saved playback and retry use the same audio.
pub fn run<M: crate::models::Models + ?Sized>(
    models: &M,
    pcm: &[f32],
    language: &str,
) -> Result<(crate::ipc::SeparatedAudio, [CandidateVoice; 2]), crate::error::ChildError> {
    let audio = prepare_audio(models.separate(pcm)?, pcm.len())?;
    let candidates = [
        decode_voice(models, &audio.channels[0], language)?,
        decode_voice(models, &audio.channels[1], language)?,
    ];
    Ok((audio, candidates))
}

/// The same quantization is used by saved work and the capture continuation.
pub fn prepare_audio(
    mut audio: crate::ipc::SeparatedAudio,
    samples: usize,
) -> Result<crate::ipc::SeparatedAudio, crate::error::ChildError> {
    audio.validate(samples)?;
    let peak = audio
        .channels
        .iter()
        .flatten()
        .map(|s| s.abs())
        .fold(1_f32, f32::max);
    for channel in &mut audio.channels {
        for sample in channel.iter_mut() {
            *sample /= peak;
        }
        let bytes = crate::audio::pcm16_bytes(channel)
            .map_err(|e| crate::error::ChildError::failed(e.to_string(), false))?;
        for (sample, bytes) in channel.iter_mut().zip(bytes.chunks_exact(2)) {
            *sample = i16::from_le_bytes([bytes[0], bytes[1]]) as f32 / 32768.;
        }
    }
    Ok(audio)
}

/// One bounded generated voice retains ordinary speech admission and Auto routing.
pub fn decode_voice<M: crate::models::Models + ?Sized>(
    models: &M,
    channel: &[f32],
    language: &str,
) -> Result<CandidateVoice, crate::error::ChildError> {
    let result = crate::import::separated_passage(models, channel, language).map_err(|e| {
        e.child
            .unwrap_or_else(|| crate::error::ChildError::failed(e.message, false))
    })?;
    Ok(CandidateVoice {
        text: result.text,
        language: result.language,
        review: result.review,
        transcription_review: result.transcription_review,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    fn job() -> Job {
        serde_json::from_value(json!({"id":"aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa","name":"mix","status":"ready","language":"en","duration":2.,"created":0.,"updated":0.,"document":{"speakers":{},"segments":[{"id":"p","start":0.,"end":2.,"speaker":"overlap","text":"mixed words","audio_revision":0,"speaker_activity":{"audio_revision":0,"observed_end_sample":32000,"regions":[{"start_sample":0,"end_sample":32000,"speakers":["speaker_0","speaker_1"]}]}}]}})).unwrap()
    }
    #[test]
    fn completed_capture_overlap_does_not_require_stop() {
        let mut job = job();
        job.status = "recording".into();
        let row = &mut job.document.as_mut().unwrap().segments[0];
        row.canonical_utterance_id = Some(row.id.clone());
        row.rest.insert("canonical_state".into(), json!("sealed"));
        assert!(
            eligible(&job, &job.document.as_ref().unwrap().segments[0]).is_err(),
            "manual routes remain saved-only"
        );
        assert!(eligible_live(&job, &job.document.as_ref().unwrap().segments[0]).is_ok());
    }
    #[test]
    fn requires_actual_concurrency_and_exactly_two_tracks() {
        let mut job = job();
        assert!(eligible(&job, &job.document.as_ref().unwrap().segments[0]).is_ok());
        for regions in [
            json!([{"start_sample":0,"end_sample":16000,"speakers":["speaker_0"]},{"start_sample":16000,"end_sample":32000,"speakers":["speaker_1"]}]),
            json!([{"start_sample":0,"end_sample":16000,"speakers":["speaker_0","speaker_1"]},{"start_sample":16000,"end_sample":32000,"speakers":["speaker_2"]}]),
        ] {
            job.document.as_mut().unwrap().segments[0]
                .speaker_activity
                .as_mut()
                .unwrap()["regions"] = regions;
            assert!(eligible(&job, &job.document.as_ref().unwrap().segments[0]).is_err());
        }
    }
    #[test]
    fn candidates_preserve_words_and_acceptance_and_restore_protect_edits() {
        let mut job = job();
        let original = job.document.as_ref().unwrap().segments[0].clone();
        let candidate = Candidate {
            id: "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb".into(),
            source_key: source_key(&job, &original),
            original: Box::new(original.clone()),
            model: MODEL.into(),
            voices: std::array::from_fn(|i| CandidateVoice {
                text: format!("voice {i}"),
                language: "en".into(),
                review: false,
                transcription_review: None,
            }),
        };
        job.document.as_mut().unwrap().segments[0].separation_candidate =
            Some(Box::new(candidate.clone()));
        assert_eq!(
            job.document.as_ref().unwrap().segments[0].text,
            "mixed words"
        );
        let serialized = serde_json::to_value(&job).unwrap();
        job = serde_json::from_value(serialized).unwrap();
        let pair = accept(&mut job, "p", &candidate.id).unwrap();
        assert_eq!(pair[0].speaker, "unassigned");
        assert_eq!(pair[1].start, pair[0].start);
        assert!(pair.iter().all(|s| s.reading_turns.is_none()));
        job.document.as_mut().unwrap().segments[1].text = "protected correction".into();
        assert!(restore(&mut job, &candidate.id).is_err());
        job.document.as_mut().unwrap().segments[1] = pair[1].clone();
        restore(&mut job, &candidate.id).unwrap();
        assert_eq!(job.document.as_ref().unwrap().segments, [original]);
    }
}
