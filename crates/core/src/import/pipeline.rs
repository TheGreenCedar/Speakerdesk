//! Bounded source reads and durable, passage-local import publication.
use super::*;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

const READ_SAMPLES: usize = 163_840;
const DIAR_READ_SAMPLES: usize = 160_000;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ImportPlan {
    pub version: u32,
    pub samples: u32,
    pub language: String,
    pub chunks: Vec<DecodeRegion>,
    pub windows: Vec<ImportWindow>,
    pub diarization: Vec<Value>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ImportWindow {
    pub chunk: usize,
    pub start_sample: i64,
    pub end_sample: i64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ImportResume {
    pub next_work: usize,
    pub plan_sha256: String,
    pub plan: ImportPlan,
    pub document: Document,
}

/// The host persists accepted rows and the cursor in one transaction. Returning
/// the current batch's stored rows makes protected edits the language authority.
pub struct ImportProgress<'a> {
    pub next_work: usize,
    pub total_work: usize,
    pub plan_sha256: &'a str,
    pub plan: &'a ImportPlan,
    pub rows: &'a [Segment],
    pub speakers: &'a BTreeMap<String, String>,
    pub provenance: &'a Value,
    pub warnings: &'a [String],
    pub diarization: &'a [Value],
}

pub fn import_reader<M: Models + ?Sized>(
    models: &M,
    reader: &audio::RecordingReader,
    language: &str,
    alignment: Option<&crate::alignment::Vocabulary>,
    resume: Option<&ImportResume>,
    publish: &mut dyn FnMut(ImportProgress<'_>) -> Result<Vec<Segment>, ImportError>,
) -> Result<Document, ImportError> {
    let samples = reader.import_samples().map_err(audio_error)?;
    run(
        models,
        &ReaderAudio { reader, samples },
        language,
        alignment,
        resume,
        publish,
    )
}

pub(super) trait ImportAudio {
    fn samples(&self) -> usize;
    fn read(&self, start: usize, end: usize) -> Result<Vec<f32>, ImportError>;
    fn diarize<M: Models + ?Sized>(&self, models: &M) -> Result<Vec<Turn>, ImportError>;
}

pub(super) struct SliceAudio<'a>(pub &'a [f32]);
impl ImportAudio for SliceAudio<'_> {
    fn samples(&self) -> usize {
        self.0.len()
    }
    fn read(&self, start: usize, end: usize) -> Result<Vec<f32>, ImportError> {
        Ok(self.0[start..end].to_vec())
    }
    fn diarize<M: Models + ?Sized>(&self, models: &M) -> Result<Vec<Turn>, ImportError> {
        models.diarize_window(self.0).map_err(child_err)
    }
}

struct ReaderAudio<'a> {
    reader: &'a audio::RecordingReader,
    samples: u32,
}
impl ImportAudio for ReaderAudio<'_> {
    fn samples(&self) -> usize {
        self.samples as usize
    }
    fn read(&self, start: usize, end: usize) -> Result<Vec<f32>, ImportError> {
        self.reader
            .read_samples(start as u32, end as u32, self.samples)
            .map_err(audio_error)
    }
    fn diarize<M: Models + ?Sized>(&self, models: &M) -> Result<Vec<Turn>, ImportError> {
        let handle = models
            .diar_open(crate::ipc::DiarPreset::Offline)
            .map_err(child_err)?;
        let result = (|| {
            let mut turns = Vec::new();
            let mut received = 0;
            let mut observed = 0;
            for start in (0..self.samples()).step_by(DIAR_READ_SAMPLES) {
                let end = (start + DIAR_READ_SAMPLES).min(self.samples());
                let pcm = self.read(start, end)?;
                received = end as u64;
                let feed = models.diar_feed(handle, &pcm, false).map_err(child_err)?;
                feed.validate(received, observed).map_err(child_err)?;
                observed = feed.observed;
                turns.extend(feed.turns.into_iter().map(|turn| Turn {
                    start: samples_to_seconds(turn.start as i64),
                    end: samples_to_seconds(turn.end as i64),
                    speaker: turn.speaker,
                }));
            }
            let feed = models.diar_feed(handle, &[], true).map_err(child_err)?;
            feed.validate(received, observed).map_err(child_err)?;
            turns.extend(feed.turns.into_iter().map(|turn| Turn {
                start: samples_to_seconds(turn.start as i64),
                end: samples_to_seconds(turn.end as i64),
                speaker: turn.speaker,
            }));
            if feed.observed != received / 160 * 160 {
                return Err(ImportError::new(messages::PROCESSING_FAILED));
            }
            Ok(turns)
        })();
        let closed = models.diar_close(handle).map_err(child_err);
        result.and_then(|turns| closed.map(|()| turns))
    }
}

fn audio_error(error: audio::AudioError) -> ImportError {
    ImportError::new(error.to_string())
}

pub(super) fn run<M: Models + ?Sized, A: ImportAudio>(
    models: &M,
    source: &A,
    language: &str,
    alignment: Option<&crate::alignment::Vocabulary>,
    resume: Option<&ImportResume>,
    publish: &mut dyn FnMut(ImportProgress<'_>) -> Result<Vec<Segment>, ImportError>,
) -> Result<Document, ImportError> {
    let choice = LanguageChoice::parse(language).map_err(language_err)?;
    let duration = samples_to_seconds(source.samples() as i64);
    if source.samples() == 0 || duration > audio::MAX_DURATION_SECONDS {
        return Err(ImportError::new(messages::RECORDING_LENGTH));
    }
    if let Some(resume) = resume {
        validate_plan(&resume.plan)?;
        if resume.plan_sha256 != plan_digest(&resume.plan)?
            || resume.next_work > resume.plan.windows.len()
        {
            return Err(invalid_plan());
        }
    }
    let turns = if let Some(resume) = resume {
        if resume.plan.samples as usize != source.samples() || resume.plan.language != language {
            return Err(ImportError::new(
                "Import source or captured language changed; accepted words are retained.",
            ));
        }
        resume
            .plan
            .diarization
            .iter()
            .map(|turn| {
                Ok(Turn {
                    start: turn["start"].as_f64().ok_or_else(invalid_plan)?,
                    end: turn["end"].as_f64().ok_or_else(invalid_plan)?,
                    speaker: turn["speaker"].as_str().ok_or_else(invalid_plan)?.into(),
                })
            })
            .collect::<Result<Vec<_>, ImportError>>()?
    } else {
        source.diarize(models)?
    };
    let turns = normalize_turns(turns, duration, limits::SPEAKER_TRACK_LIMIT)?;
    let ledger = admit_bounded(models, source)?;
    let plan = if let Some(resume) = resume {
        resume.plan.clone()
    } else {
        let mut chunks = decode_regions(&turns, Some(IMPORT_MAX_SECONDS), Some((0., duration)));
        crop_context(
            &mut chunks,
            duration,
            if matches!(choice, LanguageChoice::Auto) {
                0.
            } else {
                CROP_PADDING_SECONDS
            },
        );
        if chunks.len() > MAX_CHUNKS {
            return Err(ImportError::new(messages::TOO_MANY_TRANSITIONS));
        }
        let book = canonical_utterances(&ledger, source.samples() as i64)?;
        let chunks = utterance_tail_chunks(
            utterance_prefix_chunks(chunks, &book, source.samples() as i64),
            &book,
            source.samples() as i64,
        );
        let spans: Vec<_> = book.speech_regions().map(|(span, _)| span).collect();
        let mut windows = Vec::new();
        for (index, chunk) in chunks.iter().enumerate() {
            let start = seconds_to_sample(chunk.audio_start, source.samples() as i64);
            let end = seconds_to_sample(chunk.audio_end, source.samples() as i64);
            let mut planned = Vec::new();
            for window in plan_windows(&spans, start, end) {
                let pcm = source.read(window.start as usize, window.end as usize)?;
                let admission = ledger
                    .admission(window.start, window.end)
                    .map_err(speech_err)?;
                let blocked =
                    speech::blocked_route(admission.decision, &pcm, window.start, window.end);
                planned.push((window, blocked.is_some()));
            }
            if chunk.speakers.is_empty() && planned.iter().all(|(_, blocked)| *blocked) {
                continue;
            }
            windows.extend(planned.into_iter().map(|(window, _)| ImportWindow {
                chunk: index,
                start_sample: window.start,
                end_sample: window.end,
            }));
        }
        ImportPlan {
            version: 1,
            samples: source.samples() as u32,
            language: language.into(),
            chunks,
            windows,
            diarization: turns
                .iter()
                .map(|t| json!({"start":t.start,"end":t.end,"speaker":t.speaker}))
                .collect(),
        }
    };
    validate_plan(&plan)?;
    let plan_sha256 = plan_digest(&plan)?;
    let next = resume.map_or(0, |resume| resume.next_work);
    if resume.is_some_and(|r| r.plan_sha256 != plan_sha256 || next > plan.windows.len()) {
        return Err(invalid_plan());
    }
    let mut document = resume
        .map(|r| r.document.clone())
        .unwrap_or_else(|| Document {
            schema_version: 1,
            speakers: speaker_map(&turns),
            segments: Vec::new(),
            provenance: Some(provenance(language, limits::SPEAKER_TRACK_LIMIT)),
            warnings: warnings(language, limits::SPEAKER_TRACK_LIMIT),
            diarization: plan.diarization.clone(),
            edited_at: None,
            rest: Map::new(),
        });
    let mut summary = crate::live::MainLanguageSummary::from_segments(&document.segments);
    publish_batch(publish, &plan, &plan_sha256, next, &[], &document)?;
    for (index, window) in plan.windows.iter().enumerate().skip(next) {
        let chunk = &plan.chunks[window.chunk];
        let pcm = source.read(window.start_sample as usize, window.end_sample as usize)?;
        let admission = ledger
            .admission(window.start_sample, window.end_sample)
            .map_err(speech_err)?;
        let blocked = speech::blocked_route(
            admission.decision,
            &pcm,
            window.start_sample,
            window.end_sample,
        );
        let item = Examined {
            start: window.start_sample,
            end: window.end_sample,
            blocked,
            admission,
        };
        let target = segment_from_row(
            format!("import-{index}-0"),
            chunk.start,
            chunk.end,
            chunk_speaker(chunk),
            chunk,
            &blocked_row(
                language,
                &item,
                &BlockedSpeech {
                    reason: "planning",
                    audio_state: "planning",
                    pcm: None,
                },
            ),
            (source.samples() / 160 * 160) as i64,
        );
        let mut target = target;
        capture_language(&mut target, language);
        let context = summary.context(&document.segments, &target);
        let decoded = if let Some(blocked) = &item.blocked {
            vec![blocked_row(language, &item, blocked)]
        } else {
            decode_bounded(
                models,
                &ledger,
                &pcm,
                language,
                &chunk.speakers,
                window.start_sample,
                context,
            )?
        };
        let mut batch = Vec::new();
        for (row_index, row) in decoded.into_iter().enumerate() {
            let Some((start, end)) = clip_span(row.start, row.end, chunk.start, chunk.end)
                .and_then(|(a, b)| clip_span(a, b, 0., duration))
            else {
                continue;
            };
            let speaker = if chunk.utterance_context_boundary.is_some()
                && chunk.activity_regions.as_ref().is_some_and(|activity| {
                    activity.iter().any(|part| {
                        part.speakers.is_empty() && part.start < end && part.end > start
                    })
                }) {
                "unassigned".into()
            } else {
                chunk_speaker(chunk)
            };
            ensure_speaker(&mut document.speakers, &speaker);
            let mut segment = segment_from_row(
                format!("import-{index}-{row_index}"),
                start,
                end,
                speaker,
                chunk,
                &row,
                (source.samples() / 160 * 160) as i64,
            );
            capture_language(&mut segment, language);
            if let Some(vocabulary) = alignment
                && row.audio_state.is_none()
                && !segment.text.trim().is_empty()
                && let Some(detected) = &segment.language
                && languages::is_supported(detected)
            {
                let a = seconds_to_sample(start, source.samples() as i64);
                let b = seconds_to_sample(end, source.samples() as i64);
                let crop = source.read(a as usize, b as usize)?;
                match vocabulary.align(models, &crop, &segment.text, a, detected) {
                    Ok(result) => crate::alignment::attach(&mut segment, result),
                    Err(error) if lifecycle(&error) => return Err(child_err(error)),
                    Err(_) => {
                        segment.rest.insert("reading_alignment".into(),json!({"status":"unavailable","reason":"model_error","language":detected}));
                    }
                }
            }
            batch.push(segment);
        }
        let accepted = publish_batch(publish, &plan, &plan_sha256, index + 1, &batch, &document)?;
        for row in accepted {
            if let Some(position) = document
                .segments
                .iter()
                .position(|prior| prior.id == row.id)
            {
                summary.accept_publication(Some(&document.segments[position]), &row);
                document.segments[position] = row;
            } else {
                summary.accept_publication(None, &row);
                document.segments.push(row);
            }
        }
    }
    document.segments.sort_by(|a, b| {
        a.start
            .total_cmp(&b.start)
            .then(a.end.total_cmp(&b.end))
            .then(a.id.cmp(&b.id))
    });
    let existing: std::collections::BTreeSet<_> =
        document.segments.iter().map(|s| s.id.clone()).collect();
    review::retain_unassigned_audio(&mut document, duration);
    let coverage: Vec<_> = document
        .segments
        .iter()
        .filter(|s| !existing.contains(&s.id))
        .cloned()
        .collect();
    let accepted = publish_batch(
        publish,
        &plan,
        &plan_sha256,
        plan.windows.len(),
        &coverage,
        &document,
    )?;
    // The final coverage batch was created locally before publication.
    document.segments.retain(|s| existing.contains(&s.id));
    document.segments.extend(accepted);
    if document
        .segments
        .iter()
        .any(|s| s.rest.contains_key("reading_alignment"))
    {
        document.provenance.as_mut().expect("import provenance")["timing"] =
            json!("Supplied-text CTC timing where supported; speech crop boundaries otherwise");
        document.warnings[0]="Word timing is model-estimated. Uncertain or edited passages use speech crop boundaries; review subtitle timing.".into();
    }
    let value = serde_json::to_value(&document)
        .map_err(|_| ImportError::new(messages::PROCESSING_FAILED))?;
    let document = transcript::validate(&value, duration)?;
    export::export(&document, "txt").map_err(|_| ImportError::new(messages::EXPORT_TOO_LARGE))?;
    Ok(document)
}

fn publish_batch(
    publish: &mut dyn FnMut(ImportProgress<'_>) -> Result<Vec<Segment>, ImportError>,
    plan: &ImportPlan,
    hash: &str,
    next: usize,
    rows: &[Segment],
    document: &Document,
) -> Result<Vec<Segment>, ImportError> {
    publish(ImportProgress {
        next_work: next,
        total_work: plan.windows.len(),
        plan_sha256: hash,
        plan,
        rows,
        speakers: &document.speakers,
        provenance: document.provenance.as_ref().ok_or_else(invalid_plan)?,
        warnings: &document.warnings,
        diarization: &document.diarization,
    })
}

fn invalid_plan() -> ImportError {
    ImportError::new(
        "Import checkpoint no longer matches its source plan; accepted words are retained.",
    )
}
fn plan_digest(plan: &ImportPlan) -> Result<String, ImportError> {
    Ok(hex::encode(Sha256::digest(
        serde_json::to_vec(plan).map_err(|_| invalid_plan())?,
    )))
}
fn validate_plan(plan: &ImportPlan) -> Result<(), ImportError> {
    let duration = samples_to_seconds(i64::from(plan.samples));
    if plan.version != 1
        || plan.chunks.len() > MAX_CHUNKS
        || plan.windows.len() > MAX_CHUNKS * 100
        || plan.samples == 0
        || duration > audio::MAX_DURATION_SECONDS
    {
        return Err(invalid_plan());
    }
    for chunk in &plan.chunks {
        if !chunk.start.is_finite()
            || !chunk.end.is_finite()
            || !chunk.audio_start.is_finite()
            || !chunk.audio_end.is_finite()
            || chunk.start < 0.
            || chunk.end <= chunk.start
            || chunk.end > duration
            || chunk.audio_start < 0.
            || chunk.audio_end > duration
            || chunk.audio_end <= chunk.audio_start
            || chunk.audio_end - chunk.audio_start > limits::PASSAGE_RETRY_MAX_SECONDS
        {
            return Err(invalid_plan());
        }
    }
    for window in &plan.windows {
        if window.chunk >= plan.chunks.len()
            || window.start_sample < 0
            || window.end_sample <= window.start_sample
            || window.end_sample > i64::from(plan.samples)
            || window.end_sample - window.start_sample
                > (limits::PASSAGE_RETRY_MAX_SECONDS * 16000.) as i64
        {
            return Err(invalid_plan());
        }
        let chunk = &plan.chunks[window.chunk];
        if window.start_sample < seconds_to_sample(chunk.audio_start, i64::from(plan.samples))
            || window.end_sample > seconds_to_sample(chunk.audio_end, i64::from(plan.samples))
        {
            return Err(invalid_plan());
        }
    }
    Ok(())
}

struct BorrowedEvidence<'a> {
    ledger: &'a SpeechLedger,
}
impl crate::language::AcousticEvidence for BorrowedEvidence<'_> {
    fn admit(&mut self, start: i64, end: i64) -> Result<Admission, speech::SpeechError> {
        self.ledger.admission(start, end)
    }
}
fn lifecycle(error: &ChildError) -> bool {
    error.is_lifecycle_failure()
}

fn decode_bounded<M: Models + ?Sized>(
    models: &M,
    ledger: &SpeechLedger,
    pcm: &[f32],
    language: &str,
    speakers: &[String],
    origin: i64,
    context: Option<crate::language::LanguageContext>,
) -> Result<Vec<Decoded>, ImportError> {
    let failure = std::sync::Arc::new(std::sync::Mutex::new(None));
    let detector = (language == languages::AUTOMATIC).then(|| {
        Box::new(BorrowedModels::new(models, Some(failure.clone()))) as Box<dyn LanguageDetector>
    });
    let mut transcriber = SpeechTranscriber::new(
        language,
        Box::new(BorrowedEvidence { ledger }),
        detector,
        Box::new(BorrowedModels::new(models, Some(failure.clone()))),
        context,
    )
    .map_err(language_err)?;
    let decoded = transcriber
        .transcribe(
            pcm,
            limits::SAMPLE_RATE,
            speakers,
            TranscribeOptions {
                max_asr_seconds: IMPORT_MAX_SECONDS,
                start_sample: origin,
                asr_padding: (0, 0),
            },
        )
        .map_err(language_err)?;
    if let Some(error) = failure.lock().expect("child failure").take() {
        return Err(child_err(error));
    }
    Ok(decoded
        .into_iter()
        .map(|passage| passage_row(origin, passage))
        .collect())
}

fn admit_bounded<M: Models + ?Sized, A: ImportAudio>(
    models: &M,
    source: &A,
) -> Result<SpeechLedger, ImportError> {
    let mut ledger = SpeechLedger::new(0).map_err(speech_err)?;
    let handle = models.vad_open(None).map_err(child_err)?;
    let result = (|| {
        for block_start in (0..source.samples()).step_by(READ_SAMPLES) {
            let block_end = (block_start + READ_SAMPLES).min(source.samples());
            let block = source.read(block_start, block_end)?;
            for offset in (block_start..block_end).step_by(speech::FRAME_SAMPLES) {
                let end = (offset + speech::FRAME_SAMPLES).min(block_end);
                let feed = models
                    .vad_feed(
                        handle,
                        &block[offset - block_start..end - block_start],
                        end == source.samples(),
                    )
                    .map_err(child_err)?;
                if feed.received != end as u64 || feed.observed > feed.received {
                    return Err(ImportError::new(messages::PROCESSING_FAILED));
                }
                for frame in feed.frames {
                    if frame.start >= frame.end || frame.end > feed.observed {
                        return Err(ImportError::new(messages::PROCESSING_FAILED));
                    }
                    if frame.start >= block_start as u64 && frame.end <= block_end as u64 {
                        ledger
                            .append_pcm(
                                frame.start as i64,
                                &block[frame.start as usize - block_start
                                    ..frame.end as usize - block_start],
                                f64::from(frame.raw_p),
                                f64::from(frame.norm_p),
                            )
                            .map_err(speech_err)?;
                    } else {
                        let pcm = source.read(frame.start as usize, frame.end as usize)?;
                        ledger
                            .append_pcm(
                                frame.start as i64,
                                &pcm,
                                f64::from(frame.raw_p),
                                f64::from(frame.norm_p),
                            )
                            .map_err(speech_err)?;
                    }
                }
                if ledger.end_sample() != feed.observed as i64 {
                    return Err(ImportError::new(messages::PROCESSING_FAILED));
                }
            }
        }
        if ledger.end_sample() != source.samples() as i64 {
            return Err(ImportError::new(messages::PROCESSING_FAILED));
        }
        Ok(ledger)
    })();
    let closed = models.vad_close(handle).map_err(child_err);
    result.and_then(|ledger| closed.map(|()| ledger))
}
