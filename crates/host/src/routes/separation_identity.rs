//! Separated estimates compare with existing solo audio and consented profiles.
//! The recording owner calls one crop per step; no store lock spans inference.
use super::{ApplicationModels, CancellationToken, Path, SqliteJobStore, audio};
use serde_json::{Value, json};
use speakerdesk_core::{
    document::{Job, Segment},
    error::{ChildError, StoreError},
    ipc::SeparatedAudio,
    people::{EXPLICIT_REMEMBER_VOICE, People},
    separation,
    voice::{
        self, RecognitionMeeting, VoiceClip, VoiceProfile,
        separation::{Match, Policy, Reference, Target},
    },
};

pub(crate) struct Snapshot {
    enabled: bool,
    policy: Option<Policy>,
    profiles: Vec<VoiceProfile>,
    people: People,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::routes::AppState;
    use speakerdesk_core::store::JobStore;
    use std::sync::atomic::{AtomicUsize, Ordering};

    const EPISODE: &str = "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb";
    struct Model {
        qualified: bool,
        calls: AtomicUsize,
    }
    impl ApplicationModels for Model {
        fn infer(
            &self,
            _: &Job,
            _: &CancellationToken,
        ) -> Result<speakerdesk_core::document::Document, String> {
            unreachable!()
        }
        fn separated_voice_policy(&self) -> Option<Policy> {
            self.qualified.then(policy)
        }
        fn embed_voice(
            &self,
            pcm: &[f32],
            _: &CancellationToken,
        ) -> Result<speakerdesk_core::ipc::VoiceEmbedding, String> {
            self.calls.fetch_add(1, Ordering::SeqCst);
            let average = pcm.iter().sum::<f32>() / pcm.len() as f32;
            Ok(speakerdesk_core::ipc::VoiceEmbedding {
                model: policy().calibration.model,
                clip: voice::ClipEmbedding {
                    clean: true,
                    vector: if average < 0.2 {
                        vec![1., 0., 0.]
                    } else {
                        vec![0., 1., 0.]
                    },
                },
            })
        }
    }
    fn policy() -> Policy {
        Policy {
            separator_model: separation::MODEL.into(),
            calibration: voice::Calibration::new(
                voice::VoiceModel::new("test-model", "r1", "a".repeat(64), 3, 16000).unwrap(),
                "synthetic-separation-test",
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
    fn fixture() -> (
        tempfile::TempDir,
        AppState,
        Job,
        Segment,
        SeparatedAudio,
        Model,
    ) {
        let root = tempfile::tempdir().unwrap();
        let app = AppState::open(root.path()).unwrap();
        let job: Job = serde_json::from_value(json!({
            "id":"aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa","name":"sample","status":"ready","language":"en",
            "duration":12.,"created":0.,"updated":0.,"document":{"provenance":{"kind":"local_inference"},
                "speakers":{"speaker_0":"Ada","speaker_1":"Speaker 2"},"segments":[
                    {"id":"a1","speaker":"speaker_0","start":0.,"end":2.,"text":"first","review":false},
                    {"id":"a2","speaker":"speaker_0","start":2.,"end":4.,"text":"second","review":false},
                    {"id":"b1","speaker":"speaker_1","start":4.,"end":6.,"text":"third","review":false},
                    {"id":"b2","speaker":"speaker_1","start":6.,"end":8.,"text":"fourth","review":false},
                    {"id":"mix","speaker":"multiple_speakers","start":8.,"end":12.,"text":"mixed originals","audio_revision":0,
                        "speaker_activity":{"audio_revision":0,"observed_end_sample":192000,"regions":[{"start_sample":128000,"end_sample":192000,"speakers":["speaker_0","speaker_1"]}]}}
                ]},"speaker_assignments":{"speaker_0":{"person_id":"ada","name":"Ada","source":"manual","confirmed":true}}
        })).unwrap();
        let folder = root.path().join(job.id.as_str());
        std::fs::create_dir_all(&folder).unwrap();
        let mut writer = audio::RecordingWriter::create(folder.join("audio.wav")).unwrap();
        writer.write_samples(&vec![0.1; 64000]).unwrap();
        writer.write_samples(&vec![0.25; 64000]).unwrap();
        writer.write_samples(&vec![0.3; 64000]).unwrap();
        drop(writer);
        let source = job
            .document
            .as_ref()
            .unwrap()
            .segments
            .last()
            .unwrap()
            .clone();
        let audio = SeparatedAudio {
            model: separation::MODEL.into(),
            channels: [vec![0.25; 64000], vec![0.1; 64000]],
        };
        app.insert_job(job.clone());
        (
            root,
            app,
            job,
            source,
            audio,
            Model {
                qualified: true,
                calls: AtomicUsize::new(0),
            },
        )
    }
    fn complete(app: &AppState, model: &Model, state: &mut Identity, audio: &SeparatedAudio) {
        loop {
            let (job, snapshot) = {
                let store = app.store();
                (
                    store.get(&state.original.id, false).unwrap().unwrap(),
                    Snapshot::load(&store, model).unwrap(),
                )
            };
            let before = model.calls.load(Ordering::SeqCst);
            if !state
                .step(
                    model,
                    app.data_dir(),
                    &CancellationToken::new(),
                    &job,
                    &snapshot,
                    audio,
                )
                .unwrap()
            {
                break;
            }
            assert_eq!(
                model.calls.load(Ordering::SeqCst),
                before + 1,
                "one embedding per scheduling step"
            );
        }
    }
    fn accepted(job: &mut Job, source: &Segment) {
        let source_key = separation::source_key(job, source);
        let voices = std::array::from_fn(|channel| separation::CandidateVoice {
            text: format!("voice {channel}"),
            language: "en".into(),
            review: false,
            transcription_review: None,
        });
        job.document
            .as_mut()
            .unwrap()
            .segments
            .iter_mut()
            .find(|row| row.id == source.id)
            .unwrap()
            .separation_candidate = Some(Box::new(separation::Candidate {
            id: EPISODE.into(),
            source_key,
            original: Box::new(source.clone()),
            voices,
            model: separation::MODEL.into(),
        }));
        separation::accept(job, &source.id, EPISODE).unwrap();
    }
    fn rows(job: &Job) -> Vec<&Segment> {
        job.document
            .as_ref()
            .unwrap()
            .segments
            .iter()
            .filter(|r| r.separation.is_some())
            .collect()
    }

    #[test]
    fn association_is_acoustic_and_keeps_existing_manual_people_and_restore() {
        let (_root, app, job, source, audio, model) = fixture();
        let profiles_before = app.store().voice_profiles().unwrap();
        let mut state = Identity::begin(
            &model,
            &mut app.store(),
            &job,
            &source,
            EPISODE,
            &audio,
            &[vec![(0, 64000)], vec![(0, 64000)]],
        )
        .unwrap()
        .unwrap();
        complete(&app, &model, &mut state, &audio);
        assert_eq!(model.calls.load(Ordering::SeqCst), 6);
        let snapshot = Snapshot::load(&app.store(), &model).unwrap();
        let mut current = app.store().get(&job.id, false).unwrap().unwrap();
        current.name = "unrelated renamed meeting".into();
        current.revision += 7;
        current.document.as_mut().unwrap().segments[0].text = "retained human words".into();
        let manual = current.extra["speaker_assignments"]["speaker_0"].clone();
        accepted(&mut current, &source);
        assert!(state.apply(&mut current, &snapshot));
        assert_eq!(
            rows(&current)
                .iter()
                .map(|r| r.speaker.as_str())
                .collect::<Vec<_>>(),
            vec!["speaker_1", "speaker_0"]
        );
        assert_eq!(current.extra["speaker_assignments"]["speaker_0"], manual);
        assert_eq!(current.name, "unrelated renamed meeting");
        assert_eq!(
            current.document.as_ref().unwrap().segments[0].text,
            "retained human words"
        );
        assert_eq!(app.store().voice_profiles().unwrap(), profiles_before);
        separation::restore(&mut current, EPISODE).unwrap();
        assert_eq!(
            current
                .document
                .as_ref()
                .unwrap()
                .segments
                .last()
                .unwrap()
                .text,
            "mixed originals"
        );
    }

    #[test]
    fn pending_identity_cannot_override_source_edits_preference_reset_or_accepted_corrections() {
        for mode in ["speaker", "bounds", "source", "off_on", "row_edit"] {
            let (_root, app, job, source, audio, model) = fixture();
            let mut state = Identity::begin(
                &model,
                &mut app.store(),
                &job,
                &source,
                EPISODE,
                &audio,
                &[vec![(0, 64000)], vec![(0, 64000)]],
            )
            .unwrap()
            .unwrap();
            complete(&app, &model, &mut state, &audio);
            if mode == "off_on" {
                app.store().set_recognition(false).unwrap();
                app.store().set_recognition(true).unwrap();
            }
            let snapshot = Snapshot::load(&app.store(), &model).unwrap();
            let mut current = app.store().get(&job.id, false).unwrap().unwrap();
            match mode {
                "speaker" => {
                    current
                        .document
                        .as_mut()
                        .unwrap()
                        .speakers
                        .insert("speaker_0".into(), "Manual change".into());
                }
                "bounds" => {
                    current.document.as_mut().unwrap().segments[0].start = 0.5;
                }
                "source" => {
                    current.source_file = Some("changed.wav".into());
                }
                _ => {}
            }
            accepted(&mut current, &source);
            if mode == "row_edit" {
                let row = current
                    .document
                    .as_mut()
                    .unwrap()
                    .segments
                    .iter_mut()
                    .find(|r| r.separation.as_ref().is_some_and(|v| v.channel == 0))
                    .unwrap();
                row.speaker = "speaker_0".into();
                row.text = "human words".into();
                row.protected_fields = Some(vec!["speaker".into(), "text".into()]);
            }
            state.apply(&mut current, &snapshot);
            if mode == "row_edit" {
                assert_eq!(rows(&current)[0].text, "human words");
                assert_eq!(rows(&current)[0].speaker, "speaker_0");
                assert_eq!(
                    current.extra["voice_checks"][state.check_key(0)]["status"],
                    "cancelled"
                );
            } else {
                assert!(
                    rows(&current).iter().all(|row| row.speaker == "unassigned"),
                    "{mode}"
                );
            }
        }
    }

    #[test]
    fn absent_separated_calibration_consumes_no_model_work_or_identity_check() {
        let (_root, app, job, source, audio, mut model) = fixture();
        model.qualified = false;
        assert!(
            Identity::begin(
                &model,
                &mut app.store(),
                &job,
                &source,
                EPISODE,
                &audio,
                &[vec![(0, 64000)], vec![(0, 64000)]]
            )
            .unwrap()
            .is_none()
        );
        assert_eq!(model.calls.load(Ordering::SeqCst), 0);
        assert!(
            !app.store()
                .get(&job.id, false)
                .unwrap()
                .unwrap()
                .extra
                .contains_key("voice_checks")
        );
    }

    #[test]
    fn one_consented_profile_can_match_but_forget_replace_and_interrupt_win() {
        for mode in ["match", "forget", "replace", "interrupt"] {
            let (_root, app, mut job, source, mut audio, model) = fixture();
            job.document
                .as_mut()
                .unwrap()
                .segments
                .retain(|row| row.id == source.id);
            app.insert_job(job.clone());
            let person = app
                .store()
                .edit_person(|people| {
                    let person = people.create_with_id("Lin", 0., "lin").unwrap();
                    Ok::<_, StoreError>((person.clone(), person))
                })
                .unwrap();
            let profile = VoiceProfile {
                person_id: person.id.clone(),
                version: "v1".into(),
                enrolled_at: 0.,
                model: policy().calibration.model,
                centroid: vec![1., 0., 0.],
                clips: Vec::new(),
                consent: EXPLICIT_REMEMBER_VOICE.into(),
            };
            let generation = app.store().voice_generation(&person.id).unwrap().unwrap();
            assert!(
                app.store()
                    .publish_voice(&job.id, generation, &profile, |_| Ok::<_, StoreError>(true))
                    .unwrap()
            );
            // Channel B has no compatible known voice; A must still be useful.
            audio.channels.swap(0, 1);
            let mut state = Identity::begin(
                &model,
                &mut app.store(),
                &job,
                &source,
                EPISODE,
                &audio,
                &[vec![(0, 64000)], vec![(0, 64000)]],
            )
            .unwrap()
            .unwrap();
            complete(&app, &model, &mut state, &audio);
            match mode {
                "forget" => app.store().forget_voice(&person.id).unwrap(),
                "replace" => {
                    let mut replacement = profile.clone();
                    replacement.version = "v2".into();
                    let generation = app.store().voice_generation(&person.id).unwrap().unwrap();
                    assert!(
                        app.store()
                            .publish_voice(
                                &job.id,
                                generation,
                                &replacement,
                                |_| Ok::<_, StoreError>(true)
                            )
                            .unwrap()
                    );
                }
                _ => {}
            }
            let profiles_before = app.store().voice_profiles().unwrap();
            let snapshot = Snapshot::load(&app.store(), &model).unwrap();
            let mut current = app.store().get(&job.id, false).unwrap().unwrap();
            accepted(&mut current, &source);
            if mode == "interrupt" {
                state.interrupt(&mut current);
                current.extra["voice_checks"][state.check_key(1)] =
                    json!({"id":"newer-check","status":"checking"});
                state.interrupt(&mut current);
            }
            state.apply(&mut current, &snapshot);
            if mode == "match" {
                assert_eq!(rows(&current)[0].speaker, state.check_key(0));
                assert_eq!(rows(&current)[1].speaker, "unassigned");
                assert_eq!(
                    current.document.as_ref().unwrap().speakers[&state.check_key(0)],
                    "Lin"
                );
                assert_eq!(
                    current.extra["speaker_assignments"][state.check_key(0)]["confirmed"],
                    false
                );
            } else {
                assert!(
                    rows(&current).iter().all(|row| row.speaker == "unassigned"),
                    "{mode}"
                );
                if mode == "interrupt" {
                    assert_eq!(
                        current.extra["voice_checks"][state.check_key(1)]["id"],
                        "newer-check"
                    );
                }
            }
            assert_eq!(app.store().voice_profiles().unwrap(), profiles_before);
            assert_eq!(
                app.store().people().unwrap().get("lin").unwrap().name,
                "Lin"
            );
        }
    }
}
impl Snapshot {
    pub(crate) fn load(
        store: &SqliteJobStore,
        models: &dyn ApplicationModels,
    ) -> Result<Self, StoreError> {
        let policy = models.separated_voice_policy();
        let profiles = store
            .voice_profiles()?
            .into_values()
            .filter_map(|value| serde_json::from_value::<VoiceProfile>(value).ok())
            .filter(|profile| {
                policy
                    .as_ref()
                    .is_some_and(|p| profile.model == p.calibration.model)
                    && profile.consent == EXPLICIT_REMEMBER_VOICE
            })
            .collect();
        Ok(Self {
            enabled: store.preference("voice_recognition")?.as_deref() != Some("false"),
            policy,
            profiles,
            people: store.people()?,
        })
    }
}

struct SoloReference {
    reference: Reference,
    clips: Vec<VoiceClip>,
    vectors: Vec<Vec<f64>>,
}

pub(crate) struct Identity {
    original: Job,
    source: Segment,
    episode: String,
    policy: Policy,
    profiles: Vec<VoiceProfile>,
    solo: Vec<SoloReference>,
    query_ranges: [Option<(usize, usize)>; 2],
    queries: [Vec<Vec<f64>>; 2],
    query_attempts: [usize; 2],
    failed_channels: [bool; 2],
    failed_reference: bool,
    finished: bool,
}

fn meeting(job: &Job) -> Option<RecognitionMeeting> {
    let document = job.document.as_ref()?;
    let segments = document
        .segments
        .iter()
        .map(|row| serde_json::from_value(serde_json::to_value(row).ok()?).ok())
        .collect::<Option<Vec<_>>>()?;
    let mut meeting = RecognitionMeeting::local(job.id.as_str(), segments);
    meeting.speakers = document.speakers.clone();
    meeting.provenance_kind = document
        .provenance
        .as_ref()?
        .get("kind")?
        .as_str()
        .map(str::to_owned);
    Some(meeting)
}

fn assignment<'a>(job: &'a Job, track: &str) -> Option<&'a Value> {
    job.extra
        .get("speaker_assignments")
        .and_then(|v| v.get(track))
        .filter(|v| !v.is_null())
}

impl Identity {
    pub(crate) fn begin(
        models: &dyn ApplicationModels,
        store: &mut SqliteJobStore,
        original: &Job,
        source: &Segment,
        episode: &str,
        audio: &SeparatedAudio,
        speech_ranges: &[Vec<(usize, usize)>; 2],
    ) -> Result<Option<Self>, StoreError> {
        let snapshot = Snapshot::load(store, models)?;
        let Some(policy) = snapshot.policy else {
            return Ok(None);
        };
        let minimum = policy.calibration.minimum_clips;
        let length = audio.channels[0].len();
        let source_length = separation::sample_bounds(source)
            .ok()
            .map(|(start, end)| (end - start) as usize);
        if !snapshot.enabled
            || policy.separator_model != audio.model
            || audio.validate(length).is_err()
            || source_length != Some(length)
            || length < 32000
        {
            return Ok(None);
        }
        let Some(meeting) = meeting(original) else {
            return Ok(None);
        };
        let Some(activity) = source
            .speaker_activity
            .as_ref()
            .and_then(|v| serde_json::from_value::<voice::SpeakerActivity>(v.clone()).ok())
        else {
            return Ok(None);
        };
        let tracks: std::collections::BTreeSet<_> = activity
            .regions
            .iter()
            .flat_map(|region| region.speakers.iter())
            .collect();
        let mut solo = Vec::new();
        for track in tracks {
            if !meeting.speakers.contains_key(track) {
                continue;
            }
            let clips = voice::automatic_clips(&meeting, track, minimum);
            if clips.is_empty() {
                continue;
            }
            solo.push(SoloReference {
                reference: Reference {
                    target: Target::Track(track.clone()),
                    person_id: assignment(original, track)
                        .and_then(|a| a["person_id"].as_str())
                        .map(str::to_owned),
                    centroid: Vec::new(),
                },
                clips,
                vectors: Vec::new(),
            });
        }
        if solo.is_empty() && snapshot.profiles.is_empty() {
            return Ok(None);
        }
        let query_ranges = std::array::from_fn(|channel| {
            voice::separation::query_range(length, &speech_ranges[channel])
        });
        if query_ranges.iter().all(Option::is_none) {
            return Ok(None);
        }
        let mut state = Self {
            original: original.clone(),
            source: source.clone(),
            episode: episode.into(),
            policy,
            profiles: snapshot.profiles,
            solo,
            query_ranges,
            queries: [Vec::new(), Vec::new()],
            query_attempts: query_ranges.map(|range| usize::from(range.is_none())),
            failed_channels: query_ranges.map(|range| range.is_none()),
            failed_reference: false,
            finished: false,
        };
        for channel in 0..2 {
            let key = state.check_key(channel);
            state
                .original
                .extra
                .entry("voice_checks")
                .or_insert_with(|| json!({}))[&key] = json!({
                "id":episode,"status":"checking","manual":false,
                "model":state.policy.calibration.model,"clips":[],"separation_episode":episode
            });
        }
        store.update_job(&original.id, |current| {
            if current
                .document
                .as_ref()
                .and_then(|d| d.segments.iter().find(|s| s.id == source.id))
                .is_some_and(|row| {
                    separation::source_key(current, row) == separation::source_key(original, source)
                })
            {
                for channel in 0..2 {
                    let key = state.check_key(channel);
                    current
                        .extra
                        .entry("voice_checks")
                        .or_insert_with(|| json!({}))[&key] =
                        state.original.extra["voice_checks"][&key].clone();
                }
            }
            Ok::<_, StoreError>(())
        })?;
        Ok(Some(state))
    }

    fn check_key(&self, channel: usize) -> String {
        format!("separated_{}_{}", self.episode, channel)
    }

    pub(crate) fn interrupt(&self, current: &mut Job) {
        for channel in 0..2 {
            if let Some(check) = current
                .extra
                .get_mut("voice_checks")
                .and_then(|v| v.get_mut(self.check_key(channel)))
                && check["id"] == self.episode
                && check["status"] == "checking"
            {
                check["status"] = json!("cancelled");
            }
        }
    }

    fn unchanged(&self, current: &Job, snapshot: &Snapshot, accepted: bool) -> bool {
        if !snapshot.enabled
            || snapshot.policy.as_ref() != Some(&self.policy)
            || snapshot.profiles != self.profiles
            || (0..2).any(|channel| {
                current
                    .extra
                    .get("voice_checks")
                    .and_then(|checks| checks.get(self.check_key(channel)))
                    .is_none_or(|check| {
                        check["id"] != self.episode || check["status"] != "checking"
                    })
            })
        {
            return false;
        }
        let source = if accepted {
            current
                .extra
                .get("separation_history")
                .and_then(|h| h.get(&self.episode))
                .and_then(|h| serde_json::from_value::<separation::History>(h.clone()).ok())
                .map(|h| *h.original)
        } else {
            current
                .document
                .as_ref()
                .and_then(|d| d.segments.iter().find(|s| s.id == self.source.id))
                .cloned()
        };
        if source.as_ref().is_none_or(|row| {
            separation::source_key(current, row)
                != separation::source_key(&self.original, &self.source)
        }) {
            return false;
        }
        let Some(meeting) = meeting(current) else {
            return false;
        };
        self.solo.iter().all(|reference| {
            let Target::Track(track) = &reference.reference.target else { return false; };
            current.document.as_ref().and_then(|d| d.speakers.get(track)) == self.original.document.as_ref().and_then(|d| d.speakers.get(track))
                && assignment(current, track) == assignment(&self.original, track)
                && voice::clips_current(&meeting, &reference.clips)
                && reference.clips.iter().all(|clip| {
                    let old = self.original.document.as_ref().and_then(|d| d.segments.iter().find(|s| s.id == clip.segment_id));
                    let new = current.document.as_ref().and_then(|d| d.segments.iter().find(|s| s.id == clip.segment_id));
                    matches!((old,new),(Some(old),Some(new)) if crate::native::passage_audio_unchanged(current,&self.original,new,old))
                })
        })
    }

    pub(crate) fn step(
        &mut self,
        models: &dyn ApplicationModels,
        root: &Path,
        cancellation: &CancellationToken,
        current: &Job,
        snapshot: &Snapshot,
        audio: &SeparatedAudio,
    ) -> Result<bool, ChildError> {
        if self.finished {
            return Ok(false);
        }
        if cancellation.is_cancelled() {
            return Err(ChildError::Cancelled);
        }
        if !self.unchanged(current, snapshot, false) {
            self.finished = true;
            return Ok(false);
        }
        if let Some(reference) = self
            .solo
            .iter_mut()
            .find(|r| r.vectors.len() < r.clips.len())
        {
            let clip = &reference.clips[reference.vectors.len()];
            let row = self
                .original
                .document
                .as_ref()
                .and_then(|d| d.segments.iter().find(|s| s.id == clip.segment_id))
                .expect("selected reference");
            let pcm =
                crate::native::passage_recording_path(root, &self.original, row).and_then(|path| {
                    audio::read_pcm16_range(path, clip.start, clip.end)
                        .map_err(std::io::Error::other)
                });
            let embedded = pcm
                .map_err(|e| ChildError::failed(e.to_string(), false))
                .and_then(|pcm| models.embed_live_voice(&pcm, cancellation));
            match embedded {
                Ok(embedding)
                    if embedding.model == self.policy.calibration.model && embedding.clip.clean =>
                {
                    match voice::normalized(&embedding.clip.vector, embedding.model.dimension) {
                        Ok(vector) => reference.vectors.push(vector),
                        Err(_) => self.failed_reference = true,
                    }
                }
                Err(
                    error @ (ChildError::ProcessLost { .. }
                    | ChildError::Timeout
                    | ChildError::Cancelled),
                ) => return Err(error),
                _ => self.failed_reference = true,
            }
            if self.failed_reference {
                self.finished = true;
            }
            return Ok(true);
        }
        if let Some(channel) = (0..2).find(|channel| self.query_attempts[*channel] == 0) {
            let (start, end) = self.query_ranges[channel].expect("selected speech query");
            self.query_attempts[channel] += 1;
            let Some(pcm) = audio
                .channels
                .get(channel)
                .and_then(|pcm| pcm.get(start..end))
            else {
                self.failed_channels[channel] = true;
                return Ok(true);
            };
            let pcm = voice::separation::query_pcm(pcm)
                .map_err(|e| ChildError::failed(e.to_string(), false))?;
            match models.embed_live_voice(&pcm, cancellation) {
                Ok(embedding)
                    if embedding.model == self.policy.calibration.model && embedding.clip.clean =>
                {
                    match voice::normalized(&embedding.clip.vector, embedding.model.dimension) {
                        Ok(vector) => self.queries[channel].push(vector),
                        Err(_) => self.failed_channels[channel] = true,
                    }
                }
                Err(
                    error @ (ChildError::ProcessLost { .. }
                    | ChildError::Timeout
                    | ChildError::Cancelled),
                ) => return Err(error),
                _ => self.failed_channels[channel] = true,
            }
            return Ok(true);
        }
        self.finished = true;
        Ok(false)
    }

    /// Called in the separation publication transaction after the accepted pair
    /// exists. It never creates a People record or changes any voice profile.
    pub(crate) fn apply(&self, current: &mut Job, snapshot: &Snapshot) -> bool {
        let fresh =
            self.finished && !self.failed_reference && self.unchanged(current, snapshot, true);
        let result = if fresh {
            let mut references: Vec<_> = self
                .solo
                .iter()
                .filter_map(|solo| {
                    let mut reference = solo.reference.clone();
                    reference.centroid =
                        voice::separation::centroid(&self.policy.calibration.model, &solo.vectors)
                            .ok()?;
                    Some(reference)
                })
                .collect();
            references.extend(
                self.profiles
                    .iter()
                    .filter(|p| snapshot.people.get(&p.person_id).is_ok())
                    .map(|profile| Reference {
                        target: Target::Person {
                            person_id: profile.person_id.clone(),
                            profile_version: profile.version.clone(),
                        },
                        person_id: Some(profile.person_id.clone()),
                        centroid: profile.centroid.clone(),
                    }),
            );
            let channels = std::array::from_fn(|channel| {
                if self.failed_channels[channel] {
                    Vec::new()
                } else {
                    self.queries[channel].clone()
                }
            });
            voice::separation::propose(
                &self.policy,
                &self.policy.separator_model,
                &self.policy.calibration.model,
                &channels,
                &references,
            )
            .unwrap_or([None, None])
        } else {
            [None, None]
        };
        let mut changed = false;
        for (channel, matched) in result.iter().enumerate() {
            let assigned = matched
                .as_ref()
                .is_some_and(|matched| self.assign(current, snapshot, channel, matched));
            let status = if !fresh {
                "cancelled"
            } else if assigned {
                "matched"
            } else if matched.is_some() {
                "cancelled"
            } else {
                "unknown"
            };
            if let Some(check) = current
                .extra
                .get_mut("voice_checks")
                .and_then(|v| v.get_mut(self.check_key(channel)))
                && check["id"] == self.episode
                && check["status"] == "checking"
            {
                check["status"] = json!(status);
            }
            changed |= assigned;
        }
        changed
    }

    fn assign(
        &self,
        current: &mut Job,
        snapshot: &Snapshot,
        channel: usize,
        matched: &Match,
    ) -> bool {
        let (track, name, person) = match &matched.target {
            Target::Track(track) => {
                let Some(name) = current
                    .document
                    .as_ref()
                    .and_then(|d| d.speakers.get(track))
                    .cloned()
                else {
                    return false;
                };
                (track.clone(), name, None)
            }
            Target::Person { person_id, .. } => {
                let Ok(person) = snapshot.people.get(person_id) else {
                    return false;
                };
                let tracks: Vec<_> = current
                    .document
                    .iter()
                    .flat_map(|d| d.speakers.keys())
                    .filter(|track| {
                        assignment(current, track)
                            .is_some_and(|a| a["person_id"].as_str() == Some(person_id))
                    })
                    .collect();
                if let [track] = tracks.as_slice() {
                    let name = current
                        .document
                        .as_ref()
                        .expect("assigned document")
                        .speakers[*track]
                        .clone();
                    ((*track).clone(), name, None)
                } else {
                    (
                        self.check_key(channel),
                        person.name.clone(),
                        Some(person.id.clone()),
                    )
                }
            }
        };
        if person.is_some() && assignment(current, &track).is_some() {
            return false;
        }
        let Some(row) = current.document.as_mut().and_then(|d| {
            d.segments.iter_mut().find(|row| {
                row.separation
                    .as_ref()
                    .is_some_and(|v| v.episode_id == self.episode && v.channel == channel as u8)
            })
        }) else {
            return false;
        };
        if row.speaker != "unassigned"
            || row
                .protected_fields
                .as_ref()
                .is_some_and(|fields| !fields.is_empty())
        {
            return false;
        }
        let accepted = current
            .extra
            .get("separation_history")
            .and_then(|h| h.get(&self.episode))
            .and_then(|h| serde_json::from_value::<separation::History>(h.clone()).ok());
        if accepted.as_ref().is_none_or(|h| {
            speakerdesk_core::live::segment_version(&h.accepted[channel])
                != speakerdesk_core::live::segment_version(row)
        }) {
            return false;
        }
        row.speaker = track.clone();
        row.rest.insert("separation_identity".into(), json!({"source":"automatic_voice","score":matched.score,
            "margin":matched.margin,"dataset_id":self.policy.calibration.dataset_id,"model":self.policy.calibration.model}));
        current
            .document
            .as_mut()
            .expect("accepted document")
            .speakers
            .entry(track.clone())
            .or_insert(name.clone());
        if let Some(person_id) = person {
            current
                .extra
                .entry("speaker_assignments")
                .or_insert_with(|| json!({}))[&track] = json!({
                "person_id":person_id,"meeting_id":current.id,"track_id":track,"name":name,
                "source":"automatic_voice","confirmed":false,"evidence":{"separation_episode":self.episode,
                    "score":matched.score,"margin":matched.margin,"dataset_id":self.policy.calibration.dataset_id}
            });
        }
        // Restore compares against the machine-authored accepted version. Names
        // inferred here are not user corrections and must not prevent restoration.
        if let Some(row) = current
            .document
            .as_ref()
            .and_then(|d| {
                d.segments
                    .iter()
                    .find(|r| r.id == format!("{}-{channel}", self.episode))
            })
            .cloned()
        {
            current
                .extra
                .get_mut("separation_history")
                .expect("accepted history")[&self.episode]["accepted"][channel] =
                serde_json::to_value(row).expect("accepted row");
        }
        true
    }
}
