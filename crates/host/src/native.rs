//! Production Rust service used by the existing desktop shell.
use axum::{
    Json, Router,
    body::{Body, Bytes},
    extract::{Path as RoutePath, Request, State},
    http::{StatusCode, header},
    response::{Html, IntoResponse, Response},
    routing::{get, patch},
};
use serde_json::{Value, json};
use speakerdesk_core::{
    audio,
    document::{Document, Job},
    ids::JobId,
    languages,
    models::Models,
    store::JobStore,
};
use std::{
    io::Write,
    path::{Path, PathBuf},
    sync::{Arc, Mutex, RwLock},
};
use tokio::io::{AsyncBufReadExt, AsyncReadExt, BufReader};
use tokio_util::sync::CancellationToken;

use crate::{
    admission::Activity,
    inference::InferenceSupervisor,
    native_updates,
    routes::{AppState, ApplicationModels, blocking, confined_file, media_write_directory},
};

#[derive(rust_embed::RustEmbed)]
#[folder = "../../speakerdesk/static/"]
struct Assets;

pub(crate) struct NativeModels {
    supervisor: Arc<InferenceSupervisor>,
    root: PathBuf,
    alignment: RwLock<Option<Arc<speakerdesk_core::alignment::Vocabulary>>>,
    voice: RwLock<Option<speakerdesk_core::voice::Calibration>>,
    separation_error: RwLock<Option<String>>,
    model_root: PathBuf,
    notices: PathBuf,
    setup: crate::setup::SetupService,
    task: Mutex<Option<(CancellationToken, tokio::task::JoinHandle<()>)>>,
}

impl ApplicationModels for NativeModels {
    fn live_models(
        &self,
        cancellation: &CancellationToken,
    ) -> Option<crate::inference::ChildModels> {
        Some(self.supervisor.models(cancellation.clone()))
    }
    fn live_alignment(&self) -> Option<Arc<speakerdesk_core::alignment::Vocabulary>> {
        self.alignment.read().expect("alignment capability").clone()
    }
    fn retry_passage(
        &self,
        pcm: &[f32],
        language: &str,
        processed_microphone_catalog: Option<&str>,
        cancellation: &CancellationToken,
    ) -> Result<speakerdesk_core::import::PassageCandidate, String> {
        speakerdesk_core::import::retry_passage_source(
            &self.supervisor.models(cancellation.clone()),
            pcm,
            0.,
            pcm.len() as f64 / 16000.,
            language,
            processed_microphone_catalog,
        )
        .map_err(|error| error.to_string())
    }
    fn separate_passage(
        &self,
        pcm: &[f32],
        language: &str,
        cancellation: &CancellationToken,
    ) -> Result<
        (
            speakerdesk_core::ipc::SeparatedAudio,
            [speakerdesk_core::separation::CandidateVoice; 2],
        ),
        String,
    > {
        speakerdesk_core::separation::run(
            &self.supervisor.models(cancellation.clone()),
            pcm,
            language,
        )
        .map_err(|e| e.to_string())
    }
    fn separation_available(&self) -> bool {
        self.separation_error
            .read()
            .expect("separation capability")
            .is_none()
            && crate::setup::separation_artifact().present(&self.model_root)
    }
    fn separate_passage_with_evidence(
        &self,
        pcm: &[f32],
        language: &str,
        cancellation: &CancellationToken,
    ) -> Result<
        (
            speakerdesk_core::ipc::SeparatedAudio,
            [speakerdesk_core::separation::CandidateVoice; 2],
            [Vec<(usize, usize)>; 2],
        ),
        String,
    > {
        speakerdesk_core::separation::run_with_evidence(
            &self.supervisor.models(cancellation.clone()),
            pcm,
            language,
        )
        .map_err(|e| e.to_string())
    }
    fn voice_policy(&self) -> Option<speakerdesk_core::voice::Calibration> {
        self.voice.read().expect("voice capability").clone()
    }
    fn separated_voice_policy(&self) -> Option<speakerdesk_core::voice::separation::Policy> {
        let policy = native_separated_voice_policy()?;
        (self.separation_available()
            && self
                .voice
                .read()
                .expect("voice capability")
                .as_ref()
                .is_some_and(|active| active.model == policy.calibration.model))
        .then_some(policy)
    }
    fn embed_voice(
        &self,
        pcm: &[f32],
        cancellation: &CancellationToken,
    ) -> Result<speakerdesk_core::ipc::VoiceEmbedding, String> {
        self.supervisor
            .models(cancellation.clone())
            .embed_voice(pcm)
            .map_err(|e| e.to_string())
    }
    fn infer(&self, job: &Job, cancellation: &CancellationToken) -> Result<Document, String> {
        self.infer_progress(job, cancellation, None, &mut |progress| {
            Ok(progress.rows.to_vec())
        })
    }
    fn import_identity(&self) -> Option<String> {
        use sha2::{Digest, Sha256};
        let mut hash = Sha256::new();
        hash.update(env!("CARGO_PKG_VERSION"));
        hash.update(crate::setup::import_artifact_identity());
        Some(format!("bounded-import-v1:{:x}", hash.finalize()))
    }
    fn infer_progress(
        &self,
        job: &Job,
        cancellation: &CancellationToken,
        resume: Option<&speakerdesk_core::import::ImportResume>,
        publish: &mut dyn FnMut(
            speakerdesk_core::import::ImportProgress<'_>,
        ) -> Result<
            Vec<speakerdesk_core::document::Segment>,
            speakerdesk_core::import::ImportError,
        >,
    ) -> Result<Document, String> {
        let path = recording_path(&self.root, job).map_err(|e| e.to_string())?;
        let reader = audio::RecordingReader::open(path).map_err(|e| e.to_string())?;
        let alignment = self.alignment.read().expect("alignment capability").clone();
        speakerdesk_core::import::import_reader(
            &self.supervisor.models(cancellation.clone()),
            &reader,
            &job.language,
            alignment.as_deref(),
            resume,
            publish,
        )
        .map_err(|e| e.to_string())
    }
}

impl NativeModels {
    fn start(
        self: &Arc<Self>,
        app: AppState,
        download: bool,
        separation: bool,
    ) -> Result<(), crate::admission::AdmissionError> {
        let mut task = self.task.lock().expect("setup task");
        if task.as_ref().is_some_and(|(_, task)| !task.is_finished()) {
            return Ok(());
        }
        let lease = app.admission.lease(Activity::Setup)?;
        let cancel = CancellationToken::new();
        let token = cancel.clone();
        let native = self.clone();
        native.setup.begin(
            &if separation {
                vec![crate::setup::separation_artifact()]
            } else {
                crate::setup::artifacts()
            },
            download,
        );
        if !download {
            app.set_issues(vec!["Initializing native speech models…".into()]);
        }
        let handle = tokio::spawn(async move {
            let _lease = lease;
            if download {
                native.record_setup_attempt(&app);
            }
            let result = if download {
                native.install(&app, &token, separation).await
            } else {
                async {
                    // Recover an interrupted Windows directory replacement before the child loads it.
                    #[cfg(windows)]
                    let _setup_owner = crate::setup::Installer::open(&native.model_root)
                        .await
                        .map_err(|error| error.to_string())?;
                    native
                        .initialize_async(app.clone(), token.clone(), false)
                        .await
                }
                .await
            };
            if let Err(error) = &result {
                // Do not leave a first-run failure looking like an indefinitely pending load.
                if app.model_status().1.is_empty() {
                    app.set_issues(vec![error.clone()]);
                }
            }
            let missing = !download
                && !token.is_cancelled()
                && !app.admission.closed()
                && native.setup.progress().status == "missing";
            native.setup.finish(result, missing);
            if download {
                native.record_setup_attempt(&app);
            }
        });
        *task = Some((cancel, handle));
        Ok(())
    }
    fn record_setup_attempt(&self, app: &AppState) {
        let progress = self.setup.progress();
        let Ok(attempt) = serde_json::to_value(&progress) else {
            return;
        };
        let result = (|| {
            let mut store = app.store();
            let mut history: Vec<Value> = store
                .preference("model_setup_history")?
                .and_then(|raw| serde_json::from_str(&raw).ok())
                .unwrap_or_default();
            history.retain(|item| item["started_at"] != progress.started_at);
            history.push(attempt);
            if history.len() > 8 {
                history.drain(..history.len() - 8);
            }
            store.set_preference(
                "model_setup_history",
                &serde_json::to_string(&history).expect("setup history serializes"),
            )
        })();
        if let Err(error) = result {
            tracing::warn!(%error, "Could not retain model setup diagnostics");
        }
    }
    fn cancel(&self) {
        if let Some((cancel, task)) = self.task.lock().expect("setup task").as_ref()
            && !task.is_finished()
        {
            self.setup.phase("Cancelling model setup");
            cancel.cancel();
        }
    }
    async fn join(&self) -> Result<(), tokio::task::JoinError> {
        let task = self.task.lock().expect("setup task").take();
        if let Some((cancel, task)) = task {
            cancel.cancel();
            task.await?;
        }
        Ok(())
    }
    async fn initialize_async(
        self: &Arc<Self>,
        app: AppState,
        cancel: CancellationToken,
        all: bool,
    ) -> Result<(), String> {
        self.setup.loading();
        let native = self.clone();
        tokio::task::spawn_blocking(move || native.initialize(&app, &cancel, all))
            .await
            .map_err(|e| e.to_string())?
    }
    fn initialize(
        &self,
        app: &AppState,
        cancel: &CancellationToken,
        all: bool,
    ) -> Result<(), String> {
        let artifacts = crate::setup::artifacts();
        *self
            .separation_error
            .write()
            .expect("separation capability") = Some("Separation has not finished loading.".into());
        self.setup.check_downloads(
            &self.model_root,
            &if self.setup.progress().target.as_deref() == Some("separation") {
                vec![crate::setup::separation_artifact()]
            } else {
                artifacts.clone()
            },
        );
        let models = self.supervisor.models(cancel.clone());
        let mut reports = Vec::new();
        let mut failures = Vec::new();
        let mut core_failures = Vec::new();
        let mut ordinary_errors = true;
        let mut alignment = None;
        let mut voice = None;
        for (index, artifact) in artifacts.iter().enumerate() {
            if cancel.is_cancelled() {
                return Err("Model setup cancelled".into());
            }
            let identity = match artifact.id.as_str() {
                "silero-speech" => format!("{}@{}", artifact.repo, &artifact.revision[..7]),
                "coarse-alignment" => native_alignment_descriptor().checkpoint.into(),
                "redimnet2-b6" => native_voice_policy()
                    .map(|policy| policy.model.revision)
                    .unwrap_or_default(),
                _ => format!("{}@{}", artifact.repo, artifact.revision),
            };
            self.setup.phase(format!("Loading {}…", artifact.name));
            let checked = models.setup_check(std::slice::from_ref(&artifact.id));
            let ordinary_error = matches!(
                &checked,
                Err(speakerdesk_core::error::ChildError::Failed { .. })
            );
            let loaded = checked.map_err(|e| e.to_string()).and_then(|mut loaded| {
                if loaded.len() != 1 || !loaded[0].ok || loaded[0].model != identity {
                    return Err("Model initialization returned an unexpected identity".into());
                }
                if artifact.id == "coarse-alignment" {
                    let tokens = std::fs::read_to_string(
                        self.model_root.join("coarse-alignment/tokens.txt"),
                    )
                    .map_err(|e| e.to_string())?;
                    alignment = Some(Arc::new(speakerdesk_core::alignment::Vocabulary::parse(
                        &tokens,
                    )?));
                }
                if artifact.id == "redimnet2-b6" {
                    voice = Some(native_voice_policy().ok_or_else(|| {
                        "Model initialization returned an unexpected identity".to_owned()
                    })?);
                }
                Ok(loaded.remove(0))
            });
            match loaded {
                Ok(report) => reports.push(report),
                Err(error) => {
                    ordinary_errors &= ordinary_error;
                    let message = format!("{}: {error}", artifact.name);
                    if index < 3 {
                        core_failures.push(message.clone());
                    }
                    failures.push(message);
                    reports.push(speakerdesk_core::ipc::SetupModelReport {
                        model: identity,
                        ok: false,
                        timing_ms: 0,
                    });
                }
            }
        }
        if cancel.is_cancelled() || app.admission.closed() {
            return Err("Model setup cancelled".into());
        }
        *self.alignment.write().expect("alignment capability") = alignment;
        *self.voice.write().expect("voice capability") = voice;
        let separation = if crate::setup::separation_artifact().present(&self.model_root) {
            self.setup.phase("Loading overlapping voice separation…");
            models
                .setup_check(&["mossformer2-16k".into()])
                .map_err(|e| e.to_string())
                .and_then(|reports| {
                    if reports.len() == 1
                        && reports[0].ok
                        && reports[0].model == speakerdesk_core::separation::MODEL
                    {
                        Ok(())
                    } else {
                        Err(
                            "Separation model initialization returned an unexpected identity"
                                .into(),
                        )
                    }
                })
        } else {
            Err("Download the model to separate overlapping voices.".into())
        };
        *self
            .separation_error
            .write()
            .expect("separation capability") = separation.err();
        let missing = !all
            && ordinary_errors
            && crate::setup::initial_models_missing(&self.model_root, &artifacts, &reports);
        app.set_model_reports(reports);
        app.set_issues(core_failures.clone());
        let errors = if all { failures } else { core_failures };
        if errors.is_empty() {
            Ok(())
        } else {
            if missing {
                self.setup.missing();
            }
            Err(errors.join(" "))
        }
    }
    async fn install(
        self: &Arc<Self>,
        app: &AppState,
        cancel: &CancellationToken,
        separation: bool,
    ) -> Result<(), String> {
        let installer = crate::setup::Installer::open(&self.model_root)
            .await
            .map_err(|e| e.to_string())?;
        let mut completed = 0;
        #[cfg(windows)]
        let mut released_models = false;
        for model in if separation {
            vec![crate::setup::separation_artifact()]
        } else {
            crate::setup::artifacts()
        } {
            if !installer
                .installed(&model, cancel)
                .await
                .map_err(|e| e.to_string())?
            {
                let staging = installer
                    .stage(&model, completed, &self.setup, cancel)
                    .await
                    .map_err(|e| e.to_string())?;
                if installer
                    .needs_preparation(&model, &staging, cancel)
                    .await
                    .map_err(|e| e.to_string())?
                {
                    self.setup.phase(if separation {
                        "Preparing separation weights"
                    } else {
                        "Preparing native timing weights"
                    });
                    let supervisor = self.supervisor.clone();
                    let cancellation = cancel.clone();
                    tokio::task::spawn_blocking(move || {
                        if separation {
                            supervisor.prepare_separation(&cancellation)
                        } else {
                            supervisor.prepare_alignment(&cancellation)
                        }
                    })
                    .await
                    .map_err(|e| e.to_string())?
                    .map_err(|e| e.to_string())?;
                }
                #[cfg(windows)]
                if !released_models {
                    // Close mapped Windows model files before renaming their containing directory.
                    let supervisor = self.supervisor.clone();
                    let cancellation = cancel.clone();
                    tokio::task::spawn_blocking(move || supervisor.restart(&cancellation))
                        .await
                        .map_err(|error| error.to_string())?
                        .map_err(|error| error.to_string())?;
                    released_models = true;
                }
                installer
                    .activate(&model, &staging, cancel)
                    .await
                    .map_err(|e| e.to_string())?;
            }
            completed += model.download_bytes();
            self.setup.bytes(completed);
        }
        self.setup.loading();
        self.setup.activated();
        self.record_setup_attempt(app);
        self.setup.phase("Restarting the model worker");
        let supervisor = self.supervisor.clone();
        let cancellation = cancel.clone();
        tokio::task::spawn_blocking(move || supervisor.restart(&cancellation))
            .await
            .map_err(|e| e.to_string())?
            .map_err(|e| e.to_string())?;
        let ordinary = self
            .initialize_async(app.clone(), cancel.clone(), !separation)
            .await;
        if separation {
            // Refresh other capabilities after restart, but judge this request by
            // the model it installed. Missing timing must not turn success into failure.
            if cancel.is_cancelled() || app.admission.closed() {
                return Err("Model setup cancelled".into());
            }
            if let Some(error) = self
                .separation_error
                .read()
                .expect("separation capability")
                .clone()
            {
                return Err(error);
            }
        } else {
            ordinary?;
        }
        Ok(())
    }
}

fn native_alignment_descriptor() -> &'static speakerdesk_core::timing_policy::Descriptor {
    #[cfg(windows)]
    {
        &speakerdesk_core::timing_policy::WINDOWS
    }
    #[cfg(not(windows))]
    {
        &speakerdesk_core::timing_policy::MAC
    }
}

#[cfg(any(windows, test))]
const WINDOWS_VOICE_POLICY: &str =
    include_str!("../../../speakerdesk/voice_windows_calibration.json");

#[cfg(any(windows, test))]
fn windows_voice_calibration(
    metadata: &Value,
    separated: bool,
) -> Option<speakerdesk_core::voice::Calibration> {
    use speakerdesk_core::voice::{Calibration, VoiceModel};
    let expected = VoiceModel::new(
        "PalabraAI/redimnet2",
        "2a8d15f65b1dfb5d73fede2f11ee42bcccca3035:b6-vb2+vox2_v0-lm:sd-pcm-v1:windowsml-2.4.89-directml-v1:ort-opt-disabled",
        "89c532e48539bebddc168e773cdcbf014b1941043c50d0a4ff7ccd1139efa5f1",
        192,
        16000,
    )
    .ok()?;
    let model: VoiceModel = serde_json::from_value(metadata.get("model")?.clone()).ok()?;
    if metadata["schema_version"] != 1
        || metadata["approved_for_recognition"] != true
        || metadata["approval"]["basis"] != "preserved_mac_behavior"
        || metadata["compute_policy"] != "DirectML:ORT_DISABLE_ALL"
        || model != expected
    {
        return None;
    }
    let (section, basis, dataset, genuine, impostor, false_reject_rate) = if separated {
        (
            "separated",
            "frozen_natural_separated_query_pilot",
            "librispeech-clean-generated-query-v1:development-6-speakers",
            6,
            36,
            0.,
        )
    } else {
        (
            "ordinary",
            "owner_approved_preserved_mac_behavior",
            "librispeech-clean-ordinary-windows-v1:preserved-mac-behavior",
            12,
            4,
            1. / 12.,
        )
    };
    let policy = metadata.get(section)?;
    let measured = policy.get("calibration")?;
    let threshold = measured["threshold"].as_f64()?;
    let margin = measured["margin"].as_f64()?;
    if policy["approved_for_recognition"] != true
        || policy["admission_basis"] != basis
        || measured["dataset_id"] != dataset
        || threshold.to_bits() != 0.7264125328398843_f64.to_bits()
        || margin.to_bits() != 0.10003429964182987_f64.to_bits()
        || measured["minimum_clips"] != 2
        || measured["genuine_trials"] != genuine
        || measured["impostor_trials"] != impostor
        || measured["false_accept_rate"].as_f64()? != 0.
        || measured["false_reject_rate"].as_f64()? != false_reject_rate
        || (separated && policy["separator_model"] != speakerdesk_core::separation::MODEL)
        || (!separated && policy["original_strict_gate"]["passed"] != false)
    {
        return None;
    }
    Calibration::new(
        model,
        dataset,
        threshold,
        margin,
        genuine,
        impostor,
        0.,
        false_reject_rate,
        2,
    )
    .ok()
}

pub(crate) fn native_voice_policy() -> Option<speakerdesk_core::voice::Calibration> {
    #[cfg(windows)]
    {
        let metadata = serde_json::from_str(WINDOWS_VOICE_POLICY).ok()?;
        windows_voice_calibration(&metadata, false)
    }
    #[cfg(not(windows))]
    {
        use speakerdesk_core::voice::{Calibration, VoiceModel};
        // Frozen synthetic policy verified on native Core ML; legacy profile spaces stay distinct.
        let model = VoiceModel::new(
            "aufklarer/ReDimNet2-B6-CoreML",
            "112dd8f4f836abdf8a420e66a5e2885cf8ec64ab:sd-pcm-v1:native-coreml-v1:CPU_AND_GPU",
            "cf8cc300e240efaed56cb931bd7715ded41b66bf55af7e9a19d0b17e7fe986c9",
            192,
            16000,
        )
        .expect("pinned voice identity");
        Some(
            Calibration::new(
                model,
                "macos-compact-tts-development-v1:calibration",
                0.7264125328398843,
                0.10003429964182987,
                8,
                4,
                0.,
                0.,
                2,
            )
            .expect("measured voice policy"),
        )
    }
}

pub(crate) fn native_separated_voice_policy() -> Option<speakerdesk_core::voice::separation::Policy>
{
    #[cfg(windows)]
    {
        let metadata: Value = serde_json::from_str(WINDOWS_VOICE_POLICY).ok()?;
        let calibration = windows_voice_calibration(&metadata, true)?;
        Some(speakerdesk_core::voice::separation::Policy {
            separator_model: metadata["separated"]["separator_model"].as_str()?.into(),
            calibration,
        })
    }
    #[cfg(not(windows))]
    {
        use speakerdesk_core::voice::{Calibration, VoiceModel, separation::Policy};
        // Frozen clean public generated-query pilot; these observed rates do not
        // estimate population accuracy. New model identities require new evidence.
        let model = VoiceModel::new(
            "aufklarer/ReDimNet2-B6-CoreML",
            "112dd8f4f836abdf8a420e66a5e2885cf8ec64ab:sd-pcm-v1:native-coreml-v1:CPU_AND_GPU",
            "cf8cc300e240efaed56cb931bd7715ded41b66bf55af7e9a19d0b17e7fe986c9",
            192,
            16000,
        )
        .expect("qualified separated-query voice identity");
        Some(Policy {
            separator_model: "mossformer2-16k@407cb030cd66340918ebb6c8cc63b18f8592cdbe".into(),
            calibration: Calibration::new(
                model,
                "librispeech-clean-generated-query-v1:development-6-speakers",
                0.7264125328398843,
                0.10003429964182987,
                6,
                36,
                0.0,
                0.0,
                2,
            )
            .expect("measured separated-query policy"),
        })
    }
}

#[cfg(test)]
mod voice_policy_tests {
    use super::*;
    use speakerdesk_core::voice::{
        VoiceClip, VoiceModel, VoiceProfile, make_profile, propose_match,
    };

    fn metadata() -> Value {
        serde_json::from_str(WINDOWS_VOICE_POLICY).unwrap()
    }

    fn mac_model() -> VoiceModel {
        VoiceModel::new(
            "aufklarer/ReDimNet2-B6-CoreML",
            "112dd8f4f836abdf8a420e66a5e2885cf8ec64ab:sd-pcm-v1:native-coreml-v1:CPU_AND_GPU",
            "cf8cc300e240efaed56cb931bd7715ded41b66bf55af7e9a19d0b17e7fe986c9",
            192,
            16000,
        )
        .unwrap()
    }

    #[test]
    fn windows_voice_admission_requires_the_exact_approved_model_and_each_policy() {
        let original = metadata();
        let ordinary = windows_voice_calibration(&original, false).unwrap();
        let separated = windows_voice_calibration(&original, true).unwrap();
        assert_eq!(ordinary.model, separated.model);
        assert_ne!(ordinary.model, mac_model());
        assert_eq!((ordinary.genuine_trials, ordinary.impostor_trials), (12, 4));
        assert_eq!(ordinary.false_reject_rate, 1. / 12.);
        assert_eq!(
            original["ordinary"]["original_strict_gate"]["passed"],
            false
        );
        for (pointer, replacement) in [
            ("/schema_version", json!(2)),
            ("/approved_for_recognition", json!(false)),
            ("/approval/basis", json!("unreviewed")),
            ("/compute_policy", json!("CPU")),
            ("/model", Value::Null),
            ("/model/model_id", json!("other")),
            ("/model/revision", json!("other")),
            ("/model/artifact_sha256", json!("a".repeat(64))),
            ("/model/dimension", json!(256)),
            ("/model/sample_rate", json!(48000)),
        ] {
            let mut changed = original.clone();
            *changed.pointer_mut(pointer).unwrap() = replacement;
            assert!(
                windows_voice_calibration(&changed, false).is_none(),
                "{pointer}"
            );
            assert!(
                windows_voice_calibration(&changed, true).is_none(),
                "{pointer}"
            );
        }
        assert!(windows_voice_calibration(&Value::Null, false).is_none());
        for section in ["ordinary", "separated"] {
            let selected = section == "separated";
            let mut missing = original.clone();
            missing.as_object_mut().unwrap().remove(section);
            assert!(windows_voice_calibration(&missing, selected).is_none());
            assert!(windows_voice_calibration(&missing, !selected).is_some());
            for (field, replacement) in [
                ("approved_for_recognition", json!(false)),
                ("admission_basis", json!("unreviewed")),
                ("calibration/threshold", json!(0.7)),
                ("calibration/margin", json!(0.05)),
                ("calibration/minimum_clips", json!(1)),
                ("calibration/genuine_trials", json!(0)),
                ("calibration/false_accept_rate", json!(0.1)),
                ("calibration/dataset_id", json!("unmeasured")),
            ] {
                let mut changed = original.clone();
                *changed.pointer_mut(&format!("/{section}/{field}")).unwrap() = replacement;
                assert!(
                    windows_voice_calibration(&changed, selected).is_none(),
                    "{section}/{field}"
                );
                assert!(windows_voice_calibration(&changed, !selected).is_some());
            }
        }
        let mut changed = original;
        changed["separated"]["separator_model"] = json!("other separator");
        assert!(windows_voice_calibration(&changed, true).is_none());
        assert!(windows_voice_calibration(&changed, false).is_some());
        let active = native_voice_policy().unwrap();
        let active_separated = native_separated_voice_policy().unwrap();
        if cfg!(windows) {
            assert_eq!(active, ordinary);
            assert_eq!(active_separated.calibration, separated);
        } else {
            assert_eq!(active.model, mac_model());
            assert_eq!(active_separated.calibration.model, mac_model());
            assert_eq!((active.genuine_trials, active.impostor_trials), (8, 4));
            assert_eq!(active.false_reject_rate, 0.);
        }
    }

    #[test]
    fn windows_policy_keeps_incompatible_saved_profiles_unchanged_and_ineligible() {
        let policy = windows_voice_calibration(&metadata(), false).unwrap();
        let mut vector = vec![0.; policy.model.dimension];
        vector[0] = 1.;
        let vectors = vec![vector.clone(), vector.clone()];
        let clips = vec![
            VoiceClip::try_new("fixture", "speaker_0", "a", 0., 2.).unwrap(),
            VoiceClip::try_new("fixture", "speaker_0", "b", 2., 4.).unwrap(),
        ];
        let legacy: Value =
            serde_json::from_str(include_str!("../../../speakerdesk/voice_calibration.json"))
                .unwrap();
        let mut profiles: Vec<_> = [
            mac_model(),
            serde_json::from_value(legacy["model"].clone()).unwrap(),
        ]
        .into_iter()
        .enumerate()
        .map(|(index, model)| VoiceProfile {
            person_id: format!("retained-{index}"),
            version: "retained".into(),
            enrolled_at: 1.,
            model,
            centroid: vector.clone(),
            clips: clips.clone(),
            consent: speakerdesk_core::people::EXPLICIT_REMEMBER_VOICE.into(),
        })
        .collect();
        let before = serde_json::to_value(&profiles).unwrap();
        assert!(
            propose_match(&policy.model, &policy, &vectors, &profiles)
                .unwrap()
                .is_none()
        );
        assert_eq!(serde_json::to_value(&profiles).unwrap(), before);
        let draft = make_profile(&policy.model, &vectors, &clips).unwrap();
        profiles.push(VoiceProfile {
            person_id: "compatible".into(),
            version: "new-explicit-enrollment".into(),
            enrolled_at: 2.,
            model: draft.model,
            centroid: draft.centroid,
            clips: draft.clips,
            consent: draft.consent,
        });
        assert_eq!(
            propose_match(&policy.model, &policy, &vectors, &profiles)
                .unwrap()
                .unwrap()
                .person_id,
            "compatible"
        );
        assert_eq!(serde_json::to_value(&profiles[..2]).unwrap(), before);
    }
}

pub(crate) fn recording_path(root: &Path, job: &Job) -> std::io::Result<PathBuf> {
    let native = media_write_directory(root, job).join("audio.wav");
    if native_media_exists(root, &native)? {
        return confined_file(
            root,
            native
                .strip_prefix(root)
                .map_err(std::io::Error::other)?
                .to_str()
                .ok_or_else(|| std::io::Error::other("Invalid recording path"))?,
        );
    }
    if let Some(legacy) = job.extra.get("legacy_media_root").and_then(Value::as_str) {
        return confined_file(Path::new(legacy), &format!("{}/audio.wav", job.id));
    }
    Err(std::io::Error::new(
        std::io::ErrorKind::NotFound,
        "Recording audio is unavailable",
    ))
}

fn native_media_exists(root: &Path, path: &Path) -> std::io::Result<bool> {
    match std::fs::symlink_metadata(path) {
        Ok(_) => return Ok(true),
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => {}
        Err(error) => return Err(error),
    }
    // Windows reports NotFound for a non-directory ancestor too; that must not select legacy audio.
    for parent in path.ancestors().skip(1).take_while(|p| p.starts_with(root)) {
        match std::fs::symlink_metadata(parent) {
            Ok(metadata) if !metadata.is_dir() || metadata.is_symlink() => {
                return Err(std::io::Error::other("Invalid native media directory"));
            }
            Ok(_) => {}
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => {}
            Err(error) => return Err(error),
        }
    }
    Ok(false)
}

/// Resolve a passage on its original recording clock. Source-bound live passages
/// cannot fall back to the mixed recording when their catalog is absent or stale.
pub(crate) fn passage_recording_path(
    root: &Path,
    job: &Job,
    segment: &speakerdesk_core::document::Segment,
) -> std::io::Result<PathBuf> {
    if let Some(voice) = &segment.separation {
        return crate::routes::separation::asset_path(root, job, &voice.episode_id, voice.channel);
    }
    let invalid = || {
        std::io::Error::other(
            "Passage capture source is unavailable or no longer matches its recording",
        )
    };
    let Some(binding) = segment.capture_source.as_ref().filter(|v| !v.is_null()) else {
        if job.extra.contains_key("capture_source_catalog")
            || job
                .extra
                .get("sources")
                .and_then(Value::as_array)
                .is_some_and(|s| s.len() > 1)
        {
            return Err(invalid());
        }
        return recording_path(root, job);
    };
    let catalog = job
        .extra
        .get("capture_source_catalog")
        .ok_or_else(invalid)?;
    let hash = speakerdesk_core::memo::sha256_hex(
        &serde_json::to_vec(catalog).map_err(std::io::Error::other)?,
    );
    let source = binding["source_id"].as_str().ok_or_else(invalid)?;
    let basename = match source {
        "microphone_clean" => "microphone_clean.wav",
        "system" => "system.wav",
        _ => return Err(invalid()),
    };
    if binding["catalog_sha256"].as_str() != Some(hash.as_str())
        || binding["source_revision"] != 1
        || catalog["schema_version"] != 1
        || catalog["job_id"].as_str() != Some(job.id.as_str())
        || catalog["sample_rate"] != 16000
        || catalog["clock"] != "recording_samples"
        || catalog["policy"] != "aligned_aec_microphone_and_original_system_v1"
        || catalog["sources"]
            != json!([
            {"source_id":"microphone_clean","basename":"microphone_clean.wav","revision":1},
            {"source_id":"system","basename":"system.wav","revision":1}])
    {
        return Err(invalid());
    }
    // Prefer an owned derived source, otherwise keep migrated v1 source audio read-only.
    // A copy of audio.wav alone does not claim ownership of its separate source files.
    let native = media_write_directory(root, job);
    let (authority, folder) = match native_media_exists(root, &native.join(basename))? {
        true => (root, native),
        false => {
            if let Some(legacy) = job.extra.get("legacy_media_root").and_then(Value::as_str) {
                (Path::new(legacy), Path::new(legacy).join(job.id.as_str()))
            } else {
                (root, native)
            }
        }
    };
    if std::fs::symlink_metadata(&folder)?.file_type().is_symlink()
        || std::fs::symlink_metadata(folder.join(basename))?
            .file_type()
            .is_symlink()
    {
        return Err(invalid());
    }
    let path = folder.join(basename);
    let relative = path
        .strip_prefix(authority)
        .map_err(std::io::Error::other)?;
    confined_file(authority, relative.to_str().ok_or_else(invalid)?)
}

pub(crate) fn passage_audio_unchanged(
    current: &Job,
    original: &Job,
    current_segment: &speakerdesk_core::document::Segment,
    original_segment: &speakerdesk_core::document::Segment,
) -> bool {
    current.id == original.id
        && current.source_file == original.source_file
        && [
            "legacy_media_root",
            "audio_revision",
            "capture_source_catalog",
            "sources",
        ]
        .iter()
        .all(|key| current.extra.get(*key) == original.extra.get(*key))
        && current_segment.separation == original_segment.separation
        && current_segment.capture_source == original_segment.capture_source
        && current_segment.audio_revision == original_segment.audio_revision
        && current_segment.canonical_utterance_id == original_segment.canonical_utterance_id
}

pub fn run(
    storage: PathBuf,
    model_root: PathBuf,
    executable: PathBuf,
) -> Result<(), Box<dyn std::error::Error>> {
    let runtime = tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()?;
    runtime.block_on(serve(storage, model_root, executable))
}

async fn serve(
    storage: PathBuf,
    model_root: PathBuf,
    executable: PathBuf,
) -> Result<(), Box<dyn std::error::Error>> {
    let app = AppState::open(storage)?;
    if app.store().preference("default_language")?.is_none() {
        app.store().set_preference("default_language", "auto")?;
    }
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await?;
    let address = listener.local_addr()?;
    let notices = executable
        .parent()
        .ok_or("native executable has no parent directory")?
        .join(if cfg!(windows) {
            "THIRD_PARTY_NOTICES.txt"
        } else {
            "../Resources/THIRD_PARTY_NOTICES.txt"
        });
    let supervisor =
        InferenceSupervisor::new(executable, env!("CARGO_PKG_VERSION"), model_root.clone())?;
    let native = Arc::new(NativeModels {
        supervisor: supervisor.clone(),
        root: app.data_dir().to_path_buf(),
        model_root,
        notices,
        alignment: RwLock::new(None),
        voice: RwLock::new(None),
        separation_error: RwLock::new(Some("Separation has not been checked.".into())),
        setup: crate::setup::SetupService::default(),
        task: Mutex::new(None),
    });
    app.set_models(native.clone());
    app.native
        .set(native.clone())
        .ok()
        .expect("native setup configured once");
    app.updates.lock().expect("updates").connect();
    native.start(app.clone(), false, false)?;
    crate::meeting::configure_capture(
        &app,
        std::env::var_os("SPEAKERDESK_CAPTURE_HELPER").map(PathBuf::from),
        std::env::var_os("SPEAKERDESK_ECHO_LIBRARY").map(PathBuf::from),
    )
    .await;
    let stop = CancellationToken::new();
    let server_stop = stop.clone();
    let router = crate::router(app.clone());
    let mut server = tokio::spawn(async move {
        axum::serve(listener, router)
            .with_graceful_shutdown(server_stop.cancelled_owned())
            .await
    });
    let output_error = writeln!(std::io::stdout(), "SPEAKERDESK_URL=http://{address}/")
        .and_then(|_| std::io::stdout().flush())
        .err();
    let mut input = BufReader::new(tokio::io::stdin());
    let mut update_shutdown = false;
    let control_result = loop {
        if let Some(error) = output_error {
            break Err(error);
        }
        let mut line = Vec::new();
        let count = match (&mut input)
            .take((native_updates::MAX_FRAME + 1) as u64)
            .read_until(b'\n', &mut line)
            .await
        {
            Ok(count) => count,
            Err(error) => break Err(error),
        };
        if count > native_updates::MAX_FRAME {
            break Err(std::io::Error::other("Oversized native control frame"));
        }
        if count == 0 || line == b"shutdown\n" {
            break Ok(());
        }
        let Some(body) = line.strip_prefix(native_updates::PREFIX.as_bytes()) else {
            continue;
        };
        let Ok(message) = serde_json::from_slice(body) else {
            continue;
        };
        if app
            .updates
            .lock()
            .expect("updates")
            .handle(&app.admission, message)
        {
            update_shutdown = true;
            break Ok(());
        }
    };
    app.admission.close();
    stop.cancel();
    native.cancel();
    let close_app = app.clone();
    let close_supervisor = supervisor.clone();
    let cleanup = tokio::spawn(async move {
        let invalidate = close_app.clone();
        let cancelled = tokio::task::spawn_blocking(move || invalidate.cancel_all())
            .await
            .map_err(|e| speakerdesk_core::error::StoreError::message(e.to_string()))?;
        let meeting_cleanup = crate::meeting::shutdown(&close_app).await;
        tokio::task::spawn_blocking(move || {
            close_supervisor.shutdown();
            close_app.drain();
        })
        .await
        .map_err(|e| speakerdesk_core::error::StoreError::message(e.to_string()))?;
        cancelled.and(meeting_cleanup)
    });
    // Reaping is independent of clients finishing uploads or audio responses.
    let drained = tokio::time::timeout(crate::inference::CONTROL_TIMEOUT, &mut server).await;
    let drained = match drained {
        Ok(result) => result?.is_ok(),
        Err(_) => {
            server.abort();
            let _ = server.await;
            false
        }
    };
    let cleanup = cleanup.await?;
    native.join().await?;
    if update_shutdown {
        native_updates::emit(
            &app.admission
                .confirm_exit(drained && cleanup.is_ok() && supervisor.child_pid().is_none())?,
        )?;
    }
    cleanup?;
    control_result?;
    if !drained {
        return Err(std::io::Error::other(
            "Native HTTP service did not drain before the control deadline",
        )
        .into());
    }
    Ok(())
}

pub(crate) fn routes() -> Router<AppState> {
    Router::new()
        .merge(crate::recovery::routes())
        .route("/", get(index))
        .route("/static/{*path}", get(asset))
        .route("/api/jobs/{jid}/audio", get(playback))
        .route("/api/preferences/language", patch(language))
        .route("/api/preferences", patch(preferences))
        .route("/api/setup", get(setup).post(start_setup))
        .route("/api/setup/cancel", axum::routing::post(cancel_setup))
        .route(
            "/api/setup/separation",
            axum::routing::post(start_separation_setup),
        )
        .route("/api/notices", get(notices))
}

async fn index(State(app): State<AppState>) -> Html<String> {
    let theme = app
        .store()
        .preference("theme")
        .ok()
        .flatten()
        .filter(|theme| matches!(theme.as_str(), "system" | "light" | "dark"))
        .unwrap_or_else(|| "system".into());
    Html(
        include_str!("../../../speakerdesk/templates/index.html")
            .replace("{{token}}", app.token())
            .replace("{{theme}}", &theme)
            .replace("{{platform}}", std::env::consts::OS),
    )
}

async fn preferences(State(app): State<AppState>, body: Bytes) -> Response {
    let value: serde_json::Value = match serde_json::from_slice(&body) {
        Ok(value) => value,
        Err(_) => {
            return (
                StatusCode::BAD_REQUEST,
                Json(json!({"error":"Invalid preference."})),
            )
                .into_response();
        }
    };
    let Some(theme) = value
        .get("theme")
        .and_then(serde_json::Value::as_str)
        .filter(|theme| matches!(*theme, "system" | "light" | "dark"))
    else {
        return (
            StatusCode::BAD_REQUEST,
            Json(json!({"error":"Choose System, Light or Dark."})),
        )
            .into_response();
    };
    match app.store().set_preference("theme", theme) {
        Ok(()) => Json(json!({"theme":theme})).into_response(),
        Err(error) => (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(json!({"error":error.to_string()})),
        )
            .into_response(),
    }
}

async fn asset(RoutePath(path): RoutePath<String>) -> Response {
    let Some(asset) = Assets::get(&path) else {
        return StatusCode::NOT_FOUND.into_response();
    };
    let content_type = match Path::new(&path).extension().and_then(|s| s.to_str()) {
        Some("js") => "text/javascript; charset=utf-8",
        Some("css") => "text/css; charset=utf-8",
        Some("svg") => "image/svg+xml",
        Some("woff2") => "font/woff2",
        Some("png") => "image/png",
        _ => "application/octet-stream",
    };
    (
        [(header::CONTENT_TYPE, content_type)],
        asset.data.into_owned(),
    )
        .into_response()
}

async fn playback(
    State(app): State<AppState>,
    RoutePath(jid): RoutePath<String>,
    request: Request,
) -> Response {
    let path = tokio::task::spawn_blocking(move || {
        let id = JobId::parse(jid).ok()?;
        let job = app.store().get(&id, false).ok()??;
        recording_path(app.data_dir(), &job).ok()
    })
    .await;
    let Ok(Some(path)) = path else {
        return StatusCode::NOT_FOUND.into_response();
    };
    match tower_http::services::ServeFile::new(path)
        .try_call(request)
        .await
    {
        Ok(response) => response.map(Body::new),
        Err(_) => StatusCode::INTERNAL_SERVER_ERROR.into_response(),
    }
}

async fn language(State(app): State<AppState>, body: Bytes) -> Response {
    let Ok(body) = serde_json::from_slice::<Value>(&body) else {
        return failure("Choose a supported language mode.");
    };
    let Some(language) = body["language"]
        .as_str()
        .filter(|s| *s == "auto" || languages::is_supported(s))
        .map(str::to_owned)
    else {
        return failure("Choose a supported language mode.");
    };
    blocking(
        move || match app.store().set_preference("default_language", &language) {
            Ok(()) => Json(json!({"default_language":language})).into_response(),
            Err(_) => failure("Could not save the default language."),
        },
    )
    .await
}

async fn setup(State(app): State<AppState>) -> Response {
    blocking(move || {
        let enabled = match app.store().preference("voice_recognition") {
            Ok(value) => value.as_deref()!=Some("false"),
            Err(_) => return StatusCode::INTERNAL_SERVER_ERROR.into_response(),
        };
        let (issues, reports) = app.model_status();
        let core_ready = issues.is_empty();
        let voice = app.models().voice_policy().is_some();
        let automatic = core_ready && app.automatic_language_ready();
        let timing = core_ready && app.alignment_ready();
        let ready = core_ready && automatic && timing && voice;
        let progress = app.native.get().map(|native| native.setup.progress()).unwrap_or_default();
        let history: Vec<Value> = app.store().preference("model_setup_history").ok().flatten()
            .and_then(|raw| serde_json::from_str(&raw).ok()).unwrap_or_default();
        let operation = if progress.target.is_some() {
            serde_json::to_value(&progress).ok()
        } else {
            history.last().cloned().map(|mut last| {
                if matches!(last["status"].as_str(), Some("downloading" | "loading")) {
                    last["status"] = json!("interrupted");
                    last["error"] = json!("Setup stopped before it finished. Retry to continue.");
                }
                last
            })
        };
        let busy = matches!(progress.status,"downloading" | "loading");
        let supported = cfg!(any(all(target_os="macos",target_arch="aarch64"),all(windows,target_arch="x86_64")));
        let can_download = supported && app.native.get().is_some() && !app.admission.busy() && !app.admission.closed() && !app.admission.reserved();
        let catalog = crate::setup::artifacts();
        let mut models: Vec<_> = catalog.iter().enumerate().map(|(index, model)| json!({"id":model.id,"name":model.name,"installed":reports.get(index).is_some_and(|r| r.ok),"bytes":model.download_bytes(),"checkpoint":reports.get(index).map(|r| &r.model)})).collect();
        let alignment = &catalog[4];
        let timing_languages = native_alignment_descriptor().accuracy_languages;
        let timing_accuracy = if timing_languages.is_empty() { "Word times are uncalibrated estimates." } else { "Non-English word times are uncalibrated estimates." };
        let separation = crate::setup::separation_artifact();
        let separation_setup = progress.models.iter().any(|id|id==&separation.id);
        let separation_installed = app.native.get().is_some_and(|native|separation.present(&native.model_root));
        models.push(json!({"id":separation.id,"name":separation.name,"installed":separation_installed,"optional":true,"bytes":separation.download_bytes(),"checkpoint":speakerdesk_core::separation::MODEL}));
        let status = if busy || progress.status == "missing" {progress.status} else if progress.error.is_some() {"failed"} else if ready {"ready"} else {"idle"};
        Json(json!({
            "supported":supported,"ready":ready,"core_ready":core_ready,"automatic_language":automatic,"operation":operation,"setup_history":history,
            "status":status,"phase":progress.phase,"error":progress.error,
            "download_required":progress.download_required,
            "models":models,"total_bytes":if separation_setup {separation.download_bytes()} else {catalog.iter().map(|m| m.download_bytes()).sum::<u64>()},
            "separation":{"installed":separation_installed,"ready":app.models().separation_available(),"bytes":separation.download_bytes(),"can_download":can_download,"active":separation_setup && busy},
            "downloaded_bytes":progress.downloaded_bytes,"required_free_bytes":if separation_setup {separation.stored_bytes()} else {catalog.iter().map(|m| m.stored_bytes()).sum::<u64>()},
            "can_download":can_download,"can_cancel":busy && app.native.get().is_some(),
            "alignment":{"id":alignment.id,"status":if busy {progress.status}else if timing {"ready"}else {"idle"},"ready":timing,"can_download":false,
                "supported_languages":languages::LANGUAGES.iter().map(|(code,_)| *code).collect::<Vec<_>>(),"timing_accuracy_calibrated_languages":timing_languages,"total_bytes":alignment.download_bytes(),"stored_bytes":alignment.stored_bytes(),"required_free_bytes":alignment.stored_bytes(),
                "message":"Supplied-text timing preserves authoritative transcript words. Use Download models below to finish setup for all local models.","phase":progress.phase,"downloaded_bytes":0},
            "voice":{"status":if voice {"ready"}else {"unavailable"},"enabled":enabled,"available":voice,"released":false,
                "message":if voice {"Native voice enrollment and manual and automatic checks are available."}else {"The native voice model did not initialize. Saved voices and names are kept."}},
            "message": if busy {String::new()} else if !core_ready {issues.join(" ")} else {format!("Native preview. Automatic language: {}. Supplied-text timing for imports and supported sealed live passages: {}. Live and saved accuracy passes recheck sealed, unedited canonical passages using their captured Auto or explicit language. Long passages require one confidently routed language and unambiguous context timing. {timing_accuracy} Uncertain, mixed-language or unsupported passages retain their previous words. Live recording continues during accuracy work; pause it independently when needed.",if automatic {"available"}else {"unavailable"},if timing {"available"}else {"unavailable"})}
        })).into_response()
    }).await
}

async fn start_setup(State(app): State<AppState>) -> Response {
    let Some(native) = app.native.get().cloned() else {
        return failure("Native model setup is unavailable in this runtime");
    };
    match native.start(app, true, false) {
        Ok(()) => Json(json!({"status":native.setup.progress().status})).into_response(),
        Err(error) => (
            StatusCode::CONFLICT,
            Json(json!({"error":error.to_string()})),
        )
            .into_response(),
    }
}
async fn start_separation_setup(State(app): State<AppState>) -> Response {
    let Some(native) = app.native.get().cloned() else {
        return failure("Native model setup is unavailable in this runtime");
    };
    match native.start(app, true, true) {
        Ok(()) => Json(json!({"status":native.setup.progress().status})).into_response(),
        Err(error) => (
            StatusCode::CONFLICT,
            Json(json!({"error":error.to_string()})),
        )
            .into_response(),
    }
}
async fn cancel_setup(State(app): State<AppState>) -> Response {
    if let Some(native) = app.native.get() {
        native.cancel();
    }
    Json(json!({"status":"cancelling"})).into_response()
}
async fn notices(State(app): State<AppState>) -> Response {
    let bytes = match app.native.get() {
        Some(native) => tokio::fs::read(&native.notices).await.ok(),
        None => None,
    };
    let Some(bytes) = bytes else {
        return (
            StatusCode::SERVICE_UNAVAILABLE,
            Json(json!({"error":"Third-party notices are unavailable in this build."})),
        )
            .into_response();
    };
    (
        [
            (header::CONTENT_TYPE, "text/plain; charset=utf-8"),
            (
                header::CONTENT_DISPOSITION,
                "attachment; filename=Speakerdesk-third-party-notices.txt",
            ),
        ],
        bytes,
    )
        .into_response()
}
fn failure(message: &str) -> Response {
    (StatusCode::BAD_REQUEST, Json(json!({"error":message}))).into_response()
}

#[cfg(test)]
mod setup_tests {
    use super::*;
    #[test]
    fn passage_source_resolution_rejects_missing_stale_and_escaping_catalogs() {
        use speakerdesk_core::{document::Segment, ids::JobId};
        let root = tempfile::tempdir().unwrap();
        let id = JobId::parse("0123456789abcdef0123456789abcdef").unwrap();
        let mut job: Job = serde_json::from_value(json!({"id":id,"name":"sources","kind":"meeting","language":"en","status":"ready","created":0.,"updated":0.,"revision":0,"sources":["microphone","system"]})).unwrap();
        let folder = root.path().join(job.id.as_str());
        std::fs::create_dir(&folder).unwrap();
        std::fs::write(folder.join("audio.wav"), "mix").unwrap();
        std::fs::write(folder.join("system.wav"), "system").unwrap();
        std::fs::write(folder.join("microphone_clean.wav"), "microphone").unwrap();
        let catalog = json!({"schema_version":1,"job_id":job.id,"sample_rate":16000,"clock":"recording_samples",
            "policy":"aligned_aec_microphone_and_original_system_v1","sources":[
            {"source_id":"microphone_clean","basename":"microphone_clean.wav","revision":1},
            {"source_id":"system","basename":"system.wav","revision":1}]});
        let hash = speakerdesk_core::memo::sha256_hex(&serde_json::to_vec(&catalog).unwrap());
        job.extra.insert("capture_source_catalog".into(), catalog);
        let segment: Segment = serde_json::from_value(
            json!({"id":"s","start":0.,"end":1.,"speaker":"speaker_0","text":"words",
            "capture_source":{"source_id":"system","source_revision":1,"catalog_sha256":hash}}),
        )
        .unwrap();
        assert_eq!(
            std::fs::read_to_string(passage_recording_path(root.path(), &job, &segment).unwrap())
                .unwrap(),
            "system"
        );
        for case in ["missing", "source", "revision", "hash", "job", "basename"] {
            let mut bad = job.clone();
            let mut row = segment.clone();
            match case {
                "missing" => row.capture_source = None,
                "source" => row.capture_source.as_mut().unwrap()["source_id"] = json!("audio"),
                "revision" => row.capture_source.as_mut().unwrap()["source_revision"] = json!(2),
                "hash" => row.capture_source.as_mut().unwrap()["catalog_sha256"] = json!("wrong"),
                "job" => {
                    bad.extra.get_mut("capture_source_catalog").unwrap()["job_id"] =
                        json!("another")
                }
                "basename" => {
                    let catalog = bad.extra.get_mut("capture_source_catalog").unwrap();
                    catalog["sources"][1]["basename"] = json!("../outside.wav");
                    row.capture_source.as_mut().unwrap()["catalog_sha256"] = json!(
                        speakerdesk_core::memo::sha256_hex(&serde_json::to_vec(catalog).unwrap())
                    );
                }
                _ => unreachable!(),
            }
            assert!(
                passage_recording_path(root.path(), &bad, &row).is_err(),
                "{case}"
            );
        }
        let native_root = root.path().join("native");
        std::fs::create_dir(&native_root).unwrap();
        let mut migrated = job.clone();
        migrated
            .extra
            .insert("legacy_media_root".into(), json!(root.path()));
        let before = std::fs::read(folder.join("system.wav")).unwrap();
        assert_eq!(
            std::fs::read(passage_recording_path(&native_root, &migrated, &segment).unwrap())
                .unwrap(),
            before
        );
        let derived = media_write_directory(&native_root, &migrated);
        std::fs::create_dir_all(&derived).unwrap();
        std::fs::write(derived.join("audio.wav"), "derived mix only").unwrap();
        assert_eq!(
            std::fs::read(passage_recording_path(&native_root, &migrated, &segment).unwrap())
                .unwrap(),
            before
        );
        std::fs::write(derived.join("system.wav"), "derived source").unwrap();
        assert_eq!(
            std::fs::read(passage_recording_path(&native_root, &migrated, &segment).unwrap())
                .unwrap(),
            b"derived source"
        );
        assert_eq!(std::fs::read(folder.join("system.wav")).unwrap(), before);
        let invalid_native = root.path().join("invalid-native");
        std::fs::create_dir(&invalid_native).unwrap();
        std::fs::write(invalid_native.join("derived"), "not a directory").unwrap();
        assert!(
            passage_recording_path(&invalid_native, &migrated, &segment).is_err(),
            "native I/O errors must not silently select legacy audio"
        );
        assert!(recording_path(&invalid_native, &migrated).is_err());
        #[cfg(unix)]
        {
            let linked_native = root.path().join("linked-native");
            std::fs::create_dir(&linked_native).unwrap();
            std::os::unix::fs::symlink(native_root.join("derived"), linked_native.join("derived"))
                .unwrap();
            assert!(
                passage_recording_path(&linked_native, &migrated, &segment).is_err(),
                "a derived parent symlink must not escape native media ownership"
            );
            let outside = root.path().join("outside.wav");
            std::fs::write(&outside, "outside").unwrap();
            std::fs::remove_file(folder.join("system.wav")).unwrap();
            std::os::unix::fs::symlink(outside, folder.join("system.wav")).unwrap();
            assert!(passage_recording_path(root.path(), &job, &segment).is_err());
        }
        let mut single = job;
        single.extra.remove("capture_source_catalog");
        single.extra.insert("sources".into(), json!(["microphone"]));
        let mut row = segment;
        row.capture_source = None;
        assert_eq!(
            std::fs::read_to_string(passage_recording_path(root.path(), &single, &row).unwrap())
                .unwrap(),
            "mix"
        );
    }

    #[tokio::test]
    async fn actual_router_allows_edits_and_cancellation_until_setup_cleanup_finishes() {
        // Model work is a held replay. Requests use the production router and file-backed store.
        let temp = tempfile::tempdir().unwrap();
        let app = AppState::open(temp.path().join("data")).unwrap();
        let jid = "12345678901234567890123456789012";
        let document = json!({"speakers":{"speaker_0":"Speaker 1"},"segments":[{"id":"one","start":0.,"end":1.,"speaker":"speaker_0","text":"Original words."}]});
        app.insert_job(serde_json::from_value(json!({"id":jid,"name":"Replay","status":"ready","language":"en","duration":1.,"created":0.,"updated":0.,"revision":0,"document":document})).unwrap());
        let invalid_root = temp.path().join("invalid-model-directory");
        std::fs::write(&invalid_root, b"fixture prevents network access").unwrap();
        let supervisor =
            InferenceSupervisor::new(std::env::current_exe().unwrap(), "test", &invalid_root)
                .unwrap();
        let native = Arc::new(NativeModels {
            supervisor: supervisor.clone(),
            root: app.data_dir().into(),
            model_root: invalid_root,
            notices: temp.path().join("THIRD_PARTY_NOTICES.txt"),
            alignment: RwLock::new(None),
            voice: RwLock::new(None),
            separation_error: RwLock::new(Some("Separation has not been checked.".into())),
            setup: crate::setup::SetupService::default(),
            task: Mutex::new(None),
        });
        app.native.set(native.clone()).ok().unwrap();
        let cancel = CancellationToken::new();
        let token = cancel.clone();
        let cleanup = Arc::new(tokio::sync::Notify::new());
        let gate = cleanup.clone();
        let lease = app.admission.lease(Activity::Setup).unwrap();
        native.setup.begin(&crate::setup::artifacts(), false);
        *native.task.lock().unwrap() = Some((
            cancel.clone(),
            tokio::spawn(async move {
                let _lease = lease;
                token.cancelled().await;
                gate.notified().await;
            }),
        ));
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let address = listener.local_addr().unwrap();
        let router = crate::router(app.clone());
        let stop = CancellationToken::new();
        let stopped = stop.clone();
        let server = tokio::spawn(async move {
            axum::serve(listener, router)
                .with_graceful_shutdown(stopped.cancelled_owned())
                .await
                .unwrap();
        });
        let client = reqwest::Client::new();
        let notice_url = format!("http://{address}/api/notices");
        assert_eq!(
            client.get(&notice_url).send().await.unwrap().status(),
            StatusCode::SERVICE_UNAVAILABLE
        );
        std::fs::write(&native.notices, b"Fixture dependency notice").unwrap();
        let response = client.get(notice_url).send().await.unwrap();
        assert_eq!(response.status(), StatusCode::OK);
        assert_eq!(
            response.headers()[header::CONTENT_TYPE],
            "text/plain; charset=utf-8"
        );
        assert_eq!(response.text().await.unwrap(), "Fixture dependency notice");
        let url = format!("http://{address}");
        let post = |path: &str| {
            client
                .post(format!("{url}{path}"))
                .header("X-Speakerdesk-Token", app.token())
        };
        let duplicate = post("/api/setup").send().await.unwrap();
        assert_eq!(
            duplicate.status(),
            200,
            "duplicate setup should reach the owner"
        );
        assert_eq!(
            duplicate.json::<Value>().await.unwrap()["status"],
            "loading"
        );
        let setup_status = || client.get(format!("{url}/api/setup"));
        let loading = setup_status()
            .send()
            .await
            .unwrap()
            .json::<Value>()
            .await
            .unwrap();
        assert_eq!(loading["status"], "loading");
        assert_eq!(loading["alignment"]["status"], "loading");
        assert_eq!(loading["can_cancel"], true);
        assert_eq!(loading["can_download"], false);
        assert_eq!(loading["download_required"], false);
        let optional = loading["models"]
            .as_array()
            .unwrap()
            .iter()
            .find(|model| model["id"] == "mossformer2-16k")
            .expect("Settings must expose optional separation setup");
        assert_eq!(optional["installed"], false);
        assert_eq!(optional["optional"], true);
        assert_eq!(
            loading["total_bytes"],
            crate::setup::artifacts()
                .iter()
                .map(|model| model.download_bytes())
                .sum::<u64>()
        );
        // Transfers and resident initialization share one owned setup task, but not a UI state.
        native.setup.begin(&crate::setup::artifacts(), true);
        native.setup.bytes(12);
        let downloading = setup_status()
            .send()
            .await
            .unwrap()
            .json::<Value>()
            .await
            .unwrap();
        assert_eq!(downloading["status"], "downloading");
        assert_eq!(downloading["downloaded_bytes"], 12);
        assert_eq!(downloading["download_required"], true);
        native.setup.loading();
        native.setup.phase("A phase description is not the status");
        let loading = setup_status()
            .send()
            .await
            .unwrap()
            .json::<Value>()
            .await
            .unwrap();
        assert_eq!(loading["status"], "loading");
        assert_eq!(loading["alignment"]["status"], "loading");
        assert_eq!(
            loading["downloaded_bytes"], 12,
            "loading preserves completed transfer accounting"
        );
        let mut edited = document;
        edited["segments"][0]["text"] = json!("Saved while setup waits.");
        let saved = client
            .put(format!("{url}/api/jobs/{jid}/transcript"))
            .header("X-Speakerdesk-Token", app.token())
            .json(&json!({"revision":0,"document":edited}))
            .send()
            .await
            .unwrap();
        assert_eq!(saved.status(), 200, "{}", saved.text().await.unwrap());
        assert_eq!(
            post(&format!("/api/jobs/{jid}/run"))
                .send()
                .await
                .unwrap()
                .status(),
            409
        );
        assert_eq!(
            post("/api/setup/cancel").send().await.unwrap().status(),
            200
        );
        assert!(cancel.is_cancelled());
        let cancelling = setup_status()
            .send()
            .await
            .unwrap()
            .json::<Value>()
            .await
            .unwrap();
        assert_eq!(cancelling["status"], "loading");
        assert_eq!(
            cancelling["can_cancel"], true,
            "cancel acknowledgement is not task completion"
        );
        assert!(
            app.admission.busy(),
            "cancel acknowledgement must retain admission through cleanup"
        );
        assert!(app.admission.reserve(1, 1).is_err());
        assert_eq!(
            app.store()
                .get(&JobId::parse(jid).unwrap(), false)
                .unwrap()
                .unwrap()
                .revision,
            1
        );
        cleanup.notify_one();
        native.join().await.unwrap();
        assert!(!app.admission.busy());
        assert_eq!(
            post("/api/setup").send().await.unwrap().status(),
            200,
            "fresh setup must not hold an Import lease in middleware"
        );
        let task = native.task.lock().unwrap().take().unwrap().1;
        task.await.unwrap();
        assert_eq!(native.setup.progress().status, "failed");
        assert!(native.setup.progress().error.is_some());
        let failed = setup_status()
            .send()
            .await
            .unwrap()
            .json::<Value>()
            .await
            .unwrap();
        assert_eq!(failed["status"], "failed");
        assert_eq!(failed["can_cancel"], false);
        assert_eq!(failed["operation"]["target"], "models");
        assert_eq!(failed["operation"]["status"], "failed");
        let first_failure = failed["operation"]["error"].clone();
        assert_eq!(
            post("/api/setup/separation").send().await.unwrap().status(),
            200
        );
        let task = native.task.lock().unwrap().take().unwrap().1;
        task.await.unwrap();
        let separation_failed = setup_status()
            .send()
            .await
            .unwrap()
            .json::<Value>()
            .await
            .unwrap();
        assert_eq!(separation_failed["operation"]["target"], "separation");
        assert_eq!(separation_failed["operation"]["status"], "failed");
        assert!(
            separation_failed["setup_history"]
                .as_array()
                .unwrap()
                .iter()
                .any(|attempt| attempt["target"] == "models" && attempt["error"] == first_failure)
        );
        let persisted: Vec<Value> = serde_json::from_str(
            &app.store()
                .preference("model_setup_history")
                .unwrap()
                .unwrap(),
        )
        .unwrap();
        assert_eq!(persisted.last().unwrap()["target"], "separation");
        stop.cancel();
        server.await.unwrap();
        tokio::task::spawn_blocking(move || supervisor.shutdown())
            .await
            .unwrap();
    }
}
