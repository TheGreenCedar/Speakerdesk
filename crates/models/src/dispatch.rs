use std::sync::atomic::{AtomicBool, Ordering};

use speakerdesk_core::error::ChildError;
use speakerdesk_core::ipc::{Request, Response};

use crate::{cohere, ctc, memory, redimnet, sortformer, whisper_lid};

#[cfg(target_os = "macos")]
pub struct Runtime {
    model_root: std::path::PathBuf,
    silero: Option<crate::silero::network::Network>,
    nemotron: Option<sortformer::network::Network>,
    cohere: Option<cohere::network::Network>,
    whisper: Option<whisper_lid::network::Network>,
    ctc: Option<ctc::network::Network>,
    voice: Option<redimnet::coreml::Network>,
    vad_sessions: std::collections::HashMap<u64, crate::silero::session::Session>,
    next_session: u64,
}

#[cfg(target_os = "macos")]
impl Runtime {
    pub fn new(model_root: std::path::PathBuf) -> Result<Self, ChildError> {
        crate::metallib::prepare(None)?;
        // Weights stay resident; released intermediate buffers need not stay cached.
        mlx_rs::memory::set_cache_limit(0).map_err(crate::weights::error)?;
        Ok(Self {
            model_root,
            silero: None,
            nemotron: None,
            cohere: None,
            whisper: None,
            ctc: None,
            voice: None,
            vad_sessions: Default::default(),
            next_session: 1,
        })
    }

    pub fn execute(
        &mut self,
        request: Request,
        cancel: &AtomicBool,
    ) -> Result<Response, ChildError> {
        use speakerdesk_core::ipc::Handle;
        if cancel.load(Ordering::Acquire) {
            return Err(ChildError::Cancelled);
        }
        if let Request::SetupCheck { models } = &request {
            let mut reports = Vec::new();
            for model in models {
                let started = std::time::Instant::now();
                let pin = match model.as_str() {
                    "silero-speech" => {
                        if self.silero.is_none() {
                            self.silero = Some(crate::silero::network::Network::load(
                                &self.model_root.join(model),
                            )?);
                        }
                        crate::silero::WEIGHT_PIN.to_string()
                    }
                    "nemotron" => {
                        if self.nemotron.is_none() {
                            self.nemotron = Some(sortformer::network::Network::load(
                                &self.model_root.join(model),
                            )?);
                        }
                        sortformer::WEIGHT_PIN.to_string()
                    }
                    "cohere-speech" => {
                        if self.cohere.is_none() {
                            self.cohere = Some(cohere::network::Network::load(
                                &self.model_root.join(model),
                            )?);
                        }
                        cohere::WEIGHT_PIN.to_string()
                    }
                    "whisper-language" => {
                        if self.whisper.is_none() {
                            self.whisper = Some(whisper_lid::network::Network::load(
                                &self.model_root.join(model),
                            )?);
                        }
                        whisper_lid::WEIGHT_PIN.to_string()
                    }
                    "coarse-alignment" => {
                        if self.ctc.is_none() {
                            self.ctc =
                                Some(ctc::network::Network::load(&self.model_root.join(model))?);
                        }
                        speakerdesk_core::alignment::CHECKPOINT.to_string()
                    }
                    "redimnet2-b6" => {
                        if self.voice.is_none() {
                            self.voice = Some(redimnet::coreml::Network::load(
                                &self.model_root.join(model),
                                redimnet::coreml::ComputePolicy::CpuAndGpu,
                            )?);
                        }
                        self.voice
                            .as_ref()
                            .expect("loaded above")
                            .identity()
                            .revision
                    }
                    _ => return Err(ChildError::Unimplemented),
                };
                if cancel.load(Ordering::Acquire) {
                    return Err(ChildError::Cancelled);
                }
                reports.push(speakerdesk_core::ipc::SetupModelReport {
                    model: pin,
                    ok: true,
                    timing_ms: started.elapsed().as_millis() as u64,
                });
            }
            return Ok(Response::Setup(reports));
        }
        match request {
            Request::Hello { .. } => Ok(Response::Ready {
                build_id: env!("CARGO_PKG_VERSION").into(),
                device: "metal".into(),
                models: [
                    self.silero
                        .as_ref()
                        .map(|_| crate::silero::WEIGHT_PIN.into()),
                    self.nemotron
                        .as_ref()
                        .map(|_| sortformer::WEIGHT_PIN.into()),
                    self.cohere.as_ref().map(|_| cohere::WEIGHT_PIN.into()),
                    self.whisper
                        .as_ref()
                        .map(|_| whisper_lid::WEIGHT_PIN.into()),
                    self.ctc
                        .as_ref()
                        .map(|_| speakerdesk_core::alignment::CHECKPOINT.into()),
                    self.voice
                        .as_ref()
                        .map(|network| network.identity().revision),
                ]
                .into_iter()
                .flatten()
                .collect(),
            }),
            Request::VadOpen { conditioning } => {
                let session = crate::silero::session::Session::new(conditioning)?;
                if self.silero.is_none() {
                    self.silero = Some(crate::silero::network::Network::load(
                        &self.model_root.join("silero-speech"),
                    )?);
                }
                let id = self.next_session;
                self.next_session = id
                    .checked_add(1)
                    .ok_or_else(|| ChildError::failed("model session id exhausted", false))?;
                self.vad_sessions.insert(id, session);
                Ok(Response::Handle(Handle {
                    generation: 0,
                    session: id,
                }))
            }
            Request::VadFeed { handle, pcm, last } => {
                if handle.generation != 0 {
                    return Err(ChildError::failed("invalid child generation", false));
                }
                let session = self
                    .vad_sessions
                    .get_mut(&handle.session)
                    .ok_or_else(|| ChildError::failed("unknown Silero session", false))?;
                session
                    .feed(
                        self.silero.as_mut().expect("loaded before open"),
                        &pcm,
                        last,
                        cancel,
                    )
                    .map(Response::Vad)
            }
            Request::VadClose { handle } => {
                if handle.generation != 0 {
                    return Err(ChildError::failed("invalid child generation", false));
                }
                self.vad_sessions.remove(&handle.session);
                Ok(Response::Ack)
            }
            Request::DiarizeWindow { pcm } => {
                if self.nemotron.is_none() {
                    self.nemotron = Some(sortformer::network::Network::load(
                        &self.model_root.join("nemotron"),
                    )?);
                }
                let probabilities = self
                    .nemotron
                    .as_mut()
                    .expect("loaded above")
                    .probabilities(&pcm, cancel)?;
                Ok(Response::Turns(sortformer::frames_to_turns(&probabilities)))
            }
            Request::Transcribe {
                pcm,
                language,
                padding,
            } => {
                if self.cohere.is_none() {
                    self.cohere = Some(cohere::network::Network::load(
                        &self.model_root.join("cohere-speech"),
                    )?);
                }
                self.cohere
                    .as_mut()
                    .expect("loaded above")
                    .transcribe(&pcm, &language, padding, cancel)
                    .map(Response::Text)
            }
            Request::LanguageProbe { pcm } => {
                if self.whisper.is_none() {
                    self.whisper = Some(whisper_lid::network::Network::load(
                        &self.model_root.join("whisper-language"),
                    )?);
                }
                self.whisper
                    .as_mut()
                    .expect("loaded above")
                    .probe(&pcm, cancel)
                    .map(Response::Probe)
            }
            Request::CtcScores { pcm, token_ids } => {
                if self.ctc.is_none() {
                    self.ctc = Some(ctc::network::Network::load(
                        &self.model_root.join("coarse-alignment"),
                    )?);
                }
                self.ctc
                    .as_mut()
                    .expect("loaded above")
                    .scores(&pcm, &token_ids, cancel)
                    .map(Response::Ctc)
            }
            Request::EmbedVoice { pcm } => {
                if self.voice.is_none() {
                    self.voice = Some(redimnet::coreml::Network::load(
                        &self.model_root.join("redimnet2-b6"),
                        redimnet::coreml::ComputePolicy::CpuAndGpu,
                    )?);
                }
                let network = self.voice.as_ref().expect("loaded above");
                Ok(Response::Embedding(speakerdesk_core::ipc::VoiceEmbedding {
                    model: network.identity(),
                    clip: network.embed(&pcm, cancel)?,
                }))
            }
            other => dispatch(other, cancel),
        }
    }
}

pub fn device_name() -> String {
    // Model implementations must establish readiness before advertising a backend.
    "unavailable".to_string()
}

pub fn dispatch(request: Request, cancel: &AtomicBool) -> Result<Response, ChildError> {
    if cancel.load(Ordering::Relaxed)
        && !matches!(
            request,
            Request::Cancel { .. } | Request::Shutdown | Request::Hello { .. }
        )
    {
        return Err(ChildError::Cancelled);
    }
    match request {
        Request::Hello { build_id } => Ok(Response::Ready {
            build_id,
            device: device_name(),
            models: Vec::new(),
        }),
        Request::VadOpen { .. } | Request::VadFeed { .. } | Request::VadClose { .. } => {
            Err(ChildError::Unimplemented)
        }
        Request::DiarOpen { preset } => sortformer::diar_open(preset).map(Response::Handle),
        Request::DiarFeed { handle, pcm, last } => {
            sortformer::diar_feed(handle, &pcm, last).map(Response::Diar)
        }
        Request::DiarClose { handle } => sortformer::diar_close(handle).map(|()| Response::Ack),
        Request::DiarizeFile { path } => sortformer::diarize_file(&path).map(Response::Turns),
        Request::DiarizeWindow { pcm } => sortformer::diarize_window(&pcm).map(Response::Turns),
        Request::LanguageProbe { pcm } => whisper_lid::language_probe(&pcm).map(Response::Probe),
        Request::Transcribe {
            pcm,
            language,
            padding,
        } => cohere::transcribe(&pcm, &language, padding).map(Response::Text),
        Request::CtcScores { pcm, token_ids } => {
            ctc::ctc_scores(&pcm, &token_ids).map(Response::Ctc)
        }
        Request::EmbedVoice { pcm } => redimnet::embed_voice(&pcm).map(Response::Embedding),
        Request::SetupCheck { .. } => Err(ChildError::Unimplemented),
        Request::Metrics => memory::metrics().map(Response::Metrics),
        Request::Cancel { .. } | Request::Shutdown => Ok(Response::Ack),
    }
}
