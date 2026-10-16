use std::sync::atomic::{AtomicBool, Ordering};

use speakerdesk_core::error::ChildError;
use speakerdesk_core::ipc::{Request, Response};

use crate::{cohere, ctc, memory, redimnet, sortformer, whisper_lid};

#[cfg(target_os = "macos")]
use redimnet::coreml::Network as VoiceNetwork;
#[cfg(windows)]
use redimnet::network_windows::Network as VoiceNetwork;

#[cfg(any(target_os = "macos", windows))]
macro_rules! load_network {
    ($runtime:expr, $network:ty, $model:expr) => {{
        let directory = $runtime.model_root.join($model);
        #[cfg(target_os = "macos")]
        {
            <$network>::load(&directory)
        }
        #[cfg(windows)]
        {
            <$network>::load(&directory, &$runtime.backend)
        }
    }};
}

#[cfg(target_os = "macos")]
const ALIGNMENT_CHECKPOINT: &str = speakerdesk_core::alignment::CHECKPOINT;
#[cfg(windows)]
const ALIGNMENT_CHECKPOINT: &str = speakerdesk_core::timing_policy::WINDOWS.checkpoint;
#[cfg(target_os = "macos")]
const RUNTIME_DEVICE: &str = "metal";
#[cfg(windows)]
const RUNTIME_DEVICE: &str = "directml";

#[cfg(any(target_os = "macos", windows))]
pub struct Runtime {
    model_root: std::path::PathBuf,
    silero: Option<crate::silero::network::Network>,
    nemotron: Option<sortformer::network::Network>,
    cohere: Option<cohere::network::Network>,
    whisper: Option<whisper_lid::network::Network>,
    ctc: Option<ctc::network::Network>,
    voice: Option<VoiceNetwork>,
    separation: Option<crate::mossformer2::network::Network>,
    separation_session: Option<(u64, crate::mossformer2::network::Session)>,
    transcription_session: Option<(u64, cohere::network::Session)>,
    vad_sessions: std::collections::HashMap<u64, crate::silero::session::Session>,
    diar_sessions: std::collections::HashMap<u64, sortformer::session::Session>,
    next_session: u64,
    // Drop native sessions and owned tensors before releasing private DLL search state.
    #[cfg(windows)]
    backend: crate::windows_runtime::Backend,
}

#[cfg(any(target_os = "macos", windows))]
impl Runtime {
    fn load_voice(&self) -> Result<VoiceNetwork, ChildError> {
        let directory = self.model_root.join("redimnet2-b6");
        #[cfg(target_os = "macos")]
        {
            VoiceNetwork::load(&directory, redimnet::coreml::ComputePolicy::CpuAndGpu)
        }
        #[cfg(windows)]
        {
            VoiceNetwork::load(&directory, &self.backend)
        }
    }

    pub fn new(model_root: std::path::PathBuf) -> Result<Self, ChildError> {
        #[cfg(target_os = "macos")]
        {
            crate::metallib::prepare(None)?;
            // Weights stay resident; released intermediate buffers need not stay cached.
            mlx_rs::memory::set_cache_limit(0).map_err(crate::weights::error)?;
        }
        #[cfg(windows)]
        let backend = crate::windows_runtime::Backend::new()?;
        Ok(Self {
            model_root,
            silero: None,
            nemotron: None,
            cohere: None,
            whisper: None,
            ctc: None,
            voice: None,
            separation: None,
            separation_session: None,
            transcription_session: None,
            vad_sessions: Default::default(),
            diar_sessions: Default::default(),
            next_session: 1,
            #[cfg(windows)]
            backend,
        })
    }

    pub fn execute(
        &mut self,
        request: Request,
        cancel: &crate::operation::Operation,
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
                            self.silero =
                                Some(load_network!(self, crate::silero::network::Network, model)?);
                        }
                        crate::silero::WEIGHT_PIN.to_string()
                    }
                    "nemotron" => {
                        if self.nemotron.is_none() {
                            self.nemotron =
                                Some(load_network!(self, sortformer::network::Network, model)?);
                        }
                        sortformer::WEIGHT_PIN.to_string()
                    }
                    "cohere-speech" => {
                        if self.cohere.is_none() {
                            self.cohere =
                                Some(load_network!(self, cohere::network::Network, model)?);
                        }
                        cohere::WEIGHT_PIN.to_string()
                    }
                    "whisper-language" => {
                        if self.whisper.is_none() {
                            self.whisper =
                                Some(load_network!(self, whisper_lid::network::Network, model)?);
                        }
                        whisper_lid::WEIGHT_PIN.to_string()
                    }
                    "coarse-alignment" => {
                        if self.ctc.is_none() {
                            self.ctc = Some(load_network!(self, ctc::network::Network, model)?);
                        }
                        ALIGNMENT_CHECKPOINT.to_string()
                    }
                    "redimnet2-b6" => {
                        if self.voice.is_none() {
                            self.voice = Some(self.load_voice()?);
                        }
                        self.voice
                            .as_ref()
                            .expect("loaded above")
                            .identity()
                            .revision
                    }
                    "mossformer2-16k" => {
                        if self.separation.is_none() {
                            self.separation = Some(load_network!(
                                self,
                                crate::mossformer2::network::Network,
                                model
                            )?);
                        }
                        speakerdesk_core::separation::MODEL.into()
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
            Request::PrepareAlignment | Request::PrepareSeparation => {
                #[cfg(windows)]
                {
                    if matches!(request, Request::PrepareAlignment) {
                        ctc::prepare_windows::convert_cancellable(
                            &self.model_root.join(".staging/coarse-alignment"),
                            cancel,
                        )?;
                        Ok(Response::Ack)
                    } else {
                        Err(ChildError::failed(
                            "Windows separation requires downloaded prebuilt artifacts",
                            false,
                        ))
                    }
                }
                #[cfg(target_os = "macos")]
                {
                    let separation = matches!(request, Request::PrepareSeparation);
                    let directory = self.model_root.join(if separation {
                        ".staging/mossformer2-16k"
                    } else {
                        ".staging/coarse-alignment"
                    });
                    // The parent owns this private staging directory; never follow an old partial link.
                    let output = directory.join("model.safetensors.part");
                    match std::fs::symlink_metadata(&output) {
                        Ok(meta) if !meta.is_file() || meta.is_symlink() => {
                            return Err(crate::weights::error(
                                "Model staging output is not a regular file",
                            ));
                        }
                        Ok(_) => std::fs::remove_file(&output).map_err(crate::weights::error)?,
                        Err(error) if error.kind() == std::io::ErrorKind::NotFound => {}
                        Err(error) => return Err(crate::weights::error(error)),
                    }
                    if separation {
                        crate::mossformer2::prepare::convert_cancellable(
                            &directory, &output, cancel,
                        )?;
                    } else {
                        ctc::prepare::convert_cancellable(&directory, &output, cancel)?;
                    }
                    if cancel.load(Ordering::Acquire) {
                        return Err(ChildError::Cancelled);
                    }
                    std::fs::rename(output, directory.join("model.safetensors"))
                        .map_err(crate::weights::error)?;
                    Ok(Response::Ack)
                }
            }
            Request::Hello { .. } => Ok(Response::Ready {
                build_id: env!("CARGO_PKG_VERSION").into(),
                device: RUNTIME_DEVICE.into(),
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
                    self.ctc.as_ref().map(|_| ALIGNMENT_CHECKPOINT.into()),
                    self.voice
                        .as_ref()
                        .map(|network| network.identity().revision),
                    self.separation
                        .as_ref()
                        .map(|_| speakerdesk_core::separation::MODEL.into()),
                ]
                .into_iter()
                .flatten()
                .collect(),
            }),
            Request::VadOpen { conditioning } => {
                let session = crate::silero::session::Session::new(conditioning)?;
                if self.silero.is_none() {
                    self.silero = Some(load_network!(
                        self,
                        crate::silero::network::Network,
                        "silero-speech"
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
            Request::DiarOpen { preset } => {
                if self.nemotron.is_none() {
                    self.nemotron = Some(load_network!(
                        self,
                        sortformer::network::Network,
                        "nemotron"
                    )?);
                }
                let session = sortformer::session::Session::new(preset)?;
                let id = self.next_session;
                self.next_session = id
                    .checked_add(1)
                    .ok_or_else(|| ChildError::failed("model session id exhausted", false))?;
                self.diar_sessions.insert(id, session);
                Ok(Response::Handle(Handle {
                    generation: 0,
                    session: id,
                }))
            }
            Request::DiarFeed { handle, pcm, last } => {
                if handle.generation != 0 {
                    return Err(ChildError::failed("invalid child generation", false));
                }
                let session = self
                    .diar_sessions
                    .get_mut(&handle.session)
                    .ok_or_else(|| ChildError::failed("unknown Nemotron session", false))?;
                let result = session.feed(
                    self.nemotron.as_mut().expect("loaded before open"),
                    &pcm,
                    last,
                    cancel,
                );
                if result.is_err() {
                    self.diar_sessions.remove(&handle.session);
                }
                result.map(Response::Diar)
            }
            Request::DiarClose { handle } => {
                if handle.generation != 0 {
                    return Err(ChildError::failed("invalid child generation", false));
                }
                self.diar_sessions.remove(&handle.session);
                Ok(Response::Ack)
            }
            Request::Transcribe {
                pcm,
                language,
                padding,
            } => {
                if self.cohere.is_none() {
                    self.cohere = Some(load_network!(
                        self,
                        cohere::network::Network,
                        "cohere-speech"
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
                    self.whisper = Some(load_network!(
                        self,
                        whisper_lid::network::Network,
                        "whisper-language"
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
                    self.ctc = Some(load_network!(
                        self,
                        ctc::network::Network,
                        "coarse-alignment"
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
                    self.voice = Some(self.load_voice()?);
                }
                let network = self.voice.as_mut().expect("loaded above");
                Ok(Response::Embedding(speakerdesk_core::ipc::VoiceEmbedding {
                    model: network.identity(),
                    clip: network.embed(&pcm, cancel)?,
                }))
            }
            Request::SeparateOpen { pcm } => {
                if self.separation_session.is_some() {
                    return Err(ChildError::failed("separation session already open", false));
                }
                if self.separation.is_none() {
                    self.separation = Some(load_network!(
                        self,
                        crate::mossformer2::network::Network,
                        "mossformer2-16k"
                    )?);
                }
                let session = self
                    .separation
                    .as_mut()
                    .expect("loaded")
                    .begin(&pcm, cancel)?;
                let id = self.next_session;
                self.next_session = id
                    .checked_add(1)
                    .ok_or_else(|| ChildError::failed("model session id exhausted", false))?;
                self.separation_session = Some((id, session));
                Ok(Response::Handle(Handle {
                    generation: 0,
                    session: id,
                }))
            }
            Request::SeparateStep { handle } => {
                if handle.generation != 0 {
                    return Err(ChildError::failed("invalid child generation", false));
                }
                let (id, mut session) = self
                    .separation_session
                    .take()
                    .ok_or_else(|| ChildError::failed("unknown separation session", false))?;
                if id != handle.session {
                    self.separation_session = Some((id, session));
                    return Err(ChildError::failed("unknown separation session", false));
                }
                let result = self
                    .separation
                    .as_mut()
                    .expect("loaded")
                    .step(&mut session, cancel)?;
                if result.is_none() {
                    self.separation_session = Some((id, session));
                }
                Ok(Response::SeparationProgress(result.map(|channels| {
                    speakerdesk_core::ipc::SeparatedAudio {
                        model: speakerdesk_core::separation::MODEL.into(),
                        channels,
                    }
                })))
            }
            Request::SeparateClose { handle } => {
                if handle.generation != 0 {
                    return Err(ChildError::failed("invalid child generation", false));
                }
                if self
                    .separation_session
                    .as_ref()
                    .is_some_and(|(id, _)| *id == handle.session)
                {
                    self.separation_session = None;
                }
                Ok(Response::Ack)
            }
            Request::TranscribeOpen {
                pcm,
                language,
                padding,
            } => {
                if self.transcription_session.is_some() {
                    return Err(ChildError::failed(
                        "transcription session already open",
                        false,
                    ));
                }
                if self.cohere.is_none() {
                    self.cohere = Some(load_network!(
                        self,
                        cohere::network::Network,
                        "cohere-speech"
                    )?);
                }
                let session = self
                    .cohere
                    .as_mut()
                    .expect("loaded")
                    .begin(&pcm, &language, padding, cancel)?;
                let id = self.next_session;
                self.next_session = id
                    .checked_add(1)
                    .ok_or_else(|| ChildError::failed("model session id exhausted", false))?;
                self.transcription_session = Some((id, session));
                Ok(Response::Handle(Handle {
                    generation: 0,
                    session: id,
                }))
            }
            Request::TranscribeStep { handle } => {
                if handle.generation != 0 {
                    return Err(ChildError::failed("invalid child generation", false));
                }
                let (id, mut session) = self
                    .transcription_session
                    .take()
                    .ok_or_else(|| ChildError::failed("unknown transcription session", false))?;
                if id != handle.session {
                    self.transcription_session = Some((id, session));
                    return Err(ChildError::failed("unknown transcription session", false));
                }
                let result = self
                    .cohere
                    .as_mut()
                    .expect("loaded")
                    .step(&mut session, cancel)?;
                if result.is_none() {
                    self.transcription_session = Some((id, session));
                }
                Ok(Response::TranscriptionProgress(result))
            }
            Request::TranscribeClose { handle } => {
                if handle.generation != 0 {
                    return Err(ChildError::failed("invalid child generation", false));
                }
                if self
                    .transcription_session
                    .as_ref()
                    .is_some_and(|(id, _)| *id == handle.session)
                {
                    self.transcription_session = None;
                }
                Ok(Response::Ack)
            }
            Request::Separate { pcm } => {
                if self.separation_session.is_some() {
                    return Err(ChildError::failed("separation session already open", false));
                }
                if self.separation.is_none() {
                    self.separation = Some(load_network!(
                        self,
                        crate::mossformer2::network::Network,
                        "mossformer2-16k"
                    )?);
                }
                let channels = self
                    .separation
                    .as_mut()
                    .expect("loaded above")
                    .separate(&pcm, cancel)?;
                Ok(Response::Separated(speakerdesk_core::ipc::SeparatedAudio {
                    model: speakerdesk_core::separation::MODEL.into(),
                    channels,
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
        Request::SetupCheck { .. }
        | Request::PrepareAlignment
        | Request::PrepareSeparation
        | Request::Separate { .. }
        | Request::SeparateOpen { .. }
        | Request::SeparateStep { .. }
        | Request::SeparateClose { .. }
        | Request::TranscribeOpen { .. }
        | Request::TranscribeStep { .. }
        | Request::TranscribeClose { .. } => Err(ChildError::Unimplemented),
        Request::Metrics => memory::metrics().map(Response::Metrics),
        Request::Cancel { .. } | Request::Shutdown => Ok(Response::Ack),
    }
}
