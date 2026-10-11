use std::sync::atomic::{AtomicBool, Ordering};

use speakerdesk_core::error::ChildError;
use speakerdesk_core::ipc::{Request, Response};

use crate::{cohere, ctc, memory, redimnet, sortformer, whisper_lid};

#[cfg(target_os = "macos")]
pub struct Runtime {
    model_root: std::path::PathBuf,
    silero: Option<crate::silero::network::Network>,
    vad_sessions: std::collections::HashMap<u64, crate::silero::session::Session>,
    next_session: u64,
}

#[cfg(target_os = "macos")]
impl Runtime {
    pub fn new(model_root: std::path::PathBuf) -> Result<Self, ChildError> {
        crate::metallib::prepare(None)?;
        Ok(Self {
            model_root,
            silero: None,
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
        match request {
            Request::Hello { .. } => Ok(Response::Ready {
                build_id: env!("CARGO_PKG_VERSION").into(),
                device: "metal".into(),
                models: if self.silero.is_some() {
                    vec![crate::silero::WEIGHT_PIN.into()]
                } else {
                    vec![]
                },
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
