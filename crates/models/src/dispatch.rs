use std::sync::atomic::{AtomicBool, Ordering};

use speakerdesk_core::error::ChildError;
use speakerdesk_core::ipc::{Request, Response};

use crate::{cohere, ctc, memory, redimnet, silero, sortformer, whisper_lid};

pub fn device_name() -> String {
    if cfg!(target_os = "macos") {
        "metal".to_string()
    } else {
        "unavailable".to_string()
    }
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
        }),
        Request::VadOpen { conditioning } => silero::vad_open(conditioning).map(Response::Handle),
        Request::VadFeed { handle, pcm, last } => {
            silero::vad_feed(handle, &pcm, last).map(Response::Vad)
        }
        Request::VadClose { handle } => silero::vad_close(handle).map(|()| Response::Ack),
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
