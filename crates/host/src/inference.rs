//! Inference supervisor over the child stdio protocol.
//!
//! A frame is a big-endian `u32` length plus a postcard [`Envelope`]. The
//! resident child is this executable started as `speakerdesk --inference`
//! (ADR-0001). Its stdin and stdout are the streams
//! `speakerdesk_models::serve_on` reads and writes. [`inference_child_command`]
//! builds that process. Unit tests do not spawn it or MLX. They pass a
//! postcard peer to [`InferenceSupervisor::spawn_thread`] on a [`UnixStream`]
//! pair.

use std::ffi::OsStr;
use std::io::{self, Read, Write};
use std::os::unix::net::UnixStream;
use std::path::Path;
use std::process::{Command, Stdio};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Mutex, MutexGuard};
use std::thread::JoinHandle;
use std::time::Duration;

use speakerdesk_core::error::ChildError;
use speakerdesk_core::ipc::{
    AsrText, CtcScoresOut, DiarFeedOut, DiarPreset, Envelope, EnvelopeBody, GainPolicy, Handle,
    LanguageProbeOut, MAX_FRAME_BYTES, MetricsOut, Request, Response, SetupModelReport, Turn,
    VadFrame, request_timeout,
};
use speakerdesk_core::models::Models;

/// Stand-in for the duration-scaled `DiarizeFile` limit.
///
/// [`request_timeout`] returns `None` for that request. This two-hour ceiling
/// matches the legacy inference bound and is not a measurement of the file.
const DIARIZE_FILE_TIMEOUT_DOUBLE: Duration = Duration::from_secs(2 * 60 * 60);

struct Link {
    stream: UnixStream,
    child: Option<JoinHandle<()>>,
    closed: bool,
    failed: bool,
}

/// Parent end of the inference-child frame protocol.
pub struct InferenceSupervisor {
    link: Mutex<Link>,
    next_id: AtomicU64,
    /// Replaces [`request_timeout`] when a unit test needs a short deadline.
    #[cfg(test)]
    timeout_override: Mutex<Option<Duration>>,
}

/// Command line for the resident inference child.
///
/// The parent spawns this with piped stdin and stdout and speaks
/// length-prefixed postcard frames. The child loop is
/// `speakerdesk_models::serve_on`.
pub fn inference_child_command(executable: impl AsRef<OsStr>) -> Command {
    let mut command = Command::new(executable);
    command
        .arg("--inference")
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::inherit());
    command
}

impl InferenceSupervisor {
    /// Spawn `serve` on a background thread connected by a Unix socket pair.
    ///
    /// Sends [`Request::Hello`] and requires [`Response::Ready`] to echo
    /// `build_id`. Tests pass a postcard peer. The production child is a
    /// process from [`inference_child_command`], not this thread.
    pub fn spawn_thread<F>(build_id: &str, serve: F) -> Result<Self, ChildError>
    where
        F: FnOnce(UnixStream) + Send + 'static,
    {
        let (child_end, parent_end) = UnixStream::pair().map_err(io_to_child)?;
        let child = std::thread::spawn(move || serve(child_end));
        let supervisor = Self::from_stream(parent_end, child);
        match supervisor.hello(build_id) {
            Ok((ready_build, _)) if ready_build == build_id => Ok(supervisor),
            Ok((ready_build, _)) => {
                let _ = supervisor.shutdown();
                Err(ChildError::failed(
                    format!("inference child build id mismatch: {ready_build}"),
                    false,
                ))
            }
            Err(error) => {
                let _ = supervisor.shutdown();
                Err(error)
            }
        }
    }

    fn from_stream(stream: UnixStream, child: JoinHandle<()>) -> Self {
        Self {
            link: Mutex::new(Link {
                stream,
                child: Some(child),
                closed: false,
                failed: false,
            }),
            next_id: AtomicU64::new(1),
            #[cfg(test)]
            timeout_override: Mutex::new(None),
        }
    }

    fn lock(&self) -> MutexGuard<'_, Link> {
        self.link
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
    }

    fn timeout_for(&self, request: &Request) -> Duration {
        timeout_for_request(request, self.timeout_override())
    }

    #[cfg(test)]
    fn timeout_override(&self) -> Option<Duration> {
        *self
            .timeout_override
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
    }

    #[cfg(not(test))]
    fn timeout_override(&self) -> Option<Duration> {
        let _ = self;
        None
    }

    #[cfg(test)]
    fn set_timeout_override(&self, timeout: Duration) {
        *self
            .timeout_override
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner()) = Some(timeout);
    }

    fn exchange(&self, request: Request) -> Result<Response, ChildError> {
        let timeout = self.timeout_for(&request);
        let mut link = self.lock();
        if link.closed {
            return Err(ChildError::failed("inference child is closed", false));
        }
        if link.failed {
            return Err(ChildError::failed(
                "inference child connection failed",
                false,
            ));
        }
        let request_id = self.next_id.fetch_add(1, Ordering::Relaxed);
        match write_and_read(&mut link.stream, request_id, request, timeout) {
            Ok(result) => result,
            Err(error) => {
                link.failed = true;
                Err(error)
            }
        }
    }
}

fn timeout_for_request(request: &Request, override_timeout: Option<Duration>) -> Duration {
    if let Some(timeout) = override_timeout {
        return timeout;
    }
    request_timeout(request).unwrap_or(DIARIZE_FILE_TIMEOUT_DOUBLE)
}

impl Models for InferenceSupervisor {
    fn hello(&self, build_id: &str) -> Result<(String, String), ChildError> {
        match self.exchange(Request::Hello {
            build_id: build_id.to_owned(),
        })? {
            Response::Ready { build_id, device } => Ok((build_id, device)),
            other => Err(unexpected(&other)),
        }
    }

    fn vad_open(&self, conditioning: Option<GainPolicy>) -> Result<Handle, ChildError> {
        match self.exchange(Request::VadOpen { conditioning })? {
            Response::Handle(handle) => Ok(handle),
            other => Err(unexpected(&other)),
        }
    }

    fn vad_feed(
        &self,
        handle: Handle,
        pcm: &[f32],
        last: bool,
    ) -> Result<Vec<VadFrame>, ChildError> {
        match self.exchange(Request::VadFeed {
            handle,
            pcm: pcm.to_vec(),
            last,
        })? {
            Response::Vad(frames) => Ok(frames),
            other => Err(unexpected(&other)),
        }
    }

    fn vad_close(&self, handle: Handle) -> Result<(), ChildError> {
        ack(self.exchange(Request::VadClose { handle })?)
    }

    fn diar_open(&self, preset: DiarPreset) -> Result<Handle, ChildError> {
        match self.exchange(Request::DiarOpen { preset })? {
            Response::Handle(handle) => Ok(handle),
            other => Err(unexpected(&other)),
        }
    }

    fn diar_feed(
        &self,
        handle: Handle,
        pcm: &[f32],
        last: bool,
    ) -> Result<DiarFeedOut, ChildError> {
        match self.exchange(Request::DiarFeed {
            handle,
            pcm: pcm.to_vec(),
            last,
        })? {
            Response::Diar(feed) => Ok(feed),
            other => Err(unexpected(&other)),
        }
    }

    fn diar_close(&self, handle: Handle) -> Result<(), ChildError> {
        ack(self.exchange(Request::DiarClose { handle })?)
    }

    fn diarize_file(&self, path: &Path) -> Result<Vec<Turn>, ChildError> {
        match self.exchange(Request::DiarizeFile {
            path: path.to_path_buf(),
        })? {
            Response::Turns(turns) => Ok(turns),
            other => Err(unexpected(&other)),
        }
    }

    fn diarize_window(&self, pcm: &[f32]) -> Result<Vec<Turn>, ChildError> {
        match self.exchange(Request::DiarizeWindow { pcm: pcm.to_vec() })? {
            Response::Turns(turns) => Ok(turns),
            other => Err(unexpected(&other)),
        }
    }

    fn language_probe(&self, pcm: &[f32]) -> Result<LanguageProbeOut, ChildError> {
        match self.exchange(Request::LanguageProbe { pcm: pcm.to_vec() })? {
            Response::Probe(probe) => Ok(probe),
            other => Err(unexpected(&other)),
        }
    }

    fn transcribe(
        &self,
        pcm: &[f32],
        language: &str,
        padding: (u32, u32),
    ) -> Result<AsrText, ChildError> {
        match self.exchange(Request::Transcribe {
            pcm: pcm.to_vec(),
            language: language.to_owned(),
            padding,
        })? {
            Response::Text(text) => Ok(text),
            other => Err(unexpected(&other)),
        }
    }

    fn ctc_scores(&self, pcm: &[f32], token_ids: &[i32]) -> Result<CtcScoresOut, ChildError> {
        match self.exchange(Request::CtcScores {
            pcm: pcm.to_vec(),
            token_ids: token_ids.to_vec(),
        })? {
            Response::Ctc(scores) => Ok(scores),
            other => Err(unexpected(&other)),
        }
    }

    fn embed_voice(&self, pcm: &[f32]) -> Result<Vec<f32>, ChildError> {
        match self.exchange(Request::EmbedVoice { pcm: pcm.to_vec() })? {
            Response::Embedding(embedding) => Ok(embedding),
            other => Err(unexpected(&other)),
        }
    }

    fn setup_check(&self, models: &[String]) -> Result<Vec<SetupModelReport>, ChildError> {
        match self.exchange(Request::SetupCheck {
            models: models.to_vec(),
        })? {
            Response::Setup(reports) => Ok(reports),
            other => Err(unexpected(&other)),
        }
    }

    fn metrics(&self) -> Result<MetricsOut, ChildError> {
        match self.exchange(Request::Metrics)? {
            Response::Metrics(metrics) => Ok(metrics),
            other => Err(unexpected(&other)),
        }
    }

    fn cancel(&self, request_id: u64) -> Result<(), ChildError> {
        ack(self.exchange(Request::Cancel { request_id })?)
    }

    fn shutdown(&self) -> Result<(), ChildError> {
        let (child, result) = {
            let mut link = self.lock();
            if link.closed {
                return Ok(());
            }
            let result = if link.failed {
                Ok(())
            } else {
                let request_id = self.next_id.fetch_add(1, Ordering::Relaxed);
                let timeout = self.timeout_for(&Request::Shutdown);
                match write_and_read(&mut link.stream, request_id, Request::Shutdown, timeout) {
                    Ok(Ok(Response::Ack)) => Ok(()),
                    Ok(Ok(other)) => {
                        link.failed = true;
                        Err(unexpected(&other))
                    }
                    Ok(Err(error)) => Err(error),
                    Err(error) => {
                        link.failed = true;
                        Err(error)
                    }
                }
            };
            link.closed = true;
            let child = if result.is_ok() && !link.failed {
                link.child.take()
            } else {
                // A wedged or already-dead peer must not block shutdown on join.
                link.child.take();
                None
            };
            (child, result)
        };
        if let Some(child) = child
            && child.join().is_err()
        {
            return Err(ChildError::failed("inference child panicked", false));
        }
        result
    }
}

impl Drop for InferenceSupervisor {
    fn drop(&mut self) {
        let _ = self.shutdown();
    }
}

fn ack(response: Response) -> Result<(), ChildError> {
    match response {
        Response::Ack => Ok(()),
        other => Err(unexpected(&other)),
    }
}

fn unexpected(response: &Response) -> ChildError {
    ChildError::failed(
        format!("unexpected inference response: {response:?}"),
        false,
    )
}

fn io_to_child(error: io::Error) -> ChildError {
    match error.kind() {
        io::ErrorKind::TimedOut | io::ErrorKind::WouldBlock => ChildError::Timeout,
        _ => ChildError::failed(error.to_string(), true),
    }
}

fn write_and_read(
    stream: &mut UnixStream,
    request_id: u64,
    request: Request,
    timeout: Duration,
) -> Result<Result<Response, ChildError>, ChildError> {
    stream
        .set_read_timeout(Some(timeout))
        .map_err(io_to_child)?;
    stream
        .set_write_timeout(Some(timeout))
        .map_err(io_to_child)?;
    write_envelope(
        stream,
        &Envelope {
            request_id,
            body: EnvelopeBody::Request(request),
        },
    )?;
    let envelope = read_envelope(stream)?;
    if envelope.request_id != request_id {
        return Err(ChildError::failed(
            "inference response request_id mismatch",
            false,
        ));
    }
    match envelope.body {
        EnvelopeBody::Response(result) => Ok(result),
        EnvelopeBody::Request(_) => Err(ChildError::failed(
            "inference child sent a request frame",
            false,
        )),
    }
}

fn read_envelope(reader: &mut impl Read) -> Result<Envelope, ChildError> {
    let mut length_bytes = [0u8; 4];
    reader.read_exact(&mut length_bytes).map_err(io_to_child)?;
    let length = usize::try_from(u32::from_be_bytes(length_bytes))
        .map_err(|error| ChildError::failed(error.to_string(), false))?;
    if length > MAX_FRAME_BYTES {
        return Err(ChildError::failed("frame exceeds 64 MiB", false));
    }
    let mut payload = vec![0u8; length];
    reader.read_exact(&mut payload).map_err(io_to_child)?;
    postcard::from_bytes(&payload).map_err(|error| ChildError::failed(error.to_string(), false))
}

fn write_envelope(writer: &mut impl Write, envelope: &Envelope) -> Result<(), ChildError> {
    let payload = postcard::to_allocvec(envelope)
        .map_err(|error| ChildError::failed(error.to_string(), false))?;
    if payload.len() > MAX_FRAME_BYTES {
        return Err(ChildError::failed("frame exceeds 64 MiB", false));
    }
    let length = u32::try_from(payload.len())
        .map_err(|error| ChildError::failed(error.to_string(), false))?;
    writer
        .write_all(&length.to_be_bytes())
        .map_err(io_to_child)?;
    writer.write_all(&payload).map_err(io_to_child)?;
    writer.flush().map_err(io_to_child)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use std::io::{Read, Write};
    use std::os::unix::net::UnixStream;
    use std::path::PathBuf;
    use std::sync::atomic::{AtomicBool, Ordering};
    use std::sync::{Arc, Mutex};
    use std::thread;
    use std::time::Duration;

    use speakerdesk_core::error::ChildError;
    use speakerdesk_core::ipc::{
        Envelope, EnvelopeBody, MAX_FRAME_BYTES, Request, Response, request_timeout,
    };
    use speakerdesk_core::models::Models;

    use super::{
        DIARIZE_FILE_TIMEOUT_DOUBLE, InferenceSupervisor, inference_child_command,
        timeout_for_request,
    };

    #[test]
    fn hello_checks_ready_build_and_device() {
        let seen = Arc::new(Mutex::new(Vec::new()));
        let record = Arc::clone(&seen);
        let supervisor = InferenceSupervisor::spawn_thread("build-7", move |stream| {
            serve_fake(stream, Some(record), None);
        })
        .unwrap();
        let requests = seen.lock().unwrap().clone();
        assert!(
            matches!(requests.first(), Some(Request::Hello { build_id }) if build_id == "build-7")
        );
        let (build_id, device) = supervisor.hello("build-7").unwrap();
        assert_eq!(build_id, "build-7");
        assert_eq!(device, "unavailable");
        supervisor.shutdown().unwrap();
    }

    #[test]
    fn transcribe_returns_unimplemented() {
        let supervisor =
            InferenceSupervisor::spawn_thread("build", |stream| serve_fake(stream, None, None))
                .unwrap();
        let error = supervisor.transcribe(&[], "en", (0, 0)).unwrap_err();
        assert_eq!(error, ChildError::Unimplemented);
        supervisor.shutdown().unwrap();
    }

    #[test]
    fn shutdown_writes_shutdown_and_joins() {
        let joined = Arc::new(AtomicBool::new(false));
        let flag = Arc::clone(&joined);
        let supervisor = InferenceSupervisor::spawn_thread("build", move |stream| {
            serve_fake(stream, None, Some(flag));
        })
        .unwrap();
        supervisor.shutdown().unwrap();
        assert!(joined.load(Ordering::Acquire));
        supervisor.shutdown().unwrap();
    }

    #[test]
    fn timed_out_response_is_child_timeout() {
        let supervisor = InferenceSupervisor::spawn_thread("build", hang_after_hello).unwrap();
        supervisor.set_timeout_override(Duration::from_millis(200));
        let error = supervisor.metrics().unwrap_err();
        assert_eq!(error, ChildError::Timeout);
    }

    #[test]
    fn diarize_file_uses_the_generous_timeout_double() {
        let hello = Request::Hello {
            build_id: "b".into(),
        };
        assert_eq!(
            timeout_for_request(&hello, None),
            request_timeout(&hello).unwrap()
        );
        let file = Request::DiarizeFile {
            path: PathBuf::from("meeting.wav"),
        };
        assert!(request_timeout(&file).is_none());
        assert_eq!(
            timeout_for_request(&file, None),
            DIARIZE_FILE_TIMEOUT_DOUBLE
        );
    }

    #[test]
    fn documents_the_inference_process_spawn() {
        let command = inference_child_command("speakerdesk");
        assert_eq!(command.get_program(), "speakerdesk");
        let args: Vec<_> = command.get_args().collect();
        assert_eq!(args, ["--inference"]);
    }

    fn serve_fake(
        mut stream: UnixStream,
        seen: Option<Arc<Mutex<Vec<Request>>>>,
        joined: Option<Arc<AtomicBool>>,
    ) {
        loop {
            let envelope = match read_frame(&mut stream) {
                Ok(envelope) => envelope,
                Err(error) if error.kind() == std::io::ErrorKind::UnexpectedEof => return,
                Err(error) => panic!("fake child read failed: {error}"),
            };
            let EnvelopeBody::Request(request) = envelope.body else {
                panic!("fake child received a response frame");
            };
            if let Some(seen) = &seen {
                seen.lock().unwrap().push(request.clone());
            }
            let shutting_down = matches!(request, Request::Shutdown);
            write_frame(
                &mut stream,
                &Envelope {
                    request_id: envelope.request_id,
                    body: fake_response(request),
                },
            )
            .unwrap();
            if shutting_down {
                if let Some(joined) = joined {
                    thread::sleep(Duration::from_millis(80));
                    joined.store(true, Ordering::Release);
                }
                return;
            }
        }
    }

    fn hang_after_hello(mut stream: UnixStream) {
        let hello = read_frame(&mut stream).unwrap();
        let EnvelopeBody::Request(Request::Hello { build_id }) = hello.body else {
            panic!("expected hello, got {:?}", hello.body);
        };
        write_frame(
            &mut stream,
            &Envelope {
                request_id: hello.request_id,
                body: EnvelopeBody::Response(Ok(Response::Ready {
                    build_id,
                    device: "unavailable".into(),
                })),
            },
        )
        .unwrap();
        let _request = read_frame(&mut stream).unwrap();
        let mut byte = [0u8; 1];
        let _ = stream.read(&mut byte);
    }

    fn fake_response(request: Request) -> EnvelopeBody {
        match request {
            Request::Hello { build_id } => EnvelopeBody::Response(Ok(Response::Ready {
                build_id,
                device: "unavailable".into(),
            })),
            Request::Shutdown => EnvelopeBody::Response(Ok(Response::Ack)),
            Request::Transcribe { .. } => EnvelopeBody::Response(Err(ChildError::Unimplemented)),
            _ => EnvelopeBody::Response(Err(ChildError::Unimplemented)),
        }
    }

    /// Independent length-prefixed postcard reader, matching `serve_on`.
    fn read_frame(stream: &mut UnixStream) -> std::io::Result<Envelope> {
        let mut length_bytes = [0u8; 4];
        stream.read_exact(&mut length_bytes)?;
        let length = usize::try_from(u32::from_be_bytes(length_bytes)).unwrap();
        assert!(length <= MAX_FRAME_BYTES);
        let mut payload = vec![0u8; length];
        stream.read_exact(&mut payload)?;
        postcard::from_bytes(&payload).map_err(|error| {
            std::io::Error::new(std::io::ErrorKind::InvalidData, error.to_string())
        })
    }

    fn write_frame(stream: &mut UnixStream, envelope: &Envelope) -> std::io::Result<()> {
        let payload = postcard::to_allocvec(envelope).map_err(|error| {
            std::io::Error::new(std::io::ErrorKind::InvalidData, error.to_string())
        })?;
        assert!(payload.len() <= MAX_FRAME_BYTES);
        let length = u32::try_from(payload.len()).unwrap();
        stream.write_all(&length.to_be_bytes())?;
        stream.write_all(&payload)?;
        stream.flush()?;
        Ok(())
    }
}
