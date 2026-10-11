//! One resident process, portable pipes, and a serial parent queue.
//! Model calls run on blocking workers. Lifecycle and operation cancellation belong here.
use std::io;
use std::path::{Path, PathBuf};
use std::process::Stdio;
use std::sync::atomic::{AtomicU32, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use speakerdesk_core::error::ChildError;
use speakerdesk_core::ipc::{
    AsrText, CtcScoresOut, DiarFeedOut, DiarPreset, Envelope, EnvelopeBody, GainPolicy, Handle,
    LanguageProbeOut, MAX_FRAME_BYTES, MetricsOut, Request, Response, SetupModelReport, Turn,
    VadFrame, request_timeout,
};
use speakerdesk_core::models::Models;
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::process::{Child, ChildStdin, ChildStdout, Command};
use tokio::sync::{mpsc, oneshot};
use tokio_util::sync::CancellationToken;

const CONTROL_TIMEOUT: Duration = Duration::from_secs(10);
const FILE_TIMEOUT: Duration = Duration::from_secs(2 * 60 * 60);

struct Call {
    request: Request,
    cancellation: CancellationToken,
    reply: oneshot::Sender<Result<Response, ChildError>>,
}

pub struct InferenceSupervisor {
    calls: mpsc::Sender<Call>,
    stop: CancellationToken,
    worker: Mutex<Option<std::thread::JoinHandle<()>>>,
    pid: Arc<AtomicU32>,
}

/// A model view belongs to one operation; cancelling it cannot poison another operation.
pub struct ChildModels {
    supervisor: Arc<InferenceSupervisor>,
    cancellation: CancellationToken,
}

impl InferenceSupervisor {
    pub fn new(
        executable: impl Into<PathBuf>,
        build_id: impl Into<String>,
    ) -> Result<Arc<Self>, ChildError> {
        let (calls, receiver) = mpsc::channel(1);
        let stop = CancellationToken::new();
        let pid = Arc::new(AtomicU32::new(0));
        let runtime = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .map_err(io_to_child)?;
        let executable = executable.into();
        let build_id = build_id.into();
        let actor_stop = stop.clone();
        let actor_pid = pid.clone();
        let worker = std::thread::Builder::new()
            .name("inference-control".into())
            .spawn(move || {
                runtime.block_on(run(receiver, actor_stop, actor_pid, executable, build_id));
            })
            .map_err(io_to_child)?;
        Ok(Arc::new(Self {
            calls,
            stop,
            worker: Mutex::new(Some(worker)),
            pid,
        }))
    }

    pub fn models(self: &Arc<Self>, cancellation: CancellationToken) -> ChildModels {
        ChildModels {
            supervisor: self.clone(),
            cancellation,
        }
    }

    pub fn child_pid(&self) -> Option<u32> {
        match self.pid.load(Ordering::Acquire) {
            0 => None,
            pid => Some(pid),
        }
    }

    /// Waits for the owned process to be reaped. Call off the async executor.
    pub fn shutdown(&self) {
        self.stop.cancel();
        if let Some(worker) = self.worker.lock().expect("inference worker").take() {
            let _ = worker.join();
        }
    }
}

impl Drop for InferenceSupervisor {
    fn drop(&mut self) {
        self.shutdown();
    }
}

impl ChildModels {
    fn exchange(&self, request: Request) -> Result<Response, ChildError> {
        // Session cleanup remains available after its owning operation is cancelled.
        let cancellation = if matches!(
            request,
            Request::VadClose { .. } | Request::DiarClose { .. }
        ) {
            CancellationToken::new()
        } else {
            self.cancellation.clone()
        };
        if cancellation.is_cancelled() || self.supervisor.stop.is_cancelled() {
            return Err(ChildError::Cancelled);
        }
        let (reply, receive) = oneshot::channel();
        self.supervisor
            .calls
            .blocking_send(Call {
                request,
                cancellation,
                reply,
            })
            .map_err(|_| ChildError::failed("inference supervisor stopped", false))?;
        receive
            .blocking_recv()
            .map_err(|_| ChildError::failed("inference supervisor stopped", true))?
    }
}

impl Models for ChildModels {
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
}

struct Process {
    child: Child,
    input: ChildStdin,
    replies: mpsc::Receiver<Result<Envelope, ChildError>>,
    reader: tokio::task::JoinHandle<()>,
    generation: u64,
}

impl Process {
    async fn start(executable: &Path, generation: u64) -> Result<Self, ChildError> {
        let mut child = Command::new(executable)
            .arg("--inference")
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::inherit())
            .kill_on_drop(true)
            .spawn()
            .map_err(io_to_child)?;
        let input = child
            .stdin
            .take()
            .ok_or_else(|| ChildError::failed("child stdin unavailable", false))?;
        let output = child
            .stdout
            .take()
            .ok_or_else(|| ChildError::failed("child stdout unavailable", false))?;
        let (send, replies) = mpsc::channel(2);
        let reader = tokio::spawn(read_responses(output, send));
        Ok(Self {
            child,
            input,
            replies,
            reader,
            generation,
        })
    }

    async fn send(&mut self, id: u64, request: Request) -> Result<(), ChildError> {
        let bytes = postcard::to_allocvec(&Envelope {
            request_id: id,
            body: EnvelopeBody::Request(request),
        })
        .map_err(|e| ChildError::failed(e.to_string(), false))?;
        if bytes.len() > MAX_FRAME_BYTES {
            return Err(ChildError::failed("frame exceeds 64 MiB", false));
        }
        self.input
            .write_u32(bytes.len() as u32)
            .await
            .map_err(io_to_child)?;
        self.input.write_all(&bytes).await.map_err(io_to_child)?;
        self.input.flush().await.map_err(io_to_child)
    }

    async fn response(&mut self, id: u64) -> Result<Result<Response, ChildError>, ChildError> {
        let envelope = self
            .replies
            .recv()
            .await
            .ok_or_else(|| ChildError::failed("inference child exited", true))??;
        if envelope.request_id != id {
            return Err(ChildError::failed(
                "inference response request_id mismatch",
                true,
            ));
        }
        match envelope.body {
            EnvelopeBody::Response(result) => Ok(result),
            _ => Err(ChildError::failed("child sent a request", true)),
        }
    }

    async fn reap(mut self) {
        if self.child.try_wait().ok().flatten().is_none() {
            let _ = self.child.kill().await;
        }
        let _ = self.child.wait().await;
        self.reader.abort();
        let _ = self.reader.await;
    }
}

async fn read_responses(mut output: ChildStdout, send: mpsc::Sender<Result<Envelope, ChildError>>) {
    loop {
        let result = async {
            let len = output.read_u32().await.map_err(io_to_child)? as usize;
            if len > MAX_FRAME_BYTES {
                return Err(ChildError::failed("frame exceeds 64 MiB", false));
            }
            let mut payload = vec![0; len];
            output.read_exact(&mut payload).await.map_err(io_to_child)?;
            speakerdesk_core::ipc::decode_envelope(&payload).map_err(io_to_child)
        }
        .await;
        let failed = result.is_err();
        if send.send(result).await.is_err() || failed {
            break;
        }
    }
}

async fn run(
    mut calls: mpsc::Receiver<Call>,
    stop: CancellationToken,
    pid: Arc<AtomicU32>,
    executable: PathBuf,
    build_id: String,
) {
    let mut process: Option<Process> = None;
    let mut generation = 0_u64;
    let mut next_id = 1_u64;
    loop {
        let call = tokio::select! { biased; _ = stop.cancelled() => break, call = calls.recv() => match call {Some(call) => call,None => break} };
        if call.cancellation.is_cancelled() {
            let _ = call.reply.send(Err(ChildError::Cancelled));
            continue;
        }
        if process.is_none() {
            generation += 1;
            let started = Process::start(&executable, generation).await;
            match started {
                Ok(mut child) => {
                    pid.store(child.child.id().unwrap_or(0), Ordering::Release);
                    let hello_id = next_id;
                    next_id += 1;
                    let hello = tokio::time::timeout(CONTROL_TIMEOUT, async {
                        child
                            .send(
                                hello_id,
                                Request::Hello {
                                    build_id: build_id.clone(),
                                },
                            )
                            .await?;
                        child.response(hello_id).await
                    })
                    .await;
                    match hello {
                        Ok(Ok(Ok(Response::Ready {
                            build_id: ready, ..
                        }))) if ready == build_id => process = Some(child),
                        _ => {
                            child.reap().await;
                            pid.store(0, Ordering::Release);
                            let _ = call
                                .reply
                                .send(Err(ChildError::failed("inference handshake failed", true)));
                            continue;
                        }
                    }
                }
                Err(error) => {
                    let _ = call.reply.send(Err(error));
                    continue;
                }
            }
        }
        let child = process.as_mut().expect("started child");
        let mut request = call.request;
        if let Some(handle) = request_handle(&mut request) {
            if handle.generation != child.generation {
                let _ = call.reply.send(Err(ChildError::failed(
                    "stale model session after child restart",
                    false,
                )));
                continue;
            }
            handle.generation = 0;
        }
        let id = next_id;
        next_id += 1;
        let timeout = request_timeout(&request).unwrap_or(FILE_TIMEOUT);
        let result = tokio::select! {
            _ = stop.cancelled() => Err(ChildError::Cancelled),
            result = execute(child, id, request, timeout, &call.cancellation, &mut next_id) => result,
        };
        let result = match result {
            Ok(Ok(Response::Handle(mut handle))) => {
                handle.generation = generation;
                Ok(Response::Handle(handle))
            }
            Ok(result) => result,
            Err(error) => {
                process.take().expect("child").reap().await;
                pid.store(0, Ordering::Release);
                Err(error)
            }
        };
        let _ = call.reply.send(result);
    }
    if let Some(mut child) = process.take() {
        let id = next_id;
        let _ = tokio::time::timeout(CONTROL_TIMEOUT, async {
            child.send(id, Request::Shutdown).await?;
            child.response(id).await
        })
        .await;
        child.reap().await;
        pid.store(0, Ordering::Release);
    }
}

async fn execute(
    child: &mut Process,
    id: u64,
    request: Request,
    timeout: Duration,
    cancellation: &CancellationToken,
    next_id: &mut u64,
) -> Result<Result<Response, ChildError>, ChildError> {
    tokio::time::timeout(timeout, async {
        let cleanup = match &request {
            Request::VadFeed { handle, .. } => Some(Request::VadClose { handle: *handle }),
            Request::DiarFeed { handle, .. } => Some(Request::DiarClose { handle: *handle }),
            _ => None,
        };
        let vad_open = matches!(request, Request::VadOpen { .. });
        let diar_open = matches!(request, Request::DiarOpen { .. });
        tokio::select! {
            // A partial frame cannot be followed by a Cancel frame. Reap this child.
            biased;
            _ = cancellation.cancelled() => return Err(ChildError::Cancelled),
            result = child.send(id, request) => result?,
        }
        let result = tokio::select! {
            result = child.response(id) => result?,
            _ = cancellation.cancelled() => {
                let cancel_id = *next_id; *next_id += 1;
                tokio::time::timeout(CONTROL_TIMEOUT, async {
                    child.send(cancel_id, Request::Cancel { request_id: id }).await?;
                    let mut acked = false;
                    let mut terminal = None;
                    while !acked || terminal.is_none() {
                        let envelope = child.replies.recv().await.ok_or_else(|| ChildError::failed("child exited during cancellation", true))??;
                        match (envelope.request_id, envelope.body) {
                            (received, EnvelopeBody::Response(result)) if received == id && terminal.is_none() => terminal = Some(result),
                            (received, EnvelopeBody::Response(Ok(Response::Ack))) if received == cancel_id && !acked => acked = true,
                            _ => return Err(ChildError::failed("unexpected cancellation response", true)),
                        }
                    }
                    Ok(terminal.expect("terminal reply"))
                }).await.unwrap_or(Err(ChildError::Timeout))?
            }
        };
        if cancellation.is_cancelled() {
            // A successful response can race Cancel after the child considers work complete.
            let close = match &result {
                Ok(Response::Handle(handle)) if vad_open => Some(Request::VadClose { handle: *handle }),
                Ok(Response::Handle(handle)) if diar_open => Some(Request::DiarClose { handle: *handle }),
                Err(ChildError::Cancelled) => None,
                _ => cleanup,
            };
            if let Some(close) = close {
                let close_id = *next_id; *next_id += 1;
                tokio::time::timeout(CONTROL_TIMEOUT, async {
                    child.send(close_id, close).await?;
                    ack(child.response(close_id).await??)
                }).await.unwrap_or(Err(ChildError::Timeout))?;
            }
            Ok(Err(ChildError::Cancelled))
        } else {
            Ok(result)
        }
    }).await.unwrap_or(Err(ChildError::Timeout))
}

fn request_handle(request: &mut Request) -> Option<&mut Handle> {
    match request {
        Request::VadFeed { handle, .. }
        | Request::VadClose { handle }
        | Request::DiarFeed { handle, .. }
        | Request::DiarClose { handle } => Some(handle),
        _ => None,
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
    ChildError::failed(error.to_string(), true)
}
