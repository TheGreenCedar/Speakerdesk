use std::{path::Path, process::Stdio, time::Duration};

use tokio::{
    io::{AsyncBufReadExt, AsyncRead, AsyncReadExt, AsyncWriteExt, BufReader},
    process::Command as Process,
    sync::{mpsc, watch},
    task::JoinHandle,
    time::{Instant, sleep_until, timeout},
};
use tokio_util::sync::CancellationToken;

use super::processing::{EXPORT, Helper, invalid, private_create};
use super::writer::{SourceWriter, run_writer};
use super::{
    CaptureConfig, CaptureError, CapturePhase, CaptureSnapshot, Command, EventKind, STALL_STOP,
    parse_event,
};

// One maximum stereo/16k float packet, base64 encoded, plus bounded JSON metadata.
const MAX_LINE: usize = 176 * 1024;
// The baseline's 120 packets retain roughly 30 seconds at the helper's tile size.
const PACKETS: usize = 120;

pub struct CaptureSupervisor;
impl CaptureSupervisor {
    /// Inspect the configured app-owned helper without launching it.
    pub fn check_helper(path: &Path) -> Result<(), CaptureError> {
        Helper::inspect(path).map(|_| ())
    }
    /// Construction does filesystem work off the async reader. No model or store is involved.
    pub async fn spawn(
        helper: impl AsRef<Path>,
        config: CaptureConfig,
    ) -> Result<CaptureSession, CaptureError> {
        Self::spawn_with_deadline(helper.as_ref(), config, STALL_STOP).await
    }

    pub(super) async fn spawn_with_deadline(
        path: &Path,
        config: CaptureConfig,
        stop_bound: Duration,
    ) -> Result<CaptureSession, CaptureError> {
        let path = path.to_path_buf();
        let prepared_config = config.clone();
        let (helper, writer, log) = tokio::task::spawn_blocking(move || {
            let helper = Helper::inspect(&path)?;
            let writer = SourceWriter::create(prepared_config.clone(), helper.clone())?;
            let log = private_create(&prepared_config.folder.join("capture.log"))?;
            Ok::<_, CaptureError>((helper, writer, log))
        })
        .await
        .map_err(|e| invalid(e.to_string()))??;
        Self::start(helper, writer, log, config, stop_bound)
    }

    pub(super) fn start(
        helper: Helper,
        writer: SourceWriter,
        log: std::fs::File,
        config: CaptureConfig,
        stop_bound: Duration,
    ) -> Result<CaptureSession, CaptureError> {
        let mut child = Process::new(&helper.path)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .kill_on_drop(true)
            .spawn()?;
        let pid = child
            .id()
            .ok_or_else(|| invalid("Capture helper has no process identity."))?;
        let stdin = child.stdin.take().expect("piped stdin");
        let stdout = child.stdout.take().expect("piped stdout");
        let stderr = child.stderr.take().expect("piped stderr");
        let (commands_tx, commands_rx) = mpsc::channel(8);
        let (snapshots_tx, snapshots_rx) = watch::channel(writer.snapshot.clone());
        let stop = CancellationToken::new();
        let actor_stop = stop.clone();
        let start = Command::Start {
            microphone: config.microphone,
            system: config.system,
            job_id: (config.microphone && config.system).then_some(config.job_id),
            capture_export_contract: (config.microphone && config.system).then(|| EXPORT.into()),
        };
        let task = tokio::spawn(async move {
            let (packets_tx, packets_rx) = mpsc::channel(PACKETS);
            let writer_output = snapshots_tx.clone();
            let writer_task =
                tokio::task::spawn_blocking(move || run_writer(writer, packets_rx, writer_output));
            let (stdin_tx, mut stdin_rx) = mpsc::channel::<Command>(8);
            // A blocked stdin write cannot block stdout, cancellation, or the stop deadline.
            let mut stdin_task = tokio::spawn(async move {
                let mut stdin = stdin;
                while let Some(command) = stdin_rx.recv().await {
                    let line = command.to_line().map_err(std::io::Error::other)?;
                    stdin.write_all(line.as_bytes()).await?;
                    stdin.flush().await?;
                }
                Ok::<_, std::io::Error>(())
            });
            let mut stderr_task = tokio::spawn(async move {
                let mut reader = stderr;
                let mut log = tokio::fs::File::from_std(log);
                let mut written = 0;
                let mut buffer = [0; 4096];
                loop {
                    let count = reader.read(&mut buffer).await?;
                    if count == 0 {
                        break;
                    }
                    let keep = count.min(65536_usize.saturating_sub(written));
                    if keep > 0 {
                        log.write_all(&buffer[..keep]).await?;
                        written += keep;
                    }
                    // Continue draining after the bounded diagnostic prefix.
                }
                log.flush().await
            });
            let _ = stdin_tx.try_send(start);
            let mut lines = Lines::new(stdout);
            let mut commands = commands_rx;
            let mut observations = snapshots_tx.subscribe();
            let mut deadline = None;
            let mut stopping = false;
            let mut stopped = false;
            let mut commands_open = true;
            let mut failure = None;
            let mut stdin_joined = false;
            loop {
                tokio::select! {
                    _=actor_stop.cancelled(), if !stopping => {
                        stopping=true; deadline=Some(Instant::now()+stop_bound);
                        if stdin_tx.try_send(Command::Stop).is_err() { failure=Some("Capture command queue is full.".into());break; }
                    }
                    command=commands.recv(), if commands_open => {
                        let command=command.unwrap_or(Command::Stop);
                        if commands.is_closed() {commands_open=false;}
                        if matches!(command,Command::Stop) {
                            if stopping {continue;}
                            stopping=true;deadline=Some(Instant::now()+stop_bound);
                        } else if stopping {continue;}
                        if stdin_tx.try_send(command).is_err() {failure=Some("Capture command queue is full.".into());break;}
                    }
                    _=async { if let Some(deadline)=deadline {sleep_until(deadline).await} else {std::future::pending::<()>().await} } => {
                        failure=Some("Capture helper did not stop before its deadline; audio is preserved.".into());break;
                    }
                    changed=observations.changed() => {
                        if changed.is_err() {failure=Some("Capture writer stopped unexpectedly.".into());break;}
                        if let Some(error)=observations.borrow().error.clone() {failure=Some(error);break;}
                    }
                    result=&mut stdin_task => {
                        stdin_joined=true;
                        failure=Some(match result {Ok(Err(e))=>format!("Capture command failed: {e}"),Err(e)=>e.to_string(),Ok(Ok(()))=>"Capture command pipe closed unexpectedly.".into()});break;
                    }
                    line=lines.next() => {
                        let line=match line {
                            Ok(Some(line))=>line,
                            Ok(None)=>{if !stopped {failure=Some("Capture helper closed without a stopped acknowledgement.".into());}break;},
                            Err(error)=>{failure=Some(error.to_string());break;}
                        };
                        let event=match parse_event(&line) {Ok(event)=>event,Err(error)=>{failure=Some(error.to_string());break;}};
                        if event.kind()==EventKind::Stopped {
                            stopped=true;
                            deadline.get_or_insert_with(|| Instant::now()+stop_bound);
                        }
                        if packets_tx.try_send(event).is_err() {failure=Some("Capture writer fell behind; recording stopped and accepted audio is being saved.".into());break;}
                    }
                }
            }
            drop(stdin_tx);
            if failure.is_some() {
                let _ = child.start_kill();
            }
            let until = deadline.unwrap_or_else(|| Instant::now() + stop_bound);
            match timeout(
                until.saturating_duration_since(Instant::now()),
                child.wait(),
            )
            .await
            {
                Ok(Ok(status)) if status.success() => (),
                Ok(Ok(status)) => {
                    failure.get_or_insert_with(|| format!("Capture helper exited with {status}."));
                }
                Ok(Err(error)) => {
                    failure.get_or_insert(error.to_string());
                    let _ = child.start_kill();
                    let _ = child.wait().await;
                }
                Err(_) => {
                    failure.get_or_insert_with(|| {
                        "Capture helper did not exit after stopping.".into()
                    });
                    let _ = child.start_kill();
                    let _ = child.wait().await;
                }
            }
            if !stdin_joined {
                stdin_task.abort();
                let _ = stdin_task.await;
            }
            // An inherited stderr descriptor or stalled diagnostic write cannot extend Stop.
            if timeout(
                until.saturating_duration_since(Instant::now()),
                &mut stderr_task,
            )
            .await
            .is_err()
            {
                stderr_task.abort();
                let _ = stderr_task.await;
            }
            drop(packets_tx); // Blocking writer drains every accepted packet, then its held tail.
            let mut snapshot = match writer_task.await {
                Ok(snapshot) => snapshot,
                Err(error) => {
                    let mut snapshot = snapshots_tx.borrow().clone();
                    snapshot.error = Some(error.to_string());
                    snapshot
                }
            };
            if snapshot.error.is_none() {
                snapshot.error = failure;
            }
            snapshot.phase = if snapshot.error.is_some() {
                CapturePhase::Failed
            } else {
                CapturePhase::Stopped
            };
            snapshots_tx.send_replace(snapshot.clone());
            snapshot
        });
        Ok(CaptureSession {
            pid,
            commands: commands_tx,
            stop,
            updates: snapshots_rx,
            task: Some(task),
        })
    }
}

/// Dropping the handle requests Stop; the owned task still kills/reaps on timeout.
/// Call `finish` to wait for both process reaping and final WAV/receipt cleanup.
pub struct CaptureSession {
    pub pid: u32,
    commands: mpsc::Sender<Command>,
    stop: CancellationToken,
    pub updates: watch::Receiver<CaptureSnapshot>,
    task: Option<JoinHandle<CaptureSnapshot>>,
}
impl CaptureSession {
    pub fn command(&self, command: Command) -> Result<(), CaptureError> {
        if matches!(command, Command::Start { .. }) {
            return Err(invalid("Capture is already started."));
        }
        if matches!(command, Command::Stop) {
            self.stop.cancel();
            return Ok(());
        }
        if self.stop.is_cancelled() {
            return Err(invalid("Capture is stopping."));
        }
        self.commands
            .try_send(command)
            .map_err(|_| invalid("Capture command queue is full or closed."))
    }
    pub async fn finish(mut self) -> Result<CaptureSnapshot, CaptureError> {
        self.stop.cancel();
        self.task
            .take()
            .expect("owned capture task")
            .await
            .map_err(|e| invalid(e.to_string()))
    }
}
impl Drop for CaptureSession {
    fn drop(&mut self) {
        self.stop.cancel();
    }
}

/// The partial line belongs to the reader, so selecting commands never loses a prefix.
struct Lines<R> {
    reader: BufReader<R>,
    partial: Vec<u8>,
}
impl<R: AsyncRead + Unpin> Lines<R> {
    fn new(reader: R) -> Self {
        Self {
            reader: BufReader::new(reader),
            partial: Vec::new(),
        }
    }
    async fn next(&mut self) -> Result<Option<String>, CaptureError> {
        loop {
            let bytes = self.reader.fill_buf().await?;
            if bytes.is_empty() {
                return if self.partial.is_empty() {
                    Ok(None)
                } else {
                    Err(invalid("Capture helper closed during a JSON line."))
                };
            }
            let newline = bytes.iter().position(|b| *b == b'\n');
            let count = newline.map_or(bytes.len(), |i| i + 1);
            if self.partial.len() + count > MAX_LINE {
                return Err(invalid("Capture event exceeded its line bound."));
            }
            self.partial.extend_from_slice(&bytes[..count]);
            self.reader.consume(count);
            if newline.is_some() {
                let line = String::from_utf8(std::mem::take(&mut self.partial))
                    .map_err(|_| invalid("Capture event is not UTF-8."))?;
                return Ok(Some(line));
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn selecting_a_control_during_a_partial_line_keeps_the_prefix() {
        let (mut sender, receiver) = tokio::io::duplex(128);
        let mut lines = Lines::new(receiver);
        sender.write_all(b"{\"type\":\"record").await.unwrap();
        assert!(
            timeout(Duration::from_millis(10), lines.next())
                .await
                .is_err()
        );
        sender.write_all(b"ing\"}\n").await.unwrap();
        assert_eq!(
            lines.next().await.unwrap().unwrap(),
            "{\"type\":\"recording\"}\n"
        );
    }
}
