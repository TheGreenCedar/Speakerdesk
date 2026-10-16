use serde_json::Value;
use std::{
    collections::VecDeque,
    io::Write,
    sync::{
        Arc,
        atomic::{AtomicBool, Ordering},
        mpsc::{self, SyncSender, TrySendError},
    },
    thread,
    time::{Duration, Instant},
};

pub struct Failure {
    pub message: String,
    pub retained: VecDeque<Value>,
}
impl From<String> for Failure {
    fn from(message: String) -> Self {
        Self {
            message,
            retained: VecDeque::new(),
        }
    }
}

pub struct Output {
    sender: Option<SyncSender<Value>>,
    done: mpsc::Receiver<Result<(), String>>,
    broken: Arc<AtomicBool>,
}
impl Output {
    pub fn new() -> Self {
        let (sender, receiver) = mpsc::sync_channel::<Value>(32);
        let (finished, done) = mpsc::sync_channel(1);
        let broken = Arc::new(AtomicBool::new(false));
        let failed = broken.clone();
        thread::spawn(move || {
            let result = (|| -> Result<(), String> {
                let mut stdout = std::io::stdout().lock();
                while let Ok(value) = receiver.recv() {
                    serde_json::to_writer(&mut stdout, &value)
                        .map_err(|error| error.to_string())?;
                    stdout
                        .write_all(b"\n")
                        .and_then(|_| stdout.flush())
                        .map_err(|error| error.to_string())?;
                }
                Ok(())
            })();
            if result.is_err() {
                failed.store(true, Ordering::Release);
            }
            let _ = finished.send(result);
        });
        Self {
            sender: Some(sender),
            done,
            broken,
        }
    }
    pub fn batch(&self, mut pending: VecDeque<Value>) -> Result<(), Failure> {
        while let Some(value) = pending.pop_front() {
            if let Err(error) = self.sender.as_ref().expect("open output").try_send(value) {
                let message = match error {
                    TrySendError::Full(value) => {
                        pending.push_front(value);
                        "Capture output queue reached its limit; producers stopped while accepted output drains."
                    }
                    TrySendError::Disconnected(value) => {
                        pending.push_front(value);
                        "Capture output consumer disappeared."
                    }
                };
                return Err(Failure {
                    message: message.into(),
                    retained: pending,
                });
            }
        }
        Ok(())
    }
    pub fn event(&self, value: Value) -> Result<(), Failure> {
        self.batch(VecDeque::from([value]))
    }
    pub fn drain(&self, mut pending: VecDeque<Value>) -> Result<(), String> {
        let deadline = Instant::now() + Duration::from_secs(5);
        loop {
            match self.batch(pending) {
                Ok(()) => return Ok(()),
                Err(failure) => {
                    pending = failure.retained;
                    if self.broken.load(Ordering::Acquire) || Instant::now() >= deadline {
                        return Err(failure.message);
                    }
                    thread::sleep(Duration::from_millis(1));
                }
            }
        }
    }
    pub fn finish(mut self) -> Result<(), String> {
        self.sender.take();
        self.done
            .recv_timeout(Duration::from_secs(5))
            .map_err(|_| "Capture stdout did not drain before its deadline.".to_string())?
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn full_queue_retains_every_unsent_record_including_the_overflow_record() {
        let (sender, receiver) = mpsc::sync_channel(1);
        let (_finished, done) = mpsc::sync_channel(1);
        let output = Output {
            sender: Some(sender),
            done,
            broken: Arc::new(AtomicBool::new(false)),
        };
        let failure = output
            .batch(VecDeque::from([
                Value::from(1),
                Value::from(2),
                Value::from(3),
            ]))
            .err()
            .unwrap();
        assert_eq!(receiver.recv().unwrap(), Value::from(1));
        assert_eq!(
            failure.retained,
            VecDeque::from([Value::from(2), Value::from(3)])
        );
        let failure = output.batch(failure.retained).err().unwrap();
        assert_eq!(receiver.recv().unwrap(), Value::from(2));
        assert_eq!(failure.retained, VecDeque::from([Value::from(3)]));
    }
}
