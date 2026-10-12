//! Serial import job worker. One job at a time, matching the v0.6.6 executor.
//!
//! Submitted import runs wait on a single thread. A run is a closure so tests
//! can pass a fake model. This module does not download weights or spawn MLX.

use std::collections::VecDeque;
use std::fmt;
use std::panic::{AssertUnwindSafe, catch_unwind};
use std::sync::{Arc, Condvar, Mutex};
use std::thread::{self, JoinHandle};

struct Queue {
    pending: VecDeque<Box<dyn FnOnce() + Send>>,
    running: bool,
    stop: bool,
}

struct Shared {
    queue: Mutex<Queue>,
    wake: Condvar,
}

/// Queues import runs and executes one at a time.
pub struct JobWorker {
    shared: Arc<Shared>,
    thread: Mutex<Option<JoinHandle<()>>>,
}

impl JobWorker {
    pub fn new() -> Self {
        let shared = Arc::new(Shared {
            queue: Mutex::new(Queue {
                pending: VecDeque::new(),
                running: false,
                stop: false,
            }),
            wake: Condvar::new(),
        });
        let thread_state = Arc::clone(&shared);
        let thread = thread::Builder::new()
            .name("speakerdesk".to_string())
            .spawn(move || run_queue(thread_state))
            .expect("import worker thread starts");
        Self {
            shared,
            thread: Mutex::new(Some(thread)),
        }
    }

    /// Add an import run. It starts only when no other run is in progress.
    pub fn enqueue(&self, run: impl FnOnce() + Send + 'static) {
        let mut queue = self.shared.queue.lock().expect("import queue lock");
        if queue.stop {
            return;
        }
        queue.pending.push_back(Box::new(run));
        self.shared.wake.notify_one();
    }

    /// Block until every queued run has finished.
    pub fn drain(&self) {
        let mut queue = self.shared.queue.lock().expect("import queue lock");
        while queue.running || !queue.pending.is_empty() {
            queue = self.shared.wake.wait(queue).expect("import queue wait");
        }
    }
}

impl Default for JobWorker {
    fn default() -> Self {
        Self::new()
    }
}

impl fmt::Debug for JobWorker {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str("JobWorker")
    }
}

impl Drop for JobWorker {
    fn drop(&mut self) {
        {
            let mut queue = self.shared.queue.lock().expect("import queue lock");
            queue.stop = true;
            self.shared.wake.notify_all();
        }
        if let Some(thread) = self.thread.lock().expect("worker thread").take() {
            let _ = thread.join();
        }
    }
}

fn run_queue(shared: Arc<Shared>) {
    loop {
        let task = {
            let mut queue = shared.queue.lock().expect("import queue lock");
            loop {
                if let Some(task) = queue.pending.pop_front() {
                    queue.running = true;
                    break task;
                }
                if queue.stop {
                    queue.running = false;
                    shared.wake.notify_all();
                    return;
                }
                queue.running = false;
                shared.wake.notify_all();
                queue = shared.wake.wait(queue).expect("import queue wait");
            }
        };
        if let Err(_panic) = catch_unwind(AssertUnwindSafe(task)) {
            tracing::error!("import run panicked");
        }
        let mut queue = shared.queue.lock().expect("import queue lock");
        queue.running = false;
        shared.wake.notify_all();
    }
}

#[cfg(test)]
mod tests {
    use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
    use std::sync::{Arc, Mutex, mpsc};
    use std::time::Duration;

    use super::JobWorker;

    #[test]
    fn runs_one_import_at_a_time_in_submit_order() {
        let worker = JobWorker::new();
        let (started_tx, started_rx) = mpsc::channel();
        let (release_tx, release_rx) = mpsc::channel();
        let second = Arc::new(AtomicBool::new(false));
        let order = Arc::new(Mutex::new(Vec::new()));

        let first_order = Arc::clone(&order);
        worker.enqueue(move || {
            first_order.lock().expect("order").push(1);
            started_tx.send(()).expect("first started");
            release_rx.recv().expect("release");
        });
        let second_flag = Arc::clone(&second);
        let second_order = Arc::clone(&order);
        worker.enqueue(move || {
            second_flag.store(true, Ordering::SeqCst);
            second_order.lock().expect("order").push(2);
        });

        started_rx
            .recv_timeout(Duration::from_secs(2))
            .expect("first import starts");
        assert!(
            !second.load(Ordering::SeqCst),
            "second import must wait for the first"
        );
        release_tx.send(()).expect("release first");
        worker.drain();
        assert!(second.load(Ordering::SeqCst));
        assert_eq!(*order.lock().expect("order"), vec![1, 2]);
    }

    #[test]
    fn a_panicking_import_does_not_block_the_next_run() {
        let worker = JobWorker::new();
        let ran = Arc::new(AtomicUsize::new(0));
        worker.enqueue(|| panic!("import failed"));
        let flag = Arc::clone(&ran);
        worker.enqueue(move || {
            flag.fetch_add(1, Ordering::SeqCst);
        });
        worker.drain();
        assert_eq!(ran.load(Ordering::SeqCst), 1);
    }
}
