//! Native update admission and bounded stdout framing, independent of Tauri.
pub const MAX_FRAME_BYTES: usize = 8 * 1024;
pub const MAX_REQUEST_ID: u64 = (1 << 53) - 1;

#[derive(Default)]
pub struct Lines {
    pending: Vec<u8>,
    discarding: bool,
}
impl Lines {
    /// A rejected oversized line is discarded through its newline, never reparsed.
    pub fn push(&mut self, bytes: &[u8]) -> Vec<Vec<u8>> {
        let mut output = Vec::new();
        for &byte in bytes {
            if byte == b'\n' {
                if !self.discarding {
                    if self.pending.last() == Some(&b'\r') {
                        self.pending.pop();
                    }
                    output.push(std::mem::take(&mut self.pending));
                }
                self.pending.clear();
                self.discarding = false;
            } else if !self.discarding {
                if self.pending.len() == MAX_FRAME_BYTES {
                    self.pending.clear();
                    self.discarding = true;
                } else {
                    self.pending.push(byte);
                }
            }
        }
        output
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Phase {
    Idle,
    Checking,
    Current,
    Available,
    Downloading,
    Verifying,
    Ready,
    Preparing,
    Reserving,
    Releasing,
    Stopping,
    Blocked,
    Installing,
    Error,
    Unavailable,
}
impl Phase {
    pub fn status(self) -> &'static str {
        match self {
            Self::Idle => "idle",
            Self::Checking => "checking",
            Self::Current => "current",
            Self::Available => "available",
            Self::Downloading => "downloading",
            Self::Verifying => "verifying",
            Self::Ready => "ready",
            Self::Preparing => "preparing",
            Self::Reserving | Self::Stopping | Self::Blocked => "stopping",
            Self::Releasing => "preparing",
            Self::Installing => "installing",
            Self::Error => "error",
            Self::Unavailable => "unavailable",
        }
    }
}

pub struct Control {
    pub id: u64,
    pub phase: Phase,
    pub preparation: u64,
    pub preparation_id: u64,
    shutdown_ready: bool,
    terminated_clean: bool,
}
impl Default for Control {
    fn default() -> Self {
        Self {
            id: 0,
            phase: Phase::Idle,
            preparation: 0,
            preparation_id: 0,
            shutdown_ready: false,
            terminated_clean: false,
        }
    }
}
impl Control {
    pub fn check(&mut self, id: u64) -> bool {
        if id <= self.id
            || id > MAX_REQUEST_ID
            || !matches!(
                self.phase,
                Phase::Idle
                    | Phase::Current
                    | Phase::Available
                    | Phase::Ready
                    | Phase::Error
                    | Phase::Unavailable
            )
        {
            return false;
        }
        self.id = id;
        self.preparation_id = 0;
        self.phase = Phase::Checking;
        self.shutdown_ready = false;
        self.terminated_clean = false;
        true
    }
    pub fn current(&self, id: u64) -> bool {
        id != 0 && self.id == id
    }
    pub fn download(&mut self, id: u64) -> bool {
        if !self.current(id) || self.phase != Phase::Available {
            return false;
        }
        self.phase = Phase::Downloading;
        true
    }
    pub fn finished_download(&mut self, id: u64) {
        if self.current(id) && self.phase == Phase::Downloading {
            self.phase = Phase::Verifying;
        }
    }
    pub fn verified(&mut self, id: u64) -> bool {
        if !self.current(id) || !matches!(self.phase, Phase::Downloading | Phase::Verifying) {
            return false;
        }
        self.phase = Phase::Ready;
        true
    }
    pub fn prepare(&mut self, id: u64) -> bool {
        if !self.current(id) || self.phase != Phase::Ready || self.preparation == MAX_REQUEST_ID {
            return false;
        }
        self.preparation += 1;
        self.preparation_id = id;
        self.phase = Phase::Preparing;
        true
    }
    pub fn preparation_matches(&self, id: u64, preparation: Option<u64>) -> bool {
        self.current(id)
            && self.preparation_id == id
            && self.preparation > 0
            && preparation == Some(self.preparation)
    }
    pub fn editor_ready(&mut self, id: u64, preparation: Option<u64>) -> bool {
        if !self.preparation_matches(id, preparation) || self.phase != Phase::Preparing {
            return false;
        }
        self.phase = Phase::Reserving;
        true
    }
    pub fn reserved(&mut self, id: u64, preparation: Option<u64>) -> bool {
        if !self.preparation_matches(id, preparation) || self.phase != Phase::Reserving {
            return false;
        }
        self.phase = Phase::Stopping;
        true
    }
    pub fn shutdown_ack(&mut self, id: u64, preparation: Option<u64>, ok: bool) -> bool {
        if !self.preparation_matches(id, preparation)
            || !matches!(self.phase, Phase::Stopping | Phase::Blocked)
        {
            return false;
        }
        self.shutdown_ready = ok;
        if !ok {
            self.phase = Phase::Blocked;
        }
        true
    }
    pub fn terminated(&mut self, clean: bool) {
        self.terminated_clean = clean;
    }
    pub fn safe_exit(&self) -> bool {
        self.shutdown_ready && self.terminated_clean
    }
    pub fn install(&mut self) -> bool {
        if self.phase != Phase::Stopping || !self.safe_exit() {
            return false;
        }
        self.phase = Phase::Installing;
        true
    }
    pub fn cancellable(&self) -> bool {
        matches!(
            self.phase,
            Phase::Checking
                | Phase::Available
                | Phase::Downloading
                | Phase::Verifying
                | Phase::Ready
                | Phase::Preparing
                | Phase::Reserving
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    fn ready() -> Control {
        let mut c = Control::default();
        assert!(c.check(4));
        c.phase = Phase::Available;
        assert!(c.download(4));
        assert!(c.verified(4));
        c
    }
    #[test]
    fn fragmented_and_coalesced_stdout_has_exact_line_boundaries() {
        let mut lines = Lines::default();
        assert!(lines.push(b"SPEAKERDESK_UP").is_empty());
        assert_eq!(
            lines.push(b"DATE={\"id\":4}\r\nSPEAKERDESK_URL=http://127.0.0.1:9\n"),
            vec![
                b"SPEAKERDESK_UPDATE={\"id\":4}".to_vec(),
                b"SPEAKERDESK_URL=http://127.0.0.1:9".to_vec()
            ]
        );
    }
    #[test]
    fn oversized_frame_cannot_smuggle_a_valid_suffix() {
        let mut lines = Lines::default();
        assert!(lines.push(&vec![b'x'; MAX_FRAME_BYTES + 1]).is_empty());
        assert_eq!(
            lines.push(b"SPEAKERDESK_UPDATE={\"id\":4}\nvalid\n"),
            vec![b"valid".to_vec()]
        );
        assert_eq!(
            lines.push(&[vec![b'x'; MAX_FRAME_BYTES], vec![b'\n']].concat())[0].len(),
            MAX_FRAME_BYTES
        );
    }
    #[test]
    fn download_finish_is_not_a_verified_installable_payload() {
        let mut c = Control::default();
        assert!(c.check(1));
        c.phase = Phase::Available;
        assert!(c.download(1));
        c.finished_download(1);
        assert!(!c.prepare(1));
        assert!(!c.install());
        assert!(!c.verified(2));
        assert!(c.verified(1));
        assert!(c.prepare(1));
    }
    #[test]
    fn editor_ack_is_required_before_reservation_and_stale_ack_is_ignored() {
        let mut c = ready();
        assert!(!c.editor_ready(4, Some(1)));
        assert!(c.prepare(4));
        assert!(!c.editor_ready(3, Some(1)));
        assert!(!c.reserved(4, Some(1)));
        assert!(c.editor_ready(4, Some(1)));
        assert!(c.reserved(4, Some(1)));
        assert!(!c.check(5));
        assert!(!c.cancellable());
    }
    #[test]
    fn clean_termination_and_shutdown_ack_are_both_required() {
        let mut c = ready();
        c.prepare(4);
        c.editor_ready(4, Some(1));
        c.reserved(4, Some(1));
        assert!(c.shutdown_ack(4, Some(1), true));
        assert!(!c.install());
        c.terminated(false);
        assert!(!c.install());
        c.terminated(true);
        assert!(c.install());
        assert!(!c.install());
    }
    #[test]
    fn lost_ack_or_timeout_never_installs_on_late_process_exit() {
        let mut c = ready();
        c.prepare(4);
        c.editor_ready(4, Some(1));
        c.reserved(4, Some(1));
        c.terminated(true);
        assert!(!c.install());
        c.phase = Phase::Blocked;
        c.shutdown_ack(4, Some(1), true);
        assert!(c.safe_exit());
        assert!(!c.install());
        assert!(!c.cancellable());
    }
    #[test]
    fn duplicate_and_old_requests_cannot_replace_a_current_attempt() {
        let mut c = ready();
        assert!(!c.check(4));
        assert!(!c.check(3));
        assert!(c.check(5));
        assert!(!c.download(4));
        assert!(!c.check(6));
        assert!(!c.check(MAX_REQUEST_ID + 1));
    }
    #[test]
    fn late_editor_or_shutdown_ack_cannot_authorize_a_retried_preparation() {
        let mut c = ready();
        assert!(c.prepare(4));
        let first = c.preparation;
        // A completed cancellation reopens the verified payload for a retry.
        c.phase = Phase::Ready;
        assert!(c.prepare(4));
        let second = c.preparation;
        assert!(second > first);
        assert!(!c.editor_ready(4, Some(first)));
        assert!(!c.editor_ready(4, None));
        assert!(c.editor_ready(4, Some(second)));
        assert!(!c.reserved(4, Some(first)));
        assert!(c.reserved(4, Some(second)));
        assert!(!c.shutdown_ack(4, Some(first), true));
        c.terminated(true);
        assert!(!c.install());
        assert!(c.shutdown_ack(4, Some(second), true));
        assert!(c.install());
    }
    #[test]
    fn new_check_clears_token_ownership_without_reusing_preparation_number() {
        let mut c = ready();
        c.prepare(4);
        c.phase = Phase::Ready;
        let first = c.preparation;
        assert!(c.check(5));
        assert_eq!(c.preparation_id, 0);
        assert!(!c.preparation_matches(5, Some(first)));
        c.phase = Phase::Ready;
        assert!(c.prepare(5));
        assert!(c.preparation > first);
    }
}
