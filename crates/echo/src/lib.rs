//! Explicit-reference DSP with the shipped ABI-3 WebRTC component.
mod clock;
mod fft;
pub use clock::{
    CorrelationDecision, DELAY, LOOKAHEAD, RATE, ReferenceClock, SCORE_THRESHOLD,
    correlation_decision,
};
use libloading::Library;
use speakerdesk_core::messages;
use std::{collections::VecDeque, ffi::c_void, path::Path, ptr::NonNull};
pub const FRAME: usize = 160;
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EchoError {
    Missing,
    IncompatibleVersion,
    FailedToStart,
    InvalidInput,
    Processing,
    Reset,
}
impl std::fmt::Display for EchoError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(match self {
            Self::Missing=>messages::ECHO_MISSING,
            Self::IncompatibleVersion=>messages::ECHO_INCOMPATIBLE,
            Self::FailedToStart=>messages::ECHO_FAILED,
            Self::InvalidInput=>"Invalid aligned echo input.",
            Self::Processing=>"Speaker echo cancellation rejected an audio frame. Original audio is preserved.",
            Self::Reset=>"Speaker echo cancellation could not reset after Pause. Original audio is preserved.",
        })
    }
}
impl std::error::Error for EchoError {}
type Destroy = unsafe extern "C" fn(*mut c_void);
type Reset = unsafe extern "C" fn(*mut c_void) -> i32;
type Process =
    unsafe extern "C" fn(*mut c_void, *const f32, *const f32, *mut f32, usize, i32) -> i32;
/// Created, used and dropped on the one blocking recording-writer thread.
/// NonNull deliberately keeps this state !Send; the library outlives every call.
pub struct EchoMixer {
    state: NonNull<c_void>,
    destroy: Destroy,
    reset: Reset,
    process: Process,
    _library: Library,
    clock: ReferenceClock,
    input: VecDeque<f32>,
    pending_system: VecDeque<f32>,
    position: usize,
    discard: usize,
    raw_delay: [f32; DELAY],
    failed: bool,
}
pub struct EchoOutput {
    pub clean: Vec<f32>,
    pub system: Vec<f32>,
}

fn native_delay_ms(clock: &ReferenceClock) -> i32 {
    clock.matches().last().map_or(-1, |(_, delay)| {
        // A stronger reflection can trail the direct path. Keep the original
        // 16 ms causal room allowance here, outside clock/skew measurement.
        ((delay + clock.phase()) * 1000. / RATE as f64 - 16.)
            .round_ties_even()
            .clamp(0., 1000.) as i32
    })
}

impl EchoMixer {
    pub fn start(path: &Path) -> Result<Self, EchoError> {
        let metadata = std::fs::symlink_metadata(path).map_err(|_| EchoError::Missing)?;
        if !metadata.is_file() || metadata.file_type().is_symlink() {
            return Err(EchoError::IncompatibleVersion);
        }
        #[cfg(unix)]
        {
            use std::os::unix::fs::MetadataExt;
            if (metadata.uid() != 0 && metadata.uid() != unsafe { libc::getuid() })
                || metadata.mode() & 0o022 != 0
            {
                return Err(EchoError::IncompatibleVersion);
            }
        }
        // Resolve before loading: a relative leaf must not invoke dylib search rules.
        let path = path.canonicalize().map_err(|_| EchoError::Missing)?;
        #[cfg(windows)]
        let library = unsafe {
            use libloading::os::windows::{
                LOAD_LIBRARY_SEARCH_DLL_LOAD_DIR, LOAD_LIBRARY_SEARCH_SYSTEM32,
            };
            libloading::os::windows::Library::load_with_flags(
                &path,
                LOAD_LIBRARY_SEARCH_DLL_LOAD_DIR | LOAD_LIBRARY_SEARCH_SYSTEM32,
            )
            .map(Library::from)
        }
        .map_err(|_| EchoError::IncompatibleVersion)?;
        #[cfg(not(windows))]
        let library = unsafe { Library::new(path) }.map_err(|_| EchoError::IncompatibleVersion)?;
        unsafe {
            let abi: unsafe extern "C" fn() -> i32 = *library
                .get(b"speakerdesk_echo_abi\0")
                .map_err(|_| EchoError::IncompatibleVersion)?;
            if abi() != 3 {
                return Err(EchoError::IncompatibleVersion);
            }
            let create: unsafe extern "C" fn() -> *mut c_void = *library
                .get(b"speakerdesk_echo_create\0")
                .map_err(|_| EchoError::IncompatibleVersion)?;
            let destroy = *library
                .get::<Destroy>(b"speakerdesk_echo_destroy\0")
                .map_err(|_| EchoError::IncompatibleVersion)?;
            let reset = *library
                .get::<Reset>(b"speakerdesk_echo_reset\0")
                .map_err(|_| EchoError::IncompatibleVersion)?;
            let process = *library
                .get::<Process>(b"speakerdesk_echo_process\0")
                .map_err(|_| EchoError::IncompatibleVersion)?;
            let state = NonNull::new(create()).ok_or(EchoError::FailedToStart)?;
            Ok(Self {
                state,
                destroy,
                reset,
                process,
                _library: library,
                clock: ReferenceClock::default(),
                input: VecDeque::new(),
                pending_system: VecDeque::new(),
                position: 0,
                discard: DELAY,
                raw_delay: [0.; DELAY],
                failed: false,
            })
        }
    }
    /// At most one second per aligned writer tile. Steady held input is half a
    /// second plus one frame. Final drains the native 64-sample framing delay.
    pub fn add(
        &mut self,
        microphone: &[f32],
        system: &[f32],
        reference: &[[f32; 2]],
        final_tail: bool,
    ) -> Result<EchoOutput, EchoError> {
        if self.failed {
            return Err(EchoError::Processing);
        }
        let result = self.add_inner(microphone, system, reference, final_tail);
        if result.is_err() {
            self.failed = true;
        }
        result
    }
    fn add_inner(
        &mut self,
        microphone: &[f32],
        system: &[f32],
        reference: &[[f32; 2]],
        final_tail: bool,
    ) -> Result<EchoOutput, EchoError> {
        if microphone.len() != system.len()
            || microphone.len() != reference.len()
            || microphone.len() > RATE
            || microphone
                .iter()
                .chain(system)
                .chain(reference.iter().flatten())
                .any(|v| !v.is_finite())
        {
            return Err(EchoError::InvalidInput);
        }
        self.clock.append(reference, microphone);
        self.input.extend(microphone);
        self.pending_system.extend(system);
        let length = self.input.len();
        let process_length = if final_tail {
            length.div_ceil(FRAME) * FRAME + FRAME
        } else {
            length.saturating_sub(LOOKAHEAD) / FRAME * FRAME
        };
        let mut clean = Vec::with_capacity(process_length.min(self.pending_system.len()));
        for _ in (0..process_length).step_by(FRAME) {
            let original =
                std::array::from_fn::<_, FRAME, _>(|_| self.input.pop_front().unwrap_or(0.));
            let microphone = original.map(|v| v.clamp(-1., 1.));
            let mut render = self.clock.reference(self.position, FRAME);
            for value in &mut render {
                *value = value.clamp(-1., 1.);
            }
            let delay = native_delay_ms(&self.clock);
            let mut output = [0.; FRAME];
            // Fixed-size validated buffers satisfy the ABI; only this owner calls it.
            if unsafe {
                (self.process)(
                    self.state.as_ptr(),
                    render.as_ptr(),
                    microphone.as_ptr(),
                    output.as_mut_ptr(),
                    FRAME,
                    delay,
                )
            } != 0
                || output.iter().any(|v| !v.is_finite())
            {
                return Err(EchoError::Processing);
            }
            if !self.clock.echo() {
                output[..DELAY].copy_from_slice(&self.raw_delay);
                output[DELAY..].copy_from_slice(&original[..FRAME - DELAY]);
            }
            self.raw_delay.copy_from_slice(&original[FRAME - DELAY..]);
            let skip = self.discard.min(FRAME);
            self.discard -= skip;
            let count = (FRAME - skip).min(self.pending_system.len() - clean.len());
            clean.extend_from_slice(&output[skip..skip + count]);
            self.position += FRAME;
        }
        let system = self.pending_system.drain(..clean.len()).collect();
        if final_tail {
            if !self.pending_system.is_empty() || !self.input.is_empty() {
                return Err(EchoError::Processing);
            }
            if unsafe { (self.reset)(self.state.as_ptr()) } != 0 {
                return Err(EchoError::Reset);
            }
            self.clock = ReferenceClock::default();
            self.position = 0;
            self.discard = DELAY;
            self.raw_delay = [0.; DELAY];
        }
        Ok(EchoOutput { clean, system })
    }
}
impl Drop for EchoMixer {
    fn drop(&mut self) {
        unsafe { (self.destroy)(self.state.as_ptr()) }
    }
}

#[cfg(all(test, any(target_os = "macos", windows)))]
mod tests {
    use super::*;

    #[test]
    fn dominant_reflection_keeps_the_earlier_room_path_causal() {
        assert_eq!(native_delay_ms(&ReferenceClock::default()), -1);
        for (direct, reflection, polarity) in [(1120, 1376, 1.), (1120, 1376, -1.), (0, 128, 1.)] {
            let mut seed = 7_u32;
            let noise: Vec<f32> = (0..RATE * 3)
                .map(|_| {
                    seed = seed.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
                    ((seed >> 8) as f32 / 16_777_216. - 0.5) * 0.4
                })
                .collect();
            let reference: Vec<_> = noise.iter().map(|&v| [v, 0.]).collect();
            let microphone: Vec<_> = (0..noise.len())
                .map(|i| {
                    i.checked_sub(direct).map_or(0., |p| 0.5 * noise[p])
                        + i.checked_sub(reflection)
                            .map_or(0., |p| polarity * noise[p])
                })
                .collect();
            let mut clock = ReferenceClock::default();
            for (far, near) in reference.chunks(RATE).zip(microphone.chunks(RATE)) {
                clock.append(far, near);
            }
            assert!(clock.echo());
            let measured = clock.matches().last().unwrap().1;
            assert!((measured - reflection as f64).abs() < 0.1);
            let sent = native_delay_ms(&clock);
            assert!(sent >= 0);
            let aligned_sample = sent as usize * RATE / 1000;
            // AEC's linear filter is causal. Aligning to the strongest later
            // reflection must leave the weaker direct path representable too.
            assert!(
                aligned_sample <= direct,
                "direct {direct}, reflection {reflection}, sent {sent}ms"
            );
            assert!(reflection - aligned_sample < RATE * 128 / 1000);
        }
    }

    #[test]
    #[ignore = "Requires the pinned native echo component built by scripts/setup_echo.py"]
    fn native_echo_bounds_queues_and_drains_partial_frames_exactly() {
        let name = if cfg!(windows) {
            "speakerdesk_echo.dll"
        } else {
            "libspeakerdesk_echo.dylib"
        };
        let library = Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../desktop/capture/echo")
            .join(name);
        let mut mixer = EchoMixer::start(&library).unwrap();
        let mut expected = Vec::new();
        let mut actual = Vec::new();
        for i in 0..500 {
            let count = [1, 37, 159, 511, 4000][i % 5];
            let input: Vec<_> = (expected.len()..expected.len() + count)
                .map(|sample| if sample % 2 == 0 { 0.125 } else { -0.25 })
                .collect();
            expected.extend_from_slice(&input);
            let output = mixer
                .add(&input, &vec![0.; count], &vec![[0.; 2]; count], false)
                .unwrap();
            assert_eq!(output.system, vec![0.; output.clean.len()]);
            actual.extend(output.clean);
            assert!(mixer.input.len() < LOOKAHEAD + FRAME);
            assert!(mixer.pending_system.len() < LOOKAHEAD + FRAME + DELAY);
            assert_eq!(
                mixer.pending_system.len(),
                mixer.input.len() + DELAY - mixer.discard
            );
        }
        actual.extend(mixer.add(&[], &[], &[], true).unwrap().clean);
        assert_eq!(actual, expected);
        assert!(mixer.input.is_empty());
        assert!(mixer.pending_system.is_empty());
        assert_eq!(mixer.discard, DELAY);
        assert_eq!(mixer.position, 0);
        let resumed = mixer
            .add(&[0.375; 513], &[0.; 513], &[[0.; 2]; 513], true)
            .unwrap();
        assert_eq!(resumed.clean, vec![0.375; 513]);
        assert!(mixer.input.is_empty() && mixer.pending_system.is_empty());
    }
}
