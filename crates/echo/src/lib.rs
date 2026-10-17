//! Explicit-reference DSP with the shipped ABI-6 WebRTC + LocalVQE component.
mod clock;
mod fft;
pub use clock::{
    CorrelationDecision, DELAY, LOOKAHEAD, RATE, ReferenceClock, SCORE_THRESHOLD,
    correlation_decision,
};
use libloading::Library;
use speakerdesk_core::messages;
use std::{
    collections::VecDeque,
    ffi::{CString, c_char, c_void},
    path::Path,
    ptr::NonNull,
};
pub const FRAME: usize = 160;
pub const ENHANCE_FRAME: usize = 256;
pub const LINEAR_DELAY: usize = 64;
pub const ENHANCE_DELAY: usize = 256;
pub const MODEL_FILE: &str = "localvqe-v1.2-1.3M-f32.gguf";
pub const MODEL_BYTES: u64 = 5_173_088;
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
type Enhance = unsafe extern "C" fn(*mut c_void, *const f32, *const f32, *mut f32, usize) -> i32;
/// Created, used and dropped on the one blocking recording-writer thread.
/// NonNull deliberately keeps this state !Send; the library outlives every call.
pub struct EchoMixer {
    state: NonNull<c_void>,
    destroy: Destroy,
    reset: Reset,
    process: Process,
    enhance: Enhance,
    _library: Library,
    clock: ReferenceClock,
    input: VecDeque<f32>,
    pending_system: VecDeque<f32>,
    model_reference: VecDeque<f32>,
    model_input: VecDeque<[f32; 2]>,
    linear_discard: usize,
    position: usize,
    discard: usize,
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
        let library = unsafe { Library::new(&path) }.map_err(|_| EchoError::IncompatibleVersion)?;
        unsafe {
            let abi: unsafe extern "C" fn() -> i32 = *library
                .get(b"speakerdesk_echo_abi\0")
                .map_err(|_| EchoError::IncompatibleVersion)?;
            if abi() != 6 {
                return Err(EchoError::IncompatibleVersion);
            }
            let create: unsafe extern "C" fn(*const c_char) -> *mut c_void = *library
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
            let enhance = *library
                .get::<Enhance>(b"speakerdesk_echo_enhance\0")
                .map_err(|_| EchoError::IncompatibleVersion)?;
            let model = path.with_file_name(MODEL_FILE);
            let metadata = std::fs::symlink_metadata(&model).map_err(|_| EchoError::Missing)?;
            if !metadata.is_file()
                || metadata.file_type().is_symlink()
                || metadata.len() != MODEL_BYTES
            {
                return Err(EchoError::IncompatibleVersion);
            }
            #[cfg(unix)]
            {
                use std::os::unix::fs::MetadataExt;
                if (metadata.uid() != 0 && metadata.uid() != libc::getuid())
                    || metadata.mode() & 0o022 != 0
                {
                    return Err(EchoError::IncompatibleVersion);
                }
            }
            let model = CString::new(model.to_str().ok_or(EchoError::IncompatibleVersion)?)
                .map_err(|_| EchoError::IncompatibleVersion)?;
            let state = NonNull::new(create(model.as_ptr())).ok_or(EchoError::FailedToStart)?;
            Ok(Self {
                state,
                destroy,
                reset,
                process,
                enhance,
                _library: library,
                clock: ReferenceClock::default(),
                input: VecDeque::new(),
                pending_system: VecDeque::new(),
                model_reference: VecDeque::new(),
                model_input: VecDeque::new(),
                linear_discard: LINEAR_DELAY,
                position: 0,
                discard: ENHANCE_DELAY,
                failed: false,
            })
        }
    }
    /// At most one second per aligned writer tile. Steady held input is half a
    /// second plus one frame. Final drains both stages before resetting their shared state.
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
        let content_end = self.position + length;
        let process_length = if final_tail {
            // Complete every model hop that can contribute to original content.
            let model_end = (content_end + ENHANCE_DELAY).div_ceil(ENHANCE_FRAME) * ENHANCE_FRAME;
            (model_end + LINEAR_DELAY).div_ceil(FRAME) * FRAME - self.position
        } else {
            length.saturating_sub(LOOKAHEAD) / FRAME * FRAME
        };
        let mut clean = Vec::with_capacity(process_length.min(self.pending_system.len()));
        for _ in (0..process_length).step_by(FRAME) {
            let original =
                std::array::from_fn::<_, FRAME, _>(|_| self.input.pop_front().unwrap_or(0.));
            let microphone = original.map(|v| v.clamp(-1., 1.));
            let mut render = self.clock.reference(self.position, FRAME);
            for (index, frame) in render.chunks_exact_mut(2).enumerate() {
                for value in frame.iter_mut() {
                    *value = if self.position + index >= content_end {
                        0.
                    } else {
                        value.clamp(-1., 1.)
                    };
                }
                self.model_reference.push_back((frame[0] + frame[1]) * 0.5);
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
            let skip = self.linear_discard.min(FRAME);
            self.linear_discard -= skip;
            for &sample in &output[skip..] {
                self.model_input.push_back([
                    sample.clamp(-1., 1.),
                    self.model_reference
                        .pop_front()
                        .ok_or(EchoError::Processing)?,
                ]);
            }
            self.enhance_ready(&mut clean)?;
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
            self.discard = ENHANCE_DELAY;
            self.linear_discard = LINEAR_DELAY;
            self.model_reference.clear();
            self.model_input.clear();
        }
        Ok(EchoOutput { clean, system })
    }
    fn enhance_ready(&mut self, clean: &mut Vec<f32>) -> Result<(), EchoError> {
        while self.model_input.len() >= ENHANCE_FRAME {
            let pairs: [[f32; 2]; ENHANCE_FRAME] =
                std::array::from_fn(|_| self.model_input.pop_front().unwrap());
            let linear = pairs.map(|pair| pair[0]);
            let reference = pairs.map(|pair| pair[1]);
            let mut output = [0.; ENHANCE_FRAME];
            if unsafe {
                (self.enhance)(
                    self.state.as_ptr(),
                    reference.as_ptr(),
                    linear.as_ptr(),
                    output.as_mut_ptr(),
                    ENHANCE_FRAME,
                )
            } != 0
                || output.iter().any(|v| !v.is_finite())
            {
                return Err(EchoError::Processing);
            }
            let skip = self.discard.min(ENHANCE_FRAME);
            self.discard -= skip;
            let count = (ENHANCE_FRAME - skip).min(self.pending_system.len() - clean.len());
            clean.extend_from_slice(&output[skip..skip + count]);
        }
        Ok(())
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

    fn native_library() -> std::path::PathBuf {
        let name = if cfg!(windows) {
            "speakerdesk_echo.dll"
        } else {
            "libspeakerdesk_echo.dylib"
        };
        std::env::var_os("SPEAKERDESK_TEST_ECHO")
            .map(std::path::PathBuf::from)
            .unwrap_or_else(|| {
                Path::new(env!("CARGO_MANIFEST_DIR"))
                    .join("../../desktop/capture/echo")
                    .join(name)
            })
    }

    // Independent two-stage oracle retains continued linear output before framing.
    fn oracle(state: &EchoMixer, mic: &[f32], reference: &[[f32; 2]]) -> Vec<f32> {
        let count = (mic.len() + 1024).div_ceil(FRAME) * FRAME;
        let mut linear = Vec::new();
        let mut far = Vec::new();
        for start in (0..count).step_by(FRAME) {
            let near: [f32; FRAME] =
                std::array::from_fn(|i| mic.get(start + i).copied().unwrap_or(0.).clamp(-1., 1.));
            let render: [f32; FRAME * 2] = std::array::from_fn(|i| {
                reference
                    .get(start + i / 2)
                    .map_or(0., |v| v[i % 2].clamp(-1., 1.))
            });
            let mut out = [0.; FRAME];
            assert_eq!(
                unsafe {
                    (state.process)(
                        state.state.as_ptr(),
                        render.as_ptr(),
                        near.as_ptr(),
                        out.as_mut_ptr(),
                        FRAME,
                        -1,
                    )
                },
                0
            );
            linear.extend(out);
            far.extend(render.chunks_exact(2).map(|v| (v[0] + v[1]) * 0.5));
        }
        let aligned = &linear[LINEAR_DELAY..];
        let mut output = Vec::new();
        for start in (0..aligned.len()).step_by(ENHANCE_FRAME) {
            let near: [f32; ENHANCE_FRAME] = std::array::from_fn(|i| {
                aligned.get(start + i).copied().unwrap_or(0.).clamp(-1., 1.)
            });
            let render: [f32; ENHANCE_FRAME] = std::array::from_fn(|i| {
                if start + i < aligned.len() {
                    far.get(start + i).copied().unwrap_or(0.)
                } else {
                    0.
                }
            });
            let mut out = [0.; ENHANCE_FRAME];
            assert_eq!(
                unsafe {
                    (state.enhance)(
                        state.state.as_ptr(),
                        render.as_ptr(),
                        near.as_ptr(),
                        out.as_mut_ptr(),
                        ENHANCE_FRAME,
                    )
                },
                0
            );
            output.extend(out);
        }
        assert_eq!(unsafe { (state.reset)(state.state.as_ptr()) }, 0);
        output[ENHANCE_DELAY..ENHANCE_DELAY + mic.len()].to_vec()
    }

    #[test]
    #[ignore = "Requires the pinned native hybrid built by scripts/setup_echo.py"]
    fn native_hybrid_all_frame_residues_preserve_short_tails_and_reset_both_histories() {
        let library = native_library();
        let mut assembled = EchoMixer::start(&library).unwrap();
        let direct = EchoMixer::start(&library).unwrap();
        let microphone: Vec<f32> = (0..LOOKAHEAD + FRAME * 8 + 37)
            .map(|i| ((i * 7919 % 65521) as f32 / 65521. - 0.5) * 0.4)
            .collect();
        let reference: Vec<[f32; 2]> = (0..LOOKAHEAD + FRAME * 8 + 37)
            .map(|i| {
                [
                    ((i * 3571 % 65521) as f32 / 65521. - 0.5) * 0.3,
                    ((i * 733 % 65521) as f32 / 65521. - 0.5) * 0.2,
                ]
            })
            .collect();
        // lcm(160,256)=1280 covers every possible joint frame residue, including
        // recordings shorter than either hop. Every iteration reuses reset state.
        for length in (1..=1280).chain((1..=8).map(|phase| LOOKAHEAD + FRAME * phase + 37)) {
            let expected = oracle(&direct, &microphone[..length], &reference[..length]);
            let system: Vec<f32> = (0..length).map(|i| i as f32 / 1280.).collect();
            let split = if length > LOOKAHEAD {
                length - 37
            } else {
                length / 2
            };
            let mut out = assembled
                .add(
                    &microphone[..split],
                    &system[..split],
                    &reference[..split],
                    false,
                )
                .unwrap();
            let tail = assembled
                .add(
                    &microphone[split..length],
                    &system[split..],
                    &reference[split..length],
                    true,
                )
                .unwrap();
            out.clean.extend(tail.clean);
            out.system.extend(tail.system);
            assert_eq!(out.clean, expected, "joint frame residue {length}");
            assert_eq!(out.system, system);
            assert!(
                assembled.input.is_empty()
                    && assembled.pending_system.is_empty()
                    && assembled.model_input.is_empty()
                    && assembled.model_reference.is_empty()
            );
        }
        eprintln!(
            "1280 joint residues plus8 nonzero global160/256phases; zero mic/ref after each original EOF; complete linear drain precedes enhancement/reset; exact original N clean/system samples"
        );
    }

    #[test]
    #[ignore = "Requires the pinned hybrid; warmed continuous-zero EOF regression"]
    fn native_hybrid_warmed_eof_keeps_the_whole_last_model_hop() {
        // Tone rejects broadband Clock hints, independently keeping both paths at
        // native delay=-1 while exercising four seconds of nonzero echo history.
        let length: usize = 64_128;
        let reference: Vec<[f32; 2]> = (0..length)
            .map(|i| {
                let v = (i as f32 * std::f32::consts::TAU * 431. / RATE as f32).sin() * 0.2;
                [v, v]
            })
            .collect();
        let microphone: Vec<f32> = (0..length)
            .map(|i| {
                i.checked_sub(1120).map_or(0., |p| reference[p][0] * 0.55)
                    + (i as f32 * std::f32::consts::TAU * 173. / RATE as f32).sin() * 0.05
            })
            .collect();
        let library = native_library();
        let direct = EchoMixer::start(&library).unwrap();
        let expected = oracle(&direct, &microphone, &reference);
        let mut mixer = EchoMixer::start(&library).unwrap();
        let mut clean = Vec::new();
        for (mic, far) in microphone.chunks(RATE).zip(reference.chunks(RATE)) {
            clean.extend(
                mixer
                    .add(mic, &vec![0.; mic.len()], far, false)
                    .unwrap()
                    .clean,
            );
        }
        assert!(
            mixer.clock.matches().is_empty(),
            "independent native -1 oracle requires tonal Clock rejection"
        );
        clean.extend(mixer.add(&[], &[], &[], true).unwrap().clean);
        assert_eq!(clean.len(), length);
        let error = clean
            .iter()
            .zip(&expected)
            .map(|(a, b)| (a - b).abs())
            .fold(0_f32, f32::max);
        assert_eq!(
            error, 0.,
            "complete native response is required through the final model hop"
        );
        eprintln!(
            "warmed N=64128 agrees bit-for-bit with1024sample continued native-input zero oracle; last128original samples retained"
        );
    }

    #[test]
    #[ignore = "Requires the pinned hybrid model and library"]
    fn native_hybrid_model_admission_rejects_missing_corrupt_and_supports_unicode_paths() {
        struct Folder(std::path::PathBuf);
        impl Drop for Folder {
            fn drop(&mut self) {
                let _ = std::fs::remove_dir_all(&self.0);
            }
        }
        let folder = Folder(std::env::temp_dir().join(format!(
                "speakerdesk-echo-模型-é-{}-{}",
                std::process::id(),
                std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .unwrap()
                    .as_nanos()
            )));
        std::fs::create_dir(&folder.0).unwrap();
        let original = native_library();
        let library = folder.0.join(original.file_name().unwrap());
        std::fs::copy(&original, &library).unwrap();
        assert!(matches!(
            EchoMixer::start(&library),
            Err(EchoError::Missing)
        ));
        let model = folder.0.join(MODEL_FILE);
        std::fs::write(&model, b"bad GGUF").unwrap();
        assert!(matches!(
            EchoMixer::start(&library),
            Err(EchoError::IncompatibleVersion)
        ));
        std::fs::File::create(&model)
            .unwrap()
            .set_len(MODEL_BYTES)
            .unwrap();
        assert!(
            matches!(EchoMixer::start(&library), Err(EchoError::FailedToStart)),
            "same-size corrupt model must fail before parsing"
        );
        std::fs::copy(original.with_file_name(MODEL_FILE), &model).unwrap();
        let mut mixer = EchoMixer::start(&library).expect("UTF-8 bundled path");
        assert_eq!(
            mixer
                .add(&[0.1; 513], &[0.2; 513], &[[0.; 2]; 513], true)
                .unwrap()
                .system,
            vec![0.2; 513]
        );
        drop(mixer);
    }

    #[test]
    #[ignore = "Requires the pinned hybrid and independently frozen spoken control Float32 file"]
    fn native_hybrid_assembled_spoken_latency_is_compensated_without_fitting() {
        let path = std::env::var_os("SPEAKERDESK_ECHO_SPOKEN_CONTROL")
            .expect("independent Float32 calibration path");
        let bytes = std::fs::read(path).unwrap();
        assert_eq!(bytes.len() % 4, 0);
        let control: Vec<f32> = bytes
            .chunks_exact(4)
            .map(|v| f32::from_le_bytes(v.try_into().unwrap()))
            .collect();
        assert_eq!(control.len(), RATE * 6);
        assert!(control.iter().all(|v| v.is_finite() && v.abs() <= 1.));
        let voiced: Vec<_> = (0..5)
            .map(|i| (RATE / 2 + i * RATE, RATE / 2 + (i + 1) * RATE))
            .filter(|&(start, end)| {
                control[start..end]
                    .iter()
                    .map(|&v| f64::from(v).powi(2))
                    .sum::<f64>()
                    / RATE as f64
                    >= 0.01_f64.powi(2)
            })
            .collect();
        assert!(
            voiced.len() >= 4,
            "independent control requires at least4 input-defined voiced1s windows"
        );
        let mut mixer = EchoMixer::start(&native_library()).unwrap();
        let mut output = Vec::new();
        for mic in control.chunks(997) {
            output.extend(
                mixer
                    .add(mic, &vec![0.; mic.len()], &vec![[0.; 2]; mic.len()], false)
                    .unwrap()
                    .clean,
            );
        }
        output.extend(mixer.add(&[], &[], &[], true).unwrap().clean);
        assert_eq!(output.len(), control.len());
        for window in std::iter::once((0, control.len())).chain(voiced) {
            let (start, end) = window;
            let score = |shift: isize| {
                let mut xy = 0.;
                let mut xx = 0.;
                let mut yy = 0.;
                for (i, &x) in control.iter().enumerate().take(end).skip(start) {
                    if let Some(&y) = i.checked_add_signed(shift).and_then(|p| output.get(p)) {
                        xy += f64::from(x) * f64::from(y);
                        xx += f64::from(x).powi(2);
                        yy += f64::from(y).powi(2);
                    }
                }
                xy / (xx * yy).sqrt().max(1e-30)
            };
            let peak = (-32..=32)
                .max_by(|&a, &b| score(a).total_cmp(&score(b)))
                .unwrap();
            assert!(
                score(0) >= 0.8 && peak.abs() <= 1,
                "fixed320 compensation: {window:?}, correlation {}, peak {peak}",
                score(0)
            );
            eprintln!(
                "fixed64+256 compensation: input window {window:?}, corr={}, diagnostic peak={peak}",
                score(0)
            );
        }
    }

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
        let library = native_library();
        let mut mixer = EchoMixer::start(&library).unwrap();
        let mut expected = Vec::new();
        let mut actual = Vec::new();
        let mut system = Vec::new();
        let mut expected_system = Vec::new();
        for i in 0..500 {
            let count = [1, 37, 159, 511, 4000][i % 5];
            let input: Vec<_> = (expected.len()..expected.len() + count)
                .map(|sample| if sample % 2 == 0 { 0.125 } else { -0.25 })
                .collect();
            expected.extend_from_slice(&input);
            let reference_system: Vec<_> = (0..count).map(|s| (s % 17) as f32 / 32.).collect();
            expected_system.extend_from_slice(&reference_system);
            let output = mixer
                .add(&input, &reference_system, &vec![[0.; 2]; count], false)
                .unwrap();
            actual.extend(output.clean);
            system.extend(output.system);
            assert!(mixer.input.len() < LOOKAHEAD + FRAME);
            assert!(mixer.pending_system.len() < LOOKAHEAD + FRAME + DELAY + ENHANCE_FRAME);
            assert!(mixer.model_input.len() < ENHANCE_FRAME);
            assert!(mixer.model_reference.len() <= LINEAR_DELAY);
        }
        let final_tail = mixer.add(&[], &[], &[], true).unwrap();
        actual.extend(final_tail.clean);
        system.extend(final_tail.system);
        assert_eq!(actual.len(), expected.len());
        assert_eq!(system, expected_system);
        assert!(actual.iter().all(|v| v.is_finite()));
        // A zero reference must no longer bypass noise suppression.
        assert_ne!(actual, expected);
        let mut contiguous = EchoMixer::start(&library).unwrap();
        let mut contiguous_clean = Vec::new();
        let mut contiguous_system = Vec::new();
        for (mic, original_system) in expected.chunks(RATE).zip(system.chunks(RATE)) {
            let out = contiguous
                .add(mic, original_system, &vec![[0.; 2]; mic.len()], false)
                .unwrap();
            contiguous_clean.extend(out.clean);
            contiguous_system.extend(out.system);
        }
        let tail = contiguous.add(&[], &[], &[], true).unwrap();
        contiguous_clean.extend(tail.clean);
        contiguous_system.extend(tail.system);
        assert_eq!(actual, contiguous_clean);
        assert_eq!(system, contiguous_system);
        assert!(mixer.input.is_empty());
        assert!(mixer.pending_system.is_empty());
        assert_eq!(mixer.discard, ENHANCE_DELAY);
        assert_eq!(mixer.linear_discard, LINEAR_DELAY);
        assert!(mixer.model_input.is_empty() && mixer.model_reference.is_empty());
        assert_eq!(mixer.position, 0);
        let resumed = mixer
            .add(&[0.375; 513], &[0.; 513], &[[0.; 2]; 513], true)
            .unwrap();
        let fresh = EchoMixer::start(&library)
            .unwrap()
            .add(&[0.375; 513], &[0.; 513], &[[0.; 2]; 513], true)
            .unwrap();
        assert_eq!(resumed.clean.len(), 513);
        assert_eq!(resumed.clean, fresh.clean);
        assert_eq!(resumed.system, vec![0.; 513]);
        assert!(mixer.input.is_empty() && mixer.pending_system.is_empty());
    }
}
