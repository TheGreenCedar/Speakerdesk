//! Verbatim v0.6.6 user-facing strings. Do not edit the text.

pub const INVALID_REQUEST: &str = "Invalid request.";
pub const LOCAL_OPERATION_FAILED: &str =
    "Local operation failed. Check disk space and the app terminal.";
pub const UPDATE_RESERVED: &str =
    "Speakerdesk is preparing an update. Retry after it finishes or is cancelled.";
pub const TOKEN_MISMATCH: &str = "Reload the app before making changes.";
pub const SETUP_BEFORE_TRANSCRIPTION: &str =
    "Finish local model setup before starting transcription.";
pub const CHOOSE_LANGUAGE_MODE: &str = "Choose a supported language mode.";
pub const KNOWN_REVISION: &str = "Supply a nonnegative integer known_revision.";
pub const BATCH_SIZE: &str = "Choose 1–16 audio files per batch.";
pub const SELECT_LANGUAGE: &str = "Select a supported recording language.";
pub const SUPPORTED_FORMATS: &str =
    "Supported formats: WAV, MP3, M4A, FLAC, OGG, AIFF, MP4, AAC and WebM.";
pub const QUEUE_FULL: &str = "The local queue is full. Wait for current jobs to finish.";
pub const SUPPLY_PASSAGE_EDITS: &str = "Supply passage edits.";
pub const ONLY_WORDS_SPEAKER_BOUNDS: &str = "Only words, speaker and passage bounds can be edited.";
pub const WAIT_FOR_PASSAGES: &str = "Wait for transcript passages.";
pub const PASSAGE_STALE: &str = "This passage changed. Review its latest words before saving.";
pub const AUDIO_NOT_READY: &str = "Audio is not ready yet.";
pub const JOB_ALREADY_RUNNING: &str = "This job is already running.";
pub const RERUN_WOULD_OVERWRITE: &str =
    "Create another upload to rerun inference without overwriting edits.";
pub const FINISH_MEETING_BEFORE_IMPORT: &str =
    "Finish the active meeting before running import inference.";
pub const WAIT_FOR_INFERENCE: &str = "Wait for audio preparation and inference to finish.";
pub const TRANSCRIPT_STALE: &str = "This transcript changed in another tab. Reload before saving.";
pub const CHOOSE_PASSAGE_AND_LANGUAGE: &str = "Choose a passage and its language.";
pub const RETRY_LANGUAGE: &str =
    "Choose the language for this passage; Automatic is not a retry override.";
pub const RETRY_TOO_LONG: &str = "Choose a passage of at most 24.5 seconds before retrying.";
pub const RETRY_BUSY: &str = "Finish current recording or transcription before retrying a passage.";
pub const RETRY_STALE: &str = "This transcript changed. Save or reload before retrying.";
pub const RETRY_NEEDS_AUDIO: &str = "This passage needs its original saved audio.";
pub const EXPORT_NEEDS_TRANSCRIPT: &str = "Create a transcript before exporting.";
pub const EXPORT_TOO_LARGE: &str =
    "Export exceeds the 16 MiB download limit. The saved meeting is unchanged.";
pub const DELETE_BUSY: &str = "Wait for this job to finish before deleting it.";
pub const RECOGNITION_CHOICE: &str = "Choose whether to recognize saved voices.";
pub const TRANSCRIPT_OBJECT: &str = "Transcript must be a JSON object.";
pub const SPEAKER_COUNT: &str = "Supply 1–64 speaker names.";
pub const SPEAKER_NAMES: &str = "Speaker names must be nonempty text, up to 100 characters.";
pub const SEGMENT_LIST: &str = "Supply a segment list with at most 20,000 entries.";
pub const SEGMENT_OBJECT: &str = "Each segment must be an object.";
pub const SEGMENT_IDS: &str = "Segment IDs must be unique strings.";
pub const PERSON_NAME: &str = "Enter a name of 1–100 characters.";
pub const PEOPLE_CAP: &str = "People can hold up to 500 entries.";
pub const PERSON_MISSING: &str = "This person is no longer in People.";
pub const NAME_BEFORE_FINAL: &str =
    "Wait for finalized transcript passages before naming speakers.";
pub const NAME_STALE: &str = "The transcript changed. Review the current passages and try again.";
pub const SUGGESTION_STALE: &str = "This suggestion is no longer current.";
pub const CONFIRM_OR_CORRECT: &str =
    "Confirm the suggested person, or make a separate name correction.";
pub const REMEMBER_CONSENT: &str = "Choose Remember voice to save a local voice profile.";
pub const REMEMBER_BEFORE_CONFIRM: &str =
    "Confirm this speaker as the person before remembering their voice.";
pub const VOICE_UNAVAILABLE_BUILD: &str =
    "Voice recognition is not available in this build. Names can still be saved in People.";
pub const VOICE_BUSY: &str =
    "Finish the active recording or transcription before remembering a voice.";
pub const MEETING_AUDIO_UNAVAILABLE: &str = "Meeting audio is unavailable.";
pub const VOICE_UNAVAILABLE_SHORT: &str = "Voice recognition is not available in this build.";
pub const VOICE_CHECK_BUSY: &str =
    "Finish the active recording or transcription before checking voices.";
pub const SPEAKER_ALREADY_NAMED: &str =
    "This speaker already has a name. Choose a name correction to change it.";
pub const PREVIEW_REFRESH: &str = "Refresh passages before previewing this audio.";
pub const PREVIEW_STALE: &str = "This voice passage is no longer current.";
pub const PREVIEW_AUDIO_MISSING: &str = "Original speaker audio is unavailable.";
pub const ALIGNMENT_SELECTION: &str = "Invalid optional alignment selection.";
pub const TIMING_REQUIRED_DOWNLOAD: &str =
    "Transcript timing is required for transcription. Download all required models.";
pub const NEEDS_APPLE_SILICON: &str =
    "This version requires an Apple Silicon Mac with macOS 15 or later.";
pub const SETUP_BUSY_MEETING: &str =
    "Finish the active meeting or transcription before model setup.";
pub const WAIT_FOR_VOICE_SETUP: &str = "Wait for voice setup to finish.";
pub const VOICE_NEEDS_APPLE_SILICON: &str =
    "Update Speakerdesk on an Apple Silicon Mac with macOS 15 or later to use voice recognition.";
pub const VOICE_NEEDS_GPU: &str =
    "Voice recognition needs a verified GPU/ANE runtime. Saved names and voices are kept.";
pub const WAIT_FOR_MEETING_MODELS: &str = "Wait for meeting model setup to finish.";
pub const TIMING_UNAVAILABLE_BUILD: &str =
    "Transcript timing is unavailable in this build. Update Speakerdesk.";
pub const TIMING_NEEDS_APPLE_SILICON: &str =
    "Timing requires an Apple Silicon Mac with macOS 15 or later.";
pub const TIMING_SETUP_BUSY: &str =
    "Finish the active meeting or transcription before timing setup.";
pub const WAIT_FOR_MODEL_SETUP: &str = "Wait for model setup to finish.";
pub const TIMING_REQUEST: &str = "Invalid transcript timing request.";
pub const TIMING_CANNOT_DISABLE: &str =
    "Transcript timing is required for transcription and cannot be disabled.";
pub const TIMING_CHANGE_BUSY: &str =
    "Finish the active meeting, transcription or setup before changing timing.";
pub const TIMING_ENABLE_AFTER_CHECK: &str =
    "Download and check the timing model before enabling it.";
pub const MEETING_SOURCES: &str = "Select valid meeting audio sources.";
pub const MEETING_TITLE: &str = "Enter a meeting title and supported language.";
pub const MEETING_ROUTE: &str = "Choose microphone audio, Mac audio, or both.";
pub const MEETING_ACTIVE: &str = "A meeting is already active.";
pub const MEETING_WAIT_IMPORT: &str =
    "Wait for queued import inference to finish before starting a meeting.";
pub const LIVE_CAPTURE_MISSING: &str =
    "Live capture is not included in this build yet. Recording imports remain available.";
pub const OPEN_SETTINGS_SETUP: &str = "Open Settings to finish local model setup.";
pub const MEETING_FREE_SPACE: &str = "Free at least512MB before recording a meeting.";
pub const MEETING_INACTIVE: &str = "This meeting is no longer active.";
pub const MEETING_NOT_READY: &str = "The meeting is not ready for this action.";
pub const WAIT_FOR_RECORDING_START: &str = "Wait for recording to start.";
pub const CAPTURE_STOPPED: &str = "Audio capture stopped. Wait for the saved recording.";
pub const CHOOSE_LIVE_LANGUAGE: &str = "Choose a live language.";
pub const LIVE_LANGUAGE_REVISION: &str =
    "Choose a supported live language and its current revision.";
pub const LANGUAGE_WHILE_RECORDING: &str = "Language can change while recording or paused.";
pub const LIVE_LANGUAGE_STALE: &str =
    "The live language changed in another window. Try again with its current setting.";
pub const LIVE_LANGUAGE_BACKLOG: &str =
    "Live transcription is catching up. Try changing language again shortly.";
pub const NOT_REFINING: &str = "This meeting is not being refined.";
pub const WAIT_FOR_LOCAL_INFERENCE: &str = "Wait for current local inference to finish.";
pub const NO_SAVED_AUDIO: &str = "This meeting has no saved audio to refine.";
pub const SETUP_BEFORE_REFINE: &str = "Finish local model setup before refining saved audio.";
pub const DISMISS_UPDATE_VERSION: &str = "Dismiss the currently available update version.";
pub const CHOOSE_UPDATE_ACTION: &str = "Choose check, download, install or cancel.";
pub const UPDATE_EDITOR_RESULT: &str =
    "Supply the current update attempt and a brief editor result.";
pub const UPDATES_DESKTOP_ONLY: &str = "App updates are available in the desktop app.";
pub const UPDATE_PROTECTED: &str =
    "Wait for the protected update operation to finish or be safely cancelled.";
pub const UPDATE_BUSY: &str = "Finish the active recording, transcription and local setup before downloading or installing an update.";
pub const UPDATE_OPERATION_BUSY: &str = "Wait for the current update operation or cancel it.";
pub const UPDATE_CHECK_FIRST: &str = "Check for an available update before downloading.";
pub const UPDATE_DOWNLOAD_FIRST: &str = "Download and verify the update before installing.";
pub const UPDATE_STALE: &str = "This update preparation is no longer current.";
pub const UPDATE_INSTALLING: &str = "Installation is finishing. Keep Speakerdesk open.";
pub const UPDATE_NOTHING_TO_CANCEL: &str = "There is no update to cancel.";
pub const MISSING_DIARIZATION: &str =
    "Missing or incompatible local nemotron_diarization MLX checkpoint.";
pub const MISSING_ASR: &str = "Missing or incompatible local cohere_asr MLX checkpoint.";
pub const DIAR_PYTHON: &str = "Set DIAR_PYTHON to an inference environment Python executable.";
pub const ASR_PYTHON: &str = "Set ASR_PYTHON to an inference environment Python executable.";
pub const MLX_APPLE_SILICON: &str = "This Mac setup uses MLX on Apple Silicon.";
pub const SPEECH_MODEL_MISSING: &str =
    "Speech detection needs its verified local model. Download models in Settings.";
pub const LID_MODEL_MISSING: &str = "Automatic language detection needs its local model. Download models in Settings, or choose a language override.";
pub const TIMING_NOT_READY: &str =
    "Required transcript timing is not ready. Finish local model setup or retry timing setup.";
pub const MEMORY_PRESSURE_UNKNOWN: &str = "Unable to check Mac memory pressure before inference.";
pub const MEMORY_PRESSURE_HIGH: &str =
    "Mac memory pressure is high. Close memory-heavy apps and retry.";
pub const CAPTURE_DURATION_INVALID: &str = "Capture duration must be finite and nonnegative.";
pub const MEETING_INTERRUPTED: &str =
    "The meeting was interrupted. Captured audio and transcript were preserved.";
pub const IMPORT_INTERRUPTED: &str = "The app stopped during processing. Retry this job.";
pub const AUDIO_PREP_FAILED: &str =
    "Audio preparation failed. Check file format and free disk space.";
pub const PROCESSING_FAILED: &str =
    "Processing failed. Check the inference environment and storage.";
pub const TRANSCRIPT_READY: &str = "Transcript ready. Review names, text and timing.";
pub const ECHO_MISSING: &str =
    "Speaker echo cancellation is missing from this build. Rebuild the capture audio component.";
pub const ECHO_INCOMPATIBLE: &str = "Capture echo component has an incompatible version.";
pub const ECHO_FAILED: &str = "Speaker echo cancellation could not start.";
pub const FREE_SPACE_SETUP: &str =
    "There is not enough free disk space. Free at least 2.4 GB and retry.";
pub const RECORDING_DISK_LOW: &str =
    "Recording stopped because disk space is low. Audio has been saved.";
pub const VOICE_FREE_SPACE: &str = "Free at least 0.6 GB of disk space, then retry voice setup.";
pub const DOWNLOAD_PAUSED_DISK: &str =
    "Download paused because disk space is low. Free space, then resume.";
pub const REMEMBER_PROFILE: &str = "Choose Remember voice to save a voice profile.";
pub const RETRY_RETAINED: &str =
    "This passage could not be transcribed. Its audio and existing words are retained.";
pub const CAPTURE_FINITE_MONO: &str = "Capture requires finite mono audio.";
pub const EXPECTED_PCM_WAV: &str = "Expected 16 kHz mono 16-bit PCM WAV.";
pub const RECORDING_LENGTH: &str = "Recordings must be nonempty and no longer than two hours.";
pub const PCM_TRUNCATED: &str = "The PCM recording is truncated.";
pub const PCM_INCOMPLETE_SAMPLE: &str = "The PCM recording has an incomplete sample.";
pub const PCM_DECODE_FAILED: &str = "The file could not be decoded as PCM audio.";
pub const FFMPEG_REQUIRED: &str =
    "Install FFmpeg to decode this format, or supply mono 16 kHz 16-bit PCM WAV.";
pub const AUDIO_CONVERSION_TIMEOUT: &str = "Audio conversion exceeded three minutes.";
pub const COULD_NOT_DECODE: &str =
    "Could not decode this recording. Supply a supported audio file.";
pub const TOO_MANY_TRANSITIONS: &str =
    "Too many speaker transitions. Split the recording into smaller files.";
pub const INCOMPLETE_COHERE: &str = "Cohere returned an incomplete transcript.";
pub const CLOSING: &str = "Speakerdesk is closing.";
pub const MEETING_STOPPED_BEFORE_CAPTURE: &str = "Meeting stopped before capture began.";
pub const CAPTURE_TIMING_JUMPED: &str =
    "Audio capture timing jumped. Recording stopped to preserve its timeline.";
pub const ECHO_REJECTED_FRAME: &str = "Speaker echo cancellation rejected an audio frame.";
pub const USE_FINALIZED_CLIPS: &str =
    "Use finalized local transcription clips for voice recognition.";
pub const CLEAN_PASSAGE_LENGTH: &str =
    "Choose clean, finalized passages from one speaker, each 2–10 seconds long.";
pub const NONOVERLAPPING_PASSAGES: &str = "Choose separate, nonoverlapping passages.";
pub const PASSAGE_UNUSABLE: &str =
    "A selected passage is unusable for voice recognition. Choose another clean passage.";
pub const PROFILE_NEEDS_CLIPS: &str = "A voice profile requires multiple clean passages.";
pub const VOICE_PASSAGE_LENGTH: &str = "Voice passages must be 2–10 seconds long.";
pub const VOICE_FINITE_MONO: &str = "Supply 2–10 seconds of finite mono audio.";
pub const VOICE_PCM: &str = "Voice recognition needs normalized mono 16 kHz PCM16 meeting audio.";
pub const VOICE_PASSAGE_BEYOND_AUDIO: &str = "A voice passage extends beyond the saved audio.";
pub const VOICE_PASSAGE_INCOMPLETE: &str = "A voice passage is incomplete.";
pub const VOICE_READ_FAILED: &str = "Voice recognition could not read the meeting audio.";
pub const MODEL_VERIFICATION_FAILED: &str = "Model verification failed. Retry the download.";
pub const MODEL_METADATA_VERIFICATION_FAILED: &str =
    "Model metadata verification failed. Retry the download.";
pub const MISSING_CHECKPOINT: &str = "Missing local model checkpoint.";
pub const TIMING_FAILED_TO_LOAD: &str =
    "Required transcript timing failed to load. Retry local model setup.";
pub const CAPTURE_CLOCK_STOPPED: &str = "Capture clock stopped updating. Audio has been saved.";
pub const AUDIO_FORMAT_CHANGED: &str = "Audio format changed too frequently. Audio has been saved.";
pub const MODELS_COULD_NOT_START: &str =
    "Local models could not start. Audio capture did not begin.";
pub const CAPTURE_STOPPED_UNEXPECTEDLY: &str =
    "Audio capture stopped unexpectedly. Captured audio has been saved.";
pub const TRANSCRIPTION_FINISH_TIMEOUT: &str =
    "Transcription took too long to finish. Captured audio has been saved.";
pub const REFINEMENT_PAUSED: &str =
    "Refinement paused. Previous words and saved audio are retained.";

pub fn clean_passage_count(minimum: usize) -> String {
    format!("Choose {minimum}–12 separate clean passages.")
}

pub fn segment_times(index: usize) -> String {
    format!("Segment {index}: supply numeric start and end times.")
}

pub fn segment_bounds(index: usize) -> String {
    format!("Segment {index}: times must fall within the recording and end after start.")
}

pub fn segment_speaker_or_text(index: usize) -> String {
    format!("Segment {index}: invalid speaker or text.")
}

#[cfg(test)]
mod tests {
    use super::{PASSAGE_STALE, TRANSCRIPT_STALE};

    #[test]
    fn stale_revision_messages_match_v0_6_6() {
        assert_eq!(
            TRANSCRIPT_STALE,
            "This transcript changed in another tab. Reload before saving."
        );
        assert_eq!(
            PASSAGE_STALE,
            "This passage changed. Review its latest words before saving."
        );
        assert_eq!(
            super::RETRY_RETAINED,
            "This passage could not be transcribed. Its audio and existing words are retained."
        );
        assert_eq!(
            super::clean_passage_count(2),
            "Choose 2–12 separate clean passages."
        );
        assert_ne!(super::REMEMBER_CONSENT, super::REMEMBER_PROFILE);
    }
}
