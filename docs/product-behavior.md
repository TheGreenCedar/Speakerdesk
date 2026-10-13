# Product behavior

These are the user-visible rules the product promises. Terms follow [GLOSSARY.md](../GLOSSARY.md). Each section names the code that owns the rule. When you change a rule, update this file in the same change.

## Language

Owner: `core/src/language.rs`, `core/src/languages.rs`, `core/src/limits.rs`.

- New meetings and imports use the language default from Settings, which is **Auto** unless the user picks one of Cohere's 14 supported languages. Changing Settings never changes an existing job.
- In Auto mode, Whisper tiny scores language on short probes before Cohere runs. A supported winner applies when its probability is at least **0.90** and its margin is at least **0.20**. These are closed-set scores, not accuracy guarantees.
- An uncertain probe can borrow the most recent confident language from the same language epoch, if that language ended no more than **60 seconds** of audio earlier. Without such context, the words are a marked best-effort guess. Unsupported or failed detection leaves the audio available for an explicit language choice.
- Short, quiet and overlapping speech are still attempted. Only exactly constant PCM skips detection and ASR. It is recorded as `digital_silence` (all zero) or `constant_signal` (nonzero DC).
- **During a meeting**, the header selector changes the language from the current saved-recording sample onward. This starts a new language epoch. It is accepted only while recording or paused, and only after the previous boundary has been acknowledged. A stale `language_revision` returns 409. Earlier text keeps its language. To revisit earlier words, retry the passage.

## Speech admission

Owner: `core/src/speech.rs`.

Silero v6 runs on two views of the original audio: raw, and a peak-conditioned view (gain `min(256, 0.25 / peak)`). It uses the maximum of the two probabilities, with 0.5 onset and 0.35 release hysteresis. Diarization and Cohere always receive original audio. No amplitude threshold or word filter decides what counts as speech. A missing or failed inspection never establishes silence, and empty ASR output never does either.

## Live text and refinement

Owner: `core/src/live.rs`, `host/src/routes/refinement.rs`.

- Live meetings show provisional text, which becomes canonical utterances as speech settles. Rolling refinement improves canonical text during the meeting. Stop runs a bounded larger-context pass before the meeting is finalized.
- Complete words can finish the accuracy pass even when automatic language routing is uncertain. The language evidence stays in passage details; short passages containing multiple supported languages keep separate timing receipts. Empty, truncated, unsupported or failed results retain the previous words. Stop retries failed live passages once; it does not repeat failed work indefinitely.
- Words remain in one chronological reading column as accuracy work finishes. Pending machine work does not create a manual **Needs review** region. The header reports queued, active, paused or incomplete accuracy work; specific uncertainty and repair options remain in passage details. A retained correction whose original passage was replaced stays in its own section.
- Speaker badges match unnamed labels such as **Speaker 2**, then show initials when a person is named. Track colors remain stable through naming and save/reopen; internal track IDs never determine displayed speaker numbers. Mixed and unassigned audio do not take a speaker number.
- Refinement never overwrites a **protected edit** (user words, speaker or bounds). It never changes names or neighbouring passages, and never alters the original audio.
- Transcript timing (supplied-text CTC) is required for transcription. It functionally covers all 14 languages, but its accuracy is calibrated in English only. CJK text uses character units. Units that are unsupported or uncertain stay unresolved instead of being guessed.

## Passage retry and review

Owner: `host/src/routes/retry.rs`, `core/src/review.rs`, `core/src/export.rs`.

- After saving edits, the user can retry one saved passage of at most **24.5 s** in an explicitly chosen language. Auto is not offered for retry. The result is a separate **retry candidate**. It replaces the passage only when the user chooses **Use this text** and saves. Empty or failed candidates never erase existing words.
- Passages carry review reasons: unknown language, empty ASR, model error, partial output, overlap, unassigned audio and uncertain speaker. **Mark words reviewed** resolves the words. Overlap and unassigned-speaker warnings remain, because reviewing words does not settle who spoke.
- TXT/SRT/VTT exports skip blank passages and print `[Needs review: …]` markers. JSON keeps every range, its reasons and any candidates. Exported times are audio anchors, not phonetic word boundaries.

## Overlapping voices

Owner: `core/src/separation.rs`, `host/src/routes/separation.rs`.

- During recording, Speakerdesk automatically processes completed, unedited passages with observed overlap between exactly two voices, within one language setting and at most **24.5 s** long. The existing recording worker drains pending live audio before each separation or transcription phase. An in-flight model request must finish before the worker can return to live processing. The same automatic processing runs after imports and saved accuracy work. It uses the optional, locally prepared separation model and never downloads weights automatically.
- A successful attempt publishes two ordinary editable rows, each with its own generated audio and a **Check separated voices** hint. Both rows remain unassigned: output channel order does not identify a speaker, and current voice calibration does not qualify separated estimates for identity matching or voice enrollment. The original words and audio remain available in passage history, and an unchanged pair can restore the original passage.
- Protected edits, stale source observations, pending candidates and already separated rows are skipped. Each original passage's source/model pair is attempted once automatically, including failed or interrupted attempts and restored originals; live child-loss recovery may retry the interrupted passage under the new child generation. Failure, cancellation or edits during processing retain the current words and original recording. Estimates saved as pending candidates by an older runtime still offer explicit use or discard.

## People and voices

Owner: `core/src/people.rs`, `core/src/voice.rs`, `host/src/people_routes/`.

- People stores reusable names. Naming a speaker track applies to the current meeting only, and never renames a person or changes a saved voice.
- **Remember voice** is the single explicit consent action. It needs a confirmed person and at least two finalized, non-overlapping clips of **2–10 s** each. Long passages offer separate 6-second windows. Adding a name, downloading models, or confirming a recognition never enrolls a voice.
- Recognition is on by default and can be turned off in Settings. Automatic matches are labelled Recognized and stay unconfirmed. They never create or update a profile. Profiles from an incompatible model identity are kept but take no part in recognition until the user replaces them with consent. **Forget voice** removes only the stored voice; names, recordings and text remain.
- Saved-voice recognition runs during recording as soon as enough observed solo audio is available, including a stable prefix of an ongoing passage. It uses the same calibrated matching rules as saved recordings and yields to pending live audio between clips. Later source changes, manual naming, disabled recognition, or profile replacement/forgetting invalidate an outstanding match.
- Introduction suggestions come from conservative English patterns ("I'm …", "my name is …") and need the user's confirmation.

## Updates and shortcuts

Owner: `desktop/src-tauri/src/updates.rs`, `menus.rs`, `host/src/native_updates.rs`.

- The app checks for an update once when it opens. A notice offers **Download update…**, and installing needs a separate **Install and restart**. Dismissing a notice hides that version across restarts, and a newer version shows a new notice. **Speakerdesk → Check for Updates…** checks manually.
- Download and install are refused while recording, processing, model setup or saves are active. Downgrades are disabled.

| Shortcut | Action |
| --- | --- |
| ⌘, | Settings |
| ⌘N | New Meeting |
| ⌘O | Import recordings |
| ⌘F | Find in transcript / search meetings |
| ⌘S | Save transcript edits |
| Escape | Close Settings; clear an active search |
