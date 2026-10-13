# Product behavior

These are the user-visible rules the product promises. Terms follow [GLOSSARY.md](../GLOSSARY.md). Each section names the code that owns the rule. When you change a rule, update this file in the same change.

## Language

Owner: `core/src/language.rs`, `core/src/languages.rs`, `core/src/limits.rs`.

- New meetings and imports use the language default from Settings, which is **Auto** unless the user picks one of Cohere's 14 supported languages. Changing Settings never changes an existing job.
- In Auto mode, Whisper tiny scores language on short probes before Cohere runs. A supported winner applies when its probability is at least **0.90** and its margin is at least **0.20**. These are closed-set scores, not accuracy guarantees.
- An uncertain probe first uses the speaker’s main language when independent, confident acoustic observations establish it. Otherwise it uses the meeting’s main language. This evidence belongs to the same audio source, language epoch and stream generation; a pause does not erase it. Saved accuracy can use later observed speech to resolve an earlier uncertain passage. Borrowed language never becomes a new acoustic vote. Confident local language switches and confident unsupported-language detections take precedence; explicit language choices remain authoritative. Without qualified context, the existing best-effort or explicit-language repair behavior applies.
- Short, quiet and overlapping speech are still attempted. Only exactly constant PCM skips detection and ASR. It is recorded as `digital_silence` (all zero) or `constant_signal` (nonzero DC).
- **During a meeting**, the header selector changes the language from the current saved-recording sample onward. This starts a new language epoch. It is accepted only while recording or paused, and only after the previous boundary has been acknowledged. A stale `language_revision` returns 409. Earlier text keeps its language. To revisit earlier words, retry the passage.

## Speech admission

Owner: `core/src/speech.rs`.

Silero v6 runs on two views of the original audio: raw, and a peak-conditioned view (gain `min(256, 0.25 / peak)`). It uses the maximum of the two probabilities, with 0.5 onset and 0.35 release hysteresis. Diarization and Cohere always receive original audio. No amplitude threshold or word filter decides what counts as speech. A missing or failed inspection never establishes silence, and empty ASR output never does either.

## Live text and refinement

Owner: `core/src/live.rs`, `host/src/routes/refinement.rs`.

- Live meetings show provisional text, which becomes canonical utterances as speech settles. Rolling refinement improves canonical text during the meeting. Stop runs a bounded larger-context pass before the meeting is finalized.
- Complete words can finish the accuracy pass even when automatic language routing is uncertain. The language evidence stays in passage details; short passages containing multiple supported languages keep separate timing receipts. Empty, truncated, unsupported or failed results retain the previous words. Stop retries failed live passages once; it does not repeat failed work indefinitely.
- Words remain in one chronological reading column as accuracy work finishes. Pending machine work does not create a manual **Needs review** region. After recording stops, passages blocked by language appear at their timestamps with a reason and visible language controls, including a failed language recheck whose previous words were retained. The incomplete-accuracy header points to those passages, including when search has hidden them. Successfully accepted readable uncertainty keeps its evidence in passage details; silence does not become a repair task. A retained correction whose original passage was replaced stays in its own section.
- Speaker badges match unnamed labels such as **Speaker 2**, then show initials when a person is named. Track colors remain stable through naming and save/reopen; internal track IDs never determine displayed speaker numbers. Mixed and unassigned audio do not take a speaker number.
- Refinement never overwrites a **protected edit** (user words, speaker or bounds). It never changes names or neighbouring passages, and never alters the original audio.
- Transcript timing (supplied-text CTC) is required for transcription. It functionally covers all 14 languages, but its accuracy is calibrated in English only. CJK text uses character units. Units that are unsupported or uncertain stay unresolved instead of being guessed.

## Passage retry and review

Owner: `host/src/routes/retry.rs`, `core/src/review.rs`, `core/src/export.rs`.

- After saving edits, the user can retry one saved passage of at most **24.5 s** in an explicitly chosen language. Auto is not offered for retry. The result is a separate **retry candidate**. It replaces the passage only when the user chooses **Use this text** and saves. Empty or failed candidates never erase existing words.
- Saving a newly accepted language repair resumes the remaining incomplete accuracy work automatically. The repaired text stays protected, and the other passages retain their captured language settings and epochs. Repeated saves, stale candidates and unrelated edits do not start another pass. Longer language-blocked passages show the retry limit and expose timing adjustment. The language-repair timing control and passage retry require outstanding passage drafts or saves to finish.
- Passages carry review reasons: unknown language, empty ASR, model error, partial output, overlap, unassigned audio and uncertain speaker. **Mark words reviewed** resolves the words. Overlap and unassigned-speaker warnings remain, because reviewing words does not settle who spoke.
- TXT/SRT/VTT exports skip blank passages and print `[Needs review: …]` markers. JSON keeps every range, its reasons and any candidates. Exported times are audio anchors, not phonetic word boundaries.

## Overlapping voices

Owner: `core/src/separation.rs`, `core/src/voice/separation.rs`, `host/src/routes/separation.rs`, `host/src/routes/separation_identity.rs`.

- During recording, Speakerdesk automatically processes completed, unedited passages with observed overlap between exactly two voices, within one language setting and at most **24.5 s** long. The existing recording worker resumes long separation and separated transcription in steps, processing pending live audio and controls between steps. Neural work remains serial; a running model step must finish before it yields. The same automatic processing runs after imports and saved accuracy work. It uses the optional, locally prepared separation model and never downloads weights automatically.
- A successful attempt publishes two ordinary editable rows, each with its own generated audio and a **Check separated voices** hint. With recognition enabled and a qualified separated-voice policy, acoustic evidence can associate each row with an existing meeting track or compatible saved voice. Output channel order never identifies a speaker. A strong match can be used while its sibling stays unassigned; ambiguous estimates or two estimates matching the same identity stay unassigned. Saved-voice matches create unconfirmed assignments; reusing a meeting track preserves its current name and manual assignment. Generated estimates remain ineligible for voice enrollment. The original words and audio remain available in passage history, and an unchanged pair can restore the original passage.
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
