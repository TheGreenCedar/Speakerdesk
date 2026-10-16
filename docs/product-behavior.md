# Product behavior

These are the user-visible rules the product promises. Terms follow [GLOSSARY.md](../GLOSSARY.md). Each section names the code that owns the rule. When you change a rule, update this file in the same change.

## Language

Owner: `core/src/language.rs`, `core/src/languages.rs`, `core/src/limits.rs`.

- New meetings and imports use the language default from Settings, which is **Auto** unless the user picks one of Cohere's 14 supported languages. Changing Settings never changes an existing job.
- In Auto mode, Whisper tiny scores language on short probes before Cohere runs. A supported winner applies when its probability is at least **0.90** and its margin is at least **0.20**. These are closed-set scores, not accuracy guarantees.
- An uncertain probe first uses the speaker’s main language when independent, confident acoustic observations establish it. Otherwise it uses the meeting’s main language. This evidence belongs to the same audio source, language epoch and stream generation; a pause does not erase it. Saved accuracy can use later observed speech to resolve an earlier uncertain passage. Borrowed language never becomes a new acoustic vote. Confident local language switches and confident unsupported-language detections take precedence; explicit language choices remain authoritative. Without qualified context, the existing best-effort or explicit-language repair behavior applies.
- Accepted speech and accuracy results update the main-language evidence during the same operation, including changes to an earlier direct observation. Stop retries use the evidence accepted by then; a pause or a new saved pass is not needed to learn later speech.
- Short, quiet and overlapping speech are still attempted. Only exactly constant PCM skips detection and ASR. It is recorded as `digital_silence` (all zero) or `constant_signal` (nonzero DC).
- **During a meeting**, the header selector changes the language from the current saved-recording sample onward. This starts a new language epoch. It is accepted only while recording or paused, and only after the previous boundary has been acknowledged. A stale `language_revision` returns 409. Earlier text keeps its language. To revisit earlier words, retry the passage.

## Speech admission

Owner: `core/src/speech.rs`.

Silero v6 runs on two views of the original audio: raw, and a peak-conditioned view (gain `min(256, 0.25 / peak)`). It uses the maximum of the two probabilities, with 0.5 onset and 0.35 release hysteresis. Diarization and Cohere always receive original audio. No amplitude threshold or word filter decides what counts as speech. A missing or failed inspection never establishes silence, and empty ASR output never does either.

## Live text and refinement

Owner: `core/src/live.rs`, `host/src/routes/refinement.rs`.

- Windows microphone recording includes echo cancellation and noise suppression. Its playback reference follows the default speaker endpoint even when **System audio** is unchecked. That checkbox controls what is retained and transcribed; a private cancellation reference never becomes a saved system track. Capture fails visibly if required processing is unavailable instead of silently substituting unprocessed audio.
- Live meetings show provisional text, which becomes canonical utterances as speech settles. Rolling refinement improves canonical text during the meeting. Stop runs a bounded larger-context pass before the meeting is finalized.
- Accuracy keeps each short saved passage's complete original audio anchor for decoding. Fresh speech admission and language routing still apply; a new speech boundary does not trim away that passage's opening context.
- Complete words can finish the accuracy pass even when automatic language routing is uncertain. The language evidence stays in passage details; short passages containing multiple supported languages keep separate timing receipts. Empty, truncated, unsupported or failed results retain the previous words. Accuracy continues through other eligible passages and finishes with passage exceptions when local work fails. Each exception keeps its reason beside the retained passage; runtime interruptions remain global and retryable. Unanchored coverage gaps and protected edits are skipped. Stop retries failed live passages once; it does not repeat failed work indefinitely.
- Words remain in one chronological reading column as accuracy work finishes. Pending machine work does not create a manual **Needs review** region. After recording stops, passages blocked by language appear at their timestamps with a reason and visible language controls, including a failed language recheck whose previous words were retained. **Processing & help** points to those passages, including when search has hidden them, and holds routine accuracy, separation readiness and keyboard help. Recording/import progress, save state and service failures remain visible. Successfully accepted readable uncertainty keeps its evidence in passage details; silence does not become a repair task. A retained correction whose original passage was replaced stays in its own section.
- Server-only language warnings and candidates do not block autosave. Returned warning metadata refreshes beside passages while focused or newer drafts stay intact; concurrent changes to passage authority still conflict.
- Speaker badges match unnamed labels such as **Speaker 2**, then show initials when a person is named. Unlinked track colors remain stable through naming and save/reopen. Tracks assigned to the same saved person share one color and reading continuity; internal track IDs never determine displayed speaker numbers. Mixed and unassigned audio do not take a speaker number.
- Refinement never overwrites a **protected edit** (user words, speaker or bounds). It never changes names or neighbouring passages, and never alters the original audio.
- A short unassigned tail in an otherwise single-speaker passage can read inline, explicitly marked as unassigned. Its acoustic ownership and timing remain unresolved; distinct speakers, overlap and standalone unknown speech retain their own paragraphs.
- Transcript timing (supplied-text CTC) is required for transcription. The Mac provider functionally covers all 14 languages, with accuracy calibrated in English only. The Windows candidate uses a separate provider identity and claims no calibrated timing languages, including English. CJK text uses character units. Units that are unsupported or uncertain stay unresolved instead of being guessed. Invalid or missing acoustic scores never acquire a timing calibration claim.
- While a live passage grows, unchanged words retain their validated timing on the previously transcribed audio prefix. Speaker attribution is recalculated against current observations. New audio does not extend the words' timing evidence; changed words, sources, languages or protected edits invalidate reuse.

- An ordinary language, transcription or timing failure affects its passage; later
  speech can still transcribe. A timing-model failure retains decoded words without
  claiming resolved timing. Child loss and hardware/runtime failures still use the
  recording recovery path.
- Late audio or capture overload interrupts the meeting visibly. Already accepted
  original audio is retained, and playback is repaired where possible. Corrections
  survive repair; text and acoustic claims affected by the repaired audio are marked
  incomplete instead of being silently trusted.

## Imports and editing

Owner: `core/src/import/pipeline.rs`, `host/src/routes/import_progress.rs`,
`speakerdesk/static/app.js`, `speakerdesk/static/live_editor.js`.

- Imports show accepted passages while the remaining audio is processing. Those
  passages can be corrected immediately; later batches and Resume preserve the
  accepted corrections. Whole-document replacement and structural controls wait
  until the progressive import finishes.
- Cancel stops preparation or import work and retains the source and any accepted
  passages. Resume continues a compatible checkpoint; changing source or processing
  identity invalidates stale work. Cancel is safe to repeat while ownership drains.
- Imports retain their captured language setting and use the same main-language
  policy and passage eligibility as live recordings and saved accuracy.
- Timing edits are staged until valid and saved. Failed saves or failed navigation
  retain the current words, timing drafts and playback. Saved accuracy can update
  other passages while a correction remains focused. Save and export state includes
  every outstanding draft, including invalid timing and recoverable corrections.
- **Delete passage** removes the full underlying passage, including all its speaker paragraphs, from reading and every export. It keeps the original audio and a protected deletion record so background accuracy, live publication, import continuation and stale document content cannot recreate it. The existing **Undo** notice restores the passage through the same revision-checked edit path.
- Hovering or focusing a continuation passage reveals its compact **Passage actions** button without moving the transcript. Opening that button expands the controls; repairable passage problems stay visible.
- The native app keeps private recovery copies of unsaved corrections. On normal
  close it waits for a durable copy; on failure it stays open with a copy path.
  After a host restart, recovered corrections can be reviewed and copied without
  automatically changing the saved transcript. Abrupt crashes may lose edits made
  after the last periodic copy.
- Delete removes owned recording media and retained recovery copies. Cleanup failure
  remains visible as **Cleanup failed** and offers retry; it is not reported as
  successful deletion. Original media from a migrated library remains read-only.
- Theme follows the saved host preference across restarts. Service loss and startup
  failure show an explicit status and retry/copy actions. Models controls preserve
  focus as status refreshes, and recording controls remain reachable on narrow windows.

## Passage retry and review

Owner: `host/src/routes/retry.rs`, `core/src/review.rs`, `core/src/export.rs`.

- After saving edits, the user can retry one saved passage of at most **24.5 s** in an explicitly chosen language. Auto is not offered for retry. The result is a separate **retry candidate**. It replaces the passage only when the user chooses **Use this text** and saves. Empty or failed candidates never erase existing words.
- The passage action is **Retry transcription**. **Edit full passage** identifies the number of speaker paragraphs and the full time range being edited. Passage **Details** holds **Compare prior words**, which can restore the text before the first accuracy replacement; restoration is a protected user edit. This choice does not establish which machine version is acoustically correct. Raw inference metadata lives in collapsed **Technical diagnostics**, with a copy action.
- Saving a newly accepted language repair resumes the remaining incomplete accuracy work automatically. The repaired text stays protected, and the other passages retain their captured language settings and epochs. Repeated saves, stale candidates and unrelated edits do not start another pass. Longer language-blocked passages show the retry limit and expose timing adjustment. The language-repair timing control and passage retry require outstanding passage drafts or saves to finish.
- Passages carry review reasons: unknown language, empty ASR, model error, partial output, overlap, unassigned audio and uncertain speaker. **Mark words reviewed** resolves the words. Overlap and unassigned-speaker warnings remain, because reviewing words does not settle who spoke.
- TXT/SRT/VTT exports skip blank passages and print `[Needs review: …]` markers. JSON keeps every range, its reasons and any candidates. Exported times are audio anchors, not phonetic word boundaries.

## Overlapping voices

Owner: `core/src/separation.rs`, `core/src/voice/separation.rs`, `host/src/routes/separation.rs`, `host/src/routes/separation_identity.rs`.

- During recording, Speakerdesk automatically processes completed, unedited passages with observed overlap between exactly two voices, within one language setting and at most **24.5 s** long. The existing recording worker resumes long separation and separated transcription in steps, processing pending live audio and controls between steps. Neural work remains serial; a running model step must finish before it yields. The same automatic processing runs after imports and saved accuracy work. It uses the required, locally prepared separation model from standard model setup.
- A successful attempt saves two voice estimates beside the original passage. The original words and speaker paragraphs remain the reading view. **Compare voice estimates** offers original, Voice A and Voice B playback with play/pause controls. Nonempty generated text alone never replaces the original. After recording and processing finish, **Use both voices** explicitly accepts the estimates as two editable rows; **Keep original** discards them.
- With recognition enabled and a qualified separated-voice policy, accepting a new estimate can then associate each row with an existing meeting track or compatible saved voice. Output channel order never identifies a speaker. A strong match can be used while its sibling stays unassigned; ambiguous estimates or two estimates matching the same identity stay unassigned. Saved-voice matches create unconfirmed assignments; reusing a meeting track preserves its current name and manual assignment. Generated estimates remain ineligible for voice enrollment. The original words and audio remain in passage history, and an unchanged pair offers **Use original transcript and audio**. Older candidates without saved speech ranges remain usable without automatic identity inference.
- Protected edits, stale source observations, pending candidates and already separated rows are skipped. Each original passage's source/model pair is attempted once automatically, including failed or interrupted attempts and restored originals; live child-loss recovery may retry the interrupted passage under the new child generation. Failure, cancellation or edits during processing retain the current words and original recording. A new accuracy result invalidates estimates based on the superseded passage.

## Model setup

Owner: `host/src/setup.rs`, `host/src/native.rs`, `speakerdesk/static/app.js`.

- All installations use the same required local models, including overlapping voice separation. First setup and upgrades missing separation offer the normal **Download models** action before new transcription or recording starts. Downloads require that explicit action; startup only checks installed models.
- A setup attempt succeeds when all required models have activated and initialized with the expected identities. A prepared file alone does not establish separation readiness. Already verified models are reused when finishing or retrying setup.
- Meeting start and the transcript show separation readiness before users need Settings. Settings puts the shared download/retry action before model explanations. A failed or interrupted attempt offers **Retry setup** for the required model set. Detailed model and timing explanations remain collapsed. Saved recordings and corrections remain accessible during repair, and successful setup restores transcription and recording readiness.
- The last eight explicit setup attempts retain their target, start time, activation state and final error on this device. Retrying does not erase the first failure. An interrupted attempt remains identifiable after restart.

## People and voices

Owner: `core/src/people.rs`, `core/src/voice.rs`, `host/src/people_routes/`.

- People stores reusable names. Naming a speaker track applies to the current meeting only, and never renames a person or changes a saved voice.
- Assigning several detected voices to the same saved person shows one person in the speaker strip and passage selectors, including after reopening and later accuracy updates. Equal names alone never combine speakers. The person strip changes all voices currently assigned to that person in the meeting. A passage's name control, under **Timing & details** for a saved passage, changes only its detected voice and can unlink or reassign it. Original acoustic tracks, passages and voice evidence remain separate; voice enrollment still requires choosing clips from one detected voice. JSON exports retain those tracks and their saved-person assignments.
- **Remember voice** is the single explicit consent action. It needs a confirmed person and at least two finalized, non-overlapping clips of **2–10 s** each. Long passages offer separate 6-second windows. Adding a name, downloading models, or confirming a recognition never enrolls a voice.
- Recognition is on by default and can be turned off in Settings. Automatic matches are labelled Recognized and stay unconfirmed. They never create or update a profile. Profiles from an incompatible model identity are kept but take no part in recognition until the user replaces them with consent. **Forget voice** removes only the stored voice; names, recordings and text remain.
- Saved-voice recognition runs during recording as soon as enough observed solo audio is available, including a stable prefix of an ongoing passage. It uses the same calibrated matching rules as saved recordings and yields to pending live audio between clips. Later source changes, manual naming, disabled recognition, or profile replacement/forgetting invalidate an outstanding match.
- Introduction suggestions come from conservative English patterns ("I'm …", "my name is …") and need the user's confirmation.

## Updates and shortcuts

Owner: `desktop/src-tauri/src/updates.rs`, `menus.rs`, `host/src/native_updates.rs`.

- The app checks for an update once when it opens. A notice offers **Download update…**, and installing needs a separate **Install and restart**. Dismissing a notice hides that version across restarts, and a newer version shows a new notice. **Speakerdesk → Check for Updates…** checks manually.
- Download and install are refused while recording, processing, model setup or saves are active. Downgrades are disabled.
- Windows setup offers a per-user installation or an all-users installation in Program Files. All-users setup requires administrator authorization. Updates retain the running installation's scope and registered folder; an all-users update can request elevation while Speakerdesk itself restarts as the desktop user. An unwritable destination is refused before copying the app or recording a new installation. Meetings and models remain in the user's existing data directory in either scope.

| Shortcut | Action |
| --- | --- |
| ⌘, | Settings |
| ⌘N | New Meeting |
| ⌘O | Import recordings |
| ⌘F | Find in transcript / search meetings |
| ⌘S | Save transcript edits |
| Escape | Close Settings; clear an active search |
