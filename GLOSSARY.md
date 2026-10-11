# Speakerdesk

A local-only Mac app that turns imported recordings and live meetings into speaker-attributed, editable transcripts.

## Language

### Jobs and transcripts

**Job**:
One saved unit of work with its audio and transcript. It is either an **Import** (an uploaded recording) or a **Meeting** (a live capture).
_Avoid_: Task, session, file

**Transcript**:
The ordered set of passages, speakers and warnings that belongs to one job.
_Avoid_: Document (except when talking about the stored shape), result

**Passage**:
One speaker-attributed span of transcript text with start and end times. It is stored as a "segment".
_Avoid_: Line, utterance (an utterance is a speech span, not text), chunk

**Review reason**:
A machine-raised flag telling the user that a passage needs a human look, for example uncertain speaker or uncertain language.
_Avoid_: Warning (warnings are transcript-level), error

**Retry candidate**:
An alternative passage text produced by re-running transcription on request. The user must accept it before it replaces the passage.
_Avoid_: Suggestion, revision

**Protected edit**:
A user change to a passage field that later machine refinement must not overwrite.
_Avoid_: Lock, manual override

### Speakers and people

**Speaker track**:
A diarizer slot local to one job, such as "Speaker 2". It is not a person.
_Avoid_: Speaker (when the cross-meeting person is meant), voice

**Speaker name**:
The label the user gives a speaker track within one job.
_Avoid_: Person (unless linked)

**Person**:
A People entry that persists across jobs and can be linked to speaker tracks.
_Avoid_: Contact, user, speaker

**Overlap**:
A stretch where more than one speaker track is active, so attribution of the text is uncertain.
_Avoid_: Crosstalk

**Unknown speaker**:
Speech that no speaker track could be confidently assigned to. Its audio is kept so it can be reviewed.
_Avoid_: Unassigned speaker, Speaker 0

**Introduction suggestion**:
A proposed speaker name drawn from someone introducing themselves in the transcript ("I'm Dana"). It is never applied automatically.
_Avoid_: Auto-name

### Voice

**Voice profile**:
A saved voice signature for a person, built only from clips the user chose.
_Avoid_: Voiceprint, embedding (an embedding is one clip's vector)

**Enrollment**:
The explicit, consented act of creating or extending a voice profile. In the UI it is "Remember voice".
_Avoid_: Training, auto-enroll

**Recognition**:
Matching a new speaker track against saved voice profiles to propose a person. One decision is made per track.
_Avoid_: Identification, verification

### Audio and capture

**Capture source**:
Where meeting audio came from: the cleaned microphone or system audio. It is provenance, not speaker identity.
_Avoid_: Channel, input, speaker

**Recording clock**:
The single sample timeline of a meeting's retained recording. All transcript times refer to it.
_Avoid_: Wall clock, stream time

**Speech admission**:
The decision that a span of audio contains speech worth transcribing.
_Avoid_: VAD result, gating

**Canonical utterance**:
A stable speech span in a meeting whose boundaries have stopped changing. Its text is the one the meeting keeps.
_Avoid_: Final segment, chunk

**Provisional text**:
Live transcript text shown before its canonical utterance settles. It may still change.
_Avoid_: Partial, draft

**Finalized**:
The state of a meeting transcript after Stop, once every canonical utterance has been transcribed.
_Avoid_: Complete, done

**Refinement**:
Re-processing an already-transcribed meeting to improve speakers and text while keeping protected edits. **Rolling** refinement runs during the meeting; **saved** refinement runs on a stored meeting.
_Avoid_: Reprocessing, second pass

### Language and timing

**Language mode**:
Either **Automatic** (detected per stretch of speech) or an **override** chosen by the user.
_Avoid_: Locale

**Language epoch**:
A span of a meeting governed by one language mode. Changing the language mode starts a new epoch.
_Avoid_: Language segment

**Reading turns**:
Coarse word timing for passage text, used to follow along and to export captions. It may be explicitly unknown.
_Avoid_: Word timestamps, alignment (alignment is the technique)

### Setup

**Model setup**:
The one-time download and verification of the local models needed before transcription is admitted.
_Avoid_: Install, onboarding
