# Speakerdesk

The product language for Speakerdesk. These terms describe what a person sees and decides, not how a build is put together.

## Jobs and transcripts

**Job**:
An Import, which starts from audio files the user chooses, or a Meeting, which is recorded live.
_Avoid_: Session

**Transcript**:
The text of a job, with speakers and passages, that the user can read, edit and export.

**Passage**:
One stretch of transcript text with a start and an end.
_Avoid_: Segment (the storage name), utterance, row

**Reading turns**:
Coarse word timing inside a passage, used for highlighting and per-speaker turns.
_Avoid_: Word timestamps, alignment

**Review reason**:
Why a passage is marked for the user to check.

## Speakers and people

**Speaker track**:
A meeting-local slot from diarization, not a person. Tracks are numbered inside one job.
_Avoid_: Speaker ID, slot

**Person**:
An entry in People. A person can appear in more than one meeting.
_Avoid_: Speaker, contact

**Speaker name**:
The label shown for a speaker track or a person on a passage.

**Overlap**:
Audio where more than one person is speaking at the same time.

**Unassigned**:
Words or audio not attributed to any speaker track. It is shown as "Speaker unassigned" and is a placeholder, not one shared person.
_Avoid_: Unknown speaker

**Voice profile**:
The saved description of a person's voice, kept only after they agree to remember it.
_Avoid_: Voiceprint, embedding

**Enrollment**:
"Remember voice": the user agrees to save a voice profile from clips they confirm.

**Recognition**:
Matching a speaker track to an enrolled person. A match stays unconfirmed until the user confirms it.
_Avoid_: Identification

**Introduction suggestion**:
A possible person name heard in a meeting's words, offered for the user to confirm.

## Audio and language

**Capture source**:
Where a sound came from: the microphone or the Mac's system audio. This is provenance, not identity.

**Recording clock**:
The single sample timeline a meeting's tracks share.

**Speech admission**:
The decision that a stretch of audio is speech and may be transcribed.

**Language mode**:
Automatic detection (Auto), or a language the user sets as an override.

**Language epoch**:
The span of a meeting that shares one language choice. A language change starts a new epoch.

## Text lifecycle

**Provisional text**:
Transcript text shown before that stretch is canonical.

**Canonical utterance**:
A finished spoken stretch that later refinement is allowed to improve.

**Finalized**:
A meeting or passage that will not change unless the user edits it or asks for refinement.

**Refinement**:
A second pass over audio. Rolling refinement runs during a meeting; saved refinement runs after a canonical meeting is stored.

**Protected edit**:
A user change to words, speaker or bounds that refinement must not overwrite.

**Retry candidate**:
A proposed replacement for one passage, produced when the user asks for that passage again. It does not become the passage until the user accepts it.

**Model setup**:
Downloading and checking the local models a transcription needs before it can run.
