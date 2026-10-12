# Glossary

These names are the product language for Speakerdesk. They describe what a person sees and decides, not how a build is put together.

**Job**
An Import or a Meeting. An Import starts from audio files the user chooses. A Meeting is recorded live.

**Transcript**
The text of a job, with speakers and passages, that the user can read, edit, and export.

**Passage**
One stretch of transcript text with a start and an end. It is stored as a "segment".

**Speaker track**
A meeting-local slot from diarization, not a person. Tracks are numbered inside one job.

**Person**
An entry in People. A person can appear in more than one meeting.

**Speaker name**
The label shown for a speaker track or a person on a passage.

**Overlap**
Audio where more than one person is speaking at the same time.

**Unknown speaker**
Audio that is not assigned to a speaker track. The product calls this unassigned.

**Capture source**
Where a sound came from: microphone_clean or system. This is provenance, not identity.

**Recording clock**
The single sample timeline a meeting's tracks share.

**Language mode**
Automatic detection, or a language the user sets as an override.

**Language epoch**
The span of a meeting that shares one language choice. A language change starts a new epoch.

**Speech admission**
The decision that a stretch of audio is speech and may be transcribed.

**Canonical utterance**
A finished spoken stretch that later refinement is allowed to improve.

**Provisional text**
Transcript text shown before that stretch is canonical.

**Finalized**
A meeting or passage that will not change unless the user edits it or asks for refinement.

**Refinement**
A second pass over audio. It is either rolling, during a meeting, or saved, after a canonical meeting is stored.

**Protected edit**
A user change to words, speaker, or bounds that refinement must not overwrite.

**Retry candidate**
A proposed replacement for one passage, produced when the user asks for that passage again. It is not the passage until the user accepts it.

**Reading turns**
Coarse word timing inside a passage.

**Review reason**
Why a passage is marked for the user to check.

**Voice profile**
The saved description of a person's voice, used only after they agree to remember it.

**Enrollment**
"Remember voice": the user agrees to save a voice profile from clips they confirm.

**Recognition**
Matching a speaker track to an enrolled person.

**Introduction suggestion**
A possible person name heard in the words of a meeting, offered for the user to confirm.

**Model setup**
Downloading and checking the local models a transcription needs before it can run.
