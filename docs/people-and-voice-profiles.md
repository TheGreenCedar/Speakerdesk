# People and speaker identity

People saves local names with stable person IDs. A meeting's diarizer track and
each transcript segment have separate IDs. The same `speaker_0` in a later
meeting starts unknown; Nemotron slots and cache never supply cross-meeting
identity. Selecting a saved person confirms a name for the current meeting.
Ordinary name corrections do not rename People or update voice profiles.

Finalized local transcript passages can suggest an explicit introduction such
as “I'm Priya” with its quote and speech-region time. Addressed names such as
“Priya, what do you think?”, quoted/reported speech, overlap, pending text and
lowercase uncertain names remain unknown. These conservative English rules
produce proposals only. They do not create a person or infer an identity
without confirmation; there is no LLM call.

SQLite adds `people` and `voice_profiles` tables alongside existing jobs.
Confirmed assignments and their evidence belong to the job and are server
owned. Project JSON exports include confirmation evidence but never embeddings.
Imported projects cannot claim confirmed identities. Existing meetings need no
migration or reenrollment. Renaming a saved person preserves meeting name
snapshots. Forget voice deletes the stored centroid and enrollment evidence;
names, original recordings and transcript text remain. It also invalidates
pending voice matches.

## Voice adapter gate

The local ReDimNet2 Core ML extractor, explicit downloader and offline measurement
harness are implemented. Model bytes, optional runtime installation and measured
calibration remain pending approval and evaluation. `create_app()` keeps voice
controls unavailable until a locally reviewed compatible calibration is supplied.
Adding a name or confirming any suggestion never enrolls or refreshes a voice.

An approved local adapter must implement `LocalVoiceBackend.embed(audio, clip)`
and return a finite vector plus a waveform eligibility result. The current
extractor screens silence/clipping, not acoustic noise or overlapping voices;
passage selection and difficult-negative evaluation remain necessary.
`VoiceModel` pins model ID, revision, artifact SHA-256, dimension and 16 kHz
input. Different versions or conversions are incompatible spaces. Clip IDs
are qualified by meeting and track. Enrollment requires explicit consent, a
confirmed saved person, and at least two distinct finalized, nonoverlapping
2–10 second clean passages. The adapter can reject unusable clips; a rejected
clip saves no partial profile. Only a normalized centroid and source evidence
are saved, with a fresh enrollment version and consent timestamp.

`Calibration` requires a named measured dataset, threshold, winner margin,
genuine/impostor trial counts, false accept/reject rates and a minimum of two
clips. No numeric production policy is guessed. Every clean query clip must
pass the threshold, and the mean winner must clear the runner-up margin.
Low scores, ambiguity, insufficient evidence or incompatible profiles return
unknown. Proposals carry model/calibration/clip evidence and require a separate
confirmation. Confirmation never updates the enrolled centroid. Profile
replacement, forgetting, a different adapter/calibration, or transcript edits
invalidate pending voice evidence.

## Pinned artifact and measured plan awaiting approval

See [the integration plan](redimnet2-integration.md) for exact source/license,
artifact hashes, dependencies, installation and bounded test actions. The current
artifact totals 30,036,479 bytes; the old 24.7 MiB card figure is stale. Apple
Silicon macOS 15+ is required. The first gated step is an eight-prediction synthetic
extraction smoke, after CodeStory releases its resource window. It does not enroll
People or establish cross-meeting accuracy. Independent calibration/held-out
meetings and reviewed results are required before enabling recognition.

## Proof in this patch

CPU tests use synthetic vectors and temporary SQLite databases. They cover
persistence across app restarts/meetings, token enforcement, name-only defaults,
intro/address negatives, stale evidence, explicit enrollment, clean-clip gates,
model compatibility, unknown/ambiguity, confirmation without profile mutation,
forgetting without transcript loss, server-owned import/export provenance and
live name preservation. They do not prove real voice recognition quality,
Core ML conversion behavior, native capture or packaged model parity.
