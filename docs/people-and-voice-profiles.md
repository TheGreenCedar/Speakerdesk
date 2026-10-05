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

This patch includes a local backend interface and opt-in storage/matching flow.
It does **not** include a model extractor, installer, model weights, or measured
calibration. `create_app()` therefore keeps voice controls unavailable. Adding
a name or confirming any suggestion never enrolls or refreshes a voice.

An approved local adapter must implement `LocalVoiceBackend.embed(audio, clip)`
and return a finite vector plus its independent clean-audio assessment.
`VoiceModel` pins model ID, revision, artifact SHA-256, dimension and 16 kHz
input. Different versions or conversions are incompatible spaces. Clip IDs
are qualified by meeting and track. Enrollment requires explicit consent, a
confirmed saved person, and at least two distinct finalized, nonoverlapping
2–10 second clean passages. The adapter can reject noisy clips; a rejected
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

## Candidate and measured plan awaiting approval

The candidate is [ReDimNet2-B6](https://github.com/PalabraAI/redimnet2), whose
official repository lists MIT licensing and 12.3 million parameters. The
[aufklarer Core ML conversion](https://huggingface.co/aufklarer/ReDimNet2-B6-CoreML)
lists MIT licensing, 24.7 MiB compiled float16 weights, 16 kHz mono input,
96,000 samples (six seconds), a 192-dimensional normalized output, and
macOS 15 or newer. Its card describes repeating clean 2–6 second clips and
center-cropping longer clips. These are publisher claims, not measurements
from this lane. Newer Apple Silicon macOS is an accepted requirement.

Before any model installation or inference, obtain approval for the exact
community artifact and measured run plan. Inspect the conversion source,
license and config metadata; pin revision/checksums and a local-only extractor.
Account for compiled weights, dependency changes, compilation cache, audio
fixtures and measurement output. Preserve the 40 GB free-space floor and
wait for CodeStory's resource window to finish. No recordings are uploaded,
no user voice is enrolled by a benchmark, and no OS permissions are changed.

The proposed measured run uses explicitly approved synthetic/licensed fixtures
with speakers split between enrollment and held-out meetings, including
unseen voices, similar voices, mixed microphones/languages, noise and overlap.
Verify conversion parity, six-second preprocessing, finite normalized outputs,
warm/cold latency and peak memory on the target Mac. Measure genuine/impostor
score distributions and false accepts/rejects. Select a threshold and margin
on a calibration split, then freeze and test on held-out meetings, preserving
unknown where evidence is insufficient. Report per-condition failure rates,
not just an equal-error rate. No acceptance target or production threshold is
claimed until the plan and measured results are accepted.

## Proof in this patch

CPU tests use synthetic vectors and temporary SQLite databases. They cover
persistence across app restarts/meetings, token enforcement, name-only defaults,
intro/address negatives, stale evidence, explicit enrollment, clean-clip gates,
model compatibility, unknown/ambiguity, confirmation without profile mutation,
forgetting without transcript loss, server-owned import/export provenance and
live name preservation. They do not prove real voice recognition quality,
Core ML conversion behavior, native capture or packaged model parity.
