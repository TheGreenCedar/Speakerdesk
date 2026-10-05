# People and speaker identity

People stores reusable local names and stable person IDs. Each meeting has separate diarizer track IDs, and each transcript passage has a segment ID. Nemotron slots and cache identify tracks within the current meeting; they never supply cross-meeting identity.

Voice recognition defaults on and can be turned off in Settings. The pinned B6 model is included in standard local model setup. A new unknown track collects at least two separate clean audio clips, then the local extractor checks compatible saved voice profiles once. A passing measured score and runner-up margin apply the saved person's name to that meeting track. The result is carried forward by diarization; later phrases do not repeatedly embed audio. Failed, ambiguous, incompatible or insufficient evidence leaves the track unknown. Long import crops can contribute separate six-second windows without changing the transcript.

Automatic assignments are labelled Recognized and retain model, policy, profile version, clip time and meeting/track/segment provenance. They are explicitly unconfirmed. Choose the speaker name to correct or confirm it. User corrections win over pending work and apply to the current meeting; they never rename People or update a saved voice. A forgotten/replaced profile, changed selected passage, disabled preference or changed runtime invalidates a pending result. Existing transcript names are retained when recognition is turned off or a voice is forgotten.

Finalized local transcript text can suggest an explicit introduction such as “I'm Priya,” with its quote and speech-region time. Addressed names such as “Priya, what do you think?”, quoted/reported speech, overlap, pending text and lowercase uncertain names produce no introduction proposal. These conservative English rules require confirmation and use no LLM. A conflicting explicit introduction keeps an automatic voice decision unknown.

## Explicit enrollment

Adding a name, downloading models, recognizing a voice or confirming a name never enrolls or refreshes a voice. Remember voice requires explicit consent, a confirmed saved person, and at least two selected finalized, nonoverlapping 2–10 second passages. Long import passages offer separate six-second audio windows for selection. An automatically assigned name must be explicitly confirmed first. Enrollment and manual voice checks wait until active recording/transcription ends. Rejected audio saves no partial profile.

Only the normalized centroid and enrollment evidence are stored in SQLite, with an exact model identity, fresh profile version and consent timestamp. No extra audio copy is made, and no audio is uploaded. Forget voice removes the centroid and enrollment evidence; names, source recordings and transcript text remain. Renaming People preserves existing meeting name snapshots. JSON transcript exports contain assignment provenance, never embeddings. Imported projects cannot manufacture server-owned identities or clean-audio evidence.

## Model and policy contracts

`VoiceModel` pins model ID, revision, artifact digest, dimension, preprocessing and compute policy. Different conversions or versions are separate embedding spaces. `Calibration` records a measured dataset, score threshold, winner margin, genuine/impostor counts and observed error rates. Every query clip must pass the threshold, and the mean winner must clear the runner-up margin. No numeric policy is guessed. A passing frozen held-out evaluation enables the shipped policy without a separate user configuration or approval step; extraction smoke alone cannot enable it.

The extractor screens silence and saturation. It does not independently classify room noise, overlap or speaker changes. Single-speaker diarization evidence is required. Six-second phrase boundaries can need text review while the audio remains eligible; overlapping/ambiguous tracks stay excluded. The current live worker keeps stable track IDs for the meeting. A future diarizer reset must create a new logical track rather than reuse and relabel a historical track.

See [the model integration and measurement evidence](redimnet2-integration.md) for the artifact, license, resource budget and measured limits. Synthetic fixtures exercise different scripted sessions and unseen voices; they do not establish human cross-meeting accuracy across microphones or rooms.

## Verification

CPU tests cover persistence, default-on/off controls, one check per track, new-meeting slot reuse, unknown fallback, long import windows, introduction/address negatives, source-owned clip eligibility, correction/forget/toggle races, explicit enrollment, model compatibility, import/export provenance and live finalization. A negative control removing the unknown-track cache caused twelve extractions instead of two; restoring the cache passed. The shipped policy passed eight held-out genuine and eight unknown synthetic trials with no errors; native smoke and resource results are recorded separately. Packaged runtime behavior and recognition running alongside real capture/transcription remain unverified in this source-only integration.
