# Unknown names, unassigned words and acoustic tracks

Source: released 0.6.1 `01d36a05108cf1fffb2bf149ddc4698c11e9a345`.
These CPU controls exercise `Engine.handle`, Stop, canonical batch refinement,
host reconciliation, saved SQLite jobs and GET `/api/jobs`. Acoustic outputs and
calibrated timings are explicitly fabricated, so they do not measure recognition
or diarization accuracy on the user's meeting.

## Distinct source mechanisms

With sequential `speaker_0`/`speaker_1` activity and qualified English timing,
saved reading turns retain both identities and their default Speaker 1/2 names.
When timing is missing, both activity IDs still survive, while unsplit Cohere
words are `unassigned`. A generic Unknown speaker label can consequently look
like one shared person. It is an attribution placeholder, not an acoustic ID.
The frontend owner is correcting that presentation separately.

An actual single-track diarizer output cannot establish two acoustic identities.
The source control retains one identity in that case. Person-name matching is a
separate process: synthetic unmatched saved-voice checks preserve both acoustic
track IDs and default names, rather than merging or clearing them.

A further reproducible source loss occurs when the live pass used one global
track but the batch pass separates two disjoint local slots. Temporal matching
correctly refuses two ambiguous claims on the same old global identity. Canonical
refinement then replaces both local slots with `unknown_mixed`, discarding their
distinct raw identities from the saved result. That data loss is reproducible
from supplied model outputs alone.

## Narrow correction

Canonical refinement now retains `speaker_track_mapping` with the exact batch
turns, mapping scores/reasons, successful global correspondences, and bounded
local activity. The receipt is scoped by utterance, refinement operation, original
audio anchor and audio revision. Physical anchor growth invalidates it.

Unpaired local slots are evidence from that one batch operation. They are not new
meeting-wide identities, person names or ownership of unaligned words. Existing
global identities, raw Cohere text, qualified reading turns and ambiguous-mapping
decisions remain unchanged. Failed refinement restores the previous receipt and
saved document. The frontend owner is adding the receipt to Inference evidence,
without numbering generic unassigned passages or assigning their words.

The independent checkout changes only `canonical_runtime.py` and the existing
anchor-invalidation list in `utterances.py`. No shared integration checkout,
user audio, capture, GUI or model/dependency copies were involved.

## Verification and coordinated model replay

`tests/test_unknown_speaker_identity.py` covers qualified two-track identity,
missing timing, unpaired batch slots, successful named correspondences,
single-track output, failed refinement rollback and unmatched person names. The
two receipt regressions fail on released source because the evidence is absent;
the identity/uncertainty controls already pass.

The integrator should replay the existing owned `a-b` and `a-b-a` two-voice fixtures
from `tests/acceptance/sequential-attribution-recipes.json`, in English and
Automatic, through real Models/Engine and saved host/API. The existing generation
and serial resource instructions are in `docs/backend-attribution-replay.md`.
No user audio, new downloads or active app interaction is needed.

Retain every live NVIDIA track, raw batch slot, temporal mapping score, timing
qualification/failure, raw Cohere revision and displayed identity. Evaluate the
stages separately: actual same-slot output is a diarizer limitation; different
slots with unresolved correspondence remain local evidence; distinct mapped
tracks with absent timing do not authorize word assignment. Confirm that an
unmatched person name leaves stable acoustic IDs intact. CPU results alone cannot
establish the cause of the reported real-meeting symptom or qualify accuracy.
