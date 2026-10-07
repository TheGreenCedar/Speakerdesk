# Reading attribution at artificial decode seams

The completed Piers Morgan original 54–84 second public crop on source
`fedd5bb510b229173ae9af57bb4364a6b7862bc8` has 95 Cohere words, nine saved
reading slices, and active qualified English CTC timing. Its parent
`multiple_speakers` label does not mean that its readable speaker turns merged.
The actual remaining defect affects six words around artificial 14 second
decode cores. The assembler clears their canonical timing and core ownership,
then the reading projection receives those nulls despite retaining exact
calibrated envelopes in the chosen raw provider output.

All times below are relative to that 30 second crop; add 54 seconds for the
original source position. They are retained provider emission envelopes, not
phonetic boundaries or independently measured acoustic truth.

| Word | Selected envelope (seconds) | Existing evidence and disposition |
| --- | --- | --- |
| That's | 3.30–3.58 | Emission crosses single/overlap activity and a gap; remain unknown. |
| what | 3.60–3.68 | Full 250 ms guard touches single and overlap activity; remain unknown. |
| I'm | 3.76–3.82 | Full guard touches single and overlap activity; remain unknown. |
| Okay, | 4.00–4.14 | Full guard touches overlap and single activity; remain unknown. |
| well, | 4.24–4.36 | Full guard touches overlap and single activity; remain unknown. |
| let | 4.38–4.48 | Full guard touches overlap and single activity; remain unknown. |
| talked | 13.60–13.78 | Only Speaker 2 in the emission and full guard; recover reading owner. |
| to | 13.82–13.86 | Only Speaker 2 in the emission and full guard; recover reading owner. |
| him | 13.88–13.96 | Only Speaker 2 in the emission and full guard; recover reading owner. |
| specifically | 14.08–14.80 | Only Speaker 2 in the emission and full guard; recover reading owner. |
| Donald | 27.70–27.94 | Only Speaker 2 in the emission and full guard; recover reading owner. |
| Trump | 28.02–28.24 | Only Speaker 2 in the emission and full guard; recover reading owner. |

## Narrow source change

`authoritative_tail.assemble` retains `source_emission_envelope` only when its
existing qualified, exact raw-offset candidate loses timing to a rollover guard.
The canonical `start_sample`/`end_sample` stay null, `core_ownership` stays
`unknown`, and `alignment_complete` stays false. Selected raw text, source
decode identity/hash, both rollover receipts, and every text cut stay unchanged.

`authoritative_tail.reading_words` makes a separate projection using those
already produced envelopes. It rejects conflicts with both earlier and later
retained word timings. `CanonicalRuntime.decode` binds that projection to the
existing reading evidence. The existing native activity checks still require
observed coverage, an emission contained in one nonempty owner set, and no
competing owner set anywhere in the full 250 ms uncertainty window.

No neighbor-label fill, guessed timings, language extrapolation, word matching,
new global speaker IDs, or Cohere text changes are introduced. This changes
actual reading attribution, not just an evidence label. Unsupported or absent
provider timing still supplies no recoverable envelope.

## Evidence and tests

Frozen pre-change controls: six cases ran; three failed at the intended single
owner recovery, localized overlap recovery, and real retained-receipt recovery
assertions. Real transitions, gaps, and unobserved coverage already passed.
Nine final targeted tests pass, including conflicting envelopes, missing timing,
and protected human text. All 547 source tests pass (28.234 seconds).

The actual receipt case verifies SHA-256 for the final refined source authorities,
reconstructs only their referenced token IDs from their own character receipts,
then invokes production `CanonicalRuntime.decode` with retained parts instead of
any new inference. It uses the original Automatic routing metadata and current
production assembler/binding/projection. The publication reaches the actual host
SQLite and GET `/api/jobs`, passes server reading-turn validation, and renders
through the existing CPU DOM with the shipped JS. All characters are conserved,
and the two artificial uncertainty spans disappear from the Speaker 2 paragraph.

Result: 88 single-owner words (11 Speaker 1, 77 Speaker 2), one localized overlap
word, six still-unassigned transition words. Canonical core timing and ownership
values and both rollover receipts exactly match the retained baseline.

The test uses only completed evaluator artifacts, never the running full-clip
output or any original/private audio. This proves deterministic source behavior
from actual model receipts; it does not newly measure native acoustic accuracy.
The evaluator remains the sole owner of any full-clip/native-model replay.

```sh
PIERS_RETAINED_EVIDENCE=/Users/albert/Documents/Codex/2026-10-06/task-11/evidence/v062-piers/source-auto-interruption-candidate \
  /Users/albert/Documents/Codex/2026-10-04/task-2/.venv-package/bin/python -B \
  -m unittest discover -s tests -p test_reading_core_seams.py -v
```

The optional external receipt case is skipped without the explicit evidence
path; the eight synthetic source controls remain portable. No new model replay
harness, provider, downloads, dependency copies, GUI/capture, build, publication,
push, merge, or shared integration edits were performed.
