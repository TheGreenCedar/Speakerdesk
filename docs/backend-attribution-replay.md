# Sequential speaker attribution replay

Base: `71f48c9d81307e57a6a12bf29515f72c1e21eb67` (released 0.6.0).

## Established source behavior

`tests/test_canonical_attribution.py` sends one-second audio notifications through
the real `Engine`, canonical VAD book, temporal NVIDIA activity ledger, reading
projection, host reconciliation, and TXT export. Its acoustic providers are
explicit fabricated CPU inputs, not recognition or diarization accuracy proof.
All seven new cases fail on the released source. Separate calibrated sealed turns
already work on the release; the confirmed defect is the sealed-only live gate,
and the fallback parent label uses the entire utterance's speaker union. Removing
only the gate still loses turns when audio grows between recognition ticks.

The repair enables timing on qualified open revisions and retains the unchanged
aligned text prefix across monotone recording extension. The original timing
anchor and source audio revision remain intact; current temporal activity
requalifies each word. Shrinks, text revisions, and protected edits invalidate
the evidence. Null timing retains unknown words and original Cohere text. An
ambiguous batch speaker mapping stays unknown rather than claiming an overlap.

## Minimal actual-model proposal (not executed by this worker)

The coordinator should admit one serial source replay after combining the source
patches. Use an owned directory and process with bounded logs/time, no app launch
or native capture, and existing model/dependency paths read in place. No model or
dependency copies or downloads are needed. The user's 40 GB figure is advisory;
admission should use measured free space, actual installed model memory, and the
small WAV/log output reservation. Do not invoke an old replay harness's hard-coded
40 GB gate without reconciling it with the user's current instruction.

Start with `a-b` from `tests/acceptance/sequential-attribution-recipes.json`.
`scripts/acceptance_fixtures.generate` produces 16 kHz mono PCM using two installed
listed voice selectors, validates selectors, strips only digital-zero padding,
and records exact synthesis metadata. These are synthetic voices, not human
accuracy evaluation. Missing selectors fail closed; do not download replacements.
Synthesized durations are measured, not forced to five seconds or cropped through
words. Exact A5/B5 boundaries are supplied only in the explicit CPU regression.

Run real `live_refinement.Models(config)` and `Engine(config, models, emit, Inbox())`
from the combined source in an offline environment. Supply `diar_path`,
`cohere_path`, `speech_path`, and installed `alignment_path` explicitly, with
`canonical_utterances=True`, `language='en'`, an owned `audio_path` and fresh
`job_id`. Verify model files against their existing pinned artifacts. The optional
English timing provider must actually be installed to claim timed turn success.
If absent, preserve the unknown fallback and report the missing resource.

Feed the generated WAV via ordered one-second `Engine.handle` notifications and
retain **each** emitted canonical revision, not just the final row. Run `stop`,
then canonical `refine` with `meeting_refinement.activity_references` derived from
the emitted source. Record raw NVIDIA activity, qualified/null timing envelopes,
unaltered raw Cohere revisions, reading turns, refinement mapping, hashes,
source commit, elapsed time, and peak owned memory. Keep ground-truth recipe text
out of the model inputs. Reconcile emissions through the owned host/API and actual
shipped renderer using the coordinator's existing replay harness.

Evaluate the following independently:

- Attributed live and stopped words concatenate byte-for-byte to their source
  Cohere revision; every character is retained exactly once.
- When actual NVIDIA output distinguishes sequential voices, supported timed
  words in the two stable interiors have separate owners, with no overlap label.
- A later append preserves those unchanged text-prefix turns between decode ticks
  without an additional alignment call.
- Batch refinement preserves temporal speaker correspondence. If a model merges
  both voices into one ambiguous batch track, attribution remains unknown and
  the trace identifies that acoustic/model limitation.
- A real mixed-wave replay (`overlap`) has overlap labels only for qualified words
  whose envelopes lie within actual simultaneous NVIDIA activity.
- Gaps and null timing stay unknown; do not move or fabricate word boundaries to
  satisfy expected speaker/text pairs.

After the first two serial controls, `a-b-a` and `gap` expand the same fixture set.
If real NVIDIA activity or alignment does not qualify a case, retain all outputs
and report the failing stage. CPU passing results alone cannot qualify the real
model behavior or a release.

Run the sequential case in both explicit English and the default Automatic mode.
When Automatic produces multiple contiguous successful English passages, retain
their exact raw aggregate and all routing probes. Verify aggregate timing is
qualified across the complete original anchor, with separate reading owners in
live and sealed revisions. A mixed-language route, discontinuity, or non-English
probe must not borrow the first piece's English calibration. These source cases
are covered by the Automatic regressions in `test_canonical_attribution.py`.
