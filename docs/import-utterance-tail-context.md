# Import speech context at a trailing diarization gap

**Blocked as a complete correction.** Independent saved API, shipped renderer
and TXT review showed that commit `069285d` removed the tail decode while
leaving all eight previously known sentence words unassigned. Retaining a
candidate and an activity ledger alone does not preserve displayed ownership.
The provider-dependent projection below is preparation for the GPU alignment
owner, not an executed acoustic correction.

The retained signed 0.6.2 synthetic `genuine-thanks` import decoded an intended
sentence over samples `[0,44800)` and an extra `you` over `[44800,50816)`.
Actual archived dual-view Silero frames put both crops in one global VAD
utterance. The second crop began at a NVIDIA activity boundary, not a VAD
utterance boundary. Its nonconstant quiet samples remained admitted; rejecting
them by amplitude or by requiring raw-only VAD would lose demonstrated quiet
speech evidence.

The integrator's coordinated GPU probe decoded unchanged `[0,50816)` through
the pinned Cohere checkpoint and returned exactly
`Thank you, thank you, the folder is ready.`. It retained the intended spoken
thanks without the isolated extra word. That supports preserving acoustic
context. It does not establish that every tail sample is noise or provide word
ownership or timing.

`inference_worker.utterance_tail_chunks` now extends a clean single-speaker
import crop through its admitted continuation into the following unassigned
gap. Qualification requires an actual admitted speech region to cross the
boundary, the same global utterance to finish inside that gap, and the unchanged
18-second decode bound to accommodate the context. A later detected speaker,
overlap, a separate speech object, and context that is only model-negative do
not qualify. Original PCM is read in a bounded slice; no input padding, lexical
stitching, phrase filtering or change to speech thresholds is introduced.

The worker returns the adjusted crop plan with its raw outputs. `pipeline.infer`
retains the original activity ranges and returns one parent passage covering
the continuous decode. A passage crossing the unknown activity edge remains
speaker-unassigned and reviewed, with the known candidate preserved and voice
enrollment disabled. Import has no qualified word alignment, so a continuous
decode cannot establish that every word belongs to the adjacent known speaker.
The original diarization ledger remains available. Negative residual audio
keeps its original coverage and admission/constant-signal guards.

`pipeline.attach_import_reading_turns` now accepts a `reading_alignment` field
from a continuous-decode region, bound to its `audio_float32_sha256`. The
existing `reading_turns.alignment_words` verifies the supplied displayed text,
text hash, exact absolute audio anchor, pinned model hash, calibrated frame
clock and score policy. Existing `bind_words`/`project_turns` apply that timing
to original NVIDIA activity with the full calibrated uncertainty margin.
Validated reading turns then retain known labels in the actual renderer and
TXT export while words in empty activity remain locally unknown. The parent
crop remains unassigned; neither its candidate union nor context gains word
ownership. Text edits invalidate the saved projection normally.

No aligner is loaded by this adapter. The production import worker currently
does not supply a GPU-qualified `reading_alignment` receipt, so the original
independent product regression still fails. The current `CoarseAlignment`
provider uses CPU execution and must not be called for this task. Completion
requires the accelerated alignment owner to provide a verified GPU-only
provider, a calibration-compatible receipt, and its actual continuous-context
receipt. A different model/calibration must qualify through the shared reading
contract rather than being accepted by renaming receipt fields.

The worker/provider handoff must supply `reading_alignment` alongside each
eligible region's `audio_float32_sha256`. Its anchor and supplied text must
match that exact region; a raw full-context receipt cannot be borrowed for a
different language passage or clipped crop. Existing `cohere_raw_text` remains
unaltered. Partial or missing alignment keeps each unresolved word unknown;
no timestamps may be interpolated to satisfy this contract.

`tests/metadata/import_tail_receipts.json` contains actual archived frame
observations and the retained GPU continuous-decode outcome from source
`0b929a2bca7d52656cbb341a66af9361a1c2f668`. The bound input SHA-256 is
`fcdd05ca73c8de73d7ebc598e810da7ef152b1a34b0ccccbc2d163bd559d9c08`;
the actual float32 probe crop SHA-256 is
`c2df45f212c59b6f3ef22ca05302c106e1396091fd3ed06deb8a553480ae88e8`.
No original audio or model dependency is copied into source.

The new CPU regressions replay those observations through the real worker,
language router and import/export pipeline with fabricated PCM and ASR/model
boundary peers. The release baseline fails the one-decode tail contract;
the candidate passes it. Controls cover quiet speech with a zero tail, manual
and weak automatic language routing, A→B, A→B→A, a different speaker after an
unknown gap, and true overlap. Existing import/retry tests cover exact zero,
nonzero DC, a single quiet positive frame, long negative coverage and repeated
raw words. These verify source boundaries; the actual GPU probe is separate
evidence, and this patch has not received a new end-to-end model run.

The five new `test_import_reading_contract.py` controls inject explicitly
fabricated qualified timing values at the provider boundary and execute
upload/Run/saved API/shipped renderer/TXT export. They check known words, a
local unknown word, partial timing, invalid/stale/malformed receipts and a
human edit. They demonstrate the projection contract only. The reviewer's
unmodified `test_import_tail_product_path.py` supplies no alignment and remains
an intentional product blocker: three controls pass, known attribution fails.

This patch intentionally leaves continuations after overlap, speech objects
crossing a later known-speaker boundary, and contexts exceeding the existing
decode bound on their established path. It neither assigns words at those
boundaries nor changes the canonical/reading alignment framework.
