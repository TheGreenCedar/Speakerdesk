# Speech admission and canonical utterances: correction checkpoint

This branch is an isolated source checkpoint based on the withheld 0.5.1
candidate. It is not a replacement release. Hard speech admission is wired to
production Cohere call paths in this source, but canonical utterance assembly
is not connected to the controller yet. CPU tests exercise
control flow with supplied probabilities, not speech recognition accuracy.

The capture writer now rounds the inverse of PCM16/32768 and saturates signed
16-bit values. The resident audio reader also divides by32768. An actual WAV
round trip covers all65,536 integer codes, including quiet samples and negative
full scale. The frozen 0.5.1 fixtures and results remain unchanged. The new
transport expectation requires exact PCM bytes; it does not retrospectively
qualify the old candidate or fix its separate Pause acknowledgement defect.

## Speech evidence

`SpeechSession` carries the pinned neural branch's LSTM state and64 original
context samples over contiguous packets. Frames contain512 new16kHz samples.
Only final partial frames are zero padded, and their evidence endpoint remains
the original recording endpoint. A probability of0.5 starts speech;0.35 releases
it. No minimum speech duration or loudness cutoff drops a short word. Missing
coverage remains pending. A neural or probability-validation failure permanently
poisons that session; it cannot retry altered recurrent state as verified silence.

The proposed checkpoint is `mlx-community/silero-vad-v6` at
`2ebf4a5e10726a2e78ddd4d70eedfb6f1c33eb06`, weights SHA256
`65b6c5f0293cbc44d109e58bef78b474d9c65dedbee814cf0b90ef5f0d9150ff`.
All three retained files have pinned digests. The adapter uses the existing
`mlx_audio`0.5.7 implementation at
`feb25a37b07923bae556e59111995071d66afa0d`, strict16k branch loading only.
The conversion script is reviewed provenance, never executed. The approved weights have now been acquired and run against eight controlled
synthetic fixtures. All exact file hashes matched; streamed and whole-input
probabilities matched. Six decisions matched expectations, but the quiet Blue
and 250ms quiet fixtures were missed. This is a release blocker. Original
probabilities, old failures and thresholds are preserved in the external
`evidence/silero-canonical/SILERO-RELEASED-HANDOFF.json` receipt. Official binary
parity has not been measured. The initial unexecuted proposal remains historical evidence.

Every live, refinement, retry and imported Cohere call requires complete pinned
Silero evidence for its exact input range. Missing evidence stays pending;
neither manual language, recent language context nor speaker names bypass it.
Whisper's no-speech head is no longer consulted. Refinement clears its historical
evidence on success, cancellation and error, restoring the live provider. The
packaged acceptance inventory now includes the proposed Silero files and hashes;
the inventory is an identity contract, separate from acoustic acceptance.

Live processing is limited by both NVIDIA's processed horizon and Silero's
completed frames. Coalesced inbox audio is fed in one-second packets. A final
partial Silero frame retains the true input endpoint.

## Pause acknowledgement

A Pause request now creates a unique receipt before sending capture control.
A duplicate request before capture acknowledgement preserves that ID and sends
no second helper command. Capture drains and receipt binding precede publication
of Paused status. Legacy one-pass workers mark only `capture_complete`; they
cannot claim a worker processing acknowledgement.
After the mixer drains through the exact captured sample, an ordered worker flush
binds that ID and endpoint. The worker publishes available revisions first, then
acknowledges received samples, available processing horizon, Silero horizon and
fast sequence. The controller rejects mismatched endpoints/horizons/sequences and
ignores stale IDs. Any lookahead tail is explicit `deferred_audio`; Pause neither
feeds fake future audio nor finalizes the recurrent VAD state, so Resume remains
valid. Stop retains its separate full finalization path.

Packaged traces retain the original POST request ID independently of the receipt,
so both the runtime wait and later evidence revalidation bind that exact request.
Packaged acceptance waits for the matching receipt, rather than demanding that
NVIDIA's available horizon already equal the capture endpoint during Pause. It
still requires exact saved PCM, actual speech/text expectations and complete Stop
refinement. Old candidate assertions/results remain preserved. CPU tests cover
lagged available horizons through real pipes, SQLite and repeated Pause/Resume;
this source fix has not yet passed signed packaged acoustic acceptance.

## Canonical protocol

`UtteranceBook` identifies a recording/language-epoch/audio-origin object before
ASR. The retained start/end are original recording sample positions, including
bounded context; `speech_regions` separately record admitted speech. These
positions are audio ranges, not word timestamps. Growing live and full refined
versions replace text on the same identity. Protected human words remain primary.
Empty or incomplete machine text remains a candidate and cannot erase prior words.
Stop closes admission; late audio/language messages cannot reopen it.

A text revision and a separate audio revision bind asynchronous results. Changing
the audio endpoint invalidates alignment even if the text is unchanged. The
last accepted text retains its own `text_audio_anchor`. An alignment must match
both revisions and the exact text hash and provide nonempty ordered word ranges
within the original retained audio. Word/text coverage and alignment accuracy
remain the aligner's verified contract; bounds validation alone is not proof.

NVIDIA activity is an independent original-audio ledger. It cannot split Cohere
words at a speaker boundary. Two sequential speakers, overlap, unassigned speech,
missing coverage, open utterances and stale activity cannot authorize clean
voice enrollment. Only current complete activity with one owner over all admitted
speech on a sealed utterance can set `voice_eligible` in this prototype. Production
voice duration, quality and consent requirements still apply separately.

## Remaining integration gates

The alignment owner has confirmed the shared sample/revision/text contract against
the initial1599e815 checkpoint. Its exact source adapter is imported from fe8dc0f696347b77fbfe68d90846439cf223451f.
The alignment owner subsequently ran the approved native runtime at579ea781,
but its frame-clock and score policy remain unverified for production attachment.
A permissive diagnostic forced supplied words onto silence: alignment is never
speech admission or transcript validation. Continuous speech can exceed Cohere's24.5s
input cap: the planner supplies18s disjoint cores with3s context on each side
while one canonical reading identity survives.
Words can be assigned to nonoverlapping cores only after verified alignment;
failed alignment retains an unresolved candidate, never greedy lexical glue.
`canonical_assembly.py` now implements a pure-data bounded partition contract:
all current requests must match whole-utterance identity, both revisions, epoch
and exact core/context bounds. Every context must have complete raw-text coverage
and consistent alignment provenance. Only emission envelopes wholly inside a
disjoint core own words; crossing envelopes or empty ownership keep the candidate
unresolved. Internal raw Unicode/spacing is retained, with one explicitly
documented separator between context substrings. Numbers and real repetitions
are never normalized or deduplicated. This conservative policy may defer real
boundary words; it does not establish phonetic edges or acoustic accuracy.

`UtteranceBook.apply_decode_parts` journals every raw context and result before
pruning hot history or replacing accepted text. Archive failures leave the book
unchanged. Human corrections remain primary. Complete assemblies attach only
to the new machine revision/current audio revision/raw hash. The direct book
attachment also now requires complete raw display-unit coverage and exact text
audio anchor. CPU fixtures contain explicitly invented envelopes only.

Speech frames now have a60s hot cap with private durable indexed raw archives;
archive failures poison the session before pruning. Canonical version history
can be configured with a hot cap only when a durable raw version/edit journal is
provided. Production must choose that cap when it wires the book. Original audio
remains retained for review. The canonical protocol must be mapped to the existing
controller and production VAD-driven scheduler. Existing real Flask targeted
and whole-document edit routes now preserve server-owned canonical evidence,
reject client-forged alignment/revision metadata and discard stale attachments
after edits. Edited display bounds discard the text anchor while original audio
evidence remains retained. These API checks do not prove Engine emission,
controller projection, long-utterance scheduling or signed package integration. Hard Silero admission is wired in source, with CPU gate/route checks;
official neural parity remains unmeasured and short/quiet acoustic quality failed. `fallback_vad=True` is prohibited. Whisper remains language
detection only; Cohere remains the sole transcriber.

Before a new release: resolve the measured short/quiet admission failures,
obtain immutable actual Cohere snapshots and independently reviewed alignment
calibration, complete production canonical scheduling/projection, freeze that source,
then run the identical packaged capture/pause/stop/refinement/edit/context
lifecycle with exact model pins and PCM receipts. All old failures remain evidence.
