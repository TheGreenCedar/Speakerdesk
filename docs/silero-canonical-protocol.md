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
The conversion script is reviewed provenance, never executed. No weights have
been downloaded or run at this checkpoint. The recorded provenance/resource
proposal outside the checkout is the acquisition handoff.

Every live, refinement, retry and imported Cohere call requires complete pinned
Silero evidence for its exact input range. Missing evidence stays pending;
neither manual language, recent language context nor speaker names bypass it.
Whisper's no-speech head is no longer consulted. Refinement clears its historical
evidence on success, cancellation and error, restoring the live provider. The
packaged acceptance inventory now includes the proposed Silero files and hashes;
this is an identity contract, not evidence that its weights have been acquired.

Live processing is limited by both NVIDIA's processed horizon and Silero's
completed frames. Coalesced inbox audio is fed in one-second packets. A final
partial Silero frame retains the true input endpoint. This does not yet fix the
separate controller Pause acknowledgement requirement.

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
the initial1599e815 checkpoint. Its source-only adapter remains separate and has
not been integrated or executed here. Continuous speech can exceed Cohere's24.5s
input cap: the planner supplies18s disjoint cores with3s context on each side
while one canonical reading identity survives.
Words can be assigned to nonoverlapping cores only after verified alignment;
failed alignment retains an unresolved candidate, never greedy lexical glue.
The prototype does not yet implement that partition/assembly protocol.

Speech frames now have a60s hot cap with private durable indexed raw archives;
archive failures poison the session before pruning. Canonical version history
can be configured with a hot cap only when a durable raw version/edit journal is
provided. Production must choose that cap when it wires the book. Original audio
remains retained for review. The canonical protocol must be mapped to the existing
controller/CAS/editor/export persistence; standalone object tests do not prove
that mapping. Hard Silero admission is wired in source, with CPU gate/route checks;
neural parity and acoustic quality remain unmeasured. `fallback_vad=True` is prohibited. Whisper remains language
detection only; Cohere remains the sole transcriber.

Before a new release: coordinate the small Silero-only window, measure actual
quiet/short speech and no-speech fixtures, freeze the complete integrated source,
then run the identical packaged capture/pause/stop/refinement/edit/context
lifecycle with exact model pins and PCM receipts. All old failures remain evidence.
