# Speech admission and canonical utterances: correction checkpoint

This branch is an isolated source checkpoint based on the withheld 0.5.1
candidate. It is not a replacement release. The new speech and utterance
modules are not connected to production inference yet. CPU tests exercise
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

The alignment owner must verify the shared sample/revision/text contract before
integration. Continuous speech can exceed Cohere's24.5s input cap: decode requests
must use bounded context windows while one canonical reading identity survives.
Words can be assigned to nonoverlapping cores only after verified alignment;
failed alignment retains an unresolved candidate, never greedy lexical glue.
The prototype does not yet implement that partition/assembly protocol.

The complete frame and machine-version ledgers currently have no retention cap.
Production wiring requires bounded hot history with archived raw evidence and
audio retained for review. The canonical protocol must be mapped to the existing
controller/CAS/editor/export persistence; standalone object tests do not prove
that mapping. Hard Silero admission must cover every live, refinement, retry and
imported Cohere call. `fallback_vad=True` is prohibited. Whisper remains language
detection only; Cohere remains the sole transcriber.

Before a new release: coordinate the small Silero-only window, measure actual
quiet/short speech and no-speech fixtures, freeze the complete integrated source,
then run the identical packaged capture/pause/stop/refinement/edit/context
lifecycle with exact model pins and PCM receipts. All old failures remain evidence.
