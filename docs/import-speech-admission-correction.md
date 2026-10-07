# Imported speech admission bounds

Base: released 0.6.1 source `01d36a05108cf1fffb2bf149ddc4698c11e9a345`.
The released 0.6 source `71f48c9d81307e57a6a12bf29515f72c1e21eb67`
has byte-identical `inference_worker.py`, `pipeline.py` and
`language_detection.py` Git blobs. This source defect predates 0.6.1.

Import and passage retry already evaluate the pinned Silero model on original
audio. A complete crop receipt says `speech` if any frame is positive. Previously
that decision admitted the whole diarization or unknown-speaker crop to Cohere,
even when most of its physical samples were classified negative. Manual routing
could admit an entire 18-second import crop; Automatic routing could admit a
three-second language probe around a much smaller positive region.

The worker now uses the existing live `UtteranceBook` policy: 500 ms settling,
200 ms original context, no minimum positive duration. Its speech-object bounds
partition each original crop. Every resulting slice still passes independent
speech admission; negative gaps return blank reviewed coverage. Original samples,
speaker labels, archive receipts and raw Cohere strings remain intact. No words,
word times, amplitude thresholds or phrase filters define these bounds.

Canonical bounds alone cannot exclude a tiny all-zero crop inside positive
neighboring 32 ms frames. `SpeechTranscriber` therefore also checks the actual
physical PCM before positive admission can trigger language detection or ASR.
Exactly equal samples are recorded separately as `digital_silence` (zero) or
`constant_signal` (nonzero DC), with sample bounds and the DC value. The original
positive neural receipt remains unchanged. Any waveform variation passes this
check, including one PCM quantum on a much larger DC offset; no loudness threshold
is used. Existing model-negative and pending receipts retain their prior states.

A passage retry can consequently have several speech and blank pieces. The
pipeline checks their full contiguous coverage and presents whole raw speech
strings separated by one space. `retry_parts` retains separate returned receipts
and raw strings; the current API candidate retains the presented text and review
status, as before. A failed later decode leaves earlier words available with a
partial-result review. No existing transcript is automatically replaced.

## CPU evidence

`tests/test_import_speech_bounds.py` drives actual `inference_worker.run`,
`SpeechSession`, frame ledger, language routing, `infer`, validation/export and
`retry_passage`. Only trained model/GPU boundaries are substituted. Prescribed
frame scores are independent of waveform amplitude and prove source behavior,
not acoustic accuracy.

On the released source, six regressions fail and the entirely model-negative
control passes. A 1.024-second positive span authorizes 12 seconds of manual PCM
or 3 seconds of Automatic PCM. The correction admits 1.224 seconds. A quiet
single positive frame remains admissible, genuine `Thank you.` text remains,
repeated words from separate speakers remain, original sample offsets survive,
and separated retry speech gets two independent decodes. Original WAV bytes and
owned temporary crop cleanup are checked.

Five further cases cover the independent 400-sample zero gap, all-zero positive
import/retry, exact nonzero DC and one-quantum variation on a DC offset. Zero/DC
controls fail before the corresponding guard and pass afterward; the varying
quiet control passes throughout. Two older positive
speech fixtures used constant DC sentinels; those inputs now vary at the same
quiet amplitude while all their original routing/timing assertions remain.

The independent reviewer file `test_import_silence_boundary.py` runs against this
checkout through `SILENCE_SOURCE_ROOT`, without copying or editing it. All four
methods pass through real upload, Run, pipeline, WAV crop, SQLite, API and TXT
export, including zero/DC positive controls and retained quiet/gratitude words.

## Required acoustic replay

The source defect does not establish why the reported recording contains
`Thank you.` in alleged silence. A Silero false positive inside a bounded speech
object can still authorize hallucinated words. The integrator owns that acoustic
investigation; this worker has not accessed user audio or run models.

Use the same small, disposable integrator fixtures on released and corrected
source with the existing pinned local models: nonconstant room tone alone,
quiet genuine speech, spoken gratitude, and two speech islands separated by
room tone. Exercise both imported Automatic and a selected language, then the
selected-language retry. Record every admission span, physical Cohere input
length and raw decoder result. Check that classified-negative gaps make zero
decoder calls and that all positive quiet/gratitude spans remain available.
Run sequentially under the integrator's memory coordination. Preserve raw
evidence whether recognition succeeds or fails; a failure is not a silence proof.
