# Rolling live refinement

Status: scheduling/reconciliation foundation implemented and CPU tested in an
isolated branch based on released source `846286eed2f358d3dfc6afced8ecf8b0cef2a96a`.
Worker, persistence/API and UI integration remain to be implemented. This is not
a released feature or evidence that a trained model's overlap accuracy improved.

## Diagnosis from 0.3.0 source

`live_worker.py` calls streaming NVIDIA with threshold .5, minimum duration zero
and merge gap zero. Feed activity is merged only within .20 seconds per speaker;
every simultaneous active speaker then creates a separate mixed region. Short
secondary activations can therefore fragment otherwise useful speech. Actual
user audio has not been inspected, so the amount of acoustic false overlap is
unknown; raising a threshold without measurement would not establish a fix.

AUTO `SpeechTranscriber` explicitly abstains for every mixed-speaker crop. Its
three-second language probes can produce multiple blank rows per crop. The live
worker also marks a six-second phrase for review solely for reaching its length
cap. `app.js` repeats a Review badge, language warning and explanatory paragraph
on those rows while they are still arriving. Earlier chunks are never revisited.
These rules explain excessive review presentation even when the model has not
made a final decision. The original audio is already written and flushed before
each inference packet, so a second pass can read retained local audio safely.

## Intended two passes

The resident worker keeps provisional phrase output for low latency. Provisional
rows get one compact state label; warning paragraphs and manual review badges
wait until refinement leaves a passage unresolved. A six-second transport/phrase
limit alone will no longer be a review reason. Mixed audio remains mixed audio.

A separate reader drains commands while the same model executor serializes
streaming and refinement work. Capture and the UI continue asynchronously;
multiple MLX/Metal passes do not run concurrently. Streaming backlog has priority.
Refinement runs only with at most two seconds of live inference backlog.

The planner requests an owned core of at most 18 seconds with up to three seconds
of retained context on each side: at most 24 seconds total. It runs after three
completed sections and an eight-second scheduling delay, or when a full core and
right context are available. Existing machine-owned utterance ends determine
ownership boundaries. Pause and stop flush a partial tail. No request contains
the whole meeting, and no automatic pass repeatedly reruns an old window.

Larger context can improve NVIDIA activity decisions and ASR grouping. Independent
batch diarization speaker slots must be conservatively mapped to original live
tracks using temporal evidence; ambiguous mapping keeps mixed/unknown ownership.
Confirmed names are never inferred again. Mixed audio may be transcribed with a
confident local language decision but cannot become a clean voice-enrollment clip
or a claim that the voices were separated. This behavior needs trained-model
evaluation before release; no language threshold is tuned on the held-out set.

Cohere input crops remain at most 24.5 seconds. Context audio is used for diarization
and deciding crop boundaries; ASR text may enter only the disjoint owned interval.
Without word timestamps it is unsafe to transcribe both overlapping contexts and
clip their text by time. This design avoids that duplicate-text mechanism but
cannot guarantee model word completeness or exact word-boundary cuts. Original
audio and previous window revisions remain available for correction.

## Reconciliation and lifecycle contracts

`rolling_refinement.py` implements a bounded two-request plan, durable completed
audio cursor, unique operation IDs, bounded retries and cancel/resume behavior.
After an explicit recovery, in-flight work is requeued from the same retained
audio with a new operation ID. Late results from cancelled or previous attempts
cannot acknowledge a newer operation. Backlog resides in a cursor into the WAV,
rather than an unbounded in-memory PCM or request queue.

The reconciler copies its input, rejects context text outside ownership and
overlapping duplicate candidates, preserves IDs by audio overlap, and returns
the previous window revision. A source snapshot hash covers words, timing,
speaker, machine revision and protected fields. Changed or protected rows and
intersecting candidates remain untouched. Blank/partial replacement cannot erase
previous legible words; protection propagates across shared candidates so a
neighbor cannot lose part of its source audio. Confirmed/custom names use
`setdefault`, never replacement by guessed names.

Integration will persist the plan, result and previous revision atomically under
the existing job lock/SQLite transaction. Each provisional row gets a deterministic
audio anchor plus a machine revision. A targeted passage-edit endpoint records
server-owned protected fields and uses the row revision, allowing edits without
racing the constantly advancing job revision. Confirmed identity assignments
remain protected. Unchanged rows are patched in the DOM; focused/dirty rows and
the viewport's first visible audio anchor retain focus and scroll position.

Pause flushes settled audio without restarting capture or changing permissions.
Stop closes capture, drains the provisional tail and performs bounded final
refinement; any unfinished window remains persisted/resumable rather than
holding Stop indefinitely. Cancel only cancels refinement. A process crash keeps
the last persisted transcript, prior revision and audio; recovery never resumes
device capture automatically. Failed refinement leaves usable provisional words
and one compact unresolved state instead of repeated warning rows.

## Validation so far and next step

Sixteen CPU tests cover bounded backlog, disjoint ownership, section/delay triggers,
pause/stop tail planning, crash recovery, cancellation and late results, bounded
failures, source-row boundaries, split/merge IDs, confirmed names, edits/stale
results, blank/partial preservation, protection closure and context duplication.
They execute no trained models, GPU, capture device or permission prompt.

Next: wire the resident-worker protocol, atomic persistence and targeted edits;
then test the complete host/worker protocol with substituted CPU outputs and real
headless UI updates. Actual model comparison on synthetic overlap needs a newly
coordinated resource budget. No big build or native inference is authorized by
the first source-only phase, and 0.3.0 release assets remain unchanged.
