# Live continuity and separated-voice attribution validation

Local checks on 2026-10-13 support cooperative auxiliary inference and conservative
automatic attribution of separated voices. They used an Apple M5, 24 GiB RAM,
macOS 27.0.1, Rust 1.99.0 and optimized native MLX/Core ML builds. Public audio,
disposable stores and copied candidate executables were used; installed app data
was untouched. This records candidate qualification, not a 0.7.2 release,
notarization or physical-device capture check.

The source base was `ea197a2350aa39421e2eb77de1599485e7a4fe79` on
`codex/live-continuity`. Different gates used successive candidate snapshots;
the retained source hashes and executable hashes distinguish them. The final
import and assembled identity replay executable SHA-256 is
`f17816340baf8eddecdb5016bd485b123cd66bcd029c7989bd5efaff3b20c787`.
Earlier separation quality and decoder measurements remain in
[the original validation record](separation-validation.md).

## Cooperative continuation

The same serial inference child retains full-window Moss state and full-crop
Cohere state. It yields between existing Moss block evaluations and auxiliary
ASR decoder tokens so primary transcription and capture control can proceed.
No waveform splitting, channel stitching or second GPU owner was introduced.
The full Cohere encoder, individual Moss evaluations, VAD/probe calls and primary
ASR requests remain indivisible work.

The direct native Runtime gate interleaved primary VAD/ASR with these sessions.
All six channel PCM16 outputs from the three retained clean mixtures were byte
exact against the prior native outputs. The independent Cohere golden remained
exact at 31 tokens, including primary ASR run during pending auxiliary work.
The longest measured auxiliary call was **224.723 ms**, the first evaluated
Moss step on a **13.055 s** input; full auxiliary encoder time was 223.658 ms.
Sampled peak physical footprint of that test process was **3.387 GB**.
These are tested phase times, not a universal latency bound or whole-app peak;
model load and primary requests are measured separately.

The mounted CPU fixture used the same admitted 7–11 s speech interval before and
after the change. The baseline stopped canonical words at 4.2 s despite 13.5 s
capture while an auxiliary request was held. Both held-Moss and held-ASR candidate
cases published later words and canonical EOF before auxiliary completion, then
saved the pair. An earlier baseline with inadmissible later speech was withdrawn.
Process/router/store matrices cover cancellation, child loss/reaping, stale
handles/generations, protected edits, terminal check retirement, export and real
SQLite reopen. Their model outputs are labelled fixtures, not acoustic proof.

## Separated identity qualification

The generated-query policy is separately declared for the pinned Moss checkpoint
and ReDimNet Core ML CPU_AND_GPU identity. It retains score threshold
**0.7264125328398843** and different-identity margin **0.10003429964182987**;
neither was lowered or tuned on the fresh holdout. Two independent reference
utterances per person were distinct from query utterances.

Queries use admitted acoustic speech ranges: the longest contiguous span of at
least two seconds, capped at six seconds. Generated embedding input alone receives
downward headroom to `32766/32768`, avoiding the single full-scale normalization
sample. Saved audio, transcription input and ordinary clipping checks are
unchanged. Fixed equal halves had exposed clipping and padded-tail failures; the
speakers used to develop that correction are counted as development data.

| Public clean-speech set | Observed outcome |
| --- | --- |
| Six development speakers, three mixtures | Six correct accepted queries; true scores 0.741633–0.897665. Thirty wrong-person comparisons peaked at 0.257943; six noise/profile comparisons at 0.082232. Silence and clipped controls were rejected. |
| Four fresh speakers, two mixtures; ten candidate identities | Three correct accepted queries (0.780514, 0.796644, 0.885394), zero wrong accepted matches; the fourth remained unassigned at 0.699563. |

Policy metadata was frozen before the fresh holdout ran. This is 75% accepted
coverage on four holdout queries, not population accuracy or a broad false-accept
estimate. Matching is independent per channel, with collision/ambiguity
abstention; an unknown sibling does not suppress a strong match. Existing track
names and manual assignments survive; newly created automatic assignments remain
unconfirmed. Generated audio never silently enrolls a person. Synthetic hostile
matrices separately verify edits, source changes, forget/replace/off–on races and
publication abandonment.

## Assembled live replay

The real host/child processed a **45.995 s** public PCM sequence, with replay
replacing capture hardware. Profiles were seeded from real independent reference
embeddings; this is not UI enrollment proof. Two known names appeared at 9.071 s
and 21.890 s. Separation was pending around 33.56–41.34 s; later English words
appeared at **40.366 s** during that work. At **41.499 s**, both generated rows
were associated with existing named tracks, before Stop at **46.076 s**. The job
reached Ready with no failed refinement and no new enrollment.

| Single sequential replay pair | Baseline 0.7.1 | Candidate |
| --- | ---: | ---: |
| Named-track/overlap sequence: Stop to Ready | 7.841 s | 1.134 s |
| Named-track/overlap sequence: peak owned physical footprint | 4.768 GB | 4.719 GB |
| French/English switch: Stop to Ready | 1.957 s | 1.956 s |
| French/English switch: peak owned physical footprint | 3.514 GB | 3.520 GB |

The switch replay's last pre-Stop snapshot had 1.36 s pending audio in both runs.
It contained no separated pair and used the earlier candidate, before the final
import-boundary correction. These individual trials do not establish a first-word
speedup, sustained throughput or microphone/Mac-audio device behavior.

A repeat control using the unchanged first ten seconds of the named-track public
fixture produced identical first words at **7.156232 s** in 0.7.1 and
**7.160783 s** in the final candidate. The earlier single full-run difference
(7.140 versus 7.799 s) did not reproduce in this control; neither an early-text
speedup nor a repeatable regression is established.

The final candidate reopened a copy of the completed live store through its real
HTTP API. Both separated rows and automatic names survived the first whole-host
restart. Passage PATCH changed words and a boundary; whole-transcript PUT changed
a name. Stale PATCH and PUT requests returned 409. A second whole-host restart
preserved the document, assignments, revision and byte-identical TXT, SRT, VTT and
JSON exports. Original and copied audio hashes stayed unchanged and owned child
processes were reaped. This is service/editor-contract proof, not a new desktop
UI interaction.

The joined local regression gate passed 367 native/core/host/tool tests and 27
desktop tests, with 26 model/device-dependent tests explicitly ignored. The
separate native gates above cover the changed acoustic paths. JavaScript passed
138 tests with one skip; build/release tooling passed 56 tests with one skip.

## French/English import comparison

The manifest froze the first two validation rows per language from
`google/fleurs@70bb2e84b976b7e960aa89f1c648e09c59f894dd` before inference.
Derived cases concatenate French/English utterances with 0.8 s silences or add
seeded white noise at 10 dB SNR. They are constructed stress cases, not natural
code switching or far-field recordings. Scoring uses NFKC/casefold, apostrophe
removal and punctuation-to-space while preserving accented letters and numbers;
no word aliases. This differs from the older clean-pilot ASCII tokenizer.

| Whole import | 0.7.1 word errors | Final candidate word errors | Candidate first editable text |
| --- | ---: | ---: | ---: |
| French row 0 | 2/38 | 0/38 | 1.360 s |
| French row 1 | 0/23 | 0/23 | 0.711 s |
| English row 0 | 0/15 | 0/15 | 0.595 s |
| English row 1 | 8/33 | 8/33 | 1.473 s |
| Constructed French/English switch | 16/109 | 8/109 | 3.704 s |
| French row 1 + seeded noise | 9/23 | 9/23 | 0.705 s |

All six imports reached Ready. These single trials cover the final adjacent
language-boundary correction. Persistent English substitutions and noisy-speech
errors remain; the table is not a multilingual meeting accuracy claim.

## Core ML device-policy comparison

The same 33 public crops and waveform controls were run sequentially under each
configured policy. Warm medians exclude the first call and rejected outputs
(27 calls per policy).

| Configured policy | Warm clean median | Sampled test-process peak physical footprint |
| --- | ---: | ---: |
| CPU_AND_GPU | 30.281 ms | 426.94 MB |
| CPU_AND_NEURAL_ENGINE | 117.076 ms | 101.83 MB |
| ALL | 121.521 ms | 142.43 MB |

All policies returned 28 valid vectors and the same five rejections. Minimum
paired cosine was 0.999953 for GPU versus CPU+ANE. CPU+ANE was about 3.87 times
slower here; production keeps CPU_AND_GPU. Footprints are sampled standalone
loads, not equal whole-app/concurrent-load measurements. No actual ANE placement
trace, power benefit or concurrent-device improvement was established.

## Retained evidence and remaining limits

Receipts are retained under
`/Users/albert/.codex/work-records/speakerdesk-post07/`:

- `live-continuity-source-receipt.json`, `live-continuity-before-valid.log`,
  `live-continuity-eof.log`, `live-continuity-process-all.log` and
  `live-continuity-saved-identity.log`: source hashes and CPU lifecycle proof.
- `live-continuation-native-cases.json`, `live-continuation-native-result.json`
  and `live-continuation-native-run-2/`: native parity, phases and supervision.
- `identity-pilot/{prepared-query-scores.json,policy-frozen.json,holdout/scores.json}`
  and `identity-pilot/{gpu,ane,all}-run/`: identity and device-policy receipts.
- `quality-pilot/{manifest.json,baseline-071,candidate-072,live-fr-en-baseline,live-fr-en-candidate,live-identity-baseline,live-identity-072-2}`:
  fixed public inputs, executable hashes, transcripts and timed replay traces.
- `quality-pilot/live-identity-durability/` and `joined-072-{rust,desktop,node,tooling}.log`:
  real model-output editing/restart/export proof and joined regression results.
- `quality-pilot/firsttext-{baseline,candidate}-repeat/result.json`: repeated
  first-word control with identical output text.

No general real-meeting identity/accuracy, noisy separated attribution, far-field,
three-or-more-source, Windows or physical capture qualification follows from
these checks. Final package/UI/release acceptance remains separate from the
source, native Runtime and replay evidence recorded here.
