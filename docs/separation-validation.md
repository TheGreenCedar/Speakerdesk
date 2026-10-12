# Native two-voice separation validation

Local checks on 2026-10-12 used an Apple M5, 24 GiB RAM, macOS 27.0.1
(26A434), Rust 1.99.0 and optimized builds with MLX 0.32.2. The application
uses the Metal device; Python/PyTorch and MeetEval were separate development
oracles, never application processes. All recordings and stores were disposable.
The installed application and its data were not modified.

The comparison application was the published 0.7.0 Apple Silicon app, executable
SHA-256 `9d4955e210d1ab2c55f56ccda28c9e9fd9dbc8ff256943e4226c40d6438f2dec`.
The signed local candidate after the terminal-status correction is
`c87385ec8b3734d047ad459653303a42df2821764e98d04d7d0b240747fbfc2e`.
This records local qualification, not a release or notarization claim.

## Product and failure boundaries

The import/edit/export UI check used signed executable
`dca0beef25325474ac431632ddf708bf153c4207329a0b10ff26e5a8179f6b76`,
before the decoder and terminal-status corrections. Its editor and automatic
publication path are unchanged in the final candidate; real-child output and
cancellation checks cover those corrections separately.
The actual Tauri app imported the first public fixture through its file picker.
Transcribe showed progress, then automatically replaced two detected overlap
passages with four ordinary editable rows. No comparison/acceptance step was
required. The UI retained a compact check hint and original-recording history.

Words, a speaker assignment/name and an end boundary were edited in the UI.
After quitting the entire app, the desktop, host and inference child exited.
Reopening retained the exact document and revision, words, name, boundaries and
generated assets. Native TXT export matched the HTTP export; TXT, SRT, VTT and
JSON included the edits. A stale PUT returned 409. Original source bytes matched
the imported fixture. Generated playback controls were exercised; this does not
substitute for a listening study.

The final signed candidate also captured Mac audio with the microphone disabled,
using two playbacks of the fixed public mixture. The 102.377-second recording
included idle silence; its two detected overlap passages became four editable
rows automatically after capture stopped. No separation ran while recording.
The observed saved-accuracy completion preceded separation completion by 8.345 s;
the existing words stayed available during that interval. This is a workflow
check, not a paired Stop-latency benchmark. Peak sampled owned physical footprint
was 4.332 GB. A whole-app quit and reopen retained the exact document and revision
37, including both generated pairs. The capture helper, inference child, host and
desktop were all reaped after their respective shutdowns.

The real resident child completed repeated requests, cancelled an active maximum
window and successfully processed the following request. Cancellation preserved
the document and left no generated assets for the cancelled operation. Sampled
idle owned memory after the three successful requests was 3.120, 3.120 and
3.126 GB. The same cancellation exposed a stale “Separating…” status; the
correction covers cancellation, failed/rejected output and interrupted reopen,
while stale work cannot change a newer operation's message.

File-backed router tests cover automatic publication, protected/concurrent edits,
missing models, setup/capture admission, cancellation, stale generations, failed
or empty output, names, exports, restart, restoration and no repeated automatic
attempt of the same source. These use labelled model replays. Real inference and
desktop checks above cover the assembled path separately.

## Independent model references

The source checkpoint is pinned to
`alibabasglab/MossFormer2_SS_16K@407cb030cd66340918ebb6c8cc63b18f8592cdbe`.
The independently executed reference is unmodified ClearerVoice source at
`6b3774dc79c46ae8bed2a4fa5f706f0ac8c75c61`, using PyTorch 2.8.0 on CPU.
The reference generator and provenance instructions are in
[`crates/models/tests/data`](../crates/models/tests/data/README.md).

Native full-window outputs were checked against independent raw forward outputs,
including a solo input and a 392,000-sample stress input. After the decoder memory
correction, the stress case's two normalized RMS errors were 1.706e-6 and
2.572e-6; maximum absolute errors were 0.000165 and 0.000186. The fixed decoder
also has an independent scalar test covering frame coordinates, both batches,
signed weights, adjacent overlap and every 0–7 sample tail. Invalid input and
pre-cancelled requests remain rejected.

## Public clean-audio pilot

Libri2Mix test-clean metadata was pinned at
`b898ca375c03e4039cdb3dbe3dd485aad58ec7d1`; LibriSpeech source assets at
`openslr/librispeech_asr@71cacbfb7e2354c4226d01e70d77d5fca3d04ba1`.
The recipe used complete utterances, official gains, maximum-length zero padding,
float32 mixing and ties-to-even PCM16 conversion. The two additional cases were
selected and their words/hashes frozen before inference.

| Complete fixture | Duration | Mixed 0.7.0 ORC errors | Separated oracle-window ORC errors | SI-SDR improvement, two channels |
| --- | ---: | ---: | ---: | ---: |
| `4077-13754-0001_5142-33396-0065` | 5.195 s | 12/24 (50.00%) | 2/24 (8.33%) | 16.68 / 21.81 dB |
| `6930-76324-0027_5683-32879-0011` | 9.210 s | 24/46 (52.17%) | 0/46 (0.00%) | 15.98 / 17.34 dB |
| `1188-133604-0025_4992-23283-0016` | 7.450 s | 20/33 (60.61%) | 1/33 (3.03%) | 19.98 / 18.32 dB |

MeetEval 0.4.3 scoring used NFKC, casefold, apostrophe removal and ASCII
alphanumeric tokens equally on reference and hypothesis. No aliases or filler
deletion were applied. SI-SDR used one output/reference permutation for each
whole episode, including its tails. These are individual clean fixture results.
Oracle windows deliberately supply the full known two-voice region; their scores
do not establish automatic overlap detection or speaker identity.

The automatic import path also detected and published one pair on each additional
fixture. Its segmented transcript retains local channel identities per episode,
so it was not scored as though each row were a complete independent speaker stream.
Single sequential baseline/candidate trials observed first editable text at
0.917/0.876 s and 0.882/0.872 s. Candidate automatic separation finished at
6.403 s and 3.545 s from transcription start; the original transcript remained
editable during that work. Sampled owned peaks were 3.275/4.384 GB and
3.330/3.971 GB. These trials show added completion cost; they do not establish a
statistically significant first-text speedup.

## Decoder allocation correction

MLX 0.32.2 implements this stride-8 transposed convolution using an input-dilated
convolution with a large explicit unfold buffer. The equivalent Rust graph
projects each encoded frame into 16 samples and overlaps its two eight-sample
halves. It preserves the full window and avoids arbitrary chunk stitching.

| Real resident child, same input/settings | Before: time / peak owned memory | After: time / peak owned memory |
| --- | ---: | ---: |
| 5.195 s whole mixture | 4.951 s / 7.277 GB | 3.840 s / 3.355 GB |
| 24.500 s repeated-mixture stress | 19.341 s / 18.384 GB | 16.555 s / 6.315 GB |

Times include separation and per-voice transcription; output text was unchanged.
Memory is sampled simultaneous host/child physical footprint, not RSS. These are
single paired optimized runs, not a latency distribution. The maximum-length
input tests cost, length and tails; repeated speech is not a quality corpus.

The backend pin serves reproducibility and cache reuse. It is not a requirement
to stay on 0.32.2. Source inspection found byte-identical Metal convolution in
released MLX 0.32.3; current upstream development also retains the relevant
input-dilation fallback. That upgrade would require adjusting the vendored C
wrapper for the changed `gather_qmm` signature without fixing this allocation.
A future backend with demonstrated product benefits remains an upgrade candidate.

## Limits

No noisy or far-field quality claim is made. An AMI crop and manual references
were frozen, but selective audio retrieval timed out; it was not scored. Other
languages, automatic person assignment from separated embeddings, saved voice
compatibility for those embeddings and three-or-more overlapping voices are not
qualified here. Generated voices are not enrolled or matched using the existing
unseparated-voice calibration. Three-or-more overlap is tracked in issue #36.
Windows execution and a published package remain separate work.
