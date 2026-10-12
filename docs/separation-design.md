# Saved-passage separation proposal — issue #25

Status: proposed product behavior and architecture, for review before implementation.
Source baseline: PR #27 commit `92bac44662832468cdc38b67f8137c1547536f9e`.
This design is separate from the 0.7.0 release; it does not establish separation support.

The proposed first feature is **Separate overlapping voices** on one saved passage.
It produces two local voice estimates and transcript candidates, lets the user listen
and compare, and replaces the original passage only when the user accepts both.
Keep the original recording and prior passage available. Start with a small public-audio
prototype to decide whether separation improves the words enough to justify its cost.

The deleted `dev/episode-scoped-transcripts` implementation supplies no source or
behavioral contract. This proposal uses the current Rust owners and upstream model
sources; recovery of deleted work is neither required nor planned.

## Proposed product behavior

These are proposed defaults, not decisions already made by the owner.

| Situation | Proposed behavior |
| --- | --- |
| Ready Import or saved Meeting, local source audio available | Offer separation when current activity contains exactly two distinct tracks within the episode, with a region where both speak simultaneously. Sequential speaker changes alone are ineligible. |
| Recording, paused recording, or finishing | Keep the existing transcript path. Offer separation after Ready; do not add separation to Start, live decoding or Stop. |
| No observed overlap, more than two distinct tracks within the episode, missing/stale source, or unsupported length | Explain the unavailable action beside the passage. Keep words and audio. A union of candidates alone cannot establish concurrency; three tracks are ineligible even if never all concurrent. |
| Eligible saved passage | Save pending edits before requesting work. Show progress and Cancel; the passage remains editable. Use the existing exclusive refinement admission and serial worker. |
| Completed candidate | Show the original words and recording plus Voice A and Voice B, each with its own playback and words. Label the pair “Separated voice estimates · check both”; A/B apply only to this passage. |
| Use both voices | Server accepts the current candidate in one transaction, replacing the target slot with the pair. Preserve adjacent passages and current speaker names. |
| Keep original / cancel / model failure | Keep the target words, names, bounds and recording. Discard only this attempt's unreferenced outputs. A failure has a retry action and does not turn the saved job into a failed import. |

A separation **episode** is the original sample window of the selected passage on
one capture source and one recording clock. The initial candidate covers the whole
selected passage; it does not reconstruct the whole meeting or expand into neighboring
passages. Enable only the input lengths qualified on the actual model path, within
the existing 24.5-second retry ceiling. A new product length cap needs measured
latency, memory and quality evidence; none is selected by this design. Longer
passages remain unavailable until a qualified bounded strategy exists. Arbitrary
chunk stitching is excluded.

The episode must stay within one captured language epoch. Explicit language mode is
reused. Auto performs the existing language decision separately on each estimated
voice; uncertain or unsupported results remain reviewable candidates and cannot
silently replace supported words. The first acoustic prototype uses English; it
does not qualify the other configured languages.

Separation does not identify a person or establish that both estimates are clean.
Initially both rows have unassigned speaker identity and distinct A/B badges. The
user may assign each row to an existing speaker track through the normal picker.
There is no mapping from channel zero to `speaker_0`, no continuity of A across
episodes, and no automatic People enrollment or profile update. Keep generated
voices ineligible for Remember voice and automatic recognition until a separate
quality/identity policy is qualified.

### How the pair appears and plays

Before acceptance the pair is a candidate under the existing passage, outside the
active transcript and all four transcript exports. After acceptance it is a small
group at that passage's location, with two ordinary editable passages and the
shared original time range. Example: an overlap passage at 00:12–00:18 becomes
Voice A at 00:12–00:18 and Voice B at 00:12–00:18; the next passage keeps its ID and
position. The group can display the original mixed passage from history without
adding its words a second time to the active transcript.

Voice playback uses that channel's saved generated audio. **Original recording**
plays the original selected source. General recording playback continues on the
original recording clock. Generated samples use a crop-local clock starting at zero;
the parent maps that clock back to the episode's absolute sample origin. A missing
generated asset exposes “Separated audio unavailable” and the original playback
action; it must not masquerade as successful solo playback.

Each accepted row uses the episode bounds until fresh supplied-text timing on its
own estimated audio is available. Do not copy mixed-audio CTC, reading turns or
speaker ownership onto the two rows. Keep original activity as source evidence,
distinct from any timing of the generated voice. The review label must say that
separation was attempted, rather than retaining “voices are not separated” or
claiming that uncertainty is resolved. JSON retains group/channel provenance;
TXT/SRT/VTT include both active rows in a stable A/B order with their real overlapping
time anchors and their chosen or local voice label. Subtitle readability across
players remains a limitation to inspect, not a reason to serialize concurrent
speech into fabricated consecutive times.

### Edits, refinement and restart

- A protected edit is never replaced by background work. Generating a candidate
  leaves all edits intact. **Use both voices** is an explicit replacement of the
  currently displayed passage, including its words; show that replacement in the
  comparison before acceptance.
- Capture target ID/version, original source identity/revision, sample bounds,
  protected fields, language mode/epoch and the separation attempt's generation.
  An edit, deletion, source change, language change covering the episode or newer
  attempt invalidates the result. Recheck at completion and acceptance. Unrelated
  passage edits and later language epochs need not invalidate it; preserve them.
- After acceptance, edits to one row protect that row and cannot relabel, replace
  or erase its sibling. Bounds remain inside the saved episode. Both generated
  assets retain their original episode extent; playback converts edited bounds
  using the stored origin.
- Existing automatic saved/rolling accuracy work skips accepted separation rows.
  An explicit Retry passage runs on that row's estimated audio, uses the normal
  candidate/acceptance rules and cannot change group structure or rerun separation.
  It must never quietly retry the mixed recording as though it were the voice.
- A new separation attempt requires the original mixed passage. Offer Restore
  original for an unchanged accepted pair: atomically replace both accepted rows
  with the prior original, which becomes active and exported again. If either row
  was edited, show the prior original for copying/comparison instead of discarding
  those edits.
- Candidates and accepted pairs survive restart with their playback assets.
  Interrupted attempts return a retryable stopped state without automatic reruns.
  Historical originals and rejected candidates are not active export content.

## Current owners and the required change surface

The current source supports the following boundaries. It does not yet implement
the pair, asset selection or acceptance rules described above.

| Owner | Current behavior and implication |
| --- | --- |
| [`Models`](../crates/core/src/models.rs), [`ipc`](../crates/core/src/ipc.rs), [`ChildModels`](../crates/host/src/inference.rs) | Sync model seam, bounded framed requests, parent cancellation/generations. Add one finite-window separation call here; keep model calls outside the store lock. |
| [`Runtime`](../crates/models/src/dispatch.rs), [`serve`](../crates/models/src/serve.rs) | Models load lazily in the resident child; native execution is serial, controls run separately. A separation network belongs here, with existing load/setup/error and cancellation ownership. |
| [`import`](../crates/core/src/import.rs), [`live`](../crates/core/src/live.rs), [`review`](../crates/core/src/review.rs) | Activity distinguishes concurrency from speaker unions; overlap is currently mixed and reviewed. Core should plan/validate one episode and compose per-voice admission, language, ASR and timing. The live decoder remains unchanged for the first feature. |
| [`retry`](../crates/host/src/routes/retry.rs), [`refinement`](../crates/host/src/routes/refinement.rs), [`admission`](../crates/host/src/admission.rs) | Reuse Ready-job admission, OperationGuard, worker and freshness pattern. Retry carries one text candidate; saved accuracy replaces one slot. Neither is an existing two-row replacement mechanism. The host performs HTTP/I/O/publication; episode rules stay in core. |
| [`Document`/`Segment`](../crates/core/src/document.rs), [`SqliteJobStore::update_job`](../crates/core/src/store/sqlite.rs), [`edit`](../crates/core/src/edit.rs) | Add minimal typed product fields for the candidate and accepted group. `update_job` already uses an immediate transaction. Extend server-owned field restoration; client PUT must not forge assets, episodes or model success. |
| [`native` source resolution](../crates/host/src/native.rs), [`audio`](../crates/core/src/audio.rs), [`voice`](../crates/core/src/voice.rs) | Original capture-source identity is independent of edited speaker labels. Keep it; add explicit generated-audio selection/origin for playback and retry, confined to owned media. Voice eligibility must reject the generated variant. |
| [`transcript`](../crates/core/src/transcript.rs), [`alignment`](../crates/core/src/alignment.rs), [`timing`](../crates/core/src/timing.rs), [`export`](../crates/core/src/export.rs) | Validation permits overlapping rows and unique IDs. Add stable channel tie ordering and variant-aware timing provenance/review. Current export renders active rows only; local channel labels must remain distinguishable even while identity is unassigned. |
| [`app.js`](../speakerdesk/static/app.js), [`live_editor.js`](../speakerdesk/static/live_editor.js), template/style and existing frontend tests | The editor uses whole-transcript PUT, metadata conflict recovery, retained drafts and original-recording playback. Add the comparison/group and explicit server acceptance; retain drafts on structural conflicts. Review both playback modes, keyboard controls and stale acceptance. |
| [`model setup`](../crates/host/src/setup.rs), [`artifact inventory`](../crates/host/src/model-artifacts.json), [`model setup contract`](model-setup-contract.md), native license collector | Optional pinned model setup after behavior/quality review. Ordinary transcription must still start without separation weights. No setup entry or support claim belongs in the 0.7.0 release. |

The trait addition also reaches `UnimplementedModels`, import `Scripted` and live
`FakeModels`; the application adapter is `ApplicationModels` in
[`routes.rs`](../crates/host/src/routes.rs) and `NativeModels` in `native.rs`.
Keep its unavailable/replay implementations explicit. Extend the existing core,
mounted route/process, model-reference and frontend test owners for the behavior
they actually cover; scripted waveforms cannot claim model quality. The Linux
inference fixture/stub must return the same typed unsupported behavior as other
unqualified model calls.

Add one typed candidate on the target passage: attempt ID, captured target/source
version, episode origin/length, captured language choice, two owned asset IDs,
per-channel transcript candidates and visible state/error. On acceptance add a
small group reference on each new row: episode ID, original passage ID, channel,
origin/length, asset ID and model identity. Keep the full prior original passage
in the existing history owner so restoration preserves its product attachments.
These fields serve comparison, playback, edit guards and recovery; do not add
qualification receipts or model-source hashes to saved transcripts.

The host stages the two assets under the job's owned derived media directory,
using generated IDs and no caller-supplied paths. Commit their references only
after both files exist and pass the boundary checks. Completion publishes the
candidate; acceptance validates and replaces the group in one `update_job`.
Crash/cancel cleanup removes only this attempt's unreferenced files. Original
and migrated audio remain read-only. No model or filesystem I/O belongs inside
the SQLite transaction. Reuse the existing authenticated/host-confined routes.

## Upstream facts and proposed model boundary

The official [MossFormer2 paper](https://arxiv.org/abs/2312.11825) describes a
time-domain monaural separator. The official
[16 kHz checkpoint](https://huggingface.co/alibabasglab/MossFormer2_SS_16K/tree/407cb030cd66340918ebb6c8cc63b18f8592cdbe)
is the candidate checkpoint, not the enhancement, super-resolution or target-speaker
model. Its card declares Apache-2.0 and training on public and private data; unseen
training membership of evaluation recordings cannot be established from that card.
Revision observed October 12: `407cb030cd66340918ebb6c8cc63b18f8592cdbe`.
Weights were not downloaded or inspected in this design pass.

The inspected upstream source is ClearerVoice-Studio commit
`6b3774dc79c46ae8bed2a4fa5f706f0ac8c75c61`:

- [Inference config](https://github.com/modelscope/ClearerVoice-Studio/blob/6b3774dc79c46ae8bed2a4fa5f706f0ac8c75c61/clearvoice/clearvoice/config/inference/MossFormer2_SS_16K.yaml)
  selects 16 kHz, two outputs, kernel 16, width 512 and 24 layers; its decode window
  and single-pass limit are both two seconds.
- [Model forward](https://github.com/modelscope/ClearerVoice-Studio/blob/6b3774dc79c46ae8bed2a4fa5f706f0ac8c75c61/clearvoice/clearvoice/models/mossformer2_ss/mossformer2.py)
  pads/trims decoded output to the input extent and returns two waveform estimates.
- [Decoder](https://github.com/modelscope/ClearerVoice-Studio/blob/6b3774dc79c46ae8bed2a4fa5f706f0ac8c75c61/clearvoice/clearvoice/utils/decode.py)
  copies each window's same-index channel into the output without matching channel
  permutations between windows. It also scales each channel to input RMS without
  a zero-output guard. These are upstream behaviors to examine, not native product
  requirements. A direct transcription of this wrapper would leave channel swaps,
  silence normalization and tail handling unresolved.

The two-second config is a decoding decision, not the exact tensor extent of the
wrapper's single forward. For an exactly two-second input, the inspected decoder
pads to `window + stride` (3.5 seconds). Golden inputs must distinguish raw model
forward from the padded/normalized wrapper, recording padded and original valid
lengths explicitly. Do not compare one path with the other's expected outputs.

Propose `Models::separate(pcm)` returning exactly two finite
16 kHz arrays of the original valid sample length and the activated model identity.
The parent already owns sample origin, job identity, source and operation; the child
needs no filename, database access, speaker names or transcript fields. Reject empty,
nonfinite, oversized and invalid output before publication. Never assign identity
from output order. No configurable source count or streaming session is needed for
the bounded two-voice prototype.

Use the existing Rust/MLX model lane as the first implementation candidate under
[ADR-0001](adr/0001-rust-core-inference-child.md). Establish operator/weight mapping
and numerical behavior on public golden output before choosing any alternate
runtime. Prefer one full-window forward for the small episode; validate length,
tail, silence and within-window channel consistency against an independent
reference. Full-window memory/latency is unknown. If it cannot support the chosen episode,
reduce the qualified episode length or bring the measured blocker back for a
runtime/chunking decision. Do not silently stitch arbitrary two-second channels,
add a Python application runtime, or invent a project-owned C++ model.

## Baseline and acceptance candidates

The #25 comparison is **unmodified Rust #24 versus the same Rust pipeline with
separation enabled**, on identical PCM, language, source windows and model/runtime
identities. Pin the exact baseline executable/package when the coordinator selects
it; the released 0.7.0 artifact may become that baseline once its identity is known.
Do not substitute the retired episode branch or 0.6.6 for this comparison.

The [#24 implementation record](rust-port/implementation.md) reports real app
import/edit/restart/export, failure and disposable package tests. Its retained
7.06-second public import pair measured native 9.051 s from process launch to
HTTP Ready and 3.478 GB sampled owned footprint; its 82.010-second warm native
import measured 7.974 s to Ready and 3.597 GB. The retained 54.456-second live
fixture measured 2.570 s native Stop versus 1.952–1.959 s for 0.6.6, explicitly
accepted by the owner for #24's final refinement. These are reported observations
on an M5/24 GiB/macOS 27 machine, with OS caches and sampling limits; they are not
overlap quality evidence, #25 latency estimates or permission for additional Stop
work. This design pass did not rerun them.

### Public-audio experiment

Use [LibriMix's immutable published test metadata](https://github.com/JorisCos/LibriMix)
for 16 kHz Libri2Mix clean and noisy mixtures, retaining isolated references and
the LibriSpeech words. Add one fixed far-field microphone channel from
[AMI](https://groups.inf.ed.ac.uk/ami/corpus/overview.shtml) for meeting overlap with
its manual transcripts. AMI headset recordings are useful listening references;
they are not clean acoustic targets for far-field SI-SDR. Its
[public download/annotation page](https://groups.inf.ed.ac.uk/ami/download/)
provides selective downloads and source attribution.

Proposed fixture-driven run order:

1. Choose one fixed public two-voice fixture from metadata and reference words.
   Start numerical comparison at the official two-second single-pass window; do
   not interpret that window as a usable ASR episode or a shipping product limit.
   Replay the complete product path using retained/scripted outputs before a model
   port. Add silence/solo, tail-length and swapped-channel checks to the same model
   boundary matrix. A three-voice activity fixture verifies ineligibility in core.
2. On that fixture, compare actual downstream words against the unmodified Rust
   path using exactly the same original audio anchor and language choice. If the
   initial window is too short for speech admission/ASR, report that result and
   evaluate the smallest complete spoken episode supported by the fixture. A
   longer full forward is a new measured candidate, not an automatic extension of
   the product limit. Stop expansion on invalid outputs, unresolved channel/tail
   behavior or an unusable assembled transcript.
3. Only after the fixture establishes a useful path, select held-out clean/noisy
   Libri2Mix examples and AMI meeting windows for a bounded pilot scheduled by the
   coordinator. Freeze IDs, crops, hashes, text normalization and scoring versions
   before outputs. Do not generate the entire LibriMix dataset or tune on failed
   held-out cases. Report detected eligibility/missed overlaps separately from
   an oracle-window model comparison.

Run paired baseline/candidate inputs sequentially with the same ordinary loaded
model set. Record fresh-child setup/load separately from warm work and keep each
individual measurement. Repeat only when variance can change the decision. One
coordinator owns Mac builds and inference, using disposable storage and separate
qualification assets. The fixture result decides whether a larger pilot is worth
running; it does not establish corpus-wide quality or performance.

| Claim | Minimum measurement / proposed acceptance |
| --- | --- |
| More recoverable words in two-voice overlap | Compare ORC-WER on the same reference utterances and actual output streams, retaining substitutions/deletions/insertions. [MeetEval](https://github.com/fgnt/meeteval) supports this comparison without assuming speaker identity. Report exact fixture differences first, then aggregate and per-corpus results only if the pilot supplies enough data. The prototype must show a concrete useful recovery without hiding lost or duplicated words; no arbitrary percentage target is asserted. |
| Useful acoustic estimates | On Libri2Mix only, SI-SDR improvement over the mixture, maximizing the two-output/reference permutation once per episode. Retain both channels' results; a permutation may not change within the episode to conceal a swap. Listen to candidate failures and selected examples for leakage, duplicate words and channel changes. Do not infer transcription benefit from SI-SDR alone. |
| Identity remains honest | A/B remain local and unassigned until user mapping. ORC-WER and best-permutation SI-SDR do not qualify identity. No recognition/enrollment claim. |
| Acceptable interactive cost | Report individual request-to-candidate times, real-time factor, separation versus per-voice ASR/timing cost, fresh model load and saved asset I/O, with paired baseline deltas. Use distributions only when the sample/repetitions support them. The observed cost informs the product length/scheduling decision; this design invents no latency ceiling. |
| Acceptable memory | Sample simultaneous desktop/host/child owned physical footprint with the same sampler/interval as baseline; retain child allocator peak separately. Report loaded idle, request peak, retained post-request memory, largest sampler gap and delta. Repeated requests check retained growth where needed. The measured delta informs support on the reference machine; this design invents no memory ceiling or RSS-only parity claim. |
| Existing workflow unaffected | Same selected unseparated inputs retain identical product output and original PCM hashes when the feature is unused; startup/ordinary inference remains available without separation assets. No separation call occurs during recording or Stop. |
| Safe publication | Replay matrix proves stale target, edited sibling, language/source change, cancellation, timeout, child loss, missing/escaping asset, invalid samples/length, partial pair and restart retain the previous document and original audio. Candidate completion cannot partially accept a pair. |
| Usable saved feature | One public overlap through the actual app: compare original/A/B playback → accept → edit one row → save → full restart → both playbacks and four exports → original history. Check pending drafts, stale acceptance, cancel and keyboard focus on that same surface. Replays do not substitute for this model/UI proof. |

These are acceptance **candidates** and evidence requirements, not claims that
the model already meets them. Supported lengths and acceptable added cost need
the measured fixture and product decision. A failed prototype should retain a
research result, not enable the feature or trigger an unbounded search for models.

## Smallest next prototype

The next coherent batch is one saved English overlap episode and
one candidate pair, on disposable public audio. First exercise core planning,
publication, edits, playback routing and restart with a scripted `Models` result;
then implement the bounded resident-child model and rerun that same assembled path
with actual estimates. Reuse existing test doubles and mounted integration tests.
Keep model references/scoring outside application hot paths. This sequence avoids
discovering transcript-shape or source-clock failures only after expensive inference.

Proposed defaults for that bounded prototype:

- Use on-demand Ready-only candidates, local unassigned A/B identity and explicit
  pair acceptance.
- Retry on generated audio and qualify supported length from the fixed public
  fixture rather than selecting an unsupported cap.
- Start with the smallest fixture prototype; expand the public-corpus pilot only
  when its actual words, latency, memory and failure results justify the next run.

No model/runtime code, setup entry, database migration, inference, download, build,
installation, GitHub mutation or release change is part of this design artifact.
Windows, three-voice support, cross-episode linking, automatic naming, voice profiles,
live scheduling and arbitrary long-episode stitching are outside the first prototype.
