# Meeting language, retained speech and compact transcript

The released 0.4.0 UI placed the current-language selector beside capture controls
and repeated visible language-retry selectors on saved passages. Uncertain LID,
short/quiet audio and overlap could bypass Cohere; NVIDIA gaps were only filled
with blank audio placeholders. Those placeholders appeared as ordinary cards.

The corrective branch moves the single current-meeting selector into the header.
Settings remains the default for new meetings. Passage repair is contextual.
Retained audio reaches bounded ASR crops regardless of speaker assignment;
uncertain Auto uses fresh preceding context or a marked supported-language guess.
Historical refinement uses its own frozen context, with the captured epoch.

Blank machine passages do not create transcript cards or TXT/SRT/VTT cues. Their
WAV ranges, IDs and diagnostics remain in canonical JSON and a collapsed audio
review surface. Exact-zero PCM is marked separately, skips both models, and is
ineligible for enrollment. Nonzero quiet speech is still attempted. User drafts,
saved corrections and manually reviewed words remain protected.

## Browser evidence

These are actual Chromium renders of the app's HTML/JS and real local API routes,
with representative synthetic text and disposable CPU fixtures. They are not
screenshots of native capture or trained-model inference.

At 1280×800, the same one-hour, 360-passage fixture measured:

| Measurement | Released UI | Corrective UI |
| --- | ---: | ---: |
| Fully readable live passages | 2 | 8 |
| Fully readable saved passages | 2 | 7 |
| Readable live words | 40 | 160 |
| Default visible passage-language selectors | 360 | 0 |

The revised text is 16px. All 360 segment IDs remain independent despite visual
speaker grouping. Thirteen density/language/silence/import checks and twelve
rolling-editor checks passed without page errors, including stale-language polls,
real current-meeting PATCH, auto-follow, reading position, draft preservation,
async empty refinement, quiet/unknown-speaker words and reviewed ASR warnings.

Released live layout:

![Released layout: two fully readable passages](screenshots/meeting-density-before-1280x800.png)

Corrective live layout:

![Header language selector and dense transcript](screenshots/meeting-density-after-1280x800.png)

Forty digital-silence pause ranges retain internal bookkeeping without cards or
review noise; recovered quiet/uncertain words remain visible:

![Silence omitted, words retained, failures in collapsed audio review](screenshots/meeting-silence-after-1280x800.png)

An additional 1280×720 fixture contains 360 six-second speech passages and 360
four-second unassigned gaps, without claiming those gaps are proven silence.
All 720 internal rows remain: 360 speech cards, zero blank cards, one collapsed
360-range review surface and zero naming controls for blanks. Six passages are
fully readable in a 493px pane; scroll content is 25,003px. Its speech wording
is synthetic and is not asserted identical to the independent audit's baseline.

![Dense hour transcript with unassigned gaps collapsed](screenshots/meeting-density-gaps-after-1280x720.png)

## Validation limits

CPU tests exercise production routing, import workers, sample ranges, subprocess
protocols, persistence, epochs, failure barriers, exports and protected edits with
explicit acoustic/model peers. They do not measure acoustic accuracy or native
performance. All **202 CPU tests passed**; independent source review found no
remaining actionable backend issue after correcting historical context and
review-resolution regressions. This correction has not yet received new real NVIDIA/Whisper/Cohere
validation: shared free disk remains below the 40 GB floor plus CodeStory reserve,
so no native model run or package build was started. Existing installed models
and the signed 0.4.0 release are preserved. No new release is published.

Cohere supplies no word timestamps or trustworthy no-speech score. Nonzero room
noise and overlapping voices may still yield incorrect text; this change does
not invent confidence or discard returned words based on amplitude. Empty/failed
results remain available for audio review and an explicit language retry.
