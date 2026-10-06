# Silence and conversational turn correction after 0.5.0

This branch is a corrective candidate, not a published acoustic fix. Both user
screenshots were inspected, and released source explains three mechanisms:
nonzero audio can reach Cohere without speech admission; activity gaps partition
one sentence into separate decodes; and weak automatic-language probes impose
additional text boundaries. Existing refinement runs NVIDIA with more context,
but gives Cohere those fine-grained crops again. Empty/incomplete replacements
can preserve earlier machine hallucinations through the exact-coverage guard.
Screenshots do not measure microphone volume or prove which model made an error.

The candidate uses the same decode-envelope builder for live input/finalization,
file imports and rolling refinement. Internal gaps of at most 650ms between the
same single speaker supply continuous ASR audio. That value is an engineering
candidate awaiting actual-model controls, not an acoustically calibrated limit.
Clean speaker switches and longer pauses remain separate. Existing overlap
handling keeps all original activity ranges. Eighteen-second decode limits,
language/configuration epochs and ownership boundaries remain enforced.

The original activity ledger travels with merged rows and is clipped to the
row's recording range. Speaker mapping reads that ledger rather than assigning
unclassified gaps to the envelope's speaker. Bridged/mixed envelopes cannot
supply clean voice-enrollment evidence. Merging rows refreshes the audio anchor
to the actual candidate bounds while previous revisions retain old words and
ranges. User edits, stale revisions and incomplete-coverage guards remain intact.
No word timestamps are invented.

Automatic language detection still records every balanced <=3s probe, including
its uncertainty. Probes with compatible routed language can share continuous
Cohere input even when uncertain. Clear contradictory/unsupported language
remains a boundary. Language history replays each original confident observation
and endpoint; weak fallback neither extends expiry nor masks a failed switch.
This does not change model confidence or make uncertain words certain.

Live and saved reading views omit routine per-passage review badges, counts and
verbose explanations. Explicit Details retains confidence, language decisions,
original timing, uncovered audio and review controls. Playback/editing remain
available. JSON evidence and text export warning semantics are unchanged.

Actual-model silence/quiet controls and admission design remain pending a fresh
shared-resource window. No phrase blacklist or amplitude-only quiet-speech
cutoff is implemented. No microphone/system capture or user recording was used
for this investigation. CPU/browser tests prove routing, ownership and reading
behavior; they do not establish natural-room recognition accuracy.
