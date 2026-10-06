# Pause-aware original-audio cores

Long unaligned utterances use disjoint original PCM crops of at most 18 seconds, with the existing optional 200 ms boundary padding on each side. The physical crop plus padding stays below Cohere's 24.5-second input bound. The calibrated manual-English context/ownership path is unchanged.

Before decoding, a revision-bound plan prefers nearby internal speech-region gaps whose complete admission receipt independently confirms current-policy Silero model-negative classification. This does not prove acoustic silence or word boundaries. Uncertain, historical, pending, or unavailable evidence falls back to balanced bounded crops. At most three gap receipts are queried per boundary. Every sample is covered exactly once, and all crops in a multi-crop plan are at least six seconds; no tiny tail is created.

The exact plan is reused for result validation, retained in the raw machine-version journal and bounded decode provenance, and invalidated when text/audio revisions or physical endpoints change. It is never supplied by a client. Raw Cohere crop texts are joined with the existing one-space separator, without deduplicating, substituting or dropping words. Human corrections and existing compare-and-swap guards retain priority.

This changes model input boundaries, so actual Cohere words can change. CPU tests establish physical coverage and provenance, not recognition accuracy. The same public interview must be rerun with pinned local models before making an improvement claim. In particular, the previous cuts at 62.903, 80.748 and 312.734 seconds are diagnostic targets. Compare raw outputs and exact new crop boundaries, retain all old evidence, and do not use captions as auditory truth.

Resource qualification: the first 24-second-cap full-interview attempt exceeded the existing 4 GiB RSS guard (4,376,297,472 bytes). Its evidence is retained, and its owned process/group termination independently verified. The cap therefore remains at the already-qualified 18-second duration while pause placement changes; the guard is not raised.
