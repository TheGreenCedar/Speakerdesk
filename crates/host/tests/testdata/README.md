`capture-v066.json` contains independent retained-PCM reference cases from
Speakerdesk 0.6.6, commit `91f7593fc9c890c69befd6663cdcdeb0072d5f70`.
The fixture records the source digest and NumPy version. Its generator executed
the unchanged `SourceMixer` class extracted from that tag, with
`echo_factory=None`, and quantized the resulting tracks to PCM16 independently.

The cases cover split microphone packets and an exact final tail, delayed
phase-inverted stereo, missing/late source packets, and half-even timestamp
rounding. Each contains input helper events and expected PCM bytes and hashes.
This qualifies timestamp alignment, retention, and the identity mix used after
a trusted processed-microphone receipt. It supplies no evidence about AEC,
physical capture devices, or acoustic voice-processing quality.
