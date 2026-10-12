# Port progress

The Python-free Mac implementation is locally qualified at code revision
`0201c8063c0a56bd597f721ddedce3946bcbeafe`. The signed native package passes
actual import/capture, editor, save/reopen/export, setup, voice and lifecycle
checks on Apple M5/macOS 27. Source, Linux Rust and Mac Rust CI all pass.

[Implementation and evidence](implementation.md) records the exact artifacts,
replays versus real-model tests, performance comparisons and limitations. The
owner accepted the measured extra Stop time for final refinement. Other
performance requirements remain in force; the filename ledger is not acceptance
evidence.

Legacy Python sources remain test/reference inputs, not application runtime
dependencies. Windows, Neural Engine scheduling and issue #25's separation and
episode work remain follow-ups. This work does not install over the working app,
merge the PR, notarize or publish a release.
