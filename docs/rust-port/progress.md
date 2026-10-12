# Port progress

The Python-free Mac replacement needs changes before shipment. Fresh review at
`7d94c8011a9501bd2357e07c73d61964c55a3886` confirmed that native startup never
invokes legacy migration, and background voice-check metadata can strand a dirty
import edit on a stale revision. The Python cleanup does not change either path.

The signed package at code revision `0201c8063c0a56bd597f721ddedce3946bcbeafe`
passed the recorded import/capture, editor, save/reopen/export, setup, voice and
lifecycle checks on Apple M5/macOS 27, plus all three CI jobs. Those paths did not
cover the two newly reproduced failures. The later docs-only head's Mac CI failed
on an echo-source HTTP 503; its source and Linux jobs passed.

[Implementation and evidence](implementation.md) records the exact artifacts,
replays versus real-model tests, performance comparisons and limitations. The
owner accepted the measured extra Stop time for final refinement. Other
performance requirements remain in force; the filename ledger is not acceptance
evidence.

Legacy Python application sources and their dedicated tests/packaging paths have
been removed. Shared editor/model assets and independent numerical fixtures remain;
[source-removal scope](python-source-removal.md) records the retained test owners. Windows, Neural Engine scheduling and issue #25's separation and
episode work remain follow-ups. This work does not install over the working app,
merge the PR, notarize or publish a release.
