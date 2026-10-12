# Port progress

The confirmed startup-migration and metadata/autosave defects from the fresh review
are fixed and independently reviewed. The actual signed host imports all three
synthetic legacy layouts, preserves legacy bytes, and retains edits after restart.
The Tauri editor recovers from an injected metadata-only conflict and shows the
saved correction after whole-app restart. The final signed package also passes
the setup-card check: one clear download action, collapsed technical errors, and
an unclipped heading and Start controls at the original window size. Missing,
loading and failed setup states have separate behavior and regression coverage.

Borrowed model lifetimes replace the unsafe adapter; unused foundation helpers
are removed; ID generation and timestamp arithmetic reuse existing Rust code.
Duplicated license dumps are replaced by one build-generated packaged notice,
which also passes the native Settings export. The pinned mlx-sys build patch stays
because it provides prebuilt MLX linkage absent from this version's upstream crate.

[Implementation and evidence](implementation.md) records exact artifacts, the
UI harness relaunch incident, replays versus real-model tests, performance
comparisons and limitations. The owner accepted the measured extra Stop time for
final refinement. Other performance requirements remain in force.

Legacy Python application sources and their dedicated tests/packaging paths are
removed. Shared editor/model assets and independent numerical fixtures remain;
[source-removal scope](python-source-removal.md) records the retained test owners.
Python remains build/test tooling. Native release promotion automation is still
unsupported. Windows, Neural Engine scheduling and issue #25's separation/episode
work remain follow-ups. No app installation, merge, notarization or publication
is part of this work.
