# Python removal

Desktop startup now launches the native executable's `--native-host` mode and
one native inference child. The signed package at code revision
`0201c8063c0a56bd597f721ddedce3946bcbeafe` runs the real editor/model workflow
without a Python application runtime or a developer checkout. See
[the implementation record](../implementation.md) for process, library and
actual-app evidence; the static checker alone does not establish this result.

The legacy Python application/reference implementation, its sidecar packaging and
its dedicated tests are now removed. Shared editor assets and model metadata under
`speakerdesk/` remain embedded by Rust. Python build/test tools and independent
reference generators remain outside the application runtime. See
[source-removal scope](../python-source-removal.md) for deletion rationale and
fixture provenance. Native release promotion automation remains unsupported.
