# Python removal

Desktop startup now launches the native executable's `--native-host` mode and
one native inference child. The signed package at code revision
`0201c8063c0a56bd597f721ddedce3946bcbeafe` runs the real editor/model workflow
without a Python application runtime or a developer checkout. See
[the implementation record](../implementation.md) for process, library and
actual-app evidence; the static checker alone does not establish this result.

Reference source deletion is separate work. CI, numerical-reference generators
and source checks still use the legacy Python implementation. The `speakerdesk`
directory also contains shared editor assets and model metadata embedded by the
Rust build. Remove obsolete source only with its callers; do not remove shared
assets or confuse build/test Python with the packaged application's runtime.
