# Retained v0.6.6 oracle provenance

`tests/goldens/` contains committed outputs from the independent Python oracle.
The original function names, source hashes and generator hash in
`tests/goldens/manifest.json` describe the generator that produced those bytes;
they are historical provenance, not imports from the current application.
Rust tests consume the retained JSON directly. The legacy application and
`tools/oracle/emit.py` are removed from the current tree.

To inspect or reproduce the oracle, use a separate historical checkout containing
both the source and generator, such as commit
`7d94c8011a9501bd2357e07c73d61964c55a3886`. Run its own `tools/oracle/emit.py`
inside that checkout with its documented environment. Do not regenerate the
current goldens merely to make a native test pass.

`tests/goldens/legacy/jobs.sqlite.json` includes timestamps from `JobStore.put`,
`PeopleStore.create` and `JobStore.recover`, plus the temporary SQLite file's
SHA-256. A later run can produce a different digest. These fixtures retain the
legacy schema evidence required by native migration tests.
