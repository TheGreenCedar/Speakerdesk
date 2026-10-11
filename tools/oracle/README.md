# v0.6.6 oracle harness

This directory is a development-only Python harness for the Rust port. It is not on the build path, the CI path, or the packaging path. Nothing in `tools/oracle/` is imported by the application, the test suite, or the packager.

`tests/goldens/` holds committed goldens. `tools/oracle/emit.py` produces them by calling the in-repo Python modules. The harness imports no Rust.

The `speakerdesk` package in this repository matches tag `v0.6.6`. `emit.py` inserts `<repo>/speakerdesk` at the front of `sys.path` and imports those modules.

## Reproduce

From a clone that contains tag `v0.6.6` and this harness:

```bash
git worktree add /tmp/speakerdesk-v0.6.6 v0.6.6
python3 -m venv .venv-oracle
. .venv-oracle/bin/activate
python3 -m pip install 'flask==3.1.3'
PYTHONPATH=/tmp/speakerdesk-v0.6.6/speakerdesk python3 tools/oracle/emit.py
```

`people.py` imports Flask, so the virtualenv needs that package before `emit.py` can load `PeopleStore` and `introduction_suggestions`. The `PYTHONPATH` above is the `speakerdesk` package directory inside the `v0.6.6` worktree. `emit.py` still inserts this repository's `speakerdesk` directory first; that package is the same code as the tag.

Running `emit.py` rewrites the JSON files under `tests/goldens/`. `tests/goldens/legacy/jobs.sqlite.json` includes timestamps from `JobStore.put`, `PeopleStore.create`, and `JobStore.recover`, plus the SHA-256 of the temporary SQLite file those calls created. A later run produces a new digest.
