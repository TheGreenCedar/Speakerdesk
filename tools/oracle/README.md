# tools/oracle — v0.6.6 behavior oracle

Dev-only regeneration harness. The Rust port must reproduce v0.6.6 behavior;
these generators run the **v0.6.6 implementation** (from a detached git
worktree, never the working tree) and emit the JSON fixtures under
`tests/goldens/` that Rust tests compare against.

Nothing here is invoked by build, CI, or tests. Goldens are committed;
regenerating them requires this setup.

## Setup

```sh
# Detached worktree at the v0.6.6 tag (commit 91f7593)
git worktree add .cache/v066 v0.6.6

# Oracle venv — the light CI lockfile is enough for every JSON golden.
# There is no .venv* in the repo that satisfies this; create one:
python3 -m venv .cache/oracle-venv
.cache/oracle-venv/bin/pip install -r .cache/v066/requirements-ci.lock.txt
```

`requirements-ci.lock.txt` (flask, numpy, certifi, soundfile, cffi, pycparser,
filelock) covers all of the golden generators below, including the Flask
`create_app` edit cases — the v0.6.6 modules keep heavyweight MLX imports
lazy, so pure logic and API-surface behavior run without the model stack.

Lock files needed for **real-model** work (used only by the legacy DB
fixtures, which instead run the packaged app): none of these generators use
them. If a future oracle needs in-process MLX inference, install the full
`.cache/v066/requirements-*.lock.txt` stack per v0.6.6 packaging docs; the
legacy fixtures below sidestep this by driving `/Applications/Speakerdesk.app`
(the installed v0.6.6 build) directly.

## Generating goldens

All commands run from the repo root with the oracle venv's Python. Each
generator inserts `.cache/v066/speakerdesk` at the front of `sys.path` and
asserts every imported module resolved inside the worktree — the working
tree's mutated copy is never consulted.

```sh
PYTHONPATH=.cache/v066/speakerdesk .cache/oracle-venv/bin/python tools/oracle/gen_transcript.py
PYTHONPATH=.cache/v066/speakerdesk .cache/oracle-venv/bin/python tools/oracle/gen_regions.py
PYTHONPATH=.cache/v066/speakerdesk .cache/oracle-venv/bin/python tools/oracle/gen_audio.py
PYTHONPATH=.cache/v066/speakerdesk .cache/oracle-venv/bin/python tools/oracle/gen_edit.py
```

Outputs:

| Area                  | Exercises |
|-----------------------|-----------|
| `tests/goldens/transcript/` | `validate()` accept/reject/bounds; `export()` txt/srt/vtt/json incl. reading slices, overlap, unknown speaker, unknown/coarse timing, unicode, blank passages, and the 16 MiB cap (described parametrically) |
| `tests/goldens/edit/` | `PUT /transcript` + `PATCH /segments` through a real `create_app` test client: protected-field marking, server-owned field restoration, retry-candidate acceptance/retention, imported/manual provenance, stale revision 409s, exact `{"error": ...}` bodies, stored payloads |
| `tests/goldens/regions/` | `speech_crops`, `decode_regions`, `crop_context`, `utterance_tail_chunks` on synthetic turns/utterances (overlap, gaps, >18 s tails, coverage, padding edges) |
| `tests/goldens/audio/` | `pcm16_bytes` quantization, clipping, round-half-even boundaries, NaN/Inf/2-D rejection |

### Golden format

One case per file:

```json
{"input": {...}, "output": {...}}
```

`output` is `{"value": ...}` on success, `{"error": <type>, "message": ...}`
when the oracle raises, or a case-specific shape (`{"content", "mimetype"}`
for exports, `{"status", "body", "stored_jobs"}` for API calls,
`{"pcm16_hex": ...}` for audio). Oversized inputs are described
parametrically via `$repeat_segments` / `$repeat_text` / `$float` recipes
expanded by `oracle_lib.expand` — see `validate-reject-too-many-segments.json`
and `export-too-large.json`.

### Determinism

Files are written with `json.dumps(sort_keys=True, ensure_ascii=False,
indent=2)`; Python float repr is the fixed representation. The edit
generator freezes `time.time` at 1700000000.0 and uses fixed job ids, so
server-set fields (`edited_at`, `accepted_at`) are stable. Every generator
must produce **byte-identical output on a second run**:

```sh
find tests/goldens -name '*.json' | sort | xargs shasum -a 256 > /tmp/g1.sha
# re-run all four generators
find tests/goldens -name '*.json' | sort | xargs shasum -a 256 | diff /tmp/g1.sha -
```

## Legacy DB fixtures

`tests/goldens/legacy-db/*.sqlite` are real v0.6.6 databases produced by the
packaged app (not fabricated rows). Regenerate with **system** python3:

```sh
python3 tools/oracle/make_legacy_fixtures.py            # all fixtures
python3 tools/oracle/make_legacy_fixtures.py import-ready meeting-canonical
```

Prerequisites on this machine:

- `/Applications/Speakerdesk.app` installed (the shipped v0.6.6 build; the
  driver uses `Contents/MacOS/speakerdesk-runtime`).
- `~/Library/Application Support/com.speakerdesk.desktop/models` present —
  cloned per fixture with `cp -cR` into an isolated `SPEAKERDESK_HOME` under
  `.cache/legacy-fixtures/`. **Nothing else under the real app data dir is
  ever read or written.**
- macOS `say` + `afconvert` for the synthetic two-voice audio, and
  `.cache/v066` for `scripts/replay_capture.py` (the synthetic capture
  helper used for the canonical-meeting fixture).

Per-fixture scenarios, hashes, and row counts live in
`tests/goldens/legacy-db/README.md`.
