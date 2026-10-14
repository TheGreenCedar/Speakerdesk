# Maintaining Speakerdesk

Speakerdesk records or imports audio, transcribes it locally, and lets the user
correct, name and export passages. Start with [product behavior](product-behavior.md)
for the rule you are changing and [architecture](architecture.md) for process and
storage boundaries. The [glossary](../GLOSSARY.md) distinguishes a speaker track,
a saved person and a voice profile; those distinctions affect consent and edits.

## Find the operation

Paths below are relative to the repository root. HTTP routes are composed in
`crates/host/src/routes.rs`; that router also applies host, origin and mutation-token
checks to the merged feature routes. This is a local application with one local
library, not a multi-tenant service.

| Change | Start here | Follow only when needed |
| --- | --- | --- |
| Name a speaker or confirm a suggestion | [People routes](../crates/host/src/people_routes.rs) | [Name/suggestion rules](../crates/core/src/people.rs); [explicit voice enrollment](../crates/host/src/people_routes/voice_http.rs) |
| Import, cancel or resume audio | [Import lifecycle](../crates/host/src/routes/imports.rs), [progress publication](../crates/host/src/routes/import_progress.rs) | [Bounded reads and checkpoints](../crates/core/src/import/pipeline.rs) |
| Record, pause or stop | [Meeting routes](../crates/host/src/meeting.rs), [worker](../crates/host/src/meeting/worker.rs) | [Accepted audio](../crates/host/src/capture/); [transcript publication](../crates/core/src/live.rs) |
| Change accuracy or passage retry | [Refinement](../crates/host/src/routes/refinement.rs), [retry](../crates/host/src/routes/retry.rs) | [Saved refinement](../crates/core/src/live/context.rs), [protected edits](../crates/core/src/edit.rs) |
| Change speech or language rules | [Speech admission](../crates/core/src/speech.rs), [language policy](../crates/core/src/language.rs), [language context](../crates/core/src/live/language_context.rs) | [Import decoding](../crates/core/src/import.rs), [live engine](../crates/core/src/live.rs); [model graphs](../crates/models/src/) |
| Change editing or playback controls | [Editor](../speakerdesk/static/app.js), [live updates](../speakerdesk/static/live_editor.js) | [Naming UI](../speakerdesk/static/people.js); [existing DOM harness](../tests/support/frontend_dom.cjs) |
| Debug persistence or reopen | [SQLite store](../crates/core/src/store/sqlite.rs) | [Saved fields](../crates/core/src/document.rs); [migration](../crates/core/src/store/legacy.rs) |
| Debug setup or child failure | [Native host](../crates/host/src/native.rs), [setup](../crates/host/src/setup.rs), [supervisor](../crates/host/src/inference.rs) | [Pinned identities](../crates/host/src/model-artifacts.json); [protocol](../crates/core/src/ipc.rs) |
| Change native window, menu or recovery behavior | [Desktop shell](../desktop/src-tauri/src/) | [Startup UI](../desktop/frontend/); the transcript editor is embedded by the host |

## Trace a mutation

For speaker naming, follow the HTTP handler to `update_identity`, then the typed
job mutation. `SqliteJobStore::update_job` owns the transaction, revision increment,
timestamp and persistence. A failed callback rolls back. Keep model calls and
other external I/O outside that callback. Naming changes the meeting's track;
only **Remember voice** may enroll a profile.

Saved-voice work declares `VoiceRequest::Enroll` or `Check`. After validation,
`VoicePublication` carries the matching publication authority: a person's profile
generation, or a pending check ID and catalog snapshot. Model work runs after
releasing the store lock; publication rechecks the current audio and authority.

For saved speech admission, `import/pipeline.rs::feed_speech` owns frame reads,
receipt validation and ledger append for both reader imports and passage retry.
The callers retain session opening, conditioning and closing. Their documented
EOF-versus-close error order differs; the replay regression covers both. Live
streaming and old-audio rechecks keep their own clock and lifecycle rules.

For imports, `routes/imports.rs` owns upload staging, preparation, queue admission,
execution and cancellation. Its queued jobs may coexist; do not replace that
admission with the exclusive saved-refinement path. `import_progress.rs` commits
accepted passages and their resume cursor in one store transaction. Cancellation
persists a new generation before notifying the worker, and ownership remains held
until that worker exits.

For automatic language fallback, `live/language_context.rs` owns direct-observation
counts, speaker-before-meeting selection and witness freshness. The live engine
and saved refinement own when accepted publications update that evidence. Keep
physical-probe identity separate from display names: duplicate, conflicting and
overlapping observations must not become independent votes.

For asynchronous work, follow the operation's admission and generation through
publication, not just its model call. Cancellation acknowledgement does not mean
the worker has released ownership. A stale model result must not overwrite a
newer passage or a protected edit. Existing import and meeting integration tests
exercise those races with a disposable inference child and real SQLite.

Do not serialize a whole typed `Job` merely to edit a known field. Use its fields
and keep extension metadata in `job.extra`; preserve unknown keys. JSON belongs
at the HTTP boundary and in genuinely extensible saved metadata. Add a feature
module only when it gives a coherent operation a clearer home.

## Run the smallest useful check

Use the pinned Rust toolchain, Node 24 and Python 3.13 described in the README.
Check `node --version` and `python3 --version`: macOS may resolve `python3` to its
older system runtime. Install Python tooling in a virtual environment:

```sh
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements-ci.lock.txt
```

The following checks need no downloaded neural models or microphone permission:

```sh
# Speaker naming, consent, stale responses, persistence and export.
cargo test --locked --features speakerdesk-host/fixtures -p speakerdesk-host --lib people_routes

# Speech admission and import rules, including independent reference fixtures.
cargo test --locked -p speakerdesk-core --lib speech::
cargo test --locked -p speakerdesk-core --lib import::

# Actual host/worker/SQLite interactions using a disposable model protocol replay.
cargo test --locked --features speakerdesk-host/fixtures -p speakerdesk-host --test live_meeting

# Shipped editor behavior with the existing DOM harness.
node --test tests/speaker_identity.test.cjs tests/reading_turns.test.cjs

# Build and release tooling, including executable Swift checks on macOS.
.venv/bin/python -m unittest discover -s tests -v
```

After a change crosses owners, run the broader checks in [AGENTS.md](../AGENTS.md).
`cargo test` compiles and type-checks the selected targets. Use `cargo clippy`
with the same packages/features for lint diagnostics and `rustfmt --check` on
edited Rust files. A passing fixture test establishes its modeled interaction;
physical capture, installed-app behavior and real-model quality need their own
evidence. Do not regenerate a golden file to repair a regression.

For a rendered editor smoke check, `tests/live_partition_browser.cjs` serves the
shipped frontend against synthetic snapshots in a disposable Chrome profile. It
accepts a Chromium executable and an output directory. It does not validate the
real host, capture or model output. `cargo xtask --help` lists the implemented
bundle, Python-sidecar and native model-probe tools; the tests above own the
import, capture and editing regression checks.

## Run the editor without models

```sh
cargo run --locked -p speakerdesk-host --features fixtures --bin speakerdesk-dev -- --fixture people
```

Open the printed local URL. The fixture serves the real editor and API with a
fictional two-speaker transcript and silent placeholder audio in a fresh SQLite
library. Name speakers, edit words, reload the editor and export through the
normal routes. Model-dependent actions retain their normal unavailable state;
no models are downloaded. The silence exercises playback controls, not speech
or recognition quality.

Each invocation uses its own temporary directory and an available loopback port.
Ctrl-C stops the server and removes that invocation's data, including edits. The
fixture never opens the personal library. Use this for interactive editing and
identity work; use the inference and capture checks for their separate boundaries.

## Debug without losing state

Use a disposable storage root and model directory for native experiments. The
desktop accepts `SPEAKERDESK_HOME` and `SPEAKERDESK_NATIVE_MODELS`; do not point a
test at a personal library. GET, reopen and export must remain passive. A setup
failure belongs to the requested model target, and prepared files alone do not
prove that the child initialized them. Keep ordinary passage uncertainty separate
from failures that require repair, and verify continuation after an accepted
repair as well as save/reopen when changing that flow.
