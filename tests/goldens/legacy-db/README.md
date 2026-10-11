# Legacy v0.6.6 database fixtures

Real `recordings/jobs.sqlite` files produced by the shipped v0.6.6 build
(`/Applications/Speakerdesk.app`, `speakerdesk-runtime`) — no fabricated
rows. They exist so the Rust migration (D5) can be tested against databases
the old app actually wrote.

Regeneration: `python3 tools/oracle/make_legacy_fixtures.py [fixture …]`
(see `tools/oracle/README.md`). For every fixture the driver:

1. Creates an isolated `SPEAKERDESK_HOME` under
   `.cache/legacy-fixtures/home-<name>/` and APFS-clones the installed model
   set into it (`cp -cR ~/Library/Application\
   Support/com.speakerdesk.desktop/models <home>/models`). The real app data
   directory is never touched beyond that one read.
2. Synthesizes 16 kHz mono WAV with macOS `say` (two voices) + `afconvert`.
3. Drives the app over loopback (`POST /api/jobs`, `/run`,
   `PUT /transcript`, `/api/meetings`, `/api/people`, …) with the CSRF token
   from `GET /`.
4. Copies `recordings/jobs.sqlite` here and runs `wal_checkpoint(TRUNCATE)`.

Schema (all fixtures): tables `jobs`, `job_metadata`, `job_progress`,
`people`, `preferences`, `voice_profiles`.

## Fixtures

| File | Scenario | Rows (jobs / people / preferences / voice_profiles) | sha256 |
|------|----------|------------------------------------------------------|--------|
| `import-ready.sqlite` | WAV upload → inference to `ready` with real models → `PUT /transcript` text edit + `PATCH` speaker change. Protected edits exist (`protected_fields`, `machine_revision`, `refinement_state='edited'`); revision 3. | 1 / 0 / 0 / 0 | `0ad20592fa9ad3b4c2a97392ac1c20f8d51486e28366028acafe97452426e108` |
| `manual-transcript.sqlite` | Two uploads held at `uploaded`, then `PUT /transcript` without inference: one manual transcript (`provenance.kind='manual'`) and one with `imported: true` (`provenance.kind='imported'` + unverified warnings). | 2 / 0 / 0 / 0 | `17017fda75a292c0d365ed3387d81a2c92d148e662d58b77e999600e2f9d7116` |
| `active-run.sqlite` | Upload → `POST /run` → job observed `processing` ("Transcribing 10 audio regions with Cohere…"), then the whole runtime process group was `SIGKILL`ed mid-run. Status stays `processing` for recovery mapping. | 1 / 0 / 0 / 0 | `58c520587fbb7de045be3bc795f8598362c4d06d7c41485201144d3177b71555` |
| `meeting-canonical.sqlite` | Canonical meeting (`kind='meeting'`, `sources=['microphone']`) recorded through the synthetic capture helper `scripts/replay_capture.py` (`SPEAKERDESK_CAPTURE_HELPER` + `SPEAKERDESK_REPLAY_DIR`) against the live worker with real models; stopped and finalized to `ready`. Sealed canonical segments with speaker-activity evidence; default language preference stored. | 1 / 0 / 1 / 0 | `b652509efc11361313b1446610d6778398d517813b25fa59db94ce23fc7fe253` |
| `people-voice.sqlite` | Canonical meeting → person created via `POST /api/people` → speaker identity confirmed → voice enrolled via `POST /api/people/<pid>/voice` with recommended clean clips (real ReDimNet2-B6 extraction; `voice_profiles` row holds a 192-dim centroid + model identity) → default language set to `de`. | 1 / 1 / 1 / 1 | `39a1344b5fbbc13c4a15431689587741db0a2e019553b7038e36c2de2b72f229` |

Sizes: 60–108 KiB each (110,592 / 61,440 / 61,440 / 110,592 / 110,592 bytes
in the table order above).

## Sanitization

None required: the v0.6.6 store persists only basenames (`source.wav`,
`audio.wav`) — a scan of every table payload for the temp-home path,
`/Users/…`, and `/tmp/…` found zero stored paths in all five databases.

## Runtime notes

- Import inference on ~27 s audio: ~13 s end-to-end on this machine.
- Meeting fixture: ~27 s real-time replay + ~1 min finalization.
- Voice enrollment adds ~15 s of ReDimNet extraction.
- The packaged sidecar forks a worker process tree; the driver puts it in
  its own process group so the mid-run kill reaches every process.
- Two TTS voices are **not** separated by Nemotron during import (all speech
  lands on one track) but **are** separated by the meeting pipeline — the
  voice fixture therefore enrolls from the canonical meeting.
