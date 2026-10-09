# Transcript timing setup contract

This patch closes the missing product setup action. The selected provider declares
functional language support separately from independently calibrated timing
accuracy. The authorized production CTC target is all fourteen product languages;
English is the independently calibrated accuracy scope. Missing independent word
references for other languages is a disclosed accuracy limitation, not an
activation prohibition after functional/safety validation. GPU execution alone
does not establish word-boundary accuracy. Tiny Whisper remains rejected.

The multilingual CTC provider retains the immutable weights and runtime and adds
supplied-text normalization and explicit character timing units for Chinese and
Japanese. The previous English-only provider's readiness receipt cannot certify
this behavior. Retry reuses verified weights and creates a receipt for the new
provider identity after the GPU compatibility check succeeds.

Normal first-use `POST /api/setup` now includes timing by default. Existing installs
can choose **Settings → Models → Transcript timing → Download & enable** after the
core download control has disappeared. Setup verifies pinned downloads, prepares
GPU data, verifies artifacts and the exact packaged runtime/native extension, then
executes a bounded GPU compatibility check in an owned inference child. Only a
completed, identity-matching receipt makes timing ready. A folder, original ONNX
file, converted weights alone, or matching package version alone cannot do that.

Downloads keep partial bytes after interruption, validate server resumption ranges,
verify every pinned source digest, and atomically publish complete files. Corrupt
prepared cache repair is explicit in setup; the inference loader still refuses
corruption. Failed checks clear earlier readiness receipts. Errors persist across
restart; retry reuses verified source data. General setup preserves a user's off
preference and reuses a previously checked provider without another neural check.

## API

`GET /api/setup` keeps its existing core/voice readiness fields and adds `alignment`:

- `id`, `provider_id`, `provider_identity_sha256`: selected cache and behavioral
  identity. `supported_languages` describes validated functional coverage.
- `timing_accuracy_calibrated_languages` describes independent accuracy calibration;
  Settings displays both scopes. An empty accuracy list does not block activation.
- `status`, `phase`, `error`, `downloaded_bytes`, `total_bytes`: independent timing
  progress and actionable idle/downloading/interrupted/failed/ready state.
- `ready`, `enabled`, `active`: verified capability, persisted preference, and their
  conjunction. Core transcription readiness remains independent.
- `stored_bytes`, `required_free_bytes`, `can_download`, `message`: truthful prepared
  footprint, setup capacity requirement, admission and user guidance.

`POST /api/setup/alignment` starts/resumes/repairs only timing and enables it on
success. `PATCH /api/setup/alignment {"enabled":true|false}` persists activation for
new meetings/imports. Enable refuses unverified capability. Active transcription,
recording, refinement and setup block preference changes and setup admission.
Starting new neural work while setup owns its compatibility check is rejected;
editing, playback and readonly polling remain available. The existing CSRF and
update admission apply unchanged.

`pipeline.model_config()` supplies the timing path only when the preference and
full verified readiness agree. No change is made to Cohere words, alignment score
policy, calibrated English gates, canonical publication, or transcript ordering.
The provider's neural forward requires Metal GPU and completed evaluation; there
is no CPU neural fallback. Unsupported language timing retains original-audio
anchors rather than inventing word times.

## Provider handoff

`alignment_provider.py` owns these replacement hooks together:

- `manifest()` returns immutable artifact/calibration/runtime identity.
- `require_runtime(metadata)` checks exact versions, component identity and native
  ABI without neural initialization.
- `prepare_model(directory)` deterministically prepares verified downloaded data.
- `verify(directory)` verifies every complete source and prepared artifact plus
  runtime compatibility without neural initialization.
- `gpu_check(directory)` performs a bounded real GPU forward and returns execution
  and artifact/runtime provenance. Compatibility is not acoustic qualification.
- `validate_receipt(receipt, metadata)` refuses stale/different/uncompleted checks.

`alignment_artifact.ALIGNMENT_SPEC` supplies immutable repo/revision, directory,
weight filename/size, all source filename/digest pairs, `download_bytes` for the
complete source inventory and `file_sources` (relative filename → exact bytes,
optional immutable repo/revision override). This supports weights and tokenizer
metadata from separate pinned origins. `prepared_files` (relative
filename → bytes/digest), provider id, functional `supported_languages` and
`timing_accuracy_calibrated_languages`. Empty
prepared_files supports a provider requiring no derived weights. The lifecycle
uses these fields for repair and resource reporting. A provider with no functional
language coverage cannot become available or active through setup. The selected provider's
actual inference adapter and calibrated application gates remain a separate
integration responsibility.

### Behavioral receipt identity

`manifest()`/`verify()` preserve tensor manifest/model/runtime identity and add
`provider_identity` with schema_version=1, provider_id, supported_languages,
timing_accuracy_calibrated_languages, normalization_policy_id,
timing_unit_policy_id and implementation_source_sha256 (actual behavior module
relative path → digest). Runtime verification checks those source bytes without
neural initialization. The identity excludes its own digest.

`provider_identity_sha256` is SHA-256 of UTF-8 canonical JSON for that object:
`json.dumps(identity, sort_keys=True, separators=(',', ':'), ensure_ascii=False)`.
The GPU receipt additionally contains provider_id and provider_identity_sha256.
`validate_receipt()` requires exact agreement with the current metadata alongside
existing model/tensor/runtime/completed-Metal checks. Missing or legacy identity
cannot activate new behavior; retry reuses verified weights and performs the owned
compatibility check. The provider owns this validation, so setup never weakens it.

The new `alignment-check` inference task uses the existing worker registry and
shutdown cleanup. Its timeout is 120 seconds, with kill and join on timeout. The
packager includes both setup and provider modules. A frozen application build and
actual functional/safety validation remain acceptance gates. Independent accuracy
coverage stays explicit and does not silently widen during activation.

## Validation

The new lifecycle suite owns first-use/default selection, upgrade cache reuse,
corruption, safe range resumption, interrupted/restarted errors, runtime/GPU refusal,
persisted off/on behavior, low disk, serial admission and child timeout cleanup.
It replaces the older single mocked optional-source preparation test with stronger
HTTP lifecycle coverage. Small CPU fixtures stand in for large model bytes;
conversion math/atomicity/backend guards retain their existing tests.

The real frontend DOM tests exercise Download/Retry/PATCH actions after core setup,
verified toggle readiness and focus preservation during polling. A baseline run
against 0.6.3 fails on the missing action state. These tests are not browser layout
or hardware inference evidence.
