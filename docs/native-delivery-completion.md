# Native delivery v3 index and completion adapter

`replay.native_delivery_completion.assess(index, *, root, delivery)` is a new
read-only API. `index` is an absolute `{path, sha256}` reference. Explicit v3/native
backend dispatch is connected to `edit_completion check`; its structured return
is never treated as the legacy failure dictionary. `completion_report` includes
`nativeAssessment`, preserving candidate and recorded stages alongside strict failures.

## Index boundary

Exact top-level keys are `version` (integer 3), `backend`
(`native-fcp-partitioned-whole`), `recording`, `configuration`, `selection`,
`descriptor`, `identities`, `master`, with optional `observation` and `measurements`.
The three metadata references bind the exact
recording edit-config, its selected candidate header, and the version-one native
recorded-evidence descriptor. Unknown keys, including caller quality approvals,
are rejected. Existing edit-config acquisition/storage authorization still belongs
to `edit_completion.load_config`; this adapter additionally checks enabled stages,
source size/path/digest when supplied, and selected configuration SHA.

`identities` has exactly seven roles: `source`, `movie`, `srt`, `semanticXML`,
`template`, `modelConfig`, `modelWeights`. Each is a seven-field row:
`path`, `bytes`, `sha256`, `device`, `inode`, `mtimeNs`, `ctimeNs`. The identities
must be unique, canonical and regular. Current stat identity is compared both
before and after metadata assessment. **Stable stat is not fresh SHA verification.**
Default `assess` is metadata-only and does not read media/model bytes. Explicit
`verify_content=True` captures and finishes an `InputHashSnapshot` against all
seven original expected SHAs. Only that successful live scope sets currentCryptoVerified.
Mandatory `edit_completion check` uses live mode. SRT SHA does not establish actual display.

Missing historical stat is supported without inventing provenance. The generic CLI
`python -m replay.native_delivery_completion INDEX --sha256 SHA --root EDIT_ROOT
--delivery SELECTED_MOVIE --observe-output NEW_JSON` accepts role references
`{path, sha256}` instead of rows in its input index, checks them against the original
descriptor/header, and writes a new index with current rows and explicit
`current-identity-observation`. The original expected references remain inside
that observation; old producer records are untouched. Publication is exclusive,
0600 and repository `edits/work` only, after initial/terminal SHA and identity checks.
The output can be a declared workflow output; its read-only domain verifier is
the same CLI pinned to the ORIGINAL request, with
`--verify-generated-observation GENERATED_INDEX`. The public
`verify_generated_observation(request, generated, *, root, delivery)` compares
all unchanged request fields and expected role references to the generated
observation. The workflow receipt binds the actual generated output SHA; it need
not be guessed before the producer runs. The verifier bounds generated JSON size
before hashing it, rejects links, and never executes the observer. For a separately
known generated SHA, `--verify-observation` remains available.
It requires explicit observation provenance, validates original domain records
and current stat identity, returns observationVerified (not strictComplete),
and does not recapture media SHA or rerun the producer. Graph inputs must declare
all underlying refs; original workflow completion fingerprints/output SHA still
govern reuse. Plain assessment returns 2 when strict requirements are missing.
Use original expected pins, not a fresh digest chosen
to conceal changed bytes. A persisted observation never proves current crypto by itself.

Context source/movie/semantic pins, selected movie target/SRT/config, head-config
template and original ASR model pins must agree with these roles. Caller delivery
must be the exact selected library path. A publication alias is accepted only when
it currently resolves to the explicitly selected canonical target. No other
symlink identity, escaping relative name, alternate model or template is accepted.
Original frame/sample clock checks remain in the existing native evidence reader.

`master` is `{phase, reference}`. `historical` binds the original whole master
reference without inferring availability. `retired` binds the original immutable
retirement outcome, intent and deletion records, checks master/request/identity
and reconstruction lineage equality, and retains fresh/rebuild false. This does
not require the retired media to exist and does not reverify current retained
parts or authorize deletion/rebuilding. It never rewrites history.

## Return dimensions and strict contract

`candidateAvailable` reports current selected file existence/stat identity within
this API's scope. It is not playback, fresh crypto or semantic acceptance.
`recordedStages` maps the eight original native domains onto the existing five
required stages. `strictFailures` names actual missing evidence;
`recordedRequiredStagesComplete` summarizes only original execution. Optional
`measurements` maps `audioQuality`, `cutExecution`, `maskQuality` to pinned original
v2-schema measured receipts. Live mode calls public
`delivery_completion_v2.validate_native_measurement`, binding the actual native
descriptor, source, movie, cuts and voice. Passing numeric/coverage checks clears
only the associated measured requirements. Audio's original separation-model
lineage remains pending even after post-AAC numeric success: that validator does
not establish the model. Failed, absent or unsupported receipts never
clear failures. Metadata-only mode does not execute these media-reading validators.
`strictComplete` requires no remaining failures and a successful live crypto scope.
Native paint/display contracts remain unsupported, so current candidates still
fail strict completion. Quality/sync/narrative approvals remain false. No human
flag or bare accepted label can unlock completion.

Safe strict success would require all of:

1. Current selected source/config/movie/SRT identities and applicable cryptographic
   verification, independent full AV/frame/sample clocks and original producer
   closure, with archival master semantics preserved.
2. All five recorded required stages, with complete native part inventories and
   exact source-to-cut-to-caption clocks, models and template lineage.
3. Real full-coverage separation, protected-speech/music, selected cut protection,
   local repair quality and actual painted caption/FCP evidence below, verified
   by their typed producers and unchanged thresholds. No unsupported native paint
   schema may be interpreted as successful alpha evidence.

Existing reusable validators are `delivery_completion_v2._audio`, `_cuts`, and
`_mask`. Their numerical policies and coverage checks are independent of alpha,
**but their store/render/source/movie inputs must be supplied by an explicitly
validated native binding context**. `_audio` needs real post-AAC speech/music
windows and protected sample coverage; `_cuts` needs protected-plan/execution
projection; `_mask` needs selected local-region compositor provenance and complete
residual/readability/instability coverage. Calling them directly with an old
render configuration or a metadata boolean would not satisfy that context.
The public wrapper reuses these exact checks without alpha evidence or threshold changes.

`_remux` and `_caption_quality` are not interchangeable native validators: they
expect direct-CLI job layout/alpha parts. A native actual-paint contract still
needs every caption state, style20 font/template, line bounds/gaps/head-near
geometry, exact SRT display and original editable FCP import. Existing plain
head/template lineage cannot prove these measurements. This is the precise
remaining strict-success implementation gap, not a request for human approval.

Dispatch preserves v1/v2 and their strict exit semantics. CLI assessment returns
0 only for strict completion, 2 for valid candidates with outstanding strict
requirements, and 1 for invalid evidence. Observation generation returns 0 only
for its new index, not delivery acceptance. No original producer was changed.

## Verification and actual mapping limits

Tests isolate this new index/identity/report boundary from the separately tested
173-case original-record reader. Synthetic all-domain reports test aggregation;
the actual reader is also tested to reject incomplete descriptors, not replaced
by a mock in production. Native no-alpha availability, changed files/pins,
selection/config/model/template mismatch, incomplete parts, malformed schemas,
mutation during assessment and immutable retired master cases are covered.

Actual Oct7 eight-domain data already exists under
`edits/work/native-delivery-evidence-verification/actual-oct07-v1`. Creating a v3
index needs its corresponding retained/current seven-field rows for **all seven
roles**, the exact current header/config, and pinned retirement outcome. The old
final receipt supplies some final input identities, not all model/template/SRT
identities. Missing identities must be reconciled from original terminal evidence
or obtained by an authorized observation, never fabricated from a SHA or copied
from a different revision. No actual v3 assessment or new media/stat scan is
claimed by these synthetic tests.
# Recorded master retirement

The native reader validates the original version-1 retirement transformation:
remove exactly the original master pin from the prepared inventory, then add the
pinned intent, deletion, final disposition and that disposition's declared
reference evidence. Duplicate paths, conflicting pins, additional unknown inputs,
missing request/master pins and unresolved consumer scopes are rejected. The
domain and original identity remain bound; an absent retired master is never
reported as freshly verified or materialized. This is metadata interpretation of
the original producer, not permission to delete or rerun retirement.
