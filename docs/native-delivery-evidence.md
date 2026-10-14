# Recorded native-FCP delivery evidence

`replay.native_delivery_evidence` is an optional, read-only metadata assessor for
`native-fcp-partitioned-whole`. It is **not connected to `edit_completion` or any
production workflow**. It neither invokes producers nor decodes, hashes or stats
media/model/tool files. It does not import historical/private processing code.

Run against an explicitly pinned JSON descriptor:

```sh
uv run python -m replay.native_delivery_evidence DESCRIPTOR.json --sha256 DESCRIPTOR_SHA256
```

Version 1 requires `backend: "native-fcp-partitioned-whole"`, a `context`, and
`domains`. Context binds `{path, sha256}` references named `source`, `cuts`,
`voice`, `semanticXML`, `wholeMaster`, `finalMovie`, and positive integer `frames`
and `fps` for the edited whole clock. This first contract supports integral frame
rates; unsupported clocks must not be rounded into this schema. Cross-domain
references and frame clocks must agree. Unknown domains are rejected. Missing
domains return pending, not successful completion.

Every read record is an absolute canonical, non-linked `.json` or `.log` file containing a JSON object with an
explicit lowercase SHA256. The reader rejects duplicate JSON keys, changed file
identity, conflicting pins and nonregular files. Limits are 16 MiB per record,
128 MiB per domain, 192 MiB aggregate and 4,096 unique records. Domain accounting
includes cached records once per domain; aggregate accounting counts unique reads.
The finite limits reflect the retained 631-record Oct7 inventory (135,014,443
bytes total; largest record 13,103,119 bytes; largest domain 91,246,761 bytes).
They are bounds on serialized bytes, not on Python object memory. Historical media/tool pins remain
strings: validating their enclosing metadata does not freshly validate those
files. The result reports actual metadata bytes, unique records read and per-domain totals.
JSON-looking binary data, malformed logs, duplicate keys and array roots are
rejected; the `.log` allowance does not parse arbitrary process text. Original
files are read in place without copying or renaming their history.

## Original contracts

Domain entries refer to original records, not caller assertions that a stage was
applied. The public typed Python helpers and synthetic fixtures provide complete
schemas for the supported producer versions.

| Domain | Required original evidence and checks |
| --- | --- |
| `cleanup` | Approved `source`, cleanup assembly/index and every chunk producer/candidate/eligibility record; source and eligibility pins, contiguous presentation clocks, complete observed frame inventory, active repair frames classified as subtitles. Unknown inactive frames are not approved. |
| `voice` | `receipt`, exact `expectedInputs`, `output`, `frames`, `fps`; version-one context-atempo exact cut-slot receipt, contiguous samples/crops and sample duration equal to whole frame duration. |
| `cuts` | `receipt`, `frames`, `fps`; source-presentation frame cuts, source bounds, monotonic nonoverlapping cuts and contiguous edited output endpoint. |
| `asr` | Original input/completion/parent records and refinement/projection lineage; exact model config/weights and runtime pins, non-turbo large-v3 identity, CPU runtime, complete unique windows, original PCM clock and one projection into source then edited clock. |
| `caption-layout` | `headReceipt`, `headConfig`, `refreshReceipt`, `refreshInputs`, `output`; original reflow/cuts/template/measurement pins, retained caption stages, refresh head pin, text/media-clock preservation and exact semantic output. |
| `native-parts` | `frames` configuration: pinned partition, semantic XML, ordered publications, frame-proof tool, dimensions, whole frames/fps; every original native completion and publication intent. Optional `retirements` must cover exactly the same ordered publications and bind post-delete evidence. Unretired parts are supported without inferring their present availability. |
| `whole-assembly` | `stageEvidence`, `wholeRecords`, `source`: four original whole phase records (join, frame clock, metadata, PCM) bound within the final verification checkpoint's declared input inventory. Individual whole producer fingerprints remain explicitly unproven by this projection. |
| `final-delivery` | `stageEvidence`, `receipt`, `submit`, `status`, `request`, `success`, `output`, `frames`, `fps`; exact completed typed final-verifier checkpoint, unchanged seven-field input identity rows, full decode, selected media pin, whole clock and original official successful Compressor job/batch identity. Failed/false/truncated responses and 100%-but-failed states are rejected. |

`stageEvidence` contains pinned `workflow`, `checkpoint`, `inventory` (JSON with
an `inputs` SHA map), selected `stage`, absolute original `cwd`, and exact
`outputs` SHA map. This specific wrapper accepts the original
`python -B -m replay.delivery_verification ...` final stage. It reconstructs its
command/verifier/input/output/timeout/cwd fingerprint. In the whole domain it
anchors original whole-phase records as **inputs to that final stage**; it does
not pretend the final verifier was the join producer or accept arbitrary
join/assembly commands. Unsupported original stage schemas remain unsupported.
The separate `followup_history` helper reconstructs retained native followup
checkpoint bindings without executing them.

## Evidence dimensions and limits

- `stageApplied` / `executionRecorded` mean the supported original record
  contract is consistent. They do not mean its semantic quality is approved.
- `recordedEvidenceBound` means the bounded JSON record chain was SHA checked.
  Scoped original checkpoint reconstruction reports `historicalExecutionVerified`.
- `currentCryptoVerified`, `qualityApproved`, `syncApproved`, `deliveryApproved`
  and `finished` remain false. `candidateAvailable` is unknown (`null`), because
  this reader does not inspect the media filesystem.
- An archived or retired whole master may have valid historical evidence. That
  does not imply current materialization, freshly verified retained parts, or a
  byte-identical rebuild. Existing publication/retirement records are immutable.
- Exact atempo/cut-slot execution alone does not prove upstream music separation
  or separation-model quality. ASR model pins do not prove present model
  availability, transcription semantics or every token's numeric projection.
- Head/reflow/template lineage does not prove painted style-20 font, line layout,
  face distance, or current full-length visual quality. A historical calibration
  must not be silently treated as a newer template's rendered proof.
- Official Compressor success is not a per-job native-provider runtime freeze,
  global absence of duplicate jobs, strict AAC/FastStart acceptance, or AV-sync
  approval. Original quality flags remain distinct and are never upgraded.

The adapter reports these pending facts instead of fabricating obsolete alpha
execution or human review. Connecting it to a production completion route needs
an explicit reviewed caller and supported producer contracts; this addition
changes no existing completion gate. Synthetic tests use tiny local JSON only,
with changed source/model/clock, incomplete records, bad native envelopes,
publication identity and checkpoint fingerprint regressions. No real media
acceptance is claimed by those tests.

Original final-verifier fingerprints are bounded arrays with exactly `path`,
`bytes`, `sha256`, `device`, `inode`, `mtimeNs`, `ctimeNs`. Paths must be unique
and lexically canonical; counts/identities/timestamps are positive integers, not
booleans. Before/after equality compares every field, not only hashes. Media
paths are not resolved or opened: these are original recorded identities, not
current observations. Unsupported maps or extra fields are rejected.

Only the official submission field enables the typed native `.log` reader. It
accepts at most one exact Executor settings-sync diagnostic before and/or after
one JSON object, retains those diagnostic strings in `submissionDiagnostics`,
and still requires a successful nontruncated native envelope. The diagnostic is
recorded text, not an instruction to run settings synchronization. Arbitrary
prefix/suffix text, repeated diagnostics, extra JSON objects and duplicate keys
are rejected. Other fields do not enable this log-envelope parser.
