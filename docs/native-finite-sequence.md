# Finite native sequence orchestration

These modules connect the existing queue, pump, workflow, publication and
retirement validators. They do not replace the renderer or certify media quality.
All runtime data belongs in a canonical private directory under `edits/work/`.

## Prepare one deferred member

```sh
.venv/bin/python -B -m replay.sequence_adapters prepare-member \
  edits/work/sequence/request.json --sha256 REQUEST_SHA256 --part PART_NUMBER
```

The version-1 `native-sequence-data` request contains ordered `members`. Each
member supplies `part`, pinned `preparationRequest`, `candidateRoot`, `queuePath`,
pinned `queueTemplate`, `pumpPath`, pinned `pumpTemplate`, named future
`dependencies`, and `predecessor`. The first member's predecessor is null;
later members bind the preceding part, its original `retirementConfig` and
`postDeleteStatus`. Future dependency paths need not exist at initial planning.
They are resolved and pinned only after original predecessor verification succeeds.
An exact `{"$ref":"name"}` object resolves a declared dependency; ordinary strings
are never rewritten. Candidate reuse checks original input pins and the original
queue validator. Existing queue/pump files are immutable and must match exactly.

This CLI prepares DATA, returning `canDispatch:false`. Original live grants,
actual provider/process identities and fresh admission remain separate inputs.
`replay.native_sequence.run(parts, journal, steps)` is the finite execution API.
Use `sequence_journal.DurableJournal` for exclusive hash-chained checkpoints and
`sequence_adapters.BoundSteps` with explicitly typed members/runtime factories.
The adapters call original pump/workflow/release APIs. There is no daemon or
dynamic import of arbitrary callback names from JSON.

The remaining tail and each prepared config are immutable. A durable started
intent precedes the pump. Started or uncertain effects are never resubmitted.
Original retirement/post-verification and `verify_release` precede successor
admission. A resource refusal preserves reusable state and exits finitely.

## Exact redundant BOOT-copy retirement

`replay.boot_duplicate_retirement.prepare_graph(request, output, python)` creates
an original `replay.workflow` graph. Run its `--check` before execution. The request
uses `kind:exact-boot-duplicate-retirement`, `version:1`, `root`, `boot`, pinned
`retirement`, `verification`, `mandatoryBefore`, `disposition`, `archive`, `code`,
and exact ownership `leases`. `code_paths()` supplies the bounded actual import
closure whose current hashes must be pinned. The original publication authority,
completed before-delete readonly domain, unchanged source/survivor identities,
archival proof and all current/queued/resume/FCP/Viewer/diagnostic dispositions
must exist first. The adapter does not infer authorization or dependencies from
names. It unlinks only that exact redundant BOOT copy, under original ownership,
strict open-file and FD/path guards. It never retires a compressed-source original.

```sh
.venv/bin/python -B -m replay.boot_duplicate_retirement execute REQUEST.json
.venv/bin/python -B -m replay.boot_duplicate_retirement verify REQUEST.json
```

An uncertain pre-delete intent is preserved and refused. Once actual source
absence is established, the original after-delete domain must succeed before the
adapter emits post status, audit and release. Those artifacts connect to the
existing `OriginalReleaseFactory`; they do not themselves grant successor dispatch.

If the old owner died or its observation expired after unlink, use a separate
pinned `exact-boot-post-only-resume` record with `--resume-observation PATH
--resume-sha256 SHA`. It binds the original request/operation/source, survivor
identity, archive/code/leases and a new fresh disposition. Preserve the original
request and graph; prepare a separate resume graph/state. Post-only continuation
cannot unlink, even if a file reappears at the source path. Code-version migration
is not permitted by renewing an observation.

## Optional resource recovery

`run(..., recovery=sequence_recovery.BoundedRecovery(...))` can invoke a typed
callback only for unleased, prepared resource refusal. It records one attempt per
saved project, actual-progress content hash and refusal reason. Unchanged,
ineffective and uncertain attempts cannot repeat. Normal quit must prove exact
old-process absence before one launch; force additionally requires failed normal
quit and unresponsive evidence. The new owner must read back the same saved
project and outputs, with no modal or export and all nine native categories idle.
Pressure and measured physical footprint must improve before future candidate
DATA is prepared. The old queue/checkpoint is never replaced or automatically
resubmitted.

The recovery `Runtime` to current Executor binding is **not implemented**. It must
validate actual provider schemas, PID/birth, original observation freshness,
saved project, complete owned outputs and original resource gates. Existing
`app_quit` targets an edition and has no expectedPID argument; the live binding
must explicitly fence ownership/identity rather than invent an atomic PID guard.
Synthetic tests do not prove native app effects or production acceptance.
