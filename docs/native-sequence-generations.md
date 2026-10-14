# Immutable native sequence generations

`replay.sequence_generation` connects successful resource recovery to a new
unstarted sequence generation. It reads bounded metadata only; it performs no
app operation, media processing, output approval or GUI grant creation. Existing
queue/pump validators and fresh live capacity admission still precede every
native effect.

The original journal, checkpoint, request and recovery result stay unchanged.
The target checkpoint must be the current `prepared` head of its hash chain.
Every earlier member must have a pinned `released` head and its original proof;
any started, published, retired or uncertain member in the remaining suffix
refuses continuation. The new Config keeps exactly that ordered suffix while
changing the actual pump configuration bound to the recovered PID/birth.

## Request and observed future DATA

The version-1 `native-sequence-generation-request` has:

- `orderedParts`: the complete original ordered part identities.
- `oldJournal`: `root`, pinned `generation`, pinned current `prepared`, and
  ordered pinned `predecessors` (the last checkpoint of each released predecessor).
- `recovery`: the original pinned `finite-native-recovery-result`.
- `ledgerRoot` and `journalRoot`: distinct, existing canonical work directories,
  disjoint from the original journal. The new journal must initially be empty.

The ledger is the unique sibling named `<old-journal-name>-generation-ledger`.
A different caller-supplied ledger cannot bypass an existing adoption claim.

References use `{path, sha256}`. The original generation and every referenced
metadata file must retain their exact bytes. Its existing lock is opened
read-only with a shared nonblocking lease; missing locks are not created.

The original recovery result must report `improved:true`, a different current
PID/birth, exact intent, quit and after-evidence pins, and pinned `futureData`.
The original process identity must be absent. Its future DATA is version 1,
kind `native-sequence-continuation-data`, `candidateOnly:true`, `canDispatch:false`,
and `continuation` with these exact keys:

- `originalConfig`: original `{identity, sha256}`.
- `newConfig`: `{identity, tail, sha256}` for the actual new pump JSON.
- `observation`: pinned `native-recovery-runtime-observation`, also present in
  the recovery result's `afterEvidence`.

The observation contains version 1, `target:{pid,birth}`, `project`,
`savedProject`, `progress`, `outputs`, `observedAtNs`, `maximumAgeSeconds`
(1–30), and `providerEvidence`. Its project/output/progress references must
match the original intent. Its observation time is retained from the actual
native readback, never generated or refreshed during adoption.

The new pump's queue, admission template and runtime evidence must all bind the
recovered target. Runtime and queue provider/executor identities must agree;
the observed provider evidence includes the pinned runtime capability artifact.
A supplied original-schema GUI handoff must bind the new target. These checks
do not replace complete original pump/queue domain validation.
The ordered queue members and their original operation references stay identical;
recovery cannot substitute changed source XML, captions or export operations.

## Adoption, interruption and execution

```
python -m replay.sequence_generation adopt REQUEST.json --sha256 HASH
python -m replay.sequence_generation resume REQUEST.json --sha256 HASH
python -m replay.sequence_generation verify REQUEST.json --sha256 HASH
```

`adopt` validates the fresh observation and exclusively reserves the original
prepared checkpoint in the adoption ledger before writing new journal data.
A second adoption, including one using another destination, is refused.
`resume` only completes that exact reserved DATA adoption after interruption;
it cannot create another claim or retry native effects. Completed adoption is
opened using `verify` or the typed `open_generation(request, host=...)` API.

The new journal starts with the new Config's `prepared` checkpoint and pins the
old history, successful recovery, observed future DATA and current implementation.
Its `continuation.json` carries released predecessor evidence. Call
`native_sequence.run(new_config.tail, opened_journal, original_bound_steps)`
with BoundSteps for the newly supplied queue/pump. No old released predecessor
is re-rendered, and no old prepared Config is replaced. Subsequent resume treats
the original observation as historical evidence; the live PID/birth and all
immutable inputs still match, and original live admission observes capacity anew.

Synthetic tests establish metadata and restart behavior. They do not prove
successful native recovery, rendered video quality, synchronization or delivery.
