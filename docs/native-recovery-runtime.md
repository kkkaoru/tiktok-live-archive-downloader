# Finite native recovery binding

`replay.fcp_recovery_runtime.OwnedRecovery` is an optional callback for
`native_sequence.run`. It requires the exact pinned, granted original pump
configuration and an unstarted prepared checkpoint. It acquires the original
exclusive GUI lease only after the sequence has observed no lease. It never
imports, exports, deletes media, or starts a successor. An uncertain call retains
the lease and immutable request/response for reconciliation.

Construct `Settings` with canonical repository evidence storage, the current
Executor provider, exact app and controller PID/birth, original pump config,
provider evidence, saved-project record, stable progress record, complete owned
output identity records, and localized background window/menu values. Supply
`queue.ExecutorTransport`, `SystemHost`, and a typed `FutureFactory`. A successful
recovery releases only its exact lease after the new saved project and full
nine-category idle state are observed. The factory prepares separately pinned
candidate DATA; generation adoption is a separate verified operation.

Each output pin references small metadata with `version: 1`,
`kind: native-recovery-output-identity`, canonical `path`, and `identity` containing
device, inode, bytes, modification nanoseconds and change nanoseconds. The adapter
compares actual stats and exact-path `lsof` access modes; it does not hash media
through the bounded metadata reader. Read-only consumers may remain open; writers
or unknown access modes refuse recovery. The caller must provide the complete
authoritative inventory, not a directory-derived subset.

Pressure uses actual Darwin values 1/2/4 and is separate from libproc physical
footprint. Stable progress is the digest of actual completed effects, never a
poll timestamp or process ID. The immutable runtime observation pins saved
project, progress, output records, provider evidence and original observation
time. It is included in the recovery result's `afterEvidence`.

The installed `app_quit` schema is not atomic with an expected PID. The adapter
checks the exact process immediately before dispatch and requires its actual
absence afterward, but cannot claim an atomic OS compare-and-quit operation.
Force termination remains unsupported by this binding without independent
unresponsive evidence. A slow/unready launch refuses subsequent processing;
it does not relaunch, reimport or retry a possibly applied effect.

Synthetic tests exercise the original sequence callback and acquired lease,
native response schemas, actual-observation guards and SDK footprint decoding.
No live recovery or production successor acceptance is established by those tests.
