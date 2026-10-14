---
name: disk-maintenance
description: Plan repository-local disk usage and safely clean up approved, unnecessary intermediate artifacts. Use before large processing jobs, when free space is low, or when asked to inventory or reclaim space. Preserve originals, deliverables, models, active/resumable work and required evidence; do not scan outside this checkout.
---

# Repository disk maintenance

[AGENTS.md](../../../AGENTS.md) owns the free-space floor, protected data and
approval boundaries. This skill supplies the procedure, not broader permission.
For live job ownership and recovery, read [repo-jobs](../repo-jobs/SKILL.md).

## 1. Measure before allocating

- Resolve the checkout and requested output/scratch paths; reject symlinks that
  escape the repository. Work artifacts belong in ignored `edits/work/`.
- Use bounded `df -k`/`df -h` calls on the actual output volume. Include the startup
  volume when output is elsewhere: build caches, logs and temporary work may still
  consume it. Do not infer physical free space by summing directory sizes.
- Estimate peak **additional** allocation: new inputs, concurrent intermediates,
  final outputs, concurrent jobs and any temporary backup/compression copies.
  Reuse verified results; avoid unnecessary full-length copies or re-encoding.
- Apply the floor in AGENTS.md throughout execution, not just after completion.
  If headroom cannot be established, defer the affected new large job, report the
  deficit and continue independent work that does not increase disk pressure.
  Re-measure at stage boundaries; free-space shortage alone is not a diagnosis
  of a Mac hang. Do not interrupt someone else's job to create headroom.

## 2. Inventory narrowly

Use metadata-only, bounded-depth `du` and targeted file listings first. Detach a
potentially slow scan with the repository-local job procedure and limit load.
Do not recursively scan `private/`, `downloads/`, external symlink targets or
other user directories. Do not read media/credentials just to estimate sizes.

Record exact candidate paths, sizes and their roles. Classify each as **keep**,
**eligible for approved cleanup**, or **dependency unresolved**. A candidate list
is not approval or proof of disuse. Age, size, a name such as `failed`, or absence
from one `lsof` snapshot is insufficient.

## 3. Establish eligibility

- Check current configurations, manifests, completion/quality gates, live/queued/
  resumable jobs and FCP/Viewer references, including indirect dependencies.
- Keep all protected data from AGENTS.md, including a sole source of failure or
  verification evidence. A regenerable file may still be a current dependency.
- Retire obsolete composite/mux/alpha media only when those dependencies no longer
  require it. If a verified replacement is the reason, validate that replacement
  and its identity first; do not infer equivalence from filenames.
- Keep settings, manifests, hashes, verification results and failure records.
  Never rewrite a receipt or relax a check to hide a missing file. An uncertain
  item is held individually; it does not block independently safe cleanup, and
  unrelated videos need not finish first.

## 4. Execute only the approved scope

A new direct deletion instruction takes precedence over an older user preference
to retain that item or an agent-authored keep list, subject to actual governing
rules and current dependency checks. Apply the [latest-request procedure](../../../docs/blocker-audit.md#latest-user-requests):
resolve an already identified target from context instead of repeatedly asking
for it. Do not substitute "stop using" for "delete", or mark an unexecuted removal
complete. A genuine higher-priority scope conflict remains an explicit unmet
request with its exact governing source; repeated consent alone does not remove
that conflict. Hold only the affected action and continue independent work.

1. Save a private cleanup record in `edits/work/`: exact paths, size accounting,
   why each file is unnecessary, approval scope and current dependency evidence.
2. Immediately before deletion, recheck canonical path, file identity and current
   references/use; exclude changed/uncertain entries. Coordinate with the owner
   where future use cannot be ruled out. Do not move live files or delete shared
   directories wholesale.
3. For disposable tool caches prefer their supported cleanup command, preserving
   in-use checks (no `--force`). For media use only the exact individually verified
   paths, never a broad glob/root removal. Prior authorization does not expire
   merely because the safety check took time; do not repeatedly request it.
4. Measure volume free space afterwards and report attempted/succeeded/held items.
   Distinguish removed logical allocation from the observed free-space increase;
   hardlinks, APFS clones/snapshots and concurrent jobs can change the result.

There is intentionally no pattern-based or automatic media-deletion script: dependency
and approval judgment is not encoded by a filename pattern. Two reusable, judgment-free
tools exist: `python -m replay.disk_budget report|preflight` measures physical use and the
free-space floor, and `python -m replay.leaf_retirement` unlinks only the exact rows an
owner marked `eligible` in a SHA-pinned disposition file, after per-row identity,
symlink-target and open-handle checks (dry run by default). See
[operations](../../../docs/operations.md#keeping-disk-and-ssd-use-bounded). `python -m replay.delivery_retention` builds that disposition from producer declarations for a
gate-passing delivery and is called by `delivery_run`; it never discovers files by pattern. Wait for the
owner's disposition before deleting anything under an open ownership review.
