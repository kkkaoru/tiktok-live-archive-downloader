---
name: repo-jobs
description: Manage this checkout's long-running jobs, local scratch, logs, status and safe recovery. Use before starting, checking, resuming or recovering processing/tests/builds. Reuse existing job and native-build scripts; inspect outcomes before retries and never interrupt unrelated agents or applications.
---

# Repository jobs and recovery

[AGENTS.md](../../../AGENTS.md) owns authorization, storage and completion rules.
Use [disk-maintenance](../disk-maintenance/SKILL.md) before large allocations,
[replay-apple-workflow](../replay-apple-workflow/SKILL.md) for processing contracts,
and the generic Apple app skills for actual app operations.

## Latest user instruction first

Before the next affected action, reconcile a direct user addition/correction using
[latest user requests](../../../docs/blocker-audit.md#latest-user-requests). The new
user instruction replaces conflicting old user preferences, plans and agent notes;
it does not replace actual higher-priority rules or unrelated requirements.
Use the version-2 request ledger with `--latest-request` pinned from the current
original message. Assign missing tasks instead of merely quoting the request in
a handoff. Keep blocked requests unfulfilled, retain explicit supersession and
verify the requested outcome. Reconcile existing jobs before any cancellation or
resubmission; a new instruction is not a reason to duplicate an uncertain job.

## Preflight and launch

1. Identify the job owner, requested stage, inputs, output namespace, existing
   process/job IDs, receipts and partial results. No status file or a timeout is
   not proof that a job never ran. Reconcile an uncertain outcome before retrying.
2. Resolve the checkout and a private, stable, purpose-named work directory under
   `edits/work/` (or, for large render/offload jobs, under the approved external root
   `/Volumes/m.2-ssd/movie/replay-work/`; `scripts/tmux-job.sh` accepts both); reuse it across tasks, retaining prior reports under descriptive
   non-overwriting names. Do not create random work roots or per-video scripts.
   Reject escaping symlinks, including tmp/cache/log destinations. Set child
   `TMPDIR`, `TMP`, `TEMP` and tool-specific cache/profile/output paths locally.
   Foundation may ignore these variables: inspect actual scratch paths and use
   explicit scratch-directory APIs before any writes. Reject tools that force
   external scratch; never substitute `/tmp` or another checkout.
3. Run from the repository root. Read the existing entry point before using it:

   | Need | Entry point |
   | --- | --- |
   | Read-only workflow dependency preflight | `python -m replay.workflow PLAN.json --state STATE_DIR --check` |
   | Verified stage execution/resume | `python -m replay.workflow PLAN.json --state STATE_DIR` |
   | Detached command, log and exit status | [scripts/tmux-job.sh](../../../scripts/tmux-job.sh) `WORK_DIR NAME COMMAND [ARGS...]` |
   | Shared native build/test | [scripts/native-swift.sh](../../../scripts/native-swift.sh) `build\|test [SwiftPM options]` |
   | Format/lint/type/tests and packaging | [docs/development.md](../../../docs/development.md#quality-gate) and the existing Makefile targets |

   The job helper requires an existing work directory and a unique name. It atomically
   reserves `<name>.launch`, writes exclusive private `<name>.log`/`<name>.status`,
   rejects linked destinations, isolates tmux configuration and uses a short relative
   socket with explicit Bash. A failed dispatch retains its reservation for diagnosis;
   do not remove it or change the name to blindly retry. Inspect the exact job first.
4. For multi-stage processing, use the existing workflow manifest, not shell polling
   or a new runner. Declare every producer output and consumer input; graph preflight
   rejects multiple writers, cycles, overlapping file trees and linked outputs before
   any command runs and orders producers before consumers. Keep artifacts/manifests
   outside the state directory. `--check` prints stage names/dependencies without
   running commands, writing state or proving media readiness. For an independent
   branch, repeat `--target NAME` to select only those stages and all their ancestors;
   check that selection before executing it. Unknown targets and invalid unselected
   graph branches still fail. Selected external inputs must be nonempty regular files
   before any command starts. A targeted result is not a complete delivery.
   Use the same state directory for a job: `.workflow.lock` prevents interleaving
   workflow invocations there, not writers using another state directory or hidden
   command side effects. Do not delete lock files to force a retry. Existing receipt
   hashes and domain verifiers still govern reuse. Bind helper/model/config files as
   inputs and supply actual domain verifiers: graph validity is not quality acceptance.
   The shell helper still has no implicit timeout or cross-plan resource scheduler.
   `dispatched` is not started/complete. For its socket, inspect from the work directory
   with `tmux -S tmux.sock ...`; verify ownership before any control action.
5. Give uncertain/long commands a realistic runtime bound and use detached
   execution. Monitoring windows must be bounded too. Prefer existing completion
   notifications or scheduled checks over tight polling. Do not launch heavy
   duplicate builds or workers while another agent owns their inputs/output state.

## Observe and resume

- Read bounded log/status/receipt slices for the exact job. Do not expose secret
  environment values, signed URLs or unrelated process arguments in reports.
- Distinguish running, timed out, failed, interrupted and verified complete.
  Exit zero proves command success, not whole-media quality or delivery acceptance.
- Reuse outputs only when input, implementation and model identities and completion
  records match. Missing/changed completed output is a stop, not a retry candidate.
- Reconnect verified stages to downstream processing. Keep blocked dependencies
  local; continue independent authorized implementation, synthetic tests or
  evidence checks. Do not pass unverified media downstream as approved.
- Record concrete changes, remaining work, expected next check and recovery steps
  in the handoff. Preserve evidence needed for restart; cleanup uses disk-maintenance.

## Before a whole-goal stop

Use the [blocker audit](../../../docs/blocker-audit.md) before reporting the whole
goal blocked or ending its loop for input. Preserve the original author and entry
reference, affected task, attempted checks, next action and recheck condition.
An agent-authored wakeup delivered with a user role is still an agent note, not
new human authorization or prohibition. A helper's "prepare and return to main"
limit requires main to review the returned evidence, not automatically wait for
human permission or automatically run the deferred operation.

Run `python -m replay.blocker_audit SNAPSHOT.json --require-blocked` against the
complete current task graph in private repository-local storage; after user
additions use version 2 and `--latest-request SOURCE_REFERENCE` as above. Exit 1 means
source review, local investigation, independent work, running-job observation or
completion verification remains; exit 2 means no valid decision was established.
Only exit 0 permits a **recommendation** to report blocked on the declared facts.
Do not turn it into an execution permit. Retain the snapshot/result; on another
continuation, check the recorded condition rather than repeat an untraced ban or
relaunch the same job. Real external permissions remain external dependencies.
This CLI checks declared structure, not the truth/completeness of evidence, and
does not hook external goal/loop tools. See the guide for schema and limits.

## Task-related application recovery

Application restarts within the task are pre-authorized by AGENTS.md, including
force termination after a failed normal quit or an unresponsive app. This does
not authorize interfering with another agent's work, unrelated apps or the OS.

Verify the exact target/PID and its state, save recoverable work where possible,
then use the generic app skill's scoped quit/launch procedure. Unsaved settings or
diagnostic jobs may be lost. Independently confirm termination, a new responsive
instance and readiness; inspect old job IDs and outputs before continuing.
Never blindly repeat restarts or resubmit an uncertain job. Restart authorization
grants no permission to delete saved data, approve OS prompts, log in, purchase
or bypass challenges. Record the result and resume authorized work without asking
for the same approval again.

Do not copy the existing runners into this skill or create another daemon/runner
for orchestration alone. Keep script fixes in the shared entry point with tests;
only the documented mechanical checks are enforced; arbitrary child commands still
require trusted manifests, explicit scratch APIs and appropriate authorization.
