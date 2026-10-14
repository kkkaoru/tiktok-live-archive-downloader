# Whole-goal blocker audit

Use this read-only check when the user adds or corrects instructions, and before
a whole-goal `blocked` report or an input-wait loop termination. It complements workflow execution preflight; it never executes
commands, reads citation targets, changes goals, signals jobs or grants permission.
Run from the checkout, with the repository-local environment described in
[repo-jobs](../.agents/skills/repo-jobs/SKILL.md).

```sh
uv run --locked python -m replay.blocker_audit edits/work/task-purpose/blockers.json --require-blocked
```

Keep the snapshot and stdout privately under the task's existing `edits/work/`
root; do not include credentials or raw account/session transcripts. Resolve all
input/output parent directories inside the checkout before use. The CLI rejects
symlink leaves, nonregular input, oversized input, duplicate JSON keys, malformed
snapshots, inconsistent running/completed dependencies and dependency cycles.
It reads at most 1 MiB plus one byte and never follows the snapshot's citations.
Parent-directory confinement remains the caller's responsibility.

## Trace the restriction before classification

1. Find the original instruction or observed failure, not just a handoff/summary.
   Record its author and immutable entry/reference in `source`. A self-authored
   wakeup delivered with a user message role does **not** become a human instruction.
   If the origin cannot be established, use `source_origin: "unverified"`.
2. Distinguish an authoritative restriction from an unknown prerequisite and a
   failed operation. An unverified setting is not proof that human permission is
   required or that a safe configuration exists. Determine a bounded next check.
3. Preserve scope. A helper's "prepare the conditions and return to main" limit
   is not a whole-goal prohibition. Main must review the returned evidence and
   actual governing instructions before scheduling anything. Neither automatic
   execution nor automatic human approval is implied.
4. Describe all remaining tasks, including independent work and running jobs.
   Cite actual observations in `evidence`/`attempts`; give `next_action` and
   `recheck_when` for every blocked task. Do not label an unknown local question
   `external` merely to stop the goal. Authentication/OS permissions and explicit
   human stops must not be overridden.
5. On a later continuation, inspect the stated recheck condition and existing job
   before revising the snapshot. A timeout is not completion; unchanged evidence
   is not a new successful check. Never fabricate a check to fill these fields.

## Version-1 snapshot

This **synthetic** example demonstrates an inherited helper restriction alongside
independent work; it is not an authorization or an actual media-processing plan.

```json
{
  "version": 1,
  "task_reference": "original authorized outcome, message/reference",
  "tasks": [
    {
      "name": "encode",
      "needs": [],
      "state": "blocked",
      "evidence": "helper returned a proposed bounded execution plan",
      "blocker": {
        "kind": "restriction",
        "authority": "agent",
        "source_origin": "agent-note",
        "scope": "other-task",
        "source": "original helper assignment entry; prepare only and return to main",
        "resolution": "local",
        "attempts": "read assignment and helper result; parent authorization not yet reconciled",
        "next_action": "main compares the original request and resource checks with the returned plan",
        "recheck_when": "the original scope and required safety evidence have been checked"
      }
    },
    {
      "name": "synthetic-integration",
      "needs": [],
      "state": "pending",
      "evidence": "independent authorized synthetic integration test"
    },
    {
      "name": "delivery",
      "needs": ["encode", "synthetic-integration"],
      "state": "pending",
      "evidence": "not yet delivered"
    }
  ]
}
```

Required fields are exact; unknown/missing fields fail. There are 1–512 tasks,
unique names matching `[a-z0-9][a-z0-9_.-]{0,63}`, and named dependencies without
duplicates or cycles. Text fields are nonempty, at most 4,096 characters, without
NUL. `version` must be integer 1, not a JSON boolean.

- Task `state`: `pending`, `running`, `complete`, `blocked`. Only blocked tasks
  carry `blocker`; running/complete tasks need all dependencies complete.
- Blocker `kind`: `restriction`, `prerequisite`, `failure`.
- `authority`: `system`, `developer`, `user`, `agent`, `observation`. Record the
  actual author/authority, not the transport role of a quoted/generated message.
- `source_origin`: `original` means the caller inspected the originating source;
  `agent-note` means an assistant-authored summary/delegation/wakeup;
  `unverified` means provenance is unresolved. This declaration is not verified
  automatically. Quoting a claimed user prohibition inside a note does not make
  that note `original`.
- `scope`: `current-task`, `other-task`, `unverified`.
- `resolution`: `local` for a safe, authorized investigation/review still possible;
  `external` only for an established dependency outside that work. The next action
  for a restriction must respect it (for example, review scope), never bypass it.
- `source`, `attempts`, `next_action`, `recheck_when`: required evidence text.
  Use an actual condition or evidence event, not a fabricated retry deadline.

## Routing and exit status

The JSON result contains task names only, not the evidence text:

- `review_restriction_sources`: noncurrent/unknown scope, or a restriction without
  original authoritative provenance. Review first; these are **not** runnable tasks.
- `investigate_locally`: declared local checks with completed prerequisites.
- `candidate_tasks`: pending tasks with completed prerequisites, not execution permits.
- `observe_existing`: running jobs. Inspect them, do not duplicate them.
- `deferred`: tasks waiting on dependencies or established external conditions.

`decision` is `continue` if review/investigation/candidates exist, otherwise
`observe-existing` for running jobs, `verify-completion` if all tasks are declared
complete, or `blocked` if only deferred work remains.
`execution_authorized` is **always false**. Complete declarations still require
independent completion verification.

With `--require-blocked`, exit 0 means the declared graph supports only a blocked
recommendation; exit 1 means another route remains; exit 2 means invalid input.
Without that flag, any valid classification exits 0, so ordinary CLI success
must not be used as a blocked gate.

## Latest user requests

**A new direct user instruction takes precedence over an older user instruction,
plan, assistant note or monitoring preference wherever they conflict.** Preserve
unchanged requirements. This is not an exception to actual higher-priority rules.
Do not classify an assistant's interpretation as such a rule.

Apply a correction **before the next affected action**, not only when a goal stops:

1. Read the latest original message and the immediately relevant context. Split
   separate requests into separate outcomes. A request to delete an old template
   is not a request to redesign captions; preserving style does not discharge deletion.
2. Resolve the referenced target from the already shown artifact, path, UID or
   prior selection. Do not ask for the same target/approval again if it is known.
   If multiple targets remain plausible, perform safe local inspection first and
   ask only for the unresolved distinction; never guess a destructive target.
3. Update the affected plan and acceptance conditions. An explicit later user
   reversal supersedes an earlier user preference, not unrelated requirements or
   higher-priority constraints. Repeated corrections normally refer to the same
   unmet outcome: do not apply a numeric change repeatedly just because it was
   repeated, and do not duplicate an existing job.
4. Reconcile affected running jobs and artifacts before control actions. A stale
   plan is not permission to continue an affected action, but supersession is
   not permission to blindly cancel another agent's job or delete its inputs.
5. Bind each active request to real tasks and independently verify its requested
   outcome. Quoting a request in a checklist is not implementation. "Not used"
   is not "deleted"; a diagnostic render is not an adopted style correction.
   Report `requested → actual result → unmet part → next action`.
6. If a real higher-priority rule prevents exactly that operation, cite its actual
   source and scope. Keep the user's request **unfulfilled**, state what governing
   condition would have to change, and continue independent authorized work.
   Repeating user consent cannot itself change a higher-priority restriction;
   do not present another consent request or an agent-edited note as its solution.

### Version-2 request ledger

Version 1 remains unchanged for legacy blocker-only callers. After an instruction
change, use version 2 with `requests` alongside `version`, `task_reference`, `tasks`.
Use **`--latest-request SOURCE_REFERENCE`** from the actual current direct user
message, not from the old ledger. A mismatch or version-1 ledger exits 2. This
checks the caller's supplied reference; it does not read live conversation history.

```sh
uv run --locked python -m replay.blocker_audit edits/work/task-purpose/requests.json \
  --latest-request message-latest --require-blocked
```

This synthetic ledger deliberately has an unassigned deletion request. A completed
disuse task cannot make that separate missing action disappear:

```json
{
  "version": 2,
  "task_reference": "original authorized task",
  "tasks": [
    {"name": "disuse", "needs": [], "state": "complete", "evidence": "old template no longer selected"}
  ],
  "requests": [
    {
      "name": "remove-template",
      "source": "message-latest",
      "origin": "direct-user",
      "instruction": "Delete the old template",
      "target": "the exact old template identified in the preceding context",
      "acceptance": "the approved target is absent after verified removal",
      "tasks": [],
      "supersedes": []
    }
  ]
}
```

The result is `continue`, `unassigned_requests: ["remove-template"]`, exit 1 with
`--require-blocked`, and `execution_authorized: false`. This does not authorize deletion.

Request entries are chronological, nonempty (at most 512), with unique `name`
identifiers and exactly the eight fields above. All four free-text fields are
nonempty and bounded as in version 1; do not put secrets in them. `origin` must be
`direct-user`: do not reclassify a monitor's paraphrase or assistant note as an
original direct instruction. Separate instructions in one message may share a
source reference. Active requests' task references must exist and be unique; an
empty binding is valid **only as an explicit assignment gap**, never fulfillment.
Superseded requests may retain historical task references no longer in the current
graph, so retiring an obsolete plan does not require executing it or falsifying
completion. Preserve the prior snapshot and inspect any affected existing job
before retiring its task from the current graph. Version 2 allows
zero tasks so that a wholly missing plan can be honestly recorded.

`supersedes` is an explicit, nonduplicated list of earlier request names. Only
conflicting same-level user intents belong here. Unknown, self and forward
references fail; omission means the earlier request remains active. The tool
does not infer natural-language conflicts or weaken task-level safety blockers.
A task's `user` restriction that cites the source of an explicitly superseded
request is routed to `review_restriction_sources`, not accepted as a whole-goal
stop. Matching source references requires reconciliation, not automatic execution;
this rule never supersedes a `system` or `developer` restriction.

Additional output fields (identifiers only):

- `unassigned_requests`: active requests with no task binding; forces `continue`.
- `unfulfilled_requests`: active bound requests with incomplete tasks; a genuine
  blocker can remain `blocked`, but the unmet user intent is still explicit.
- `requests_to_verify`: all bound tasks declared complete; actual acceptance
  still requires independent evidence, never automatic fulfillment.
- `superseded_requests`: explicitly replaced intents; preserved in the ledger.
- `review_superseded_tasks`: tasks referenced only by replaced intents; forces
  plan review and removes them from `candidate_tasks`. It does not cancel jobs,
  discard results or prove those tasks are unnecessary. A dependency still needed
  by active work needs reconciliation rather than blind removal.

Before execution/completion, the caller must verify that bindings and evidence
actually implement the request's target and acceptance condition. The checker
cannot detect a dishonest binding of a deletion request to an unrelated completed
disuse task. It also cannot detect an omitted instruction when the caller supplies
an equally stale latest-message reference. Those are explicit enforcement limits,
not permissions to omit the check or claim a fulfilled outcome.

## Enforcement limits

The parser, graph validation and exit-code routing are mechanical. Truth and
completeness of citations, author attribution, safe local checks and the task
inventory are not. The checker retains no hidden cross-run state and cannot tell
whether repeated text is a meaningful new observation. It does not verify that
an app's scratch is confined, approve media quality, resume paused work, or remove
an actual prohibition. An agent can bypass this advisory CLI; repository routing
instructions do **not** intercept the external `update_goal`/`loop_complete` APIs.
Do not claim harness-level enforcement without a separately authorized integration.

Tests use synthetic snapshots only. Private incident records belong under
`edits/work/`, not in public source or test fixtures.
