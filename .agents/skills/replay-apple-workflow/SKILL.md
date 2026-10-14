---
name: replay-apple-workflow
description: Coordinate this repository's TikTok/YouTube replay processing with Motion, Compressor and Final Cut Pro. Use for replay captions, native rendering, resumable jobs, required-stage validation, editable projects and Viewer delivery. Delegate direct app operations to the generic apple-pro-apps skills.
---

# Replay processing with Apple Pro apps

## Ownership and entry point

This skill owns **repository-specific orchestration and acceptance**, not Apple
app APIs. Read [AGENTS.md](../../../AGENTS.md), [README.md](../../../README.md),
[SECURITY.md](../../../SECURITY.md) and [architecture](../../../docs/architecture.md).
Use [development](../../../docs/development.md) for checks and
[operations](../../../docs/operations.md) for troubleshooting.

For direct app operations, load [apple-pro-apps](../apple-pro-apps/SKILL.md)
and the relevant sibling: [Motion](../apple-motion/SKILL.md),
[Compressor](../apple-compressor/SKILL.md),
[Final Cut Pro](../apple-final-cut-pro/SKILL.md),
[Logic Pro](../apple-logic-pro/SKILL.md) or
[MainStage](../apple-mainstage/SKILL.md).
Those skills own discovery, schemas, app/edition selection, interchange, UI/MIDI
side effects and operation-level verification. They do not own replay policy.
Neither layer grants permissions or treats historical approvals as current consent.

For video creation/editing, also follow the installed `hyperframes` entry skill;
retain an explicitly selected Apple-native workflow rather than silently changing
the deliverable's framework. A routing skill is not a rendering backend.

## Repository contracts

- Keep private work/logs/scratch in `edits/work/`, models in `edits/models/`, and
  recording edit configuration in `edits/replay-<id>/edit-config.json`. Resolve
  paths, reject escaping symlinks and verify actual child-process scratch paths.
  Do not reuse a home-level Movies test root or another checkout. Storage and
  cleanup authorization, free-space checks and protected data follow AGENTS.md.
- Keep Python processing in `src/replay/` and native processing in
  `native/apple-pro-apps/`; read its [README](../../../native/apple-pro-apps/README.md).
  Historical snapshots/work directories are evidence, not implementation roots.
  Never introduce replay-ID/date-specific processing scripts or fallback backends.
- Use the selected non-turbo MLX ASR and verified hashes from
  [README — local models](../../../README.md#local-models). Do not substitute generic
  MCP transcription, Core ML or turbo, or auto-download during inference. Explicit
  alternate paths need matching verified hashes. WhisperKit is a separate native
  test backend requiring compiled Core ML, not MLX weights. Record origin, revision,
  full manifest, expected/actual sizes and hashes, local path and test outcomes in
  `edits/models/` or `edits/work/`; validate each LFS SHA-256 or Git blob SHA-1 before
  use. Preserve old models, recognition and partial/failed transfers; do not call
  them installed. Set `WHISPER_TEST_MODEL`, `WHISPER_TEST_TOKENIZER` and
  `DEMUCS_TEST_MODEL` only to verified directories in the test child environment.
  Limited tokenizer/sanitizer success does not pass the full native gate.
- Caption constants live in `src/replay/caption_style_defaults.py` (style 20), not
  copied palettes in skills. Follow [caption finishing](../../../docs/development.md#layered-motion-caption-finishing):
  preserve the white outline, inner/outer black rims and shadow; scale SVG reference
  geometry proportionally rather than treating it as raw Motion units. Do not
  equate sRGB with Motion controls or claim AAA/calibration from adopted defaults.
  Preserve existing projects/evidence and independently verify the actual render.
- Native face tracking must use `VNDetectFaceRectanglesRequest` →
  `VNTrackObjectRequest` → `VNSequenceRequestHandler`. Supply validated media paths,
  clocks, model/tool fingerprints and evidence-derived policy through configuration;
  missing/low-confidence observations do not authorize invented positions.
- Measure representative end-to-end throughput before full execution. Reuse verified
  results and frame-local detections, prefer bounded batching/persistent native
  workers over per-frame launches/decodes and avoid unnecessary intermediates.
  For required edits prefer verified QuickTime-compatible hardware HEVC when
  supported; independently check clocks, frames, streams, decode and file size.
  Never trade coverage, quality, provenance, resource bounds or resumability for speed.
- Before a new large job, check disk headroom and existing jobs, outputs, receipts
  and dependencies. Reuse hash-matching results; never erase completion/partial
  records or resubmit an uncertain job to force progress. Isolate a blocked stage
  and continue independent authorized work without interrupting another agent.

## Per-video optimization through shared processing

Minimize video-specific logic, not video-specific measurement. Keep one shared
processing path; adapt its parameters to each video's validated characteristics.

- Separate **facts → policy → execution**: measure media clocks, resolution,
  speech/noise, observed faces/source text, caption geometry and throughput as
  needed; let reusable policy derive cuts, masks, layout, batching and encoding
  parameters. Use only observations actually available; missing/low-confidence
  evidence must remain unresolved, not become invented positions or timings.
- Per-video configuration may contain input references, verified measurements,
  explicit editing intent and derived parameters/results. It must not become a
  second implementation or a collection of patches keyed by recording ID, date,
  exact duration or frame count. Selecting a reusable strategy by measured
  characteristics is allowed; selecting it by a video's identity is not.
- Record the evidence, policy/tool/model versions and reasons behind derived
  choices so results can be reproduced and safely reused. Do not add unexplained
  per-video constants or overrides to conceal a shared defect or pass a gate.
- Fix failures in the common implementation. Test contrasting synthetic inputs
  and relevant boundaries, not only the failing video's shape, then reconnect
  processing and re-run only stages invalidated by the change.
- Preserve required stages, quality, provenance and storage bounds for every
  video. Automatic adaptation is not permission to lower acceptance criteria.

## Workflow

When the user adds or corrects a requirement, apply the
[latest-request procedure](../../../docs/blocker-audit.md#latest-user-requests)
before the next affected step. Update conflicting old plans/agent notes, preserve
unchanged style and delivery requirements, and bind each separate requested
outcome to tasks and actual acceptance evidence. Template deletion, template
selection and visible style correction are different outcomes; a diagnostic
project or disuse note is not completion of the other requests.

1. Record required stages, inputs/outputs, clocks, fingerprints and acceptance
   conditions. Prefer one full-length delivery first. Run a representative short
   section through the selected real caption/audio/render/playback path early;
   that is integration evidence, not full-length completion.
2. Prepare inputs through the shared replay modules. Preserve the source-frame
   and audio-sample mappings, protected cuts, source-caption mask evidence, ASR
   provenance and measured caption placement. Missing observations do not permit
   invented positions or timestamps.
3. Build Motion/FCP artifacts through the shared implementation. For unavailable
   non-UI operations, establish the missing contract, implement/test a reusable
   bounded adapter within authorized scope, and reconnect it to processing.
   Do not repeat UI configuration or substitute a historical source snapshot.
4. Delegate actual app operations to the generic skills. Pin the selected installed
   edition, template/preset, submitted inputs, job IDs and returned output paths.
   Separate Motion documents, real Motion renders, AVFoundation composites and
   editable FCP projects; none proves the others.
5. Verify decoded media, clocks, audio, captions, masks and actual FCP import as
   distinct outcomes. Do not silently replace requested editable caption titles
   with flattened video. Test bounded import/memory/editing performance before a
   whole large timeline; fewer title instances alone do not establish performance.
6. Pass `python -m replay.edit_completion check` for the recording using the
   configured environment. Music removal, non-speech cuts, source-caption masks,
   transcription and burned dynamic captions remain mandatory. Do not publish or
   call a delivery finished with a skipped/degraded stage or missing quality proof.
7. Complete requested Viewer playback/registration and summaries separately, using
   existing shared processing. Report completed videos/requested videos, intermediate
   progress, real blockers and the next action. Preserve resumable evidence and
   clean up only confirmed unnecessary intermediates under AGENTS.md.

## Harness delegation

Use [repo-jobs](../repo-jobs/SKILL.md) for launch, local scratch, monitoring, native
builds and recovery; use [disk-maintenance](../disk-maintenance/SKILL.md) for
capacity planning and approved cleanup. Processing/resume stays in the shared
`src/replay/` modules. Verification commands and gates stay in
[development](../../../docs/development.md#quality-gate) and the native README.
Do not add alternate runners/renderers or copy the shared scripts into this skill.

## Project acceptance and historical observations

For the established video-generation change checks, **both 60-second and
90-second cases are required**: at 30fps verify all 1800/2700 frames, whole audio,
caption/mask/outline samples and transitions, SE onset/control windows and source
preservation. Respect each tool's duration/frame bounds; longer ASR inputs require
documented chunks and boundary review. Neither sample replaces whole-delivery
validation. This is project acceptance, not a universal Apple-app requirement.

The old app skills recorded the following narrow host observations. Retain them
as troubleshooting context, **not current readiness, permission or sign-off**:

- An Executor prototype had editing/measurement and sanitizer checks, with native
  boundary exceptions documented by its provider. Locate the deployed provider's
  verification/boundary reports; do not assume the old dotfiles checkout exists.
- All five Creator Studio apps plus standalone MainStage, and Screen Recording,
  Accessibility and Event Synthesizing permissions, had been observed. Rediscover
  installed editions and permissions after host/session changes.
- An unchanged Snap Lower Third `.motn` rendered via Compressor to 180 H.264 frames
  at about 29.97003fps / 6.006s; this did not prove arbitrary Motion compatibility.
  An `ap4h.setting` alpha attempt failed with `className is null` and needs separate
  preset validation, not blind retry.
- Compressor 5.3 CLI help exposed submission/monitor/control options; a job stalled
  at 0% until its app ran. Check current capabilities and existing job/controller
  state before retrying.
- Foreground/clipboard FCP operations had task-scoped approval on 2026-10-08. That
  approval is not transferable; current app actions must remain within the user's
  authorized task. Recovery follows AGENTS.md and the generic app procedure.
