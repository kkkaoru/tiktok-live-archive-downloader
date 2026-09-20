# Automatic Japanese captioning implementation plan

## Scope

Build an explicit, local-only, non-overwriting caption workflow for new videos,
not another recording-specific shell transcript. Preserve acquisition/session code,
existing Git changes, original media and earlier exports. No cloud audio/video
upload, automatic model download, login or desktop automation is required.

## Contracts and stages

1. Extend the existing Executor native speech adapter with opt-in recognizer-run
   timestamps and bounded contextual vocabulary. Use final results and installed
   Japanese assets; do not estimate word timestamps from character counts. Keep
   the old phrase API compatible. Untimed lexical text is an explicit failure,
   never a silent fallback to proportional timing.
2. Add typed Python caption/timeline models and boundary validation. Deduplicate
   overlapped recognition chunks by source-time ownership, not fuzzy global text
   deletion. Split at speech gaps and cut boundaries. A new utterance must not
   inherit old text. Reject ambiguous/unsupported timing and retain raw results.
3. Break Japanese punctuation at display-line boundaries; never emit a standalone
   punctuation cue, trailing blank line or isolated punctuation-prefix fragment.
   Bound each displayed phrase by the timed tokens actually spoken in that span.
4. Detect source subtitle rectangles using local Vision OCR over a configurable
   broad source search area. Derive padded spatial/time masks from detections;
   do not restrict discovery to the previous fixed 120-pixel band. Preserve
   evidence of empty/missing samples and expose sampling limits. Inspect the
   reported missed-caption point and other positions using this same logic.
5. Provide a reusable CLI with explicit source/output/configuration, optional
   clean voice and source/cut mapping, dictionary input, deterministic manifests,
   content-hash-bound resume, subprocess deadlines and interruption-safe output.
   Reuse local separation/encoding primitives via Executor; normal downloader
   commands do not acquire an editing/model/runtime dependency.
6. Run on the current replay, including the reported output time 01:03:39.
   Compare original/noisy versus separated-voice Japanese recognition on bounded
   samples. Measure timing/punctuation/ownership invariants, not invented speech
   accuracy scores. Keep ambiguous words uncorrected and report them.
7. Render a new full output from source, preserving the above-face caption
   placement, voice and earlier outputs. Gate publication on full decode, exact
   frame/sample duration, source hashes and local subtitle-region QA.

## Headless execution checkpoints

`uv run --locked python -m replay.workflow /absolute/workflow.json --state
/absolute/edit-run/state` executes a **trusted local command manifest**. The JSON
has `version: 1` and an ordered `stages` array. Each stage declares a unique `name`,
`command` argument array, absolute `inputs`/`outputs` file arrays and a positive
`timeout_seconds` (per child command, at most 86400). Optional `verify_command`
is a read-only, input-provenance verifier; `adopt_existing: true` explicitly
permits it to register outputs from a previously verified external worker.

Include adapter scripts and model/configuration files in `inputs`, not just their
paths in command arguments. Run from the same checkout: the working directory is
also fingerprinted. Plans are executable configuration, not untrusted downloaded
metadata. Child stdin is closed; logs and checkpoint directories are private.
Do not pass credential values in arguments or print them from stage commands.

Completed mutations are skipped only when input and output hashes still match.
Configured read-only verifiers also run on reuse: a manifest's unchanged hash
alone does not prove that its referenced artifacts still exist or are unchanged.
Process locks prevent duplicate stage execution. Interrupted mutations are not
blindly repeated: only an explicit read-only verifier can adopt their results.
Missing/changed completed outputs stop the workflow without overwriting anything.
Failures preserve intermediate artifacts and private attempt logs.

This checkpoint runner is implemented and tested. It does **not** by itself
create a complete editing plan, repair uncertain speech, establish exhaustive
subtitle concealment, or demonstrate Final Cut import fidelity. Those domain
adapters and their final-delivery checks remain separate acceptance requirements.

## Evidence and automatic review

- `caption_io.sound_windows` validates native classification results and preserves
  their scores. Classification remains at threshold 0.2; an empty result is not
  evidence of silence. Ordinary word-boundary protection remains 0.4 seconds.
- `waveform_quiet.measured_quiet_runs` checks complete, sample-exact stereo peak
  measurements. Quiet requires both channels below -60dBFS for at least 20ms;
  mono cancellation and ASR absence cannot authorize deletion.
- `caption_repair.repair_native_points` can replace a complete, identical-text
  block with an independently measured native-clock block. It never apportions
  a point across characters. Unresolved points remain explicit.
- `caption_review.select_native_context` can propose different native ASR text
  between shared, timed context anchors. It preserves donor words and clocks,
  requires safe seams and display limits, and can require unresolved lexical
  content to remain represented. A short, complete provider phrase can also be
  retained at its actual native phrase clock if text, neighbors and the unchanged
  display budget all fit; it is explicitly phrase-level, never interpolated into
  word clocks. Punctuation normalization is for anchor comparison, not spelling correction. Matching tolerances do not change cut
  thresholds or permit clock adjustment.
- Context proposals are not accuracy assertions. Keep original transcripts,
  alternatives, provider failures and ownership records. Protect original speech
  evidence independently of whichever transcription supplies display captions.
  Failed recognition is never promoted to silence or blindly retried. A bounded,
  wider-context measurement must use a fresh plan and output location.

The evidence parsers and selection policies are implemented and tested. The
single-entry editing-plan generator, final rendering/packaging and end-to-end
acceptance checks are still required; these modules alone do not complete them.

## Verification

- Synthetic tests for stale partials, chunk seams, speech restarts, repeated valid
  words, punctuation-only runs, Japanese/Latin punctuation, short fragments,
  overlapping/out-of-range tokens, cut projection and dynamic mask extents.
- Swift strict formatting/warnings-as-errors, per-file >=95% coverage, separate
  TSan/ASan, restored normal coverage and release build. Native Japanese TTS
  fixtures test the time-indexed API without uploading or playing audio.
- Python Ruff format/lint, strict mypy, focused and full pytest, per-production-file
  >=90% coverage, `make check`, `uv lock --check`, `uv build`.
- Actual Executor calls plus bounded real-media checks establish different evidence
  from synthetic tests. Sampled OCR is not proof of complete concealment or ASR
  correctness; report any residual subtitles or unsupported cases explicitly.
