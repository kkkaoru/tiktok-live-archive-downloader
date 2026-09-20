---
name: apple-compressor
description: >-
  Control Apple Compressor through Executor's Swift MCP using its official CLI: source analysis, job submission, bounded status and per-job/batch pause, resume or cancel. Prefer native batch encoding over GUI automation.
---

# Compressor — official CLI through typed MCP

Load **apple-pro-apps** and discover tools in Executor namespace `apple-pro-apps`.
Do not run arbitrary shell command strings. Swift constructs argument arrays for
the installed, explicitly selected Compressor executable. This Mac's default is
`compressor` / `com.apple.CompressorApp` (Creator Studio 5.3).

## Procedure

1. Inspect the requested source, desired codec, resolution/rate, color/HDR tags,
   audio channels, range and output destination. Use `media_inspect` for scalar
   AVFoundation metadata and first-frame decoding. `compressor_inspect` invokes
   official `-checkstream`, but empty output is not successful media analysis.
2. Obtain a **trusted, existing** `.cmprstng` preset or an Apple-bundled `.setting`. Presets/actions can carry
   behavior beyond encoding; never use an unreviewed downloaded preset. This MCP
   does not author arbitrary preset internals or modify shared-computer settings.
3. Describe `compressor_submit`, then supply `sourcePath`, `presetPath`,
   `outputDirectory`, `outputName`, `batchName` and optional installed `bundleID`.
   All paths are absolute; source/preset must exist, output parent must exist.
   The server reserves a unique private `compressor-<UUID>` child directory and
   returns its output path. Existing exports and sources are never overwritten.
   For a short smoke test, pass `range: {startSeconds: 0, durationSeconds: 5}`.
   This user's verification output root is `~/Movies/Apple-Pro-Apps-Verification/`.
   Resolve it to an absolute path and ensure it exists before supplying
   `outputDirectory`; never pass literal `~` to the MCP. Keep diagnostic logs
   private and separate from the viewing copy.
   Start is bounded to 0–86399 seconds; duration to 1–600 seconds. These become
   source `-in`/`-out` timecodes before the output target; frame rounding belongs
   to Compressor. Verify the exported duration, do not assume trimming succeeded.
4. Save the returned native submission result and exact job/batch ID. Exit zero
   is **not** encode completion. The response reports `completionVerified: false`.
   If submission fails/times out, retain its reserved output path and investigate
   before any retry; the native application may already have accepted the job.
5. Describe/call `compressor_status` with that ID, `job: true` for a job or false
   for a batch. It uses `-monitor -once` and bounded timeouts. Schedule another
   check only when useful, not a tight polling loop. Never invent a job ID.
6. Use `compressor_control` only for an explicitly authorized `pause`, `resume`
   or `cancel` of the exact ID. Cancellation can lose processing work. There is
   no cancel-all/reset/repair/network-sharing tool.
7. Confirm native completion status and the actual output file's properties before
   reporting success. An empty/partial file or merely returned ID is insufficient.
   For outputs up to 30 seconds, `media_verify_video` decodes every video frame
   (maximum 1800), `audio_measure` checks normalized mono PCM levels, and
   `video_frame_measure` samples region RGB. These complement native job success;
   they do not prove every effect, every channel or HDR fidelity.

For trim/reorder/speed/geometry, gain/fades/mix, SDR color or static-title work,
`media_edit` is a separate native AVFoundation renderer with reusable JSON recipes.
It is not a Compressor preset editor or Motion renderer. Prefer it when its exact
schema covers the requested edit; use Compressor for the chosen trusted encoding
preset and subsequent job management.

The installed 5.3 CLI was inspected with `-help`; it supports submission JSON,
`-monitor`, `-once`, `-timeout`, `-pause`, `-resume` and `-kill` by job/batch ID.
These are version-specific observations; recheck capability on another version.
The current typed submission covers one regular source and one preset; repeat
explicitly for a bounded set of sources. Image-sequence URL parameters, multi-
computer distribution, relabel-in-place and job-action XML are not exposed.

For creating/editing presets, configuring destinations or other UI-only features,
use `apple-pro-apps-ui` only after identifying that gap and following shared safety
rules. Native encoding of exported media is preferred to repeated GUI workflows.
Do not assume the CLI can render arbitrary Motion timelines without Motion.

## Apple references

- https://support.apple.com/guide/compressor/cpsr9be73312/mac — submission syntax
- https://support.apple.com/guide/compressor/cpsr9be734f8/mac — command options
- https://support.apple.com/guide/compressor/cpsr9be73659/mac — examples
- https://support.apple.com/guide/compressor/welcome/mac
