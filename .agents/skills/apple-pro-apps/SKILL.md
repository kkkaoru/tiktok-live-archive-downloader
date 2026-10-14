---
name: apple-pro-apps
description: Operate Motion, Compressor, Final Cut Pro, Logic Pro and MainStage through Executor. Prefer the custom Swift native MCP (CLI, XML interchange, LaunchServices, CoreMIDI, OSC, background Accessibility UI); use the separate Peekaboo MCP only as a last-resort auxiliary.
---

# Apple Pro Apps — machine integration first

Use **Executor** for all app operations. The custom Swift `apple-pro-apps` MCP is
primary. `apple-pro-apps-ui` is a separate, optional Peekaboo MCP using the existing
Homebrew installation. Do not silently substitute shell UI scripts, another MCP
client, a hosted model or an autonomous Peekaboo agent.

## Scope and verification

This is a reusable app-operation skill, not a project's processing or delivery
policy. Obtain paths, acceptance criteria, media-retention rules and authorization
from the current task/project; do not infer them from another installation.
Project-specific workflow skills may call this skill, but this skill does not
require a particular repository, media model, caption style or output directory.

Check the deployed MCP's verification report and documented native boundaries
before real-project use. Historical test results or boundary exceptions are not
current approval. Recheck Screen Recording, Accessibility and Event Synthesizing
through the actual host after session/host changes. Permissions, registration and
unit tests do not prove app-specific action success.

## Discovery and authorization

1. Load the matching sibling skill through the harness's skill loader or read its
   `SKILL.md`: `apple-motion`, `apple-compressor`, `apple-final-cut-pro`,
   `apple-logic-pro`, `apple-mainstage`. If using Executor's `local-skills` loader,
   discover its tools first. Skills describe procedures, not authorization.
2. `executor tools search app_capabilities --namespace apple-pro-apps --limit 2`;
   describe its exact returned path, then call the described schema. This reports
   installed editions and **actual limitations**, without launching apps.
3. Search for the required native tool, describe it, then call its exact path and
   arguments. Do not invent a tool because an Apple feature exists. Keep outputs
   bounded. Check both Executor's envelope and MCP `isError` / structured result:
   exit zero can still mean an approval pause or a tool failure.
4. Show an approval request and stop if paused. Never blindly resume, install an
   allow-all policy, manipulate TCC databases or accept OS/license/subscription
   prompts. Setup's registration flag covers only adding this server/connection.

## Native operations in order of preference

| Need | Native MCP tools | Boundary |
|---|---|---|
| Edit video/audio offline | `media_edit_plan`, `media_edit`, `media_project_read` | Trim/reorder/speed, geometry, gain/fades/mix, MP4/M4A and reusable JSON; not live editor control |
| Cut + blur + positioned captions + encode in ONE pass | `media_edit` with `video.encoding`, `styledCaptions`, `captionAppearance` | Video-only (like `ffmpeg -an`); exact frame count; hardware H.264/HEVC, BT.709, fast start |
| Render several parts at once | `media_edit_batch` | 1–32 requests, concurrency 1–4 (default 3); per-item results, no automatic retry |
| Prepare an approved Speech locale for this app | `speech_locale_reserve` | Persistent mutation, macOS 26+; no download/release/eviction. Require user approval and `readyForTranscription:true` |
| Transcribe a local audio file | `audio_transcribe` | ≤60s, prepared locale; read-only, on-device, approximate phrases, unreviewed; no implicit reservation/download/cloud fallback |
| Decode output fully with a bounded budget | `media_verify_video` | Defaults to 30s/1800 frames; explicit maximumDurationSeconds ≤120 and maximumFrames ≤7200; audio is separate |
| Generate caption-onset SE | `audio_cue_track` | New private WAV, ≤120s/120 cues, 80ms smooth 880Hz sounds; no playback; mix as one audio layer with headroom |
| Locate existing visible subtitles | `video_text_recognize` | Local Japanese/English Vision OCR, 1–8 frames, cropped/full-frame top-left bounds; no images uploaded/exported; OCR confidence is not speech accuracy |
| Measure audio/selected image regions | `audio_measure`, `video_frame_measure` | Mono PCM RMS/peak/zero crossings and device-RGB averages; not LUFS, robust pitch or all-pixel proof |
| Verify a local video without opening an editor | `media_inspect` | AVFoundation scalar metadata + first decoded frame, bounded child process; not full-file or app import validation |
| Installed editions/capabilities | `app_capabilities` | Does not launch apps or prove license activation |
| Open/import project, FCPXML or MIDI | `app_open_document` | LaunchServices/Open Document, not GUI clicking; acceptance is not import completion |
| Validate FCPXML structure against Apple DTD | `fcpxml_validate` | Uses the selected installed edition's self-contained DTD; not media-reference or import validation |
| Inspect/query interchange | `interchange_inspect`, `interchange_query` | Local UTF-8 XML only; bounded fragments; **not DTD validation** |
| Generate/copy/change interchange | `interchange_write`, `interchange_patch` | New files only; patch existing unique leaf/attribute values; no original overwrite |
| Encode media | `compressor_inspect`, `compressor_submit`, `compressor_status`, `compressor_control` | Official installed CLI; explicit source/preset and job/batch ID |
| Create musical note arrangement | `midi_file_create` | Standard type-0 MIDI file; no playback; import into Logic separately |
| Assigned live controls/patch changes | `midi_destinations`, `midi_send` | Exact destination ID AND name; CC/program/pitch bend only; routing must already be verified |
| Logic OSC assignment | `osc_send` | Exact path and explicit loopback UDP port; no receiver acknowledgment |
| Observe windows, dialogs, sheets and UI trees | `ui_windows`, `ui_inspect`, `ui_menu_list` | Background Accessibility; bounded trees with child-index paths; no activation |
| Press, select, set text, run menu items | `ui_perform`, `ui_set_value`, `ui_menu_select` | Exactly one located element; read back; text is written as AXValue, so input methods do not matter |
| Wait for UI state | `ui_wait` | Polls ≤30s; use instead of sleeping in Executor code (`setTimeout` is undefined there) |
| Capture one window | `ui_capture` | ScreenCaptureKit, background/occluded windows, NEW private PNG; needs Screen Recording |
| Live Final Cut Pro timeline: read, select, open project, move/seek playhead, blade/delete, Share export | `fcp_timeline_read`, `fcp_timeline_select`, `fcp_project_open`, `fcp_playhead_move`, `fcp_playhead_seek`, `fcp_timeline_edit`, `fcp_export` | Background except `fcp_export`; frame-exact seek without keyboard; no undo; see `apple-final-cut-pro` |
| Final Cut Pro effects, inspector parameters, keyframes, FCPXML export, library close | `fcp_effect_catalog`, `fcp_effect_parameters`, `fcp_inspector_read`, `fcp_inspector_set`, `fcp_effects_paste`, `fcp_xml_export`, `fcp_library_close` | Pass `project` to timeline tools (guard); merge paste by default, overwrites the clipboard; open/paste/export/close briefly activate FCP |
| Launch / quit an exact edition | `app_launch`, `app_quit` | Background launch by default; force quit needs `discardUnsavedChanges: true` |
| Keyboard input source | `input_source_status`, `input_source_select` | Global user state; always restore `previousInputSource` |

Offline editing creates a private new output directory and `edit-request.json`.
Every video render also writes `<outputName>.fcpxml` beside the media (clips on the
spine, captions/titles as Basic Title connected clips, masks as markers) and
DTD-validates it against the installed Final Cut Pro; check `render.fcpxml`
(`written`, `validDTD`, `unrepresented`, `reason`). Keep this FCPXML even in fully
automatic runs; retimed/transition/crop/layered recipes report a reason instead.

Single-pass rendering (`video.encoding`) replaces FFmpeg part renders: masks, color,
titles and `styledCaptions` are composited per frame and encoded once. Variable
frame-rate sources are filled to the output cadence like FFmpeg's `fps` filter.
For libass parity take caption text, times and `\pos` from the burned ASS (it
re-balances line breaks), drop zero-length events, use the font's exact PostScript
name (for example `HiraginoSans-W6`) and `assFontSize` equal to the ASS size.
Clip spans must already be trimmed to the planned frame count.
Use an explicitly authorized absolute output/scratch path that meets the current
project's storage policy. Do not assume a home-directory test root.
Ranges are source seconds; fades and additional-audio offsets are output seconds.
Optional `video.color` controls brightness/contrast/saturation via an extra native
encoding pass with SDR-clamped input, not HDR preservation. Describe the current
schema before using it. Optional static `video.titles` uses white bold system text
and top-left pixel positions; overflow is refused. `video.captions` supplies timed
cues; `captionStyle` supports a real black outline, box opacity, optional font size
and bottom margin. `video.masks` covers source rectangles before text; full opacity
conceals burned-in subtitles but does not restore their background. These effects
share the extra encoding pass. Keep raw ASR/review evidence; never treat repeated
recognition or source-subtitle OCR as ground truth. Only native-clock end rounding
within one tick is normalized, with `originalDurationSeconds` retained.

Choose verification durations from the current task's acceptance criteria, within
the discovered tool bounds. A first-frame probe does not verify a whole output.
Audio decoding beyond 30s requires explicit `maximumDurationSeconds`;
transcription remains ≤60s per input, so longer material needs documented chunks
and boundary review.

Optional clip `transitionInSeconds` overlaps adjacent clips with
video dissolve and linear audio crossfade, shortening the timeline. It is limited
to 5 seconds and half either adjacent output clip; explicit/transition audio fades
use the longer duration, not competing ramps. Inspect export, full decode, measured effects and app
import as separate evidence; a successful export is not all four.

Motion XML (`ozml`) is **undocumented**. Reading is allowed; writes/patches require
`allowUndocumentedFormat: true`, a version-matched template/disposable project,
and explicit authorization to use this experimental method. Never describe it as
an official API. No public headless Motion renderer has been established.

Before MIDI/OSC writes, confirm the assignment, channel, scaling, destination and
intended application. An endpoint name is not proof of exclusive application
routing. Do not enable IAC/network MIDI or remap control surfaces automatically;
these can affect other apps/hardware. GUI automation is not sample-accurate MIDI
performance. Never emit audible test notes, start recording or alter a live concert
as setup. MIDI file generation is safe for offline tests; live routes need a
separately agreed test. MainStage has no documented native OSC listener.

For Compressor, use a **trusted** `.cmprstng` or Apple `.setting` preset. Submission creates a private
unique subdirectory under the requested output directory, so existing outputs are
not overwritten. Preserve returned output path and job/batch ID. Check status by
that ID; a dispatch/exit-zero response is not completion. Never retry a failed or
timed-out submission blindly: it may already have created a job. Cancellation
must target only the explicitly authorized ID, never all jobs or the service.

## Native background UI — before Peekaboo

Order of preference: internal channels (CLI, files, Open Document, MIDI/OSC) →
native `ui_*` Accessibility tools → Peekaboo only for a remaining, explained gap.

- The `ui_*` tools never activate an app or synthesize keyboard/pointer input.
  Check `focus.frontmostChanged` in every result; report it if true.
  The exceptions are the Final Cut Pro operations that need an active app:
  `fcp_export`, `fcp_xml_export` (with `allowForeground: true`), `fcp_project_open`,
  the paste step of `fcp_effects_paste` and `fcp_library_close`. Foreground and
  clipboard side effects require authorization for the current task; past approvals
  do not transfer. These tools restore the previous frontmost app, also on failure.
- Locate elements with `ui_inspect` first. Prefer `identifier`/`role`/`title` or
  `containsText` (for table rows) over raw paths; ambiguity is an error, use `index`.
  UI titles are localized (for example 選択, ライブラリを開く); read them, never assume.
- Put text into fields with `ui_set_value` (`attribute: value`), not typing.
  Synthetic typing under a Japanese input method turns `/` into `・`.
- After `app_open_document` or a menu item, call `ui_wait` for the expected window
  in a separate Executor call. Never place a dispatching call and polling code that
  can throw in one `execute`: a later throw hides that the dispatch already happened.
- `dispatched: true` is not success. Confirm the resulting state (window gone,
  file created, project listed) before reporting completion.
- Keyboard-only steps (for example a Save panel's Go to Folder sheet) cannot run in
  the background. Prefer a design that avoids them; if unavoidable, explain the gap,
  serialize with the user and restore any input source you changed.

## Installed editions — verify on every Mac

| App selector | Creator Studio bundle ID | Standalone bundle ID |
|---|---|---|
| `motion` | `com.apple.motionappApp` | `com.apple.motionapp` |
| `compressor` | `com.apple.CompressorApp` | `com.apple.Compressor` |
| `finalCutPro` | `com.apple.FinalCutApp` | `com.apple.FinalCut` |
| `logicPro` | `com.apple.mobilelogic` | `com.apple.logic10` |
| `mainStage` | `com.apple.MainStageApp` | `com.apple.mainstage3` |

The adapter defaults to Creator Studio; discover installed editions and pass an
explicit installed `bundleID` when selecting another edition. Do not assume any
edition is installed, derive IDs by appending `App`, or open one project in two editions.
Never purchase, subscribe, accept licenses, download sound packs, install plugins
or change audio/MIDI device settings as an incidental automation step.

## Continually expand machine interfaces

Prefer native APIs, typed official CLI calls and validated file interchange.
Before repeating a UI workflow, check whether a bounded native adapter can replace
it. Record missing contracts; when adapter development is within the authorized
scope, use the component's quality gates and verify real artifacts.
Do not broaden unsafe/private API use or make unsupported headless claims just
to avoid UI. Motion project editing, Motion rendering and AVFoundation rendering
remain distinct capabilities and need separate evidence.

## Peekaboo fallback — auxiliary, only after the native `ui_*` tools

Peekaboo is auxiliary. Its errors can be false negatives (an action reported as
failed may have happened) and false positives; always observe state before any
retry. Once a Save panel's Go to Folder sheet is open, Peekaboo cannot target that
app (the sheet has no matching accessibility window); close or cancel it with the
native `ui_perform` (`CloseButton`, `CancelButton`) instead of force quitting.

Explain which requested operation has no suitable native method. Discover tools
in namespace **`apple-pro-apps-ui`**, not the primary namespace. Verify the
configured existing Peekaboo executable and launcher; do not assume a Homebrew
prefix or source checkout, or install a second runtime automatically.

1. Describe/call `permissions` from the **Executor-hosted MCP**. A standalone CLI
   can check a different Bridge host. Screen Recording and Accessibility are
   required; keyboard/physical pointer operations also need Event Synthesizing.
   The user grants these permissions to the identity reported by the MCP.
2. Resolve exact app/PID and window ID using `app`/`window`. Inspect `menu` and
   `see` for that exact window, including project, selection, mode and dialogs.
3. Prefer semantic menu or fresh-snapshot AX `action`/`set_value`. Otherwise use
   screenshot-bound coordinates. OCR text is evidence, not an AX action target.
   Use `coordinate_context`; don't assume a Retina scale or reuse stale snapshots.
4. `drag`/`move` and some input paths manipulate the physical cursor. The fallback
   intentionally enables foreground capability; serialize with the user and stop
   on focus changes. Never parallelize GUI mutations or target unrelated apps.
5. Re-observe after every mutation; use bounded `verify_state` where supported.
   `verified: false`, partial dispatch or retry-unsafe responses require inspection,
   not automatic replay. Respect actual local keymaps and UI language.

The fallback is **general macOS automation, not a five-app sandbox**. Clipboard,
foreground and captures can affect shared state. Existing Executor approvals
remain in force. Dedicated shell, browser, nested-agent and recording tools are
not allowlisted. A tool-name filter does not constrain every argument: do not use
optional AI analysis/provider features without explicit authorization. Keep
captures/large results in an explicitly configured, policy-compliant private
scratch directory (`umask 077`); verify the tool's actual destination rather than
assuming environment variables control it. Bound terminal output and inspect
only the returned target image.
Never dump base64 images, commit screenshots, read unrelated clipboard data or
upload project views to another model provider.

## Integrity / completion

Confirm project, inputs, desired edit, destination and collisions. Use disposable
copies for experiments; never directly mutate `.fcpbundle`, `.logicx` or `.concert`
package internals. Confirm destructive changes, original-media consolidation,
recording, publishing and overwrite/delete unless already explicitly requested.
For export verify range, codec, frame/sample rate, color space, channels and actual
artifact. Separate submission, dispatch and verified completion in the report.

There is no universal public all-functions API for these apps. Native capabilities,
experimental file editing, UI fallback and human-only steps are distinct; never
claim every operation is supported because tool discovery succeeded.

For setup/build/check procedures, locate the deployed server's own documentation;
do not assume another checkout exists. Its native code is a Swift Package with
Swift Testing and warnings-as-errors checks.
No new Shell implementation, duplicate Peekaboo installation, permanent daemon,
provider API key or blanket approval policy is required.

## Sources

- https://developer.apple.com/documentation/professional-video-applications/sending-data-programmatically-to-final-cut-pro
- https://support.apple.com/guide/compressor/cpsr9be73312/mac
- https://support.apple.com/guide/logicpro/osc-message-paths-ctlsf67f4bdc/mac
- https://peekaboo.sh/MCP.html
- https://peekaboo.sh/commands/mcp.html
- https://github.com/openclaw/Peekaboo/blob/main/docs/permissions.md
- https://support.apple.com/127131
