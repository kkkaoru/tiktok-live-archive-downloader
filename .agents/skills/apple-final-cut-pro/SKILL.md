---
name: apple-final-cut-pro
description: Integrate Final Cut Pro through Executor using FCPXML inspection, generation, copy/patch and native Open Document import first. Covers timelines, metadata, media interchange and limits of live editing/export; Peekaboo is UI-only fallback.
---

# Final Cut Pro — FCPXML first

Load **apple-pro-apps**. Discover the actual `apple-pro-apps` tool schemas before
calling them. This Mac uses `finalCutPro` / `com.apple.FinalCutApp` (Creator Studio).
Never target the standalone edition implicitly or edit a live `.fcpbundle` database.

## Mechanical timeline/media workflow

1. Obtain a user-approved FCPXML export or build FCPXML following Apple's current
   reference. `interchange_inspect` and `interchange_query` with `kind: "fcpxml"`
   inspect structure, assets, resource IDs, timelines and metadata without the UI.
   Queries return bounded/truncated fragments; narrow the XPath rather than
   assuming a truncated node is the full XML.
2. `interchange_write` creates a **new** `.fcpxml`; `interchange_patch` creates a
   copy while changing existing uniquely selected leaf/attribute values. Use the
   former for structural edits, the latter for precise values. No overwrite,
   arbitrary entity declarations or external DTD loading is allowed.
3. Keep resource references consistent, media URLs correctly escaped, lanes and
   parent-relative offsets intentional, and time values in FCPXML rational seconds
   (for example, `1001/30000s`). Do not substitute decimal frame approximations.
   Match source media formats and the intended sequence rate/resolution/color space.
4. `interchange_inspect` checks UTF-8, well-formed XML and the `fcpxml` root only.
   Separately discover/describe `fcpxml_validate`: it uses the matching DTD from
   the selected installed Final Cut edition, private snapshots and network/catalog-
   disabled system validation. External schema references are refused. DTD success
   **does not resolve media, prove effect fidelity or establish importability**.
   Use version-matched Apple documentation and an isolated library for import tests.
5. Deliver the saved file with `app_open_document`, `app: "finalCutPro"`. This uses
   the documented Open Document path, not UI clicking. FCP may ask for an event,
   library, missing media or license; tool acceptance does not verify the timeline.
   Don't repeat an indeterminate import—it may create duplicate events/projects.

Current exports can also be `.fcpxmld` bundles containing `Info.fcpxml`. The native
open tool accepts that bundle, but XML tools operate on its explicit regular
`Info.fcpxml` file. Write a new standalone `.fcpxml` for experiments; never mutate
a live bundle or assume every bundled asset can be omitted.

## Offline edits before import

For a new rendered asset rather than a live FCP timeline change, use
`media_edit_plan` → `media_edit` → `media_project_read`. Recipes support trim,
reorder, speed, crop/fit/fill/rotation, audio mixing/fades, SDR color controls and
static titles (describe the deployed schema). This produces MP4/M4A plus reusable
JSON, not an FCPXML timeline or an imported FCP project. Preserve the source and
use the Movies verification root for viewing tests. Check `media_verify_video`,
`audio_measure` and `video_frame_measure` for bounded decoded evidence; none
proves FCP import or all-channel/HDR fidelity.

## Export and live-operation limits

FCPXML interchange is not a universal live timeline control API. Apple documents
Custom Share Destinations (media/FCPXML delivery by Apple events), Workflow
Extensions (in-app timeline integration) and FxPlug (effects). These require
additional app/extension implementations and are **not** magically provided by
this MCP. The installed scripting dictionary includes library inspection, but a
complete read/write timeline automation surface is not established.

For UI-only selection, browser organization, effects/color work, keyframe editing
or Share/export, explain the native gap before using `apple-pro-apps-ui` via the
shared safe procedure. Export to a new destination and verify actual media. Use
Compressor's native MCP for subsequent encoding, not blind repeated Share actions.

## Apple references

- https://developer.apple.com/documentation/professional-video-applications/importing-fcpxml-data
- https://developer.apple.com/documentation/professional-video-applications/sending-data-programmatically-to-final-cut-pro
- https://developer.apple.com/documentation/professional-video-applications/receiving-media-and-data-through-a-custom-share-destination
- https://developer.apple.com/documentation/professional-video-applications/workflow-extensions
- https://support.apple.com/guide/final-cut-pro/welcome/mac
