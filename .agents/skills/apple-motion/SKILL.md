---
name: apple-motion
description: Automate Motion through Executor using template-based native XML inspection/copy editing and verified Compressor CLI rendering first. Expand missing typed MCP capabilities before repeating GUI steps; distinguish experimental project edits, actual Motion renders and AVFoundation composites.
---

# Motion — native file integration first

Load **apple-pro-apps** first. Discover native tools in `apple-pro-apps`; the
separate `apple-pro-apps-ui` namespace is fallback, not the primary method.
Use `app_capabilities` and verify `motion` / `com.apple.motionappApp` on this Mac.

## Do not regress to repetitive UI work

Read `MOTION-NATIVE-PLAN.md` and `VERIFICATION.md` in the native Swift package
before authoring. The current user explicitly prioritizes expanding UI-free
capabilities. Do not configure Project Browser fields one click at a time when
a template-copy adapter can express the operation. Identify the missing contract,
implement/test a bounded native adapter, and verify an actual render. Do not
claim an adapter exists until its schema is discovered from Executor.

The canonical skill may live in the originating dotfiles checkout under
`.agents/skills-stroage/apple-motion/SKILL.md` even when the previously advertised
home-directory skill path is absent. A missing symlink is not evidence that
Motion has no native integration; inspect the canonical guidance and repair only
this skill's discovery, without replacing unrelated skill/configuration files.

## Machine workflow

1. Start from a version-compatible, known-good `.motn`, `.moti`, `.motr`, `.moef`
   or `.mogen` document. Do not guess the internal node/class layout.
2. Use `interchange_inspect` with `kind: "motion"`, then `interchange_query` to
   inspect bounded fragments at specific XPath locations. The expected root is
   `ozml`; well-formedness does **not** establish that Motion can open/render it.
3. For text, prefer the discovered `motion_text_inspect` and `motion_text_copy`.
   Inspect gives a source SHA-256, layer IDs/current text and editable flags. Copy
   requires that hash, explicit expected text and replacement per layer, a NEW
   output path and `allowUndocumentedFormat: true`. It updates character objects,
   kerning IDs and style-run lengths together. Supports observed ozml 4.0,
   single-style, neutral-kerning, single-line BMP text only (up to 120 units).
   Never fall back to editing only `<text>` when a character table also exists.
   Unsupported formatting must be investigated, not flattened.
4. For an existing parameter/keyframe value use `interchange_patch` with an
   explicit new output path and a uniquely selecting XPath. Inspect identifiers,
   units and temporal representation first. In the observed Snap template,
   sceneSettings.frameRate=30 plus NTSC=1 means 30000/1001, not exact 30 fps.
   Copy editing is experimental and requires the same explicit authorization.
5. `interchange_write` can create a new file from fully authored XML with the same
   opt-in. It is not a supported schema generator: prefer a known-good template
   over invented project XML. Originals are never overwritten.
6. Use `app_open_document` with `app: "motion"` and the new path for native
   Open Document delivery. Opening is not render validation. Inspect the result
   with the user or the explicitly chosen UI fallback before production use.

## Native rendered-media alternative

If the goal is a new MP4 rather than editing a `.motn`, discover `media_edit`:
typed recipes offer geometry, SDR brightness/contrast/saturation and static
single-line white titles. Color/title effects share an additional native encoding
pass; no Motion project, rig, FxPlug or reusable Motion title template is created.
Use `media_project_read` for the reusable JSON recipe, and full-video/region/audio
measurement tools for bounded export evidence. These do not establish Motion
compatibility, animated text support or HDR preservation.

## Native Motion rendering through Compressor

Discover and describe `compressor_submit` and `compressor_status`. Apple documents
Motion-project input, and this host has previously rendered an unchanged Snap
Lower Third `.motn` to H.264 using the official CLI without UI: 180 fully decoded
frames at 29.97003 fps, 6.006 seconds, with graphics/text visible later in the clip.
That is a narrow verified case, not proof of arbitrary project support.

Use a trusted preset, filesystem source path (not an incorrectly percent-encoded
path), unique output directory and returned job ID. Inspect the job before any
retry. Verify actual output duration/frame count, decoded animation at multiple
times, Japanese text, audio and hashes separately. Submission is not completion.
The first `ap4h.setting` alpha attempt failed with `className is null`; do not
repeat it blindly or promise alpha output. Verify a corrected preset independently.
Do not call an AVFoundation-composited video an entirely Motion-rendered timeline.

## Where GUI fallback may still be needed

No supported universal Motion parameter API has been established. Unknown
factories, rigs, tracking, particles, plugins or template publishing need concrete
format/API research; template editing cannot be assumed to support them all.
Explain the exact gap before using the shared Peekaboo procedure, and do not
return to repetitive GUI work while native development remains actionable.

- New projects: confirm dimensions, frame rate, duration and color space; choose
  Motion Project versus Final Cut Effect/Title/Transition/Generator deliberately.
- Layers/animation: confirm exact layer, playhead and active tool; distinguish
  static changes, keyframes and animation recording. Verify at multiple times.
- Templates: publish selected Inspector parameter controls or rig widgets;
  review Project Inspector → Publishing, save under a new name/category, then
  verify in the corresponding Final Cut browser. Not all items can be published.
- Rendering: prefer the bounded Compressor path above. Only if the particular
  project/preset cannot render through it, investigate Share as an explicit
  fallback. Preserve the failed job and distinguish opaque from alpha exports.

Do not directly alter template bundles shipped by Apple, overwrite user templates,
change custom command sets or assume factory shortcuts. Treat media/fonts/plugins
referenced by XML as part of the version-sensitive dependency set.

## Apple references

- https://support.apple.com/guide/motion/welcome/mac
- https://support.apple.com/guide/motion/motn17691fe6/mac — template types
- https://support.apple.com/guide/motion/motna47583a5/mac — publish controls
- https://support.apple.com/guide/motion/motn13f21017/mac — publish rigs
- https://support.apple.com/guide/motion/motn72925de5/mac — convert project types
