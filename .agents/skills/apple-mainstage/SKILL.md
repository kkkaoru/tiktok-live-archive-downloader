---
name: apple-mainstage
description: Operate MainStage through Executor using native concert/patch opening and explicitly routed CoreMIDI controls first. Covers patches, sets, assignments, layouts and safe live-performance boundaries; Peekaboo is UI-only fallback.
---

# MainStage — explicit MIDI mappings first

Load **apple-pro-apps**. Use native Executor namespace `apple-pro-apps`; discover
and describe every tool before calling it. This Mac has both Creator Studio
`com.apple.MainStageApp` and standalone `com.apple.mainstage3`. Choose one edition
explicitly and never open the same concert in both.

## Machine integration

- `app_open_document` with `app: "mainStage"` delivers an existing approved
  `.concert` or `.patch` through Open Document. App acceptance does not establish
  successful loading; plugins, audio access, sound packs or licensing may intervene.
  Do not edit a live `.concert` bundle's internals or test on a performance concert.
- MainStage's normal external control path is **MIDI assignments and mappings**:
  hardware/MIDI control → screen control → parameter or action. A MIDI number has
  no universal meaning independent of the selected concert and mappings.
- Obtain the user's intended route, channel, mapping and target patch/parameter.
  `midi_destinations` lists current endpoint ID/name pairs. Select an exact pair,
  not the first port. IAC/shared routes can reach other apps and devices; do not
  enable or reconfigure them as setup.
- Use `midi_send` only for an already verified destination: CC 0–127/value 0–127,
  program change number 0–127 with value zero, or pitch bend value 0–16383 (center
  8192, number zero). Channel is 1–16. Account for MainStage's program-change source
  and patch mappings; a dispatched packet does not verify a patch switch.
- MainStage has no documented native OSC listener. `osc_send` is not automatically
  a MainStage API. An explicitly configured OSC-to-MIDI bridge would be needed;
  do not install a new bridge/daemon or claim OSC support without it.
- `midi_file_create` is useful for portable offline note data/Logic preparation,
  not evidence that MainStage imports a MIDI file as a concert or plays it directly.

## Offline backing-media preparation

`media_edit` can trim/reorder/time-scale/fade/mix sources into a new M4A when video
settings are omitted. `media_project_read` restores the reusable JSON recipe;
`audio_measure` provides bounded mono RMS/peak/zero-crossing evidence without
playback. Use these for offline preparation, not as a concert editor, playback
plug-in assignment, routing change or verification of MainStage's live behavior.
Loading/assigning prepared media still requires an explicitly scoped app workflow.

## UI-only authoring fallback

Use `apple-pro-apps-ui` only for an identified native gap, following shared rules.
MainStage modes have different meanings:

- **Layout:** arrange screen controls and assign hardware inputs.
- **Edit:** create/organize patches and sets, edit channel strips/plugins, keyboard
  layers/splits, screen-control parameter mappings and concert-level settings.
- **Perform:** use the prepared mappings; avoid editing a live concert, changing
  output devices or entering full screen unexpectedly.

Confirm whether an edit belongs to patch, set or concert scope. Save experiments
as a new concert, preserve assets and verify mappings after reload. In performance,
patch changes can affect active notes and sound; gain/transport/record/mute/panic
commands require explicit intent. Do not emit test audio, start recording, change
sample rate/buffer size or install sound libraries as setup.

No public general-purpose concert editor or guaranteed real-time API has been
established. Expose the exact native capability and state the remaining UI/human
step rather than calling every operation supported.

## Apple references

- https://support.apple.com/guide/mainstage/welcome/mac
- https://help.apple.com/pdf/mainstage/en_US/mainstage-user-guide.pdf — mappings, modes, program changes and actions
- https://help.apple.com/pdf/mainstage-effects/en_US/mainstage-effects-user-guide.pdf — Scripter and plugin scope
