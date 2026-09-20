---
name: apple-logic-pro
description: Integrate Logic Pro through Executor with native MIDI file generation/import, CoreMIDI assigned controls and loopback OSC. Prefer machine interchange over GUI; covers routing, automation, key-command and live-edit limitations.
---

# Logic Pro — MIDI / OSC / files first

Load **apple-pro-apps** and discover native tools in `apple-pro-apps`. This Mac's
Creator Studio edition uses **`com.apple.mobilelogic`**, not a guessed
`com.apple.logic10App`. Pass `app: "logicPro"` to the native open tool.

## Native musical content

Use `midi_file_create` to build a new standard MIDI type-0 file. Supply explicit
`bpm`, optional `ticksPerQuarter` (default 480), and notes with `note` (0–127),
`velocity` (1–127), `channel` (1–16), `startTick` and positive `durationTick`.
Note-off events are paired and sorted before simultaneous note-on events. This
creates a file only; it does not play sound or mutate a Logic project.

Open/import the approved `.mid` with `app_open_document`. Logic can treat opening
MIDI as project creation/import and may show choices; receipt is not proof of the
intended track placement. Preserve the source `.logicx` and use a new/disposable
project for testing. Do not edit Logic project-package internals.

## Offline audio editing and measurement

For media-file edits, discover `media_edit_plan`, `media_edit` and
`media_project_read`. Omit `recipe.video` for M4A: source selection, ordering,
rate, linear gain, fades and timed extra/replacement/mixed audio need no playback
or Logic routing. Fades/offsets use output time; sources remain untouched.
`audio_measure` decodes up to 30 seconds to temporary mono PCM16/16000 Hz and
reports RMS/peak/zero crossings for whole clips and windows. This is not LUFS,
a reliable voice/music pitch detector, per-channel verification or a Logic bounce.
Keep viewing outputs under the Movies verification root; import separately into
a disposable Logic project if actual editor integration is requested.

## Assigned controls

1. Ask for the intended control, assignment, channel and routing. Apple's control
   surfaces and Controller Assignments can map MIDI controls to mixer/plugin
   parameters and key commands. There is no universal factory mapping to invent.
2. Use `midi_destinations` to obtain exact current endpoint IDs/names. A CoreMIDI
   endpoint is **not** proof that only Logic receives it. Do not create IAC ports,
   enable network MIDI, install control-surface profiles or alter mappings silently.
3. For an already verified route, `midi_send` emits one `controlChange`,
   `programChange` or `pitchBend`. Channels are 1-based; CC/program numbers are
   0-based. Program change uses `number` and `value: 0`; pitch bend uses
   `number: 0`, value 0–16383 (center 8192). Receiver effect is unverified.
4. Logic documents OSC paths in Controller Assignments. If its actual loopback
   receiver port and assignment are known, use `osc_send` with that explicit port,
   exact path and float value. There is no port guessing, network scan, wildcard,
   auto-configuration or semantic acknowledgment. Do not assume `/play` or `/volume`
   exists unless the user's controller assignment defines it.

Treat Read/Touch/Latch/Write automation modes, armed tracks and live playback as
important state: an external controller can write automation or change sound.
Never test record/transport/audio-device changes on an existing session. SMF
creation is an offline test; live control needs an agreed route and harmless test.

## Gaps / fallback

Scripter is a MIDI-FX JavaScript environment; controller Lua profiles are another
specialized integration. Neither is a general-purpose external project API.
This MCP does not install either, create arbitrary tracks/plugins through a public
API, guarantee sample-accurate performance or provide automatic bounce/export.
For those UI-only tasks, identify the gap and then use `apple-pro-apps-ui` under
shared rules. Read the actual current key-command set; don't remap it or assume
factory shortcuts. Verify bounce range, sample rate, bit depth and output artifact.

## Apple references

- https://support.apple.com/guide/logicpro-css/control-surfaces-overview-ctls036b3e21/mac
- https://support.apple.com/guide/logicpro/osc-message-paths-ctlsf67f4bdc/mac
- https://support.apple.com/guide/logicpro/assign-buttons-to-key-commands-ctls71c31056/mac
- https://support.apple.com/guide/logicpro/key-commands-overview-lgcp32e85cd9/mac
- https://support.apple.com/guide/logicpro/welcome/mac
