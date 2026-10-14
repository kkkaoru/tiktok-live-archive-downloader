# Motion caption fitting before paragraph layout

An outer FCP title transform scales an already laid-out paragraph. It cannot make
more glyphs fit the original paragraph width. For fitted captions, publish and
set text geometry before paragraph layout instead.

`python -m replay.motion_pre_layout SOURCE.moti NEW.moti RECEIPT.json` creates a
new candidate from the observed three-layer style20 Motion5.15 title. It verifies
the selected font, proportional outline/shadow defaults, bottom paragraph layout
and shared non-fading two-frame entrance. It publishes three font sizes, three
outline widths, and outer-rim shadow distance/blur X/blur Y without changing any
base values. Inputs/outputs must stay inside the canonical checkout. It never
modifies installed templates or operates an app.

Save and publish a new name through Motion, then preserve a repository evidence
copy and run `python -m replay.motion_pre_layout SAVED.moti READBACK.json --inspect`.
This verifies the published scalar contract; actual FCP glyph rendering remains a
separate test. Published controls are synchronized by the caller, not by a rig.

`scaled_values(document, factor)` returns nine `(name, key, value)` tuples as a
verified template geometry contract. Do not emit these Motion style paths as
direct FCP title parameters: the native import test dropped all titles when those
nine parameters were present, while the same template without them retained its
titles. Motion publication alone does not establish a usable FCP parameter key.

The supported v6 consumer expands each of the three text layers' left and right
paragraph margins by `1 / factor`, then scales their common Motion parent group
before canvas clipping. Generate its publication candidate with `--group-scale`;
save/publish through Motion and verify `group_controls` readback. The two published
scale component controls authenticate the template, but FCP consumes their parent
vector parameter `9999/999166631/1/100/105` with value `factor factor`, together
with the six paragraph-margin parameters. Separate direct component parameters
did not survive native import. `fcp_pre_layout` validates this shared contract.

All three text nodes remain plain text. No `text-style-def` font or paint
overrides are emitted. Native trials of those overrides lost template colors or
fell back to an unintended font; paragraph expansion with an outer FCP scale
clipped the text to the canvas before shrinking it. Those approaches are not
accepted alternatives. Parent-group scaling retains the template's font,
gradient, outline, shadow and two-frame entrance together. Its identity position,
anchor and rotation, and the children's bottom origin, are validated. At factor1
the consumer emits no geometry overrides.

Keep the outer FCP scale at1. The factor must be finite, positive and at most1.
Derive it from measured glyph/outline bounds and available width/height, not
character count alone. The consumer replaces prior geometry parameters after
title cloning. Recompute placement using
the actual fitted bounds while preserving recognized speech clock.

`python -m replay.fcp_pre_layout_migrate CONFIG ABS_OUTPUT` migrates an existing
XML using a version1 configuration with pinned `sourceXML` and template inputs.
It preserves text, clock and Y placement while replacing the outer-scale
representation. Migration does not repair an incorrect existing Y position.

Verify ordinary factor1, long text, multiline text, all outlines and shadow in
native frames; native save/publication or XML readback alone does not prove
complete visible text, readability, face clearance or time synchronization.
