"""Whole-replay editable FCPXML: source cuts and template captions.

One project per replay lets a person fine-tune the edit in Final Cut Pro. Every
element stays editable: source cuts are asset clips, processed voice audio is
attached to each cut, and each displayed caption event is one instance
of a caller-supplied, locally installed Motion title.

The title must hold two text objects, or three with a separate keyline face. Each
instance carries identical plain ``<text>`` elements for every object and no
``text-style`` overrides, because overrides replace the template's gradient with a
solid face. The title's look, palette and line spacing therefore come only from
the template; this module neither chooses nor certifies them.

Captions are shown on the output frames whose time ``t`` satisfies
``start <= t < end``. Final Cut Pro edits on frame boundaries, so each event is
snapped to exactly those frames, events shown on no frame are counted and left
out, and pieces of one event split at part boundaries are joined again.

Calibration measured against Final Cut Pro renders of a 1080x1920 project:

- ``adjust-transform`` positions and ``trim-rect`` values are percentages of the
  frame height.
- Multi-line titles are anchored at their first line. The anchor offset was
  measured on Basic Title and is not yet calibrated for two-object templates.

Historical mask records remain readable for migration, but cannot be exported.
Source-caption removal must use detected local repair or verified clean media.

Other canvas sizes are refused until they are calibrated.

EXPERIMENTAL, opt-in ``motion_reveal``: one title per strict ``caption_chains``
chain instead of one per stage. The title carries the chain's final full text,
spans the first stage's start to the last stage's end, and animates the published
Type On End so each original stage frame shows that stage's prefix, counted in
Unicode extended grapheme clusters (``\\X``) as Motion consumed them in native
checks. The title is anchored and laid out as the final full string, so earlier
stages are not claimed to match per-stage titles; template fades apply once per
chain; emoji glyph fidelity failed natively. This mode is not approved for delivery.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import subprocess
from collections.abc import Callable, Iterable, Mapping
from dataclasses import asdict, dataclass, field, replace
from fractions import Fraction
from itertools import pairwise
from pathlib import Path
from xml.etree import ElementTree as ET

import regex

from replay import vertical_captions
from replay.caption_projection import CutSpan
from replay.caption_style_defaults import MAX_CAPTION_LINES
from replay.caption_timing import DISPLAY_TIMESCALE, CaptionCue, display_time
from replay.core import ReplayError
from replay.fcp_measured import CaptionAnchor, read_measured_captions
from replay.fcp_pre_layout import apply_geometry, require_production_geometry, unique_style_ids
from replay.motion_caption import MAX_XML_CHARACTERS
from replay.motion_entrance import validate_entrance_transform
from replay.recordings import record

FCPXML_VERSION = "1.11"
GAUSSIAN_UID = ".../Effects.localized/Blur.localized/Gaussian.localized/Gaussian.moef"
SOURCE_TIMESCALE = DISPLAY_TIMESCALE
CAPTION_LANE_BASE = 10
# FCP generator clips conventionally start their local clock at one hour.
TITLE_START = Fraction(3600)
# Rim and face text objects; each receives the same plain caption text.
TEXT_OBJECTS = 2
MOTION_TEXT_FACTORY = "16"
MOTION_TEMPLATES_DIRECTORY = "Motion Templates.localized"
TITLES_DIRECTORY = "Titles.localized"
MOTION_DOCTYPE = "<!DOCTYPE ozxmlscene>"
PROBE_TIMEOUT_SECONDS = 60
MAX_AUDIO_CHANNELS = 64
# Final Cut Pro keys a published Motion parameter as
# ``9999/<layer id>/<object id>/<channel path>`` (observed in an FCP export).
PUBLISHED_PARAMETER_ROOT = "9999"
# Type On > End of a text object, published once per text object.
REVEAL_CHANNEL = "./2/319/321"
# Publish target names, in ``CaptionTemplate.reveal_keys`` order.
REVEAL_TARGETS: tuple[str, str] = ("Caption Reveal Face", "Caption Reveal Rim")
_MOTION_ID = re.compile(r"[0-9]+")
_REVEAL_KEY = re.compile(
    rf"{PUBLISHED_PARAMETER_ROOT}/[0-9]+/[0-9]+/{re.escape(REVEAL_CHANNEL.removeprefix('./'))}"
)
# Candidate reveal unit supported by bounded native probes, not universal font compatibility.
_GRAPHEME = regex.compile(r"\X")
# ``(title-relative seconds, reveal 0..1)`` keyframe samples.
RevealSamples = tuple[tuple[Fraction, Fraction], ...]


@dataclass(frozen=True, slots=True)
class Calibration:
    """Final Cut Pro rendering constants measured on one canvas size."""

    width: int
    height: int
    # Basic Title first-line anchor offset, in em.
    first_line_offset_em: float
    # Blur radius (output pixels) = slope * slider + intercept, valid for 0.5..1.

    def percent(self, pixels: float) -> float:
        """Output pixels to a percentage of the frame height."""
        return pixels / self.height * 100


VERTICAL_1080 = Calibration(
    width=1080,
    height=1920,
    first_line_offset_em=12.5 / (66 * 1000 / (1058 + 274)),
)


@dataclass(frozen=True, slots=True)
class FontMetrics:
    """Font identity and the OS/2 metrics libass uses for its real-dimension size."""

    family: str
    face: str
    units_per_em: int
    win_ascent: int
    win_descent: int

    def em(self, ass_size: float) -> float:
        return ass_size * self.units_per_em / (self.win_ascent + self.win_descent)


HIRAGINO_SANS_W6 = FontMetrics("Hiragino Sans", "W6", 1000, 1058, 274)


@dataclass(frozen=True, slots=True)
class CaptionStyle:
    ass_size: float
    fill: str
    border: float
    border_color: str
    rim: float
    rim_color: str
    font: FontMetrics = HIRAGINO_SANS_W6


@dataclass(frozen=True, slots=True)
class Caption:
    start: Fraction
    end: Fraction
    x: float
    bottom: float
    lines: tuple[str, ...]
    fill: str
    motion_origin_y: float | None = field(default=None, kw_only=True)
    uniform_scale: float = field(default=1.0, kw_only=True)


@dataclass(frozen=True, slots=True)
class Mask:
    start: Fraction
    end: Fraction
    x: float
    y: float
    width: float
    height: float
    radius: float


@dataclass(frozen=True, slots=True)
class Cut:
    source: Path
    source_start: Fraction
    output_start: Fraction
    duration: Fraction


@dataclass(frozen=True, slots=True)
class Timeline:
    rate: int
    width: int
    height: int
    cuts: tuple[Cut, ...]
    masks: tuple[Mask, ...]
    captions: tuple[Caption, ...]
    style: CaptionStyle | None

    @property
    def duration(self) -> Fraction:
        return sum((cut.duration for cut in self.cuts), Fraction(0))


@dataclass(frozen=True, slots=True)
class CaptionTemplate:
    """An installed two/three-text-object Motion title as FCPXML refers to it."""

    name: str
    uid: str
    src: str
    # Validated Type On End keys (face, rim, optional keyline) in publishing order.
    reveal_keys: tuple[str, ...] | None = None
    text_objects: int = field(default=TEXT_OBJECTS, kw_only=True)
    anchor: CaptionAnchor = field(default=CaptionAnchor.TOP, kw_only=True)
    pre_layout_geometry: tuple[tuple[str, str, float], ...] = field(default=(), kw_only=True)

    def __post_init__(self) -> None:
        if type(self.text_objects) is not int or self.text_objects not in (2, 3):
            raise ReplayError("Caption template needs two or three text objects")


@dataclass(frozen=True, slots=True)
class MediaInfo:
    duration: Fraction
    has_video: bool
    has_audio: bool
    # One audio stream's probed channels; unknown is not permission to guess stereo.
    audio_channels: int | None = field(default=None, kw_only=True)

    def __post_init__(self) -> None:
        if self.audio_channels is not None and (
            type(self.audio_channels) is not int
            or not 1 <= self.audio_channels <= MAX_AUDIO_CHANNELS
            or not self.has_audio
        ):
            raise ReplayError("Invalid audio channel count")


@dataclass(frozen=True, slots=True)
class ProjectReport:
    cuts: int
    masks: int
    mask_clips: int
    captions: int
    # Emitted titles: one per caption event, or one per chain with ``motion_reveal``.
    titles: int
    # Displayed caption events grouped by ``caption_chains`` (singletons included, so
    # at most the per-event title count; equal to ``titles`` with ``motion_reveal``).
    # A structural prefix-growth opportunity only: not verified utterances, layout or
    # pixel equivalence, or rendering performance.
    caption_chains: int
    frames: int
    # Plan entries that no output frame displays, so no clip represents them.
    hidden_masks: int
    hidden_captions: int


@dataclass(frozen=True, slots=True)
class CaptionChain:
    """Displayed caption events whose text grows in place, each with its own clock."""

    stages: tuple[Caption, ...]


@dataclass(frozen=True, slots=True)
class _Display:
    """One emitted title: its caption span and, for a revealed chain, its reveal keys."""

    caption: Caption
    reveal: tuple[CaptionTemplate, RevealSamples] | None = None
    smooth: bool = False


@dataclass(frozen=True, slots=True)
class _Part:
    """One plan entry on the global clock."""

    offset: Fraction
    end: Fraction
    recipe: dict[str, object]
    video: dict[str, object]
    burn: object


# ASS -----------------------------------------------------------------------------

_DIALOGUE = re.compile(
    r"^Dialogue:\s*(\d+),([^,]+),([^,]+),[^,]*,[^,]*,[^,]*,[^,]*,[^,]*,[^,]*,(.*)$"
)
_POSITION = re.compile(r"\\pos\(([-0-9.]+),([-0-9.]+)\)")
_FILL = re.compile(r"\\1c&H([0-9A-Fa-f]{6})&")


def ass_clock(value: str) -> Fraction:
    parts = value.strip().split(":")
    if len(parts) != 3:
        raise ReplayError("Invalid ASS clock")
    hours, minutes, seconds = parts
    return Fraction(int(hours) * 3600 + int(minutes) * 60) + Fraction(seconds)


def parse_ass_captions(text: str, offset: Fraction = Fraction(0)) -> list[Caption]:
    """Captions from the unstroked fill events (``\\bord0``) of a bordered ASS file.

    Zero-length events are dropped because libass draws nothing for them.
    """
    captions: list[Caption] = []
    for line in text.splitlines():
        match = _DIALOGUE.match(line)
        if match is None:
            continue
        body = match.group(4)
        tags = re.match(r"\{(.*?)\}(.*)$", body)
        if tags is None or "\\bord0" not in tags.group(1):
            continue
        position = _POSITION.search(tags.group(1))
        fill = _FILL.search(tags.group(1))
        if position is None or fill is None:
            raise ReplayError("ASS fill event lacks a position or fill color")
        start, end = ass_clock(match.group(2)), ass_clock(match.group(3))
        if end <= start:
            continue
        bgr = fill.group(1).upper()
        lines = tuple(tags.group(2).split("\\N"))
        if not all(part.strip() for part in lines):
            raise ReplayError("ASS caption contains a blank line")
        captions.append(
            Caption(
                start + offset,
                end + offset,
                float(position.group(1)),
                float(position.group(2)),
                lines,
                f"#{bgr[4:6]}{bgr[2:4]}{bgr[0:2]}",
            )
        )
    return captions


# Plan ----------------------------------------------------------------------------


def _number(value: object, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ReplayError(f"Invalid {label}")
    return float(value)


def _frames(value: object, label: str) -> int:
    if type(value) is not int or value < 0:
        raise ReplayError(f"Invalid {label}")
    return value


def _seconds(value: float) -> Fraction:
    return display_time(value)


def _items(value: object, label: str) -> list[object]:
    if value is None:
        return []
    if not isinstance(value, list):
        raise ReplayError(f"Invalid {label} list")
    return value


def _part_cuts(part: _Part) -> list[Cut]:
    """Source clips trimmed to the planned frames (like ``ffmpeg -frames:v``)."""
    cuts: list[Cut] = []
    cursor = part.offset
    for item in _items(part.recipe.get("clips"), "clip"):
        if cursor >= part.end:
            break
        clip = record(item)
        selection = record(clip.get("selection"))
        source = clip.get("sourcePath")
        if not isinstance(source, str) or _number(selection.get("rate"), "rate") != 1:
            raise ReplayError("Only unretimed source clips can be exported")
        take = min(
            _seconds(_number(selection.get("durationSeconds"), "duration")), part.end - cursor
        )
        start = _seconds(_number(selection.get("startSeconds"), "start"))
        cuts.append(Cut(Path(source), start, cursor, take))
        cursor += take
    if cursor < part.end:
        raise ReplayError("Part clips are shorter than the planned frames")
    return cuts


def _part_masks(part: _Part) -> list[Mask]:
    masks: list[Mask] = []
    for item in _items(part.video.get("masks"), "mask"):
        mask = record(item)
        region = record(mask.get("region"))
        radius = mask.get("blurRadius")
        if _number(mask.get("opacity"), "mask opacity") != 1 or radius is None:
            raise ReplayError("Only opaque blur masks can be exported")
        start = part.offset + _seconds(_number(mask.get("startSeconds"), "mask start"))
        end = min(part.offset + _seconds(_number(mask.get("endSeconds"), "mask end")), part.end)
        if end <= start:
            continue
        masks.append(
            Mask(
                start,
                end,
                _number(region.get("x"), "mask x"),
                _number(region.get("y"), "mask y"),
                _number(region.get("width"), "mask width"),
                _number(region.get("height"), "mask height"),
                _number(radius, "mask radius"),
            )
        )
    return masks


def _part_captions(
    part: _Part, render_directory: Path
) -> tuple[list[Caption], CaptionStyle | None]:
    """Captions of the part's burned ASS, clipped to the part, and their style."""
    if not isinstance(part.burn, dict) or not (part.burn.get("ass") or part.burn.get("style")):
        return [], None
    burn = record(part.burn)
    parsed: list[Caption] = []
    if burn.get("ass"):
        ass = render_directory / str(burn["ass"])
        parsed = parse_ass_captions(ass.read_text(encoding="utf-8"), part.offset)
    captions = [
        Caption(item.start, min(item.end, part.end), item.x, item.bottom, item.lines, item.fill)
        for item in parsed
        if item.start < part.end
    ]
    described = record(burn.get("style"))
    style = CaptionStyle(
        _number(described.get("fontSize"), "caption size"),
        str(described.get("fill")),
        _number(described.get("whiteBorder"), "caption border"),
        "#FFFFFF",
        _number(described.get("blackRim"), "caption rim"),
        "#000000",
    )
    return captions, style


def _parts(plan: list[object]) -> tuple[list[_Part], tuple[int, int, int]]:
    """Validate contiguous parts sharing one fit canvas; return them on the global clock."""
    parts: list[_Part] = []
    canvas: tuple[int, int, int] | None = None
    expected_frame = 0
    for raw in plan:
        part = record(raw)
        recipe = record(record(part.get("request")).get("recipe"))
        video = record(recipe.get("video"))
        size = (
            _frames(video.get("width"), "width"),
            _frames(video.get("height"), "height"),
            _frames(video.get("frameRate"), "frame rate"),
        )
        canvas = canvas or size
        if size != canvas or video.get("resizeMode") != "fit":
            raise ReplayError("Every part must share one fit canvas")
        start_frame = _frames(part.get("startFrame"), "start frame")
        frames = _frames(part.get("durationFrames"), "frame count")
        if start_frame != expected_frame or frames == 0:
            raise ReplayError("Parts must be contiguous and non-empty")
        expected_frame += frames
        offset = Fraction(start_frame, size[2])
        parts.append(
            _Part(
                offset, offset + Fraction(frames, size[2]), recipe, video, part.get("captionBurn")
            )
        )
    if canvas is None:
        raise ReplayError("Render plan is empty")
    return parts, canvas


def load_timeline(plan: object, render_directory: Path) -> Timeline:
    """Global timeline from a render plan; parts are trimmed to their planned frames."""
    if not isinstance(plan, list) or not plan:
        raise ReplayError("Render plan is empty")
    parts, (width, height, rate) = _parts(plan)
    cuts = [cut for part in parts for cut in _part_cuts(part)]
    masks = [mask for part in parts for mask in _part_masks(part)]
    captions: list[Caption] = []
    styles: set[CaptionStyle] = set()
    for part in parts:
        part_captions, style = _part_captions(part, render_directory)
        captions.extend(part_captions)
        if style is not None:
            styles.add(style)
    if len(styles) > 1:
        raise ReplayError("Every part must share one caption style")
    style = next(iter(styles), None)
    return Timeline(rate, width, height, tuple(cuts), tuple(masks), tuple(captions), style)


def replace_source(timeline: Timeline, source: Path) -> Timeline:
    """Point every cut at ``source``, an explicitly supplied copy of the plan's one source.

    The caller is responsible for ``source`` having the same video timestamps.
    """
    if len({cut.source for cut in timeline.cuts}) != 1:
        raise ReplayError("Only a single-source plan can have its source replaced")
    if source.is_symlink() or not source.is_file():
        raise ReplayError(f"Replacement source must be a regular file: {source.name}")
    return replace(timeline, cuts=tuple(replace(cut, source=source) for cut in timeline.cuts))


def place_captions_above_heads(
    timeline: Timeline,
    samples: tuple[vertical_captions.HeadSample, ...],
    *,
    top_limit: int = vertical_captions.TOP_LIMIT,
    allow_top_overlap: bool = False,
    caption_scale: float = 1.0,
) -> Timeline:
    """Apply the FFmpeg height policy to an editable, single-source FCP timeline.

    Samples must be bound to this source on its presentation clock. Keep source
    cuts, masks, words and times unchanged. An explicit caption_scale uniformly
    scales the calibrated base typography and outlines, never the source footage.
    Rewrap text to the new width before placement; verify rendered bounds separately.
    Strict observed headroom is required unless allow_top_overlap
    explicitly permits clamping at the upper limit. Original framing is unchanged.
    This plans coordinates,
    not a calibration or acceptance of the installed Motion title's glyph bounds.
    """
    if any(caption.motion_origin_y is not None for caption in timeline.captions):
        raise ReplayError("Measured Motion origins cannot use ASS head placement")
    style = timeline.style
    if (
        (timeline.width, timeline.height) != vertical_captions.FRAME
        or timeline.rate != vertical_captions.FRAME_RATE
        or style is None
        or (style.ass_size, style.border, style.rim)
        != (vertical_captions.FONT_SIZE, vertical_captions.BORDER, vertical_captions.RIM)
        or len({cut.source for cut in timeline.cuts}) != 1
    ):
        raise ReplayError("Head placement requires the calibrated vertical style and one source")
    cuts: list[CutSpan] = []
    cursor = 0
    for cut in timeline.cuts:
        values = (cut.source_start, cut.source_start + cut.duration, cut.output_start)
        frames = tuple(value * timeline.rate for value in values)
        if any(value.denominator != 1 for value in frames) or frames[2] != cursor:
            raise ReplayError("Head placement requires contiguous frame-aligned cuts")
        span = CutSpan(*(int(value) for value in frames))
        cuts.append(span)
        cursor += span.duration_frames
    cues: list[CaptionCue] = []
    end = Fraction(0)
    for caption in timeline.captions:
        if (
            not end <= caption.start < caption.end <= timeline.duration
            or not 1 <= len(caption.lines) <= vertical_captions.MAXIMUM_LINES
        ):
            raise ReplayError("Head placement requires ordered captions within the timeline")
        cues.append(CaptionCue("\n".join(caption.lines), float(caption.start), float(caption.end)))
        end = caption.end
    bottoms = vertical_captions.layout(
        tuple(cues),
        tuple(cuts),
        samples,
        top_limit,
        require_headroom=True,
        allow_top_overlap=allow_top_overlap,
        caption_scale=caption_scale,
    )
    return replace(
        timeline,
        style=replace(
            style,
            ass_size=style.ass_size * caption_scale,
            border=style.border * caption_scale,
            rim=style.rim * caption_scale,
        ),
        captions=tuple(
            replace(caption, x=timeline.width / 2, bottom=bottom)
            for caption, bottom in zip(timeline.captions, bottoms, strict=True)
        ),
    )


# Frames --------------------------------------------------------------------------


def frame_span(start: Fraction, end: Fraction, rate: int) -> tuple[Fraction, Fraction] | None:
    """Frame-aligned interval of the frames ``t`` with ``start <= t < end``, if any."""
    first = Fraction(math.ceil(start * rate), rate)
    last = Fraction(math.ceil(end * rate), rate)
    return (first, last) if last > first else None


def _caption_start(caption: Caption) -> Fraction:
    return caption.start


def caption_events(captions: Iterable[Caption], rate: int) -> tuple[list[Caption], int]:
    """Frame-aligned displayed caption events and the number shown on no frame.

    A caption that continues the same display (text, position and fill) at the frame
    where an earlier one ends is one event split at a part boundary and is joined.
    """
    events: list[Caption] = []
    latest: dict[tuple[tuple[str, ...], float, float, str, float | None], int] = {}
    hidden = 0
    for caption in sorted(captions, key=_caption_start):
        span = frame_span(caption.start, caption.end, rate)
        if span is None:
            hidden += 1
            continue
        display = (caption.lines, caption.x, caption.bottom, caption.fill, caption.motion_origin_y)
        index = latest.get(display)
        if index is not None and events[index].end == span[0]:
            events[index] = replace(events[index], end=span[1])
            continue
        latest[display] = len(events)
        events.append(replace(caption, start=span[0], end=span[1]))
    return events, hidden


def _text(caption: Caption) -> str:
    return "\n".join(caption.lines)


def _extends(stage: Caption, following: Caption) -> bool:
    """Whether the line-joined text of ``following`` strictly extends that of ``stage``."""
    text = _text(stage)
    grown = _text(following)
    return len(grown) > len(text) and grown.startswith(text)


def _chain_to_extend(
    event: Caption, *, candidates: Iterable[int], chains: list[list[Caption]]
) -> int | None:
    """The one candidate chain whose last stage ``event`` extends; ambiguity extends none."""
    matches = [index for index in candidates if _extends(chains[index][-1], event)]
    return matches[0] if len(matches) == 1 else None


def caption_chains(events: Iterable[Caption]) -> tuple[CaptionChain, ...]:
    """Group frame-aligned ``caption_events`` into strict prefix-growth chains.

    A stage continues a chain only when it starts on the exact frame where the
    chain's last stage ends, keeps its position and fill, and its line-joined text
    strictly extends that stage's text, compared code point by code point. Text and
    clocks are never normalized or changed; gaps, overlaps, resets and rebalanced
    line breaks start a new chain. Growth may still add a line (``("ab",)`` to
    ``("ab", "c")``), so one full-text title would be anchored unlike each stage.
    """
    chains: list[list[Caption]] = []
    tails: dict[tuple[Fraction, float, float, str, float | None], list[int]] = {}
    for event in sorted(events, key=_caption_start):
        key = (event.start, event.x, event.bottom, event.fill, event.motion_origin_y)
        index = _chain_to_extend(event, candidates=tails.get(key, ()), chains=chains)
        if index is None:
            index = len(chains)
            chains.append([event])
        else:
            tails[key].remove(index)
            chains[index].append(event)
        tails.setdefault(
            (event.end, event.x, event.bottom, event.fill, event.motion_origin_y), []
        ).append(index)
    return tuple(CaptionChain(tuple(stages)) for stages in chains)


def _visible_fraction(stage: Caption, *, final: str, ends: list[int]) -> Fraction:
    """Interior Type On End value showing exactly the stage's ``k`` of ``N`` graphemes.

    ``(2k - 1) / 2N`` lies strictly inside the ``k``-th grapheme, so Motion's
    observed ceiling behaviour (with FadeIn 0) shows exactly ``k``.
    """
    text = _text(stage)
    if not final.startswith(text):
        raise ReplayError("Caption reveal stage is not a prefix of its chain's final text")
    if len(text) not in ends:
        raise ReplayError("Caption reveal stage splits a composed character of the final text")
    return Fraction(2 * (ends.index(len(text)) + 1) - 1, 2 * len(ends))


def chain_reveal_samples(chain: CaptionChain, rate: int) -> RevealSamples:
    """Linear Type On End keys holding each stage's prefix on exactly its own frames.

    Times are relative to the first stage: its value is keyed at 0, and every later
    stage keys the preceding value one frame before its start and its own value at
    its start. Keys identical in time and value (after a one-frame stage) are kept
    once; clocks are never shifted. Text is never normalized: a stage ending inside
    a final-text grapheme cluster is refused.
    """
    if not chain.stages:
        raise ReplayError("Caption chain has no stages")
    first = chain.stages[0]
    final = _text(chain.stages[-1])
    ends = [match.end() for match in _GRAPHEME.finditer(final)]
    samples = [(Fraction(0), _visible_fraction(first, final=final, ends=ends))]
    for previous, stage in pairwise(chain.stages):
        if stage.start != previous.end:
            raise ReplayError("Caption chain stages must be contiguous")
        time = stage.start - first.start
        hold = (time - Fraction(1, rate), samples[-1][1])
        if hold != samples[-1]:
            samples.append(hold)
        samples.append((time, _visible_fraction(stage, final=final, ends=ends)))
    return tuple(samples)


def smooth_chain_reveal_samples(
    chain: CaptionChain, rate: int, *, ramp_frames: int = 8
) -> RevealSamples:
    """Sparse word-onset ramps for Motion Fade In, not pop-in samples.

    Each recognized prefix starts appearing at its own displayed onset, reaches
    its full grapheme fraction within at most ``ramp_frames``, and holds until
    the next onset. Short stages shorten the ramp, never delay the following word.
    Use smooth interpolation in both renderers and independently verify the output.
    """
    if (
        type(rate) is not int
        or not 1 <= rate <= 240
        or type(ramp_frames) is not int
        or not 1 <= ramp_frames <= rate
        or not chain.stages
    ):
        raise ReplayError("Invalid smooth caption rate, ramp or empty chain")
    origin = chain.stages[0].start
    final = _text(chain.stages[-1])
    ends = [match.end() for match in _GRAPHEME.finditer(final)]
    samples = [(Fraction(0), Fraction(0))]
    previous_end = origin
    for stage in chain.stages:
        if (
            stage.start != previous_end
            or not 0 <= stage.start < stage.end
            or (stage.start * rate).denominator != 1
            or (stage.end * rate).denominator != 1
        ):
            raise ReplayError("Smooth caption stages must be contiguous on the frame grid")
        # Reuse strict prefix/grapheme validation, then use the full fade fraction.
        fraction = _visible_fraction(stage, final=final, ends=ends) + Fraction(1, 2 * len(ends))
        if fraction <= samples[-1][1]:
            raise ReplayError("Smooth caption prefixes must strictly grow")
        start = stage.start - origin
        if start != samples[-1][0]:
            samples.append((start, samples[-1][1]))
        support_end = stage.end - origin
        if stage is chain.stages[-1]:
            support_end -= Fraction(1, rate)
        finish = min(support_end, start + Fraction(ramp_frames, rate))
        if finish <= start:
            raise ReplayError("Final smooth prefix needs two visible frames")
        samples.append((finish, fraction))
        previous_end = stage.end
    return tuple(samples)


def _chain_display(
    chain: CaptionChain, *, rate: int, template: CaptionTemplate, smooth: bool
) -> _Display:
    """The chain's final caption, with mode-specific reveal (including smooth singletons)."""
    caption = replace(chain.stages[-1], start=chain.stages[0].start)
    if smooth:
        return _Display(caption, (template, smooth_chain_reveal_samples(chain, rate)), smooth=True)
    if len(chain.stages) == 1:
        return _Display(caption)
    return _Display(caption, (template, chain_reveal_samples(chain, rate)))


def _displays(
    events: list[Caption],
    chains: tuple[CaptionChain, ...],
    *,
    rate: int,
    reveal: CaptionTemplate | None,
    smooth: bool,
) -> list[_Display]:
    """One display per caption event, or per chain when ``reveal`` names the template."""
    if reveal is None:
        return [_Display(event) for event in events]
    return [_chain_display(chain, rate=rate, template=reveal, smooth=smooth) for chain in chains]


def framed_masks(masks: Iterable[Mask], rate: int) -> tuple[list[Mask], int]:
    """Frame-aligned masks and the number covering no frame."""
    framed: list[Mask] = []
    hidden = 0
    for mask in masks:
        span = frame_span(mask.start, mask.end, rate)
        if span is None:
            hidden += 1
            continue
        framed.append(replace(mask, start=span[0], end=span[1]))
    return framed, hidden


# Template ------------------------------------------------------------------------


def _motion_root(text: str) -> ET.Element:
    # Only Motion's observed empty declaration is allowed; never load a DTD/entity.
    text = text.replace(MOTION_DOCTYPE, "")
    if re.search(r"<!\s*(?:DOCTYPE|ENTITY)\b", text, re.IGNORECASE):
        raise ReplayError("External or custom Motion XML declarations are not allowed")
    try:
        root = ET.fromstring(text)
    except ET.ParseError as error:
        raise ReplayError("Invalid Motion XML") from error
    if root.tag != "ozml":
        raise ReplayError("Expected a Motion ozml document")
    return root


def _reveal_key(root: ET.Element, target: ET.Element) -> str:
    """FCPXML key of one reveal publish target, checked against the scene it names."""
    if target.get("channel") != REVEAL_CHANNEL:
        raise ReplayError("Unsupported caption reveal channel")
    identifier = target.get("object", "")
    if _MOTION_ID.fullmatch(identifier) is None:
        raise ReplayError("Caption reveal object ID must be numeric")
    nodes = root.findall(f"scene/layer/scenenode[@id='{identifier}']")
    if len(nodes) != 1 or nodes[0].get("factoryID") != MOTION_TEXT_FACTORY:
        raise ReplayError("Caption reveal must target exactly one text object")
    parent = root.find(f"scene/layer/scenenode[@id='{identifier}']/..")
    layer = "" if parent is None else parent.get("id", "")
    if _MOTION_ID.fullmatch(layer) is None:
        raise ReplayError("Caption reveal layer ID must be numeric")
    channel = REVEAL_CHANNEL.removeprefix("./")
    control = nodes[0]
    for part in channel.split("/"):
        matches = control.findall(f"parameter[@id='{part}']")
        if len(matches) != 1:
            raise ReplayError("Caption reveal text object lacks the unique Type On control")
        control = matches[0]
    return "/".join((PUBLISHED_PARAMETER_ROOT, layer, identifier, channel))


def _reveal_keys(root: ET.Element) -> tuple[str, ...] | None:
    """Published (face, rim) Type On End keys, or ``None`` for a title without them.

    Identity comes from the publish target names, never from object order or IDs.
    """
    targets = [
        target
        for target in root.findall("scene/publishSettings/target")
        if target.get("name") in (*REVEAL_TARGETS, "Caption Reveal Keyline")
    ]
    if not targets:
        return None
    names = sorted(target.attrib["name"] for target in targets)
    count = len(root.findall(f"scene/layer/scenenode[@factoryID='{MOTION_TEXT_FACTORY}']"))
    expected = (*REVEAL_TARGETS, "Caption Reveal Keyline") if count == 3 else REVEAL_TARGETS
    if names != sorted(expected):
        if count == 3:
            raise ReplayError("Caption reveal needs exactly one face, rim and keyline target")
        raise ReplayError("Caption reveal controls must be exactly one face and one rim target")
    by_name = {target.attrib["name"]: target for target in targets}
    ordered = tuple(by_name[name] for name in expected)
    if len({target.get("object") for target in ordered}) != len(ordered):
        raise ReplayError("Caption reveal face and rim must target different text objects")
    return tuple(_reveal_key(root, target) for target in ordered)


def _bottom_template(root: ET.Element, objects: list[ET.Element]) -> None:
    """Require the native bottom layout and zero local origin, without retiming."""
    if len(objects) != 3:
        raise ReplayError("Bottom captions require three Motion text objects")
    if any(marker.text != "8" for marker in root.findall("scene/timemarkerset/timemarker/type")):
        raise ReplayError("Bottom captions require linear Motion timing")
    for node in objects:
        controls = node.find("parameter[@id='2']")
        transform = node.find("parameter[@id='1']/parameter[@id='100']")
        position = node.find("parameter[@id='1']/parameter[@id='100']/parameter[@id='101']")
        if controls is None or node.find("parameter[@id='1']") is None:
            raise ReplayError("Missing bottom caption controls")
        layout = controls.find("parameter[@id='314']")
        alignment = controls.find("parameter[@id='373']/parameter[@id='2']")
        bottom = controls.find("parameter[@id='326']")
        if layout is None or alignment is None:
            raise ReplayError("Missing bottom paragraph controls")
        if (
            layout.get("value") != "1"
            or alignment.get("value") != "2"
            or (bottom is not None and float(bottom.get("value", "nan")) != 0)
        ):
            raise ReplayError("Expected paragraph bottom alignment at zero")
        # Motion's native Save omits the complete default-zero Position subtree
        # and the zero lower paragraph margin. An unexpected subtree is not zero.
        entrance = transform is not None and transform.find("parameter[@id='105']") is not None
        if entrance and transform is not None:
            validate_entrance_transform(transform)
        if position is None:
            if transform is not None and not entrance:
                raise ReplayError("Unsupported bottom caption position structure")
            continue
        axes = position.findall("parameter")
        if {axis.get("id") for axis in axes} != {"1", "2"} or any(
            float(axis.get("value", "nan")) != 0 or list(axis) for axis in axes
        ):
            raise ReplayError("Bottom caption local origin must be static zero")


def caption_template(path: Path, *, anchor: CaptionAnchor = CaptionAnchor.TOP) -> CaptionTemplate:
    """Reference an installed title with two or three synchronized text objects.

    ``uid`` is the path below the user's Motion Templates folder with FCP's ``~/``
    prefix and ``src`` the file URL, as Final Cut Pro writes for installed titles.
    """
    if not path.is_absolute() or path.suffix != ".moti":
        raise ReplayError("Caption template must be an absolute .moti path")
    if path.is_symlink() or not path.is_file():
        raise ReplayError(f"Caption template is not an installed file: {path.name}")
    parts = path.parts
    if MOTION_TEMPLATES_DIRECTORY not in parts:
        raise ReplayError("Caption template must be inside Motion Templates.localized")
    relative = parts[parts.index(MOTION_TEMPLATES_DIRECTORY) + 1 :]
    if relative[0] != TITLES_DIRECTORY:
        raise ReplayError("Caption template must be an installed title")
    if path.stat().st_size > MAX_XML_CHARACTERS:
        raise ReplayError("Motion template exceeds the XML size limit")
    document = path.read_text(encoding="utf-8")
    root = _motion_root(document)
    objects = root.findall(f"scene/layer/scenenode[@factoryID='{MOTION_TEXT_FACTORY}']")
    if len(objects) not in (2, 3):
        raise ReplayError("Caption template must hold exactly two or three text objects")
    if anchor is CaptionAnchor.BOTTOM:
        _bottom_template(root, objects)
    geometry: tuple[tuple[str, str, float], ...] = ()
    if any(
        target.get("name", "").startswith("Caption ")
        and target.get("name", "").endswith(" Font Size")
        for target in root.findall("scene/publishSettings/target")
    ):
        from replay.fcp_pre_layout import paragraph_values

        geometry = paragraph_values(document)
    return CaptionTemplate(
        path.stem,
        "~/" + "/".join(relative),
        path.as_uri(),
        reveal_keys=None if anchor is CaptionAnchor.BOTTOM else _reveal_keys(root),
        text_objects=len(objects),
        anchor=anchor,
        pre_layout_geometry=geometry,
    )


# FCPXML --------------------------------------------------------------------------


def rational(value: Fraction) -> str:
    if value == 0:
        return "0s"
    return (
        f"{value.numerator}s"
        if value.denominator == 1
        else f"{value.numerator}/{value.denominator}s"
    )


def _number_text(value: float) -> str:
    rounded = round(value, 4)
    return str(int(rounded)) if rounded == int(rounded) else str(rounded)


def _calibration(width: int, height: int) -> Calibration:
    if (width, height) != (VERTICAL_1080.width, VERTICAL_1080.height):
        raise ReplayError("Final Cut Pro rendering is only calibrated for 1080x1920")
    return VERTICAL_1080


def _cut_index(cuts: tuple[Cut, ...], time: Fraction) -> int:
    for index in range(len(cuts) - 1, -1, -1):
        if cuts[index].output_start <= time:
            return index
    return 0


def _local(cut: Cut, time: Fraction) -> Fraction:
    return cut.source_start + (time - cut.output_start)


def caption_title(
    caption: Caption,
    style: CaptionStyle,
    calibration: Calibration,
    *,
    title_ref: str,
    text_objects: int = TEXT_OBJECTS,
    pre_layout_geometry: tuple[tuple[str, str, float], ...] = (),
) -> ET.Element:
    """One template title positioned like ``\\an2\\pos(x,bottom)``, with plain text.

    Explicit measured origins use canvas-top-left pixels, not ASS/font baselines.
    Every text object gets the same text; no style is overridden.
    """
    if type(text_objects) is not int or text_objects not in (2, 3):
        raise ReplayError("Caption title needs two or three text objects")
    if caption.motion_origin_y is None:
        em = style.font.em(style.ass_size)
        first_line = (
            caption.bottom
            - (len(caption.lines) - 1) * style.ass_size
            - calibration.first_line_offset_em * em
        )
    else:
        first_line = _number(caption.motion_origin_y, "measured Motion origin")
        if not 0 <= first_line <= calibration.height:
            raise ReplayError("Measured Motion origin is outside the canvas")
    position = (
        f"{_number_text(calibration.percent(caption.x - calibration.width / 2))} "
        f"{_number_text(calibration.percent(calibration.height / 2 - first_line))}"
    )
    title = ET.Element("title", ref=title_ref)
    for _ in range(text_objects):
        ET.SubElement(title, "text").text = "\n".join(caption.lines)
    ET.SubElement(title, "adjust-transform", position=position)
    require_production_geometry(pre_layout_geometry, caption.uniform_scale)
    apply_geometry(title, pre_layout_geometry, caption.uniform_scale)
    return title


def _reveal_value(value: Fraction) -> str:
    """Nearest-double fixed-point text (never exponent notation) for a reveal value."""
    return f"{float(value):.17f}".rstrip("0").rstrip(".")


def _check_reveal_samples(samples: tuple[tuple[Fraction, Fraction], ...]) -> None:
    if not samples:
        raise ReplayError("Caption reveal needs at least one sample")
    times = [time for time, _ in samples]
    if times[0] < 0 or any(later <= earlier for earlier, later in pairwise(times)):
        raise ReplayError("Caption reveal times must be nonnegative and strictly increasing")
    if any(not 0 <= value <= 1 for _, value in samples):
        raise ReplayError("Caption reveal values must be between 0 and 1")


def _reveal_param(
    name: str, key: str, samples: tuple[tuple[Fraction, Fraction], ...], *, smooth: bool
) -> ET.Element:
    param = ET.Element("param", name=name, key=key)
    animation = ET.SubElement(param, "keyframeAnimation")
    for time, value in samples:
        ET.SubElement(
            animation,
            "keyframe",
            time=rational(TITLE_START + time),
            value=_reveal_value(value),
            interp="ease" if smooth else "linear",
            curve="smooth" if smooth else "linear",
        )
    return param


def set_caption_reveal(
    title: ET.Element,
    template: CaptionTemplate,
    *,
    samples: Iterable[tuple[Fraction, Fraction]],
    smooth: bool = False,
) -> None:
    """Animate every text object's published Type On End.

    Smooth mode writes the bounded, synthetically rendered ease/smooth candidate;
    it does not enable Fade In or certify arbitrary-text Motion/FCP equivalence.

    ``samples`` are explicit ``(title-relative seconds, reveal 0..1)`` pairs; all
    text objects receive the same keys on the title-local clock (``TITLE_START``).
    Everything is validated before ``title`` is changed.
    """
    if not isinstance(smooth, bool):
        raise ReplayError("Caption smoothing must be a boolean")
    keys = template.reveal_keys
    if keys is None:
        raise ReplayError("Caption template has no validated reveal controls")
    if (
        len(keys) != template.text_objects
        or len(set(keys)) != len(keys)
        or any(_REVEAL_KEY.fullmatch(key) is None for key in keys)
    ):
        raise ReplayError("Caption template reveal keys are malformed")
    if any(param.get("key") in keys for param in title.findall("param")):
        raise ReplayError("Title already has caption reveal controls")
    first_text = next((i for i, child in enumerate(title) if child.tag == "text"), None)
    if first_text is None:
        raise ReplayError("Title has no text to reveal")
    pairs = tuple(samples)
    _check_reveal_samples(pairs)
    names = (
        (*REVEAL_TARGETS, "Caption Reveal Keyline")
        if template.text_objects == 3
        else REVEAL_TARGETS
    )
    for index, (name, key) in enumerate(zip(names, keys, strict=True)):
        title.insert(first_text + index, _reveal_param(name, key, pairs, smooth=smooth))


def _display_start(display: _Display) -> Fraction:
    return display.caption.start


def _caption_groups(displays: Iterable[_Display]) -> list[tuple[int, _Display]]:
    """Lane group per display: overlapping captions take separate lanes."""
    lane_ends: list[Fraction] = []
    grouped: list[tuple[int, _Display]] = []
    for display in sorted(displays, key=_display_start):
        caption = display.caption
        group = next((i for i, end in enumerate(lane_ends) if end <= caption.start), len(lane_ends))
        if group == len(lane_ends):
            lane_ends.append(caption.end)
        else:
            lane_ends[group] = caption.end
        grouped.append((group, display))
    return grouped


def _asset(resources: ET.Element, identifier: str, path: Path, media: MediaInfo) -> None:
    asset = ET.SubElement(
        resources,
        "asset",
        id=identifier,
        name=path.stem,
        start="0s",
        duration=rational(media.duration),
        hasVideo=str(int(media.has_video)),
        hasAudio=str(int(media.has_audio)),
    )
    ET.SubElement(asset, "media-rep", kind="original-media", src=path.resolve().as_uri())


def _resources(
    root: ET.Element,
    timeline: Timeline,
    sources: Mapping[Path, MediaInfo],
    voice: tuple[Path, MediaInfo] | None,
    template: CaptionTemplate | None,
) -> dict[Path, str]:
    resources = ET.SubElement(root, "resources")
    ET.SubElement(
        resources,
        "format",
        id="r1",
        frameDuration=rational(Fraction(1, timeline.rate)),
        width=str(timeline.width),
        height=str(timeline.height),
        colorSpace="1-1-1 (Rec. 709)",
    )
    asset_ids: dict[Path, str] = {}
    for source in dict.fromkeys(cut.source for cut in timeline.cuts):
        media = sources.get(source)
        if media is None:
            raise ReplayError("Missing source duration")
        if not media.has_video:
            raise ReplayError(f"Source has no video stream: {source.name}")
        asset_ids[source] = f"a{len(asset_ids) + 1}"
        _asset(resources, asset_ids[source], source, media)
    if voice is not None:
        _asset(resources, "voice", *voice)
    if template is not None:
        ET.SubElement(
            resources, "effect", id="t1", name=template.name, uid=template.uid, src=template.src
        )
    return asset_ids


def _check_voice(
    timeline: Timeline,
    voice: tuple[Path, MediaInfo] | None,
    offsets: tuple[Fraction, ...] | None = None,
) -> None:
    if offsets is not None and (voice is None or len(offsets) != len(timeline.cuts)):
        raise ReplayError("Voice selection requires one processed-clock offset per cut")
    if voice is None:
        return
    if not voice[1].has_audio:
        raise ReplayError(f"Voice has no audio stream: {voice[0].name}")
    if offsets is not None:
        if any(
            start < 0 or start + cut.duration > voice[1].duration
            for start, cut in zip(offsets, timeline.cuts, strict=True)
        ):
            raise ReplayError("Voice selection exceeds processed audio")
    elif abs(voice[1].duration - timeline.duration) > Fraction(1, timeline.rate):
        raise ReplayError("Voice audio must be on the output clock")


def _caption_style(timeline: Timeline, template: CaptionTemplate | None) -> CaptionStyle | None:
    """The style for template captions, or ``None`` when there are no captions."""
    if not timeline.captions:
        return None
    if timeline.style is None:
        raise ReplayError("Captions need a caption style")
    if template is None:
        raise ReplayError("Captions need an installed caption template")
    if template.anchor is CaptionAnchor.BOTTOM and any(
        caption.motion_origin_y is None
        or caption.x != timeline.width / 2
        or not 1 <= len(caption.lines) <= MAX_CAPTION_LINES
        for caption in timeline.captions
    ):
        raise ReplayError("Bottom captions require measured centered origins and bounded lines")
    if len({caption.fill for caption in timeline.captions}) > 1:
        raise ReplayError("The caption template cannot show per-caption fill colours")
    return timeline.style


def _append_titles(
    clips: list[ET.Element],
    timeline: Timeline,
    displays: list[_Display],
    style: CaptionStyle,
    calibration: Calibration,
    *,
    text_objects: int,
    pre_layout_geometry: tuple[tuple[str, str, float], ...],
) -> None:
    """Anchor each display's title to the cut holding its start, on the parent's clock."""
    for number, (group, display) in enumerate(_caption_groups(displays), start=1):
        caption = display.caption
        index = _cut_index(timeline.cuts, caption.start)
        title = caption_title(
            caption,
            style,
            calibration,
            title_ref="t1",
            text_objects=text_objects,
            pre_layout_geometry=pre_layout_geometry,
        )
        if display.reveal is not None:
            set_caption_reveal(
                title, display.reveal[0], samples=display.reveal[1], smooth=display.smooth
            )
        title.set("lane", str(CAPTION_LANE_BASE + group))
        title.set("offset", rational(_local(timeline.cuts[index], caption.start)))
        title.set("name", f"Caption {number}")
        title.set("start", rational(TITLE_START))
        title.set("duration", rational(caption.end - caption.start))
        clips[index].append(title)


def _caption_compound(
    resources: ET.Element,
    clip: ET.Element,
    titles: list[ET.Element],
    *,
    cut: Cut,
    identifier: str,
    lane: str,
) -> None:
    """Move one non-overlapping caption lane to a source-cut-local compound.

    A gap has no intrinsic video/audio (Apple FCPXML reference), unlike a black
    generator. Actual transparent compositing still needs an FCP render check.
    Connected titles may extend beyond their anchor cut; retain that full extent.
    """
    offsets = [Fraction(title.attrib["offset"][:-1]) - cut.source_start for title in titles]
    ends = [
        offset + Fraction(title.attrib["duration"][:-1])
        for offset, title in zip(offsets, titles, strict=True)
    ]
    duration = rational(max(cut.duration, max(ends)))
    media = ET.SubElement(resources, "media", id=identifier, name=f"Captions {identifier}")
    sequence = ET.SubElement(
        media, "sequence", format="r1", duration=duration, tcStart="0s", tcFormat="NDF"
    )
    gap = ET.SubElement(
        ET.SubElement(sequence, "spine"),
        "gap",
        name="Caption timing",
        offset="0s",
        start="0s",
        duration=duration,
    )
    for title, offset in zip(titles, offsets, strict=True):
        clip.remove(title)
        title.set("offset", rational(offset))
        title.set("lane", "1")
        gap.append(title)
    ET.SubElement(
        clip,
        "ref-clip",
        ref=identifier,
        lane=lane,
        name=f"Captions {identifier}",
        offset=rational(cut.source_start),
        start="0s",
        duration=duration,
        srcEnable="video",
    )


def _group_cut_captions(resources: ET.Element, clip: ET.Element, *, cut: Cut, number: int) -> None:
    lanes: dict[str, list[ET.Element]] = {}
    for title in clip.findall("title"):
        lanes.setdefault(title.attrib["lane"], []).append(title)
    for lane, titles in lanes.items():
        _caption_compound(
            resources, clip, titles, cut=cut, identifier=f"captions-{number}-{lane}", lane=lane
        )


def _disable_source_audio(
    root: ET.Element,
    *,
    sources: Mapping[Path, MediaInfo],
    asset_ids: Mapping[Path, str],
) -> None:
    """FCP imports can drop srcEnable; deactivate all actual source channels too."""
    media_by_id = {identifier: sources[path] for path, identifier in asset_ids.items()}
    for clip in root.iter("asset-clip"):
        if clip.get("srcEnable") != "video":
            continue
        media = media_by_id[clip.attrib["ref"]]
        if not media.has_audio:
            continue
        if media.audio_channels is None:
            raise ReplayError("Muting source audio requires a verified audio channel count")
        component = ET.Element(
            "audio-channel-source",
            srcCh=", ".join(str(channel) for channel in range(1, media.audio_channels + 1)),
            role="dialogue.dialogue-1",
            active="0",
        )
        # Connected clips precede components; any video filters follow them.
        index = next((i for i, child in enumerate(clip) if child.tag == "filter-video"), len(clip))
        clip.insert(index, component)


def build_fcpxml(
    timeline: Timeline,
    *,
    name: str,
    sources: Mapping[Path, MediaInfo],
    voice: tuple[Path, MediaInfo] | None = None,
    voice_offsets: tuple[Fraction, ...] | None = None,
    source_audio: bool = False,
    template: CaptionTemplate | None = None,
    group_captions: bool = False,
    motion_reveal: bool = False,
    motion_smooth: bool = False,
) -> tuple[bytes, ProjectReport]:
    """Serialize one editable project with exact rational times.

    ``voice`` is processed audio already on the output clock. ``source_audio``
    instead retains each original clip's audio on its own source clock; it cannot
    be combined with processed voice or claimed as music removal. The default is
    still silent without ``voice``. Every displayed
    caption event becomes one ``template`` title. ``group_captions`` keeps those
    titles in cut-local, per-lane compounds rather than a huge flat main timeline;
    it changes neither the number of rendered titles nor their display timing.

    EXPERIMENTAL ``motion_reveal`` instead emits one title per ``caption_chains``
    chain, revealing each stage's prefix on its own frames (see the module notes on
    its full-string anchoring and unverified rendering). It needs a template with
    validated reveal controls. ``motion_smooth`` opts into sparse word fades,
    including singleton chains, using the same samples as the Motion sequence.
    Callers must separately verify that both template objects have Fade In enabled.
    """
    if timeline.masks:
        raise ReplayError("Source-caption blur masks were removed; use verified local repair")
    if not isinstance(motion_smooth, bool) or (motion_smooth and not motion_reveal):
        raise ReplayError("Smooth Motion captions require motion_reveal and a boolean mode")
    if motion_reveal and (template is None or template.reveal_keys is None):
        raise ReplayError("Motion reveal needs a caption template with validated reveal controls")
    if not isinstance(source_audio, bool) or (source_audio and voice is not None):
        raise ReplayError("Source audio needs a boolean mode and cannot be mixed with voice")
    calibration = _calibration(timeline.width, timeline.height)
    _check_voice(timeline, voice, voice_offsets)
    style = _caption_style(timeline, template)
    root = ET.Element("fcpxml", version=FCPXML_VERSION)
    asset_ids = _resources(root, timeline, sources, voice, template)
    if source_audio and any(not sources[cut.source].has_audio for cut in timeline.cuts):
        raise ReplayError("Source audio requested for a clip without an audio stream")
    event = ET.SubElement(ET.SubElement(root, "library"), "event", name=name)
    sequence = ET.SubElement(
        ET.SubElement(event, "project", name=name),
        "sequence",
        format="r1",
        duration=rational(timeline.duration),
        tcStart="0s",
        tcFormat="NDF",
    )
    spine = ET.SubElement(sequence, "spine")
    clips = [
        ET.SubElement(
            spine,
            "asset-clip",
            ref=asset_ids[cut.source],
            name=f"Cut {index + 1}",
            offset=rational(cut.output_start),
            start=rational(cut.source_start),
            duration=rational(cut.duration),
            srcEnable="all" if source_audio else "video",
            tcFormat="NDF",
        )
        for index, cut in enumerate(timeline.cuts)
    ]
    events, hidden_captions = caption_events(timeline.captions, timeline.rate)
    chains = caption_chains(events)
    displays = _displays(
        events,
        chains,
        rate=timeline.rate,
        reveal=template if motion_reveal else None,
        smooth=motion_smooth,
    )
    if style is not None:
        _append_titles(
            clips,
            timeline,
            displays,
            style,
            calibration,
            text_objects=TEXT_OBJECTS if template is None else template.text_objects,
            pre_layout_geometry=() if template is None else template.pre_layout_geometry,
        )
    if group_captions:
        for number, (clip, cut) in enumerate(zip(clips, timeline.cuts, strict=True), start=1):
            _group_cut_captions(root[0], clip, cut=cut, number=number)
    if voice is not None:
        for index, (clip, cut) in enumerate(zip(clips, timeline.cuts, strict=True)):
            ET.SubElement(
                clip,
                "asset-clip",
                ref="voice",
                lane="-1",
                name="Voice",
                offset=rational(cut.source_start),
                start=rational(cut.output_start if voice_offsets is None else voice_offsets[index]),
                duration=rational(cut.duration),
                srcEnable="audio",
            )
    _disable_source_audio(root, sources=sources, asset_ids=asset_ids)
    # FCP imported mixed-prefix IDs (r1/a1/t1/g1) as the wrong resource kinds
    # despite DTD validity. Use one numeric namespace, including compound media.
    resource_ids = {
        resource.attrib["id"]: f"r{index}" for index, resource in enumerate(root[0], start=1)
    }
    for element in root.iter():
        for attribute in ("id", "ref", "format"):
            identifier = element.get(attribute)
            if identifier in resource_ids:
                element.set(attribute, resource_ids[identifier])
    unique_style_ids(root)
    body = ET.tostring(root, encoding="unicode")
    document = f'<?xml version="1.0" encoding="UTF-8"?>\n<!DOCTYPE fcpxml>\n{body}\n'
    report = ProjectReport(
        cuts=len(timeline.cuts),
        masks=0,
        mask_clips=0,
        captions=len(timeline.captions),
        titles=len(displays),
        caption_chains=len(chains),
        frames=int(timeline.duration * timeline.rate),
        hidden_masks=0,
        hidden_captions=hidden_captions,
    )
    return document.encode("utf-8"), report


def _media_info(output: str) -> MediaInfo:
    document: object = json.loads(output)
    probed = record(document)
    streams = [record(stream) for stream in _items(probed.get("streams"), "stream")]
    audio = [stream for stream in streams if stream.get("codec_type") == "audio"]
    if len(audio) > 1:
        raise ReplayError("Multiple source audio streams are not supported")
    channels = _frames(audio[0].get("channels"), "audio channel count") if audio else None
    video = any(stream.get("codec_type") == "video" for stream in streams)
    if audio and not video and ("duration_ts" in audio[0] or "time_base" in audio[0]):
        ticks, time_base = audio[0].get("duration_ts"), audio[0].get("time_base")
        if isinstance(ticks, bool) or not isinstance(ticks, int) or ticks <= 0:
            raise ValueError("invalid audio duration ticks")
        if not isinstance(time_base, str):
            raise ValueError("missing audio time base")
        try:
            exact_duration = ticks * Fraction(time_base)
        except ZeroDivisionError as error:
            raise ValueError("invalid audio time base") from error
        if exact_duration <= 0:
            raise ValueError("invalid audio duration")
        return MediaInfo(exact_duration, False, True, audio_channels=channels)
    duration = record(probed.get("format")).get("duration")
    if not isinstance(duration, str):
        raise ValueError("missing duration")
    return MediaInfo(
        Fraction(duration).limit_denominator(SOURCE_TIMESCALE),
        video,
        bool(audio),
        audio_channels=channels,
    )


def probe_media(path: Path) -> MediaInfo:
    """Use exact stream ticks for audio-only media, container duration otherwise."""
    result = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "format=duration:stream=codec_type,channels,duration_ts,time_base",
            "-of",
            "json",
            str(path),
        ],
        capture_output=True,
        text=True,
        check=False,
        timeout=PROBE_TIMEOUT_SECONDS,
    )
    try:
        return _media_info(result.stdout)
    except (ValueError, ReplayError) as error:
        raise ReplayError(f"Cannot read media duration: {path.name}") from error


def main(argv: list[str] | None = None, probe: Callable[[Path], MediaInfo] = probe_media) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("plan", type=Path, help="render-plan.json")
    parser.add_argument("output", type=Path, help="new .fcpxml file")
    parser.add_argument("--name", required=True)
    parser.add_argument("--audio", type=Path, help="processed voice audio on the output clock")
    parser.add_argument(
        "--measured-captions", type=Path, help="measured candidate replacing ASS captions"
    )
    parser.add_argument(
        "--measured-captions-sha256", help="required SHA-256 of the measured candidate"
    )
    parser.add_argument(
        "--source-audio",
        action="store_true",
        help="retain original clip audio, not music removal; cannot be combined with --audio",
    )
    parser.add_argument(
        "--group-captions", action="store_true", help="use cut-local editable caption compounds"
    )
    parser.add_argument(
        "--motion-reveal",
        action="store_true",
        help="EXPERIMENTAL: one Type On title per caption chain; not approved for delivery",
    )
    parser.add_argument(
        "--motion-smooth",
        action="store_true",
        help="EXPERIMENTAL: sparse smooth fades; requires --motion-reveal and a Fade In template",
    )
    parser.add_argument(
        "--caption-template", type=Path, help="installed two/three-text-object Motion title (.moti)"
    )
    parser.add_argument(
        "--caption-anchor",
        type=CaptionAnchor,
        choices=list(CaptionAnchor),
        default=CaptionAnchor.TOP,
    )
    parser.add_argument(
        "--source-video", type=Path, help="same-timestamp replacement for the plan's one source"
    )
    parser.add_argument(
        "--render-dir", type=Path, help="directory holding the plan's ASS files (default: plan's)"
    )
    args = parser.parse_args(argv)
    if args.output.exists() or args.output.is_symlink() or args.output.suffix != ".fcpxml":
        raise ReplayError(f"Refusing to write {args.output}")
    plan: object = json.loads(args.plan.read_text(encoding="utf-8"))
    timeline = load_timeline(plan, args.render_dir or args.plan.parent)
    if args.caption_anchor is CaptionAnchor.BOTTOM and (
        args.measured_captions is None or args.caption_template is None or args.motion_reveal
    ):
        raise ReplayError("Bottom mode requires measured captions and a per-stage Motion template")
    if (args.measured_captions is None) != (args.measured_captions_sha256 is None):
        raise ReplayError("Measured captions require both a path and SHA-256")
    if args.measured_captions is not None:
        measured = read_measured_captions(
            args.measured_captions,
            sha256=args.measured_captions_sha256,
            width=timeline.width,
            height=timeline.height,
            rate=timeline.rate,
            duration=timeline.duration,
            anchor=args.caption_anchor,
        )
        if timeline.style is None:
            raise ReplayError("Measured captions require a template caption style")
        timeline = replace(
            timeline,
            captions=tuple(
                Caption(
                    _seconds(item.cue.start),
                    _seconds(item.cue.end),
                    item.x,
                    item.y,
                    tuple(item.cue.text.split("\n")),
                    timeline.style.fill,
                    motion_origin_y=item.y,
                )
                for item in measured
            ),
        )
    if args.source_video is not None:
        timeline = replace_source(timeline, args.source_video)
    template = (
        caption_template(args.caption_template, anchor=args.caption_anchor)
        if args.caption_template
        else None
    )
    sources = {source: probe(source) for source in dict.fromkeys(c.source for c in timeline.cuts)}
    voice = (args.audio, probe(args.audio)) if args.audio is not None else None
    document, report = build_fcpxml(
        timeline,
        name=args.name,
        sources=sources,
        voice=voice,
        source_audio=args.source_audio,
        template=template,
        group_captions=args.group_captions,
        motion_reveal=args.motion_reveal,
        motion_smooth=args.motion_smooth,
    )
    with args.output.open("xb") as handle:
        handle.write(document)
    print(json.dumps(asdict(report)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
