"""Measured, POS phrase captions with canonical style colours and no growth pulses."""

from __future__ import annotations

import argparse
import json
import math
import re
from collections.abc import Callable, Mapping
from dataclasses import dataclass, replace
from pathlib import Path

from replay import caption_style_defaults as style
from replay.caption_linguistics import (
    PROHIBITED_LINE_END,
    PROHIBITED_LINE_START,
    CaptionLayoutError,
)
from replay.caption_motion_bundle import read_output_tokens
from replay.caption_native_layout import NativeCaptionLayout, NativeCaptionStyle
from replay.caption_timing import CaptionCue, CaptionPolicy, SpokenToken, phrase_units
from replay.core import ReplayError
from replay.recordings import record
from replay.speaker_subtitles import SpeakerCue, ass_clock, bgr, clean_text, render_srt
from replay.stages import file_digest

GRADIENT_BANDS = 32
MAXIMUM_PAGE_SECONDS = 3.0
MAXIMUM_GAP_SECONDS = 0.25
HOLD_SECONDS = 0.12


@dataclass(frozen=True, slots=True)
class Appearance:
    font_size: float
    center_x: float
    bottom_y: float
    safe_left: float
    safe_right: float
    font_name: str = style.FONT_POSTSCRIPT_NAME

    def __post_init__(self) -> None:
        if (
            (
                self.font_name != style.FONT_FAMILY
                and re.fullmatch(r"NotoSansJP-[A-Za-z0-9_-]{1,80}", self.font_name) is None
            )
            or not all(
                math.isfinite(v)
                for v in (
                    self.font_size,
                    self.center_x,
                    self.bottom_y,
                    self.safe_left,
                    self.safe_right,
                )
            )
            or not (
                16 <= self.font_size <= 128
                and 0 <= self.safe_left < self.center_x < self.safe_right <= 1920
                and 3 * self.font_size < self.bottom_y <= 1066
            )
        ):
            raise ReplayError("Invalid measured caption appearance")

    @property
    def border(self) -> float:
        return style.WHITE_STROKE_WIDTH / 2 * self.font_size / style.REFERENCE_FONT_SIZE

    @property
    def rim(self) -> float:
        return (
            (style.OUTER_BLACK_STROKE_WIDTH - style.WHITE_STROKE_WIDTH)
            / 2
            * (self.font_size / style.REFERENCE_FONT_SIZE)
        )


def phrase_cues(
    tokens: tuple[SpokenToken, ...],
    cuts: tuple[tuple[int, int, int], ...],
    *,
    layout: Callable[[str], str],
    minimum_seconds: float = 0.6,
) -> tuple[CaptionCue, ...]:
    """Display whole phrases at observed onsets, without cumulative prefix updates.

    Minimum duration is a grouping target. A hard cut or real silence is never
    crossed to manufacture support; remaining short phrases are reported by CLI.
    """
    if not math.isfinite(minimum_seconds) or not 0.2 <= minimum_seconds <= 2:
        raise ReplayError("Invalid phrase display minimum")
    units = phrase_units(
        tokens,
        policy=CaptionPolicy(
            linguistic_boundaries=True,
            complete_phrases=True,
            measured_layout=layout,
            maximum_characters=80,
            maximum_lines=3,
            maximum_gap=MAXIMUM_GAP_SECONDS,
            hold_seconds=HOLD_SECONDS,
        ),
    )
    result: list[CaptionCue] = []
    pending: list[SpokenToken] = []

    def fits(text: str) -> bool:
        try:
            layout(text)
        except CaptionLayoutError:
            return False
        return True

    def flush(next_start: float | None = None) -> None:
        if not pending:
            return
        start, end = pending[0].start, max(unit.end for unit in pending)
        cut_end = next(
            (last / 30 for first, last, _ in cuts if first / 30 <= start < last / 30), None
        )
        if cut_end is None:
            raise ReplayError("Phrase onset has no retained cut")
        end = min(end + HOLD_SECONDS, cut_end, next_start if next_start is not None else cut_end)
        text = "".join(unit.text for unit in pending)
        result.append(CaptionCue(layout(text), start, end))
        pending.clear()

    for unit in units:
        if not fits(unit.text):
            raise ReplayError(f"Intact phrase exceeds measured display area: {unit.text!r}")
        if pending and (
            unit.utterance != pending[-1].utterance
            or unit.start - pending[-1].end > MAXIMUM_GAP_SECONDS
            or unit.end - pending[0].start > MAXIMUM_PAGE_SECONDS
            or not fits("".join(item.text for item in pending) + unit.text)
            or (
                pending[-1].text.endswith(("。", "！", "？", "!", "?"))
                and pending[-1].end - pending[0].start >= minimum_seconds
            )
        ):
            flush(unit.start)
        pending.append(unit)
    flush()
    return tuple(result)


def _rgb(hexadecimal: str) -> tuple[int, int, int]:
    return int(hexadecimal[1:3], 16), int(hexadecimal[3:5], 16), int(hexadecimal[5:7], 16)


def render_ass(
    cues: tuple[CaptionCue, ...],
    appearance: Appearance,
    *,
    glyph_bounds: Mapping[str, tuple[int, int]] | None = None,
) -> str:
    """Render independent bottom-anchored lines, layered outlines and sRGB gradient."""
    header = (
        "[Script Info]\nScriptType: v4.00+\nYCbCr Matrix: TV.709\nPlayResX: 1920\nPlayResY: 1080\n"
        "WrapStyle: 2\nScaledBorderAndShadow: yes\n\n[V4+ Styles]\n"
        "Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, "
        "BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, "
        "BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding\n"
        f"Style: caption,{appearance.font_name},{appearance.font_size},&H00FFFFFF,&H00FFFFFF,"
        "&H00000000,&H00000000,-1,0,0,0,100,100,0,0,1,0,0,2,0,0,0,1\n\n[Events]\n"
        "Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text\n"
    )
    events: list[str] = []
    top_rgb, bottom_rgb = _rgb(style.GRADIENT_TOP_HEX), _rgb(style.GRADIENT_BOTTOM_HEX)
    for cue in cues:
        lines = cue.text.splitlines()
        if not 1 <= len(lines) <= 3 or any(not line.strip() for line in lines):
            raise ReplayError("Caption contains an empty or unsupported line")
        for index, line in enumerate(lines):
            bottom = appearance.bottom_y - (len(lines) - 1 - index) * appearance.font_size
            position = f"\\an2\\pos({appearance.center_x:.3f},{bottom:.3f})"
            prefix = f"{ass_clock(cue.start)},{ass_clock(cue.end)},caption,,0,0,0,,"
            text = clean_text(line)
            scale = appearance.font_size / style.REFERENCE_FONT_SIZE
            shadow_y = bottom + style.SHADOW_OFFSET_Y * scale
            events.append(
                f"Dialogue: 0,{prefix}{{\\an2\\pos({appearance.center_x:.3f},{shadow_y:.3f})"
                f"\\bord{appearance.border + appearance.rim:.3f}\\shad0"
                f"\\blur{style.SHADOW_BLUR_STD_DEVIATION * scale:.3f}"
                f"\\1c&H000000&\\3c&H000000&\\alpha&H66&}}{text}"
            )
            events.append(
                f"Dialogue: 0,{prefix}{{{position}"
                f"\\bord{appearance.border + appearance.rim:.3f}"
                f"\\shad0\\3c&H000000&\\1a&HFF&}}{text}"
            )
            events.append(
                f"Dialogue: 2,{prefix}{{{position}"
                f"\\bord{style.INNER_BLACK_STROKE_WIDTH * scale / 2:.3f}"
                f"\\shad0\\3c&H000000&\\1a&HFF&}}{text}"
            )
            glyph_top, glyph_bottom = -appearance.font_size, 0.0
            if glyph_bounds is not None:
                if line not in glyph_bounds:
                    raise ReplayError("Missing actual libass glyph bounds")
                glyph_top, glyph_bottom = glyph_bounds[line]
                if not -2 * appearance.font_size <= glyph_top < glyph_bottom <= 0:
                    raise ReplayError("Invalid libass glyph ink bounds")
            gradient_top, gradient_bottom = bottom + glyph_top, bottom + glyph_bottom
            events.append(
                f"Dialogue: 1,{prefix}{{{position}\\bord{appearance.border:.3f}"
                f"\\shad0\\3c&HFFFFFF&\\1a&HFF&}}{text}"
            )
            for band in range(GRADIENT_BANDS):
                amount = min(
                    1.0,
                    max(
                        0.0,
                        (band / (GRADIENT_BANDS - 1) - style.GRADIENT_TOP_HOLD)
                        / (style.GRADIENT_BOTTOM_START - style.GRADIENT_TOP_HOLD),
                    ),
                )
                color = bgr(
                    (
                        round(top_rgb[0] + (bottom_rgb[0] - top_rgb[0]) * amount),
                        round(top_rgb[1] + (bottom_rgb[1] - top_rgb[1]) * amount),
                        round(top_rgb[2] + (bottom_rgb[2] - top_rgb[2]) * amount),
                    )
                )
                span = gradient_bottom - gradient_top
                top = 0 if band == 0 else round(gradient_top + span * band / GRADIENT_BANDS)
                last = (
                    1080
                    if band == GRADIENT_BANDS - 1
                    else round(gradient_top + span * (band + 1) / GRADIENT_BANDS)
                )
                events.append(
                    f"Dialogue: 3,{prefix}{{{position}\\bord0\\shad0\\1c{color}"
                    f"\\clip(0,{top},1920,{last})}}{text}"
                )
    return header + "\n".join(events) + "\n"


def restyle(
    plan_path: Path,
    output: Path,
    appearance: Appearance,
    *,
    bounds_path: Path | None = None,
) -> None:
    """Reuse verified timing/layout data for a new, independently tested style render."""
    document = record(json.loads(plan_path.read_text(encoding="utf-8")))
    rows = document.get("captions")
    if not isinstance(rows, list) or not 1 <= len(rows) <= 100_000:
        raise ReplayError("Expected bounded phrase plan")
    captions: list[CaptionCue] = []
    for raw in rows:
        row = record(raw)
        text, start, end = row.get("text"), row.get("start"), row.get("end")
        if (
            not isinstance(text, str)
            or isinstance(start, bool)
            or not isinstance(start, (int, float))
            or isinstance(end, bool)
            or not isinstance(end, (int, float))
            or not math.isfinite(start)
            or not math.isfinite(end)
            or not 0 <= start < end
        ):
            raise ReplayError("Invalid phrase plan timing")
        captions.append(CaptionCue(text, float(start), float(end)))
    bounds: dict[str, tuple[int, int]] | None = None
    if bounds_path is not None:
        measurement = record(json.loads(bounds_path.read_text(encoding="utf-8")))
        if measurement.get("fontSize") != appearance.font_size:
            raise ReplayError("Glyph measurement size differs from the render")
        bounds = {}
        for text, raw in record(measurement.get("records")).items():
            row = record(raw)
            top, bottom = row.get("inkTopOffsetPx"), row.get("inkBottomOffsetPx")
            if type(top) is not int or type(bottom) is not int:
                raise ReplayError("Glyph bounds must be integral")
            bounds[text] = (top, bottom)
        if not bounds:
            raise ReplayError("Glyph bounds are empty")
        scale = appearance.font_size / style.REFERENCE_FONT_SIZE
        reserve = (
            appearance.border
            + appearance.rim
            + scale * (style.SHADOW_OFFSET_Y + 3 * style.SHADOW_BLUR_STD_DEVIATION)
        )
        anchor = math.floor(
            1080
            - style.CAPTION_SCREEN_INSET_PX
            - max(value[1] for value in bounds.values())
            - reserve
        )
        appearance = replace(appearance, bottom_y=min(appearance.bottom_y, anchor))
        document["glyphBoundsSHA256"] = file_digest(bounds_path)
        document["bottomAnchorPx"] = appearance.bottom_y
    output.mkdir(mode=0o700)
    (output / "captions.ass").write_text(
        render_ass(tuple(captions), appearance, glyph_bounds=bounds), encoding="utf-8"
    )
    speakers = tuple(SpeakerCue(cue.start, cue.end, "unassigned", cue.text) for cue in captions)
    (output / "captions.srt").write_text(render_srt(speakers), encoding="utf-8")
    document["restyledFromSHA256"] = file_digest(plan_path)
    document["producerSHA256"] = file_digest(Path(__file__))
    (output / "plan.json").write_text(
        json.dumps(document, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("cues", type=Path)
    parser.add_argument("cuts", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--native", type=Path, required=True)
    parser.add_argument("--evidence", type=Path, required=True)
    parser.add_argument("--font-size", type=float, required=True)
    parser.add_argument("--center-x", type=float, required=True)
    parser.add_argument("--bottom-y", type=float, required=True)
    parser.add_argument("--safe-left", type=float, required=True)
    parser.add_argument("--safe-right", type=float, required=True)
    parser.add_argument("--from-plan", action="store_true")
    parser.add_argument("--font-name", default=style.FONT_POSTSCRIPT_NAME)
    parser.add_argument("--glyph-bounds", type=Path)
    parser.add_argument("--reflow-existing", action="store_true")
    args = parser.parse_args(argv)
    appearance = Appearance(
        args.font_size,
        args.center_x,
        args.bottom_y,
        args.safe_left,
        args.safe_right,
        args.font_name,
    )
    if args.reflow_existing:
        if args.glyph_bounds is None:
            raise ReplayError("Existing-page reflow needs actual glyph bounds")
        reflow_existing(
            args.cues,
            args.output,
            appearance=appearance,
            native=args.native,
            evidence=args.evidence,
            bounds_path=args.glyph_bounds,
        )
        return 0
    if args.from_plan:
        restyle(args.cues, args.output, appearance, bounds_path=args.glyph_bounds)
        return 0
    args.output.mkdir(mode=0o700)
    tokens, cuts, clipped = read_output_tokens(args.cues, args.cuts)
    measured = NativeCaptionLayout(
        executable=args.native,
        directory=args.output / "measurements",
        style=NativeCaptionStyle(
            style.FONT_POSTSCRIPT_NAME,
            appearance.font_size,
            0,
            0,
            appearance.font_size,
        ),
        width=appearance.safe_right
        - appearance.safe_left
        - 2 * (appearance.border + appearance.rim),
        height=3 * appearance.font_size + 16,
        evidence=(args.evidence,),
    )
    cues = phrase_cues(tokens, cuts, layout=measured)
    measured.verify_inputs()
    speakers = tuple(SpeakerCue(cue.start, cue.end, "unassigned", cue.text) for cue in cues)
    (args.output / "captions.ass").write_text(render_ass(cues, appearance), encoding="utf-8")
    (args.output / "captions.srt").write_text(render_srt(speakers), encoding="utf-8")
    result = {
        "version": 1,
        "backend": "libass-pos-phrase",
        "candidateOnly": True,
        "cueSHA256": file_digest(args.cues),
        "cutSHA256": file_digest(args.cuts),
        "styleSHA256": file_digest(Path(style.__file__)),
        "frames": cuts[-1][1],
        "sourceTokens": len(tokens),
        "phraseCues": len(cues),
        "cutClippedHolds": clipped,
        "minimumTargetSeconds": 0.6,
        "shortCues": sum(cue.end - cue.start < 0.6 for cue in cues),
        "animation": "none",
        "captions": [{"text": cue.text, "start": cue.start, "end": cue.end} for cue in cues],
    }
    (args.output / "plan.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps({key: value for key, value in result.items() if key != "captions"}))
    return 0


def reflow_existing(
    plan_path: Path,
    output: Path,
    *,
    appearance: Appearance,
    native: Path,
    evidence: Path,
    bounds_path: Path,
) -> None:
    """Repair only unsafe word breaks or measured width overflow; clocks stay intact."""
    document = record(json.loads(plan_path.read_text(encoding="utf-8")))
    rows = document.get("captions")
    if not isinstance(rows, list) or not rows:
        raise ReplayError("Expected existing phrase pages")
    actual = record(record(json.loads(bounds_path.read_text(encoding="utf-8"))).get("records"))
    output.mkdir(mode=0o700)
    scale = appearance.font_size / style.REFERENCE_FONT_SIZE
    margin = 2 * (appearance.border + appearance.rim + 3 * style.SHADOW_BLUR_STD_DEVIATION * scale)
    width = appearance.safe_right - appearance.safe_left - margin
    measured = NativeCaptionLayout(
        executable=native,
        directory=output / "measurements",
        style=NativeCaptionStyle(
            style.FONT_POSTSCRIPT_NAME, appearance.font_size, 0, 0, appearance.font_size
        ),
        width=width,
        height=3 * appearance.font_size + 16,
        evidence=(evidence, bounds_path),
    )
    changes: list[dict[str, object]] = []
    for index, raw in enumerate(rows):
        cue = record(raw)
        text = cue.get("text")
        if not isinstance(text, str):
            raise ReplayError("Invalid existing phrase text")
        too_wide = False
        for line in text.splitlines():
            ink = record(actual.get(line)).get("inkWidthPx")
            if type(ink) is not int:
                raise ReplayError("Actual glyph width is missing")
            too_wide = too_wide or ink > width
        broken = re.search(r"[ァ-ヺーA-Za-z0-9]\n[ァ-ヺーA-Za-z0-9]", text) is not None
        lines = text.splitlines()
        broken = broken or any(
            right[0] in PROHIBITED_LINE_START or left[-1] in PROHIBITED_LINE_END
            for left, right in zip(lines, lines[1:], strict=False)
        )
        if not too_wide and not broken:
            continue
        fixed = measured(text.replace("\n", ""))
        if fixed.replace("\n", "") != text.replace("\n", ""):
            raise ReplayError("Reflow changed phrase text")
        cue["text"] = fixed
        rows[index] = cue
        changes.append({"index": index, "before": text, "after": fixed})
    measured.verify_inputs()
    document["reflowedFromSHA256"] = file_digest(plan_path)
    document["wordWrapRepairs"] = changes
    document["measuredSafeInkWidthPx"] = width
    document["producerSHA256"] = file_digest(Path(__file__))
    (output / "plan.json").write_text(
        json.dumps(document, ensure_ascii=False, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    raise SystemExit(main())
