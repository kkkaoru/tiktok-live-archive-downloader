"""Read pinned measured-caption candidates; not template or delivery approval."""

import argparse
import json
import math
from collections.abc import Sequence
from dataclasses import asdict, replace
from enum import StrEnum
from fractions import Fraction
from pathlib import Path

from replay.caption_io import _array, _fields, _number, _string
from replay.caption_measured_job import _merge, _pairs
from replay.caption_measured_plan import MAX_TOKENS, PositionedCue
from replay.caption_native_layout import NativeCaptionLayout
from replay.caption_style_defaults import MAX_CAPTION_LINES
from replay.caption_timing import CaptionCue, display_time
from replay.foreground_evidence import _bound_bytes
from replay.stages import file_digest
from replay.tracking_observations import _checked_output

MAX_CANDIDATE_BYTES = 64 * 1024 * 1024
# A native job can bind 128 request/response pairs plus 32 foreground windows.
MAX_PART_BINDINGS = 512


class CaptionAnchor(StrEnum):
    """Explicit coordinate meaning; old candidates must never be relabelled."""

    TOP = "top"
    BOTTOM = "bottom"

    @property
    def coordinate_system(self) -> str:
        if self is CaptionAnchor.BOTTOM:
            return "top-left canvas; measured paragraph bottom origin"
        return "top-left canvas; calibrated text origin"


def _input_bindings(value: object) -> dict[str, str]:
    if not isinstance(value, dict) or not 1 <= len(value) <= MAX_PART_BINDINGS:
        raise ValueError("Expected bounded nonempty measured input provenance")
    return {_string(name): _string(digest) for name, digest in value.items()}


def read_measured_captions(
    path: Path,
    *,
    sha256: str,
    width: int,
    height: int,
    rate: int,
    duration: Fraction,
    anchor: CaptionAnchor = CaptionAnchor.TOP,
) -> tuple[PositionedCue, ...]:
    """Preserve candidate text/clocks/origins; upstream correspondence is caller-owned."""
    raw = _fields(
        json.loads(_bound_bytes(path, sha256, MAX_CANDIDATE_BYTES), object_pairs_hook=_pairs)
    )
    if (
        type(raw.get("version")) is not int
        or raw.get("version") != 1
        or raw.get("candidateOnly") is not True
        or raw.get("coordinateSystem") != anchor.coordinate_system
        or raw.get("canvas") != [width, height]
        or _fields(raw.get("policy")).get("frame_rate") != rate
    ):
        raise ValueError("Measured caption coordinate contract differs from the timeline")
    result: list[PositionedCue] = []
    previous_start = Fraction(0)
    for value in _array(raw.get("candidates"), MAX_TOKENS):
        item = _fields(value)
        cue = _fields(item.get("cue"))
        text = _string(cue.get("text"))
        start, end = _number(cue.get("start")), _number(cue.get("end"))
        x, y = _number(item.get("x")), _number(item.get("y"))
        if (
            not text.strip()
            or len(text.encode("utf-8")) > 32768
            or any(ord(character) < 32 and character != "\n" for character in text)
            or not 1
            <= len(text.split("\n"))
            <= (MAX_CAPTION_LINES if anchor is CaptionAnchor.BOTTOM else 2)
            or any(not line.strip() for line in text.split("\n"))
            or not previous_start <= display_time(start) < display_time(end) <= duration
            or not 0 <= x <= width
            or not 0 <= y <= height
            or (anchor is CaptionAnchor.BOTTOM and x != width / 2)
        ):
            raise ValueError("Invalid measured caption text, clock or origin")
        result.append(PositionedCue(CaptionCue(text, start, end), x, y))
        previous_start = display_time(start)
    if not result:
        raise ValueError("Measured captions must not be empty")
    return tuple(result)


def _boundary_hold(
    previous: PositionedCue, following: PositionedCue, policy: dict[str, object]
) -> PositionedCue:
    """Apply the planner's own hold rule across independently planned candidates.

    Inside one candidate a caption's hold ends at the next token's onset. A part
    boundary hides that onset, so only an overlap no longer than the policy hold
    is trimmed to the next candidate's first onset; text, starts and positions
    are unchanged and larger overlaps still fail.
    """
    start, end = following.cue.start, previous.cue.end
    if not start < end or start <= previous.cue.start:
        return previous
    if end > start + _number(policy.get("hold_seconds")):
        return previous
    return replace(previous, cue=replace(previous.cue, end=start))


def drop_retained_hidden_prefixes(
    captions: list[PositionedCue], *, rate: int
) -> tuple[list[PositionedCue], int]:
    """Remove captions shown on no frame whose text the next caption extends.

    The planner accepts such a hidden prefix because the following caption
    displays the same text plus more, so removing it loses no text. Any other
    caption without a display frame is kept for the exporter to refuse.
    """
    kept: list[PositionedCue] = []
    for index, item in enumerate(captions):
        following = captions[index + 1] if index + 1 < len(captions) else None
        visible = math.ceil(display_time(item.cue.end) * rate) > math.ceil(
            display_time(item.cue.start) * rate
        )
        prefix = item.cue.text.replace("\n", "")
        retained = following is not None and following.cue.text.replace("\n", "").startswith(prefix)
        if visible or not retained:
            kept.append(item)
    return kept, len(captions) - len(kept)


def ensure_display_frames(
    captions: list[PositionedCue], *, rate: int, duration: Fraction
) -> tuple[list[PositionedCue], int]:
    """Give every caption at least one display frame without losing its text.

    A caption shown on no frame is extended to the end of its first frame, and
    the following caption's first display frame may move by at most one frame. Denser independent
    replacements must be replanned; delay must never accumulate silently.
    Text, order and positions are unchanged.
    """
    result: list[PositionedCue] = []
    widened = 0
    previous_end = Fraction(0)
    for item in captions:
        start = max(display_time(item.cue.start), previous_end)
        end = display_time(item.cue.end)
        first = Fraction(math.ceil(start * rate), rate)
        if end <= first:
            end = min(first + Fraction(1, rate), duration)
            widened += 1
        # The budget is one display frame: the caption's first shown frame may move by one,
        # however close to a frame boundary its recognized start falls.
        if first * rate - math.ceil(display_time(item.cue.start) * rate) > 1:
            raise ValueError("Caption density exceeds one-frame display-delay budget")
        if end > duration or end <= first or end <= start:
            raise ValueError("Caption has no frame support inside the declared duration")
        if start != display_time(item.cue.start) or end != display_time(item.cue.end):
            cue = replace(item.cue, start=float(start), end=float(end))
            item = replace(item, cue=cue)
        result.append(item)
        previous_end = display_time(item.cue.end)
    return result, widened


def combine_measured_captions(
    references: Sequence[tuple[Path, str]],
    output: Path,
    *,
    width: int,
    height: int,
    rate: int,
    duration: Fraction,
    calibration: tuple[Path, str] | None = None,
) -> None:
    """Combine ordered candidates without retiming or approving their geometry.

    Part boundaries must already be non-overlapping policy resets. Verify every
    retained input fingerprint; do not remeasure or relabel an obsolete producer.
    """
    if (
        any(type(value) is not int or not 1 <= value <= 8192 for value in (width, height))
        or type(rate) is not int
        or not 1 <= rate <= 240
        or not isinstance(duration, Fraction)
        or not 0 < duration <= 120
    ):
        raise ValueError("Invalid measured canvas, rate or duration bounds")
    if not 1 <= len(references) <= 128:
        raise ValueError("Expected bounded measured candidate references")
    output = _checked_output(output)
    if output.exists():
        raise FileExistsError(output)
    if calibration is None:
        raise ValueError("A common pinned render calibration is required")
    calibration_path, calibration_digest = calibration
    calibration_name = str(_checked_output(calibration_path))
    bindings: dict[str, str] = {}
    policy: dict[str, object] | None = None
    combined: list[PositionedCue] = []
    for path, digest in references:
        raw = _fields(
            json.loads(_bound_bytes(path, digest, MAX_CANDIDATE_BYTES), object_pairs_hook=_pairs)
        )
        current_policy = _fields(raw.get("policy"))
        if policy is not None and current_policy != policy:
            raise ValueError("Measured candidates have conflicting policies")
        policy = current_policy
        part_bindings = _input_bindings(raw.get("inputs"))
        if part_bindings.get(calibration_name) != calibration_digest:
            raise ValueError("Measured candidates do not share the pinned render calibration")
        _merge(bindings, part_bindings)
        _merge(bindings, {str(path.absolute()): digest})
        members = read_measured_captions(
            path, sha256=digest, width=width, height=height, rate=rate, duration=duration
        )
        if combined:
            combined[-1] = _boundary_hold(combined[-1], members[0], policy)
        for member in members:
            if combined and display_time(member.cue.start) < display_time(combined[-1].cue.end):
                raise ValueError("Measured candidate clocks overlap or are out of order")
            combined.append(member)
        if len(combined) > MAX_TOKENS:
            raise ValueError("Combined measured candidate budget exceeded")
    combined, hidden = drop_retained_hidden_prefixes(combined, rate=rate)
    combined, widened = ensure_display_frames(combined, rate=rate, duration=duration)
    for name, digest in bindings.items():
        path = Path(name)
        if not path.is_absolute():
            raise ValueError("Measured candidate inputs must be absolute paths")
        if file_digest(_checked_output(path)) != digest:
            raise ValueError("Measured candidate input fingerprint changed")
    NativeCaptionLayout._save(
        output,
        {
            "version": 1,
            "candidateOnly": True,
            "inputs": bindings,
            "canvas": [width, height],
            "renderCalibration": {"path": calibration_name, "sha256": calibration_digest},
            "policy": policy,
            "coordinateSystem": "top-left canvas; calibrated text origin",
            "candidates": [asdict(member) for member in combined],
            "hiddenPrefixesDropped": hidden,
            "captionsWidenedToOneFrame": widened,
            "editorCorrespondenceVerified": False,
            "completeMotionSupportVerified": False,
            "motionRenderVerified": False,
            "semanticPlacementApproved": False,
            "captionGeometryApproved": False,
            "aaaContrastApproved": False,
            "deliveryComplete": False,
        },
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Combine pinned measured candidates; not delivery approval"
    )
    parser.add_argument("output", type=Path)
    parser.add_argument(
        "--candidate", action="append", nargs=2, required=True, metavar=("PATH", "SHA256")
    )
    parser.add_argument("--calibration", nargs=2, required=True, metavar=("PATH", "SHA256"))
    parser.add_argument("--width", type=int, required=True)
    parser.add_argument("--height", type=int, required=True)
    parser.add_argument("--rate", type=int, required=True)
    parser.add_argument("--frames", type=int, required=True)
    args = parser.parse_args()
    if min(args.width, args.height, args.rate, args.frames) <= 0:
        parser.error("Canvas, rate and frame count must be positive")
    combine_measured_captions(
        [(Path(path), digest) for path, digest in args.candidate],
        args.output,
        width=args.width,
        height=args.height,
        rate=args.rate,
        duration=Fraction(args.frames, args.rate),
        calibration=(Path(args.calibration[0]), args.calibration[1]),
    )


if __name__ == "__main__":
    main()
