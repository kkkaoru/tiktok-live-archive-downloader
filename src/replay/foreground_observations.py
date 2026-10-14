"""Validate exact-frame foreground measurements, never face or caption approval."""

from dataclasses import dataclass
from fractions import Fraction
from itertools import pairwise

from replay.core import ReplayError
from replay.recordings import record
from replay.vertical_captions import Box

MAX_SAMPLES = 8
MAX_DIMENSION = 16384
MAX_PIXELS = 16_777_216
MAX_TIMESCALE = 1_000_000_000


@dataclass(frozen=True, slots=True)
class ForegroundMask:
    nonzero_pixels: int
    bounds: Box | None


@dataclass(frozen=True, slots=True)
class ForegroundFrame:
    time: Fraction
    labels: tuple[int, ...] | None
    mask: ForegroundMask | None


def _integer(value: object) -> int:
    if type(value) is not int:
        raise ReplayError("Foreground measurements require integer fields")
    return value


def _time(value: object) -> Fraction:
    raw = record(value)
    ticks, scale = _integer(raw.get("value")), _integer(raw.get("timescale"))
    if ticks < 0 or not 1 <= scale <= MAX_TIMESCALE:
        raise ReplayError("Invalid foreground rational clock")
    return Fraction(ticks, scale)


def _mask(value: object, *, width: int, height: int) -> ForegroundMask:
    raw = record(value)
    if _integer(raw.get("width")) != width or _integer(raw.get("height")) != height:
        raise ReplayError("Foreground mask dimensions differ from source")
    pixels = _integer(raw.get("nonzeroPixels"))
    if not 0 <= pixels <= width * height:
        raise ReplayError("Invalid foreground pixel count")
    bounds = raw.get("nonzeroBounds")
    if pixels == 0:
        if bounds is not None:
            raise ReplayError("Empty foreground mask cannot have bounds")
        return ForegroundMask(0, None)
    box = record(bounds)
    x, y = _integer(box.get("x")), _integer(box.get("y"))
    w, h = _integer(box.get("width")), _integer(box.get("height"))
    if not 0 <= x < x + w <= width or not 0 <= y < y + h <= height or pixels > w * h:
        raise ReplayError("Foreground bounds or count exceed their measured extent")
    return ForegroundMask(pixels, Box(x, y, x + w, y + h))


def foreground_frames(
    value: object, *, times: tuple[Fraction, ...], width: int, height: int
) -> tuple[ForegroundFrame, ...]:
    """Preserve absent/empty masks; local instance labels are not persistent identity.

    The caller binds media, native executable, requests and responses. Nonzero L8
    bounds are a model/quantization result, not pixel coverage of their rectangle,
    head geometry, absence of a subject, or a safe caption position.
    """
    if (
        type(width) is not int
        or type(height) is not int
        or not 1 <= width <= MAX_DIMENSION
        or not 1 <= height <= MAX_DIMENSION
        or width * height > MAX_PIXELS
    ):
        raise ReplayError("Foreground source geometry budget exceeded")
    if (
        not isinstance(times, tuple)
        or not 1 <= len(times) <= MAX_SAMPLES
        or any(
            not isinstance(time, Fraction)
            or not 0 <= time <= 86400
            or time.denominator > MAX_TIMESCALE
            for time in times
        )
        or any(right <= left for left, right in pairwise(times))
    ):
        raise ReplayError("Invalid expected foreground sample clocks")
    if not isinstance(value, list) or len(value) != len(times):
        raise ReplayError("Foreground response frame count differs")
    result = []
    for item, expected in zip(value, times, strict=True):
        raw = record(item)
        if _time(raw.get("requested")) != expected or _time(raw.get("actual")) != expected:
            raise ReplayError("Foreground requested/actual source clock differs")
        if _integer(raw.get("width")) != width or _integer(raw.get("height")) != height:
            raise ReplayError("Foreground display dimensions differ")
        if (
            raw.get("captionPlacementApproved") is not False
            or _integer(raw.get("maskThreshold8Bit")) != 1
        ):
            raise ReplayError(
                "Foreground observations cannot approve placement or change mask threshold"
            )
        present, labels, mask = (
            raw.get("observationPresent"),
            raw.get("instanceLabels"),
            raw.get("unionMask"),
        )
        if type(present) is not bool:
            raise ReplayError("Foreground observation presence must be explicit")
        if not present:
            if labels is not None or mask is not None:
                raise ReplayError("Absent foreground observation contains geometry")
            result.append(ForegroundFrame(expected, None, None))
            continue
        if not isinstance(labels, list):
            raise ReplayError("Foreground instance labels are missing")
        identifiers = tuple(_integer(label) for label in labels)
        if (
            any(not 1 <= label <= 2**63 - 1 for label in identifiers)
            or tuple(sorted(set(identifiers))) != identifiers
        ):
            raise ReplayError("Foreground instance labels must be positive, unique and ordered")
        if not identifiers:
            if mask is not None:
                raise ReplayError("No foreground instances can have a union mask")
            result.append(ForegroundFrame(expected, (), None))
            continue
        result.append(
            ForegroundFrame(expected, identifiers, _mask(mask, width=width, height=height))
        )
    return tuple(result)
