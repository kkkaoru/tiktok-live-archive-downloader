"""Measured uniform scaling evidence for indivisible, finely timed caption units."""

from dataclasses import asdict, dataclass
from math import isfinite

from replay.core import ReplayError
from replay.recordings import record

FOCUSED_TIMING_SECONDS = 2.8


@dataclass(frozen=True, slots=True)
class CaptionScale:
    width: float
    height: float
    available_width: float
    available_height: float
    fragments: int
    maximum_fragment_seconds: float
    provenance: str = "indivisible-word"

    def __post_init__(self) -> None:
        numbers = (
            self.width,
            self.height,
            self.available_width,
            self.available_height,
            self.maximum_fragment_seconds,
        )
        if (
            any(isinstance(value, bool) or not isfinite(value) or value <= 0 for value in numbers)
            or type(self.fragments) is not int
            or not 2 <= self.fragments <= 16384
            or self.maximum_fragment_seconds > FOCUSED_TIMING_SECONDS
            or not 0 < self.factor < 1
            or self.provenance not in {"indivisible-word", "recognition-anchor-group"}
        ):
            raise ReplayError("Invalid indivisible caption scale evidence")

    @property
    def factor(self) -> float:
        return min(1.0, self.available_width / self.width, self.available_height / self.height)


def scale_fields(scale: CaptionScale | None) -> dict[str, object]:
    if scale is None:
        return {}
    raw = asdict(scale)
    raw.pop("provenance")
    key = "indivisibleScale" if scale.provenance == "indivisible-word" else "anchoredGroupScale"
    return {key: raw}


def read_scale(row: dict[str, object]) -> CaptionScale | None:
    if "indivisibleScale" in row and "anchoredGroupScale" in row:
        raise ReplayError("Ambiguous caption scale provenance")
    anchored = "anchoredGroupScale" in row
    value = row.get("anchoredGroupScale" if anchored else "indivisibleScale")
    if value is None:
        return None
    raw = record(value)
    if set(raw) != {
        "width",
        "height",
        "available_width",
        "available_height",
        "fragments",
        "maximum_fragment_seconds",
    }:
        raise ReplayError("Unexpected caption scale evidence fields")
    numbers: dict[str, float] = {}
    for name in (
        "width",
        "height",
        "available_width",
        "available_height",
        "maximum_fragment_seconds",
    ):
        number = raw[name]
        if isinstance(number, bool) or not isinstance(number, (int, float)):
            raise ReplayError("Expected measured caption scale number")
        numbers[name] = float(number)
    fragments = raw["fragments"]
    if type(fragments) is not int:
        raise ReplayError("Expected caption scale fragment count")
    return CaptionScale(
        **numbers,
        fragments=fragments,
        provenance="recognition-anchor-group" if anchored else "indivisible-word",
    )
