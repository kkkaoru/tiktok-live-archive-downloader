"""Per-frame repair regions from sampled OCR regions and detector regions (never raw OCR).

Every frame gets the bounding rectangle of the two bracketing sampled-OCR regions united with
the temporally widened detector region. A region is a conservative search window; the
repair stage still detects the actual glyph pixels inside it for every frame.
"""

from collections.abc import Mapping, Sequence

Region = tuple[int, int, int, int]  # x0, y0, x1, y1 (half-open pixel box)


def union(regions: Sequence[Region]) -> Region:
    """Smallest rectangle containing every region."""
    if not regions:
        raise ValueError("Cannot unite zero regions")
    return (
        min(region[0] for region in regions),
        min(region[1] for region in regions),
        max(region[2] for region in regions),
        max(region[3] for region in regions),
    )


def widen(regions: Sequence[Region | None], radius: int) -> tuple[Region | None, ...]:
    """Union each frame's region with its neighbours within radius frames (None = no text)."""
    if radius < 0:
        raise ValueError("Radius must not be negative")
    widened: list[Region | None] = []
    for frame in range(len(regions)):
        near = [
            region
            for region in regions[max(0, frame - radius) : frame + radius + 1]
            if region is not None
        ]
        widened.append(union(near) if near else None)
    return tuple(widened)


def bracket(count: int, sampled: Mapping[int, Region | None]) -> tuple[Region | None, ...]:
    """Each frame takes the union of its previous and next sampled region."""
    indices = sorted(sampled)
    if not indices or indices[0] != 0 or indices[-1] != count - 1:
        raise ValueError("Samples must include the first and last frame")
    result: list[Region | None] = []
    for frame in range(count):
        previous = max(index for index in indices if index <= frame)
        following = min(index for index in indices if index >= frame)
        pair = [region for region in (sampled[previous], sampled[following]) if region is not None]
        result.append(union(pair) if pair else None)
    return tuple(result)


def combine(
    sampled: Sequence[Region | None], detected: Sequence[Region | None]
) -> tuple[Region | None, ...]:
    """Per-frame union of the sampled-OCR fill and the detector fill."""
    if len(sampled) != len(detected):
        raise ValueError("Region sequences cover different frame counts")
    combined: list[Region | None] = []
    for first, second in zip(sampled, detected, strict=True):
        pair = [region for region in (first, second) if region is not None]
        combined.append(union(pair) if pair else None)
    return tuple(combined)


def coverage_gaps(
    required: Sequence[Region | None], filled: Sequence[Region | None]
) -> tuple[int, ...]:
    """Frames whose required region is not contained in the filled region."""
    if len(required) != len(filled):
        raise ValueError("Region sequences cover different frame counts")
    gaps: list[int] = []
    for frame, (need, have) in enumerate(zip(required, filled, strict=True)):
        if need is None:
            continue
        if have is None or union([need, have]) != have:
            gaps.append(frame)
    return tuple(gaps)
