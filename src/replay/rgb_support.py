"""Bounded RGB predicate support, never recovered alpha or native paint approval."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from numpy.typing import NDArray

from replay.caption_pixels import MAX_PIXELS, _bounds
from replay.stages import file_digest

MAX_BYTES = 32 * 1024 * 1024


@dataclass(frozen=True)
class Predicate:
    name: str
    lower: tuple[int, int, int]
    upper: tuple[int, int, int]

    def __post_init__(self) -> None:
        if (
            not self.name
            or len(self.name) > 64
            or any(
                type(low) is not int or type(high) is not int or not 0 <= low <= high <= 255
                for low, high in zip(self.lower, self.upper, strict=True)
            )
        ):
            raise ValueError("Invalid RGB predicate")


def measure(
    image: NDArray[np.uint8], *, region: tuple[int, int, int, int], predicate: Predicate
) -> dict[str, object]:
    """Measure all matching pixels without morphology or background assumptions."""
    if image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("Expected RGB8 image")
    left, top, right, bottom = region
    if not (
        0 < image.shape[0] * image.shape[1] <= MAX_PIXELS
        and 0 <= left < right <= image.shape[1]
        and 0 <= top < bottom <= image.shape[0]
    ):
        raise ValueError("Invalid bounded RGB region")
    crop = image[top:bottom, left:right]
    support = np.all((crop >= predicate.lower) & (crop <= predicate.upper), axis=2)
    bounds = _bounds(support)
    clipped = bounds is not None and (
        bounds[0] == 0 or bounds[1] == 0 or bounds[2] == right - left or bounds[3] == bottom - top
    )
    absolute = (
        None
        if bounds is None
        else [bounds[0] + left, bounds[1] + top, bounds[2] + left, bounds[3] + top]
    )
    colors, counts = np.unique(crop[support], axis=0, return_counts=True)
    order = np.argsort(-counts, kind="stable")[:16]
    return {
        "label": predicate.name,
        "lowerRGB": predicate.lower,
        "upperRGB": predicate.upper,
        "region": region,
        "pixels": int(np.count_nonzero(support)),
        "boundsHalfOpen": absolute,
        "touchesRegionBoundary": clipped,
        "rowCounts": np.count_nonzero(support, axis=1).tolist(),
        "dominantRGB": [{"rgb": colors[i].tolist(), "count": int(counts[i])} for i in order],
        "uncertainty": "no-support"
        if bounds is None
        else "clipped-or-background"
        if clipped
        else "colour-support-only-background-and-shadow-unresolved",
        "nativePaintAccepted": False,
    }


def run(
    source: Path, output: Path, *, region: tuple[int, int, int, int], predicate: Predicate
) -> None:
    """Read a pinned bounded PNG and retain input identity with exclusive output."""
    if source.is_symlink() or not source.is_file() or not 0 < source.stat().st_size <= MAX_BYTES:
        raise ValueError("Expected bounded regular PNG")
    before = file_digest(source)
    header = source.read_bytes()[:24]
    if header[:8] != b"\x89PNG\r\n\x1a\n" or len(header) != 24:
        raise ValueError("Expected PNG header")
    width, height = int.from_bytes(header[16:20]), int.from_bytes(header[20:24])
    if not 0 < width * height <= MAX_PIXELS:
        raise ValueError("PNG pixel budget exceeded")
    decoded = cv2.imread(str(source), cv2.IMREAD_COLOR)
    if decoded is None or decoded.shape[:2] != (height, width):
        raise ValueError("PNG decode differs from declared canvas")
    image = np.asarray(decoded[:, :, ::-1], dtype=np.uint8)
    result = measure(image, region=region, predicate=predicate)
    if file_digest(source) != before:
        raise ValueError("PNG input changed")
    result.update(source=str(source.resolve()), sourceSHA256=before)
    with output.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2)
        stream.write("\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--region", type=int, nargs=4, required=True)
    parser.add_argument("--lower", type=int, nargs=3, required=True)
    parser.add_argument("--upper", type=int, nargs=3, required=True)
    parser.add_argument("--label", required=True)
    args = parser.parse_args()
    run(
        args.source,
        args.output,
        region=tuple(args.region),
        predicate=Predicate(args.label, tuple(args.lower), tuple(args.upper)),
    )


if __name__ == "__main__":
    main()
