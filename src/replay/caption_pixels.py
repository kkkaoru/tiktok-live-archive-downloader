"""Exact support and colour-code counts for bounded, decoded RGBA caption frames.

No crop-detection heuristic, font metrics or inferred colour-space conversion.
RGB counts describe the caller's decoded bytes, not automatic WCAG approval.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from replay.core import ReplayError
from replay.stages import file_digest

MAX_PIXELS = 4096 * 2160


def _bounds(mask: NDArray[np.bool_]) -> list[int] | None:
    rows, columns = np.nonzero(mask)
    if not rows.size:
        return None
    return [int(columns.min()), int(rows.min()), int(columns.max()) + 1, int(rows.max()) + 1]


def measure(image: NDArray[np.uint8]) -> dict[str, object]:
    """Count every alpha sample, retaining sparse edges and half-open bounds."""
    if (
        image.dtype != np.uint8
        or image.ndim != 3
        or image.shape[2] != 4
        or not 0 < image.shape[0] * image.shape[1] <= MAX_PIXELS
    ):
        raise ReplayError("Expected a bounded nonempty RGBA8 frame")
    alpha = image[:, :, 3]
    opaque = alpha == 255
    colors, counts = np.unique(image[:, :, :3][opaque], axis=0, return_counts=True)
    order = np.argsort(-counts, kind="stable")[:8]
    return {
        "width": image.shape[1],
        "height": image.shape[0],
        "nonzeroAlphaPixels": int(np.count_nonzero(alpha)),
        "opaquePixels": int(np.count_nonzero(opaque)),
        "nonzeroAlphaBounds": _bounds(alpha > 0),
        "opaqueBounds": _bounds(opaque),
        "dominantOpaqueRGB": [
            {"rgb": colors[index].tolist(), "count": int(counts[index])} for index in order
        ],
        "colourSpaceVerified": False,
        "captionQualityApproved": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("rgba", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--width", type=int, required=True)
    parser.add_argument("--height", type=int, required=True)
    args = parser.parse_args()
    pixels = args.width * args.height
    if min(args.width, args.height) <= 0 or pixels > MAX_PIXELS:
        raise ReplayError("RGBA dimensions exceed the pixel budget")
    if args.rgba.is_symlink() or not args.rgba.is_file() or args.rgba.stat().st_size != pixels * 4:
        raise ReplayError("Expected one complete regular RGBA frame")
    before = file_digest(args.rgba)
    data = args.rgba.read_bytes()
    if len(data) != pixels * 4:
        raise ReplayError("RGBA input changed during measurement")
    image = np.frombuffer(data, dtype=np.uint8).reshape(args.height, args.width, 4)
    result = measure(image)
    if file_digest(args.rgba) != before:
        raise ReplayError("RGBA input changed during measurement")
    result.update(source=str(args.rgba.resolve()), sourceSHA256=before)
    with args.output.open("x", encoding="utf-8") as handle:
        json.dump(result, handle, ensure_ascii=False, indent=2)
        handle.write("\n")


if __name__ == "__main__":
    main()
