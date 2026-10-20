"""Check a rendered caption's colour gradient on its pixels, not on its settings.

The style's gradient once passed every test while the render showed none: Motion maps gradient
locations onto each line's text box, so stops placed on glyph fractions fell below the glyphs.
This audit reads a rendered alpha movie and measures, per text line, the fill colour (opaque,
saturated pixels; the white band and black outlines are excluded) in the line's upper fifth and
in its middle. The style's upper colour is the brighter one, so every measured line's upper
fill must be brighter than its middle by ``MINIMUM_LIFT`` (sRGB luma, 0-255).

    python -m replay.caption_gradient_audit ALPHA.mov
"""

from __future__ import annotations

import argparse
import json
import subprocess
from collections.abc import Iterator, Sequence

import numpy as np
from numpy.typing import NDArray

from replay.core import ReplayError

MINIMUM_LIFT = 12.0
SATURATION = 25  # max-min channel spread that separates the fill from white, grey and black
LINE_GAP = 4  # rows without fill that separate two lines
SAMPLE_STEP = 60
MINIMUM_LINES = 3
LUMA = np.array([0.2126, 0.7152, 0.0722])


def fill_mask(frame: NDArray[np.uint8]) -> NDArray[np.bool_]:
    """Opaque coloured pixels of an RGBA frame: the caption fill only."""
    rgb = frame[..., :3].astype(np.int16)
    spread = rgb.max(axis=2) - rgb.min(axis=2)
    return np.asarray((frame[..., 3] > 250) & (spread > SATURATION))


def line_lifts(frame: NDArray[np.uint8]) -> list[float]:
    """Per text line: luma of the upper fifth's fill minus the middle fifth's fill."""
    fill = fill_mask(frame)
    rows = np.nonzero(fill.sum(axis=1) > 3)[0]
    if not len(rows):
        return []
    lifts = []
    for run in np.split(rows, np.nonzero(np.diff(rows) > LINE_GAP)[0] + 1):
        top, height = int(run[0]), int(run[-1]) - int(run[0]) + 1
        upper = frame[top : top + height // 5, :, :3][fill[top : top + height // 5]]
        middle_rows = slice(top + 2 * height // 5, top + 3 * height // 5)
        middle = frame[middle_rows, :, :3][fill[middle_rows]]
        if len(upper) and len(middle):
            lifts.append(float((upper.mean(axis=0) - middle.mean(axis=0)) @ LUMA))
    return lifts


def frames(path: str, *, step: int = SAMPLE_STEP) -> Iterator[NDArray[np.uint8]]:
    """Every ``step``-th frame of a movie as RGBA."""
    probe = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
         "stream=width,height", "-of", "json", path],
        capture_output=True, text=True, check=True,
    )  # fmt: skip
    stream = json.loads(probe.stdout)["streams"][0]
    width, height = int(stream["width"]), int(stream["height"])
    command = ["ffmpeg", "-v", "error", "-nostdin", "-i", path, "-vf",
               f"select=not(mod(n\\,{step}))", "-fps_mode", "passthrough",
               "-f", "rawvideo", "-pix_fmt", "rgba", "-"]  # fmt: skip
    size = width * height * 4
    with subprocess.Popen(command, stdout=subprocess.PIPE) as process:
        if process.stdout is None:
            raise ReplayError("Missing FFmpeg frame stream")
        while len(block := process.stdout.read(size)) == size:
            yield np.frombuffer(block, np.uint8).reshape(height, width, 4)
    if process.returncode != 0:
        raise ReplayError(f"FFmpeg could not decode {path}")


def audit(lifts: Sequence[float]) -> dict[str, object]:
    """Pass when enough lines were measured and every one is brighter at the top."""
    if len(lifts) < MINIMUM_LINES:
        raise ReplayError(f"Only {len(lifts)} caption lines were measurable")
    weakest = min(lifts)
    if weakest < MINIMUM_LIFT:
        raise ReplayError(
            f"Rendered caption gradient is missing: a line's top is only {weakest:.1f} brighter"
        )
    return {"lines": len(lifts), "weakestLift": round(weakest, 1), "minimumLift": MINIMUM_LIFT}


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("movie")
    args = parser.parse_args(argv)
    lifts = [lift for frame in frames(args.movie) for lift in line_lifts(frame)]
    print(json.dumps(audit(lifts)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
