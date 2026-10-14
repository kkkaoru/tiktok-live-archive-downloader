"""Model sensitivity of supplied full-range, nonlinear BT.709 RGB-prime samples.

A BT.709 tag alone does not choose camera inversion versus display EOTF. These
explicit models are diagnostics, not proof of a current Apple viewing transform.
The CLI reads one RGBA64LE signal frame (matrix/range conversion only, NO transfer
conversion), retains 16-bit codes, and reports dominant opaque colours. It does
not segment glyphs, cover an entire render or approve a caption.
"""

import argparse
import json
import math
import os
from collections.abc import Mapping, Sequence
from pathlib import Path
from types import MappingProxyType
from typing import Literal

import numpy as np
from numpy.typing import NDArray

from replay.caption_contrast import (
    UPPER_MINIMUM,
    WHITE,
    Rgb,
    _hex_color,
    contrast_ratio,
    require_white_contrast,
    sampled_colour_correspondence,
)
from replay.caption_pixels import MAX_PIXELS
from replay.core import ReplayError
from replay.stages import file_digest

SignalModel = Literal["bt709-camera", "bt1886-display", "apple-video-1.961"]
SIGNAL_MODELS: tuple[SignalModel, ...] = (
    "bt709-camera",
    "bt1886-display",
    "apple-video-1.961",
)
SIGNAL_MAXIMUM = 65535
SIGNAL_DTYPE: np.dtype[np.uint16] = np.dtype("<u2")
MAX_REPORTED_CODES = 8
CAMERA_SLOPE = 4.5
CAMERA_OFFSET = 0.099
CAMERA_SCALE = 1.099
CAMERA_EXPONENT = 0.45
CAMERA_BREAK = 0.081
DISPLAY_GAMMA = 2.4
ARCHIVED_APPLE_GAMMA = 1.961
SRGB_LINEAR_BREAK = 0.0031308
SRGB_SLOPE = 12.92
SRGB_SCALE = 1.055
SRGB_OFFSET = 0.055
SRGB_EXPONENT = 2.4
MODEL_GAMMAS: Mapping[SignalModel, float] = MappingProxyType(
    {
        "bt709-camera": 1 / CAMERA_EXPONENT,
        "bt1886-display": DISPLAY_GAMMA,
        "apple-video-1.961": ARCHIVED_APPLE_GAMMA,
    }
)


def _decode(value: float, model: SignalModel) -> float:
    if model == "bt709-camera":
        return float(
            value / CAMERA_SLOPE
            if value < CAMERA_BREAK
            else ((value + CAMERA_OFFSET) / CAMERA_SCALE) ** MODEL_GAMMAS[model]
        )
    return float(value ** MODEL_GAMMAS[model])


def _srgb(linear: float) -> float:
    return float(
        linear * SRGB_SLOPE
        if linear <= SRGB_LINEAR_BREAK
        else SRGB_SCALE * linear ** (1 / SRGB_EXPONENT) - SRGB_OFFSET
    )


def signal_to_srgb(signal: Sequence[float], *, model: SignalModel) -> Rgb:
    """Apply an explicit SDR interpretation without clipping or 8-bit quantization."""
    if len(signal) != 3 or any(
        isinstance(value, bool) or not math.isfinite(value) or not 0 <= value <= 1
        for value in signal
    ):
        raise ReplayError(
            "Signal RGB requires three finite components in 0..1; do not clip excursions"
        )
    if model not in MODEL_GAMMAS:
        raise ReplayError("Unknown video signal interpretation")
    red, green, blue = (_srgb(_decode(value, model)) for value in signal)
    return red, green, blue


def _interpret(signal: Rgb, intent: Rgb | None, model: SignalModel) -> dict[str, object]:
    color = signal_to_srgb(signal, model=model)
    ratio = contrast_ratio(WHITE, color)
    correspondence = (
        None
        if intent is None
        else (ratio >= UPPER_MINIMUM and sampled_colour_correspondence((intent,), (color,)))
    )
    return {
        "sRGB": color,
        "sRGB8DisplayOnly": [round(value * 255) for value in color],
        "whiteContrast": ratio,
        "suppliedSampleCorrespondence": correspondence,
    }


def measure_signal(image: NDArray[np.uint16], *, intent: Rgb | None = None) -> dict[str, object]:
    """Count supplied opaque signal codes; decorations remain in the report."""
    if (
        image.dtype != SIGNAL_DTYPE
        or image.ndim != 3
        or image.shape[2] != 4
        or not 0 < image.shape[0] * image.shape[1] <= MAX_PIXELS
    ):
        raise ReplayError("Expected one bounded nonempty RGBA64LE signal frame")
    if intent is not None:
        require_white_contrast((intent,))
    opaque = image[:, :, 3] == SIGNAL_MAXIMUM
    colors, counts = np.unique(image[:, :, :3][opaque], axis=0, return_counts=True)
    order = np.argsort(-counts, kind="stable")[:MAX_REPORTED_CODES]
    records: list[dict[str, object]] = []
    for index in order:
        codes = [int(colors[index, channel]) for channel in range(3)]
        signal = (codes[0] / SIGNAL_MAXIMUM, codes[1] / SIGNAL_MAXIMUM, codes[2] / SIGNAL_MAXIMUM)
        models = {model: _interpret(signal, intent, model) for model in SIGNAL_MODELS}
        records.append({"rgb16": codes, "count": int(counts[index]), "models": models})
    return {
        "width": image.shape[1],
        "height": image.shape[0],
        "encoding": "full-range nonlinear BT.709 RGB-prime RGBA64LE",
        "opaquePixels": int(np.count_nonzero(opaque)),
        "reportedPixels": sum(int(counts[index]) for index in order),
        "uniqueOpaqueCodes": len(colors),
        "dominantOpaqueCodes": records,
        "selectedSRGB": intent,
        "coverage": "dominant-opaque-codes-only; not glyph segmentation",
        "transferInterpretationVerified": False,
        "renderedContrastVerified": False,
        "captionQualityApproved": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("signal", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--width", type=int, required=True)
    parser.add_argument("--height", type=int, required=True)
    parser.add_argument("--intent", type=_hex_color)
    args = parser.parse_args()
    pixels = args.width * args.height
    if min(args.width, args.height) <= 0 or pixels > MAX_PIXELS:
        raise ReplayError("Signal dimensions exceed the pixel budget")
    if (
        args.signal.is_symlink()
        or not args.signal.is_file()
        or args.signal.stat().st_size != pixels * 8
    ):
        raise ReplayError("Expected one complete regular RGBA64LE signal frame")
    before = file_digest(args.signal)
    data = args.signal.read_bytes()
    if len(data) != pixels * 8:
        raise ReplayError("Signal input changed during measurement")
    image = np.frombuffer(data, dtype=SIGNAL_DTYPE).reshape(args.height, args.width, 4)
    result = measure_signal(image, intent=args.intent)
    if file_digest(args.signal) != before:
        raise ReplayError("Signal input changed during measurement")
    result.update(source=str(args.signal.resolve()), sourceSHA256=before)
    with args.output.open("x", encoding="utf-8") as handle:
        os.fchmod(handle.fileno(), 0o600)
        json.dump(result, handle, indent=2)
        handle.write("\n")


if __name__ == "__main__":
    main()
