"""WCAG contrast arithmetic for externally colour-managed sRGB caption samples.

Inputs are sRGB-encoded components in 0..1 that the caller has already
converted from decoded output using its verified matrix, range, primaries and
transfer function. Merely relabelling BT.709 code values as sRGB is not conversion.
Motion/FCP controls have an unestablished encoding and are not accepted here.
Only explicitly selected or independently converted sRGB values are meaningful.
Callers also supply the upper/lower/intermediate segmentation and its
evidence; this module neither segments glyphs nor certifies a whole image.
"""

import argparse
import json
import math
from collections.abc import Sequence
from dataclasses import dataclass

from replay.core import ReplayError

Rgb = tuple[float, float, float]

WHITE: Rgb = (1.0, 1.0, 1.0)
SRGB_LINEAR_LIMIT: float = 0.04045
SRGB_LINEAR_SLOPE: float = 12.92
SRGB_OFFSET: float = 0.055
SRGB_SCALE: float = 1.055
SRGB_EXPONENT: float = 2.4
LUMINANCE_WEIGHTS: Rgb = (0.2126, 0.7152, 0.0722)
FLARE: float = 0.05
RGB_COMPONENTS: int = 3
UPPER_MINIMUM: float = 7.0
LOWER_MINIMUM: float = 7.0
INTERMEDIATE_MINIMUM: float = 7.0


@dataclass(frozen=True, slots=True)
class SampledContrast:
    """Unrounded minimum ratios against white; ``failures`` names groups below threshold."""

    upper_minimum: float
    lower_minimum: float
    intermediate_minimum: float
    failures: tuple[str, ...]


def _linear(channel: float) -> float:
    if isinstance(channel, bool) or not math.isfinite(channel) or not 0.0 <= channel <= 1.0:
        raise ReplayError("sRGB components must be finite and within 0..1")
    if channel <= SRGB_LINEAR_LIMIT:
        return channel / SRGB_LINEAR_SLOPE
    return float(((channel + SRGB_OFFSET) / SRGB_SCALE) ** SRGB_EXPONENT)


def relative_luminance(color: Sequence[float]) -> float:
    """W3C relative luminance of one sRGB-encoded colour."""
    if len(color) != RGB_COMPONENTS:
        raise ReplayError("sRGB colours must have exactly three components")
    red, green, blue = (_linear(channel) for channel in color)
    red_weight, green_weight, blue_weight = LUMINANCE_WEIGHTS
    return red_weight * red + green_weight * green + blue_weight * blue


def contrast_ratio(first: Sequence[float], second: Sequence[float]) -> float:
    """W3C contrast ratio (lighter + 0.05) / (darker + 0.05), independent of order."""
    first_luminance = relative_luminance(first)
    second_luminance = relative_luminance(second)
    lighter = max(first_luminance, second_luminance)
    darker = min(first_luminance, second_luminance)
    return (lighter + FLARE) / (darker + FLARE)


def require_white_contrast(colors: Sequence[Sequence[float]]) -> tuple[float, ...]:
    """Reject any selected sRGB fill below 7:1 against white, without rounding."""
    if not colors:
        raise ReplayError("Select at least one caption colour")
    ratios = tuple(contrast_ratio(WHITE, color) for color in colors)
    if min(ratios) < UPPER_MINIMUM:
        raise ReplayError(f"Caption fill/white contrast {min(ratios):.8f}:1 is below 7:1")
    return ratios


def _minimum_ratio(group: str, samples: Sequence[Sequence[float]]) -> float:
    if not samples:
        raise ReplayError(f"No sampled {group} caption pixels; contrast is unproven")
    return min(contrast_ratio(WHITE, sample) for sample in samples)


def audit_sampled_contrast(
    *,
    upper: Sequence[Sequence[float]],
    lower: Sequence[Sequence[float]],
    intermediate: Sequence[Sequence[float]],
) -> SampledContrast:
    """Check every caller-segmented fill group against white at 7:1.

    Every group is mandatory: an empty group raises instead of passing vacuously.
    Thresholds compare unrounded ratios. The result covers only the supplied
    samples, not unsampled pixels, frames, or full-image WCAG conformance.
    """
    upper_minimum = _minimum_ratio("upper", upper)
    lower_minimum = _minimum_ratio("lower", lower)
    intermediate_minimum = _minimum_ratio("intermediate", intermediate)
    checks = (
        ("upper", upper_minimum, UPPER_MINIMUM),
        ("lower", lower_minimum, LOWER_MINIMUM),
        ("intermediate", intermediate_minimum, INTERMEDIATE_MINIMUM),
    )
    failures = tuple(group for group, ratio, minimum in checks if ratio < minimum)
    return SampledContrast(upper_minimum, lower_minimum, intermediate_minimum, failures)


def sampled_colour_correspondence(
    selected: Sequence[Sequence[float]], rendered: Sequence[Sequence[float]]
) -> bool:
    """Compare paired sRGB samples at 8-bit precision, allowing one code step.

    This checks supplied samples only. The caller must establish conversion,
    stop correspondence, frame coverage and provenance independently. Near-black
    cannot stand in for selected cyan merely because both exceed 7:1.
    """
    require_white_contrast(selected)
    require_white_contrast(rendered)
    if len(selected) != len(rendered):
        raise ReplayError("Selected and rendered colour samples must correspond one-to-one")
    return all(
        abs(round(wanted * 255) - round(actual * 255)) <= 1
        for target, sample in zip(selected, rendered, strict=True)
        for wanted, actual in zip(target, sample, strict=True)
    )


def _hex_color(value: str) -> Rgb:
    value = value.removeprefix("#")
    if len(value) != 6 or any(char not in "0123456789abcdefABCDEF" for char in value):
        raise argparse.ArgumentTypeError("Expected an sRGB hex colour, e.g. #005A9C")
    return (int(value[:2], 16) / 255, int(value[2:4], 16) / 255, int(value[4:], 16) / 255)


def main() -> None:
    parser = argparse.ArgumentParser(description="Check selected sRGB fills against white at 7:1")
    parser.add_argument("colors", type=_hex_color, nargs="+")
    parser.add_argument("--rendered-colors", type=_hex_color, nargs="+")
    args = parser.parse_args()
    ratios = [contrast_ratio(WHITE, color) for color in args.colors]
    correspondence: bool | None = None
    if args.rendered_colors is not None:
        try:
            correspondence = sampled_colour_correspondence(args.colors, args.rendered_colors)
        except ReplayError:
            correspondence = False
    passed = min(ratios) >= UPPER_MINIMUM and correspondence is not False
    print(
        json.dumps(
            {
                "ratios": ratios,
                "minimum": min(ratios),
                "required": 7.0,
                "passed": passed,
                "renderedContrastVerified": False,
                "suppliedSampleCorrespondence": correspondence,
            }
        )
    )
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
