"""Measure how closely a candidate caption renderer matches Motion's alpha renders.

A native renderer may replace the Motion/Compressor caption alpha only if its captions look the
same. Two alpha renders of the same captions (RGBA as FFmpeg decodes ProRes 4444, straight alpha)
are compared frame by frame, inside the union of both alphas dilated by ``SUPPORT_DILATION``:

- SSIM (Wang et al., TIP 2004) on premultiplied RGBA; alpha is a channel because the rim and the
  shadow are black, so premultiplied RGB alone would not see them;
- FLIP (Andersson et al., HPG 2020; this is a NumPy port of the LDR variant, not the reference
  code) after compositing both renders over mid-grey: pooled mean and the mean per-frame 99th
  percentile;
- CIEDE2000 between the fills' mean colours at 10/20/30/50% of each text line's ink height;
- the edge profile, mean alpha against the distance from the fill: where it falls to one half,
  its 90%-to-10% fall width, where the shadow fades below 5%, and where the white band ends;
- each text line's placement: the displacement of every glyph-sized window of its fill's column
  profile and of its row profile (the sub-pixel peak of their cross-correlation, so threshold
  noise at anti-aliased edges does not move it as it moves a centroid), and its ink bottom.

Motion renders a document bit-exactly, so run-to-run noise is zero. The floor is therefore Motion
against Motion with every caption shifted half a pixel in x and y, the sub-pixel glyph placement
a second rasterizer cannot be expected to beat. A candidate passes a metric when its error is at
most ``FLOOR_MULTIPLE`` times the floor's, or ``MINIMUM_ERROR`` where the floor is about zero,
and always within the absolute ``GUARD`` (fill colour, border width at half alpha), which does
not depend on the shift. A floor measured once for the same style can be reused as JSON (the
``floor`` object of an earlier result).

    python -m replay.caption_parity MOTION.mov CANDIDATE.mov --floor MOTION_SHIFTED.mov
    python -m replay.caption_parity MOTION.mov CANDIDATE.mov --floor-json EARLIER_RESULT.json
"""

from __future__ import annotations

import argparse
import json
import math
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Final

import cv2
import numpy as np
from numpy.typing import NDArray

from replay.caption_gradient_audit import _lines, fill_mask, frames
from replay.core import ReplayError

Image = NDArray[np.float64]

# Pinned acceptance: one place for every threshold.
FLOOR_MULTIPLE: Final[float] = 1.0
MINIMUM_ERROR: Final[dict[str, float]] = {
    "ssimLoss": 0.002,
    "ssimWorstLoss": 0.01,
    "flipMean": 0.005,
    "flipP99": 0.05,
    "deltaE10": 1.0,
    "deltaE20": 1.0,
    "deltaE30": 1.0,
    "deltaE50": 1.0,
    "alphaHalf": 0.25,
    "alphaFall": 0.25,
    "shadowFade": 0.5,
    "whiteHalf": 0.25,
    "glyphX": 0.25,
    "glyphY": 0.25,
    "baseline": 0.5,
    "lineMismatch": 0.0,
}
GUARD: Final[dict[str, float]] = {
    "deltaE10": 2.0,
    "deltaE20": 2.0,
    "deltaE30": 2.0,
    "deltaE50": 2.0,
    "alphaHalf": 0.5,
}
FRACTIONS: Final[tuple[int, ...]] = (10, 20, 30, 50)
SUPPORT_DILATION: Final[int] = 5  # the SSIM window's radius
CROP_MARGIN: Final[int] = 24  # covers the FLIP filters' radius
SSIM_SIGMA: Final[float] = 1.5
SSIM_C1: Final[float] = 0.01**2
SSIM_C2: Final[float] = 0.03**2
GREY: Final[float] = 0.5
MIN_SAMPLES: Final[int] = 8  # fill pixels a sampled row band needs
LINE_ROWS: Final[int] = 3  # rows with fill that make a text line
LINE_GAP: Final[int] = 4
MAX_LAG: Final[int] = 6  # pixels searched for a glyph displacement
# Edge-profile distance bin, pixels: one pixel, so every bin mixes axis and diagonal distances.
BIN: Final[float] = 1.0
MAX_DISTANCE: Final[float] = 48.0
WHITE_MINIMUM: Final[int] = 200
WHITE_SPREAD: Final[int] = 25
OPAQUE: Final[int] = 250
# FLIP (LDR) constants from the paper: viewing at 67 pixels per degree.
PIXELS_PER_DEGREE: Final[float] = 67.0
CSF: Final[tuple[tuple[float, float, float, float], ...]] = (
    (1.0, 0.0047, 0.0, 1e-5),
    (1.0, 0.0053, 0.0, 1e-5),
    (34.1, 0.04, 13.5, 0.025),
)
FLIP_QC: Final[float] = 0.7
FLIP_PC: Final[float] = 0.4
FLIP_PT: Final[float] = 0.95
FLIP_QF: Final[float] = 0.5
FEATURE_WIDTH: Final[float] = 0.082  # degrees
RGB_TO_XYZ: Final[Image] = np.array(
    [[0.4124564, 0.3575761, 0.1804375],
     [0.2126729, 0.7151522, 0.0721750],
     [0.0193339, 0.1191920, 0.9503041]]
)  # fmt: skip
XYZ_TO_RGB: Final[Image] = np.linalg.inv(RGB_TO_XYZ)
WHITE: Final[Image] = RGB_TO_XYZ.sum(axis=1)


@dataclass(slots=True)
class Tally:
    """Per-frame measurements of one comparison, pooled at the end."""

    ssim: list[float] = field(default_factory=list)
    flip_sum: float = 0.0
    flip_count: int = 0
    flip_p99: list[float] = field(default_factory=list)
    delta_e: dict[int, list[float]] = field(default_factory=lambda: {f: [] for f in FRACTIONS})
    glyph_x: list[float] = field(default_factory=list)
    glyph_y: list[float] = field(default_factory=list)
    baseline: list[float] = field(default_factory=list)
    mismatched: int = 0
    frames: int = 0
    profiles: NDArray[np.float64] = field(
        default_factory=lambda: np.zeros((2, 3, int(MAX_DISTANCE / BIN)))
    )


@dataclass(frozen=True, slots=True)
class Line:
    """A text line's ink rows, its fill's column and row profiles and the fill colour at each
    sampled fraction."""

    top: int
    bottom: int
    columns: Image
    rows: Image
    colours: dict[int, Image]


def srgb_to_linear(rgb: Image) -> Image:
    return np.where(rgb <= 0.04045, rgb / 12.92, ((rgb + 0.055) / 1.055) ** 2.4)


def xyz_to_lab(xyz: Image) -> Image:
    t = xyz / WHITE
    delta = 6 / 29
    f = np.where(t > delta**3, np.cbrt(np.maximum(t, 0)), t / (3 * delta**2) + 4 / 29)
    return np.stack(
        (116 * f[..., 1] - 16, 500 * (f[..., 0] - f[..., 1]), 200 * (f[..., 1] - f[..., 2])),
        axis=-1,
    )


def srgb_to_lab(rgb: Image) -> Image:
    """CIELAB (D65) of sRGB values in [0, 1]."""
    return xyz_to_lab(srgb_to_linear(rgb) @ RGB_TO_XYZ.T)


def delta_e2000(first: Image, second: Image) -> Image:
    """CIEDE2000 colour difference (Sharma, Wu and Dalal 2005) between CIELAB arrays."""
    l1, a1, b1 = first[..., 0], first[..., 1], first[..., 2]
    l2, a2, b2 = second[..., 0], second[..., 1], second[..., 2]
    chroma = (np.hypot(a1, b1) + np.hypot(a2, b2)) / 2
    g = 0.5 * (1 - np.sqrt(chroma**7 / (chroma**7 + 25.0**7)))
    a1p, a2p = (1 + g) * a1, (1 + g) * a2
    c1p, c2p = np.hypot(a1p, b1), np.hypot(a2p, b2)
    h1p = np.degrees(np.arctan2(b1, a1p)) % 360
    h2p = np.degrees(np.arctan2(b2, a2p)) % 360
    neutral = c1p * c2p == 0
    dh = h2p - h1p
    dh = np.where(dh > 180, dh - 360, np.where(dh < -180, dh + 360, dh))
    dh = np.where(neutral, 0, dh)
    big_h = 2 * np.sqrt(c1p * c2p) * np.sin(np.radians(dh / 2))
    mean_l, mean_c = (l1 + l2) / 2, (c1p + c2p) / 2
    total = h1p + h2p
    mean_h = np.where(
        np.abs(h1p - h2p) <= 180, total / 2, np.where(total < 360, total + 360, total - 360) / 2
    )
    mean_h = np.where(neutral, total, mean_h)
    t = (
        1
        - 0.17 * np.cos(np.radians(mean_h - 30))
        + 0.24 * np.cos(np.radians(2 * mean_h))
        + 0.32 * np.cos(np.radians(3 * mean_h + 6))
        - 0.20 * np.cos(np.radians(4 * mean_h - 63))
    )
    rotation = 30 * np.exp(-(((mean_h - 275) / 25) ** 2))
    rc = 2 * np.sqrt(mean_c**7 / (mean_c**7 + 25.0**7))
    sl = 1 + 0.015 * (mean_l - 50) ** 2 / np.sqrt(20 + (mean_l - 50) ** 2)
    sc, sh = 1 + 0.045 * mean_c, 1 + 0.015 * mean_c * t
    rt = -np.sin(np.radians(2 * rotation)) * rc
    dl, dc = (l2 - l1) / sl, (c2p - c1p) / sc
    dhh = big_h / sh
    return np.asarray(np.sqrt(dl**2 + dc**2 + dhh**2 + rt * dc * dhh))


def ssim(first: Image, second: Image, support: NDArray[np.bool_]) -> float:
    """Mean SSIM over every channel inside the support (images in [0, 1])."""

    def blur(image: Image) -> Image:
        return np.asarray(cv2.GaussianBlur(image, (11, 11), SSIM_SIGMA))

    mu1, mu2 = blur(first), blur(second)
    var1 = blur(first * first) - mu1**2
    var2 = blur(second * second) - mu2**2
    cov = blur(first * second) - mu1 * mu2
    index = ((2 * mu1 * mu2 + SSIM_C1) * (2 * cov + SSIM_C2)) / (
        (mu1**2 + mu2**2 + SSIM_C1) * (var1 + var2 + SSIM_C2)
    )
    return float(index[support].mean())


def _csf_kernel(a1: float, b1: float, a2: float, b2: float) -> Image:
    radius = math.ceil(3 * math.sqrt(max(b1, b2) / (2 * math.pi**2)) * PIXELS_PER_DEGREE)
    x = np.arange(-radius, radius + 1) / PIXELS_PER_DEGREE
    squared = x[None, :] ** 2 + x[:, None] ** 2
    kernel = a1 * np.sqrt(np.pi / b1) * np.exp(-(np.pi**2) * squared / b1) + a2 * np.sqrt(
        np.pi / b2
    ) * np.exp(-(np.pi**2) * squared / b2)
    return np.asarray(kernel / kernel.sum())


def _feature_kernels() -> tuple[Image, Image]:
    """Gaussian first- and second-derivative kernels, each sign normalised to unit mass."""
    sigma = 0.5 * FEATURE_WIDTH * PIXELS_PER_DEGREE
    radius = math.ceil(3 * sigma)
    x = np.arange(-radius, radius + 1, dtype=np.float64)
    gauss = np.exp(-(x[None, :] ** 2 + x[:, None] ** 2) / (2 * sigma**2))
    kernels = []
    for kernel in (-x[None, :] * gauss, (x[None, :] ** 2 / sigma**2 - 1) * gauss):
        positive, negative = kernel.clip(min=0), kernel.clip(max=0)
        kernels.append(positive / positive.sum() - negative / negative.sum())
    return kernels[0], kernels[1]


CSF_KERNELS: Final[tuple[Image, ...]] = tuple(_csf_kernel(*row) for row in CSF)
EDGE_KERNEL, POINT_KERNEL = _feature_kernels()


def _filter(image: Image, kernel: Image) -> Image:
    return np.asarray(cv2.filter2D(image, -1, kernel, borderType=cv2.BORDER_REPLICATE))


def _ycxcz(linear: Image) -> Image:
    t = (linear @ RGB_TO_XYZ.T) / WHITE
    return np.stack(
        (116 * t[..., 1] - 16, 500 * (t[..., 0] - t[..., 1]), 200 * (t[..., 1] - t[..., 2])),
        axis=-1,
    )


def _hunt_lab(ycxcz: Image) -> Image:
    """Spatially filtered opponent colour back to Hunt-adjusted CIELAB."""
    filtered = np.stack(
        [_filter(ycxcz[..., index], kernel) for index, kernel in enumerate(CSF_KERNELS)], axis=-1
    )
    y = (filtered[..., 0] + 16) / 116
    xyz = np.stack((filtered[..., 1] / 500 + y, y, y - filtered[..., 2] / 200), axis=-1) * WHITE
    linear = (xyz @ XYZ_TO_RGB.T).clip(0, 1)
    lab = xyz_to_lab(linear @ RGB_TO_XYZ.T)
    return _hunt(lab)


def _hunt(lab: Image) -> Image:
    scale = 0.01 * lab[..., 0]
    return np.stack((lab[..., 0], scale * lab[..., 1], scale * lab[..., 2]), axis=-1)


def _hyab(first: Image, second: Image) -> Image:
    difference = first - second
    return np.asarray(np.abs(difference[..., 0]) + np.hypot(difference[..., 1], difference[..., 2]))


CMAX: Final[float] = float(
    _hyab(
        _hunt(xyz_to_lab(np.array([0.0, 1.0, 0.0]) @ RGB_TO_XYZ.T)),
        _hunt(xyz_to_lab(np.array([0.0, 0.0, 1.0]) @ RGB_TO_XYZ.T)),
    )
    ** FLIP_QC
)


def _features(luminance: Image) -> tuple[Image, Image]:
    edge = np.hypot(_filter(luminance, EDGE_KERNEL), _filter(luminance, EDGE_KERNEL.T))
    point = np.hypot(_filter(luminance, POINT_KERNEL), _filter(luminance, POINT_KERNEL.T))
    return edge, point


def flip(reference: Image, test: Image) -> Image:
    """Per-pixel LDR FLIP error of two sRGB images in [0, 1]."""
    first, second = _ycxcz(srgb_to_linear(reference)), _ycxcz(srgb_to_linear(test))
    error = _hyab(_hunt_lab(first), _hunt_lab(second)) ** FLIP_QC
    knee = FLIP_PC * CMAX
    colour = np.where(
        error < knee,
        error * FLIP_PT / knee,
        FLIP_PT + (error - knee) / (CMAX - knee) * (1 - FLIP_PT),
    )
    edge1, point1 = _features((first[..., 0] + 16) / 116)
    edge2, point2 = _features((second[..., 0] + 16) / 116)
    feature = (np.maximum(np.abs(edge1 - edge2), np.abs(point1 - point2)) / math.sqrt(2)) ** FLIP_QF
    return np.asarray(colour ** (1 - feature))


def premultiplied(frame: NDArray[np.uint8]) -> Image:
    rgba = frame.astype(np.float64) / 255
    return np.concatenate((rgba[..., :3] * rgba[..., 3:], rgba[..., 3:]), axis=-1)


def over_grey(frame: NDArray[np.uint8]) -> Image:
    rgba = frame.astype(np.float64) / 255
    return np.asarray(rgba[..., :3] * rgba[..., 3:] + GREY * (1 - rgba[..., 3:]))


def text_lines(frame: NDArray[np.uint8]) -> list[Line]:
    """Every text line of a frame with its fill colours at the sampled fractions."""
    fill = fill_mask(frame)
    rows = np.nonzero(fill.sum(axis=1) > LINE_ROWS)[0]
    if not len(rows):
        return []
    runs = np.split(rows, np.nonzero(np.diff(rows) > LINE_GAP)[0] + 1)
    # Colours come from the fill's interior: edge pixels mix in the keyline by sub-pixel position.
    interior = cv2.erode(fill.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
    result = []
    for top, bottom in _lines([(int(run[0]), int(run[-1])) for run in runs]):
        profile = np.zeros(len(fill))
        profile[top : bottom + 1] = fill[top : bottom + 1].sum(axis=1)
        colours = {}
        for fraction in FRACTIONS:
            row = top + round((bottom - top) * fraction / 100)
            band = slice(max(top, row - 1), row + 2)
            pixels = frame[band, :, :3][interior[band]]
            if len(pixels) >= MIN_SAMPLES:
                colours[fraction] = pixels.mean(axis=0) / 255
        columns = fill[top : bottom + 1].sum(axis=0).astype(np.float64)
        result.append(Line(top, bottom, columns, profile, colours))
    return result


def edge_profile(frame: NDArray[np.uint8]) -> NDArray[np.float64]:
    """Per distance bin from the fill: summed alpha, summed white-band share, pixel count."""
    fill = fill_mask(frame)
    bins = int(MAX_DISTANCE / BIN)
    if not fill.any():
        return np.zeros((3, bins))
    distance = cv2.distanceTransform((~fill).astype(np.uint8), cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
    index = (distance / BIN).astype(np.int64)
    inside = index < bins
    rgb = frame[..., :3].astype(np.int16)
    white = (
        (frame[..., 3] > OPAQUE)
        & (rgb.min(axis=2) > WHITE_MINIMUM)
        & (rgb.max(axis=2) - rgb.min(axis=2) < WHITE_SPREAD)
    )
    alpha = frame[..., 3] / 255
    return np.stack(
        (
            np.bincount(index[inside], alpha[inside], bins),
            np.bincount(index[inside], white[inside].astype(np.float64), bins),
            np.bincount(index[inside], minlength=bins).astype(np.float64),
        )
    )


def crossing(distances: NDArray[np.float64], profile: NDArray[np.float64], level: float) -> float:
    """Distance (pixels) where the profile first falls below ``level`` after its peak."""
    peak = int(np.argmax(profile))
    below = np.nonzero(profile[peak:] < level)[0]
    if not len(below):
        return MAX_DISTANCE
    index = peak + int(below[0])
    if index == peak:
        return float(distances[index])
    high, low = profile[index - 1], profile[index]
    near, far = distances[index - 1], distances[index]
    return float(near + (high - level) / (high - low) * (far - near))


def _crop(first: NDArray[np.uint8], second: NDArray[np.uint8]) -> tuple[slice, slice] | None:
    ys, xs = np.nonzero((first[..., 3] > 0) | (second[..., 3] > 0))
    if not len(ys):
        return None
    height, width = first.shape[:2]
    return (
        slice(max(0, int(ys.min()) - CROP_MARGIN), min(height, int(ys.max()) + CROP_MARGIN + 1)),
        slice(max(0, int(xs.min()) - CROP_MARGIN), min(width, int(xs.max()) + CROP_MARGIN + 1)),
    )


def lag(first: Image, second: Image) -> float:
    """Sub-pixel displacement of profile ``second`` against ``first`` (parabolic peak)."""
    padded = np.pad(second, MAX_LAG)
    scores = np.array(
        [float(first @ padded[shift : shift + len(first)]) for shift in range(2 * MAX_LAG + 1)]
    )
    peak = int(np.argmax(scores))
    if not 0 < peak < 2 * MAX_LAG:
        return float(peak - MAX_LAG)
    left, centre, right = scores[peak - 1], scores[peak], scores[peak + 1]
    curvature = left - 2 * centre + right
    offset = 0.0 if curvature == 0 else (left - right) / (2 * curvature)
    return float(peak - MAX_LAG + offset)


def glyph_lags(first: Line, second: Line) -> list[float]:
    """Displacements of glyph-sized windows (the line's height) along the line."""
    inked = np.nonzero(first.columns)[0]
    if not len(inked):
        return []
    size = first.bottom - first.top + 1
    result = []
    for start in range(int(inked[0]), int(inked[-1]) + 1, size):
        window = slice(start, start + size)
        if first.columns[window].sum() >= MIN_SAMPLES:
            result.append(lag(first.columns[window], second.columns[window]))
    return result


def _compare_lines(tally: Tally, first: list[Line], second: list[Line]) -> None:
    if len(first) != len(second):
        tally.mismatched += 1
        return
    for one, two in zip(first, second, strict=True):
        tally.glyph_x += [abs(value) for value in glyph_lags(one, two)]
        tally.glyph_y.append(abs(lag(one.rows, two.rows)))
        tally.baseline.append(abs(one.bottom - two.bottom))
        for fraction in set(one.colours) & set(two.colours):
            difference = delta_e2000(srgb_to_lab(one.colours[fraction]),
                                     srgb_to_lab(two.colours[fraction]))  # fmt: skip
            tally.delta_e[fraction].append(float(difference))


def add_frame(tally: Tally, first: NDArray[np.uint8], second: NDArray[np.uint8]) -> None:
    """Measure one pair of RGBA frames; frames where neither shows a caption are skipped."""
    if first.shape != second.shape:
        raise ReplayError("The two renders have different frame sizes")
    window = _crop(first, second)
    if window is None:
        return
    a, b = first[window], second[window]
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * SUPPORT_DILATION + 1,) * 2)
    support = cv2.dilate(((a[..., 3] > 0) | (b[..., 3] > 0)).astype(np.uint8), kernel) > 0
    tally.frames += 1
    tally.ssim.append(ssim(premultiplied(a), premultiplied(b), support))
    error = flip(over_grey(a), over_grey(b))[support]
    tally.flip_sum += float(error.sum())
    tally.flip_count += int(error.size)
    tally.flip_p99.append(float(np.percentile(error, 99)))
    _compare_lines(tally, text_lines(a), text_lines(b))
    tally.profiles[0] += edge_profile(a)
    tally.profiles[1] += edge_profile(b)


def _mean(values: Sequence[float]) -> float:
    return float(np.mean(values)) if values else 0.0


def _edges(sums: NDArray[np.float64]) -> dict[str, float]:
    """Edge distances of a pooled profile; distance bins no pixel fell into are left out."""
    used = sums[2] > 0
    distances = (np.nonzero(used)[0] + 0.5) * BIN
    alpha, white = sums[0][used] / sums[2][used], sums[1][used] / sums[2][used]
    return {
        "alphaHalf": crossing(distances, alpha, 0.5),
        "alphaFall": crossing(distances, alpha, 0.1) - crossing(distances, alpha, 0.9),
        "shadowFade": crossing(distances, alpha, 0.05),
        "whiteHalf": crossing(distances, white, 0.5),
    }


def summary(tally: Tally) -> dict[str, float]:
    """Every metric as an error (lower is closer), plus the pooled values behind them."""
    if not tally.frames:
        raise ReplayError("Neither render shows a caption")
    first, second = _edges(tally.profiles[0]), _edges(tally.profiles[1])
    result = {
        "ssimLoss": 1 - _mean(tally.ssim),
        "ssimWorstLoss": 1 - min(tally.ssim),
        "flipMean": tally.flip_sum / tally.flip_count,
        "flipP99": _mean(tally.flip_p99),
        **{f"deltaE{fraction}": _mean(tally.delta_e[fraction]) for fraction in FRACTIONS},
        **{name: abs(first[name] - second[name]) for name in first},
        "glyphX": _mean(tally.glyph_x),
        "glyphY": _mean(tally.glyph_y),
        "baseline": _mean(tally.baseline),
        "lineMismatch": tally.mismatched / tally.frames,
    }
    return {name: round(value, 6) for name, value in result.items()}


def compare(first: Iterable[NDArray[np.uint8]], second: Iterable[NDArray[np.uint8]]) -> Tally:
    tally = Tally()
    try:
        for one, two in zip(first, second, strict=True):
            add_frame(tally, one, two)
    except ValueError as error:
        raise ReplayError("The two renders have different frame counts") from error
    return tally


def limits(floor: dict[str, float]) -> dict[str, float]:
    allowed = {
        name: max(FLOOR_MULTIPLE * floor[name], MINIMUM_ERROR[name]) for name in MINIMUM_ERROR
    }
    return {name: min(limit, GUARD.get(name, limit)) for name, limit in allowed.items()}


def verdict(floor: dict[str, float], candidate: dict[str, float]) -> dict[str, object]:
    allowed = limits(floor)
    failed = [name for name, limit in allowed.items() if candidate[name] > limit]
    return {"pass": not failed, "failed": failed, "limits": allowed}


def stored_floor(path: Path) -> dict[str, float]:
    """The floor of an earlier result, refused unless it holds every metric as a number."""
    value = json.loads(path.read_text(encoding="utf-8"))
    floor = value.get("floor") if isinstance(value, dict) else None
    if not isinstance(floor, dict) or set(floor) != set(MINIMUM_ERROR):
        raise ReplayError(f"{path} holds no complete parity floor")
    if not all(
        isinstance(item, (int, float)) and not isinstance(item, bool) for item in floor.values()
    ):
        raise ReplayError(f"{path} holds no complete parity floor")
    return {str(name): float(item) for name, item in floor.items()}


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reference", help="Motion alpha render")
    parser.add_argument("candidate", help="candidate alpha render of the same captions")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--floor", help="Motion render shifted half a pixel")
    source.add_argument("--floor-json", type=Path, help="an earlier result holding the floor")
    parser.add_argument("--step", type=int, default=1, help="compare every n-th frame")
    args = parser.parse_args(argv)
    floor = (
        stored_floor(args.floor_json)
        if args.floor is None
        else summary(
            compare(frames(args.reference, step=args.step), frames(args.floor, step=args.step))
        )
    )
    candidate = summary(
        compare(frames(args.reference, step=args.step), frames(args.candidate, step=args.step))
    )
    result = {"floor": floor, "candidate": candidate, **verdict(floor, candidate)}
    print(json.dumps(result, indent=1))
    return 0 if result["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
