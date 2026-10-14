"""Local character masks from Apple's VNDetectTextRectanglesRequest, not OCR."""

import hashlib
import json
import logging
import math
import subprocess
from pathlib import Path
from tempfile import TemporaryDirectory
from time import perf_counter

import cv2
import numpy as np
from numpy.typing import NDArray

from replay.caption_masks import TextRectangle
from replay.recordings import record

Image = NDArray[np.uint8]
DEFAULT_EXECUTABLE = Path("edits/work/native-swift/build/release/apple-pro-apps")
DEFAULT_WORK_DIRECTORY = Path("edits/work/vision-text-rectangles")
LOGGER = logging.getLogger(__name__)


def character_mask(value: object, *, width: int, height: int) -> Image:
    """Rasterize measured top-left character boxes, never whole line envelopes."""
    return _box_mask(value, width=width, height=height, text_regions=False)


def text_region_mask(value: object, *, width: int, height: int) -> Image:
    """Rasterize actual VNTextObservation regions, without OCR or inferred boxes.

    Zero-area native envelopes are explicitly reported and represented only by
    valid measured character support, never inflated or silently called absent.
    This explicitly broader support is intended for neural region repair. It does
    not certify text coverage or authorize replacing pixels outside these regions.
    Legacy measurements lacking textBoxes fail rather than falling back.
    """
    return _box_mask(value, width=width, height=height, text_regions=True)


def _box_mask(value: object, *, width: int, height: int, text_regions: bool) -> Image:
    document = record(value)
    if (
        type(document.get("width")) is not int
        or type(document.get("height")) is not int
        or document["width"] != width
        or document["height"] != height
        or min(width, height) < 8
        or width * height > 2_000_000
    ):
        raise ValueError("Vision mask dimensions differ from the input ROI")
    kind = "text" if text_regions else "character"
    boxes = document.get(f"{kind}Boxes")
    if not isinstance(boxes, list) or len(boxes) > (64 if text_regions else 4096):
        raise ValueError(f"Missing or excessive Vision {kind} boxes")
    mask = np.zeros((height, width), dtype=np.uint8)
    for item in boxes:
        box = record(item)
        coordinates: list[float] = []
        for name in ("x", "y", "width", "height"):
            number = box.get(name)
            if isinstance(number, bool) or not isinstance(number, (int, float)):
                raise ValueError("Invalid Vision character coordinate")
            if not math.isfinite(number):
                raise ValueError("Non-finite Vision character coordinate")
            coordinates.append(float(number))
        x, y, w, h = coordinates
        if min(x, y) < 0 or min(w, h) <= 0 or x + w > width or y + h > height:
            raise ValueError("Vision character box lies outside the ROI")
        mask[math.floor(y) : math.ceil(y + h), math.floor(x) : math.ceil(x + w)] = 255
    if text_regions and document.get("diagnostics") is not None:
        diagnostics = record(document["diagnostics"])
        degenerate = diagnostics.get("degenerateTextObservationCount", 0)
        if type(degenerate) is not int or not 0 <= degenerate <= 64:
            raise ValueError("Invalid degenerate Vision observation count")
        if degenerate:
            if document.get("characterBoxes") == []:
                raise ValueError("Degenerate Vision envelope lacks measured character support")
            # Native text envelopes can have zero area despite valid characters.
            # Use only the independently measured character pixels, not an
            # invented envelope, padding, clipping or an OCR-derived rectangle.
            mask |= _box_mask(document, width=width, height=height, text_regions=False)
            LOGGER.warning(
                "Vision zero-area text observations=%d; "
                "measured character support retained; quality unapproved",
                degenerate,
            )
    return mask


def vision_text_mask(
    image: Image,
    *,
    executable: Path = DEFAULT_EXECUTABLE,
    work_directory: Path = DEFAULT_WORK_DIRECTORY,
    contrast_region: TextRectangle | None = None,
    text_regions: bool = False,
) -> Image:
    """Single-ROI convenience wrapper; stream callers use bounded batches."""
    return vision_text_masks(
        (image,),
        executable=executable,
        work_directory=work_directory,
        contrast_regions=(contrast_region,),
        text_regions=text_regions,
    )[0]


def _validate_image(image: Image) -> None:
    if (
        image.dtype != np.uint8
        or image.ndim != 3
        or image.shape[2] != 3
        or min(image.shape[:2]) < 8
        or image.shape[0] * image.shape[1] > 2_000_000
    ):
        raise ValueError("Expected a bounded uint8 BGR subtitle ROI")


def _contrast_image(image: Image, region: TextRectangle) -> Image:
    """Use a caller-observed region's contrast, never a whole-scene histogram."""
    _validate_image(image)
    if region.x + region.width > image.shape[1] or region.y + region.height > image.shape[0]:
        raise ValueError("Contrast reference region lies outside the ROI")
    left, top = math.floor(region.x), math.floor(region.y)
    right, bottom = math.ceil(region.x + region.width), math.ceil(region.y + region.height)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    reference = gray[top:bottom, left:right]
    if reference.min() == reference.max():
        raise ValueError("Contrast reference has no intensity separation")
    threshold, _ = cv2.threshold(reference, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    _, binary = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)
    LOGGER.debug(
        "vision contrast reference=(%d,%d,%d,%d) threshold=%.6f source_unchanged=true",
        left,
        top,
        right - left,
        bottom - top,
        threshold,
    )
    return np.asarray(cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR), dtype=np.uint8)


def _encode_roi(image: Image) -> bytes:
    _validate_image(image)
    success, encoded = cv2.imencode(".png", image)
    if not success or encoded.nbytes > 8 * 1024 * 1024:
        raise ValueError("Cannot encode bounded subtitle ROI")
    return encoded.tobytes()


def vision_text_masks(
    images: tuple[Image, ...],
    *,
    executable: Path = DEFAULT_EXECUTABLE,
    work_directory: Path = DEFAULT_WORK_DIRECTORY,
    contrast_regions: tuple[TextRectangle | None, ...] | None = None,
    text_regions: bool = False,
) -> tuple[Image, ...]:
    """Observe 1–8 exact ROIs with one process/context, preserving request order.

    No color/OCR fallback. Empty detections are not certified text absence.
    Default masks use characters; explicit text_regions=True uses measured Vision
    observation regions for neural repair, never envelopes inferred from OCR.
    Only transient files created by this invocation are removed. Each decoded ROI
    retains the two-megapixel limit; native decoding is sequential, not parallel.
    Optional contrast references must be observed on these exact input images and
    supplied in the same order; no region is guessed. Their Otsu threshold prepares
    detector input only. Masks come exclusively from the selected Vision geometry.
    Original pixels, dimensions and coordinates are preserved. This is not coverage
    or repair-quality approval, and the default raw-image policy is unchanged.
    """
    if not 1 <= len(images) <= 8:
        raise ValueError("Supply 1–8 subtitle ROIs")
    if contrast_regions is not None and len(contrast_regions) != len(images):
        raise ValueError("Contrast reference count differs from input ROIs")
    if not executable.is_file():
        raise FileNotFoundError("Build or explicitly supply the native apple-pro-apps executable")
    started = perf_counter()
    LOGGER.debug("vision batch start images=%d", len(images))
    work_directory.mkdir(parents=True, exist_ok=True, mode=0o700)
    with TemporaryDirectory(prefix="text-rectangles-", dir=work_directory) as name:
        directory = Path(name).resolve()
        paths: list[str] = []
        for index, image in enumerate(images):
            image_path = directory / f"roi-{index}.png"
            region = None if contrast_regions is None else contrast_regions[index]
            prepared = image if region is None else _contrast_image(image, region)
            encoded = _encode_roi(prepared)
            if LOGGER.isEnabledFor(logging.DEBUG):
                decoded = cv2.imdecode(np.frombuffer(encoded, dtype=np.uint8), cv2.IMREAD_COLOR)
                if decoded is None or not np.array_equal(prepared, decoded):
                    raise ValueError("PNG round-trip changed the detector input")
                LOGGER.debug(
                    "vision input index=%d width=%d height=%d source_bgr_sha256=%s "
                    "detector_bgr_sha256=%s png_sha256=%s "
                    "png_bytes=%d round_trip_equal=true channel_min=%s channel_max=%s",
                    index,
                    image.shape[1],
                    image.shape[0],
                    hashlib.sha256(image.tobytes()).hexdigest(),
                    hashlib.sha256(prepared.tobytes()).hexdigest(),
                    hashlib.sha256(encoded).hexdigest(),
                    len(encoded),
                    image.min(axis=(0, 1)).tolist(),
                    image.max(axis=(0, 1)).tolist(),
                )
            with image_path.open("xb") as handle:
                image_path.chmod(0o600)
                handle.write(encoded)
            paths.append(str(image_path))
        request = directory / "request.json"
        with request.open("x", encoding="utf-8") as handle:
            request.chmod(0o600)
            json.dump({"paths": paths}, handle)
        LOGGER.debug("vision native start images=%d", len(paths))
        native_started = perf_counter()
        try:
            result = subprocess.run(
                [
                    str(executable.resolve()),
                    "measure-media",
                    "image_text_rectangles_batch",
                    str(request),
                ],
                check=True,
                capture_output=True,
                timeout=60,
            )
        except (OSError, subprocess.SubprocessError) as error:
            LOGGER.debug(
                "vision native failed kind=%s seconds=%.6f",
                type(error).__name__,
                perf_counter() - native_started,
            )
            raise
        LOGGER.debug(
            "vision native end seconds=%.6f stdout_bytes=%d stderr_bytes=%d",
            perf_counter() - native_started,
            len(result.stdout),
            len(result.stderr),
        )
    if len(result.stdout) > len(images) * 1024 * 1024:
        raise ValueError("Excessive Vision character response")
    values: object = json.loads(result.stdout)
    if not isinstance(values, list) or len(values) != len(images):
        raise ValueError("Vision result count differs from requested ROIs")
    masks: list[Image] = []
    for index, (image, value) in enumerate(zip(images, values, strict=True)):
        mask = _box_mask(
            value, width=image.shape[1], height=image.shape[0], text_regions=text_regions
        )
        if LOGGER.isEnabledFor(logging.DEBUG):
            measurement = record(value)
            LOGGER.debug(
                "vision mask index=%d native_diagnostics=%s mask_pixels=%d "
                "text_absence_certified=false",
                index,
                measurement.get("diagnostics"),
                np.count_nonzero(mask),
            )
        masks.append(mask)
    LOGGER.debug("vision batch end seconds=%.6f", perf_counter() - started)
    return tuple(masks)
