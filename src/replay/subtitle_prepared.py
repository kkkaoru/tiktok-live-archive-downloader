"""Observe each input ROI before bounded Vision detector preparation.

OCR locates contrast contexts, never repair support. Only measured Vision text
regions on two fixed prepared views seed masks; original pixels stay untouched.
Empty detections remain unknown, not proof of subtitle absence or removal.
"""

import json
import math
import subprocess
from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory

import cv2
import numpy as np

from .caption_io import FrameObservation, native_ocr_frames
from .recordings import record
from .subtitle_inpaint import Image, validate_roi
from .subtitle_vision import vision_text_masks


@dataclass(frozen=True, slots=True)
class ImageContext:
    observation: FrameObservation
    clipped_lines: int


@dataclass(frozen=True, slots=True)
class PreparedMasks:
    masks: tuple[Image, ...]
    observed_lines: tuple[int, ...]
    clipped_lines: tuple[int, ...]


def image_observations(
    images: tuple[Image, ...], *, executable: Path, work_directory: Path
) -> tuple[ImageContext, ...]:
    """Recognize exact local pixels; zero-valued image clocks are not media PTS."""
    if not 1 <= len(images) <= 8 or not executable.is_file():
        raise ValueError("Supply 1–8 images and an existing native executable")
    for image in images:
        validate_roi(image)
    work_directory.mkdir(parents=True, exist_ok=True, mode=0o700)
    with TemporaryDirectory(prefix="image-context-", dir=work_directory) as temporary:
        root = Path(temporary)
        paths: list[str] = []
        for index, image in enumerate(images):
            success, encoded = cv2.imencode(".png", image)
            if not success:
                raise ValueError("Cannot encode observation input")
            path = root / f"{index}.png"
            with path.open("xb") as stream:
                path.chmod(0o600)
                stream.write(encoded.tobytes())
            paths.append(str(path.resolve()))
        request = root / "request.json"
        with request.open("x", encoding="utf-8") as stream:
            json.dump({"paths": paths}, stream)
        response = subprocess.run(
            [
                str(executable.resolve()),
                "measure-media",
                "image_text_recognize_batch",
                str(request.resolve()),
            ],
            capture_output=True,
            check=True,
            timeout=60,
        )
    if len(response.stdout) > 8 * 1024 * 1024:
        raise ValueError("Excessive image recognition response")
    raw: object = json.loads(response.stdout)
    if not isinstance(raw, list) or len(raw) != len(images):
        raise ValueError("Image observation count differs from inputs")
    frames: list[ImageContext] = []
    for image, value in zip(images, raw, strict=True):
        data = record(value)
        if data.get("width") != image.shape[1] or data.get("height") != image.shape[0]:
            raise ValueError("Image observation dimensions changed")
        lines = data.get("lines")
        if not isinstance(lines, list) or any(
            type(record(line).get("boundsClipped")) is not bool for line in lines
        ):
            raise ValueError("Unknown OCR clipping provenance")
        frame = native_ocr_frames([{**data, "requestedTimeSeconds": 0, "actualTimeSeconds": 0}])[0]
        frames.append(
            ImageContext(frame, sum(record(line)["boundsClipped"] is True for line in lines))
        )
    return tuple(frames)


def prepared_text_masks(
    images: tuple[Image, ...], *, executable: Path, work_directory: Path
) -> PreparedMasks:
    """Use same-frame observed line contexts, fixed 2x binary/closed preparation.

    This is a fixed shared policy, not a transform search or OCR rectangle mask.
    A clipped OCR line supplies only its visible, bounded contrast pixels, not a
    complete glyph claim. Clipping counts are retained for downstream review.
    Actual repair support still requires independently measured, in-bounds Vision
    geometry. Masks map measured 2x pixels back to their source footprints.
    No resizing touches the source image; this never approves removal quality.
    """
    observations = image_observations(images, executable=executable, work_directory=work_directory)
    contexts: list[Image] = []
    placements: list[tuple[int, int, int, int, int]] = []
    outputs = [np.zeros(image.shape[:2], dtype=np.uint8) for image in images]

    def apply() -> None:
        masks = vision_text_masks(
            tuple(contexts), executable=executable, work_directory=work_directory, text_regions=True
        )
        for mask, (index, left, top, width, height) in zip(masks, placements, strict=True):
            local = mask.reshape(height, 2, width, 2).max(axis=(1, 3))
            outputs[index][top : top + height, left : left + width] |= local
        contexts.clear()
        placements.clear()

    for index, (image, observed) in enumerate(zip(images, observations, strict=True)):
        for line in observed.observation.detections:
            box = line.rectangle
            if (
                line.confidence < 0.3
                or not 16 <= box.height <= 96
                or not any(c.isalpha() for c in line.text)
            ):
                continue
            padding = math.ceil(box.height / 2)
            left = max(0, math.floor(box.x) - padding)
            top = max(0, math.floor(box.y) - padding)
            right = min(image.shape[1], math.ceil(box.x + box.width) + padding)
            bottom = min(image.shape[0], math.ceil(box.y + box.height) + padding)
            width, height = right - left, bottom - top
            if min(width, height) < 8 or width * height * 4 > 2_000_000:
                raise ValueError("Observed contrast context exceeds the detector budget")
            gray = cv2.cvtColor(image[top:bottom, left:right], cv2.COLOR_BGR2GRAY)
            if gray.min() == gray.max():
                raise ValueError("Observed contrast context has no intensity separation")
            _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
            closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, np.ones((3, 3), dtype=np.uint8))
            for pixels in (binary, closed):
                contexts.append(
                    np.asarray(
                        cv2.resize(
                            cv2.cvtColor(pixels, cv2.COLOR_GRAY2BGR),
                            (width * 2, height * 2),
                            interpolation=cv2.INTER_NEAREST,
                        ),
                        dtype=np.uint8,
                    )
                )
                placements.append((index, left, top, width, height))
                if len(contexts) == 8:
                    apply()
    if contexts:
        apply()
    return PreparedMasks(
        tuple(outputs),
        tuple(len(item.observation.detections) for item in observations),
        tuple(item.clipped_lines for item in observations),
    )
