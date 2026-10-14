"""Synthetic registered observations; no private footage or model inference."""

import subprocess
from collections.abc import Sequence
from dataclasses import replace
from fractions import Fraction
from pathlib import Path

import cv2
import numpy as np
import pytest

from replay import subtitle_quality_observe as observe
from replay.caption_io import FrameObservation
from replay.caption_masks import TextDetection, TextRectangle
from replay.subtitle_clock import VideoFrameClock
from replay.subtitle_inpaint import Image
from replay.subtitle_prepared import ImageContext


@pytest.fixture
def frame() -> observe.ObservationFrame:
    background = np.random.default_rng(42).integers(20, 200, (192, 192, 3), dtype=np.uint8)
    mask = np.zeros((192, 192), np.bool_)
    mask[90:100, 90:100] = True
    original = background.copy()
    original[mask] = 255
    return observe.ObservationFrame(
        observe.OriginalContext(original, mask, Fraction(1, 30)),
        background,
        mask,
        mask,
        observe.OriginalContext(background, np.zeros_like(mask), Fraction(0)),
        observe.OriginalContext(background, np.zeros_like(mask), Fraction(2, 30)),
    )


def test_clean_registered_background(frame: observe.ObservationFrame) -> None:
    result = observe.temporal_observation(frame)
    assert result.evaluated
    assert np.count_nonzero(result.unstable) == 0
    assert result.previous is not None
    assert result.previous.displacement == (0, 0)
    assert result.previous.donor_xy[95, 95].tolist() == [95.0, 95.0]


def test_residual_change_is_unstable(frame: observe.ObservationFrame) -> None:
    result = observe.temporal_observation(replace(frame, candidate=frame.original.image))
    assert result.evaluated
    assert np.count_nonzero(result.unstable) == 100


@pytest.mark.parametrize("neighbor", ["previous", "following"])
def test_boundary_unknown(frame: observe.ObservationFrame, neighbor: str) -> None:
    changed = (
        replace(frame, previous=None) if neighbor == "previous" else replace(frame, following=None)
    )
    result = observe.temporal_observation(changed)
    assert not result.evaluated
    assert np.count_nonzero(result.unknown) == 100


def test_donor_text_is_unknown(frame: observe.ObservationFrame) -> None:
    assert frame.previous is not None
    result = observe.temporal_observation(
        replace(frame, previous=replace(frame.previous, excluded=frame.repair))
    )
    assert not result.evaluated
    assert np.count_nonzero(result.unknown) == 100


def test_natural_motion_is_unknown(frame: observe.ObservationFrame) -> None:
    assert frame.following is not None
    moving = frame.following.image.copy()
    moving[frame.repair] = 0
    result = observe.temporal_observation(
        replace(frame, following=replace(frame.following, image=moving))
    )
    assert not result.evaluated
    assert np.count_nonzero(result.unstable) == 0


def test_translation(frame: observe.ObservationFrame) -> None:
    assert frame.previous is not None
    shifted = np.asarray(
        cv2.warpAffine(
            frame.previous.image, np.array([[1, 0, 2], [0, 1, 0]], dtype=np.float32), (192, 192)
        ),
        dtype=np.uint8,
    )
    result = observe.register(frame.original, replace(frame.previous, image=shifted))
    assert result.displacement == pytest.approx((2, 0), abs=0.01)
    assert result.known[95, 95]
    assert result.donor_xy[95, 95].tolist() == pytest.approx([97.0, 95.0], abs=0.01)


def test_rotation_not_translation(frame: observe.ObservationFrame) -> None:
    assert frame.previous is not None
    rotated = cv2.warpAffine(
        frame.previous.image, cv2.getRotationMatrix2D((96, 96), 15, 1), (192, 192)
    )
    result = observe.register(
        frame.original, replace(frame.previous, image=np.asarray(rotated, dtype=np.uint8))
    )
    assert not np.any(result.known)
    assert result.displacement is None


def test_untextured_unknown(frame: observe.ObservationFrame) -> None:
    flat = np.zeros_like(frame.candidate)
    assert frame.previous is not None
    result = observe.register(
        replace(frame.original, image=flat), replace(frame.previous, image=flat)
    )
    assert not np.any(result.known)


@pytest.mark.parametrize(
    "fault", ["dtype", "shape", "pts", "mask", "candidate", "outside", "order", "equal"]
)
def test_invalid(frame: observe.ObservationFrame, fault: str) -> None:
    assert frame.previous is not None
    if fault == "dtype":
        frame = replace(
            frame, original=replace(frame.original, image=frame.original.image.astype(np.float32))
        )
    elif fault == "shape":
        frame = replace(frame, original=replace(frame.original, image=frame.original.image[:4]))
    elif fault == "pts":
        frame = replace(frame, original=replace(frame.original, pts=Fraction(-1)))
    elif fault == "mask":
        frame = replace(frame, original=replace(frame.original, excluded=frame.repair[:5]))
    elif fault == "candidate":
        frame = replace(frame, candidate=frame.candidate[:5])
    elif fault == "outside":
        frame = replace(
            frame, original=replace(frame.original, excluded=np.zeros_like(frame.repair))
        )
    elif fault == "order":
        assert frame.previous is not None
        frame = replace(frame, previous=replace(frame.previous, pts=Fraction(9)))
    else:
        with pytest.raises(ValueError, match="distinct"):
            observe.register(frame.original, frame.original)
        return
    with pytest.raises(ValueError):
        observe.temporal_observation(frame)


def no_glyph(images: Sequence[Image]) -> tuple[Image, ...]:
    return tuple(np.zeros(image.shape[:2], np.uint8) for image in images)


def empty_ocr(
    images: tuple[Image, ...], *, executable: Path, work_directory: Path
) -> tuple[ImageContext, ...]:
    return tuple(ImageContext(FrameObservation(0, 0, 192, 192, ()), 0) for image in images)


def test_batch(
    frame: observe.ObservationFrame, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(observe, "image_observations", empty_ocr)
    result = observe.observe_batch(
        (frame,), detect=no_glyph, executable=tmp_path, work_directory=tmp_path
    )
    assert result[0].evidence.technically_valid
    assert result[0].evidence.residual_fraction == 0
    assert result[0].evidence.unstable_fraction == 0
    assert result[0].evidence.readable_text is False
    missing = observe.observe_batch(
        (replace(frame, previous=None),),
        detect=no_glyph,
        executable=tmp_path,
        work_directory=tmp_path,
    )
    assert not missing[0].evidence.technically_valid
    assert missing[0].evidence.unstable_fraction is None
    with pytest.raises(ValueError, match="one to eight"):
        observe.observe_batch((), detect=no_glyph, executable=tmp_path, work_directory=tmp_path)


@pytest.mark.parametrize("clipped", [0, 1])
def test_readable_ocr_is_not_detector_absence(
    frame: observe.ObservationFrame,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    clipped: int,
) -> None:
    def recognize(
        images: tuple[Image, ...], *, executable: Path, work_directory: Path
    ) -> tuple[ImageContext, ...]:
        return (
            ImageContext(
                FrameObservation(
                    0, 0, 192, 192, (TextDetection("text", 0.01, TextRectangle(90, 90, 10, 10)),)
                ),
                clipped,
            ),
        )

    monkeypatch.setattr(observe, "image_observations", recognize)
    result = observe.observe_batch(
        (frame,), detect=no_glyph, executable=tmp_path, work_directory=tmp_path
    )
    assert result[0].evidence.readable_text is True
    assert result[0].readable_lines == 1
    if clipped:
        assert result[0].evidence.technically_valid is False


def test_decode(tmp_path: Path) -> None:
    import shutil

    ffmpeg = shutil.which("ffmpeg")
    assert ffmpeg is not None
    source = tmp_path / "synthetic.mkv"
    subprocess.run(
        [
            ffmpeg,
            "-v",
            "error",
            "-f",
            "lavfi",
            "-i",
            "testsrc2=size=32x32:rate=30",
            "-frames:v",
            "4",
            "-c:v",
            "ffv1",
            str(source),
        ],
        check=True,
    )
    frames = observe.decode_context(
        source,
        clock=VideoFrameClock((33, 67), Fraction(1, 1000), 33),
        width=32,
        height=32,
        ffmpeg=Path(ffmpeg),
    )
    assert len(frames) == 2
    assert frames[0].shape == (32, 32, 3)
    with pytest.raises(ValueError, match="actual PTS"):
        observe.decode_context(
            source,
            clock=VideoFrameClock((34, 67), Fraction(1, 1000), 33),
            width=32,
            height=32,
            ffmpeg=Path(ffmpeg),
        )
    with pytest.raises(ValueError, match="budget"):
        observe.decode_context(
            source,
            clock=VideoFrameClock((0,), Fraction(1, 30), 1),
            width=1,
            height=1,
            ffmpeg=Path(ffmpeg),
        )
