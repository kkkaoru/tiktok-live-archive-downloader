"""Synthetic RGBA measurements, not private media or colour-management acceptance."""

import json
import runpy
import sys
from pathlib import Path

import numpy as np
import pytest
from numpy.typing import NDArray

from replay import caption_pixels
from replay.core import ReplayError


def test_sparse_support_and_opaque_rgb_are_exact() -> None:
    image = np.zeros((3, 4, 4), dtype=np.uint8)
    image[0, 0] = [0, 0, 0, 1]
    image[1, 1] = [90, 207, 226, 255]
    image[2, 3] = [90, 207, 226, 255]
    assert caption_pixels.measure(image) == {
        "width": 4,
        "height": 3,
        "nonzeroAlphaPixels": 3,
        "opaquePixels": 2,
        "nonzeroAlphaBounds": [0, 0, 4, 3],
        "opaqueBounds": [1, 1, 4, 3],
        "dominantOpaqueRGB": [{"rgb": [90, 207, 226], "count": 2}],
        "colourSpaceVerified": False,
        "captionQualityApproved": False,
    }


def test_transparent_frame_keeps_absence_explicit() -> None:
    assert caption_pixels.measure(np.zeros((1, 2, 4), dtype=np.uint8)) == {
        "width": 2,
        "height": 1,
        "nonzeroAlphaPixels": 0,
        "opaquePixels": 0,
        "nonzeroAlphaBounds": None,
        "opaqueBounds": None,
        "dominantOpaqueRGB": [],
        "colourSpaceVerified": False,
        "captionQualityApproved": False,
    }


@pytest.mark.parametrize(
    "image",
    [
        np.zeros((2, 3), dtype=np.uint8),
        np.zeros((2, 3, 3), dtype=np.uint8),
        np.zeros((0, 3, 4), dtype=np.uint8),
    ],
)
def test_invalid_shapes_fail(image: NDArray[np.uint8]) -> None:
    with pytest.raises(ReplayError):
        caption_pixels.measure(image)


def test_pixel_budget_is_enforced(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(caption_pixels, "MAX_PIXELS", 2)
    with pytest.raises(ReplayError):
        caption_pixels.measure(np.zeros((1, 3, 4), dtype=np.uint8))


def test_cli_is_exclusive_and_preserves_input(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source, output = tmp_path / "frame.rgba", tmp_path / "measurement.json"
    source.write_bytes(bytes([90, 207, 226, 255]))
    monkeypatch.setattr(
        sys, "argv", ["caption_pixels", str(source), str(output), "--width", "1", "--height", "1"]
    )
    runpy.run_path(caption_pixels.__file__, run_name="__main__")
    document = json.loads(output.read_text(encoding="utf-8"))
    assert document["dominantOpaqueRGB"] == [{"rgb": [90, 207, 226], "count": 1}]
    assert document["nonzeroAlphaBounds"] == [0, 0, 1, 1]
    assert len(document["sourceSHA256"]) == 64
    with pytest.raises(FileExistsError):
        caption_pixels.main()
    assert source.read_bytes() == bytes([90, 207, 226, 255])


@pytest.mark.parametrize("width,height", [(0, 1), (-1, -1), (9000, 9000)])
def test_cli_rejects_invalid_dimensions(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    width: int,
    height: int,
) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "caption_pixels",
            str(tmp_path / "none"),
            str(tmp_path / "out"),
            "--width",
            str(width),
            "--height",
            str(height),
        ],
    )
    with pytest.raises(ReplayError, match="pixel budget"):
        caption_pixels.main()


@pytest.mark.parametrize("kind", ["missing", "short", "link", "changed-size", "changed-digest"])
def test_cli_rejects_invalid_or_changing_inputs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    kind: str,
) -> None:
    source, output = tmp_path / "frame.rgba", tmp_path / "measurement.json"
    if kind == "link":
        source.symlink_to(tmp_path / "absent")
    elif kind != "missing":
        source.write_bytes(b"x" if kind == "short" else b"1234")
    if kind == "changed-size":
        monkeypatch.setattr(Path, "read_bytes", lambda _: b"12")
    if kind == "changed-digest":
        digests = iter(["a" * 64, "b" * 64])
        monkeypatch.setattr(caption_pixels, "file_digest", lambda _: next(digests))
    monkeypatch.setattr(
        sys, "argv", ["caption_pixels", str(source), str(output), "--width", "1", "--height", "1"]
    )
    with pytest.raises(ReplayError):
        caption_pixels.main()
    assert not output.exists()
