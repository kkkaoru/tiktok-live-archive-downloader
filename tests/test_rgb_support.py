"""Synthetic RGB support including ambiguity, clipping and input validation."""

import json
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

from replay import rgb_support as rgb


@pytest.mark.parametrize("background", [80, 140])
def test_sparse_lower_pixel_preserved(background: int) -> None:
    image = np.full((8, 9, 3), background, dtype=np.uint8)
    image[2:4, 3:5] = 10
    image[6, 4] = 0
    result = rgb.measure(
        image, region=(0, 0, 9, 8), predicate=rgb.Predicate("dark", (0, 0, 0), (20, 20, 20))
    )
    assert result["boundsHalfOpen"] == [3, 2, 5, 7]
    assert result["pixels"] == 5
    assert result["nativePaintAccepted"] is False
    assert result["touchesRegionBoundary"] is False


def test_background_matching_and_clipping_are_unknown() -> None:
    result = rgb.measure(
        np.zeros((2, 3, 3), dtype=np.uint8),
        region=(0, 0, 3, 2),
        predicate=rgb.Predicate("black", (0, 0, 0), (0, 0, 0)),
    )
    assert result["uncertainty"] == "clipped-or-background"
    assert result["nativePaintAccepted"] is False


def test_absence_is_not_pass() -> None:
    result = rgb.measure(
        np.full((3, 4, 3), 90, dtype=np.uint8),
        region=(1, 1, 3, 3),
        predicate=rgb.Predicate("black", (0, 0, 0), (20, 20, 20)),
    )
    assert result["boundsHalfOpen"] is None
    assert result["uncertainty"] == "no-support"


def test_noisy_region_reports_all_support() -> None:
    image = np.full((8, 9, 3), 90, dtype=np.uint8)
    image[2, 3] = 20
    image[7, 0] = 19
    result = rgb.measure(
        image, region=(0, 0, 9, 8), predicate=rgb.Predicate("noise", (0, 0, 0), (20, 20, 20))
    )
    assert result["boundsHalfOpen"] == [0, 2, 4, 8]
    assert result["touchesRegionBoundary"] is True


def test_invalid_models() -> None:
    with pytest.raises(ValueError, match="predicate"):
        rgb.Predicate("", (0, 0, 0), (255, 255, 255))
    with pytest.raises(ValueError, match="RGB8"):
        rgb.measure(
            np.zeros((2, 2, 4), dtype=np.uint8),
            region=(0, 0, 2, 2),
            predicate=rgb.Predicate("x", (0, 0, 0), (255, 255, 255)),
        )
    with pytest.raises(ValueError, match="region"):
        rgb.measure(
            np.zeros((2, 2, 3), dtype=np.uint8),
            region=(0, 0, 3, 2),
            predicate=rgb.Predicate("x", (0, 0, 0), (255, 255, 255)),
        )


def test_cli(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    source, output = tmp_path / "input.png", tmp_path / "output.json"
    assert cv2.imwrite(str(source), np.full((3, 4, 3), 250, dtype=np.uint8))
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "rgb_support",
            str(source),
            str(output),
            "--region",
            "0",
            "0",
            "4",
            "3",
            "--lower",
            "230",
            "230",
            "230",
            "--upper",
            "255",
            "255",
            "255",
            "--label",
            "white",
        ],
    )
    rgb.main()
    assert json.loads(output.read_text())["pixels"] == 12
    with pytest.raises(FileExistsError):
        rgb.main()


def test_bad_file_and_header(tmp_path: Path) -> None:
    source = tmp_path / "input.png"
    predicate = rgb.Predicate("x", (0, 0, 0), (255, 255, 255))
    with pytest.raises(ValueError, match="regular"):
        rgb.run(source, tmp_path / "out.json", region=(0, 0, 1, 1), predicate=predicate)
    source.write_bytes(b"not png")
    with pytest.raises(ValueError, match="header"):
        rgb.run(source, tmp_path / "out.json", region=(0, 0, 1, 1), predicate=predicate)


def test_mutation_and_decode(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    source = tmp_path / "input.png"
    assert cv2.imwrite(str(source), np.zeros((2, 2, 3), dtype=np.uint8))
    predicate = rgb.Predicate("x", (0, 0, 0), (255, 255, 255))
    values = iter(("first", "changed"))
    monkeypatch.setattr(rgb, "file_digest", lambda _: next(values))
    with pytest.raises(ValueError, match="changed"):
        rgb.run(source, tmp_path / "out.json", region=(0, 0, 2, 2), predicate=predicate)
    monkeypatch.setattr(rgb, "file_digest", lambda _: "same")
    monkeypatch.setattr(cv2, "imread", lambda *_: None)
    with pytest.raises(ValueError, match="decode"):
        rgb.run(source, tmp_path / "out.json", region=(0, 0, 2, 2), predicate=predicate)
    monkeypatch.setattr(rgb, "MAX_PIXELS", 1)
    with pytest.raises(ValueError, match="budget"):
        rgb.run(source, tmp_path / "out.json", region=(0, 0, 2, 2), predicate=predicate)
