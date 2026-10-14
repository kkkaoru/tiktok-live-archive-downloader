"""Saved synthetic writer ROIs; no private media or actual model calls."""

import json
from collections.abc import Sequence
from pathlib import Path

import cv2
import numpy as np
import pytest

from replay import subtitle_quality_diagnostic as diagnostic
from replay.core import ReplayError
from replay.delivery_mask_quality import digest
from replay.subtitle_inpaint import Image


@pytest.fixture
def receipt(tmp_path: Path) -> Path:
    source = np.zeros((16, 20, 3), np.uint8)
    source[4:10, 5:12] = [10, 90, 230]
    pixels = tmp_path / "pixels.npz"
    np.savez(
        pixels,
        source=source,
        candidate=source.copy(),
        glyph=np.zeros((16, 20), np.uint8),
        repair=np.zeros((16, 20), np.uint8),
    )
    path = tmp_path / "receipt.json"
    path.write_text(
        json.dumps(
            {
                "losslessEvidence": {
                    "pixelFormat": "bgr24",
                    "frames": [
                        {
                            "path": str(pixels),
                            "sha256": digest(pixels),
                            "frame": 1,
                            "sourcePTS": 3060,
                            "timeBase": "1/90000",
                            "region": {"x": 10, "y": 20, "width": 20, "height": 16},
                            "glyphPixels": 0,
                            "repairPixels": 0,
                        }
                    ],
                }
            }
        ),
        encoding="utf-8",
    )
    return path


def detection(images: Sequence[Image]) -> tuple[Image, ...]:
    return tuple(np.full(image.shape[:2], 255, np.uint8) for image in images)


def test_export_and_cli(receipt: Path, tmp_path: Path) -> None:
    directory = tmp_path / "export"
    report = diagnostic.export_capture(receipt, directory, sha256=digest(receipt), detect=detection)
    assert report["qualityApproved"] is False
    assert report["sourceMediaReverified"] is False
    source = cv2.imread(str(directory / "00000001-source.png"))
    assert source is not None
    assert source.shape == (16, 20, 3)
    assert source[5, 6].tolist() == [10, 90, 230]
    assert (directory / "00000001-source.png").stat().st_mode & 0o777 == 0o600
    glyph = cv2.imread(str(directory / "00000001-glyph.png"), 0)
    independent = cv2.imread(str(directory / "00000001-independent-source.png"), 0)
    assert glyph is not None and independent is not None
    assert np.count_nonzero(glyph) == 0
    assert np.count_nonzero(independent) == 320
    assert diagnostic.main([str(receipt), str(tmp_path / "cli"), "--sha256", digest(receipt)]) == 0
    raw = json.loads((tmp_path / "cli" / "diagnostic.json").read_text(encoding="utf-8"))
    assert raw["frames"][0]["sourceEqualsCandidate"] is True
    assert raw["frames"][0]["independentSourcePixels"] is None
    with pytest.raises(ReplayError, match="overwrite"):
        diagnostic.export_capture(receipt, directory, sha256=digest(receipt))


@pytest.mark.parametrize("fault", ["count", "duplicate", "format", "shape", "dtype", "pin"])
def test_reject_bad_capture(receipt: Path, tmp_path: Path, fault: str) -> None:
    raw = json.loads(receipt.read_text(encoding="utf-8"))
    entry = raw["losslessEvidence"]["frames"][0]
    if fault == "count":
        entry["glyphPixels"] = 1
    elif fault == "duplicate":
        raw["losslessEvidence"]["frames"].append(entry.copy())
    elif fault == "format":
        raw["losslessEvidence"]["pixelFormat"] = "rgb24"
    elif fault == "pin":
        entry["sha256"] = "0" * 64
    else:
        pixels = Path(entry["path"])
        source = np.zeros((16, 20, 3), np.uint8)
        np.savez(
            pixels,
            source=source,
            candidate=source[:8] if fault == "shape" else source,
            glyph=np.zeros((16, 20), np.float32 if fault == "dtype" else np.uint8),
            repair=np.zeros((16, 20), np.uint8),
        )
        entry["sha256"] = digest(pixels)
    receipt.write_text(json.dumps(raw), encoding="utf-8")
    assert diagnostic.main([str(receipt), str(tmp_path / "bad"), "--sha256", digest(receipt)]) == 1


def test_bad_detector_and_png(
    receipt: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def wrong(images: Sequence[Image]) -> tuple[Image, ...]:
        return ()

    with pytest.raises(ReplayError, match="source detection"):
        diagnostic.export_capture(receipt, tmp_path / "bad", sha256=digest(receipt), detect=wrong)
    monkeypatch.setattr(cv2, "imencode", lambda *args: (False, np.zeros(1, np.uint8)))
    with pytest.raises(ReplayError, match="encoding"):
        diagnostic.export_capture(receipt, tmp_path / "png", sha256=digest(receipt))
    monkeypatch.setattr(cv2, "imencode", lambda *args: (True, np.zeros(1, np.uint8)))
    monkeypatch.setattr(cv2, "imdecode", lambda *args: np.zeros((1, 1), np.uint8))
    with pytest.raises(ReplayError, match="roundtrip"):
        diagnostic.export_capture(receipt, tmp_path / "roundtrip", sha256=digest(receipt))


def test_cli_detector(receipt: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    model = tmp_path / "model"
    model.write_bytes(b"synthetic")

    class Detector:
        def __init__(self, path: Path, *, sha256: str) -> None:
            pass

        def detect(self, images: Sequence[Image]) -> tuple[Image, ...]:
            return detection(images)

    monkeypatch.setattr(diagnostic, "LearnedTextDetector", Detector)
    args = [
        str(receipt),
        str(tmp_path / "cli-model"),
        "--sha256",
        digest(receipt),
        "--detector",
        str(model),
    ]
    assert diagnostic.main(args) == 1
    assert diagnostic.main([*args, "--detector-sha256", digest(model)]) == 0
    raw = json.loads((tmp_path / "cli-model" / "diagnostic.json").read_text(encoding="utf-8"))
    assert raw["frames"][0]["independentSourcePixels"] == 320
    assert raw["frames"][0]["glyphPixels"] == 0
