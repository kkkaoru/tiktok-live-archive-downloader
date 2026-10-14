"""Original-donor entry point checks using synthetic images only."""

import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from replay import subtitle_context_candidate as candidate
from replay.delivery_mask_quality import digest, read_json, rows
from replay.recordings import record


@pytest.fixture
def config(tmp_path: Path) -> Path:
    image = np.random.default_rng(32).integers(20, 220, (256, 256, 3), dtype=np.uint8)
    image = cv2.GaussianBlur(image, (5, 5), 0)
    glyph = np.zeros((256, 256), np.uint8)
    glyph[120:124, 120:124] = 255
    target = image.copy()
    target[glyph > 0] = 255
    assert cv2.imwrite(str(tmp_path / "target.png"), target)
    assert cv2.imwrite(str(tmp_path / "donor.png"), image)
    assert cv2.imwrite(str(tmp_path / "glyph.png"), glyph)
    assert cv2.imwrite(str(tmp_path / "clear.png"), np.zeros_like(glyph))
    review = tmp_path / "review.json"
    review.write_text(
        json.dumps(
            {
                "reviewerKind": "agent-visual-review",
                "originalDonorClearanceReviewed": True,
                "scope": "Synthetic original donors are entirely clear.",
            }
        ),
        encoding="utf-8",
    )
    source = {"path": str(tmp_path / "target.png"), "sha256": digest(tmp_path / "target.png")}
    donor = {"path": str(tmp_path / "donor.png"), "sha256": digest(tmp_path / "donor.png")}
    mask = {"path": str(tmp_path / "glyph.png"), "sha256": digest(tmp_path / "glyph.png")}
    clear = {"path": str(tmp_path / "clear.png"), "sha256": digest(tmp_path / "clear.png")}
    path = tmp_path / "config.json"
    path.write_text(
        json.dumps(
            {
                "version": 1,
                "timeBase": "1/30",
                "review": {"path": str(review), "sha256": digest(review)},
                "frames": [
                    {"sourcePTS": 1, "image": source, "excluded": mask},
                    {"sourcePTS": 0, "image": donor, "excluded": clear},
                    {"sourcePTS": 2, "image": donor, "excluded": clear},
                ],
            }
        ),
        encoding="utf-8",
    )
    return path


def test_real_kernel(config: Path, tmp_path: Path) -> None:
    output = tmp_path / "result"
    assert candidate.main([str(config), str(output), "--sha256", digest(config)]) == 0
    report = read_json(output / "context.json")
    assert report["targetPixels"] == 16
    assert report["supportedPixels"] == 16
    assert report["unknownPixels"] == 0
    assert report["qualityApproved"] is False
    assert report["fullCaptionSupport"] is False
    geometry = record(rows(report["forwardLocalGeometry"])[0])
    diagnostics = record(geometry["neighborhoodDiagnostics"])
    assert diagnostics["invalid_tracks"] == 0
    assert diagnostics["invalid_neighbors"] == 0
    assert diagnostics["residual"] == 0
    restored = cv2.imread(str(output / "candidate.png"))
    original = cv2.imread(str(tmp_path / "donor.png"))
    assert restored is not None
    assert original is not None
    assert np.array_equal(restored, original)
    with pytest.raises(FileExistsError):
        candidate.restore_context(config, output, sha256=digest(config))


@pytest.mark.parametrize("fault", ["version", "review", "base", "count", "clock", "shape", "path"])
def test_invalid(config: Path, tmp_path: Path, fault: str) -> None:
    data = read_json(config)
    frames = rows(data["frames"])
    output = tmp_path / "result"
    if fault == "version":
        data["version"] = 2
    elif fault == "review":
        review = tmp_path / "review.json"
        review.write_text("{}", encoding="utf-8")
        data["review"] = {"path": str(review), "sha256": digest(review)}
    elif fault == "base":
        data["timeBase"] = "0"
    elif fault == "count":
        data["frames"] = []
    elif fault == "clock":
        record(frames[1])["sourcePTS"] = 2
    elif fault == "shape":
        mask = tmp_path / "wrong.png"
        assert cv2.imwrite(str(mask), np.zeros((32, 32), np.uint8))
        record(frames[0])["excluded"] = {"path": str(mask), "sha256": digest(mask)}
    else:
        output = Path("relative-output")
    config.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(ValueError):
        candidate.restore_context(config, output, sha256=digest(config))


def test_write_failure(config: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(cv2, "imwrite", lambda *args: False)
    with pytest.raises(OSError, match="Cannot write"):
        candidate.restore_context(config, tmp_path / "failed", sha256=digest(config))
