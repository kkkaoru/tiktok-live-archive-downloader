import json
from pathlib import Path
from unittest.mock import Mock

import cv2
import numpy as np
import pytest

from replay.head_seed import (
    main,
    measure_contour_tracks,
    propagate_contour,
    propagate_reviewed_arc,
    review_seed,
)
from replay.local_asr import fingerprint


def test_contour_translation_and_occlusion() -> None:
    source = np.random.default_rng(19).integers(0, 255, (96, 96, 3), dtype=np.uint8)
    source = np.asarray(cv2.GaussianBlur(source, (5, 5), 0), np.uint8)
    points = np.array([[32, 32], [48, 40], [64, 48]], np.float32)
    donor = np.asarray(
        cv2.warpAffine(source, np.array([[1, 0, 2], [0, 1, 0]], np.float32), (96, 96)), np.uint8
    )
    moved, valid = propagate_contour(source, donor, points)
    assert valid.tolist() == [True, True, True]
    np.testing.assert_allclose(moved, [[34, 32], [50, 40], [66, 48]], atol=0.1)
    donor[:] = 0
    _, supported = propagate_contour(source, donor, points)
    assert supported.tolist() == [False, False, False]


@pytest.mark.parametrize("reverse", [False, True])
def test_missing_tracks_are_unknown(monkeypatch: pytest.MonkeyPatch, reverse: bool) -> None:
    points = np.array([[32, 32], [48, 40], [64, 48]], np.float32)
    response = (points.reshape(-1, 1, 2), np.ones((3, 1), np.uint8), None)
    monkeypatch.setattr(
        cv2,
        "calcOpticalFlowPyrLK",
        Mock(side_effect=[response, (None, None, None)] if reverse else [(None, None, None)]),
    )
    source = np.zeros((96, 96, 3), np.uint8)
    _, valid = propagate_contour(source, source, points)
    assert valid.tolist() == [False, False, False]


def test_invalid_contour() -> None:
    source = np.zeros((96, 96, 3), np.uint8)
    with pytest.raises(ValueError, match="contour"):
        propagate_contour(source, source, np.array([[0, 0]], np.float32))


def test_metrics_keep_failure_causes_separate(monkeypatch: pytest.MonkeyPatch) -> None:
    points = np.array([[32, 32], [48, 40], [64, 48]], np.float32)
    backward = points.copy()
    backward[1, 0] += 1
    status = np.ones((3, 1), np.uint8)
    monkeypatch.setattr(
        cv2,
        "calcOpticalFlowPyrLK",
        Mock(
            side_effect=[
                (points.reshape(-1, 1, 2), status, None),
                (backward.reshape(-1, 1, 2), status, None),
            ]
        ),
    )
    source = np.zeros((96, 96, 3), np.uint8)
    donor = source.copy()
    donor[48, 64] = 20
    result = measure_contour_tracks(source, donor, points)
    assert result.supported.tolist() == [True, False, False]
    assert result.roundtrip_error.tolist() == [0, 1, 0]
    assert result.appearance_error.tolist() == [0, 0, 20]
    assert result.forward_status.all() and result.backward_status.all() and result.in_bounds.all()


@pytest.fixture
def config(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.chdir(tmp_path)
    source = tmp_path / "source.png"
    assert cv2.imwrite(str(source), np.zeros((96, 96, 3), np.uint8))
    path = tmp_path / "seed.json"
    path.write_text(
        json.dumps(
            {
                "sourcePNG": {"path": str(source), "sha256": fingerprint(source)},
                "sourcePTS": 120,
                "timeBase": "1/10",
                "sourceMediaSHA256": "a" * 64,
                "points": [[32, 32], [48, 40], [64, 48]],
            }
        ),
        encoding="utf-8",
    )
    return path


def test_review_is_actual_image_not_approval(config: Path) -> None:
    output = config.parent / "review"
    assert main([str(config), str(output)]) == 0
    result = json.loads((output / "seed-review.json").read_text(encoding="utf-8"))
    assert result["seedReviewed"] is False
    assert result["independentHeadObservations"] == 0
    assert result["candidateTop"] == 32
    image = cv2.imread(str(output / "seed-review.png"))
    assert image is not None
    assert image[32, 32].tolist() == [0, 255, 0]
    with pytest.raises(ValueError, match="new checkout"):
        review_seed(config, output)


def test_changed_source_refused(config: Path) -> None:
    (config.parent / "source.png").write_bytes(b"changed")
    with pytest.raises(ValueError, match="binding"):
        review_seed(config, config.parent / "review")


def test_failed_image_write(config: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(cv2, "imwrite", Mock(return_value=False))
    with pytest.raises(OSError, match="write"):
        review_seed(config, config.parent / "review")


def test_bad_png(config: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(cv2, "imread", Mock(return_value=None))
    with pytest.raises(ValueError, match="decode"):
        review_seed(config, config.parent / "review")


@pytest.fixture
def arc(config: Path) -> Path:
    data = json.loads(config.read_text(encoding="utf-8"))
    review = config.parent / "arc-review.json"
    review.write_text(
        json.dumps(
            {
                "reviewerKind": "agent-visual-review",
                "visibleArcReviewed": True,
                "scope": "Synthetic partial arc only",
            }
        ),
        encoding="utf-8",
    )
    data["visibleArcReview"] = {"path": str(review), "sha256": fingerprint(review)}
    data["donors"] = [{"sourcePTS": 121, "sourcePNG": data["sourcePNG"]}]
    config.write_text(json.dumps(data), encoding="utf-8")
    return config


def test_arc_retains_unknown_without_whole_head_claim(arc: Path) -> None:
    output = arc.parent / "tracks"
    assert main([str(arc), str(output), "--propagate-reviewed-arc"]) == 0
    result = json.loads((output / "arc-tracks.json").read_text(encoding="utf-8"))
    assert result["visibleArcReviewed"] is True
    assert result["seedReviewed"] is False
    assert result["headContourComplete"] is False
    assert result["independentHeadObservations"] == 0
    assert result["observations"][0]["supported"] == [False, False, False]
    assert result["observations"][0]["unknownPoints"] == 3


def test_arc_nonfinite_tracks_remain_unknown(arc: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "replay.head_seed.propagate_contour",
        Mock(return_value=(np.full((3, 2), np.nan, np.float32), np.zeros(3, np.bool_))),
    )
    receipt = propagate_reviewed_arc(arc, arc.parent / "tracks")
    result = json.loads(receipt.read_text(encoding="utf-8"))
    assert result["observations"][0]["points"] == [None, None, None]
    assert result["observations"][0]["unknownPoints"] == 3
    assert result["sourceClockIndependentlyVerified"] is False


@pytest.mark.parametrize("fault", ["review", "clock", "duplicate", "empty", "png"])
def test_arc_invalid(arc: Path, fault: str) -> None:
    data = json.loads(arc.read_text(encoding="utf-8"))
    if fault == "review":
        path = arc.parent / "arc-review.json"
        path.write_text("{}", encoding="utf-8")
        data["visibleArcReview"]["sha256"] = fingerprint(path)
    elif fault == "clock":
        data["timeBase"] = "0"
    elif fault == "duplicate":
        data["donors"][0]["sourcePTS"] = 120
    else:
        path = arc.parent / "bad.png"
        path.write_bytes(b"bad" if fault == "png" else b"")
        data["donors"][0]["sourcePNG"] = {"path": str(path), "sha256": fingerprint(path)}
    arc.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(ValueError):
        propagate_reviewed_arc(arc, arc.parent / "tracks")
