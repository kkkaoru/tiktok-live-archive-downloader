"""Synthetic saved-image entry points, without media/models or project paths."""

import json
from pathlib import Path
from typing import Literal

import cv2
import numpy as np
import pytest

from replay import subtitle_saved_observe as saved
from replay.core import ReplayError
from replay.delivery_mask_quality import digest, read_json, rows
from replay.recordings import record
from replay.subtitle_quality_observe import (
    ObservationFrame,
    TemporalObservation,
    temporal_observation,
)


@pytest.fixture
def config(tmp_path: Path) -> Path:
    source = np.random.default_rng(42).integers(0, 255, (96, 128, 3), dtype=np.uint8)
    support = np.zeros((96, 128), np.uint8)
    support[40:46, 50:58] = 255
    assert cv2.imwrite(str(tmp_path / "source.png"), source)
    assert cv2.imwrite(str(tmp_path / "support.png"), support)
    source_ref = {"path": str(tmp_path / "source.png"), "sha256": digest(tmp_path / "source.png")}
    support_ref = {
        "path": str(tmp_path / "support.png"),
        "sha256": digest(tmp_path / "support.png"),
    }
    receipt = tmp_path / "receipt.json"
    receipt.write_text('{"synthetic": true}', encoding="utf-8")
    row = {
        "source": source_ref,
        "candidate": source_ref,
        "glyph": [support_ref],
        "repair": [support_ref],
    }
    path = tmp_path / "config.json"
    path.write_text(
        json.dumps(
            {
                "version": 1,
                "receipts": [{"path": str(receipt), "sha256": digest(receipt)}],
                "clock": {"pts": [0, 2, 5, 9], "timeBase": "1/100", "lastDuration": 3},
                "frames": [
                    {**row, "sourcePTS": 0},
                    {**row, "sourcePTS": 2},
                    {**row, "sourcePTS": 5},
                    {
                        **row,
                        "sourcePTS": 9,
                        "comparisons": [{"name": "saved-alternative", "image": source_ref}],
                    },
                ],
            }
        ),
        encoding="utf-8",
    )
    return path


def test_cli_four_frames(config: Path, tmp_path: Path) -> None:
    output = tmp_path / "result.json"
    assert saved.main([str(config), str(output), "--sha256", digest(config)]) == 0
    result = read_json(output)
    frames = rows(result["frames"])
    terminal = record(rows(record(frames[3])["states"])[0])
    metric = record(rows(record(frames[0])["metrics"])[0])
    assert result["inputsBefore"] == result["inputsAfter"]
    assert result["qualityApproved"] is False
    assert result["sourceMediaReverified"] is False
    assert len(frames) == 4
    assert terminal["registration"] is None
    assert terminal["unknownPixels"] == 48
    assert metric["outsideChangedPixels"] == 0
    assert output.stat().st_mode & 0o777 == 0o600
    assert saved.main([str(config), str(output), "--sha256", digest(config)]) == 1


@pytest.mark.parametrize(
    "fault",
    [
        "version",
        "empty",
        "large",
        "base",
        "clock",
        "comparison",
        "name",
        "canvas",
        "pin",
        "binary",
        "support",
        "glyph",
        "png",
        "bytes",
        "budget",
        "channels",
        "order",
        "escape",
    ],
)
def test_reject_invalid(config: Path, tmp_path: Path, fault: str) -> None:
    raw = read_json(config)
    clock = record(raw["clock"])
    frames = rows(raw["frames"])
    first = record(frames[0])
    last = record(frames[3])
    comparisons = rows(last["comparisons"])
    comparison = record(comparisons[0])
    if fault == "version":
        raw["version"] = 2
    elif fault == "empty":
        raw["frames"] = []
    elif fault == "large":
        clock["pts"] = rows(clock["pts"]) * 3
    elif fault == "base":
        clock["timeBase"] = 100
    elif fault == "clock":
        first["sourcePTS"] = 1
    elif fault == "comparison":
        last["comparisons"] = comparisons * 17
    elif fault == "name":
        comparison["name"] = None
    elif fault == "pin":
        record(first["source"])["sha256"] = "0" * 64
    elif fault == "order":
        clock["pts"] = [0, 5, 2, 9]
    elif fault == "escape":
        linked = tmp_path / "external"
        linked.symlink_to("/etc/hosts")
        record(first["source"])["path"] = str(linked)
    else:
        path = tmp_path / "bad.png"
        image = np.zeros((96, 128, 3), np.uint8)
        if fault in ("binary", "support", "glyph"):
            image = np.full((96, 128), 100 if fault == "binary" else 255, np.uint8)
        if fault in ("canvas", "support"):
            image = image[:32]
        if fault == "channels":
            image = np.zeros((96, 128, 4), np.uint8)
        assert cv2.imwrite(str(path), image)
        if fault == "png":
            path.write_bytes(b"x" * 40)
        elif fault == "bytes":
            path.write_bytes(b"x")
        elif fault == "budget":
            data = path.read_bytes()
            path.write_bytes(data[:16] + b"\xff" * 8 + data[24:])
        ref = {"path": str(path), "sha256": digest(path)}
        if fault in ("binary", "support", "glyph"):
            first["glyph"] = [ref]
        elif fault == "canvas":
            comparison["image"] = ref
        else:
            first["source"] = ref
    config.write_text(json.dumps(raw), encoding="utf-8")
    assert saved.main([str(config), str(tmp_path / "bad.json"), "--sha256", digest(config)]) == 1
    assert not (tmp_path / "bad.json").exists()


def test_single_empty_support(config: Path, tmp_path: Path) -> None:
    raw = read_json(config)
    assert cv2.imwrite(str(tmp_path / "empty.png"), np.zeros((96, 128), np.uint8))
    ref = {"path": str(tmp_path / "empty.png"), "sha256": digest(tmp_path / "empty.png")}
    raw["frames"] = [{**record(rows(raw["frames"])[0]), "glyph": [ref], "repair": [ref]}]
    record(raw["clock"])["pts"] = [0]
    config.write_text(json.dumps(raw), encoding="utf-8")
    result = saved.observe_saved(config, sha256=digest(config))
    metrics = rows(record(rows(result["frames"])[0])["metrics"])
    assert record(metrics[0])["boundaryAdjacentCodeDifference"] == {
        "count": 0,
        "mean": None,
        "p95": None,
        "max": None,
    }


def test_changed_during_measurement(config: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    def change(
        frame: ObservationFrame,
        *,
        registration_model: Literal["translation", "local-affine"] = "translation",
    ) -> TemporalObservation:
        config.write_text("{}", encoding="utf-8")
        return temporal_observation(frame, registration_model=registration_model)

    monkeypatch.setattr(saved, "temporal_observation", change)
    with pytest.raises(ReplayError, match="changed during"):
        saved.observe_saved(config, sha256=digest(config))
