import json
from pathlib import Path
from unittest.mock import Mock

import pytest
from pytest import MonkeyPatch

from replay import delivery_mask_probe as probe
from replay.core import ReplayError
from replay.delivery_mask_quality import digest


@pytest.fixture
def producer(tmp_path: Path, monkeypatch: MonkeyPatch) -> Path:
    monkeypatch.chdir(tmp_path)
    path = tmp_path / "producer.json"
    path.write_text(
        json.dumps(
            {
                "clock": {"pts": [30, 32, 35], "timeBase": "1/30", "lastDuration": 4},
                "width": 16,
                "height": 16,
                "masks": [
                    {
                        "startSeconds": 0,
                        "endSeconds": 1,
                        "region": {"x": 0, "y": 0, "width": 16, "height": 16},
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(probe, "render_candidate", Mock(side_effect=lambda config: config))
    return path


def test_probe_preserves_pts_ramp_and_parent(producer: Path) -> None:
    result = probe.probe(
        producer, producer.parent / "new", sha256=digest(producer), start=1, count=2
    )
    data = json.loads(result.read_text())
    assert data["clock"] == {"pts": [32, 35], "timeBase": "1/30", "lastDuration": 4}
    assert data["evidenceFrames"] == [0, 1]
    assert data["rampStartFrame"] == 1
    assert data["rampTotalFrames"] == 3
    assert data["probeParent"]["sha256"] == digest(producer)
    assert data["masks"] == [
        {
            "startSeconds": 0.0,
            "endSeconds": 0.1,
            "region": {"x": 0, "y": 0, "width": 16, "height": 16},
        },
        {
            "startSeconds": 0.1,
            "endSeconds": 0.23333333333333334,
            "region": {"x": 0, "y": 0, "width": 16, "height": 16},
        },
    ]
    assert result.stat().st_mode & 0o777 == 0o600
    with pytest.raises(FileExistsError):
        probe.probe(producer, producer.parent / "new", sha256=digest(producer), start=1, count=2)


def test_cli_inner_clock(producer: Path) -> None:
    assert (
        probe.main(
            [
                str(producer),
                str(producer.parent / "new"),
                "--sha256",
                digest(producer),
                "--start-frame",
                "0",
                "--count",
                "2",
            ]
        )
        == 0
    )
    data = json.loads((producer.parent / "new/probe-config.json").read_text())
    assert data["clock"]["lastDuration"] == 3


@pytest.mark.parametrize("start,count", [(-1, 1), (True, 1), (0, 0), (0, 9), (0, True), (2, 2)])
def test_invalid_bounds(producer: Path, start: int, count: int) -> None:
    with pytest.raises(ReplayError):
        probe.probe(
            producer, producer.parent / "new", sha256=digest(producer), start=start, count=count
        )
    assert not (producer.parent / "new").exists()


@pytest.mark.parametrize("change", [{"timeBase": None}, {"timeBase": "0"}])
def test_invalid_clock(producer: Path, change: dict[str, object]) -> None:
    data = json.loads(producer.read_text())
    data["clock"].update(change)
    producer.write_text(json.dumps(data))
    with pytest.raises(ReplayError):
        probe.probe(producer, producer.parent / "new", sha256=digest(producer), start=0, count=1)


def test_empty_mask(producer: Path) -> None:
    data = json.loads(producer.read_text())
    data["masks"] = []
    producer.write_text(json.dumps(data))
    with pytest.raises(ReplayError, match="lacks planned mask"):
        probe.probe(producer, producer.parent / "new", sha256=digest(producer), start=0, count=1)


def test_changed_parent(producer: Path, monkeypatch: MonkeyPatch) -> None:
    with pytest.raises(ReplayError, match="configuration changed"):
        probe.probe(producer, producer.parent / "new", sha256="0" * 64, start=0, count=1)

    def change(config: Path) -> Path:
        producer.write_text("{}")
        return config

    monkeypatch.setattr(probe, "render_candidate", change)
    with pytest.raises(ReplayError, match="during probe"):
        probe.probe(producer, producer.parent / "new", sha256=digest(producer), start=0, count=1)
