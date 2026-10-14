import json
import os
import sys
from pathlib import Path

import pytest

from replay import fcp_native_progress as progress
from replay.core import ReplayError
from replay.stages import file_digest


@pytest.fixture
def plan(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    movie = tmp_path / "part.mov"
    movie.write_bytes(b"media")
    os.utime(movie, (1, 1))
    proof = tmp_path / "proof.json"
    proof.write_text(
        json.dumps(
            {
                "movie": {"path": str(movie), "sha256": file_digest(movie)},
                "observed": {"frames": [{"pts": 0}, {"pts": 1}]},
            }
        )
    )
    assembly = tmp_path / "assembly.json"
    assembly.write_text("{}")
    state = tmp_path / "progress.json"
    state.write_text(
        json.dumps(
            {
                "version": 1,
                "libraryAndMasterRoot": str(tmp_path),
                "parts": [{"part": 1, "master": "part.mov", "state": "rendering"}],
            }
        )
    )
    status = tmp_path / "decode.status"
    status.write_text("0\n")
    background = tmp_path / "background.json"
    background.write_text(
        json.dumps(
            {
                "elements": [
                    {"role": "AXStaticText", "value": "Sharing"},
                    {"identifier": "_NS:166", "value": "Idle"},
                ]
            }
        )
    )
    config = tmp_path / "plan.json"
    config.write_text(
        json.dumps(
            {
                "version": 1,
                "part": 1,
                "progress": {"path": str(state), "sha256": file_digest(state)},
                "assembly": {"path": str(assembly), "sha256": file_digest(assembly)},
                "decodeStatus": {"path": str(status), "sha256": file_digest(status)},
                "background": {"path": str(background), "sha256": file_digest(background)},
                "receiptPath": str(tmp_path / "receipt.json"),
            }
        )
    )
    monkeypatch.setattr(progress, "verify", lambda config, *, part: proof)
    return config


def test_updates_only_verified_completion(plan: Path) -> None:
    progress.update(plan)
    row = json.loads((plan.parent / "progress.json").read_text())["parts"][0]
    assert row["state"] == "complete"
    assert row["videoFrames"] == 2
    assert row["sizeBytes"] == 5
    assert row["qualityApproved"] is False
    receipt = json.loads((plan.parent / "receipt.json").read_text())
    assert receipt["previousDocument"]["parts"][0]["state"] == "rendering"
    assert receipt["updatedSHA256"] == file_digest(plan.parent / "progress.json")


@pytest.mark.parametrize("change", ["decode", "old-background", "movie", "progress", "receipt"])
def test_rejects_incomplete_or_changed_evidence(plan: Path, change: str) -> None:
    raw = json.loads(plan.read_text())
    if change == "decode":
        status = Path(raw["decodeStatus"]["path"])
        status.write_text("1\n")
        raw["decodeStatus"]["sha256"] = file_digest(status)
    elif change == "old-background":
        os.utime(raw["background"]["path"], (0, 0))
    elif change == "movie":
        (plan.parent / "part.mov").write_bytes(b"changed")
    elif change == "progress":
        Path(raw["progress"]["path"]).write_text("{}")
    else:
        Path(raw["receiptPath"]).write_text("{}")
    plan.write_text(json.dumps(raw))
    with pytest.raises(ReplayError):
        progress.update(plan)


@pytest.mark.parametrize("value", [0, True, "1"])
def test_rejects_invalid_part(plan: Path, value: object) -> None:
    raw = json.loads(plan.read_text())
    raw["part"] = value
    plan.write_text(json.dumps(raw))
    with pytest.raises(ReplayError, match="positive"):
        progress.update(plan)


@pytest.mark.parametrize(
    "state",
    [
        {"version": 1, "parts": []},
        {"version": 1, "parts": [{"part": 1, "master": "wrong.mov"}]},
        {"version": 2, "parts": []},
    ],
)
def test_rejects_progress_identity(plan: Path, state: dict[str, object]) -> None:
    raw = json.loads(plan.read_text())
    state["libraryAndMasterRoot"] = str(plan.parent)
    path = Path(raw["progress"]["path"])
    path.write_text(json.dumps(state))
    raw["progress"]["sha256"] = file_digest(path)
    plan.write_text(json.dumps(raw))
    with pytest.raises(ReplayError):
        progress.update(plan)


def test_cli(plan: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sys, "argv", ["native-progress", str(plan)])
    progress.main()
    assert (plan.parent / "receipt.json").exists()


def test_concurrent_progress_change_is_rejected(
    plan: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def changed(config: Path, *, part: int) -> Path:
        (plan.parent / "progress.json").write_text('{"concurrent":true}')
        return plan.parent / "proof.json"

    monkeypatch.setattr(progress, "verify", changed)
    with pytest.raises(ReplayError, match="changed during"):
        progress.update(plan)
    assert not (plan.parent / "receipt.json").exists()
