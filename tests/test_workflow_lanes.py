"""Lane splitting keeps dependencies together and balances independent stages."""

import json
import subprocess
from pathlib import Path

import pytest

from replay import workflow_lanes
from replay.core import ReplayError


def _stage(root: Path, name: str, source: str, output: str) -> dict[str, object]:
    return {
        "name": name,
        "command": ["/bin/echo", name],
        "inputs": [str(root / source)],
        "outputs": [str(root / output)],
        "timeout_seconds": 60,
    }


def _write(root: Path, stages: list[dict[str, object]]) -> Path:
    manifest = root / "manifest.json"
    manifest.write_text(json.dumps({"version": 1, "stages": stages}), encoding="utf-8")
    return manifest


def _independent(root: Path, count: int) -> list[dict[str, object]]:
    return [_stage(root, f"win-{n}", f"in-{n}", f"out-{n}") for n in range(count)]


def test_independent_stages_are_balanced_across_lanes(tmp_path: Path) -> None:
    manifest = _write(tmp_path, _independent(tmp_path, 6))
    lanes = workflow_lanes.split(manifest, tmp_path / "lanes", 3)
    assert [len(lane.stages) for lane in lanes] == [2, 2, 2]


def test_every_stage_appears_in_exactly_one_lane(tmp_path: Path) -> None:
    manifest = _write(tmp_path, _independent(tmp_path, 5))
    lanes = workflow_lanes.split(manifest, tmp_path / "lanes", 2)
    assert sorted(name for lane in lanes for name in lane.stages) == [
        "win-0",
        "win-1",
        "win-2",
        "win-3",
        "win-4",
    ]


def test_dependent_stages_stay_together_in_order(tmp_path: Path) -> None:
    stages = [
        _stage(tmp_path, "second", "mid", "last"),
        _stage(tmp_path, "first", "in-a", "mid"),
        _stage(tmp_path, "other", "in-b", "out-b"),
    ]
    lanes = workflow_lanes.split(_write(tmp_path, stages), tmp_path / "lanes", 2)
    assert [lane.stages for lane in lanes] == [("first", "second"), ("other",)]


def test_lane_count_is_capped_by_independent_groups(tmp_path: Path) -> None:
    manifest = _write(tmp_path, _independent(tmp_path, 2))
    lanes = workflow_lanes.split(manifest, tmp_path / "lanes", 8)
    assert len(lanes) == 2


def test_lane_manifest_is_an_ordinary_workflow_with_a_private_state(tmp_path: Path) -> None:
    manifest = _write(tmp_path, _independent(tmp_path, 2))
    lanes = workflow_lanes.split(manifest, tmp_path / "lanes", 2)
    written = json.loads(lanes[0].manifest.read_text(encoding="utf-8"))
    assert (written["version"], [item["name"] for item in written["stages"]]) == (1, ["win-0"])
    assert oct(lanes[0].state.stat().st_mode & 0o777) == "0o700"


def test_nonempty_output_is_refused(tmp_path: Path) -> None:
    manifest = _write(tmp_path, _independent(tmp_path, 2))
    (tmp_path / "lanes").mkdir()
    (tmp_path / "lanes" / "keep").write_text("x", encoding="utf-8")
    with pytest.raises(ReplayError, match="new or empty directory"):
        workflow_lanes.split(manifest, tmp_path / "lanes", 2)


def test_lane_count_is_bounded(tmp_path: Path) -> None:
    manifest = _write(tmp_path, _independent(tmp_path, 2))
    with pytest.raises(ReplayError, match="between 1 and 16"):
        workflow_lanes.split(manifest, tmp_path / "lanes", 17)


def test_manifest_with_extra_fields_is_refused(tmp_path: Path) -> None:
    manifest = tmp_path / "manifest.json"
    stages = _independent(tmp_path, 1)
    manifest.write_text(
        json.dumps({"version": 1, "stages": stages, "sourceCaptionProduction": {}}),
        encoding="utf-8",
    )
    with pytest.raises(ReplayError, match="plain version and stages"):
        workflow_lanes.split(manifest, tmp_path / "lanes", 2)


def test_launch_starts_one_detached_job_per_lane(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    calls: list[list[str]] = []

    def fake_run(command: list[str], **_: object) -> subprocess.CompletedProcess[str]:
        calls.append(command)
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(subprocess, "run", fake_run)
    manifest = _write(tmp_path, _independent(tmp_path, 2))
    lanes = workflow_lanes.split(manifest, tmp_path / "lanes", 2)
    names = workflow_lanes.launch(lanes, tmp_path / "lanes", "track")
    assert (names, [call[2] for call in calls]) == (
        ["track-lane00", "track-lane01"],
        ["track-lane00", "track-lane01"],
    )


def test_cli_prints_the_lane_summary(capsys: pytest.CaptureFixture[str], tmp_path: Path) -> None:
    manifest = _write(tmp_path, _independent(tmp_path, 3))
    code = workflow_lanes.main([str(manifest), str(tmp_path / "lanes"), "--lanes", "3"])
    summary = json.loads(capsys.readouterr().out)
    assert (code, [item["stages"] for item in summary["lanes"]], summary["jobs"]) == (
        0,
        [1, 1, 1],
        [],
    )


def test_cli_reports_a_refused_split(capsys: pytest.CaptureFixture[str], tmp_path: Path) -> None:
    manifest = _write(tmp_path, _independent(tmp_path, 1))
    code = workflow_lanes.main([str(manifest), str(tmp_path / "lanes"), "--lanes", "0"])
    assert (code, "WORKFLOW_LANES_ERROR" in capsys.readouterr().err) == (1, True)
