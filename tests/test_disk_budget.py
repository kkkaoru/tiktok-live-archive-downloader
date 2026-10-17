from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from replay import disk_budget
from replay.core import ReplayError

GIB = 1024**3
MIB = 1024 * 1024


def _write(path: Path, size: int) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"x" * size)
    return path


def test_floor_is_thirty_gib_on_a_small_volume() -> None:
    assert disk_budget.floor_bytes(100 * GIB) == 30 * GIB


def test_floor_is_five_percent_on_a_large_volume() -> None:
    assert disk_budget.floor_bytes(1000 * GIB) == 50 * GIB


def test_spare_subtracts_need_and_floor() -> None:
    volume = disk_budget.Headroom(Path("/v"), 1000 * GIB, 200 * GIB, 50 * GIB, 20 * GIB)
    assert volume.spare == 130 * GIB


def test_measure_reads_the_real_volume(tmp_path: Path) -> None:
    volume = disk_budget.measure(tmp_path, 7)
    assert volume.need == 7
    assert volume.total > 0
    assert volume.floor >= 30 * GIB


def test_require_headroom_returns_every_measurement(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(
        disk_budget,
        "measure",
        lambda path, need: disk_budget.Headroom(path, 1000 * GIB, 400 * GIB, 50 * GIB, need),
    )
    volumes = disk_budget.require_headroom([tmp_path, tmp_path / "other"], 10 * GIB)
    assert [volume.spare for volume in volumes] == [340 * GIB, 340 * GIB]


def test_require_headroom_refuses_below_the_floor(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(
        disk_budget,
        "measure",
        lambda path, need: disk_budget.Headroom(path, 1000 * GIB, 60 * GIB, 50 * GIB, need),
    )
    with pytest.raises(ReplayError, match="Free space on .* is 60.0 GiB; 70.0 GiB is required"):
        disk_budget.require_headroom([tmp_path], 20 * GIB)


def test_shared_volume_needs_are_summed_once(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    (tmp_path / "work").mkdir()
    (tmp_path / "storage").mkdir()
    monkeypatch.setattr(
        disk_budget,
        "measure",
        lambda path, need: disk_budget.Headroom(path, 1000 * GIB, 400 * GIB, 50 * GIB, need),
    )
    volumes = disk_budget.require_shared_headroom(
        {tmp_path / "work": 100 * GIB, tmp_path / "storage": 60 * GIB}
    )
    assert [(volume.path, volume.need) for volume in volumes] == [(tmp_path / "work", 160 * GIB)]


def test_shared_volume_refuses_when_the_summed_need_breaks_the_floor(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    (tmp_path / "work").mkdir()
    (tmp_path / "storage").mkdir()
    monkeypatch.setattr(
        disk_budget,
        "measure",
        lambda path, need: disk_budget.Headroom(path, 1000 * GIB, 200 * GIB, 50 * GIB, need),
    )
    with pytest.raises(ReplayError, match="is 200.0 GiB; 250.0 GiB is required"):
        disk_budget.require_shared_headroom(
            {tmp_path / "work": 100 * GIB, tmp_path / "storage": 100 * GIB}
        )


def test_physical_usage_counts_real_allocation_by_directory_and_extension(tmp_path: Path) -> None:
    _write(tmp_path / "a" / "b" / "c" / "d" / "big.mov", 2 * MIB)
    _write(tmp_path / "a" / "audio.wav", 3 * MIB)
    _write(tmp_path / "small.wav", 1000)
    report = disk_budget.physical_usage(tmp_path, min_bytes=MIB, depth=2)
    assert report.by_directory["a/b"].files == 1
    assert report.by_directory["a"].files == 1
    assert report.by_extension[".mov"].files == 1
    assert report.by_extension[".wav"].files == 1
    assert ".wav" in report.by_extension
    assert report.allocated >= 5 * MIB


def test_physical_usage_counts_a_hardlinked_inode_once(tmp_path: Path) -> None:
    original = _write(tmp_path / "x" / "one.mov", 2 * MIB)
    os.link(original, tmp_path / "x" / "two.mov")
    report = disk_budget.physical_usage(tmp_path, min_bytes=MIB)
    assert report.by_extension[".mov"].files == 1


def test_physical_usage_ignores_sparse_files(tmp_path: Path) -> None:
    sparse = tmp_path / "sparse.mp4"
    with sparse.open("wb") as stream:
        stream.truncate(500 * MIB)
    report = disk_budget.physical_usage(tmp_path, min_bytes=MIB)
    assert report.allocated == 0


def test_physical_usage_skips_protected_folders_and_symlinks(tmp_path: Path) -> None:
    _write(tmp_path / "private" / "secret.mov", 2 * MIB)
    _write(tmp_path / "downloads" / "original.mov", 2 * MIB)
    outside = _write(tmp_path / "keep" / "real.mov", 2 * MIB)
    (tmp_path / "alias.mov").symlink_to(outside)
    report = disk_budget.physical_usage(tmp_path, min_bytes=MIB)
    assert list(report.by_directory) == ["keep"]
    assert report.by_extension[".mov"].files == 1


def test_files_at_the_root_are_reported_under_a_dot(tmp_path: Path) -> None:
    _write(tmp_path / "root.m4a", 2 * MIB)
    report = disk_budget.physical_usage(tmp_path, min_bytes=MIB)
    assert list(report.by_directory) == ["."]


def test_files_without_extension_are_grouped(tmp_path: Path) -> None:
    _write(tmp_path / "blob", 2 * MIB)
    report = disk_budget.physical_usage(tmp_path, min_bytes=MIB)
    assert list(report.by_extension) == ["(none)"]


def test_render_ranks_by_allocated_bytes(tmp_path: Path) -> None:
    _write(tmp_path / "small" / "a.wav", 2 * MIB)
    _write(tmp_path / "large" / "b.wav", 4 * MIB)
    report = disk_budget.physical_usage(tmp_path, min_bytes=MIB)
    rendered = json.loads(disk_budget.render(report, 1))
    assert [row["name"] for row in rendered["byDirectory"]] == ["large"]


def test_volume_table_shows_spare_bytes() -> None:
    table = disk_budget.volume_table(
        [disk_budget.Headroom(Path("/v"), 1000 * GIB, 200 * GIB, 50 * GIB, 0)]
    )
    assert table == {
        "/v": {
            "totalBytes": 1000 * GIB,
            "freeBytes": 200 * GIB,
            "floorBytes": 50 * GIB,
            "spareBytes": 150 * GIB,
        }
    }


def test_report_command_prints_json(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    _write(tmp_path / "d" / "a.wav", 2 * MIB)
    assert disk_budget.main(["report", str(tmp_path), "--min-mib", "1"]) == 0
    assert json.loads(capsys.readouterr().out)["byExtension"][0]["name"] == ".wav"


def test_preflight_command_passes_when_space_is_ample(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(
        disk_budget,
        "measure",
        lambda path, need: disk_budget.Headroom(path, 1000 * GIB, 400 * GIB, 50 * GIB, need),
    )
    assert disk_budget.main(["preflight", "--path", str(tmp_path)]) == 0
    assert json.loads(capsys.readouterr().out)[str(tmp_path)]["spareBytes"] == 350 * GIB


def test_preflight_command_fails_below_the_floor(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(
        disk_budget,
        "measure",
        lambda path, need: disk_budget.Headroom(path, 1000 * GIB, 40 * GIB, 50 * GIB, need),
    )
    assert disk_budget.main(["preflight", "--path", str(tmp_path)]) == 1
    assert "DISK_BUDGET_ERROR" in capsys.readouterr().out
