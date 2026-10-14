import json
import sys
from pathlib import Path

import pytest

from replay import motion_measured
from replay.recordings import record
from replay.stages import file_digest


@pytest.fixture
def candidate(tmp_path: Path) -> Path:
    evidence = tmp_path / "calibration.json"
    evidence.write_text("{}", encoding="utf-8")
    path = tmp_path / "candidate.json"
    path.write_text(
        json.dumps(
            {
                "version": 1,
                "candidateOnly": True,
                "canvas": [1080, 1920],
                "coordinateSystem": "top-left canvas; calibrated text origin",
                "policy": {"frame_rate": 30},
                "renderCalibration": {"path": str(evidence), "sha256": file_digest(evidence)},
                "inputs": {str(evidence): file_digest(evidence)},
                "candidates": [
                    {"cue": {"text": "あ\nい", "start": 0.1, "end": 0.9}, "x": 540, "y": 500}
                ],
            }
        ),
        encoding="utf-8",
    )
    return path


def test_export_local_clock_origin_and_false_approval(candidate: Path, tmp_path: Path) -> None:
    result = motion_measured.export_plans(
        candidate,
        tmp_path / "parts",
        sha256=file_digest(candidate),
        width=1080,
        height=1920,
        rate=30,
        frames=30,
        expected_text="Title",
    )
    raw = record(json.loads(result.read_text(encoding="utf-8")))
    assert raw["motionRenderVerified"] is False
    assert raw["deliveryComplete"] is False
    assert json.loads((tmp_path / "parts/part-0000.json").read_text(encoding="utf-8")) == {
        "frames": 24,
        "width": 1080,
        "height": 1920,
        "rate": 30,
        "expectedText": "Title",
        "placements": [{"text": "あ\nい", "startFrame": 0, "endFrame": 24, "x": 0, "y": 460}],
    }
    assert (result.stat().st_mode & 0o777) == 0o600
    assert ((tmp_path / "parts").stat().st_mode & 0o777) == 0o700


def test_partition_preserves_all_33_stages(candidate: Path, tmp_path: Path) -> None:
    raw = record(json.loads(candidate.read_text(encoding="utf-8")))
    raw["candidates"] = [
        {"cue": {"text": "字幕", "start": i / 30, "end": (i + 1) / 30}, "x": 540, "y": 500}
        for i in range(33)
    ]
    candidate.write_text(json.dumps(raw), encoding="utf-8")
    result = motion_measured.export_plans(
        candidate,
        tmp_path / "parts",
        sha256=file_digest(candidate),
        width=1080,
        height=1920,
        rate=30,
        frames=33,
        expected_text="Title",
    )
    manifest = json.loads(result.read_text(encoding="utf-8"))
    assert [(p["offsetFrame"], p["frames"], p["stages"]) for p in manifest["parts"]] == [
        (0, 32, 32),
        (32, 1, 1),
    ]
    assert json.loads((tmp_path / "parts/part-0001.json").read_text(encoding="utf-8"))[
        "placements"
    ] == [{"text": "字幕", "startFrame": 0, "endFrame": 1, "x": 0, "y": 460}]


@pytest.mark.parametrize("frames", [0, True, 3601])
def test_refuse_invalid_bounds(candidate: Path, tmp_path: Path, frames: int) -> None:
    with pytest.raises(ValueError, match="bounded"):
        motion_measured.export_plans(
            candidate,
            tmp_path / "parts",
            sha256=file_digest(candidate),
            width=1080,
            height=1920,
            rate=30,
            frames=frames,
            expected_text="Title",
        )
    assert not (tmp_path / "parts").exists()


@pytest.mark.parametrize("times", [(0.101, 0.102), (0.1, 0.9)])
def test_no_hidden_or_overlapping_caption_loss(
    candidate: Path,
    tmp_path: Path,
    times: tuple[float, float],
) -> None:
    raw = record(json.loads(candidate.read_text(encoding="utf-8")))
    raw["candidates"] = [
        {"cue": {"text": "あ", "start": times[0], "end": times[1]}, "x": 540, "y": 500},
        {"cue": {"text": "い", "start": 0.2, "end": 0.8}, "x": 540, "y": 500},
    ]
    candidate.write_text(json.dumps(raw), encoding="utf-8")
    with pytest.raises(ValueError, match="display frame|overlap"):
        motion_measured.export_plans(
            candidate,
            tmp_path / "parts",
            sha256=file_digest(candidate),
            width=1080,
            height=1920,
            rate=30,
            frames=30,
            expected_text="Title",
        )
    assert not (tmp_path / "parts").exists()


def test_changed_input_and_existing_output_refused(candidate: Path, tmp_path: Path) -> None:
    (tmp_path / "calibration.json").write_text("changed", encoding="utf-8")
    with pytest.raises(ValueError, match="fingerprint"):
        motion_measured.export_plans(
            candidate,
            tmp_path / "parts",
            sha256=file_digest(candidate),
            width=1080,
            height=1920,
            rate=30,
            frames=30,
            expected_text="Title",
        )
    with pytest.raises(FileExistsError):
        motion_measured.export_plans(
            candidate,
            tmp_path,
            sha256=file_digest(candidate),
            width=1080,
            height=1920,
            rate=30,
            frames=30,
            expected_text="Title",
        )


@pytest.mark.parametrize("bindings", [None, {}, {"relative": "wrong"}])
def test_invalid_provenance(bindings: object) -> None:
    with pytest.raises(ValueError):
        motion_measured._verify_bindings(
            {"inputs": bindings, "renderCalibration": {"path": "relative", "sha256": "wrong"}}
        )


def test_missing_common_calibration() -> None:
    with pytest.raises(ValueError, match="common calibration"):
        motion_measured._verify_bindings(
            {"inputs": {"x": "a"}, "renderCalibration": {"path": "x", "sha256": "b"}}
        )


def test_invalid_binding_type(tmp_path: Path) -> None:
    path = tmp_path / "calibration"
    path.write_text("x", encoding="utf-8")
    with pytest.raises(ValueError, match="input binding"):
        motion_measured._verify_bindings(
            {
                "inputs": {str(path): file_digest(path), "bad": 1},
                "renderCalibration": {"path": str(path), "sha256": file_digest(path)},
            }
        )


def test_cli(candidate: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "motion_measured",
            str(candidate),
            str(tmp_path / "parts"),
            "--sha256",
            file_digest(candidate),
            "--width",
            "1080",
            "--height",
            "1920",
            "--rate",
            "30",
            "--frames",
            "30",
            "--expected-text",
            "Title",
        ],
    )
    motion_measured.main()
    assert (tmp_path / "parts/manifest.json").is_file()
