import json
from fractions import Fraction
from pathlib import Path

import pytest

from replay import caption_clamp, caption_measured_job, foreground_geometry
from replay.caption_source_support import OutputSourceSupport
from replay.core import ReplayError
from replay.foreground_observations import ForegroundFrame


def test_clamped_support_never_reports_above_floor(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(OutputSourceSupport, "__post_init__", lambda _self: None)
    monkeypatch.setattr(OutputSourceSupport, "__call__", lambda _self, _start, _end: 120.0)
    support = caption_clamp.clamped(300)((), ())
    assert support(Fraction(0), Fraction(1)) == 300


def test_clamped_support_keeps_lower_tops(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(OutputSourceSupport, "__post_init__", lambda _self: None)
    monkeypatch.setattr(OutputSourceSupport, "__call__", lambda _self, _start, _end: 900.0)
    support = caption_clamp.clamped(300)((), ())
    assert support(Fraction(0), Fraction(1)) == 900


def test_run_records_the_floor(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    original = caption_measured_job.__dict__["OutputSourceSupport"]

    def plan(_manifest: Path, directory: Path, *, manifest_sha256: str) -> Path:
        directory.mkdir()
        assert manifest_sha256 == "a" * 64
        assert caption_measured_job.__dict__["OutputSourceSupport"] is not original
        return directory / "candidate.json"

    monkeypatch.setattr(caption_measured_job, "plan", plan)
    try:
        candidate = caption_clamp.run(
            tmp_path / "manifest.json", tmp_path / "out", sha256="a" * 64, minimum_top=300
        )
    finally:
        caption_measured_job.__dict__["OutputSourceSupport"] = original
    assert candidate == tmp_path / "out" / "candidate.json"
    assert json.loads((tmp_path / "out" / "clamp.json").read_text(encoding="utf-8")) == {
        "minimumSupportTop": 300,
        "unknownSupportTop": None,
        "reason": "subject reaches the frame top; captions held in the upper band",
        "abovePlacementApproved": False,
    }


def test_run_rejects_floor_outside_canvas(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="inside the canvas"):
        caption_clamp.run(tmp_path / "m.json", tmp_path / "out", sha256="a", minimum_top=0)


def test_cli_reports_errors(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    args = [str(tmp_path / "m.json"), str(tmp_path / "out"), "--manifest-sha256", "a"]
    assert caption_clamp.main([*args, "--minimum-top", "0"]) == 1
    assert "CAPTION_CLAMP_ERROR" in capsys.readouterr().err


def test_cli_success(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(caption_clamp, "run", lambda *_args, **_kwargs: tmp_path)
    args = [str(tmp_path / "m.json"), str(tmp_path / "out"), "--manifest-sha256", "a"]
    assert caption_clamp.main([*args, "--minimum-top", "300"]) == 0


def test_unknown_frames_use_stated_default(monkeypatch: pytest.MonkeyPatch) -> None:
    def unknown(_frame: object) -> float:
        raise ReplayError("Required foreground support is unknown")

    monkeypatch.setattr(foreground_geometry, "_support_top", unknown)
    support_top = caption_clamp.unknown_frames(900)
    assert support_top(cast_frame()) == 900


def test_other_frame_errors_still_fail(monkeypatch: pytest.MonkeyPatch) -> None:
    def broken(_frame: object) -> float:
        raise ReplayError("Invalid box")

    monkeypatch.setattr(foreground_geometry, "_support_top", broken)
    support_top = caption_clamp.unknown_frames(900)
    with pytest.raises(ReplayError, match="Invalid box"):
        support_top(cast_frame())


def cast_frame() -> ForegroundFrame:
    return ForegroundFrame.__new__(ForegroundFrame)
