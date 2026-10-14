"""Streamed glyph calibration is independent of private footage and app UI."""

import io
import json
from pathlib import Path

import numpy as np
import pytest

from replay import caption_ass_glyphs as glyphs
from replay.caption_phrase_ass import Appearance
from replay.core import ReplayError
from replay.stages import file_digest


def test_actual_green_ink_bounds() -> None:
    image = np.zeros((4, 4, 3), dtype=np.uint8)
    image[1:3, 1:3, 1] = 255
    assert glyphs.ink_bounds(image) == (-539, -537, 2)


def test_absent_ink_is_unresolved() -> None:
    with pytest.raises(ReplayError, match="no resolved"):
        glyphs.ink_bounds(np.zeros((4, 4, 3), dtype=np.uint8))


def test_streamed_renderer_records_real_frame_geometry(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(glyphs, "WIDTH", 4)
    monkeypatch.setattr(glyphs, "HEIGHT", 4)
    image = np.zeros((4, 4, 3), dtype=np.uint8)
    image[1:3, 1:3, 1] = 255

    class Process:
        stdout = io.BytesIO(image.tobytes())

        def __enter__(self) -> "Process":
            return self

        def __exit__(self, *args: object) -> None:
            pass

        def wait(self, timeout: int) -> int:
            assert timeout == 60
            return 0

    monkeypatch.setattr(glyphs.subprocess, "Popen", lambda *args, **kwargs: Process())
    result = glyphs.rasterize(
        ("字幕",),
        appearance=Appearance(52, 960, 1032, 0, 1920),
        ffmpeg=tmp_path / "ffmpeg",
        fonts=tmp_path,
        output=tmp_path / "atlas",
    )
    assert result == {"字幕": {"inkTopOffsetPx": -539, "inkBottomOffsetPx": -537, "inkWidthPx": 2}}


def test_empty_atlas_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(ReplayError):
        glyphs.rasterize(
            (),
            appearance=Appearance(52, 960, 1032, 0, 1920),
            ffmpeg=tmp_path / "ffmpeg",
            fonts=tmp_path,
            output=tmp_path / "atlas",
        )


def test_cli_uses_hash_bound_render_instance(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    plan, font, receipt, out = (
        tmp_path / "plan.json",
        tmp_path / "font.ttf",
        tmp_path / "receipt.json",
        tmp_path / "out",
    )
    plan.write_text(json.dumps({"captions": [{"text": "字幕"}]}))
    font.write_bytes(b"synthetic font")
    receipt.write_text(
        json.dumps(
            {
                "family": "Noto Sans JP",
                "renderInstance": str(font),
                "renderInstanceSHA256": file_digest(font),
            }
        )
    )

    def render(*args: object, **kwargs: object) -> dict[str, object]:
        out.mkdir()
        return {"字幕": {"inkTopOffsetPx": -80, "inkBottomOffsetPx": -14, "inkWidthPx": 200}}

    monkeypatch.setattr(glyphs, "rasterize", render)
    assert (
        glyphs.main([str(plan), str(receipt), str(out), "--ffmpeg", "ffmpeg", "--font-size", "104"])
        == 0
    )
    assert json.loads((out / "bounds.json").read_text())["lineCount"] == 1


@pytest.mark.parametrize("rows", [[], [{"text": ""}]])
def test_invalid_plan_is_rejected(tmp_path: Path, rows: object) -> None:
    plan = tmp_path / "plan.json"
    plan.write_text(json.dumps({"captions": rows}))
    with pytest.raises(ReplayError):
        glyphs.main(
            [
                str(plan),
                "font.json",
                str(tmp_path / "out"),
                "--ffmpeg",
                "ffmpeg",
                "--font-size",
                "104",
            ]
        )
