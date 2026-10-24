"""Rendered caption gradient audit on synthetic RGBA frames and a tiny FFmpeg movie."""

import json
import subprocess
from pathlib import Path

import numpy as np
import pytest

from replay import caption_gradient_audit as audit_module
from replay.caption_gradient_audit import audit, frames, full_lines, line_lifts, line_measures, main
from replay.core import ReplayError


def caption(top_colour: tuple[int, int, int], base: tuple[int, int, int]) -> np.ndarray:
    """Two 20-row text lines on transparency: the upper 4 rows top_colour, the rest base."""
    frame = np.zeros((60, 40, 4), dtype=np.uint8)
    for first in (5, 35):
        frame[first : first + 20, 5:35] = (*base, 255)
        frame[first : first + 4, 5:35] = (*top_colour, 255)
    return frame


def test_a_brighter_top_is_measured_on_every_line() -> None:
    lifts = line_lifts(caption((151, 216, 247), (47, 176, 238)))
    assert [round(lift, 1) for lift in lifts] == [51.4, 51.4]


def test_white_pixels_are_not_fill_so_the_line_starts_below_them() -> None:
    frame = caption((47, 176, 238), (47, 176, 238))
    frame[5:9, 5:35] = (255, 255, 255, 255)
    assert [round(lift, 1) for lift in line_lifts(frame)] == [0.0, 0.0]


def test_an_empty_frame_has_no_lines() -> None:
    assert line_lifts(np.zeros((10, 10, 4), dtype=np.uint8)) == []


def test_the_audit_passes_only_lifted_lines_and_needs_enough_of_them() -> None:
    assert audit([30.0, 20.5, 40.0]) == {"lines": 3, "weakestLift": 20.5, "minimumLift": 12.0}
    with pytest.raises(ReplayError, match="gradient is missing"):
        audit([30.0, -6.3, 40.0])
    with pytest.raises(ReplayError, match="Only 2 caption lines"):
        audit([30.0, 40.0])


def movie(path: Path, frame: np.ndarray) -> None:
    height, width = frame.shape[:2]
    subprocess.run(
        ["ffmpeg", "-v", "error", "-f", "rawvideo", "-pix_fmt", "rgba", "-s", f"{width}x{height}",
         "-r", "30", "-i", "-", "-c:v", "qtrle", "-pix_fmt", "argb", str(path)],
        input=frame.tobytes() * 2, check=True,
    )  # fmt: skip


def test_frames_are_read_from_a_movie_with_alpha(tmp_path: Path) -> None:
    path = tmp_path / "alpha.mov"
    movie(path, caption((151, 216, 247), (47, 176, 238)))
    decoded = list(frames(str(path), step=1))
    assert len(decoded) == 2
    assert decoded[0][10, 10].tolist() == [47, 176, 238, 255]


def test_an_undecodable_movie_is_refused(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    path = tmp_path / "alpha.mov"
    movie(path, caption((151, 216, 247), (47, 176, 238)))
    monkeypatch.setattr(audit_module, "SAMPLE_STEP", 1)
    original = subprocess.Popen

    def failing(
        command: list[str], *, stdout: int, stderr: int | None = None, text: bool = False
    ) -> object:
        failed = ["sh", "-c", "exit 3"] if command[0] == "ffmpeg" else command
        return original(failed, stdout=stdout, stderr=stderr, text=text)

    monkeypatch.setattr(subprocess, "Popen", failing)
    with pytest.raises(ReplayError, match="could not decode"):
        list(frames(str(path)))


def test_the_cli_prints_the_audit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    path = tmp_path / "alpha.mov"
    movie(path, caption((151, 216, 247), (47, 176, 238)))
    monkeypatch.setattr(audit_module, "SAMPLE_STEP", 1)
    monkeypatch.setattr(audit_module, "frames", lambda movie: frames(movie, step=1))
    assert main([str(path)]) == 0
    assert json.loads(capsys.readouterr().out) == {
        "lines": 4,
        "weakestLift": 51.4,
        "minimumLift": 12.0,
    }


def test_a_detached_stroke_above_a_glyph_belongs_to_its_line() -> None:
    frame = np.zeros((60, 40, 4), dtype=np.uint8)
    frame[5:9, 5:35] = (240, 200, 120, 255)  # the dash above え, then fill-free rows
    frame[15:40, 5:35] = (150, 80, 40, 255)
    frame[15:20, 5:35] = (240, 200, 120, 255)
    frame[50:54, 5:35] = (150, 80, 40, 255)  # a trailing fragment joins the last line
    lifts = line_lifts(frame)
    assert len(lifts) == 1 and lifts[0] > 12


def test_a_line_of_glyphs_below_the_text_top_is_not_measured() -> None:
    full = caption((151, 216, 247), (47, 176, 238))
    partial = np.zeros((60, 40, 4), dtype=np.uint8)
    partial[40:50, 5:20] = (47, 176, 238, 255)  # a lone っ, revealed before the rest of its line
    measures = line_measures(full) + line_measures(partial)
    assert [height for height, _ in measures] == [20, 20, 10]
    assert [round(lift, 1) for lift in full_lines(measures)] == [51.4, 51.4]
    assert full_lines([]) == []
