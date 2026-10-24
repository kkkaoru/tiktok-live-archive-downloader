"""With the source's frame clock, compose reads each cut at the phase that keeps every frame."""

import json
from pathlib import Path

import pytest

from replay import caption_compose as compose
from replay.core import ReplayError


def flv_clock(tmp_path: Path) -> Path:
    """300 frames from 17 ms in 33/34 ms steps (29.85 fps) at a millisecond time base."""
    pts = [17]
    for index in range(299):
        pts.append(pts[-1] + (33 if index % 2 else 34))
    frames = [{"pts": value, "duration": 33} for value in pts]
    path = tmp_path / "video-clock.json"
    path.write_text(json.dumps({"frames": frames, "streams": [{"time_base": "1/1000"}]}), "utf-8")
    return path


def plan(tmp_path: Path, frames: int) -> dict[str, object]:
    parts = tmp_path / "parts.json"
    parts.write_text("[]", encoding="utf-8")
    cuts = tmp_path / "cuts.json"
    rows = [{"source_start_frame": 150, "source_end_frame": 250, "output_start_frame": 0}]
    cuts.write_text(json.dumps({"cuts": rows}), encoding="utf-8")
    compose.plan(
        cuts, parts, Path("/src.mov"), tmp_path / "out", tmp_path / "req",
        packets=lambda _path: frames, frame_clock=flv_clock(tmp_path),
    )  # fmt: skip
    body: dict[str, object] = json.loads((tmp_path / "req/g000.json").read_text("utf-8"))
    return body


def test_a_cut_reads_each_run_at_its_phase(tmp_path: Path) -> None:
    body = plan(tmp_path, 300)
    assert body["recipe"] == {
        "video": {
            "width": 1080,
            "height": 1920,
            "frameRate": 30,
            "resizeMode": "fit",
            "encoding": {"codec": "hevc", "averageBitRate": 20_000_000,
                         "allowFrameReordering": False},
        },
        "clips": [
            {
                "sourcePath": "/src.mov",
                "selection": {"startSeconds": 150.511 / 30, "durationSeconds": 51 / 30, "rate": 1},
            },
            {
                "sourcePath": "/src.mov",
                "selection": {"startSeconds": 201.5 / 30, "durationSeconds": 49 / 30, "rate": 1},
            },
        ],
        "additionalVideo": [],
    }  # fmt: skip


def test_a_clock_of_another_source_is_refused(tmp_path: Path) -> None:
    with pytest.raises(ReplayError, match="does not describe the composed source"):
        plan(tmp_path, 299)
