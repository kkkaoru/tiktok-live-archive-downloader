import json
import subprocess
from pathlib import Path

import pytest

from replay.core import ReplayError
from replay.part_render import Clip, Part, command, filtergraph, load_part, main

CLIPS: list[object] = [
    {
        "sourcePath": "/s.mp4",
        "selection": {"startSeconds": 224.8, "durationSeconds": 10.6, "rate": 1},
    },
    {"sourcePath": "/s.mp4", "selection": {"startSeconds": 0, "durationSeconds": 1.4, "rate": 1}},
]
MASKS: list[object] = [
    {
        "startSeconds": 0.0,
        "endSeconds": 1.2,
        "region": {"x": -10.0, "y": 1288.0, "width": 1100.0, "height": 152.0},
        "opacity": 1,
        "blurRadius": 64,
    },
    {
        "startSeconds": 2.0,
        "endSeconds": 3.0,
        "region": {"x": 2000.0, "y": 10.0, "width": 50.0, "height": 50.0},
        "opacity": 1,
        "blurRadius": 64,
    },
]


def entry(*, clips: list[object], masks: list[object], resize: str = "fit") -> dict[str, object]:
    return {
        "durationFrames": 318,
        "request": {
            "recipe": {
                "clips": clips,
                "video": {
                    "width": 1080,
                    "height": 1920,
                    "frameRate": 30,
                    "resizeMode": resize,
                    "masks": masks,
                },
            }
        },
    }


PART = Part(
    clips=(Clip(Path("/s.mp4"), 224.8, 10.6), Clip(Path("/s.mp4"), 0.0, 1.4)),
    width=1080,
    height=1920,
    rate=30,
    frames=318,
)


def test_plan_entry_becomes_clips_masks_and_canvas() -> None:
    assert load_part(entry(clips=CLIPS, masks=[])) == PART
    with pytest.raises(ReplayError, match="detected local repair"):
        load_part(entry(clips=CLIPS, masks=MASKS))


def test_invalid_entries_are_refused() -> None:
    with pytest.raises(ReplayError, match="fit-to-canvas"):
        load_part(entry(clips=CLIPS, masks=MASKS, resize="fill"))
    with pytest.raises(ReplayError, match="no source clips"):
        load_part(entry(clips=[], masks=MASKS))
    with pytest.raises(ReplayError, match="Invalid part clip span"):
        load_part(
            entry(
                clips=[
                    {
                        "sourcePath": "/s.mp4",
                        "selection": {"startSeconds": 1, "durationSeconds": 0, "rate": 1},
                    }
                ],
                masks=[],
            )
        )
    with pytest.raises(ReplayError, match="Invalid part clip"):
        load_part(
            entry(
                clips=[{"selection": {"startSeconds": 1, "durationSeconds": 1, "rate": 1}}],
                masks=[],
            )
        )
    with pytest.raises(ReplayError, match="detected local repair"):
        load_part(
            entry(
                clips=CLIPS,
                masks=[
                    {
                        "startSeconds": 0,
                        "endSeconds": 1,
                        "region": {"x": 0, "y": 0, "width": 9, "height": 9},
                        "opacity": 0.5,
                        "blurRadius": 64,
                    }
                ],
            )
        )
    with pytest.raises(ReplayError, match="detected local repair"):
        load_part(
            entry(
                clips=CLIPS,
                masks=[
                    {
                        "startSeconds": 0,
                        "endSeconds": 1,
                        "region": {"x": True, "y": 0, "width": 9, "height": 9},
                        "opacity": 1,
                        "blurRadius": 64,
                    }
                ],
            )
        )


def test_filtergraph_joins_resamples_and_burns_once_without_blur() -> None:
    assert filtergraph(PART, Path("/r/captions/a:b.ass")) == (
        "[0:v:0]setpts=PTS-STARTPTS,fps=30,tpad=stop_mode=clone:stop_duration=10.6,"
        "trim=duration=10.6,setpts=PTS-STARTPTS,"
        "scale=1080:1920:force_original_aspect_ratio=decrease,"
        "pad=1080:1920:(ow-iw)/2:(oh-ih)/2,setsar=1[c0];"
        "[1:v:0]setpts=PTS-STARTPTS,fps=30,tpad=stop_mode=clone:stop_duration=1.4,"
        "trim=duration=1.4,setpts=PTS-STARTPTS,"
        "scale=1080:1920:force_original_aspect_ratio=decrease,"
        "pad=1080:1920:(ow-iw)/2:(oh-ih)/2,setsar=1[c1];"
        "[c0][c1]concat=n=2:v=1:a=0[v0];"
        "[v0]format=yuv420p,ass=filename='/r/captions/a\\:b.ass'[out]"
    )


def test_command_seeks_each_clip_and_cuts_to_the_planned_frames() -> None:
    arguments = command(PART, None, Path("/o/part.mp4"), "ffmpeg")
    assert arguments[:18] == [
        "ffmpeg",
        "-v",
        "error",
        "-xerror",
        "-nostdin",
        "-n",
        "-ss",
        "224.8",
        "-t",
        "10.6",
        "-i",
        "/s.mp4",
        "-ss",
        "0",
        "-t",
        "1.4",
        "-i",
        "/s.mp4",
    ]
    assert arguments[arguments.index("-frames:v") + 1] == "318"
    assert arguments[arguments.index("-c:v") + 1] == "h264_videotoolbox"
    assert arguments[-1] == "/o/part.mp4"
    assert arguments[arguments.index("-filter_complex") + 1].endswith("[v0]format=yuv420p[out]")


def test_cli_renders_one_planned_part(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    plan = tmp_path / "render-plan.json"
    plan.write_text(json.dumps([entry(clips=CLIPS, masks=[])]), encoding="utf-8")
    calls: list[list[str]] = []

    def run(arguments: list[str], *, check: bool) -> subprocess.CompletedProcess[str]:
        calls.append(arguments)
        return subprocess.CompletedProcess(arguments, 0 if check else 1)

    monkeypatch.setattr(subprocess, "run", run)
    output = tmp_path / "part.mp4"
    assert main([str(plan), "0", str(output), "--ffmpeg", "/bin/ffmpeg"]) == 0
    assert calls[0][0] == "/bin/ffmpeg"
    assert calls[0][-1] == str(output)
    with pytest.raises(ReplayError, match="outside the render plan"):
        main([str(plan), "1", str(output)])
    output.write_bytes(b"x")
    with pytest.raises(ReplayError, match="overwrite"):
        main([str(plan), "0", str(output)])
