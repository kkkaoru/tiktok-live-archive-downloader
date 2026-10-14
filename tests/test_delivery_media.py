import json
import shutil
import subprocess
from fractions import Fraction
from pathlib import Path

import pytest

from replay import delivery_media as media
from replay.core import ReplayError
from replay.foreground_collection import load_job

FFMPEG = shutil.which("ffmpeg")
FFPROBE = shutil.which("ffprobe")


def _probe(pts: list[int], durations: list[int]) -> dict[str, object]:
    return {
        "streams": [{"time_base": "1/90000"}],
        "frames": [
            {"pts": value, "duration": length} for value, length in zip(pts, durations, strict=True)
        ],
    }


def test_canvas_rejects_odd_dimensions() -> None:
    with pytest.raises(ReplayError, match="even dimensions"):
        media.Canvas(1081, 1920, 30)


def test_canvas_rejects_unbounded_rate() -> None:
    with pytest.raises(ReplayError, match="bounded rate"):
        media.Canvas(1080, 1920, 240)


def test_parse_clock_keeps_exact_ticks() -> None:
    clock = media.parse_clock(_probe([0, 3000, 6030], [3000, 3030, 2970]))
    assert clock == media.FrameClock((0, 3000, 6030), Fraction(1, 90000), 2970)


def test_parse_clock_rejects_non_increasing_ticks() -> None:
    with pytest.raises(ReplayError, match="strictly increase"):
        media.parse_clock(_probe([0, 3000, 3000], [3000, 3000, 3000]))


def test_parse_clock_rejects_missing_duration() -> None:
    data = {"streams": [{"time_base": "1/90000"}], "frames": [{"pts": 0}]}
    with pytest.raises(ReplayError, match="integer pts and duration"):
        media.parse_clock(data)


def test_parse_clock_requires_single_stream() -> None:
    with pytest.raises(ReplayError, match="one video stream"):
        media.parse_clock({"streams": [], "frames": []})


def test_parse_clock_requires_time_base() -> None:
    with pytest.raises(ReplayError, match="time base"):
        media.parse_clock({"streams": [{}], "frames": []})


def test_proxy_command_fits_without_crop(tmp_path: Path) -> None:
    command = media.proxy_command(
        tmp_path / "in.mkv", tmp_path / "out.mov", media.Canvas(1080, 1920, 30), ffmpeg="ff"
    )
    assert command[command.index("-vf") + 1] == (
        "scale=1080:1920:force_original_aspect_ratio=decrease:flags=lanczos,"
        "pad=1080:1920:(ow-iw)/2:(oh-ih)/2:black,fps=30,format=yuv420p"
    )
    assert command[command.index("-c:a") + 1] == "pcm_s16le"


def test_proxy_command_rejects_unreadable_suffix(tmp_path: Path) -> None:
    with pytest.raises(ReplayError, match=r"\.mov"):
        media.proxy_command(
            tmp_path / "in.mkv", tmp_path / "out.mkv", media.Canvas(2, 2, 1), ffmpeg="ff"
        )


def test_foreground_plan_round_trips_through_collection_loader(tmp_path: Path) -> None:
    source = tmp_path / "source.mp4"
    source.write_bytes(b"synthetic")
    native = tmp_path / "native"
    native.write_bytes(b"#!/bin/sh\n")
    native.chmod(0o700)
    pts = [index * 3000 for index in range(130)]
    clock = media.FrameClock(tuple(pts), Fraction(1, 90000), 3000)
    configs = media.foreground_configs(
        clock, source=source, native=native, canvas=media.Canvas(1080, 1920, 30)
    )
    workflows = media.write_foreground_plan(tmp_path / "plan", configs, workers=2)
    first = load_job(tmp_path / "plan" / "configs" / "0000.json")
    second = load_job(tmp_path / "plan" / "configs" / "0001.json")
    assert len(configs) == 2
    assert first.clock.last_duration == 3000
    assert second.clock.pts == (384000, 387000)
    assert [path.name for path in workflows] == ["workflow-0-0000.json", "workflow-1-0000.json"]
    stages = json.loads(workflows[1].read_text(encoding="utf-8"))["stages"]
    assert stages[0]["name"] == "foreground-0001"


def test_foreground_plan_refuses_existing_directory(tmp_path: Path) -> None:
    (tmp_path / "plan" / "configs").mkdir(parents=True)
    with pytest.raises(FileExistsError):
        media.write_foreground_plan(tmp_path / "plan", [{"version": 1}], workers=1)


def test_foreground_plan_rejects_worker_count(tmp_path: Path) -> None:
    with pytest.raises(ReplayError, match="1-8 workers"):
        media.write_foreground_plan(tmp_path / "plan", [{"version": 1}], workers=9)


@pytest.mark.skipif(FFMPEG is None or FFPROBE is None, reason="FFmpeg is required")
def test_cli_proxy_and_clock_with_real_ffmpeg(tmp_path: Path) -> None:
    assert FFMPEG is not None
    assert FFPROBE is not None
    source = tmp_path / "source.mkv"
    subprocess.run(
        (
            FFMPEG,
            "-v",
            "error",
            "-f",
            "lavfi",
            "-i",
            "testsrc=size=72x128:rate=25:duration=1",
            "-f",
            "lavfi",
            "-i",
            "sine=frequency=440:duration=1",
            "-shortest",
            str(source),
        ),
        check=True,
    )
    proxy = tmp_path / "proxy.mov"
    assert media.main(["proxy", str(source), str(proxy), "--width", "108", "--height", "192"]) == 0
    clock = tmp_path / "clock.json"
    assert media.main(["clock", str(proxy), str(clock)]) == 0
    assert len(media.read_clock(clock).pts) == 30


def test_cli_reports_failures(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert media.main(["clock", str(tmp_path / "missing.mp4"), str(tmp_path / "c.json")]) == 1
    assert "DELIVERY_MEDIA_ERROR" in capsys.readouterr().err


def test_cli_foreground_plan(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    source = tmp_path / "source.mp4"
    source.write_bytes(b"synthetic")
    native = tmp_path / "native"
    native.write_bytes(b"#!/bin/sh\n")
    native.chmod(0o700)
    clock = tmp_path / "clock.json"
    clock.write_text(json.dumps(_probe([0, 3000], [3000, 3000])), encoding="utf-8")
    code = media.main(
        ["foreground-plan", str(source), str(clock), str(native), str(tmp_path / "plan")]
    )
    assert code == 0
    assert capsys.readouterr().out.strip().endswith("workflow-0-0000.json")


def render_config(path: Path, output: Path, pts: list[int]) -> Path:
    output.mkdir(parents=True)
    (output / "candidate.mp4").write_bytes(b"movie")
    clock = {"pts": pts, "timeBase": "1/90000", "lastDuration": 3000}
    path.write_text(json.dumps({"clock": clock, "outputDirectory": str(output)}), encoding="utf-8")
    return path


def test_chunk_index_orders_contiguous_chunks(tmp_path: Path) -> None:
    second = render_config(tmp_path / "b.json", tmp_path / "b", [6000, 9000])
    first = render_config(tmp_path / "a.json", tmp_path / "a", [0, 3000])
    index = media.chunk_index([second, first])
    assert index["ranges"] == [
        {
            "media": str(tmp_path / "a" / "candidate.mp4"),
            "sourceFirstPTS": 0,
            "sourceEndPTS": 6000,
            "frames": 2,
            "timeBase": "1/90000",
        },
        {
            "media": str(tmp_path / "b" / "candidate.mp4"),
            "sourceFirstPTS": 6000,
            "sourceEndPTS": 12000,
            "frames": 2,
            "timeBase": "1/90000",
        },
    ]


def test_chunk_index_rejects_gaps(tmp_path: Path) -> None:
    first = render_config(tmp_path / "a.json", tmp_path / "a", [0])
    second = render_config(tmp_path / "b.json", tmp_path / "b", [9000])
    with pytest.raises(ReplayError, match="contiguous"):
        media.chunk_index([first, second])


def test_chunk_index_requires_rendered_media(tmp_path: Path) -> None:
    path = render_config(tmp_path / "a.json", tmp_path / "a", [0])
    (tmp_path / "a" / "candidate.mp4").unlink()
    with pytest.raises(ReplayError, match="missing"):
        media.chunk_index([path])


def test_chunk_index_requires_clock(tmp_path: Path) -> None:
    path = tmp_path / "a.json"
    path.write_text('{"clock": {}}', encoding="utf-8")
    with pytest.raises(ReplayError, match="clock or output"):
        media.chunk_index([path])


def planned(tmp_path: Path) -> Path:
    source = tmp_path / "source.mp4"
    source.write_bytes(b"synthetic")
    native = tmp_path / "native"
    native.write_bytes(b"#!/bin/sh\n")
    native.chmod(0o700)
    clock = media.FrameClock(tuple(index * 3000 for index in range(300)), Fraction(1, 90000), 3000)
    configs = media.foreground_configs(
        clock, source=source, native=native, canvas=media.Canvas(1080, 1920, 30)
    )
    media.write_foreground_plan(tmp_path / "plan", configs, workers=1)
    return tmp_path / "plan"


def test_retry_foreground_keeps_partial_evidence(tmp_path: Path) -> None:
    plan = planned(tmp_path)
    (plan / "evidence-0000" / "batch-0000").mkdir(parents=True)
    (plan / "evidence-0001").mkdir()
    (plan / "evidence-0001" / "provenance.json").write_text("{}", encoding="utf-8")
    [workflow] = media.retry_foreground(plan, first=0, last=2, workers=1)
    stages = json.loads(workflow.read_text(encoding="utf-8"))["stages"]
    assert [stage["name"] for stage in stages] == ["foreground-0000", "foreground-0002"]
    assert stages[0]["command"][7] == str(plan / "retry-0000" / "collection-state")
    assert (plan / "evidence-0000.failed-0" / "batch-0000").is_dir()


def test_retry_foreground_rejects_unknown_window(tmp_path: Path) -> None:
    plan = planned(tmp_path)
    with pytest.raises(ReplayError, match="Unknown foreground window"):
        media.retry_foreground(plan, first=0, last=9, workers=1)


def test_retry_foreground_needs_pending_windows(tmp_path: Path) -> None:
    plan = planned(tmp_path)
    (plan / "evidence-0000").mkdir()
    (plan / "evidence-0000" / "provenance.json").write_text("{}", encoding="utf-8")
    with pytest.raises(ReplayError, match="already has a receipt"):
        media.retry_foreground(plan, first=0, last=0, workers=1)


def test_retry_foreground_validates_range(tmp_path: Path) -> None:
    with pytest.raises(ReplayError, match="window range"):
        media.retry_foreground(tmp_path, first=2, last=1, workers=1)


def test_rebind_resolves_linked_executable(tmp_path: Path) -> None:
    plan = planned(tmp_path)
    link = tmp_path / "linked-native"
    link.symlink_to(tmp_path / "native")
    config = json.loads((plan / "configs" / "0000.json").read_text(encoding="utf-8"))
    linked = tmp_path / "linked.json"
    linked.write_text(json.dumps({**config, "nativeExecutable": str(link)}), encoding="utf-8")
    [workflow] = media.rebind_foreground(tmp_path / "rebound", [linked])
    rebound = json.loads(
        (tmp_path / "rebound" / "configs" / "0000.json").read_text(encoding="utf-8")
    )
    assert rebound["nativeExecutable"] == str((tmp_path / "native").resolve())
    assert workflow.name == "workflow-0-0000.json"


def test_rebind_rejects_changed_executable(tmp_path: Path) -> None:
    plan = planned(tmp_path)
    (tmp_path / "native").write_bytes(b"changed")
    with pytest.raises(ReplayError, match="executable changed"):
        media.rebind_foreground(tmp_path / "rebound", [plan / "configs" / "0000.json"])


def test_cli_chunk_retry_and_rebind(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    plan = planned(tmp_path)
    assert media.main(["foreground-retry", str(plan), "--first", "0", "--last", "0"]) == 0
    assert (
        media.main(["foreground-rebind", str(tmp_path / "re"), str(plan / "configs" / "0001.json")])
        == 0
    )
    config = render_config(tmp_path / "a.json", tmp_path / "a", [0])
    assert media.main(["chunk-index", str(tmp_path / "chunks.json"), str(config)]) == 0
    assert capsys.readouterr().out.strip().endswith("workflow-0-0000.json")
