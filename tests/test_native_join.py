"""Native part join; the join tests require ffmpeg and ffprobe on PATH."""

import json
import subprocess
from fractions import Fraction
from pathlib import Path

import pytest

from replay import native_join
from replay.core import ReplayError
from replay.native_join import PartFile, VideoStream, concat_list, join, probe_video


def make_part(path: Path, frames: int, *, codec: str = "libx264", rate: int = 30) -> Path:
    subprocess.run(
        [
            "ffmpeg",
            "-nostdin",
            "-v",
            "error",
            "-f",
            "lavfi",
            "-i",
            f"testsrc=s=64x64:r={rate}",
            "-frames:v",
            str(frames),
            "-c:v",
            codec,
            "-video_track_timescale",
            "600",
            str(path),
        ],
        check=True,
        capture_output=True,
        timeout=20,
    )
    return path


def make_audio(path: Path) -> Path:
    subprocess.run(
        [
            "ffmpeg",
            "-nostdin",
            "-v",
            "error",
            "-f",
            "lavfi",
            "-i",
            "sine=frequency=440:sample_rate=48000",
            "-t",
            "1",
            "-c:a",
            "aac",
            str(path),
        ],
        check=True,
        capture_output=True,
        timeout=20,
    )
    return path


def test_concat_list_quotes_paths_and_lands_on_frames(tmp_path: Path) -> None:
    parts = [PartFile(tmp_path / "a'1.mp4", 15), PartFile(tmp_path / "b.mp4", 16)]
    assert concat_list(parts, 30, Fraction(1, 600)) == (
        "ffconcat version 1.0\n"
        f"file '{tmp_path.resolve()}/a'\\''1.mp4'\n"
        "duration 0.500000\n"
        f"file '{tmp_path.resolve()}/b.mp4'\n"
        "duration 0.533333\n"
    )


def test_probe_counts_packets_of_a_real_render(tmp_path: Path) -> None:
    assert probe_video(make_part(tmp_path / "a.mp4", 7)) == VideoStream("h264", Fraction(1, 600), 7)


def test_join_stream_copies_parts_and_audio_into_a_new_file(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    parts = tmp_path / "parts"
    parts.mkdir()
    make_part(parts / "part-000000.mp4", 15)
    make_part(parts / "part-000001.mp4", 16)
    plan = tmp_path / "render-plan.json"
    plan.write_text(json.dumps([{"durationFrames": 15}, {"durationFrames": 16}]), encoding="utf-8")
    output = tmp_path / "joined.mp4"
    audio = make_audio(tmp_path / "voice.m4a")
    argv = [str(plan), str(parts), str(output), "--audio", str(audio)]
    assert native_join.main(argv) == 0
    assert capsys.readouterr().out == '{"parts": 2, "frames": 31}\n'
    assert probe_video(output) == VideoStream("h264", Fraction(1, 600), 31)
    streams = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "stream=codec_type", "-of", "csv=p=0"]
        + [str(output)],
        check=True,
        capture_output=True,
        text=True,
        timeout=20,
    ).stdout
    assert streams == "video\naudio\n"
    assert sorted(path.name for path in tmp_path.iterdir()) == [
        "joined.mp4",
        "parts",
        "render-plan.json",
        "voice.m4a",
    ]
    with pytest.raises(ReplayError, match="Refusing to overwrite"):
        native_join.main(argv)


def test_join_without_audio_keeps_only_video(tmp_path: Path) -> None:
    part = make_part(tmp_path / "a.mp4", 5)
    output = tmp_path / "joined.mp4"
    assert join([PartFile(part, 5)], rate=30, output=output, audio=None) == 5
    assert probe_video(output) == VideoStream("h264", Fraction(1, 600), 5)


def test_join_rejects_parts_that_do_not_match_the_plan(tmp_path: Path) -> None:
    first = make_part(tmp_path / "a.mp4", 5)
    other_rate = make_part(tmp_path / "b.mp4", 5, codec="mpeg4")
    output = tmp_path / "joined.mp4"
    with pytest.raises(ReplayError, match="No part renders"):
        join([], rate=30, output=output, audio=None)
    with pytest.raises(ReplayError, match="does not hold its planned frames"):
        join([PartFile(first, 6)], rate=30, output=output, audio=None)
    with pytest.raises(ReplayError, match="share one codec and time base"):
        join([PartFile(first, 5), PartFile(other_rate, 5)], rate=30, output=output, audio=None)
    assert not output.exists()


def test_join_reports_ffmpeg_failures_and_short_outputs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    part = make_part(tmp_path / "a.mp4", 5)
    output = tmp_path / "joined.mp4"
    with pytest.raises(ReplayError, match="could not join"):
        join([PartFile(part, 5)], rate=30, output=output, audio=tmp_path / "missing.m4a")
    probes = iter(
        [VideoStream("h264", Fraction(1, 600), 5), VideoStream("h264", Fraction(1, 600), 4)]
    )
    monkeypatch.setattr(native_join, "probe_video", lambda path: next(probes))
    with pytest.raises(ReplayError, match="does not hold the planned frames"):
        join([PartFile(part, 5)], rate=30, output=output, audio=None)
    assert sorted(path.name for path in tmp_path.iterdir()) == ["a.mp4"]


@pytest.mark.parametrize(
    ("stdout", "message"),
    [
        ("{}", "no single video stream"),
        ('{"streams": [{"codec_name": "h264"}]}', "Invalid local ffprobe"),
        (
            '{"streams": [{"index": 0, "codec_name": "h264", '
            '"time_base": "1/0", "nb_read_packets": "1"}]}',
            "Invalid local ffprobe",
        ),
    ],
)
def test_probe_rejects_malformed_ffprobe_output(
    monkeypatch: pytest.MonkeyPatch, stdout: str, message: str
) -> None:
    def fake_run(command: list[str], **options: object) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(command, 0, stdout, "")

    monkeypatch.setattr(subprocess, "run", fake_run)
    with pytest.raises(ReplayError, match=message):
        probe_video(Path("a.mp4"))


def test_probe_reports_ffprobe_failure(tmp_path: Path) -> None:
    with pytest.raises(ReplayError, match="cannot verify a part render"):
        probe_video(tmp_path / "missing.mp4")


def test_cli_rejects_malformed_plans(tmp_path: Path) -> None:
    plan = tmp_path / "render-plan.json"
    plan.write_text("{}", encoding="utf-8")
    with pytest.raises(ReplayError, match="Render plan is empty"):
        native_join.main([str(plan), str(tmp_path), str(tmp_path / "out.mp4")])
    plan.write_text('[{"durationFrames": "1"}]', encoding="utf-8")
    with pytest.raises(ReplayError, match="Invalid render plan frames"):
        native_join.main([str(plan), str(tmp_path), str(tmp_path / "out.mp4")])


def test_join_refuses_mixed_stream_order_then_lossless_single_part_remux_repairs_it(
    tmp_path: Path,
) -> None:
    video = make_part(tmp_path / "video.mp4", 30, codec="libx265")
    audio = make_audio(tmp_path / "audio.m4a")
    audio_first = tmp_path / "audio-first.mov"
    subprocess.run(
        [
            "ffmpeg",
            "-nostdin",
            "-v",
            "error",
            "-i",
            str(audio),
            "-i",
            str(video),
            "-map",
            "0:a:0",
            "-map",
            "1:v:0",
            "-c",
            "copy",
            "-video_track_timescale",
            "600",
            str(audio_first),
        ],
        check=True,
        capture_output=True,
        timeout=20,
    )
    assert probe_video(audio_first).index == 1
    output = tmp_path / "joined.mov"
    with pytest.raises(ReplayError, match="video stream index"):
        join([PartFile(audio_first, 30), PartFile(video, 30)], rate=30, output=output, audio=None)
    assert not output.exists()
    normalized = tmp_path / "normalized.mov"
    assert join([PartFile(audio_first, 30)], rate=30, output=normalized, audio=None) == 30
    assert probe_video(normalized).index == 0
    assert (
        join([PartFile(normalized, 30), PartFile(video, 30)], rate=30, output=output, audio=None)
        == 60
    )
    decoded = subprocess.run(
        ["ffmpeg", "-nostdin", "-v", "error", "-xerror", "-i", str(output), "-f", "null", "-"],
        check=True,
        capture_output=True,
        timeout=20,
    )
    assert decoded.stderr == b""
    original_hash = subprocess.run(
        [
            "ffmpeg",
            "-v",
            "error",
            "-i",
            str(audio_first),
            "-map",
            "0:v:0",
            "-c",
            "copy",
            "-f",
            "hash",
            "-hash",
            "sha256",
            "-",
        ],
        check=True,
        capture_output=True,
        timeout=20,
    )
    copied_hash = subprocess.run(
        [
            "ffmpeg",
            "-v",
            "error",
            "-i",
            str(normalized),
            "-map",
            "0:v:0",
            "-c",
            "copy",
            "-f",
            "hash",
            "-hash",
            "sha256",
            "-",
        ],
        check=True,
        capture_output=True,
        timeout=20,
    )
    assert original_hash.stdout == copied_hash.stdout
    assert original_hash.stderr == copied_hash.stderr == b""


def test_probe_refuses_invalid_stream_index(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        native_join,
        "_ffprobe",
        lambda path: {
            "streams": [
                {"codec_name": "h264", "time_base": "1/600", "nb_read_packets": "5", "index": -1}
            ]
        },
    )
    with pytest.raises(ReplayError, match="Invalid local ffprobe stream data"):
        probe_video(Path("synthetic.mp4"))


def test_join_rejects_a_non_unit_mux_time_base(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    part = make_part(tmp_path / "a.mp4", 5)
    monkeypatch.setattr(
        native_join, "probe_video", lambda path: VideoStream("h264", Fraction(2, 75), 5)
    )
    with pytest.raises(ReplayError, match="1/N mux time base"):
        join([PartFile(part, 5)], rate=30, output=tmp_path / "joined.mp4", audio=None)
