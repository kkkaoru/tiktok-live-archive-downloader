"""A late-starting video gets lead-in frames; every source frame keeps its PTS, audio is copied."""

import json
import subprocess
from pathlib import Path

import pytest

from replay import source_lead_in as lead_in
from replay.core import ReplayError


def movie(tmp_path: Path, *, offset: str) -> Path:
    """A 2 s 160x120 H.264 picture starting at ``offset`` beside a 2.5 s tone starting at zero."""
    video, output = tmp_path / "v.mp4", tmp_path / f"late-{offset}.mp4"
    subprocess.run(
        ["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i", "testsrc=size=160x120:rate=30:d=2",
         "-c:v", "libx264", "-pix_fmt", "yuv420p", str(video)],
        check=True,
    )  # fmt: skip
    subprocess.run(
        ["ffmpeg", "-v", "error", "-y", "-itsoffset", offset, "-i", str(video), "-f", "lavfi",
         "-i", "sine=duration=2.5", "-map", "0:v", "-map", "1:a", "-c:v", "copy", "-c:a", "aac",
         str(output)],
        check=True,
    )  # fmt: skip
    return output


def audio_packets(path: Path) -> str:
    return subprocess.run(
        ["ffmpeg", "-v", "error", "-i", str(path), "-map", "0:a", "-c", "copy", "-f", "framemd5",
         "-"],
        capture_output=True, text=True, check=True,
    ).stdout.split("\n", 4)[-1]  # fmt: skip


def test_a_zero_origin_source_is_used_as_is(tmp_path: Path) -> None:
    source = movie(tmp_path, offset="0")
    store = tmp_path / "store"
    assert lead_in.prepared(source, store, ffmpeg="ffmpeg", ffprobe="ffprobe") == source
    assert store.exists() is False


def test_a_late_video_gets_lead_in_frames_and_keeps_every_source_pts(tmp_path: Path) -> None:
    source = movie(tmp_path, offset="0.5")
    output = lead_in.prepared(source, tmp_path / "store", ffmpeg="ffmpeg", ffprobe="ffprobe")
    assert output == tmp_path / "store/late-0.5.mp4"
    pts = lead_in.video_pts(output, "ffprobe")
    assert pts[:16] == [0, 512, 1024, 1536, 2048, 2560, 3072, 3584, 4096, 4608, 5120, 5632, 6144,
                        6656, 7168, 7680]  # fmt: skip
    assert pts[15:] == lead_in.video_pts(source, "ffprobe")
    assert audio_packets(output) == audio_packets(source)
    receipt = json.loads((output.parent / "lead-in.json").read_text(encoding="utf-8"))
    assert receipt["videoStartPTS"] == 7680
    assert receipt["timeBase"] == "1/15360"
    assert receipt["leadInFrames"] == 15
    assert receipt["outputSHA256"] == lead_in.sha256(output)
    assert sorted(path.name for path in output.parent.iterdir()) == ["late-0.5.mp4", "lead-in.json"]
    again = lead_in.prepared(source, tmp_path / "store", ffmpeg="/missing", ffprobe="ffprobe")
    assert again == output


def test_a_changed_lead_in_copy_is_refused(tmp_path: Path) -> None:
    source = movie(tmp_path, offset="0.5")
    output = lead_in.prepared(source, tmp_path / "store", ffmpeg="ffmpeg", ffprobe="ffprobe")
    output.write_bytes(b"changed")
    with pytest.raises(ReplayError, match="no longer matches its record"):
        lead_in.prepared(source, tmp_path / "store", ffmpeg="ffmpeg", ffprobe="ffprobe")


def test_a_copy_that_moved_a_source_frame_is_not_used(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = movie(tmp_path, offset="0.5")
    measured = iter(([7680, 8192], [0, 7680, 8200]))
    monkeypatch.setattr(lead_in, "video_pts", lambda _movie, _ffprobe: next(measured))
    with pytest.raises(ReplayError, match="moved a source frame"):
        lead_in.prepared(source, tmp_path / "store", ffmpeg="ffmpeg", ffprobe="ffprobe")
    assert (tmp_path / "store/late-0.5.mp4").exists() is False


def test_a_video_starting_before_zero_is_refused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(lead_in, "video_stream", lambda _s, _p: {"start_pts": "-3"})
    with pytest.raises(ReplayError, match="starts before zero"):
        lead_in.prepared(tmp_path / "s.mp4", tmp_path, ffmpeg="ffmpeg", ffprobe="ffprobe")


def test_a_codec_without_a_lead_in_encoder_is_refused(tmp_path: Path) -> None:
    stream: dict[str, object] = {"codec_name": "vp9"}
    with pytest.raises(ReplayError, match="No lead-in encoder for vp9 video"):
        lead_in.lead_in_command(tmp_path / "s", tmp_path / "p", tmp_path / "o", stream, ffmpeg="f")


def test_a_source_without_exactly_one_video_stream_is_refused(tmp_path: Path) -> None:
    tone = tmp_path / "tone.m4a"
    subprocess.run(
        ["ffmpeg", "-v", "error", "-f", "lavfi", "-i", "sine=duration=1", str(tone)], check=True
    )
    with pytest.raises(ReplayError, match="exactly one video stream"):
        lead_in.video_stream(tone, "ffprobe")


def test_the_copy_is_never_written_beside_its_source(tmp_path: Path) -> None:
    source = movie(tmp_path, offset="0.5")
    with pytest.raises(ReplayError, match="apart from its source"):
        lead_in.prepared(source, tmp_path, ffmpeg="ffmpeg", ffprobe="ffprobe")
