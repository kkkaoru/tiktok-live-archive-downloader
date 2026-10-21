"""VP9/Opus sources get a natively decodable copy; frame PTS and audio samples are kept."""

import json
import subprocess
from pathlib import Path

import numpy as np
import pytest

from replay import source_decodable as decodable
from replay.core import ReplayError
from replay.source_lead_in import sha256, video_pts


def movie(tmp_path: Path, video: str, audio: str) -> Path:
    """A 1 s 64x96 picture with a 1 s tone in the given codecs."""
    output = tmp_path / "2026-10-21_07-30-17_JST_youtube-Eg9-gl3Z954.mp4"
    subprocess.run(
        ["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i", "testsrc=size=64x96:rate=30:d=1",
         "-f", "lavfi", "-i", "sine=duration=1:sample_rate=48000", "-c:v", video,
         "-pix_fmt", "yuv420p", "-c:a", audio, str(output)],
        check=True,
    )  # fmt: skip
    return output


def pcm(path: Path) -> bytes:
    return subprocess.run(
        ["ffmpeg", "-v", "error", "-i", str(path), "-map", "0:a", "-f", "s16le", "-"],
        capture_output=True, check=True,
    ).stdout  # fmt: skip


def test_a_decodable_source_is_used_as_is(tmp_path: Path) -> None:
    source = movie(tmp_path, "libx264", "aac")
    store = tmp_path / "store"
    assert decodable.prepared(source, store, ffmpeg="ffmpeg", ffprobe="ffprobe") == source
    assert store.exists() is False


def test_vp9_and_opus_are_reencoded_keeping_every_pts_and_the_audio(tmp_path: Path) -> None:
    source = movie(tmp_path, "libvpx-vp9", "libopus")
    store = tmp_path / "store"
    output = decodable.prepared(source, store, ffmpeg="ffmpeg", ffprobe="ffprobe")
    assert output == store / "2026-10-21_07-30-17_JST_youtube-Eg9-gl3Z954.mp4"
    video, audio = decodable.streams(output, "ffprobe")
    assert (video["codec_name"], audio["codec_name"]) == ("hevc", "alac")
    assert video_pts(output, "ffprobe") == video_pts(source, "ffprobe")
    original = np.frombuffer(pcm(source), dtype="<i2").astype(np.int32)
    copied = np.frombuffer(pcm(output), dtype="<i2").astype(np.int32)
    assert copied.shape == original.shape
    assert int(np.abs(copied - original).max()) <= 1
    receipt = json.loads((store / "decodable.json").read_text(encoding="utf-8"))
    assert receipt["sourceCodecs"] == ["vp9", "opus"]
    assert receipt["outputCodecs"] == ["hevc", "alac"]
    assert receipt["outputSHA256"] == sha256(output)
    again = decodable.prepared(source, store, ffmpeg="/missing", ffprobe="ffprobe")
    assert again == output
    output.write_bytes(b"changed")
    with pytest.raises(ReplayError, match="no longer matches"):
        decodable.prepared(source, store, ffmpeg="ffmpeg", ffprobe="ffprobe")


def test_only_the_undecodable_stream_is_reencoded(tmp_path: Path) -> None:
    source = movie(tmp_path, "libx264", "libopus")
    output = decodable.prepared(source, tmp_path / "store", ffmpeg="ffmpeg", ffprobe="ffprobe")
    video, audio = decodable.streams(output, "ffprobe")
    assert (video["codec_name"], audio["codec_name"]) == ("h264", "alac")


def test_the_source_directory_and_bad_copies_are_refused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = movie(tmp_path, "libvpx-vp9", "libopus")
    with pytest.raises(ReplayError, match="apart from its source"):
        decodable.prepared(source, tmp_path, ffmpeg="ffmpeg", ffprobe="ffprobe")
    real = decodable.streams
    monkeypatch.setattr(
        decodable,
        "streams",
        lambda path, ffprobe: (
            ({"codec_name": "vp9"}, {"codec_name": "opus"})
            if path.name.startswith("partial-")
            else real(path, ffprobe)
        ),
    )
    with pytest.raises(ReplayError, match="not in decodable codecs"):
        decodable.prepared(source, tmp_path / "a", ffmpeg="ffmpeg", ffprobe="ffprobe")
    monkeypatch.setattr(decodable, "streams", real)
    monkeypatch.setattr(
        decodable,
        "video_pts",
        lambda path, ffprobe: [1] if path.name.startswith("partial-") else [0],
    )
    with pytest.raises(ReplayError, match="moved a source frame"):
        decodable.prepared(source, tmp_path / "b", ffmpeg="ffmpeg", ffprobe="ffprobe")


def test_a_source_without_one_video_and_one_audio_stream_is_refused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        subprocess,
        "run",
        lambda *args, **kwargs: subprocess.CompletedProcess(args, 0, '{"streams": []}', ""),
    )
    with pytest.raises(ReplayError, match="one video and one audio stream"):
        decodable.streams(tmp_path / "x.mp4", "ffprobe")
