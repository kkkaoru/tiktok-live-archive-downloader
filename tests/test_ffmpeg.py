"""Local integration smoke test; requires ffmpeg and ffprobe on PATH."""

import json
import subprocess
from pathlib import Path

import httpx
import pytest

from replay.core import Candidate, Scope
from replay.download import Downloader


@pytest.mark.parametrize("segment_type", ["mpegts", "fmp4"])
def test_real_remux(tmp_path: Path, segment_type: str) -> None:
    source = tmp_path / "source"
    source.mkdir()
    subprocess.run(
        [
            "ffmpeg",
            "-nostdin",
            "-hide_banner",
            "-loglevel",
            "error",
            "-f",
            "lavfi",
            "-i",
            "color=c=blue:s=64x64:r=10",
            "-f",
            "lavfi",
            "-i",
            "sine=frequency=440:sample_rate=44100",
            "-t",
            "1",
            "-c:v",
            "libx264",
            "-c:a",
            "aac",
            "-g",
            "5",
            "-hls_time",
            "0.5",
            "-hls_playlist_type",
            "vod",
            "-hls_segment_type",
            segment_type,
            str(source / "index.m3u8"),
        ],
        check=True,
        capture_output=True,
        timeout=20,
    )

    def serve(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200, content=(source / request.url.path.removeprefix("/")).read_bytes()
        )

    with httpx.Client(transport=httpx.MockTransport(serve)) as client:
        Downloader(
            client=client,
            scope=Scope(("cdn.test",)),
            candidate=Candidate("https://cdn.test/index.m3u8", {}, "hls"),
        ).download(tmp_path / "result.mp4")
    probe = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "stream=codec_name:format=duration",
            "-of",
            "json",
            str(tmp_path / "result.mp4"),
        ],
        check=True,
        capture_output=True,
        text=True,
        timeout=10,
    )
    result = json.loads(probe.stdout)
    assert result["streams"] == [{"codec_name": "h264"}, {"codec_name": "aac"}]
    assert 1 <= float(result["format"]["duration"]) < 1.2
