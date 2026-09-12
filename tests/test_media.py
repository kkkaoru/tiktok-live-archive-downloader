import json
import subprocess
from pathlib import Path
from unittest.mock import Mock

import pytest

from replay.core import ReplayError
from replay.media import quicktime_video_tags


def test_hevc_tag(monkeypatch: pytest.MonkeyPatch) -> None:
    run = Mock(
        return_value=subprocess.CompletedProcess(
            [], 0, stdout='{"streams":[{"codec_name":"h264"},{"codec_name":"hevc"}]}'
        )
    )
    monkeypatch.setattr("replay.media.subprocess.run", run)
    assert quicktime_video_tags(Path("local.m3u8")) == ["-tag:v:1", "hvc1"]
    assert run.call_args.kwargs["timeout"] == 30


def test_h264_keeps_default_tag(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "replay.media.subprocess.run",
        Mock(
            return_value=subprocess.CompletedProcess(
                [], 0, stdout='{"streams":[{"codec_name":"h264"}]}'
            )
        ),
    )
    assert quicktime_video_tags(Path("local.m3u8")) == []


@pytest.mark.parametrize("data", [[], {}, {"streams": [None]}, {"streams": [{}]}])
def test_bad_stream_data(monkeypatch: pytest.MonkeyPatch, data: object) -> None:
    monkeypatch.setattr(
        "replay.media.subprocess.run",
        Mock(return_value=subprocess.CompletedProcess([], 0, stdout=json.dumps(data))),
    )
    with pytest.raises(ReplayError, match="Invalid"):
        quicktime_video_tags(Path("local.m3u8"))


def test_probe_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "replay.media.subprocess.run", Mock(side_effect=subprocess.TimeoutExpired("ffprobe", 30))
    )
    with pytest.raises(ReplayError, match="ffprobe failed"):
        quicktime_video_tags(Path("local.m3u8"))
