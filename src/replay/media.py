"""Local media compatibility checks, without remote resource access."""

import json
import subprocess
from pathlib import Path

from replay.core import ReplayError


def quicktime_video_tags(path: Path) -> list[str]:
    try:
        probe = subprocess.run(
            [
                "ffprobe",
                "-v",
                "error",
                "-protocol_whitelist",
                "file,crypto,data",
                "-allowed_extensions",
                "ALL",
                "-select_streams",
                "v",
                "-show_entries",
                "stream=codec_name",
                "-of",
                "json",
                str(path),
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=30,
        )
    except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired) as error:
        raise ReplayError("Local ffprobe failed; cannot determine QuickTime video tags.") from error
    data: object = json.loads(probe.stdout)
    streams: object = data.get("streams") if isinstance(data, dict) else None
    if not isinstance(streams, list):
        raise ReplayError("Invalid local ffprobe stream data.")
    result: list[str] = []
    for index, stream in enumerate(streams):
        if not isinstance(stream, dict) or not isinstance(stream.get("codec_name"), str):
            raise ReplayError("Invalid local ffprobe stream data.")
        if stream["codec_name"] == "hevc":
            result.extend([f"-tag:v:{index}", "hvc1"])
    return result
