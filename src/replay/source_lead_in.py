"""A zero-origin copy of a source whose first video frame is presented after time zero.

Source-caption cleanup refuses a video clock that does not start at zero, and rebasing it would
move the picture against the sound. When the measured first video PTS is later than zero, the
video is encoded once more with lead-in frames (the first picture, at the source frame rate)
covering [0, first PTS); every source frame keeps its exact PTS and the audio is copied, so the
sound, including speech before the first picture, stays where it was. The derived file keeps the
source's name (delivery names derive from it) in its own directory, beside a record (source size
and SHA-256, measured start, output SHA-256); a rerun reuses it only when the record still
matches. A zero-origin source is returned as is.
"""

import hashlib
import json
import subprocess
from fractions import Fraction
from pathlib import Path

from replay.core import ReplayError
from replay.recordings import record

DIRECTORY = "source-lead-in"
# Measured on a 4 Mbit/s 1080x1920 HEVC replay: SSIM 0.995 against the source at 10 Mbit/s.
BITRATE = "10M"
ENCODERS = {"hevc": ("hevc_videotoolbox", "hvc1"), "h264": ("h264_videotoolbox", "avc1")}
COLOUR = (
    ("color_range", "-color_range"),
    ("color_space", "-colorspace"),
    ("color_primaries", "-color_primaries"),
    ("color_transfer", "-color_trc"),
)
CHUNK = 1 << 24


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(CHUNK):
            digest.update(chunk)
    return digest.hexdigest()


def video_stream(source: Path, ffprobe: str) -> dict[str, object]:
    """The single video stream's codec, clock and colour fields."""
    result = subprocess.run(
        [ffprobe, "-v", "error", "-select_streams", "v", "-show_entries",
         "stream=codec_name,pix_fmt,time_base,start_pts,r_frame_rate,"
         "color_range,color_space,color_primaries,color_transfer",
         "-of", "json", str(source)],
        capture_output=True, text=True, check=True,
    )  # fmt: skip
    streams = record(json.loads(result.stdout)).get("streams")
    if not isinstance(streams, list) or len(streams) != 1:
        raise ReplayError("The source must hold exactly one video stream")
    return record(streams[0])


def video_pts(movie: Path, ffprobe: str) -> list[int]:
    """Every video packet's PTS in presentation order."""
    result = subprocess.run(
        [ffprobe, "-v", "error", "-select_streams", "v", "-show_entries", "packet=pts",
         "-of", "csv=p=0", str(movie)],
        capture_output=True, text=True, check=True,
    )  # fmt: skip
    return sorted(int(line) for line in result.stdout.split())


def lead_in_command(
    source: Path, picture: Path, output: Path, stream: dict[str, object], *, ffmpeg: str
) -> list[str]:
    """Interleave lead-in frames before the source frames; source PTS and audio are kept."""
    codec = str(stream["codec_name"])
    if codec not in ENCODERS:
        raise ReplayError(f"No lead-in encoder for {codec} video")
    encoder, tag = ENCODERS[codec]
    base = Fraction(str(stream["time_base"]))
    step = round(1 / (Fraction(str(stream["r_frame_rate"])) * base))
    start = int(str(stream["start_pts"]))
    frames = -(-start // step)
    graph = (
        f"[1:v]format={stream['pix_fmt']},settb={base},setpts=N*{step},"
        f"trim=end_frame={frames}[lead];[0:v]settb={base}[source];[lead][source]interleave[v]"
    )
    colour = [
        argument
        for key, option in COLOUR
        if isinstance(stream.get(key), str) and stream[key] != "unknown"
        for argument in (option, str(stream[key]))
    ]
    return [
        ffmpeg, "-v", "error", "-nostdin", "-i", str(source), "-loop", "1", "-i", str(picture),
        "-filter_complex", graph, "-map", "[v]", "-map", "0:a", "-c:a", "copy",
        "-fps_mode", "passthrough", "-enc_time_base", str(base), "-c:v", encoder,
        "-b:v", BITRATE, "-tag:v", tag, *colour, "-movflags", "+faststart", str(output),
    ]  # fmt: skip


def prepared(source: Path, directory: Path, *, ffmpeg: str, ffprobe: str) -> Path:
    """The source itself when its video starts at zero, else its verified lead-in copy."""
    stream = video_stream(source, ffprobe)
    start = int(str(stream["start_pts"]))
    if start == 0:
        return source
    if start < 0:
        raise ReplayError("The source video starts before zero; no lead-in can be added")
    if directory.resolve() == source.resolve().parent:
        raise ReplayError("The lead-in copy needs a directory apart from its source")
    output, receipt = directory / source.name, directory / "lead-in.json"
    expected = {
        "source": str(source),
        "sourceSize": source.stat().st_size,
        "sourceSHA256": sha256(source),
        "videoStartPTS": start,
        "timeBase": str(stream["time_base"]),
        "output": str(output),
    }
    if receipt.exists():
        recorded = record(json.loads(receipt.read_text(encoding="utf-8")))
        kept = {key: recorded.get(key) for key in expected}
        if kept != expected or recorded.get("outputSHA256") != sha256(output):
            raise ReplayError("The lead-in copy no longer matches its record; remove it to redo")
        return output
    directory.mkdir(mode=0o700, parents=True, exist_ok=True)
    picture, partial = directory / "first-frame.png", directory / f"partial-{output.name}"
    for stale in (picture, partial):
        stale.unlink(missing_ok=True)
    subprocess.run(
        [ffmpeg, "-v", "error", "-nostdin", "-i", str(source), "-map", "0:v", "-frames:v", "1",
         str(picture)],
        check=True,
    )  # fmt: skip
    subprocess.run(lead_in_command(source, picture, partial, stream, ffmpeg=ffmpeg), check=True)
    picture.unlink()
    original, derived = video_pts(source, ffprobe), video_pts(partial, ffprobe)
    if derived[0] != 0 or [pts for pts in derived if pts >= start] != original:
        raise ReplayError("The lead-in copy moved a source frame; it is not used")
    partial.rename(output)
    receipt.write_text(
        json.dumps({**expected, "leadInFrames": len(derived) - len(original),
                    "outputSHA256": sha256(output)}, indent=2) + "\n",
        encoding="utf-8",
    )  # fmt: skip
    return output
