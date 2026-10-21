"""A copy of a source in codecs the native (AVFoundation) tools decode, at the source's size.

OCR, caption composition and other native stages read the source through AVFoundation, which
refuses VP9/AV1 video (``AVFoundationErrorDomain -11869``) that YouTube serves. Such a source is
encoded once: video to HEVC with every frame's PTS kept, audio losslessly to ALAC (Opus is not
copied either), at the source's own size; scaling to the delivery canvas stays the composition's
last step. The derived file keeps the source's name in its own directory, beside a record
(source size and SHA-256, codecs, output SHA-256); a rerun reuses it only when the record still
matches. A source already in decodable codecs is returned as is.
"""

import json
import subprocess
from fractions import Fraction
from pathlib import Path

from replay.core import ReplayError
from replay.recordings import record
from replay.source_lead_in import BITRATE, COLOUR, sha256, video_pts

DIRECTORY = "source-decodable"
RECORD = "decodable.json"
VIDEO_CODECS = frozenset({"h264", "hevc"})
AUDIO_CODECS = frozenset({"aac", "alac", "pcm_s16le", "pcm_s24le"})


def streams(source: Path, ffprobe: str) -> tuple[dict[str, object], dict[str, object]]:
    """The single video and single audio stream of a source."""
    result = subprocess.run(
        [ffprobe, "-v", "error", "-show_entries",
         "stream=codec_type,codec_name,time_base,color_range,color_space,color_primaries,"
         "color_transfer", "-of", "json", str(source)],
        capture_output=True, text=True, check=True,
    )  # fmt: skip
    found = record(json.loads(result.stdout)).get("streams")
    rows = [record(row) for row in found] if isinstance(found, list) else []
    video = [row for row in rows if row.get("codec_type") == "video"]
    audio = [row for row in rows if row.get("codec_type") == "audio"]
    if len(video) != 1 or len(audio) != 1:
        raise ReplayError("The source must hold exactly one video and one audio stream")
    return video[0], audio[0]


def encode_command(
    source: Path, output: Path, video: dict[str, object], audio: dict[str, object], *, ffmpeg: str
) -> list[str]:
    """Re-encode only the codecs native tools cannot decode; PTS (passthrough) are kept."""
    colour = [
        argument
        for key, option in COLOUR
        if isinstance(video.get(key), str) and video[key] != "unknown"
        for argument in (option, str(video[key]))
    ]
    video_codec = (
        ["-c:v", "copy"]
        if video.get("codec_name") in VIDEO_CODECS
        else ["-c:v", "hevc_videotoolbox", "-b:v", BITRATE, "-tag:v", "hvc1", *colour]
    )
    audio_codec = (
        ["-c:a", "copy"]
        if audio.get("codec_name") in AUDIO_CODECS
        # Opus decodes to floats: 24-bit ALAC keeps them to within one 16-bit step.
        else ["-c:a", "alac", "-sample_fmt", "s32p"]
    )
    return [
        ffmpeg, "-v", "error", "-nostdin", "-i", str(source), "-map", "0:v", "-map", "0:a",
        "-fps_mode", "passthrough", "-enc_time_base", str(Fraction(str(video["time_base"]))),
        *video_codec, *audio_codec, "-movflags", "+faststart", str(output),
    ]  # fmt: skip


def prepared(source: Path, directory: Path, *, ffmpeg: str, ffprobe: str) -> Path:
    """The source itself when native tools decode it, else its verified decodable copy."""
    video, audio = streams(source, ffprobe)
    codecs = [str(video.get("codec_name")), str(audio.get("codec_name"))]
    if codecs[0] in VIDEO_CODECS and codecs[1] in AUDIO_CODECS:
        return source
    if directory.resolve() == source.resolve().parent:
        raise ReplayError("The decodable copy needs a directory apart from its source")
    output, receipt = directory / source.name, directory / RECORD
    expected = {
        "source": str(source),
        "sourceSize": source.stat().st_size,
        "sourceSHA256": sha256(source),
        "sourceCodecs": codecs,
        "output": str(output),
    }
    if receipt.exists():
        recorded = record(json.loads(receipt.read_text(encoding="utf-8")))
        kept = {key: recorded.get(key) for key in expected}
        if kept != expected or recorded.get("outputSHA256") != sha256(output):
            raise ReplayError("The decodable copy no longer matches its record; remove it to redo")
        return output
    directory.mkdir(mode=0o700, parents=True, exist_ok=True)
    partial = directory / f"partial-{output.name}"
    partial.unlink(missing_ok=True)
    subprocess.run(encode_command(source, partial, video, audio, ffmpeg=ffmpeg), check=True)
    derived = [str(stream.get("codec_name")) for stream in streams(partial, ffprobe)]
    if derived[0] not in VIDEO_CODECS or derived[1] not in AUDIO_CODECS:
        raise ReplayError("The decodable copy is not in decodable codecs; it is not used")
    if video_pts(partial, ffprobe) != video_pts(source, ffprobe):
        raise ReplayError("The decodable copy moved a source frame; it is not used")
    partial.rename(output)
    receipt.write_text(
        json.dumps({**expected, "outputCodecs": derived, "outputSHA256": sha256(output)}, indent=2)
        + "\n",
        encoding="utf-8",
    )
    return output
