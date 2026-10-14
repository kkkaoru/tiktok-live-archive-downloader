"""Decode original audio samples continuously without timestamp-driven repair."""

from pathlib import Path

from replay.core import ReplayError


def continuous_decode_command(source: Path, output: Path) -> list[str]:
    """Prepare float WAV without resampling, silence insertion, drops or fades.

    WAV uses a continuous sample clock; original presentation timing is not
    retained or certified. Callers separately bind the source and measure sample
    conservation. No model inference, cuts or synchronization is performed here.
    """
    if source.is_symlink() or not source.is_file():
        raise ReplayError("Continuous decoding requires a regular source file")
    if output.exists() or output.is_symlink() or not output.parent.is_dir():
        raise ReplayError("Continuous decoding requires a new output in an existing directory")
    return [
        "ffmpeg",
        "-hide_banner",
        "-nostdin",
        "-v",
        "error",
        "-n",
        "-i",
        str(source.resolve()),
        "-map",
        "0:a:0",
        "-vn",
        "-af",
        "asetpts=N/SR/TB",
        "-c:a",
        "pcm_f32le",
        str(output.absolute()),
    ]
