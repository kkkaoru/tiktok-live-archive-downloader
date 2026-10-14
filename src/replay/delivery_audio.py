"""Separate the voice stem of any recording's original audio for delivery.

The original audio is decoded on its continuous sample clock, converted once to
the separation model's 16-bit 44.1 kHz input with only clipping-prevention gain
(no loudness normalization, time stretch, padding or fades), and then passed to
the pinned `replay.bgm_separation.separate_vocals` model. The result is the
music-removed voice track that later stages cut on the same clock.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import wave
from collections.abc import Callable, Sequence
from pathlib import Path

from replay.core import ReplayError
from replay.native_invocation import read, require

PEAK_HEADROOM_DB = 0.1
DECODE_TIMEOUT_SECONDS = 7200
MAX_VOLUME = re.compile(r"max_volume:\s*(-?[0-9]+(?:\.[0-9]+)?|-inf) dB")

Runner = Callable[[Sequence[str]], str]


def _run(arguments: Sequence[str]) -> str:
    result = subprocess.run(
        arguments,
        stdin=subprocess.DEVNULL,
        capture_output=True,
        text=True,
        timeout=DECODE_TIMEOUT_SECONDS,
        check=True,
    )
    return result.stderr


def peak_command(source: Path, *, ffmpeg: str) -> list[str]:
    return [
        ffmpeg,
        "-nostdin",
        "-hide_banner",
        "-i",
        str(source),
        "-map",
        "0:a:0",
        "-vn",
        "-af",
        "asetpts=N/SR/TB,aresample=44100,volumedetect",
        "-f",
        "null",
        "-",
    ]


def clip_safe_gain(log: str) -> float:
    """Gain in dB that keeps the decoded peak below full scale; never positive."""
    match = MAX_VOLUME.search(log)
    if match is None:
        raise ReplayError("Peak measurement did not report max_volume")
    if match[1] == "-inf":
        raise ReplayError("Decoded audio is silent")
    return min(0.0, -PEAK_HEADROOM_DB - float(match[1]))


def input_command(source: Path, output: Path, *, gain: float, ffmpeg: str) -> list[str]:
    return [
        ffmpeg,
        "-nostdin",
        "-hide_banner",
        "-v",
        "error",
        "-n",
        "-i",
        str(source),
        "-map",
        "0:a:0",
        "-vn",
        "-af",
        f"asetpts=N/SR/TB,aresample=44100,volume={gain:.6f}dB",
        "-c:a",
        "pcm_s16le",
        str(output),
    ]


def separate(
    source: Path,
    directory: Path,
    *,
    ffmpeg: str = "ffmpeg",
    device: str = "auto",
    prepared_input: bool = False,
    runner: Runner = _run,
    separator: Callable[[Path, Path], object] | None = None,
) -> Path:
    """Write a candidate in a new directory; prepared PCM bypasses decoding/copying."""
    require(device in ("auto", "mps", "cpu"), "Invalid separation device")
    if prepared_input:
        require(source.is_file() and not source.is_symlink(), "Expected regular prepared PCM")
        with wave.open(str(source), "rb") as stream:
            require(
                stream.getnchannels() == 2
                and stream.getsampwidth() == 2
                and stream.getframerate() == 44100
                and 0 < stream.getnframes() <= 21600 * 44100,
                "Prepared input must be stereo PCM16/44100 within six hours",
            )
    directory.mkdir(mode=0o700)
    model_input = source if prepared_input else directory / "model-input.wav"
    vocals = directory / "vocals.wav"
    gain = 0.0 if prepared_input else clip_safe_gain(runner(peak_command(source, ffmpeg=ffmpeg)))
    if not prepared_input:
        runner(input_command(source, model_input, gain=gain, ffmpeg=ffmpeg))
    result: object
    if separator is None:
        from replay.bgm_separation import SeparationConfig, _import, separate_vocals

        if device == "cpu":
            from replay.audio_cpu_probe import audit_descriptors

            audit_descriptors(directory, "before-model-import")
            _import("mlx.core")
            audit_descriptors(directory, "after-model-import")
        result = separate_vocals(model_input, vocals, SeparationConfig(device=device))
        if device == "cpu":
            audit_descriptors(directory, "after-separation")
    else:
        result = separator(model_input, vocals)
    receipt = directory / "result.json"
    with receipt.open("x", encoding="utf-8") as stream:
        os.fchmod(stream.fileno(), 0o600)
        json.dump(
            {
                "source": str(source),
                "gainDB": gain,
                "preparedInput": prepared_input,
                "device": device,
                "modelInput": str(model_input),
                "vocals": str(vocals),
                "separation": repr(result),
                "timestampCorrection": False,
                "loudnessNormalization": False,
                "deliveryComplete": False,
            },
            stream,
            indent=1,
        )
    return receipt


def verify_material(source: Path, directory: Path) -> None:
    """Check complete separated PCM and receipt, not perceptual music/speech quality."""
    receipt = read(directory / "result.json")
    model_input = source if receipt.get("preparedInput") is True else directory / "model-input.wav"
    vocals = directory / "vocals.wav"
    require(
        receipt.get("source") == str(source)
        and receipt.get("modelInput") == str(model_input)
        and receipt.get("vocals") == str(vocals)
        and receipt.get("timestampCorrection") is False
        and receipt.get("deliveryComplete") is False,
        "Separation receipt changed",
    )
    counts: list[int] = []
    for path in (model_input, vocals):
        require(path.is_file() and not path.is_symlink(), "Expected regular separation PCM")
        with wave.open(str(path), "rb") as stream:
            require(
                stream.getnchannels() == 2
                and stream.getsampwidth() == 2
                and stream.getframerate() == 44100
                and stream.getcomptype() == "NONE",
                "Unexpected separation PCM format",
            )
            count = stream.getnframes()
            require(0 < count <= 21600 * 44100, "Invalid separation duration")
            decoded = 0
            while block := stream.readframes(65536):
                decoded += len(block)
            require(decoded == count * 4, "Truncated separation PCM")
            counts.append(count)
    require(counts[0] == counts[1], "Separation changed the sample clock")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--ffmpeg", default="ffmpeg")
    parser.add_argument("--device", choices=("auto", "mps", "cpu"), default="auto")
    parser.add_argument("--prepared-input", action="store_true")
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args(argv)
    try:
        if args.verify:
            verify_material(args.source, args.directory)
        else:
            print(
                separate(
                    args.source,
                    args.directory,
                    ffmpeg=args.ffmpeg,
                    device=args.device,
                    prepared_input=args.prepared_input,
                )
            )
    except (ReplayError, OSError, wave.Error, subprocess.SubprocessError) as error:
        print(f"DELIVERY_AUDIO_ERROR {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
