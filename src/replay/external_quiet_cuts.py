"""Caller-declared quiet-cut planning for externally produced vocals/PCM.

Mirrors continuous_quiet_plan.run stage for stage (FFmpeg astats peaks,
measured_quiet_runs, source_quiet, quiet_cut_frames) but binds caller-declared
lineage instead of preparation-plan receipts: the stereo vocals and word
recognition may come from an authorized external producer, as long as every
artifact is hash-bound and the stereo PCM is duration-preserving against the
measured source audio clock. No video-ID, date, duration or resolution branch.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import wave
from dataclasses import asdict
from fractions import Fraction
from math import ceil
from pathlib import Path
from typing import Final

from replay.audio_clock import AudioFrameClock
from replay.continuous_quiet_plan import _labels as _projected_labels
from replay.continuous_quiet_plan import source_quiet
from replay.core import ReplayError
from replay.delivery_media import read_clock
from replay.quiet_cuts import quiet_cut_frames
from replay.recordings import record
from replay.stages import file_digest
from replay.waveform_quiet import measured_quiet_runs

MAX_JSON_BYTES: Final[int] = 128 * 1024 * 1024


def _read(path: Path) -> dict[str, object]:
    if path.is_symlink() or not path.is_file() or path.stat().st_size > MAX_JSON_BYTES:
        raise ReplayError("Cut planning input must be a bounded regular file")
    return record(json.loads(path.read_text(encoding="utf-8")))


def _binding(value: object) -> tuple[Path, str]:
    item = record(value)
    path, digest = item.get("path"), item.get("sha256")
    if set(item) != {"path", "sha256"} or not isinstance(path, str) or not isinstance(digest, str):
        raise ReplayError("Expected explicit artifact path and SHA256")
    candidate = Path(path)
    if file_digest(candidate) != digest:
        raise ReplayError("Artifact digest does not match its bytes")
    return candidate, digest


def run(config: Path, output: Path, *, ffmpeg: str) -> int:
    """Plan conservative cuts from measured vocals quiet; never invent silence."""
    if output.exists() or output.is_symlink():
        raise ReplayError("Expected a new cut planning output directory")
    config_digest = file_digest(config)
    data = _read(config)
    fps = data.get("fps")
    if (
        data.get("version") != 1
        or data.get("pcmClock") != "continuous-decoded-pcm"
        or type(fps) is not int
        or not 1 <= fps <= 120
    ):
        raise ReplayError("Invalid source-clock planning contract")
    names = ("labels", "stereoPCM", "audioClock", "videoClock")
    artifacts = record(data.get("artifacts"))
    if set(artifacts) != set(names):
        raise ReplayError("Incomplete quiet/recognition clock lineage")
    bindings = {name: _binding(artifacts[name]) for name in names}
    paths = {name: binding[0] for name, binding in bindings.items()}
    labels = _projected_labels(_read(paths["labels"]))
    clock = AudioFrameClock.from_probe(_read(paths["audioClock"]))
    with wave.open(str(paths["stereoPCM"]), "rb") as stream:
        samples, rate = stream.getnframes(), stream.getframerate()
        if (
            stream.getnchannels() != 2
            or stream.getsampwidth() != 2
            or stream.getcomptype() != "NONE"
            or not 8000 <= rate <= 192000
            or not 0 < samples <= rate * 21600
            or abs(Fraction(samples, rate) - Fraction(clock.sample_ends[-1], clock.sample_rate))
            > Fraction(1, rate)
        ):
            raise ReplayError("Quiet input must preserve stereo decoded-sample duration")
    video = read_clock(paths["videoClock"])
    source_frames = ceil((video.pts[-1] + video.last_duration) * video.time_base * fps)
    output.mkdir(mode=0o700)
    peaks = output / "peaks.txt"
    window = max(1, rate // 50)
    command = [
        ffmpeg,
        "-nostdin",
        "-v",
        "error",
        "-i",
        str(paths["stereoPCM"]),
        "-af",
        f"asetnsamples=n={window}:p=0,astats=metadata=1:reset=1,ametadata=print:key=lavfi.astats.Overall.Peak_level:file=-",
        "-f",
        "null",
        "-",
    ]
    with peaks.open("xb") as stdout, (output / "ffmpeg.stderr.log").open("xb") as stderr:
        subprocess.run(
            command,
            stdin=subprocess.DEVNULL,
            stdout=stdout,
            stderr=stderr,
            check=True,
            timeout=3600,
        )
    with peaks.open(encoding="utf-8") as lines:
        quiet = measured_quiet_runs(
            lines, sample_count=samples, sample_rate=rate, window_samples=window
        )
    mapped = source_quiet(quiet, clock, measurement_rate=rate)
    cuts = quiet_cut_frames(mapped, labels, source_frames=source_frames, fps=fps)
    result = {
        "version": 1,
        "clock": "source-presentation",
        "fps": fps,
        "sourceFrames": source_frames,
        "cuts": [asdict(cut) for cut in cuts],
        "sourceQuiet": [asdict(span) for span in mapped],
        "parentSHA256": {
            str(paths[name]): str(record(artifacts[name])["sha256"]) for name in names
        },
        "configSHA256": config_digest,
        "peakSHA256": file_digest(peaks),
        "command": command,
        "lineageIsCallerDeclared": True,
        "nonSpeechRemovalVerified": False,
        "humanReviewed": False,
        "deliveryComplete": False,
    }
    with (output / "cuts.json").open("x", encoding="utf-8") as stream:
        os.fchmod(stream.fileno(), 0o600)
        json.dump(result, stream, ensure_ascii=False, allow_nan=False, indent=2)
    return len(cuts)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--ffmpeg", default="ffmpeg")
    args = parser.parse_args(argv)
    print(f"Planned quiet cuts: {run(args.config, args.output, ffmpeg=args.ffmpeg)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
