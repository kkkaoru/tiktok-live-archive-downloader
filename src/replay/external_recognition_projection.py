"""Project externally produced continuous-PCM recognition onto source presentation.

The shared asr_clock_projection adapter requires preparation-plan receipts that
only exist when separation and recognition ran inside this repository's
factory. When an authorized external producer supplies the vocals and word
recognition (e.g. a GPU separation plus whisper-large-v3-mlx batch), this
module applies the identical shared math (AudioFrameClock.from_probe plus
project_labels) and the identical duration-preservation checks, with the
caller declaring the lineage in a config file. Hashes bind every input;
measured durations are recorded, never assumed. Nothing here branches on a
video ID, date, duration or resolution.
"""

from __future__ import annotations

import argparse
import json
import os
import wave
from dataclasses import asdict
from fractions import Fraction
from pathlib import Path
from typing import Final

from replay.asr_clock_projection import project_labels
from replay.audio_clock import AudioFrameClock
from replay.caption_repair import ClockLabel
from replay.core import ReplayError
from replay.recordings import record
from replay.stages import file_digest

MAX_JSON_BYTES: Final[int] = 128 * 1024 * 1024


def _read(path: Path) -> dict[str, object]:
    if path.is_symlink() or not path.is_file() or path.stat().st_size > MAX_JSON_BYTES:
        raise ReplayError("Projection input must be a bounded regular file")
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


def _number(value: object, message: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ReplayError(message)
    return float(value)


def run(config: Path, output: Path) -> int:
    """Project caller-declared external recognition; never infer missing times."""
    if (
        output.exists()
        or output.is_symlink()
        or output.parent != output.parent.resolve(strict=True)
    ):
        raise ReplayError("Expected a new output under a canonical directory")
    config_digest = file_digest(config)
    data = _read(config)
    if data.get("version") != 1 or data.get("inputClock") != "continuous-decoded-pcm":
        raise ReplayError("Unsupported projection clock contract")
    names = ("labels", "pcm", "audioClock", "asrInputs")
    artifacts = record(data.get("artifacts"))
    if set(artifacts) != set(names):
        raise ReplayError("Projection requires complete caller-declared lineage")
    paths = {name: _binding(artifacts[name])[0] for name in names}
    digests = {name: record(artifacts[name])["sha256"] for name in names}
    assembled, manifest = _read(paths["labels"]), _read(paths["asrInputs"])
    if (
        assembled.get("version") != 1
        or assembled.get("clock") != "input-pcm-not-projected"
        or manifest.get("clock") != "input-pcm-not-projected"
        or record(manifest.get("inputs")).get(str(paths["pcm"])) != digests["pcm"]
    ):
        raise ReplayError("Recognition does not bind the declared PCM")
    values = assembled.get("labels")
    if not isinstance(values, list) or not 1 <= len(values) <= 100_000:
        raise ReplayError("Expected bounded unprojected assembled recognition")
    labels: list[ClockLabel] = []
    for value in values:
        item = record(value)
        text, provenance = item.get("text"), item.get("provenance")
        if not isinstance(text, str) or not isinstance(provenance, str):
            raise ReplayError("Recognition text/provenance is missing")
        labels.append(
            ClockLabel(
                text,
                _number(item.get("start"), "Recognition clock must be numeric"),
                _number(item.get("end"), "Recognition clock must be numeric"),
                provenance,
            )
        )
    clock = AudioFrameClock.from_probe(_read(paths["audioClock"]))
    with wave.open(str(paths["pcm"]), "rb") as stream:
        samples, rate = stream.getnframes(), stream.getframerate()
        if (
            rate != 16000
            or stream.getnchannels() != 1
            or stream.getsampwidth() != 2
            or stream.getcomptype() != "NONE"
            or samples != manifest.get("samples")
            or rate != manifest.get("sampleRate")
            or abs(Fraction(samples, rate) - Fraction(clock.sample_ends[-1], clock.sample_rate))
            > Fraction(1, rate)
        ):
            raise ReplayError("PCM is not duration-preserving mono PCM16/16k recognition input")
    projected = project_labels(tuple(labels), clock)
    pcm_seconds = Fraction(samples, rate)
    clock_seconds = Fraction(clock.sample_ends[-1], clock.sample_rate)
    result = {
        "version": 1,
        "clock": "source-presentation",
        "labels": [asdict(label) for label in projected],
        "parentSHA256": {str(paths[name]): str(digests[name]) for name in names},
        "configSHA256": config_digest,
        "measuredDuration": {
            "pcmSeconds": str(pcm_seconds),
            "clockSeconds": str(clock_seconds),
            "driftSeconds": str(abs(pcm_seconds - clock_seconds)),
        },
        "unresolvedPoints": sum(label.start == label.end for label in projected),
        "lineageIsCallerDeclared": True,
        "humanReviewed": False,
        "speechAccuracyVerified": False,
        "deliveryComplete": False,
    }
    with output.open("x", encoding="utf-8") as stream:
        os.fchmod(stream.fileno(), 0o600)
        json.dump(result, stream, ensure_ascii=False, allow_nan=False, indent=2)
    return len(projected)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args(argv)
    print(f"Projected external recognition labels: {run(args.config, args.output)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
