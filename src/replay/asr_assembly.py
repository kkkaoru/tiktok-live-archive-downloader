"""Assemble completed raw ASR windows without inventing or repairing word clocks."""

import argparse
import json
import os
from dataclasses import asdict
from pathlib import Path

from . import caption_io
from .audio_windows import AudioWindow, plan_audio_windows
from .batch_asr import MAX_JSON_BYTES, MAX_SECONDS, RATE
from .caption_io import mlx_groups
from .caption_repair import ClockLabel
from .core import ReplayError
from .local_asr import fingerprint
from .recordings import record


def _read(path: Path) -> dict[str, object]:
    if path.is_symlink() or not path.is_file() or path.stat().st_size > MAX_JSON_BYTES:
        raise ReplayError("Expected bounded regular ASR evidence")
    return record(json.loads(path.read_text(encoding="utf-8")))


def _windows(manifest: dict[str, object]) -> tuple[AudioWindow, ...]:
    samples = manifest.get("samples")
    if (
        type(samples) is not int
        or not 0 < samples <= RATE * MAX_SECONDS
        or manifest.get("version") != 1
        or manifest.get("sampleRate") != RATE
        or manifest.get("windowSamples") != 60 * RATE
        or manifest.get("strideSamples") != 30 * RATE
        or manifest.get("clock") != "input-pcm-not-projected"
    ):
        raise ReplayError("Unsupported ASR window definition")
    return plan_audio_windows(samples, window_samples=60 * RATE, stride_samples=30 * RATE)


def _labels(
    raw: dict[str, object], *, window: AudioWindow, owner_start: float, owner_end: float
) -> list[ClockLabel]:
    # Empty raw windows remain explicit evidence, never a silence decision.
    if raw.get("segments") == [] and raw.get("text") == "" and raw.get("language") == "ja":
        return []
    groups = mlx_groups(raw, duration=window.sample_count / RATE)
    offset = window.start_sample / RATE
    # Same half-open midpoint ownership as caption_timing.owned_tokens, including
    # unresolved points that SpokenToken intentionally cannot represent.
    return [
        ClockLabel(
            group.text,
            group.start + offset,
            group.end + offset,
            f"{window.start_sample:012d}:{index}",
        )
        for index, group in enumerate(groups)
        if owner_start <= (group.start + group.end) / 2 + offset < owner_end
    ]


def assemble(source: Path, output: Path) -> int:
    """Select native groups from immutable completed windows on the INPUT PCM clock.

    Midpoints partition overlap, not semantic duplicates. Points, interval overlaps
    and onset reversals remain unresolved; no sorting, stretching, source-media
    projection, silence detection or caption/delivery approval occurs here.
    """
    if source.is_symlink() or output.exists() or output.is_symlink():
        raise ReplayError("Expected regular input directory and new output")
    manifest_path, complete_path = source / "inputs.json", source / "complete.json"
    manifest, complete = _read(manifest_path), _read(complete_path)
    windows = _windows(manifest)
    hashes = complete.get("rawSHA256")
    if (
        type(complete.get("windows")) is not int
        or complete.get("windows") != len(windows)
        or not isinstance(hashes, list)
        or len(hashes) != len(windows)
        or complete.get("deliveryComplete") is not False
    ):
        raise ReplayError("Incomplete ASR collection")
    inputs = record(manifest.get("inputs"))
    if not inputs or len(inputs) > 16:
        raise ReplayError("Invalid ASR input bindings")
    for name, digest in inputs.items():
        if not Path(name).is_absolute() or fingerprint(Path(name)) != digest:
            raise ReplayError("ASR input fingerprint changed")
    bindings = {
        str(manifest_path): fingerprint(manifest_path),
        str(complete_path): fingerprint(complete_path),
    }
    labels: list[ClockLabel] = []
    empty_windows = 0
    for index, window in enumerate(windows):
        raw_path = source / f"{window.start_sample:012d}.raw.json"
        clock_path = source / f"{window.start_sample:012d}.clock.json"
        digest = fingerprint(raw_path)
        expected = {
            "startSample": window.start_sample,
            "endSample": window.end_sample,
            "sampleRate": RATE,
            "humanReviewed": False,
            "emptyRecognitionIsNotSilence": True,
            "rawSHA256": digest,
        }
        if digest != hashes[index] or _read(clock_path) != expected:
            raise ReplayError("ASR window fingerprint or clock changed")
        raw = _read(raw_path)
        left = (windows[index - 1].end_sample + window.start_sample) / (2 * RATE) if index else 0.0
        right = (
            (window.end_sample + windows[index + 1].start_sample) / (2 * RATE)
            if index + 1 < len(windows)
            else window.end_sample / RATE
        )
        labels.extend(_labels(raw, window=window, owner_start=left, owner_end=right))
        empty_windows += raw.get("segments") == []
        bindings[str(raw_path)] = digest
        bindings[str(clock_path)] = fingerprint(clock_path)
    # Recheck evidence after parsing; raw files are never rewritten by assembly.
    if any(fingerprint(Path(name)) != digest for name, digest in bindings.items()):
        raise ReplayError("ASR evidence changed during assembly")
    result = {
        "version": 1,
        "clock": "input-pcm-not-projected",
        "labels": [asdict(label) for label in labels],
        "parentSHA256": bindings,
        "assemblySHA256": fingerprint(Path(__file__)),
        "parserSHA256": fingerprint(Path(caption_io.__file__)),
        "emptyWindows": empty_windows,
        "unresolvedPoints": sum(label.start == label.end for label in labels),
        "onsetReversals": sum(b.start < a.start for a, b in zip(labels, labels[1:], strict=False)),
        "adjacentOverlaps": sum(b.start < a.end for a, b in zip(labels, labels[1:], strict=False)),
        "humanReviewed": False,
        "speechAccuracyVerified": False,
        "deliveryComplete": False,
    }
    with output.open("x", encoding="utf-8") as stream:
        os.fchmod(stream.fileno(), 0o600)
        json.dump(result, stream, ensure_ascii=False, allow_nan=False, indent=2)
    return len(labels)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args(argv)
    assemble(args.source, args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
