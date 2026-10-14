"""Select whole native words from an overlapping completed ASR window.

An explicit label block can be replaced by a bounded native-clock alternative.
The old assembly is immutable; text differences remain unreviewed candidates.
"""

import argparse
import json
import os
from dataclasses import asdict
from itertools import pairwise
from pathlib import Path

from replay.asr_assembly import _labels as window_labels
from replay.asr_assembly import _windows
from replay.asr_clock_projection import _binding, _labels
from replay.batch_asr import RATE
from replay.caption_measured_job import _json
from replay.caption_repair import ClockLabel
from replay.input_hash_snapshot import InputHashSnapshot
from replay.recordings import record
from replay.stages import file_digest

MAX_BYTES = 128 * 1024 * 1024
MAX_REPLACEMENT_SECONDS = 60


def _read(path: Path) -> dict[str, object]:
    if path.stat().st_size > MAX_BYTES:
        raise ValueError("Refinement JSON exceeds byte budget")
    return record(_json(path.read_bytes()))


def replace_block(
    original: tuple[ClockLabel, ...],
    alternative: tuple[ClockLabel, ...],
    *,
    first: int,
    stop: int,
    start_sample: int,
    end_sample: int,
) -> tuple[tuple[ClockLabel, ...], tuple[ClockLabel, ...]]:
    """Preserve order and every external label; never cut a recognized word."""
    if (
        any(type(value) is not int for value in (first, stop, start_sample, end_sample))
        or not 0 <= first < stop <= len(original)
        or not 0 <= start_sample < end_sample
        or end_sample - start_sample > MAX_REPLACEMENT_SECONDS * RATE
    ):
        raise ValueError("Expected bounded label indices and PCM interval")
    start, end = start_sample / RATE, end_sample / RATE
    removed = original[first:stop]
    if max(label.end for label in removed) - min(
        label.start for label in removed
    ) > MAX_REPLACEMENT_SECONDS or any(
        label.end < start or label.start >= end for label in removed
    ):
        raise ValueError("Removed labels exceed or miss the bounded replacement support")
    selected = tuple(
        label
        for label in alternative
        if label.end > start and label.start < end or start <= label.start == label.end < end
    )
    if not selected or any(label.start < start or label.end > end for label in selected):
        raise ValueError("Alternative is empty or replacement boundary splits a word")
    if first and original[first - 1].end > selected[0].start:
        raise ValueError("Alternative overlaps the unchanged left neighbor")
    if stop < len(original) and max(label.end for label in selected) > original[stop].start:
        raise ValueError("Alternative overlaps the unchanged right neighbor")
    result = original[:first] + selected + original[stop:]
    if any(b.start < a.start for a, b in pairwise(result)):
        raise ValueError("Candidate recognition still reverses word onsets")
    return result, selected


def _alternative(
    paths: dict[str, Path], pins: dict[str, str], original: dict[str, object]
) -> tuple[ClockLabel, ...]:
    inputs, complete, clock = (_read(paths[name]) for name in ("inputs", "complete", "clock"))
    windows = _windows(inputs)
    matches = [window for window in windows if window.start_sample == clock.get("startSample")]
    hashes = complete.get("rawSHA256")
    if (
        len(matches) != 1
        or not isinstance(hashes, list)
        or len(hashes) != len(windows)
        or complete.get("windows") != len(windows)
        or complete.get("deliveryComplete") is not False
    ):
        raise ValueError("Incomplete or unsupported alternative window")
    window = matches[0]
    if (
        hashes[windows.index(window)] != pins["raw"]
        or clock
        != {
            "startSample": window.start_sample,
            "endSample": window.end_sample,
            "sampleRate": RATE,
            "humanReviewed": False,
            "emptyRecognitionIsNotSilence": True,
            "rawSHA256": pins["raw"],
        }
        or any(
            record(original.get("parentSHA256")).get(str(paths[name])) != pins[name]
            for name in ("inputs", "complete", "raw", "clock")
        )
    ):
        raise ValueError("Alternative must be bound by the original completed assembly")
    return tuple(
        window_labels(
            _read(paths["raw"]),
            window=window,
            owner_start=window.start_sample / RATE,
            owner_end=window.end_sample / RATE,
        )
    )


def run(config: Path, output: Path) -> int:
    """Adopt an explicit overlap candidate with before/after pins and text evidence."""
    if output.exists() or output.is_symlink() or output.parent != output.parent.resolve():
        raise ValueError("Expected a new output under a canonical directory")
    config_digest = file_digest(config)
    data = _read(config)
    if set(data) != {"version", "artifacts", "replacement"} or data["version"] != 1:
        raise ValueError("Unsupported overlap refinement configuration")
    artifacts = record(data["artifacts"])
    if set(artifacts) != {"labels", "inputs", "complete", "raw", "clock"}:
        raise ValueError("Expected completed original and alternative ASR bindings")
    bindings = {name: _binding(value) for name, value in artifacts.items()}
    paths = {name: value[0] for name, value in bindings.items()}
    pins = {name: value[1] for name, value in bindings.items()}
    dependencies = record(_read(paths["inputs"]).get("inputs"))
    extra = [_binding({"path": name, "sha256": value}) for name, value in dependencies.items()]
    snapshot = InputHashSnapshot.capture([(config, config_digest), *bindings.values(), *extra])
    original = _read(paths["labels"])
    labels = _labels(original)
    alternative = _alternative(paths, pins, original)
    interval = record(data["replacement"])
    names = ("firstLabel", "stopLabel", "startSample", "endSample")
    if set(interval) != set(names) or any(type(interval[name]) is not int for name in names):
        raise ValueError("Expected exact integer replacement coordinates")
    first, stop, start, end = (int(str(interval[name])) for name in names)
    result, selected = replace_block(
        labels, alternative, first=first, stop=stop, start_sample=start, end_sample=end
    )
    snapshot.finish()
    document = {
        **original,
        "labels": [asdict(label) for label in result],
        "parentSHA256": {
            **record(original["parentSHA256"]),
            **{str(path): digest for path, digest in bindings.values()},
        },
        "refinementSHA256": file_digest(Path(__file__)),
        "refinement": {
            "configSHA256": config_digest,
            "replacement": interval,
            "removed": [asdict(label) for label in labels[first:stop]],
            "selected": [asdict(label) for label in selected],
            "clockOffsetApplications": 1,
            "sourceProjectionApplications": 0,
            "candidateOnly": True,
        },
        "onsetReversals": 0,
        "adjacentOverlaps": sum(b.start < a.end for a, b in pairwise(result)),
        "unresolvedPoints": sum(label.start == label.end for label in result),
        "humanReviewed": False,
        "speechAccuracyVerified": False,
        "deliveryComplete": False,
    }
    with output.open("x", encoding="utf-8") as stream:
        os.fchmod(stream.fileno(), 0o600)
        json.dump(document, stream, ensure_ascii=False, allow_nan=False, indent=2)
    return len(result)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args(argv)
    run(args.config, args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
