"""Intersect candidate cuts with the decoded audio envelope without inventing silence.

Only outer support is considered: internal gaps/overlaps retain their existing
clock policy. Recognition is evidence against removal, never proof of silence.
"""

import argparse
import json
import os
from bisect import bisect_right
from dataclasses import asdict, dataclass
from fractions import Fraction
from math import ceil, floor
from pathlib import Path

from replay.audio_clock import AudioFrameClock
from replay.caption_projection import CutSpan
from replay.caption_repair import ClockLabel
from replay.continuous_quiet_plan import _labels
from replay.core import ReplayError
from replay.input_hash_snapshot import InputHashSnapshot
from replay.recordings import record
from replay.stages import file_digest

MAX_BYTES = 128 * 1024 * 1024
MAX_CUTS = 3000
MAX_LABELS = 100_000


@dataclass(frozen=True, slots=True)
class SupportIntersection:
    cuts: tuple[CutSpan, ...]
    removed: tuple[tuple[int, int], ...]
    first_frame: int
    end_frame: int


def _protect_labels(
    removed: tuple[tuple[int, int], ...],
    labels: tuple[ClockLabel, ...],
    fps: int,
    support: tuple[Fraction, Fraction],
) -> None:
    """Reject every label in a removed frame, except speech wholly inside decoded audio.

    Trimming to whole frames removes the partial frame at each end of the decoded audio. An
    interval label inside the audio may start (or end) in that frame; the shared token
    projection keeps it and clamps it to the retained cut. Points are never exempt.
    """
    ends = tuple(Fraction(end, fps) for _, end in removed)
    for label in labels:
        start, end = Fraction(str(label.start)), Fraction(str(label.end))
        index = bisect_right(ends, start)
        if index == len(removed):
            continue
        low, high = (Fraction(value, fps) for value in removed[index])
        point_removed = start == end and low <= start < high
        supported = start < end and support[0] <= start and end <= support[1]
        if point_removed or (start < high and end > low and not supported):
            raise ReplayError("Audio-support trimming intersects recognized speech or a point")


def intersect(
    cuts: tuple[CutSpan, ...],
    labels: tuple[ClockLabel, ...],
    clock: AudioFrameClock,
    *,
    fps: int,
) -> SupportIntersection:
    """Trim only unsupported outer frames; reject every intersecting native label.

    Whole video frames must fit the decoded envelope: ceil the start and floor
    the end. Removed points are rejected conservatively as unresolved evidence.
    Output offsets are rebuilt once; source and recognition clocks are unchanged.
    """
    if type(fps) is not int or not 1 <= fps <= 120:
        raise ReplayError("Invalid integral video frame rate")
    if not 1 <= len(cuts) <= MAX_CUTS or len(labels) > MAX_LABELS:
        raise ReplayError("Audio-support intersection exceeds its input budget")
    audio = (
        clock.pts[0] * clock.time_base,
        clock.pts[-1] * clock.time_base + Fraction(clock.samples[-1], clock.sample_rate),
    )
    first, end = ceil(audio[0] * fps), floor(audio[1] * fps)
    if first >= end:
        raise ReplayError("No complete video frame lies inside decoded audio support")
    result: list[CutSpan] = []
    removed: list[tuple[int, int]] = []
    source_end = old_offset = new_offset = 0
    for cut in cuts:
        if (
            any(type(value) is not int for value in asdict(cut).values())
            or cut.source_start_frame < source_end
            or cut.output_start_frame != old_offset
        ):
            raise ReplayError("Expected ordered disjoint cuts with contiguous output offsets")
        source_end = cut.source_end_frame
        old_offset += cut.duration_frames
        low, high = max(first, cut.source_start_frame), min(end, cut.source_end_frame)
        if low >= high:
            removed.append((cut.source_start_frame, cut.source_end_frame))
            continue
        if cut.source_start_frame < low:
            removed.append((cut.source_start_frame, low))
        if high < cut.source_end_frame:
            removed.append((high, cut.source_end_frame))
        result.append(CutSpan(low, high, new_offset))
        new_offset += high - low
    _protect_labels(tuple(removed), labels, fps, audio)
    if not result:
        raise ReplayError("Audio-support intersection removes every retained cut")
    return SupportIntersection(tuple(result), tuple(removed), first, end)


def _local(path: Path, *, output: bool = False) -> Path:
    path = path.absolute()
    if path != path.resolve(strict=not output) or not path.is_relative_to(Path.cwd()):
        raise ReplayError("Audio-support artifacts must be canonical checkout paths")
    if output:
        if (
            path.exists()
            or not path.parent.is_dir()
            or not path.is_relative_to(Path.cwd() / "edits/work")
        ):
            raise ReplayError("Expected a new repository work output")
    elif not path.is_file() or not 0 < path.stat().st_size <= MAX_BYTES:
        raise ReplayError("Expected a bounded regular JSON input")
    return path


def _read(path: Path) -> dict[str, object]:
    return record(json.loads(path.read_text(encoding="utf-8")))


def _binding(value: object) -> tuple[Path, str]:
    item = record(value)
    name, digest = item.get("path"), item.get("sha256")
    if set(item) != {"path", "sha256"} or not isinstance(name, str) or not isinstance(digest, str):
        raise ReplayError("Expected an explicit JSON artifact path and SHA256")
    return _local(Path(name)), digest


def _cuts(document: dict[str, object]) -> tuple[CutSpan, ...]:
    rows = document.get("cuts")
    if not isinstance(rows, list) or not 1 <= len(rows) <= MAX_CUTS:
        raise ReplayError("Expected bounded candidate cut rows")
    result: list[CutSpan] = []
    for value in rows:
        row = record(value)
        start, end, offset = (
            row.get(key) for key in ("source_start_frame", "source_end_frame", "output_start_frame")
        )
        if type(start) is not int or type(end) is not int or type(offset) is not int:
            raise ReplayError("Cut frame coordinates must be integral")
        result.append(CutSpan(start, end, offset))
    return tuple(result)


def run(config: Path, output: Path) -> SupportIntersection:
    """Publish a separate candidate revision bound to the original cuts and labels."""
    config, output = _local(config), _local(output, output=True)
    document = _read(config)
    refs = record(document.get("artifacts"))
    if document.get("version") != 1 or set(refs) != {"cuts", "labels", "audioClock"}:
        raise ReplayError("Expected cuts, source labels and decoded audio-clock bindings")
    bindings = {name: _binding(value) for name, value in refs.items()}
    config_digest = file_digest(config)
    source = Path(__file__).resolve()
    parents = {str(path): digest for path, digest in bindings.values()}
    parents[str(config)] = config_digest
    parents[str(source)] = file_digest(source)
    for name in ("audio_clock.py", "continuous_quiet_plan.py"):
        dependency = source.with_name(name)
        parents[str(dependency)] = file_digest(dependency)
    snapshot = InputHashSnapshot.capture((Path(name), digest) for name, digest in parents.items())
    cuts = _read(bindings["cuts"][0])
    labels = _read(bindings["labels"][0])
    fps, source_frames = cuts.get("fps"), cuts.get("sourceFrames")
    if (
        cuts.get("version") != 1
        or cuts.get("clock") != "source-presentation"
        or type(fps) is not int
        or type(source_frames) is not int
        or source_frames <= 0
    ):
        raise ReplayError("Expected a source-presentation candidate cut plan")
    cut_parents = record(cuts.get("parentSHA256"))
    label_parents = record(labels.get("parentSHA256"))
    clock_path, clock_hash = bindings["audioClock"]
    label_path, label_hash = bindings["labels"]
    if (
        cut_parents.get(str(label_path)) != label_hash
        or cut_parents.get(str(clock_path)) != clock_hash
        or label_parents.get(str(clock_path)) != clock_hash
    ):
        raise ReplayError("Cuts and recognition must bind the identical decoded audio clock")
    original = _cuts(cuts)
    if original[-1].source_end_frame > source_frames:
        raise ReplayError("Candidate cut exceeds its source frame extent")
    clock = AudioFrameClock.from_probe(_read(clock_path))
    result = intersect(original, _labels(labels), clock, fps=fps)
    snapshot.finish()
    revised = dict(cuts)
    revised.update(
        {
            "cuts": [asdict(cut) for cut in result.cuts],
            "parentSHA256": {**cut_parents, **parents},
            "audioSupport": {
                "policy": "outer-envelope-whole-frames-no-recognition-removal",
                "firstFrame": result.first_frame,
                "endFrame": result.end_frame,
                "removedFrames": [list(span) for span in result.removed],
                "recognizedSpeechRemoved": False,
                "internalGapsVerified": False,
            },
            "nonSpeechRemovalVerified": False,
            "humanReviewed": False,
            "deliveryComplete": False,
        }
    )
    with output.open("x", encoding="utf-8") as stream:
        os.fchmod(stream.fileno(), 0o600)
        json.dump(revised, stream, ensure_ascii=False, allow_nan=False, indent=2)
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args(argv)
    run(args.config, args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
