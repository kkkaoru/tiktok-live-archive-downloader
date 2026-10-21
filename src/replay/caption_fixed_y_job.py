"""Plan one fixed origin per caption run for a time range of a paged recording.

Inputs are verified evidence only: the paging output (text, display interval and line breaks), the
cached glyph measurements it used, the style calibration, the cut plan mapping the output clock to
the source clock, and the person's top edge over source time. Each run (see ``caption_fixed_y``)
gets one top origin that keeps its tallest page above the person; a run that cannot is recorded as
a failure rather than moved, shrunk or guessed. The result is a candidate-only measured-caption
document bound to the calibration and to every input it used.
"""

import argparse
import hashlib
import json
import sys
from bisect import bisect_right
from collections.abc import Sequence
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path

from replay import caption_fixed_y as fixed
from replay.core import ReplayError
from replay.fcp_tracking_caption import (
    FaceObservation,
    TrackingPosition,
    read_verified_face_observations,
)
from replay.vertical_captions import Box

COORDINATE_SYSTEM = "top-left canvas; calibrated text origin"
MAX_LINES_TOP_READER = 2
UNKNOWN_PERSON_RULES = ("fail", "region-top")
ABOVE_PERSON = "cannot fit above foreground support"  # foreground_geometry's refusal
NO_INDEX = Path("-")


@dataclass(frozen=True, slots=True)
class CutMap:
    """Output frame to source frame, from a cut plan's rows."""

    starts: tuple[int, ...]
    rows: tuple[tuple[int, int, int], ...]

    def source_frame(self, frame: int) -> int:
        index = bisect_right(self.starts, frame) - 1
        if index < 0:
            raise ReplayError("Output frame precedes the cut plan")
        output_start, length, source_start = self.rows[index]
        return source_start + min(frame - output_start, length - 1)


def cut_map(cuts: Sequence[dict[str, int]]) -> CutMap:
    """Build the output-to-source frame map from `output_start_frame`/`source_*_frame` rows."""
    rows = tuple(
        (
            c["output_start_frame"],
            c["source_end_frame"] - c["source_start_frame"],
            c["source_start_frame"],
        )
        for c in cuts
    )
    if not rows or any(b[0] <= a[0] for a, b in zip(rows, rows[1:], strict=False)):
        raise ReplayError("Cut plan rows must be ordered by output frame")
    return CutMap(tuple(row[0] for row in rows), rows)


def top_envelope(
    sizes: dict[str, tuple[float, float]], *, padding: float, ascent: float, line_advance: float
) -> fixed.Envelope:
    """The top-origin glyph envelope of a page, from cached per-line sizes and the calibration."""

    def envelope(text: str) -> Box:
        lines = text.split("\n")
        measured = [sizes[line] for line in lines]
        width = max(size[0] for size in measured)
        height = max(size[1] for size in measured) + (len(lines) - 1) * line_advance
        top = -padding - ascent
        return Box(-width / 2, top, width / 2, top + height)

    return envelope


def select_cues(
    pages: Sequence[dict[str, object]], *, start: Fraction, end: Fraction, max_lines: int
) -> list[fixed.RunCue]:
    """Pages inside [start, end) with at most `max_lines` lines, as ordered run cues."""
    if not 1 <= max_lines <= 3 or start >= end:
        raise ReplayError("Invalid cue selection")
    selected: list[fixed.RunCue] = []
    for page in pages:
        text = str(page["text"])
        first = Fraction(str(page["start"])).limit_denominator(100000)
        last = Fraction(str(page["end"])).limit_denominator(100000)
        if first >= start and last <= end and len(text.split("\n")) <= max_lines:
            selected.append(fixed.RunCue(text, first, last))
    if not selected:
        raise ReplayError("No page matches the time range and line limit")
    return selected


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def person_top_from(
    observed: Sequence[TrackingPosition], mapping: CutMap, *, rate: int, maximum_drift: float
) -> fixed.PersonTop:
    """A person-top function over output seconds; observed samples live in source time."""

    def person_top(start: Fraction, end: Fraction) -> float:
        first = int(start * rate)
        last = max(first + 1, int(end * rate)) - 1
        source_first, source_last = mapping.source_frame(first), mapping.source_frame(last)
        estimate = fixed.person_top_with_hold(
            observed,
            Fraction(source_first, rate),
            Fraction(max(source_last, source_first) + 1, rate),
            maximum_drift=maximum_drift,
        )
        return estimate.top

    return person_top


def plan_runs(
    cues: Sequence[fixed.RunCue],
    *,
    rate: int,
    region: Box,
    center_x: float,
    clearance: float,
    envelope: fixed.Envelope,
    person_top: fixed.PersonTop,
    unknown_person: str = "fail",
) -> tuple[list[fixed.FixedCue], list[dict[str, object]], list[dict[str, object]]]:
    """Fixed origins for every run that fits; the others are returned as recorded failures.

    With ``unknown_person="region-top"`` a run whose person position no observation bounds is
    placed at the top of the region, the origin farthest from any person below it, instead of
    being dropped; such runs are returned separately so the choice stays visible. The same rule
    covers a run that cannot clear the person at one origin (the head rises during it): each of
    its cues gets its own origin, and a cue that still cannot clear the person goes to the top.
    """
    if unknown_person not in UNKNOWN_PERSON_RULES:
        raise ReplayError(f"Unknown unobserved-person rule: {unknown_person}")
    placed: list[fixed.FixedCue] = []
    failures: list[dict[str, object]] = []
    topped: list[dict[str, object]] = []
    for number, run in enumerate(fixed.split_runs(cues, rate=rate)):
        members = [cues[i] for i in run]
        try:
            placed.extend(
                fixed.fixed_origins(
                    members, rate=rate, region=region, center_x=center_x, clearance=clearance,
                    envelope=envelope, person_top=person_top,
                )
            )  # fmt: skip
        except fixed.UnknownPerson as error:
            if unknown_person == "fail":
                failures.append(failure(number, members, error))
                continue
            placed.extend(top_origins(members, number, region=region, center_x=center_x,
                                      envelope=envelope))  # fmt: skip
            topped.append(failure(number, members, error))
        except ReplayError as error:
            if unknown_person == "fail" or ABOVE_PERSON not in str(error):
                failures.append(failure(number, members, error))
                continue
            for cue in members:
                try:
                    placed.extend(
                        fixed.fixed_origins(
                            [cue], rate=rate, region=region, center_x=center_x,
                            clearance=clearance, envelope=envelope, person_top=person_top,
                        )
                    )  # fmt: skip
                except (fixed.UnknownPerson, ReplayError) as single:
                    reason: Exception = single
                    if isinstance(single, fixed.UnknownPerson) or ABOVE_PERSON in str(single):
                        try:
                            top = top_origins(
                                [cue], number, region=region, center_x=center_x, envelope=envelope
                            )
                            placed.extend(top)
                            topped.append(failure(number, [cue], single))
                            continue
                        except ReplayError as unplaced:
                            reason = unplaced
                    failures.append(failure(number, [cue], reason))
    return placed, failures, topped


def failure(number: int, members: Sequence[fixed.RunCue], error: Exception) -> dict[str, object]:
    return {"run": number, "cues": len(members), "start": float(members[0].start),
            "error": str(error)}  # fmt: skip


def top_origins(
    members: Sequence[fixed.RunCue],
    number: int,
    *,
    region: Box,
    center_x: float,
    envelope: fixed.Envelope,
) -> list[fixed.FixedCue]:
    """One origin for the run at the top of the region: every cue's whole envelope inside it."""
    boxes = [envelope(cue.text) for cue in members]
    for box in boxes:
        if center_x + box.left < region.left or center_x + box.right > region.right:
            raise ReplayError("Measured caption cannot fit at the fixed horizontal origin")
    origin = max(region.top - box.top for box in boxes)
    if any(origin + box.bottom > region.bottom for box in boxes):
        raise ReplayError("A caption run does not fit at one origin inside the region")
    return [fixed.FixedCue(cue, number, center_x, origin) for cue in members]


def rebased(cues: Sequence[fixed.FixedCue], *, offset: Fraction) -> list[fixed.FixedCue]:
    """The same cues with their times moved so the range starts at zero."""
    return [
        fixed.FixedCue(
            fixed.RunCue(item.cue.text, item.cue.start - offset, item.cue.end - offset),
            item.run,
            item.x,
            item.y,
        )
        for item in cues
    ]


def observed_only(samples: Sequence[FaceObservation]) -> list[TrackingPosition]:
    """The observed face-top samples in time order; frames without a face are not invented."""
    ordered = sorted(samples, key=lambda sample: sample.time)
    return [
        TrackingPosition(item.time, item.position.x, item.position.y)
        for item in ordered
        if item.position is not None
    ]


def load_observed(window_index: Path) -> list[TrackingPosition]:
    """Observed face-top samples (source time) from every verified raw face window."""
    samples: list[FaceObservation] = []
    for window in json.loads(window_index.read_text(encoding="utf-8"))["windows"]:
        series = read_verified_face_observations(
            Path(window["config"]), Path(window["evidence"]), anchor="top"
        )
        samples.extend(series.samples)
    return observed_only(samples)


def load_yolo(
    track: Path, *, minimum_confidence: float, canvas_width: float
) -> list[TrackingPosition]:
    """Face-top samples from a `yolo_track` evidence file; frames with 0 or 2+ faces are skipped.

    A frame with several faces is ambiguous, exactly as Vision's ambiguous frames are, and is not
    guessed. The sample time is the frame's presentation timestamp (v2 tracks); v1 tracks derived
    time from the frame index and drift on variable-frame-rate sources, so they are refused. The
    source's nominal rate is therefore irrelevant (it need not equal the output rate). The sample
    position is the face box's horizontal centre and its top edge, scaled from the tracked
    source's pixels to the caption canvas (a 720x1280 source is composed onto 1080x1920).
    """
    data = json.loads(track.read_text(encoding="utf-8"))
    if data.get("protocol") != "yolo-face-caption-track-v2" or data.get("provenance") != "yolo":
        raise ReplayError("Not a timestamped (v2) YOLO face track")
    base = Fraction(str(data["timeBase"]))
    width = data.get("width")
    if isinstance(width, bool) or not isinstance(width, int) or width <= 0:
        raise ReplayError("The YOLO face track records no source width")
    scale = canvas_width / width
    samples: list[TrackingPosition] = []
    for row in data["frames"]:
        box = row["face"]
        if box is None or row["faceCount"] != 1 or box[4] < minimum_confidence:
            continue
        samples.append(
            TrackingPosition(int(row["pts"]) * base, (box[0] + box[2]) / 2 * scale, box[1] * scale)
        )
    return sorted(samples, key=lambda sample: sample.time)


def merge_observed(
    vision: Sequence[TrackingPosition], yolo: Sequence[TrackingPosition], *, gap: Fraction
) -> list[TrackingPosition]:
    """Vision samples, plus YOLO samples that have no Vision sample within `gap` seconds."""
    times = [sample.time for sample in vision]
    added = []
    for sample in yolo:
        index = bisect_right(times, sample.time)
        near = [times[i] for i in (index - 1, index) if 0 <= i < len(times)]
        if all(abs(sample.time - time) > gap for time in near):
            added.append(sample)
    return sorted([*vision, *added], key=lambda sample: sample.time)


def candidate_for(
    placed: Sequence[fixed.FixedCue],
    *,
    canvas: tuple[int, int],
    rate: int,
    calibration: Path,
    inputs: Sequence[Path],
) -> dict[str, object]:
    """The candidate document bound to the calibration and to every input file it used."""
    digests = {str(path): sha256(path) for path in (calibration, *inputs)}
    return fixed.candidate_document(
        placed,
        canvas=canvas,
        rate=rate,
        coordinate_system=COORDINATE_SYSTEM,
        inputs=digests,
        calibration=(str(calibration), digests[str(calibration)]),
    )


def line_sizes(paging: Path) -> dict[str, tuple[float, float]]:
    """Cached single-line (width, height) from the paging run's native measurement responses."""
    sizes: dict[str, tuple[float, float]] = {}
    for response in sorted(paging.glob("metrics-*/batch-*.response.json")):
        for item in json.loads(response.read_text(encoding="utf-8"))["bounds"]:
            sizes[item["text"]] = (float(item["width"]), float(item["height"]))
    if not sizes:
        raise ReplayError("The paging directory holds no cached line measurements")
    return sizes


def parse(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paging", type=Path, help="paging directory with reflow.json and metrics-*")
    parser.add_argument(
        "window_index", type=Path, help="verified raw face windows index, or - for none"
    )
    parser.add_argument("cuts", type=Path, help="supported-cuts.json (output to source frames)")
    parser.add_argument("calibration", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--start-seconds", type=Fraction, required=True)
    parser.add_argument("--end-seconds", type=Fraction, required=True)
    parser.add_argument("--max-lines", type=int, default=MAX_LINES_TOP_READER)
    parser.add_argument("--rate", type=int, default=30)
    parser.add_argument("--region-top", type=float, required=True)
    parser.add_argument("--region-bottom", type=float, required=True)
    parser.add_argument("--clearance", type=float, required=True)
    parser.add_argument("--maximum-drift", type=float, required=True)
    parser.add_argument("--rebase", action="store_true")
    parser.add_argument("--unknown-person", choices=UNKNOWN_PERSON_RULES, default="fail")
    parser.add_argument("--yolo-track", type=Path, help="yolo_track evidence for unobserved frames")
    parser.add_argument("--yolo-minimum-confidence", type=float, default=0.5)
    parser.add_argument("--yolo-gap-seconds", type=Fraction, default=Fraction(1, 2))
    return parser.parse_args(argv)


def run_plan(args: argparse.Namespace, observed: Sequence[TrackingPosition]) -> dict[str, object]:
    """Plan the range from the arguments and observed samples; write the candidate document."""
    if args.output.exists() or args.output.is_symlink():
        raise ReplayError("Refusing to overwrite the output")
    calibration = json.loads(args.calibration.read_text(encoding="utf-8"))
    width, height = calibration["canvas"]
    reflow = args.paging / "reflow.json"
    cues = select_cues(
        json.loads(reflow.read_text(encoding="utf-8"))["captions"],
        start=args.start_seconds,
        end=args.end_seconds,
        max_lines=args.max_lines,
    )
    mapping = cut_map(json.loads(args.cuts.read_text(encoding="utf-8"))["cuts"])
    placed, failures, topped = plan_runs(
        cues,
        rate=args.rate,
        region=Box(0.0, args.region_top, float(width), args.region_bottom),
        center_x=width / 2,
        clearance=args.clearance,
        envelope=top_envelope(
            line_sizes(args.paging),
            padding=float(calibration["padding"]),
            ascent=float(calibration["nativeFont"]["renderedAscent"]),
            line_advance=float(calibration["style"]["motion_line_advance"]),
        ),
        person_top=person_top_from(
            observed, mapping, rate=args.rate, maximum_drift=args.maximum_drift
        ),
        unknown_person=args.unknown_person,
    )
    chosen = rebased(placed, offset=args.start_seconds) if args.rebase else placed
    document = candidate_for(
        chosen,
        canvas=(int(width), int(height)),
        rate=args.rate,
        calibration=args.calibration,
        inputs=[
            reflow,
            *([args.window_index] if args.window_index != NO_INDEX else []),
            args.cuts,
            *([args.yolo_track] if args.yolo_track is not None else []),
        ],
    )
    document["plannedRuns"] = {
        "placedCues": len(placed),
        "failedRuns": failures,
        "regionTopRuns": topped,
    }
    args.output.write_text(json.dumps(document, ensure_ascii=False, indent=1), encoding="utf-8")
    return {"placedCues": len(placed), "failedRuns": len(failures)}


def main(argv: list[str] | None = None) -> int:
    args = parse(argv)
    if args.output.exists() or args.output.is_symlink():
        raise ReplayError("Refusing to overwrite the output")
    vision = [] if args.window_index == NO_INDEX else load_observed(args.window_index)
    if args.yolo_track is None:
        observed = vision
    else:
        canvas = json.loads(args.calibration.read_text(encoding="utf-8"))["canvas"]
        yolo = load_yolo(
            args.yolo_track,
            minimum_confidence=args.yolo_minimum_confidence,
            canvas_width=float(canvas[0]),
        )
        observed = merge_observed(vision, yolo, gap=args.yolo_gap_seconds)
    print(json.dumps(run_plan(args, observed)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
