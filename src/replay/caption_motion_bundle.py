"""Re-page output-clock caption cues with shared Japanese POS boundaries.

This creates an experimental Motion title-plan bundle. It preserves recognized
onsets, cut ownership, source text and the caller's measured horizontal slot; it
does not certify native rendering or delivery.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path

from replay.caption_timing import CaptionPolicy, SpokenToken, make_captions
from replay.core import ReplayError
from replay.recordings import record

MAX_BYTES = 64 * 1024 * 1024
MAX_CUES = 100_000
MAX_CUTS = 3_000
MAX_TITLE_CUES = 32
MAX_TITLE_SECONDS = 120


def _read(path: Path) -> tuple[bytes, dict[str, object]]:
    absolute = path.absolute()
    if absolute.resolve() != absolute or absolute.is_symlink():
        raise ReplayError("Caption plan inputs must be canonical regular files")
    try:
        info = absolute.stat()
        if not absolute.is_file() or info.st_nlink != 1 or not 0 < info.st_size <= MAX_BYTES:
            raise ReplayError("Caption plan input is not a bounded regular file")
        payload = absolute.read_bytes()
    except OSError as error:
        raise ReplayError("Caption plan input is unavailable") from error
    if len(payload) != info.st_size:
        raise ReplayError("Caption plan input changed while reading")
    try:
        result = record(json.loads(payload))
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as error:
        raise ReplayError("Caption plan input is invalid JSON") from error
    return payload, result


def _cuts(value: dict[str, object]) -> tuple[tuple[int, int, int], ...]:
    rows = value.get("cuts")
    if value.get("version") != 1 or value.get("clock") != "source-presentation":
        raise ReplayError("Cut plan must retain its source-presentation clock")
    if value.get("fps") != 30 or not isinstance(rows, list) or not 1 <= len(rows) <= MAX_CUTS:
        raise ReplayError("Cut plan exceeds its frame-clock contract")
    result: list[tuple[int, int, int]] = []
    source_end = 0
    output_end = 0
    for raw in rows:
        row = record(raw)
        start = row.get("source_start_frame")
        end = row.get("source_end_frame")
        output = row.get("output_start_frame")
        if (
            type(start) is not int
            or type(end) is not int
            or type(output) is not int
            or not source_end <= start < end
            or output != output_end
        ):
            raise ReplayError("Cut plan is unordered or its output clock has a gap")
        result.append((output, output + end - start, len(result)))
        source_end = end
        output_end += end - start
    return tuple(result)


def _owner(start: float, end: float, cuts: tuple[tuple[int, int, int], ...]) -> tuple[int, float]:
    first = math.floor(start * 30 + 1e-9)
    for output_start, output_end, index in cuts:
        if output_start <= first < output_end:
            return index, min(end, output_end / 30)
    raise ReplayError("Caption onset is outside the retained output clock")


def _input_tokens(
    value: dict[str, object], cuts: tuple[tuple[int, int, int], ...]
) -> tuple[tuple[SpokenToken, ...], int]:
    rows = value.get("cues")
    if value.get("version") != 1 or value.get("clock") != "output":
        raise ReplayError("Caption cues must use the output clock")
    if not isinstance(rows, list) or not 1 <= len(rows) <= MAX_CUES:
        raise ReplayError("Caption cue count is outside the supported bound")
    result: list[SpokenToken] = []
    previous = 0.0
    clipped = 0
    for raw in rows:
        row = record(raw)
        text, start, end = row.get("text"), row.get("start"), row.get("end")
        if (
            not isinstance(text, str)
            or not text.strip()
            or "\n" in text
            or "\r" in text
            or isinstance(start, bool)
            or not isinstance(start, (int, float))
            or isinstance(end, bool)
            or not isinstance(end, (int, float))
            or not math.isfinite(start)
            or not math.isfinite(end)
            or not 0 <= start < end
            or start < previous
        ):
            raise ReplayError("Caption cue order, text or interval is invalid")
        owner, display_end = _owner(float(start), float(end), cuts)
        if display_end < float(end):
            clipped += 1
        if display_end <= float(start):
            raise ReplayError("Caption has no support before its retained cut boundary")
        result.append(SpokenToken(text, float(start), display_end, f"output-cut:{owner}"))
        previous = float(start)
    return tuple(result), clipped


def read_output_tokens(
    cue_path: Path, cut_path: Path
) -> tuple[tuple[SpokenToken, ...], tuple[tuple[int, int, int], ...], int]:
    """Validate shared output-clock recognition and retained-cut ownership."""
    _, cue_doc = _read(cue_path)
    _, cut_doc = _read(cut_path)
    cuts = _cuts(cut_doc)
    tokens, clipped = _input_tokens(cue_doc, cuts)
    return tokens, cuts, clipped


def read_retained_cuts(cut_path: Path) -> tuple[tuple[int, int, int], ...]:
    """Expose the validated source-to-output frame ownership boundary."""
    _, document = _read(cut_path)
    return _cuts(document)


def reflow(
    cue_path: Path,
    cut_path: Path,
    base_bundle_path: Path,
    output_path: Path,
) -> dict[str, object]:
    """Write a pinned candidate title bundle using morphology and real cue clocks."""
    cue_bytes, cue_doc = _read(cue_path)
    cut_bytes, cut_doc = _read(cut_path)
    bundle_bytes, bundle = _read(base_bundle_path)
    cuts = _cuts(cut_doc)
    tokens, clipped_cues = _input_tokens(cue_doc, cuts)
    selected = CaptionPolicy(
        maximum_characters=36,
        maximum_lines=2,
        maximum_gap=0.25,
        maximum_context_seconds=3.0,
        linguistic_boundaries=True,
        complete_phrases=False,
        frame_rate=30,
    )
    captions = make_captions(tokens, policy=selected)
    if not captions:
        raise ReplayError("POS caption planning produced no display cues")
    if any(
        not math.isfinite(cue.start)
        or not math.isfinite(cue.end)
        or not 0 <= cue.start < cue.end
        or (index > 0 and captions[index - 1].start > cue.start)
        for index, cue in enumerate(captions)
    ):
        raise ReplayError("POS caption planning produced invalid clocks")

    canvas = record(bundle.get("canvas"))
    slot = record(bundle.get("slot"))
    width, height, rate = canvas.get("width"), canvas.get("height"), canvas.get("rate")
    center, left, right, slot_width = (
        slot.get("centerX"),
        slot.get("left"),
        slot.get("right"),
        slot.get("width"),
    )
    if any(type(item) is not int or item <= 0 for item in (width, height, rate)) or rate != 30:
        raise ReplayError("Base title bundle has invalid canvas, safe slot or style")
    if any(
        isinstance(item, bool) or not isinstance(item, (float, int))
        for item in (center, left, right, slot_width)
    ):
        raise ReplayError("Base title bundle has invalid canvas, safe slot or style")
    if not isinstance(bundle.get("fontSize"), (float, int)):
        raise ReplayError("Base title bundle has invalid canvas, safe slot or style")
    assert type(width) is int and type(height) is int and type(rate) is int
    assert isinstance(center, (float, int)) and not isinstance(center, bool)
    assert isinstance(left, (float, int)) and not isinstance(left, bool)
    assert isinstance(right, (float, int)) and not isinstance(right, bool)
    assert isinstance(slot_width, (float, int)) and not isinstance(slot_width, bool)
    if not left < center < right <= width or not math.isclose(
        right - left, slot_width, rel_tol=0, abs_tol=1e-6
    ):
        raise ReplayError("Base title bundle has invalid canvas, safe slot or style")

    total_frames = cuts[-1][1]
    placements: list[dict[str, object]] = []
    for caption in captions:
        start = round(caption.start * rate)
        end = max(start + 1, round(caption.end * rate))
        if not 0 <= start < end <= total_frames:
            raise ReplayError("POS caption does not fit the retained output duration")
        placements.append({"text": caption.text, "startFrame": start, "endFrame": end})

    parts: list[dict[str, object]] = []
    pending: list[dict[str, object]] = []
    part_start = 0

    def flush() -> None:
        nonlocal pending, part_start
        if not pending:
            return
        local: list[dict[str, object]] = []
        for placement in pending:
            start_frame = placement["startFrame"]
            end_frame_value = placement["endFrame"]
            if type(start_frame) is not int or type(end_frame_value) is not int:
                raise ReplayError("Invalid normalized POS caption frame")
            local.append(
                {
                    **placement,
                    "startFrame": start_frame - part_start,
                    "endFrame": end_frame_value - part_start,
                    "x": center,
                    "y": height * 8 / 9,
                }
            )
        end_frame = max(
            int(placement["endFrame"]) for placement in local if type(placement["endFrame"]) is int
        )
        parts.append(
            {
                "version": 1,
                "width": width,
                "height": height,
                "rate": rate,
                "frames": end_frame,
                "offsetSeconds": part_start / rate,
                "offsetFrame": part_start,
                "expectedText": "Title",
                "placements": local,
            }
        )
        pending = []

    for placement in placements:
        frame_start = placement["startFrame"]
        if type(frame_start) is not int:
            raise ReplayError("Invalid POS caption start frame")
        if pending and (
            len(pending) >= MAX_TITLE_CUES or frame_start - part_start >= MAX_TITLE_SECONDS * rate
        ):
            flush()
        if not pending:
            part_start = frame_start
        pending.append(placement)
    flush()
    output_path = output_path.absolute()
    evidence_path = output_path.with_suffix(".evidence.json")
    if (
        output_path.resolve() != output_path
        or output_path.is_symlink()
        or output_path.exists()
        or evidence_path.is_symlink()
        or evidence_path.exists()
    ):
        raise ReplayError("Refusing to overwrite or follow the POS plan output")
    output_path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    candidate: dict[str, object] = {
        "version": 1,
        "canvas": canvas,
        "slot": slot,
        "fontSize": bundle["fontSize"],
        "parts": len(parts),
        "placements": len(placements),
        "titleParts": parts,
    }
    evidence: dict[str, object] = {
        "version": 1,
        "candidateOnly": True,
        "clock": "output",
        "sourceEvidence": {
            "cueSHA256": hashlib.sha256(cue_bytes).hexdigest(),
            "cutPlanSHA256": hashlib.sha256(cut_bytes).hexdigest(),
            "baseBundleSHA256": hashlib.sha256(bundle_bytes).hexdigest(),
            "sourceCueCount": len(tokens),
            "posCaptionCount": len(captions),
            "displayEndsClippedAtCut": clipped_cues,
            "retainedOutputFrames": total_frames,
        },
        "nativeImportVerified": False,
        "compositeVerified": False,
        "deliveryComplete": False,
    }
    with output_path.open("x", encoding="utf-8") as stream:
        json.dump(candidate, stream, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    with evidence_path.open("x", encoding="utf-8") as stream:
        json.dump(evidence, stream, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    return candidate


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("cues", type=Path)
    parser.add_argument("cuts", type=Path)
    parser.add_argument("base_bundle", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args(argv)
    reflow(args.cues, args.cuts, args.base_bundle, args.output)
    evidence_path = args.output.with_suffix(".evidence.json")
    evidence = record(json.loads(evidence_path.read_text(encoding="utf-8")))
    print(json.dumps(evidence["sourceEvidence"], ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
