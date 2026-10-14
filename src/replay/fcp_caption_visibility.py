"""Bind explicit candidate visibility to final head paging and exact XML clocks."""

import argparse
import json
from dataclasses import dataclass
from fractions import Fraction
from math import floor
from pathlib import Path

from replay.artifact_json import _pin
from replay.caption_io import _array, _number
from replay.caption_measured_job import _object
from replay.caption_timing import display_time
from replay.caption_visibility_candidate import MAX_TARGETS, _read, clip_holds, plan
from replay.fcp_caption_srt import Cue, caption_cues
from replay.fcp_project import frame_span
from replay.input_hash_snapshot import InputHashSnapshot
from replay.recordings import record
from replay.stages import file_digest


@dataclass(frozen=True, slots=True)
class VisibilityBinding:
    """Verified overlay and recognized output-clock exclusions, never accuracy."""

    regions: tuple[tuple[float, float], ...]
    inputs: dict[str, str]
    proof: dict[str, object]


def read_binding(value: object, *, reflow_inputs: dict[str, object]) -> VisibilityBinding:
    """Require the exact explicit overlay pinned by the native paging producer."""
    raw = _object(value, "config overlay")
    config, overlay = (_pin(raw[name]) for name in ("config", "overlay"))
    inputs = {
        str(path): file_digest(path)
        for path in (config, overlay, Path(__file__), Path(__file__).with_name("fcp_project.py"))
    }
    snapshot = InputHashSnapshot.capture([(Path(name), digest) for name, digest in inputs.items()])
    if reflow_inputs.get(str(overlay)) != inputs[str(overlay)]:
        raise ValueError("Head reflow must pin the explicit visibility overlay")
    document = _read(overlay)
    if document != plan(config):
        raise ValueError("Head visibility overlay differs from its recognized proof")
    proof = record(document.get("candidateVisibility"))
    producer = _pin(document.get("visibilityProducer"))
    inputs[str(producer)] = file_digest(producer)
    original = _pin(proof.get("original"))
    dependencies = record(proof.get("dependencies"))
    inputs[str(original)] = file_digest(original)
    for name, digest in dependencies.items():
        path = _pin({"path": name, "sha256": digest})
        inputs[str(path)] = file_digest(path)
    rows = _array(proof.get("regions"), MAX_TARGETS)
    regions = tuple(
        (_number(record(row).get("start")), _number(record(row).get("end"))) for row in rows
    )
    snapshot.finish()
    return VisibilityBinding(regions, inputs, proof)


def apply_binding(
    captions: list[dict[str, object]], binding: VisibilityBinding, *, rate: int = 30
) -> list[dict[str, object]]:
    """Keep final display frame cells outside exclusions; leave raw clocks intact.

    Exact Fraction ends are transient writer inputs, not replacement recognition
    timestamps. The shared FCP writer otherwise ceils both ends to whole frames.
    """
    if type(rate) is not int or not 1 <= rate <= 120:
        raise ValueError("Expected a bounded integer display frame rate")
    cutoffs = {
        display_time(start): Fraction(floor(display_time(start) * rate), rate)
        for start, _ in binding.regions
    }
    result = clip_holds(list(captions), binding.regions)
    for row in result:
        cutoff = cutoffs.get(display_time(_number(row.get("end"))))
        if cutoff is None:
            continue
        start = Fraction(str(_number(row.get("start"))))
        if frame_span(start, cutoff, rate) is None:
            raise ValueError("Explicit visibility leaves no complete display frame for caption")
        row["end"] = cutoff
    return result


def interval_audit(cues: tuple[Cue, ...], binding: VisibilityBinding) -> dict[str, object]:
    """Check exact serialized display intervals after the shared clock conversion."""
    regions = tuple((display_time(start), display_time(end)) for start, end in binding.regions)
    if any(start < right and end > left for start, end, _ in cues for left, right in regions):
        raise ValueError("FCP title display intersects explicitly hidden recognition")
    return {
        "version": 1,
        "candidateOnly": True,
        "titleIntervals": len(cues),
        "hiddenRegionCount": len(regions),
        "displayRegions": [{"start": str(start), "end": str(end)} for start, end in regions],
        "displayClock": "existing-60000-timescale-recognized-output-conversion",
        "displayEndPolicy": "complete-frame-cells-before-explicit-recognized-exclusion",
        "recognizedClocksChanged": False,
        "visibleTitleIntervalsVerified": True,
        "nativeRenderedCaptionVisibilityVerified": False,
        "completeCaptionsVerified": False,
        "speechAccuracyVerified": False,
        "deliveryComplete": False,
        "proof": binding.proof,
    }


def audit_xml(path: Path, binding: VisibilityBinding) -> dict[str, object]:
    """Independently parse full XML title/source clock composition before acceptance."""
    digest = file_digest(path)
    result = interval_audit(caption_cues(path), binding)
    if file_digest(path) != digest:
        raise ValueError("FCPXML changed during candidate visibility audit")
    return {**result, "semanticXML": {"path": str(path), "sha256": digest}}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("binding", type=Path)
    parser.add_argument("reflow", type=Path)
    parser.add_argument("xml", type=Path)
    arguments = parser.parse_args(argv)
    binding = read_binding(
        _read(arguments.binding), reflow_inputs=record(_read(arguments.reflow).get("inputs"))
    )
    print(json.dumps(audit_xml(arguments.xml, binding), ensure_ascii=False, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
