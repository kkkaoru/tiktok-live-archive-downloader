"""Build plan-bound alpha constraints from independent native layout measurements.

This does not read the alpha movie or infer font/shadow calibration from it.
Calibration must already bind the selected template and native font metrics;
changed outline/shadow conversions require fresh independent calibration.
"""

import argparse
import sys
from pathlib import Path

from replay import delivery_caption_audit as audit
from replay.core import ReplayError
from replay.delivery_render import load_config

GEOMETRY_KEYS = frozenset(
    {
        "region",
        "centerX",
        "centerTolerancePixels",
        "clearance",
        "foregroundTop",
        "faceTop",
        "headTop",
    }
)


def build_constraints(
    config_path: Path,
    plan_path: Path,
    metrics_path: Path,
    geometry_path: Path,
    output: Path,
    *,
    name: str = "calibration",
) -> dict[str, object]:
    """Use retained native request/response pairs, never target-alpha observed bounds."""
    if not name:
        raise ReplayError("Constraint part name is required")
    inputs = audit.Inputs()
    config = load_config(audit.local(config_path))
    inputs.bind(config.path)
    plan_path = audit.local(plan_path)
    plan = audit.document(plan_path, inputs)
    if (plan.get("width"), plan.get("height"), plan.get("rate")) != (
        config.width,
        config.height,
        config.rate,
    ):
        raise ReplayError("Constraint plan canvas/clock mismatch")
    frames = audit.integer(plan.get("frames"), minimum=1)
    if frames > audit.MAX_FRAMES:
        raise ReplayError("Constraint frame budget exceeded")
    cues = tuple(audit.cue(row, frames) for row in audit.rows(plan.get("placements")))
    audit.ordered(cues)
    calibration = audit.document(config.calibration, inputs)
    if inputs.reference(calibration.get("template")) != config.template:
        raise ReplayError("Calibration selects another template")
    evidence = audit.rows(calibration.get("evidence"))
    if not evidence:
        raise ReplayError("Independent native calibration evidence is required")
    for reference in evidence:
        inputs.reference(reference)
    metrics = audit.document(metrics_path, inputs)
    envelopes = audit.measured_envelopes(metrics, config.calibration, inputs)
    geometry = audit.document(geometry_path, inputs)
    regions = audit.rows(geometry.get("states"))
    if not cues or len(cues) != len(regions) or inputs.missing:
        raise ReplayError("Incomplete independent metrics or per-state geometry")
    states: list[dict[str, object]] = []
    for cue, region in zip(cues, regions, strict=True):
        if set(region) != GEOMETRY_KEYS or cue.text not in envelopes:
            raise ReplayError("Missing native text metrics or invalid geometry fields")
        audit.box(region["region"])
        for key in GEOMETRY_KEYS - {"region"}:
            value = audit.number(region[key])
            if key in {"clearance", "centerTolerancePixels"} and value < 0:
                raise ReplayError("Negative geometric constraint")
        left, top, right, bottom = envelopes[cue.text]
        x, y = cue.x + config.width / 2, config.height / 2 - cue.y
        states.append(
            {
                **region,
                "startFrame": cue.start,
                "endFrame": cue.end,
                "text": cue.text,
                "envelope": [left + x, top + y, right + x, bottom + y],
            }
        )
    inputs.bind(Path(__file__))
    inputs.bind(Path(audit.__file__))
    result: dict[str, object] = {
        "version": 1,
        "kind": "delivery-caption-constraints",
        "parts": {
            name: {"plan": inputs.bind(plan_path), "inputs": inputs.rows(), "states": states},
        },
    }
    inputs.verify()
    audit.save(output, result)
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path)
    parser.add_argument("plan", type=Path)
    parser.add_argument("metrics", type=Path)
    parser.add_argument("geometry", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--name", default="calibration")
    args = parser.parse_args(argv)
    try:
        build_constraints(
            args.config, args.plan, args.metrics, args.geometry, args.output, name=args.name
        )
    except audit.ERRORS as error:
        print(f"CAPTION_CONSTRAINTS_ERROR {error}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
