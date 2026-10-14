"""Synthetic rejection checks for independent alpha constraint provenance."""

import json
from pathlib import Path

import pytest
from test_caption_probe_constraints import geometry as geometry
from test_delivery_caption_audit import config as config
from test_delivery_caption_audit import constraints as constraints
from test_delivery_caption_audit import part as part
from test_delivery_caption_audit import write_json

from replay import caption_probe_constraints as builder
from replay import delivery_caption_audit as audit
from replay.core import ReplayError
from replay.delivery_render import Config


@pytest.mark.parametrize(
    ("field", "value"),
    [("headTop", None), ("faceTop", None), ("centerTolerancePixels", -1)],
)
def test_unknown_geometry_is_not_approval(
    config: Config, part: audit.Part, geometry: Path, field: str, value: object
) -> None:
    data = audit.document(geometry, audit.Inputs())
    audit.rows(data["states"])[0][field] = value
    write_json(geometry, data)
    with pytest.raises(ReplayError):
        builder.build_constraints(
            config.path,
            part.plan,
            config.captions / "combined-0000.json",
            geometry,
            config.work / "rejected.json",
        )
    assert not (config.work / "rejected.json").exists()


def test_missing_head_is_not_foreground_fallback(
    config: Config, part: audit.Part, geometry: Path
) -> None:
    data = audit.document(geometry, audit.Inputs())
    del audit.rows(data["states"])[0]["headTop"]
    write_json(geometry, data)
    with pytest.raises(ReplayError, match="invalid geometry fields"):
        builder.build_constraints(
            config.path,
            part.plan,
            config.captions / "combined-0000.json",
            geometry,
            config.work / "rejected.json",
        )


@pytest.mark.parametrize("changed", ["plan", "policy"])
def test_consumer_rejects_stale_plan_or_changed_policy_pin(
    config: Config, part: audit.Part, geometry: Path, changed: str
) -> None:
    output = config.work / "constraints.json"
    builder.build_constraints(
        config.path,
        part.plan,
        config.captions / "combined-0000.json",
        geometry,
        output,
        name=part.name,
    )
    path = part.plan if changed == "plan" else geometry
    path.write_text(path.read_text(encoding="utf-8") + "\n", encoding="utf-8")
    with pytest.raises(ReplayError, match="fingerprint mismatch"):
        audit.constraints_for(output, part, audit.Inputs())


def test_constraints_preserve_local_clocks_and_global_offset(
    config: Config, part: audit.Part, geometry: Path
) -> None:
    output = config.work / "constraints.json"
    builder.build_constraints(
        config.path,
        part.plan,
        config.captions / "combined-0000.json",
        geometry,
        output,
        name=part.name,
    )
    states = audit.constraints_for(output, part, audit.Inputs())
    assert part.offset == 1
    assert [(s["startFrame"], s["endFrame"]) for s in states] == [(0, 2), (3, 5)]
    assert sum(c.end - c.start for c in part.cues) == 4
    manifest = json.loads((part.plan.parent / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["frames"] == 7
    assert audit.canonical_srt([part], 30).startswith("1\n00:00:00,033 --> 00:00:00,100\n")
