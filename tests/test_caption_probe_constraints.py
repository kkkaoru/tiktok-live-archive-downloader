"""Independent native-metric constraints; the target alpha is never an input."""

import json
from pathlib import Path

import pytest
from test_delivery_caption_audit import config as config
from test_delivery_caption_audit import constraints as constraints
from test_delivery_caption_audit import part as part
from test_delivery_caption_audit import ref, write_json

from replay import caption_probe_constraints as builder
from replay import delivery_caption_audit as audit
from replay.delivery_render import Config
from replay.recordings import record


@pytest.fixture
def geometry(
    config: Config, constraints: list[dict[str, object]], monkeypatch: pytest.MonkeyPatch
) -> Path:
    monkeypatch.setattr(builder, "load_config", lambda _: config)
    data = json.loads(config.calibration.read_text())
    data["template"] = ref(config.template)
    data["evidence"] = [ref(config.path)]
    write_json(config.calibration, data)
    return write_json(
        config.work / "geometry.json",
        {
            "states": [
                {key: value for key, value in row.items() if key in builder.GEOMETRY_KEYS}
                for row in constraints
            ]
        },
    )


def test_native_metrics_make_plan_bound_constraints(
    config: Config, part: audit.Part, geometry: Path
) -> None:
    output = config.work / "constraints-built.json"
    assert (
        builder.main(
            [
                str(config.path),
                str(part.plan),
                str(config.captions / "combined-0000.json"),
                str(geometry),
                str(output),
                "--name",
                part.name,
            ]
        )
        == 0
    )
    inputs = audit.Inputs()
    states = audit.constraints_for(output, part, inputs)
    assert states[0]["envelope"] == [1.0, 1.0, 3.0, 3.0]
    assert states[1]["text"] == "あい"
    inputs.verify()
    assert not (config.work / "cache").exists()


@pytest.mark.parametrize(
    "fault",
    [
        "empty-name",
        "canvas",
        "frames",
        "evidence",
        "template",
        "count",
        "field",
        "negative",
        "missing-metric",
    ],
)
def test_invalid_independent_inputs_fail(
    config: Config, part: audit.Part, geometry: Path, fault: str
) -> None:
    plan = json.loads(part.plan.read_text())
    calibration = json.loads(config.calibration.read_text())
    region = json.loads(geometry.read_text())
    if fault == "canvas":
        plan["width"] = 3
    elif fault == "frames":
        plan["frames"] = 7201
    elif fault == "evidence":
        calibration["evidence"] = []
    elif fault == "template":
        calibration["template"] = ref(part.plan)
    elif fault == "count":
        region["states"] = []
    elif fault == "field":
        region["states"][0]["unexpected"] = 1
    elif fault == "negative":
        region["states"][0]["clearance"] = -1
    elif fault == "missing-metric":
        data = json.loads((config.captions / "combined-0000.json").read_text())
        data["inputs"] = {}
        write_json(config.captions / "combined-0000.json", data)
    write_json(part.plan, plan)
    write_json(config.calibration, calibration)
    write_json(geometry, region)
    assert (
        builder.main(
            [
                str(config.path),
                str(part.plan),
                str(config.captions / "combined-0000.json"),
                str(geometry),
                str(config.work / "bad.json"),
                "--name",
                "" if fault == "empty-name" else part.name,
            ]
        )
        == 2
    )


def test_constraints_shift_with_plan_not_observed_alpha(
    config: Config, part: audit.Part, geometry: Path
) -> None:
    plan = json.loads(part.plan.read_text())
    plan["placements"][0]["x"] = 1
    write_json(part.plan, plan)
    result = builder.build_constraints(
        config.path,
        part.plan,
        config.captions / "combined-0000.json",
        geometry,
        config.work / "shifted.json",
    )
    states = audit.rows(record(record(result["parts"])["calibration"])["states"])
    assert states[0]["envelope"] == [2.0, 1.0, 4.0, 3.0]
