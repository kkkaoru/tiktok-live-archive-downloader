"""Synthetic evidence only: no apps, decode, recordings or native process."""

import json
import sys
from dataclasses import replace
from fractions import Fraction
from pathlib import Path

import pytest
from test_motion_caption import sequence_template as sequence_template
from test_motion_caption import text_template as text_template

from replay import motion_calibration_verify as audit
from replay import motion_finish, motion_native_envelope, motion_stages
from replay.core import ReplayError
from replay.recordings import record
from replay.stages import StageError, _fingerprint, file_digest
from replay.workflow import load_stages


@pytest.fixture
def raw_plan() -> dict[str, object]:
    return {
        "frames": 2,
        "width": 1080,
        "height": 1920,
        "rate": 30,
        "expectedText": "Title",
        "placements": [{"text": "A", "startFrame": 0, "endFrame": 2, "x": 10, "y": 20}],
    }


@pytest.fixture
def font() -> dict[str, object]:
    return {
        "postScriptName": "NotoSansJP-Thin_Bold",
        "family": "Synthetic",
        "face": "Bold",
        "pointSize": 80,
        "lineAdvance": 120,
        "lineAscent": 90,
        "renderedAscent": 90,
    }


@pytest.fixture
def calibration(font: dict[str, object]) -> dict[str, object]:
    return {
        "version": 1,
        "canvas": [1080, 1920],
        "padding": 24,
        "nativeFont": font,
        "style": {
            "font_name": "NotoSansJP-Thin_Bold",
            "ass_font_size": 120,
            "border": 16,
            "rim": 5,
            "motion_line_advance": 121,
        },
    }


@pytest.fixture
def bounds() -> str:
    return (
        "frame:0 pts:0 pts_time:0\nlavfi.bbox.x1=530\nlavfi.bbox.x2=570\n"
        "lavfi.bbox.y1=850\nlavfi.bbox.y2=950\nlavfi.bbox.w=41\nlavfi.bbox.h=101\n"
        "frame:1 pts:1000 pts_time:0.0333333\nlavfi.bbox.x1=530\nlavfi.bbox.x2=570\n"
        "lavfi.bbox.y1=850\nlavfi.bbox.y2=950\nlavfi.bbox.w=41\nlavfi.bbox.h=101\n"
    )


@pytest.fixture
def md5() -> str:
    return (
        "#tb 0: 1/30\n#dimensions 0: 1080x1920\n"
        "0,0,0,1,100,aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa\n"
        "0,1,1,1,100,bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb\n"
    )


@pytest.fixture
def probe() -> dict[str, object]:
    return {
        "streams": [
            {
                "codec_type": "video",
                "width": 1080,
                "height": 1920,
                "start_pts": 0,
                "nb_frames": "2",
                "nb_read_frames": "2",
                "avg_frame_rate": "30/1",
                "duration_ts": 2000,
                "time_base": "1/30000",
            }
        ]
    }


def save(path: Path, value: object) -> None:
    path.write_text(json.dumps(value), encoding="utf-8")


def workflow_receipt(workflow: Path, state: Path) -> None:
    state.mkdir()
    stage = load_stages(workflow)[0]
    save(
        state / f"{stage.name}.json",
        {
            "status": "complete",
            "fingerprint": _fingerprint(stage),
            "outputs": {str(p): file_digest(p) for p in stage.outputs},
        },
    )


@pytest.fixture
def config(
    tmp_path: Path,
    sequence_template: str,
    raw_plan: dict[str, object],
    calibration: dict[str, object],
    font: dict[str, object],
    bounds: str,
    md5: str,
    probe: dict[str, object],
) -> Path:
    paths = {
        key: str(tmp_path / key)
        for key in (
            "plan",
            "template",
            "twoLayer",
            "calibration",
            "nativeRequest",
            "nativeResponse",
            "paddedRequest",
            "paddedResponse",
            "probe",
            "framemd5",
            "bounds",
            "alpha",
            "renderReceipt",
            "prepareWorkflow",
            "prepareState",
            "verifyWorkflow",
            "verifyState",
        )
    }
    save(tmp_path / "plan", raw_plan)
    two_layer = motion_stages.prepare(sequence_template, raw_plan)
    (tmp_path / "twoLayer").write_text(two_layer, encoding="utf-8")
    style = motion_finish.selected_finish_style(
        top=(0.1, 0.2, 0.3),
        bottom=(0.4, 0.5, 0.6),
        font_name="NotoSansJP-Thin_Bold",
        font_size=80.0,
        stroke_scale=0.5,
        shadow_scale=1,
    )
    (tmp_path / "template").write_text(
        motion_finish.finish_sequence(two_layer, style=style), encoding="utf-8"
    )
    request = {
        "fontName": "NotoSansJP-Thin_Bold",
        "assFontSize": 120,
        "border": 0,
        "rim": 0,
        "texts": ["A"],
    }
    response = {
        "motionRenderVerified": False,
        "font": font,
        "bounds": [{"text": "A", "width": 50, "height": 130, "lineWidths": [50]}],
    }
    save(tmp_path / "nativeRequest", request)
    save(tmp_path / "nativeResponse", response)
    save(tmp_path / "paddedRequest", request | {"border": 16, "rim": 5})
    save(tmp_path / "paddedResponse", response)
    calibration["evidence"] = [
        {"path": paths[key], "sha256": file_digest(Path(paths[key]))}
        for key in ("nativeRequest", "nativeResponse", "paddedRequest", "paddedResponse", "plan")
    ]
    save(tmp_path / "calibration", calibration)
    save(tmp_path / "probe", probe)
    (tmp_path / "framemd5").write_text(md5, encoding="utf-8")
    (tmp_path / "bounds").write_text(bounds, encoding="utf-8")
    (tmp_path / "alpha").write_bytes(b"synthetic movie identity; not a media fixture")
    save(
        tmp_path / "renderReceipt",
        {
            "status": "Successful",
            "terminalInputsVerified": True,
            "output": paths["alpha"],
            "inputs": [{"path": paths["template"], "sha256": file_digest(tmp_path / "template")}],
        },
    )
    save(
        tmp_path / "prepareWorkflow",
        {
            "version": 1,
            "stages": [
                {
                    "name": "prepare",
                    "command": ["synthetic"],
                    "inputs": [paths["plan"], paths["twoLayer"]],
                    "outputs": [paths["template"]],
                    "timeout_seconds": 1,
                }
            ],
        },
    )
    save(
        tmp_path / "verifyWorkflow",
        {
            "version": 1,
            "stages": [
                {
                    "name": "observe",
                    "command": ["synthetic"],
                    "inputs": [paths["alpha"]],
                    "outputs": [paths["probe"], paths["framemd5"], paths["bounds"]],
                    "timeout_seconds": 1,
                }
            ],
        },
    )
    workflow_receipt(tmp_path / "prepareWorkflow", tmp_path / "prepareState")
    workflow_receipt(tmp_path / "verifyWorkflow", tmp_path / "verifyState")
    result = tmp_path / "config.json"
    save(
        result,
        {
            "paths": paths,
            "finish": {
                "top": [0.1, 0.2, 0.3],
                "bottom": [0.4, 0.5, 0.6],
                "strokeScale": 0.5,
                "shadowScale": 1,
            },
        },
    )
    return result


def test_cli_complete_geometry_only(
    config: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    output = config.parent / "receipt.json"
    monkeypatch.setattr(sys, "argv", ["audit", str(config), str(output)])
    audit.main()
    result = record(json.loads(output.read_text(encoding="utf-8")))
    assert result["states"] == [
        {
            "text": "A",
            "startFrame": 0,
            "endFrame": 2,
            "framesChecked": 2,
            "expectedEnvelope": [525, 826, 575, 956],
            "observedBounds": [[530, 850, 571, 951]],
            "contained": True,
        }
    ]
    assert result["scope"] == "geometry-only"
    assert result["newCalibrationIssued"] is False
    assert result["deliveryComplete"] is False
    assert '"geometryContained": true' in capsys.readouterr().out
    with pytest.raises(FileExistsError):
        audit.verify(config, output)


@pytest.mark.parametrize(
    "key,value",
    [
        ("frames", 0),
        ("rate", 0),
        ("width", 0),
        ("placements", []),
        ("extra", 1),
        ("frames", True),
        ("placements", [{"text": "A", "startFrame": 0, "endFrame": 3, "x": 0, "y": 0}]),
    ],
)
def test_invalid_plan(raw_plan: dict[str, object], key: str, value: object) -> None:
    raw_plan[key] = value
    with pytest.raises((ValueError, ReplayError)):
        audit.parse_plan(raw_plan)


@pytest.mark.parametrize(
    "old,new",
    [
        ("frame:1", "frame:0"),
        ("pts:1000", "pts:1001"),
        ("pts_time:0.0333333", "pts_time:0.2"),
        ("lavfi.bbox.x1=530", "lavfi.bbox.z=530"),
        ("lavfi.bbox.x2=570", "lavfi.bbox.x1=570"),
        ("lavfi.bbox.w=41", "lavfi.bbox.w=40"),
        ("lavfi.bbox.y2=950", "lavfi.bbox.y2=1920"),
        ("frame:1 pts:1000 pts_time:0.0333333\n", ""),
    ],
)
def test_invalid_bounds(raw_plan: dict[str, object], bounds: str, old: str, new: str) -> None:
    with pytest.raises(ValueError):
        audit.parse_bounds(
            bounds.replace(old, new), plan=audit.parse_plan(raw_plan), base=Fraction(1, 30000)
        )


@pytest.mark.parametrize(
    "change", ["missing", "extra", "duplicate", "hash", "timebase", "dimensions"]
)
def test_bad_md5(
    raw_plan: dict[str, object], probe: dict[str, object], md5: str, change: str
) -> None:
    mutations = {
        "missing": md5.rsplit("\n", 2)[0],
        "extra": md5 + md5.splitlines()[-1],
        "duplicate": md5.replace("0,1,1,1", "0,0,0,1"),
        "hash": md5.replace("aaaaaaaa", "zzzzzzzz"),
        "timebase": md5.replace("1/30", "1/60"),
        "dimensions": md5.replace("1080x1920", "1920x1080"),
    }
    with pytest.raises(ValueError):
        audit.check_clocks(probe, mutations[change], audit.parse_plan(raw_plan))


@pytest.mark.parametrize(
    "probe", [{}, {"streams": []}, {"streams": [{"codec_type": "video", "time_base": "1/30"}]}]
)
def test_bad_probe(raw_plan: dict[str, object], probe: dict[str, object], md5: str) -> None:
    with pytest.raises(ValueError):
        audit.check_clocks(probe, md5, audit.parse_plan(raw_plan))


def test_multiline_and_containment_failure(
    raw_plan: dict[str, object], calibration: dict[str, object]
) -> None:
    plan = audit.parse_plan(raw_plan)
    plan = replace(plan, states=(replace(plan.states[0], text="A\nB"),))
    result = audit.compare_geometry(
        plan=plan,
        bounds=[(525, 826, 575, 1077), (525, 826, 576, 1077)],
        sizes={"A": (50, 130), "B": (40, 100)},
        calibration=calibration,
    )
    assert result == [
        {
            "text": "A\nB",
            "startFrame": 0,
            "endFrame": 2,
            "framesChecked": 2,
            "expectedEnvelope": (525, 826, 575, 1077),
            "observedBounds": [(525, 826, 575, 1077), (525, 826, 576, 1077)],
            "contained": False,
        }
    ]


def test_missing_metric_gap_and_padding(
    raw_plan: dict[str, object], calibration: dict[str, object]
) -> None:
    plan = audit.parse_plan(raw_plan)
    with pytest.raises(ValueError, match="metrics missing"):
        audit.compare_geometry(
            plan=plan, bounds=[(0, 0, 1, 1)] * 2, sizes={}, calibration=calibration
        )
    with pytest.raises(ValueError, match="full state coverage"):
        audit.compare_geometry(
            plan=replace(plan, frames=3),
            bounds=[(0, 0, 1, 1)] * 3,
            sizes={"A": (50, 130)},
            calibration=calibration,
        )
    calibration["padding"] = 65
    with pytest.raises(ValueError, match="padding/canvas"):
        audit.compare_geometry(plan=plan, bounds=[], sizes={}, calibration=calibration)


@pytest.mark.parametrize(
    "native_request,response",
    [
        ({}, {}),
        (
            {"texts": ["A"]},
            {"motionRenderVerified": False, "bounds": [{"text": "B", "lineWidths": [2]}]},
        ),
        (
            {"texts": ["A"]},
            {
                "motionRenderVerified": False,
                "bounds": [{"text": "A", "width": 0, "height": 1, "lineWidths": [0]}],
            },
        ),
    ],
)
def test_invalid_metrics(native_request: dict[str, object], response: dict[str, object]) -> None:
    with pytest.raises((ValueError, ReplayError)):
        audit.native_sizes(native_request, response)


def test_changed_inputs_and_paths(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    inputs = audit.Inputs()
    with pytest.raises(ValueError):
        inputs.path(42)
    with pytest.raises(ValueError):
        inputs.path(str(Path.cwd().parent / "escape"))
    file = tmp_path / "input"
    file.write_text("before", encoding="utf-8")
    inputs.bind(file)
    file.write_text("after", encoding="utf-8")
    with pytest.raises(ValueError, match="changed"):
        inputs.verify()
    monkeypatch.setattr(audit, "MAX_BYTES", 1)
    with pytest.raises(ValueError, match="budget"):
        audit.Inputs().text(file)


@pytest.mark.parametrize(
    "target",
    [
        "template",
        "nativeResponse",
        "calibration",
        "renderReceipt",
        "verifyState/observe.json",
        "prepareWorkflow",
        "config.json",
    ],
)
def test_corrupt_evidence_fails(config: Path, target: str) -> None:
    path = config.parent / target
    if target == "template":
        path.write_text(
            path.read_text(encoding="utf-8").replace('value="80.0"', 'value="81.0"'),
            encoding="utf-8",
        )
    else:
        save(path, {})
    with pytest.raises((ValueError, ReplayError, StageError, KeyError)):
        audit.verify(config, config.parent / "rejected.json")
    assert not (config.parent / "rejected.json").exists()


def test_template_mismatch(
    config: Path, raw_plan: dict[str, object], font: dict[str, object]
) -> None:
    document = (config.parent / "template").read_text(encoding="utf-8")
    parsed_font = motion_native_envelope._font(font)
    with pytest.raises(ValueError, match="stage count"):
        audit.check_template(
            document.replace('factoryID="16"', 'factoryID="17"'),
            plan=audit.parse_plan(raw_plan),
            font=parsed_font,
        )
    with pytest.raises(ValueError, match="differs from plan"):
        audit.check_template(
            document.replace("<text>A</text>", "<text>B</text>"),
            plan=audit.parse_plan(raw_plan),
            font=parsed_font,
        )


@pytest.mark.parametrize(
    "target,field,value",
    [
        ("nativeRequest", "fontName", "Other-Font"),
        ("calibration", "evidence", []),
        ("calibration", "evidence", None),
        ("calibration", "version", True),
        (
            "paddedResponse",
            "bounds",
            [{"text": "A", "width": 10, "height": 10, "lineWidths": [10]}],
        ),
    ],
)
def test_reuse_rejects_changed_metrics(
    config: Path, target: str, field: str, value: object
) -> None:
    path = config.parent / target
    raw = record(json.loads(path.read_text(encoding="utf-8")))
    raw[field] = value
    save(path, raw)
    with pytest.raises(ValueError):
        audit.verify(config, config.parent / "rejected.json")


def test_workflow_requires_actual_sources(config: Path) -> None:
    with pytest.raises(ValueError, match="required source"):
        audit.bind_workflow(
            audit.Inputs(),
            workflow=config.parent / "verifyWorkflow",
            state=config.parent / "verifyState",
            outputs=(config.parent / "bounds",),
            required_inputs=(config.parent / "template",),
        )
    with pytest.raises(ValueError, match="unique evidence producer"):
        audit.bind_workflow(
            audit.Inputs(),
            workflow=config.parent / "verifyWorkflow",
            state=config.parent / "verifyState",
            outputs=(config.parent / "template",),
            required_inputs=(),
        )


@pytest.mark.parametrize("precision", [".6g", ".6f"])
def test_long_clock_print_precision(raw_plan: dict[str, object], precision: str) -> None:
    plan = replace(audit.parse_plan(raw_plan), frames=3600)
    metadata = "".join(
        f"frame:{index} pts:{index * 1000} pts_time:{format(index / 30, precision)}\n"
        "lavfi.bbox.x1=0\nlavfi.bbox.x2=1\nlavfi.bbox.y1=0\n"
        "lavfi.bbox.y2=1\nlavfi.bbox.w=2\nlavfi.bbox.h=2\n"
        for index in range(3600)
    )
    result = audit.parse_bounds(metadata, plan=plan, base=Fraction(1, 30000))
    assert len(result) == 3600
    assert result[-1] == (0, 0, 2, 2)


def test_failed_geometry_cli(config: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    path = config.parent / "bounds"
    path.write_text(
        path.read_text(encoding="utf-8").replace("x1=530", "x1=520").replace("w=41", "w=51"),
        encoding="utf-8",
    )
    stage = load_stages(config.parent / "verifyWorkflow")[0]
    save(
        config.parent / "verifyState/observe.json",
        {
            "status": "complete",
            "fingerprint": _fingerprint(stage),
            "outputs": {str(p): file_digest(p) for p in stage.outputs},
        },
    )
    monkeypatch.setattr(sys, "argv", ["audit", str(config), str(config.parent / "failed.json")])
    with pytest.raises(SystemExit, match="1"):
        audit.main()
