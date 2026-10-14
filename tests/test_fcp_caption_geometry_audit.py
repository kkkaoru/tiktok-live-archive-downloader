import json
import sys
from pathlib import Path
from xml.etree import ElementTree as ET

import pytest

from replay import fcp_caption_geometry_audit as audit
from replay.core import ReplayError
from replay.stages import file_digest


@pytest.fixture
def config(tmp_path: Path) -> Path:
    before = tmp_path / "before.fcpxml"
    after = tmp_path / "after.fcpxml"
    text = (
        '<fcpxml><library><event><project name="old"><sequence duration="2s"><spine>'
        '<asset-clip offset="0s" start="10s" duration="1s">'
        '<title offset="10s" duration="1s"><text>甲</text><adjust-transform position="0 10"/>'
        '</title></asset-clip><asset-clip offset="1s" start="20s" duration="1s">'
        '<title offset="20s" duration="1s"><text>乙\n丙</text><adjust-transform position="0 20"/>'
        "</title></asset-clip></spine></sequence></project></event></library></fcpxml>"
    )
    before.write_text(text, encoding="utf-8")
    after.write_text(
        text.replace('name="old"', 'name="new"').replace("0 10", "0 11"), encoding="utf-8"
    )
    path = tmp_path / "config.json"
    path.write_text(
        json.dumps(
            {
                "version": 1,
                "before": {"path": str(before), "sha256": file_digest(before)},
                "after": {"path": str(after), "sha256": file_digest(after)},
            }
        ),
        encoding="utf-8",
    )
    return path


def test_cli_exact_cue_changes_and_optional_partition(
    config: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    data = json.loads(config.read_text(encoding="utf-8"))
    manifest = tmp_path / "partition.json"
    manifest.write_text(
        json.dumps(
            {
                "source": data["before"],
                "parts": [
                    {**data["before"], "globalStart": "0s", "duration": "1s"},
                    {**data["after"], "globalStart": "1s", "duration": "1s"},
                ],
            }
        ),
        encoding="utf-8",
    )
    data["partition"] = {"path": str(manifest), "sha256": file_digest(manifest)}
    config.write_text(json.dumps(data), encoding="utf-8")
    output = tmp_path / "report.json"
    monkeypatch.setattr(sys, "argv", ["audit", str(config), str(output)])
    audit.main()
    report = json.loads(output.read_text(encoding="utf-8"))
    assert report["changedTitles"] == 1
    assert report["captions"] == [
        {
            "title": 0,
            "start": "0",
            "end": "1",
            "changed": True,
            "text": "甲",
            "lineCount": 1,
            "position": "0 11",
            "keys": [],
            "outerScale": "1 1",
        },
        {
            "title": 1,
            "start": "1",
            "end": "2",
            "changed": False,
            "text": "乙\n丙",
            "lineCount": 2,
            "position": "0 20",
            "keys": [],
            "outerScale": "1 1",
        },
    ]
    assert [p["renderReuseCandidate"] for p in report["partitionImpact"]] == [False, True]
    assert report["nativeRenderVerified"] is False
    assert report["headNearVerified"] is False


def test_without_partition(config: Path, tmp_path: Path) -> None:
    report = audit.audit(config, tmp_path / "result.json")
    assert report["partitionImpact"] == []


@pytest.mark.parametrize(
    ("old", "new"),
    [
        ("甲", "丁"),
        ('position="0 11"', 'position="0 11" scale="0.5 0.5"'),
        ('position="0 11"', 'position="1 11"'),
    ],
)
def test_changed_non_y_content_is_rejected(
    config: Path, tmp_path: Path, old: str, new: str
) -> None:
    data = json.loads(config.read_text(encoding="utf-8"))
    after = Path(data["after"]["path"])
    after.write_text(after.read_text(encoding="utf-8").replace(old, new), encoding="utf-8")
    data["after"]["sha256"] = file_digest(after)
    config.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(ReplayError, match="unchanged"):
        audit.audit(config, tmp_path / "result.json")


def test_bad_version(config: Path, tmp_path: Path) -> None:
    config.write_text('{"version":2}', encoding="utf-8")
    with pytest.raises(ReplayError, match="version"):
        audit.audit(config, tmp_path / "result.json")


@pytest.mark.parametrize(
    ("xml", "message"),
    [
        ("<fcpxml/>", "matching"),
        (
            "<fcpxml><asset-clip><title><text>甲\n\n乙</text></title></asset-clip></fcpxml>",
            "nonempty",
        ),
        ("<fcpxml><asset-clip><title><text>甲</text></title></asset-clip></fcpxml>", "transform"),
    ],
)
def test_invalid_rows(xml: str, message: str) -> None:
    with pytest.raises(ReplayError, match=message):
        audit._caption_rows(ET.fromstring(xml), [{"changed": False}])


def test_project_shape_and_input_mutation(
    config: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    with pytest.raises(ReplayError, match="one comparison"):
        audit._structure(ET.fromstring("<fcpxml/>"))
    original = audit._caption_rows

    def mutate(root: ET.Element, changes: list[dict[str, object]]) -> list[dict[str, object]]:
        result = original(root, changes)
        config.write_text(config.read_text(encoding="utf-8") + "\n", encoding="utf-8")
        return result

    monkeypatch.setattr(audit, "_caption_rows", mutate)
    with pytest.raises(ReplayError, match="input changed"):
        audit.audit(config, tmp_path / "result.json")


def _animated(values: list[tuple[str, str]], *, curve: str = "linear") -> ET.Element:
    transform = ET.Element("adjust-transform")
    param = ET.SubElement(transform, "param", name="position")
    animation = ET.SubElement(param, "keyframeAnimation")
    for time, value in values:
        ET.SubElement(animation, "keyframe", time=time, value=value, curve=curve)
    return transform


def test_constant_x_allows_y_resampling_and_static_to_animated() -> None:
    static = ET.fromstring('<adjust-transform position="+0.000 10"/>')
    before = _animated([("0s", "0 10"), ("1s", "0 11")])
    after = _animated([("0s", "0 1e-12"), ("1/2s", "0 25"), ("1s", "0 30")])
    assert audit._horizontal(static) == audit._horizontal(before) == audit._horizontal(after)
    after.set("position", "0 15")
    assert audit._horizontal(static) == audit._horizontal(after)
    assert audit._horizontal(ET.Element("adjust-transform")) != audit._horizontal(static)


@pytest.mark.parametrize("curve", ["linear", "smooth"])
def test_moving_x_preserves_clock_curve_and_x_values(curve: str) -> None:
    before = _animated([("0s", "0 10"), ("1s", "1 11")], curve=curve)
    y_only = _animated([("0s", "0 20"), ("1s", "1 30")], curve=curve)
    shifted_clock = _animated([("0s", "0 20"), ("2s", "1 30")], curve=curve)
    shifted_x = _animated([("0s", "0 20"), ("1s", "2 30")], curve=curve)
    assert audit._horizontal(before) == audit._horizontal(y_only)
    assert audit._horizontal(before) != audit._horizontal(shifted_clock)
    assert audit._horizontal(before) != audit._horizontal(shifted_x)


def test_unknown_position_metadata_is_preserved() -> None:
    before = _animated([("0s", "0 10"), ("1s", "0 20")])
    after = _animated([("0s", "0 30"), ("1s", "0 40")])
    before[0].set("enabled", "1")
    after[0].set("enabled", "1")
    assert audit._horizontal(before) == audit._horizontal(after)
    after[0].set("enabled", "0")
    assert audit._horizontal(before) != audit._horizontal(after)
    static_param = ET.fromstring(
        '<adjust-transform><param name="position" value="0 3"/></adjust-transform>'
    )
    other_param = ET.fromstring(
        '<adjust-transform><param name="position" value="1 3"/></adjust-transform>'
    )
    assert audit._horizontal(static_param) != audit._horizontal(other_param)


@pytest.mark.parametrize(
    "value",
    [
        None,
        "0",
        "0 1 2",
        "nan 1",
        "0 inf",
        "1e999999 0",
        "0 " + "1" * 65,
        "1e309 0",
        "0 1e309",
        "٠ 1",
    ],
)
def test_invalid_position_vectors_are_rejected(value: str | None) -> None:
    with pytest.raises(ReplayError, match="position vector"):
        audit._position_x(value)


@pytest.mark.parametrize(
    ("value", "expected"),
    [("1e-12 2E+2", "1/1000000000000"), ("-.5e+2 +1.e-2", "-50"), ("0 -1.5e2", "0")],
)
def test_scientific_x_remains_exact_and_finite(value: str, expected: str) -> None:
    assert audit._position_x(value) == expected


def test_duplicate_channels_and_nonincreasing_keys_are_rejected() -> None:
    duplicate = ET.fromstring(
        '<adjust-transform><param name="position"/><param name="position"/></adjust-transform>'
    )
    with pytest.raises(ReplayError, match="one caption"):
        audit._horizontal(duplicate)
    with pytest.raises(ReplayError, match="increasing"):
        audit._horizontal(_animated([("0s", "0 10"), ("0s", "0 20")]))
