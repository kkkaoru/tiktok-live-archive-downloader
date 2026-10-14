import copy
import json
import sys
from pathlib import Path
from xml.etree import ElementTree as ET

import pytest

from replay import fcp_template_reference as reference
from replay.core import ReplayError
from replay.stages import file_digest


@pytest.fixture
def config(tmp_path: Path) -> Path:
    template = tmp_path / "style.moti"
    template.write_text("<ozml/>")
    source = tmp_path / "source.fcpxml"
    source.write_text(
        '<!DOCTYPE fcpxml><fcpxml version="1.11"><resources><effect id="r1" '
        f'uid="~/old" src="{template.as_uri()}"/></resources><library><event><project '
        'name="old"><sequence duration="60s"><spine><asset-clip start="30s">'
        '<title offset="31s"><text>word</text><adjust-transform position="0 12"/>'
        "</title></asset-clip></spine></sequence></project></event></library></fcpxml>"
    )
    path = tmp_path / "config.json"
    path.write_text(
        json.dumps(
            {
                "version": 1,
                "source": {"path": str(source), "sha256": file_digest(source)},
                "template": {"path": str(template), "sha256": file_digest(template)},
                "effectID": "r1",
                "expectedUID": "~/old",
                "projectName": "new",
            }
        )
    )
    return path


def run(config: Path) -> dict[str, object]:
    return reference.rewrite(config, config.parent / "out.fcpxml", config.parent / "proof.json")


def test_only_uid_and_name(config: Path) -> None:
    report = run(config)
    assert report["nativeAcceptanceVerified"] is False
    assert report["allOtherXMLUnchanged"] is True
    root = ET.parse(config.parent / "out.fcpxml")
    project = root.find(".//project")
    assert project is not None and project.get("name") == "new"
    assert b"<!DOCTYPE fcpxml>" in (config.parent / "out.fcpxml").read_bytes()
    with pytest.raises(ReplayError, match="distinct and new"):
        run(config)


@pytest.mark.parametrize(
    "key,value",
    [
        ("version", 2),
        ("effectID", "absent"),
        ("expectedUID", None),
        ("expectedUID", "wrong"),
        ("projectName", ""),
        ("projectName", 4),
    ],
)
def test_invalid_config(config: Path, key: str, value: object) -> None:
    data = json.loads(config.read_text())
    data[key] = value
    config.write_text(json.dumps(data))
    with pytest.raises(ReplayError):
        run(config)


@pytest.mark.parametrize(
    "xml",
    [
        '<!DOCTYPE fcpxml SYSTEM "evil"><fcpxml/>',
        '<fcpxml><!ENTITY a "b"></fcpxml>',
        "<other/>",
        '<fcpxml><resources><effect id="r1" uid="~/old"/></resources></fcpxml>',
    ],
)
def test_invalid_xml(config: Path, xml: str) -> None:
    data = json.loads(config.read_text())
    source = Path(data["source"]["path"])
    source.write_text(xml)
    data["source"]["sha256"] = file_digest(source)
    config.write_text(json.dumps(data))
    with pytest.raises(ReplayError):
        run(config)


def test_no_rename_and_missing_project_name(config: Path) -> None:
    data = json.loads(config.read_text())
    data.pop("projectName")
    source = Path(data["source"]["path"])
    source.write_text(source.read_text().replace(' name="old"', ""))
    data["source"]["sha256"] = file_digest(source)
    config.write_text(json.dumps(data))
    assert run(config)["projectName"] is None


def test_changed_input_and_limits(config: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(reference, "MAX_XML_ELEMENTS", 1)
    with pytest.raises(ReplayError, match="bounded FCPXML"):
        run(config)
    monkeypatch.setattr(reference, "MAX_XML_ELEMENTS", 100)
    original = file_digest
    calls = 0

    def changed(path: Path) -> str:
        nonlocal calls
        calls += 1
        return "0" * 64 if calls > 4 else original(path)

    monkeypatch.setattr(reference, "file_digest", changed)
    with pytest.raises(ReplayError, match="input changed"):
        run(config)


def test_path_and_template_guards(config: Path) -> None:
    with pytest.raises(ReplayError, match="checkout"):
        reference._local(Path("/outside"))
    with pytest.raises(ReplayError, match="distinct"):
        reference.rewrite(config, config.parent / "same", config.parent / "same")
    data = json.loads(config.read_text())
    data["template"] = data["source"]
    config.write_text(json.dumps(data))
    with pytest.raises(ReplayError, match="Motion template"):
        run(config)


def test_cli(config: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        ["reference", str(config), str(config.parent / "out"), str(config.parent / "proof")],
    )
    reference.main()
    assert (config.parent / "proof").exists()


def test_byte_budget(config: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(reference, "MAX_XML_BYTES", 1)
    with pytest.raises(ValueError):
        run(config)


def test_output_growth_is_bounded(config: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    source = config.parent / "source.fcpxml"
    monkeypatch.setattr(reference, "MAX_XML_BYTES", source.stat().st_size + 1)
    with pytest.raises(ReplayError, match="output exceeds"):
        run(config)


@pytest.fixture
def observed(config: Path) -> Path:
    data = json.loads(config.read_text())
    target = config.parent / "managed.moti"
    target.write_text("<ozml/>")
    native = config.parent / "native.fcpxml"
    native.write_text(
        '<fcpxml><resources><effect id="native" name="Managed" uid="../observed" '
        f'src="{target.as_uri()}"/></resources></fcpxml>'
    )
    data["observedEffect"] = {
        "roundtrip": {"path": str(native), "sha256": file_digest(native)},
        "targetTemplate": {"path": str(target), "sha256": file_digest(target)},
        "effectID": "native",
        "expectedUID": "../observed",
    }
    config.write_text(json.dumps(data))
    return config


def test_observed_binding_preserves_body(observed: Path) -> None:
    report = run(observed)
    assert report["candidateUID"] == "../observed"
    assert report["observedEffectBound"] is True
    assert report["allOtherXMLUnchanged"] is True
    root = ET.parse(observed.parent / "out.fcpxml")
    effect = root.find("./resources/effect")
    assert effect is not None and effect.get("name") == "Managed"
    assert effect.get("src") == (observed.parent / "managed.moti").as_uri()
    assert root.findtext(".//title/text") == "word"


@pytest.mark.parametrize("defect", ["bytes", "uid", "missing", "multiple", "src", "name"])
def test_observed_rejects_mismatches(observed: Path, defect: str) -> None:
    data = json.loads(observed.read_text())
    settings = data["observedEffect"]
    native = Path(settings["roundtrip"]["path"])
    target = Path(settings["targetTemplate"]["path"])
    if defect == "bytes":
        target.write_text("<ozml>different</ozml>")
        settings["targetTemplate"]["sha256"] = file_digest(target)
    elif defect == "uid":
        settings["expectedUID"] = "wrong"
    elif defect == "missing":
        settings["effectID"] = "missing"
    elif defect == "multiple":
        native.write_text(
            native.read_text().replace("</resources>", '<effect id="native"/></resources>')
        )
    elif defect == "src":
        native.write_text(native.read_text().replace(target.as_uri(), "file:///elsewhere"))
    else:
        native.write_text(native.read_text().replace(' name="Managed"', ""))
    settings["roundtrip"]["sha256"] = file_digest(native)
    observed.write_text(json.dumps(data))
    with pytest.raises(ReplayError):
        run(observed)


def test_unexpected_title_mutation_rejected(
    observed: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    original = copy.deepcopy

    def tamper(root: ET.Element) -> ET.Element:
        title = root.find(".//title/text")
        assert title is not None
        title.text = "changed"
        return original(root)

    monkeypatch.setattr(copy, "deepcopy", tamper)
    with pytest.raises(ReplayError, match="Unexpected non-resource"):
        run(observed)
