import json
import sys
from dataclasses import replace
from pathlib import Path
from xml.etree import ElementTree as ET

import pytest
from test_fcp_pre_layout import geometry as geometry
from test_fcp_pre_layout import internal_geometry as internal_geometry

from replay import fcp_pre_layout_migrate as migration
from replay.core import ReplayError
from replay.fcp_pre_layout import GeometryValues, geometry_factor
from replay.fcp_project import CaptionTemplate
from replay.stages import file_digest


@pytest.fixture
def template(geometry: GeometryValues) -> CaptionTemplate:
    return CaptionTemplate(
        "New geometry", "~/new", "file:///new.moti", pre_layout_geometry=geometry
    )


@pytest.fixture
def source() -> ET.Element:
    return ET.fromstring(
        '<fcpxml><resources><effect id="t" uid="old"/></resources><library><event><project>'
        '<title ref="t" start="3600s" offset="3s" duration="1s">'
        "<text>long</text><text>long</text><text>long</text>"
        '<adjust-transform position="0 10" scale=".142 .142"/></title>'
        "</project></event></library></fcpxml>"
    )


def test_migration_changes_geometry_only_and_is_repeatable(
    source: ET.Element, template: CaptionTemplate
) -> None:
    original = ET.tostring(source)
    result = migration.migrate(source, template)
    assert ET.tostring(source) == original
    title = result.find(".//title")
    assert title is not None
    original_title = source.find(".//title")
    assert original_title is not None
    assert title.attrib == original_title.attrib
    assert ["".join(node.itertext()) for node in title.findall("text")] == ["long"] * 3
    assert geometry_factor(title, template.pre_layout_geometry) == pytest.approx(0.142)
    assert ET.tostring(migration.migrate(result, template)) == ET.tostring(result)


def test_template_only_diagnostic_preserves_old_geometry_exactly(
    source: ET.Element, template: CaptionTemplate
) -> None:
    result = migration.migrate(source, template, template_only_diagnostic=True)
    old_title, new_title = source.find(".//title"), result.find(".//title")
    assert old_title is not None and new_title is not None
    assert ET.tostring(old_title) == ET.tostring(new_title)
    assert new_title.findall("param") == []


def test_internal_group_migration_is_idempotent(
    source: ET.Element, template: CaptionTemplate, internal_geometry: GeometryValues
) -> None:
    template = replace(template, pre_layout_geometry=internal_geometry)
    first = migration.migrate(source, template)
    second = migration.migrate(first, template)
    assert ET.tostring(first) == ET.tostring(second)
    title = second.find(".//title")
    assert title is not None
    assert geometry_factor(title, internal_geometry) == pytest.approx(0.142)


@pytest.mark.parametrize(
    "fault", ["contract", "title", "effect", "text", "transform", "nonnumeric", "nonuniform"]
)
def test_migration_rejects_ambiguous_contract(
    source: ET.Element, template: CaptionTemplate, fault: str
) -> None:
    title = source.find(".//title")
    assert title is not None
    if fault == "contract":
        template = replace(template, pre_layout_geometry=())
    elif fault == "title":
        title.tag = "clip"
    elif fault == "effect":
        title.set("ref", "missing")
    elif fault == "text":
        title.remove(title.findall("text")[0])
    else:
        transform = title.find("adjust-transform")
        assert transform is not None
        if fault == "transform":
            title.remove(transform)
        else:
            transform.set("scale", "x y" if fault == "nonnumeric" else ".1 .2")
    with pytest.raises(ReplayError):
        migration.migrate(source, template)


@pytest.mark.parametrize(
    "fault", ["none", "version", "budget", "elements", "name", "existing", "mode"]
)
def test_cli_binds_new_template_and_never_claims_native_verification(
    tmp_path: Path,
    template: CaptionTemplate,
    source: ET.Element,
    monkeypatch: pytest.MonkeyPatch,
    fault: str,
) -> None:
    media = tmp_path / "source.fcpxml"
    media.write_bytes(ET.tostring(source))
    motion = tmp_path / "new.moti"
    motion.write_text("synthetic", encoding="utf-8")
    config = tmp_path / "config.json"
    config.write_text(
        json.dumps(
            {
                "version": 2 if fault == "version" else 1,
                "sourceXML": {"path": str(media), "sha256": file_digest(media)},
                "template": {"path": str(motion), "sha256": file_digest(motion)},
                "name": "" if fault == "name" else "Test",
                "mode": "bad" if fault == "mode" else "pre-layout-geometry",
            }
        ),
        encoding="utf-8",
    )
    output = tmp_path / "new.fcpxml"
    monkeypatch.setattr(migration, "caption_template", lambda *args, **kwargs: template)
    if fault == "budget":
        monkeypatch.setattr(migration, "MAX_XML_BYTES", 1)
    if fault == "elements":
        monkeypatch.setattr(migration, "MAX_XML_ELEMENTS", 1)
    if fault == "existing":
        output.write_text("existing", encoding="utf-8")
    monkeypatch.setattr(sys, "argv", ["migration", str(config), str(output)])
    if fault != "none":
        with pytest.raises(ReplayError):
            migration.main()
        return
    migration.main()
    receipt = json.loads(output.with_suffix(".geometry-migration.json").read_text())
    assert receipt["nativeRenderVerified"] is False
    assert receipt["headNearVerified"] is False
    assert receipt["placementPolicy"] == "source-position-track-unchanged"
