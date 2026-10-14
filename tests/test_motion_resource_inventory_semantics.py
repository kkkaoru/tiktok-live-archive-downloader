"""Synthetic ozml fixtures exercise slot semantics without production recordings."""

import plistlib
from pathlib import Path

import pytest

from replay import motion_resource_inventory as resources
from replay.core import ReplayError

SCENE = """<ozml version="5.13"><scene><publishSettings>
<target object="10" channel="./2/319/321"/>
<target object="10" channel="./5/20/14/15"/>
</publishSettings><layer><scenenode id="10" factoryID="16" version="5">
<textPathModified textPathModifiedVal="0"/>
<style id="20"><parameter name="Font" id="83">
<font>Test-Bold</font><defaultFont>Helvetica</defaultFont></parameter>
<parameter id="14"><parameter id="15" value="1"/>
<parameter name="Texture" id="18"><parameter name="Image" id="1" value="0"/>
<parameter name="Wrap Mode" id="5" value="1"/></parameter>
<parameter name="Source Media" id="85"/>
</parameter></style>
<parameter name="Object" id="2"><parameter id="319"><parameter id="321" value="1"/></parameter>
<parameter name="Path Options" id="329"><parameter name="Shape Source" id="339" value="0"/>
<parameter name="Align to Path" id="333" value="1"/></parameter></parameter>
</scenenode></layer></scene></ozml>"""

PRESET = {
    "Description": "Synthetic authoring settings",
    "tag": "test",
    "DisplayName": "Test",
    "FieldDominance": 0,
    "FrameRate": 29.97,
    "Height": 1080,
    "IsRegionalDefault": False,
    "IsHidden": False,
    "PixelAspect": 1,
    "Width": 1920,
}
AUTHORING = """<ozml version="5.13"><scene><sceneSettings>
<width>1920</width><height>1080</height><frameRate>30</frameRate><NTSC>1</NTSC>
<pixelAspectRatio>1</pixelAspectRatio><fieldRenderingMode>0</fieldRenderingMode>
<presetPath>/missing.preset</presetPath></sceneSettings></scene></ozml>"""


def test_supported_slots_and_font_names(tmp_path: Path) -> None:
    template = tmp_path / "title.moti"
    template.write_text(SCENE, encoding="utf-8")
    result = resources.inventory(template, allowed_roots=(tmp_path,))
    assert [(field.kind, field.status) for field in result.fields] == [
        ("graph-channel", "internal"),
        ("graph-channel", "internal"),
        ("container", "internal"),
        ("boolean", "internal"),
        ("font-name", "requires-font-supplier"),
        ("font-name", "requires-font-supplier"),
        ("container", "internal"),
        ("empty-binding", "internal"),
        ("empty-binding", "internal"),
        ("enum", "internal"),
        ("enum", "internal"),
        ("empty-binding", "internal"),
        ("container", "internal"),
        ("empty-binding", "internal"),
        ("empty-binding", "internal"),
        ("boolean", "internal"),
        ("boolean", "internal"),
    ]
    assert all(field.reason and field.evidence for field in result.fields)
    assert result.render_closure_verified is False


@pytest.mark.parametrize(
    "old,new",
    [
        ("./2/319/321", "./2/319/999"),
        ("./2/319/321", "./2/../file"),
        ("./2/319/321", "https://example.test/file"),
        ("./2/319/321", "not-a-path"),
        ('object="10"', 'object="99"'),
        ("./5/20/14/15", "./5/99/14/15"),
        ("./5/20/14/15", "./5/20"),
        ('<parameter id="321" value="1"/>', '<parameter id="321"/><parameter id="321"/>'),
    ],
)
def test_invalid_graph_is_never_a_file_or_internal(tmp_path: Path, old: str, new: str) -> None:
    template = tmp_path / "title.moti"
    template.write_text(SCENE.replace(old, new), encoding="utf-8")
    fields = resources.inventory(template, allowed_roots=(tmp_path,)).fields
    invalid = [field for field in fields if field.status == "invalid-graph"]
    assert invalid
    assert all(field.kind == "unknown" and field.path is None for field in invalid)


@pytest.mark.parametrize(
    "old,new",
    [
        ('version="5.13"', 'version="6.0"'),
        ('textPathModifiedVal="0"', 'textPathModifiedVal="2"'),
        ('name="Image" id="1" value="0"', 'name="Image" id="1" value="42"'),
        ('name="Shape Source" id="339" value="0"', 'name="Shape Source" id="339" value="7"'),
        ('name="Align to Path" id="333" value="1"', 'name="Align to Path" id="333" value="8"'),
        ('factoryID="16"', 'factoryID="99"'),
        ('version="5"', 'version="6"'),
        ('name="Wrap Mode" id="5" value="1"', 'name="Wrap Mode" id="5" value="9"'),
    ],
)
def test_unknown_numeric_slots_fail_closed(tmp_path: Path, old: str, new: str) -> None:
    template = tmp_path / "title.moti"
    template.write_text(SCENE.replace(old, new), encoding="utf-8")
    result = resources.inventory(template, allowed_roots=(tmp_path,))
    assert any(field.kind == "unknown" for field in result.fields)


@pytest.mark.parametrize("wrapper", ["publishSettings", "other"])
def test_published_target_outside_scene_is_not_supported(tmp_path: Path, wrapper: str) -> None:
    template = tmp_path / "title.moti"
    template.write_text(
        f'<ozml version="5.13"><{wrapper}><target object="1" channel="./1/2"/></{wrapper}></ozml>',
        encoding="utf-8",
    )
    result = resources.inventory(template, allowed_roots=(tmp_path,))
    assert result.fields[0].status == "invalid-graph"
    assert result.fields[0].kind == "unknown"


def test_external_media_in_known_texture_is_not_dropped(tmp_path: Path) -> None:
    template = tmp_path / "title.moti"
    template.write_text(
        SCENE.replace('name="Image" id="1" value="0"', 'name="Image" id="1" value="./missing.png"'),
        encoding="utf-8",
    )
    result = resources.inventory(template, allowed_roots=(tmp_path,))
    field = next(field for field in result.fields if field.value == "./missing.png")
    assert (field.kind, field.status) == ("external-file", "missing")


def test_copy_binds_only_one_field_and_inventory_follows_plist(tmp_path: Path) -> None:
    preset = tmp_path / "new & selected.preset"
    preset.write_bytes(plistlib.dumps(PRESET))
    output = resources.bind_project_preset(
        AUTHORING, preset=preset, expected_previous="/missing.preset"
    )
    assert output.replace(str(preset).replace("&", "&amp;"), "/missing.preset") == AUTHORING
    template = tmp_path / "new.moti"
    template.write_text(output, encoding="utf-8")
    result = resources.inventory(template, allowed_roots=(tmp_path,))
    assert len(result.documents) == 2
    assert result.fields[0].status == "resolved"
    assert result.fields[0].kind == "external-file"
    assert resources.project_preset(preset)["FrameRate"] == 29.97


@pytest.mark.parametrize(
    "change",
    [
        {"Height": 0},
        {"Width": "1920"},
        {"FrameRate": True},
        {"FrameRate": 0},
        {"PixelAspect": 2},
        {"FieldDominance": 1},
        {"Description": []},
        {"resource": "/media"},
    ],
)
def test_reject_invalid_preset(tmp_path: Path, change: dict[str, object]) -> None:
    preset = tmp_path / "bad.preset"
    preset.write_bytes(plistlib.dumps(PRESET | change))
    with pytest.raises(ReplayError, match="preset"):
        resources.project_preset(preset)


@pytest.mark.parametrize(
    "old,new,match",
    [
        ('version="5.13"', 'version="6.0"', "Unsupported"),
        ("/missing.preset", "/other.preset", "mismatch"),
        ("<width>1920</width>", "<width>invalid</width>", "Invalid"),
        ("<width>1920</width>", "<width>1080</width>", "incompatible"),
        ("<NTSC>1</NTSC>", "<NTSC>2</NTSC>", "incompatible"),
        ("<frameRate>30</frameRate>", "<frameRate>inf</frameRate>", "incompatible"),
        (
            "<pixelAspectRatio>1</pixelAspectRatio>",
            "<pixelAspectRatio>2</pixelAspectRatio>",
            "incompatible",
        ),
        ("<presetPath>", "<presetPath >", "lexical"),
    ],
)
def test_reject_incompatible_binding(tmp_path: Path, old: str, new: str, match: str) -> None:
    preset = tmp_path / "test.preset"
    preset.write_bytes(plistlib.dumps(PRESET))
    with pytest.raises(ReplayError, match=match):
        resources.bind_project_preset(
            AUTHORING.replace(old, new), preset=preset, expected_previous="/missing.preset"
        )


def test_missing_and_malformed_presets(tmp_path: Path) -> None:
    preset = tmp_path / "bad.preset"
    with pytest.raises(ReplayError, match="canonical"):
        resources.project_preset(preset)
    preset.write_bytes(b"not a plist")
    with pytest.raises(ReplayError, match="Invalid"):
        resources.project_preset(preset)
    preset.write_bytes(
        b'<?xml version="1.0"?><!DOCTYPE plist [<!ENTITY file SYSTEM "file:///outside">]><plist><string>&file;</string></plist>'
    )
    with pytest.raises(ReplayError):
        resources.project_preset(preset)
