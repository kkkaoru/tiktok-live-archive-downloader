import json
import sys
from pathlib import Path
from xml.etree import ElementTree as ET

import pytest

from replay import motion_title_fcpxml
from replay.core import ReplayError
from replay.motion_title_fcpxml import build
from replay.stages import file_digest


@pytest.fixture
def plan(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict[str, object]:
    monkeypatch.setattr(motion_title_fcpxml, "ROOT", tmp_path)
    path = tmp_path / "Motion Templates.localized/Titles.localized/Test & Category/Title/Title.moti"
    path.parent.mkdir(parents=True)
    path.write_text(
        '<ozml version="5.13"><template><flags>1</flags></template><scene><sceneSettings>'
        "<width>1080</width><height>1920</height><frameRate>30</frameRate>"
        "<duration>60</duration><NTSC>0</NTSC></sceneSettings><layer>"
        '<scenenode factoryID="16"><parameter id="2"><parameter id="369">'
        "<text>検証一行目</text></parameter></parameter></scenenode>"
        '<scenenode factoryID="16"><parameter id="2"><parameter id="369">'
        "<text>検証一行目\n検証二行目 &amp; 字</text></parameter></parameter></scenenode>"
        "</layer></scene></ozml>",
        encoding="utf-8",
    )
    return {
        "version": 1,
        "name": "Synthetic & title",
        "width": 1080,
        "height": 1920,
        "rate": 30,
        "frames": 90,
        "titles": [
            {"path": str(path), "sha256": file_digest(path), "offsetFrame": 30, "frames": 60}
        ],
    }


def title_item(plan: dict[str, object]) -> dict[str, object]:
    titles = plan["titles"]
    assert isinstance(titles, list)
    item = titles[0]
    assert isinstance(item, dict)
    return item


def test_file_reference_clocks_and_no_flattened_media(plan: dict[str, object]) -> None:
    root = ET.fromstring(build(plan))
    effect = root.find("resources/effect")
    assert effect is not None
    assert effect.get("uid") == "~/Titles.localized/Test & Category/Title/Title.moti"
    assert (effect.get("src") or "").endswith("/Test%20%26%20Category/Title/Title.moti")
    assert root.findall("resources/asset") == []
    title = root.find("library/event/project/sequence/spine/gap/title")
    assert title is not None
    assert title.attrib == {
        "ref": "t0",
        "name": "Title",
        "lane": "1",
        "offset": "1s",
        "start": "3600s",
        "duration": "2s",
    }
    assert [node.text for node in title.findall("text")] == [
        "検証一行目",
        "検証一行目\n検証二行目 & 字",
    ]
    assert title.findall("text-style-def") == []
    assert title.findall(".//text-style") == []
    assert root.findtext("library/event/project/sequence/spine/gap") is not None


@pytest.mark.parametrize(
    "key,value",
    [
        ("version", True),
        ("name", ""),
        ("rate", 0),
        ("width", 0),
        ("height", 8193),
        ("frames", 3601),
        ("titles", []),
        ("titles", "bad"),
        ("rate", True),
    ],
)
def test_invalid_timeline_fields(plan: dict[str, object], key: str, value: object) -> None:
    plan[key] = value
    with pytest.raises((ValueError, ReplayError)):
        build(plan)


def test_unknown_fields_fail(plan: dict[str, object]) -> None:
    plan["ignored"] = True
    with pytest.raises(ValueError, match="fields"):
        build(plan)


def test_unknown_title_fields_fail(plan: dict[str, object]) -> None:
    title_item(plan)["ignored"] = True
    with pytest.raises(ValueError, match="reference"):
        build(plan)


@pytest.mark.parametrize(
    "key,value",
    [
        ("sha256", "0" * 64),
        ("frames", 61),
        ("offsetFrame", -1),
        ("offsetFrame", 31),
    ],
)
def test_mismatched_title_fails(plan: dict[str, object], key: str, value: object) -> None:
    title_item(plan)[key] = value
    with pytest.raises(ValueError):
        build(plan)


def test_overlap_fails(plan: dict[str, object]) -> None:
    item = title_item(plan)
    plan["titles"] = [item, item]
    with pytest.raises(ValueError, match="overlap"):
        build(plan)


@pytest.mark.parametrize("change", ["settings", "flags", "tree", "loose", "link", "empty"])
def test_template_contract_is_not_guessed(
    plan: dict[str, object], tmp_path: Path, change: str
) -> None:
    item = title_item(plan)
    path = Path(str(item["path"]))
    if change == "settings":
        path.write_text('<ozml version="5.13"/>', encoding="utf-8")
    elif change == "flags":
        path.write_text(
            path.read_text(encoding="utf-8").replace("<flags>1", "<flags>2"), encoding="utf-8"
        )
    elif change in {"tree", "loose"}:
        new = tmp_path / (
            "Motion Templates.localized/Generators.localized/X.moti"
            if change == "tree"
            else "loose.moti"
        )
        new.parent.mkdir(parents=True, exist_ok=True)
        path.rename(new)
        path = new
    elif change == "link":
        new = path.with_name("link.moti")
        new.symlink_to(path)
        item["path"] = str(new)
        with pytest.raises(ValueError, match="unlinked"):
            build(plan)
        return
    else:
        path.write_bytes(b"")
        with pytest.raises(ValueError, match="pinned"):
            build(plan)
        return
    item.update(path=str(path), sha256=file_digest(path))
    with pytest.raises(ValueError):
        build(plan)


@pytest.mark.parametrize(
    "body",
    [
        "<scene><layer/></scene>",
        '<scene><layer><scenenode factoryID="16"/></layer></scene>',
        '<scene><layer><scenenode factoryID="16"><parameter id="2">'
        '<parameter id="369"><text/><text>x</text></parameter></parameter>'
        "</scenenode></layer></scene>",
        '<scene><layer><scenenode factoryID="16"><parameter id="2">'
        '<parameter id="369"><text/></parameter></parameter></scenenode></layer></scene>',
        '<scene><layer><scenenode factoryID="16"><parameter id="2">'
        '<parameter id="369"><text><span>x</span></text></parameter></parameter>'
        "</scenenode></layer></scene>",
        '<scene><layer><scenenode factoryID="16"><parameter id="2">'
        '<parameter id="369"><text>' + "x" * 121 + "</text></parameter></parameter>"
        "</scenenode></layer></scene>",
        '<scene><layer><scenenode factoryID="16"><parameter id="2">'
        '<parameter id="369"><text>x</text></parameter></parameter></scenenode>'
        '<layer><scenenode factoryID="16"/></layer></layer></scene>',
        "<scene><layer>" + '<scenenode factoryID="16"/>' * 97 + "</layer></scene>",
    ],
)
def test_text_contract_refuses_ambiguous_or_unbounded_objects(body: str) -> None:
    with pytest.raises(ValueError, match="Motion"):
        motion_title_fcpxml._title_texts(ET.fromstring('<ozml version="5.13">' + body + "</ozml>"))


def test_title_text_binding_preserves_each_object_and_source(plan: dict[str, object]) -> None:
    path = Path(str(title_item(plan)["path"]))
    before = path.read_bytes()
    first = ET.fromstring(build(plan))
    assert [node.text for node in first.findall(".//title/text")] == [
        "検証一行目",
        "検証一行目\n検証二行目 & 字",
    ]
    assert path.read_bytes() == before


def test_outside_root_refused(plan: dict[str, object]) -> None:
    title_item(plan)["path"] = "/not-in-checkout/title.moti"
    with pytest.raises(ValueError, match="checkout"):
        build(plan)


def test_cli_non_overwrite_and_bounded_config(
    plan: dict[str, object], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source, output = tmp_path / "plan.json", tmp_path / "out.fcpxml"
    source.write_text(json.dumps(plan), encoding="utf-8")
    monkeypatch.setattr(sys, "argv", ["motion_title_fcpxml", str(source), str(output)])
    motion_title_fcpxml.main()
    assert ET.parse(output).getroot().tag == "fcpxml"
    with pytest.raises(FileExistsError):
        motion_title_fcpxml.main()
    source.write_bytes(b"")
    with pytest.raises(ValueError, match="bounded"):
        motion_title_fcpxml.main()
