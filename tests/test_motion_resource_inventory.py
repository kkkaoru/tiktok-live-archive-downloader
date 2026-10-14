from pathlib import Path

import pytest

from replay import motion_resource_inventory as resources
from replay.core import ReplayError


def test_missing_and_unknown_are_not_empty_resources(tmp_path: Path) -> None:
    template = tmp_path / "title.moti"
    template.write_text(
        '<ozml version="5.13"><!-- comment --><presetPath>./missing.preset</presetPath>'
        "<texture/><resource>unknown</resource><url>https://example.test/image.png</url>"
        '<parameter name="Image" value="42"/></ozml>',
        encoding="utf-8",
    )
    result = resources.inventory(template, allowed_roots=(tmp_path,))
    assert [field.status for field in result.fields] == [
        "missing",
        "empty-slot",
        "unresolved-symbolic-value",
        "unresolved-uri",
        "empty-slot",
        "unresolved-symbolic-value",
    ]
    assert result.tags == {
        "ozml": 1,
        "presetPath": 1,
        "texture": 1,
        "resource": 1,
        "url": 1,
        "parameter": 1,
    }
    assert result.attributes == {"version": 1, "name": 1, "value": 1}
    assert result.examined_fields == 9
    assert result.report()["render_closure_verified"] is False


def test_empty_named_resource_slot(tmp_path: Path) -> None:
    template = tmp_path / "title.moti"
    template.write_text(
        '<ozml version="5.13"><parameter name="Source Media"/></ozml>', encoding="utf-8"
    )
    result = resources.inventory(template, allowed_roots=(tmp_path,))
    assert len(result.fields) == 1
    assert result.fields[0].status == "empty-slot"
    assert result.fields[0].field == "parameter"


def test_recursive_cycle_and_exact_file_uri(tmp_path: Path) -> None:
    template = tmp_path / "title.moti"
    child = tmp_path / "child.moti"
    image = tmp_path / "image.bin"
    image.write_bytes(b"not-decoded")
    template.write_text(
        f'<ozml version="5.13"><path>{child.as_uri()}</path></ozml>', encoding="utf-8"
    )
    child.write_text(
        f'<ozml version="5.13"><path>{template}</path><resource>{image}</resource></ozml>',
        encoding="utf-8",
    )
    result = resources.inventory(template, allowed_roots=(tmp_path,))
    assert len(result.documents) == 2
    assert len(result.fields) == 3
    assert {field.status for field in result.fields} == {"resolved"}


def test_scope_and_symlinks(tmp_path: Path) -> None:
    scope = tmp_path / "scope"
    scope.mkdir()
    original = scope / "data.bin"
    original.write_bytes(b"x")
    link = scope / "linked.bin"
    link.symlink_to(original)
    template = scope / "title.moti"
    template.write_text(
        f'<ozml version="5.13"><path>{link}</path><path>/outside/scope/file.bin</path></ozml>',
        encoding="utf-8",
    )
    result = resources.inventory(template, allowed_roots=(scope,))
    assert [field.status for field in result.fields] == ["noncanonical-or-linked", "outside-scope"]
    with pytest.raises(ReplayError, match="scope roots"):
        resources.inventory(template, allowed_roots=())
    with pytest.raises(ReplayError, match="scope roots"):
        resources.inventory(template, allowed_roots=(link,))
    with pytest.raises(ReplayError, match="outside scope"):
        resources.inventory(template, allowed_roots=(original,))


@pytest.mark.parametrize("value", ["file://host/file", "file:///file?query", "file:///file#part"])
def test_uri_ambiguity(tmp_path: Path, value: str) -> None:
    template = tmp_path / "title.moti"
    template.write_text(f'<ozml version="5.13"><url>{value}</url></ozml>', encoding="utf-8")
    result = resources.inventory(template, allowed_roots=(tmp_path,))
    assert result.fields[0].status == "unresolved-uri"


@pytest.mark.parametrize("budget", ["MAX_DOCUMENTS", "MAX_FIELDS", "MAX_XML_CHARACTERS"])
def test_budgets(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, budget: str) -> None:
    template = tmp_path / "title.moti"
    template.write_text('<ozml version="5.13"/>', encoding="utf-8")
    monkeypatch.setattr(resources, budget, 0)
    with pytest.raises(ReplayError, match="budget exceeded"):
        resources.inventory(template, allowed_roots=(tmp_path,))


def test_hostile_xml(tmp_path: Path) -> None:
    template = tmp_path / "title.moti"
    template.write_text('<!DOCTYPE x [<!ENTITY x "bad">]><ozml version="5.13"/>', encoding="utf-8")
    with pytest.raises(ReplayError):
        resources.inventory(template, allowed_roots=(tmp_path,))
