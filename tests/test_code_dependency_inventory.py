"""Synthetic source closure regressions; no producer imports or private data."""

import json
from pathlib import Path

import pytest

from replay import code_dependency_inventory as inventory


@pytest.fixture
def package(tmp_path: Path) -> Path:
    path = tmp_path / "replay"
    path.mkdir()
    (path / "__init__.py").write_text("", encoding="utf-8")
    return path


def test_lazy_relative_and_absolute_imports_are_transitive(package: Path) -> None:
    (package / "entry.py").write_text(
        "import os\nimport replay\nfrom replay import alpha as a\n"
        "def unused():\n    from . import beta\n    from .gamma import value\n",
        encoding="utf-8",
    )
    (package / "alpha.py").write_text("import replay.beta\n", encoding="utf-8")
    (package / "beta.py").write_text("from replay.entry import unused\n", encoding="utf-8")
    (package / "gamma.py").write_text("raise RuntimeError('never import me')\n", encoding="utf-8")
    sources = inventory.capture(("entry",), package=package)
    assert tuple(item.path.name for item in sources) == (
        "__init__.py",
        "alpha.py",
        "beta.py",
        "entry.py",
        "gamma.py",
    )
    assert sources[0].sha256 == "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"


def test_replay_submodule_and_external_from_import(package: Path) -> None:
    (package / "entry.py").write_text(
        "from replay.helper import value\nfrom pathlib import Path\n", encoding="utf-8"
    )
    (package / "helper.py").write_text("value = 1\n", encoding="utf-8")
    assert tuple(item.path.name for item in inventory.capture(("entry",), package=package)) == (
        "__init__.py",
        "entry.py",
        "helper.py",
    )


@pytest.mark.parametrize("modules", [(), ("../private",), ("replay.entry",), ("entry/other",)])
def test_invalid_entry_names_rejected(package: Path, modules: tuple[str, ...]) -> None:
    with pytest.raises(ValueError):
        inventory.capture(modules, package=package)


@pytest.mark.parametrize(
    "text", ["from .. import private\n", "import replay.nested.module\n", "from replay import *\n"]
)
def test_out_of_scope_imports_rejected(package: Path, text: str) -> None:
    (package / "entry.py").write_text(text, encoding="utf-8")
    with pytest.raises(ValueError):
        inventory.capture(("entry",), package=package)


def test_linked_source_rejected(package: Path) -> None:
    (package / "real.py").write_text("", encoding="utf-8")
    (package / "entry.py").symlink_to("real.py")
    with pytest.raises(ValueError, match="canonical regular"):
        inventory.capture(("entry",), package=package)


def test_linked_package_rejected(package: Path) -> None:
    alias = package.parent / "alias"
    alias.symlink_to(package, target_is_directory=True)
    with pytest.raises(ValueError, match="canonical package"):
        inventory.capture(("entry",), package=alias)


def test_missing_source_does_not_omit_dependency(package: Path) -> None:
    (package / "entry.py").write_text("from .missing import value\n", encoding="utf-8")
    with pytest.raises(FileNotFoundError):
        inventory.capture(("entry",), package=package)


def test_oversized_source_rejected(package: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    (package / "entry.py").write_text("# too long\n", encoding="utf-8")
    monkeypatch.setattr(inventory, "MAX_SOURCE_BYTES", 3)
    with pytest.raises(ValueError, match="bounded canonical"):
        inventory.capture(("entry",), package=package)


def test_aggregate_budget_rejected(package: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    (package / "entry.py").write_text("# large\n", encoding="utf-8")
    monkeypatch.setattr(inventory, "MAX_TOTAL_BYTES", 3)
    with pytest.raises(ValueError, match="aggregate"):
        inventory.capture(("entry",), package=package)


def test_module_budget_rejected(package: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    (package / "entry.py").write_text("", encoding="utf-8")
    monkeypatch.setattr(inventory, "MAX_MODULES", 1)
    with pytest.raises(ValueError, match="module budget"):
        inventory.capture(("entry",), package=package)


def test_dependency_modified_after_read_rejected(
    package: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (package / "entry.py").write_text("from .helper import value\n", encoding="utf-8")
    (package / "helper.py").write_text("value = 1\n", encoding="utf-8")
    original = inventory._read

    def changed(path: Path) -> tuple[bytes, inventory.Source]:
        result = original(path)
        if path.name == "helper.py":
            (package / "entry.py").write_text("# changed\n", encoding="utf-8")
        return result

    monkeypatch.setattr(inventory, "_read", changed)
    with pytest.raises(ValueError, match="during closure"):
        inventory.capture(("entry",), package=package)


def test_content_digest_is_from_the_actual_source_bytes(package: Path) -> None:
    path = package / "entry.py"
    path.write_bytes(b"# original\n")
    before = inventory.capture(("entry",), package=package)
    path.write_bytes(b"# modified\n")
    after = inventory.capture(("entry",), package=package)
    assert before[1].sha256 != after[1].sha256
    assert before[1].identity[:2] == after[1].identity[:2]


def test_package_replaced_by_alias_after_read_rejected(
    package: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (package / "entry.py").write_text("value = 1\n", encoding="utf-8")
    relocated = package.parent / "relocated"
    original = inventory._read

    def changed(path: Path) -> tuple[bytes, inventory.Source]:
        result = original(path)
        if path.name == "entry.py":
            package.rename(relocated)
            package.symlink_to(relocated, target_is_directory=True)
        return result

    monkeypatch.setattr(inventory, "_read", changed)
    with pytest.raises(ValueError, match="during closure"):
        inventory.capture(("entry",), package=package)


def test_cli_json_pins_the_inventory_implementation(capsys: pytest.CaptureFixture[str]) -> None:
    assert inventory.main(["core", "--json"]) == 0
    value = json.loads(capsys.readouterr().out)
    assert str(inventory.PACKAGE / "core.py") in value["inputs"]
    assert str(inventory.PACKAGE / "code_dependency_inventory.py") in value["inputs"]


def test_cli_sha256_check_format(capsys: pytest.CaptureFixture[str]) -> None:
    assert inventory.main(["__init__"]) == 0
    lines = capsys.readouterr().out.splitlines()
    assert len(lines) == 2
    assert lines[0].split("  ")[1].endswith("/replay/__init__.py")
