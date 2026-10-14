"""Variable-font source identity and the selected static render weight are required."""

import json
import struct
from pathlib import Path

import pytest

from replay.caption_font_instance import inspect_instance, main, materialize
from replay.core import ReplayError
from replay.stages import file_digest

FONT = struct.pack(">IHHHH", 0x10000, 1, 16, 0, 0) + b"glyf" + bytes(12)
METADATA = (
    '<ttFont><OS_2><usWeightClass value="700"/></OS_2><name>'
    '<namerecord nameID="1">Noto Sans JP</namerecord>'
    '<namerecord nameID="2">Bold</namerecord>'
    '<namerecord nameID="6">NotoSansJP-Bold</namerecord></name></ttFont>'
)


def execute(argv: tuple[str, ...], log: Path, timeout: int) -> None:
    assert timeout in (60, 600)
    output = Path(argv[argv.index("-o") + 1])
    if "fontTools.ttx" in argv:
        output.write_text(METADATA)
    else:
        output.write_bytes(FONT)
    log.write_text("synthetic command success")


def test_materialize_source_unchanged_and_pinned(tmp_path: Path) -> None:
    source = tmp_path / "NotoSansJP-VariableFont_wght.ttf"
    source.write_bytes(b"variable synthetic source")
    receipt = materialize(
        source, tmp_path / "instance", expected_sha256=file_digest(source), execute=execute
    )
    data = json.loads(receipt.read_text())
    assert data["sourceUnchanged"] is True
    assert data["weight"] == 700
    assert data["postScriptName"] == "NotoSansJP-Bold"
    assert source.read_bytes() == b"variable synthetic source"


def test_reject_unpinned_source(tmp_path: Path) -> None:
    source = tmp_path / "NotoSansJP-VariableFont_wght.ttf"
    source.write_bytes(b"source")
    with pytest.raises(ReplayError, match="identity"):
        materialize(source, tmp_path / "instance", expected_sha256="0" * 64, execute=execute)


@pytest.mark.parametrize("tag,weight", [(b"fvar", "700"), (b"glyf", "100")])
def test_reject_unresolved_axis_and_thin_weight(tmp_path: Path, tag: bytes, weight: str) -> None:
    font, metadata = tmp_path / "font.ttf", tmp_path / "font.ttx"
    font.write_bytes(struct.pack(">IHHHH", 0x10000, 1, 16, 0, 0) + tag + bytes(12))
    metadata.write_text(METADATA.replace('value="700"', f'value="{weight}"'))
    with pytest.raises(ReplayError):
        inspect_instance(font, metadata)


@pytest.mark.parametrize("payload", [b"short", bytes(12), FONT[:-1]])
def test_reject_invalid_table_extent(tmp_path: Path, payload: bytes) -> None:
    font, metadata = tmp_path / "font.ttf", tmp_path / "font.ttx"
    font.write_bytes(payload)
    metadata.write_text(METADATA)
    with pytest.raises(ReplayError):
        inspect_instance(font, metadata)


def test_reject_missing_metadata_and_unsafe_name(tmp_path: Path) -> None:
    font, metadata = tmp_path / "font.ttf", tmp_path / "font.ttx"
    font.write_bytes(FONT)
    metadata.write_text("")
    with pytest.raises(ReplayError):
        inspect_instance(font, metadata)
    metadata.write_text(METADATA.replace("NotoSansJP-Bold", "injected,style"))
    with pytest.raises(ReplayError):
        inspect_instance(font, metadata)


def test_source_mutation_is_rejected(tmp_path: Path) -> None:
    source = tmp_path / "NotoSansJP-VariableFont_wght.ttf"
    source.write_bytes(b"source")

    def mutate(argv: tuple[str, ...], log: Path, timeout: int) -> None:
        execute(argv, log, timeout)
        source.write_bytes(b"changed")

    with pytest.raises(ReplayError, match="changed"):
        materialize(
            source, tmp_path / "instance", expected_sha256=file_digest(source), execute=mutate
        )


def test_cli_returns_receipt(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "replay.caption_font_instance.materialize",
        lambda *args, **kwargs: tmp_path / "receipt.json",
    )
    assert main([str(tmp_path / "source.ttf"), str(tmp_path / "out"), "--sha256", "0" * 64]) == 0
