"""Resolve the selected variable font weight into a private libass render instance.

The source asset stays unchanged. This avoids CoreText/libass silently choosing
the variable font's default Thin face or an unrelated Latin fallback.
"""

import argparse
import json
import re
import struct
import sys
import xml.etree.ElementTree as ET
from collections.abc import Callable
from pathlib import Path

from replay import caption_style_defaults as style
from replay.core import ReplayError
from replay.stages import execute_command, file_digest

MAX_FONT_BYTES = 64 * 1024 * 1024


def inspect_instance(font: Path, metadata: Path) -> str:
    """Require a static Bold weight and an unambiguous, ASS-safe native name."""
    if not font.is_file() or font.is_symlink() or not 12 <= font.stat().st_size <= MAX_FONT_BYTES:
        raise ReplayError("Invalid font instance artifact")
    with font.open("rb") as stream:
        header = stream.read(12)
        tables = struct.unpack_from(">H", header, 4)[0]
        if not 1 <= tables <= 128:
            raise ReplayError("Invalid font table budget")
        directory = stream.read(tables * 16)
    if len(directory) != tables * 16:
        raise ReplayError("Font instance retains unresolved variation axes")
    tags = {directory[index : index + 4] for index in range(0, len(directory), 16)}
    if b"fvar" in tags:
        raise ReplayError("Font instance retains unresolved variation axes")
    if metadata.is_symlink() or not 0 < metadata.stat().st_size <= 2 * 1024 * 1024:
        raise ReplayError("Invalid font metadata artifact")
    root = ET.parse(metadata).getroot()
    weight = root.find("OS_2/usWeightClass")
    names = root.findall("name/namerecord")
    families = {str(node.text).strip() for node in names if node.get("nameID") in {"1", "16"}}
    faces = {str(node.text).strip() for node in names if node.get("nameID") in {"2", "17"}}
    postscript = {str(node.text).strip() for node in names if node.get("nameID") == "6"}
    if (
        weight is None
        or weight.get("value") != str(int(style.FONT_WEIGHT))
        or style.FONT_FAMILY not in families
        or style.FONT_FACE not in faces
        or len(postscript) != 1
    ):
        raise ReplayError("Font instance differs from selected source family/weight")
    name = postscript.pop()
    if re.fullmatch(r"[A-Za-z0-9_-]{1,100}", name) is None:
        raise ReplayError("Invalid PostScript font instance name")
    return name


def materialize(
    source: Path,
    output: Path,
    *,
    expected_sha256: str,
    execute: Callable[[tuple[str, ...], Path, int], None] = execute_command,
) -> Path:
    """Pin the source, interpolate wght700 once, and preserve exact tool evidence."""
    if (
        source.name != style.FONT_FILE_NAME
        or source.is_symlink()
        or file_digest(source) != expected_sha256
    ):
        raise ReplayError("Selected variable font source identity differs")
    output.mkdir(mode=0o700)
    font_directory = output / "fonts"
    font_directory.mkdir(mode=0o700)
    font, metadata = font_directory / "render-instance.ttf", output / "render-instance.ttx"
    execute(
        (
            sys.executable,
            "-B",
            "-m",
            "fontTools.varLib.instancer",
            str(source),
            f"wght={style.FONT_WEIGHT:g}",
            "--static",
            "--update-name-table",
            "--no-recalc-timestamp",
            "-o",
            str(font),
        ),
        output / "instance.log",
        600,
    )
    execute(
        (
            sys.executable,
            "-B",
            "-m",
            "fontTools.ttx",
            "-t",
            "name",
            "-t",
            "OS/2",
            "-o",
            str(metadata),
            str(font),
        ),
        output / "metadata.log",
        60,
    )
    name = inspect_instance(font, metadata)
    if file_digest(source) != expected_sha256:
        raise ReplayError("Variable font changed during instancing")
    receipt = output / "receipt.json"
    receipt.write_text(
        json.dumps(
            {
                "version": 1,
                "source": str(source),
                "sourceSHA256": expected_sha256,
                "sourceUnchanged": True,
                "weightAxis": "wght",
                "weight": style.FONT_WEIGHT,
                "family": style.FONT_FAMILY,
                "face": style.FONT_FACE,
                "postScriptName": name,
                "renderInstance": str(font),
                "renderInstanceSHA256": file_digest(font),
                "metadataSHA256": file_digest(metadata),
                "producerSHA256": file_digest(Path(__file__)),
                "fontRasterVerified": False,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return receipt


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--sha256", required=True)
    args = parser.parse_args(argv)
    print(materialize(args.source, args.output, expected_sha256=args.sha256))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
