"""Join many rendered parts through one hash-pinned manifest instead of one input per part.

`delivery_render.join` declares every rendered part as a separate stage input, and the
workflow plan caps each array at 256 entries, so a 916-part replay cannot be planned.
This module keeps the per-part guarantee but moves it into the join command: the stage
pins only the manifest, and `join_manifest` re-hashes every part before the lossless
copy-join runs. The pinned render and workflow modules stay untouched.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path

from replay import delivery_render, native_join
from replay.core import ReplayError
from replay.recordings import record
from replay.stages import file_digest

MANIFEST_VERSION = 1
MAX_MANIFEST_BYTES = 64 * 1024 * 1024
MAX_PARTS = 4096
MANIFEST_KEYS = frozenset({"version", "parts"})
PART_KEYS = frozenset({"sourcePath", "durationFrames", "sha256"})
DIGEST_LENGTH = 64
WORKFLOW_ARRAY_LIMIT = 256
JOIN_FIXED_INPUTS = 3  # join plan, cuts index, chunks index


@dataclass(frozen=True, slots=True)
class ManifestPart:
    path: Path
    frames: int
    sha256: str


def _text(value: object) -> str:
    if not isinstance(value, str) or not value or "\0" in value:
        raise ReplayError("Manifest values must be nonempty text")
    return value


def _frames(value: object) -> int:
    if type(value) is not int or value < 1:
        raise ReplayError("Manifest frame counts must be positive integers")
    return value


def _digest(value: object) -> str:
    text = _text(value)
    if len(text) != DIGEST_LENGTH or any(char not in "0123456789abcdef" for char in text):
        raise ReplayError("Manifest digests must be lowercase SHA-256")
    return text


def _part(raw: object) -> ManifestPart:
    if not isinstance(raw, dict) or set(raw) != PART_KEYS:
        raise ReplayError("Manifest part requires exactly path, frames and digest")
    path = Path(_text(raw["sourcePath"]))
    if not path.is_absolute():
        raise ReplayError("Manifest part paths must be absolute")
    return ManifestPart(path, _frames(raw["durationFrames"]), _digest(raw["sha256"]))


def parse_manifest(data: object) -> tuple[ManifestPart, ...]:
    """Validate a decoded manifest and return its ordered parts."""
    if not isinstance(data, dict) or set(data) != MANIFEST_KEYS:
        raise ReplayError("Join manifest requires exactly version and parts")
    if type(data["version"]) is not int or data["version"] != MANIFEST_VERSION:
        raise ReplayError("Unsupported join manifest version")
    rows = data["parts"]
    if not isinstance(rows, list) or not 1 <= len(rows) <= MAX_PARTS:
        raise ReplayError("Join manifest part count is outside its budget")
    parts = tuple(_part(row) for row in rows)
    if len({part.path for part in parts}) != len(parts):
        raise ReplayError("Join manifest repeats a part path")
    return parts


def write_manifest(path: Path, parts: Iterable[ManifestPart]) -> Path:
    """Write a new manifest; an existing file is never replaced."""
    rows = [
        {"sourcePath": str(part.path), "durationFrames": part.frames, "sha256": part.sha256}
        for part in parts
    ]
    parse_manifest({"version": MANIFEST_VERSION, "parts": rows})
    document = {"version": MANIFEST_VERSION, "parts": rows}
    with path.open("x", encoding="utf-8") as stream:
        json.dump(document, stream, sort_keys=True)
    return path


def pin_part(path: Path, frames: int) -> ManifestPart:
    """Pin one regular, non-symlink rendered movie by its current SHA-256."""
    if path.is_symlink() or not path.is_file():
        raise ReplayError("Join requires regular immutable native movies")
    return ManifestPart(path, frames, file_digest(path))


def _load(manifest: Path) -> tuple[ManifestPart, ...]:
    if manifest.is_symlink() or not manifest.is_file():
        raise ReplayError("Join manifest must be a regular file")
    if manifest.stat().st_size > MAX_MANIFEST_BYTES:
        raise ReplayError("Join manifest exceeds its byte budget")
    with manifest.open("rb") as stream:
        return parse_manifest(json.load(stream))


def verify_parts(parts: Sequence[ManifestPart]) -> tuple[native_join.PartFile, ...]:
    """Re-hash every part; any change since pinning stops the join."""
    for part in parts:
        if part.path.is_symlink() or not part.path.is_file():
            raise ReplayError("Join requires regular immutable native movies")
        if file_digest(part.path) != part.sha256:
            raise ReplayError(f"Pinned native part changed: {part.path.name}")
    return tuple(native_join.PartFile(part.path, part.frames) for part in parts)


def join_manifest(manifest: Path, output: Path, rate: int, audio: Path | None = None) -> int:
    """Verify the pinned parts, then run the existing lossless join in manifest order."""
    files = verify_parts(_load(manifest))
    return native_join.join(files, rate=rate, output=output, audio=audio)


def join_stage(
    manifest: Path, output: Path, rate: int, command: Sequence[str], extra_inputs: Sequence[Path]
) -> dict[str, object]:
    """Describe the join stage with a constant-size input list, whatever the part count."""
    return {
        "name": "join",
        "command": [
            *command,
            "join-manifest",
            str(manifest),
            "--output",
            str(output),
            "--rate",
            str(rate),
        ],
        "inputs": [str(manifest), *map(str, extra_inputs)],
        "outputs": [str(output)],
    }


def needs_manifest(part_count: int, evidence_count: int) -> bool:
    """True when the classic join stage cannot be planned within the workflow array cap.

    The classic stage lists plan, every part, cuts, chunks and cut evidence as inputs;
    only beyond the cap is the manifest form needed, so the classic form (which the cut
    audit inspects) stays in use everywhere else.
    """
    return part_count + JOIN_FIXED_INPUTS + evidence_count > WORKFLOW_ARRAY_LIMIT


def _rendered_part(request: Path, frames: int) -> ManifestPart:
    response = record(delivery_render._json(request.with_name("render.response.json")))
    if response.get("frameCount") != frames:
        raise ReplayError("Native part frame count differs from its plan")
    return pin_part(delivery_render._path(response.get("outputPath")), frames)


def generate_join(rendered: Iterable[tuple[Path, int]], config: delivery_render.Config) -> Path:
    """Write the manifest and a one-stage join workflow for the rendered parts, in order."""
    parts = [_rendered_part(request, frames) for request, frames in rendered]
    manifest = write_manifest(config.work / "join-manifest.json", parts)
    stage = join_stage(
        manifest,
        config.storage / "joined-video.mp4",
        config.rate,
        [sys.executable, "-m", "replay.mechanical_join"],
        [config.cuts, config.chunks, *config.cut_evidence],
    )
    stage["timeout_seconds"] = delivery_render.JOIN_TIMEOUT_SECONDS
    workflow = config.work / "join-workflow.json"
    with workflow.open("x", encoding="utf-8") as stream:
        json.dump({"version": 1, "stages": [stage]}, stream, sort_keys=True)
    return workflow


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="replay.mechanical_join")
    commands = parser.add_subparsers(dest="command", required=True)
    join = commands.add_parser("join-manifest")
    join.add_argument("manifest", type=Path)
    join.add_argument("--output", type=Path, required=True)
    join.add_argument("--rate", type=int, required=True)
    join.add_argument("--audio", type=Path)
    args = parser.parse_args(argv)
    return join_manifest(args.manifest, args.output, args.rate, args.audio)


if __name__ == "__main__":
    raise SystemExit(main())
