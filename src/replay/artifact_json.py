"""Canonical pinned JSON artifacts, independent of caption/render imports."""

import json
from pathlib import Path

from replay.core import ReplayError
from replay.foreground_evidence import _bound_bytes
from replay.recordings import record
from replay.stages import file_digest

MAX_JSON_BYTES = 8 * 1024 * 1024
MAX_REFLOW_JSON_BYTES = 64 * 1024 * 1024


def _path(value: object) -> Path:
    if not isinstance(value, str) or not value or "\x00" in value:
        raise ReplayError("Expected canonical absolute path")
    path = Path(value)
    if not path.is_absolute() or path != path.resolve():
        raise ReplayError("Expected canonical absolute path")
    return path


def _json(path: Path) -> dict[str, object]:
    return record(json.loads(_bound_bytes(path, file_digest(path), MAX_JSON_BYTES)))


def caption_reflow_json(path: Path) -> dict[str, object]:
    """Read caption reflow lineage within the existing retarget reader's budget.

    Ordinary configuration artifacts retain their smaller limit. This reader
    checks storage, bytes and JSON shape; the reflow consumer must still verify
    its layout schema, input pins, caption count and source clock coverage.
    """
    return record(json.loads(_bound_bytes(path, file_digest(path), MAX_REFLOW_JSON_BYTES)))


def _pin(value: object) -> Path:
    row = record(value)
    path = _path(row.get("path"))
    if row.get("sha256") != file_digest(path):
        raise ReplayError("Assembly input hash changed")
    return path
