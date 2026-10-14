"""Read exact edited clock expectations without inventing a render configuration.

This metadata loader neither processes media nor approves delivery or sync. A
consumer must preserve its complete decoder, packet and initial/terminal input
checks and declare the returned input closure in its workflow.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from xml.parsers import expat

from replay.caption_font_supplier import MAX_FCPXML_BYTES, MAX_TITLES, _doctype
from replay.caption_measured_job import _json as strict_json
from replay.core import ReplayError
from replay.delivery_render import read_cuts
from replay.edit_completion import CONFIG_NAME, load_config
from replay.fcp_caption_srt import _time, caption_cues
from replay.foreground_evidence import _bound_bytes
from replay.recordings import record
from replay.tracking_observations import _read

ROOT = Path(__file__).resolve().parents[2]
MAX_CONTRACT_BYTES = 65536
MAX_JSON_BYTES = 8 * 1024 * 1024
MAX_FRAMES = 4_000_000
MAX_SECONDS = 86400
MAX_DIMENSION = 16384
MAX_RATE = 240
MAX_XML_DEPTH = 64
MAX_XML_ELEMENTS = MAX_TITLES * 64
CONTRACT_FIELDS = frozenset({"version", "kind", "editConfig", "cuts", "semanticXML"})
SEQUENCE_PATH = ("fcpxml", "library", "event", "project", "sequence")


@dataclass(frozen=True, slots=True)
class BoundInput:
    path: Path
    sha256: str


@dataclass(frozen=True, slots=True)
class ClockExpectation:
    path: Path
    cuts: Path
    rate: int
    width: int
    height: int
    delivery_root: Path
    expected_frames: int
    inputs: tuple[BoundInput, ...]


class _Geometry:
    def __init__(self) -> None:
        self.stack: list[str] = []
        self.elements = 0
        self.formats: dict[str, dict[str, str]] = {}
        self.sequence: dict[str, str] | None = None

    def start(self, name: str, attributes: dict[str, str]) -> None:
        path = (*self.stack, name)
        self.elements += 1
        if len(path) > MAX_XML_DEPTH or self.elements > MAX_XML_ELEMENTS:
            raise ReplayError("Clock XML exceeds its structural budget")
        if not self.stack and name != "fcpxml":
            raise ReplayError("Clock XML must have an FCPXML root")
        if path == ("fcpxml", "resources", "format"):
            identifier = attributes.get("id", "")
            if not identifier or identifier in self.formats:
                raise ReplayError("Clock XML format identifiers must be unique and nonempty")
            self.formats[identifier] = attributes
        if name == "sequence":
            if path != SEQUENCE_PATH or self.sequence is not None:
                raise ReplayError("Clock XML requires exactly one direct project sequence")
            if attributes.get("tcStart", "0s") != "0s":
                raise ReplayError("Clock XML sequence must start at zero")
            self.sequence = attributes
        self.stack.append(name)

    def end(self, _name: str) -> None:
        self.stack.pop()


def _local(value: object, *, work_only: bool = False) -> Path:
    if not isinstance(value, str) or not value or "\0" in value:
        raise ReplayError("Clock inputs need canonical absolute paths")
    path = Path(value)
    allowed = ROOT / "edits" / "work" if work_only else ROOT / "edits"
    if (
        not path.is_absolute()
        or path != path.resolve()
        or not path.is_relative_to(allowed)
        or any(part in {"private", "downloads"} for part in path.parts)
    ):
        raise ReplayError("Clock inputs must be unlinked files in authorized checkout storage")
    return path


def _object(data: bytes) -> dict[str, object]:
    try:
        return record(strict_json(data))
    except ValueError as error:
        raise ReplayError("Clock JSON must be a strict object") from error


def _read_pin(value: object, *, limit: int) -> tuple[BoundInput, bytes]:
    row = record(value)
    if set(row) != {"path", "sha256"}:
        raise ReplayError("Clock input references need exactly path and SHA256")
    path = _local(row["path"])
    digest = row["sha256"]
    if not isinstance(digest, str) or re.fullmatch(r"[0-9a-f]{64}", digest) is None:
        raise ReplayError("Clock input needs an exact SHA256")
    return BoundInput(path, digest), _bound_bytes(path, digest, limit)


def _integer(value: object, *, maximum: int) -> int:
    if type(value) is not int or not 0 < value <= maximum:
        raise ReplayError("Clock counts and dimensions require bounded positive integers")
    return value


def _dimension(value: str | None) -> int:
    if value is None or re.fullmatch(r"[1-9][0-9]{0,4}", value) is None:
        raise ReplayError("Clock XML needs canonical positive dimensions")
    return _integer(int(value), maximum=MAX_DIMENSION)


def _geometry(data: bytes) -> tuple[int, int, int, Fraction]:
    geometry = _Geometry()
    parser = expat.ParserCreate()
    parser.StartElementHandler = geometry.start
    parser.EndElementHandler = geometry.end
    parser.StartDoctypeDeclHandler = _doctype
    try:
        parser.Parse(data, True)
    except expat.ExpatError as error:
        raise ReplayError("Clock XML is invalid") from error
    sequence = geometry.sequence
    if sequence is None or sequence.get("format") not in geometry.formats:
        raise ReplayError("Clock XML needs one referenced sequence format")
    format_row = geometry.formats[sequence["format"]]
    frame = _time(format_row, "frameDuration")
    if frame <= 0 or (1 / frame).denominator != 1:
        raise ReplayError("Clock XML needs an integer positive frame rate")
    rate = _integer((1 / frame).numerator, maximum=MAX_RATE)
    return (
        rate,
        _dimension(format_row.get("width")),
        _dimension(format_row.get("height")),
        _time(sequence, "duration"),
    )


def _expected(path: Path, data: bytes, *, rate: int) -> int:
    raw = _object(data)
    if (
        raw.get("clock") != "source-presentation"
        or type(raw.get("fps")) is not int
        or raw.get("fps") != rate
    ):
        raise ReplayError("Cut presentation clock and frame rate must match the semantic XML")
    rows = raw.get("cuts")
    if not isinstance(rows, list) or not 0 < len(rows) <= MAX_FRAMES:
        raise ReplayError("Clock cut rows must be a bounded nonempty array")
    cuts = read_cuts(path)  # Reuse the shared ordered, nonoverlapping source-cut validation.
    total = 0
    for cut, value in zip(cuts, rows, strict=True):
        start = record(value).get("output_start_frame")
        if type(start) is not int or start != total:
            raise ReplayError("Cut output clock must have exact contiguous offsets")
        total += cut.end - cut.start
    if not 0 < total <= MAX_FRAMES or Fraction(total, rate) > MAX_SECONDS:
        raise ReplayError("Clock expected frames or duration exceed the decoder budget")
    return total


def load_clock_contract(path: Path) -> ClockExpectation:
    """Derive one pinned edited-output expectation; never write or decode media."""
    path = _local(str(path), work_only=True)
    data = _read(path, MAX_CONTRACT_BYTES)
    raw = _object(data)
    if (
        set(raw) != CONTRACT_FIELDS
        or type(raw["version"]) is not int
        or raw["version"] != 1
        or raw["kind"] != "delivery-clock-expectation"
    ):
        raise ReplayError("Unsupported or incomplete clock expectation schema")
    contract = BoundInput(path, hashlib.sha256(data).hexdigest())
    edit_ref, edit_data = _read_pin(raw["editConfig"], limit=MAX_JSON_BYTES)
    cuts_ref, cuts_data = _read_pin(raw["cuts"], limit=MAX_JSON_BYTES)
    xml_ref, xml_data = _read_pin(raw["semanticXML"], limit=MAX_FCPXML_BYTES)
    if len({path, edit_ref.path, cuts_ref.path, xml_ref.path}) != 4:
        raise ReplayError("Clock input roles must name distinct files")
    if edit_ref.path.name != CONFIG_NAME:
        raise ReplayError("Clock edit configuration must be named edit-config.json")
    _object(edit_data)
    edit = load_config(edit_ref.path.parent)
    if edit.storage_root is None:
        raise ReplayError("Clock edit configuration requires authorized delivery storage")
    rate, width, height, duration = _geometry(xml_data)
    caption_cues(xml_ref.path)  # Keep existing bounded direct edit/caption-domain validation.
    frames = _expected(cuts_ref.path, cuts_data, rate=rate)
    if duration != Fraction(frames, rate):
        raise ReplayError("Semantic XML duration differs from the exact edited cut clock")
    inputs = (contract, edit_ref, cuts_ref, xml_ref)
    budgets = (MAX_CONTRACT_BYTES, MAX_JSON_BYTES, MAX_JSON_BYTES, MAX_FCPXML_BYTES)
    for ref, budget in zip(inputs, budgets, strict=True):
        _bound_bytes(ref.path, ref.sha256, budget)
    return ClockExpectation(
        path, cuts_ref.path, rate, width, height, edit.storage_root, frames, inputs
    )
