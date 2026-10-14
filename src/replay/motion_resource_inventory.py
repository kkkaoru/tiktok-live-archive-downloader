"""Read-only, bounded Motion XML reference inventory; never a render-closure proof."""

from __future__ import annotations

import math
import plistlib
import re
import xml.etree.ElementTree as ET
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal
from urllib.parse import unquote, urlsplit
from xml.sax.saxutils import escape

from replay.core import ReplayError
from replay.motion_caption import MAX_XML_CHARACTERS, _parse

MAX_DOCUMENTS = 32
MAX_FIELDS = 100_000
XML_SUFFIXES = frozenset({".moti", ".motn", ".motr", ".moef", ".preset", ".xml"})
REFERENCE_NAME = re.compile(r"(?:url|uri|path|file|resource|media|texture|image|source)", re.I)
URI = re.compile(r"^[a-zA-Z][a-zA-Z0-9+.-]*:")
Kind = Literal[
    "external-file",
    "graph-channel",
    "boolean",
    "enum",
    "empty-binding",
    "container",
    "font-name",
    "unknown",
]
# Observed ozml 5.13 text-style slots, not arbitrary numeric parameters.
TEXTURES = {"14": "18", "21": "25", "30": "34", "38": "42"}
SOURCES = {"14": "85", "21": "88", "30": "86", "38": "87"}
PRESET_KEYS = frozenset(
    {
        "Description",
        "tag",
        "DisplayName",
        "FieldDominance",
        "FrameRate",
        "Height",
        "IsRegionalDefault",
        "IsHidden",
        "PixelAspect",
        "Width",
    }
)


@dataclass(frozen=True)
class ResourceField:
    document: str
    element: int
    field: str
    label: str
    value: str
    status: str
    path: str | None
    kind: Kind
    reason: str
    evidence: str


@dataclass(frozen=True)
class ResourceInventory:
    documents: tuple[str, ...]
    fields: tuple[ResourceField, ...]
    tags: dict[str, int]
    attributes: dict[str, int]
    examined_fields: int
    # Lexical XML discovery cannot certify undocumented plugin/resource semantics.
    render_closure_verified: bool = False

    def report(self) -> dict[str, object]:
        return asdict(self)


def project_preset(path: Path) -> dict[str, object]:
    """Read a scalar project-settings plist, never a Motion/media document or DTD."""
    if path != path.resolve() or not path.is_file() or path.stat().st_size > MAX_XML_CHARACTERS:
        raise ReplayError("Project preset must be a bounded canonical file")
    try:
        value: object = plistlib.loads(path.read_bytes())
    except (ValueError, plistlib.InvalidFileException) as error:
        raise ReplayError("Invalid project preset plist") from error
    if not isinstance(value, dict) or set(value) != PRESET_KEYS:
        raise ReplayError("Unsupported project preset keys")
    result: dict[str, object] = {}
    for key, item in value.items():
        if not isinstance(key, str) or type(item) not in (str, int, float, bool):
            raise ReplayError("Project preset requires scalar settings")
        result[key] = item
    dimensions = (result["Width"], result["Height"])
    if (
        any(type(dimension) is not int or dimension <= 0 for dimension in dimensions)
        or not isinstance(result["FrameRate"], (int, float))
        or isinstance(result["FrameRate"], bool)
        or not 1 <= result["FrameRate"] <= 120
        or result["PixelAspect"] != 1
        or result["FieldDominance"] != 0
    ):
        raise ReplayError("Unsupported project preset canvas/clock")
    return result


def bind_project_preset(document: str, *, preset: Path, expected_previous: str) -> str:
    """Select a NEW compatible authoring preset; preserve every other XML byte.

    No historical identity or render equivalence is asserted. The caller publishes
    this copy exclusively and must reverify the actual prepared render.
    """
    root = _parse(document)
    settings = root.findall("scene/sceneSettings")
    if root.get("version") != "5.13" or len(settings) != 1:
        raise ReplayError("Unsupported preset binding document")
    nodes = settings[0].findall("presetPath")
    if len(nodes) != 1 or nodes[0].text != expected_previous:
        raise ReplayError("Preset binding source mismatch")
    data = project_preset(preset)
    try:
        width = int(settings[0].findtext("width", ""))
        height = int(settings[0].findtext("height", ""))
        rate = float(settings[0].findtext("frameRate", ""))
        ntsc = settings[0].findtext("NTSC")
        aspect = float(settings[0].findtext("pixelAspectRatio", ""))
        fields = int(settings[0].findtext("fieldRenderingMode", ""))
    except ValueError as error:
        raise ReplayError("Invalid preset binding scene settings") from error
    if (
        ntsc not in ("0", "1")
        or not math.isfinite(rate)
        or (
            aspect != data["PixelAspect"]
            or fields != data["FieldDominance"]
            or width != data["Width"]
            or height != data["Height"]
            or round(rate / (1.001 if ntsc == "1" else 1), 2) != data["FrameRate"]
        )
    ):
        raise ReplayError("Selected project preset is incompatible with authoring settings")
    old = f"<presetPath>{escape(expected_previous)}</presetPath>"
    if document.count(old) != 1:
        raise ReplayError("Preset binding requires one exact lexical slot")
    return document.replace(old, f"<presetPath>{escape(str(preset))}</presetPath>")


def _graph(root: ET.Element, element: ET.Element, value: str) -> bool:
    if re.fullmatch(r"\./[0-9]+(?:/[0-9]+)*", value) is None:
        return False
    identifier = element.get("object", "")
    nodes = [node for node in root.iter("scenenode") if node.get("id") == identifier]
    if not identifier.isdecimal() or len(nodes) != 1:
        return False
    node = nodes[0]
    parts = value[2:].split("/")
    if parts[0] == "5" and len(parts) > 1:
        styles = [style for style in node.findall("style") if style.get("id") == parts[1]]
        if len(styles) != 1:
            return False
        node, parts = styles[0], parts[2:]
    for part in parts:
        children = [child for child in node.findall("parameter") if child.get("id") == part]
        if len(children) != 1:
            return False
        node = children[0]
    return bool(parts)


def _semantic(
    root: ET.Element,
    element: ET.Element,
    name: str,
    value: str,
    parents: dict[ET.Element, ET.Element],
) -> tuple[Kind, str, str] | None:
    parent = parents.get(element)
    if element.tag == "target" and name == "channel":
        if (
            root.get("version") != "5.13"
            or parent not in root.findall("scene/publishSettings")
            or not _graph(root, element, value)
        ):
            return (
                "unknown",
                "invalid-graph",
                "Published target must resolve a unique numeric object/control chain",
            )
        return (
            "graph-channel",
            "internal",
            "Unique object and parameter chain; style namespace 5 is resolved locally",
        )
    if root.get("version") != "5.13" or parent is None:
        return None
    ancestors = []
    node = element
    while node in parents:
        node = parents[node]
        ancestors.append(node)
    if (
        not any(
            node.tag == "scenenode" and node.get("factoryID") == "16" and node.get("version") == "5"
            for node in ancestors
        )
        or root.find("scene") not in ancestors
    ):
        return None
    if (
        element.tag in ("font", "defaultFont")
        and parent.tag == "parameter"
        and parent.get("id") == "83"
        and re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,99}", value)
    ):
        return (
            "font-name",
            "requires-font-supplier",
            "Active face or fallback name is not a filesystem path",
        )
    if element.tag == "textPathModified" and parent.tag == "scenenode":
        if element.get("textPathModifiedVal") in ("0", "1") and not value:
            return "container", "internal", "Text-path modification flag container"
        if name == "textPathModifiedVal" and value in ("0", "1"):
            return "boolean", "internal", "ozml 5.13 text-path modification flag"
    if element.tag != "parameter" or name not in ("parameter", "value"):
        return None
    label, identifier = element.get("name"), element.get("id")
    parent_id = parent.get("id", "")
    in_style = any(node.tag == "style" for node in ancestors)
    if in_style and parent.tag == "parameter":
        if label == "Texture" and identifier == TEXTURES.get(parent_id) and not value:
            return (
                "container",
                "internal",
                "Text appearance texture group; child bindings scanned separately",
            )
        if (
            label == "Source Media"
            and identifier == SOURCES.get(parent_id)
            and not value
            and not element.attrib.get("value")
        ):
            return (
                "empty-binding",
                "internal",
                "Unassigned text appearance media slot; all children still scanned",
            )
        if (
            label == "Image"
            and identifier == "1"
            and parent.get("name") == "Texture"
            and (grandparent := parents.get(parent)) is not None
            and parent_id == TEXTURES.get(grandparent.get("id", ""))
            and element.get("value") == "0"
            and value in ("", "0")
        ):
            return "empty-binding", "internal", "Texture image object ID zero denotes no binding"
        if (
            label == "Wrap Mode"
            and identifier == "5"
            and parent.get("name") == "Texture"
            and (grandparent := parents.get(parent)) is not None
            and parent_id == TEXTURES.get(grandparent.get("id", ""))
            and element.get("value") in ("0", "1")
            and value in ("", "0", "1")
        ):
            return "enum", "internal", "Observed texture wrap-mode scalar; not a file binding"
    if (
        parent.tag == "parameter"
        and parent_id == "2"
        and label == "Path Options"
        and identifier == "329"
        and not value
    ):
        return (
            "container",
            "internal",
            "Text object path options group; children independently validated",
        )
    if parent.tag == "parameter" and parent_id == "329" and parent.get("name") == "Path Options":
        if (
            label == "Shape Source"
            and identifier == "339"
            and element.get("value") == "0"
            and value in ("", "0")
        ):
            return "empty-binding", "internal", "Text-path shape object ID zero denotes no binding"
        if (
            label == "Align to Path"
            and identifier == "333"
            and element.get("value") in ("0", "1")
            and value in ("", "0", "1")
        ):
            return "boolean", "internal", "Text path alignment boolean, not an asset name"
    return None


def _reference(value: str, *, document: Path, roots: tuple[Path, ...]) -> tuple[str, Path | None]:
    if not value:
        return "empty-slot", None
    if URI.match(value):
        uri = urlsplit(value)
        if uri.scheme != "file" or uri.netloc or uri.query or uri.fragment:
            return "unresolved-uri", None
        path = Path(unquote(uri.path))
    elif value.startswith(("/", "./", "../")):
        path = Path(value)
    else:
        return "unresolved-symbolic-value", None
    if not path.is_absolute():
        path = document.parent / path
    canonical = path.resolve()
    if not any(canonical.is_relative_to(root) for root in roots):
        return "outside-scope", canonical
    if path.absolute() != canonical:
        return "noncanonical-or-linked", canonical
    if not canonical.is_file():
        return "missing", canonical
    return "resolved", canonical


def inventory(template: Path, *, allowed_roots: tuple[Path, ...]) -> ResourceInventory:
    """Follow only exact, in-scope file references; retain unknown/absent references.

    All element text and attributes are scanned. Named resource slots are retained
    even if empty or symbolic. Embedded text may cause conservative false positives.
    No app, font registry, network, or output file is touched.
    """
    roots = tuple(root.absolute() for root in allowed_roots)
    if not roots or any(root != root.resolve() for root in roots):
        raise ReplayError("Resource inventory requires canonical explicit scope roots")
    pending = [template.absolute()]
    seen: set[Path] = set()
    fields: list[ResourceField] = []
    tags: Counter[str] = Counter()
    attributes: Counter[str] = Counter()
    examined = 0
    while pending:
        path = pending.pop()
        if path in seen:
            continue
        if len(seen) >= MAX_DOCUMENTS:
            raise ReplayError("Resource XML document budget exceeded")
        if path != path.resolve() or not any(path.is_relative_to(root) for root in roots):
            raise ReplayError("Resource XML path is linked or outside scope")
        if path.stat().st_size > MAX_XML_CHARACTERS:
            raise ReplayError("Resource XML byte budget exceeded")
        if path.suffix.lower() == ".preset":
            project_preset(path)
            seen.add(path)
            continue
        root = _parse(path.read_text(encoding="utf-8"))
        parents = {child: parent for parent in root.iter() for child in parent}
        seen.add(path)
        for index, element in enumerate(root.iter()):
            if not isinstance(element.tag, str):
                continue  # Preserved XML comments are not executable resource references.
            tags[element.tag] += 1
            attributes.update(element.attrib.keys())
            values = [(element.tag, (element.text or "").strip()), *element.attrib.items()]
            for name, value in values:
                examined += 1
                if examined > MAX_FIELDS:
                    raise ReplayError("Resource XML field budget exceeded")
                label = element.attrib.get("name", "") if name in ("value", element.tag) else ""
                if not (
                    REFERENCE_NAME.search(name + label)
                    or element.tag in ("font", "defaultFont")
                    or (element.tag == "parameter" and label == "Wrap Mode")
                    or (element.tag == "target" and name == "channel")
                    or URI.match(value)
                    or value.startswith(("/", "./", "../"))
                ):
                    continue
                semantic = _semantic(root, element, name, value, parents)
                kind: Kind = "unknown"
                reason = "Lexical candidate has no supported internal-slot contract"
                target = None
                if semantic is not None:
                    kind, status, reason = semantic
                else:
                    status, target = _reference(value, document=path, roots=roots)
                    if target is not None:
                        kind = "external-file"
                        reason = "Explicit file reference; scene preset retained as a dependency"
                evidence = (
                    "/".join(
                        (
                            str(root.tag),
                            str(parent.tag) if (parent := parents.get(element)) is not None else "",
                            str(element.tag),
                        )
                    )
                    + f" id={element.get('id', '')}"
                    + f" parentID={parent.get('id', '') if parent is not None else ''}"
                    + f" ozml={root.get('version', '')}"
                )
                fields.append(
                    ResourceField(
                        str(path),
                        index,
                        name,
                        label,
                        value,
                        status,
                        str(target) if target is not None else None,
                        kind,
                        reason,
                        evidence,
                    )
                )
                if (
                    status == "resolved"
                    and target is not None
                    and target.suffix.lower() in XML_SUFFIXES
                ):
                    pending.append(target)
    return ResourceInventory(
        tuple(sorted(map(str, seen))), tuple(fields), dict(tags), dict(attributes), examined
    )
