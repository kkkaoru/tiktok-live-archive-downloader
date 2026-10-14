"""Inspect and derive encoding-only Compressor presets from pinned native samples."""

import argparse
import base64
import hashlib
import json
import plistlib
from pathlib import Path
from xml.etree import ElementTree as ET

from replay.artifact_json import _json, _pin
from replay.core import ReplayError
from replay.recordings import record

MAX_BYTES = 1024 * 1024
EDITABLE = frozenset(
    {
        "description",
        "nameKey",
        "descriptionKey",
        "encoder/file-extension",
        "encoder/encode-cc",
        "encoder/encode-chapters",
        "encoder/auto-bit-rate-purpose",
        "encoder/optimize-for-network",
        "encoder/audio-encode/audio-format-info",
        "encoder/audio-encode/auto-channel_layout",
        "encoder/audio-encode/auto-format_sampleRate",
        "encoder/audio-encode/quality",
        "encoder/audio-encode/audio-encoding-strategy",
        "encoder/audio-encode/audio-encoding-bitrate",
        "encoder/video-encode/bounds",
        "encoder/video-encode/frame-rate",
        "encoder/video-encode/automatic",
        "encoder/video-encode/video-conversion",
        "encoder/video-encode/color-spec",
        "encoder/video-encode/color-space",
        "encoder/video-encode/spatial",
        "encoder/video-encode/temporal",
        "encoder/video-encode/icm2-options",
        "encoder/video-encode/data-rate",
        "encoder/video-encode/data-rate-limit-size",
        "encoder/video-encode/data-rate-limit-duration",
        "encoder/video-encode/bit-rate-mode",
        "encoder/video-encode/max-frame-rate",
    }
)


def read(path: Path) -> ET.Element:
    if path.is_symlink() or not path.is_file() or path.stat().st_size > MAX_BYTES:
        raise ReplayError("Expected a bounded regular native preset")
    root = ET.fromstring(path.read_bytes())
    if root.tag != "setting" or len(root.findall("encoder")) != 1:
        raise ReplayError("Expected one Compressor setting encoder")
    if any("action" in element.tag.lower() for element in root.iter()):
        raise ReplayError("Encoding presets must not include job actions")
    filters = root.find("filter-set")
    if filters is None or len(filters):
        raise ReplayError("Expected an encoding-only preset without filters")
    return root


def property_envelope(root: ET.Element) -> dict[str, object]:
    node = root.find("encoder/video-encode/encoder-properties")
    if node is None or not node.text:
        raise ReplayError("Native preset has no video encoder properties")
    decoded: object = plistlib.loads(base64.b64decode(node.text, validate=True))
    return record(decoded)


def properties(root: ET.Element) -> dict[str, object]:
    result = property_envelope(root)
    return record(result["codec-properties"]) if "codec-properties" in result else result


def inspect(path: Path) -> dict[str, object]:
    root = read(path)
    values: dict[str, object] = {}
    for selector in sorted(EDITABLE | {"encoder", "encoder/audio-encode", "encoder/video-encode"}):
        node = root.find(selector)
        if node is not None:
            values[selector] = {"text": node.text or "", "attributes": dict(node.attrib)}
    return {
        "path": str(path.resolve()),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "name": root.get("name"),
        "values": values,
        "codecProperties": properties(root),
        "nativeApplicationReadback": False,
        "encodedArtifactVerified": False,
    }


def derive(config_path: Path, output: Path) -> dict[str, object]:
    """Apply explicit native-field edits, requiring the exact observed old value."""
    config = _json(config_path)
    if config.get("version") != 1:
        raise ReplayError("Unsupported preset derivation version")
    source = _pin(config.get("source"))
    root = read(source)
    encoder = root.find("encoder")
    if encoder is None or encoder.get("name") != "MPEG4":
        raise ReplayError("MP4 derivation requires a native MPEG4 preset")
    name = config.get("name")
    edits = config.get("edits")
    if not isinstance(name, str) or not name or not isinstance(edits, list):
        raise ReplayError("Expected a preset name and native edits")
    root.set("name", name)
    seen: set[str] = set()
    for raw in edits:
        edit = record(raw)
        selector = edit.get("path")
        if not isinstance(selector, str) or selector not in EDITABLE or selector in seen:
            raise ReplayError("Unknown or duplicate encoding field")
        seen.add(selector)
        node = root.find(selector)
        if node is None:
            raise ReplayError("Encoding field is absent from native source sample")
        before = {"text": node.text or "", "attributes": dict(node.attrib)}
        if edit.get("before") != before:
            raise ReplayError("Native encoding field differs from observed value")
        after = record(edit.get("after"))
        text, attributes = after.get("text"), record(after.get("attributes"))
        if not isinstance(text, str) or any(not isinstance(v, str) for v in attributes.values()):
            raise ReplayError("Native field values must be text")
        node.text = text
        node.attrib.clear()
        for key, value in attributes.items():
            if not isinstance(value, str):
                raise ReplayError("Native field values must be text")
            node.set(key, value)
    donor = config.get("profileSource")
    if donor is not None:
        source_properties = properties(read(_pin(donor)))
        wanted: dict[str, object] = {
            key: source_properties[key]
            for key in ("H264EntropyMode", "ProfileLevel")
            if key in source_properties
        }
        if len(wanted) != 2:
            raise ReplayError("Native donor lacks H264 profile or entropy")
        target = root.find("encoder/video-encode/encoder-properties")
        if target is None:
            raise ReplayError("Native destination lacks codec properties")
        envelope = property_envelope(root)
        destination = (
            record(envelope["codec-properties"]) if "codec-properties" in envelope else envelope
        )
        destination.update(wanted)
        if "codec-properties" in envelope:
            envelope["codec-properties"] = destination
        target.text = base64.b64encode(plistlib.dumps(envelope, fmt=plistlib.FMT_BINARY)).decode(
            "ascii"
        )
    if root.findtext("encoder/file-extension") != "mp4":
        raise ReplayError("Native container extension must remain mp4")
    with output.open("xb") as handle:
        handle.write(ET.tostring(root, encoding="utf-8", xml_declaration=True))
    return inspect(output)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="action", required=True)
    show = commands.add_parser("inspect")
    show.add_argument("preset", type=Path)
    build = commands.add_parser("derive")
    build.add_argument("config", type=Path)
    build.add_argument("output", type=Path)
    args = parser.parse_args()
    report = inspect(args.preset) if args.action == "inspect" else derive(args.config, args.output)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
