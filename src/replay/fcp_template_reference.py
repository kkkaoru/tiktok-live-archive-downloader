"""Create a pinned UID-only FCPXML candidate; native acceptance remains unverified."""

import argparse
import copy
import json
from pathlib import Path
from xml.etree import ElementTree as ET

from replay.artifact_json import _json, _path, _pin
from replay.core import ReplayError
from replay.fcp_output_window import MAX_XML_BYTES, MAX_XML_ELEMENTS
from replay.foreground_evidence import _bound_bytes
from replay.recordings import record
from replay.stages import file_digest


def _local(path: Path) -> Path:
    root = Path(__file__).resolve().parents[2]
    if not path.is_relative_to(root) or path != path.resolve():
        raise ReplayError("Template reference paths must stay inside the checkout")
    return path


def _xml(path: Path, digest: str) -> ET.Element:
    content = _bound_bytes(path, digest, MAX_XML_BYTES).replace(b"<!DOCTYPE fcpxml>", b"")
    if b"<!DOCTYPE" in content or b"<!ENTITY" in content:
        raise ReplayError("External XML declarations are not supported")
    root = ET.fromstring(content)
    if root.tag != "fcpxml" or sum(1 for _ in root.iter()) > MAX_XML_ELEMENTS:
        raise ReplayError("Expected bounded FCPXML")
    return root


def _observed(value: object, template: Path, inputs: dict[str, str]) -> dict[str, str]:
    data = record(value)
    native, target = (_local(_pin(data.get(key))) for key in ("roundtrip", "targetTemplate"))
    if target.suffix != ".moti" or file_digest(target) != inputs[str(template)]:
        raise ReplayError("Observed Motion template bytes differ from source template")
    inputs.update({str(path): file_digest(path) for path in (native, target)})
    root = _xml(native, inputs[str(native)])
    effects = [e for e in root.findall("./resources/effect") if e.get("id") == data.get("effectID")]
    if len(effects) != 1:
        raise ReplayError("Expected exactly one observed effect")
    effect = effects[0]
    uid, name = effect.get("uid"), effect.get("name")
    if (
        not uid
        or uid != data.get("expectedUID")
        or not name
        or effect.get("src") != target.as_uri()
    ):
        raise ReplayError("Observed effect identity or template source does not match")
    return {"uid": uid, "src": target.as_uri(), "name": name}


def rewrite(config_path: Path, output: Path, proof: Path) -> dict[str, object]:
    """Change exactly one pinned effect UID and, optionally, the project name."""
    config_path, output, proof = (_local(_path(str(p))) for p in (config_path, output, proof))
    config = _json(config_path)
    if config.get("version") != 1:
        raise ReplayError("Expected template reference version 1")
    source, template = (_local(_pin(config.get(k))) for k in ("source", "template"))
    if template.suffix != ".moti":
        raise ReplayError("Expected a pinned Motion template")
    if output == proof or output.exists() or proof.exists():
        raise ReplayError("Template reference outputs must be distinct and new")
    inputs = {str(p): file_digest(p) for p in (config_path, source, template, Path(__file__))}
    root = _xml(source, inputs[str(source)])
    effects = [
        e for e in root.findall("./resources/effect") if e.get("id") == config.get("effectID")
    ]
    if len(effects) != 1 or not isinstance(config.get("expectedUID"), str):
        raise ReplayError("Expected one identified effect and its original UID")
    effect = effects[0]
    if effect.get("uid") != config["expectedUID"] or effect.get("src") != template.as_uri():
        raise ReplayError("Template source or original UID does not match")
    projects = root.findall("./library/event/project")
    if len(projects) != 1:
        raise ReplayError("Expected one project")
    name = config.get("projectName")
    if name is not None and (not isinstance(name, str) or not name.strip() or len(name) > 256):
        raise ReplayError("Expected a bounded nonempty project name")
    before = ET.tostring(root)
    old_name = projects[0].get("name")
    old_attributes = dict(effect.attrib)
    attributes = (
        _observed(config["observedEffect"], template, inputs)
        if "observedEffect" in config
        else {"uid": str(template)}
    )
    effect.attrib.update(attributes)
    if isinstance(name, str):
        projects[0].set("name", name)
    candidate = copy.deepcopy(root)
    effect.attrib.clear()
    effect.attrib.update(old_attributes)
    if old_name is None:
        projects[0].attrib.pop("name", None)
    else:
        projects[0].set("name", old_name)
    if ET.tostring(root) != before:
        raise ReplayError("Unexpected non-resource XML change")
    payload = b'<?xml version="1.0" encoding="UTF-8"?>\n<!DOCTYPE fcpxml>\n' + ET.tostring(
        candidate
    )
    if len(payload) > MAX_XML_BYTES:
        raise ReplayError("Template reference output exceeds byte budget")
    if any(file_digest(Path(p)) != digest for p, digest in inputs.items()):
        raise ReplayError("Template reference input changed")
    with output.open("xb") as stream:
        stream.write(payload)
    report: dict[str, object] = {
        "version": 1,
        "inputs": inputs,
        "output": {"path": str(output), "sha256": file_digest(output)},
        "effectID": config["effectID"],
        "previousUID": config["expectedUID"],
        "candidateUID": attributes["uid"],
        "observedEffectBound": "observedEffect" in config,
        "previousProjectName": old_name,
        "projectName": name if name is not None else old_name,
        "titleCount": len(candidate.findall(".//title")),
        "allOtherXMLUnchanged": True,
        "captionTextClockGeometryMediaUnchanged": True,
        "candidateOnly": True,
        "nativeAcceptanceVerified": False,
    }
    with proof.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("proof", type=Path)
    args = parser.parse_args()
    rewrite(args.config, args.output, args.proof)


if __name__ == "__main__":
    main()
