"""Bind unchanged native part bytes to a new caption-position-only partition."""

import argparse
import copy
from pathlib import Path
from xml.etree import ElementTree as ET

from replay.artifact_json import _json, _path, _pin
from replay.core import ReplayError
from replay.fcp_caption_geometry_audit import _structure
from replay.fcp_part_assembly import _config, _reference, _save, verify
from replay.fcp_partition import _required
from replay.part_master_publication import verify_published
from replay.recordings import record
from replay.stages import file_digest

PRODUCER_FILES = (
    "fcp_part_reuse.py",
    "fcp_part_assembly.py",
    "fcp_partition.py",
    "fcp_caption_geometry_audit.py",
    "fcp_head_refresh.py",
    "fcp_caption_srt.py",
    "part_master_publication.py",
    "artifact_json.py",
)


def _content(path: Path) -> bytes:
    root = copy.deepcopy(ET.parse(path).getroot())
    _required(root, "./library/event/project").attrib.pop("name", None)
    return ET.tostring(root)


def _build(config_path: Path) -> dict[str, object]:
    config_digest = file_digest(config_path)
    producers = tuple(Path(__file__).with_name(name) for name in PRODUCER_FILES)
    producer_pins = {str(path): file_digest(path) for path in producers}
    config = _json(config_path)
    part = config.get("part")
    if config.get("version") != 1 or type(part) is not int or part < 1:
        raise ReplayError("Expected a positive reuse part number")
    old_assembly = _pin(config.get("priorAssembly"))
    target_assembly = _pin(config.get("targetAssembly"))
    prior, prior_config, prior_rows = _config(old_assembly)
    target, target_config, target_rows = _config(target_assembly)
    if part > len(prior.parts) or len(prior.parts) != len(target.parts):
        raise ReplayError("Reuse partitions must have matching ordered parts")
    fields = ("globalStart", "duration", "clips", "titles")
    if (
        prior.frame != target.frame
        or prior.duration != target.duration
        or prior_config.get("voice") != target_config.get("voice")
        or prior_config.get("voiceReceipt") != target_config.get("voiceReceipt")
        or any(
            tuple(before.get(key) for key in fields) != tuple(after.get(key) for key in fields)
            for before, after in zip(prior.parts, target.parts, strict=True)
        )
    ):
        raise ReplayError("Reuse partition frame ranges differ")
    if _structure(ET.parse(prior.semantic).getroot()) != _structure(
        ET.parse(target.semantic).getroot()
    ):
        raise ReplayError("Reuse requires unchanged text, media, clocks and non-Y geometry")
    old_xml, new_xml = _pin(prior.parts[part - 1]), _pin(target.parts[part - 1])
    if _content(old_xml) != _content(new_xml):
        raise ReplayError("Reuse part XML changed beyond its project display name")
    old_row, target_row = record(prior_rows[part - 1]), record(target_rows[part - 1])
    old_proof_path = _path(old_row.get("proof"))
    old_proof = _json(old_proof_path)
    if "crossRevisionReuse" in old_proof:
        raise ReplayError("Reuse must originate from a direct native frame proof")
    if verify(old_assembly, part=part) != old_proof_path:
        raise ReplayError("Reuse prior assembly proof differs")
    publication = _pin(config.get("publication"))
    retirement = _pin(config["retirement"]) if "retirement" in config else None
    completion, intent = verify_published(
        publication, old_proof_path, part=part, retirement=retirement
    )
    movie = _pin(old_proof.get("movie"))
    if target_row.get("movie") != str(movie):
        raise ReplayError("Reuse target movie must be the verified published bytes")
    paths = (
        config_path,
        old_assembly,
        target_assembly,
        prior.manifest,
        target.manifest,
        prior.semantic,
        target.semantic,
        old_xml,
        new_xml,
        old_proof_path,
        publication,
        completion,
        intent,
        *(() if retirement is None else (retirement,)),
        *producers,
    )
    inputs = {str(path): file_digest(path) for path in paths}
    if inputs[str(config_path)] != config_digest or any(
        inputs[name] != digest for name, digest in producer_pins.items()
    ):
        raise ReplayError("Reuse configuration or producer changed during verification")
    _pin(config.get("priorAssembly"))
    _pin(config.get("targetAssembly"))
    _pin(config.get("publication"))
    if retirement is not None:
        _pin(config["retirement"])
    _pin(prior_config.get("partition"))
    _pin(target_config.get("partition"))
    _pin(old_proof.get("movie"))
    return {
        "version": 1,
        "part": part,
        "partition": target_config["partition"],
        "movie": old_proof["movie"],
        "observed": old_proof["observed"],
        "audioIgnored": True,
        "crossRevisionReuse": {
            "config": _reference(config_path),
            "priorFrameProof": _reference(old_proof_path),
            "priorPartition": prior_config["partition"],
            "completion": _reference(completion),
            "publication": _reference(publication),
            "intent": _reference(intent),
            "retirement": None if retirement is None else _reference(retirement),
            "inputs": inputs,
            "projectDisplayNameOnlyIgnored": True,
            "partXMLIdentical": True,
            "frameRangesUnchanged": True,
            "qualityApproved": False,
        },
    }


def prepare(config_path: Path) -> Path:
    """Write an immutable new proof without modifying media or earlier receipts."""
    config_path = _path(str(config_path))
    config = _json(config_path)
    target = _pin(config.get("targetAssembly"))
    part = config.get("part")
    plan, _, rows = _config(target)
    if type(part) is not int or not 1 <= part <= len(plan.parts):
        raise ReplayError("Reuse part index outside target assembly")
    output = _path(record(rows[part - 1]).get("proof"))
    if output.exists() or output.is_symlink() or not output.parent.is_dir():
        raise ReplayError("Reuse proof requires an absent output in an existing directory")
    expected = _build(config_path)
    _save(output, expected)
    return output


def verify_proof(path: Path, *, partition: object, part: int) -> None:
    """Recheck the original completion, explicit retirement and new immutable binding."""
    proof = _json(path)
    reuse = record(proof.get("crossRevisionReuse"))
    if proof.get("partition") != partition or proof.get("part") != part:
        raise ReplayError("Reuse proof does not bind its ordered target partition")
    config_path = _pin(reuse.get("config"))
    config = _json(config_path)
    target = _pin(config.get("targetAssembly"))
    _, _, rows = _config(target)
    if not 1 <= part <= len(rows) or _path(record(rows[part - 1]).get("proof")) != path:
        raise ReplayError("Reuse proof path differs from target assembly")
    if proof != _build(config_path):
        raise ReplayError("Reuse completion or producer inputs changed")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path)
    print(prepare(parser.parse_args().config))


if __name__ == "__main__":
    main()
