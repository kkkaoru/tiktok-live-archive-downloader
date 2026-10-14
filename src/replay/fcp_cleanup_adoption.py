"""Verify every partition's cleanup in one fresh, bounded byte-check operation.

Historical source names remain receipt identities. Only explicitly inventoried
current repair and approved original media are opened; this read-only operation
renders and publishes nothing, and never grants delivery or quality approval.
"""

import argparse
import hashlib
import json
import os
import re
import stat
from dataclasses import dataclass
from pathlib import Path

from replay import fcp_part_followup as followup
from replay.core import ReplayError
from replay.input_hash_cache import MAX_INPUTS, InputHashCache, input_hash_context
from replay.recordings import record
from replay.stages import file_digest

ROOT = Path(__file__).resolve().parents[2]
MAX_JSON_BYTES = 8 * 1024 * 1024
CODE_NAMES = tuple(dict.fromkeys((*followup.CODE_NAMES, "fcp_cleanup_adoption.py", "core.py")))


def _unique(pairs: list[tuple[str, object]]) -> dict[str, object]:
    if len({key for key, _ in pairs}) != len(pairs):
        raise ReplayError("Cleanup batch JSON keys must be unique")
    return dict(pairs)


def _path(value: object, *, work: bool = False, evidence: bool = False) -> Path:
    if not isinstance(value, str) or not value or any(char in value for char in "\0\r\n"):
        raise ReplayError("Expected canonical cleanup batch path")
    path = Path(value)
    if (
        not path.is_absolute()
        or str(path) != value
        or path.resolve() != path
        or ".." in path.parts
        or (work and not path.is_relative_to(ROOT / "edits/work"))
        or (evidence and not path.is_relative_to(ROOT / "edits"))
        or any(path.is_relative_to(ROOT / name) for name in ("private", "downloads"))
    ):
        raise ReplayError("Expected canonical repository-work cleanup batch evidence")
    return path


def _sha(value: object) -> str:
    if not isinstance(value, str) or re.fullmatch(r"[a-f0-9]{64}", value) is None:
        raise ReplayError("Cleanup batch requires explicit SHA256 fingerprints")
    return value


def _ref(
    value: object, *, exact: bool = True, work: bool = False, evidence: bool = False
) -> tuple[Path, str]:
    row = record(value)
    if exact and set(row) != {"path", "sha256"}:
        raise ReplayError("Cleanup batch reference fields differ")
    return _path(row.get("path"), work=work, evidence=evidence), _sha(row.get("sha256"))


def _bytes(path: Path) -> bytes:
    with os.fdopen(os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK), "rb") as stream:
        before = os.fstat(stream.fileno())
        if not stat.S_ISREG(before.st_mode) or not 0 < before.st_size <= MAX_JSON_BYTES:
            raise ReplayError("Cleanup batch metadata must be bounded nonempty regular files")
        data = stream.read(MAX_JSON_BYTES + 1)
        after = os.fstat(stream.fileno())
        named = path.lstat()
    identity = (
        before.st_dev,
        before.st_ino,
        before.st_size,
        before.st_mtime_ns,
        before.st_ctime_ns,
    )
    if len(data) != before.st_size or any(
        identity != (info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns, info.st_ctime_ns)
        for info in (after, named)
    ):
        raise ReplayError("Cleanup batch metadata changed during reading")
    return data


def _json(path: Path, digest: str) -> dict[str, object]:
    data = _bytes(path)
    if hashlib.sha256(data).hexdigest() != _sha(digest):
        raise ReplayError("Cleanup batch metadata fingerprint changed")
    return record(json.loads(data, object_pairs_hook=_unique))


@dataclass(frozen=True)
class Batch:
    semantic: Path
    configs: tuple[Path, ...]
    assembly: Path
    partition: Path
    inputs: dict[str, str]


class Inventory:
    """Discover saved current references without hashing media before cache prime."""

    def __init__(self) -> None:
        self.inputs: dict[str, str] = {}
        self.documents: dict[Path, dict[str, object]] = {}

    def bind(self, path: Path, digest: str) -> None:
        if self.inputs.get(str(path), digest) != digest:
            raise ReplayError("Cleanup batch has conflicting input fingerprints")
        self.inputs[str(path)] = digest
        if len(self.inputs) >= MAX_INPUTS:
            raise ReplayError("Cleanup batch input closure exceeds its reference budget")

    def metadata(
        self, value: object, *, exact: bool = True, operation: bool = False
    ) -> tuple[Path, dict[str, object]]:
        # Existing domain references include both work and edits/replay evidence.
        # Newly supplied operation configurations remain work-only.
        path, digest = _ref(value, exact=exact, work=operation, evidence=True)
        self.bind(path, digest)
        if path not in self.documents:
            self.documents[path] = _json(path, digest)
        return path, self.documents[path]

    def media(self, path: object, digest: object) -> None:
        self.bind(_path(path), _sha(digest))

    def cleanup(self, value: object) -> None:
        _path_, binding = self.metadata(value)
        _manifest_path, manifest = self.metadata(binding.get("assemblyManifest"))
        _index_path, index = self.metadata(manifest.get("index"))
        _receipt_path, receipt = self.metadata(binding.get("assemblyReceipt"))
        self.metadata(binding.get("cleanIndex"))
        self.metadata(binding.get("voiceReceipt"))
        consumer, consumer_sha = _ref(binding.get("consumer"), exact=False, evidence=True)
        self.bind(consumer, consumer_sha)
        rows, chunks = index.get("ranges"), manifest.get("chunks")
        if (
            not isinstance(rows, list)
            or not isinstance(chunks, list)
            or not (1 <= len(rows) == len(chunks) <= 512)
        ):
            raise ReplayError("Cleanup batch needs every bounded cleanup chunk")
        for row, entry in zip(rows, chunks, strict=True):
            chunk = record(entry)
            _candidate_path, candidate = self.metadata(chunk.get("candidate"))
            self.metadata(chunk.get("producer"))
            self.media(record(row).get("media"), candidate.get("videoSHA256"))
        probes = record(receipt.get("probeEvidence"))
        if not 1 <= len(probes) <= 1024:
            raise ReplayError("Cleanup batch probe evidence is missing or excessive")
        for path, digest in probes.items():
            self.bind(_path(path, evidence=True), _sha(digest))
        clean, digest = _ref(receipt.get("output"))
        self.media(str(clean), digest)


def _discover(document: dict[str, object]) -> Batch:
    inventory = Inventory()
    semantic, semantic_digest = _ref(document.get("semantic"), evidence=True)
    inventory.bind(semantic, semantic_digest)
    rows = document.get("configs")
    if not isinstance(rows, list) or not 1 <= len(rows) <= followup.MAX_PARTS:
        raise ReplayError("Cleanup batch requires bounded nonempty part configurations")
    configs: list[Path] = []
    parts: list[int] = []
    assemblies: set[Path] = set()
    partitions: set[Path] = set()
    for value in rows:
        config, raw = inventory.metadata(value, operation=True)
        if (
            not {"version", "part", "assembly", "sourceCleanup"}.issubset(raw)
            or set(raw)
            - {"version", "part", "assembly", "sourceCleanup", "recover", "approvedOriginal"}
            or type(raw.get("version")) is not int
            or raw.get("version") != 1
            or type(raw.get("part")) is not int
            or type(raw.get("recover", False)) is not bool
        ):
            raise ReplayError("Cleanup batch requires minimal exact part configurations")
        part = raw["part"]
        if not isinstance(part, int) or not 1 <= part <= followup.MAX_PARTS:
            raise ReplayError("Cleanup batch part number is invalid")
        assembly, metadata = inventory.metadata(raw["assembly"])
        partition, partition_data = inventory.metadata(metadata.get("partition"))
        xmls, assembly_parts = partition_data.get("parts"), metadata.get("parts")
        if (
            type(metadata.get("version")) is not int
            or metadata.get("version") != 1
            or type(partition_data.get("version")) is not int
            or partition_data.get("version") != 1
            or not isinstance(xmls, list)
            or not isinstance(assembly_parts, list)
            or not 1 <= len(xmls) == len(rows) == len(assembly_parts) <= followup.MAX_PARTS
            or _ref(partition_data.get("source")) != (semantic, semantic_digest)
        ):
            raise ReplayError("Cleanup batch must cover one complete semantic partition")
        selected: set[Path] = set()
        for xml in xmls:
            path, digest = _ref(xml, exact=False, evidence=True)
            if path in selected:
                raise ReplayError("Cleanup batch partition XML paths must be unique")
            selected.add(path)
            inventory.bind(path, digest)
        if "approvedOriginal" in raw:
            original, digest = _ref(raw["approvedOriginal"])
            inventory.media(str(original), digest)
        inventory.cleanup(raw["sourceCleanup"])
        configs.append(config)
        parts.append(part)
        assemblies.add(assembly)
        partitions.add(partition)
    if (
        len(set(configs)) != len(configs)
        or parts != list(range(1, len(configs) + 1))
        or len(assemblies) != 1
        or len(partitions) != 1
    ):
        raise ReplayError("Cleanup batch requires unique ordered parts from one assembly")
    for name in CODE_NAMES:
        path = _path(str(ROOT / "src/replay" / name))
        inventory.bind(path, file_digest(path))
    return Batch(semantic, tuple(configs), assemblies.pop(), partitions.pop(), inventory.inputs)


def _inventory(value: object) -> dict[str, str]:
    if not isinstance(value, list) or not 1 <= len(value) < MAX_INPUTS:
        raise ReplayError("Cleanup batch needs a bounded explicit input inventory")
    inputs: dict[str, str] = {}
    for row in value:
        path, digest = _ref(row)
        if str(path) in inputs:
            raise ReplayError("Cleanup batch inventory paths must be unique")
        inputs[str(path)] = digest
    return inputs


def _write(path: Path, value: dict[str, object]) -> None:
    _path(str(path), work=True)
    if not path.parent.is_dir():
        raise ReplayError("Cleanup batch output parent must already exist")
    with path.open("x", encoding="utf-8") as stream:
        os.fchmod(stream.fileno(), 0o600)
        stream.write(json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n")


def prepare(configs: tuple[Path, ...], semantic: Path, output: Path) -> None:
    """Write exact current references; preparation does not hash or approve media."""
    document: dict[str, object] = {
        "version": 1,
        "semantic": {
            "path": str(_path(str(semantic), evidence=True)),
            "sha256": file_digest(semantic),
        },
        "configs": [
            {"path": str(_path(str(path), work=True)), "sha256": file_digest(path)}
            for path in configs
        ],
    }
    batch = _discover(document)
    document["inputs"] = [{"path": path, "sha256": sha} for path, sha in batch.inputs.items()]
    _write(output, document)


def _validate(config: Path, digest: str) -> dict[str, object]:
    config = _path(str(config), work=True)
    document = _json(config, digest)
    if set(document) != {"version", "semantic", "configs", "inputs"} or (
        type(document.get("version")) is not int or document.get("version") != 1
    ):
        raise ReplayError("Cleanup batch manifest fields differ")
    declared = _inventory(document["inputs"])
    batch = _discover(document)
    if batch.inputs != declared or str(config) in declared:
        raise ReplayError("Cleanup batch inventory must equal the complete current input closure")
    inputs = {**declared, str(config): _sha(digest)}
    cache = InputHashCache((Path(path) for path in inputs), allow_uncached_overflow=True)
    cache.prime()
    with input_hash_context(cache):
        if any(file_digest(Path(path)) != sha for path, sha in inputs.items()):
            raise ReplayError("Cleanup batch current input fingerprint changed")
        if _discover(_json(config, digest)) != batch:
            raise ReplayError("Cleanup batch closure changed after initial byte verification")
        results: list[dict[str, object]] = []
        for number, path in enumerate(batch.configs, start=1):
            cache.guard()
            assembly, part, recover = followup.validate_cleanup(path, batch.semantic)
            if (assembly, part) != (batch.assembly, number):
                raise ReplayError("Cleanup batch domain returned a different part identity")
            cache.guard()
            results.append({"part": part, "config": str(path), "recover": recover})
        cache.verify()
    return {
        "version": 1,
        "kind": "verified-cleanup-part-batch",
        "config": {"path": str(config), "sha256": digest},
        "semantic": document["semantic"],
        "assembly": str(batch.assembly),
        "partition": str(batch.partition),
        "parts": results,
        "inputs": inputs,
        "currentReaderCode": {
            str(ROOT / "src/replay" / name): inputs[str(ROOT / "src/replay" / name)]
            for name in CODE_NAMES
        },
        "hashReads": cache.hash_reads,
        "bytesRead": cache.bytes_read,
        "cacheHits": cache.cache_hits,
        "cleanupDomainVerified": True,
        "initialAndTerminalBytesVerified": True,
        "historicalSourceOpened": False,
        "candidateOnly": True,
        "qualityApproved": False,
        "nativeRenderVerified": False,
        "deliveryComplete": False,
    }


def adopt(config: Path, digest: str, receipt: Path) -> None:
    _path(str(receipt), work=True)
    if receipt.exists() or receipt.is_symlink() or not receipt.parent.is_dir():
        raise ReplayError("Cleanup batch receipt must be absent with an existing parent")
    _write(receipt, _validate(config, digest))


def verify(config: Path, digest: str, receipt: Path) -> None:
    """Use a new empty cache and full domain verification for every reuse."""
    receipt = _path(str(receipt), work=True)
    receipt_digest = file_digest(receipt)
    original = _json(receipt, receipt_digest)
    expected = _validate(config, digest)
    if json.dumps(original, sort_keys=True) != json.dumps(expected, sort_keys=True):
        raise ReplayError("Cleanup batch receipt differs from fresh full verification")
    _json(receipt, receipt_digest)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    preparation = commands.add_parser("prepare")
    preparation.add_argument("output", type=Path)
    preparation.add_argument("--semantic", type=Path, required=True)
    preparation.add_argument("--part-config", type=Path, action="append", required=True)
    for name in ("adopt", "verify"):
        command = commands.add_parser(name)
        command.add_argument("config", type=Path)
        command.add_argument("receipt", type=Path)
        command.add_argument("--sha256", required=True)
    args = parser.parse_args()
    if args.command == "prepare":
        prepare(tuple(args.part_config), args.semantic, args.output)
    elif args.command == "verify":
        verify(args.config, args.sha256, args.receipt)
    else:
        adopt(args.config, args.sha256, args.receipt)


if __name__ == "__main__":
    main()
