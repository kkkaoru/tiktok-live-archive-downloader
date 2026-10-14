"""Read historical upper-pixel evidence using explicit code-only attestations.

Archived Python is never executed. Original cells, raw support and candidate
status remain unchanged; the current reader has a separate dependency fingerprint.
"""

import argparse
import hashlib
import json
import os
from dataclasses import dataclass, replace
from itertools import batched
from pathlib import Path

from replay import head_image_collection as collection
from replay import head_image_plan as planning
from replay import producer_code_evidence
from replay import tracking_observations as storage
from replay.fcp_visible_arc import VisibleArc
from replay.foreground_observations import MAX_SAMPLES
from replay.input_hash_cache import InputHashCache, input_hash_context
from replay.recordings import record
from replay.stages import file_digest

ROOT = Path(__file__).resolve().parents[2]
PROTOCOL = "head-image-collection-v1"
PRODUCER_FILES = tuple(dict.fromkeys((*collection.CODE_FILES, *planning.CODE_FILES)))
READER_FILES = (
    *PRODUCER_FILES,
    "head_image_retained.py",
    "producer_code_evidence.py",
    "input_hash_cache.py",
)


@dataclass(frozen=True, slots=True)
class RetainedUpperBatch:
    cells: tuple[planning.ImageFrameCell, ...]
    response: Path
    response_sha256: str


@dataclass(frozen=True, slots=True)
class RetainedUpperPixels:
    job: collection.ImageJob
    arc: VisibleArc
    batches: tuple[RetainedUpperBatch, ...]
    inputs: dict[str, str]
    producer_code: dict[str, str]
    provenance_sha256: str

    def receipt(self) -> dict[str, object]:
        return {
            "version": 1,
            "protocol": PROTOCOL,
            "inputs": self.inputs,
            "config": str(self.job.config),
            "provenanceSHA256": self.provenance_sha256,
            "historicalProducerCode": self.producer_code,
            "currentReaderCode": {
                str(ROOT / "src" / "replay" / name): self.inputs[
                    str(ROOT / "src" / "replay" / name)
                ]
                for name in READER_FILES
            },
            "frames": len(self.arc.cells),
            "supportedFrames": sum(cell.visible_top is not None for cell in self.arc.cells),
            "mediaBytesVerified": True,
            "nativeBytesVerified": True,
            "requestAndResponseClocksVerified": True,
            "historicalExecutionReissued": False,
            "candidateOnly": True,
            "semanticHeadVerified": False,
            "captionPlacementApproved": False,
            "deliveryComplete": False,
        }


def _json(path: Path, digest: str, limit: int = storage.CONFIG_BYTES) -> object:
    data = storage._read(path, limit)
    if hashlib.sha256(data).hexdigest() != storage._digest(digest):
        raise ValueError("Pinned retained upper-pixel artifact changed")
    return json.loads(data, object_pairs_hook=producer_code_evidence._unique_object)


def _reference(value: object) -> tuple[Path, str]:
    raw = record(value)
    if set(raw) != {"path", "sha256"} or not isinstance(raw["path"], str):
        raise ValueError("Expected exact retained upper-pixel path and hash")
    path = producer_code_evidence._path(raw["path"], root=ROOT)
    if not path.is_relative_to(ROOT / "edits" / "work"):
        raise ValueError("Retained upper-pixel evidence must stay under repository work")
    return path, storage._digest(raw["sha256"])


def _bind(inputs: dict[str, str], path: Path, digest: str) -> None:
    expected = storage._digest(digest)
    if str(path) in inputs and inputs[str(path)] != expected:
        raise ValueError("Conflicting retained upper-pixel dependency hashes")
    inputs[str(path)] = expected


def _job(
    config: Path,
    config_hash: str,
    original: dict[str, object],
    evidence: producer_code_evidence.CodeEvidence,
) -> tuple[collection.ImageJob, dict[str, str], dict[str, str]]:
    raw = record(_json(config, config_hash))
    fresh = collection.load_job(config)
    producer = {
        name: storage._digest(digest) for name, digest in record(original.get("inputs")).items()
    }
    declared = {name: storage._digest(digest) for name, digest in record(raw.get("inputs")).items()}
    collected_code = {str(ROOT / "src" / "replay" / name) for name in collection.CODE_FILES}
    code = {str(ROOT / "src" / "replay" / name) for name in PRODUCER_FILES}
    if (
        set(producer) != {*declared, str(config), *collected_code}
        or producer.get(str(config)) != config_hash
    ):
        raise ValueError("Retained upper-pixel producer input roles differ")
    if any(
        producer[name] != digest for name, digest in declared.items() if name not in collected_code
    ):
        raise ValueError("Original upper-pixel configuration input pins differ")
    inputs = evidence.inputs.copy()
    producer_code: dict[str, str] = {}
    for name, digest in producer.items():
        path = Path(name)
        if name in code:
            producer_code[name] = digest
            saved = evidence.archives.get(name)
            if saved is not None:
                if saved.sha256 != digest:
                    raise ValueError("Archive differs from the original upper-pixel code pin")
                path = saved.path
        _bind(inputs, path, digest)
    if not os.access(fresh.native, os.X_OK):
        raise ValueError("Retained upper-pixel native executable is not executable")
    return replace(fresh, inputs=producer), inputs, producer_code


def read_image_binding(
    value: object, *, source_hash: str, width: int, height: int
) -> RetainedUpperPixels:
    """Verify exact historical raw bytes and return unchanged partial ArcCells."""
    binding = record(value)
    if set(binding) != {"config", "provenance", "producerEvidence"}:
        raise ValueError(
            "Retained upper pixels require explicit config, provenance and code evidence"
        )
    config, config_hash = _reference(binding["config"])
    provenance, provenance_hash = _reference(binding["provenance"])
    descriptor, descriptor_hash = _reference(binding["producerEvidence"])
    if provenance.name != "provenance.json":
        raise ValueError("Retained upper pixels require the original provenance filename")
    evidence = producer_code_evidence.read_archives(
        descriptor,
        descriptor_hash,
        protocol=PROTOCOL,
        source_names=PRODUCER_FILES,
        root=ROOT,
        binding_fields=frozenset({"measurement"}),
    )
    measurement = record(evidence.document.get("measurement"))
    if set(measurement) != {"config", "provenance"} or (
        _reference(measurement["config"]) != (config, config_hash)
        or _reference(measurement["provenance"]) != (provenance, provenance_hash)
    ):
        raise ValueError("Historical code evidence does not bind this upper-pixel measurement")
    original = record(_json(provenance, provenance_hash))
    job, inputs, producer_code = _job(config, config_hash, original, evidence)
    if (
        type(width) is not int
        or type(height) is not int
        or (job.width, job.height) != (width, height)
        or job.inputs[str(job.source)] != storage._digest(source_hash)
    ):
        raise ValueError("Retained upper pixels differ from the protected source or geometry")
    _bind(inputs, provenance, provenance_hash)
    stored = provenance.parent / "inputs.json"
    _bind(inputs, stored, file_digest(stored))
    if _json(stored, inputs[str(stored)]) != job.inputs:
        raise ValueError("Original upper-pixel input receipt changed")
    batches = tuple(batched((cell for cell in job.cells if cell.face is not None), MAX_SAMPLES))
    recorded_batches = original.get("batches")
    if not isinstance(recorded_batches, list) or len(recorded_batches) != len(batches):
        raise ValueError("Retained upper-pixel batch count differs")
    retained_batches: list[RetainedUpperBatch] = []
    for index, (cells, row) in enumerate(zip(batches, recorded_batches, strict=True)):
        paths = record(record(row).get("inputs"))
        directory = provenance.parent / f"batch-{index:04d}"
        request, response = directory / "request.json", directory / "response.json"
        if set(paths) != {str(request), str(response)}:
            raise ValueError("Retained upper-pixel batch input roles differ")
        _bind(inputs, request, storage._digest(paths[str(request)]))
        _bind(inputs, response, storage._digest(paths[str(response)]))
        # Duplicate keys cannot make different raw requests/responses look equal.
        _json(request, inputs[str(request)])
        _json(response, inputs[str(response)], collection.RESPONSE_BYTES)
        retained_batches.append(RetainedUpperBatch(cells, response, inputs[str(response)]))
    for name in READER_FILES:
        path = ROOT / "src" / "replay" / name
        _bind(inputs, path, file_digest(path))
    cache = InputHashCache(Path(name) for name in inputs)
    cache.prime()
    with input_hash_context(cache):
        if any(file_digest(Path(name)) != digest for name, digest in inputs.items()):
            raise ValueError("Protected retained upper-pixel input bytes changed")
        expected, arc = collection._receipt(job, provenance.parent)
        if json.dumps(original, sort_keys=True) != json.dumps(expected, sort_keys=True):
            raise ValueError("Original upper-pixel provenance or raw cell measurements changed")
        cache.verify()
    for index in range(len(batches)):
        storage._check_diagnostics(provenance.parent / f"batch-{index:04d}")
    return RetainedUpperPixels(
        job,
        VisibleArc(arc.cells, inputs),
        tuple(retained_batches),
        inputs,
        producer_code,
        provenance_hash,
    )


def read_response(batch: RetainedUpperBatch) -> object:
    """Read the still-bound raw support for a consumer's explicit seed policy."""
    return _json(batch.response, batch.response_sha256, collection.RESPONSE_BYTES)


def write_receipt(path: Path, verified: RetainedUpperPixels) -> None:
    destination = producer_code_evidence._path(str(path), root=ROOT)
    if not destination.is_relative_to(ROOT / "edits" / "work") or not destination.parent.is_dir():
        raise ValueError("Retained upper-pixel receipts require existing repository work parents")
    storage._write(destination, verified.receipt())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path)
    parser.add_argument("receipt", type=Path)
    parser.add_argument("--sha256", required=True)
    args = parser.parse_args()
    path, digest = _reference({"path": str(args.config), "sha256": args.sha256})
    config = record(_json(path, digest))
    if set(config) != {"binding", "sourceSHA256", "width", "height"}:
        raise ValueError("Retained upper-pixel consumer config fields differ")
    verified = read_image_binding(
        config["binding"],
        source_hash=storage._digest(config["sourceSHA256"]),
        width=storage._integer(config["width"]),
        height=storage._integer(config["height"]),
    )
    _bind(verified.inputs, path, digest)
    _json(path, digest)
    write_receipt(args.receipt, verified)


if __name__ == "__main__":
    main()
