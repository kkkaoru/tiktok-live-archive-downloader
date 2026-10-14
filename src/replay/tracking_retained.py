"""Revalidate immutable tracking with explicit historical code evidence.

Archived source bytes are attestations only, never imported or executed. Every
source/native byte, request, response and recognized source clock is verified by
the current reader before the existing face consumer can use the observation.
"""

import argparse
import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path

from replay import producer_code_evidence
from replay import tracking_observations as storage
from replay.head_observations import tracking_frames
from replay.input_hash_cache import InputHashCache, input_hash_context
from replay.recordings import record
from replay.stages import file_digest

ROOT = Path(__file__).resolve().parents[2]
PROTOCOL = "tracking-observations-v1"
MAX_WINDOWS = 512
READER_FILES = (
    *storage.CODE_FILES,
    "tracking_retained.py",
    "producer_code_evidence.py",
    "input_hash_cache.py",
)


@dataclass(frozen=True, slots=True)
class RetainedWindow:
    job: storage.TrackingJob
    output: Path
    provenance_sha256: str
    response_sha256: str
    frames: int
    producer_code: dict[str, str]


@dataclass(frozen=True, slots=True)
class RetainedTracking:
    windows: tuple[RetainedWindow, ...]
    inputs: dict[str, str]

    def window(self, config: Path, output: Path) -> RetainedWindow:
        """Require the exact pair selected by the pinned face index."""
        matches = tuple(
            item for item in self.windows if (item.job.config, item.output) == (config, output)
        )
        if len(matches) != 1:
            raise ValueError("Retained tracking does not bind this face window")
        return matches[0]

    def receipt(self) -> dict[str, object]:
        return {
            "version": 1,
            "protocol": PROTOCOL,
            "inputs": self.inputs,
            "currentReaderCode": {
                str(Path(__file__).with_name(name)): self.inputs[
                    str(Path(__file__).with_name(name))
                ]
                for name in READER_FILES
            },
            "windows": [
                {
                    "config": str(item.job.config),
                    "evidence": str(item.output),
                    "provenanceSHA256": item.provenance_sha256,
                    "responseSHA256": item.response_sha256,
                    "frames": item.frames,
                    "historicalProducerCode": item.producer_code,
                }
                for item in self.windows
            ],
            "mediaBytesVerified": True,
            "nativeBytesVerified": True,
            "requestAndResponseClocksVerified": True,
            "historicalExecutionReissued": False,
            "captionPlacementApproved": False,
            "deliveryComplete": False,
        }


def _local(path: Path) -> Path:
    if (
        not path.is_absolute()
        or path != path.resolve(strict=True)
        or not path.is_relative_to(ROOT / "edits" / "work")
    ):
        raise ValueError("Retained tracking evidence must be canonical repository work files")
    return path


def _bound_bytes(path: Path, digest: str, limit: int) -> bytes:
    data = storage._read(path, limit)
    if hashlib.sha256(data).hexdigest() != storage._digest(digest):
        raise ValueError("Pinned retained tracking artifact changed")
    return data


def _unique_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for name, value in pairs:
        if name in result:
            raise ValueError("Retained tracking JSON keys must be unique")
        result[name] = value
    return result


def _json(data: bytes) -> object:
    return json.loads(data, object_pairs_hook=_unique_object)


def _bind(inputs: dict[str, str], path: Path, digest: str) -> None:
    value = storage._digest(digest)
    previous = inputs.get(str(path))
    if previous is not None and previous != value:
        raise ValueError("Conflicting retained tracking input fingerprints")
    inputs[str(path)] = value


def _reference(value: object, inputs: dict[str, str]) -> tuple[Path, str]:
    raw = record(value)
    if set(raw) != {"path", "sha256"} or not isinstance(raw["path"], str):
        raise ValueError("Expected exact retained tracking path and hash")
    path = _local(Path(raw["path"]))
    digest = storage._digest(raw["sha256"])
    _bind(inputs, path, digest)
    return path, digest


def _prepare_window(
    value: object, evidence: producer_code_evidence.CodeEvidence, inputs: dict[str, str]
) -> tuple[RetainedWindow, dict[str, object]]:
    raw = record(value)
    if set(raw) != {"config", "provenance"}:
        raise ValueError("Expected exact retained face-window roles")
    config, config_hash = _reference(raw["config"], inputs)
    provenance, provenance_hash = _reference(raw["provenance"], inputs)
    _json(_bound_bytes(config, config_hash, storage.CONFIG_BYTES))
    job = storage.load_job(config)
    if job.config_hash != config_hash or not os.access(job.native, os.X_OK):
        raise ValueError("Pinned tracking configuration or native executable changed")
    output = provenance.parent
    if provenance.name != "provenance.json":
        raise ValueError("Retained tracking requires the original provenance filename")
    original = record(_json(_bound_bytes(provenance, provenance_hash, storage.CONFIG_BYTES)))
    old_inputs = {
        name: storage._digest(digest) for name, digest in record(original.get("inputs")).items()
    }
    code = {str(Path(__file__).with_name(name)): name for name in storage.CODE_FILES}
    expected = {str(job.config), str(job.source), str(job.native), *code}
    if len(expected) != 3 + len(code) or set(old_inputs) != expected:
        raise ValueError("Retained tracking original input roles differ")
    producer_code: dict[str, str] = {}
    for name, digest in old_inputs.items():
        path = Path(name)
        if name in code:
            producer_code[name] = digest
            archived = evidence.archives.get(name)
            if archived is not None:
                if archived.sha256 != digest:
                    raise ValueError("Historical archive differs from tracking producer pin")
                path = archived.path
        _bind(inputs, path, digest)
    if (
        old_inputs[str(job.source)] != job.source_hash
        or old_inputs[str(job.native)] != job.native_hash
    ):
        raise ValueError("Protected tracking source or native pins differ")
    response_hash = storage._digest(original.get("responseSHA256"))
    _bind(inputs, output / "request.json", storage._digest(original.get("requestSHA256")))
    _bind(inputs, output / "response.json", response_hash)
    return RetainedWindow(
        job, output, provenance_hash, response_hash, len(job.clock.pts), producer_code
    ), original


def _verify_window(window: RetainedWindow, original: dict[str, object]) -> None:
    request = window.output / "request.json"
    if (
        _json(
            _bound_bytes(
                request, storage._digest(original.get("requestSHA256")), storage.CONFIG_BYTES
            )
        )
        != window.job.request()
    ):
        raise ValueError("Retained tracking request differs from the verified plan")
    frames = tracking_frames(
        _json(
            _bound_bytes(
                window.output / "response.json", window.response_sha256, storage.RESPONSE_BYTES
            )
        ),
        window.job.clock,
        width=window.job.width,
        height=window.job.height,
    )
    expected = {
        "inputs": original["inputs"],
        "requestSHA256": file_digest(request),
        "responseSHA256": window.response_sha256,
        "frames": len(frames),
        "nativeExitCode": 0,
        "captionPlacementApproved": False,
    }
    if json.dumps(original, sort_keys=True) != json.dumps(expected, sort_keys=True):
        raise ValueError("Retained tracking original provenance changed")
    storage._check_diagnostics(window.output)


def verify_catalog(
    descriptor: Path, *, sha256: str, references: list[tuple[Path, Path]] | None = None
) -> RetainedTracking:
    """Read current bytes at both boundaries; do not rewrite historical receipts."""
    evidence = producer_code_evidence.read_archives(
        _local(descriptor),
        sha256,
        protocol=PROTOCOL,
        source_names=storage.CODE_FILES,
        root=ROOT,
        binding_fields=frozenset({"windows"}),
    )
    rows = evidence.document.get("windows")
    if not isinstance(rows, list) or not 1 <= len(rows) <= MAX_WINDOWS:
        raise ValueError("Expected bounded retained tracking windows")
    inputs = evidence.inputs.copy()
    prepared = tuple(_prepare_window(row, evidence, inputs) for row in rows)
    pairs = tuple((window.job.config, window.output) for window, _ in prepared)
    if len(set(pairs)) != len(pairs) or (
        references is not None and (len(references) != len(pairs) or set(references) != set(pairs))
    ):
        raise ValueError("Retained tracking catalog differs from the selected face index")
    for name in READER_FILES:
        path = Path(__file__).with_name(name)
        # The original producer code is separate from the current reader hash.
        if (
            name in storage.CODE_FILES
            and str(path) in inputs
            and inputs[str(path)] != file_digest(path)
        ):
            raise ValueError("Changed producer code requires explicit archive evidence")
        _bind(inputs, path, file_digest(path))
    cache = InputHashCache(Path(name) for name in inputs)
    cache.prime()
    with input_hash_context(cache):
        if any(file_digest(Path(name)) != digest for name, digest in inputs.items()):
            raise ValueError("Retained tracking protected input bytes changed")
        for window, original in prepared:
            cache.guard()
            _verify_window(window, original)
        cache.verify()
    for window, _ in prepared:
        storage._check_diagnostics(window.output)
    return RetainedTracking(tuple(window for window, _ in prepared), inputs)


def write_receipt(path: Path, verified: RetainedTracking) -> None:
    destination = storage._checked_output(path.absolute())
    if not destination.is_relative_to(ROOT / "edits" / "work") or not destination.parent.is_dir():
        raise ValueError("Retained tracking receipts require repository work parents")
    storage._write(destination, verified.receipt())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("descriptor", type=Path)
    parser.add_argument("receipt", type=Path)
    parser.add_argument("--sha256", required=True)
    arguments = parser.parse_args()
    write_receipt(arguments.receipt, verify_catalog(arguments.descriptor, sha256=arguments.sha256))


if __name__ == "__main__":
    main()
