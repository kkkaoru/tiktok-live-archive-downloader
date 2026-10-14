"""Batch existing readonly retirement checks within one fresh digest scope.

Before-delete and after-delete operations are distinct. This adapter never
publishes, authorizes deletion, unlinks, renders or grants successor admission.
"""

import argparse
import fcntl
import hashlib
import json
import os
import re
import stat
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from replay import part_master_publication as publication
from replay.core import ReplayError
from replay.input_hash_cache import MAX_INPUTS, InputHashCache, input_hash_context
from replay.native_export_receipt import save_new
from replay.recordings import record
from replay.stages import file_digest
from replay.whole_technical_batch import reader_code_paths

ROOT = Path(__file__).resolve().parents[2]
MEDIA_ROOT = Path("/Volumes/m.2-ssd/movie")
MAX_JSON_BYTES = 8 * 1024**2
MAX_PARTS = 1024
Phase = Literal["before-delete", "after-delete"]
CODE_NAMES = (
    "part_retirement_verification_batch.py",
    "part_master_publication.py",
    "fcp_part_assembly.py",
    "fcp_partition.py",
    "fcp_caption_srt.py",
    "caption_font_supplier.py",
    "delivery_render.py",
    "native_export_receipt.py",
    "artifact_json.py",
    "recordings.py",
    "core.py",
    "stages.py",
    "digest_memo.py",
    "input_hash_cache.py",
    "foreground_evidence.py",
    "tracking_observations.py",
    "fcpxml_chapters.py",
)


def _path(value: object, *, work: bool = False) -> Path:
    if not isinstance(value, str) or not value or "\0" in value:
        raise ReplayError("Expected a canonical absolute retirement path")
    path = Path(value)
    if not path.is_absolute() or path != path.resolve():
        raise ReplayError("Retirement paths must not escape through aliases or symlinks")
    roots = (
        (ROOT / "edits/work",)
        if work
        else (
            ROOT / "edits",
            ROOT / "src/replay",
            MEDIA_ROOT,
            Path("/opt/homebrew/Cellar/ffmpeg"),
        )
    )
    if not any(path.is_relative_to(root) for root in roots):
        raise ReplayError("Retirement path is outside authorized evidence roots")
    return path


@dataclass(frozen=True)
class Reference:
    path: Path
    sha256: str

    def json(self) -> dict[str, str]:
        return {"path": str(self.path), "sha256": self.sha256}


def _ref(value: object) -> Reference:
    raw = record(value)
    digest = raw.get("sha256")
    if (
        set(raw) != {"path", "sha256"}
        or not isinstance(digest, str)
        or re.fullmatch(r"[a-f0-9]{64}", digest) is None
    ):
        raise ReplayError("Retirement references require exact path and SHA256")
    return Reference(_path(raw["path"]), digest)


def _unique(pairs: list[tuple[str, object]]) -> dict[str, object]:
    if len({key for key, _ in pairs}) != len(pairs):
        raise ReplayError("Retirement metadata contains duplicate keys")
    return dict(pairs)


def _read(ref: Reference) -> dict[str, object]:
    path = _path(str(ref.path))
    value = path.lstat()
    if not stat.S_ISREG(value.st_mode) or not 0 < value.st_size <= MAX_JSON_BYTES:
        raise ReplayError("Retirement metadata is not bounded regular evidence")
    with os.fdopen(os.open(path, os.O_RDONLY | os.O_NOFOLLOW), "rb") as handle:
        data = handle.read(MAX_JSON_BYTES + 1)
    if len(data) > MAX_JSON_BYTES or hashlib.sha256(data).hexdigest() != ref.sha256:
        raise ReplayError("Retirement metadata bytes changed")
    return record(json.loads(data, object_pairs_hook=_unique))


def _current(path: Path) -> Reference:
    return Reference(_path(str(path)), file_digest(path))


def _phase(value: object) -> Phase:
    if value == "before-delete":
        return "before-delete"
    if value == "after-delete":
        return "after-delete"
    raise ReplayError("Retirement verification requires an explicit deletion phase")


def _source(path: Path, phase: Phase) -> None:
    _path(str(path))
    if phase == "after-delete":
        if path.exists() or path.is_symlink():
            raise ReplayError("After-delete verification requires actual staging absence")
    elif not path.is_file():
        raise ReplayError("Before-delete verification requires the original staging file")


@dataclass(frozen=True)
class Part:
    number: int
    config: Reference
    source: Path

    def json(self) -> dict[str, object]:
        return {"part": self.number, "verification": self.config.json()}


class Inventory:
    def __init__(self) -> None:
        self.refs: dict[Path, Reference] = {}

    def add(self, value: object) -> Reference:
        ref = _ref(value)
        old = self.refs.get(ref.path)
        if old is not None and old != ref:
            raise ReplayError("Conflicting retirement input digests")
        self.refs[ref.path] = ref
        if len(self.refs) >= MAX_INPUTS:
            raise ReplayError("Retirement input closure exceeds its budget")
        return ref

    def read(self, value: object) -> dict[str, object]:
        return _read(self.add(value))

    def ordered(self) -> tuple[Reference, ...]:
        return tuple(self.refs[path] for path in sorted(self.refs))


def _proof(inventory: Inventory, value: object) -> dict[str, object]:
    proof = inventory.read(value)
    if "crossRevisionReuse" in proof:
        raise ReplayError("Cross-revision retirement requires a separate explicit consumer")
    if proof.get("probeProvenance") is not None:
        inventory.add(record(proof["probeProvenance"]).get("executable"))
    return proof


def _discover(configs: tuple[Reference, ...], phase: Phase) -> tuple[tuple[Part, ...], Inventory]:
    if not 1 <= len(configs) <= MAX_PARTS or len({ref.path for ref in configs}) != len(configs):
        raise ReplayError("Retirement verification requires unique bounded configurations")
    inventory = Inventory()
    parts: list[Part] = []
    for ref in configs:
        _path(str(ref.path), work=True)
        config = inventory.read(ref.json())
        if type(config.get("version")) is not int or config.get("version") != 1:
            raise ReplayError("Expected version-one retirement verification")
        retired = inventory.read(config.get("retirement"))
        number = retired.get("part")
        if type(number) is not int or not 1 <= number <= MAX_PARTS:
            raise ReplayError("Invalid retired part number")
        source = _ref(retired.get("source"))
        _source(source.path, phase)
        if phase == "before-delete":
            inventory.add(source.json())
        replacement = inventory.add(retired.get("movie"))
        for key in ("config", "intent"):
            inventory.read(retired.get(key))
        published = inventory.read(retired.get("publication"))
        for key in ("completion", "intent", "frameProof", "movie"):
            if published.get(key) != retired.get(key):
                raise ReplayError("Retirement and publication bindings differ")
        completed = inventory.read(retired.get("completion"))
        if completed.get("movie") != source.json() or replacement.sha256 != source.sha256:
            raise ReplayError("Retirement replacement must preserve original completed bytes")
        for name, digest in record(completed.get("inputs")).items():
            original = _path(name)
            inventory.add(
                {
                    "path": str(replacement.path if original == source.path else original),
                    "sha256": digest,
                }
            )
        _proof(inventory, completed.get("frameProof"))
        proof = _proof(inventory, retired.get("frameProof"))
        inventory.add(proof.get("movie"))
        assembly = inventory.read(retired.get("assembly"))
        partition = inventory.read(assembly.get("partition"))
        inventory.add(partition.get("source"))
        rows = partition.get("parts")
        if not isinstance(rows, list) or not 1 <= len(rows) <= MAX_PARTS:
            raise ReplayError("Retirement partition requires bounded complete XML references")
        for row in rows:
            # Partition rows also carry clock fields; the reference itself is exact.
            item = record(row)
            inventory.add({"path": item.get("path"), "sha256": item.get("sha256")})
        masters = assembly.get("parts")
        if not isinstance(masters, list) or len(masters) != len(rows) or number > len(masters):
            raise ReplayError("Retirement assembly does not preserve complete part membership")
        selected = record(masters[number - 1])
        if selected.get("proof") != record(retired.get("frameProof")).get("path") or selected.get(
            "movie"
        ) != str(replacement.path):
            raise ReplayError("Retirement assembly selects a different published proof")
        if assembly.get("voiceReceipt") is not None:
            inventory.read(assembly["voiceReceipt"])
            inventory.add(assembly.get("voice"))
        if proof.get("partition") != assembly.get("partition"):
            raise ReplayError("Retirement proof and assembly partitions differ")
        parts.append(Part(number, ref, source.path))
    if len({part.number for part in parts}) != len(parts):
        raise ReplayError("Retirement batch contains duplicate part numbers")
    # Keep the ordinary consumer's lazy reader dependencies. The public bounded
    # scanner owns traversal; explicit leaf/verifier seeds only extend its union.
    code_paths = {*reader_code_paths(), *(ROOT / "src/replay" / name for name in CODE_NAMES)}
    for path in sorted(code_paths):
        inventory.add(_current(path).json())
    return tuple(parts), inventory


def prepare(configs: tuple[Path, ...], output: Path, *, phase: Phase) -> Path:
    """Discover pinned metadata without reading any movie or voice payload."""
    output = _path(str(output), work=True)
    phase = _phase(phase)
    refs = tuple(_current(_path(str(path), work=True)) for path in configs)
    parts, inventory = _discover(refs, phase)
    save_new(
        output,
        {
            "version": 1,
            "kind": "retirement-verification-batch",
            "phase": phase,
            "parts": [part.json() for part in parts],
            "inputs": [ref.json() for ref in inventory.ordered()],
        },
    )
    return output


def _manifest(path: Path, digest: str) -> tuple[Phase, tuple[Part, ...], tuple[Reference, ...]]:
    ref = _ref({"path": str(_path(str(path), work=True)), "sha256": digest})
    raw = _read(ref)
    rows, inputs = raw.get("parts"), raw.get("inputs")
    if (
        set(raw) != {"version", "kind", "phase", "parts", "inputs"}
        or type(raw.get("version")) is not int
        or raw.get("version") != 1
        or raw.get("kind") != "retirement-verification-batch"
        or not isinstance(rows, list)
        or not 1 <= len(rows) <= MAX_PARTS
        or not isinstance(inputs, list)
        or not 1 <= len(inputs) < MAX_INPUTS
    ):
        raise ReplayError("Invalid retirement batch manifest")
    phase = _phase(raw["phase"])
    configs: list[Reference] = []
    for value in rows:
        row = record(value)
        if set(row) != {"part", "verification"} or type(row["part"]) is not int:
            raise ReplayError("Invalid retirement batch part")
        configs.append(_ref(row["verification"]))
    parts, inventory = _discover(tuple(configs), phase)
    expected = tuple(_ref(value) for value in inputs)
    if [part.json() for part in parts] != rows or expected != inventory.ordered():
        raise ReplayError("Retirement batch requires the exact current input closure")
    return phase, parts, expected


def run(manifest: Path, digest: str, output: Path, *, resume: bool = False) -> Path:
    """Recheck all domains even on resume; checkpoints never substitute verification."""
    manifest = _path(str(manifest), work=True)
    output = _path(str(output), work=True)
    phase, parts, refs = _manifest(manifest, digest)
    bound = _ref({"path": str(manifest), "sha256": digest})
    inputs = (*refs, bound)
    cache = InputHashCache((ref.path for ref in inputs), allow_uncached_overflow=True)
    cache.prime()
    with input_hash_context(cache):
        if any(file_digest(ref.path) != ref.sha256 for ref in inputs):
            raise ReplayError("Retirement batch input digest changed")
        if _manifest(manifest, digest) != (phase, parts, refs):
            raise ReplayError("Retirement discovery changed after initial hashing")
        operation = {"version": 1, "manifest": bound.json(), "phase": phase}
        if output.exists():
            if not resume or not output.is_dir() or not (output / "operation.json").is_file():
                raise ReplayError(
                    "Occupied retirement output requires a complete matching checkpoint"
                )
        else:
            output.mkdir()
            save_new(output / "operation.json", operation)
        allowed = {"operation.json", "operation.lock", "receipt.json"} | {
            f"part-{part.number:04d}.json" for part in parts
        }
        if any(
            child.name not in allowed or _path(str(child), work=True) != child
            for child in output.iterdir()
        ):
            raise ReplayError("Retirement output contains uncertain additional artifacts")
        lock = output / "operation.lock"
        _path(str(lock), work=True)
        with os.fdopen(
            os.open(lock, os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW, 0o600), "r+"
        ) as handle:
            try:
                fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as error:
                raise ReplayError("Retirement batch is already owned") from error
            operation_ref = _current(output / "operation.json")
            if _read(operation_ref) != operation:
                raise ReplayError("Retirement operation checkpoint differs")
            generated = [operation_ref]
            for part in parts:
                checkpoint = output / f"part-{part.number:04d}.json"
                expected = {
                    "version": 1,
                    "manifest": bound.json(),
                    "phase": phase,
                    "part": part.number,
                    "verification": part.config.json(),
                    "retirementDomainVerified": True,
                }
                checkpoint_ref = _current(checkpoint) if checkpoint.exists() else None
                if checkpoint_ref is not None and _read(checkpoint_ref) != expected:
                    raise ReplayError("Retirement part checkpoint differs")
                cache.guard()
                _source(part.source, phase)
                publication.verify_retirement(part.config.path)
                _source(part.source, phase)
                cache.guard()
                if checkpoint_ref is None:
                    save_new(checkpoint, expected)
                    checkpoint_ref = _current(checkpoint)
                    if _read(checkpoint_ref) != expected:
                        raise ReplayError("New retirement checkpoint changed")
                generated.append(checkpoint_ref)
            verified = cache.verify()
            if any(file_digest(ref.path) != ref.sha256 for ref in generated):
                raise ReplayError("Retirement generated checkpoint changed")
            for part in parts:
                _source(part.source, phase)
            cache.guard()
            receipt = output / "receipt.json"
            result: dict[str, object] = {
                **operation,
                "kind": "verified-retirement-batch",
                "parts": [part.number for part in parts],
                "inputs": verified,
                "retirementDomainVerified": True,
                "initialAndTerminalBytesVerified": True,
                "sourceAbsenceObserved": phase == "after-delete",
                "deletionPerformed": False,
                "releaseDecision": None,
                "qualityApproved": False,
                "deliveryApproved": False,
                "hashReads": cache.hash_reads,
                "bytesRead": cache.bytes_read,
                "cacheHits": cache.cache_hits,
                "physicalIOBytes": None,
                "metricScope": "InputHashCache payload reads; bounded metadata parsing excluded",
            }
            if receipt.exists():
                old = _read(_current(receipt))
                stable = {
                    key: value
                    for key, value in result.items()
                    if key not in {"hashReads", "bytesRead", "cacheHits"}
                }
                if {
                    key: value
                    for key, value in old.items()
                    if key not in {"hashReads", "bytesRead", "cacheHits"}
                } != stable:
                    raise ReplayError("Existing retirement receipt differs")
            else:
                save_new(receipt, result)
            return receipt


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    prepare_parser = sub.add_parser("prepare")
    prepare_parser.add_argument("output", type=Path)
    prepare_parser.add_argument("configs", nargs="+", type=Path)
    prepare_parser.add_argument("--phase", choices=("before-delete", "after-delete"), required=True)
    run_parser = sub.add_parser("run")
    run_parser.add_argument("manifest", type=Path)
    run_parser.add_argument("sha256")
    run_parser.add_argument("output", type=Path)
    run_parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.command == "prepare":
        prepare(tuple(args.configs), args.output, phase=_phase(args.phase))
    else:
        run(args.manifest, args.sha256, args.output, resume=args.resume)


if __name__ == "__main__":
    main()
