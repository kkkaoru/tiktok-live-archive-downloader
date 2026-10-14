"""Prepare ready native parts with one fresh, bounded input-digest scope.

This metadata adapter delegates the existing canonical factory. It never renders,
executes a workflow or substitutes a verifier. The existing common batch factory
may be called afterwards, outside this operation's deliberately nonnested cache.
"""

import argparse
import fcntl
import json
import os
import stat
import sys
import time
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

from replay import fcp_part_followup as canonical
from replay.artifact_json import _path
from replay.core import ReplayError
from replay.fcp_part_followup_batch import MAX_PARTS, Reference
from replay.fcp_part_followup_batch import _save_atomic as _save_atomic
from replay.foreground_evidence import _bound_bytes
from replay.input_hash_cache import MAX_INPUTS, InputHashCache, input_hash_context
from replay.recordings import record
from replay.stages import file_digest

ROOT = Path(__file__).resolve().parents[2]
MAX_JSON_BYTES = 8 * 1024 * 1024
CODE_NAMES = tuple(
    dict.fromkeys(
        (*canonical.CODE_NAMES, "fcp_part_followup_batch.py", "fcp_part_followup_prepare_batch.py")
    )
)


@dataclass(frozen=True)
class Part:
    number: int
    config: Reference
    output: Path

    def json(self) -> dict[str, object]:
        return {"part": self.number, "config": self.config.json(), "output": str(self.output)}


@dataclass(frozen=True)
class Request:
    path: Path
    parts: tuple[Part, ...]
    inputs: tuple[Reference, ...]
    max_native_bytes: int


@dataclass(frozen=True)
class Result:
    receipt: Path
    workflows: tuple[Path, ...]
    prepared: int
    adopted: int
    hash_reads: int
    bytes_read: int
    cache_hits: int
    seconds: float


def _work(path: Path) -> Path:
    path = _path(str(path))
    if not path.is_relative_to(ROOT / "edits/work"):
        raise ReplayError("Preparation artifacts require repository-work paths")
    return path


def _input(path: Path) -> Path:
    path = _path(str(path))
    if any(path.is_relative_to(ROOT / name) for name in ("private", "downloads")):
        raise ReplayError("Preparation inputs cannot scan private or downloads data")
    return path


def _ref(value: object) -> Reference:
    row = record(value)
    digest = row.get("sha256")
    if set(row) != {"path", "sha256"} or not isinstance(digest, str):
        raise ReplayError("Preparation references require exact path and SHA256 fields")
    return Reference(_input(_path(row.get("path"))), digest)


def _unique(pairs: list[tuple[str, object]]) -> dict[str, object]:
    if len({key for key, _ in pairs}) != len(pairs):
        raise ReplayError("Preparation metadata keys must be unique")
    return dict(pairs)


def _read(path: Path, digest: str | None = None) -> dict[str, object]:
    digest = file_digest(path) if digest is None else digest
    return record(json.loads(_bound_bytes(path, digest, MAX_JSON_BYTES), object_pairs_hook=_unique))


def _references(paths: tuple[Path, ...]) -> tuple[Reference, ...]:
    return tuple(Reference(_input(path), file_digest(_input(path))) for path in paths)


def _load(path: Path) -> Request:
    path = _work(path)
    raw = _read(path)
    rows, inputs = raw.get("parts"), raw.get("inputs")
    maximum, byte_limit = raw.get("maxReadyParts"), raw.get("maxNativeBytes")
    if (
        set(raw) != {"version", "kind", "parts", "inputs", "maxReadyParts", "maxNativeBytes"}
        or type(raw.get("version")) is not int
        or raw.get("version") != 1
        or raw.get("kind") != "native-followup-preparation-request"
        or type(maximum) is not int
        or not 1 <= maximum <= MAX_PARTS
        or type(byte_limit) is not int
        or byte_limit <= 0
        or not isinstance(rows, list)
        or not 1 <= len(rows) <= maximum
        or not isinstance(inputs, list)
        or not 1 <= len(inputs) < MAX_INPUTS
    ):
        raise ReplayError(
            "Expected bounded ready parts, explicit input closure and native-byte budget"
        )
    parts: list[Part] = []
    for value in rows:
        row = record(value)
        number = row.get("part")
        if (
            set(row) != {"part", "config", "output"}
            or type(number) is not int
            or not 1 <= number <= canonical.MAX_PARTS
        ):
            raise ReplayError("Preparation needs explicit bounded assembly part identities")
        config = _ref(row.get("config"))
        _work(config.path)
        parts.append(Part(number, config, _work(_path(row.get("output")))))
    refs = tuple(_ref(value) for value in inputs)
    if (
        len({part.number for part in parts}) != len(parts)
        or len({part.config.path for part in parts}) != len(parts)
        or len({ref.path for ref in refs}) != len(refs)
    ):
        raise ReplayError("Part identities, configs and explicit input references must be unique")
    return Request(path, tuple(parts), refs, byte_limit)


def _inventory(request: Request) -> tuple[Reference, ...]:
    code = tuple(Path(__file__).with_name(name) for name in CODE_NAMES)
    refs = (*request.inputs, *_references((request.path, *code)))
    unique: dict[Path, Reference] = {}
    for ref in refs:
        if ref.path in unique and unique[ref.path] != ref:
            raise ReplayError("Preparation input fingerprints conflict")
        unique[ref.path] = ref
    if len(unique) >= MAX_INPUTS or any(part.config not in refs for part in request.parts):
        raise ReplayError("Preparation input closure is excessive or omits a part configuration")
    return tuple(unique.values())


def _plans(request: Request, refs: tuple[Reference, ...]) -> tuple[canonical.Followup, ...]:
    expected = {request.path, *(Path(__file__).with_name(name) for name in CODE_NAMES)}
    plans: list[canonical.Followup] = []
    for part in request.parts:
        plan = canonical._load(part.config.path, allow_existing=True)
        if plan.part != part.number:
            raise ReplayError("Requested assembly part differs from the canonical configuration")
        plans.append(plan)
        expected.update(plan.inputs)
        expected.update(canonical._cleanup_inputs(plan.paths["sourceCleanup"]))
    if expected != {ref.path for ref in refs}:
        raise ReplayError("Explicit preparation closure differs from the current canonical factory")
    if (
        len({plan.semantic for plan in plans}) != 1
        or len({plan.paths["sourceCleanup"] for plan in plans}) != 1
        or sum(plan.movie.stat().st_size for plan in plans) > request.max_native_bytes
    ):
        raise ReplayError(
            "Ready parts must share semantic cleanup and fit their observed byte budget"
        )
    return tuple(plans)


def _namespace(
    request: Request, refs: tuple[Reference, ...], output: Path, *, effects: tuple[Path, ...] = ()
) -> None:
    roots = (output, *(part.output for part in request.parts), *effects)
    for index, path in enumerate(roots):
        if not path.parent.is_dir() or any(
            path.is_relative_to(other) or other.is_relative_to(path)
            for other in (*roots[:index], *(ref.path for ref in refs))
        ):
            raise ReplayError(
                "Preparation output namespaces overlap protected inputs or each other"
            )


@contextmanager
def _lease(output: Path, *, resume: bool) -> Iterator[None]:
    if output.exists():
        if not resume or output.is_symlink() or not output.is_dir():
            raise ReplayError("Occupied preparation root requires explicit verified resume")
    elif resume:
        raise ReplayError("Resume requires an existing immutable preparation operation")
    else:
        output.mkdir(mode=0o700)
    flags = os.O_RDWR | os.O_NOFOLLOW
    if not resume:
        flags |= os.O_CREAT | os.O_EXCL
    try:
        descriptor = os.open(output / "prepare.lock", flags, 0o600)
    except FileNotFoundError as error:
        raise ReplayError("Resume requires the original preparation lease") from error
    with os.fdopen(descriptor, "rb") as stream:
        value = os.fstat(stream.fileno())
        if not stat.S_ISREG(value.st_mode) or value.st_nlink != 1:
            raise ReplayError("Preparation lease must be a regular single-link file")
        try:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise ReplayError("Preparation operation is owned by another caller") from error
        yield


def _identity(request: Request, refs: tuple[Reference, ...]) -> dict[str, object]:
    return {
        "version": 1,
        "kind": "immutable-native-followup-preparation-operation",
        "request": _references((request.path,))[0].json(),
        "inputs": [ref.json() for ref in refs],
        "parts": [part.json() for part in request.parts],
        "qualityApproved": False,
    }


def _initialize(output: Path, identity: dict[str, object], *, resume: bool) -> None:
    operation, checkpoints = output / "operation.json", output / "checkpoints"
    if resume:
        if _read(operation) != identity or not checkpoints.is_dir() or checkpoints.is_symlink():
            raise ReplayError("Existing preparation operation differs from current pinned inputs")
    else:
        _save_atomic(operation, identity)
        checkpoints.mkdir(mode=0o700)


def _artifacts(part: Part, plan: canonical.Followup) -> tuple[Reference, ...]:
    paths = tuple(
        part.output / name for name in ("workflow.json", "operation.json", "completion-config.json")
    )
    refs = _references(paths)
    data, inputs = canonical._operation(paths[1], plan.semantic)
    current = canonical._integrated_plan(paths[1], plan.semantic, data, inputs)
    graph = _read(paths[0])
    rows = graph.get("stages")
    if (
        current.part != part.number
        or data.get("config") != str(part.config.path)
        or not isinstance(rows, list)
        or len(rows) != 1
        or record(rows[0]).get("command")
        != canonical._command(
            "fcp_part_followup", "run-integrated", str(paths[1]), str(plan.semantic)
        )
        or record(rows[0]).get("verify_command")
        != canonical._command(
            "fcp_part_followup", "verify-integrated", str(paths[1]), str(plan.semantic)
        )
        or graph.get("sourceCaptionProduction")
        != {"bindings": {"followup": str(plan.paths["sourceCleanup"])}}
    ):
        raise ReplayError("Prepared metadata no longer binds its canonical operation and verifier")
    return refs


def _part(part: Part, plan: canonical.Followup, output: Path, identity: dict[str, object]) -> bool:
    checkpoint = output / "checkpoints" / f"part-{part.number:04d}.json"
    if checkpoint.exists() or checkpoint.is_symlink():
        raw = _read(_work(checkpoint))
        if raw != _checkpoint_value(part, plan, identity):
            raise ReplayError("Preparation checkpoint differs from its exact metadata bytes")
        return False
    if (
        part.output.exists()
        or part.output.is_symlink()
        or any(
            path.exists() or path.is_symlink()
            for path in (plan.proof, plan.destination, plan.published_proof, plan.publication)
        )
    ):
        raise ReplayError(
            "Uncheckpointed or uncertain preparation effects require separate recovery"
        )
    canonical.prepare(part.config.path, part.output, integrated=True)
    _save_atomic(checkpoint, _checkpoint_value(part, plan, identity))
    return True


def _checkpoint_value(
    part: Part, plan: canonical.Followup, identity: dict[str, object]
) -> dict[str, object]:
    return {
        "version": 1,
        "kind": "pending-native-followup-preparation",
        "operation": identity,
        "part": part.json(),
        "artifacts": [ref.json() for ref in _artifacts(part, plan)],
        "qualityApproved": False,
    }


def prepare(request_path: Path, output: Path, *, resume: bool = False) -> Result:
    """Delegate metadata preparation only; resume never executes a child workflow."""
    if type(resume) is not bool:
        raise ReplayError("Preparation resume must be explicit boolean intent")
    started = time.monotonic()
    request, output = _load(request_path), _work(output)
    refs = _inventory(request)
    _namespace(request, refs, output)
    cache = InputHashCache((ref.path for ref in refs), allow_uncached_overflow=True)
    prepared = 0
    with input_hash_context(cache):
        cache.prime()
        if any(file_digest(ref.path) != ref.sha256 for ref in refs):
            raise ReplayError("Preparation input bytes differ from their immutable fingerprints")
        plans = _plans(request, refs)
        _namespace(
            request,
            refs,
            output,
            effects=tuple(
                path
                for plan in plans
                for path in (plan.proof, plan.destination, plan.published_proof, plan.publication)
            ),
        )
        identity = _identity(request, refs)
        cache.guard()
        with _lease(output, resume=resume):
            _initialize(output, identity, resume=resume)
            for part, plan in zip(request.parts, plans, strict=True):
                cache.guard()
                prepared += _part(part, plan, output, identity)
                cache.guard()
            for part, plan in zip(request.parts, plans, strict=True):
                checkpoint = output / "checkpoints" / f"part-{part.number:04d}.json"
                if _read(checkpoint) != _checkpoint_value(part, plan, identity):
                    raise ReplayError("Prepared metadata changed after its immutable checkpoint")
            checkpoints = _references(
                tuple(
                    output / "checkpoints" / f"part-{part.number:04d}.json"
                    for part in request.parts
                )
            )
            metadata = tuple(
                ref
                for part, plan in zip(request.parts, plans, strict=True)
                for ref in _artifacts(part, plan)
            )
            generated = InputHashCache(ref.path for ref in (*checkpoints, *metadata))
            generated.prime()
            if any(file_digest(ref.path) != ref.sha256 for ref in (*checkpoints, *metadata)):
                raise ReplayError(
                    "Generated preparation metadata changed before terminal verification"
                )
            verified = cache.verify()
            generated_verified = generated.verify()
            receipt = output / "prepared.json"
            acceptance = {
                "version": 1,
                "kind": "verified-native-followup-preparation",
                "operation": identity,
                "inputs": verified,
                "checkpoints": [ref.json() for ref in checkpoints],
                "generatedMetadata": generated_verified,
                "qualityApproved": False,
                "deliveryApproved": False,
                "workflowsExecuted": False,
            }
            if receipt.exists() or receipt.is_symlink():
                raw = _read(_work(receipt))
                if raw.get("acceptance") != acceptance:
                    raise ReplayError(
                        "Completed preparation receipt changed; preserve it for investigation"
                    )
            else:
                _save_atomic(
                    receipt,
                    {
                        "acceptance": acceptance,
                        "metrics": {
                            "parentCacheHashReads": cache.hash_reads,
                            "parentCacheBytesRead": cache.bytes_read,
                            "parentCacheHits": cache.cache_hits,
                            "generatedMetadataHashReads": generated.hash_reads,
                            "generatedMetadataBytesRead": generated.bytes_read,
                            "seconds": time.monotonic() - started,
                            "delegatedOrdinaryHashBytes": None,
                            "childHashBytes": None,
                            "physicalIOBytes": None,
                        },
                    },
                )
    return Result(
        receipt,
        tuple(part.output / "workflow.json" for part in request.parts),
        prepared,
        len(request.parts) - prepared,
        cache.hash_reads,
        cache.bytes_read,
        cache.cache_hits,
        time.monotonic() - started,
    )


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("request", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args(argv)
    result = prepare(args.request, args.output, resume=args.resume)
    print(
        json.dumps(
            {"receipt": str(result.receipt), "prepared": result.prepared, "adopted": result.adopted}
        )
    )


if __name__ == "__main__":
    main(sys.argv[1:])
