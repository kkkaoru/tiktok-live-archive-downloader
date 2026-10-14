"""Fresh outer boundaries for provisional, explicitly scoped measurement caches.

Existing foreground/caption producers and domain readers remain the backend.
Child cache hits cannot approve a unit or batch. This does not freeze filesystem
bytes atomically, prove raster quality, or change a producer completion record.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os
import re
import stat
import sys
from collections.abc import Iterator
from contextlib import ExitStack, contextmanager
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Literal

from replay import (
    caption_linguistics,
    caption_measured_job,
    caption_resource_inputs,
    digest_store,
    fcp_measured,
    foreground_collection,
    foreground_evidence,
    foreground_retained,
    workflow,
)
from replay.caption_measured_plan import MAX_TOKENS
from replay.caption_source_support import MAX_CUTS, MAX_WINDOWS
from replay.core import ReplayError
from replay.input_hash_cache import InputHashCache, input_hash_context
from replay.recordings import record
from replay.stages import execution_lock, file_digest

ROOT = Path(__file__).resolve().parents[2]
WORK = ROOT / "edits/work"
MAX_JSON = 8 * 1024 * 1024
MAX_UNITS = 4096
MAX_INPUTS = 4096
CODE_LIMIT = 512
Kind = Literal["foreground", "caption"]
Identity = tuple[int, int, int, int, int, int, int]


@dataclass(frozen=True, slots=True)
class Reference:
    path: Path
    sha256: str

    def json(self) -> dict[str, str]:
        return {"path": str(self.path), "sha256": self.sha256}


@dataclass(frozen=True, slots=True)
class Unit:
    number: int
    kind: Kind
    configuration: Reference
    output: Path
    resources: Reference | None


@dataclass(frozen=True, slots=True)
class Batch:
    configuration: Reference
    inputs: tuple[Reference, ...]
    units: tuple[Unit, ...]
    store: Path
    state: Path
    receipt: Path


def _path(value: object, *, work: bool = False) -> Path:
    if not isinstance(value, str) or "\0" in value:
        raise ReplayError("Measurement paths require explicit strings")
    path = Path(value)
    if not path.is_absolute() or path != path.resolve() or ".." in path.parts:
        raise ReplayError("Measurement paths must be canonical without links")
    if work and not path.is_relative_to(WORK):
        raise ReplayError("Measurement outputs must stay in repository work storage")
    return path


def _reference(value: object) -> Reference:
    raw = record(value)
    if set(raw) != {"path", "sha256"}:
        raise ReplayError("Measurement references require exact path/SHA256")
    digest = raw["sha256"]
    if not isinstance(digest, str) or re.fullmatch(r"[a-f0-9]{64}", digest) is None:
        raise ReplayError("Measurement references require lowercase SHA256")
    return Reference(_path(raw["path"]), digest)


def _identity(path: Path) -> Identity:
    _path(str(path))
    value = path.lstat()
    if not stat.S_ISREG(value.st_mode) or value.st_size <= 0:
        raise ReplayError("Measurement inputs must be nonempty regular files")
    return (
        value.st_dev,
        value.st_ino,
        value.st_size,
        value.st_mtime_ns,
        value.st_ctime_ns,
        value.st_mode,
        value.st_nlink,
    )


def _pairs(values: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for name, value in values:
        if name in result:
            raise ReplayError("Duplicate measurement metadata fields")
        result[name] = value
    return result


def _constant(value: str) -> object:
    raise ReplayError(f"Non-finite measurement JSON: {value}")


def _document(reference: Reference) -> dict[str, object]:
    before = _identity(reference.path)
    if before[2] > MAX_JSON:
        raise ReplayError("Measurement metadata exceeds its byte budget")
    descriptor = os.open(reference.path, os.O_RDONLY | os.O_NOFOLLOW)
    with os.fdopen(descriptor, "rb") as stream:
        opened = os.fstat(stream.fileno())
        if (opened.st_dev, opened.st_ino) != before[:2]:
            raise ReplayError("Measurement metadata changed before opening")
        data = stream.read(MAX_JSON + 1)
        if _identity(reference.path) != before or len(data) > MAX_JSON:
            raise ReplayError("Measurement metadata changed during reading")
    if hashlib.sha256(data).hexdigest() != reference.sha256:
        raise ReplayError("Measurement metadata SHA256 changed")
    value: object = json.loads(data, object_pairs_hook=_pairs, parse_constant=_constant)
    return record(value)


def _rows(value: object, *, maximum: int) -> list[object]:
    if not isinstance(value, list) or not 1 <= len(value) <= maximum:
        raise ReplayError("Missing or excessive measurement rows")
    return value


def code_paths() -> tuple[Path, ...]:
    """Bound the current local import closure; never scan user data directories."""
    pending = {"measurement_digest_batch"}
    names: set[str] = set()
    while pending:
        name = pending.pop()
        if name in names:
            continue
        if re.fullmatch(r"[a-z][a-z0-9_]*", name) is None or len(names) >= CODE_LIMIT:
            raise ReplayError("Measurement code closure exceeds its bounded package scope")
        names.add(name)
        path = ROOT / "src/replay" / f"{name}.py"
        if not 0 < _identity(path)[2] <= MAX_JSON:
            raise ReplayError("Measurement code must be bounded regular source")
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            if isinstance(node, ast.ImportFrom) and node.module == "replay":
                pending.update(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module is not None:
                if node.module.startswith("replay."):
                    pending.add(node.module.removeprefix("replay."))
            elif isinstance(node, ast.Import):
                pending.update(
                    alias.name.removeprefix("replay.")
                    for alias in node.names
                    if alias.name.startswith("replay.")
                )
    return tuple(ROOT / "src/replay" / f"{name}.py" for name in sorted(names))


def _require(reference: Reference, pins: dict[Path, str], required: set[Path]) -> None:
    if pins.get(reference.path) != reference.sha256:
        raise ReplayError("Measurement transitive pin is absent or conflicts")
    required.add(reference.path)


def _foreground(reference: Reference, pins: dict[Path, str], required: set[Path]) -> None:
    _require(reference, pins, required)
    _document(reference)
    job = foreground_collection.load_job(reference.path)
    if job.config_hash != reference.sha256:
        raise ReplayError("Foreground configuration identity changed")
    _require(Reference(job.source, job.source_hash), pins, required)
    _require(Reference(job.native, job.native_hash), pins, required)
    for name in foreground_collection.CODE_FILES:
        path = ROOT / "src/replay" / name
        if path not in pins:
            raise ReplayError("Foreground producer code is absent from the closure")
        required.add(path)


def _retained(
    value: object, pins: dict[Path, str], required: set[Path]
) -> tuple[Reference, Reference, Reference | None]:
    raw = record(value)
    if set(raw) not in ({"config", "provenance"}, {"config", "provenance", "retainedVerification"}):
        raise ReplayError("Unsupported measurement foreground reference")
    config, provenance = _reference(raw["config"]), _reference(raw["provenance"])
    _require(config, pins, required)
    _document(config)
    job = foreground_collection.load_job(config.path)
    _require(Reference(job.source, job.source_hash), pins, required)
    _require(Reference(job.native, job.native_hash), pins, required)
    descriptor = _reference(raw["retainedVerification"]) if "retainedVerification" in raw else None
    archives: dict[str, Path] = {}
    if descriptor is not None:
        _require(descriptor, pins, required)
        archives = foreground_retained._archive_paths(descriptor.path, descriptor.sha256)
    else:
        _foreground(config, pins, required)
    _require(provenance, pins, required)
    receipt = _document(provenance)
    if provenance.path.name != "provenance.json":
        raise ReplayError("Foreground provenance must name its original receipt")
    for name, digest in record(receipt.get("inputs")).items():
        original = _path(name)
        historical = archives.get(original.name) if original.parent == ROOT / "src/replay" else None
        _require(
            _reference({"path": str(historical or original), "sha256": digest}), pins, required
        )
    for number, value in enumerate(_rows(receipt.get("batches"), maximum=16)):
        batch = record(value)
        directory = provenance.path.parent / f"batch-{number:04d}"
        _require(
            _reference(
                {"path": str(directory / "request.json"), "sha256": batch.get("requestSHA256")}
            ),
            pins,
            required,
        )
        _require(
            _reference(
                {"path": str(directory / "response.json"), "sha256": batch.get("responseSHA256")}
            ),
            pins,
            required,
        )
    for path in archives.values():
        if path not in pins:
            raise ReplayError("Retained producer archive is not pinned")
        required.add(path)
    return config, provenance, descriptor


def _caption(unit: Unit, pins: dict[Path, str], required: set[Path]) -> None:
    raw = _document(unit.configuration)
    expected = (
        "version tokens spans foreground native calibration outputFrameRate "
        "region centerX clearance"
    )
    if set(raw) != set(expected.split()) or type(raw["version"]) is not int or raw["version"] != 1:
        raise ReplayError("Unsupported original caption measurement manifest")
    tokens, calibration = _reference(raw["tokens"]), _reference(raw["calibration"])
    _require(tokens, pins, required)
    _require(calibration, pins, required)
    token_data = _document(tokens)
    if (
        set(token_data) != {"version", "clock", "tokens"}
        or type(token_data["version"]) is not int
        or token_data["version"] != 1
        or token_data["clock"] != "output"
    ):
        raise ReplayError("Caption tokens must preserve their original output clock")
    tokens_raw = _rows(token_data["tokens"], maximum=MAX_TOKENS)
    spoken = tuple(caption_measured_job._token(row) for row in tokens_raw)
    _require(_reference(raw["native"]), pins, required)
    calibrated = _document(calibration)
    _require(_reference(calibrated.get("template")), pins, required)
    for value in _rows(calibrated.get("evidence"), maximum=32):
        _require(_reference(value), pins, required)
    for value in _rows(raw["foreground"], maximum=MAX_WINDOWS):
        _retained(value, pins, required)
    for row in _rows(raw["spans"], maximum=MAX_CUTS):
        caption_measured_job._span(row)
    if unit.resources is None:
        raise ReplayError("Caption admission requires observed font resource evidence")
    _require(unit.resources, pins, required)
    resources = _document(unit.resources)
    if set(resources) != {"motion", "evidence"}:
        raise ReplayError("Caption resources require the original font evidence contract")
    store = caption_resource_inputs.ResourceInputs()
    texts = {token.text for token in spoken}
    caption_resource_inputs._font_evidence(
        store, record(resources["motion"]), record(resources["evidence"]), texts
    )
    for path, digest in store.pins.items():
        _require(Reference(path, digest), pins, required)
    dictionary = ROOT / caption_linguistics.DICTIONARY_PATH
    _require(Reference(dictionary, caption_linguistics.DICTIONARY_SHA256), pins, required)
    for name in caption_measured_job.CODE_FILES:
        path = ROOT / "src/replay" / name
        if path not in pins:
            raise ReplayError("Caption producer code is absent from the closure")
        required.add(path)


def _closure(batch: Batch) -> None:
    pins = {reference.path: reference.sha256 for reference in batch.inputs}
    required = {batch.configuration.path}
    for path in code_paths():
        if path not in pins:
            raise ReplayError("Measurement controller code is absent from the closure")
        required.add(path)
    for unit in batch.units:
        _require(unit.configuration, pins, required)
        if unit.kind == "foreground":
            _foreground(unit.configuration, pins, required)
        else:
            _caption(unit, pins, required)
    if required != set(pins):
        raise ReplayError("Measurement input closure contains unrelated or missing roles")


def load(path: Path, sha256: str) -> Batch:
    """Validate a bounded explicit input closure without any producer effects."""
    reference = _reference({"path": str(path), "sha256": sha256})
    _path(str(path), work=True)
    raw = _document(reference)
    if (
        set(raw) != {"version", "inputs", "units", "store", "state", "receipt"}
        or type(raw["version"]) is not int
        or raw["version"] != 1
    ):
        raise ReplayError("Unsupported measurement batch configuration")
    inputs = tuple(_reference(value) for value in _rows(raw["inputs"], maximum=MAX_INPUTS - 1))
    if len({item.path for item in inputs}) != len(inputs) or path in {item.path for item in inputs}:
        raise ReplayError("Duplicate or recursive measurement pins")
    units: list[Unit] = []
    for number, value in enumerate(_rows(raw["units"], maximum=MAX_UNITS)):
        row = record(value)
        if set(row) != {"kind", "configuration", "output", "resources"} or row["kind"] not in {
            "foreground",
            "caption",
        }:
            raise ReplayError("Unsupported measurement unit")
        kind: Kind = "foreground" if row["kind"] == "foreground" else "caption"
        resources = _reference(row["resources"]) if row["resources"] is not None else None
        if kind == "foreground" and resources is not None:
            raise ReplayError("Foreground units cannot introduce caption resource roles")
        units.append(
            Unit(
                number,
                kind,
                _reference(row["configuration"]),
                _path(row["output"], work=True),
                resources,
            )
        )
    batch = Batch(
        reference,
        (reference, *inputs),
        tuple(units),
        _path(raw["store"], work=True),
        _path(raw["state"], work=True),
        _path(raw["receipt"], work=True),
    )
    outputs = (batch.store, batch.state, batch.receipt, *(unit.output for unit in batch.units))
    if len(set(outputs)) != len(outputs) or any(
        left != right and left.is_relative_to(right) for left in outputs for right in outputs
    ):
        raise ReplayError("Measurement storage roles must be distinct and non-nested")
    if any(
        reference.path == output or reference.path.is_relative_to(output)
        for reference in batch.inputs
        for output in outputs
    ):
        raise ReplayError("Measurement outputs cannot contain declared inputs")
    if any(not output.parent.is_dir() for output in outputs):
        raise ReplayError("Measurement storage needs existing canonical parents")
    _closure(batch)
    return batch


def _candidate(unit: Unit, batch: Batch) -> dict[str, object]:
    manifest = _document(unit.configuration)
    path = unit.output / "candidate.json"
    reference = Reference(path, file_digest(path))
    raw = _document(reference)
    pins = {item.path: item.sha256 for item in batch.inputs}
    candidate_inputs = fcp_measured._input_bindings(raw.get("inputs"))
    required = (
        unit.configuration,
        _reference(manifest["tokens"]),
        _reference(manifest["native"]),
        _reference(manifest["calibration"]),
    )
    if any(candidate_inputs.get(str(item.path)) != item.sha256 for item in required):
        raise ReplayError("Candidate lacks its original producer input roles")
    for name, digest in candidate_inputs.items():
        item = _reference({"path": name, "sha256": digest})
        if item.path in pins:
            if pins[item.path] != item.sha256:
                raise ReplayError("Candidate shared input pin changed")
        elif not item.path.is_relative_to(unit.output):
            raise ReplayError("Candidate introduced an undeclared preexisting input")
        elif item.path.suffix != ".json" or _identity(item.path)[2] > MAX_JSON:
            raise ReplayError("Candidate growth must remain bounded generated metadata")
        if file_digest(item.path) != item.sha256:
            raise ReplayError("Candidate generated or shared evidence changed")
    foreground = _rows(manifest["foreground"], maximum=MAX_WINDOWS)
    config, provenance, descriptor = _retained(foreground[0], pins, set())
    job = foreground_collection.load_job(config.path)
    for value in foreground:
        config, provenance, descriptor = _retained(value, pins, set())
        if descriptor is None:
            foreground_evidence.read_window(
                config.path,
                provenance.path.parent,
                config_sha256=config.sha256,
                provenance_sha256=provenance.sha256,
            )
        else:
            foreground_retained.verify(
                config.path,
                provenance.path.parent,
                config_sha256=config.sha256,
                provenance_sha256=provenance.sha256,
                descriptor=descriptor.path,
                descriptor_sha256=descriptor.sha256,
            )
    spans = tuple(
        caption_measured_job._span(value) for value in _rows(manifest["spans"], maximum=MAX_CUTS)
    )
    duration = max(span.output_start + span.duration for span in spans)
    rate = manifest["outputFrameRate"]
    if type(rate) is not int:
        raise ReplayError("Caption output frame rate must remain an integer")
    fcp_measured.read_measured_captions(
        path,
        sha256=reference.sha256,
        width=job.width,
        height=job.height,
        rate=rate,
        duration=duration,
    )
    if any(
        raw.get(key) is not False
        for key in (
            "editorCorrespondenceVerified",
            "completeMotionSupportVerified",
            "motionRenderVerified",
            "semanticPlacementApproved",
            "captionGeometryApproved",
            "aaaContrastApproved",
            "deliveryComplete",
        )
    ):
        raise ReplayError("Cached measurements cannot acquire quality or delivery approval")
    return {"unit": unit.number, "kind": unit.kind, "output": reference.json(), "provisional": True}


def verify_unit(batch: Batch, number: int) -> dict[str, object]:
    if not 0 <= number < len(batch.units):
        raise ReplayError("Unknown measurement unit")
    unit = batch.units[number]
    if unit.kind == "caption":
        return _candidate(unit, batch)
    foreground_collection.verify(unit.configuration.path, unit.output)
    reference = Reference(
        unit.output / "provenance.json", file_digest(unit.output / "provenance.json")
    )
    return {"unit": number, "kind": unit.kind, "output": reference.json(), "provisional": True}


@contextmanager
def _router(batch: Batch, unit: Unit) -> Iterator[dict[str, int]]:
    """Child-only allowlist; unknown inputs reject and generated evidence stays fresh."""
    original = file_digest
    pins = {reference.path: reference.sha256 for reference in batch.inputs}
    identities = {path: _identity(path) for path in pins}
    counts = {"cacheCalls": 0, "generatedFreshCalls": 0, "cacheByteReads": 0, "cacheBytesRead": 0}
    original_store_hash = digest_store.hash_file

    def measured_store_hash(path: Path) -> str:
        size = _identity(path)[2]
        result = original_store_hash(path)
        counts["cacheByteReads"] += 1
        counts["cacheBytesRead"] += size
        return result

    def digest(path: Path) -> str:
        path = _path(str(path.absolute()))
        if path not in pins:
            if not path.is_relative_to(unit.output):
                raise ReplayError("Child attempted an undeclared measurement input")
            if path.suffix != ".json" or _identity(path)[2] > MAX_JSON:
                raise ReplayError("Child generated inputs must remain bounded metadata")
            counts["generatedFreshCalls"] += 1
            return original(path)
        if _identity(path) != identities[path]:
            raise ReplayError("Child shared input identity changed")
        counts["cacheCalls"] += 1
        actual = digest_store.cached_digest(batch.store, path)
        if actual != pins[path] or _identity(path) != identities[path]:
            raise ReplayError("Child shared input pin changed")
        return actual

    with ExitStack() as restore:
        digest_store.hash_file = measured_store_hash
        restore.callback(setattr, digest_store, "hash_file", original_store_hash)
        for name, module in tuple(sys.modules.items()):
            if (
                name.startswith("replay.")
                and isinstance(module, ModuleType)
                and getattr(module, "file_digest", None) is original
            ):
                module.__dict__["file_digest"] = digest
                restore.callback(module.__dict__.__setitem__, "file_digest", original)
        yield counts


def child(batch: Batch, number: int, *, verify_only: bool) -> None:
    """Provisional child adapter only; it never publishes the outer acceptance."""
    if not 0 <= number < len(batch.units):
        raise ReplayError("Unknown measurement child")
    unit = batch.units[number]
    with _router(batch, unit) as counts:
        if verify_only:
            verify_unit(batch, number)
        elif unit.kind == "foreground":
            foreground_collection.observe(unit.configuration.path, unit.output)
        else:
            caption_measured_job.plan(
                unit.configuration.path, unit.output, manifest_sha256=unit.configuration.sha256
            )
    if not verify_only:
        _write(
            unit.output / "cache-observation.json",
            {
                "version": 1,
                "configuration": batch.configuration.json(),
                "unit": number,
                "provisional": True,
                "scope": "producer-store-reads-only",
                "physicalIOBytes": None,
                "counts": counts,
            },
        )


def _write(path: Path, value: object) -> None:
    data = (json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n").encode()
    if len(data) > MAX_JSON:
        raise ReplayError("Measurement output metadata exceeds its byte budget")
    _path(str(path), work=True)
    with ExitStack() as close:
        parent = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
        close.callback(os.close, parent)
        _path(str(path.parent), work=True)
        opened = os.fstat(parent)
        before = path.parent.lstat()
        if (opened.st_dev, opened.st_ino) != (before.st_dev, before.st_ino):
            raise ReplayError("Measurement output parent changed")
        descriptor = os.open(
            path.name, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600, dir_fd=parent
        )
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
            after = os.fstat(stream.fileno())
            observed = path.lstat()
            if (
                _path(str(path), work=True) != path
                or (after.st_dev, after.st_ino) != (observed.st_dev, observed.st_ino)
                or after.st_nlink != 1
                or after.st_mode & 0o077
            ):
                raise ReplayError("Measurement output identity changed")
        os.fsync(parent)
        observed_parent = path.parent.lstat()
        if (observed_parent.st_dev, observed_parent.st_ino) != (before.st_dev, before.st_ino):
            raise ReplayError("Measurement output parent changed after publication")


def _result(batch: Batch, units: list[dict[str, object]]) -> dict[str, object]:
    return {
        "version": 1,
        "kind": "fresh-measurement-batch",
        "configuration": batch.configuration.json(),
        "inputs": [item.json() for item in batch.inputs],
        "units": units,
        "boundaryScope": "outer-batch",
        "independentPerUnitFreshness": False,
        "complete": True,
        "qualityApproved": False,
        "deliveryApproved": False,
        "physicalIOBytes": None,
        "coldCacheReads": "not-inferred-from-cache-call-count",
    }


def _scope(batch: Batch) -> InputHashCache:
    cache = InputHashCache((item.path for item in batch.inputs), allow_uncached_overflow=True)
    cache.prime()
    return cache


def _expected(batch: Batch) -> None:
    if any(file_digest(item.path) != item.sha256 for item in batch.inputs):
        raise ReplayError("Measurement fresh initial input differs from its declared pin")
    _closure(batch)


def verify(batch: Batch, receipt: Reference) -> dict[str, object]:
    """Fresh resume acceptance, all original domains, and no producer effects."""
    cache = _scope(batch)
    with input_hash_context(cache):
        _expected(batch)
        units = [verify_unit(batch, unit.number) for unit in batch.units]
        expected = _result(batch, units)
        if _document(receipt) != expected:
            raise ReplayError("Measurement batch receipt differs from actual domain evidence")
        cache.verify()
        cache.guard()
    return expected


def run(batch: Batch, *, resume: bool = False) -> Path:
    """Reuse the public workflow runner; accept only after fresh terminal verification."""
    if batch.receipt.exists():
        if not resume:
            raise ReplayError("Occupied measurement receipt requires explicit readonly resume")
        verify(batch, Reference(batch.receipt, file_digest(batch.receipt)))
        return batch.receipt
    cache = _scope(batch)
    with input_hash_context(cache):
        _expected(batch)
    batch.state.mkdir(mode=0o700, exist_ok=resume)
    with execution_lock(batch.state, "measurement-batch") as acquired:
        if not acquired:
            raise ReplayError("Another process owns this measurement batch")
        with input_hash_context(cache):
            _expected(batch)
            stages: list[dict[str, object]] = []
            for unit in batch.units:
                command = [sys.executable, "-m", "replay.measurement_digest_batch"]
                arguments = [
                    str(batch.configuration.path),
                    batch.configuration.sha256,
                    str(unit.number),
                ]
                output = unit.output / (
                    "candidate.json" if unit.kind == "caption" else "provenance.json"
                )
                stages.append(
                    {
                        "name": f"measurement-{unit.number:04d}",
                        "command": [*command, "child-run", *arguments],
                        "verify_command": [*command, "child-verify", *arguments],
                        "inputs": [str(item.path) for item in batch.inputs],
                        "outputs": [str(output)],
                        "timeout_seconds": foreground_collection.STAGE_TIMEOUT_SECONDS
                        if unit.kind == "foreground"
                        else caption_measured_job.JOB_SECONDS,
                        "adopt_existing": True,
                    }
                )
            plan = batch.state / "workflow.json"
            definition = {"version": 1, "stages": stages}
            if plan.exists():
                if not resume or _document(Reference(plan, file_digest(plan))) != definition:
                    raise ReplayError("Occupied measurement workflow differs")
            else:
                _write(plan, definition)
            workflow.run_plan(plan, state_directory=batch.state / "stages")
            units = [verify_unit(batch, unit.number) for unit in batch.units]
            cache.verify()
            _closure(batch)
            cache.guard()
            _write(batch.receipt, _result(batch, units))
            cache.guard()
    return batch.receipt


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("run", "verify", "child-run", "child-verify"))
    parser.add_argument("configuration", type=Path)
    parser.add_argument("sha256")
    parser.add_argument("number", nargs="?", type=int)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    batch = load(args.configuration, args.sha256)
    if args.action.startswith("child-"):
        if args.number is None:
            parser.error("child actions require a unit number")
        child(batch, args.number, verify_only=args.action == "child-verify")
    elif args.action == "verify":
        print(json.dumps(verify(batch, Reference(batch.receipt, file_digest(batch.receipt)))))
    else:
        print(run(batch, resume=args.resume))


if __name__ == "__main__":
    main()
