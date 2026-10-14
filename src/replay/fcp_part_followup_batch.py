"""Bounded, resumable control primitives for existing native followup backends.

This module never submits apps, encodes media or supplies a replacement verifier.
The backend must retain the existing frame/AV/native/publication domain checks.
Checkpoints remain pending until the batch's fresh terminal input check succeeds.

After an interrupted workflow, first prove its runner stopped and is unowned,
then use replay.stage_recovery on that same workflow and state. It preserves the
failed outer state and these child checkpoints; rerunning the pinned workflow
then resumes the controller. Plain reruns intentionally use read-only verification.
"""

import argparse
import fcntl
import json
import os
import re
import sys
import tempfile
import time
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Protocol

from replay import fcp_part_followup as canonical
from replay.artifact_json import _json, _path, _pin
from replay.core import ReplayError
from replay.fcp_part_assembly import _reference, probe
from replay.fcp_part_assembly import verify as verify_frames
from replay.input_hash_cache import MAX_INPUTS, InputHashCache, input_hash_context
from replay.native_export_receipt import complete
from replay.part_master_publication import _completed, publish, verify_retirement
from replay.recordings import record
from replay.source_caption_binding import clean_source
from replay.stages import file_digest

ROOT = Path(__file__).resolve().parents[2]
MAX_PARTS = 32
MAX_OUTPUTS = 16


class Step(StrEnum):
    FRAMES = "frames"
    COMPLETE = "complete"
    PUBLICATION_CONFIG = "publication-config"
    PUBLISH = "publish"


@dataclass(frozen=True)
class Reference:
    path: Path
    sha256: str

    def __post_init__(self) -> None:
        _path(str(self.path))
        if re.fullmatch(r"[a-f0-9]{64}", self.sha256) is None:
            raise ReplayError("Batch references require SHA256 fingerprints")

    def json(self) -> dict[str, object]:
        return {"path": str(self.path), "sha256": self.sha256}


@dataclass(frozen=True)
class Part:
    number: int
    operation: Reference
    native: Reference
    retirement: Reference | None = None
    adoption: tuple[Reference, ...] = ()

    def __post_init__(self) -> None:
        if type(self.number) is not int or not 1 <= self.number <= 1024:
            raise ReplayError("Batch requires a bounded native partition number")


class Backend(Protocol):
    """Existing primitive calls with explicit output namespaces and domain checks."""

    def bind_inputs(self, inputs: tuple[Reference, ...]) -> None: ...

    def outputs(self, part: Part, step: Step) -> tuple[Path, ...]: ...

    def effects(self, part: Part, step: Step) -> tuple[Path, ...]: ...

    def execute(self, part: Part, step: Step) -> None: ...

    def verify(self, part: Part, step: Step, outputs: tuple[Reference, ...]) -> None: ...


@dataclass(frozen=True)
class Plan:
    common: tuple[Reference, ...]
    parts: tuple[Part, ...]
    checkpoints: Path
    terminal: Path
    backend_code: tuple[Reference, ...]


@dataclass(frozen=True)
class Result:
    executed: int
    adopted: int
    hash_reads: int
    bytes_read: int


@dataclass(frozen=True)
class Task:
    part: Part
    step: Step
    outputs: tuple[Path, ...]
    effects: tuple[Path, ...]


def _work(path: Path) -> Path:
    path = _path(str(path))
    if not path.is_relative_to(ROOT / "edits/work"):
        raise ReplayError("Batch checkpoints require canonical repository work paths")
    return path


def _inventory(plan: Plan) -> tuple[Reference, ...]:
    if not plan.common or not 1 <= len(plan.parts) <= MAX_PARTS:
        raise ReplayError("Batch common inputs or partition count are invalid")
    if not plan.backend_code or any(ref not in plan.common for ref in plan.backend_code):
        raise ReplayError("Batch backend implementation pins must be declared common inputs")
    if len({part.number for part in plan.parts}) != len(plan.parts):
        raise ReplayError("Batch partition numbers must be unique")
    refs = list(plan.common)
    for part in plan.parts:
        refs.extend((part.operation, part.native))
        if part.retirement is not None:
            refs.append(part.retirement)
            refs.extend(part.adoption)
        elif part.adoption:
            raise ReplayError("Batch adoption inputs require explicit retirement evidence")
    unique: dict[Path, Reference] = {}
    for ref in refs:
        if ref.path in unique and unique[ref.path] != ref:
            raise ReplayError("Batch input fingerprints conflict")
        unique[ref.path] = ref
    if len(unique) >= MAX_INPUTS:
        raise ReplayError("Batch input closure exceeds its bounded inventory")
    _work(plan.checkpoints)
    _work(plan.terminal)
    if not plan.checkpoints.is_dir() or not plan.terminal.parent.is_dir():
        raise ReplayError("Batch evidence directories must already exist")
    return tuple(unique.values())


def _outputs(backend: Backend, part: Part, step: Step) -> tuple[Path, ...]:
    outputs = backend.outputs(part, step)
    if not 1 <= len(outputs) <= MAX_OUTPUTS or len(set(outputs)) != len(outputs):
        raise ReplayError("Batch step output namespace is missing or excessive")
    return tuple(_path(str(path)) for path in outputs)


def _tasks(plan: Plan, backend: Backend, inputs: tuple[Reference, ...]) -> tuple[Task, ...]:
    tasks = tuple(
        Task(part, step, _outputs(backend, part, step), backend.effects(part, step))
        for part in plan.parts
        for step in Step
    )
    protected = (*tuple(ref.path for ref in inputs), plan.checkpoints, plan.terminal)
    claimed: list[Path] = []
    for task in tasks:
        if len(task.effects) > MAX_OUTPUTS:
            raise ReplayError("Batch side-effect namespace exceeds its budget")
        for raw in (*task.outputs, *task.effects):
            path = _path(str(raw))
            retired_readonly = task.part.retirement is not None and path in {
                task.part.native.path,
                *(ref.path for ref in task.part.adoption),
            }
            if any(
                (path.is_relative_to(other) or other.is_relative_to(path))
                and not (retired_readonly and path == other)
                for other in protected
            ):
                raise ReplayError("Batch output namespace overlaps protected inputs or checkpoints")
            if any(path.is_relative_to(other) or other.is_relative_to(path) for other in claimed):
                raise ReplayError("Batch output namespaces collide across parts or steps")
            claimed.append(path)
    return tasks


def _references(paths: tuple[Path, ...]) -> tuple[Reference, ...]:
    return tuple(Reference(path, file_digest(path)) for path in paths)


def _save_atomic(path: Path, value: dict[str, object]) -> None:
    """Keep interrupted writes unpublished and retain their local diagnostic file."""
    path = _work(path)
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=path.parent, prefix=".batch-record-", delete=False
    ) as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
        os.chmod(stream.name, 0o600)
        partial = Path(stream.name)
        os.link(partial, path)
    directory = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        os.fsync(directory)
    finally:
        os.close(directory)
    partial.unlink()


def _identity(plan: Plan, part: Part, step: Step) -> dict[str, object]:
    return {
        "version": 1,
        "kind": "pending-common-context-native-step",
        "common": [ref.json() for ref in plan.common],
        "part": part.number,
        "operation": part.operation.json(),
        "nativeSHA256": part.native.sha256,
        "step": str(step),
        "qualityApproved": False,
    }


def _step(plan: Plan, backend: Backend, task: Task, *, require_existing: bool = False) -> bool:
    part, step, outputs = task.part, task.step, task.outputs
    checkpoint = plan.checkpoints / f"part-{part.number:04d}-{step}.json"
    expected = _identity(plan, part, step)
    if checkpoint.exists() or checkpoint.is_symlink():
        saved = _json(_path(str(checkpoint)))
        raw = saved.pop("outputs", None)
        if saved != expected or not isinstance(raw, list):
            raise ReplayError("Batch checkpoint input or implementation identity changed")
        refs = _references(outputs)
        if raw != [ref.json() for ref in refs]:
            raise ReplayError("Batch checkpoint output bytes changed")
        backend.verify(part, step, refs)
        return False
    if require_existing:
        raise ReplayError("Batch checkpoint disappeared before terminal verification")
    present = tuple(path.exists() or path.is_symlink() for path in outputs)
    side_effects = any(path.exists() or path.is_symlink() for path in task.effects)
    if (any(present) or side_effects) and not all(present):
        raise ReplayError("Batch step has uncertain partial effects; reconcile before resuming")
    if part.retirement is not None and not all(present):
        raise ReplayError("Retired parts permit verified adoption only, never new effects")
    executed = not any(present)
    if executed:
        backend.execute(part, step)
    refs = _references(outputs)
    backend.verify(part, step, refs)
    _save_atomic(checkpoint, {**expected, "outputs": [ref.json() for ref in refs]})
    return executed


def _checkpoint(plan: Plan, task: Task) -> Path:
    return plan.checkpoints / f"part-{task.part.number:04d}-{task.step}.json"


def run(plan: Plan, backend: Backend) -> Result:
    """Freshly bind common bytes, adopt valid checkpoints, execute missing steps.

    An explicit retired source may use a byte-identical SSD replacement. Its backend
    must verify the immutable retirement and original operation lineage; absence is
    never an implicit replacement. Terminal files are new immutable attempt records.
    """
    started = time.monotonic()
    refs = _inventory(plan)
    if plan.terminal.exists() or plan.terminal.is_symlink():
        raise ReplayError("Batch terminal attempt already exists")
    lock = plan.checkpoints / ".batch.lock"
    _path(str(lock))
    executed = adopted = 0
    generated: list[InputHashCache] = []
    cache = InputHashCache((ref.path for ref in refs), allow_uncached_overflow=True)
    descriptor = os.open(lock, os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW, 0o600)
    with os.fdopen(descriptor, "r+b") as stream:
        try:
            fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise ReplayError("Batch checkpoints are owned by another operation") from error
        with input_hash_context(cache):
            cache.prime()
            if any(file_digest(ref.path) != ref.sha256 for ref in refs):
                raise ReplayError("Batch input bytes differ from their explicit fingerprints")
            backend.bind_inputs(refs)
            tasks = _tasks(plan, backend, refs)
            for task in tasks:
                cache.guard()
                for output_cache in generated:
                    output_cache.guard()
                print(
                    f"BATCH_SUBSTEP {task.part.number} {task.step} start "
                    f"{time.monotonic() - started:.3f}",
                    flush=True,
                )
                did_execute = _step(plan, backend, task)
                print(
                    f"BATCH_SUBSTEP {task.part.number} {task.step} complete "
                    f"{time.monotonic() - started:.3f}",
                    flush=True,
                )
                executed += int(did_execute)
                adopted += int(not did_execute)
                cache.guard()
                for output_cache in generated:
                    output_cache.guard()
                generated.append(InputHashCache((*task.outputs, _checkpoint(plan, task))))
            for task in tasks:
                for output_cache in generated:
                    output_cache.guard()
                _step(plan, backend, task, require_existing=True)
            verified = cache.verify()
            for output_cache in generated:
                output_cache.guard()
            _save_atomic(
                plan.terminal,
                {
                    "version": 1,
                    "kind": "verified-common-context-native-batch",
                    "inputs": verified,
                    "checkpoints": [
                        ref.json()
                        for ref in _references(
                            tuple(
                                plan.checkpoints / f"part-{task.part.number:04d}-{task.step}.json"
                                for task in tasks
                            )
                        )
                    ],
                    "executed": executed,
                    "adopted": adopted,
                    "cacheHashReads": cache.hash_reads,
                    "cacheBytesRead": cache.bytes_read,
                    "cacheCountersExcludeGeneratedOutputReads": True,
                    "totalBytesRead": None,
                    "elapsedBeforeTerminalSeconds": time.monotonic() - started,
                    "qualityApproved": False,
                    "deliveryApproved": False,
                },
            )
    return Result(executed, adopted, cache.hash_reads, cache.bytes_read)


def _read_ref(value: object) -> Reference:
    raw = record(value)
    digest = raw.get("sha256")
    if not isinstance(digest, str):
        raise ReplayError("Batch reference is missing its fingerprint")
    return Reference(_path(raw.get("path")), digest)


def _read_refs(value: object) -> tuple[Reference, ...]:
    if not isinstance(value, list) or not 1 <= len(value) < MAX_INPUTS:
        raise ReplayError("Batch requires a bounded reference inventory")
    return tuple(_read_ref(item) for item in value)


def _operation(part: Part) -> tuple[dict[str, object], tuple[Path, ...], Path]:
    data = _json(part.operation.path)
    semantic = _path(data.get("semantic"))
    data, inputs = canonical._operation(part.operation.path, semantic)
    return data, inputs, semantic


def _retirement_inputs(verification: Reference) -> tuple[Reference, ...]:
    config = _json(verification.path)
    retirement = _read_ref(config.get("retirement"))
    retired = _json(retirement.path)
    refs = [retirement]
    refs.extend(
        _read_ref(retired.get(name))
        for name in ("publication", "completion", "frameProof", "intent", "config", "assembly")
    )
    completed = _json(_read_ref(retired.get("completion")).path)
    refs.append(_read_ref(completed.get("frameProof")))
    return tuple(dict.fromkeys(refs))


@dataclass(frozen=True)
class Binding:
    data: dict[str, object]
    config: Path
    semantic: Path
    assembly: Path
    published_assembly: Path
    original: Path
    proof: Path
    completion: Path
    publication_config: Path
    destination: Path
    published_proof: Path
    publication: Path
    recover: bool


class CanonicalBackend:
    """Use the existing four primitives and their domain verifiers, without app calls."""

    def __init__(self) -> None:
        self._bindings: dict[int, Binding] = {}
        self._inputs: dict[Path, Reference] = {}

    def bind_inputs(self, inputs: tuple[Reference, ...]) -> None:
        self._inputs = {ref.path: ref for ref in inputs}
        codes = tuple(Path(__file__).with_name(name) for name in canonical.CODE_NAMES) + (
            Path(__file__).resolve(),
        )
        if any(path not in self._inputs for path in codes):
            raise ReplayError("Canonical batch lacks its complete current backend code closure")

    def _bind(self, part: Part) -> Binding:
        if part.number in self._bindings:
            return self._bindings[part.number]
        data, inputs, semantic = _operation(part)
        config_path = _path(data.get("config"))
        config = _json(config_path)
        assembly = _pin(config.get("assembly"))
        published_assembly = _pin(config.get("publishedAssembly"))
        staged = canonical._member(_json(assembly), part.number)
        published = canonical._member(_json(published_assembly), part.number)
        original = _path(staged.get("movie"))
        for ref in _read_refs(data.get("inputs")):
            path = (
                part.native.path
                if part.retirement is not None and ref.path == original
                else ref.path
            )
            if self._inputs.get(path) != Reference(path, ref.sha256):
                raise ReplayError("Canonical batch omitted a mandatory operation dependency")
        if config.get("part") != part.number:
            raise ReplayError("Batch operation identifies a different native part")
        if part.retirement is None:
            native = canonical._integrated_plan(part.operation.path, semantic, data, inputs)
            if native.movie != part.native.path:
                raise ReplayError("Batch native movie differs from its operation")
        else:
            self._retired(part, data, inputs, original)
        canonical.validate_cleanup(config_path, semantic)
        recover = config.get("recover", False)
        if type(recover) is not bool:
            raise ReplayError("Batch recovery mode must be explicit")
        result = Binding(
            data,
            config_path,
            semantic,
            assembly,
            published_assembly,
            original,
            _path(staged.get("proof")),
            part.operation.path.parent / "completion.json",
            part.operation.path.parent / "publication-config.json",
            _path(published.get("movie")),
            _path(published.get("proof")),
            _path(config.get("receiptOutput")),
            recover,
        )
        self._bindings[part.number] = result
        return result

    def _retired(
        self, part: Part, data: dict[str, object], inputs: tuple[Path, ...], original: Path
    ) -> None:
        verification = part.retirement
        if verification is None:
            raise ReplayError("Retired adoption is missing explicit verification")
        if part.adoption != _retirement_inputs(verification):
            raise ReplayError("Retired adoption lacks its exact immutable history closure")
        verify_retirement(verification.path)
        retired = _json(_pin(_json(verification.path).get("retirement")))
        if (
            original.exists()
            or original.is_symlink()
            or _path(record(retired.get("source")).get("path")) != original
            or _pin(retired.get("movie")) != part.native.path
            or record(retired.get("source")).get("sha256") != part.native.sha256
            or retired.get("part") != part.number
        ):
            raise ReplayError("Retired replacement differs from the original operation")
        refs = _read_refs(data.get("inputs"))
        if original not in inputs or data.get("completionConfig") != str(
            part.operation.path.parent / "completion-config.json"
        ):
            raise ReplayError("Retired operation lacks its original completion or native input")
        for ref in refs:
            path = part.native.path if ref.path == original else ref.path
            if file_digest(path) != ref.sha256:
                raise ReplayError("Retired operation input changed outside its one substitution")

    def outputs(self, part: Part, step: Step) -> tuple[Path, ...]:
        bound = self._bind(part)
        return {
            Step.FRAMES: (bound.proof,),
            Step.COMPLETE: (bound.completion,),
            Step.PUBLICATION_CONFIG: (bound.publication_config,),
            Step.PUBLISH: (bound.destination, bound.published_proof, bound.publication),
        }[step]

    def effects(self, part: Part, step: Step) -> tuple[Path, ...]:
        bound = self._bind(part)
        return (
            (
                bound.publication.with_suffix(".intent.json"),
                bound.destination.with_name(f".{bound.destination.name}.publication-partial"),
            )
            if step == Step.PUBLISH
            else ()
        )

    def execute(self, part: Part, step: Step) -> None:
        bound = self._bind(part)
        canonical.validate_cleanup(bound.config, bound.semantic)
        if part.retirement is not None:
            raise ReplayError("Canonical retired adoption must never execute media primitives")
        if step == Step.FRAMES:
            probe(bound.assembly, part=part.number, tool=canonical.frame_probe_tool(bound.config))
        elif step == Step.COMPLETE:
            complete(
                _path(bound.data.get("completionConfig")), bound.completion, recover=bound.recover
            )
        elif step == Step.PUBLICATION_CONFIG:
            canonical.publication_config(
                bound.completion,
                destination=bound.destination,
                receipt=bound.publication,
                proof=bound.published_proof,
                output=bound.publication_config,
            )
        else:
            publish(bound.publication_config)

    def verify(self, part: Part, step: Step, outputs: tuple[Reference, ...]) -> None:
        bound = self._bind(part)
        canonical.validate_cleanup(bound.config, bound.semantic)
        if tuple(ref.path for ref in outputs) != self.outputs(part, step):
            raise ReplayError("Canonical output paths differ from their operation")
        if part.retirement is not None:
            verify_retirement(part.retirement.path)
            _completed(bound.completion, replacement=part.native.path)
        elif step == Step.FRAMES:
            verify_frames(bound.assembly, part=part.number)
        elif step == Step.COMPLETE:
            _completed(bound.completion)
        if step == Step.PUBLICATION_CONFIG:
            expected = {
                "version": 1,
                "completion": _reference(bound.completion),
                "destination": str(bound.destination),
                "receiptOutput": str(bound.publication),
                "frameProofOutput": str(bound.published_proof),
            }
            if _json(bound.publication_config) != expected:
                raise ReplayError("Publication configuration differs from its completion")
        if step == Step.PUBLISH:
            partial = self.effects(part, step)[1]
            if partial.exists() or partial.is_symlink():
                raise ReplayError("Publication partial effect needs explicit reconciliation")
            if part.retirement is None:
                canonical.verify_publication(
                    bound.published_assembly, part=part.number, publication=bound.publication
                )


def load(path: Path) -> Plan:
    data = _json(_work(path))
    rows = data.get("parts")
    if (
        data.get("version") != 1
        or data.get("kind") != "bounded-native-followup-batch"
        or not (isinstance(rows, list) and 1 <= len(rows) <= MAX_PARTS)
    ):
        raise ReplayError("Invalid bounded canonical batch plan")
    parts: list[Part] = []
    for value in rows:
        row = record(value)
        number = row.get("part")
        if type(number) is not int:
            raise ReplayError("Batch part number must be an integer")
        retirement = row.get("retirement")
        parts.append(
            Part(
                number,
                _read_ref(row.get("operation")),
                _read_ref(row.get("native")),
                None if retirement is None else _read_ref(retirement),
                () if retirement is None else _read_refs(row.get("adoption")),
            )
        )
    return Plan(
        _read_refs(data.get("common")),
        tuple(parts),
        _work(_path(data.get("checkpoints"))),
        _work(_path(data.get("terminal"))),
        _read_refs(data.get("backendCode")),
    )


def prepare(config_path: Path, output: Path) -> Path:
    """Pin only ready observed operations; generate a bounded, resumable batch graph."""
    config_path, output = _work(config_path), _work(output)
    config = _json(config_path)
    rows = config.get("operations")
    max_parts, max_bytes = config.get("maxReadyParts"), config.get("maxNativeBytes")
    if (
        config.get("version") != 1
        or type(max_parts) is not int
        or not 1 <= max_parts <= MAX_PARTS
        or type(max_bytes) is not int
        or max_bytes <= 0
        or not isinstance(rows, list)
        or not 1 <= len(rows) <= max_parts
        or output.exists()
        or output.is_symlink()
        or not output.parent.is_dir()
    ):
        raise ReplayError(
            "Batch preparation needs ready operations, explicit budgets and a fresh root"
        )
    common: list[Reference] = [Reference(config_path, file_digest(config_path))]
    parts: list[Part] = []
    cleanup: set[Path] = set()
    for value in rows:
        row = record(value)
        operation = _read_ref(row.get("operation"))
        data = _json(operation.path)
        refs = _read_refs(data.get("inputs"))
        original_config = _json(_path(data.get("config")))
        number = original_config.get("part")
        if type(number) is not int:
            raise ReplayError("Prepared operation is missing its native part number")
        assembly = _json(_path(record(original_config.get("assembly")).get("path")))
        original = _path(canonical._member(assembly, number).get("movie"))
        originals = tuple(ref for ref in refs if ref.path == original)
        if len(originals) != 1:
            raise ReplayError("Operation must pin exactly its one original native movie")
        retirement_raw = row.get("retirement")
        retirement = None if retirement_raw is None else _read_ref(retirement_raw)
        adoption = () if retirement is None else _retirement_inputs(retirement)
        native = originals[0]
        if retirement is not None:
            retired = _json(_read_ref(_json(retirement.path).get("retirement")).path)
            native = _read_ref(retired.get("movie"))
            if native.sha256 != originals[0].sha256:
                raise ReplayError("Retired batch movie is not byte-identical to its original")
        parts.append(Part(number, operation, native, retirement, adoption))
        common.extend(ref for ref in refs if ref.path != original)
        cleanup.add(_path(record(original_config.get("sourceCleanup")).get("path")))
    if len(cleanup) != 1 or sum(part.native.path.stat().st_size for part in parts) > max_bytes:
        raise ReplayError("Batch must share one cleanup binding and fit its measured native budget")
    code = tuple(Path(__file__).with_name(name) for name in canonical.CODE_NAMES) + (
        Path(__file__).resolve(),
    )
    implementation = _references(code)
    common.extend(implementation)
    output.mkdir(mode=0o700)
    checkpoints = output / "checkpoints"
    checkpoints.mkdir(mode=0o700)
    plan = Plan(
        tuple(dict.fromkeys(common)),
        tuple(parts),
        checkpoints,
        output / "verified-batch.json",
        implementation,
    )
    refs = _inventory(plan)
    cache = InputHashCache((ref.path for ref in refs), allow_uncached_overflow=True)
    with input_hash_context(cache):
        cache.prime()
        if any(file_digest(ref.path) != ref.sha256 for ref in refs):
            raise ReplayError("Batch preparation input bytes changed")
        backend = CanonicalBackend()
        backend.bind_inputs(refs)
        tasks = _tasks(plan, backend, refs)
        binding_path = next(iter(cleanup))
        _, clean, _ = clean_source(_json(binding_path))
        semantic = _operation(plan.parts[0])[2]
        if any(_operation(part)[2] != semantic for part in plan.parts):
            raise ReplayError("Batch operations must share their exact semantic XML")
        cache.verify()
    plan_path = output / "plan.json"
    _save_atomic(
        plan_path,
        {
            "version": 1,
            "kind": "bounded-native-followup-batch",
            "common": [ref.json() for ref in plan.common],
            "backendCode": [ref.json() for ref in plan.backend_code],
            "parts": [
                {
                    "part": part.number,
                    "operation": part.operation.json(),
                    "native": part.native.json(),
                    "retirement": None if part.retirement is None else part.retirement.json(),
                    "adoption": [ref.json() for ref in part.adoption],
                }
                for part in plan.parts
            ],
            "checkpoints": str(checkpoints),
            "terminal": str(plan.terminal),
            "generatedOutputs": [str(path) for task in tasks for path in task.outputs],
        },
    )
    direct = (
        plan_path,
        *code,
        *(part.operation.path for part in plan.parts),
        binding_path,
        clean,
        semantic,
    )
    graph = output / "workflow.json"
    _save_atomic(
        graph,
        {
            "version": 1,
            "stages": [
                {
                    "name": "followup-batch",
                    "command": [
                        sys.executable,
                        "-B",
                        "-m",
                        "replay.fcp_part_followup_batch",
                        "run",
                        str(plan_path),
                        "--semantic",
                        str(semantic),
                    ],
                    "inputs": list(map(str, direct)),
                    # Individual files are immutable child checkpoints owned by
                    # this manifest. The terminal owns their exact references;
                    # the terminal-only output leaves child checkpoints intact
                    # during explicit stage_recovery of a stopped workflow.
                    "outputs": [str(plan.terminal)],
                    "timeout_seconds": len(parts) * 12800,
                    "verify_command": [
                        sys.executable,
                        "-B",
                        "-m",
                        "replay.fcp_part_followup_batch",
                        "verify",
                        str(plan_path),
                        "--semantic",
                        str(semantic),
                    ],
                }
            ],
            "sourceCaptionProduction": {"bindings": {"followup-batch": str(next(iter(cleanup)))}},
        },
    )
    return graph


def verify(plan: Plan, backend: Backend) -> None:
    """Freshly revalidate every checkpoint and canonical domain after completion."""
    refs = _inventory(plan)
    cache = InputHashCache((ref.path for ref in refs), allow_uncached_overflow=True)
    with input_hash_context(cache):
        cache.prime()
        if any(file_digest(ref.path) != ref.sha256 for ref in refs):
            raise ReplayError("Batch verification input bytes changed")
        backend.bind_inputs(refs)
        tasks = _tasks(plan, backend, refs)
        generated = InputHashCache(
            tuple(path for task in tasks for path in (*task.outputs, _checkpoint(plan, task)))
        )
        for task in tasks:
            generated.guard()
            _step(plan, backend, task, require_existing=True)
            generated.guard()
        terminal = _json(plan.terminal)
        checkpoints = _references(tuple(_checkpoint(plan, task) for task in tasks))
        if (
            terminal.get("version") != 1
            or terminal.get("kind") != "verified-common-context-native-batch"
            or terminal.get("checkpoints") != [ref.json() for ref in checkpoints]
            or terminal.get("inputs") != cache.verify()
            or terminal.get("qualityApproved") is not False
            or terminal.get("deliveryApproved") is not False
        ):
            raise ReplayError("Batch terminal differs from its current inputs or checkpoints")
        generated.guard()


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("operation", choices=("prepare", "run", "verify"))
    parser.add_argument("plan", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--semantic", type=Path)
    arguments = parser.parse_args(argv)
    if arguments.operation == "prepare":
        if arguments.output is None:
            raise ReplayError("Batch prepare needs an explicit fresh output root")
        prepare(arguments.plan, arguments.output)
        return
    if arguments.output is not None:
        raise ReplayError("Only preparation accepts a new output root")
    plan = load(arguments.plan)
    if arguments.semantic is not None and any(
        _operation(part)[2] != _path(str(arguments.semantic)) for part in plan.parts
    ):
        raise ReplayError("Batch CLI semantic differs from its pinned operations")
    backend = CanonicalBackend()
    if arguments.operation == "run":
        result = run(plan, backend)
        print(json.dumps({"executed": result.executed, "adopted": result.adopted}))
    else:
        verify(plan, backend)


if __name__ == "__main__":
    main(sys.argv[1:])
