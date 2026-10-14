"""Checkpoint existing whole-media checks within one bounded parent hash context.

Preparation is metadata-only. Admission retains the original complete ancestor
receipts, transitive inventory and readonly domains. Verification never invokes
a producer or recovery; stopped-stage recovery remains a separate operation.
"""

import argparse
import ast
import hashlib
import json
import os
import re
import stat
from collections.abc import Iterator, Sequence
from contextlib import ExitStack, contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import NoReturn

from replay.core import ReplayError
from replay.input_hash_cache import MAX_INPUTS, InputHashCache, input_hash_context
from replay.recordings import record
from replay.source_caption_binding import validate_binding
from replay.stages import (
    Stage,
    _fingerprint,
    _read_receipt,
    execute_command,
    execution_lock,
    file_digest,
)
from replay.workflow import load_stages, run_plan, stage_graph

ROOT = Path(__file__).resolve().parents[2]
MAX_METADATA_BYTES = 8 * 1024 * 1024
MAX_DECODER_THREADS = 256
ROLE_NAMES = frozenset({"wholeJoin", "fullAV", "clock", "cleanup", "retirement"})
CODE_NAMES = (
    "__init__.py",
    "whole_technical_batch.py",
    "workflow.py",
    "stages.py",
    "input_hash_cache.py",
    "core.py",
    "recordings.py",
    "source_caption_binding.py",
    "stage_recovery.py",
)


@dataclass(frozen=True)
class Reference:
    path: Path
    sha256: str

    def json(self) -> dict[str, str]:
        return {"path": str(self.path), "sha256": self.sha256}


@dataclass(frozen=True)
class FullAVReuse:
    stage: Stage
    decoder: Path


@dataclass(frozen=True)
class Plan:
    original: Reference
    source_state: Path
    stages: tuple[Stage, ...]
    ancestors: tuple[Stage, ...]
    inputs: tuple[Reference, ...]
    state: Path
    cwd: Path
    tools: tuple[tuple[Path, Path], ...]
    cleanup_bindings: tuple[tuple[Stage, Path], ...]
    full_av_reuse: tuple[FullAVReuse, ...] = ()


@dataclass(frozen=True)
class Prepared:
    manifest: Reference
    workflow: Reference
    config: Reference
    plan: Plan


@dataclass(frozen=True)
class Counters:
    hash_reads: int
    bytes_read: int
    cache_hits: int

    def json(self) -> dict[str, object]:
        return {
            "hashReads": self.hash_reads,
            "bytesRead": self.bytes_read,
            "cacheHits": self.cache_hits,
            "scope": "parent-cache-only",
            "childIO": {"measured": False, "bytesRead": None},
            "outerIO": {"measured": False, "bytesRead": None},
        }


def _path(value: object, *, work: bool = False) -> Path:
    if not isinstance(value, str) or not value or any(c in value for c in "\0\r\n"):
        raise ReplayError("Expected a canonical technical-batch path")
    path = Path(value)
    if (
        not path.is_absolute()
        or str(path) != value
        or ".." in path.parts
        or any(part in {"private", "downloads"} for part in path.parts)
        or (work and not path.is_relative_to(ROOT / "edits/work"))
    ):
        raise ReplayError("Technical-batch path is outside its declared scope")
    if path.resolve() != path:
        raise ReplayError("Technical-batch paths must not traverse symlinks")
    return path


def _sha(value: object) -> str:
    if not isinstance(value, str) or re.fullmatch(r"[a-f0-9]{64}", value) is None:
        raise ReplayError("Technical batch requires an explicit SHA256")
    return value


def _reference(value: object) -> Reference:
    row = record(value)
    if set(row) != {"path", "sha256"}:
        raise ReplayError("Technical-batch reference fields differ")
    return Reference(_path(row["path"]), _sha(row["sha256"]))


def _unique(pairs: list[tuple[str, object]]) -> dict[str, object]:
    if len({name for name, _ in pairs}) != len(pairs):
        raise ReplayError("Technical-batch JSON keys must be unique")
    return dict(pairs)


def _bounded_bytes(path: Path) -> bytes:
    _path(str(path))
    with ExitStack() as stack:
        descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
        stack.callback(os.close, descriptor)
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode) or not 0 < before.st_size <= MAX_METADATA_BYTES:
            raise ReplayError("Technical-batch metadata must be bounded regular files")
        stream = stack.enter_context(os.fdopen(descriptor, "rb", closefd=False))
        data = stream.read(MAX_METADATA_BYTES + 1)
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
        raise ReplayError("Technical-batch metadata changed during reading")
    return data


def _nonfinite(value: str) -> NoReturn:
    raise ReplayError(f"Nonfinite technical-batch JSON number: {value}")


def _metadata(path: Path, digest: str | None = None) -> dict[str, object]:
    data = _bounded_bytes(path)
    if digest is not None and hashlib.sha256(data).hexdigest() != _sha(digest):
        raise ReplayError("Technical-batch metadata fingerprint changed")
    return record(json.loads(data, object_pairs_hook=_unique, parse_constant=_nonfinite))


def _write(path: Path, document: dict[str, object]) -> Reference:
    _path(str(path), work=True)
    if not path.parent.is_dir():
        raise ReplayError("Technical-batch output needs an existing parent")
    data = (json.dumps(document, sort_keys=True, ensure_ascii=False) + "\n").encode("utf-8")
    with path.open("xb") as stream:
        os.fchmod(stream.fileno(), 0o600)
        stream.write(data)
    return Reference(path, hashlib.sha256(data).hexdigest())


def _texts(value: object) -> tuple[str, ...]:
    if not isinstance(value, list) or not 1 <= len(value) <= 256:
        raise ReplayError("Technical batch requires a bounded name list")
    if any(not isinstance(item, str) or not item for item in value):
        raise ReplayError("Technical-batch names must be nonempty text")
    return tuple(item for item in value if isinstance(item, str))


def _inputs(value: object) -> tuple[Reference, ...]:
    if not isinstance(value, list) or not 1 <= len(value) <= MAX_INPUTS:
        raise ReplayError("Technical-batch input inventory exceeds its bound")
    refs = tuple(_reference(item) for item in value)
    if len({ref.path for ref in refs}) != len(refs):
        raise ReplayError("Technical-batch inputs must be unique")
    return refs


def _stage_json(stage: Stage) -> dict[str, object]:
    return {
        "name": stage.name,
        "command": list(stage.command),
        "inputs": list(map(str, stage.inputs)),
        "outputs": list(map(str, stage.outputs)),
        "timeout_seconds": stage.timeout_seconds,
        "verify_command": list(stage.verify_command),
        "adopt_existing": stage.adopt_existing,
    }


def _ancestors(graph: dict[str, tuple[str, ...]], targets: tuple[str, ...]) -> set[str]:
    pending = list(targets)
    names: set[str] = set()
    while pending:
        name = pending.pop()
        if name not in graph:
            raise ReplayError("Unknown technical target")
        if name not in names:
            names.add(name)
            pending.extend(graph[name])
    return names.difference(targets)


def _tool(command: tuple[str, ...]) -> tuple[Path, Path]:
    alias = Path(command[0])
    if not alias.is_absolute() or any(part in {"private", "downloads"} for part in alias.parts):
        raise ReplayError("Technical-batch executables must be explicitly absolute")
    resolved = _path(str(alias.resolve(strict=True)))
    if not resolved.is_file():
        raise ReplayError("Technical-batch executable is not a regular file")
    return alias, resolved


def _source_stages(reference: Reference) -> tuple[Stage, ...]:
    document = _metadata(reference.path, reference.sha256)
    rows = document.get("stages")
    if not isinstance(rows, list):
        raise ReplayError("Technical batch needs its original trusted workflow")
    for row in rows:
        for key in ("inputs", "outputs"):
            values = record(row).get(key)
            if not isinstance(values, list):
                raise ReplayError("Original workflow paths are malformed")
            for value in values:
                _path(value)
    return load_stages(reference.path)


def reader_code_paths() -> tuple[Path, ...]:
    """Require the bounded current reader import closure, including lazy domains."""
    pending = {name.removesuffix(".py") for name in CODE_NAMES}
    found: set[str] = set()
    while pending:
        name = pending.pop()
        if name in found:
            continue
        if re.fullmatch(r"(?:[a-z][a-z0-9_]*|__init__)", name) is None or len(found) >= 512:
            raise ReplayError("Technical reader code closure exceeds its module scope")
        found.add(name)
        path = _path(str(ROOT / "src/replay" / f"{name}.py"))
        if not path.is_file() or not 0 < path.stat().st_size <= MAX_METADATA_BYTES:
            raise ReplayError("Technical reader source must be bounded regular code")
        for node in ast.walk(ast.parse(_bounded_bytes(path).decode("utf-8"))):
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
    return tuple(ROOT / "src/replay" / f"{name}.py" for name in sorted(found))


def _full_av_reuse(
    admission: dict[str, object], roles: dict[str, object], ancestors: tuple[Stage, ...]
) -> tuple[FullAVReuse, ...]:
    if "fullAVReuse" not in admission:
        return ()
    declaration = record(admission["fullAVReuse"])
    if (
        set(declaration) != {"version", "kind", "stages"}
        or type(declaration["version"]) is not int
        or declaration["version"] != 1
        or declaration["kind"] != "completed-canonical-ffmpeg-full-av"
    ):
        raise ReplayError("Unsupported typed full AV evidence reuse")
    names = _texts(declaration["stages"])
    eligible = set(_texts(roles["wholeJoin"])) & set(_texts(roles["fullAV"]))
    if len(set(names)) != len(names) or not set(names).issubset(eligible):
        raise ReplayError("Full AV reuse must identify distinct whole-join/full-AV ancestors")
    by_name = {stage.name: stage for stage in ancestors}
    return tuple(_full_av_contract(by_name[name]) for name in names)


def _full_av_contract(stage: Stage) -> FullAVReuse:
    command = stage.verify_command
    if len(stage.outputs) != 1 or not command:
        raise ReplayError("Full AV reuse requires one exact media output and verifier")
    _, decoder = _tool(command)
    arguments = command[1:]
    prefix = ("-nostdin", "-v", "error", "-xerror")
    if arguments[:4] != prefix:
        raise ReplayError("Full AV reuse requires the canonical local MOV verifier")
    arguments = arguments[4:]
    if arguments[:1] == ("-threads",):
        if len(arguments) < 2 or re.fullmatch(r"0|[1-9][0-9]{0,2}", arguments[1]) is None:
            raise ReplayError("Full AV decoder thread count is malformed")
        if int(arguments[1]) > MAX_DECODER_THREADS:
            raise ReplayError("Full AV decoder thread count exceeds its bound")
        arguments = arguments[2:]
    expected = (
        "-protocol_whitelist",
        "file",
        "-f",
        "mov",
        "-enable_drefs",
        "0",
        "-use_absolute_path",
        "0",
        "-i",
        str(stage.outputs[0]),
        "-map",
        "0:v:0",
        "-map",
        "0:a:0",
        "-f",
        "null",
        "-",
    )
    if arguments != expected:
        raise ReplayError("Full AV reuse requires the canonical local MOV verifier")
    required = {decoder, *reader_code_paths()}
    if not required.issubset(stage.inputs):
        raise ReplayError("Original full AV checkpoint must directly pin decoder and reader code")
    return FullAVReuse(stage, decoder)


def _discover(document: dict[str, object]) -> Plan:
    if set(document) != {
        "version",
        "workflow",
        "sourceState",
        "targets",
        "admission",
        "state",
        "cwd",
    }:
        raise ReplayError("Technical-batch configuration fields differ")
    if type(document["version"]) is not int or document["version"] != 1:
        raise ReplayError("Unsupported technical-batch configuration version")
    original = _reference(document["workflow"])
    all_stages = _source_stages(original)
    graph = stage_graph(all_stages)
    targets = _texts(document["targets"])
    if len(targets) != 3 or len(set(targets)) != 3:
        raise ReplayError("Technical batch requires exactly three distinct targets")
    ancestor_names = _ancestors(graph, targets)
    selected = tuple(stage for stage in all_stages if stage.name in targets)
    ancestors = tuple(stage for stage in all_stages if stage.name in ancestor_names)
    bindings = record(_metadata(original.path, original.sha256).get("sourceCaptionProduction", {}))
    binding_paths = record(bindings.get("bindings", {}))
    if set(targets).intersection(binding_paths):
        raise ReplayError("Technical targets must not discard a source-cleanup binding")
    admission = record(document["admission"])
    if set(admission) not in ({"roles", "inputs"}, {"roles", "inputs", "fullAVReuse"}):
        raise ReplayError("Technical-batch admission fields differ")
    roles = record(admission["roles"])
    if set(roles) != ROLE_NAMES:
        raise ReplayError(
            "All whole-join, full-AV, clock, cleanup and retirement roles are required"
        )
    ancestor_domains = {stage.name for stage in ancestors if stage.verify_command}
    if any(not set(_texts(value)).issubset(ancestor_domains) for value in roles.values()):
        raise ReplayError("Admission roles must bind original ancestor domain verifiers")
    full_av_reuse = _full_av_reuse(admission, roles, ancestors)
    if any(not stage.verify_command for stage in selected):
        raise ReplayError("Each technical target needs its original readonly verifier")
    state = _path(document["state"], work=True)
    source_state = _path(document["sourceState"], work=True)
    cwd = _path(document["cwd"])
    if cwd != ROOT or not cwd.is_dir() or not source_state.is_dir():
        raise ReplayError("Technical batch requires its original canonical working directory/state")
    refs = _inputs(admission["inputs"])
    tools = tuple(
        dict.fromkeys(
            _tool(command)
            for stage in (*ancestors, *selected)
            for command in (stage.command, stage.verify_command)
            if command
        )
    )
    selected_outputs = {path for stage in selected for path in stage.outputs}
    needed = {original.path, *reader_code_paths()}
    needed.update(
        path
        for stage in (*ancestors, *selected)
        for path in stage.inputs
        if path not in selected_outputs
    )
    needed.update(path for stage in ancestors for path in stage.outputs)
    needed.update(source_state / f"{stage.name}.json" for stage in ancestors)
    needed.update(resolved for _, resolved in tools)
    needed.update(
        source_state / f"{stage.name}.json"
        for stage in selected
        if (source_state / f"{stage.name}.json").exists()
    )
    supplied = {ref.path for ref in refs}
    if not needed.issubset(supplied) or supplied.intersection(selected_outputs):
        raise ReplayError("Admission must pin the complete closure and exclude generated outputs")
    if any(path.is_relative_to(state) or state.is_relative_to(path) for path in supplied):
        raise ReplayError("Technical state must be separate from immutable admission inputs")
    if any(path.is_relative_to(state) or state.is_relative_to(path) for path in selected_outputs):
        raise ReplayError("Technical outputs must be separate from their workflow state")
    stage_graph(selected)
    cleanup_bindings = tuple(
        (stage, _path(binding_paths[stage.name]))
        for stage in ancestors
        if stage.name in binding_paths
    )
    return Plan(
        original,
        source_state,
        selected,
        ancestors,
        refs,
        state,
        cwd,
        tools,
        cleanup_bindings,
        full_av_reuse,
    )


def prepare(config: Path, *, manifest: Path, workflow: Path) -> None:
    """Pin metadata and exact original stages without hashing large media."""
    config = _path(str(config), work=True)
    document = _metadata(config)
    plan = _discover(document)
    workflow = _path(str(workflow), work=True)
    manifest = _path(str(manifest), work=True)
    protected = {config, plan.original.path, *(ref.path for ref in plan.inputs)}
    outputs = {path for stage in plan.stages for path in stage.outputs}
    if workflow == manifest or any(
        path in protected | outputs or path.exists() for path in (workflow, manifest)
    ):
        raise ReplayError("Technical preparation outputs must be distinct and absent")
    _separate(plan, (workflow, manifest), protected | outputs)
    technical = _write(
        workflow, {"version": 1, "stages": [_stage_json(stage) for stage in plan.stages]}
    )
    _write(
        manifest,
        {
            "version": 1,
            "config": Reference(config, file_digest(config)).json(),
            "technicalWorkflow": technical.json(),
        },
    )


def load(manifest: Path, digest: str) -> Prepared:
    """Validate exact data graph and complete admission inventory, without running."""
    manifest = _path(str(manifest), work=True)
    data = _metadata(manifest, digest)
    if (
        set(data) != {"version", "config", "technicalWorkflow"}
        or type(data["version"]) is not int
        or data["version"] != 1
    ):
        raise ReplayError("Invalid prepared technical batch")
    config = _reference(data["config"])
    plan = _discover(_metadata(config.path, config.sha256))
    technical = _reference(data["technicalWorkflow"])
    if _source_stages(technical) != plan.stages:
        raise ReplayError("Technical graph changed the original commands or stage definitions")
    paths = {ref.path for ref in plan.inputs}
    if {manifest, config.path, technical.path}.intersection(paths):
        raise ReplayError("Prepared control artifacts must be outside the admission inventory")
    _separate(plan, (manifest, config.path, technical.path), paths)
    return Prepared(Reference(manifest, _sha(digest)), technical, config, plan)


def _cache(prepared: Prepared) -> InputHashCache:
    refs = (*prepared.plan.inputs, prepared.manifest, prepared.workflow, prepared.config)
    if len(refs) > MAX_INPUTS:
        raise ReplayError("Prepared technical closure exceeds its reference budget")
    cache = InputHashCache((ref.path for ref in refs), allow_uncached_overflow=True)
    cache.prime()
    with input_hash_context(cache):
        if any(file_digest(ref.path) != ref.sha256 for ref in refs):
            raise ReplayError("Technical-batch input fingerprint changed")
    return cache


def _tools(plan: Plan) -> None:
    if Path.cwd().resolve() != plan.cwd:
        raise ReplayError("Technical-batch working directory changed")
    if any(alias.resolve(strict=True) != resolved for alias, resolved in plan.tools):
        raise ReplayError("Technical-batch executable binding changed")


def _checkpoint(stage: Stage, state: Path) -> Reference:
    path = _path(str(state / f"{stage.name}.json"), work=True)
    saved = record(_read_receipt(path))
    expected: dict[str, object] = {
        "status": "complete",
        "fingerprint": _fingerprint(stage),
        "outputs": {str(path): file_digest(path) for path in stage.outputs},
    }
    if saved != expected:
        raise ReplayError("Technical-batch checkpoint or completed outputs changed")
    return Reference(path, file_digest(path))


def _domains(stages: tuple[Stage, ...], logs: Path, cache: InputHashCache) -> None:
    for stage in stages:
        cache.guard()
        if stage.verify_command:
            execute_command(stage.verify_command, logs / f"{stage.name}.log", stage.timeout_seconds)
        cache.guard()


@contextmanager
def _ownership(plan: Plan, *, generated: bool) -> Iterator[None]:
    with ExitStack() as stack:
        owners = (
            (plan.state.parent / f".{plan.state.name}-batch", "owner"),
            (plan.source_state, ".workflow"),
        )
        for directory, name in owners:
            if not stack.enter_context(execution_lock(directory, name)):
                raise ReplayError("Another process owns technical-batch admission/state")
        for stage in (*plan.ancestors, *plan.stages):
            if not stack.enter_context(execution_lock(plan.source_state, stage.name)):
                raise ReplayError("Another process owns an original technical stage")
        if not generated:
            for name in (".workflow", *(stage.name for stage in plan.stages)):
                with execution_lock(plan.state, name) as acquired:
                    if not acquired:
                        raise ReplayError("Another process owns technical workflow state")
        if generated:
            for name in (".workflow", *(stage.name for stage in plan.stages)):
                if not stack.enter_context(execution_lock(plan.state, name)):
                    raise ReplayError("Another process owns completed technical outputs")
        yield


def _admit(plan: Plan, logs: Path, cache: InputHashCache) -> None:
    _tools(plan)
    for stage, path in plan.cleanup_bindings:
        cache.guard()
        validate_binding(path, consumer_inputs=stage.inputs, consumer_command=stage.command)
        cache.guard()
    for stage in plan.ancestors:
        _checkpoint(stage, plan.source_state)
    reused = {reuse.stage.name for reuse in plan.full_av_reuse}
    _domains(tuple(stage for stage in plan.ancestors if stage.name not in reused), logs, cache)


def _seed(plan: Plan) -> None:
    with execution_lock(plan.state, ".workflow") as acquired:
        if not acquired:
            raise ReplayError("Another process owns technical workflow state")
        for stage in plan.stages:
            source = plan.source_state / f"{stage.name}.json"
            destination = plan.state / f"{stage.name}.json"
            if source.exists() and not destination.exists():
                _checkpoint(stage, plan.source_state)
                with destination.open("xb") as stream:
                    os.fchmod(stream.fileno(), 0o600)
                    stream.write(source.read_bytes())


def _logs(path: Path) -> Path:
    path = _path(str(path), work=True)
    path.mkdir(mode=0o700)
    return path


def _counters(cache: InputHashCache) -> Counters:
    return Counters(cache.hash_reads, cache.bytes_read, cache.cache_hits)


def _metrics(value: object) -> None:
    row = record(value)
    if set(row) != {"hashReads", "bytesRead", "cacheHits", "scope", "childIO", "outerIO"}:
        raise ReplayError("Technical-batch cache counters have invalid fields")
    for key in ("hashReads", "bytesRead", "cacheHits"):
        value = row[key]
        if type(value) is not int or value < 0:
            raise ReplayError("Technical-batch cache counters must be nonnegative integers")
    unknown = {"measured": False, "bytesRead": None}
    if (
        row["scope"] != "parent-cache-only"
        or row["childIO"] != unknown
        or row["outerIO"] != unknown
    ):
        raise ReplayError("Technical-batch counters must not claim child or outer IO")


def _terminal_path(prepared: Prepared, receipt: Path) -> Path:
    receipt = _path(str(receipt), work=True)
    protected = {ref.path for ref in prepared.plan.inputs} | {
        prepared.manifest.path,
        prepared.workflow.path,
        prepared.config.path,
        *(path for stage in prepared.plan.stages for path in stage.outputs),
    }
    _separate(prepared.plan, (receipt,), protected)
    return receipt


def _separate(plan: Plan, controls: tuple[Path, ...], protected: set[Path]) -> None:
    states = (plan.state, plan.source_state)
    for control in controls:
        if any(
            control.is_relative_to(path) or path.is_relative_to(control)
            for path in (*states, *protected)
        ):
            raise ReplayError("Technical controls must be separate from inputs, outputs and states")


def _log_directory(prepared: Prepared, logs: Path, receipt: Path) -> Path:
    logs = _path(str(logs), work=True)
    protected = {
        prepared.manifest.path,
        prepared.workflow.path,
        prepared.config.path,
        receipt,
        *(ref.path for ref in prepared.plan.inputs),
        *(path for stage in prepared.plan.stages for path in stage.outputs),
    }
    _separate(prepared.plan, (logs,), protected)
    return logs


def _result(prepared: Prepared, cache: InputHashCache) -> dict[str, object]:
    checkpoints = [_checkpoint(stage, prepared.plan.state).json() for stage in prepared.plan.stages]
    return {
        "version": 1,
        "kind": "verified-whole-technical-batch",
        "manifest": prepared.manifest.json(),
        "workflow": prepared.workflow.json(),
        "inputs": cache.verify(),
        "checkpoints": checkpoints,
        "reusedFullAV": [
            {
                "kind": "completed-canonical-ffmpeg-full-av",
                "stage": reuse.stage.name,
                "checkpoint": _checkpoint(reuse.stage, prepared.plan.source_state).json(),
                "media": Reference(
                    reuse.stage.outputs[0], file_digest(reuse.stage.outputs[0])
                ).json(),
                "decoder": Reference(reuse.decoder, file_digest(reuse.decoder)).json(),
                "verifyCommand": list(reuse.stage.verify_command),
                "newDecodePerformed": False,
            }
            for reuse in prepared.plan.full_av_reuse
        ],
        "domains": [stage.name for stage in prepared.plan.stages],
        "candidateOnly": True,
        "qualityApproved": False,
        "syncApproved": False,
        "deliveryApproved": False,
        "overallThroughputApproved": False,
    }


def run(manifest: Path, digest: str, *, receipt: Path, logs: Path) -> Counters:
    """Run original checks with original checkpoints in one parent process."""
    prepared = load(manifest, digest)
    receipt = _terminal_path(prepared, receipt)
    logs = _log_directory(prepared, logs, receipt)
    if receipt.exists() or not receipt.parent.is_dir():
        raise ReplayError("Technical-batch receipt must be absent with an existing parent")
    _tools(prepared.plan)
    with _ownership(prepared.plan, generated=False):
        cache = _cache(prepared)
        log_directory = _logs(logs)
        with input_hash_context(cache):
            _admit(prepared.plan, log_directory, cache)
            _seed(prepared.plan)
            for stage in prepared.plan.stages:
                cache.guard()
                _tools(prepared.plan)
                run_plan(
                    prepared.workflow.path,
                    state_directory=prepared.plan.state,
                    targets=(stage.name,),
                )
                cache.guard()
            result = _result(prepared, cache)
            counters = _counters(cache)
            result["runCache"] = counters.json()
            _write(receipt, result)
            return counters


def verify(manifest: Path, digest: str, *, receipt: Path, logs: Path) -> Counters:
    """Fresh readonly receipt/domain verification; never run or recover producers."""
    prepared = load(manifest, digest)
    receipt = _terminal_path(prepared, receipt)
    logs = _log_directory(prepared, logs, receipt)
    original = _metadata(receipt)
    original_digest = file_digest(receipt)
    _tools(prepared.plan)
    with _ownership(prepared.plan, generated=True):
        cache = _cache(prepared)
        log_directory = _logs(logs)
        with input_hash_context(cache):
            generated = InputHashCache(
                tuple(
                    path
                    for stage in prepared.plan.stages
                    for path in (*stage.outputs, prepared.plan.state / f"{stage.name}.json")
                ),
                allow_uncached_overflow=True,
            )
            _admit(prepared.plan, log_directory, cache)
            for stage in prepared.plan.stages:
                _checkpoint(stage, prepared.plan.state)
            generated.guard()
            _domains(prepared.plan.stages, log_directory, cache)
            generated.guard()
            expected = _result(prepared, cache)
            metrics = original.pop("runCache", None)
            _metrics(metrics)
            if original != expected:
                raise ReplayError(
                    "Technical-batch receipt differs from fresh readonly verification"
                )
            if file_digest(receipt) != original_digest:
                raise ReplayError("Technical-batch terminal receipt changed during verification")
            generated.guard()
            return _counters(cache)


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="operation", required=True)
    preparation = commands.add_parser("prepare")
    preparation.add_argument("config", type=Path)
    preparation.add_argument("--manifest", type=Path, required=True)
    preparation.add_argument("--workflow", type=Path, required=True)
    for name in ("run", "verify"):
        command = commands.add_parser(name)
        command.add_argument("manifest", type=Path)
        command.add_argument("--sha256", required=True)
        command.add_argument("--receipt", type=Path, required=True)
        command.add_argument("--logs", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.operation == "prepare":
        prepare(args.config, manifest=args.manifest, workflow=args.workflow)
        return
    operations = {"run": run, "verify": verify}
    result = operations[args.operation](
        args.manifest, args.sha256, receipt=args.receipt, logs=args.logs
    )
    print(json.dumps(result.json(), sort_keys=True))


if __name__ == "__main__":
    main()
