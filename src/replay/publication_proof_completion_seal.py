"""Authenticate publication proof inputs without substituting historical source pins.

Source review is deliberately conservative: imported modules and module-level
execution must remain identical. Only definitions unreachable from authenticated
consumer references may differ. This is evidence for a future completion seal,
not permission to skip material, lock or completion-domain verification.
"""

import argparse
import ast
import hashlib
import json
import re
import stat
import sys
from collections.abc import Mapping
from contextlib import ExitStack
from dataclasses import dataclass
from pathlib import Path

from replay import cleanup_finalization_workflow as proof_common
from replay import cleanup_lane_successor as proof_successor
from replay import code_dependency_inventory as proof_inventory
from replay import code_inventory_memo_run as proof_memo
from replay import leaf_retirement as proof_leaves
from replay import native_retired_hardlink_verification as history_aliases
from replay import recovery_source_finalization as proof_original
from replay import subtitle_cleanup_assembly as proof_assembly
from replay import verified_hardlink_publication as proof_retirement
from replay import workflow
from replay.artifact_json import _path
from replay.core import ReplayError
from replay.fcp_part_assembly import _reference, _save
from replay.foreground_evidence import _bound_bytes
from replay.input_hash_cache import InputHashCache, input_hash_context
from replay.recordings import record
from replay.sequence_journal import bounded_read
from replay.stages import Stage, file_digest

ROOT = Path(__file__).resolve().parents[2]
MAX_SOURCE_BYTES = 2 * 1024 * 1024
MAX_SOURCES = 512
SHA256 = re.compile(r"[a-f0-9]{64}")
DYNAMIC_NAMES = frozenset(
    {
        "eval",
        "exec",
        "getattr",
        "setattr",
        "delattr",
        "globals",
        "locals",
        "vars",
        "dir",
        "__import__",
    }
)
REFLECTIVE_ATTRIBUTES = frozenset(
    {"__globals__", "__dict__", "__code__", "__closure__", "__getattribute__"}
)
DEFINITIONS = (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)


@dataclass(frozen=True)
class PublicationHistory:
    job: proof_common.Job
    retirement_request: dict[str, object]
    historical_inputs: dict[str, str]
    source_pins: dict[Path, str]
    adoption_stage: Stage
    child_stage: Stage
    child_outputs: dict[str, str]
    evidence_pins: tuple[history_aliases.Reference, ...]
    alias: Path
    keeper: Path


def _history_merge(pins: dict[str, str], path: Path, digest: object) -> None:
    expected = proof_assembly._sha(digest)
    if str(path) in pins and pins[str(path)] != expected:
        raise ReplayError("Conflicting authenticated historical pins")
    pins[str(path)] = expected


def _history_reference(value: object) -> tuple[Path, str]:
    raw = record(value)
    if set(raw) != {"path", "sha256"}:
        raise ReplayError("Expected exact historical reference")
    return proof_successor._path(raw["path"], work=False), proof_assembly._sha(raw["sha256"])


def _history_read(path: Path, pins: Mapping[str, str]) -> dict[str, object]:
    digest = pins.get(str(path))
    if digest is None:
        raise ReplayError("Historical metadata lacks an authenticated binding")
    try:
        return record(json.loads(_bound_bytes(path, digest, history_aliases.MAX_BYTES)))
    except ValueError as error:
        raise ReplayError("Authenticated historical metadata is invalid") from error


def _history_complete(
    stage: Stage,
    *,
    checkpoint: dict[str, object],
    pins: Mapping[str, str],
    outputs: Mapping[str, str],
) -> None:
    if not stage.verify_command or set(outputs) != {str(path) for path in stage.outputs}:
        raise ReplayError("Historical post-command output definition differs")
    selected = {str(path): pins[str(path)] for path in stage.inputs if str(path) in pins}
    if (
        set(checkpoint) != {"status", "fingerprint", "outputs"}
        or checkpoint["status"] != "complete"
        or checkpoint["fingerprint"] != historical_stage_fingerprint(stage, selected)
        or checkpoint["outputs"] != dict(outputs)
    ):
        raise ReplayError("Historical complete checkpoint differs from authenticated inputs")


def load_publication_history(
    *,
    operation: Path,
    custody_seal: Path,
    publication_workflow: Path,
    adoption_checkpoint: Path,
    child_workflow: Path,
    child_checkpoint: Path,
) -> PublicationHistory:
    """Authenticate past verification without executing or accepting past reader code.

    The caller separately verifies current dependency bytes and current code.
    Media output hashes here originate from the genuine custody seal, not a
    stat cache or an invented completion of the original running child.
    """
    seal_raw = proof_common._read(custody_seal)
    alias = proof_successor._path(seal_raw.get("alias"))
    owner = seal_raw.get("owner")
    if not isinstance(owner, str):
        raise ReplayError("Historical custody owner is missing")
    verified = history_aliases.verify(custody_seal, alias=alias, owner=owner)
    pins: dict[str, str] = {}
    for reference in verified.pins:
        _history_merge(pins, reference.path, reference.sha256)
    authenticated_seal = _history_read(custody_seal, pins)
    completed = record(authenticated_seal.get("completedStage"))
    for name, digest in record(completed.get("inputs")).items():
        _history_merge(pins, proof_successor._path(name, work=False), digest)
    for name, digest in record(completed.get("outputs")).items():
        _history_merge(pins, proof_successor._path(name, work=False), digest)
    seal_request_path, seal_request_sha = _history_reference(authenticated_seal["request"])
    _history_merge(pins, seal_request_path, seal_request_sha)
    seal_request = _history_read(seal_request_path, pins)
    retirement_path, retirement_sha = _history_reference(seal_request["retirementRequest"])
    _history_merge(pins, retirement_path, retirement_sha)
    retirement = _history_read(retirement_path, pins)
    operation_path, operation_sha = _history_reference(retirement["publicationOperation"])
    if operation_path != operation:
        raise ReplayError("Historical publication operation differs")
    _history_merge(pins, operation_path, operation_sha)
    raw_operation = _history_read(operation, pins)
    if (
        set(raw_operation) != {"version", "protocol", "request", "workRoot", "readerCode"}
        or raw_operation["version"] != 1
        or raw_operation["protocol"] != "retained-cleanup-finalization-v1"
    ):
        raise ReplayError("Unsupported historical retained operation")
    historical_code = record(raw_operation["readerCode"])
    code = {name: proof_assembly._sha(digest) for name, digest in historical_code.items()}
    if any(pins.get(name) != digest for name, digest in code.items()):
        raise ReplayError("Operation reader code differs from sealed history")
    request_path, request_sha = _history_reference(raw_operation["request"])
    _history_merge(pins, request_path, request_sha)
    request = _history_read(request_path, pins)
    for field in ("generation", "ffmpeg", "ffprobe"):
        path, digest = _history_reference(request[field])
        _history_merge(pins, path, digest)
    _history_read(publication_workflow, pins)
    _history_read(child_workflow, pins)
    stages = workflow.load_stages(publication_workflow)
    adoption = [stage for stage in stages if stage.name == "adopt-verified-existing-source"]
    children = workflow.load_stages(child_workflow)
    if len(adoption) != 1 or len(children) != 1:
        raise ReplayError("Expected exact historical adoption and child stages")
    parent, child = adoption[0], children[0]
    parent_command = (
        sys.executable,
        "-B",
        "-m",
        "replay.workflow",
        str(child_workflow),
        "--state",
        str(child_checkpoint.parent),
    )
    child_command = (
        sys.executable,
        "-B",
        "-m",
        "replay.code_inventory_memo_run",
        "run",
        "replay.retained_cleanup_finalization",
        "verify-assembly",
        str(operation),
    )
    if (
        parent.command != parent_command
        or parent.verify_command != parent_command
        or child.command != child_command
        or child.verify_command != child_command
        or not child.adopt_existing
        or child.name != "assemble-clean-source"
        or adoption_checkpoint.name != f"{parent.name}.json"
        or child_checkpoint.name != f"{child.name}.json"
    ):
        raise ReplayError("Historical adoption command or checkpoint identity differs")
    child_digest = file_digest(child_checkpoint)
    child_outputs = {str(verified.keeper): proof_assembly._sha(authenticated_seal["keeperSHA256"])}
    assembly = record(retirement["assembly"])
    for field in ("assemblyReceipt", "cleanIndex"):
        path, digest = _history_reference(assembly[field])
        if pins.get(str(path)) != digest:
            raise ReplayError("Historical assembly output is not sealed")
        child_outputs[str(path)] = digest
    _history_merge(pins, child_checkpoint, child_digest)
    _history_complete(
        child, checkpoint=_history_read(child_checkpoint, pins), pins=pins, outputs=child_outputs
    )
    adoption_digest = file_digest(adoption_checkpoint)
    _history_merge(pins, adoption_checkpoint, adoption_digest)
    _history_complete(
        parent,
        checkpoint=_history_read(adoption_checkpoint, pins),
        pins=pins,
        outputs={str(child_checkpoint): child_digest},
    )
    root = proof_successor._path(raw_operation["workRoot"])
    job = proof_original._request(request_path, operation, root, code)
    source_pins = {
        Path(name): digest for name, digest in pins.items() if _scope_historical_source(Path(name))
    }
    return PublicationHistory(
        job,
        retirement,
        pins,
        source_pins,
        parent,
        child,
        child_outputs,
        verified.pins,
        verified.alias,
        verified.keeper,
    )


@dataclass(frozen=True)
class SourceBytes:
    """Actual bytes whose hash matches a separately authenticated historical pin."""

    path: Path
    sha256: str

    def read(self) -> bytes:
        if SHA256.fullmatch(self.sha256) is None:
            raise ReplayError("Expected an exact source SHA256")
        return _bound_bytes(self.path, self.sha256, MAX_SOURCE_BYTES)


@dataclass(frozen=True)
class SourceReview:
    module: str
    historical_sha256: str
    current_sha256: str
    used_symbols: tuple[str, ...]
    changed_unused_symbols: tuple[str, ...]

    def json(self) -> dict[str, object]:
        return {
            "module": self.module,
            "historicalSHA256": self.historical_sha256,
            "currentSHA256": self.current_sha256,
            "usedSymbols": list(self.used_symbols),
            "changedUnusedSymbols": list(self.changed_unused_symbols),
            "historicalBytesExecuted": False,
        }


def _tree(source: bytes) -> ast.Module:
    try:
        return ast.parse(source)
    except (SyntaxError, ValueError) as error:
        raise ReplayError("Source evidence must be valid Python") from error


def _dump(node: ast.AST) -> str:
    return ast.dump(node, include_attributes=False)


def _declarations(tree: ast.Module) -> tuple[dict[str, ast.AST], tuple[str, ...]]:
    declarations: dict[str, ast.AST] = {}
    execution: list[str] = []
    for node in tree.body:
        if isinstance(node, DEFINITIONS):
            if node.name in declarations:
                raise ReplayError("Ambiguous duplicate source definition")
            declarations[node.name] = node
        else:
            execution.append(_dump(node))
    return declarations, tuple(execution)


def _consumer_symbols(source: bytes, module: str) -> set[str]:
    tree = _tree(source)
    aliases: dict[str, str | None] = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            if any(alias.name == module and alias.asname is None for alias in node.names):
                raise ReplayError("Qualified imports require an explicit stronger review")
            aliases.update(
                (alias.asname or alias.name, None) for alias in node.names if alias.name == module
            )
        elif isinstance(node, ast.ImportFrom):
            if node.module == module:
                if any(alias.name == "*" for alias in node.names):
                    raise ReplayError("Wildcard consumers need an explicit stronger review")
                aliases.update((alias.asname or alias.name, alias.name) for alias in node.names)
            elif node.module == "replay":
                aliases.update(
                    (alias.asname or alias.name, None)
                    for alias in node.names
                    if f"replay.{alias.name}" == module
                )
    if aliases and any(
        (isinstance(node, ast.Name) and node.id in DYNAMIC_NAMES)
        or (isinstance(node, ast.Attribute) and node.attr in REFLECTIVE_ATTRIBUTES)
        for node in ast.walk(tree)
    ):
        raise ReplayError("Reflective consumer access requires a stronger source review")
    parents = {child: parent for parent in ast.walk(tree) for child in ast.iter_child_nodes(parent)}
    used: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Name) and node.id in aliases and isinstance(node.ctx, ast.Load):
            symbol = aliases[node.id]
            if symbol is not None:
                used.add(symbol)
            elif isinstance(parents.get(node), ast.Attribute):
                parent = parents[node]
                if not isinstance(parent, ast.Attribute) or parent.value is not node:
                    raise ReplayError("Module alias escapes the static consumer contract")
                used.add(parent.attr)
            else:
                raise ReplayError("Module alias escapes the static consumer contract")
    return used


def _reachable(declarations: Mapping[str, ast.AST], initial: set[str]) -> set[str]:
    pending = list(initial)
    visited: set[str] = set()
    while pending:
        symbol = pending.pop()
        if symbol in visited:
            continue
        visited.add(symbol)
        node = declarations.get(symbol)
        if node is None:
            # Imports/constants are covered by identical module-level AST.
            continue
        names = {child.id for child in ast.walk(node) if isinstance(child, ast.Name)}
        if names & DYNAMIC_NAMES:
            raise ReplayError("Dynamic symbol access requires a stronger source review")
        pending.extend((names & declarations.keys()) - visited)
    return visited


def _definition_time_symbols(tree: ast.Module) -> set[str]:
    """Include entry calls, aliases, defaults, decorators and class execution."""
    executed: list[ast.AST] = []
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            executed.extend((node.args, *node.decorator_list, *node.type_params))
            if node.returns is not None:
                executed.append(node.returns)
        else:
            # Class bodies execute at import time, including method defaults.
            # Conservatively retain their method references as well.
            executed.append(node)
    names = {
        child.id for node in executed for child in ast.walk(node) if isinstance(child, ast.Name)
    }
    if names & DYNAMIC_NAMES:
        raise ReplayError("Reflective module execution requires a stronger source review")
    return names


def review_source_transition(
    *,
    module: str,
    historical: SourceBytes,
    current: SourceBytes,
    authenticated_historical_sha256: str,
    consumers: tuple[SourceBytes, ...],
) -> SourceReview:
    """Compare genuine old/new ASTs and every supplied authenticated consumer.

    The owner must supply the complete consumer closure authenticated by the
    historical proof. This function never treats a source hash manifest as bytes,
    executes old source, or overrides an original generation's loader.
    """
    if (
        re.fullmatch(r"replay\.[a-z][a-z0-9_]*", module) is None
        or historical.sha256 != authenticated_historical_sha256
        or not 1 <= len(consumers) <= MAX_SOURCES
    ):
        raise ReplayError("Require exact historical identity and bounded consumer closure")
    old_bytes, current_bytes = historical.read(), current.read()
    old_tree, new_tree = _tree(old_bytes), _tree(current_bytes)
    old, old_execution = _declarations(old_tree)
    new, new_execution = _declarations(new_tree)
    if old_execution != new_execution:
        raise ReplayError("Module execution or imported dependencies changed")
    used = _definition_time_symbols(old_tree) | _definition_time_symbols(new_tree)
    used.update({"__getattr__", "__dir__"} & (old.keys() | new.keys()))
    for consumer in consumers:
        used.update(_consumer_symbols(consumer.read(), module))
    used = _reachable(old, used) | _reachable(new, used)
    if any(
        name not in old or name not in new or _dump(old[name]) != _dump(new[name])
        for name in used
        if name in old or name in new
    ):
        raise ReplayError("An actually referenced source symbol changed")
    changed = {
        name
        for name in old.keys() | new.keys()
        if name not in old or name not in new or _dump(old[name]) != _dump(new[name])
    }
    # Decorators, defaults, annotations and class bodies execute at definition
    # time. An unused definition change is safe only for undecorated functions
    # with unchanged signature/default/annotation AST, never class changes.
    for name in changed:
        before, after = old.get(name), new.get(name)
        if (
            not isinstance(before, (ast.FunctionDef, ast.AsyncFunctionDef))
            or not isinstance(after, (ast.FunctionDef, ast.AsyncFunctionDef))
            or type(before) is not type(after)
        ):
            raise ReplayError("Definition-time changes require a stronger source review")
        if before.decorator_list or after.decorator_list:
            raise ReplayError("Decorated definition changes require a stronger source review")
        if (
            _dump(before.args) != _dump(after.args)
            or (None if before.returns is None else _dump(before.returns))
            != (None if after.returns is None else _dump(after.returns))
            or tuple(_dump(node) for node in before.type_params)
            != tuple(_dump(node) for node in after.type_params)
        ):
            raise ReplayError("Definition signature changed")
    historical.read()
    current.read()
    return SourceReview(
        module, historical.sha256, current.sha256, tuple(sorted(used)), tuple(sorted(changed))
    )


def historical_stage_fingerprint(stage: Stage, inputs: Mapping[str, str]) -> str:
    """Use a genuine sealed hash table without claiming current source equality."""
    if set(inputs) != {str(path) for path in stage.inputs} or any(
        SHA256.fullmatch(value) is None for value in inputs.values()
    ):
        raise ReplayError("Historical stage input table is incomplete or invalid")
    definition = {
        "command": stage.command,
        "verifyCommand": stage.verify_command,
        "inputs": dict(inputs),
        "outputs": [str(path) for path in stage.outputs],
        "timeoutSeconds": stage.timeout_seconds,
        "workingDirectory": str(Path.cwd().resolve()),
    }
    return hashlib.sha256(json.dumps(definition, sort_keys=True).encode("utf-8")).hexdigest()


def review_reader_generation(
    *,
    historical_pins: Mapping[Path, str],
    current_pins: Mapping[Path, str],
    historical_bytes: Mapping[Path, SourceBytes],
) -> tuple[SourceReview, ...]:
    """Bind every actual consumer to an exact authenticated reader inventory.

    The historical pin table must originate from the verified immutable stage
    seal. Unchanged modules supply their actual matching current bytes; changed
    modules require separately retained authentic old bytes. Added/removed
    dependencies are rejected rather than implicitly classified as unused.
    """
    if not 1 <= len(historical_pins) <= MAX_SOURCES or set(current_pins) != set(historical_pins):
        raise ReplayError("Historical and current dependency inventories differ")
    changed = {path for path, digest in historical_pins.items() if current_pins[path] != digest}
    if set(historical_bytes) != changed:
        raise ReplayError("Require genuine old bytes for the exact changed source set")
    old_sources = {
        path: historical_bytes[path] if path in changed else SourceBytes(path, digest)
        for path, digest in historical_pins.items()
    }
    if any(old_sources[path].sha256 != digest for path, digest in historical_pins.items()):
        raise ReplayError("Historical byte evidence belongs to a different source generation")
    current_sources = {path: SourceBytes(path, digest) for path, digest in current_pins.items()}
    consumers = (*old_sources.values(), *current_sources.values())
    if len(consumers) > MAX_SOURCES:
        raise ReplayError("Consumer source closure exceeds its bounded review budget")
    for source in consumers:
        source.read()
    reviews = tuple(
        review_source_transition(
            module=f"replay.{path.stem}",
            historical=old_sources[path],
            current=current_sources[path],
            authenticated_historical_sha256=historical_pins[path],
            consumers=consumers,
        )
        for path in sorted(changed)
    )
    for source in consumers:
        source.read()
    return reviews


def verify_completed_stage(
    stage: Stage, *, checkpoint: Mapping[str, object], authenticated_inputs: Mapping[str, str]
) -> None:
    """Authenticate a real post-command completion; never promote a running stage."""
    if not stage.verify_command:
        raise ReplayError("A normal post-command verifier is required")
    if (
        set(checkpoint) != {"status", "fingerprint", "outputs"}
        or checkpoint.get("status") != "complete"
        or checkpoint.get("fingerprint")
        != historical_stage_fingerprint(stage, authenticated_inputs)
        or checkpoint.get("outputs") != {str(path): file_digest(path) for path in stage.outputs}
    ):
        raise ReplayError("Genuine completed stage or its output bytes changed")
    record(checkpoint.get("outputs"))


def _scope_historical_source(path: Path) -> bool:
    return path.parent == ROOT / "src/replay" and path.suffix == ".py" and path == path.resolve()


def _scope_pin(pins: dict[str, str], path: Path, digest: object) -> None:
    expected = proof_assembly._sha(digest)
    if _scope_historical_source(path):
        return
    name = str(path)
    if name in pins and pins[name] != expected:
        raise ReplayError("Conflicting current dependency bindings")
    pins[name] = expected


def _scope_recorded(
    pins: dict[str, str], data: dict[str, object], *, selected: set[str] | None = None
) -> None:
    for name, digest in record(data["inputs"]).items():
        if selected is None or name in selected:
            _scope_pin(pins, proof_successor._path(name, work=False), digest)


def _scope_read(path: Path, pins: dict[str, str]) -> dict[str, object]:
    return proof_successor._read(path, pins)


def _scope_reference(pins: dict[str, str], value: object) -> Path:
    reference = record(value)
    if set(reference) != {"path", "sha256"}:
        raise ReplayError("Expected exact current dependency reference")
    path = proof_successor._path(reference["path"], work=False)
    if _scope_historical_source(path):
        raise ReplayError("A material reference cannot be classified as historical source")
    if path.stat().st_size <= 0:
        raise ReplayError("Current material references must be nonempty")
    _scope_pin(pins, path, reference["sha256"])
    return path


def _scope_stage_pins(
    pins: dict[str, str],
    *,
    graph: Path,
    state: Path,
    names: set[str] | None = None,
    recorded_sources: Mapping[str, object] | None = None,
) -> None:
    """Guard current Stage snapshots; historical authentication is separate."""
    _scope_read(graph, pins)
    stages = workflow.load_stages(graph)
    selected = [stage for stage in stages if names is None or stage.name in names]
    if names is not None and {stage.name for stage in selected} != names:
        raise ReplayError("Selected publication producer Stage is missing")
    for stage in selected:
        checkpoint = _scope_read(state / f"{stage.name}.json", pins)
        outputs = record(checkpoint.get("outputs"))
        if checkpoint.get("status") != "complete" or set(outputs) != {
            str(path) for path in stage.outputs
        }:
            raise ReplayError("Publication producer lacks genuine complete outputs")
        for path in stage.outputs:
            _scope_pin(pins, path, outputs[str(path)])
        missing = {
            str(path)
            for path in stage.inputs
            if not _scope_historical_source(path) and str(path) not in pins
        }
        if missing:
            examples = ", ".join(sorted(missing)[:8])
            raise ReplayError(
                f"Publication producer {stage.name} input lacks an authenticated binding: "
                f"{examples}"
            )
        historical: dict[str, str] = {}
        for path in stage.inputs:
            name = str(path)
            if _scope_historical_source(path):
                if recorded_sources is None or name not in recorded_sources:
                    raise ReplayError("Publication producer lacks a historical source binding")
                historical[name] = proof_assembly._sha(recorded_sources[name])
            else:
                historical[name] = pins[name]
        if set(checkpoint) != {"status", "fingerprint", "outputs"} or checkpoint.get(
            "fingerprint"
        ) != historical_stage_fingerprint(stage, historical):
            raise ReplayError(f"Publication producer {stage.name} historical fingerprint changed")


def _scope_selected_input_union(data: dict[str, object]) -> set[str]:
    """Union the unchanged successor selection with one canonical recorded scan."""
    selected = {
        str(proof_successor._path(data[field]))
        for field in ("request", "originalPlan", "handoff", "settings")
    }
    request = proof_common._read(proof_successor._path(data["request"]))
    for value in proof_successor._list(request["oldWorkers"]):
        worker = record(value)
        selected.add(str(proof_successor._path(record(worker["workflow"])["path"])))
    directories: set[Path] = set()
    for value in proof_successor._list(data["chunks"]):
        row = record(value)
        directories.add(proof_successor._path(row["original"]).parent)
        for value_stage in proof_successor._list(row["oldStages"]):
            stage = record(value_stage)
            selected.add(str(proof_successor._path(stage["state"]) / f"{stage['name']}.json"))
    handoff = proof_common._read(proof_successor._path(data["handoff"]))
    selected.update(
        str(proof_successor._path(record(value)["path"]))
        for value in proof_successor._list(handoff["evidence"])
    )
    settings = proof_common._read(proof_successor._path(data["settings"]))
    for field in (
        "native",
        "decoder",
        "model",
        "sourceClock",
        "classificationEvidence",
        "profileInputs",
        "profileComplete",
    ):
        path = proof_successor._path(record(settings[field])["path"], work=False)
        selected.add(str(path))
        if field in {"profileInputs", "profileComplete"}:
            directories.add(path.parent)
    for name in record(data["inputs"]):
        path = proof_successor._path(name)
        if path in directories or any(parent in directories for parent in path.parents):
            selected.add(name)
    return selected


def _scope_source_table(data: dict[str, object]) -> dict[str, object]:
    sources: dict[str, object] = {}
    for field in ("inputs", "readerCode"):
        for name, digest in record(data.get(field, {})).items():
            path = Path(name)
            if _scope_historical_source(path):
                proof_successor._path(name, work=False)
                expected = proof_assembly._sha(digest)
                if name in sources and sources[name] != expected:
                    raise ReplayError("Conflicting historical generation source bindings")
                sources[name] = expected
    return sources


def current_dependency_pins(
    job: proof_common.Job, *, verified_lineage_pins: Mapping[str, str] | None = None
) -> dict[str, str]:
    """Derive expected selected material pins without hashing large dependencies.

    Historical source hashes are deliberately excluded from current material
    comparisons. The caller separately authenticates the historical normal
    completion proofs and pins its current reader. This function neither runs a
    historical reader nor proves source compatibility.
    """
    metadata = proof_original._generation(job.plan)
    pins = dict(verified_lineage_pins or {})
    manifest = _scope_read(job.root / "finalized/manifest.json", pins)
    for value in proof_successor._list(manifest["chunks"]):
        chunk = record(value)
        _scope_reference(pins, chunk["candidate"])
        _scope_reference(pins, chunk["producer"])
    _scope_recorded(pins, metadata)
    _scope_read(job.plan, pins)
    operation = _scope_read(job.operation, pins)
    request = _scope_reference(pins, operation["request"])
    request_data = _scope_read(request, pins)
    if _scope_reference(pins, request_data["generation"]) != job.plan:
        raise ReplayError("Publication generation binding differs")
    legacy = _scope_read(proof_successor._path(metadata["successor"]), pins)
    rows = proof_successor._list(metadata["chunks"])
    prior_rows = proof_successor._list(legacy["chunks"])
    if len(rows) != len(prior_rows):
        raise ReplayError("Publication selection coverage differs")
    selected = _scope_selected_input_union(legacy)
    _scope_recorded(pins, legacy, selected=selected)
    for field in ("settings", "originalPlan", "handoff"):
        _scope_read(proof_successor._path(metadata[field]), pins)
    settings = _scope_read(proof_successor._path(metadata["settings"]), pins)
    for field in (
        "source",
        "native",
        "decoder",
        "model",
        "sourceClock",
        "classificationEvidence",
        "profileInputs",
        "profileComplete",
    ):
        _scope_reference(pins, settings[field])
    for tool in job.tools:
        _scope_reference(pins, tool)
    for value in rows:
        row = record(value)
        _scope_read(proof_successor._path(row["chunk"]), pins)
        proof = _scope_read(proof_successor._path(row["proof"]), pins)
        repair = record(proof["repairEvidence"])
        _scope_reference(pins, repair["receipt"])
        _scope_reference(pins, repair["media"])
        if row["adoptedProof"] is not None:
            _scope_reference(pins, row["adoptedProof"])
    historical_sources = _scope_source_table(metadata)
    for graph_name in proof_successor._strings(metadata["workflows"]):
        _scope_read(Path(graph_name), pins)
    for graph, state, status in job.lanes:
        outcome = bounded_read(status)
        if outcome.strip() != b"0":
            raise ReplayError("Publication recovery parent is not terminal zero")
        _scope_pin(pins, status, hashlib.sha256(outcome).hexdigest())
        _scope_stage_pins(pins, graph=graph, state=state, recorded_sources=historical_sources)
    for graph_name, state_name in zip(
        proof_successor._strings(metadata["workflows"]),
        proof_successor._strings(metadata["states"]),
        strict=True,
    ):
        _scope_stage_pins(
            pins,
            graph=Path(graph_name),
            state=Path(state_name),
            recorded_sources=historical_sources,
        )
    legacy_sources = _scope_source_table(legacy)
    for index, value in enumerate(prior_rows):
        if record(rows[index])["mode"] != "retained-completed":
            continue
        row = record(value)
        lane = row["lane"]
        if type(lane) is not int or not 0 <= lane < len(
            proof_successor._strings(legacy["workflows"])
        ):
            raise ReplayError("Invalid publication producer lane")
        _scope_stage_pins(
            pins,
            graph=Path(proof_successor._strings(legacy["workflows"])[lane]),
            state=Path(proof_successor._strings(legacy["states"])[lane]),
            names=set(proof_successor._strings(record(rows[index])["oldStageNames"])),
            recorded_sources=legacy_sources,
        )
    return pins


def verify_current_dependency_pins(pins: Mapping[str, str]) -> None:
    """Compare expected material bindings inside the caller's one hash context."""
    for name, expected in pins.items():
        path = proof_successor._path(name, work=False)
        if _scope_historical_source(path):
            raise ReplayError("Historical source cannot enter current material pins")
        if file_digest(path) != proof_assembly._sha(expected):
            raise ReplayError("Current publication dependency bytes changed")


def _scope_stage_leases(graph: Path, state: Path) -> set[Path]:
    return {state / f"{stage.name}.lock" for stage in workflow.load_stages(graph)}


def publication_leases(
    job: proof_common.Job, retirement_request: dict[str, object]
) -> tuple[Path, ...]:
    """Derive the same producer, lane and workflow writer leases as retention."""
    metadata = proof_original._generation(job.plan)
    legacy = proof_common._read(proof_successor._path(metadata["successor"]))
    paths = {
        proof_successor._path(value)
        for value in proof_successor._strings(retirement_request["leases"])
    }
    states = [state for _, state, _ in job.lanes]
    for document in (metadata, legacy):
        states.extend(Path(name) for name in proof_successor._strings(document["states"]))
        states.extend(Path(name) for name in proof_successor._strings(document["oldStates"]))
    paths.update(state / ".workflow.lock" for state in states)
    for graph, state, _ in job.lanes:
        paths.update(_scope_stage_leases(graph, state))
    for document in (metadata, legacy):
        for graph_name, state_name in zip(
            proof_successor._strings(document["workflows"]),
            proof_successor._strings(document["states"]),
            strict=True,
        ):
            paths.update(_scope_stage_leases(Path(graph_name), Path(state_name)))
    for value in proof_successor._list(metadata["chunks"]):
        row = record(value)
        paths.add(job.plan.parent / f"recovery-lane-{row['lane']}.lock")
        state = proof_successor._path(row["oldState"])
        paths.update(
            state / f"{name}.lock" for name in proof_successor._strings(row["oldStageNames"])
        )
    for value in proof_successor._list(legacy["chunks"]):
        for stage_value in proof_successor._list(record(value)["oldStages"]):
            stage = record(stage_value)
            paths.add(proof_successor._path(stage["state"]) / f"{stage['name']}.lock")
    return tuple(sorted(proof_successor._path(str(path)) for path in paths))


def _lineage_authenticated_read(
    path: Path, *, historical_pins: Mapping[str, str], pins: dict[str, str]
) -> dict[str, object]:
    digest = historical_pins.get(str(path))
    if digest is None:
        raise ReplayError("Publication lineage lacks an authenticated historical pin")
    _scope_pin(pins, path, digest)
    return _scope_read(path, pins)


def _lineage_metadata_digest(path: Path, pins: dict[str, str]) -> str:
    digest = hashlib.sha256(bounded_read(proof_successor._path(str(path)))).hexdigest()
    _scope_pin(pins, path, digest)
    return digest


def _lineage_current_proof(
    *,
    job: proof_common.Job,
    row: dict[str, object],
    reference: dict[str, object],
    indexed: dict[str, object],
    pins: dict[str, str],
) -> None:
    receipt_path = _scope_reference(pins, reference["candidate"])
    producer = _scope_reference(pins, reference["producer"])
    receipt = _scope_read(receipt_path, pins)
    _scope_read(producer, pins)
    media = proof_successor._path(indexed["media"], work=False)
    media_sha = proof_assembly._sha(receipt["videoSHA256"])
    _scope_pin(pins, media, media_sha)
    repair = {
        "receipt": {"path": str(receipt_path), "sha256": pins[str(receipt_path)]},
        "media": {"path": str(media), "sha256": media_sha},
    }
    if row["repairEvidence"] is not None and row["repairEvidence"] != repair:
        raise ReplayError("Current repair evidence differs from authenticated lineage")
    chunk = proof_successor._path(row["chunk"])
    verification = {
        "version": 1,
        "kind": "readonly-original-cleanup-admission",
        "chunk": {"path": str(chunk), "sha256": _lineage_metadata_digest(chunk, pins)},
        "outputs": {
            str(chunk.parent / name): _lineage_metadata_digest(chunk.parent / name, pins)
            for name in (
                "ocr-batches.json",
                "admission.json",
                "eligibility.json",
                "candidate-config.json",
            )
        },
        "repairVerified": True,
        "originalStageCompleted": False,
        "qualityApproved": False,
    }
    if record(verification["outputs"]).get(str(producer)) != pins[str(producer)]:
        raise ReplayError("Current admission proof differs from its authenticated producer")
    expected = {
        "version": 1,
        "generation": {"path": str(job.plan), "sha256": pins[str(job.plan)]},
        "index": row["index"],
        "mode": row["mode"],
        "originalChunk": row["originalChunk"],
        "adoptedProof": row["adoptedProof"],
        "verification": verification,
        "repairEvidence": repair,
        "qualityApproved": False,
    }
    if _scope_read(proof_successor._path(row["proof"]), pins) != expected:
        raise ReplayError(
            "Current recovery proof differs from authenticated selection and material"
        )


def current_lineage_validation(
    job: proof_common.Job, historical_pins: Mapping[str, str]
) -> dict[str, str]:
    """Validate current proof structure against genuinely authenticated lineage.

    The old normal post-command proofs authenticate historical verification.
    Fresh recovery proof and admission artifact hashes below are current
    structural snapshots, never assertions of historically sealed proof hashes
    or renewed admission algorithm verification.
    """
    pins: dict[str, str] = {}
    finalized = _lineage_authenticated_read(
        job.root / "finalized.json", historical_pins=historical_pins, pins=pins
    )
    manifest_path = job.root / "finalized/manifest.json"
    manifest = _lineage_authenticated_read(
        manifest_path, historical_pins=historical_pins, pins=pins
    )
    lineage = _lineage_authenticated_read(
        job.root / "finalized/lineage.json", historical_pins=historical_pins, pins=pins
    )
    index_path = _scope_reference(pins, manifest["index"])
    indexed = _scope_read(index_path, pins)
    _scope_pin(pins, job.plan, record(lineage["generation"])["sha256"])
    metadata = _scope_read(job.plan, pins)
    proof_original._generation(job.plan)
    operation = {
        "path": str(job.operation),
        "sha256": _lineage_metadata_digest(job.operation, pins),
    }
    prior = _scope_read(proof_successor._path(metadata["originalPlan"]), pins)
    settings = _scope_read(proof_successor._path(metadata["settings"]), pins)
    outputs = {
        str(job.root / "finalized" / name): historical_pins.get(str(job.root / "finalized" / name))
        for name in proof_original.FILES
    }
    if any(digest is None for digest in outputs.values()) or finalized != {
        "version": 1,
        "protocol": "retained-cleanup-finalization-v1",
        "operation": operation,
        "sourceClock": settings["sourceClock"],
        "physicalSourceFrames": prior["plannedFrames"],
        "outputs": outputs,
    }:
        raise ReplayError("Finalized publication differs from authenticated metadata outputs")
    if lineage != {
        "version": 1,
        "generation": {"path": str(job.plan), "sha256": pins[str(job.plan)]},
        "manifest": {"path": str(manifest_path), "sha256": pins[str(manifest_path)]},
        "selectedResults": metadata["chunks"],
        "requiredNextStage": "delivery_source_assembly full packet/decoded PTS verification",
        "qualityApproved": False,
        "deliveryComplete": False,
    }:
        raise ReplayError("Publication lineage selection differs from its authenticated generation")
    rows = proof_successor._list(metadata["chunks"])
    references = proof_successor._list(manifest["chunks"])
    ranges = proof_successor._list(indexed["ranges"])
    if not len(rows) == len(references) == len(ranges):
        raise ReplayError("Publication lineage and material coverage differ")
    for row, reference, indexed_row in zip(rows, references, ranges, strict=True):
        _lineage_current_proof(
            job=job,
            row=record(row),
            reference=record(reference),
            indexed=record(indexed_row),
            pins=pins,
        )
    return pins


BOUNDARY_KIND = "current-publication-proof-seal-request"
BOUNDARY_PROTOCOL = "current-source-clean-publication-boundary-v1"
BOUNDARY_REFS = (
    "operation",
    "custodySeal",
    "publicationWorkflow",
    "adoptionCheckpoint",
    "childWorkflow",
    "childCheckpoint",
)


def _boundary_local(value: object) -> Path:
    path = _path(value)
    if not path.is_relative_to(ROOT / "edits/work"):
        raise ReplayError("Current publication metadata must remain in checkout work")
    return path


def _boundary_unique(rows: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for name, value in rows:
        if name in result:
            raise ReplayError("Duplicate current publication metadata key")
        result[name] = value
    return result


def _boundary_read(path: Path, digest: str | None = None) -> dict[str, object]:
    try:
        return record(
            json.loads(
                _bound_bytes(path, digest or file_digest(path), 8 * 1024 * 1024),
                object_pairs_hook=_boundary_unique,
            )
        )
    except ValueError as error:
        raise ReplayError("Invalid current publication JSON") from error


def _boundary_reference(value: object) -> Path:
    raw = record(value)
    digest = raw.get("sha256")
    if (
        set(raw) != {"path", "sha256"}
        or not isinstance(digest, str)
        or SHA256.fullmatch(digest) is None
    ):
        raise ReplayError("Exact current publication reference is required")
    path = _boundary_local(raw["path"])
    if file_digest(path) != digest:
        raise ReplayError("Current publication reference changed")
    return path


def _boundary_request(request: Path) -> dict[str, object]:
    _boundary_local(str(request))
    raw = _boundary_read(request)
    if (
        set(raw) != {"version", "kind", "captures", *BOUNDARY_REFS}
        or type(raw["version"]) is not int
        or raw["version"] != 1
        or raw["kind"] != BOUNDARY_KIND
        or not isinstance(raw["captures"], list)
        or not 1 <= len(raw["captures"]) <= 8
    ):
        raise ReplayError("Invalid current publication proof request")
    for name in BOUNDARY_REFS:
        _boundary_reference(raw[name])
    for value in raw["captures"]:
        _boundary_reference(value)
    return raw


def _boundary_code() -> dict[str, str]:
    return {
        str(source.path): source.sha256
        for source in proof_inventory.capture(
            ("publication_proof_completion_seal", "workflow", "phase_timer")
        )
    }


def _boundary_history(raw: dict[str, object]) -> "PublicationHistory":
    return load_publication_history(
        operation=_boundary_reference(raw["operation"]),
        custody_seal=_boundary_reference(raw["custodySeal"]),
        publication_workflow=_boundary_reference(raw["publicationWorkflow"]),
        adoption_checkpoint=_boundary_reference(raw["adoptionCheckpoint"]),
        child_workflow=_boundary_reference(raw["childWorkflow"]),
        child_checkpoint=_boundary_reference(raw["childCheckpoint"]),
    )


def _boundary_material(history: "PublicationHistory") -> dict[str, str]:
    pins: dict[str, str] = {}
    for name, digest in history.historical_inputs.items():
        _scope_pin(pins, _path(name), digest)
    for ref in history.evidence_pins:
        _scope_pin(pins, ref.path, ref.sha256)
    for name, digest in history.child_outputs.items():
        _scope_pin(pins, _path(name), digest)
    lineage_pins = current_lineage_validation(history.job, history.historical_inputs)
    for name, digest in lineage_pins.items():
        _scope_pin(pins, _path(name), digest)
    # The history loader authenticates the manifest before this reader derives
    # selected current material. Conflicts never become fresh replacement pins.
    for name, digest in current_dependency_pins(
        history.job, verified_lineage_pins=lineage_pins
    ).items():
        _scope_pin(pins, _path(name), digest)
    return pins


def prepare_current_boundary(request: Path, output: Path) -> Path:
    """Prepare a new current reader, preserving every historical operation pin."""
    _boundary_local(str(output))
    code = _boundary_code()
    raw = _boundary_request(request)
    history = _boundary_history(raw)
    pins = _boundary_material(history)
    if code != _boundary_code():
        raise ReplayError("Current reader changed during boundary preparation")
    if output.exists():
        raise ReplayError("Current boundary namespace already exists")
    output.mkdir(parents=True)
    dependencies = output / "material-pins.json"
    _save(dependencies, {"version": 1, "inputs": pins})
    operation = output / "operation.json"
    _save(
        operation,
        {
            "version": 1,
            "protocol": BOUNDARY_PROTOCOL,
            "request": _reference(request),
            "materialPins": _reference(dependencies),
            "currentReader": code,
            "outputRoot": str(output),
        },
    )
    command = [sys.executable, "-B", "-m", "replay.publication_proof_completion_seal"]
    inputs = [
        operation,
        request,
        dependencies,
        *map(Path, code),
        *(_boundary_reference(raw[name]) for name in BOUNDARY_REFS),
        *(_boundary_reference(value) for value in proof_successor._list(raw["captures"])),
        history.keeper,
    ]
    graph = output / "workflow.json"
    _save(
        graph,
        {
            "version": 1,
            "stages": [
                {
                    "name": "seal-current-source-clean-boundary",
                    "inputs": [str(path) for path in dict.fromkeys(inputs)],
                    "outputs": [
                        str(output / name)
                        for name in ("completion.json", "source-cleanup.json", "current-proof.json")
                    ],
                    "command": [*command, "run", str(operation)],
                    "verify_command": [*command, "verify", str(operation)],
                    "timeout_seconds": 1200,
                }
            ],
        },
    )
    workflow.load_stages(graph)
    return graph


def _boundary_guard(
    history: "PublicationHistory", raw: dict[str, object]
) -> list[dict[str, object]]:
    _boundary_history(raw)
    actors = proof_retirement._actors(raw["captures"])
    # The retired alias is absent by the typed custody check. Query its surviving
    # inode through the keeper; a missing-path lsof error is not an FD observation.
    # Unknown/deleted alias names returned for that inode remain fail-closed.
    if proof_leaves.lsof_open_among([str(history.keeper)]) != set():
        raise ReplayError("Publication endpoints have an open or unobservable descriptor")
    # Re-run the typed custody endpoint check, including exact inode, nlink=1,
    # alias absence and all immutable old completion evidence. No media decoder.
    _boundary_history(raw)
    return actors


def _boundary_identities(pins: Mapping[str, str]) -> dict[str, tuple[int, int, int, int, int]]:
    identities: dict[str, tuple[int, int, int, int, int]] = {}
    for name in pins:
        path = _path(name)
        info = path.lstat()
        if not stat.S_ISREG(info.st_mode) or (
            info.st_size == 0 and pins[name] != hashlib.sha256(b"").hexdigest()
        ):
            raise ReplayError("Current publication input is not a bound regular file")
        identities[name] = (
            info.st_dev,
            info.st_ino,
            info.st_size,
            info.st_mtime_ns,
            info.st_ctime_ns,
        )
    return identities


def _boundary_digest(path: Path, expected: str) -> str:
    """Preserve explicitly recorded empty logs without changing Stage artifacts."""
    if expected == hashlib.sha256(b"").hexdigest():
        return hashlib.sha256(bounded_read(path)).hexdigest()
    return file_digest(path)


def _boundary_terminal(
    pins: Mapping[str, str],
    cache: InputHashCache,
    identities: Mapping[str, tuple[int, int, int, int, int]],
) -> dict[str, str]:
    # Cache a bounded large subset; read every other dependency freshly at both
    # endpoints and guard every identity, including the uncached metadata.
    if _boundary_identities(pins) != dict(identities):
        raise ReplayError("Current publication input identity changed")
    fresh = cache.verify()
    result = {
        name: fresh[name] if name in fresh else _boundary_digest(Path(name), expected)
        for name, expected in pins.items()
    }
    if result != dict(pins) or _boundary_identities(pins) != dict(identities):
        raise ReplayError("Current publication terminal input bytes changed")
    return result


def execute_current_boundary(operation: Path, *, verify: bool = False) -> Path:
    """Revalidate actual material under a current reader; never claim old compatibility."""
    _boundary_local(str(operation))
    document = _boundary_read(operation)
    if (
        set(document)
        != {"version", "protocol", "request", "materialPins", "currentReader", "outputRoot"}
        or type(document["version"]) is not int
        or document["version"] != 1
        or document["protocol"] != BOUNDARY_PROTOCOL
    ):
        raise ReplayError("Invalid current publication operation")
    output = _boundary_local(document["outputRoot"])
    if operation != output / "operation.json" or document["currentReader"] != _boundary_code():
        raise ReplayError("Current publication reader or operation namespace changed")
    request = _boundary_reference(document["request"])
    raw = _boundary_request(request)
    history = _boundary_history(raw)
    bound = _boundary_read(_boundary_reference(document["materialPins"]))
    pins = _boundary_material(history)
    if bound != {"version": 1, "inputs": pins}:
        raise ReplayError("Current material scope differs from its prepared bindings")
    current = record(document["currentReader"])
    for name, digest in current.items():
        if not isinstance(digest, str) or SHA256.fullmatch(digest) is None:
            raise ReplayError("Invalid current reader pin")
        _scope_pin_reader(pins, name, digest)
    for path in (operation, request, _boundary_reference(document["materialPins"])):
        _scope_pin_reader(pins, str(path), file_digest(path))
    if not verify and any(
        (output / name).exists()
        for name in ("completion.json", "source-cleanup.json", "current-proof.json")
    ):
        raise ReplayError("Current boundary outputs already exist; verify them")
    with ExitStack() as stack:
        for path in publication_leases(history.job, history.retirement_request):
            stack.enter_context(proof_successor._unlocked(path))
        stack.enter_context(proof_memo.capture_scope())
        identities = _boundary_identities(pins)
        cache_paths = tuple(
            Path(name)
            for name in sorted(
                (name for name in pins if identities[name][2] > 0),
                key=lambda name: (-identities[name][2], name),
            )[:1024]
        )
        cache = InputHashCache(cache_paths, allow_uncached_overflow=True)
        stack.enter_context(input_hash_context(cache))
        cache.prime()
        for name, expected in pins.items():
            if _boundary_digest(Path(name), expected) != expected:
                raise ReplayError("Current publication input bytes changed")
        actors = _boundary_guard(history, raw)
        assembled = proof_common._assembled(history.job)
        completion = {
            **assembled,
            "protocol": BOUNDARY_PROTOCOL,
            "operation": _reference(operation),
            "historicalOperation": raw["operation"],
            "childStageReceipt": raw["childCheckpoint"],
            "maskQualityApproved": False,
            "editCompletion": False,
            "asrAccuracyApproved": False,
            "historicalReaderEqualsCurrent": False,
            "mediaReassembled": False,
        }
        cleanup = {
            "assemblyManifest": _reference(history.job.root / "finalized/manifest.json"),
            "assemblyReceipt": assembled["assemblyReceipt"],
            "cleanIndex": assembled["cleanIndex"],
        }
        if _boundary_guard(history, raw) != actors or document["currentReader"] != _boundary_code():
            raise ReplayError("Current publication endpoints or reader changed")
        terminal = _boundary_terminal(pins, cache, identities)
        proof = {
            "version": 1,
            "kind": "current-source-clean-publication-proof",
            "operation": _reference(operation),
            "custodySeal": raw["custodySeal"],
            "historicalSourcePins": {
                str(path): digest for path, digest in history.source_pins.items()
            },
            "currentReader": current,
            "inputs": terminal,
            "actors": actors,
            "historicalCompatibilityClaimed": False,
            "sourceCleanMaterialVerified": True,
            "qualityApproved": False,
            "originalOuterStatusRewritten": False,
            "mediaReassembled": False,
        }
        for name, expected_document in (
            ("completion.json", completion),
            ("source-cleanup.json", cleanup),
            ("current-proof.json", proof),
        ):
            destination = output / name
            if verify:
                if _boundary_read(destination) != expected_document:
                    raise ReplayError("Current publication completion proof changed")
            else:
                _save(destination, expected_document)
    return output / "completion.json"


def _scope_pin_reader(pins: dict[str, str], name: str, digest: str) -> None:
    path = _path(name)
    if name in pins and pins[name] != digest:
        raise ReplayError("Current reader conflicts with current material")
    pins[str(path)] = digest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("prepare", "run", "verify"))
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path, nargs="?")
    args = parser.parse_args()
    try:
        if args.mode == "prepare":
            if args.output is None:
                raise ReplayError("Prepare requires a fresh output namespace")
            result = prepare_current_boundary(args.input.absolute(), args.output.absolute())
        else:
            if args.output is not None:
                raise ReplayError("Execution does not accept an output override")
            result = execute_current_boundary(args.input.absolute(), verify=args.mode == "verify")
    except (ReplayError, ValueError, OSError) as error:
        print(f"CURRENT_PUBLICATION_ERROR {error}", file=sys.stderr)
        raise SystemExit(1) from error
    print(result)


if __name__ == "__main__":
    main()
