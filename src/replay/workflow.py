"""Run a trusted local stage manifest without a shell or interactive operations."""

import argparse
import heapq
import json
import signal
import stat
import subprocess
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from graphlib import CycleError, TopologicalSorter
from pathlib import Path
from types import FrameType

from replay.core import ReplayError
from replay.stages import Stage, StageError, execution_lock, run_stage

MAX_PLAN_BYTES = 1_048_576
MAX_STAGES = 512
STAGE_FIELDS = frozenset(
    {"name", "command", "inputs", "outputs", "timeout_seconds", "verify_command", "adopt_existing"}
)
SOURCE_CAPTION_CONSUMERS = frozenset(
    {
        "replay.native_render",
        "replay.native_invocation",
        "replay.part_render",
        "replay.fcp_transformed_source",
        "replay.delivery_render",
    }
)


class Arguments(argparse.Namespace):
    plan: Path
    state: Path
    check: bool
    target: list[str]


@dataclass(frozen=True)
class _Workflow:
    stages: tuple[Stage, ...]
    dependencies: dict[str, tuple[str, ...]]
    paths: dict[Path, Path]
    source_caption_bindings: dict[str, Path]


def _texts(value: object) -> tuple[str, ...]:
    if not isinstance(value, list) or len(value) > 256:
        raise StageError("Expected a bounded argument/path array")
    items: list[str] = []
    for item in value:
        if not isinstance(item, str) or not item or "\0" in item:
            raise StageError("Arguments and paths must be nonempty text")
        items.append(item)
    return tuple(items)


def _stage(value: object) -> Stage:
    if not isinstance(value, dict) or not set(value).issubset(STAGE_FIELDS):
        raise StageError("Unknown stage fields")
    name: object = value.get("name")
    timeout: object = value.get("timeout_seconds")
    adopt: object = value.get("adopt_existing", False)
    if not isinstance(name, str) or type(timeout) is not int or type(adopt) is not bool:
        raise StageError("Invalid stage name, timeout or adoption policy")
    return Stage(
        name=name,
        command=_texts(value.get("command")),
        inputs=tuple(Path(path) for path in _texts(value.get("inputs"))),
        outputs=tuple(Path(path) for path in _texts(value.get("outputs"))),
        timeout_seconds=timeout,
        verify_command=_texts(value.get("verify_command", [])),
        adopt_existing=adopt,
    )


def _parse_workflow(value: object) -> _Workflow:
    if (
        not isinstance(value, dict)
        or not {"version", "stages"}.issubset(value)
        or set(value).difference({"version", "stages", "sourceCaptionProduction"})
    ):
        raise StageError("Invalid workflow manifest fields")
    version: object = value.get("version")
    entries: object = value.get("stages")
    if type(version) is not int or version != 1:
        raise StageError("Unsupported workflow version")
    if not isinstance(entries, list) or not 1 <= len(entries) <= MAX_STAGES:
        raise StageError("Invalid workflow stage count")
    stages = tuple(_stage(entry) for entry in entries)
    if len({stage.name for stage in stages}) != len(stages):
        raise StageError("Workflow stage names must be unique")
    paths = _resolved_paths(stages)
    bindings = (
        _source_caption_bindings(value["sourceCaptionProduction"], stages)
        if "sourceCaptionProduction" in value
        else {}
    )
    return _Workflow(stages, _dependency_graph(stages, paths), paths, bindings)


def _source_caption_bindings(value: object, stages: tuple[Stage, ...]) -> dict[str, Path]:
    if not isinstance(value, dict) or set(value) != {"bindings"}:
        raise StageError("Invalid source-caption production declaration")
    entries: object = value["bindings"]
    if not isinstance(entries, dict) or not entries:
        raise StageError("Source-caption production needs stage bindings")
    known = {stage.name: stage for stage in stages}
    bindings: dict[str, Path] = {}
    for name, raw in entries.items():
        if not isinstance(name, str) or name not in known:
            raise StageError("Unknown source-caption consumer stage")
        if not isinstance(raw, str) or not raw or "\0" in raw:
            raise StageError("Invalid source-caption binding path")
        path = Path(raw)
        if path not in known[name].inputs:
            raise StageError("Source-caption binding must be an explicit stage input")
        bindings[name] = path
    for stage in stages:
        modules = {
            stage.command[index + 1]
            for index, argument in enumerate(stage.command[:-1])
            if argument == "-m"
        }
        if modules.intersection(SOURCE_CAPTION_CONSUMERS) and stage.name not in bindings:
            raise StageError("Render consumer lacks a source-caption binding")
    return bindings


def parse_stages(value: object) -> tuple[Stage, ...]:
    """Parse trusted execution configuration, not downloaded metadata or prompts."""
    return _parse_workflow(value).stages


def _resolved_paths(stages: tuple[Stage, ...]) -> dict[Path, Path]:
    paths = dict.fromkeys(path for stage in stages for path in (*stage.inputs, *stage.outputs))
    return {path: path.resolve() for path in paths}


def _dependency_graph(
    stages: tuple[Stage, ...], resolved: dict[Path, Path]
) -> dict[str, tuple[str, ...]]:
    if not stages or len({stage.name for stage in stages}) != len(stages):
        raise StageError("Workflow stage names must be unique and nonempty")
    producers: dict[Path, str] = {}
    for stage in stages:
        for output in stage.outputs:
            path = resolved[output]
            if path in producers:
                raise StageError("Workflow outputs must have exactly one producer")
            producers[path] = stage.name
    artifacts = set(resolved.values())
    if any(parent in artifacts for path in artifacts for parent in path.parents):
        raise StageError("Workflow file paths overlap as ancestor and descendant")
    for output in (path for stage in stages for path in stage.outputs):
        if output.is_symlink():
            raise StageError("Workflow output must not be a symlink")
        if output.exists():
            metadata = output.stat()
            if not stat.S_ISREG(metadata.st_mode) or metadata.st_nlink != 1:
                raise StageError("Workflow output must be a single-link regular file")
    graph = {
        stage.name: tuple(
            dict.fromkeys(
                producers[resolved[path]] for path in stage.inputs if resolved[path] in producers
            )
        )
        for stage in stages
    }
    try:
        TopologicalSorter(graph).prepare()
    except CycleError as error:
        raise StageError("Workflow dependencies contain a cycle") from error
    return graph


def stage_graph(stages: tuple[Stage, ...]) -> dict[str, tuple[str, ...]]:
    """Infer file dependencies; reject ambiguous ownership and cycles before execution."""
    return _dependency_graph(stages, _resolved_paths(stages))


def _load_workflow(plan: Path) -> _Workflow:
    metadata = plan.lstat()
    if not stat.S_ISREG(metadata.st_mode) or metadata.st_size > MAX_PLAN_BYTES:
        raise StageError("Invalid workflow manifest file")
    with plan.open("rb") as stream:
        data = stream.read(MAX_PLAN_BYTES + 1)
    if len(data) > MAX_PLAN_BYTES:
        raise StageError("Invalid workflow manifest file")
    raw: object = json.loads(data.decode("utf-8"))
    return _parse_workflow(raw)


def load_stages(plan: Path) -> tuple[Stage, ...]:
    """Read a bounded trusted manifest without running commands or writing state."""
    return _load_workflow(plan).stages


def _selected_graph(
    graph: dict[str, tuple[str, ...]], targets: tuple[str, ...]
) -> dict[str, tuple[str, ...]]:
    if set(targets).difference(graph):
        raise StageError("Unknown workflow target")
    pending = list(targets or graph)
    selected: set[str] = set()
    while pending:
        name = pending.pop()
        if name not in selected:
            selected.add(name)
            pending.extend(graph[name])
    return {name: parents for name, parents in graph.items() if name in selected}


def _check_layout(workflow: _Workflow, *, plan: Path, state: Path) -> None:
    root = state.resolve()
    manifest = plan.resolve()
    artifacts = set(workflow.paths.values()) | {manifest}
    if any(path.is_relative_to(root) or root.is_relative_to(path) for path in artifacts):
        raise StageError("Workflow artifacts and manifest must be separate from state")
    outputs = {workflow.paths[path] for stage in workflow.stages for path in stage.outputs}
    if manifest in outputs:
        raise StageError("Workflow must not overwrite its manifest")


def _check_external_inputs(stages: tuple[Stage, ...], paths: dict[Path, Path]) -> None:
    produced = {paths[path] for stage in stages for path in stage.outputs}
    sources = {path for stage in stages for path in stage.inputs if paths[path] not in produced}
    for source in sources:
        try:
            metadata = source.lstat()
        except FileNotFoundError as error:
            raise StageError("Workflow external input is missing") from error
        if not stat.S_ISREG(metadata.st_mode) or metadata.st_size == 0:
            raise StageError("Workflow external input must be a nonempty regular file")


def _run_selected(
    stages: tuple[Stage, ...],
    *,
    graph: dict[str, tuple[str, ...]],
    state: Path,
    source_caption_bindings: dict[str, Path],
) -> None:
    priority = {stage.name: index for index, stage in enumerate(stages)}
    sorter = TopologicalSorter(graph)
    sorter.prepare()
    ready: list[int] = []
    while sorter.is_active():
        for name in sorter.get_ready():
            heapq.heappush(ready, priority[name])
        stage = stages[heapq.heappop(ready)]
        if stage.name in source_caption_bindings:
            from replay.source_caption_binding import validate_binding

            try:
                validate_binding(
                    source_caption_bindings[stage.name],
                    consumer_inputs=stage.inputs,
                    consumer_command=stage.command,
                )
            except ReplayError as error:
                raise StageError("Source-caption repair binding failed") from error
        result = run_stage(stage, state_directory=state)
        print(f"{stage.name}: {result}", flush=True)
        if result == "busy":
            raise StageError("Another process owns this stage; no duplicate was started")
        sorter.done(stage.name)


def run_plan(plan: Path, *, state_directory: Path, targets: tuple[str, ...] = ()) -> None:
    workflow = _load_workflow(plan)
    graph = _selected_graph(workflow.dependencies, targets)
    stages = tuple(stage for stage in workflow.stages if stage.name in graph)
    _check_layout(workflow, plan=plan, state=state_directory)
    _check_external_inputs(stages, workflow.paths)
    with execution_lock(state_directory, ".workflow") as acquired:
        if not acquired:
            raise StageError("Another process owns this workflow state; no duplicate was started")
        _run_selected(
            stages,
            graph=graph,
            state=state_directory,
            source_caption_bindings=workflow.source_caption_bindings,
        )


def _cancel(signum: int, frame: FrameType | None) -> None:
    """Unwind the stage runner so its owned process group is stopped, not orphaned."""
    raise StageError("Workflow cancelled; partial outputs remain unapproved")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("plan", type=Path, help="Trusted local JSON command manifest")
    parser.add_argument("--state", required=True, type=Path, help="Private checkpoint directory")
    parser.add_argument("--check", action="store_true", help="Print dependency graph; run nothing")
    parser.add_argument(
        "--target",
        action="append",
        default=[],
        help="Run this stage and all ancestors (repeatable)",
    )
    arguments = Arguments()
    parser.parse_args(argv, namespace=arguments)
    previous_handler = signal.signal(signal.SIGTERM, _cancel)
    try:
        if arguments.check:
            workflow = _load_workflow(arguments.plan)
            _check_layout(workflow, plan=arguments.plan, state=arguments.state)
            print(
                json.dumps(
                    _selected_graph(workflow.dependencies, tuple(arguments.target)), sort_keys=True
                )
            )
        else:
            run_plan(
                arguments.plan, state_directory=arguments.state, targets=tuple(arguments.target)
            )
    except subprocess.TimeoutExpired:
        print("Workflow deadline exceeded; inspect private stage logs.", file=sys.stderr)
        return 1
    except (StageError, OSError, ValueError) as error:
        print(
            f"Workflow stopped ({type(error).__name__}); inspect its private state.",
            file=sys.stderr,
        )
        return 1
    finally:
        signal.signal(signal.SIGTERM, previous_handler)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
