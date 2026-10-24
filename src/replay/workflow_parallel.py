"""Run a workflow manifest with every ready, independent stage at the same time.

``replay.workflow`` runs one ready stage at a time, so the audio branch (separation, ASR,
paging) waited for the video branch (caption removal, tracking) although neither reads the
other's outputs. ``replay.workflow`` is hash-pinned by other modules and stays unchanged; this
runner reuses its manifest parsing, dependency graph, layout and input checks and
``stages.run_stage`` (per-stage lock, receipts, verified reuse, no blind retry), and adds:

- several stages at once, up to ``max_parallel``;
- resource slots: a side file names a resource per stage (for example ``gpu``) and the most
  stages of that resource that may run together. The first stage of a resource starts when
  ready; for ``gpu`` each further one starts only while the measured GPU utilization (IOKit
  ``Device Utilization %``) stays below ``GPU_LIMIT`` and at least ``SETTLE_SECONDS`` after the
  previous GPU start, so the last stage's model load shows up in the measurement first;
- measured admission: ``always`` stages (default 2, so the audio branch is not starved while
  caption removal saturates the CPU) start whenever they are ready; each further stage starts
  only while the 1-minute load stays below ``LOAD_LIMIT`` x cores and the kernel's free-memory
  level (``kern.memorystatus_level``) stays at or above ``MEMORY_FREE_MIN`` percent and the
  machine pages in slower than ``memory_admission.SWAP_IN_LIMIT`` per minute. Swap use itself is
  not a signal: on this Mac resident apps keep it above 90% while memory is plentiful;
- the first failed stage stops admission; running stages finish and the run fails;
- SIGTERM asks every running stage's process group to stop (SIGTERM, so a stage can stop the
  process groups it started itself), kills what remains after ``STOP_GRACE_SECONDS``, and the
  run fails.

The order among ready stages is the manifest order, as in ``replay.workflow``.
"""

import argparse
import heapq
import json
import os
import re
import signal
import subprocess
import sys
import threading
import time
from collections.abc import Callable, Sequence
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from dataclasses import dataclass, field
from graphlib import TopologicalSorter
from pathlib import Path
from types import FrameType
from typing import Literal

from replay import memory_admission
from replay.core import ReplayError
from replay.stages import Stage, StageError, execution_lock, run_stage
from replay.workflow import (
    _check_external_inputs,
    _check_layout,
    _load_workflow,
    _selected_graph,
)

MAX_PARALLEL = 16
ALWAYS = 2
LOAD_LIMIT = 0.85
MEMORY_FREE_MIN = 20
POLL_SECONDS = 20.0
MAX_RESOURCES_BYTES = 65_536
GPU_LIMIT = 75
SETTLE_SECONDS = 60.0
STOP_GRACE_SECONDS = 20.0
STOP_POLL_SECONDS = 0.2
MEMORY_LEVEL = ("/usr/sbin/sysctl", "-n", "kern.memorystatus_level")
GPU_STATISTICS = ("/usr/sbin/ioreg", "-r", "-d", "1", "-w", "0", "-c", "IOAccelerator")
GPU_UTILIZATION = re.compile(r'"Device Utilization %"=(\d+)')
GPU = "gpu"

Result = Literal["completed", "reused", "adopted", "busy"]
RunStage = Callable[[Stage, Path], Result]


@dataclass(frozen=True, slots=True)
class Load:
    """Cores, 1-minute load average, the kernel's free-memory, the GPU busy percentage and the
    machine-wide page-in rate (bytes per minute)."""

    cpus: int
    load: float
    memory_free_percent: int
    gpu_percent: int
    swap_in_per_minute: float = 0.0


@dataclass(frozen=True)
class StageSlots:
    """Per-stage resource names and how many stages of each resource may run together."""

    slots: dict[str, int] = field(default_factory=dict)
    stages: dict[str, str] = field(default_factory=dict)


def load_resources(path: Path | None, names: Sequence[str]) -> StageSlots:
    """Validate the optional side file ``{"slots": {res: n}, "stages": {stage: res}}``."""
    if path is None:
        return StageSlots()
    data = path.read_bytes()
    if len(data) > MAX_RESOURCES_BYTES:
        raise StageError("Resource file is too large")
    raw: object = json.loads(data.decode("utf-8"))
    if not isinstance(raw, dict) or set(raw) != {"slots", "stages"}:
        raise StageError("Resource file needs exactly slots and stages")
    slots, stages = raw["slots"], raw["stages"]
    if not isinstance(slots, dict) or not isinstance(stages, dict):
        raise StageError("Resource slots and stages must be objects")
    for name, count in slots.items():
        if not isinstance(name, str) or type(count) is not int or not 1 <= count <= MAX_PARALLEL:
            raise StageError("Resource slots must be counts from 1 to 16")
    for stage, resource in stages.items():
        if stage not in names or resource not in slots:
            raise StageError("Resource stages must name known stages and resources")
    return StageSlots(
        {str(name): int(count) for name, count in slots.items()},
        {str(stage): str(resource) for stage, resource in stages.items()},
    )


class Children:
    """Process groups of running stage commands, so a cancelled run stops them all."""

    def __init__(self) -> None:
        self._groups: set[int] = set()
        self._lock = threading.Lock()

    def execute(self, command: tuple[str, ...], log: Path, timeout: int) -> None:
        """``stages.execute_command`` with the child's process group registered while it runs."""
        with log.open("xb") as stream:
            os.fchmod(stream.fileno(), 0o600)
            with subprocess.Popen(
                command,
                stdin=subprocess.DEVNULL,
                stdout=stream,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            ) as child:
                with self._lock:
                    self._groups.add(child.pid)
                try:
                    code = child.wait(timeout=timeout)
                finally:
                    with self._lock:
                        self._groups.discard(child.pid)
                    if child.poll() is None:
                        _kill(child.pid)
                        child.wait()
                if code != 0:
                    raise StageError(f"Stage command failed; inspect {log}")

    def stop_all(self, grace_seconds: float = STOP_GRACE_SECONDS) -> None:
        """SIGTERM every running group, then SIGKILL the groups still alive after the grace."""
        with self._lock:
            groups = tuple(self._groups)
        for group in groups:
            _signal(group, signal.SIGTERM)
        deadline = time.monotonic() + grace_seconds
        while any(_alive(group) for group in groups) and time.monotonic() < deadline:
            time.sleep(STOP_POLL_SECONDS)
        for group in groups:
            _signal(group, signal.SIGKILL)


def _signal(group: int, number: signal.Signals) -> None:
    try:
        os.killpg(group, number)
    except (ProcessLookupError, PermissionError):
        return  # the group exited (macOS answers EPERM while only zombies remain)


def _alive(group: int) -> bool:
    try:
        os.killpg(group, 0)
    except (ProcessLookupError, PermissionError):
        return False
    return True


def _kill(group: int) -> None:
    _signal(group, signal.SIGKILL)


def read_load() -> Load:
    """Measured load; unreadable memory counts as none free, an unreadable GPU as fully busy."""
    level = subprocess.run(MEMORY_LEVEL, capture_output=True, text=True, check=False).stdout
    free = int(level) if level.strip().isdigit() else 0
    statistics = subprocess.run(GPU_STATISTICS, capture_output=True, text=True, check=False)
    busy = [int(found) for found in GPU_UTILIZATION.findall(statistics.stdout)]
    return Load(
        os.cpu_count() or 1,
        os.getloadavg()[0],
        free,
        max(busy, default=100),
        memory_admission.swap_in_rate(),
    )


def headroom(load: Load) -> bool:
    """True while measured CPU load, free memory and paging leave room for one more stage."""
    return (
        load.load < LOAD_LIMIT * load.cpus
        and load.memory_free_percent >= MEMORY_FREE_MIN
        and load.swap_in_per_minute < memory_admission.SWAP_IN_LIMIT
    )


def _validated(stage: Stage, bindings: dict[str, Path]) -> None:
    if stage.name in bindings:
        from replay.source_caption_binding import validate_binding

        try:
            validate_binding(
                bindings[stage.name], consumer_inputs=stage.inputs, consumer_command=stage.command
            )
        except ReplayError as error:
            raise StageError("Source-caption repair binding failed") from error


def run_selected(
    stages: tuple[Stage, ...],
    *,
    graph: dict[str, tuple[str, ...]],
    state: Path,
    bindings: dict[str, Path],
    max_parallel: int,
    resources: StageSlots,
    run: RunStage,
    measure: Callable[[], Load] = read_load,
    always: int = ALWAYS,
    poll_seconds: float = POLL_SECONDS,
    clock: Callable[[], float] = time.monotonic,
    settle_seconds: float = SETTLE_SECONDS,
) -> dict[str, Result]:
    """Run the selected stages concurrently; return each stage's result."""
    if not 1 <= max_parallel <= MAX_PARALLEL or not 1 <= always <= max_parallel:
        raise StageError("Invalid parallel stage count")
    priority = {stage.name: index for index, stage in enumerate(stages)}
    sorter = TopologicalSorter(graph)
    sorter.prepare()
    ready: list[int] = []
    running: dict[Future[Result], str] = {}
    results: dict[str, Result] = {}
    failures: list[str] = []
    printing = threading.Lock()
    last_gpu_start = [float("-inf")]

    def in_use(resource: str) -> int:
        return sum(1 for name in running.values() if resources.stages.get(name) == resource)

    def admitted(resource: str | None, current: Load) -> bool:
        if resource is None:
            return True
        used = in_use(resource)
        if used >= resources.slots[resource]:
            return False
        if resource != GPU or used == 0:
            return True
        settled = clock() - last_gpu_start[0] >= settle_seconds
        return settled and current.gpu_percent < GPU_LIMIT

    def startable(current: Load) -> Stage | None:
        for index in sorted(ready):
            stage = stages[index]
            resource = resources.stages.get(stage.name)
            if admitted(resource, current):
                ready.remove(index)
                heapq.heapify(ready)
                if resource == GPU:
                    last_gpu_start[0] = clock()
                return stage
        return None

    with ThreadPoolExecutor(max_workers=max_parallel) as pool:
        while sorter.is_active() and not failures:
            for name in sorter.get_ready():
                heapq.heappush(ready, priority[name])
            while len(running) < max_parallel and ready and not failures:
                current = measure()
                if len(running) >= always and not headroom(current):
                    break
                stage = startable(current)
                if stage is None:
                    break
                _validated(stage, bindings)
                running[pool.submit(run, stage, state)] = stage.name
            if not running:
                raise StageError("No runnable stage although the workflow is unfinished")
            done, _ = wait(tuple(running), timeout=poll_seconds, return_when=FIRST_COMPLETED)
            for future in done:
                name = running.pop(future)
                try:
                    result = future.result()
                except (StageError, ReplayError, OSError, ValueError) as error:
                    failures.append(f"{name}: {type(error).__name__}: {error}")
                    continue
                with printing:
                    print(f"{name}: {result}", flush=True)
                if result == "busy":
                    failures.append(f"{name}: another process owns this stage")
                    continue
                results[name] = result
                sorter.done(name)
        for future in wait(tuple(running)).done:
            name = running.pop(future)
            try:
                results[name] = future.result()
                print(f"{name}: {results[name]}", flush=True)
            except (StageError, ReplayError, OSError, ValueError) as error:
                failures.append(f"{name}: {type(error).__name__}: {error}")
    if failures:
        raise StageError("Workflow stages failed: " + "; ".join(failures))
    return results


def run_plan(
    plan: Path,
    *,
    state_directory: Path,
    targets: tuple[str, ...] = (),
    max_parallel: int = 4,
    resources: Path | None = None,
    children: Children | None = None,
) -> dict[str, Result]:
    workflow = _load_workflow(plan)
    graph = _selected_graph(workflow.dependencies, targets)
    stages = tuple(stage for stage in workflow.stages if stage.name in graph)
    _check_layout(workflow, plan=plan, state=state_directory)
    _check_external_inputs(stages, workflow.paths)
    slots = load_resources(resources, [stage.name for stage in workflow.stages])
    owned = children or Children()

    def run(stage: Stage, state: Path) -> Result:
        return run_stage(stage, state_directory=state, execute=owned.execute)

    with execution_lock(state_directory, ".workflow") as acquired:
        if not acquired:
            raise StageError("Another process owns this workflow state; no duplicate was started")
        started = time.time()
        results = run_selected(
            stages,
            graph=graph,
            state=state_directory,
            bindings=workflow.source_caption_bindings,
            max_parallel=max_parallel,
            resources=slots,
            run=run,
        )
        print(f"workflow wall seconds: {time.time() - started:.1f}", flush=True)
        return results


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("plan", type=Path, help="Trusted local JSON command manifest")
    parser.add_argument("--state", required=True, type=Path, help="Private checkpoint directory")
    parser.add_argument("--target", action="append", default=[], help="Stage and its ancestors")
    parser.add_argument("--max-parallel", type=int, default=4)
    parser.add_argument("--resources", type=Path, help="Optional resource slot side file")
    args = parser.parse_args(argv)
    children = Children()

    def cancel(_signum: int, _frame: FrameType | None) -> None:
        children.stop_all()
        raise StageError("Workflow cancelled; partial outputs remain unapproved")

    previous = signal.signal(signal.SIGTERM, cancel)
    try:
        run_plan(
            args.plan,
            state_directory=args.state,
            targets=tuple(args.target),
            max_parallel=args.max_parallel,
            resources=args.resources,
            children=children,
        )
    except (StageError, OSError, ValueError) as error:
        print(f"Workflow stopped ({type(error).__name__}): {error}", file=sys.stderr)
        return 1
    finally:
        signal.signal(signal.SIGTERM, previous)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
