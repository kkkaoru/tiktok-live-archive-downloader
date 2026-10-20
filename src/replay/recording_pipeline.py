"""Run one recording's pipeline from download to the captioned candidate movie, idempotently.

``start REPLAY_ID --defaults DEFAULTS.json`` binds the recording (``edit_completion init`` when
its edit root is new), writes ``<root>/replay-<id>/pipeline.json`` and launches ``run``
detached through ``scripts/tmux-job.sh``. ``run`` executes the step graph with
``replay.workflow_parallel`` so the video and audio branches overlap and GPU steps share the GPU
while its measured utilization leaves room:

    clocks ─┬─ cleanup-bootstrap ── cleanup ─────────────┐
            ├─ separation (GPU) ── asr (GPU) ─────────────┴─ retire-intermediates
            └─ yolo-track (GPU) ─────────────────────────────┐
    asr ── cut-plan ── paging ── placement (+ yolo-track) ── caption-parts ── caption-render
        ── compose (+ cleanup) ── finish ── publish ── upload (web viewer sync)

The caption half (``replay.recording_caption_steps``) reads ``captions.json``, written beside
``pipeline.json`` from the defaults' ``captions`` section; changing it marks only caption steps
stale.

Idempotent and resumable:

- a step whose receipt matches its current inputs (configuration and upstream receipts) is
  skipped; a receipt that predates its inputs stops the run instead of being trusted;
- an interrupted step runs again and reuses everything it finished (``replay.recording_steps``);
- ``--force STEP`` (repeatable) redoes that step and every step downstream of it;
  ``--force-all`` redoes the whole recording. Forced outputs are deleted first.

The candidate is not a delivery: the completion gate assesses it separately.
"""

import argparse
import json
import os
import signal
import subprocess
import sys
import time
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from types import FrameType

from replay import gpu_lease
from replay import recording_caption_steps as captions
from replay import recording_steps as steps
from replay import source_lead_in as lead_in
from replay.core import ReplayError
from replay.edit_completion import bind_source, bind_storage, load_config
from replay.recordings import record
from replay.stages import Stage, StageError, execution_lock
from replay.workflow_parallel import Children, Result, StageSlots, run_selected

DEFAULT_KEYS = frozenset(
    {
        "version",
        "root",
        "python",
        "providerPython",
        "native",
        "ffmpeg",
        "ffprobe",
        "references",
        "yolo",
        "maxParallel",
        "gpuSlots",
        "captions",
        "deliveryRoot",
    }
)
HOUR = 3600
GPU = "gpu"
LAUNCHER = Path("scripts/tmux-job.sh")
MAX_LAUNCHES = 100
# Start order among ready steps: the longest branch first.
ORDER = (
    "clocks", "separation", "cleanup-bootstrap", "cleanup", "asr", "yolo-track", "cut-plan",
    "paging", "placement", "caption-parts", "caption-render", "compose", "finish", "publish",
    "upload",
)  # fmt: skip
DEPENDENCIES = {**steps.DEPENDENCIES, **captions.DEPENDENCIES}
TIMEOUTS = {
    "clocks": 2 * HOUR,
    "separation": 4 * HOUR,
    "cleanup-bootstrap": 2 * HOUR,
    "cleanup": 6 * HOUR,
    "asr": 4 * HOUR,
    "yolo-track": 2 * HOUR,
    "retire-intermediates": HOUR,
    "cut-plan": HOUR,
    "paging": 2 * HOUR,
    "placement": HOUR,
    "caption-parts": 2 * HOUR,
    "caption-render": 12 * HOUR,
    "compose": 6 * HOUR,
    "finish": 2 * HOUR,
    "publish": 2 * HOUR,
    "upload": 6 * HOUR,
}
GPU_STEPS = frozenset({"separation", "asr", "yolo-track", "caption-render", "compose"})


@dataclass(frozen=True, slots=True)
class Layout:
    """Where one recording's pipeline lives."""

    directory: Path

    @property
    def pipeline(self) -> Path:
        return self.directory / "pipeline.json"

    @property
    def steps(self) -> Path:
        return self.directory / "steps"


@dataclass(frozen=True, slots=True)
class Forcing:
    """Steps to redo; ``every`` redoes all of them."""

    names: tuple[str, ...] = ()
    every: bool = False

    def arguments(self) -> list[str]:
        named = [argument for name in self.names for argument in ("--force", name)]
        return [*named, "--force-all"] if self.every else named


NO_FORCING = Forcing()


def load_defaults(path: Path) -> dict[str, object]:
    data = record(json.loads(path.read_text(encoding="utf-8")))
    if set(data) != DEFAULT_KEYS or data["version"] != 1:
        raise ReplayError("Unsupported recording pipeline defaults")
    references, yolo = record(data["references"]), record(data["yolo"])
    if set(references) != steps.REFERENCES or set(yolo) != steps.YOLO_KEYS:
        raise ReplayError("Incomplete pipeline references or detector settings")
    if set(record(data["captions"])) != captions.KEYS - {"version"}:
        raise ReplayError("Incomplete caption pipeline defaults")
    for key in ("maxParallel", "gpuSlots"):
        if type(data[key]) is not int or not 1 <= int(str(data[key])) <= 16:
            raise ReplayError(f"{key} must be an integer from 1 to 16")
    return data


def edit_source(edit_root: Path) -> Path:
    config = record(json.loads((edit_root / "edit-config.json").read_text(encoding="utf-8")))
    source = Path(str(config.get("source")))
    if not source.is_absolute() or not source.is_file():
        raise ReplayError("The edit configuration names no existing source")
    return source


def zero_origin_source(defaults: dict[str, object], edit_root: Path) -> Path:
    """The edit source; one whose video starts late is first rebound to its lead-in copy.

    The copy lives beside the delivery storage (``<deliveryRoot>/source-lead-in/<recording>``),
    so the storage root never contains the source it was validated against.
    """
    config = load_config(edit_root)
    if config.storage_root is None:
        raise ReplayError("The recording has no bound storage for a lead-in copy")
    directory = config.storage_root.parent / lead_in.DIRECTORY / edit_root.name
    derived = lead_in.prepared(
        config.source, directory,
        ffmpeg=str(defaults["ffmpeg"]), ffprobe=str(defaults["ffprobe"]),
    )  # fmt: skip
    if derived != config.source:
        bind_source(edit_root, derived)
    return derived


def bind(replay_id: str, python: str, checkout: Path) -> Path:
    """The recording's edit root, initialized from its completed download job when new."""
    if not replay_id.isdigit():
        raise ReplayError("A replay ID is decimal digits")
    edit_root = checkout / "edits" / f"replay-{replay_id}"
    if not (edit_root / "edit-config.json").exists():
        subprocess.run(
            [python, "-B", "-m", "replay.edit_completion", "init", str(edit_root),
             "--engine", str(checkout / "src/replay"), "--executor-scope", str(checkout)],
            check=True,
        )  # fmt: skip
    return edit_root


def pipeline_config(
    defaults: dict[str, object], layout: Layout, replay_id: str, edit_root: Path
) -> dict[str, object]:
    return {
        "version": 1,
        "recording": replay_id,
        "editRoot": str(edit_root),
        "source": str(zero_origin_source(defaults, edit_root)),
        "work": str(layout.directory / f"replay-{replay_id}"),
        "steps": str(layout.steps),
        "python": defaults["python"],
        "providerPython": defaults["providerPython"],
        "native": defaults["native"],
        "ffmpeg": defaults["ffmpeg"],
        "ffprobe": defaults["ffprobe"],
        "references": defaults["references"],
        "yolo": defaults["yolo"],
    }


def write_once(path: Path, value: object) -> None:
    """Write a new file, or accept an identical existing one (an idempotent re-prepare)."""
    text = json.dumps(value, indent=2, allow_nan=False) + "\n"
    if path.exists():
        if path.read_text(encoding="utf-8") != text:
            raise ReplayError(f"{path.name} exists with different content; force a new setup")
        return
    with path.open("x", encoding="utf-8") as stream:
        os.fchmod(stream.fileno(), 0o600)
        stream.write(text)


def layout_of(replay_id: str, defaults: dict[str, object], checkout: Path) -> Layout:
    layout = Layout(Path(str(defaults["root"])) / f"replay-{replay_id}")
    if not layout.directory.resolve().is_relative_to(checkout / "edits/work"):
        raise ReplayError("The pipeline root must live under the checkout's edits/work")
    return layout


def bind_delivery_storage(edit_root: Path, deliveries: Path) -> None:
    """Give the recording its external delivery storage (``<deliveryRoot>/<recording>``) once."""
    if load_config(edit_root).storage_root is not None:
        return
    storage = deliveries / edit_root.name
    storage.mkdir(mode=0o700, parents=True, exist_ok=True)
    bind_storage(edit_root, storage)


def prepare(replay_id: str, defaults_path: Path, *, checkout: Path) -> Layout:
    defaults = load_defaults(defaults_path)
    edit_root = bind(replay_id, str(defaults["python"]), checkout)
    bind_delivery_storage(edit_root, Path(str(defaults["deliveryRoot"])))
    layout = layout_of(replay_id, defaults, checkout)
    layout.steps.mkdir(mode=0o700, parents=True, exist_ok=True)
    write_once(layout.pipeline, pipeline_config(defaults, layout, replay_id, edit_root))
    write_once(layout.directory / "captions.json", {"version": 1, **record(defaults["captions"])})
    return layout


def graph_of(name: str) -> steps.Graph:
    return captions.BACK if name in captions.STEPS else steps.FRONT


def downstream(names: Sequence[str]) -> list[str]:
    """The named steps and every step that depends on them, in graph order."""
    chosen = set(names)
    unknown = chosen - set(DEPENDENCIES)
    if unknown:
        raise ReplayError(f"Unknown steps: {', '.join(sorted(unknown))}")
    changed = True
    while changed:
        before = len(chosen)
        chosen |= {name for name, after in DEPENDENCIES.items() if chosen & set(after)}
        changed = len(chosen) != before
    return [name for name in DEPENDENCIES if name in chosen]


def apply_forcing(pipeline: steps.Pipeline, forcing: Forcing) -> list[dict[str, object]]:
    names = list(DEPENDENCIES) if forcing.every else downstream(forcing.names)
    removed: list[dict[str, object]] = []
    for name in reversed(names):  # dependents first, so no receipt outlives its inputs
        removed.extend(steps.reset(pipeline, name, graph_of(name)))
    return removed


def stage_of(layout: Layout, python: str, name: str) -> Stage:
    receipts = [layout.steps / f"{parent}.json" for parent in DEPENDENCIES[name]]
    owner = "recording_caption_steps" if name in captions.STEPS else "recording_steps"
    settings = (layout.directory / "captions.json",) if name in captions.STEPS else ()
    return Stage(
        name=name,
        command=(python, "-B", "-m", f"replay.{owner}", name, str(layout.pipeline)),
        inputs=(layout.pipeline, *settings, *receipts),
        outputs=(layout.steps / f"{name}.json",),
        timeout_seconds=TIMEOUTS[name],
    )


def run(
    replay_id: str,
    defaults_path: Path,
    *,
    checkout: Path,
    forcing: Forcing = NO_FORCING,
    children: Children | None = None,
) -> dict[str, Result]:
    """Run every unfinished step of the recording; return each step's result."""
    defaults = load_defaults(defaults_path)
    layout = layout_of(replay_id, defaults, checkout)
    pipeline = steps.load(layout.pipeline)
    owned = children or Children()
    # Every stage inherits the start time: the GPU leases serve recordings in start order.
    os.environ.setdefault(gpu_lease.STARTED_VARIABLE, str(time.time_ns()))
    with execution_lock(layout.directory, "pipeline") as acquired:
        if not acquired:
            raise ReplayError("Another run owns this recording's pipeline")
        for item in apply_forcing(pipeline, forcing):
            print(f"forced: removed {item['path']} ({item['bytes']} bytes)", flush=True)
        names = (*ORDER, "retire-intermediates")
        python = str(defaults["python"])
        stages = tuple(stage_of(layout, python, name) for name in names)

        def execute(stage: Stage, _state: Path) -> Result:
            if steps.finished(pipeline, stage.name, graph_of(stage.name)):
                return "reused"
            log = layout.steps / "logs" / f"{stage.name}-{time.time_ns()}.run.log"
            if stage.name not in GPU_STEPS:
                owned.execute(stage.command, log, stage.timeout_seconds)
                return "completed"
            # The pipeline's own GPU slot is held already; then one machine-wide lease.
            with gpu_lease.lease(gpu_lease.root(), replay_id):
                owned.execute(stage.command, log, stage.timeout_seconds)
            return "completed"

        layout.steps.joinpath("logs").mkdir(mode=0o700, exist_ok=True)
        started = time.time()
        results = run_selected(
            stages,
            graph={name: DEPENDENCIES[name] for name in names},
            state=layout.steps,
            bindings={},
            max_parallel=int(str(defaults["maxParallel"])),
            resources=StageSlots(
                {GPU: int(str(defaults["gpuSlots"]))}, dict.fromkeys(GPU_STEPS, GPU)
            ),
            run=execute,
        )
        print(f"pipeline wall seconds: {time.time() - started:.1f}", flush=True)
        return results


def launch_name(layout: Layout) -> str:
    for number in range(1, MAX_LAUNCHES + 1):
        name = f"pipeline-v{number}"
        if not (layout.directory / f"{name}.launch").exists():
            return name
    raise ReplayError("Too many pipeline launches; inspect the earlier ones")


def start(
    replay_id: str, defaults_path: Path, *, checkout: Path, forcing: Forcing = NO_FORCING
) -> str:
    """Prepare the pipeline and launch ``run`` detached; return the launch name."""
    layout = prepare(replay_id, defaults_path, checkout=checkout)
    python = str(load_defaults(defaults_path)["python"])
    name = launch_name(layout)
    command = [
        str(checkout / LAUNCHER), str(layout.directory), name, python, "-B", "-m",
        "replay.recording_pipeline", "run", replay_id, "--defaults", str(defaults_path.resolve()),
        *forcing.arguments(),
    ]  # fmt: skip
    subprocess.run(command, check=True, cwd=checkout)
    return name


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "run", "start"))
    parser.add_argument("replay_id")
    parser.add_argument("--defaults", type=Path, required=True)
    parser.add_argument("--force", action="append", default=[], choices=tuple(DEPENDENCIES))
    parser.add_argument("--force-all", action="store_true")
    args = parser.parse_args(argv)
    checkout = Path.cwd().resolve()
    forcing = Forcing(tuple(args.force), args.force_all)
    children = Children()

    def stop(_signum: int, _frame: FrameType | None) -> None:
        children.stop_all()
        raise StageError("Pipeline stopped; finished work is reused on the next run")

    previous = signal.signal(signal.SIGTERM, stop)
    try:
        if args.action == "prepare":
            print(prepare(args.replay_id, args.defaults, checkout=checkout).directory)
        elif args.action == "run":
            prepare(args.replay_id, args.defaults, checkout=checkout)
            run(
                args.replay_id, args.defaults, checkout=checkout, forcing=forcing, children=children
            )
        else:
            print(start(args.replay_id, args.defaults, checkout=checkout, forcing=forcing))
    except (ReplayError, StageError, OSError, ValueError, subprocess.CalledProcessError) as error:
        print(f"Pipeline stopped ({type(error).__name__}): {error}", file=sys.stderr)
        return 1
    finally:
        signal.signal(signal.SIGTERM, previous)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
