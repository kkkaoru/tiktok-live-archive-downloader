"""Run the render half of a delivery end to end for one recording.

Each step of `replay.delivery_render` generates its workflows or Compressor job
list once; a rerun finds the generated files and the stage runner reuses
completed stages, so an interrupted pipeline resumes where it stopped. A
step's workflows run in parallel; the pipeline stops at the first failure.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

from replay import (
    delivery_captions,
    delivery_render,
    delivery_render_resume,
    disk_budget,
    mechanical_join,
    mechanical_lane,
    mechanical_plan_provenance,
    source_caption_binding,
    stage_recovery,
    worker_budget,
)
from replay.core import ReplayError
from replay.recordings import record
from replay.stages import StageError
from replay.workflow import parse_stages

WORKFLOW_TIMEOUT_SECONDS = 6 * 3600

Launcher = Callable[[Sequence[Sequence[str]]], list[int]]


@dataclass(frozen=True, slots=True)
class Step:
    """A step's generated files: `pattern` under the work (or captions) directory."""

    name: str
    pattern: str
    compressor: bool = False
    workers: int = 1
    captions: bool = False


STEPS = (
    Step("combine", "combine-workflow.json", captions=True),
    Step("motion-export", "motion/export-*-*.json", workers=4),
    Step("motion-prepare", "motion/prepare-*-*.json", workers=4),
    Step("alpha-jobs", "alpha-jobs.json", compressor=True),
    Step("compose", "compose/partition-*-*.json", workers=4),
    Step("render", "compose/render-*-*.json", workers=3),
    Step("join", "join-workflow.json"),
    Step("audio", "audio-workflow.json"),
    Step("final-mux", "final-mux-workflow.json"),
    Step("final-job", "final-job.json", compressor=True),
)
STEP_NAMES = (*(step.name for step in STEPS), "publish")


def _launch(commands: Sequence[Sequence[str]]) -> list[int]:
    processes = [subprocess.Popen(list(command), stdin=subprocess.DEVNULL) for command in commands]
    return [process.wait(timeout=WORKFLOW_TIMEOUT_SECONDS) for process in processes]


def _files(config: delivery_render.Config, step: Step) -> list[Path]:
    if step.name in {"compose", "render"}:
        return sorted(config.composition.glob(step.pattern.removeprefix("compose/")))
    root = config.captions if step.captions else config.work
    return sorted(root.glob(step.pattern))


def _generate(config_path: Path, config: delivery_render.Config, step: Step) -> None:
    if step.captions:
        delivery_captions.write_combine(
            config.captions,
            width=config.width,
            height=config.height,
            calibration=config.calibration,
        )
        return
    if step.name == "join":
        rendered = delivery_render.rendered_parts(config)
        if mechanical_plan_provenance.join_needs_manifest(rendered, config):
            with mechanical_plan_provenance.join_fixes():
                mechanical_join.generate_join(rendered, config)
            return
    workers = (
        worker_budget.workers(step.workers) if step.name in {"compose", "render"} else step.workers
    )
    arguments = [step.name, str(config_path), "--workers", str(workers)]
    if step.name == "compose":
        with mechanical_lane.compose_fixes() as clamps:
            failed = delivery_render.main(arguments)
        mechanical_lane.write_clamps(config.work / mechanical_lane.CLAMPS_NAME, clamps)
    elif step.name == "render":
        with mechanical_lane.render_fixes():
            failed = delivery_render.main(arguments)
    elif step.name == "join":
        with mechanical_plan_provenance.join_fixes():
            failed = delivery_render.main(arguments)
    else:
        failed = delivery_render.main(arguments)
    if failed:
        raise ReplayError(f"Could not generate step {step.name}")


def commands(step: Step, files: Sequence[Path]) -> list[list[str]]:
    """Runner commands; each file keeps its own state directory beside it."""
    module = "replay.compressor_batch" if step.compressor else "replay.workflow"
    return [
        [
            sys.executable,
            "-m",
            module,
            str(path),
            "--state",
            str(path.with_name(f"{path.stem}-state")),
        ]
        for path in files
    ]


def _has_native_wrapper(raw: object) -> bool:
    if not isinstance(raw, dict) or not isinstance(raw.get("stages"), list):
        return False
    return any(
        isinstance(stage, dict)
        and any(
            isinstance(stage.get(key), list) and "replay.native_invocation" in stage[key]
            for key in ("command", "verify_command")
        )
        for stage in raw["stages"]
    )


def _attempt(step: Step, files: Sequence[Path], launch: Launcher) -> list[int]:
    """Keep native bindings in place for the workflow's read-only verifier."""
    if not step.compressor:
        for path in files:
            raw = delivery_render._json(path)
            # Even a malformed/partial wrapper must never enter destructive recovery.
            if _has_native_wrapper(raw):
                continue
            for preserved in stage_recovery.recover(path, path.with_name(f"{path.stem}-state")):
                print(f"PRESERVED {preserved}", flush=True)
    return launch(commands(step, files))


def _production_preflight(config_path: Path, files: Sequence[Path]) -> None:
    """Reused workflows have the same cleanup contract as newly generated ones."""
    cleanup = source_caption_binding.require_cleanup(config_path)
    expected = record(delivery_render._json(cleanup.path))
    for path in files:
        raw = record(delivery_render._json(path))
        declared = record(raw.get("sourceCaptionProduction"))
        bindings = record(declared.get("bindings"))
        stages = parse_stages(raw)
        if not stages or set(bindings) != {stage.name for stage in stages}:
            raise ReplayError("Production workflow requires every cleanup binding")
        for stage in stages:
            binding = delivery_render._path(bindings[stage.name])
            selected = record(delivery_render._json(binding))
            if any(
                selected.get(key) != expected.get(key)
                for key in ("assemblyManifest", "assemblyReceipt", "cleanIndex")
            ):
                raise ReplayError("Production workflow selects a different cleanup source")
            source_caption_binding.validate_binding(
                binding,
                consumer_inputs=stage.inputs,
                consumer_command=stage.command,
            )
        mechanical_plan_provenance.validate_stages(stages)


def run(config_path: Path, *, launch: Launcher = _launch, start: str | None = None) -> list[str]:
    """Run every step from `start` (default: the first); return the step names run."""
    config = delivery_render.load_config(config_path)
    source_caption_binding.require_cleanup(config_path)
    first = STEP_NAMES.index(start) if start is not None else 0
    done: list[str] = []
    for step in STEPS[first:]:
        disk_budget.require_headroom((config.work, config.storage))
        seal = config.composition / "resume.json"
        if step.name in {"render", "join"} and seal.exists():
            delivery_render_resume.verify(seal)
        if step.name in {"render", "join"}:
            partitions = _files(config, Step("compose", "compose/partition-*-*.json"))
            if partitions:
                _production_preflight(config_path, partitions)
        files = _files(config, step)
        if not files:
            _generate(config_path, config, step)
            files = _files(config, step)
        if not files:
            raise ReplayError(f"Step {step.name} generated nothing")
        if step.name in {"compose", "render"}:
            _production_preflight(config_path, files)
        codes = _attempt(step, files, launch)
        if any(code != 0 for code in codes):
            if step.name in {"compose", "render"}:
                _production_preflight(config_path, files)
            codes = _attempt(step, files, launch)
        if any(code != 0 for code in codes):
            raise ReplayError(f"Step {step.name} failed twice: {codes}")
        print(f"STEP {step.name}: completed", flush=True)
        done.append(step.name)
    for path in delivery_render.publish(config):
        print(path, flush=True)
    done.append("publish")
    return done


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path)
    parser.add_argument("--start", choices=STEP_NAMES)
    args = parser.parse_args(argv)
    try:
        run(args.config.absolute(), start=args.start)
    except (ReplayError, StageError, OSError, ValueError, subprocess.SubprocessError) as error:
        print(f"DELIVERY_PIPELINE_ERROR {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
