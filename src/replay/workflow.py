"""Run a trusted local stage manifest without a shell or interactive operations."""

import argparse
import json
import subprocess
import sys
from collections.abc import Sequence
from pathlib import Path

from replay.stages import Stage, StageError, run_stage

MAX_PLAN_BYTES = 1_048_576
MAX_STAGES = 512
STAGE_FIELDS = frozenset(
    {"name", "command", "inputs", "outputs", "timeout_seconds", "verify_command", "adopt_existing"}
)


class Arguments(argparse.Namespace):
    plan: Path
    state: Path


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


def parse_stages(value: object) -> tuple[Stage, ...]:
    """Parse trusted execution configuration, not downloaded metadata or prompts."""
    if not isinstance(value, dict) or set(value) != {"version", "stages"}:
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
    return stages


def run_plan(plan: Path, *, state_directory: Path) -> None:
    if plan.is_symlink() or plan.stat().st_size > MAX_PLAN_BYTES:
        raise StageError("Invalid workflow manifest file")
    raw: object = json.loads(plan.read_text(encoding="utf-8"))
    stages = parse_stages(raw)
    for stage in stages:
        result = run_stage(stage, state_directory=state_directory)
        print(f"{stage.name}: {result}", flush=True)
        if result == "busy":
            raise StageError("Another process owns this stage; no duplicate was started")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("plan", type=Path, help="Trusted local JSON command manifest")
    parser.add_argument("--state", required=True, type=Path, help="Private checkpoint directory")
    arguments = Arguments()
    parser.parse_args(argv, namespace=arguments)
    try:
        run_plan(arguments.plan, state_directory=arguments.state)
    except subprocess.TimeoutExpired:
        print("Workflow deadline exceeded; inspect private stage logs.", file=sys.stderr)
        return 1
    except (StageError, OSError, ValueError) as error:
        print(
            f"Workflow stopped ({type(error).__name__}); inspect its private state.",
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
