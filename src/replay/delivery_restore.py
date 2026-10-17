"""Make retired render intermediates regenerable instead of leaving the pipeline stuck.

A stage runner refuses to continue when a completed stage's output is missing, and the
sealed render path refuses a missing part movie. Deleting an intermediate therefore needs a
matching, lossless change of bookkeeping: the recipe of what was retired is recorded and
the checkpoint directories and JSON outputs of every producer plan that touched a retired
file are moved (never deleted) into an archive. The plan files stay, so the next pipeline
run sees a clean stage, re-executes its command and writes fresh checkpoints.

Native movies are not bit-identical across renders; a restored delivery is accepted by the
usual media verification, not by comparing with the digests kept here as evidence.
"""

from __future__ import annotations

import json
import os
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path

from replay import disk_budget
from replay.core import ReplayError
from replay.recordings import record

RECIPE_NAME = "restore-recipe.json"
ARCHIVE_DIRECTORY = "archive"
PLAN_NAMES = (
    "alpha-jobs.json",
    "join-workflow.json",
    "audio-workflow.json",
    "final-mux-workflow.json",
    "final-job.json",
)
RENDER_PATTERN = "render-*-*.json"
RESPONSE_NAME = "render.response.json"
MAX_JSON_BYTES = 64 * 1024 * 1024


@dataclass(frozen=True, slots=True)
class Plan:
    """A producer plan, its checkpoint directory and the JSON files its stages wrote."""

    path: Path
    state: Path
    outputs: tuple[Path, ...]
    media: frozenset[str]


def _document(path: Path) -> dict[str, object]:
    if path.stat().st_size > MAX_JSON_BYTES:
        raise ReplayError(f"JSON exceeds its size limit: {path.name}")
    return record(json.loads(path.read_text(encoding="utf-8")))


def _rows(document: dict[str, object], key: str) -> list[dict[str, object]]:
    rows = document.get(key)
    return [record(row) for row in rows] if isinstance(rows, list) else []


def _listed(row: dict[str, object], key: str) -> list[str]:
    values = row.get(key)
    return [value for value in values if isinstance(value, str)] if isinstance(values, list) else []


def _texts(row: dict[str, object], key: str) -> list[str]:
    value = row.get(key)
    return [value] if isinstance(value, str) else []


def _response_movie(response: Path) -> str | None:
    try:
        output = _document(response).get("outputPath")
    except (OSError, ValueError, ReplayError):
        return None
    return output if isinstance(output, str) else None


def load_plan(path: Path) -> Plan:
    """Every path a plan mentions, and the JSON files its stages or jobs produce."""
    document = _document(path)
    stages = _rows(document, "stages")
    jobs = _rows(document, "jobs")
    listed = [
        value for stage in stages for key in ("inputs", "outputs") for value in _listed(stage, key)
    ]
    listed += [value for job in jobs for value in _listed(job, "inputs")]
    listed += [value for job in jobs for value in _texts(job, "output")]
    outputs = tuple(
        Path(value)
        for stage in stages
        for value in _listed(stage, "outputs")
        if value.endswith(".json")
    )
    movies = {
        movie
        for output in outputs
        if output.name == RESPONSE_NAME and (movie := _response_movie(output)) is not None
    }
    return Plan(path, path.with_name(f"{path.stem}-state"), outputs, frozenset({*listed, *movies}))


def plans(work: Path, composition: Path) -> list[Plan]:
    found = [work / name for name in PLAN_NAMES if (work / name).is_file()]
    found += sorted(composition.glob(RENDER_PATTERN))
    return [load_plan(path) for path in found]


def affected(candidates: Iterable[Plan], retired: set[str]) -> list[Plan]:
    return [plan for plan in candidates if plan.media & retired]


def _digests(plan: Plan, retired: set[str]) -> dict[str, str]:
    found: dict[str, str] = {}
    for receipt in sorted(plan.state.glob("*.json")) if plan.state.is_dir() else []:
        try:
            outputs = record(_document(receipt).get("outputs"))
        except (OSError, ValueError, ReplayError):
            continue
        found.update(
            {
                path: digest
                for path, digest in outputs.items()
                if path in retired and isinstance(digest, str)
            }
        )
    return found


def prepare(work: Path, composition: Path, retired: Sequence[Path]) -> dict[str, object]:
    """The restore recipe: what is retired and which plans must be reset to rebuild it."""
    names = {str(path) for path in retired}
    chosen = affected(plans(work, composition), names)
    return {
        "version": 1,
        "retired": sorted(names),
        "plans": [
            {
                "plan": str(plan.path),
                "state": str(plan.state),
                "outputs": [str(path) for path in plan.outputs],
                "digests": _digests(plan, names),
            }
            for plan in chosen
        ],
        "moved": [],
    }


def _move(source: Path, target: Path) -> dict[str, str]:
    target.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    if target.exists():
        raise ReplayError(f"Archive target already exists: {target.name}")
    os.rename(source, target)
    return {"from": str(source), "to": str(target)}


def archive(recipe: dict[str, object], directory: Path) -> Path:
    """Move checkpoints and JSON outputs of the recipe's plans aside; write the recipe."""
    moved: list[dict[str, str]] = []
    for index, entry in enumerate(_rows(recipe, "plans")):
        plan = Path(str(entry["plan"]))
        base = directory / ARCHIVE_DIRECTORY / f"{index:04d}-{plan.stem}"
        state = Path(str(entry["state"]))
        if state.is_dir():
            moved.append(_move(state, base / state.name))
        for number, name in enumerate(_listed(entry, "outputs")):
            output = Path(name)
            if output.is_file():
                moved.append(_move(output, base / f"{number:06d}-{output.name}"))
    path = directory / RECIPE_NAME
    directory.mkdir(mode=0o700, parents=True, exist_ok=True)
    path.write_text(
        json.dumps({**recipe, "moved": moved}, indent=1, sort_keys=True), encoding="utf-8"
    )
    return path


def restore(source: Path, roots: Sequence[Path], directory: Path, receipt_name: str) -> list[str]:
    """Allow the pipeline to rebuild: set the retention receipt aside; name the plans to rerun."""
    recipe_path = directory / RECIPE_NAME
    receipt = directory / receipt_name
    if not recipe_path.is_file() or not receipt.is_file():
        raise ReplayError("Nothing was retired with a restore recipe")
    if not source.is_file():
        raise ReplayError("The original recording is missing; nothing can be rebuilt")
    disk_budget.require_headroom(roots)
    plans_to_run = [str(entry["plan"]) for entry in _rows(_document(recipe_path), "plans")]
    _move(receipt, directory / ARCHIVE_DIRECTORY / receipt_name)
    return plans_to_run
