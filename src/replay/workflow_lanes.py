"""Split a workflow manifest's independent stages into parallel lane manifests.

Stages of one workflow run one after another, so a manifest of many independent
windows (face tracking, per-chunk collection) uses a single core of an otherwise idle
machine. This tool partitions such a manifest into several lanes. Stages joined by a
file dependency stay together, in dependency order. Every lane is an ordinary
`replay.workflow` manifest with its own private state directory, so each lane keeps
the usual fingerprint, completion and verification rules. Nothing is skipped or
approximated, and a lane never shares a state directory with another lane.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from graphlib import TopologicalSorter
from pathlib import Path

from replay import workflow
from replay.core import ReplayError
from replay.stages import StageError

MAX_LANES = 16
MANIFEST_KEYS = frozenset({"version", "stages"})
TMUX_JOB = Path(__file__).resolve().parents[2] / "scripts" / "tmux-job.sh"


@dataclass(frozen=True, slots=True)
class Lane:
    index: int
    manifest: Path
    state: Path
    stages: tuple[str, ...]


def _components(graph: dict[str, tuple[str, ...]]) -> list[tuple[str, ...]]:
    """Group stages joined by dependencies; order each group by dependency."""
    parent = {name: name for name in graph}

    def find(name: str) -> str:
        while parent[name] != name:
            parent[name] = parent[parent[name]]
            name = parent[name]
        return name

    for name, dependencies in graph.items():
        for dependency in dependencies:
            parent[find(name)] = find(dependency)
    groups: dict[str, dict[str, tuple[str, ...]]] = {}
    for name, dependencies in graph.items():
        groups.setdefault(find(name), {})[name] = dependencies
    return [tuple(TopologicalSorter(group).static_order()) for group in groups.values()]


def _assign(groups: list[tuple[str, ...]], lanes: int) -> list[list[str]]:
    """Greedy balance: biggest group first, always onto the currently shortest lane."""
    count = min(lanes, len(groups))
    assigned: list[list[str]] = [[] for _ in range(count)]
    for group in sorted(groups, key=lambda item: (-len(item), item[0])):
        shortest = min(range(count), key=lambda index: (len(assigned[index]), index))
        assigned[shortest].extend(group)
    return assigned


def split(manifest: Path, output: Path, lanes: int) -> tuple[Lane, ...]:
    """Write one ordinary workflow manifest and one empty private state per lane."""
    if type(lanes) is not int or not 1 <= lanes <= MAX_LANES:
        raise ReplayError(f"Lane count must be between 1 and {MAX_LANES}")
    raw = json.loads(manifest.read_text(encoding="utf-8"))
    if not isinstance(raw, dict) or set(raw) != MANIFEST_KEYS:
        raise ReplayError("Only plain version and stages manifests can be split")
    stages = workflow.load_stages(manifest)
    definitions = {str(item["name"]): item for item in raw["stages"]}
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise ReplayError("Lane output must be a new or empty directory")
    groups = _components(workflow.stage_graph(stages))
    output.mkdir(mode=0o700, exist_ok=True)
    result: list[Lane] = []
    for index, names in enumerate(_assign(groups, lanes)):
        path = output / f"lane-{index:02d}.json"
        state = output / f"state-{index:02d}"
        path.write_text(
            json.dumps({"version": raw["version"], "stages": [definitions[n] for n in names]}),
            encoding="utf-8",
        )
        state.mkdir(mode=0o700)
        result.append(Lane(index, path, state, tuple(names)))
    return tuple(result)


def launch(lanes: Sequence[Lane], output: Path, prefix: str) -> list[str]:
    """Start each lane as a detached job through the repository job runner."""
    names: list[str] = []
    for lane in lanes:
        name = f"{prefix}-lane{lane.index:02d}"
        command = [
            str(TMUX_JOB),
            str(output),
            name,
            sys.executable,
            "-B",
            "-m",
            "replay.workflow",
            str(lane.manifest),
            "--state",
            str(lane.state),
        ]
        subprocess.run(command, check=True, timeout=60)
        names.append(name)
    return names


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="replay.workflow_lanes")
    parser.add_argument("manifest", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--lanes", type=int, required=True)
    parser.add_argument("--launch", metavar="PREFIX", help="start every lane as a detached job")
    args = parser.parse_args(argv)
    try:
        lanes = split(args.manifest, args.output, args.lanes)
        jobs = launch(lanes, args.output, args.launch) if args.launch else []
    except (ReplayError, StageError, OSError, ValueError, subprocess.SubprocessError) as error:
        print(f"WORKFLOW_LANES_ERROR {error}", file=sys.stderr)
        return 1
    print(
        json.dumps(
            {
                "lanes": [
                    {
                        "manifest": str(lane.manifest),
                        "state": str(lane.state),
                        "stages": len(lane.stages),
                    }
                    for lane in lanes
                ],
                "jobs": jobs,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
