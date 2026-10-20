"""Split a fixed-Y caption candidate into Motion alpha parts and their Compressor jobs, resumably.

The generic form of the per-recording ``prepare_parts.py``:

- the cues are split into windows of at most ``MAX_SECONDS`` at whole frames inside gaps of at
  least ``MIN_GAP`` seconds; no cue crosses a window boundary;
- per window ``replay.motion_measured`` exports the part plans, then per part
  ``replay.motion_reveal_stages`` (template) and ``replay.motion_finish`` (style settings) write the
  finished ``.moti``;
- ``jobs.json`` lists one Compressor job per part, rendering into ``<render root>/<part>/``.

Windows, plans and finished parts that already exist are reused, so an interrupted run
continues; a window whose plan export was cut short is exported again.
"""

import argparse
import hashlib
import json
import math
import os
import shutil
import subprocess
import sys
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path

from replay.core import ReplayError
from replay.recordings import record

RATE = 30
MAX_SECONDS = 110
MIN_GAP = Fraction(1, 15)
WIDTH, HEIGHT = 1080, 1920
EXPECTED_TEXT = "Title"
RENDER_NAME = "compressor-batch-v1/motion-alpha.mov"

Run = Callable[[Sequence[str]], None]


@dataclass(frozen=True, slots=True)
class Window:
    """Output frames ``[start, end)`` and the cues entirely inside them."""

    index: int
    start: int
    end: int
    cues: tuple[dict[str, object], ...]


def cue_time(cue: dict[str, object], key: str) -> Fraction:
    return Fraction(str(record(cue["cue"])[key]))


def boundaries(cues: Sequence[dict[str, object]], total: int) -> list[int]:
    """Latest whole frame inside a long-enough gap that keeps each window within the limit."""
    starts = [cue_time(cue, "start") for cue in cues]
    ends = [cue_time(cue, "end") for cue in cues]
    edges: list[int] = []
    low = 0
    while total - low > MAX_SECONDS * RATE:
        limit = low + MAX_SECONDS * RATE
        best: int | None = None
        for index in range(1, len(cues)):
            gap_low, gap_high = ends[index - 1], starts[index]
            if gap_high - gap_low < MIN_GAP:
                continue
            first = max(math.ceil(gap_low * RATE), low + 1)
            frame = min(math.floor(gap_high * RATE), limit)
            if first <= frame and (best is None or frame > best):
                best = frame
        if best is None:
            raise ReplayError(f"No caption gap allows a boundary in ({low}, {limit}]")
        edges.append(best)
        low = best
    return edges


def windows(cues: Sequence[dict[str, object]]) -> list[Window]:
    total = math.ceil(max(cue_time(cue, "end") for cue in cues) * RATE) + 1
    edges = [0, *boundaries(cues, total), total]
    result = []
    for index, (start, end) in enumerate(zip(edges, edges[1:], strict=False)):
        low, high = Fraction(start, RATE), Fraction(end, RATE)
        inside, crossing = [], []
        for cue in cues:
            first, last = cue_time(cue, "start"), cue_time(cue, "end")
            if low <= first and last <= high:
                inside.append(cue)
            elif first < high and last > low:
                crossing.append(cue)
        if crossing:
            raise ReplayError(f"A cue crosses the window boundary {start}/{end}")
        result.append(Window(index, start, end, tuple(inside)))
    if sum(len(window.cues) for window in result) != len(cues):
        raise ReplayError("Windows do not hold every cue exactly once")
    return result


def shifted(document: dict[str, object], window: Window) -> dict[str, object]:
    """The candidate restricted to the window, with cue times relative to its start."""
    offset = Fraction(window.start, RATE)
    cues = []
    for cue in window.cues:
        timing = record(cue["cue"])
        start, end = cue_time(cue, "start") - offset, cue_time(cue, "end") - offset
        cues.append({**cue, "cue": {**timing, "start": float(start), "end": float(end)}})
    part = {**document, "candidates": cues}
    part.pop("plannedRuns", None)
    return part


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_once(path: Path, text: str) -> None:
    if path.exists():
        if path.read_text(encoding="utf-8") != text:
            raise ReplayError(f"{path} exists with different content")
        return
    with path.open("x", encoding="utf-8") as stream:
        os.fchmod(stream.fileno(), 0o600)
        stream.write(text)


def module(name: str, *arguments: str) -> list[str]:
    return [sys.executable, "-B", "-m", f"replay.{name}", *arguments]


def run_checked(command: Sequence[str]) -> None:
    subprocess.run(list(command), check=True)


def export_plans(window: Window, path: Path, plans: Path, run: Run) -> list[dict[str, object]]:
    """The window's part plans, exporting them unless a complete export exists."""
    manifest = plans / "manifest.json"
    if not manifest.exists():
        if plans.exists():
            shutil.rmtree(plans)  # an export cut short
        run(
            module("motion_measured", str(path), str(plans), "--sha256", sha256(path),
                   "--width", str(WIDTH), "--height", str(HEIGHT), "--rate", str(RATE),
                   "--frames", str(window.end - window.start), "--expected-text", EXPECTED_TEXT)
        )  # fmt: skip
    listed = record(json.loads(manifest.read_text("utf-8"))).get("parts")
    if not isinstance(listed, list):
        raise ReplayError(f"{manifest} lists no parts")
    return [record(item) for item in listed]


def finished_part(
    two: Path, final: Path, plan: Path, *, template: Path, finish: Sequence[str], run: Run
) -> None:
    """Write the staged and the finished ``.moti`` unless the finished one exists."""
    if final.exists():
        return
    two.unlink(missing_ok=True)
    run(module("motion_reveal_stages", str(template), str(plan), str(two)))
    run(module("motion_finish", str(two), str(final), *finish))


def prepare(
    candidate: Path,
    output: Path,
    render_root: Path,
    *,
    template: Path,
    finish: Sequence[str],
    preset: Path,
    run: Run = run_checked,
) -> Path:
    """Write every window, plan, finished part, ``parts.json`` and ``jobs.json``."""
    document = record(json.loads(candidate.read_text(encoding="utf-8")))
    listed = document.get("candidates")
    if not isinstance(listed, list) or not listed:
        raise ReplayError("The candidate has no cues")
    cues = [record(cue) for cue in listed]
    for directory in (output / "windows", output / "moti", render_root):
        directory.mkdir(mode=0o700, parents=True, exist_ok=True)
    parts: list[dict[str, object]] = []
    for window in windows(cues):
        path = output / "windows" / f"candidate-w{window.index:03d}.json"
        write_once(path, json.dumps(shifted(document, window), ensure_ascii=False))
        plans = output / f"plans-w{window.index:03d}"
        for index, item in enumerate(export_plans(window, path, plans, run)):
            name = f"w{window.index:03d}-part-{index:04d}"
            final = output / "moti" / f"{name}.moti"
            finished_part(
                output / "moti" / f"{name}.two-layer.moti", final,
                plans / f"part-{index:04d}.json", template=template, finish=finish, run=run,
            )  # fmt: skip
            (render_root / name).mkdir(mode=0o700, exist_ok=True)
            parts.append(
                {
                    "name": name,
                    "window": window.index,
                    "offsetFrame": window.start + int(str(item["offsetFrame"])),
                    "frames": int(str(item["frames"])),
                    "stages": item["stages"],
                    "moti": str(final),
                    "renderDir": str(render_root / name),
                }
            )
    write_once(output / "parts.json", json.dumps(parts, indent=1) + "\n")
    jobs = [
        {
            "name": str(part["name"]),
            "source": str(part["moti"]),
            "preset": str(preset),
            "output": str(Path(str(part["renderDir"])) / RENDER_NAME),
        }
        for part in parts
    ]
    write_once(output / "jobs.json", json.dumps({"jobs": jobs}, indent=1) + "\n")
    return output / "parts.json"


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("candidate", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("render_root", type=Path)
    parser.add_argument("--template", type=Path, required=True)
    parser.add_argument("--preset", type=Path, required=True)
    arguments = list(sys.argv[1:] if argv is None else argv)
    split = arguments.index("--") if "--" in arguments else len(arguments)
    args = parser.parse_args(arguments[:split])
    finish = arguments[split + 1 :]  # motion_finish style arguments after ``--``
    print(
        prepare(
            args.candidate, args.output, args.render_root,
            template=args.template, finish=finish, preset=args.preset,
        )
    )  # fmt: skip
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
