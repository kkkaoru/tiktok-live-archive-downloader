"""Compose the clean source and the alpha caption parts into the final video, resumably.

The generic form of the per-recording ``compose_prepare.py``, ``worker-*.sh``, ``finish.sh``
and ``verify_join2.py`` scripts:

- ``plan``: split the cut list into ``media_edit`` requests within the schema limits
  (``MAX_CLIPS`` clips, ``MAX_LAYERS`` alpha layers, ``MAX_GROUP_SECONDS`` per request); a cut
  is subdivided wherever more alpha parts would overlap one request. For each alpha part the
  render used is the one whose packet count equals its planned frames;
- ``render``: run the group requests, ``workers`` at a time. A group whose output already holds
  its planned frames is reused; an unfinished group's partial output is removed and redone;
- ``finish``: encode the tempo audio once and join the groups losslessly with it
  (``native_join.join``, verified packet count). A finished join is reused.

Every request and output stays under the pipeline's directories; inputs are only read.
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path

from replay.core import ReplayError
from replay.native_join import PartFile, join, probe_video
from replay.recordings import record

RATE = 30
MAX_CLIPS = 60
MAX_LAYERS = 16
MAX_GROUP_SECONDS = 600
WIDTH, HEIGHT = 1080, 1920
BIT_RATE = 20_000_000
REQUEST_TIMEOUT = 7200
AUDIO_ARGUMENTS = ("-c:a", "aac", "-b:a", "256k", "-ar", "44100", "-ac", "2")

Packets = Callable[[Path], int]
Call = Callable[[Path, Path], None]


@dataclass(frozen=True, slots=True)
class Cut:
    """Source frames ``[start, end)`` placed at output frame ``output``."""

    start: int
    end: int
    output: int

    @property
    def length(self) -> int:
        return self.end - self.start


@dataclass(frozen=True, slots=True)
class Part:
    """An alpha caption part covering output frames ``[offset, offset + frames)``."""

    name: str
    offset: int
    frames: int
    render: Path

    @property
    def end(self) -> int:
        return self.offset + self.frames


def seconds(frames: int) -> float:
    return float(Fraction(frames, RATE))


def sampled(frames: int) -> float:
    """The middle of frame slot ``frames``: where a selection reads that slot's frame.

    A selection that starts exactly on a frame boundary may be rounded to the previous frame,
    and a variable-rate source presents frames a fraction of a slot away from the grid; the
    middle of the slot falls inside the intended frame in both cases.
    """
    return float((Fraction(frames) + Fraction(1, 2)) / RATE)


def packet_count(path: Path) -> int:
    """Video packets of ``path``, or -1 when it cannot be probed."""
    try:
        return probe_video(path).packets
    except ReplayError:
        return -1


def load_cuts(path: Path) -> list[Cut]:
    rows = record(json.loads(path.read_text(encoding="utf-8"))).get("cuts")
    if not isinstance(rows, list) or not rows:
        raise ReplayError("The cut list is empty")
    cuts = []
    for row in rows:
        item = record(row)
        cuts.append(
            Cut(
                int(str(item["source_start_frame"])),
                int(str(item["source_end_frame"])),
                int(str(item["output_start_frame"])),
            )
        )
    return cuts


def complete_render(directory: Path, frames: int, packets: Packets) -> Path:
    """The newest render under ``directory`` whose packet count equals the planned frames."""
    found = [
        movie
        for movie in sorted(directory.glob("compressor-*/motion-alpha*.mov"))
        if movie.stat().st_size > 0 and packets(movie) == frames
    ]
    if not found:
        raise ReplayError(f"No complete render under {directory}")
    return found[-1]


def recorded_render(state: Path, directory: Path, frames: int, packets: Packets) -> Path:
    """The output a Compressor job state records as successful, holding the planned frames.

    A movie left by a job Compressor reported as failed is never used, even when complete.
    """
    saved = record(json.loads(state.read_text(encoding="utf-8")))
    output = Path(str(saved.get("output")))
    if saved.get("status") != "Successful" or not output.is_relative_to(directory):
        raise ReplayError(f"No successful Compressor render recorded in {state}")
    if not output.is_file() or packets(output) != frames:
        raise ReplayError(f"The recorded render {output} does not hold its planned frames")
    return output


def load_parts(path: Path, packets: Packets, states: Path | None = None) -> list[Part]:
    """The planned parts and their renders; with ``states``, only recorded successful renders."""
    parts = []
    for row in json.loads(path.read_text(encoding="utf-8")):
        item = record(row)
        name, frames = str(item["name"]), int(str(item["frames"]))
        directory = Path(str(item["renderDir"]))
        render = (
            complete_render(directory, frames, packets)
            if states is None
            else recorded_render(states / f"{name}.json", directory, frames, packets)
        )
        parts.append(Part(name, int(str(item["offsetFrame"])), frames, render))
    return parts


def overlapping(parts: Sequence[Part], low: int, high: int) -> int:
    return sum(1 for part in parts if part.offset < high and part.end > low)


def split_cut(cut: Cut, parts: Sequence[Part]) -> list[Cut]:
    """The cut in pieces of at most ``MAX_GROUP_SECONDS`` and ``MAX_LAYERS`` overlapping parts."""
    pieces: list[Cut] = []
    position = 0
    while position < cut.length:
        low = cut.output + position
        high = min(cut.output + cut.length, low + MAX_GROUP_SECONDS * RATE)
        while overlapping(parts, low, high) > MAX_LAYERS:
            edges = sorted({x for part in parts for x in (part.offset, part.end) if low < x < high})
            fitting = [x for x in edges if overlapping(parts, low, x) <= MAX_LAYERS]
            if not fitting:
                raise ReplayError("More alpha parts overlap one frame than a request allows")
            high = max(fitting)
        piece = high - low
        pieces.append(Cut(cut.start + position, cut.start + position + piece, low))
        position += piece
    return pieces


def group_bounds(cuts: Sequence[Cut], parts: Sequence[Part]) -> list[tuple[int, int]]:
    """Consecutive cut ranges that each fit one request."""

    def fits(first: int, last: int) -> bool:
        base = cuts[first].output
        end = cuts[last - 1].output + cuts[last - 1].length
        within = (end - base) <= MAX_GROUP_SECONDS * RATE
        return last - first <= MAX_CLIPS and overlapping(parts, base, end) <= MAX_LAYERS and within

    bounds: list[tuple[int, int]] = []
    first = 0
    while first < len(cuts):
        last = first + 1
        while last < len(cuts) and fits(first, last + 1):
            last += 1
        bounds.append((first, last))
        first = last
    return bounds


def request(
    cuts: Sequence[Cut], parts: Sequence[Part], source: Path, directory: Path, name: str
) -> tuple[dict[str, object], int]:
    """A media_edit request for consecutive cuts and the frames it renders."""
    base, end = cuts[0].output, cuts[-1].output + cuts[-1].length
    clips = [
        {
            "sourcePath": str(source),
            "selection": {
                "startSeconds": sampled(cut.start),
                "durationSeconds": seconds(cut.length),
                "rate": 1,
            },
        }
        for cut in cuts
    ]
    layers = []
    for part in parts:
        low, high = max(part.offset, base), min(part.end, end)
        if low < high:
            layers.append(
                {
                    "sourcePath": str(part.render),
                    "selection": {
                        # Alpha parts are exact constant-rate renders: frame starts are exact.
                        "startSeconds": seconds(low - part.offset),
                        "durationSeconds": seconds(high - low),
                        "rate": 1,
                    },
                    "offsetSeconds": seconds(low - base),
                    "opacity": 1,
                }
            )
    video = {
        "width": WIDTH,
        "height": HEIGHT,
        "frameRate": RATE,
        "resizeMode": "fit",
        "encoding": {"codec": "hevc", "averageBitRate": BIT_RATE, "allowFrameReordering": False},
    }
    body: dict[str, object] = {
        "outputDirectory": str(directory),
        "outputName": f"{name}.mp4",
        "recipe": {"video": video, "clips": clips, "additionalVideo": layers},
    }
    return body, end - base


def write_json(path: Path, value: object) -> None:
    """Write a new file or accept an identical one, so a re-plan is idempotent."""
    text = json.dumps(value, indent=1, ensure_ascii=False) + "\n"
    if path.exists():
        if path.read_text(encoding="utf-8") != text:
            raise ReplayError(f"{path} exists with a different plan")
        return
    with path.open("x", encoding="utf-8") as stream:
        os.fchmod(stream.fileno(), 0o600)
        stream.write(text)


def plan(
    cuts_path: Path,
    parts_path: Path,
    source: Path,
    output_root: Path,
    requests: Path,
    *,
    packets: Packets | None = None,
    states: Path | None = None,
) -> Path:
    """Write one request per group and ``groups.json``; return ``groups.json``."""
    parts = load_parts(parts_path, packets or packet_count, states)
    cuts = [piece for cut in load_cuts(cuts_path) for piece in split_cut(cut, parts)]
    requests.mkdir(mode=0o700, parents=True, exist_ok=True)
    rows = []
    for index, (first, last) in enumerate(group_bounds(cuts, parts)):
        name = f"g{index:03d}"
        directory = output_root / name
        body, frames = request(cuts[first:last], parts, source, directory, name)
        write_json(requests / f"{name}.json", body)
        rows.append({"group": index, "name": name, "frames": frames, "directory": str(directory)})
    groups = requests / "groups.json"
    write_json(groups, rows)
    return groups


def group_output(directory: Path, name: str) -> list[Path]:
    return sorted(directory.glob(f"edit-*/{name}.mp4"))


def render_group(row: dict[str, object], requests: Path, call: Call, packets: Packets) -> str:
    """Reuse a finished group, otherwise clear its partial output and render it."""
    name, frames = str(row["name"]), int(str(row["frames"]))
    directory = Path(str(row["directory"]))
    outputs = group_output(directory, name)
    if len(outputs) == 1 and packets(outputs[0]) == frames:
        return "reused"
    if directory.exists():
        shutil.rmtree(directory)
    directory.mkdir(mode=0o700, parents=True)
    response = requests / f"{name}.response.json"
    response.unlink(missing_ok=True)
    call(requests / f"{name}.json", response)
    outputs = group_output(directory, name)
    if len(outputs) != 1 or packets(outputs[0]) != frames:
        raise ReplayError(f"{name} did not render its planned {frames} frames")
    return "completed"


def media_edit_call(server: Path, server_sha256: str) -> Call:
    def call(arguments: Path, response: Path) -> None:
        subprocess.run(
            [sys.executable, "-B", "-m", "replay.apple_pro_apps_call", "media_edit",
             str(arguments), str(response), "--server", str(server),
             "--server-sha256", server_sha256, "--timeout", str(REQUEST_TIMEOUT)],
            check=True,
        )  # fmt: skip

    return call


def groups_of(requests: Path) -> list[dict[str, object]]:
    return [record(row) for row in json.loads((requests / "groups.json").read_text("utf-8"))]


def render(
    requests: Path, call: Call, *, workers: int, packets: Packets | None = None
) -> dict[str, str]:
    """Render every group, ``workers`` at a time, longest first; return each group's result."""
    rows = sorted(groups_of(requests), key=lambda row: -int(str(row["frames"])))
    count = packets or packet_count
    with ThreadPoolExecutor(max_workers=workers) as pool:
        results = pool.map(lambda row: render_group(row, requests, call, count), rows)
        return {str(row["name"]): result for row, result in zip(rows, results, strict=True)}


def finish(
    requests: Path, tempo: Path, output: Path, *, ffmpeg: str, packets: Packets | None = None
) -> int:
    """Encode the tempo audio once and join every group with it; reuse a finished join."""
    rows = sorted(groups_of(requests), key=lambda row: int(str(row["group"])))
    expected = sum(int(str(row["frames"])) for row in rows)
    count = packets or packet_count
    if output.exists() and count(output) == expected:
        return expected
    output.unlink(missing_ok=True)
    audio = output.parent / "audio" / "tempo.m4a"
    if not audio.exists():
        audio.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        staged = audio.with_name(".tempo.partial.m4a")
        staged.unlink(missing_ok=True)
        subprocess.run(
            [ffmpeg, "-nostdin", "-v", "error", "-i", str(tempo), *AUDIO_ARGUMENTS, str(staged)],
            check=True,
        )
        staged.rename(audio)
    files = []
    for row in rows:
        name, frames = str(row["name"]), int(str(row["frames"]))
        found = group_output(Path(str(row["directory"])), name)
        if len(found) != 1 or count(found[0]) != frames:
            raise ReplayError(f"{name} is missing or does not hold its planned frames")
        files.append(PartFile(found[0], frames))
    return join(files, rate=RATE, output=output, audio=audio)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    planned = commands.add_parser("plan")
    for name in ("cuts", "parts", "source", "output_root", "requests"):
        planned.add_argument(name, type=Path)
    planned.add_argument("--states", type=Path, help="Compressor job states of the parts")
    rendered = commands.add_parser("render")
    rendered.add_argument("requests", type=Path)
    rendered.add_argument("--server", type=Path, required=True)
    rendered.add_argument("--server-sha256", required=True)
    rendered.add_argument("--workers", type=int, default=3)
    finished = commands.add_parser("finish")
    finished.add_argument("requests", type=Path)
    finished.add_argument("tempo", type=Path)
    finished.add_argument("output", type=Path)
    finished.add_argument("--ffmpeg", default="ffmpeg")
    args = parser.parse_args(argv)
    if args.command == "plan":
        arguments = (args.cuts, args.parts, args.source, args.output_root, args.requests)
        print(plan(*arguments, states=args.states))
    elif args.command == "render":
        call = media_edit_call(args.server, args.server_sha256)
        print(json.dumps(render(args.requests, call, workers=args.workers)))
    else:
        print(finish(args.requests, args.tempo, args.output, ffmpeg=args.ffmpeg))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
