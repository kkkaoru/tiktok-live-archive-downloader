"""Generic media preparation for any verified recording before caption planning.

These helpers turn one source movie into the inputs that the shared caption and
composition stages need: an AVFoundation-readable canvas proxy, the decoded
presentation clock and bounded foreground-observation windows with worker
workflows. Nothing here is specific to one recording; every policy input is
derived from the source or passed explicitly.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from fractions import Fraction
from itertools import batched
from pathlib import Path

from replay.audio_clock import AudioFrameClock
from replay.core import ReplayError
from replay.recordings import record

FOREGROUND_WINDOW_FRAMES = 128
WORKFLOW_STAGES = 512
FOREGROUND_TIMEOUT_SECONDS = 600
MAX_CLOCK_BYTES = 256 * 1024 * 1024
PROXY_QUALITY = 75
PROXY_SUFFIXES = frozenset({".mov", ".mp4", ".m4v"})


@dataclass(frozen=True, slots=True)
class Canvas:
    width: int
    height: int
    rate: int

    def __post_init__(self) -> None:
        if (
            min(self.width, self.height, self.rate) <= 0
            or self.width % 2
            or self.height % 2
            or self.rate > 120
        ):
            raise ReplayError("Canvas needs positive even dimensions and a bounded rate")


@dataclass(frozen=True, slots=True)
class FrameClock:
    """Decoded presentation timestamps in one time base, strictly increasing."""

    pts: tuple[int, ...]
    time_base: Fraction
    last_duration: int

    def __post_init__(self) -> None:
        if not self.pts or self.last_duration <= 0 or self.time_base <= 0:
            raise ReplayError("A frame clock needs frames, a positive base and last duration")
        if any(later <= earlier for earlier, later in zip(self.pts, self.pts[1:], strict=False)):
            raise ReplayError("Decoded presentation timestamps must strictly increase")


def proxy_command(source: Path, output: Path, canvas: Canvas, *, ffmpeg: str) -> tuple[str, ...]:
    """Fit the whole source inside the canvas at a constant rate without cropping.

    Audio is carried as 48 kHz PCM so later separation does not inherit a second
    lossy generation. The output suffix must be one native tools can open.
    """
    if output.suffix.lower() not in PROXY_SUFFIXES:
        raise ReplayError("Proxy output must be .mov, .mp4 or .m4v")
    fit = (
        f"scale={canvas.width}:{canvas.height}:force_original_aspect_ratio=decrease:"
        f"flags=lanczos,pad={canvas.width}:{canvas.height}:(ow-iw)/2:(oh-ih)/2:black,"
        f"fps={canvas.rate},format=yuv420p"
    )
    return (
        ffmpeg,
        "-nostdin",
        "-hide_banner",
        "-v",
        "error",
        "-n",
        "-i",
        str(source),
        "-map",
        "0:v:0",
        "-map",
        "0:a:0",
        "-vf",
        fit,
        "-c:v",
        "hevc_videotoolbox",
        "-q:v",
        str(PROXY_QUALITY),
        "-tag:v",
        "hvc1",
        "-c:a",
        "pcm_s16le",
        "-ar",
        "48000",
        "-movflags",
        "+faststart",
        str(output),
    )


def clock_command(source: Path, *, ffprobe: str) -> tuple[str, ...]:
    # ffprobe decodes on one thread unless asked; the frame clock is byte-identical threaded
    # (verified on 1.6 h TikTok and 2.5 h YouTube sources: 1,092 s -> 210 s).
    return (
        ffprobe,
        "-v",
        "error",
        "-threads",
        "0",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=time_base:frame=pts,duration",
        "-of",
        "json",
        str(source),
    )


def audio_clock_command(source: Path, *, ffprobe: str) -> tuple[str, ...]:
    """Decoded audio frame clock in the form `replay.audio_clock` reads."""
    return (
        ffprobe,
        "-v",
        "error",
        "-select_streams",
        "a:0",
        "-show_entries",
        "stream=sample_rate,time_base:frame=pts,nb_samples",
        "-of",
        "json",
        str(source),
    )


def parse_clock(data: object) -> FrameClock:
    """Validate ffprobe JSON (stream time base plus per-frame pts/duration)."""
    raw = record(data)
    streams, frames = raw.get("streams"), raw.get("frames")
    if not isinstance(streams, list) or len(streams) != 1 or not isinstance(frames, list):
        raise ReplayError("Expected one video stream and its decoded frames")
    base = record(streams[0]).get("time_base")
    if not isinstance(base, str) or "/" not in base:
        raise ReplayError("Missing video time base")
    pts: list[int] = []
    duration = 0
    for frame in frames:
        item = record(frame)
        value, length = item.get("pts"), item.get("duration")
        if type(value) is not int or type(length) is not int:
            raise ReplayError("Every decoded frame needs integer pts and duration")
        pts.append(value)
        duration = length
    return FrameClock(tuple(pts), Fraction(base), duration)


def read_clock(path: Path) -> FrameClock:
    if path.stat().st_size > MAX_CLOCK_BYTES:
        raise ReplayError("Frame clock exceeds the size limit")
    return parse_clock(json.loads(path.read_text(encoding="utf-8")))


def _digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _write_new(path: Path, value: object) -> None:
    with path.open("x", encoding="utf-8") as stream:
        os.fchmod(stream.fileno(), 0o600)
        json.dump(value, stream, indent=1, sort_keys=True)
        stream.write("\n")


def foreground_configs(
    clock: FrameClock, *, source: Path, native: Path, canvas: Canvas
) -> list[dict[str, object]]:
    """Split the whole clock into bounded windows for `replay.foreground_collection`.

    Each window keeps exact source ticks; the last frame duration of an inner
    window is the distance to the next window's first frame.
    """
    source_hash, native_hash = _digest(source), _digest(native)
    windows = list(batched(clock.pts, FOREGROUND_WINDOW_FRAMES))
    configs: list[dict[str, object]] = []
    for index, ticks in enumerate(windows):
        following = windows[index + 1][0] if index + 1 < len(windows) else None
        last = clock.last_duration if following is None else following - ticks[-1]
        configs.append(
            {
                "version": 1,
                "source": str(source),
                "sourceSHA256": source_hash,
                "nativeExecutable": str(native),
                "nativeSHA256": native_hash,
                "width": canvas.width,
                "height": canvas.height,
                "clock": {
                    "pts": list(ticks),
                    "timeBase": f"{clock.time_base.numerator}/{clock.time_base.denominator}",
                    "lastDuration": last,
                },
            }
        )
    return configs


def _foreground_stage(
    directory: Path, index: int, *, collection: str = "collection-state"
) -> dict[str, object]:
    name = f"foreground-{index:04d}"
    config = directory / "configs" / f"{index:04d}.json"
    evidence = directory / f"evidence-{index:04d}"
    module = (sys.executable, "-m", "replay.foreground_collection")
    return {
        "name": name,
        "command": [
            *module,
            "collect",
            str(config),
            str(evidence),
            "--state-directory",
            str(directory / collection),
            "--name",
            name,
        ],
        "inputs": [str(config)],
        "outputs": [str(evidence / "provenance.json")],
        "timeout_seconds": FOREGROUND_TIMEOUT_SECONDS,
        "verify_command": [*module, "verify", str(config), str(evidence)],
    }


def write_foreground_plan(
    directory: Path, configs: list[dict[str, object]], *, workers: int
) -> list[Path]:
    """Write configs and interleaved per-worker workflows; never overwrite a plan."""
    if not 1 <= workers <= 8 or not configs:
        raise ReplayError("Foreground plans need 1-8 workers and at least one window")
    directory = directory.absolute()
    (directory / "configs").mkdir(mode=0o700, parents=True)
    for index, config in enumerate(configs):
        _write_new(directory / "configs" / f"{index:04d}.json", config)
    workflows: list[Path] = []
    for worker in range(workers):
        indices = range(worker, len(configs), workers)
        for part, chunk in enumerate(batched(indices, WORKFLOW_STAGES)):
            path = directory / f"workflow-{worker}-{part:04d}.json"
            stages = [_foreground_stage(directory, index) for index in chunk]
            _write_new(path, {"version": 1, "stages": stages})
            workflows.append(path)
    return workflows


def retry_foreground(directory: Path, *, first: int, last: int, workers: int) -> list[Path]:
    """Rerun windows first..last that lack a receipt, with fresh state directories.

    The caller names the range so live workers are never duplicated. Partial
    evidence is kept under `evidence-NNNN.failed-K` and the window runs unchanged.
    """
    if not 1 <= workers <= 8 or last < first:
        raise ReplayError("Retry needs a window range and 1-8 workers")
    directory = directory.absolute()
    rounds = len(list(directory.glob("retry-[0-9][0-9][0-9][0-9]")))
    pending: list[int] = []
    for index in range(first, last + 1):
        if not (directory / "configs" / f"{index:04d}.json").is_file():
            raise ReplayError(f"Unknown foreground window {index}")
        evidence = directory / f"evidence-{index:04d}"
        if (evidence / "provenance.json").is_file():
            continue
        if evidence.exists():
            evidence.rename(evidence.with_name(f"{evidence.name}.failed-{rounds}"))
        pending.append(index)
    if not pending:
        raise ReplayError("Every window in the range already has a receipt")
    round_directory = directory / f"retry-{rounds:04d}"
    round_directory.mkdir(mode=0o700)
    collection = f"retry-{rounds:04d}/collection-state"
    workflows: list[Path] = []
    for worker in range(workers):
        for part, chunk in enumerate(batched(pending[worker::workers], WORKFLOW_STAGES)):
            stages = [_foreground_stage(directory, index, collection=collection) for index in chunk]
            path = round_directory / f"workflow-{worker}-{part:04d}.json"
            _write_new(path, {"version": 1, "stages": stages})
            workflows.append(path)
    return workflows


def rebind_foreground(directory: Path, configs: Sequence[Path]) -> list[Path]:
    """Copy window configs with a resolved native path and plan fresh collection.

    Readers reject linked executable paths (for example SwiftPM `build/release`).
    The clock, source and executable bytes are unchanged; the windows are
    observed again rather than relabelling existing evidence.
    """
    rows: list[dict[str, object]] = []
    for path in configs:
        raw = record(json.loads(path.read_text(encoding="utf-8")))
        native = raw.get("nativeExecutable")
        if not isinstance(native, str):
            raise ReplayError(f"Foreground config lacks its executable: {path}")
        resolved = Path(native).resolve(strict=True)
        if _digest(resolved) != raw.get("nativeSHA256"):
            raise ReplayError(f"Foreground executable changed: {path}")
        rows.append({**raw, "nativeExecutable": str(resolved)})
    return write_foreground_plan(directory, rows, workers=1)


def chunk_index(configs: Sequence[Path]) -> dict[str, object]:
    """Index rendered source chunks from their render configs, in source order.

    Each config names `outputDirectory` (holding `candidate.mp4`) and the exact
    decoded `clock` it rendered. Chunks must be contiguous: each ends where the
    next begins, and the last ends after its final frame duration.
    """
    rows: list[dict[str, object]] = []
    for path in configs:
        raw = record(json.loads(path.read_text(encoding="utf-8")))
        clock = record(raw.get("clock"))
        pts, base, last = clock.get("pts"), clock.get("timeBase"), clock.get("lastDuration")
        directory = raw.get("outputDirectory")
        if (
            not isinstance(pts, list)
            or not pts
            or not isinstance(base, str)
            or type(last) is not int
            or not isinstance(directory, str)
        ):
            raise ReplayError(f"Render config lacks its clock or output: {path}")
        first, final = pts[0], pts[-1]
        if type(first) is not int or type(final) is not int:
            raise ReplayError("Render clock ticks must be integers")
        media = Path(directory) / "candidate.mp4"
        if not media.is_file():
            raise ReplayError(f"Rendered chunk is missing: {media}")
        rows.append(
            {
                "media": str(media),
                "sourceFirstPTS": first,
                "sourceEndPTS": final + last,
                "frames": len(pts),
                "timeBase": base,
            }
        )
    rows.sort(key=_first_pts)
    for previous, current in zip(rows, rows[1:], strict=False):
        if previous["sourceEndPTS"] != current["sourceFirstPTS"]:
            raise ReplayError("Rendered chunks are not contiguous")
    return {"purpose": "source chunk clock index; not a quality approval", "ranges": rows}


def _first_pts(row: dict[str, object]) -> int:
    value = row["sourceFirstPTS"]
    if type(value) is not int:
        raise ReplayError("Invalid chunk start")
    return value


def _run(command: tuple[str, ...], *, stdout: Path | None = None) -> None:
    if stdout is None:
        subprocess.run(command, stdin=subprocess.DEVNULL, check=True)
        return
    with stdout.open("xb") as stream:
        subprocess.run(command, stdin=subprocess.DEVNULL, stdout=stream, check=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    proxy = commands.add_parser("proxy", help="Fit a source into a native-readable canvas")
    proxy.add_argument("source", type=Path)
    proxy.add_argument("output", type=Path)
    clock = commands.add_parser("clock", help="Save the decoded video presentation clock")
    clock.add_argument("source", type=Path)
    clock.add_argument("output", type=Path)
    sound = commands.add_parser("audio-clock", help="Save the decoded audio frame clock")
    sound.add_argument("source", type=Path)
    sound.add_argument("output", type=Path)
    sound.add_argument("--ffprobe", default="ffprobe")
    retried = commands.add_parser("foreground-retry", help="Rerun windows without receipts")
    retried.add_argument("directory", type=Path)
    retried.add_argument("--first", type=int, required=True)
    retried.add_argument("--last", type=int, required=True)
    retried.add_argument("--workers", type=int, default=2)
    rebind = commands.add_parser("foreground-rebind", help="Re-plan windows with resolved paths")
    rebind.add_argument("directory", type=Path)
    rebind.add_argument("configs", type=Path, nargs="+")
    chunks = commands.add_parser("chunk-index", help="Index rendered source chunks")
    chunks.add_argument("output", type=Path)
    chunks.add_argument("configs", type=Path, nargs="+")
    plan = commands.add_parser("foreground-plan", help="Bounded foreground windows/workflows")
    plan.add_argument("source", type=Path)
    plan.add_argument("clock", type=Path)
    plan.add_argument("native", type=Path)
    plan.add_argument("directory", type=Path)
    plan.add_argument("--workers", type=int, default=2)
    for command in (proxy, plan):
        command.add_argument("--width", type=int, default=1080)
        command.add_argument("--height", type=int, default=1920)
        command.add_argument("--rate", type=int, default=30)
    for command in (proxy, clock):
        command.add_argument("--ffmpeg", default="ffmpeg")
        command.add_argument("--ffprobe", default="ffprobe")
    args = parser.parse_args(argv)
    try:
        if args.command == "foreground-retry":
            for path in retry_foreground(
                args.directory, first=args.first, last=args.last, workers=args.workers
            ):
                print(path)
            return 0
        if args.command == "foreground-rebind":
            for path in rebind_foreground(args.directory, args.configs):
                print(path)
            return 0
        if args.command == "audio-clock":
            _run(audio_clock_command(args.source, ffprobe=args.ffprobe), stdout=args.output)
            AudioFrameClock.from_probe(json.loads(args.output.read_text(encoding="utf-8")))
            return 0
        if args.command == "chunk-index":
            _write_new(args.output, chunk_index(args.configs))
            return 0
        if args.command == "proxy":
            canvas = Canvas(args.width, args.height, args.rate)
            _run(proxy_command(args.source, args.output, canvas, ffmpeg=args.ffmpeg))
        elif args.command == "clock":
            _run(clock_command(args.source, ffprobe=args.ffprobe), stdout=args.output)
            read_clock(args.output)
        else:
            canvas = Canvas(args.width, args.height, args.rate)
            configs = foreground_configs(
                read_clock(args.clock),
                source=args.source.resolve(strict=True),
                native=args.native.resolve(strict=True),
                canvas=canvas,
            )
            for path in write_foreground_plan(args.directory, configs, workers=args.workers):
                print(path)
    except (ReplayError, OSError, ValueError, subprocess.CalledProcessError) as error:
        print(f"DELIVERY_MEDIA_ERROR {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
