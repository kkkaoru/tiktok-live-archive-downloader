"""Render one planned vertical part with a single FFmpeg pass.

The plan's native request (source clips and 30 fps fit canvas) is
reproduced with FFmpeg filters, and the part's caption ASS is burned in the same
encode. Compared with a native render followed by a caption re-encode, the part is
encoded once and no external render service is needed. Clip selections keep the
plan's source seconds; each clip is resampled to the output frame rate and holds its
last frame to fill its planned span, as the native renderer does, and the part is
cut to its planned frames.
"""

from __future__ import annotations

import argparse
import json
import math
import subprocess
from dataclasses import dataclass
from pathlib import Path

from replay.core import ReplayError
from replay.recordings import record

ENCODER = [
    "-c:v",
    "h264_videotoolbox",
    "-b:v",
    "6000k",
    "-maxrate",
    "7000k",
    "-bufsize",
    "14000k",
    "-bf",
    "0",
    "-profile:v",
    "high",
    "-pix_fmt",
    "yuv420p",
    "-tag:v",
    "avc1",
    "-color_primaries",
    "bt709",
    "-color_trc",
    "bt709",
    "-colorspace",
    "bt709",
    "-video_track_timescale",
    "15360",
    "-movflags",
    "+faststart",
]


@dataclass(frozen=True, slots=True)
class Clip:
    source: Path
    start: float
    duration: float


@dataclass(frozen=True, slots=True)
class Part:
    clips: tuple[Clip, ...]
    width: int
    height: int
    rate: int
    frames: int


def _number(value: object, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ReplayError(f"Invalid part {label}")
    return float(value)


def _integer(value: object, label: str) -> int:
    if type(value) is not int or value <= 0:
        raise ReplayError(f"Invalid part {label}")
    return value


def load_part(raw: object) -> Part:
    """Validate one render-plan entry produced by prepare-render."""
    entry = record(raw)
    recipe = record(record(entry.get("request")).get("recipe"))
    video = record(recipe.get("video"))
    if video.get("resizeMode") != "fit":
        raise ReplayError("Only fit-to-canvas parts can be rendered")
    clips: list[Clip] = []
    raw_clips = recipe.get("clips")
    for item in raw_clips if isinstance(raw_clips, list) else []:
        clip = record(item)
        selection = record(clip.get("selection"))
        source = clip.get("sourcePath")
        if not isinstance(source, str) or _number(selection.get("rate"), "rate") != 1:
            raise ReplayError("Invalid part clip")
        start = _number(selection.get("startSeconds"), "clip start")
        duration = _number(selection.get("durationSeconds"), "clip duration")
        if start < 0 or duration <= 0:
            raise ReplayError("Invalid part clip span")
        clips.append(Clip(Path(source), start, duration))
    if not clips:
        raise ReplayError("Part has no source clips")
    raw_masks = video.get("masks")
    if raw_masks is not None and raw_masks != []:
        raise ReplayError("Fixed source-caption blur is unsupported; use detected local repair")
    return Part(
        tuple(clips),
        _integer(video.get("width"), "width"),
        _integer(video.get("height"), "height"),
        _integer(video.get("frameRate"), "frame rate"),
        _integer(entry.get("durationFrames"), "frame count"),
    )


def _escape(path: Path) -> str:
    return str(path).replace("\\", "\\\\").replace("'", "\\'").replace(":", "\\:")


def filtergraph(part: Part, ass: Path | None) -> str:
    """Join locally repaired clips, resample to the output rate, then burn captions."""
    # Each clip fills exactly its planned span on the output clock, like the native
    # renderer: a source gap or an early last frame is held, never shortens the part.
    steps = [
        f"[{index}:v:0]setpts=PTS-STARTPTS,fps={part.rate},"
        f"tpad=stop_mode=clone:stop_duration={clip.duration:g},"
        f"trim=duration={clip.duration:g},setpts=PTS-STARTPTS,"
        f"scale={part.width}:{part.height}:force_original_aspect_ratio=decrease,"
        f"pad={part.width}:{part.height}:(ow-iw)/2:(oh-ih)/2,setsar=1[c{index}]"
        for index, clip in enumerate(part.clips)
    ]
    joined = "".join(f"[c{index}]" for index in range(len(part.clips)))
    steps.append(f"{joined}concat=n={len(part.clips)}:v=1:a=0[v0]")
    captions = f",ass=filename='{_escape(ass)}'" if ass is not None else ""
    steps.append(f"[v0]format=yuv420p{captions}[out]")
    return ";".join(steps)


def command(part: Part, ass: Path | None, output: Path, ffmpeg: str) -> list[str]:
    arguments = [ffmpeg, "-v", "error", "-xerror", "-nostdin", "-n"]
    for clip in part.clips:
        arguments += ["-ss", f"{clip.start:g}", "-t", f"{clip.duration:g}", "-i", str(clip.source)]
    arguments += [
        "-filter_complex",
        filtergraph(part, ass),
        "-map",
        "[out]",
        "-an",
        "-frames:v",
        str(part.frames),
        *ENCODER,
        str(output),
    ]
    return arguments


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("plan", type=Path, help="render-plan.json")
    parser.add_argument("index", type=int)
    parser.add_argument("output", type=Path)
    parser.add_argument("--ass", type=Path)
    parser.add_argument("--ffmpeg", default="ffmpeg")
    args = parser.parse_args(argv)
    if args.output.exists() or args.output.is_symlink():
        raise ReplayError(f"Refusing to overwrite {args.output}")
    plan: object = json.loads(args.plan.read_text(encoding="utf-8"))
    if not isinstance(plan, list) or not 0 <= args.index < len(plan):
        raise ReplayError("Part index is outside the render plan")
    subprocess.run(
        command(load_part(plan[args.index]), args.ass, args.output, args.ffmpeg), check=True
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
