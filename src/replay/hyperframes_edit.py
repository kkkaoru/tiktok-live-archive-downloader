"""Local-only HyperFrames companion for validated footage, cuts and SRT overlays.

Audio separation, inpainting, subtitle tracks and delivery verification remain in
existing processors. Projects live under edits/ and are never publishable assets.
"""

import argparse
import html
import json
import os
import re
import shutil
import subprocess
from dataclasses import dataclass
from math import isfinite
from pathlib import Path

from replay.core import ReplayError

_TIME = re.compile(r"(\d{2}):(\d{2}):(\d{2}),(\d{3})")
_MAX_CAPTIONS = 2500


@dataclass(frozen=True, slots=True)
class Clip:
    start: float
    end: float

    def __post_init__(self) -> None:
        if not all(isfinite(value) for value in (self.start, self.end)) or not (
            0 <= self.start < self.end <= 86400
        ):
            raise ReplayError("Invalid source clip")


@dataclass(frozen=True, slots=True)
class Cue:
    start: float
    end: float
    text: str


def _seconds(value: str) -> float:
    match = _TIME.fullmatch(value)
    if match is None:
        raise ReplayError("Invalid SRT timestamp")
    hours, minutes, seconds, milliseconds = map(int, match.groups())
    if minutes >= 60 or seconds >= 60:
        raise ReplayError("Invalid SRT clock")
    return hours * 3600 + minutes * 60 + seconds + milliseconds / 1000


def parse_srt(text: str) -> tuple[Cue, ...]:
    """Accept timed text only, never promote unknown or empty cues to captions."""
    cues: list[Cue] = []
    for block in re.split(r"\r?\n\r?\n", text.strip("\r\n")):
        rows = block.splitlines()
        if len(rows) < 3 or not rows[0].isdigit():
            raise ReplayError("Malformed SRT cue")
        times = rows[1].split(" --> ")
        if len(times) != 2:
            raise ReplayError("Malformed SRT interval")
        start, end = map(_seconds, times)
        caption = "\n".join(rows[2:]).strip()
        if not caption or not 0 <= start < end <= 86400:
            raise ReplayError("Invalid SRT cue content or interval")
        if cues and start < cues[-1].start:
            raise ReplayError("SRT cues must be ordered")
        cues.append(Cue(start, end, caption))
        if len(cues) > 100_000:
            raise ReplayError("SRT cue budget exceeded")
    return tuple(cues)


def _probe(path: Path) -> tuple[float, int, int]:
    result = subprocess.run(
        ["ffprobe", "-v", "error", "-show_format", "-show_streams", "-of", "json", str(path)],
        capture_output=True,
        text=True,
        check=True,
        timeout=30,
    )
    data: object = json.loads(result.stdout)
    if not isinstance(data, dict) or not isinstance(data.get("streams"), list):
        raise ReplayError("Invalid source media probe")
    video = next(
        (
            stream
            for stream in data["streams"]
            if isinstance(stream, dict) and stream.get("codec_type") == "video"
        ),
        None,
    )
    source_format = data.get("format")
    if not isinstance(video, dict) or not isinstance(source_format, dict):
        raise ReplayError("A video stream and duration are required")
    duration = float(source_format["duration"])
    width, height = video.get("width"), video.get("height")
    if (
        not isfinite(duration)
        or duration <= 0
        or type(width) is not int
        or type(height) is not int
        or not 1 <= width <= 4096
        or not 1 <= height <= 4096
    ):
        raise ReplayError("Unsupported video dimensions or duration")
    return duration, width, height


def _clips(value: object, duration: float) -> tuple[Clip, ...]:
    if not isinstance(value, list) or not 1 <= len(value) <= 3000:
        raise ReplayError("Expected 1-3000 source clips")
    result: list[Clip] = []
    for item in value:
        if (
            not isinstance(item, list)
            or len(item) != 2
            or any(type(number) not in (int, float) for number in item)
        ):
            raise ReplayError("Each clip needs numeric source start and end")
        clip = Clip(float(item[0]), float(item[1]))
        if clip.end > duration + 0.001:
            raise ReplayError("Clip extends beyond source")
        result.append(clip)
    return tuple(result)


def composition(
    source: str, clips: tuple[Clip, ...], cues: tuple[Cue, ...], *, width: int, height: int
) -> str:
    """Generate a seek-safe local composition; no runtime requests or ASR guesses."""
    cursor = 0.0
    visual: list[str] = []
    caption: list[str] = []
    for index, clip in enumerate(clips):
        length = clip.end - clip.start
        timing = (
            f'data-start="{cursor:.3f}" data-duration="{length:.3f}" '
            f'data-media-start="{clip.start:.3f}"'
        )
        visual.append(
            f'<video id="shot-{index}" class="clip" src="{html.escape(source, quote=True)}" '
            f'{timing} data-track-index="0" muted playsinline></video>'
            f'<audio id="sound-{index}" class="clip" src="{html.escape(source, quote=True)}" '
            f'{timing} data-track-index="10" data-volume="1"></audio>'
        )
        for cue in cues:
            start = max(cue.start, clip.start)
            end = min(cue.end, clip.end)
            if start >= end:
                continue
            caption.append(
                f'<div id="caption-{len(caption)}" class="clip caption" '
                f'data-start="{cursor + start - clip.start:.3f}" '
                f'data-duration="{end - start:.3f}" data-track-index="20">'
                f"{html.escape(cue.text)}</div>"
            )
            if len(caption) > _MAX_CAPTIONS:
                raise ReplayError("Too many captions for a local HyperFrames composition")
        cursor += length
    return f'''<!doctype html>
<html lang="ja"><head><meta charset="utf-8"><title>Local footage edit</title>
<script src="assets/gsap.min.js"></script><style>
html,body{{margin:0;width:100%;height:100%;background:#000}}
#main{{width:100%;height:100%;position:relative;overflow:hidden}}
video{{position:absolute;inset:0;width:100%;height:100%;object-fit:contain}}
.caption{{position:absolute;left:8%;right:8%;bottom:12%;padding:.15em .35em;
 text-align:center;color:#fff;background:rgba(0,0,0,.88);
 font:600 48px/1.3 system-ui,sans-serif;white-space:pre-line}}
</style></head><body>
<div id="main" data-composition-id="main" data-width="{width}" data-height="{height}"
 data-duration="{cursor:.3f}" data-start="0">
{"".join(visual)}
{"".join(caption)}
</div><script>window.__timelines["main"] = gsap.timeline({{paused:true}});</script>
</body></html>
'''


def build(config: Path) -> Path:
    """Never modify source, prior projects, or another delivery family."""
    data: object = json.loads(config.read_text(encoding="utf-8"))
    if not isinstance(data, dict) or set(data) not in (
        {"version", "source", "project", "clips", "srt", "gsap"},
        {"version", "source", "project", "clips", "srt", "gsap", "caption_offset"},
    ):
        raise ReplayError("Invalid HyperFrames edit configuration")
    if type(data["version"]) is not int or data["version"] != 1:
        raise ReplayError("Unsupported HyperFrames edit version")
    if not isinstance(data["source"], str) or not isinstance(data["project"], str):
        raise ReplayError("Absolute source and project paths are required")
    source, project = Path(data["source"]), Path(data["project"])
    if not source.is_absolute() or not project.is_absolute() or not source.is_file():
        raise ReplayError("Source must be a local file and project must be absolute")
    edits = Path.cwd().resolve() / "edits"
    if project.is_symlink() or project.exists():
        raise ReplayError("Refusing to overwrite a project")
    if not project.parent.is_dir() or project.parent.is_symlink():
        raise ReplayError("Project parent must already exist and not be a symlink")
    if project.parent.resolve() != edits and edits not in project.parent.resolve().parents:
        raise ReplayError("Local projects must stay under this checkout's edits directory")
    if not source.name.lower().endswith((".mp4", ".mov", ".mkv")):
        raise ReplayError("Unsupported source container")
    gsap = data["gsap"]
    if not isinstance(gsap, str) or not Path(gsap).is_file() or Path(gsap).stat().st_size > 300_000:
        raise ReplayError("A pinned local GSAP script is required")
    srt_path = data["srt"]
    if srt_path is not None and (not isinstance(srt_path, str) or not Path(srt_path).is_file()):
        raise ReplayError("Explicit SRT must be an existing file")
    offset = data.get("caption_offset", 0)
    if type(offset) not in (int, float) or not isfinite(offset) or not 0 <= offset < 86400:
        raise ReplayError("Invalid source caption offset")
    duration, width, height = _probe(source)
    clips = _clips(data["clips"], duration)
    cues: tuple[Cue, ...] = ()
    if isinstance(srt_path, str):
        if Path(srt_path).stat().st_size > 10_000_000:
            raise ReplayError("SRT file budget exceeded")
        cues = tuple(
            Cue(cue.start - offset, cue.end - offset, cue.text)
            for cue in parse_srt(Path(srt_path).read_text(encoding="utf-8-sig"))
            if cue.end > offset and cue.start < offset + duration
        )
    page = composition(
        "assets/source" + source.suffix.lower(), clips, cues, width=width, height=height
    )
    project.mkdir(mode=0o700)
    (project / "assets").mkdir(mode=0o700)
    os.symlink(source.resolve(strict=True), project / "assets" / ("source" + source.suffix.lower()))
    shutil.copyfile(gsap, project / "assets/gsap.min.js")
    (project / "index.html").write_text(page, encoding="utf-8")
    (project / "hyperframes.json").write_text('{"name":"local-footage-edit"}\n', encoding="utf-8")
    return project


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare a private HyperFrames video-edit project")
    parser.add_argument("config", type=Path)
    args = parser.parse_args()
    print(build(args.config))


if __name__ == "__main__":
    main()
