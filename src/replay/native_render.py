"""Native single-pass render requests from the same timeline as the editable FCPXML.

Each render-plan part becomes one ``media_edit`` request for the apple-pro-apps
native MCP: source cuts trimmed to the planned frames and the
burned ASS captions as positioned ``styledCaptions`` with a libass-compatible
appearance. The output is video-only and encoded once by the hardware encoder.
Requests are grouped into ``media_edit_batch`` payloads. An explicit hash-bound
foreground manifest can replace styled captions with already rendered per-part
Motion media. This is not a replacement for the separate editable FCP project;
these requests and the native sidecar do not certify editability or completion.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path

from replay.core import ReplayError
from replay.fcp_project import Caption, CaptionStyle, Cut, Timeline, load_timeline
from replay.stages import file_digest

DEFAULT_CODEC = "hevc"
# 4.5 Mbps HEVC measured smaller than the FFmpeg H.264 part render at VMAF >= 95.
DEFAULT_BIT_RATE = 4_500_000
CAPTION_FONT = "HiraginoSans-W6"
BATCH_LIMIT = 32


@dataclass(frozen=True, slots=True)
class Encoding:
    codec: str = DEFAULT_CODEC
    average_bit_rate: int = DEFAULT_BIT_RATE
    allow_frame_reordering: bool = True


@dataclass(frozen=True, slots=True)
class PartWindow:
    """One part's half-open output interval on the global clock."""

    index: int
    start: Fraction
    end: Fraction


def _seconds(value: Fraction) -> float:
    return float(value)


def part_windows(plan: Sequence[object], rate: int) -> list[PartWindow]:
    """Part intervals from the plan's contiguous start and duration frames."""
    windows: list[PartWindow] = []
    for index, raw in enumerate(plan):
        if not isinstance(raw, dict):
            raise ReplayError("Invalid render plan part")
        start, frames = raw.get("startFrame"), raw.get("durationFrames")
        if type(start) is not int or type(frames) is not int:
            raise ReplayError("Invalid render plan frames")
        windows.append(PartWindow(index, Fraction(start, rate), Fraction(start + frames, rate)))
    return windows


def _overlap(
    start: Fraction, end: Fraction, window: PartWindow
) -> tuple[Fraction, Fraction] | None:
    low, high = max(start, window.start), min(end, window.end)
    return (low, high) if high > low else None


def _clips(cuts: Sequence[Cut], window: PartWindow) -> list[dict[str, object]]:
    clips: list[dict[str, object]] = []
    for cut in cuts:
        span = _overlap(cut.output_start, cut.output_start + cut.duration, window)
        if span is None:
            continue
        clips.append(
            {
                "sourcePath": str(cut.source),
                "selection": {
                    "startSeconds": _seconds(cut.source_start + span[0] - cut.output_start),
                    "durationSeconds": _seconds(span[1] - span[0]),
                    "rate": 1,
                },
            }
        )
    return clips


def _captions(
    captions: Sequence[Caption], window: PartWindow
) -> tuple[list[dict[str, object]], set[str]]:
    """The window's positioned captions and the fill colors they use."""
    result: list[dict[str, object]] = []
    fills: set[str] = set()
    for caption in captions:
        span = _overlap(caption.start, caption.end, window)
        if span is None:
            continue
        fills.add(caption.fill)
        result.append(
            {
                "text": "\n".join(caption.lines),
                "startSeconds": _seconds(span[0] - window.start),
                "endSeconds": _seconds(span[1] - window.start),
                "x": caption.x,
                "bottom": caption.bottom,
            }
        )
    return result, fills


def _appearance(style: CaptionStyle, fill: str) -> dict[str, object]:
    return {
        "fontName": CAPTION_FONT,
        "assFontSize": style.ass_size,
        "fill": fill,
        "border": style.border,
        "borderColor": style.border_color,
        "rim": style.rim,
        "rimColor": style.rim_color,
    }


def part_request(
    timeline: Timeline,
    window: PartWindow,
    output_directory: Path,
    encoding: Encoding,
    *,
    foreground: Path | None = None,
) -> dict[str, object]:
    """One video-only request; foreground must already use this part's zero-based clock.

    Native decoding enforces canvas, duration and every frame before publication.
    The caller binds content/provenance; a path alone is not a render receipt.
    """
    if foreground is not None and (not foreground.is_absolute() or "\0" in str(foreground)):
        raise ReplayError("Foreground must be an absolute local path")
    if timeline.masks:
        raise ReplayError("Fixed source-caption blur is unsupported; use detected local repair")
    video: dict[str, object] = {
        "width": timeline.width,
        "height": timeline.height,
        "frameRate": timeline.rate,
        "resizeMode": "fit",
        "encoding": {
            "codec": encoding.codec,
            "averageBitRate": encoding.average_bit_rate,
            "allowFrameReordering": encoding.allow_frame_reordering,
        },
    }
    captions, fills = _captions(timeline.captions, window)
    if foreground is not None:
        video["foregroundVideoPath"] = str(foreground)
    elif captions:
        if timeline.style is None:
            raise ReplayError("Captions need a caption style")
        if len(fills) != 1:
            raise ReplayError("Native styled captions need one fill color per part")
        video["styledCaptions"] = captions
        video["captionAppearance"] = _appearance(timeline.style, fills.pop())
    return {
        "recipe": {"clips": _clips(timeline.cuts, window), "video": video},
        "outputDirectory": str(output_directory),
        "outputName": f"part-{window.index:06d}.mp4",
    }


def load_foregrounds(
    manifest: Path, *, plan: Path, windows: Sequence[PartWindow], rate: int
) -> tuple[Path, ...]:
    """Bind one explicitly supplied foreground per planned part; never trim or retime it.

    This verifies file identity and declared mapping, not decoded frame clocks,
    alpha, caption quality or correspondence to an editable Motion project.
    """
    if manifest.is_symlink() or not manifest.is_file() or manifest.stat().st_size > 1_048_576:
        raise ReplayError("Expected a bounded regular foreground manifest")
    value: object = json.loads(manifest.read_text(encoding="utf-8"))
    if (
        not isinstance(value, dict)
        or set(value) != {"version", "renderPlanSHA256", "parts"}
        or type(value["version"]) is not int
        or value["version"] != 1
        or value["renderPlanSHA256"] != file_digest(plan)
    ):
        raise ReplayError("Foreground manifest does not match the render plan")
    parts: object = value["parts"]
    if not isinstance(parts, list) or len(parts) != len(windows):
        raise ReplayError("Foreground manifest needs exactly one entry per part")
    paths: list[Path] = []
    for item, window in zip(parts, windows, strict=True):
        if (
            not isinstance(item, dict)
            or set(item) != {"startFrame", "durationFrames", "path", "sha256"}
            or type(item["startFrame"]) is not int
            or type(item["durationFrames"]) is not int
            or item["startFrame"] != window.start * rate
            or item["durationFrames"] != (window.end - window.start) * rate
        ):
            raise ReplayError("Foreground part clock does not match the render plan")
        name: object = item["path"]
        digest: object = item["sha256"]
        if not isinstance(name, str) or "\0" in name or not Path(name).is_absolute():
            raise ReplayError("Foreground must be an absolute local path")
        path = Path(name)
        if not isinstance(digest, str) or file_digest(path) != digest:
            raise ReplayError("Foreground file fingerprint differs")
        paths.append(path)
    return tuple(paths)


def batches(
    requests: Sequence[dict[str, object]], size: int = BATCH_LIMIT
) -> Iterator[list[dict[str, object]]]:
    """``media_edit_batch`` item lists of at most ``size`` requests, in order."""
    if not 1 <= size <= BATCH_LIMIT:
        raise ReplayError("Batch size must be 1-32")
    for start in range(0, len(requests), size):
        yield list(requests[start : start + size])


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("plan", type=Path, help="render-plan.json")
    parser.add_argument("output", type=Path, help="new JSON file of media_edit_batch payloads")
    parser.add_argument(
        "--render-output", type=Path, required=True, help="absolute output directory"
    )
    parser.add_argument("--render-dir", type=Path, help="directory holding the plan's ASS files")
    parser.add_argument(
        "--foreground-manifest",
        type=Path,
        help="hash-bound, zero-based foreground per part instead of styled captions",
    )
    parser.add_argument("--bit-rate", type=int, default=DEFAULT_BIT_RATE)
    parser.add_argument("--codec", choices=("h264", "hevc"), default=DEFAULT_CODEC)
    args = parser.parse_args(argv)
    if args.output.exists() or not args.render_output.is_absolute():
        raise ReplayError("Refusing to overwrite output or use a relative render directory")
    plan: object = json.loads(args.plan.read_text(encoding="utf-8"))
    if not isinstance(plan, list):
        raise ReplayError("Render plan is empty")
    timeline = load_timeline(plan, args.render_dir or args.plan.parent)
    encoding = Encoding(args.codec, args.bit_rate)
    windows = part_windows(plan, timeline.rate)
    foregrounds = (
        load_foregrounds(
            args.foreground_manifest, plan=args.plan, windows=windows, rate=timeline.rate
        )
        if args.foreground_manifest is not None
        else None
    )
    requests = [
        part_request(
            timeline,
            window,
            args.render_output,
            encoding,
            foreground=None if foregrounds is None else foregrounds[window.index],
        )
        for window in windows
    ]
    payloads = [{"items": items, "concurrency": 3} for items in batches(requests)]
    with args.output.open("x", encoding="utf-8") as handle:
        json.dump(payloads, handle, ensure_ascii=False)
    print(json.dumps({"parts": len(requests), "batches": len(payloads)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
