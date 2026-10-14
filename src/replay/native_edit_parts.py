"""Partition video-only native recipes at existing cuts without changing media clocks."""

import argparse
import hashlib
import json
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path

from replay.caption_measured_job import _pairs
from replay.caption_native_layout import NativeCaptionLayout
from replay.caption_timing import display_time
from replay.foreground_evidence import _bound_bytes
from replay.recordings import record
from replay.stages import file_digest
from replay.tracking_observations import _checked_output

MAX_REQUEST_BYTES = 4 * 1024 * 1024
MAX_CLIPS = 60
MAX_SOURCE_SECONDS = 86400
MAX_PART_SECONDS = 600
MAX_OVERLAYS = 16
STATIC_VIDEO_FIELDS = {"width", "height", "frameRate", "resizeMode", "encoding", "color"}


@dataclass(frozen=True, slots=True)
class Selection:
    start: Fraction
    duration: Fraction


def _time(value: object) -> Fraction:
    if (
        isinstance(value, bool)
        or not isinstance(value, (float, int))
        or not 0 <= value <= MAX_SOURCE_SECONDS
    ):
        raise ValueError("Native edit times must be finite nonnegative seconds")
    return display_time(float(value))


def _selection(value: object) -> Selection:
    data = record(value)
    if set(data) != {"startSeconds", "durationSeconds", "rate"} or (
        isinstance(data["rate"], bool) or data["rate"] != 1
    ):
        raise ValueError("Cut partitioning requires unretimed selections")
    start, duration = _time(data["startSeconds"]), _time(data["durationSeconds"])
    if duration <= 0:
        raise ValueError("Native edit selections must have positive duration")
    return Selection(start, duration)


def _frames(time: Fraction, rate: int) -> int:
    value = time * rate
    if value.denominator != 1:
        raise ValueError("Native output boundaries must already lie on the frame grid")
    return value.numerator


def _layers(
    values: object, *, start: Fraction, end: Fraction, total: Fraction, rate: int
) -> list[dict[str, object]]:
    if not isinstance(values, list) or len(values) > MAX_OVERLAYS:
        raise ValueError("Expected bounded native video layers")
    result: list[dict[str, object]] = []
    for value in values:
        layer = record(value)
        if not {"sourcePath", "selection", "offsetSeconds"} <= set(layer) or set(layer) - {
            "sourcePath",
            "selection",
            "offsetSeconds",
            "opacity",
            "geometry",
        }:
            raise ValueError("Unsupported native video layer fields")
        selected = _selection(layer["selection"])
        offset = _time(layer["offsetSeconds"])
        _frames(offset, rate)
        _frames(selected.duration, rate)
        _frames(selected.start, rate)
        if offset + selected.duration > total:
            raise ValueError("Native video layer exceeds the complete source timeline")
        low, high = max(start, offset), min(end, offset + selected.duration)
        if low < high:
            result.append(
                {
                    **layer,
                    "offsetSeconds": float(low - start),
                    "selection": {
                        "startSeconds": float(selected.start + low - offset),
                        "durationSeconds": float(high - low),
                        "rate": 1,
                    },
                }
            )
    return result


def partition(request: object) -> tuple[tuple[int, int, dict[str, object]], ...]:
    """Limit each reader to one source cut and only intersecting Motion layers.

    Reject timed effects, transitions, extra audio and speed changes rather than
    dropping them. The native service must still validate each resulting recipe
    and its media. This is not render, source-quality or A/V-sync acceptance.
    """
    root = record(request)
    if set(root) != {"recipe", "outputDirectory", "outputName"}:
        raise ValueError("Unsupported native edit request")
    recipe = record(root["recipe"])
    if not {"clips", "video"} <= set(recipe) or set(recipe) - {
        "clips",
        "video",
        "additionalVideo",
        "muteOriginalAudio",
    }:
        raise ValueError("Partitioning supports video-only recipes without extra audio")
    video = record(recipe["video"])
    rate = video.get("frameRate")
    if (
        set(video) - STATIC_VIDEO_FIELDS
        or "encoding" not in video
        or (type(rate) is not int or not 1 <= rate <= 60)
    ):
        raise ValueError("Partitioning requires static video settings and single-pass encoding")
    clips = recipe["clips"]
    if not isinstance(clips, list) or not 1 <= len(clips) <= MAX_CLIPS:
        raise ValueError("Expected bounded nonempty native cuts")
    total = sum((_selection(record(clip).get("selection")).duration for clip in clips), Fraction(0))
    start = Fraction(0)
    result: list[tuple[int, int, dict[str, object]]] = []
    for value in clips:
        clip = record(value)
        if (
            not {"sourcePath", "selection"} <= set(clip)
            or set(clip) - {"sourcePath", "selection", "geometry", "audio", "transitionInSeconds"}
            or clip.get("transitionInSeconds", 0) != 0
        ):
            raise ValueError("Partitioning cannot change transitions or unknown clip operations")
        selected = _selection(clip["selection"])
        frames = _frames(selected.duration, rate)
        if selected.duration > MAX_PART_SECONDS:
            raise ValueError("Source cut exceeds the native render duration budget")
        end = start + selected.duration
        layers = _layers(
            recipe.get("additionalVideo", []), start=start, end=end, total=total, rate=rate
        )
        local = {**recipe, "clips": [clip], "additionalVideo": layers}
        result.append((_frames(start, rate), frames, {**root, "recipe": local}))
        start = end
    return tuple(result)


def export(request: Path, output: Path, *, sha256: str) -> Path:
    request, output = _checked_output(request), _checked_output(output)
    if not output.is_relative_to(Path.cwd().resolve()):
        raise ValueError("Native part configuration storage must stay inside the repository")
    if output.exists():
        raise FileExistsError(output)
    raw = json.loads(_bound_bytes(request, sha256, MAX_REQUEST_BYTES), object_pairs_hook=_pairs)
    parts = partition(raw)
    destination = record(raw).get("outputDirectory")
    if not isinstance(destination, str) or not destination or not Path(destination).is_absolute():
        raise ValueError("Explicit absolute media outputDirectory is required")
    # The request authorizes this media root, not symlink redirection or shared part names.
    media_root = _checked_output(Path(destination))
    namespace = hashlib.sha256(str(output).encode("utf-8")).hexdigest()
    media_parts = _checked_output(media_root / f"parts-{namespace}")
    if media_parts.exists():
        raise FileExistsError(media_parts)
    if file_digest(request) != sha256:
        raise ValueError("Native edit request changed during planning")
    output.mkdir(mode=0o700)
    entries: list[dict[str, object]] = []
    for offset, frames, part in parts:
        directory = output / f"part-{len(entries):04d}"
        directory.mkdir(mode=0o700)
        path = directory / "request.json"
        NativeCaptionLayout._save(
            path, {**part, "outputDirectory": str(media_parts / directory.name)}
        )
        entries.append(
            {
                "path": str(path),
                "sha256": file_digest(path),
                "offsetFrame": offset,
                "frames": frames,
            }
        )
    join_plan = output / "join-plan.json"
    NativeCaptionLayout._save(join_plan, [{"durationFrames": part[1]} for part in parts])
    manifest = output / "manifest.json"
    NativeCaptionLayout._save(
        manifest,
        {
            "version": 1,
            "sourceRequest": {"path": str(request), "sha256": sha256},
            "parts": entries,
            "joinPlan": {"path": str(join_plan), "sha256": file_digest(join_plan)},
            "renderVerified": False,
            "losslessAssemblyVerified": False,
            "deliveryComplete": False,
        },
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("request", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--sha256", required=True)
    args = parser.parse_args()
    export(args.request, args.output, sha256=args.sha256)


if __name__ == "__main__":
    main()
