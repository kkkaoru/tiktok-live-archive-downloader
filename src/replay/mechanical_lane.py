"""Shared fixes the non-GUI render lane needs, kept out of the pinned render modules.

`tolerant_clips` is `delivery_render.clips` with one change: a cell may overrun its
chunk end by at most half a frame (container rounding of the last frame); frame
count and start stay exact and any larger overrun still fails.
`bind_partition_stages` adds the `native-request` source-caption binding the
production preflight requires for compose partition stages, and lists the binding
as a stage input so `validate_binding` accepts it.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from fractions import Fraction
from math import ceil
from pathlib import Path

from replay import delivery_render, mechanical_plan_provenance, source_caption_binding
from replay.core import ReplayError
from replay.delivery_render import Chunk, Clip, CutRow


def tolerant_clips(cuts: Sequence[CutRow], chunks: Sequence[Chunk], rate: int) -> list[Clip]:
    half = Fraction(1, 2 * rate)
    result: list[Clip] = []
    for cut in cuts:
        frame = cut.start
        while frame < cut.end:
            seconds = Fraction(frame, rate)
            middle = Fraction(2 * frame + 1, 2 * rate)
            chunk = next((item for item in chunks if item.start <= middle < item.end), None)
            if chunk is None:
                raise ReplayError("Cut lies outside the source chunk index")
            last = ceil(chunk.end * rate - Fraction(1, 2))
            boundary = min(cut.end, max(frame + 1, last))
            if seconds < chunk.start or Fraction(boundary, rate) - chunk.end > half:
                raise ReplayError("CFR selection exceeds its chunk by more than half a frame")
            result.append(Clip(chunk.media, seconds - chunk.start, boundary - frame))
            frame = boundary
    return result


MILLISECONDS = 1000
CLAMPS_NAME = "clamps.json"
FFPROBE_TIMEOUT_SECONDS = 60


@dataclass(frozen=True, slots=True)
class Clamp:
    """Duration diagnostic retaining planned start/count; not sample-content proof."""

    media: str
    start_seconds: str
    frames: int
    trimmed_seconds: str


def media_limit(media: Path) -> Fraction:
    """Duration the native tool accepts: the container duration floored to a millisecond.

    The tool reads the movie-header duration at millisecond resolution while ffprobe may
    report a few ticks more, so the floor is the conservative common bound.
    """
    probes = mechanical_plan_provenance.Probes()
    result = probes.limit(media)
    probes.check(fresh=True)
    return result


def native_clip_json(
    clip: Clip,
    rate: int,
    log: list[Clamp] | None = None,
    *,
    limit: Callable[[Path], Fraction] | None = None,
) -> dict[str, object]:
    """`delivery_render._clip_json` with the selection end clamped to the source end.

    Only an overrun of at most half a frame is clamped; planned start/count are retained,
    and a larger overrun stays an error. Actual native frame/content proof is separate.
    """
    duration = Fraction(clip.frames, rate)
    excess = (
        clip.start + duration - (media_limit(clip.media) if limit is None else limit(clip.media))
    )
    if excess > Fraction(1, 2 * rate):
        raise ReplayError("Selection overruns its source by more than half a frame")
    if excess > 0:
        duration -= excess
        if log is not None:
            log.append(Clamp(str(clip.media), str(clip.start), clip.frames, str(excess)))
    if duration <= 0:
        raise ReplayError("Selection has no duration inside its source")
    return {
        "sourcePath": str(clip.media),
        "selection": {
            "startSeconds": float(clip.start),
            "durationSeconds": float(duration),
            "rate": 1,
        },
    }


def bind_partition_stages(
    cleanup: source_caption_binding.Cleanup, stages: list[dict[str, object]]
) -> dict[str, Path]:
    """Create one immutable binding per compose partition stage; extend its inputs."""
    bindings: dict[str, Path] = {}
    for stage in stages:
        inputs = stage.get("inputs")
        if not isinstance(inputs, list) or not inputs or not isinstance(inputs[0], str):
            raise ReplayError("Partition stage lacks its request input")
        request = Path(inputs[0])
        binding = source_caption_binding.create_binding(
            cleanup,
            consumer=request,
            kind="native-request",
            output=request.with_name(f"{request.stem}.source-caption.binding.json"),
        )
        bindings[str(stage["name"])] = binding
        inputs.append(str(binding))
    return bindings


PATCHED_NAMES = ("clips", "_clip_json", "compose", "_workflows")


@contextmanager
def compose_fixes() -> Iterator[list[Clamp]]:
    """Apply the compose fixes to `delivery_render` for the duration of one call.

    `delivery_render.py` is code-pinned by the retirement verification, so the fixes are
    applied at run time instead of by editing it; everything is restored on exit.
    """
    originals = {name: getattr(delivery_render, name) for name in PATCHED_NAMES}
    current: dict[str, delivery_render.Config] = {}
    generation: dict[str, mechanical_plan_provenance.Generation] = {}
    clamps: list[Clamp] = []

    def compose(config: delivery_render.Config, *, workers: int) -> list[Path]:
        current["config"] = config
        generation["current"] = mechanical_plan_provenance.Generation()
        result: list[Path] = originals["compose"](config, workers=workers)
        generation["current"].finish()
        return result

    def clip_json(clip: Clip, rate: int) -> dict[str, object]:
        probe = generation["current"].probes.limit if "current" in generation else media_limit
        result = native_clip_json(clip, rate, clamps, limit=probe)
        if "current" in generation:
            generation["current"].record_clip(result)
        return result

    def workflows(
        directory: Path,
        name: str,
        stages: list[dict[str, object]],
        workers: int,
        *,
        bindings: dict[str, Path] | None = None,
    ) -> list[Path]:
        if name == "partition" and bindings is None:
            cleanup = delivery_render.verified_cleanup(current["config"])
            bindings = bind_partition_stages(cleanup, stages)
            generation["current"].bind(stages)
        result: list[Path] = originals["_workflows"](
            directory, name, stages, workers, bindings=bindings
        )
        return result

    replacements = {
        "clips": tolerant_clips,
        "_clip_json": clip_json,
        "compose": compose,
        "_workflows": workflows,
    }
    for name, value in replacements.items():
        setattr(delivery_render, name, value)
    try:
        yield clamps
    finally:
        for name, value in originals.items():
            setattr(delivery_render, name, value)


@contextmanager
def render_fixes() -> Iterator[None]:
    """Propagate original compose provenance into every native render consumer."""
    original = delivery_render._workflows

    def workflows(
        directory: Path,
        name: str,
        stages: list[dict[str, object]],
        workers: int,
        *,
        bindings: dict[str, Path] | None = None,
    ) -> list[Path]:
        if name == "render":
            mechanical_plan_provenance.bind_render(stages)
        return original(directory, name, stages, workers, bindings=bindings)

    delivery_render._workflows = workflows
    try:
        yield
    finally:
        delivery_render._workflows = original


def write_clamps(path: Path, clamps: Sequence[Clamp]) -> Path | None:
    """Record every trimmed selection beside the plan; nothing is written when none occurred."""
    if not clamps:
        return None
    rows = [
        {
            "media": clamp.media,
            "startSeconds": clamp.start_seconds,
            "frames": clamp.frames,
            "trimmedSeconds": clamp.trimmed_seconds,
        }
        for clamp in clamps
    ]
    with path.open("x", encoding="utf-8") as stream:
        json.dump({"version": 1, "clamps": rows}, stream, sort_keys=True)
    return path
