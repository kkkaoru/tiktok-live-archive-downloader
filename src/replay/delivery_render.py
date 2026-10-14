"""Generate the render-side workflows of a full-recording delivery.

After `replay.delivery_captions` has measured and combined every caption
segment, this module turns one per-recording render configuration into the
next bounded step each time it is called:

1. `motion-export`: one `replay.motion_measured` export per caption segment.
2. `motion-prepare`: `replay.motion_stages` + `replay.motion_finish` per part.
3. `alpha-jobs`: Compressor jobs rendering every Motion part with alpha.
4. `compose`: native edit requests (cut clips + Motion overlays) split into
   bounded chunks, each partitioned by `replay.native_edit_parts`.
5. `render`: native `edit-media` stages for every partitioned part.
6. `join`: link the rendered parts in timeline order and join losslessly.
7. `encode-job`: the Compressor job for the final video encode.
8. `audio`: the `replay.continuous_cut_job` selection of the cut voice track.

Every step derives its inputs from the configuration and earlier outputs; no
step is specific to one recording. Nothing is approved by generation alone.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from dataclasses import dataclass
from fractions import Fraction
from itertools import batched
from math import ceil
from pathlib import Path

from replay import (
    caption_resource_inputs,
    motion_dependencies,
    native_invocation,
    native_join,
    source_caption_binding,
)
from replay.compressor_batch import actual_output
from replay.core import ReplayError
from replay.delivery_output_names import delivery_stem as original_delivery_stem
from replay.delivery_output_names import encoded_name
from replay.recordings import record

MAX_CONFIG_BYTES = 1_048_576
MAX_JSON_BYTES = 256 * 1024 * 1024
MAX_CLIPS = 60
MAX_OVERLAYS = 16
EXPORT_TIMEOUT_SECONDS = 600
PREPARE_TIMEOUT_SECONDS = 300
PARTITION_TIMEOUT_SECONDS = 300
RENDER_TIMEOUT_SECONDS = 1800
JOIN_TIMEOUT_SECONDS = 7200
AUDIO_TIMEOUT_SECONDS = 7200
WORKFLOW_STAGES = 512
CONFIG_KEYS = frozenset(
    {
        "version",
        "recording",
        "captions",
        "work",
        "storage",
        "rate",
        "width",
        "height",
        "cuts",
        "sourceChunks",
        "motion",
        "alphaPreset",
        "encodePreset",
        "native",
        "encoding",
        "audio",
        "source",
        "deliveryRoot",
        "finalPreset",
        "calibration",
        "viewerRoot",
    }
)


@dataclass(frozen=True, slots=True)
class Chunk:
    """A source movie covering source seconds [start, end)."""

    media: Path
    start: Fraction
    end: Fraction


@dataclass(frozen=True, slots=True)
class Clip:
    media: Path
    start: Fraction
    frames: int


@dataclass(frozen=True, slots=True)
class Overlay:
    media: Path
    offset: int
    frames: int


@dataclass(frozen=True, slots=True)
class MotionPart:
    segment: int
    part: int
    plan: Path
    offset: int
    frames: int


@dataclass(frozen=True, slots=True)
class CutRow:
    """Retained source frames [start, end) on the output frame grid."""

    start: int
    end: int


@dataclass(frozen=True, slots=True)
class Config:
    path: Path
    recording: str
    captions: Path
    work: Path
    storage: Path
    rate: int
    width: int
    height: int
    cuts: Path
    chunks: Path
    template: Path
    expected_text: str
    finish: tuple[str, ...]
    alpha_preset: Path
    encode_preset: Path
    native: Path
    encoding: dict[str, object]
    audio_source: Path
    audio_clock: Path
    source: Path
    delivery_root: Path
    final_preset: Path
    calibration: Path
    viewer_root: Path
    composition_directory: Path | None = None
    cut_evidence: tuple[Path, ...] = ()
    resolved_assets: tuple[Path, ...] | None = None
    source_cleanup: Path | None = None

    @property
    def composition(self) -> Path:
        return self.composition_directory or self.work / "compose"


def _json(path: Path, limit: int = MAX_JSON_BYTES) -> object:
    if path.stat().st_size > limit:
        raise ReplayError(f"JSON exceeds its size limit: {path.name}")
    return json.loads(path.read_text(encoding="utf-8"))


def _digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _path(value: object) -> Path:
    if not isinstance(value, str) or not value or not Path(value).is_absolute():
        raise ReplayError("Render configuration paths must be absolute")
    return Path(value)


def _integer(value: object) -> int:
    if type(value) is not int or value <= 0:
        raise ReplayError("Render configuration needs positive integers")
    return value


def load_config(path: Path) -> Config:
    raw = record(_json(path, MAX_CONFIG_BYTES))
    if (
        set(raw) - {"compositionDirectory", "cutEvidence", "sourceCleanup"} != CONFIG_KEYS
        or raw.get("version") != 1
    ):
        raise ReplayError("Render configuration has unknown, missing or unsupported fields")
    motion, audio = record(raw["motion"]), record(raw["audio"])
    finish = motion.get("finish")
    text = motion.get("expectedText")
    if not isinstance(finish, list) or not all(isinstance(item, str) for item in finish):
        raise ReplayError("Motion finish arguments must be strings")
    if not isinstance(text, str) or not text:
        raise ReplayError("Motion template text is required")
    recording = raw["recording"]
    if not isinstance(recording, str) or not recording:
        raise ReplayError("Recording name is required")
    return Config(
        path=path.absolute(),
        recording=recording,
        captions=_path(raw["captions"]),
        work=_path(raw["work"]),
        storage=_path(raw["storage"]),
        rate=_integer(raw["rate"]),
        width=_integer(raw["width"]),
        height=_integer(raw["height"]),
        cuts=_path(raw["cuts"]),
        chunks=_path(raw["sourceChunks"]),
        template=_path(motion.get("template")),
        expected_text=text,
        finish=tuple(str(item) for item in finish),
        resolved_assets=_resolved_assets(motion.get("resolvedAssets")),
        alpha_preset=_path(raw["alphaPreset"]),
        encode_preset=_path(raw["encodePreset"]),
        native=_path(raw["native"]),
        encoding=record(raw["encoding"]),
        audio_source=_path(audio.get("source")),
        audio_clock=_path(audio.get("clock")),
        source=_path(raw["source"]),
        delivery_root=_path(raw["deliveryRoot"]),
        final_preset=_path(raw["finalPreset"]),
        calibration=_path(raw["calibration"]),
        viewer_root=_path(raw["viewerRoot"]),
        cut_evidence=tuple(_path(value) for value in _list(raw.get("cutEvidence", []))),
        composition_directory=(
            _path(raw["compositionDirectory"]) if "compositionDirectory" in raw else None
        ),
        source_cleanup=(
            source_caption_binding._pin(raw["sourceCleanup"]) if "sourceCleanup" in raw else None
        ),
    )


def verified_cleanup(config: Config) -> source_caption_binding.Cleanup:
    """Require the current pinned cleanup assembly for composition and rendering."""
    if config.source_cleanup is None:
        raise ReplayError("Production rendering requires pinned sourceCleanup evidence")
    cleanup = source_caption_binding.require_cleanup(config.path)
    if cleanup.path != config.source_cleanup:
        raise ReplayError("Source cleanup configuration changed after loading")
    if (
        cleanup.original != config.source
        or source_caption_binding._pin(record(_json(cleanup.path)).get("cleanIndex"))
        != config.chunks
    ):
        raise ReplayError("Render source and chunks differ from verified cleanup assembly")
    return cleanup


def _resolved_assets(value: object) -> tuple[Path, ...] | None:
    """Explicit supplier-resolved files; names alone never identify fonts/resources."""
    if value is None:
        return None
    data = record(value)
    if set(data) != {"fonts", "resources"}:
        raise ReplayError("Motion resolvedAssets requires fonts and resources references")
    fonts, resources = _list(data["fonts"]), _list(data["resources"])
    if not fonts or len(fonts) + len(resources) > 64:
        raise ReplayError("Motion resolvedAssets needs fonts within the asset budget")
    paths: list[Path] = []
    for raw in [*fonts, *resources]:
        ref = record(raw)
        if set(ref) != {"path", "sha256"}:
            raise ReplayError("Motion asset requires exact path/sha256 reference")
        path = _path(ref["path"])
        if path != path.resolve() or not path.is_file() or path.stat().st_size == 0:
            raise ReplayError("Motion asset must be a resolved nonempty regular file")
        if _digest(path) != ref["sha256"]:
            raise ReplayError("Motion resolved asset hash changed")
        paths.append(path)
    return tuple(dict.fromkeys(paths))


def preparation_inputs(config: Config, item: MotionPart) -> tuple[Path, ...]:
    """Bind future preparation and submission, never retrofit historical evidence."""
    if config.resolved_assets is None:
        raise ReplayError("Motion resolvedAssets supplier evidence is required before preparation")
    motion = record(record(_json(config.path)).get("motion"))
    evidence: tuple[Path, ...] = ()
    if "resourceEvidence" in motion:
        evidence = tuple(
            caption_resource_inputs.preparation_inputs(config=config.path, plan=item.plan).pins
        )
    elif any(
        not path.is_relative_to(Path(__file__).resolve().parents[2])
        for path in config.resolved_assets
    ):
        raise ReplayError("External Motion assets require resourceEvidence before preparation")
    return tuple(
        dict.fromkeys(
            (
                *motion_dependencies.preparation_inputs(
                    config=config.path,
                    template=config.template,
                    plan=item.plan,
                    calibration=config.calibration,
                ),
                config.captions / "segments.json",
                config.captions / f"combined-{item.segment:04d}.json",
                item.plan.parent / "manifest.json",
                _segment_dir(config, item.segment) / "plans" / "manifest.json",
                Path(__file__).resolve(),
                *config.resolved_assets,
                *evidence,
            )
        )
    )


def _write_new(path: Path, value: object) -> Path:
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        os.fchmod(stream.fileno(), 0o600)
        json.dump(value, stream, ensure_ascii=False, indent=1)
        stream.write("\n")
    return path


def _workflows(
    directory: Path,
    name: str,
    stages: list[dict[str, object]],
    workers: int,
    *,
    bindings: dict[str, Path] | None = None,
) -> list[Path]:
    if not stages:
        raise ReplayError(f"No {name} stages remain")
    paths: list[Path] = []
    for worker in range(min(workers, len(stages))):
        for part, chunk in enumerate(batched(stages[worker::workers], WORKFLOW_STAGES)):
            path = directory / f"{name}-{worker}-{part:04d}.json"
            document: dict[str, object] = {"version": 1, "stages": list(chunk)}
            if bindings is not None:
                document["sourceCaptionProduction"] = {
                    "bindings": {
                        str(stage["name"]): str(bindings[str(stage["name"])]) for stage in chunk
                    }
                }
            paths.append(_write_new(path, document))
    return paths


def segments(config: Config) -> list[dict[str, int]]:
    data = record(_json(config.captions / "segments.json"))
    rows: list[dict[str, int]] = []
    for raw in _list(data.get("segments")):
        row = record(raw)
        rows.append(
            {
                "segment": _int0(row.get("segment")),
                "offset": _int0(row.get("offsetFrame")),
                "frames": _integer(row.get("frames")),
            }
        )
    if not rows:
        raise ReplayError("Caption segment index is empty")
    return rows


def _list(value: object) -> list[object]:
    if not isinstance(value, list):
        raise ReplayError("Expected a JSON list")
    return value


def _int0(value: object) -> int:
    if type(value) is not int or value < 0:
        raise ReplayError("Expected a nonnegative integer")
    return value


def _segment_dir(config: Config, segment: int) -> Path:
    return config.work / "motion" / f"segment-{segment:04d}"


def motion_export(config: Config, *, workers: int) -> list[Path]:
    """One motion_measured export per segment from its combined candidate."""
    stages: list[dict[str, object]] = []
    for row in segments(config):
        number = row["segment"]
        combined = config.captions / f"combined-{number:04d}.json"
        if not combined.is_file():
            raise ReplayError(f"Segment {number} has not been combined")
        plans = _segment_dir(config, number) / "plans"
        plans.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        stages.append(
            {
                "name": f"export-{number:04d}",
                "command": [
                    sys.executable,
                    "-m",
                    "replay.motion_measured",
                    str(combined),
                    str(plans),
                    "--sha256",
                    _digest(combined),
                    "--width",
                    str(config.width),
                    "--height",
                    str(config.height),
                    "--rate",
                    str(config.rate),
                    "--frames",
                    str(row["frames"]),
                    "--expected-text",
                    config.expected_text,
                ],
                "inputs": [str(combined)],
                "outputs": [str(plans / "manifest.json")],
                "timeout_seconds": EXPORT_TIMEOUT_SECONDS,
            }
        )
    return _workflows(config.work / "motion", "export", stages, workers)


def motion_parts(config: Config) -> list[MotionPart]:
    """Every Motion part with its plan and global output offset."""
    parts: list[MotionPart] = []
    for row in segments(config):
        manifest = record(_json(_segment_dir(config, row["segment"]) / "plans" / "manifest.json"))
        for index, raw in enumerate(_list(manifest.get("parts"))):
            part = record(raw)
            parts.append(
                MotionPart(
                    segment=row["segment"],
                    part=index,
                    plan=_path(part.get("path")),
                    offset=row["offset"] + _int0(part.get("offsetFrame")),
                    frames=_integer(part.get("frames")),
                )
            )
    return parts


def _moti(config: Config, segment: int, part: int) -> Path:
    return _segment_dir(config, segment) / f"part-{part:04d}.moti"


def motion_prepare(config: Config, *, workers: int) -> list[Path]:
    """Stage plans become finished Motion documents with the selected style."""
    stages: list[dict[str, object]] = []
    for item in motion_parts(config):
        segment, part, plan = item.segment, item.part, item.plan
        final = _moti(config, segment, part)
        layered = final.with_suffix(".two-layer.moti")
        name = f"{segment:04d}-{part:04d}"
        dependencies = list(map(str, preparation_inputs(config, item)))
        stages.append(
            {
                "name": f"stages-{name}",
                "command": [
                    sys.executable,
                    "-m",
                    "replay.motion_stages",
                    str(config.template),
                    str(plan),
                    str(layered),
                ],
                "inputs": dependencies,
                "outputs": [str(layered)],
                "timeout_seconds": PREPARE_TIMEOUT_SECONDS,
            }
        )
        stages.append(
            {
                "name": f"finish-{name}",
                "command": [
                    sys.executable,
                    "-m",
                    "replay.motion_finish",
                    str(layered),
                    str(final),
                    *config.finish,
                ],
                "inputs": [str(layered), *dependencies],
                "outputs": [str(final)],
                "timeout_seconds": PREPARE_TIMEOUT_SECONDS,
            }
        )
    paired = [stages[index : index + 2] for index in range(0, len(stages), 2)]
    paths: list[Path] = []
    for worker in range(min(workers, len(paired))):
        ordered = [stage for pair in paired[worker::workers] for stage in pair]
        for part, chunk in enumerate(batched(ordered, WORKFLOW_STAGES)):
            path = config.work / "motion" / f"prepare-{worker}-{part:04d}.json"
            motion = record(record(_json(config.path)).get("motion"))
            if "resourceEvidence" in motion:
                preparation = record(record(motion["resourceEvidence"]).get("preparation"))
                if preparation.get("workflow") != str(path):
                    raise ReplayError(
                        "Configured resource preparation workflow differs from output"
                    )
            paths.append(_write_new(path, {"version": 1, "stages": list(chunk)}))
    if not paths:
        raise ReplayError("No motion-prepare stages remain")
    return paths


def _alpha(config: Config, segment: int, part: int) -> Path:
    return config.storage / "alpha" / f"segment-{segment:04d}-part-{part:04d}.mov"


def alpha_dependencies(config: Config, item: MotionPart) -> tuple[Path, ...]:
    """Only completed fresh preparation can authorize external alpha resources."""
    inputs = preparation_inputs(config, item)
    motion = record(record(_json(config.path)).get("motion"))
    if "resourceEvidence" not in motion:
        return inputs
    completed = caption_resource_inputs.completed_preparation(
        config=config.path, plan=item.plan, source=_moti(config, item.segment, item.part)
    )
    return tuple(dict.fromkeys((*inputs, *completed.pins)))


def alpha_jobs(config: Config) -> Path:
    jobs = [
        {
            "name": f"alpha-{item.segment:04d}-{item.part:04d}",
            "source": str(_moti(config, item.segment, item.part)),
            "preset": str(config.alpha_preset),
            "output": str(_alpha(config, item.segment, item.part)),
            "dependencies": list(map(str, alpha_dependencies(config, item))),
        }
        for item in motion_parts(config)
    ]
    return _write_new(config.work / "alpha-jobs.json", {"jobs": jobs})


def read_chunks(path: Path) -> list[Chunk]:
    """Source chunk index rows: media, sourceFirstPTS, sourceEndPTS and timeBase."""
    rows = record(_json(path)).get("ranges")
    if not isinstance(rows, list) or not rows:
        raise ReplayError("Source chunk index is empty")
    chunks: list[Chunk] = []
    for raw in rows:
        row = record(raw)
        base = row.get("timeBase")
        if not isinstance(base, str):
            raise ReplayError("Source chunk needs a time base")
        scale = Fraction(base)
        if scale <= 0:
            raise ReplayError("Source chunk time base must be positive")
        start, end = row.get("sourceFirstPTS"), row.get("sourceEndPTS")
        if type(start) is not int or type(end) is not int or end <= start:
            raise ReplayError("Source chunk needs increasing integer PTS")
        chunks.append(Chunk(_path(row.get("media")), start * scale, end * scale))
    for previous, current in zip(chunks, chunks[1:], strict=False):
        if current.start != previous.end:
            raise ReplayError("Source chunks must be contiguous")
    return chunks


def read_cuts(path: Path) -> list[CutRow]:
    rows = record(_json(path)).get("cuts")
    if not isinstance(rows, list) or not rows:
        raise ReplayError("Cut plan is empty")
    cuts: list[CutRow] = []
    for raw in rows:
        row = record(raw)
        cut = CutRow(_int0(row.get("source_start_frame")), _integer(row.get("source_end_frame")))
        if cut.end <= cut.start or (cuts and cut.start < cuts[-1].end):
            raise ReplayError("Cuts must be positive, ordered and non-overlapping")
        cuts.append(cut)
    return cuts


def clips(cuts: list[CutRow], chunks: list[Chunk], rate: int) -> list[Clip]:
    """Map frame-grid cuts onto chunk media; a crossing cut is split on the grid.

    Midpoint ownership alone cannot represent a CFR cell crossing a VFR seam.
    Reject such selections until a verified frame-clock mapping is supplied;
    never shift the start, truncate the duration or extend the source.
    """
    result: list[Clip] = []
    for cut in cuts:
        start, end = cut.start, cut.end
        frame = start
        while frame < end:
            seconds = Fraction(frame, rate)
            middle = Fraction(2 * frame + 1, 2 * rate)
            chunk = next((item for item in chunks if item.start <= middle < item.end), None)
            if chunk is None:
                raise ReplayError("Cut lies outside the source chunk index")
            last = ceil(chunk.end * rate - Fraction(1, 2))
            boundary = min(end, max(frame + 1, last))
            if seconds < chunk.start or Fraction(boundary, rate) > chunk.end:
                raise ReplayError(
                    "CFR selection crosses a native chunk boundary; "
                    "verified source/candidate frame-clock mapping is required"
                )
            result.append(Clip(chunk.media, seconds - chunk.start, boundary - frame))
            frame = boundary
    return result


def _clip_json(clip: Clip, rate: int) -> dict[str, object]:
    return {
        "sourcePath": str(clip.media),
        "selection": {
            "startSeconds": float(clip.start),
            "durationSeconds": float(Fraction(clip.frames, rate)),
            "rate": 1,
        },
    }


def _overlays(config: Config) -> list[Overlay]:
    return [
        Overlay(_alpha(config, item.segment, item.part), item.offset, item.frames)
        for item in motion_parts(config)
    ]


def compose(config: Config, *, workers: int) -> list[Path]:
    """Write bounded native requests and the workflow that partitions them."""
    if (config.composition / "resume.json").exists():
        raise ReplayError("Resume compositions must not be repartitioned")
    cleanup = verified_cleanup(config)
    rate = config.rate
    pieces = clips(read_cuts(config.cuts), read_chunks(config.chunks), rate)
    overlays = _overlays(config)
    requests: list[dict[str, object]] = []
    group: list[Clip] = []
    cursor = start = 0

    def flush() -> None:
        end = cursor
        layers = [
            {
                "sourcePath": str(layer.media),
                "selection": {
                    "startSeconds": float(Fraction(max(start - layer.offset, 0), rate)),
                    "durationSeconds": float(
                        Fraction(
                            min(end, layer.offset + layer.frames) - max(start, layer.offset), rate
                        )
                    ),
                    "rate": 1,
                },
                "offsetSeconds": float(Fraction(max(layer.offset - start, 0), rate)),
                "opacity": 1,
            }
            for layer in overlays
            if layer.offset < end and start < layer.offset + layer.frames
        ]
        if len(layers) > MAX_OVERLAYS:
            raise ReplayError("A native request would exceed its overlay budget")
        index = len(requests)
        directory = config.storage / "compose" / f"request-{index:04d}"
        requests.append(
            {
                "outputDirectory": str(directory),
                "outputName": "candidate-video.mp4",
                "recipe": {
                    "video": {
                        "width": config.width,
                        "height": config.height,
                        "frameRate": rate,
                        "resizeMode": "fit",
                        "encoding": config.encoding,
                    },
                    "clips": [_clip_json(clip, rate) for clip in group],
                    "additionalVideo": layers,
                },
            }
        )

    for clip in pieces:
        overlapping = sum(
            1
            for layer in overlays
            if layer.offset < cursor + clip.frames and start < layer.offset + layer.frames
        )
        if group and (len(group) >= MAX_CLIPS or overlapping > MAX_OVERLAYS):
            flush()
            group, start = [], cursor
        group.append(clip)
        cursor += clip.frames
    flush()
    stages: list[dict[str, object]] = []
    for index, request in enumerate(requests):
        path = _write_new(config.composition / f"request-{index:04d}.json", request)
        parts = config.composition / f"parts-{index:04d}"
        stages.append(
            {
                "name": f"partition-{index:04d}",
                "command": [
                    sys.executable,
                    "-m",
                    "replay.native_edit_parts",
                    str(path),
                    str(parts),
                    "--sha256",
                    _digest(path),
                ],
                "inputs": [
                    str(path),
                    str(config.path),
                    str(config.cuts),
                    str(config.chunks),
                    str(cleanup.path),
                    str(cleanup.clean),
                    *map(str, config.cut_evidence),
                ],
                "outputs": [str(parts / "manifest.json")],
                "timeout_seconds": PARTITION_TIMEOUT_SECONDS,
            }
        )
    return _workflows(config.composition, "partition", stages, workers)


def rendered_parts(config: Config) -> list[tuple[Path, int]]:
    """Every partitioned part request in timeline order with its frame count."""
    parts: list[tuple[Path, int]] = []
    for manifest in sorted(config.composition.glob("parts-[0-9][0-9][0-9][0-9]/manifest.json")):
        for raw in _list(record(_json(manifest)).get("parts")):
            part = record(raw)
            parts.append((_path(part.get("path")), _integer(part.get("frames"))))
    if not parts:
        raise ReplayError("No partitioned native parts exist")
    return parts


def render(config: Config, *, workers: int) -> list[Path]:
    """Native edit-media stages for every part; the response names the movie."""
    if (config.composition / "resume.json").exists():
        raise ReplayError("Resume compositions must use their sealed prepared workflows")
    cleanup = verified_cleanup(config)
    stages: list[dict[str, object]] = []
    bindings: dict[str, Path] = {}
    for index, (request, frames) in enumerate(rendered_parts(config)):
        invocation, response, binding = native_invocation.paths(request)
        document = record(_json(request))
        recipe = record(document.get("recipe"))
        media = list(
            dict.fromkeys(
                _path(record(row).get("sourcePath"))
                for row in [*_list(recipe.get("clips")), *_list(recipe.get("additionalVideo", []))]
            )
        )
        if frames <= 0:
            raise ReplayError("Native part has no frames")
        name = f"render-{index:06d}"
        cleanup_binding = source_caption_binding.create_binding(
            cleanup,
            consumer=request,
            kind="native-request",
            output=request.with_name("source-caption.binding.json"),
        )
        bindings[name] = cleanup_binding
        stages.append(
            {
                "name": name,
                "command": list(native_invocation.command(config.native, request, config.storage)),
                "verify_command": list(
                    native_invocation.command(config.native, request, config.storage, verify=True)
                ),
                "inputs": [
                    str(request),
                    str(config.native),
                    *(str(path) for path in media if path != cleanup.clean),
                    str(config.path),
                    str(config.cuts),
                    str(config.chunks),
                    str(cleanup.path),
                    str(cleanup.clean),
                    str(cleanup_binding),
                    *map(str, config.cut_evidence),
                    str(Path(native_invocation.__file__).resolve()),
                ],
                "outputs": [str(invocation), str(response), str(binding)],
                "timeout_seconds": RENDER_TIMEOUT_SECONDS,
            }
        )
    return _workflows(config.composition, "render", stages, workers, bindings=bindings)


def join(config: Config) -> list[Path]:
    """Bind ordered real movie paths without mutable symlink or hardlink aliases."""
    plan: list[dict[str, object]] = []
    inputs: list[str] = []
    for index, (request, frames) in enumerate(rendered_parts(config)):
        response = record(_json(request.with_name("render.response.json")))
        movie = _path(response.get("outputPath"))
        if response.get("frameCount") != frames:
            raise ReplayError(f"Native part {index} frame count differs from its plan")
        if movie.is_symlink() or not movie.is_file():
            raise ReplayError("Join requires regular immutable native movies")
        inputs.append(str(movie))
        plan.append({"durationFrames": frames, "sourcePath": str(movie)})
    plan_path = _write_new(config.work / "join-plan.json", plan)
    output = config.storage / "joined-video.mp4"
    stage = {
        "name": "join",
        "command": [
            sys.executable,
            "-m",
            "replay.delivery_render",
            "join-files",
            str(plan_path),
            "--output",
            str(output),
            "--rate",
            str(config.rate),
        ],
        "inputs": [
            str(plan_path),
            *inputs,
            str(config.cuts),
            str(config.chunks),
            *map(str, config.cut_evidence),
        ],
        "outputs": [str(output)],
        "timeout_seconds": JOIN_TIMEOUT_SECONDS,
    }
    return [_write_new(config.work / "join-workflow.json", {"version": 1, "stages": [stage]})]


def join_files(plan: Path, output: Path, rate: int, audio: Path | None = None) -> int:
    """Execute the pinned ordered plan through the existing lossless join implementation."""
    rows = _list(_json(plan))
    parts: list[native_join.PartFile] = []
    for raw in rows:
        row = record(raw)
        if set(row) != {"sourcePath", "durationFrames"}:
            raise ReplayError("Join plan requires exact media paths and frame counts")
        parts.append(
            native_join.PartFile(_path(row["sourcePath"]), _integer(row["durationFrames"]))
        )
    return native_join.join(parts, rate=_integer(rate), output=output, audio=audio)


def encode_job(config: Config) -> Path:
    job = {
        "name": "encode",
        "source": str(config.storage / "joined-video.mp4"),
        "preset": str(config.encode_preset),
        "output": str(config.storage / "encoded-video.mov"),
    }
    return _write_new(config.work / "encode-job.json", {"jobs": [job]})


def audio(config: Config) -> list[Path]:
    """Select the cut intervals from the voice track on its native clock."""
    intervals = [
        {
            "start": {"value": cut.start, "timescale": config.rate},
            "end": {"value": cut.end, "timescale": config.rate},
        }
        for cut in read_cuts(config.cuts)
    ]
    selection = {
        "version": 1,
        "coordinateSpace": "native-presentation-seconds",
        "source": {"path": str(config.audio_source), "sha256": _digest(config.audio_source)},
        "clock": {"path": str(config.audio_clock), "sha256": _digest(config.audio_clock)},
        "evidence": [],
        "intervals": intervals,
    }
    path = _write_new(config.work / "audio-selection.json", selection)
    output = config.work / "audio-selected"
    stage = {
        "name": "select-audio",
        "command": [
            sys.executable,
            "-m",
            "replay.continuous_cut_job",
            str(path),
            str(output),
            "--sha256",
            _digest(path),
        ],
        "inputs": [
            str(path),
            str(config.audio_source),
            str(config.audio_clock),
            str(config.cuts),
            *map(str, config.cut_evidence),
        ],
        "outputs": [str(output / "selected.wav")],
        "timeout_seconds": AUDIO_TIMEOUT_SECONDS,
    }
    return [_write_new(config.work / "audio-workflow.json", {"version": 1, "stages": [stage]})]


def _master(config: Config) -> Path:
    return config.storage / "av-master.mov"


def final_mux(config: Config) -> list[Path]:
    """Losslessly pair the joined video with the selected voice PCM for encoding."""
    selected = config.work / "audio-selected" / "selected.wav"
    joined = config.storage / "joined-video.mp4"
    stage = {
        "name": "final-mux",
        "command": [
            "ffmpeg",
            "-nostdin",
            "-v",
            "error",
            "-n",
            "-i",
            str(joined),
            "-i",
            str(selected),
            "-map",
            "0:v:0",
            "-map",
            "1:a:0",
            "-c",
            "copy",
            str(_master(config)),
        ],
        "inputs": [str(joined), str(selected), *map(str, config.cut_evidence)],
        "outputs": [str(_master(config))],
        "timeout_seconds": JOIN_TIMEOUT_SECONDS,
    }
    return [_write_new(config.work / "final-mux-workflow.json", {"version": 1, "stages": [stage]})]


def final_job(config: Config) -> Path:
    """The single Compressor pass that encodes the delivered video and audio."""
    job = {
        "name": "final",
        "source": str(_master(config)),
        "preset": str(config.final_preset),
        "output": str(configured_movie(config)),
    }
    return _write_new(config.work / "final-job.json", {"jobs": [job]})


def delivery_stem(config: Config) -> str:
    """`<JST start>_JST_<recording>-review`, from the dated original's name."""
    return original_delivery_stem(config.source, config.recording)


def configured_movie(config: Config) -> Path:
    """The dated Compressor target, preserving its native preset's container."""
    return (
        config.storage
        / "final"
        / encoded_name(config.source, config.recording, config.final_preset)
    )


def caption_cues(config: Config) -> list[tuple[Fraction, Fraction, str]]:
    """Every combined caption on the global output clock, in display order."""
    cues: list[tuple[Fraction, Fraction, str]] = []
    for row in segments(config):
        combined = record(_json(config.captions / f"combined-{row['segment']:04d}.json"))
        offset = Fraction(row["offset"], config.rate)
        for raw in _list(combined.get("candidates")):
            cue = record(record(raw).get("cue"))
            text, start, end = cue.get("text"), cue.get("start"), cue.get("end")
            if not isinstance(text, str) or not isinstance(start, (int, float)):
                raise ReplayError("Combined caption lacks text or clock")
            if not isinstance(end, (int, float)):
                raise ReplayError("Combined caption lacks its end")
            cues.append((offset + Fraction(start), offset + Fraction(end), text))
    return cues


def _srt_time(value: Fraction) -> str:
    milliseconds = round(value * 1000)
    hours, rest = divmod(milliseconds, 3_600_000)
    minutes, rest = divmod(rest, 60_000)
    seconds, milliseconds = divmod(rest, 1000)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d},{milliseconds:03d}"


def render_srt(cues: list[tuple[Fraction, Fraction, str]]) -> str:
    blocks = [
        f"{index}\n{_srt_time(start)} --> {_srt_time(end)}\n{text}\n"
        for index, (start, end, text) in enumerate(cues, start=1)
    ]
    return "\n".join(blocks)


def publish(config: Config) -> list[Path]:
    """Stream-copy the Compressor output into the dated delivery with its SRT.

    The delivery and the source probe the Viewer uses for the broadcast date go
    under `deliveryRoot`; nothing existing is overwritten.
    """
    verified_cleanup(config)
    encoded = actual_output(configured_movie(config))
    directory = config.delivery_root / "revisions" / "native-v1" / "delivery"
    directory.mkdir(mode=0o700, parents=True, exist_ok=True)
    movie = directory / f"{delivery_stem(config)}.mp4"
    if movie.exists():
        raise ReplayError(f"Refusing to overwrite a delivery: {movie}")
    subprocess.run(
        [
            "ffmpeg",
            "-nostdin",
            "-v",
            "error",
            "-n",
            "-i",
            str(encoded),
            "-map",
            "0",
            "-c",
            "copy",
            "-tag:v",
            "hvc1",
            "-movflags",
            "+faststart",
            str(movie),
        ],
        check=True,
        timeout=JOIN_TIMEOUT_SECONDS,
    )
    srt = movie.with_suffix(".srt")
    with srt.open("x", encoding="utf-8") as stream:
        stream.write(render_srt(caption_cues(config)))
    probe = config.delivery_root / "evidence" / "source-probe.json"
    if not probe.exists():
        _write_new(probe, {"format": {"filename": str(config.source)}})
    report = verify(config, movie)
    checks = record(_json(report))
    if (
        checks.get("framesMatchCutPlan") is not True
        or checks.get("fullDecodePassed") is not True
        or _int0(checks.get("audioPackets")) <= 0
    ):
        raise ReplayError(f"Delivery media verification failed; candidate retained: {report}")
    return [movie, srt, probe, report, *register(config)]


def register(config: Config) -> list[Path]:
    """Expose an unchanged candidate only after every required edit stage passes.

    This separate operation resumes acceptance after validators/import complete;
    it never re-encodes or overwrites the retained candidate.
    """
    from replay.edit_completion import assess_delivery

    directory = config.delivery_root / "revisions" / "native-v1" / "delivery"
    movie = directory / f"{delivery_stem(config)}.mp4"
    failures = assess_delivery(movie, root=config.viewer_root)
    if failures:
        raise ReplayError(
            "Required edit stages remain unverified; candidate retained: "
            + ", ".join(sorted(failures))
        )
    return link_viewer(config, movie, movie.with_suffix(".srt"))


def _packets(movie: Path, stream: str) -> int:
    result = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            stream,
            "-count_packets",
            "-show_entries",
            "stream=nb_read_packets",
            "-of",
            "csv=p=0",
            str(movie),
        ],
        check=True,
        capture_output=True,
        text=True,
        timeout=JOIN_TIMEOUT_SECONDS,
    )
    return int(result.stdout.strip() or "0")


def verify(config: Config, movie: Path) -> Path:
    """Fully decode the delivery and compare its frames with the cut plan.

    The report states measured facts only; it does not approve captions,
    source-caption removal or audio quality.
    """
    expected = sum(cut.end - cut.start for cut in read_cuts(config.cuts))
    decode = subprocess.run(
        ["ffmpeg", "-nostdin", "-v", "error", "-xerror", "-i", str(movie), "-f", "null", "-"],
        capture_output=True,
        text=True,
        timeout=JOIN_TIMEOUT_SECONDS,
        check=False,
    )
    frames = _packets(movie, "v:0")
    report = {
        "movie": str(movie),
        "bytes": movie.stat().st_size,
        "expectedFrames": expected,
        "videoFrames": frames,
        "audioPackets": _packets(movie, "a:0"),
        "framesMatchCutPlan": frames == expected,
        "fullDecodePassed": decode.returncode == 0 and not decode.stderr.strip(),
        "decodeErrors": decode.stderr.strip()[-2000:],
        "captionQualityApproved": False,
        "sourceCaptionRemovalApproved": False,
        "musicRemovalApproved": False,
    }
    path = movie.with_name(f"{movie.stem}.verification.json")
    return _write_new(path, report)


def link_viewer(config: Config, movie: Path, srt: Path) -> list[Path]:
    """Expose the delivery in the Viewer library through links; media stays external."""
    directory = config.viewer_root / "revisions" / "native-v1" / "delivery"
    directory.mkdir(mode=0o700, parents=True, exist_ok=True)
    links = [directory / movie.name, directory / srt.name]
    for link, target in zip(links, (movie, srt), strict=True):
        link.symlink_to(target)
    return links


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "step",
        choices=(
            "motion-export",
            "motion-prepare",
            "alpha-jobs",
            "compose",
            "render",
            "join",
            "join-files",
            "encode-job",
            "audio",
            "final-mux",
            "final-job",
            "publish",
            "register",
        ),
    )
    parser.add_argument("config", type=Path)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--rate", type=int, default=30)
    parser.add_argument("--audio", type=Path, help="Whole-timeline audio for join-files")
    args = parser.parse_args(argv)
    try:
        if args.step == "join-files":
            if args.output is None:
                raise ReplayError("join-files requires --output")
            print(join_files(args.config, args.output, args.rate, args.audio))
            return 0
        if args.audio is not None:
            raise ReplayError("--audio is only supported by join-files")
        config = load_config(args.config)
        steps = {
            "motion-export": lambda: motion_export(config, workers=args.workers),
            "motion-prepare": lambda: motion_prepare(config, workers=args.workers),
            "alpha-jobs": lambda: [alpha_jobs(config)],
            "compose": lambda: compose(config, workers=args.workers),
            "render": lambda: render(config, workers=args.workers),
            "join": lambda: join(config),
            "encode-job": lambda: [encode_job(config)],
            "audio": lambda: audio(config),
            "final-mux": lambda: final_mux(config),
            "final-job": lambda: [final_job(config)],
            "publish": lambda: publish(config),
            "register": lambda: register(config),
        }
        for path in steps[args.step]():
            print(path)
    except (ReplayError, OSError, ValueError) as error:
        print(f"DELIVERY_RENDER_ERROR {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
