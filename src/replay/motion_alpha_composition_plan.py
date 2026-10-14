"""Plan audited Motion alpha overlays through the existing native edit contract.

This metadata producer does not render. Cleanup and alpha acceptance use the
existing production validators; constant Motion origins do not prove native
bounding boxes, rendered colour, audio, or finished delivery.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path

from replay import delivery_render, native_edit_parts, native_invocation, source_caption_binding
from replay.caption_measured_job import _pairs
from replay.code_dependency_inventory import capture
from replay.core import ReplayError
from replay.foreground_evidence import _bound_bytes
from replay.motion_caption import MAX_XML_CHARACTERS, _parse
from replay.recordings import record
from replay.source_caption_overlays import _approved
from replay.stages import file_digest

MAX_METADATA_BYTES = 8 * 1024 * 1024
MAX_ALPHAS = 4096
MAX_REQUESTS = 4096
MAX_STAGE_INPUTS = 256
CODE_MODULES = ("motion_alpha_composition_plan", "native_invocation", "workflow")


@dataclass(frozen=True)
class Alpha:
    """One audited, immutable title and its global edited-frame interval."""

    movie: Path
    source: Path
    manifest: Path
    start: int
    frames: int


def _path(value: object) -> Path:
    if not isinstance(value, str) or not value:
        raise ReplayError("Expected a canonical absolute path")
    path = Path(value)
    if not path.is_absolute() or path != path.resolve() or path.is_symlink():
        raise ReplayError("Paths must be canonical without symlink aliases")
    return path


def _integer(value: object, *, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        raise ReplayError("Expected a bounded integer frame/count")
    return value


def _json(path: Path) -> dict[str, object]:
    return record(
        json.loads(
            _bound_bytes(path, file_digest(path), MAX_METADATA_BYTES), object_pairs_hook=_pairs
        )
    )


def _pin(value: object) -> Path:
    row = record(value)
    if set(row) != {"path", "sha256"}:
        raise ReplayError("Expected an exact path/SHA pin")
    path = _path(row["path"])
    if row["sha256"] != file_digest(path):
        raise ReplayError("Pinned input changed")
    return path


def verify_motion_origins(text: str) -> int:
    """Reject changing local Y origins; scale/bbox/colour remain unmeasured.

    Generated caption nodes must have scalar Y origins. Ancestor position
    channels, if present, must also be scalar. No evaluation or removal of an
    unsupported animation is attempted.
    """
    root = _parse(text)
    nodes = root.findall(".//scenenode[@factoryID='16']")
    if not nodes:
        raise ReplayError("Motion document has no generated captions")
    for node in nodes:
        origins = node.findall(
            "parameter[@id='1']/parameter[@id='100']/parameter[@id='101']/parameter[@id='2']"
        )
        if len(origins) != 1:
            raise ReplayError("Caption requires exactly one scalar Y origin")
        if list(origins[0]) or not math.isfinite(float(origins[0].get("value", "nan"))):
            raise ReplayError("Caption Y origin is animated or nonfinite")
    for position in root.findall(".//parameter[@id='100']/parameter[@id='101']"):
        axes = position.findall("parameter[@id='2']")
        if len(axes) != 1 or list(axes[0]):
            raise ReplayError("Animated or unknown Motion Y origin is unsupported")
        value = float(axes[0].get("value", "nan"))
        if not math.isfinite(value):
            raise ReplayError("Motion Y origin must be finite")
    return len(nodes)


def _alphas(raw: object, cleanup: source_caption_binding.Cleanup, rate: int) -> tuple[Alpha, ...]:
    if not isinstance(raw, list) or not 1 <= len(raw) <= MAX_ALPHAS:
        raise ReplayError("Expected a bounded nonempty alpha mapping")
    cleanup_data = _json(cleanup.path)
    approved = _approved(cleanup_data, cleanup.clean)
    audited_sources: dict[Path, Path] = {}
    references = cleanup_data.get("captionOverlays")
    if not isinstance(references, list):
        raise ReplayError("Cleanup has no alpha audits")
    for reference in references:
        audit = _json(_pin(record(reference).get("audit")))
        parts = record(audit.get("measurements")).get("parts")
        if not isinstance(parts, list):
            raise ReplayError("Alpha audit has no measured parts")
        for part in parts:
            row = record(part)
            movie, source = _pin(row.get("alpha")), _pin(row.get("source"))
            if movie in audited_sources and audited_sources[movie] != source:
                raise ReplayError("Conflicting alpha source authority")
            audited_sources[movie] = source
    result: list[Alpha] = []
    previous_end = 0
    movies: set[Path] = set()
    selected: dict[Path, set[str]] = {}
    for value in raw:
        row = record(value)
        if set(row) != {"manifest", "part", "alpha"}:
            raise ReplayError("Unsupported alpha mapping fields")
        manifest = _pin(row["manifest"])
        name = row["part"]
        if not isinstance(name, str) or Path(name).name != name or not name.endswith(".moti"):
            raise ReplayError("Invalid Motion manifest part name")
        data = _json(manifest)
        if data.get("version") != 1 or data.get("candidateOnly") is not True:
            raise ReplayError("Unsupported Motion expansion manifest")
        entry = record(record(data.get("completed")).get(name))
        selected.setdefault(manifest, set()).add(name)
        source = _pin({"path": str(manifest.parent / name), "sha256": entry.get("sha256")})
        payload = _bound_bytes(source, file_digest(source), MAX_XML_CHARACTERS)
        verify_motion_origins(payload.decode("utf-8"))
        movie = _pin(row["alpha"])
        start, frames = _integer(entry.get("offsetFrame")), _integer(entry.get("frames"), minimum=1)
        if (
            movie in movies
            or start < previous_end
            or approved.get(movie) != Fraction(frames, rate)
            or audited_sources.get(movie) != source
        ):
            raise ReplayError("Alpha order/source/frame clock differs from genuine audit")
        result.append(Alpha(movie, source, manifest, start, frames))
        movies.add(movie)
        previous_end = start + frames
    if any(set(record(_json(path).get("completed"))) != names for path, names in selected.items()):
        raise ReplayError("Alpha mapping must cover every completed manifest part")
    return tuple(result)


def requests(
    config: delivery_render.Config, alphas: tuple[Alpha, ...]
) -> tuple[dict[str, object], ...]:
    """Split only on original cut, alpha and finite native-reader boundaries."""
    clips = delivery_render.clips(
        delivery_render.read_cuts(config.cuts),
        delivery_render.read_chunks(config.chunks),
        config.rate,
    )
    total = sum(clip.frames for clip in clips)
    if not alphas or alphas[-1].start + alphas[-1].frames > total:
        raise ReplayError("Alpha mapping exceeds the complete edited timeline")
    boundaries = {0, total}
    for alpha in alphas:
        boundaries.update((alpha.start, alpha.start + alpha.frames))
    cursor = 0
    result: list[dict[str, object]] = []
    for clip in clips:
        end = cursor + clip.frames
        marks = sorted(
            {
                cursor,
                end,
                *(point for point in boundaries if cursor < point < end),
                *range(cursor, end, config.rate * native_edit_parts.MAX_PART_SECONDS),
            }
        )
        for low, high in zip(marks[:-1], marks[1:], strict=True):
            layers: list[dict[str, object]] = []
            for alpha in alphas:
                if alpha.start <= low < alpha.start + alpha.frames:
                    layers.append(
                        {
                            "sourcePath": str(alpha.movie),
                            "selection": {
                                "startSeconds": float(Fraction(low - alpha.start, config.rate)),
                                "durationSeconds": float(Fraction(high - low, config.rate)),
                                "rate": 1,
                            },
                            "offsetSeconds": 0,
                            "opacity": 1,
                        }
                    )
            request: dict[str, object] = {
                "outputDirectory": str(config.storage / f"motion-alpha-{len(result):06d}"),
                "outputName": "candidate-video.mp4",
                "recipe": {
                    "video": {
                        "width": config.width,
                        "height": config.height,
                        "frameRate": config.rate,
                        "resizeMode": "fit",
                        "encoding": config.encoding,
                    },
                    "muteOriginalAudio": True,
                    "clips": [
                        {
                            "sourcePath": str(clip.media),
                            "selection": {
                                "startSeconds": float(
                                    clip.start + Fraction(low - cursor, config.rate)
                                ),
                                "durationSeconds": float(Fraction(high - low, config.rate)),
                                "rate": 1,
                            },
                        }
                    ],
                    "additionalVideo": layers,
                },
            }
            native_edit_parts.partition(request)
            result.append(request)
            if len(result) > MAX_REQUESTS:
                raise ReplayError("Composition exceeds bounded request count")
        cursor = end
    return tuple(result)


def _save(path: Path, value: object) -> None:
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, sort_keys=True, indent=2)
        stream.write("\n")


def prepare(config_path: Path, output: Path, *, verify: bool = False) -> Path:
    """Write or recompute immutable requests and genuine cleanup bindings.

    Native execution is intentionally separate. The returned index is consumed
    by normal native_invocation stages, whose producer and verifier arguments
    are retained per request. No output movie or approval is fabricated.
    """
    config_path, output = _path(str(config_path.absolute())), _path(str(output.absolute()))
    if not output.is_relative_to(Path.cwd().resolve()):
        raise ReplayError("Planner output must remain in the checkout")
    original_sha = file_digest(config_path)
    code = capture(CODE_MODULES)
    raw = _json(config_path)
    if (
        set(raw) != {"version", "renderConfig", "alphas"}
        or type(raw["version"]) is not int
        or raw["version"] != 1
    ):
        raise ReplayError("Unsupported Motion composition configuration")
    render_path = _pin(raw["renderConfig"])
    config = delivery_render.load_config(render_path)
    cleanup = delivery_render.verified_cleanup(config)
    alphas = _alphas(raw["alphas"], cleanup, config.rate)
    planned = requests(config, alphas)
    if not verify:
        if any(_path(item["outputDirectory"]).exists() for item in planned):
            raise ReplayError("Native media output namespace already exists")
        output.mkdir(mode=0o700)
    entries: list[dict[str, object]] = []
    for index, request in enumerate(planned):
        directory = output / f"part-{index:06d}"
        path = directory / "request.json"
        binding = directory / "source-caption.binding.json"
        if verify:
            if _json(path) != request:
                raise ReplayError("Generated request changed")
            source_caption_binding.validate_binding(
                binding,
                consumer_inputs=(binding, cleanup.clean, path),
                consumer_command=native_invocation.command(config.native, path, config.storage),
            )
        else:
            directory.mkdir(mode=0o700)
            _save(path, request)
            source_caption_binding.create_binding(
                cleanup, consumer=path, kind="native-request", output=binding
            )
        media_inputs = native_invocation.inputs(config.native, path)
        inputs = list(
            dict.fromkeys(
                (
                    config_path,
                    render_path,
                    config.cuts,
                    config.chunks,
                    cleanup.path,
                    cleanup.clean,
                    binding,
                    *media_inputs,
                    *(item.path for item in code),
                    *(item.source for item in alphas if item.movie in media_inputs),
                    *(item.manifest for item in alphas if item.movie in media_inputs),
                )
            )
        )
        if len(inputs) > MAX_STAGE_INPUTS:
            raise ReplayError("Native stage input budget exceeded")
        stage: dict[str, object] = {
            "name": "render",
            "command": list(native_invocation.command(config.native, path, config.storage)),
            "verify_command": list(
                native_invocation.command(config.native, path, config.storage, verify=True)
            ),
            "inputs": list(map(str, inputs)),
            "outputs": list(map(str, native_invocation.paths(path))),
            "timeout_seconds": 1800,
        }
        graph = {
            "version": 1,
            "stages": [stage],
            "sourceCaptionProduction": {"bindings": {"render": str(binding)}},
        }
        workflow = directory / "workflow.json"
        if verify:
            if _json(workflow) != graph:
                raise ReplayError("Generated workflow changed")
        else:
            _save(workflow, graph)
        entries.append(
            {
                "request": str(path),
                "sha256": file_digest(path),
                "workflow": str(workflow),
                "workflowSHA256": file_digest(workflow),
                "binding": str(binding),
                "bindingSHA256": file_digest(binding),
            }
        )
    result = {
        "version": 1,
        "kind": "motion-alpha-composition-plan",
        "configSHA256": original_sha,
        "codePins": {str(item.path): item.sha256 for item in code},
        "parts": entries,
        "videoOnly": True,
        "nativeBBoxVerified": False,
        "renderedColourVerified": False,
        "audioVerified": False,
        "deliveryComplete": False,
    }
    receipt = output / "plan.json"
    if file_digest(config_path) != original_sha:
        raise ReplayError("Planner configuration changed during preparation")
    if (
        _pin(raw["renderConfig"]) != render_path
        or delivery_render.verified_cleanup(config) != cleanup
    ):
        raise ReplayError("Cleanup or render configuration changed")
    if _alphas(raw["alphas"], cleanup, config.rate) != alphas:
        raise ReplayError("Alpha input identity changed during preparation")
    if capture(CODE_MODULES) != code:
        raise ReplayError("Code changed during composition planning")
    if verify:
        if _json(receipt) != result:
            raise ReplayError("Composition plan changed")
    else:
        _save(receipt, result)
    return receipt


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args(argv)
    try:
        print(prepare(args.config, args.output, verify=args.verify))
    except (ReplayError, OSError, ValueError) as error:
        print(f"MOTION_ALPHA_PLAN_ERROR {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
