"""Require verified local cleanup at declared production consumer boundaries."""

import json
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from xml.etree import ElementTree as ET

from replay.artifact_json import _json as _json
from replay.artifact_json import _pin as _pin
from replay.core import ReplayError
from replay.fcp_project import GAUSSIAN_UID
from replay.recordings import record


def clean_source(config: dict[str, object]) -> tuple[Path, Path, Fraction]:
    from replay.fcp_transformed_source import clean_source as validate

    return validate(config)


@dataclass(frozen=True)
class Cleanup:
    path: Path
    original: Path
    clean: Path
    duration: Fraction


def require_cleanup(render_config: Path) -> Cleanup:
    """Production render configurations must select their verified clean index."""
    data = _json(render_config)
    path = _pin(data.get("sourceCleanup"))
    config = _json(path)
    original, clean, duration = clean_source(config)
    if data.get("source") != str(original) or data.get("sourceChunks") != str(
        _pin(config.get("cleanIndex"))
    ):
        raise ReplayError("Production config does not select its verified cleanup source/index")
    return Cleanup(path, original, clean, duration)


def create_binding(cleanup: Cleanup, *, consumer: Path, kind: str, output: Path) -> Path:
    """Write one immutable consumer binding; renderers still validate before execution."""
    from replay.fcp_part_assembly import _reference, _save

    config = _json(cleanup.path)
    _consumer(consumer, kind, config, cleanup.clean)
    _save(output, {**config, "version": 1, "consumer": {**_reference(consumer), "kind": kind}})
    return output


def _clips(value: object, clean: Path, config: dict[str, object]) -> None:
    recipe = record(value)
    clips = recipe.get("clips")
    if not isinstance(clips, list) or not clips:
        raise ReplayError("Production consumer has no source clips")
    for raw in clips:
        if record(raw).get("sourcePath") != str(clean):
            raise ReplayError("Production consumer references an unverified source")
    video = record(recipe.get("video"))
    if video.get("masks") not in (None, []):
        raise ReplayError("Fixed source-caption masks cannot accompany local repair")
    additional = recipe.get("additionalVideo", [])
    if not isinstance(additional, list):
        raise ReplayError("Invalid production overlay list")
    if additional:
        from replay.source_caption_overlays import validate_overlays

        validate_overlays(additional, config=config, clean=clean)


def _consumer(path: Path, kind: object, config: dict[str, object], clean: Path) -> None:
    if kind == "fcpxml":
        root = ET.fromstring(path.read_bytes())
        videos = root.findall("./resources/asset[@hasVideo='1']")
        if not videos or any(
            asset.find(f"media-rep[@src='{clean.as_uri()}']") is None for asset in videos
        ):
            raise ReplayError("Production XML references an unverified source")
        gaussian = {
            effect.get("id")
            for effect in root.findall("./resources/effect")
            if effect.get("uid") == GAUSSIAN_UID
        }
        if any(effect.get("ref") in gaussian for effect in root.findall(".//filter-video")):
            raise ReplayError("Production XML cannot reintroduce source-caption Gaussian overlays")
        return
    raw: object = json.loads(path.read_text(encoding="utf-8"))
    if kind == "part-plan":
        if not isinstance(raw, list) or not raw:
            raise ReplayError("Production part plan must be nonempty")
        for entry in raw:
            _clips(record(record(entry).get("request")).get("recipe"), clean, config)
    elif kind == "native-request":
        _clips(record(raw).get("recipe"), clean, config)
    elif kind == "transformed-source-config":
        data = record(raw)
        if (
            data.get("version") != 1
            or data.get("removeSourceMasks") is not True
            or any(
                data.get(key) != config.get(key)
                for key in ("assemblyManifest", "assemblyReceipt", "cleanIndex")
            )
        ):
            raise ReplayError("Transform consumer does not bind the verified cleanup assembly")
    elif kind == "delivery-render-config":
        if record(raw).get("sourceChunks") != str(_pin(config.get("cleanIndex"))):
            raise ReplayError("Delivery consumer does not use the verified cleanup index")
    else:
        raise ReplayError("Unsupported source-caption production consumer kind")


def validate_binding(
    path: Path, *, consumer_inputs: tuple[Path, ...], consumer_command: tuple[str, ...]
) -> None:
    """Verify both cleanup provenance and the actual input artifact before execution.

    Fresh repair and reuse have the same contract: the assembly must already
    exist with pinned producer, media and presentation-clock evidence. A failed
    upstream repair cannot authorize a raw-source substitution.
    """
    config = _json(path)
    if config.get("version") != 1:
        raise ReplayError("Unsupported source-caption binding version")
    _, clean, _ = clean_source(config)
    reference = record(config.get("consumer"))
    consumer = _pin(reference)
    inputs = {item.resolve(strict=True) for item in consumer_inputs}
    if not {path.resolve(strict=True), clean, consumer}.issubset(inputs):
        raise ReplayError(
            "Production consumer must declare binding, clean media and artifact inputs"
        )
    if str(consumer) not in consumer_command:
        raise ReplayError("Verified consumer artifact is not used by the command")
    if "-m" in consumer_command:
        index = consumer_command.index("-m") + 1
        module = consumer_command[index] if index < len(consumer_command) else ""
        expected = {
            "replay.native_render": "part-plan",
            "replay.part_render": "part-plan",
            "replay.native_invocation": "native-request",
            "replay.fcp_transformed_source": "transformed-source-config",
            "replay.delivery_render": "delivery-render-config",
        }
        if module in expected and reference.get("kind") != expected[module]:
            raise ReplayError("Binding kind does not match the production consumer")
        if module in expected:
            offset = {"replay.native_invocation": 3, "replay.delivery_render": 2}.get(module, 1)
            position = index + offset
            if position >= len(consumer_command) or consumer_command[position] != str(consumer):
                raise ReplayError("Verified artifact is not the consumer's positional input")
    _consumer(consumer, reference.get("kind"), config, clean)
