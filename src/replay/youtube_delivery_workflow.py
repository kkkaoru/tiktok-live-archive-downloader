"""Build a pinned post-Compressor candidate verification/publication workflow.

Building performs no media processing. Runtime checks technical reports, never
claims editorial approval, and publishes only through the existing explicit CAS.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from xml.etree import ElementTree as ET

from replay import (
    artifact_json,
    candidate_publication,
    core,
    delivery_render,
    delivery_verification,
    fcp_caption_srt,
    recordings,
    youtube_delivery_validation,
)
from replay import stages as stage_runner
from replay.artifact_json import _json, _path, _pin
from replay.core import ReplayError
from replay.delivery_render import load_config, read_cuts
from replay.recordings import record
from replay.stages import file_digest

REPOSITORY = Path(__file__).resolve().parents[2]
PINS = (
    "movie",
    "master",
    "preset",
    "semanticXML",
    "motionTemplate",
    "assemblyXML",
    "assemblyLineage",
    "renderConfig",
    "cuts",
    "editConfig",
    "ffmpeg",
    "ffprobe",
)
STAGES = ("artifact", "decode", "gate", "publish")


@dataclass(frozen=True)
class Config:
    path: Path
    raw: dict[str, object]
    pins: dict[str, Path]
    evidence: Path
    frames: int
    rate: int


def _save(path: Path, value: object) -> None:
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")


def _cut_clock(semantic: ET.Element, sequence: ET.Element, config: delivery_render.Config) -> None:
    """Match every retained source interval, not merely the output's total duration."""
    spine = sequence.find("spine")
    if spine is None or semantic.findall(".//timeMap") or semantic.findall(".//ref-clip"):
        raise ReplayError("Nested or retimed semantic timelines are unsupported")
    clips = list(spine)
    cuts = read_cuts(config.cuts)
    if len(clips) != len(cuts) or any(clip.tag != "asset-clip" for clip in clips):
        raise ReplayError("Semantic timeline requires one direct asset clip per cut")
    if len({clip.get("ref") for clip in clips}) != 1:
        raise ReplayError("Semantic cuts must share the uncut source clock")
    cursor = 0
    for clip, cut in zip(clips, cuts, strict=True):
        if clip.findall(".//spine") or any(
            child.get("srcEnable") != "audio" for child in clip.findall(".//asset-clip")
        ):
            raise ReplayError("Nested semantic video cuts are unsupported")
        actual = tuple(
            Fraction(clip.get(key, "-1s").removesuffix("s"))
            for key in ("offset", "start", "duration")
        )
        expected = tuple(
            Fraction(value, config.rate) for value in (cursor, cut.start, cut.end - cut.start)
        )
        if actual != expected:
            raise ReplayError("Semantic source intervals differ from the exact cut clock")
        cursor += cut.end - cut.start


def load(path: Path, *, sha256: str) -> Config:
    """Recheck all original pins for every producer and consumer invocation."""
    _pin({"path": str(path), "sha256": sha256})
    raw = _json(path)
    if (
        set(raw)
        != {
            "version",
            *PINS,
            "templateUID",
            "evidenceRoot",
            "editRoot",
            "library",
            "revision",
            "previousSelectionSHA256",
        }
        or raw["version"] != 1
    ):
        raise ReplayError("Unsupported post-Compressor workflow configuration")
    pins = {key: _pin(raw[key]) for key in PINS}
    evidence = _path(raw["evidenceRoot"])
    if not evidence.is_relative_to(REPOSITORY / "edits/work"):
        raise ReplayError("Workflow evidence must stay under checkout edits/work")
    root, library = _path(raw["editRoot"]), _path(raw["library"])
    if pins["editConfig"] != root / "edit-config.json":
        raise ReplayError("Publication edit config differs from its pinned root")
    previous = raw["previousSelectionSHA256"]
    if previous is not None and (
        not isinstance(previous, str) or re.fullmatch(r"[0-9a-f]{64}", previous) is None
    ):
        raise ReplayError("Explicit previous selection SHA256 is invalid")
    if (
        not isinstance(raw["revision"], str)
        or re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,95}", raw["revision"]) is None
    ):
        raise ReplayError("Invalid publication revision")
    config = load_config(pins["renderConfig"])
    if (
        config.cuts != pins["cuts"]
        or pins["movie"] in {config.source, pins["master"]}
        or not pins["movie"].is_relative_to(library / root.name)
        or config.delivery_root != library / root.name
    ):
        raise ReplayError("Final movie or exact cut configuration differs from delivery")
    lineage = _json(pins["assemblyLineage"])
    if any(lineage.get(key) != raw[key] for key in ("semanticXML", "assemblyXML")):
        raise ReplayError("Assembly lineage does not bind the selected semantic XML")
    frames = delivery_verification._expected(config)
    if lineage.get("duration") != f"{Fraction(frames, config.rate)}s":
        raise ReplayError("Assembly lineage duration differs from exact cuts")
    semantic = ET.parse(pins["semanticXML"]).getroot()
    sequence = semantic.find("./library/event/project/sequence")
    if sequence is None or Fraction(sequence.get("duration", "-1s").removesuffix("s")) != Fraction(
        frames, config.rate
    ):
        raise ReplayError("Semantic timeline duration differs from the exact cut clock")
    _cut_clock(semantic, sequence, config)
    effects = semantic.findall("./resources/effect")
    matching = {
        effect.get("id")
        for effect in effects
        if effect.get("uid") == raw["templateUID"]
        and effect.get("id") is not None
        and effect.get("src") == pins["motionTemplate"].as_uri()
    }
    titles = semantic.findall(".//title")
    if (
        not isinstance(raw["templateUID"], str)
        or not raw["templateUID"]
        or not matching
        or not titles
        or any(title.get("ref") not in matching for title in titles)
    ):
        raise ReplayError("Semantic titles do not reference the pinned Motion template UID/src")
    return Config(path, raw, pins, evidence, frames, config.rate)


def reports(config: Config) -> tuple[Path, Path, Path]:
    return (
        config.evidence / "artifact/result.json",
        config.evidence / "decode/receipt.json",
        config.evidence / "gate.json",
    )


def _identity(path: Path) -> list[int]:
    value = path.stat()
    return [value.st_dev, value.st_ino, value.st_size, value.st_mtime_ns, value.st_ctime_ns]


def gate(config: Config) -> dict[str, object]:
    """Both technical reports must refer to these exact current media/config bytes."""
    artifact, decoded, _ = reports(config)
    first, second = _json(artifact), _json(decoded)
    if (
        first.get("technicalPropertiesAccepted") is not True
        or first.get("movie") != str(config.pins["movie"])
        or first.get("master") != str(config.pins["master"])
        or first.get("inputStatIdentity")
        != [
            _identity(config.pins["movie"]),
            _identity(config.pins["master"]),
        ]
        or second.get("technicalAccepted") is not True
        or second.get("fullDecode") is not True
        or second.get("expectedVideoContract") != "h264"
    ):
        raise ReplayError("Technical reports failed or refer to different media")
    expected_inputs = [
        delivery_verification._fingerprint(config.pins[key])
        for key in ("renderConfig", "cuts", "movie")
    ]
    if second.get("inputs") != expected_inputs or second.get("inputsAfter") != expected_inputs:
        raise ReplayError("Full decode report inputs changed or identify another movie")
    render = load_config(config.pins["renderConfig"])
    if second.get("expected") != {
        "frames": config.frames,
        "rate": config.rate,
        "width": render.width,
        "height": render.height,
    } or second.get("videoClock") != {
        "frames": config.frames,
        "start": "0",
        "end": str(Fraction(config.frames, config.rate)),
        "exactCFR": True,
    }:
        raise ReplayError("Full decode report differs from the exact cut clock")
    for key, units in (
        ("audioDecodedClock", "sampleCount"),
        ("audioPacketClock", "durationTicksSum"),
    ):
        clock = record(second.get(key))
        count = clock.get("count")
        amount = clock.get(units)
        if type(count) is not int or count <= 0 or type(amount) is not int or amount <= 0:
            raise ReplayError("Full decode report lacks measured audio clocks")
        start, end, span = (Fraction(str(clock.get(name))) for name in ("start", "end", "span"))
        if end <= start or span != end - start:
            raise ReplayError("Measured audio clock bounds are inconsistent")
    return {
        "version": 1,
        "technicalAccepted": True,
        "deliveryApproved": False,
        "config": {"path": str(config.path), "sha256": file_digest(config.path)},
        "movie": config.raw["movie"],
        "semanticXML": config.raw["semanticXML"],
        "motionTemplate": config.raw["motionTemplate"],
        "reports": [
            {"path": str(path), "sha256": file_digest(path)} for path in (artifact, decoded)
        ],
    }


def run(path: Path, *, sha256: str, stage: str) -> None:
    config = load(path, sha256=sha256)
    movie, master = config.pins["movie"], config.pins["master"]
    artifact, decoded, accepted = reports(config)
    if stage == "artifact":
        if not youtube_delivery_validation.validate(
            movie, master=master, output=artifact.parent, ffprobe=config.pins["ffprobe"]
        ):
            raise ReplayError("YouTube artifact validation failed")
    elif stage == "decode":
        if not delivery_verification.verify(
            config.pins["renderConfig"],
            movie,
            decoded.parent,
            ffmpeg=config.pins["ffmpeg"],
            ffprobe=config.pins["ffprobe"],
            video_contract="h264",
        ):
            raise ReplayError("Full AV decode verification failed")
    elif stage == "gate":
        _save(accepted, gate(config))
    elif stage == "publish":
        if _json(accepted) != gate(config):
            raise ReplayError("Publication gate changed")
        previous = config.raw["previousSelectionSHA256"]
        selection = candidate_publication.publish(
            candidate_publication.Publication(
                _path(config.raw["editRoot"]),
                _path(config.raw["library"]),
                movie,
                config.pins["semanticXML"],
                str(config.raw["revision"]),
                previous if isinstance(previous, str) else None,
            )
        )
        _save(
            config.evidence / "publication.json",
            {
                "version": 1,
                "deliveryApproved": False,
                "selection": {"path": str(selection), "sha256": file_digest(selection)},
            },
        )
    else:
        raise ReplayError("Unknown post-Compressor workflow stage")
    load(path, sha256=sha256)


def build(path: Path, output: Path) -> Path:
    """Generate a fresh dependency graph only; no render, encode or publication."""
    sha = file_digest(path)
    config = load(path, sha256=sha)
    if output != config.evidence or output.exists():
        raise ReplayError("Require a fresh configured workflow evidence directory")
    artifact, decoded, accepted = reports(config)
    publication = config.evidence / "publication.json"
    outputs = (artifact, decoded, accepted, publication)
    stages: list[dict[str, object]] = []
    for index, name in enumerate(STAGES):
        stages.append(
            {
                "name": name,
                "command": [
                    sys.executable,
                    "-m",
                    "replay.youtube_delivery_workflow",
                    "run",
                    str(path),
                    "--sha256",
                    sha,
                    "--stage",
                    name,
                ],
                "inputs": list(
                    dict.fromkeys(
                        map(
                            str,
                            (
                                path,
                                Path(__file__).resolve(),
                                Path(delivery_verification.__file__).resolve(),
                                Path(youtube_delivery_validation.__file__).resolve(),
                                Path(candidate_publication.__file__).resolve(),
                                Path(artifact_json.__file__).resolve(),
                                Path(fcp_caption_srt.__file__).resolve(),
                                Path(delivery_render.__file__).resolve(),
                                Path(recordings.__file__).resolve(),
                                Path(stage_runner.__file__).resolve(),
                                Path(core.__file__).resolve(),
                                *config.pins.values(),
                                *outputs[:index],
                            ),
                        )
                    )
                ),
                "outputs": [str(outputs[index])],
                "timeout_seconds": 10800,
            }
        )
    output.mkdir(mode=0o700, parents=True)
    plan = output / "workflow.json"
    _save(plan, {"version": 1, "stages": stages})
    return plan


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="action", required=True)
    prepare = commands.add_parser("build")
    prepare.add_argument("config", type=Path)
    prepare.add_argument("output", type=Path)
    execute = commands.add_parser("run")
    execute.add_argument("config", type=Path)
    execute.add_argument("--sha256", required=True)
    execute.add_argument("--stage", choices=STAGES, required=True)
    args = parser.parse_args(argv)
    try:
        if args.action == "build":
            print(build(args.config, args.output))
        else:
            run(args.config, sha256=args.sha256, stage=args.stage)
    except (ReplayError, OSError, ValueError, ET.ParseError, subprocess.SubprocessError) as error:
        print(f"YOUTUBE_DELIVERY_WORKFLOW_ERROR: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
