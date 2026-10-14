"""Generate resumable, bounded OCR/admission/repair workflows from pinned evidence."""

import argparse
import json
import os
import re
import shutil
import sys
from pathlib import Path

from replay.caption_io import FrameObservation, native_ocr_frames
from replay.caption_masks import MaskPolicy
from replay.delivery_source_assembly import candidate_media_path
from replay.local_asr import fingerprint
from replay.recordings import record
from replay.subtitle_cleanup_plan import admission, clock_chunks, search_profile
from replay.subtitle_clock import VideoFrameClock
from replay.subtitle_eligibility import _reference

MAX_JSON_BYTES = 16 * 1024 * 1024


def _local(path: Path) -> Path:
    if (
        not path.is_absolute()
        or path != path.resolve()
        or not path.is_relative_to(Path.cwd().resolve())
    ):
        raise ValueError("Cleanup metadata/output must be canonical checkout-local paths")
    return path


def _read(path: Path) -> dict[str, object]:
    _local(path)
    if not path.is_file() or not 0 < path.stat().st_size <= MAX_JSON_BYTES:
        raise ValueError("Invalid bounded cleanup JSON")
    return record(json.loads(path.read_text(encoding="utf-8")))


def _pin(path: Path) -> dict[str, str]:
    return {"path": str(path), "sha256": fingerprint(path)}


def _write(path: Path, value: object) -> None:
    _local(path)
    text = json.dumps(value, indent=2, allow_nan=False) + "\n"
    if path.exists():
        if path.read_text(encoding="utf-8") != text:
            raise ValueError("Existing cleanup plan differs; preserve it and use a new revision")
        return
    with path.open("x", encoding="utf-8") as stream:
        os.fchmod(stream.fileno(), 0o600)
        stream.write(text)


def _settings(path: Path) -> dict[str, object]:
    data = _read(path)
    source = record(data.get("source"))
    name, sha = source.get("path"), source.get("sha256")
    if (
        data.get("version") != 1
        or not isinstance(name, str)
        or not Path(name).is_absolute()
        or Path(name) != Path(name).resolve()
        or not isinstance(sha, str)
        or re.fullmatch("[0-9a-f]{64}", sha) is None
    ):
        raise ValueError("Invalid pinned source settings")
    bindings: dict[str, str] = {}
    _reference(data.get("native"), bindings, executable=True)
    _reference(data.get("decoder"), bindings, executable=True)
    _reference(data.get("model"), bindings)
    _reference(data.get("sourceClock"), bindings)
    _reference(data.get("classificationEvidence"), bindings)
    _reference(data.get("profileInputs"), bindings)
    _reference(data.get("profileComplete"), bindings)
    if type(data.get("chunkFrames")) is not int:
        raise ValueError("Missing explicit chunk frame budget")
    return data


def collector_frames(
    inputs: Path, complete: Path, settings: dict[str, object]
) -> tuple[tuple[FrameObservation, ...], dict[str, object]]:
    """Verify the collector receipt and every raw batch; no inference or writes."""
    producer, receipt = _read(inputs), _read(complete)
    native = record(settings.get("native"))
    source = record(settings.get("source"))
    if (
        producer.get("sourceSHA256") != source.get("sha256")
        or producer.get("nativeSHA256") != native.get("sha256")
        or producer.get("nativeExecutable") != native.get("path")
        or producer.get("measurementTool") != "video_text_recognize"
        or producer.get("fullFrame") is not True
        or complete.parent != inputs.parent
    ):
        raise ValueError("OCR collector source/native/mode provenance differs")
    results = record(receipt.get("results"))
    if not 1 <= len(results) <= 6250:
        raise ValueError("Invalid collector batch count")
    batches: list[dict[str, object]] = []
    frames: list[FrameObservation] = []
    for index, (name, sha) in enumerate(sorted(results.items())):
        if name != f"{index:04d}.result.json" or not isinstance(sha, str):
            raise ValueError("Missing or unordered collector batch")
        path = _reference({"path": str(inputs.parent / name), "sha256": sha}, {})
        observed = native_ocr_frames(json.loads(path.read_text(encoding="utf-8")))
        if not observed or (index + 1 < len(results) and len(observed) != 8):
            raise ValueError("Incomplete internal OCR batch")
        frames.extend(observed)
        batches.append({"ocr": _pin(path), "ocrProvenance": _pin(inputs)})
    if receipt.get("observations") != len(frames):
        raise ValueError("Collector observation total differs")
    return tuple(frames), {
        "version": 1,
        "kind": "source-caption-ocr-batches",
        "sourceSHA256": source["sha256"],
        "batches": batches,
    }


def _chunk(path: Path) -> tuple[dict[str, object], dict[str, object], VideoFrameClock]:
    data = _read(path)
    settings_path = _reference(data.get("settings"), {})
    settings = _settings(settings_path)
    clocks = clock_chunks(data.get("clock"), maximum_frames=10000)
    if len(clocks) != 1:
        raise ValueError("Expected one bounded cleanup chunk")
    return data, settings, clocks[0]


def check_ocr(chunk: Path) -> None:
    data, settings, clock = _chunk(chunk)
    directory = _local(Path(str(data.get("ocrDirectory"))))
    frames, _ = collector_frames(directory / "inputs.json", directory / "complete.json", settings)
    expected = tuple(float(pts * clock.time_base) for pts in clock.pts)
    if (
        tuple(frame.requested_time for frame in frames) != expected
        or tuple(frame.actual_time for frame in frames) != expected
    ):
        raise ValueError("OCR requested/actual PTS differ from the source chunk")


def build_candidate(chunk: Path) -> None:
    """Resolve OCR references at consumer time, keeping unknown frames explicit."""
    check_ocr(chunk)
    data, settings, clock = _chunk(chunk)
    directory = chunk.parent
    ocr_directory = _local(Path(str(data.get("ocrDirectory"))))
    inputs, complete = ocr_directory / "inputs.json", ocr_directory / "complete.json"
    frames, batches = collector_frames(inputs, complete, settings)
    workers = data.get("maximumActiveChunks", 1)
    if type(workers) is not int or not 1 <= workers <= 8:
        raise ValueError("Invalid active chunk capacity budget")
    storage: dict[str, object] = {"outputDirectory": str(directory / "repair")}
    if "mediaOutput" in data or "allowedMediaRoot" in data:
        storage["mediaOutput"] = data.get("mediaOutput")
        storage["allowedMediaRoot"] = data.get("allowedMediaRoot")
    media = candidate_media_path(storage)
    usage = shutil.disk_usage(media.parent if "mediaOutput" in storage else directory)
    minimum = max(30 * 1024**3, usage.total // 20)
    # Bound each active encoded output by uncompressed RGB bytes. No raw frame
    # files are produced; this conservative reserve accounts for all workers.
    reserve = len(frames) * frames[0].width * frames[0].height * 3 * workers
    if usage.free < minimum + reserve:
        raise ValueError("Insufficient free space for bounded active cleanup chunks")
    profile_inputs = _reference(settings.get("profileInputs"), {})
    profile_complete = _reference(settings.get("profileComplete"), {})
    seeds, _ = collector_frames(profile_inputs, profile_complete, settings)
    plan = admission(frames, clock, search_profile(seeds), policy=MaskPolicy())
    _write(directory / "ocr-batches.json", batches)
    _write(directory / "admission.json", plan)
    source, native, model = (record(settings.get(key)) for key in ("source", "native", "model"))
    eligibility = {
        "version": 1,
        "kind": "source-caption-eligibility",
        "sourceSHA256": source["sha256"],
        "canvas": [frames[0].width, frames[0].height],
        "timeBase": str(clock.time_base),
        "decoder": settings.get("decoder"),
        "codec": settings["sourceClock"],
        "classificationEvidence": settings["classificationEvidence"],
        "ocrProvenance": _pin(inputs),
        "ocr": _pin(directory / "ocr-batches.json"),
        "frames": plan["frames"],
    }
    _write(directory / "eligibility.json", eligibility)
    active = (
        [
            index
            for index, row in enumerate(plan["frames"])
            if record(row).get("classification") == "overlay-subtitle"
        ]
        if isinstance(plan["frames"], list)
        else []
    )
    candidate: dict[str, object] = {
        "source": source["path"],
        "sourceSHA256": source["sha256"],
        "nativeExecutable": native["path"],
        "nativeSHA256": native["sha256"],
        "outputDirectory": str(directory / "repair"),
        "workDirectory": str(directory / "scratch"),
        "width": frames[0].width,
        "height": frames[0].height,
        "clock": data["clock"],
        "masks": plan["masks"],
        "visionTextRegions": True,
        "prepareVisionContext": False,
        "learnedTextDetector": {
            "backend": "ppocrv3-db",
            "model": model["path"],
            "sha256": model["sha256"],
        },
        "sourceEligibility": _pin(directory / "eligibility.json"),
        "hardwareEncode": False,
    }
    if "mediaOutput" in storage:
        candidate["mediaOutput"] = storage["mediaOutput"]
        candidate["allowedMediaRoot"] = storage["allowedMediaRoot"]
    if active:
        candidate["evidenceFrames"] = sorted({active[0], active[-1]})
    _write(directory / "candidate-config.json", candidate)


def check_repair(chunk: Path) -> None:
    _, _, clock = _chunk(chunk)
    path = chunk.parent / "repair" / "candidate.json"
    receipt = _read(path)
    if record(receipt.get("render")).get("frames") != len(clock.pts):
        raise ValueError("Repaired source chunk frame count differs")
    producer = _read(chunk.parent / "candidate-config.json")
    media = candidate_media_path(producer)
    if "mediaOutput" in producer and (
        receipt.get("mediaOutput") != str(media)
        or record(receipt.get("inputs")).get(str(chunk.parent / "candidate-config.json"))
        != fingerprint(chunk.parent / "candidate-config.json")
    ):
        raise ValueError("Repair receipt differs from its explicit media destination")
    if receipt.get("videoSHA256") != fingerprint(media):
        raise ValueError("Repaired media fingerprint changed")
    for name, sha in record(receipt.get("inputs")).items():
        if fingerprint(Path(name)) != sha:
            raise ValueError("Repaired chunk input fingerprint changed")


def _command(operation: str, chunk: Path) -> list[str]:
    return [sys.executable, "-B", "-m", "replay.subtitle_cleanup_workflow", operation, str(chunk)]


def _stages(
    chunk: Path, settings: dict[str, object], previous: Path | None
) -> list[dict[str, object]]:
    directory = chunk.parent
    ocr = directory / "ocr"
    source, native = record(settings["source"]), record(settings["native"])
    common = [
        str(chunk),
        str(Path(__file__).resolve()),
        str(Path(__file__).with_name("subtitle_cleanup_plan.py").resolve()),
    ]
    parent_inputs = [str(previous)] if previous is not None else []
    name = directory.name
    return [
        {
            "name": name + "-ocr",
            "command": [
                sys.executable,
                "-B",
                "-m",
                "replay.subtitle_observations",
                source["path"],
                str(directory / "times.json"),
                str(ocr),
                "--native-executable",
                native["path"],
            ],
            "inputs": [
                *common,
                str(directory / "times.json"),
                source["path"],
                native["path"],
                *parent_inputs,
            ],
            "outputs": [str(ocr / "inputs.json"), str(ocr / "complete.json")],
            "timeout_seconds": 3600,
            "verify_command": _command("check-ocr", chunk),
        },
        {
            "name": name + "-admission",
            "command": _command("build", chunk),
            "inputs": [*common, str(ocr / "inputs.json"), str(ocr / "complete.json")],
            "outputs": [
                str(directory / value)
                for value in (
                    "ocr-batches.json",
                    "admission.json",
                    "eligibility.json",
                    "candidate-config.json",
                )
            ],
            "timeout_seconds": 120,
            "verify_command": _command("build", chunk),
        },
        {
            "name": name + "-repair",
            "command": [
                sys.executable,
                "-B",
                "-m",
                "replay.subtitle_candidate",
                str(directory / "candidate-config.json"),
            ],
            "inputs": [
                *common,
                str(directory / "candidate-config.json"),
                str(directory / "eligibility.json"),
                str(directory / "ocr-batches.json"),
            ],
            "outputs": [
                str(directory / "repair" / value) for value in ("candidate.mp4", "candidate.json")
            ],
            "timeout_seconds": 3600,
            "verify_command": _command("check-repair", chunk),
        },
    ]


def prepare(config: Path, output: Path, *, workers: int, limit: int | None = None) -> None:
    """Each worker finishes repair before admitting its next chunk; no frame dumps."""
    settings = _settings(config)
    seeds, _ = collector_frames(
        _reference(settings["profileInputs"], {}),
        _reference(settings["profileComplete"], {}),
        settings,
    )
    search_profile(seeds)
    if (
        type(workers) is not int
        or not 1 <= workers <= 8
        or (limit is not None and (type(limit) is not int or limit < 1))
    ):
        raise ValueError("Invalid bounded worker/chunk selection")
    source = record(settings["source"])
    if fingerprint(Path(str(source["path"]))) != source["sha256"]:
        raise ValueError("Source media changed before planning")
    source_clock = _reference(settings["sourceClock"], {})
    size = settings["chunkFrames"]
    if not isinstance(size, int):
        raise ValueError("Invalid chunk size")
    all_chunks = clock_chunks(_read(source_clock), maximum_frames=size)
    chunks = all_chunks[:limit]
    _local(output).mkdir(mode=0o700, parents=True, exist_ok=True)
    workflows: list[list[dict[str, object]]] = [[] for _ in range(workers)]
    previous: list[Path | None] = [None] * workers
    for index, clock in enumerate(chunks):
        directory = output / f"chunk-{index:05d}"
        directory.mkdir(mode=0o700, exist_ok=True)
        clock_data = {
            "pts": list(clock.pts),
            "timeBase": str(clock.time_base),
            "lastDuration": clock.last_duration,
        }
        chunk = directory / "chunk.json"
        _write(
            chunk,
            {
                "version": 1,
                "settings": _pin(config),
                "clock": clock_data,
                "ocrDirectory": str(directory / "ocr"),
                "maximumActiveChunks": min(workers, len(chunks)),
            },
        )
        _write(directory / "times.json", [float(pts * clock.time_base) for pts in clock.pts])
        worker = index % workers
        workflows[worker].extend(_stages(chunk, settings, previous[worker]))
        previous[worker] = directory / "repair" / "candidate.json"
    for worker, stages in enumerate(workflows):
        if stages:
            _write(output / f"worker-{worker:02d}.json", {"version": 1, "stages": stages})
    _write(
        output / "plan.json",
        {
            "version": 1,
            "settings": _pin(config),
            "totalSourceFrames": sum(len(clock.pts) for clock in all_chunks),
            "plannedFrames": sum(len(clock.pts) for clock in chunks),
            "chunks": len(chunks),
            "maximumActiveChunks": min(workers, len(chunks)),
            "decodedFramesOnDisk": False,
            "wholeSourcePlanned": len(chunks) == len(all_chunks),
            "qualityApproved": False,
        },
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("operation", choices=("prepare", "check-ocr", "build", "check-repair"))
    parser.add_argument("input", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args(argv)
    if args.operation == "prepare":
        if args.output is None:
            parser.error("prepare requires --output")
        prepare(args.input, args.output, workers=args.workers, limit=args.limit)
    else:
        operations = {
            "check-ocr": check_ocr,
            "build": build_candidate,
            "check-repair": check_repair,
        }
        operations[args.operation](args.input)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
