"""Explicit CPU OCR sessions; raw eight-frame evidence is never legacy provenance."""

import argparse
import json
import math
import os
import re
import stat
import subprocess
from dataclasses import dataclass
from fractions import Fraction
from itertools import islice
from pathlib import Path

from replay import code_generation_compat as compat
from replay.caption_io import native_ocr_frames
from replay.code_dependency_inventory import capture
from replay.recordings import record
from replay.stages import execution_lock, file_digest

PROTOCOL = "source-caption-ocr-session-v1"
TOOL = "video_text_recognize_session"
MAX_JSON = 4 * 1024 * 1024
MAX_SAMPLES = 50000


@dataclass(frozen=True)
class SourceRoot:
    path: Path
    device: int
    inode: int


@dataclass(frozen=True)
class SessionPlan:
    source: Path
    native: Path
    source_sha: str
    native_sha: str
    samples: tuple[Fraction, ...]
    width: int
    height: int
    workers: int | None = None
    source_root: SourceRoot | None = None


def _path(value: object) -> Path:
    if not isinstance(value, str):
        raise ValueError("Expected a canonical checkout path")
    path = Path(value)
    if (
        not path.is_absolute()
        or str(path) != value
        or path.resolve() != path
        or not path.is_relative_to(Path.cwd())
        or {"private", "downloads"}.intersection(path.relative_to(Path.cwd()).parts)
    ):
        raise ValueError("Session path escapes checkout or traverses a link")
    return path


def _source_root(value: object) -> SourceRoot:
    if not isinstance(value, str):
        raise ValueError("Expected an explicit canonical source root")
    path = Path(value)
    if not path.is_absolute() or str(path) != value or path.resolve(strict=True) != path:
        raise ValueError("Source root is noncanonical or linked")
    info = path.lstat()
    if not stat.S_ISDIR(info.st_mode) or path == Path(path.anchor):
        raise ValueError("Source root must be a specific existing directory")
    return SourceRoot(path, info.st_dev, info.st_ino)


def _source_path(value: object, root: SourceRoot | None) -> Path:
    if root is None:
        return _path(value)
    if _source_root(str(root.path)) != root or not isinstance(value, str):
        raise ValueError("Declared source root changed")
    path = Path(value)
    if (
        not path.is_absolute()
        or str(path) != value
        or path.resolve(strict=True) != path
        or path == root.path
        or not path.is_relative_to(root.path)
    ):
        raise ValueError("Source escapes its declared root")
    if path.is_relative_to(Path.cwd()):
        _path(value)
    return path


def _identity(path: Path, root: SourceRoot | None = None) -> dict[str, int]:
    _source_path(str(path), root)
    info = path.lstat()
    if not stat.S_ISREG(info.st_mode) or info.st_size <= 0 or info.st_nlink != 1:
        raise ValueError("Session input must be a nonempty single-link regular file")
    return {
        "device": info.st_dev,
        "inode": info.st_ino,
        "bytes": info.st_size,
        "links": info.st_nlink,
        "modifiedSeconds": info.st_mtime_ns // 1000000000,
        "modifiedNanoseconds": info.st_mtime_ns % 1000000000,
        "changedSeconds": info.st_ctime_ns // 1000000000,
        "changedNanoseconds": info.st_ctime_ns % 1000000000,
    }


def _pairs(items: list[tuple[str, object]]) -> dict[str, object]:
    if len(dict(items)) != len(items):
        raise ValueError("Duplicate JSON fields")
    return dict(items)


def _read(path: Path) -> object:
    before = _identity(path)
    if before["bytes"] > MAX_JSON:
        raise ValueError("Session metadata exceeds its bound")
    descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    with os.fdopen(descriptor, "rb") as stream:
        opened = os.fstat(stream.fileno())
        if (opened.st_dev, opened.st_ino) != (before["device"], before["inode"]):
            raise ValueError("Session metadata changed before opening")
        data = stream.read(MAX_JSON + 1)
    if len(data) != before["bytes"] or _identity(path) != before:
        raise ValueError("Session metadata changed while reading")
    return json.loads(data, object_pairs_hook=_pairs)


def _write(path: Path, value: object) -> None:
    _path(str(path))
    with path.open("x", encoding="utf-8") as stream:
        os.fchmod(stream.fileno(), 0o600)
        json.dump(value, stream, sort_keys=True, allow_nan=False)
        stream.flush()
        os.fsync(stream.fileno())


def _integer(value: object, maximum: int) -> int:
    if type(value) is not int or not 0 < value <= maximum:
        raise ValueError("Expected a positive bounded integer")
    return value


def _sha(value: object) -> str:
    if not isinstance(value, str) or re.fullmatch("[a-f0-9]{64}", value) is None:
        raise ValueError("Expected an explicit SHA256")
    return value


def _times(value: object) -> tuple[Fraction, ...]:
    if not isinstance(value, list) or not 1 <= len(value) <= MAX_SAMPLES:
        raise ValueError("Invalid original rational clock inventory")
    result: list[Fraction] = []
    for item in value:
        row = record(item)
        number = row.get("value")
        if set(row) != {"value", "timescale"} or type(number) is not int or number < 0:
            raise ValueError("Invalid source clock")
        point = Fraction(number, _integer(row["timescale"], 1000000000))
        if point > 86400 or (result and point <= result[-1]):
            raise ValueError("Source clocks must be bounded and strictly increasing")
        result.append(point)
    return tuple(result)


def _clocks(samples: tuple[Fraction, ...]) -> list[dict[str, int]]:
    return [{"value": point.numerator, "timescale": point.denominator} for point in samples]


def load_plan(path: Path) -> SessionPlan:
    """Read explicit provider admission; this never hashes or opens the source video."""
    raw = record(_read(path))
    if (
        set(raw) - {"workers", "allowedSourceRoot"}
        != {
            "version",
            "protocol",
            "source",
            "sourceSHA256",
            "nativeExecutable",
            "nativeSHA256",
            "samples",
            "width",
            "height",
        }
        or raw["version"] != 1
        or type(raw["version"]) is not int
        or raw["protocol"] != PROTOCOL
    ):
        raise ValueError("Unsupported session plan/provider version")
    root = _source_root(raw["allowedSourceRoot"]) if "allowedSourceRoot" in raw else None
    result = SessionPlan(
        _source_path(raw["source"], root),
        _path(raw["nativeExecutable"]),
        _sha(raw["sourceSHA256"]),
        _sha(raw["nativeSHA256"]),
        _times(raw["samples"]),
        _integer(raw["width"], 16384),
        _integer(raw["height"], 16384),
        _integer(raw["workers"], 2) if "workers" in raw else None,
        root,
    )
    if root is not None:
        _identity(result.source, root)
    if result.source == result.native or result.width * result.height > 16000000:
        raise ValueError("Source/provider alias or oversized canvas")
    return result


def _pins(plan: SessionPlan, path: Path) -> dict[str, str]:
    expected = {str(plan.source): plan.source_sha, str(plan.native): plan.native_sha}
    expected[str(path)] = file_digest(path)
    for item in capture(("subtitle_session_observations",)):
        expected[str(item.path)] = item.sha256
    return expected


def _check(pins: dict[str, str], plan: SessionPlan | None = None) -> None:
    for name, digest in pins.items():
        root = plan.source_root if plan is not None and Path(name) == plan.source else None
        before = _identity(Path(name), root)
        if file_digest(Path(name)) != digest:
            raise ValueError("Session cryptographic inputs changed")
        if _identity(Path(name), root) != before:
            raise ValueError("Session input identity changed during hashing")


def validate_blocks(
    directory: Path, samples: tuple[Fraction, ...], *, width: int, height: int
) -> tuple[dict[str, str], int]:
    """Validate genuine raw eight-frame blocks and their separate exact clocks."""
    pins: dict[str, str] = {}
    count = 0
    for index in range((len(samples) + 7) // 8):
        raw = directory / f"block-{index}.json"
        clock = directory / f"block-{index}-clocks.json"
        if not clock.exists():
            break
        row = record(_read(clock))
        expected = samples[index * 8 : index * 8 + 8]
        if (
            set(row) != {"firstSample", "requested", "actual", "measurementsPath"}
            or type(row["firstSample"]) is not int
            or row["firstSample"] != index * 8
            or row["measurementsPath"] != str(raw)
            or _times(row["requested"]) != expected
            or _times(row["actual"]) != expected
        ):
            raise ValueError("Session block clocks or raw path differ")
        frames = native_ocr_frames(_read(raw))
        if len(frames) != len(expected) or any(
            frame.requested_time != float(point)
            or frame.actual_time != float(point)
            or frame.width != width
            or frame.height != height
            for frame, point in zip(frames, expected, strict=True)
        ):
            raise ValueError("Raw OCR clocks or canvas differ")
        pins.update({str(raw): file_digest(raw), str(clock): file_digest(clock)})
        count += len(expected)
    return pins, count


def _report(directory: Path, samples: tuple[Fraction, ...], identity: dict[str, int]) -> None:
    report = record(_read(directory / "complete.json"))
    blocks = report.get("blocks")
    if (
        report.get("protocolVersion") != PROTOCOL
        or report.get("sourceModified") is not False
        or report.get("qualityApproved") is not False
        or report.get("sourceHashVerified") is not False
        or report.get("sourceIdentity") != identity
        or type(report.get("frames")) is not int
        or report["frames"] != len(samples)
        or not isinstance(blocks, list)
        or blocks
        != [
            _read(directory / f"block-{index}-clocks.json")
            for index in range((len(samples) + 7) // 8)
        ]
    ):
        raise ValueError("Native session completion does not bind the real input")
    timing = record(report.get("timing"))
    if not timing or any(
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or value < 0
        for value in timing.values()
    ):
        raise ValueError("Invalid native timing evidence")


def _attempt(plan: SessionPlan, directory: Path, start: int, pins: dict[str, str]) -> int:
    directory.mkdir(mode=0o700)
    checkpoint = directory / "native"
    checkpoint.mkdir(mode=0o700)
    scratch = directory / "scratch"
    scratch.mkdir(mode=0o700)
    samples = plan.samples[start : start + 64]
    identity = _identity(plan.source, plan.source_root)
    request: dict[str, object] = {
        "checkoutRoot": str(Path.cwd()),
        "path": str(plan.source),
        "checkpointDirectory": str(checkpoint),
        "samples": _clocks(samples),
        "width": plan.width,
        "height": plan.height,
        "maximumSeconds": 60,
    }
    if plan.workers is not None:
        request["workers"] = plan.workers
    if plan.source_root is not None:
        request["allowedSourceRoot"] = str(plan.source_root.path)
    _write(directory / "request.json", request)
    with (
        (directory / "stdout.json").open("xb") as stdout,
        (directory / "stderr.log").open("xb") as stderr,
    ):
        os.fchmod(stdout.fileno(), 0o600)
        os.fchmod(stderr.fileno(), 0o600)
        try:
            child = subprocess.run(
                [str(plan.native), "measure-media", TOOL, str(directory / "request.json")],
                stdin=subprocess.DEVNULL,
                stdout=stdout,
                stderr=stderr,
                timeout=90,
                check=False,
                env={
                    "PATH": "/usr/bin:/bin:/usr/sbin:/sbin",
                    "TMPDIR": str(scratch),
                    "TMP": str(scratch),
                    "TEMP": str(scratch),
                    "XDG_CACHE_HOME": str(scratch),
                    "LLVM_PROFILE_FILE": str(scratch / "native-%p.profraw"),
                },
            )
            code = child.returncode
        except subprocess.TimeoutExpired:
            code = -1
    _check(pins, plan)
    if _identity(plan.source, plan.source_root) != identity:
        raise ValueError("Source identity changed during session")
    raw_pins, count = validate_blocks(checkpoint, samples, width=plan.width, height=plan.height)
    if code == 0:
        _report(checkpoint, samples, identity)
        if count != len(samples) or _read(directory / "stdout.json") != _read(
            checkpoint / "complete.json"
        ):
            raise ValueError("Native stdout or completed block coverage differs")
        raw_pins[str(checkpoint / "complete.json")] = file_digest(checkpoint / "complete.json")
    raw_pins[str(directory / "request.json")] = file_digest(directory / "request.json")
    if (directory / "stdout.json").stat().st_size:
        raw_pins[str(directory / "stdout.json")] = file_digest(directory / "stdout.json")
    _write(
        directory / "boundary.json",
        {
            "start": start,
            "count": count,
            "exit": code,
            "inputs": pins,
            "sourceIdentity": identity,
            "raw": raw_pins,
        },
    )
    if code != 0:
        raise ValueError("Session failed; verified prefix and raw failure evidence retained")
    return count


def _retained_producer_pins(
    recorded: dict[str, object], expected: dict[str, object], *, verify: bool
) -> dict[str, str]:
    inputs = record(recorded.get("inputs"))
    current = record(expected.get("inputs"))
    if recorded != expected and not (
        verify
        and recorded == {**expected, "inputs": inputs}
        and compat.historical_generation(inputs, current)
    ):
        raise ValueError("Session producer changed; use a separate revision")
    # Membership already matches the validated current pins, including the
    # explicitly admitted read-only external source root.
    return {name: _sha(digest) for name, digest in inputs.items()}


def collect(plan_path: Path, output: Path, *, verify: bool = False) -> int:
    """Resume only boundary-sealed new-provider prefixes; old collectors never merge."""
    plan = load_plan(plan_path)
    output = _path(str(output))
    if not output.is_relative_to(Path.cwd() / "edits/work"):
        raise ValueError("Session output must be repository scratch")
    if not verify:
        output.mkdir(mode=0o700, parents=True, exist_ok=True)
    pins = _pins(plan, plan_path)
    checked_pins = pins
    _check(pins, plan)
    if load_plan(plan_path) != plan:
        raise ValueError("Session plan changed during admission")
    producer = {
        "version": 1,
        "protocolVersion": PROTOCOL,
        "backend": "native-session-cli",
        "measurementTool": TOOL,
        "fullFrame": True,
        "sourceSHA256": plan.source_sha,
        "source": str(plan.source),
        "nativeExecutable": str(plan.native),
        "nativeSHA256": plan.native_sha,
        "inputs": pins,
        "candidateOnly": True,
    }
    if plan.source_root is not None:
        producer["allowedSourceRoot"] = {
            "path": str(plan.source_root.path),
            "device": plan.source_root.device,
            "inode": plan.source_root.inode,
            "readOnly": True,
        }
    with execution_lock(output, "collector") as owned:
        if not owned:
            raise ValueError("Session collector is already owned")
        if (output / "inputs.json").exists():
            pins = _retained_producer_pins(
                record(_read(output / "inputs.json")), producer, verify=verify
            )
        elif verify:
            raise ValueError("Missing session producer")
        else:
            _write(output / "inputs.json", producer)
        count = 0
        boundaries: dict[str, str] = {}
        evidence: dict[str, str] = {}
        attempts = sorted(islice(output.glob("attempt-*"), MAX_SAMPLES + 1))
        if len(attempts) > MAX_SAMPLES:
            raise ValueError("Session attempt budget exhausted")
        for index, attempt in enumerate(attempts):
            if attempt.name != f"attempt-{index:06d}":
                raise ValueError("Unordered session attempts")
            boundary = record(_read(attempt / "boundary.json"))
            if (
                set(boundary) != {"start", "count", "exit", "inputs", "sourceIdentity", "raw"}
                or boundary.get("inputs") != pins
                or type(boundary.get("start")) is not int
                or boundary.get("start") != count
                or type(boundary.get("exit")) is not int
                or boundary.get("sourceIdentity") != _identity(plan.source, plan.source_root)
            ):
                raise ValueError("Unsealed or incompatible session prefix")
            raw = record(boundary.get("raw"))
            raw_pins = {str(_path(name)): _sha(sha) for name, sha in raw.items()}
            _check(raw_pins)
            block_pins, found = validate_blocks(
                attempt / "native",
                plan.samples[count : count + 64],
                width=plan.width,
                height=plan.height,
            )
            if (
                type(boundary.get("count")) is not int
                or boundary.get("count") != found
                or any(raw_pins.get(name) != digest for name, digest in block_pins.items())
                or str(attempt / "request.json") not in raw_pins
                or any(not Path(name).is_relative_to(attempt) for name in raw_pins)
            ):
                raise ValueError("Session prefix coverage differs")
            request = record(_read(attempt / "request.json"))
            samples = plan.samples[count : count + 64]
            expected_request: dict[str, object] = {
                "checkoutRoot": str(Path.cwd()),
                "path": str(plan.source),
                "checkpointDirectory": str(attempt / "native"),
                "samples": _clocks(samples),
                "width": plan.width,
                "height": plan.height,
                "maximumSeconds": 60,
            }
            if plan.workers is not None:
                expected_request["workers"] = plan.workers
            if plan.source_root is not None:
                expected_request["allowedSourceRoot"] = str(plan.source_root.path)
            if request != expected_request:
                raise ValueError("Session request differs from original plan")
            if boundary["exit"] == 0:
                if not {
                    str(attempt / "native/complete.json"),
                    str(attempt / "stdout.json"),
                }.issubset(raw_pins):
                    raise ValueError("Successful session lacks pinned native completion")
                _report(attempt / "native", samples, _identity(plan.source, plan.source_root))
                if found != len(samples):
                    raise ValueError("Successful session has missing blocks")
            evidence.update(raw_pins)
            count += found
            boundaries[str(attempt / "boundary.json")] = file_digest(attempt / "boundary.json")
        if verify and count != len(plan.samples):
            raise ValueError("Incomplete session collection")
        while count < len(plan.samples):
            attempt = output / f"attempt-{len(boundaries):06d}"
            if (output / "complete.json").exists():
                raise ValueError("Completed session lost evidence")
            count += _attempt(plan, attempt, count, pins)
            boundaries[str(attempt / "boundary.json")] = file_digest(attempt / "boundary.json")
        _check(checked_pins, plan)
        if load_plan(plan_path) != plan:
            raise ValueError("Session plan changed at completion")
        _check({**evidence, **boundaries})
        receipt = {
            "version": 1,
            "protocolVersion": PROTOCOL,
            "observations": count,
            "boundaries": boundaries,
            "candidateOnly": True,
            "consumerConnected": False,
        }
        if (output / "complete.json").exists():
            if _read(output / "complete.json") != receipt:
                raise ValueError("Session completion changed")
        elif verify:
            raise ValueError("Missing session completion")
        else:
            _write(output / "complete.json", receipt)
    return count


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("operation", choices=("collect", "verify"))
    parser.add_argument("plan", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    print(collect(args.plan, args.output, verify=args.operation == "verify"))


if __name__ == "__main__":
    main()
