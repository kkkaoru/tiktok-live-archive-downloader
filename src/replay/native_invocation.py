"""Bind dynamic edit-media output at invocation time, never by later adoption.

An exclusive reservation survives Stage output recovery and forbids resubmission.
Interrupted attempts preserve stdout/stderr and movies; only a complete binding
is reusable. This is provenance, not decode, clock or visual-quality approval.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import stat
import subprocess
import sys
from pathlib import Path

from replay.core import ReplayError
from replay.recordings import record

VERSION = 1
MAX_JSON_BYTES = 16 * 1024 * 1024
TIMEOUT = 1740


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ReplayError(message)


def canonical(value: object) -> Path:
    require(isinstance(value, str) and bool(value), "Missing absolute path")
    path = Path(str(value))
    require(path.is_absolute() and path == path.resolve(), "Noncanonical/linked path")
    return path


def identity(path: Path) -> dict[str, int]:
    canonical(str(path))
    info = path.lstat()
    require(stat.S_ISREG(info.st_mode) and info.st_nlink == 1, "Expected regular singlelink file")
    # The device number is renumbered by remounts and differs on another volume, so it is not
    # recorded; size, inode and both times still bind the file.
    return {
        "inode": info.st_ino,
        "bytes": info.st_size,
        "mtime_ns": info.st_mtime_ns,
        "ctime_ns": info.st_ctime_ns,
    }


def pin(path: Path) -> dict[str, object]:
    before = identity(path)
    require(before["bytes"] > 0, "Empty invocation input/output")
    with path.open("rb") as stream:
        opened = os.fstat(stream.fileno())
        require(
            (opened.st_dev, opened.st_ino) == (path.lstat().st_dev, before["inode"]),
            "File replaced before hashing",
        )
        sha = hashlib.file_digest(stream, "sha256").hexdigest()
    require(identity(path) == before, "File changed during hashing")
    return {"path": str(path), "sha256": sha, "bytes": before["bytes"], "stat": before}


def read(path: Path) -> dict[str, object]:
    require(0 < identity(path)["bytes"] <= MAX_JSON_BYTES, "Invalid JSON size")
    return record(json.loads(path.read_bytes()))


def rows(value: object) -> list[object]:
    if not isinstance(value, list):
        raise ReplayError("Expected invocation list")
    return value


def inputs(native: Path, request: Path) -> tuple[Path, ...]:
    recipe = record(read(request).get("recipe"))
    clips = rows(recipe.get("clips"))
    require(bool(clips), "Missing native clips")
    # This wrapper owns the video-only edit-media route, not unbound audio inputs.
    require(
        not recipe.get("additionalAudio") and not recipe.get("audio"), "Unsupported audio input"
    )
    media = [
        canonical(record(row).get("sourcePath"))
        for row in [*clips, *rows(recipe.get("additionalVideo", []))]
    ]
    return tuple(dict.fromkeys((request, native, *media)))


def paths(request: Path) -> tuple[Path, Path, Path]:
    return (
        request.with_name("render.invocation.json"),
        request.with_name("render.response.json"),
        request.with_name("render.binding.json"),
    )


def command(
    native: Path, request: Path, output_directory: Path, *, verify: bool = False
) -> tuple[str, ...]:
    return (
        sys.executable,
        "-m",
        "replay.native_invocation",
        "verify" if verify else "run",
        str(native),
        str(request),
        str(output_directory),
    )


def durable(path: Path, data: object) -> None:
    canonical(str(path))
    with path.open("x", encoding="utf-8") as stream:
        os.fchmod(stream.fileno(), 0o600)
        json.dump(data, stream, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    sync_directory(path.parent)


def sync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def context(native: Path, request: Path, output_directory: Path) -> dict[str, object]:
    canonical(str(output_directory))
    configured = canonical(read(request).get("outputDirectory"))
    require(
        configured.is_relative_to(output_directory),
        "Request output directory escapes authorization",
    )
    return {
        "version": VERSION,
        "fingerprintMoment": "before-native-call",
        "inputContract": "readonly-immutable-through-output-binding",
        "command": [str(native), "edit-media", str(request)],
        "authorizedOutputDirectory": str(output_directory),
        "requestOutputDirectory": str(configured),
    }


def run(native: Path, request: Path, output_directory: Path) -> Path:
    start, response, binding = paths(request)
    stderr = request.with_name("render.stderr")
    reservation = request.with_name("render.attempt")
    require(
        not any(
            p.exists() or p.is_symlink() for p in (start, response, binding, stderr, reservation)
        ),
        "Invocation already attempted; reconcile unknown native outcome, never resubmit",
    )
    facts = context(native, request, output_directory)
    before = [pin(path) for path in inputs(native, request)]
    require(context(native, request, output_directory) == facts, "Request changed before call")
    # Stage recovery may rename every declared output. This separate reservation
    # remains beside the immutable request and prevents that from authorizing a call.
    canonical(str(reservation)).mkdir(mode=0o700)
    sync_directory(reservation.parent)
    durable(start, {**facts, "inputs": before})
    invocation_pin = pin(start)
    directory = canonical(facts["requestOutputDirectory"])
    directory.mkdir(parents=True, exist_ok=True, mode=0o700)
    require(directory == directory.resolve(), "Output directory changed")
    with response.open("xb") as stdout, stderr.open("xb") as errors:
        os.fchmod(stdout.fileno(), 0o600)
        os.fchmod(errors.fileno(), 0o600)
        require(
            all(identity(canonical(row["path"])) == row["stat"] for row in before),
            "Input changed before native call",
        )
        try:
            result = subprocess.run(
                [str(native), "edit-media", str(request)],
                stdin=subprocess.DEVNULL,
                stdout=stdout,
                stderr=errors,
                timeout=TIMEOUT,
                check=False,
            )
        finally:
            stdout.flush()
            errors.flush()
            os.fsync(stdout.fileno())
            os.fsync(errors.fileno())
            sync_directory(response.parent)
    require(result.returncode == 0, "Native call failed; response/output retained")
    response_pin = pin(response)
    raw = read(response)
    require(
        type(raw.get("frameCount")) is int and int(str(raw["frameCount"])) > 0,
        "Incomplete native success response",
    )
    require(
        all(
            type(raw.get(key)) in (int, float)
            and math.isfinite(float(str(raw[key])))
            and float(str(raw[key])) > 0
            for key in ("expectedDurationSeconds", "actualDurationSeconds")
        ),
        "Incomplete native success durations",
    )
    movie = canonical(raw.get("outputPath"))
    require(movie.is_relative_to(directory), "Native output escapes request output directory")
    require(movie not in inputs(native, request), "Native output aliases an input")
    actual = pin(movie)
    after = [pin(canonical(row["path"])) for row in before]
    require(after == before, "Native call changed immutable input")
    require(
        pin(start) == invocation_pin
        and pin(response) == response_pin
        and identity(movie) == actual["stat"],
        "Output/response changed",
    )
    durable(
        binding,
        {
            **facts,
            "status": "complete",
            "invocation": invocation_pin,
            "inputs": before,
            "inputsAfter": after,
            "response": response_pin,
            "output": actual,
        },
    )
    return binding


def verify(native: Path, request: Path, output_directory: Path) -> dict[str, object]:
    """Read-only verification; missing/partial records never trigger a call."""
    start, response, binding = paths(request)
    saved = read(binding)
    facts = context(native, request, output_directory)
    before = [pin(path) for path in inputs(native, request)]
    require(read(start) == {**facts, "inputs": before}, "Pre-call invocation binding differs")
    movie = canonical(read(response).get("outputPath"))
    require(
        movie.is_relative_to(canonical(facts["requestOutputDirectory"])),
        "Native output escapes authorization",
    )
    require(movie not in inputs(native, request), "Native output aliases input")
    expected = {
        **facts,
        "status": "complete",
        "invocation": pin(start),
        "inputs": before,
        "inputsAfter": before,
        "response": pin(response),
        "output": pin(movie),
    }
    require(saved == expected, "Runtime output binding changed or incomplete")
    return saved


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("run", "verify"))
    parser.add_argument("native", type=Path)
    parser.add_argument("request", type=Path)
    parser.add_argument("output_directory", type=Path)
    args = parser.parse_args(argv)
    try:
        if args.action == "run":
            run(args.native, args.request, args.output_directory)
        else:
            verify(args.native, args.request, args.output_directory)
    except (ReplayError, OSError, ValueError, subprocess.SubprocessError) as error:
        print(f"NATIVE_INVOCATION_ERROR: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
