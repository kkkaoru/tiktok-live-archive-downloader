"""Resumable local frame evidence; OCR/head samples are not full-edit approval."""

import argparse
import fcntl
import json
import math
import os
import subprocess
import tempfile
from pathlib import Path

from replay.caption_io import native_ocr_frames, ocr_frames
from replay.head_observations import head_frames
from replay.stages import file_digest

BATCH_SIZE = 8
MAX_RESULT_BYTES = 131072


def observation_times(value: object) -> tuple[float, ...]:
    """Validate a bounded ordered plan without inventing missing frame clocks."""
    if not isinstance(value, list) or not 1 <= len(value) <= 50000:
        raise ValueError("Invalid OCR observation plan")
    times: list[float] = []
    for item in value:
        if isinstance(item, bool) or not isinstance(item, (int, float)):
            raise ValueError("Invalid OCR observation time")
        if not 0 <= item <= 86400 or not math.isfinite(item):
            raise ValueError("Invalid OCR observation time")
        time = float(item)
        if times and time <= times[-1]:
            raise ValueError("OCR times must be strictly increasing")
        times.append(time)
    return tuple(times)


def _validate_result(
    path: Path, times: tuple[float, ...], *, native: bool = False, head_tracking: bool = False
) -> None:
    if path.is_symlink() or not 0 < path.stat().st_size <= MAX_RESULT_BYTES:
        raise ValueError("Invalid OCR result file")
    value = json.loads(path.read_text(encoding="utf-8"))
    if head_tracking:
        clocks = tuple(frame.requested_time for frame in head_frames(value, native=native))
    else:
        parse = native_ocr_frames if native else ocr_frames
        clocks = tuple(frame.requested_time for frame in parse(value))
    if clocks != times:
        raise ValueError("OCR result clocks differ from the request")


def _write_new(path: Path, text: str) -> None:
    with path.open("x", encoding="utf-8") as stream:
        os.fchmod(stream.fileno(), 0o600)
        stream.write(text)


def collect_observations(
    source: Path,
    plan: Path,
    output: Path,
    *,
    native_executable: Path | None = None,
    head_tracking: bool = False,
) -> int:
    """Collect full-frame OCR or head geometry; reuse only verified same-mode batches.

    Failed/interrupted attempt directories are retained. Existing result files
    must validate, never be overwritten. Source and plan fingerprints bind reuse;
    Executor remains the default. A caller may explicitly select an authorized
    local apple-pro-apps CLI; no fallback, discovery, install or approval changes
    occur. Its raw JSON is preserved, not disguised as an Executor receipt.
    """
    if native_executable is not None and (
        not native_executable.is_absolute() or not os.access(native_executable, os.X_OK)
    ):
        raise ValueError("Explicit native OCR executable must be absolute and executable")
    native_hash = file_digest(native_executable) if native_executable is not None else None
    source_hash, plan_hash = file_digest(source), file_digest(plan)
    if plan.stat().st_size > 1048576:
        raise ValueError("OCR plan exceeds its byte budget")
    times = observation_times(json.loads(plan.read_text(encoding="utf-8")))
    if output.is_symlink():
        raise ValueError("Linked OCR evidence directory is not accepted")
    output.mkdir(mode=0o700, parents=True, exist_ok=True)
    descriptor = os.open(output / "collector.lock", os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW, 0o600)
    with os.fdopen(descriptor, "a+b") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        manifest = json.dumps(
            {
                "source": str(source.resolve()),
                "sourceSHA256": source_hash,
                "planSHA256": plan_hash,
                "collectorSHA256": file_digest(Path(__file__)),
                "parserSHA256": file_digest(Path(__file__).with_name("caption_io.py")),
                "headParserSHA256": file_digest(Path(__file__).with_name("head_observations.py"))
                if head_tracking
                else None,
                "measurementTool": "video_head_detect" if head_tracking else "video_text_recognize",
                "fullFrame": True,
                "backend": "native-cli" if native_executable is not None else "executor",
                "nativeExecutable": str(native_executable)
                if native_executable is not None
                else None,
                "nativeSHA256": native_hash,
            },
            sort_keys=True,
        )
        inputs = output / "inputs.json"
        if inputs.exists() or inputs.is_symlink():
            if inputs.is_symlink() or inputs.read_text(encoding="utf-8") != manifest:
                raise ValueError("OCR collection inputs changed")
        else:
            _write_new(inputs, manifest)
        complete = output / "complete.json"
        if complete.is_symlink():
            raise ValueError("Linked OCR completion receipt")
        hashes: dict[str, str] = {}
        for start in range(0, len(times), BATCH_SIZE):
            chunk = times[start : start + BATCH_SIZE]
            result = output / f"{start // BATCH_SIZE:04d}.result.json"
            if not result.exists() and not result.is_symlink():
                if complete.exists():
                    raise ValueError("Completed OCR result is missing; refusing recollection")
                _collect_batch(
                    source,
                    chunk,
                    output,
                    result,
                    native_executable=native_executable,
                    head_tracking=head_tracking,
                )
            _validate_result(
                result, chunk, native=native_executable is not None, head_tracking=head_tracking
            )
            hashes[result.name] = file_digest(result)
        if native_executable is not None and file_digest(native_executable) != native_hash:
            raise ValueError("Native OCR executable changed during collection")
        if file_digest(source) != source_hash or file_digest(plan) != plan_hash:
            raise ValueError("OCR inputs changed during collection")
        receipt = json.dumps(
            {"observations": len(times), "samplingExhaustive": False, "results": hashes},
            sort_keys=True,
        )
        if complete.exists() or complete.is_symlink():
            if complete.is_symlink() or complete.read_text(encoding="utf-8") != receipt:
                raise ValueError("Completed OCR evidence changed")
        else:
            _write_new(complete, receipt)
    return len(times)


def _collect_batch(
    source: Path,
    times: tuple[float, ...],
    output: Path,
    result: Path,
    *,
    native_executable: Path | None,
    head_tracking: bool = False,
) -> None:
    attempt = Path(tempfile.mkdtemp(prefix="attempt-", dir=output))
    request = json.dumps(
        {"path": str(source.resolve()), "samples": [{"timeSeconds": time} for time in times]}
    )
    _write_new(attempt / "request.json", request)
    raw = attempt / "response.json"
    tool = "video_head_detect" if head_tracking else "video_text_recognize"
    with raw.open("xb") as stdout, (attempt / "stderr.log").open("xb") as stderr:
        os.fchmod(stdout.fileno(), 0o600)
        os.fchmod(stderr.fileno(), 0o600)
        response = subprocess.run(
            [
                str(native_executable),
                "measure-media",
                tool,
                str((attempt / "request.json").resolve()),
            ]
            if native_executable is not None
            else [
                "executor",
                "call",
                "apple-pro-apps",
                "org",
                "default",
                tool,
                request,
            ],
            stdin=subprocess.DEVNULL,
            stdout=stdout,
            stderr=stderr,
            timeout=120,
            check=False,
        )
    if response.returncode != 0:
        raise ValueError("Local OCR command failed; attempt evidence retained")
    _validate_result(raw, times, native=native_executable is not None, head_tracking=head_tracking)
    os.link(raw, result)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("plan", type=Path, help="JSON array of source-second observation times")
    parser.add_argument("output", type=Path, help="New or unchanged resumable evidence directory")
    parser.add_argument(
        "--native-executable",
        type=Path,
        help="Explicit authorized apple-pro-apps CLI (no fallback)",
    )
    parser.add_argument(
        "--head-tracking", action="store_true", help="Collect face/person boxes instead of OCR"
    )
    args = parser.parse_args()
    count = collect_observations(
        args.source,
        args.plan,
        args.output,
        native_executable=args.native_executable,
        head_tracking=args.head_tracking,
    )
    kind = "head" if args.head_tracking else "OCR"
    print(f"Verified local {kind} observations: {count}; exhaustive coverage not claimed")


if __name__ == "__main__":
    main()
