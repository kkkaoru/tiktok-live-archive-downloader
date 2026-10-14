"""Prepare and independently verify checkpoint-only, origin-zero audio observations."""

import argparse
import json
import math
import subprocess
import sys
import tempfile
from collections.abc import Sequence
from fractions import Fraction
from pathlib import Path

from replay.native_invocation import canonical, durable, read, require, rows
from replay.recordings import record
from replay.stages import file_digest

RATE = 48000
MAX_CHECKPOINT_SAMPLES = 60 * RATE
MAX_RECORD_BYTES = 4096


class Arguments(argparse.Namespace):
    action: str
    request: Path
    native: Path
    ffmpeg: Path
    ffprobe: Path
    directory: Path


def integer(value: object, lower: int = 0) -> int:
    require(type(value) is int, "Expected integer")
    result = int(str(value))
    require(result >= lower, "Integer below bound")
    return result


def time(value: object) -> Fraction:
    data = record(value)
    return Fraction(integer(data.get("value"), -(2**63)), integer(data.get("timescale"), 1))


def checkpoints(request: dict[str, object]) -> list[tuple[Fraction, int]]:
    selection = record(request.get("statefulSelection"))
    result = [
        (
            Fraction(integer(row.get("startValue")), integer(row.get("timescale"), 1)),
            integer(row.get("sampleCount"), 1),
        )
        for value in rows(selection.get("checkpoints"))
        for row in [record(value)]
    ]
    require(1 <= len(result) <= 3, "Expected early/late/end bounded checkpoints")
    material = selection.get("productionMaterial") is True
    ceiling = MAX_CHECKPOINT_SAMPLES
    if material:
        ceiling = integer(record(selection.get("budget")).get("maxDecodedSamples"), 1)
        require(
            len(result) == 1
            and result[0] == (Fraction(0), ceiling)
            and Fraction(ceiling, RATE)
            == Fraction(
                integer(request.get("durationValue"), 1), integer(request.get("timescale"), 1)
            )
            and ceiling <= 21600 * RATE,
            "Invalid production material extent",
        )
    end = Fraction(0)
    for start, count in result:
        require(start >= end and count <= ceiling, "Invalid source checkpoint")
        end = start + Fraction(count, RATE)
    require(request.get("startValue") == 0, "Reader must start at source zero")
    require(
        end
        <= Fraction(integer(request.get("durationValue"), 1), integer(request.get("timescale"), 1)),
        "Insufficient read extent",
    )
    return result


def verify_ledger(request: dict[str, object], directory: Path) -> dict[str, object]:
    result = read(directory / "stateful.json")
    require(
        result.get("version") == 1 and time(result.get("origin")) == 0, "Incomplete stateful result"
    )
    selection = record(request.get("statefulSelection"))
    # RSS belongs to the external Python watchdog, not Swift's native selection schema.
    native_budget = dict(record(selection.get("budget")))
    native_budget.pop("maxRSSBytes", None)
    require(result.get("selection") == {**selection, "budget": native_budget}, "Selection changed")
    selected = checkpoints(request)
    material = selection.get("productionMaterial") is True
    retained = [0] * len(selected)
    budget = record(record(request.get("statefulSelection")).get("budget"))
    ledger = directory / "ledger.jsonl"
    require(
        ledger.stat().st_size == result.get("ledgerBytes")
        and ledger.stat().st_size <= integer(budget.get("maxLedgerBytes"), 1),
        "Ledger size changed",
    )
    require(file_digest(ledger) == result.get("ledgerSHA256"), "Ledger hash changed")
    ordinal = 0
    records = 0
    previous_end = Fraction(0)
    with ledger.open("rb") as stream:
        while line := stream.readline(MAX_RECORD_BYTES + 1):
            require(len(line) <= MAX_RECORD_BYTES and line.endswith(b"\n"), "Invalid ledger line")
            row = record(json.loads(line))
            require(
                row.get("bufferOrdinal") == records and row.get("sampleOrdinal") == ordinal,
                "Ordinal units changed",
            )
            start = time(row.get("sourcePTS"))
            count = integer(row.get("samples"), 1)
            duration = time(row.get("duration"))
            require(start >= 0 and duration == Fraction(count, RATE), "Unknown sample extent")
            require(
                time(row.get("discontinuity")) == start - previous_end, "Unrecorded discontinuity"
            )
            for index, (target, expected) in enumerate(selected):
                end = target + Fraction(expected, RATE)
                require(
                    start == previous_end
                    or min(start, previous_end) >= end
                    or max(start, previous_end) <= target,
                    "Ambiguous checkpoint support",
                )
                lower, upper = max(start, target), min(start + duration, end)
                if lower < upper:
                    offset, length = (lower - start) * RATE, (upper - lower) * RATE
                    require(
                        offset.denominator == length.denominator == 1
                        and lower == target + Fraction(retained[index], RATE),
                        "Partial sample/gap",
                    )
                    retained[index] += int(length)
            records += 1
            ordinal += count
            require(
                records <= integer(budget.get("maxRecords"), 1)
                and ordinal <= integer(budget.get("maxDecodedSamples"), 1),
                "Ledger exceeds budget",
            )
            previous_end = start + duration
    require(
        records > 0
        and records == result.get("records")
        and ordinal == result.get("decodedSamples")
        and retained == ([ordinal] if material else [count for _, count in selected]),
        "Incomplete observation",
    )
    if material:
        require(
            time(result.get("supportedEnd")) == previous_end == Fraction(ordinal, RATE)
            and 0 < time(result.get("assetDuration")) <= Fraction(selected[0][1], RATE),
            "Invalid natural EOF accounting",
        )
    return result


def verify(
    request_path: Path, ffmpeg: Path, ffprobe: Path, *, guard_report: Path | None = None
) -> None:
    request = read(request_path)
    budget = record(record(request.get("statefulSelection")).get("budget"))
    if "maxRSSBytes" in budget:
        # Deferred: watchdog shares this module's strict budget parser.
        from replay.stateful_guard import verify_guard

        require(guard_report is not None, "Missing external watchdog result")
        if guard_report is not None:
            verify_guard(request_path, guard_report)
    directory = canonical(request.get("outputDirectory"))
    result = verify_ledger(request, directory)
    material = record(request.get("statefulSelection")).get("productionMaterial") is True
    for index, (_, requested_count) in enumerate(checkpoints(request)):
        count = integer(result.get("decodedSamples"), 1) if material else requested_count
        movie = directory / f"checkpoint-{index}.mov"
        raw = subprocess.run(
            [str(ffprobe), "-v", "error", "-show_streams", "-of", "json", str(movie)],
            check=True,
            capture_output=True,
            timeout=30,
        )
        streams = rows(record(json.loads(raw.stdout)).get("streams"))
        require(len(streams) == 1, "Unexpected checkpoint streams")
        audio = record(streams[0])
        require(
            audio.get("sample_rate") == "48000"
            and audio.get("channels") == 2
            and audio.get("codec_name") == "pcm_s16le",
            "Unexpected checkpoint PCM format",
        )
        duration = Fraction(str(audio.get("time_base"))) * integer(audio.get("duration_ts"), 1)
        require(duration == Fraction(count, RATE), "Inexact checkpoint presentation")
        # Explicit disk-backed readback avoids retaining a whole recording in RAM.
        with tempfile.TemporaryFile(dir=directory.parent) as decoded:
            process = subprocess.run(
                [
                    str(ffmpeg),
                    "-v",
                    "error",
                    "-nostdin",
                    "-i",
                    str(movie),
                    "-map",
                    "0:a:0",
                    "-f",
                    "s16le",
                    "-c:a",
                    "pcm_s16le",
                    "-",
                ],
                check=True,
                stdout=decoded,
                stderr=subprocess.PIPE,
                timeout=integer(budget.get("maxWallSeconds"), 1) if material else 30,
            )
            require(
                not process.stderr and decoded.tell() == count * 4, "Incomplete checkpoint decode"
            )


def prepare(request_path: Path, native: Path, ffmpeg: Path, ffprobe: Path, directory: Path) -> Path:
    request_path, native, ffmpeg, ffprobe, directory = (
        canonical(str(path)) for path in (request_path, native, ffmpeg, ffprobe, directory)
    )
    require(directory.is_relative_to(Path.cwd() / "edits/work"), "Preparation must stay local")
    request = read(request_path)
    checkpoints(request)
    budget = record(record(request.get("statefulSelection")).get("budget"))
    wall = budget.get("maxWallSeconds")
    require(type(wall) in (int, float), "Missing wall budget")
    seconds = float(str(wall))
    require(math.isfinite(seconds) and 0 < seconds <= 3600, "Invalid wall budget")
    output = canonical(request.get("outputDirectory"))
    require(
        output.is_relative_to(Path.cwd() / "edits/work") and not output.exists(),
        "Output must be a new local namespace",
    )
    source = canonical(request.get("source"))
    code = Path(__file__).resolve()
    inputs = [
        request_path,
        native,
        source,
        Path(sys.executable).resolve(),
        ffmpeg,
        ffprobe,
        code,
        Path.cwd() / "src/replay/workflow.py",
        Path.cwd() / "src/replay/stages.py",
        Path.cwd() / "src/replay/native_invocation.py",
        Path.cwd() / "src/replay/recordings.py",
        Path.cwd() / "native/apple-pro-apps/Package.swift",
        *sorted((Path.cwd() / "native/apple-pro-apps/Sources/AudioClockProbe").glob("*.swift")),
    ]
    outputs = [
        output / "stateful.json",
        output / "ledger.jsonl",
        *(output / f"checkpoint-{index}.mov" for index in range(len(checkpoints(request)))),
    ]
    command = [str(native), str(request_path)]
    if "maxRSSBytes" in budget:
        integer(budget.get("maxRSSBytes"), 1)
        guard_report = directory / "guard.json"
        command = [
            sys.executable,
            "-m",
            "replay.stateful_guard",
            str(request_path),
            str(native),
            str(guard_report),
        ]
        inputs.extend([Path.cwd() / "src/replay/stateful_guard.py", Path("/bin/ps").resolve()])
        outputs.append(guard_report)
    workflow = {
        "version": 1,
        "stages": [
            {
                "name": "source-pts-observation",
                "command": command,
                "inputs": [str(path) for path in inputs],
                "outputs": [str(path) for path in outputs],
                "timeout_seconds": math.ceil(seconds),
                "adopt_existing": False,
                "verify_command": [
                    sys.executable,
                    "-m",
                    "replay.stateful_audio",
                    "verify",
                    str(request_path),
                    str(native),
                    str(ffmpeg),
                    str(ffprobe),
                    str(directory),
                ],
            }
        ],
    }
    directory.mkdir(mode=0o700)
    plan = directory / "workflow.json"
    durable(plan, workflow)
    return plan


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "verify"))
    parser.add_argument("request", type=Path)
    parser.add_argument("native", type=Path)
    parser.add_argument("ffmpeg", type=Path)
    parser.add_argument("ffprobe", type=Path)
    parser.add_argument("directory", type=Path)
    args = Arguments()
    parser.parse_args(argv, namespace=args)
    if args.action == "prepare":
        print(prepare(args.request, args.native, args.ffmpeg, args.ffprobe, args.directory))
    else:
        verify(args.request, args.ffmpeg, args.ffprobe, guard_report=args.directory / "guard.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
