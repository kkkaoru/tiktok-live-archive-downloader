"""Three bounded, identity-stable process-family snapshots; never a peak guarantee."""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path

from replay.core import ReplayError
from replay.native_invocation import canonical, pin, require, rows
from replay.process_family_observation import SystemHost, sample_family
from replay.recordings import record

MAX_REQUEST_BYTES = 65536
MAX_FAMILIES = 16
SAMPLE_COUNT = 3
MAX_INTERVAL_SECONDS = 2.0
REPOSITORY = Path(__file__).resolve().parents[2]
SOURCE_PATHS = (
    Path(__file__).resolve(),
    REPOSITORY / "src/replay/process_family_observation.py",
    REPOSITORY / "src/replay/native_invocation.py",
    REPOSITORY / "src/replay/core.py",
    REPOSITORY / "src/replay/recordings.py",
)


@dataclass(frozen=True)
class Family:
    pid: int
    birth: str


@dataclass(frozen=True)
class Request:
    families: tuple[Family, ...]
    interval_seconds: float


def _unique(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        require(key not in result, "Duplicate JSON key")
        result[key] = value
    return result


def _path(path: Path) -> Path:
    result = canonical(str(path))
    require(
        result.is_relative_to(REPOSITORY / "edits/work") and result.parent.is_dir(),
        "Expected canonical repository work path with existing parent",
    )
    return result


def load(path: Path) -> Request:
    """Validate a bounded external request without accepting unknown fields."""
    _path(path)
    require(0 < path.stat().st_size <= MAX_REQUEST_BYTES, "Invalid request size")
    with path.open("rb") as stream:
        data = stream.read(MAX_REQUEST_BYTES + 1)
    require(len(data) <= MAX_REQUEST_BYTES, "Request grew beyond size bound")
    raw = record(json.loads(data, object_pairs_hook=_unique))
    require(
        set(raw) == {"version", "kind", "sampleCount", "intervalSeconds", "families"}
        and type(raw["version"]) is int
        and raw["version"] == 1
        and raw["kind"] == "process-resource-observation-request"
        and type(raw["sampleCount"]) is int
        and raw["sampleCount"] == SAMPLE_COUNT,
        "Invalid observation request schema/version/count",
    )
    interval = raw["intervalSeconds"]
    require(
        isinstance(interval, (int, float))
        and not isinstance(interval, bool)
        and math.isfinite(interval)
        and 0 <= interval <= MAX_INTERVAL_SECONDS,
        "Invalid sample interval",
    )
    if not isinstance(interval, (int, float)):
        raise ReplayError("Expected numeric interval")
    families = tuple(_family(row) for row in rows(raw["families"]))
    require(0 < len(families) <= MAX_FAMILIES, "Invalid family count")
    require(len({item.pid for item in families}) == len(families), "Duplicate family root")
    return Request(families, float(interval))


def _family(value: object) -> Family:
    raw = record(value)
    require(set(raw) == {"pid", "birth"}, "Invalid family schema")
    pid, birth = raw["pid"], raw["birth"]
    if type(pid) is not int or not isinstance(birth, str):
        raise ReplayError("Invalid process identity types")
    require(0 < pid <= 2147483647, "Invalid process PID")
    require(re.fullmatch(r"[1-9][0-9]*:[0-9]{1,6}", birth) is not None, "Invalid birth")
    require(int(birth.split(":")[1]) < 1000000, "Invalid birth microseconds")
    return Family(pid, birth)


def _members(sample: dict[str, object]) -> tuple[Family, ...]:
    exited = rows(sample.get("exitedDuringObservation"))
    require(all(type(pid) is int and pid > 0 for pid in exited), "Invalid exited identity evidence")
    members = tuple(_family(row) for row in rows(sample.get("members")))
    require(bool(members), "Empty family snapshot")
    require(len({item.pid for item in members}) == len(members), "Duplicate member")
    return tuple(sorted(members, key=lambda item: item.pid))


def _amount(sample: dict[str, object], key: str) -> int:
    value = sample.get(key)
    if type(value) is not int or value <= 0:
        raise ReplayError("Invalid observed byte amount")
    return value


def _summary(samples: list[dict[str, object]], key: str) -> dict[str, object]:
    amounts = [_amount(sample, key) for sample in samples]
    return {
        "maxBytes": max(amounts),
        "meanBytes": sum(amounts) / SAMPLE_COUNT,
        "variationBytes": max(amounts) - min(amounts),
    }


def observe(request: Request, host: SystemHost) -> list[dict[str, object]]:
    """Observe exactly three times and reject overlapping or changing families."""
    snapshots: list[list[dict[str, object]]] = []
    for index in range(SAMPLE_COUNT):
        if index:
            time.sleep(request.interval_seconds)
        current = [sample_family(item.pid, item.birth, host) for item in request.families]
        identities = [_members(sample) for sample in current]
        flattened = [member.pid for family in identities for member in family]
        require(len(set(flattened)) == len(flattened), "Overlapping requested families")
        if snapshots:
            require(
                identities == [_members(sample) for sample in snapshots[0]],
                "Family membership/birth changed between snapshots",
            )
        snapshots.append(current)
    result: list[dict[str, object]] = []
    for index, family in enumerate(request.families):
        samples = [snapshot[index] for snapshot in snapshots]
        members = _members(samples[-1])
        require(
            all(host.birth(member.pid) == member.birth for member in members),
            "Family identity changed after observation",
        )
        result.append(
            {
                "root": {"pid": family.pid, "birth": family.birth},
                "beforeIdentities": samples[0]["members"],
                "afterIdentities": samples[-1]["members"],
                "observerIncluded": any(member.pid == os.getpid() for member in members),
                "samples": samples,
                "physicalBytes": _summary(samples, "physicalBytes"),
                "rssBytes": _summary(samples, "rssBytes"),
            }
        )
    return result


def run(request_path: Path, output: Path) -> None:
    """Reserve a new report before observing; failed reservations remain evidence."""
    output = _path(output)
    request = load(request_path)
    request_pin = pin(request_path)
    require(load(request_path) == request, "Request changed before observation")
    source_pins = [pin(path) for path in SOURCE_PATHS]
    started = time.time_ns()
    host = SystemHost()
    producer = {"pid": os.getpid(), "birth": host.birth(os.getpid())}
    with output.open("x", encoding="utf-8") as stream:
        os.fchmod(stream.fileno(), 0o600)
        observations = observe(request, host)
        require(pin(request_path) == request_pin, "Request changed during observation")
        require([pin(path) for path in SOURCE_PATHS] == source_pins, "Observer source changed")
        json.dump(
            {
                "version": 1,
                "kind": "process-resource-observation",
                "startedAtNs": started,
                "endedAtNs": time.time_ns(),
                "producer": producer,
                "request": request_pin,
                "sources": source_pins,
                "families": observations,
                "sampleCount": SAMPLE_COUNT,
                "intervalSeconds": request.interval_seconds,
                "peakGuarantee": False,
                "quota": False,
                "reclaimGuarantee": False,
                "familyTotalsMayIncludeObserver": True,
            },
            stream,
            sort_keys=True,
        )
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("request", type=Path)
    parser.add_argument("output", type=Path)
    arguments = parser.parse_args(argv)
    try:
        run(arguments.request, arguments.output)
    except (ReplayError, OSError, ValueError) as error:
        print(str(error), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
