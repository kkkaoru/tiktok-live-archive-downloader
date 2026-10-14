"""Resume Metal recognition in a new namespace with actual retained child receipts.

Only exit-zero, family-gone windows are adopted. Uncertain source windows remain
untouched; uncertain successor windows require an explicitly prepared successor.
This produces raw recognition on the original PCM clock, never delivery approval.
"""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import os
import subprocess
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from pathlib import Path
from typing import Protocol

from replay import isolated_metal_asr as lane
from replay.audio_windows import AudioWindow, plan_audio_windows
from replay.core import ReplayError
from replay.fcp_recovery_runtime import SystemHost
from replay.native_invocation import canonical, durable, identity, pin, require
from replay.recordings import record
from replay.sequence_journal import bounded_read

POLICY = "retained-verified-children-plus-fresh-child-per-missing-window"
ARTIFACTS = (
    "complete.json",
    "intent.json",
    "launch.json",
    "child.json",
    "raw.json",
    "resources.json",
    "failed.json",
    "worker.log",
    "mlx-memory.json",
    *(f"{phase}-descriptors.json" for phase in lane.PHASES),
)


class Quiescence(Protocol):
    def absent(self, pid: int) -> bool: ...


def _local(path: Path) -> Path:
    checked = canonical(str(path))
    require(checked.is_relative_to(Path.cwd().resolve() / "edits/work"), "Work-local path required")
    return checked


@contextmanager
def _lease(directory: Path, *, create: bool = False) -> Iterator[None]:
    flags = os.O_RDWR | os.O_NOFOLLOW | (os.O_CREAT if create else 0)
    descriptor = os.open(directory / ".run.lock", flags, 0o600)
    try:
        require(
            identity(directory / ".run.lock")["inode"] == os.fstat(descriptor).st_ino,
            "Collector lease changed",
        )
        fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        yield
    finally:
        os.close(descriptor)


def _group_gone(pid: int) -> None:
    result = subprocess.run(
        ["/bin/ps", "-axo", "pid=,pgid="], check=True, capture_output=True, text=True, timeout=2
    )
    require(len(result.stdout) <= 16 * 1024 * 1024, "Oversized process observation")
    for line in result.stdout.splitlines():
        fields = line.split()
        require(
            len(fields) == 2 and all(item.isdecimal() for item in fields),
            "Invalid process group observation",
        )
        require(int(fields[1]) != pid, "Source child process group still present")


def _quiescent(work: Path, host: Quiescence) -> None:
    if not work.exists():
        return
    canonical(str(work))
    launch = work / "launch.json"
    require(launch.exists(), "Uncertain source lacks child identity; reconcile before adoption")
    actor = lane._read(launch)
    pid = lane._integer(actor.get("pid"), 1, 2**31 - 1)
    require(isinstance(actor.get("birth"), str) and bool(actor.get("birth")), "Missing child birth")
    require(host.absent(pid), "Source child still present")
    _group_gone(pid)
    resources = work / "resources.json"
    if resources.exists():
        samples = lane._read(resources).get("samples")
        if not isinstance(samples, list):
            raise ReplayError("Invalid source family measurements")
        for sample in samples:
            members = record(sample).get("members")
            if not isinstance(members, list):
                raise ReplayError("Missing source family identities")
            for member in members:
                child = lane._integer(record(member).get("pid"), 1, 2**31 - 1)
                require(host.absent(child), "Source descendant still present")


def _origin(request: lane.Request, window: AudioWindow, work: Path) -> dict[str, object]:
    _local(work)
    receipt = work / "complete.json"
    expected = {
        "version": 1,
        "device": "metal",
        "requestSHA256": request.sha256,
        "startSample": window.start_sample,
        "sampleCount": window.sample_count,
        "evidence": lane._evidence(work),
        "deliveryComplete": False,
    }
    require(lane._read(receipt) == expected, "Original window provenance changed")
    intent = lane._read(work / "intent.json")
    require(
        intent.get("requestSHA256") == request.sha256
        and intent.get("startSample") == window.start_sample
        and intent.get("sampleCount") == window.sample_count
        and intent.get("device") == "metal",
        "Original intent differs",
    )
    child = lane._read(work / "child.json")
    require(
        child.get("request") == str(request.path)
        and child.get("requestSHA256") == request.sha256
        and child.get("startSample") == window.start_sample
        and child.get("sampleCount") == window.sample_count,
        "Original child request differs",
    )
    return pin(receipt)


def _receipt(
    request: lane.Request, window: AudioWindow, output: Path, origin: Path, kind: str
) -> dict[str, object]:
    original = _origin(request, window, origin)
    raw_path = output / f"{window.start_sample:012d}.raw.json"
    clock_path = output / f"{window.start_sample:012d}.clock.json"
    require(lane._read(raw_path) == lane._read(origin / "raw.json"), "Retained raw differs")
    require(
        lane._read(clock_path) == lane._clock(window, str(pin(raw_path)["sha256"])),
        "Retained clock differs",
    )
    return {
        "version": 1,
        "kind": kind,
        "requestSHA256": request.sha256,
        "origin": original,
        "raw": pin(raw_path),
        "clock": pin(clock_path),
        "deliveryComplete": False,
    }


def _published(request: lane.Request, window: AudioWindow, source: Path) -> Path | None:
    receipt = source / f"{window.start_sample:012d}.receipt.json"
    if receipt.exists():
        saved = lane._read(receipt)
        origin = canonical(record(saved.get("origin")).get("path")).parent
        kind = saved.get("kind")
        require(kind in ("retained", "fresh"), "Invalid window kind")
        require(saved == _receipt(request, window, source, origin, str(kind)), "Receipt changed")
        return origin
    work = source / f"{window.start_sample:012d}.metal-runtime"
    if (work / "complete.json").exists():
        _origin(request, window, work)
        raw_path = source / f"{window.start_sample:012d}.raw.json"
        clock_path = source / f"{window.start_sample:012d}.clock.json"
        if raw_path.exists():
            require(
                lane._read(raw_path) == lane._read(work / "raw.json"),
                "Original published raw differs",
            )
        if clock_path.exists():
            require(
                raw_path.exists()
                and lane._read(clock_path) == lane._clock(window, str(pin(raw_path)["sha256"])),
                "Original published clock differs",
            )
        return work
    return None


def _snapshot(request: lane.Request, source: Path, host: Quiescence) -> dict[str, object]:
    manifest = lane._read(source / "inputs.json")
    inputs = {
        str(request.path): request.sha256,
        **{
            str(path): request.pins[path] for path in (request.pcm, request.weights, request.config)
        },
    }
    require(
        manifest.get("version") == 1
        and manifest.get("samples") == request.samples
        and manifest.get("sampleRate") == lane.RATE
        and manifest.get("windowSamples") == 60 * lane.RATE
        and manifest.get("strideSamples") == 30 * lane.RATE
        and manifest.get("clock") == "input-pcm-not-projected"
        and manifest.get("device") == "metal",
        "Source manifest contract differs",
    )
    bound = record(manifest.get("inputs"))
    require(
        all(bound.get(name) == digest for name, digest in inputs.items()), "Source inputs differ"
    )
    require(1 <= len(bound) <= 16, "Invalid source input bindings")
    for name, digest in bound.items():
        path = canonical(name)
        require(
            path.is_relative_to(Path.cwd().resolve()) and pin(path)["sha256"] == digest,
            "Source input fingerprint changed",
        )
    retained: dict[str, object] = {}
    artifacts: dict[str, object] = {}
    for window in _windows(request):
        work = source / f"{window.start_sample:012d}.metal-runtime"
        _quiescent(work, host)
        origin = _published(request, window, source)
        if origin is not None:
            _quiescent(origin, host)
            retained[str(window.start_sample)] = _origin(request, window, origin)
        publications = (
            source / f"{window.start_sample:012d}.raw.json",
            source / f"{window.start_sample:012d}.clock.json",
            source / f"{window.start_sample:012d}.receipt.json",
        )
        for publication in publications:
            if publication.exists():
                artifacts[str(publication)] = pin(publication)
        for name in ARTIFACTS:
            path = work / name
            if path.exists():
                artifacts[str(path)] = hashlib.sha256(bounded_read(path)).hexdigest()
    complete = source / "complete.json"
    if complete.exists():
        artifacts[str(complete)] = pin(complete)
    return {"manifest": pin(source / "inputs.json"), "retained": retained, "artifacts": artifacts}


def _windows(request: lane.Request) -> tuple[AudioWindow, ...]:
    return plan_audio_windows(
        request.samples, window_samples=60 * lane.RATE, stride_samples=30 * lane.RATE
    )


def _manifest(source: Path, snapshot: dict[str, object]) -> dict[str, object]:
    manifest = lane._read(source / "inputs.json")
    manifest.update(
        {
            "processPolicy": POLICY,
            "sourceCollection": str(source),
            "sourceSnapshot": snapshot,
            "resumeCode": pin(Path(__file__).resolve()),
        }
    )
    bindings = record(manifest.get("inputs"))
    bindings[str(source / "inputs.json")] = pin(source / "inputs.json")["sha256"]
    manifest["inputs"] = bindings
    require(len(bindings) <= 16, "Too many successor generations for assembly")
    return manifest


def prepare(
    request_path: Path, source: Path, output: Path, *, host: Quiescence | None = None
) -> Path:
    """Create an immutable adoption plan; never launch workers or change source files."""
    request = lane.load_request(request_path)
    source, output = _local(source), _local(output)
    require(
        not output.exists()
        and not output.is_relative_to(source)
        and not source.is_relative_to(output),
        "A disjoint new namespace is required",
    )
    with _lease(source):
        snapshot = _snapshot(request, source, host or SystemHost())
        lane.verify_inputs(request)
        manifest = _manifest(source, snapshot)
        output.mkdir(mode=0o700, parents=True)
        with _lease(output, create=True):
            durable(output / "inputs.json", manifest)
    return output / "inputs.json"


def _fresh(
    request: lane.Request,
    window: AudioWindow,
    output: Path,
    runtime: lane.Runtime,
    identities: dict[str, object],
) -> Path:
    work = output / f"{window.start_sample:012d}.metal-runtime"
    if (work / "complete.json").exists():
        _origin(request, window, work)
        return work
    require(
        not work.exists(), "Uncertain/failed new window: prepare explicit fresh successor; no retry"
    )
    work.mkdir(mode=0o700)
    durable(
        work / "intent.json",
        {
            "requestSHA256": request.sha256,
            "startSample": window.start_sample,
            "sampleCount": window.sample_count,
            "device": "metal",
            "inputIdentities": identities,
        },
    )
    terminal = work / "complete.json"
    try:
        runtime.execute(request, window, work, identities)
        lane._identities(identities)
        durable(
            terminal,
            {
                "version": 1,
                "device": "metal",
                "requestSHA256": request.sha256,
                "startSample": window.start_sample,
                "sampleCount": window.sample_count,
                "evidence": lane._evidence(work),
                "deliveryComplete": False,
            },
        )
    finally:
        if not terminal.exists():
            durable(
                work / "failed.json",
                {
                    "version": 1,
                    "effect": "uncertain",
                    "automaticRetryAllowed": False,
                    "requestSHA256": request.sha256,
                },
            )
    return work


def run(
    request_path: Path,
    output: Path,
    *,
    runtime: lane.Runtime | None = None,
    host: Quiescence | None = None,
    verify_only: bool = False,
    maximum_new_windows: int | None = None,
) -> Path:
    """Resume completed receipts; refuse implicit retries of interrupted new children."""
    require(
        maximum_new_windows is None
        or (
            not verify_only and type(maximum_new_windows) is int and 1 <= maximum_new_windows <= 720
        ),
        "Invalid new-window limit",
    )
    request = lane.load_request(request_path)
    identities = lane.verify_inputs(request)
    output = _local(output)
    manifest = lane._read(output / "inputs.json")
    require(
        manifest.get("processPolicy") == POLICY
        and manifest.get("resumeCode") == pin(Path(__file__).resolve()),
        "Resume implementation changed",
    )
    source = _local(canonical(manifest.get("sourceCollection")))
    with _lease(source), _lease(output):
        snapshot = _snapshot(request, source, host or SystemHost())
        require(
            manifest == _manifest(source, snapshot), "Source snapshot or resume manifest changed"
        )
        retained = record(snapshot["retained"])
        hashes: list[str] = []
        receipts: list[dict[str, object]] = []
        fresh = 0
        for window in _windows(request):
            receipt_path = output / f"{window.start_sample:012d}.receipt.json"
            ref = retained.get(str(window.start_sample))
            kind = "retained" if ref is not None else "fresh"
            origin = (
                canonical(record(ref).get("path")).parent
                if ref is not None
                else output / f"{window.start_sample:012d}.metal-runtime"
            )
            if not receipt_path.exists():
                require(not verify_only, "Missing completed window receipt")
                if ref is None:
                    if maximum_new_windows is not None and fresh >= maximum_new_windows:
                        lane.verify_inputs(request)
                        return output / "inputs.json"
                    origin = _fresh(
                        request, window, output, runtime or lane.MetalRuntime(), identities
                    )
                    fresh += 1
                raw_path = output / f"{window.start_sample:012d}.raw.json"
                lane._same(raw_path, lane._read(origin / "raw.json"))
                lane._same(
                    output / f"{window.start_sample:012d}.clock.json",
                    lane._clock(window, str(pin(raw_path)["sha256"])),
                )
                durable(receipt_path, _receipt(request, window, output, origin, kind))
            expected = _receipt(request, window, output, origin, kind)
            require(lane._read(receipt_path) == expected, "Successor window changed")
            hashes.append(str(record(expected["raw"])["sha256"]))
            receipts.append(pin(receipt_path))
        lane.verify_inputs(request)
        require(
            _snapshot(request, source, host or SystemHost()) == snapshot,
            "Source changed during run",
        )
        complete = {
            "rawSHA256": hashes,
            "windows": len(hashes),
            "device": "metal",
            "processPolicy": POLICY,
            "runtimeReceipts": receipts,
            "deliveryComplete": False,
        }
        if verify_only:
            require(lane._read(output / "complete.json") == complete, "Completion changed")
        else:
            lane._same(output / "complete.json", complete)
    return output / "complete.json"


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "run", "verify"))
    parser.add_argument("request", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--source", type=Path)
    parser.add_argument("--maximum-new-windows", type=int)
    args = parser.parse_args(argv)
    if args.action == "prepare":
        require(
            args.source is not None and args.maximum_new_windows is None,
            "Prepare requires --source and no window limit",
        )
        prepare(args.request, args.source, args.output)
    else:
        require(args.source is None, "Source is bound by the prepared manifest")
        run(
            args.request,
            args.output,
            verify_only=args.action == "verify",
            maximum_new_windows=args.maximum_new_windows,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
