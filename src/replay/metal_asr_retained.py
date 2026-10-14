"""Publish a new completed-only Metal collection from verified retained windows.

Historical actors are identified by PID and birth, with launch-command hashes for
same-birth observations. A reused PID never authorizes signalling or blocking an
unrelated actor. Original receipts, clocks and inputs remain untouched. No worker
can be launched by this module, and no media-quality approval is granted.
"""

from __future__ import annotations

import argparse
import hashlib
from collections.abc import Iterator, Sequence
from contextlib import ExitStack, contextmanager
from pathlib import Path
from typing import Protocol

from replay import isolated_metal_asr as lane
from replay import metal_asr_resume as retained
from replay.core import ReplayError
from replay.fcp_recovery_runtime import SystemHost as ProcessHost
from replay.measured_job_admission import Parent
from replay.measured_job_admission import SystemHost as IdentityHost
from replay.native_invocation import canonical, durable, pin, require
from replay.recordings import record

POLICY = "retained-verified-completed-children-only; no-worker-dispatch"
QUIESCENCE = "historical-pid-birth-and-launch-command; reused-pid-is-not-historical-actor"


class Host(Protocol):
    def absent(self, pid: int) -> bool: ...
    def identity(self, pid: int) -> Parent: ...


class CurrentHost:
    def absent(self, pid: int) -> bool:
        return ProcessHost().absent(pid)

    def identity(self, pid: int) -> Parent:
        return IdentityHost().identity(pid)


def _actor_gone(host: Host, actor: dict[str, object], *, launch: bool = False) -> bool:
    pid = lane._integer(actor.get("pid"), 1, 2**31 - 1)
    birth = actor.get("birth")
    require(isinstance(birth, str) and bool(birth), "Missing historical actor birth")
    if host.absent(pid):
        return False
    observed = host.identity(pid)
    require(host.identity(pid) == observed, "Current process identity changed during observation")
    if observed.birth != birth:
        return True
    if launch:
        command = actor.get("command")
        if (
            not isinstance(command, list)
            or not command
            or not all(isinstance(x, str) for x in command)
        ):
            raise ReplayError("Missing retained launch command")
        full = tuple(str(x) for x in command)
        worker = full[3:] if full[:2] == ("/usr/bin/sandbox-exec", "-f") else full
        expected = {
            hashlib.sha256((" ".join(args) + "\n").encode()).hexdigest() for args in (full, worker)
        }
        require(observed.command_sha256 in expected, "Same-birth launch command identity changed")
    raise ReplayError("Historical child or descendant still present")


def _quiescent(work: Path, host: Host) -> None:
    retained._local(work)
    actor = lane._read(work / "launch.json")
    reused = _actor_gone(host, actor, launch=True)
    # A PID cannot be allocated to a new actor while its old process-group ID is
    # reserved. Checking the unrelated new group after proven reuse would attach
    # the old receipt to the new actor again.
    if not reused:
        retained._group_gone(lane._integer(actor.get("pid"), 1, 2**31 - 1))
    samples = lane._read(work / "resources.json").get("samples")
    if not isinstance(samples, list):
        raise ReplayError("Missing original family measurements")
    for sample in samples:
        members = record(sample).get("members")
        if not isinstance(members, list):
            raise ReplayError("Missing historical family identities")
        for member in members:
            _actor_gone(host, record(member))


def _origins(request: lane.Request, source: Path) -> dict[str, Path]:
    result: dict[str, Path] = {}
    for window in retained._windows(request):
        origin = retained._published(request, window, source)
        require(origin is not None, "Retained-only collection requires every completed window")
        if origin is not None:
            result[str(window.start_sample)] = origin
    return result


@contextmanager
def _sources(request: lane.Request, source: Path) -> Iterator[dict[str, Path]]:
    with retained._lease(source):
        origins = _origins(request, source)
        with ExitStack() as leases:
            for directory in sorted({p.parent for p in origins.values()} - {source}):
                leases.enter_context(retained._lease(directory))
            require(_origins(request, source) == origins, "Retained origins changed during leasing")
            yield origins


def _snapshot(
    request: lane.Request, source: Path, origins: dict[str, Path], host: Host
) -> dict[str, object]:
    recipe = lane._read(source / "inputs.json")
    require(
        recipe.get("version") == 1
        and recipe.get("samples") == request.samples
        and recipe.get("sampleRate") == lane.RATE
        and recipe.get("windowSamples") == 60 * lane.RATE
        and recipe.get("strideSamples") == 30 * lane.RATE
        and recipe.get("clock") == "input-pcm-not-projected"
        and recipe.get("device") == "metal",
        "Source PCM window contract differs",
    )
    inputs = record(recipe.get("inputs"))
    expected_inputs = {
        str(request.path): request.sha256,
        **{str(p): request.pins[p] for p in (request.pcm, request.weights, request.config)},
    }
    require(
        all(inputs.get(name) == sha for name, sha in expected_inputs.items()),
        "Source inputs differ",
    )
    for name, sha in inputs.items():
        path = canonical(name)
        require(
            path.is_relative_to(Path.cwd().resolve()) and pin(path)["sha256"] == sha,
            "Source input fingerprint changed",
        )
    receipts: list[dict[str, object]] = []
    publications: dict[str, object] = {}
    for window in retained._windows(request):
        origin = origins[str(window.start_sample)]
        _quiescent(origin, host)
        receipts.append(retained._origin(request, window, origin))
        require(
            retained._published(request, window, source) == origin, "Source publication changed"
        )
        for suffix in ("raw", "clock"):
            path = source / f"{window.start_sample:012d}.{suffix}.json"
            publications[str(path)] = pin(path)
        receipt = source / f"{window.start_sample:012d}.receipt.json"
        if receipt.exists():
            publications[str(receipt)] = pin(receipt)
    complete = source / "complete.json"
    if complete.exists():
        publications[str(complete)] = pin(complete)
    return {
        "sourceManifest": pin(source / "inputs.json"),
        "originReceipts": receipts,
        "sourcePublications": publications,
    }


def _manifest(
    request: lane.Request, source: Path, snapshot: dict[str, object]
) -> dict[str, object]:
    recipe = lane._read(source / "inputs.json")
    inputs = record(recipe.get("inputs"))
    inputs[str(source / "inputs.json")] = pin(source / "inputs.json")["sha256"]
    require(len(inputs) <= 16, "Too many retained collection generations")
    return {
        "version": 1,
        "samples": request.samples,
        "sampleRate": lane.RATE,
        "windowSamples": 60 * lane.RATE,
        "strideSamples": 30 * lane.RATE,
        "clock": "input-pcm-not-projected",
        "device": "metal",
        "inputs": inputs,
        "processPolicy": POLICY,
        "quiescencePolicy": QUIESCENCE,
        "sourceCollection": str(source),
        "sourceSnapshot": snapshot,
        "retainedReader": pin(Path(__file__).resolve()),
        "deliveryComplete": False,
    }


def prepare(request_path: Path, source: Path, output: Path, *, host: Host | None = None) -> Path:
    """Prepare an immutable completed-only plan under source and origin leases."""
    request = lane.load_request(request_path)
    source, output = retained._local(source), retained._local(output)
    require(
        not output.exists()
        and not output.is_relative_to(source)
        and not source.is_relative_to(output),
        "A disjoint new namespace is required",
    )
    with _sources(request, source) as origins:
        snapshot = _snapshot(request, source, origins, host or CurrentHost())
        lane.verify_inputs(request)
        output.mkdir(mode=0o700, parents=True)
        with retained._lease(output, create=True):
            durable(output / "inputs.json", _manifest(request, source, snapshot))
    return output / "inputs.json"


def run(
    request_path: Path, output: Path, *, host: Host | None = None, verify_only: bool = False
) -> Path:
    """Publish or verify every retained window; this API cannot start recognition."""
    request = lane.load_request(request_path)
    output = retained._local(output)
    recipe = lane._read(output / "inputs.json")
    source = retained._local(canonical(recipe.get("sourceCollection")))
    current = host or CurrentHost()
    with _sources(request, source) as origins, retained._lease(output):
        snapshot = _snapshot(request, source, origins, current)
        require(recipe == _manifest(request, source, snapshot), "Retained manifest changed")
        hashes: list[str] = []
        receipts: list[dict[str, object]] = []
        for window in retained._windows(request):
            origin = origins[str(window.start_sample)]
            raw = output / f"{window.start_sample:012d}.raw.json"
            clock = output / f"{window.start_sample:012d}.clock.json"
            receipt = output / f"{window.start_sample:012d}.receipt.json"
            if verify_only:
                require(receipt.exists(), "Missing retained window publication")
            else:
                lane._same(raw, lane._read(origin / "raw.json"))
                lane._same(clock, lane._clock(window, str(pin(raw)["sha256"])))
                lane._same(receipt, retained._receipt(request, window, output, origin, "retained"))
            require(
                lane._read(receipt)
                == retained._receipt(request, window, output, origin, "retained"),
                "Retained publication changed",
            )
            hashes.append(str(pin(raw)["sha256"]))
            receipts.append(pin(receipt))
        lane.verify_inputs(request)
        require(
            _snapshot(request, source, origins, current) == snapshot,
            "Source changed during publication",
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
            require(lane._read(output / "complete.json") == complete, "Retained completion changed")
        else:
            lane._same(output / "complete.json", complete)
    return output / "complete.json"


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "run", "verify"))
    parser.add_argument("request", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--source", type=Path)
    args = parser.parse_args(argv)
    if args.action == "prepare":
        require(args.source is not None, "Prepare requires --source")
        prepare(args.request, args.source, args.output)
    else:
        require(args.source is None, "Source is bound by the manifest")
        run(args.request, args.output, verify_only=args.action == "verify")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
