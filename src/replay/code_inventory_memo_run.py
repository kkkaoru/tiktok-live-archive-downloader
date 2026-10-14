"""Own-process memo wrapper for declared future cleanup metadata/check entrypoints.

The explicit inventory-module and observations alias sites are restored on every
exit. No existing process, producer, source file or signed generation is changed.
Future workflow stages must bind this wrapper and code_inventory_memo.py as
inputs, retain the original domain verifier, and run after prior actors release.
"""

import argparse
import cProfile
import sys
import threading
import time
from collections.abc import Iterator
from contextlib import ExitStack, contextmanager
from pathlib import Path

from replay import cleanup_lane_successor as locks
from replay import code_dependency_inventory as inventory
from replay import code_inventory_memo as memo_module
from replay import recovery_source_finalization as recovery
from replay import retained_cleanup_finalization as retained
from replay import subtitle_session_cleanup_finalization as finalization
from replay import subtitle_session_cleanup_wave as wave
from replay import subtitle_session_observations as observations
from replay.sequence_journal import bounded_read, decode, encode, publish
from replay.workflow import parse_stages, stage_graph

FINAL_MODES = frozenset(
    {
        "finalize",
        "verify-finalize",
        "prepare-child",
        "verify-prepare-child",
        "verify-assembly",
        "verify-run-child",
    }
)
ALLOWED = {
    "replay.subtitle_session_cleanup_finalization": FINAL_MODES,
    "replay.recovery_source_finalization": FINAL_MODES,
    "replay.retained_cleanup_finalization": FINAL_MODES,
    "replay.subtitle_session_cleanup_wave": frozenset(
        {"verify-observe", "prepare-child", "verify-prepare-child", "verify-run-child"}
    ),
}
_SCOPE_LOCK = threading.Lock()


def _observation_alias_is(capture: memo_module.Capture) -> bool:
    # Strict mypy does not export this imported alias from the frozen module.
    # The one explicit boundary is an object identity check against a typed
    # callable; no module discovery, callable casts or unchecked invocation.
    value: object = observations.__dict__.get("capture")
    return value is capture


@contextmanager
def capture_scope() -> Iterator[None]:
    """Lease two explicit alias sites only in this single owned Python process."""
    if not _SCOPE_LOCK.acquire(blocking=False):
        raise ValueError("Inventory memo wrapper scope is already active")
    original_capture = inventory.capture
    if not _observation_alias_is(original_capture):
        _SCOPE_LOCK.release()
        raise ValueError("Inventory aliases are not owned ordinary capture sites")
    try:
        cache = memo_module.InventoryMemo(capture_function=original_capture)
    except (OSError, ValueError):
        _SCOPE_LOCK.release()
        raise
    capture_lock = threading.RLock()
    try:
        own_paths = (Path(__file__).resolve(), Path(memo_module.__file__).resolve())
        identities = tuple(memo_module._identity(path) for path in own_paths)
    except (OSError, ValueError):
        cache.close()
        _SCOPE_LOCK.release()
        raise

    def check() -> None:
        if any(
            memo_module._identity(path) != identity
            for path, identity in zip(own_paths, identities, strict=True)
        ):
            raise ValueError("Inventory wrapper implementation changed")
        cache.check()

    def captured(
        modules: tuple[str, ...], *, package: Path = inventory.PACKAGE
    ) -> tuple[inventory.Source, ...]:
        if package != cache.package:
            raise ValueError("Memo wrapper cannot change the source package")
        with capture_lock:
            check()
            result = cache.capture(modules)
            check()
            return result

    try:
        inventory.capture = captured
        observations.__dict__["capture"] = captured
        check()
        yield
        if inventory.capture is not captured or not _observation_alias_is(captured):
            raise ValueError("Inventory alias ownership changed")
        check()
    finally:
        inventory.capture = original_capture
        observations.__dict__["capture"] = original_capture
        cache.close()
        _SCOPE_LOCK.release()


def _work_path(path: Path) -> Path:
    if path != path.resolve() or not path.is_relative_to(Path.cwd().resolve() / "edits/work"):
        raise ValueError("Memo wrapper metadata must remain canonical repository-local work")
    return path


def run(module: str, arguments: list[str], *, leases: tuple[Path, ...] = ()) -> int:
    """Dispatch a typed known metadata/check CLI; producer modes are excluded."""
    if module not in ALLOWED or not arguments or arguments[0] not in ALLOWED[module]:
        raise ValueError("Expected an allowed cleanup metadata/check entrypoint")
    if len(leases) > 8 or len(set(leases)) != len(leases):
        raise ValueError("Expected bounded distinct prior actor leases")
    with ExitStack() as stack:
        for lease in leases:
            _work_path(lease)
            if not lease.is_file():
                raise ValueError("Prior actor lease must already exist")
            stack.enter_context(locks._unlocked(lease))
        stack.enter_context(capture_scope())
        if module == "replay.subtitle_session_cleanup_finalization":
            return finalization.main(arguments)
        if module == "replay.recovery_source_finalization":
            return recovery.main(arguments)
        if module == "replay.retained_cleanup_finalization":
            return retained.main(arguments)
        return wave.main(arguments)


def prepare_check_plan(request: Path, output: Path) -> Path:
    """Declare a single necessary future reuse check; never run the verifier."""
    raw = decode(bounded_read(_work_path(request)))
    if set(raw) != {"version", "module", "arguments", "inputs", "proof", "leases", "reason"}:
        raise ValueError("Expected explicit future check request")
    if type(raw["version"]) is not int or raw["version"] != 1:
        raise ValueError("Unsupported future check version")
    module, reason = raw["module"], raw["reason"]
    arguments, inputs, leases = raw["arguments"], raw["inputs"], raw["leases"]
    if (
        not isinstance(module, str)
        or module not in ALLOWED
        or not isinstance(reason, str)
        or not reason.strip()
        or not isinstance(arguments, list)
        or not arguments
        or len(arguments) > 32
        or any(not isinstance(item, str) or not item for item in arguments)
        or not isinstance(inputs, list)
        or len(inputs) > 240
        or any(not isinstance(item, str) or not item for item in inputs)
        or not isinstance(leases, list)
        or not 1 <= len(leases) <= 8
        or any(not isinstance(item, str) or not item for item in leases)
    ):
        raise ValueError("Expected bounded typed future check fields")
    # record/list validation above does not create unchecked Any values.
    argv = [str(item) for item in arguments]
    if not argv[0].startswith("verify-") or argv[0] not in ALLOWED[module]:
        raise ValueError("Future adoption requires an existing read-only domain verifier")
    lease_paths = tuple(_work_path(Path(str(item))) for item in leases)
    if len(set(lease_paths)) != len(lease_paths):
        raise ValueError("Future check lease duplicated")
    proof = _work_path(Path(str(raw["proof"])))
    _work_path(output)
    if output.exists():
        raise ValueError("Preserve the previous check plan")
    required = [str(_work_path(Path(str(item)))) for item in inputs]
    if str(proof) in required:
        raise ValueError("Existing proof is the adoption output, not its input")
    helpers = [
        str(Path(__file__).resolve()),
        str(Path(memo_module.__file__).resolve()),
        str(Path(inventory.__file__).resolve()),
    ]
    command = [sys.executable, "-B", "-m", "replay.code_inventory_memo_run", "run"]
    for lease in lease_paths:
        command.extend(["--lease", str(lease)])
    command.extend([module, *argv])
    graph = {
        "version": 1,
        "stages": [
            {
                "name": "verify-existing-domain",
                "command": command,
                "verify_command": command,
                "inputs": list(dict.fromkeys([str(request), *required, *helpers])),
                "outputs": [str(proof)],
                "timeout_seconds": 43200,
                "adopt_existing": True,
            }
        ],
    }
    stage_graph(parse_stages(graph))
    output.mkdir(mode=0o700)
    target = output / "workflow.json"
    publish(target, encode(graph))
    publish(
        output / "conditions.json",
        encode(
            {
                "version": 1,
                "reason": reason,
                "priorActorsMustReleaseLeases": [str(path) for path in lease_paths],
                "onlyIfDownstreamReuseRequiresDomainVerification": True,
                "notARequestToDuplicateCurrentParentVerification": True,
                "producerLaunched": False,
                "existingProofMayBeFuture": True,
            }
        ),
    )
    return target


def _snapshot() -> tuple[dict[str, str], dict[str, str], tuple[inventory.Source, ...]]:
    return (
        wave._code(),
        finalization._code(),
        _observation_snapshot(),
    )


def _observation_snapshot() -> tuple[inventory.Source, ...]:
    # The profiled imported alias must exactly match the typed module function
    # before dispatch. Calling that identical object avoids unchecked exports.
    if not _observation_alias_is(inventory.capture):
        raise ValueError("Inventory observation alias changed")
    return inventory.capture(("subtitle_session_observations",))


def benchmark(output: Path, *, iterations: int = 10) -> Path:
    """Bounded source-only benchmark of the three actual profiled call sites."""
    root = Path.cwd().resolve()
    if (
        output != output.resolve()
        or not output.is_relative_to(root / "edits/work")
        or output.exists()
        or not 1 <= iterations <= 20
    ):
        raise ValueError("Expected a fresh repository-local bounded benchmark output")
    output.mkdir(mode=0o700, parents=True)
    expected = _snapshot()
    records: list[dict[str, object]] = []
    for strategy in ("ordinary", "memo"):
        profile = cProfile.Profile()
        began = time.perf_counter()
        profile.enable()
        if strategy == "memo":
            with capture_scope():
                for _ in range(iterations):
                    if _snapshot() != expected:
                        raise ValueError("Memo inventory differs from ordinary capture")
        else:
            for _ in range(iterations):
                if _snapshot() != expected:
                    raise ValueError("Ordinary source inventory changed")
        profile.disable()
        seconds = time.perf_counter() - began
        profile.dump_stats(str(output / f"{strategy}.pstats"))
        records.append(
            {
                "strategy": strategy,
                "iterations": iterations,
                "captureCalls": iterations * 3,
                "seconds": seconds,
                "exactInventories": True,
                "profilerOverheadIncluded": True,
            }
        )
    target = output / "comparison.json"
    publish(
        target,
        encode(
            {
                "version": 1,
                "results": records,
                "liveProcessesModified": False,
                "producerLaunched": False,
            }
        ),
    )
    return target


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="operation", required=True)
    execute = subparsers.add_parser("run")
    execute.add_argument("module", choices=tuple(ALLOWED))
    execute.add_argument("--lease", action="append", type=Path, default=[])
    execute.add_argument("arguments", nargs=argparse.REMAINDER)
    profile = subparsers.add_parser("benchmark")
    profile.add_argument("--output", type=Path, required=True)
    profile.add_argument("--iterations", type=int, default=10)
    plan = subparsers.add_parser("prepare-check-plan")
    plan.add_argument("request", type=Path)
    plan.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        if args.operation == "benchmark":
            print(benchmark(args.output, iterations=args.iterations))
            return 0
        if args.operation == "prepare-check-plan":
            print(prepare_check_plan(args.request, args.output))
            return 0
        return run(args.module, args.arguments, leases=tuple(args.lease))
    except (OSError, ValueError) as error:
        print(f"INVENTORY_MEMO_RUN_ERROR {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
