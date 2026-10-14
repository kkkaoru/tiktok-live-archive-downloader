"""One admitted native part through the existing verified publication workflow.

This finite adapter does not delete staging, submit Compressor work or release
successors. Actual observations and checkpoints are immutable; an uncertain
native intent is never replayed. The current inventory is only an owned index
of an independently retained, accepted admission snapshot.
"""

import argparse
import json
import os
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

from replay import fcp_native_admission as admission
from replay import fcp_native_queue as queue
from replay import fcp_native_queue_followup as followup
from replay import workflow
from replay.artifact_json import _path
from replay.core import ReplayError
from replay.fcp_native_queue_followup import Ref
from replay.foreground_evidence import _bound_bytes
from replay.native_export_receipt import native_payload, save_new
from replay.phase_timer import run_timed
from replay.recordings import record
from replay.stages import execution_lock, file_digest
from replay.whole_technical_batch import reader_code_paths

ROOT = Path(__file__).resolve().parents[2]
WORK = ROOT / "edits" / "work"
MAX_BYTES = 8 * 1024 * 1024
FOLLOWUP_KEYS = frozenset(
    {
        "assembly",
        "publishedAssembly",
        "sourceCleanup",
        "ffmpeg",
        "ffprobe",
        "audioSampleRate",
        "audioChannels",
        "receiptOutput",
    }
)
REFRESH_PHASES = ("background-open", "background-wait", "background", "background-close")


@dataclass(frozen=True)
class Pump:
    config: Path
    native: queue.Queue
    root: Path
    template: dict[str, object]
    followup: dict[str, object]
    runtime: dict[str, object]
    inputs: tuple[Ref, ...]


@dataclass(frozen=True)
class Result:
    state: str
    assembly_part: int
    evidence: Path | None
    quality_approved: bool = False


def _unique(rows: list[tuple[str, object]]) -> dict[str, object]:
    raw: dict[str, object] = {}
    for key, value in rows:
        if key in raw:
            raise ReplayError("Duplicate pump JSON key")
        raw[key] = value
    return raw


def _work(value: object) -> Path:
    path = _path(value)
    if not path.is_relative_to(WORK) or path == WORK:
        raise ReplayError("Pump artifacts must remain in repository work storage")
    return path


def _ref(path: Path) -> dict[str, str]:
    return {"path": str(path), "sha256": file_digest(path)}


def _read(path: Path) -> dict[str, object]:
    return record(
        json.loads(_bound_bytes(path, file_digest(path), MAX_BYTES), object_pairs_hook=_unique)
    )


def _schema(raw: dict[str, object], kind: str, keys: set[str]) -> None:
    if raw.get("version") != 1 or raw.get("kind") != kind or set(raw) != keys | {"version", "kind"}:
        raise ReplayError("Unsupported or incomplete native delivery pump schema")


def _verify(pump: Pump) -> None:
    if any(file_digest(ref.path) != ref.sha256 for ref in pump.inputs):
        raise ReplayError("Pump input or reader bytes changed; preserve existing effects")


def load(path: Path) -> Pump:
    """Validate static, pinned configuration without observing or operating apps."""
    path = _work(str(path))
    raw = _read(path)
    _schema(
        raw,
        "native-delivery-pump",
        {"queue", "admissionTemplate", "followup", "runtimeEvidence", "outputRoot"},
    )
    inputs = followup.Inputs()
    inputs.current(path)
    native = queue.load(inputs.add(raw["queue"]).path)
    root = _work(raw["outputRoot"])
    if not root.is_dir() or native.inventory != root / "current-inventory.json":
        raise ReplayError("Pump requires an existing root and its own current inventory index")
    if any(ref == native.inventory for ref, _ in native.pins):
        raise ReplayError("Mutable inventory index cannot be a historical queue pin")
    template = inputs.read(inputs.add(raw["admissionTemplate"]))
    admission._preparation(template)
    target = {"pid": native.target.pid, "birth": native.target.birth}
    requested = record(template.get("requested"))
    policy = record(template.get("policy"))
    if (
        template.get("target") != target
        or requested.get("path") != str(native.parts[0].movie)
        or requested.get("peakAdditionalBytes") != native.peak_bytes
        or requested.get("diskAllocation")
        != record(_read(native.config).get("limits")).get("diskAllocation")
        or policy.get("maximumActiveRenders") != 1
    ):
        raise ReplayError("Admission must bind the exact queue allocation and one native writer")
    inputs.add(template["registry"])
    fixed = record(raw["followup"])
    if not FOLLOWUP_KEYS <= set(fixed) <= FOLLOWUP_KEYS | {"approvedOriginal"}:
        raise ReplayError("Unsupported followup template; no degraded fallback")
    for name in ("assembly", "publishedAssembly", "sourceCleanup", "ffmpeg", "ffprobe"):
        inputs.add(fixed[name])
    if "approvedOriginal" in fixed:
        inputs.add(fixed["approvedOriginal"])
    for name in ("audioSampleRate", "audioChannels"):
        value = fixed[name]
        if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
            raise ReplayError("Expected explicit positive audio sample clock and channels")
    _work(fixed["receiptOutput"])
    runtime = inputs.read(inputs.add(raw["runtimeEvidence"]))
    _schema(
        runtime,
        "native-delivery-runtime-evidence",
        {"executor", "provider", "target", "capabilities", "scratch"},
    )
    if (
        runtime["executor"] != _ref(native.executable)
        or runtime["provider"] != native.provider
        or runtime["target"] != target
    ):
        raise ReplayError("Runtime evidence belongs to another Executor or app identity")
    native_payload(inputs.read(inputs.add(runtime["capabilities"])))
    scratch = record(runtime["scratch"])
    if set(scratch) != {"roots", "observedPaths"}:
        raise ReplayError("Actual child scratch observations are required")
    roots, observations = scratch["roots"], scratch["observedPaths"]
    if (
        not isinstance(roots, list)
        or not 1 <= len(roots) <= 16
        or not isinstance(observations, list)
        or not 1 <= len(observations) <= 256
    ):
        raise ReplayError("Expected bounded observed scratch paths and roots")
    checked_roots = tuple(_work(value) for value in roots)
    if any(not item.is_dir() for item in checked_roots) or any(
        not any(_work(value).is_relative_to(item) for item in checked_roots)
        for value in observations
    ):
        raise ReplayError("Observed Executor scratch escaped the declared repository roots")
    for code in (
        *reader_code_paths(),
        Path(__file__).resolve(),
        Path(queue.__file__).resolve(),
        Path(admission.__file__).resolve(),
        Path(followup.__file__).resolve(),
        Path(workflow.__file__).resolve(),
        Path(run_timed.__code__.co_filename),
    ):
        inputs.current(code)
    for reference, digest in native.pins:
        inputs.add({"path": str(reference), "sha256": digest})
    inputs.verify()
    return Pump(
        path,
        native,
        root,
        template,
        fixed,
        runtime,
        tuple(followup.Ref(item, digest) for item, digest in inputs.refs.items()),
    )


def _scratch(pump: Pump) -> None:
    raw_roots = record(pump.runtime["scratch"])["roots"]
    if not isinstance(raw_roots, list):
        raise ReplayError("Expected explicit scratch roots")
    roots = tuple(_work(value) for value in raw_roots)
    values = tuple(os.environ.get(name) for name in ("TMPDIR", "TMP", "TEMP", "XDG_CACHE_HOME"))
    if any(
        value is None
        or not _work(value).is_dir()
        or not any(_work(value).is_relative_to(root) for root in roots)
        for value in values
    ):
        raise ReplayError("Executor child temporary and cache paths are not configured locally")
    if Path(tempfile.gettempdir()).resolve() != _work(values[0]):
        raise ReplayError("Actual Python temporary directory differs from checked child scratch")


def _lease(pump: Pump, host: admission.Host) -> tuple[Path, str]:
    path = queue.GUI_LEASE_ROOT / ".fcp-native-queue-owner.json"
    value = _read(path)
    if (
        value.get("pid") != os.getpid()
        or value.get("birth") != host.birth(os.getpid())
        or value.get("queueSHA256") != pump.native.digest
        or host.birth(pump.native.target.pid) != pump.native.target.birth
    ):
        raise ReplayError("Exact pump process does not own the native GUI lease")
    return path, file_digest(path)


def _index(pump: Pump, snapshot: Path, expected: str | None, *, host: admission.Host) -> None:
    lease, lease_sha = _lease(pump, host)
    path = pump.native.inventory
    actual = file_digest(path) if path.exists() else None
    if path.is_symlink() or actual != expected:
        raise ReplayError("Owned inventory index changed concurrently")
    raw = _read(snapshot)
    if (
        raw.get("admitted") is not True
        or raw.get("registry") != pump.template["registry"]
        or raw.get("target") != pump.template["target"]
    ):
        raise ReplayError("Only the exact accepted real snapshot can update current inventory")
    with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as stream:
        temporary = Path(stream.name)
        try:
            stream.write(_bound_bytes(snapshot, file_digest(snapshot), MAX_BYTES))
            stream.flush()
            os.fsync(stream.fileno())
            os.chmod(stream.name, 0o600)
            if (
                file_digest(lease) != lease_sha
                or (file_digest(path) if path.exists() else None) != expected
            ):
                raise ReplayError("GUI lease or inventory changed before atomic index update")
            os.replace(stream.name, path)
        finally:
            temporary.unlink(missing_ok=True)
    save_new(
        snapshot.with_name("index-binding.json"),
        {"snapshot": _ref(snapshot), "index": _ref(path), "previousSHA256": expected},
    )


def _refresh(pump: Pump, transport: queue.Transport, host: admission.Host, name: str) -> None:
    _verify(pump)
    lease, lease_sha = _lease(pump, host)
    root = pump.root / name
    root.mkdir(mode=0o700)
    current = file_digest(pump.native.inventory) if pump.native.inventory.exists() else None
    observation: Path | None = None
    for phase in REFRESH_PHASES:
        if _lease(pump, host) != (lease, lease_sha):
            raise ReplayError("GUI ownership bytes changed during admission refresh")
        operation = next(item for item in pump.native.parts[0].operations if item.phase == phase)
        arguments = queue._arguments(pump.native, pump.native.parts[0], operation)
        save_new(
            root / f"{phase}-intent.json",
            {"tool": queue.TOOLS[phase], "arguments": arguments, "ownerPID": os.getpid()},
        )
        captured = transport.call(
            f"{pump.native.provider}.{queue.TOOLS[phase]}", arguments, pump.native.timeout
        )
        save_new(root / f"{phase}-response.json", captured)
        if _lease(pump, host) != (lease, lease_sha):
            raise ReplayError("GUI ownership bytes changed during actual native observation")
        envelope = queue._envelope(captured)
        observed = root / f"{phase}-native.json"
        save_new(observed, envelope)
        payload = native_payload(envelope)
        if phase == "background":
            if payload.get("pid") != pump.native.target.pid or not queue._background_idle(payload):
                raise ReplayError("Fresh native background is busy or belongs to another app")
            observation = observed
        else:
            queue._accept(phase, payload, pump.native, pump.native.parts[0], host.now_ns())
    if observation is None:
        raise ReplayError("Actual native admission observation is missing")
    request = {
        **pump.template,
        "kind": "native-admission-request",
        "nativeObservation": _ref(observation),
    }
    request_path, snapshot = root / "request.json", root / "snapshot.json"
    save_new(request_path, request)
    result = admission.capture(request_path, snapshot, host=host)
    if not result.admitted:
        raise ReplayError("Measured native allocation denied: " + ", ".join(result.reasons))
    _index(pump, snapshot, current, host=host)
    _verify(pump)


@dataclass(frozen=True)
class RefreshTransport:
    pump: Pump
    delegate: queue.Transport
    host: admission.Host

    def call(self, tool: str, arguments: dict[str, object], timeout: int) -> dict[str, object]:
        captured = self.delegate.call(tool, arguments, timeout)
        if tool == f"{self.pump.native.provider}.{queue.TOOLS['project']}":
            save_new(self.pump.root / "project-trigger-response.json", captured)
            save_new(self.pump.root / "project-trigger-native.json", queue._envelope(captured))
            payload = queue._payload(captured)
            queue._accept(
                "project", payload, self.pump.native, self.pump.native.parts[0], self.host.now_ns()
            )
            _refresh(self.pump, self.delegate, self.host, "refresh-project")
        return captured


def _checkpoint(pump: Pump, name: str, paths: tuple[Path, ...]) -> Path:
    path = pump.root / name
    save_new(
        path,
        {
            "version": 1,
            "kind": "native-delivery-pump-checkpoint",
            "config": _ref(pump.config),
            "inputs": [_ref(ref.path) for ref in pump.inputs],
            "outputs": [_ref(item) for item in paths],
            "qualityApproved": False,
        },
    )
    return path


def _completed(pump: Pump, name: str) -> dict[str, object] | None:
    path = pump.root / name
    if not path.exists():
        return None
    raw = _read(path)
    if (
        raw.get("version") != 1
        or raw.get("kind") != "native-delivery-pump-checkpoint"
        or raw.get("config") != _ref(pump.config)
        or raw.get("inputs") != [_ref(ref.path) for ref in pump.inputs]
        or raw.get("qualityApproved") is not False
    ):
        raise ReplayError("Completed pump checkpoint identity changed")
    outputs = raw.get("outputs")
    if not isinstance(outputs, list) or not 1 <= len(outputs) <= 32:
        raise ReplayError("Pump checkpoint output inventory is invalid")
    for item in outputs:
        ref = record(item)
        if set(ref) != {"path", "sha256"} or _ref(_work(ref.get("path"))) != ref:
            raise ReplayError("Completed pump artifact changed; do not replay effects")
    return raw


def _native(pump: Pump, transport: queue.Transport, host: admission.Host) -> Path:
    ready = pump.native.root / "followup-ready.json"
    if _completed(pump, "native-complete.json") is not None:
        return ready
    if any(
        path.exists() or path.is_symlink()
        for path in (
            pump.root / "native-started.json",
            pump.native.root / "owner.json",
            ready,
            pump.native.parts[0].movie,
            pump.native.parts[0].movie.with_name(
                f".{pump.native.parts[0].movie.name}.native-queue-reservation.json"
            ),
            pump.root / "refresh-initial",
            pump.root / "refresh-project",
            pump.root / "followup-request.json",
            pump.root / "followup",
            pump.root / "project-trigger-response.json",
            _work(pump.followup["receiptOutput"]),
        )
    ):
        raise ReplayError("Native effect is occupied or uncertain; reconcile without resubmission")
    save_new(pump.root / "native-started.json", {"config": _ref(pump.config), "retrySafe": False})
    lease = queue.GUI_LEASE_ROOT / ".fcp-native-queue-owner.json"
    save_new(
        lease,
        {"pid": os.getpid(), "birth": host.birth(os.getpid()), "queueSHA256": pump.native.digest},
    )
    digest = file_digest(lease)
    _refresh(pump, transport, host, "refresh-initial")
    if file_digest(lease) != digest:
        raise ReplayError("Initial GUI handoff ownership changed")
    lease.unlink()
    result = queue.run(pump.native, RefreshTransport(pump, transport, host), host)
    if result.state != "followup-ready":
        raise ReplayError("Native queue is uncertain or bounded; preserve all effects")
    _checkpoint(pump, "native-complete.json", (ready,))
    return ready


def _reader_capacity(pump: Pump, host: admission.Host) -> None:
    """Count explicit other reader owners; do not infer global freedom from ours."""
    _verify(pump)
    registry_ref = record(pump.template["registry"])
    registry = admission._registry(_read(_path(registry_ref["path"])))
    completed_evidence = admission.Evidence()
    admission._completed_owners(completed_evidence, registry)
    admission._quiescent_owners(completed_evidence, registry)
    readers = tuple(
        owner
        for owner in registry.owners
        if owner.role in {"reader", "followup"}
        and owner.completion is None
        and owner.quiescence is None
        and owner.identity.pid != os.getpid()
    )
    if any(host.birth(owner.identity.pid) != owner.identity.birth for owner in readers):
        raise ReplayError("Declared read worker identity changed")
    policy = record(pump.template["policy"])
    limit = policy["maximumPressure"]
    if not isinstance(limit, int) or host.pressure() > limit or len(readers) >= 2:
        raise ReplayError("Measured pressure or declared global read capacity denies followup")
    if not queue.disk_capacity(
        host,
        pump.native.boot,
        pump.native.parts[0].movie,
        pump.native.peak_bytes,
        pump.native.floor_bytes,
        pump.native.disk_allocation,
        tuple((item.path.parent, item.remaining_bytes) for item in registry.outputs)
        if pump.native.disk_allocation is not None
        else (),
    ):
        raise ReplayError("Measured followup disk capacity is insufficient")
    admission._completed_owners(completed_evidence, registry)
    admission._quiescent_owners(completed_evidence, registry)
    completed_evidence.verify()
    _verify(pump)


def run(pump: Pump, *, transport: queue.Transport, host: admission.Host) -> Result:
    """Perform one finite part; existing workflow owns all media verification."""
    _scratch(pump)
    with execution_lock(pump.root, ".pump") as acquired:
        if not acquired:
            raise ReplayError("Another process owns this delivery pump")
        _verify(pump)
        _completed(pump, "followup-complete.json")
        ready = _native(pump, transport, host)
        _reader_capacity(pump, host)
        request_path = pump.root / "followup-request.json"
        prepared = pump.root / "followup"
        graph = prepared / "prepared" / "workflow.json"
        completed = _completed(pump, "preparation-complete.json")
        if completed is None:
            if request_path.exists() or prepared.exists():
                raise ReplayError(
                    "Partial preparation is uncheckpointed; preserve it and reconcile"
                )
            observed = _read(ready)
            request = {
                "version": 1,
                "kind": "native-queue-followup-request",
                "queue": _ref(pump.native.config),
                "ready": _ref(ready),
                "queueOrdinal": 1,
                "assemblyPart": pump.native.parts[0].assembly_part,
                **pump.followup,
                **{
                    name: observed[name]
                    for name in ("exportRequest", "exportResponse", "background")
                },
            }
            save_new(request_path, request)
            graph = followup.prepare(request_path, prepared)
            _checkpoint(
                pump,
                "preparation-complete.json",
                (
                    request_path,
                    prepared / "preparation.json",
                    graph,
                    graph.parent / "operation.json",
                ),
            )
        state = pump.root / "followup-state"
        if workflow.main([str(graph), "--state", str(state), "--check"]) != 0:
            raise ReplayError("Existing followup dependency preflight failed")
        code = run_timed(
            pump.root / "phases.jsonl",
            "native-followup",
            [
                str(Path(sys.executable).absolute()),
                "-m",
                "replay.workflow",
                str(graph),
                "--state",
                str(state),
            ],
        )
        if code != 0:
            raise ReplayError("Existing followup stopped; resume its exact original state")
        _verify(pump)
        terminal = graph.parent / "verified-followup.json"
        if _completed(pump, "followup-complete.json") is None:
            _checkpoint(
                pump, "followup-complete.json", (terminal, _work(pump.followup["receiptOutput"]))
            )
        return Result("published-candidate", pump.native.parts[0].assembly_part, terminal)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("operation", choices=("check", "run"))
    parser.add_argument("config", type=Path)
    args = parser.parse_args(argv)
    try:
        pump = load(args.config)
        if args.operation == "check":
            print(
                json.dumps(
                    {
                        "assemblyPart": pump.native.parts[0].assembly_part,
                        "canDispatch": False,
                        "readWorkers": 1,
                        "nativeWriters": 1,
                    }
                )
            )
            return 0
        result = run(
            pump,
            transport=queue.ExecutorTransport(pump.native.executable),
            host=admission.SystemHost(),
        )
        print(
            json.dumps(
                {
                    "state": result.state,
                    "assemblyPart": result.assembly_part,
                    "qualityApproved": False,
                }
            )
        )
        return 0
    except (ReplayError, OSError, ValueError) as error:
        print(f"Native delivery pump stopped: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
