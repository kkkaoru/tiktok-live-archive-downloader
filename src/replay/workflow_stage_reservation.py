"""Reserve a future cooperative stage lease and attest a finite workflow handoff.

The existing controller finishes its current stage and verifier, then reports the
reserved stage as busy. No producer, process, receipt or lock file is removed or
signalled. A proof describes that boundary observation, not permission to restart
the old graph or approval of the media. Consumers must recheck its pinned inputs
and absent transfer outputs before starting a disjoint graph.
"""

from __future__ import annotations

import argparse
import hashlib
import heapq
import json
import os
import re
import stat
import subprocess
import sys
import time
from collections.abc import Sequence
from contextlib import ExitStack
from dataclasses import asdict, dataclass
from datetime import datetime
from graphlib import TopologicalSorter
from pathlib import Path

from replay import stages as stage_runner
from replay import workflow as workflow_runner
from replay.core import ReplayError
from replay.recordings import record
from replay.stages import Stage, StageError, execution_lock
from replay.workflow import MAX_STAGES, load_stages, stage_graph

MAX_MANIFEST_BYTES = 1_048_576
MAX_LOG_BYTES = 16 * 1_048_576
MAX_LSOF_BYTES = 65_536
MAX_PROCESS_BYTES = 1_048_576
MAX_PROCESS_ROWS = 8192
MAX_WAIT_SECONDS = 3600
POLL_SECONDS = 0.2
OBSERVATION_SECONDS = 2.0
BLOCK_BYTES = 1_048_576
SHA256 = re.compile(r"[a-f0-9]{64}")
JOB_NAME = re.compile(r"[a-z0-9][a-z0-9._-]{0,63}")
REQUEST_FIELDS = frozenset(
    {
        "version",
        "workflow",
        "state",
        "job",
        "current",
        "boundary",
        "transferStages",
        "timeoutSeconds",
    }
)
JOB_FIELDS = frozenset({"root", "name", "pid", "ppid", "uid", "start", "executable"})


@dataclass(frozen=True)
class FileIdentity:
    device: int
    inode: int
    size: int
    modified_ns: int


@dataclass(frozen=True)
class NamespaceIdentity:
    device: int
    inode: int


@dataclass(frozen=True)
class ProcessIdentity:
    """Exact OS metadata; executable is the opaque comm field, not a resolved path."""

    pid: int
    ppid: int
    uid: int
    start: str
    executable: str

    @property
    def identity(self) -> tuple[int, int, str, str]:
        return self.pid, self.uid, self.start, self.executable


@dataclass(frozen=True)
class Request:
    path: Path
    sha256: str
    workflow: Path
    workflow_sha256: str
    state: Path
    job_root: Path
    job_name: str
    owner: ProcessIdentity
    current: Stage
    current_fingerprint: str
    transfer: tuple[Stage, ...]
    timeout_seconds: int

    @property
    def boundary(self) -> str:
        return self.transfer[0].name

    @property
    def log(self) -> Path:
        return self.job_root / f"{self.job_name}.log"

    @property
    def status(self) -> Path:
        return self.job_root / f"{self.job_name}.status"


def _repository() -> Path:
    root = Path(__file__).resolve().parents[2]
    if Path.cwd().resolve() != root:
        raise ReplayError("Reservation must run from the repository root")
    return root


def _path(value: object, *, local: bool = True) -> Path:
    if not isinstance(value, str) or not value or any(c in value for c in "\0\n\r"):
        raise ReplayError("Expected a canonical absolute reservation path")
    path = Path(value)
    if not path.is_absolute() or path != path.resolve():
        raise ReplayError("Reservation paths must not traverse links or parent segments")
    if local and not path.is_relative_to(_repository() / "edits" / "work"):
        raise ReplayError("Reservation paths must stay inside repository work storage")
    return path


def _identity(path: Path, *, empty: bool = False) -> FileIdentity:
    _path(str(path), local=False)
    info = path.lstat()
    if (
        not stat.S_ISREG(info.st_mode)
        or info.st_nlink != 1
        or info.st_uid != os.getuid()
        or (not empty and info.st_size == 0)
    ):
        raise ReplayError("Reservation evidence must be an owned single-link regular file")
    return FileIdentity(info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns)


def _namespace(path: Path) -> NamespaceIdentity:
    _path(str(path))
    info = path.lstat()
    if not stat.S_ISDIR(info.st_mode) or info.st_uid != os.getuid():
        raise ReplayError("Reservation namespaces must remain owned canonical directories")
    return NamespaceIdentity(info.st_dev, info.st_ino)


def _deadline(deadline: float) -> None:
    if time.monotonic() >= deadline:
        raise ReplayError("Reservation deadline reached; no handoff was verified")


def _digest(path: Path, deadline: float) -> str:
    _path(str(path), local=False)
    before = _identity(path)
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        opened = os.fstat(stream.fileno())
        if (opened.st_dev, opened.st_ino) != (before.device, before.inode):
            raise ReplayError("Reservation artifact changed before hashing")
        while block := stream.read(BLOCK_BYTES):
            _deadline(deadline)
            digest.update(block)
    if _identity(path) != before:
        raise ReplayError("Reservation artifact changed during hashing")
    return digest.hexdigest()


def _json(path: Path) -> dict[str, object]:
    before = _identity(path)
    if before.size > MAX_MANIFEST_BYTES:
        raise ReplayError("Reservation JSON exceeds its byte budget")
    with path.open("rb") as stream:
        data = stream.read(MAX_MANIFEST_BYTES + 1)
    if len(data) > MAX_MANIFEST_BYTES or _identity(path) != before:
        raise ReplayError("Reservation JSON changed or exceeds its byte budget")
    return record(json.loads(data.decode("utf-8")))


def _sha(value: object) -> str:
    if not isinstance(value, str) or SHA256.fullmatch(value) is None:
        raise ReplayError("Expected a reservation SHA-256 fingerprint")
    return value


def _text(value: object) -> str:
    if not isinstance(value, str) or not value or any(c in value for c in "\0\n\r"):
        raise ReplayError("Expected bounded reservation text")
    if len(value) > 4096:
        raise ReplayError("Reservation text exceeds its budget")
    return value


def _integer(value: object, *, minimum: int) -> int:
    if type(value) is not int or value < minimum:
        raise ReplayError("Invalid reservation process or deadline integer")
    return value


def _ordered(stages: tuple[Stage, ...]) -> tuple[Stage, ...]:
    sorter = TopologicalSorter(stage_graph(stages))
    sorter.prepare()
    priority = {stage.name: index for index, stage in enumerate(stages)}
    ready: list[int] = []
    ordered: list[Stage] = []
    while sorter.is_active():
        for name in sorter.get_ready():
            heapq.heappush(ready, priority[name])
        stage = stages[heapq.heappop(ready)]
        ordered.append(stage)
        sorter.done(stage.name)
    return tuple(ordered)


def _transfer(raw: dict[str, object], stages: tuple[Stage, ...]) -> tuple[Stage, tuple[Stage, ...]]:
    current = record(raw["current"])
    if set(current) != {"stage", "fingerprint"}:
        raise ReplayError("Invalid current-stage reservation fields")
    names = raw["transferStages"]
    if (
        not isinstance(names, list)
        or not 1 <= len(names) <= MAX_STAGES
        or any(not isinstance(name, str) for name in names)
    ):
        raise ReplayError("Expected explicit bounded transfer stage names")
    ordered = _ordered(stages)
    boundary = _text(raw["boundary"])
    positions = {stage.name: index for index, stage in enumerate(ordered)}
    if boundary not in positions or positions[boundary] == 0:
        raise ReplayError("Reservation boundary must have a preceding stage")
    index = positions[boundary]
    preceding, transfer = ordered[index - 1], ordered[index:]
    if current["stage"] != preceding.name or names != [stage.name for stage in transfer]:
        raise ReplayError("Reservation must transfer the complete ordered future suffix")
    if not preceding.verify_command:
        raise ReplayError("The preceding stage requires an explicit domain verifier")
    if preceding.name not in stage_graph(stages)[boundary]:
        raise ReplayError("The boundary must depend on the preceding stage output")
    return preceding, transfer


def load_request(path: Path) -> Request:
    """Validate a pinned local request without taking leases or changing existing state."""
    manifest = _path(str(path))
    raw = _json(manifest)
    if set(raw) != REQUEST_FIELDS or type(raw["version"]) is not int or raw["version"] != 1:
        raise ReplayError("Invalid version-one reservation request fields")
    timeout = _integer(raw["timeoutSeconds"], minimum=1)
    if timeout > MAX_WAIT_SECONDS:
        raise ReplayError("Reservation deadline exceeds its finite budget")
    deadline = time.monotonic() + timeout
    pin = record(raw["workflow"])
    if set(pin) != {"path", "sha256"}:
        raise ReplayError("Invalid workflow reservation pin")
    workflow = _path(pin["path"])
    workflow_sha = _sha(pin["sha256"])
    if _digest(workflow, deadline) != workflow_sha:
        raise ReplayError("Reservation workflow pin changed")
    state = _path(raw["state"])
    if not state.is_dir() or workflow.is_relative_to(state) or manifest.is_relative_to(state):
        raise ReplayError("Existing workflow state must be separate from manifests")
    current, transfer = _transfer(raw, load_stages(workflow))
    outputs = tuple(path for stage in (current, *transfer) for path in stage.outputs)
    if any(_path(str(output)).is_relative_to(state) for output in outputs):
        raise ReplayError("Stage output namespaces must stay outside checkpoint state")
    job = record(raw["job"])
    if set(job) != JOB_FIELDS:
        raise ReplayError("Invalid expected reservation job identity fields")
    job_name = _text(job["name"])
    job_root = _path(job["root"])
    if JOB_NAME.fullmatch(job_name) is None or not job_root.is_dir():
        raise ReplayError("Invalid existing repository job namespace")
    launch = _path(str(job_root / f"{job_name}.launch"))
    info = launch.stat()
    if not stat.S_ISDIR(info.st_mode) or info.st_uid != os.getuid():
        raise ReplayError("Expected an owned retained job launch reservation")
    owner = ProcessIdentity(
        _integer(job["pid"], minimum=1),
        _integer(job["ppid"], minimum=0),
        _integer(job["uid"], minimum=0),
        _text(job["start"]),
        _text(job["executable"]),
    )
    if owner.uid != os.getuid():
        raise ReplayError("Expected an owned exact controller executable identity")
    return Request(
        manifest,
        _digest(manifest, deadline),
        workflow,
        workflow_sha,
        state,
        job_root,
        job_name,
        owner,
        current,
        _sha(record(raw["current"])["fingerprint"]),
        transfer,
        timeout,
    )


def _lsof(path: Path, *, deadline: float, pid: int | None = None) -> str:
    _deadline(deadline)
    selector = [] if pid is None else ["-a", "-p", str(pid)]
    result = subprocess.run(
        ["/usr/sbin/lsof", "-n", "-P", *selector, "-Fpln", "--", str(path)],
        stdin=subprocess.DEVNULL,
        capture_output=True,
        text=True,
        check=False,
        timeout=min(OBSERVATION_SECONDS, deadline - time.monotonic()),
        env={**os.environ, "LC_ALL": "C"},
    )
    if result.returncode != 0 or result.stderr.strip() or len(result.stdout) > MAX_LSOF_BYTES:
        raise ReplayError("Scoped lock-owner observation failed")
    return result.stdout


def _parse_processes(body: str) -> list[ProcessIdentity]:
    lines = body.splitlines()
    if not lines or len(lines) > MAX_PROCESS_ROWS or len(body.encode("utf-8")) > MAX_PROCESS_BYTES:
        raise ReplayError("Reservation process metadata exceeds its finite budget or is empty")
    rows: list[ProcessIdentity] = []
    seen: set[int] = set()
    for line in lines:
        fields = line.split(maxsplit=8)
        if (
            len(fields) != 9
            or any(re.fullmatch(r"[0-9]+", value) is None for value in fields[:2])
            or re.fullmatch(r"-?[0-9]+", fields[2]) is None
        ):
            raise ReplayError("Malformed bounded reservation process metadata")
        pid, ppid, uid = map(int, fields[:3])
        start = " ".join(fields[3:8])
        datetime.strptime(start, "%a %b %d %H:%M:%S %Y")
        if pid <= 0 or pid in seen or len(fields[8]) > 4096:
            raise ReplayError("Invalid bounded reservation process identity")
        seen.add(pid)
        rows.append(ProcessIdentity(pid, ppid, uid, start, fields[8]))
    return rows


def _snapshot(directory: Path, *, deadline: float) -> list[ProcessIdentity]:
    _namespace(directory)
    _deadline(deadline)
    result = subprocess.run(
        ["/bin/ps", "-ww", "-ax", "-o", "pid=,ppid=,uid=,lstart=,comm="],
        stdin=subprocess.DEVNULL,
        capture_output=True,
        text=True,
        check=False,
        timeout=min(OBSERVATION_SECONDS, deadline - time.monotonic()),
        env={**os.environ, "LC_ALL": "C", "TZ": "UTC"},
    )
    if (
        result.returncode != 0
        or result.stderr.strip()
        or not result.stdout.strip()
        or len(result.stdout.encode("utf-8")) > MAX_PROCESS_BYTES
    ):
        raise ReplayError("Bounded reservation process snapshot failed")
    return _parse_processes(result.stdout)


def _lock_owner(path: Path, owner: int, deadline: float) -> None:
    """Bind a busy kernel lease to its sole observed opener and unchanged inode.

    macOS lsof reports a blank lock field even for an exclusive flock. A blank
    field is never lock evidence: an independent nonblocking kernel probe must
    refuse acquisition, with the expected sole opener observed before and after.
    These finite observations cannot prevent an uncooperative process racing them.
    """
    before = _identity(path, empty=True)
    _sole_opener(path, owner, deadline)
    with execution_lock(path.parent, path.name.removesuffix(".lock")) as acquired:
        if acquired:
            raise ReplayError("The exact controller does not exclusively own the expected lease")
    _sole_opener(path, owner, deadline)
    if _identity(path, empty=True) != before:
        raise ReplayError("The observed controller lease inode changed")


def _sole_opener(path: Path, owner: int, deadline: float) -> None:
    pid: int | None = None
    openers: set[int] = set()
    incompatible_lock = False
    for line in _lsof(path, deadline=deadline).splitlines():
        if line.startswith("p") and line[1:].isdigit():
            pid = int(line[1:])
        if line.startswith("l") and line not in {"lW", "l "}:
            incompatible_lock = True
        if line == f"n{path}" and pid is not None:
            openers.add(pid)
    if openers != {owner} or incompatible_lock:
        raise ReplayError("The exact controller does not exclusively own the expected lease")


def _same_process(actual: ProcessIdentity, expected: ProcessIdentity) -> bool:
    return actual.identity == expected.identity and actual.ppid == expected.ppid


def _ended_or_zombie(pid: int, deadline: float) -> bool:
    _deadline(deadline)
    result = subprocess.run(
        ["/bin/ps", "-p", str(pid), "-o", "state="],
        stdin=subprocess.DEVNULL,
        capture_output=True,
        text=True,
        check=False,
        timeout=min(OBSERVATION_SECONDS, deadline - time.monotonic()),
        env={**os.environ, "LC_ALL": "C"},
    )
    if result.stderr.strip() or len(result.stdout) > 32:
        raise ReplayError("Owned process exit-state observation failed")
    if result.returncode == 1 and not result.stdout.strip():
        return True
    if result.returncode != 0 or re.fullmatch(r"[A-Za-z+<>]+", result.stdout.strip()) is None:
        raise ReplayError("Owned process exit-state observation failed")
    return result.stdout.strip().startswith("Z")


def _unchanged_owned(actual: ProcessIdentity, expected: ProcessIdentity, deadline: float) -> bool:
    if actual.identity == expected.identity:
        return True
    birth = actual.pid, actual.uid, actual.start
    expected_birth = expected.pid, expected.uid, expected.start
    # ps replaces the executable of a dying process with '(name)'. Its exact
    # birth identity must still match and an independent state observation must
    # confirm exit. Keep tracking the PID until absent; a zombie is not a handoff.
    return birth == expected_birth and _ended_or_zombie(actual.pid, deadline)


def _descendants(rows: list[ProcessIdentity], owner: int) -> tuple[ProcessIdentity, ...]:
    family = {owner}
    pending = rows.copy()
    found: list[ProcessIdentity] = []
    while pending:
        children = [row for row in pending if row.ppid in family]
        if not children:
            break
        found.extend(children)
        family.update(row.pid for row in children)
        pending = [row for row in pending if row.pid not in family]
    return tuple(found)


def _future_absent(request: Request) -> None:
    for stage in request.transfer:
        paths = (*stage.outputs, request.state / f"{stage.name}.json")
        for path in paths:
            _path(str(path))
        if any(path.exists() or path.is_symlink() for path in paths):
            raise ReplayError("A transferred stage already has a receipt or producer output")
        if any(request.state.glob(f"{stage.name}-*")):
            raise ReplayError("A transferred stage already has an execution attempt")


def _pin_inputs(request: Request, deadline: float) -> dict[str, str]:
    generated = {path for stage in (request.current, *request.transfer) for path in stage.outputs}
    sources = {
        path for stage in (request.current, *request.transfer) for path in stage.inputs
    } - generated
    sources.update(
        {
            request.path,
            request.workflow,
            Path(__file__).resolve(),
            Path(stage_runner.__file__).resolve(),
            Path(workflow_runner.__file__).resolve(),
        }
    )
    return {str(path): _digest(path, deadline) for path in sorted(sources)}


def _initial(request: Request, deadline: float) -> tuple[FileIdentity, dict[int, ProcessIdentity]]:
    _future_absent(request)
    if request.status.exists() or request.status.is_symlink():
        raise ReplayError("Expected an active job without a retained terminal status")
    log = _identity(request.log, empty=True)
    if log.size > MAX_LOG_BYTES:
        raise ReplayError("Job log exceeds its observation budget")
    saved = _json(request.state / f"{request.current.name}.json")
    if saved != {"status": "running", "fingerprint": request.current_fingerprint}:
        raise ReplayError("The declared current stage is not the expected active receipt")
    rows = _snapshot(request.job_root, deadline=deadline)
    actual = next((row for row in rows if row.pid == request.owner.pid), None)
    if actual is None or not _same_process(actual, request.owner):
        raise ReplayError("Expected controller process identity changed or disappeared")
    _lock_owner(request.state / ".workflow.lock", request.owner.pid, deadline)
    _lock_owner(request.state / f"{request.current.name}.lock", request.owner.pid, deadline)
    opened = _lsof(request.log, deadline=deadline, pid=request.owner.pid)
    if f"n{request.log}" not in opened.splitlines():
        raise ReplayError("Expected controller does not write the retained job log")
    family = {row.pid: row for row in _descendants(rows, request.owner.pid)}
    family[actual.pid] = actual
    return log, family


def _wait(
    request: Request, *, stack: ExitStack, deadline: float, family: dict[int, ProcessIdentity]
) -> None:
    global_held = False
    while True:
        _deadline(deadline)
        rows = _snapshot(request.job_root, deadline=deadline)
        children = _descendants(rows, request.owner.pid)
        if any(
            row.pid in family and not _unchanged_owned(row, family[row.pid], deadline)
            for row in children
        ):
            raise ReplayError("An observed owned process PID was reused during reservation")
        family.update({row.pid: row for row in children if row.pid not in family})
        living = {row.pid: row for row in rows if row.pid in family}
        if any(not _unchanged_owned(row, family[pid], deadline) for pid, row in living.items()):
            raise ReplayError("An observed owned process PID was reused during reservation")
        if not global_held:
            with ExitStack() as probe:
                global_held = probe.enter_context(execution_lock(request.state, ".workflow"))
                if global_held:
                    stack.enter_context(probe.pop_all())
        if global_held and not living and request.status.exists():
            return
        time.sleep(min(POLL_SECONDS, max(0.0, deadline - time.monotonic())))


def _completed(request: Request, deadline: float) -> dict[str, object]:
    receipt_path = request.state / f"{request.current.name}.json"
    receipt = _json(receipt_path)
    inputs = {str(path): _digest(path, deadline) for path in request.current.inputs}
    definition = {
        "command": request.current.command,
        "verifyCommand": request.current.verify_command,
        "inputs": inputs,
        "outputs": [str(path) for path in request.current.outputs],
        "timeoutSeconds": request.current.timeout_seconds,
        "workingDirectory": str(_repository()),
    }
    fingerprint = hashlib.sha256(json.dumps(definition, sort_keys=True).encode("utf-8")).hexdigest()
    expected: dict[str, object] = {
        "status": "complete",
        "fingerprint": request.current_fingerprint,
        "outputs": {str(path): _digest(path, deadline) for path in request.current.outputs},
    }
    if fingerprint != request.current_fingerprint or receipt != expected:
        raise ReplayError("The preceding stage or its domain verifier did not complete unchanged")
    return {"receipt": str(receipt_path), "sha256": _digest(receipt_path, deadline), **receipt}


def _terminal(request: Request, log: FileIdentity, deadline: float) -> dict[str, object]:
    status_info = _identity(request.status)
    if status_info.size > 16 or request.status.read_text(encoding="utf-8").strip() != "1":
        raise ReplayError("The controller did not retain its expected busy terminal exit")
    after = _identity(request.log)
    if (after.device, after.inode) != (
        log.device,
        log.inode,
    ) or not log.size <= after.size <= MAX_LOG_BYTES:
        raise ReplayError("The retained job log changed identity or exceeded its budget")
    with request.log.open("rb") as stream:
        stream.seek(log.size)
        appended = stream.read(MAX_LOG_BYTES + 1).decode("utf-8")
    pattern = rf"(?:^|\n){re.escape(request.boundary)}: busy(?:\n|$)"
    if re.search(pattern, appended) is None:
        raise ReplayError("No new reserved-stage busy refusal was observed in the job log")
    return {
        "status": str(request.status),
        "statusSha256": _digest(request.status, deadline),
        "exitCode": 1,
        "log": str(request.log),
        "logSha256": _digest(request.log, deadline),
        "reservedStageAttempted": True,
        "reservedStageProducerStarted": False,
        "reason": "cooperative-stage-lease-busy",
    }


def _proof_path(path: Path, request: Request) -> Path:
    proof = _path(str(path))
    sources = {value for stage in (request.current, *request.transfer) for value in stage.inputs}
    outputs = {value for stage in (request.current, *request.transfer) for value in stage.outputs}
    if (
        proof.exists()
        or proof.is_symlink()
        or not proof.parent.is_dir()
        or proof.is_relative_to(request.state)
        or proof
        in sources | outputs | {request.path, request.workflow, request.log, request.status}
    ):
        raise ReplayError("Handoff proof must be distinct, absent and outside existing namespaces")
    return proof


def reserve(request: Request, *, proof: Path) -> Path:
    """Hold a finite future lease, then exclusively publish a verified boundary proof.

    Exceptions leave all existing records intact and release every acquired lease.
    Even on failure the controller may already have refused the reserved stage;
    reconcile retained status, log, outputs and process facts before any retry.
    A returned proof records an old controller exit1, not workflow or media success.
    """
    if load_request(request.path) != request:
        raise ReplayError("The reservation request changed after validation")
    destination = _proof_path(proof, request)
    deadline = time.monotonic() + request.timeout_seconds
    with ExitStack() as stack:
        if not stack.enter_context(execution_lock(request.state, request.boundary)):
            raise ReplayError("The reserved future stage is already active or leased")
        namespaces = (
            request.state,
            request.job_root,
            request.job_root / f"{request.job_name}.launch",
            destination.parent,
        )
        initial_namespaces = {str(path): _namespace(path) for path in namespaces}
        leases = (request.state / ".workflow.lock", request.state / f"{request.boundary}.lock")
        initial_leases = {str(path): _identity(path, empty=True) for path in leases}
        log, family = _initial(request, deadline)
        initial_pins = _pin_inputs(request, deadline)
        _wait(request, stack=stack, deadline=deadline, family=family)
        _future_absent(request)
        if {str(path): _identity(path, empty=True) for path in leases} != initial_leases:
            raise ReplayError("The persistent workflow lease identity changed")
        terminal = _terminal(request, log, deadline)
        completed = _completed(request, deadline)
        if _pin_inputs(request, deadline) != initial_pins:
            raise ReplayError("Pinned reservation inputs changed before handoff")
        if {str(path): _namespace(path) for path in namespaces} != initial_namespaces:
            raise ReplayError("The reservation namespace identity changed")
        rows = _snapshot(request.job_root, deadline=deadline)
        if any(row.pid in family for row in rows):
            raise ReplayError("An observed owned process is present at final handoff")
        _future_absent(request)
        result: dict[str, object] = {
            "version": 1,
            "kind": "cooperative-workflow-stage-handoff",
            "status": "verified-boundary-handoff",
            "request": {"path": str(request.path), "sha256": request.sha256},
            "workflow": {"path": str(request.workflow), "sha256": request.workflow_sha256},
            "state": str(request.state),
            "owner": asdict(request.owner),
            "terminal": terminal,
            "precedingStage": completed,
            "inputPins": initial_pins,
            "namespacePins": {
                name: asdict(identity) for name, identity in initial_namespaces.items()
            },
            "leasePins": {name: asdict(identity) for name, identity in initial_leases.items()},
            "transferStages": [
                {"name": stage.name, "outputs": [str(path) for path in stage.outputs]}
                for stage in request.transfer
            ],
            "activeOwnedProcesses": [],
            "mediaApproved": False,
            "oldWorkflowComplete": False,
        }
        _deadline(deadline)
        with destination.open("x", encoding="utf-8") as stream:
            os.fchmod(stream.fileno(), 0o600)
            stream.write(json.dumps(result, ensure_ascii=False, allow_nan=False, indent=2) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
    return destination


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("request", type=Path)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--proof", type=Path)
    action.add_argument(
        "--check", action="store_true", help="Load the request without taking leases"
    )
    args = parser.parse_args(argv)
    try:
        request = load_request(args.request)
        if args.check:
            print(
                json.dumps(
                    {
                        "status": "loaded-request",
                        "request": str(request.path),
                        "sha256": request.sha256,
                        "reservationDispatched": False,
                    },
                    allow_nan=False,
                )
            )
            return 0
        result = reserve(request, proof=args.proof)
    except (ReplayError, StageError, OSError, ValueError, subprocess.TimeoutExpired) as error:
        print(f"Reservation unfulfilled ({type(error).__name__}): {error}", file=sys.stderr)
        return 1
    print(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
