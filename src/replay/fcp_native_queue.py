"""Finite, exclusive Executor calls for native parts; never infer media completion.

The descriptor binds existing request artifacts. A live finite run owns every
call; it is not a daemon, renderer, automatic retry or replacement for followup.
"""

import argparse
import ctypes
import json
import math
import os
import re
import shutil
import stat
import subprocess
import sys
import time
from collections import deque
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Protocol

from replay.artifact_json import _json, _path, _pin
from replay.core import ReplayError
from replay.native_export_receipt import _dispatch, native_payload
from replay.recordings import record
from replay.stages import file_digest

MAX_BYTES = 8 * 1024 * 1024
GUI_LEASE_ROOT = Path(__file__).resolve().parents[2] / "edits" / "work"
TOOLS = {
    "import": "app_open_document",
    "windows": "ui_windows",
    "project": "fcp_project_open",
    "export": "fcp_export",
    "background-open": "ui_menu_select",
    "background-wait": "ui_wait",
    "background": "ui_inspect",
    "background-close": "ui_perform",
}
PHASES = tuple(TOOLS)
TOOLS.update(
    {
        "chooser-inspect": "ui_inspect",
        "chooser-select": "ui_set_value",
        "chooser-readback": "ui_inspect",
        "chooser-confirm": "ui_perform",
        "chooser-wait": "ui_wait",
        "chooser-absent": "ui_windows",
        "observation-wait": "ui_wait",
    }
)
CHOOSER_TITLES = ("Open Library", "ライブラリを開く")
READONLY_OBSERVATIONS = frozenset(
    {"windows", "chooser-inspect", "chooser-readback", "chooser-absent", "background"}
)
BUSY_ERROR = (
    "Unavailable: The app is busy or still launching (AXError -25204); "
    "wait or observe before retrying"
)


@dataclass(frozen=True)
class Identity:
    pid: int
    birth: str


@dataclass(frozen=True)
class Operation:
    phase: str
    request: Path


@dataclass(frozen=True)
class Part:
    project: str
    movie: Path
    operations: tuple[Operation, ...]
    assembly_part: int
    library_basename: str


@dataclass
class ObservationState:
    chooser_title: str = ""
    selected_path: tuple[int, ...] = ()
    output: "OutputObservation | None" = None
    exported: bool = False


@dataclass(frozen=True)
class OutputObservation:
    device: int
    inode: int
    size: int
    modified_ns: int
    changed_ns: int


@dataclass(frozen=True)
class Queue:
    config: Path
    digest: str
    provider: str
    executable: Path
    target: Identity
    bundle: str
    root: Path
    boot: Path
    parts: tuple[Part, ...]
    pins: tuple[tuple[Path, str], ...]
    inventory: Path
    max_calls: int
    timeout: int
    max_age: int
    peak_bytes: int
    floor_bytes: int
    capacity: int
    disk_allocation: Path | None = None


@dataclass(frozen=True)
class Result:
    state: str
    reason: str
    calls: int


class Host(Protocol):
    def birth(self, pid: int) -> str: ...
    def now_ns(self) -> int: ...
    def free(self, path: Path) -> int: ...
    def floor(self, path: Path) -> int: ...


class Transport(Protocol):
    def call(self, tool: str, arguments: dict[str, object], timeout: int) -> dict[str, object]: ...


class SystemHost:
    def birth(self, pid: int) -> str:
        if sys.platform != "darwin":
            raise ReplayError("Exact native process identity requires macOS libproc")
        # Darwin proc_bsdinfo: start timeval occupies the final two uint64 fields.
        # No process command/argv is used as executable or process identity.
        library = ctypes.CDLL("/usr/lib/libproc.dylib", use_errno=True)
        function = library.proc_pidinfo
        function.argtypes = [
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_uint64,
            ctypes.c_void_p,
            ctypes.c_int,
        ]
        function.restype = ctypes.c_int
        buffer = ctypes.create_string_buffer(136)
        count = function(pid, 3, 0, buffer, len(buffer))
        if count != 136:
            raise ReplayError("Exact process birth is unavailable")
        seconds = int.from_bytes(buffer.raw[120:128], sys.byteorder)
        micros = int.from_bytes(buffer.raw[128:136], sys.byteorder)
        if seconds <= 0 or not 0 <= micros < 1_000_000:
            raise ReplayError("Invalid native process birth")
        return f"{seconds}:{micros}"

    def now_ns(self) -> int:
        return time.time_ns()

    def free(self, path: Path) -> int:
        return shutil.disk_usage(path).free

    def floor(self, path: Path) -> int:
        return max(30 * 1024**3, shutil.disk_usage(path).total // 20)


@dataclass(frozen=True)
class ExecutorTransport:
    executable: Path

    def call(self, tool: str, arguments: dict[str, object], timeout: int) -> dict[str, object]:
        result = subprocess.run(
            [str(self.executable), "call", tool, json.dumps(arguments, ensure_ascii=False)],
            capture_output=True,
            timeout=timeout,
            check=False,
        )
        if len(result.stdout) + len(result.stderr) > MAX_BYTES:
            raise ReplayError("Executor response exceeds byte budget; reconcile effects")
        return {
            "exitCode": result.returncode,
            "stdout": result.stdout.decode("utf-8", errors="replace"),
            "stderr": result.stderr.decode("utf-8", errors="replace"),
        }


def _int(value: object, *, maximum: int = 2**63 - 1) -> int:
    if type(value) is not int or not 1 <= value <= maximum:
        raise ReplayError("Expected a bounded positive integer")
    return value


def _text(value: object) -> str:
    if not isinstance(value, str) or not value or len(value) > 512 or "\0" in value:
        raise ReplayError("Expected bounded nonempty text")
    return value


def _rows(value: object, maximum: int) -> list[object]:
    if not isinstance(value, list) or not 1 <= len(value) <= maximum:
        raise ReplayError("Expected a bounded nonempty array")
    return value


def disk_allocation(
    reference: object, boot: Path, movie: Path, peak: int
) -> tuple[Path | None, tuple[tuple[Path, int], ...], tuple[Path, ...]]:
    """Read explicit observed role budgets; legacy descriptors remain conservative.

    The evidence is an explicit planning input, not proof of a future peak.
    Every role binds an actual directory/device and retained observation pins.
    Unknown cache locations must be budgeted on boot, never silently omitted.
    """
    if reference is None:
        return None, ((boot, peak), (movie.parent, peak)), ()
    path = _small_pin(reference)
    raw = _json(path)
    if (
        raw.get("version") != 1
        or raw.get("kind") != "native-disk-allocation"
        or raw.get("movie") != str(movie)
        or raw.get("bootVolume") != str(boot)
    ):
        raise ReplayError("Disk allocation must bind exact movie and boot paths")
    rows = _rows(raw.get("allocations"), 3)
    roles: set[str] = set()
    budgets: list[tuple[Path, int]] = []
    pins = [path]
    for value in rows:
        row = record(value)
        role = _text(row.get("role"))
        directory = _path(row.get("directory"))
        if role in roles or role not in {"movie", "local-metadata", "fcp-cache"}:
            raise ReplayError("Expected each native disk allocation role once")
        roles.add(role)
        if (
            not directory.is_dir()
            or row.get("device") != directory.stat().st_dev
            or type(row.get("device")) is not int
            or role == "movie"
            and directory != movie.parent
            or role == "local-metadata"
            and directory.stat().st_dev != boot.stat().st_dev
            or role == "fcp-cache"
            and row.get("locationKnown") is not True
            and directory != boot
        ):
            raise ReplayError("Disk allocation directory/device or cache location changed")
        amount = _int(row.get("peakAdditionalBytes"))
        observed = _int(row.get("observedPeakBytes"))
        if amount < observed or role == "movie" and amount != peak:
            raise ReplayError("Disk allocation must cover observed peak and exact movie budget")
        evidence = _rows(row.get("evidence"), 32)
        pins.extend(_small_pin(item) for item in evidence)
        budgets.append((directory, amount))
    if roles != {"movie", "local-metadata", "fcp-cache"}:
        raise ReplayError("All native disk allocation roles are required")
    return path, tuple(budgets), tuple(pins)


def disk_capacity(
    host: Host,
    boot: Path,
    movie: Path,
    peak: int,
    floor: int,
    allocation: Path | None,
    reservations: tuple[tuple[Path, int], ...] = (),
) -> bool:
    """Check all declared devices, charging same-device roles additively."""
    reference = (
        None if allocation is None else {"path": str(allocation), "sha256": file_digest(allocation)}
    )
    _, budgets, _ = disk_allocation(reference, boot, movie, peak)
    if allocation is None:
        reserved = sum(amount for _, amount in reservations)
        return all(
            host.free(path) - peak - reserved >= max(floor, host.floor(path))
            for path in dict.fromkeys((boot, movie.parent, *(path for path, _ in reservations)))
        )
    amounts: dict[int, int] = {}
    paths = tuple(dict.fromkeys((boot, *(path for path, _ in budgets + reservations))))
    for path, amount in budgets + reservations:
        device = path.stat().st_dev
        amounts[device] = amounts.get(device, 0) + amount
    return all(
        host.free(path) - amounts.get(path.stat().st_dev, 0) >= max(floor, host.floor(path))
        for path in paths
    )


def load(path: Path) -> Queue:
    path = _path(str(path))
    raw = _json(path)
    if raw.get("version") != 1 or raw.get("kind") != "executor-native-part-queue":
        raise ReplayError("Unsupported native queue descriptor")
    provider = _text(raw.get("provider"))
    if re.fullmatch(r"[a-zA-Z0-9_.-]+", provider) is None:
        raise ReplayError("Invalid Executor provider path")
    refs = [_pin(ref) for ref in _rows(raw.get("corePins"), 1024)]
    executable = _pin(raw.get("executor"))
    refs.append(executable)
    refs.append(Path(__file__).resolve())
    parts: list[Part] = []
    for item in _rows(raw.get("parts"), 1024):
        row = record(item)
        operations: list[Operation] = []
        for item_operation in _rows(row.get("operations"), len(PHASES)):
            operation = record(item_operation)
            phase = _text(operation.get("phase"))
            request = _pin(operation.get("request"))
            refs.append(request)
            operations.append(Operation(phase, request))
        if tuple(op.phase for op in operations) != PHASES:
            raise ReplayError("Each part requires the complete ordered native operation contract")
        basename = _text(row.get("libraryBasename"))
        if Path(basename).name != basename or not basename.endswith(".fcpbundle"):
            raise ReplayError("Expected an exact library basename")
        parts.append(
            Part(
                _text(row.get("project")),
                _path(row.get("movie")),
                tuple(operations),
                _int(row.get("assemblyPart")),
                basename,
            )
        )
    if len({part.movie for part in parts}) != len(parts):
        raise ReplayError("Part outputs must be unique")
    if len({part.assembly_part for part in parts}) != len(parts):
        raise ReplayError("Assembly part identities must be unique")
    target = record(raw.get("target"))
    limits = record(raw.get("limits"))
    root, boot = _path(raw.get("eventRoot")), _path(raw.get("bootVolume"))
    inventory = _path(raw.get("inventory"))
    if not root.is_dir() or not boot.is_dir():
        raise ReplayError("Existing canonical event and boot roots are required")
    grant_path = _pin(raw.get("guiGrant"))
    grant = _json(grant_path)
    if (
        grant.get("version") != 1
        or grant.get("kind") != "native-gui-idle-handoff"
        or grant.get("released") is not True
        or grant.get("eventRoot") != str(root)
        or grant.get("target") != target
        or not isinstance(grant.get("previousOwner"), str)
        or not grant["previousOwner"]
    ):
        raise ReplayError("An explicit exact-target idle GUI ownership grant is required")
    refs.append(grant_path)
    allocation, _, allocation_pins = disk_allocation(
        limits.get("diskAllocation"), boot, parts[0].movie, _int(limits.get("peakAdditionalBytes"))
    )
    if allocation is not None and len(parts) != 1:
        raise ReplayError("Explicit disk allocation requires one exact movie")
    refs.extend(allocation_pins)
    return Queue(
        path,
        file_digest(path),
        provider,
        executable,
        Identity(_int(target.get("pid")), _text(target.get("birth"))),
        _text(target.get("bundleID")),
        root,
        boot,
        tuple(parts),
        tuple((ref, file_digest(ref)) for ref in dict.fromkeys(refs)),
        inventory,
        _int(limits.get("maxCalls"), maximum=10000),
        _int(limits.get("callSeconds"), maximum=300),
        _int(limits.get("observationSeconds"), maximum=30),
        _int(limits.get("peakAdditionalBytes")),
        _int(limits.get("minimumFreeBytes")),
        _int(limits.get("capacity"), maximum=1024),
        allocation,
    )


def _save(path: Path, value: dict[str, object]) -> None:
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2)
        handle.flush()
        os.fsync(handle.fileno())


def _guard(queue: Queue, host: Host, owner: Identity, part: Part, *, export: bool) -> None:
    if host.birth(owner.pid) != owner.birth or host.birth(queue.target.pid) != queue.target.birth:
        raise ReplayError("Owner or target process birth changed")
    if file_digest(queue.config) != queue.digest or any(
        file_digest(path) != digest for path, digest in queue.pins
    ):
        raise ReplayError("Queue core pins changed")
    if not export:
        # Observation after a long export makes no new heavy allocation. Do not
        # falsify the original inventory time or block observation on its age.
        return
    inventory = _json(queue.inventory)
    captured = _int(inventory.get("capturedAtNs"))
    if not 0 <= host.now_ns() - captured <= queue.max_age * 1_000_000_000:
        raise ReplayError("Resource inventory is stale or future dated")
    active = inventory.get("activeRenders")
    if type(active) is not int or active < 0 or active + 1 > queue.capacity:
        raise ReplayError("Native queue exceeds observed capacity")
    if not disk_capacity(
        host, queue.boot, part.movie, queue.peak_bytes, queue.floor_bytes, queue.disk_allocation
    ):
        raise ReplayError("Insufficient measured peak disk capacity")
    if not part.movie.parent.is_dir() or part.movie.is_symlink():
        raise ReplayError("Native output parent or symlink is invalid")
    if export and part.movie.exists():
        raise ReplayError("Output already exists; reconcile instead of resubmitting")


def _arguments(queue: Queue, part: Part, operation: Operation) -> dict[str, object]:
    return _request_arguments(part, operation, bundle=queue.bundle, pins=dict(queue.pins))


def _request_arguments(
    part: Part, operation: Operation, *, bundle: str, pins: dict[Path, str]
) -> dict[str, object]:
    args = _json(operation.request)
    if file_digest(operation.request) != pins[operation.request]:
        raise ReplayError("Native request changed during planning")
    if args.get("bundleID") != bundle:
        raise ReplayError("Request bundle differs from the queue target")
    if operation.phase in ("project", "export"):
        if "app" in args:
            raise ReplayError("FCP-specific requests do not accept an app field")
    elif args.get("app") != "finalCutPro":
        raise ReplayError("Open Document and UI requests require app finalCutPro")
    if operation.phase in ("project", "export") and args.get("project") != part.project:
        raise ReplayError("Request project differs from the exact queued project")
    if operation.phase == "project" and (
        args.get("library") != part.library_basename.removesuffix(".fcpbundle")
        or not isinstance(args.get("event"), str)
        or not args["event"]
    ):
        raise ReplayError("Exact library and event are required for project selection")
    if operation.phase == "export" and (
        args.get("directory") != str(part.movie.parent) or args.get("fileName") != part.movie.name
    ):
        raise ReplayError("Export request output differs from the exclusive queue output")
    if operation.phase == "background-close":
        element = record(args.get("element"))
        if args.get("action") != "press" or element.get("subrole") != "AXCloseButton":
            raise ReplayError("Background closure must target an explicit close button")
    if operation.phase == "background-open" and args.get("menuPath") not in (
        ["Window", "Background Tasks"],
        ["ウインドウ", "バックグラウンドタスク"],
    ):
        raise ReplayError("Only the Background Tasks menu is allowed")
    if operation.phase in ("background", "background-close", "background-wait") and record(
        args.get("window")
    ).get("title") not in ("Background Tasks", "バックグラウンドタスク"):
        raise ReplayError("Background operation must target its named window")
    if operation.phase in ("background-wait", "observation-wait"):
        timeout = args.get("timeoutSeconds")
        condition, role = (
            ("exists", "AXWindow")
            if operation.phase == "observation-wait"
            else ("absent", "AXProgressIndicator")
        )
        if (
            args.get("condition") != condition
            or record(args.get("element")).get("role") != role
            or type(timeout) not in (int, float)
            or not isinstance(timeout, (int, float))
            or not math.isfinite(timeout)
            or not 0 < timeout <= 30
        ):
            raise ReplayError("Native wait must be bounded readonly observation")
    if operation.phase == "import":
        source = _path(args.get("path"))
        if source not in pins or source.suffix != ".fcpxml":
            raise ReplayError("Import must use a core-pinned FCPXML")
    return args


def _small_pin(value: object) -> Path:
    path = _path(record(value).get("path"))
    if not path.is_file() or path.stat().st_size > MAX_BYTES:
        raise ReplayError("Preparation requires bounded metadata, not media")
    return _pin(value)


def _ref(path: Path) -> dict[str, str]:
    return {"path": str(path), "sha256": file_digest(path)}


def _prepared_save(path: Path, value: dict[str, object]) -> None:
    """Resume identical partial preparation without replacing an artifact."""
    if path.exists() or path.is_symlink():
        if path.is_symlink() or _json(path) != value:
            raise ReplayError("Prepared data changed; use a new generation")
        return
    _save(path, value)


def _planned_requests(
    row: dict[str, object], *, library: str, event: str, background: str
) -> dict[str, dict[str, object]]:
    export = record(row.get("export"))
    project = _text(export.get("project"))
    return {
        "import": {"path": str(_small_pin(row.get("xml")))},
        "windows": {},
        "project": {"project": project, "library": library, "event": event},
        "export": export,
        "background-open": {
            "menuPath": ["Window" if background == "Background Tasks" else "ウインドウ", background]
        },
        "background-wait": {
            "window": {"title": background},
            "condition": "absent",
            "element": {"role": "AXProgressIndicator"},
            "timeoutSeconds": 30,
        },
        "background": {"window": {"title": background}, "maxDepth": 20},
        "background-close": {
            "window": {"title": background},
            "element": {"subrole": "AXCloseButton"},
            "action": "press",
        },
    }


def validate_candidate(path: Path) -> tuple[Part, ...]:
    """Read candidate metadata only; this never returns an executable Queue."""
    raw = _json(_path(str(path)))
    if (
        raw.get("version") != 1
        or raw.get("kind") != "native-queue-candidate"
        or raw.get("candidateOnly") is not True
        or raw.get("canDispatch") is not False
        or raw.get("guiGrant") is not None
        or raw.get("readiness") is not None
    ):
        raise ReplayError("Expected explicitly non-dispatchable queue candidate")
    pins = {
        _small_pin(value): _text(record(value).get("sha256"))
        for value in _rows(raw.get("inputs"), 1024)
    }
    if Path(__file__).resolve() not in pins:
        raise ReplayError("Candidate must pin its preparation producer")
    parts: list[Part] = []
    for value in _rows(raw.get("parts"), 1024):
        row = record(value)
        operations = tuple(
            Operation(_text(record(op).get("phase")), _small_pin(record(op).get("request")))
            for op in _rows(row.get("operations"), len(PHASES))
        )
        if tuple(op.phase for op in operations) != PHASES:
            raise ReplayError("Candidate requires all ordered operations")
        part = Part(
            _text(row.get("project")),
            _path(row.get("movie")),
            operations,
            _int(row.get("assemblyPart")),
            _text(row.get("libraryBasename")),
        )
        for operation in operations:
            if operation.request not in pins:
                raise ReplayError("Candidate operation is not input-pinned")
            _request_arguments(part, operation, bundle=_text(raw.get("bundleID")), pins=pins)
        parts.append(part)
    if len({part.movie for part in parts}) != len(parts) or len(
        {part.assembly_part for part in parts}
    ) != len(parts):
        raise ReplayError("Candidate parts and outputs must be unique")
    return tuple(parts)


def prepare_data(request: Path, output: Path) -> Path:
    """Generate immutable operation data from existing manifests, without GUI ownership."""
    request = _path(str(request))
    raw = _json(request)
    if raw.get("version") != 1 or raw.get("kind") != "native-queue-preparation-request":
        raise ReplayError("Unsupported queue preparation request")
    manifest_path = _small_pin(raw.get("nativeActions"))
    manifest = _json(manifest_path)
    wanted = tuple(_int(value) for value in _rows(raw.get("assemblyParts"), 1024))
    rows = [record(value) for value in _rows(manifest.get("parts"), 1024)]
    selected = [row for row in rows if row.get("part") in wanted]
    if len(set(wanted)) != len(wanted) or sorted(
        _int(row.get("part")) for row in selected
    ) != sorted(wanted):
        raise ReplayError("Requested parts must exist exactly once in the manifest")
    library = _text(raw.get("libraryBasename"))
    if Path(library).name != library or not library.endswith(".fcpbundle"):
        raise ReplayError("Expected exact library basename")
    event, bundle = _text(raw.get("event")), _text(raw.get("bundleID"))
    background = _text(raw.get("backgroundTitle"))
    if background not in ("Background Tasks", "バックグラウンドタスク"):
        raise ReplayError("Unsupported explicit background window vocabulary")
    output = _path(str(output))
    if not output.is_relative_to(GUI_LEASE_ROOT) or not output.parent.is_dir():
        raise ReplayError("Preparation output must be repository-local")
    output.mkdir(mode=0o700, exist_ok=True)
    _prepared_save(output / "request.json", {"request": _ref(request)})
    pins = [request, manifest_path, Path(__file__).resolve()]
    parts: list[dict[str, object]] = []
    for row in selected:
        number = _int(row.get("part"))
        pins.append(_small_pin(row.get("xml")))
        requests = _planned_requests(
            row, library=library.removesuffix(".fcpbundle"), event=event, background=background
        )
        export = record(row.get("export"))
        operations: list[dict[str, object]] = []
        for phase, arguments in requests.items():
            destination = output / f"part-{number:04d}-{phase}.json"
            application = {} if phase in ("project", "export") else {"app": "finalCutPro"}
            _prepared_save(destination, {"bundleID": bundle, **application, **arguments})
            pins.append(destination)
            operations.append({"phase": phase, "request": _ref(destination)})
        parts.append(
            {
                "assemblyPart": number,
                "project": _text(export.get("project")),
                "movie": str(_path(export.get("directory")) / _text(export.get("fileName"))),
                "libraryBasename": library,
                "operations": operations,
            }
        )
    result = output / "candidate.json"
    _prepared_save(
        result,
        {
            "version": 1,
            "kind": "native-queue-candidate",
            "candidateOnly": True,
            "canDispatch": False,
            "guiGrant": None,
            "readiness": None,
            "bundleID": bundle,
            "parts": parts,
            "inputs": [_ref(path) for path in dict.fromkeys(pins)],
        },
    )
    validate_candidate(result)
    return result


def _envelope(captured: dict[str, object]) -> dict[str, object]:
    if captured.get("exitCode") != 0:
        raise ReplayError("Executor did not exit successfully")
    stdout = captured.get("stdout")
    if not isinstance(stdout, str):
        raise ReplayError("Executor stdout is absent")
    return record(json.loads(stdout))


def _payload(captured: dict[str, object]) -> dict[str, object]:
    return native_payload(_envelope(captured))


def _readonly_busy(phase: str, captured: dict[str, object]) -> bool:
    """Recognize only the deployed busy error of an idempotent observation."""
    if phase not in READONLY_OBSERVATIONS:
        return False
    raw = _envelope(captured)
    if raw.get("ok") is not False or raw.get("data") is not None:
        return False
    error = record(raw.get("error"))
    detail = error.get("message")
    if error.get("code") != "mcp_tool_error" or not isinstance(detail, str):
        return False
    failure = record(json.loads(detail))
    return (
        set(failure) == {"retrySafe", "error"}
        and failure.get("retrySafe") is False
        and failure.get("error") == BUSY_ERROR
    )


def _observation_retry(
    queue: Queue, *, operation: Operation, calls: int, stem: Path
) -> tuple[Queue, Operation]:
    """Keep the failed read and wait positively before repeating only that read."""
    request = queue.root / f"generated-{calls:04d}-00-observation-wait.json"
    _save(
        request,
        {
            "app": "finalCutPro",
            "bundleID": queue.bundle,
            "condition": "exists",
            "element": {"role": "AXWindow"},
            "timeoutSeconds": 30,
        },
    )
    _save(
        stem.with_suffix(".readonly-retry.json"),
        {
            "phase": operation.phase,
            "readOnly": True,
            "reason": BUSY_ERROR,
            "retrySafe": False,
            "nativePayloadModified": False,
            "osTargetPID": queue.target.pid,
            "osTargetBirth": queue.target.birth,
            "beforeAndAfterMatched": True,
        },
    )
    revised = replace(queue, pins=(*queue.pins, (request, file_digest(request))))
    return revised, Operation("observation-wait", request)


def _chooser_title(payload: dict[str, object]) -> str:
    windows = payload.get("windows")
    if not isinstance(windows, list) or not windows or len(windows) > 100:
        raise ReplayError("Bounded native windows are absent")
    rows = [record(row) for row in windows]
    if any(row.get("sheets") for row in rows):
        raise ReplayError("Unexpected modal sheet requires reconciliation")
    titles = [str(row["title"]) for row in rows if row.get("title") in CHOOSER_TITLES]
    if len(titles) > 1 or any(
        row.get("subrole") == "AXDialog" and row.get("title") not in CHOOSER_TITLES for row in rows
    ):
        raise ReplayError("Ambiguous or unknown native chooser")
    return titles[0] if titles else ""


def _ax_path(row: dict[str, object]) -> tuple[int, ...]:
    value = row.get("path")
    if not isinstance(value, list) or not value or len(value) > 32:
        raise ReplayError("AX element path is absent or unbounded")
    if any(type(step) is not int or not 0 <= step <= 4096 for step in value):
        raise ReplayError("AX element path is invalid")
    return tuple(step for step in value if isinstance(step, int))


def _elements(payload: dict[str, object]) -> list[dict[str, object]]:
    if payload.get("truncated") is not False:
        raise ReplayError("Native observation is truncated or incomplete")
    return [record(row) for row in _rows(payload.get("elements"), 500)]


def _library_row(payload: dict[str, object], basename: str) -> dict[str, object]:
    elements = _elements(payload)
    names = [
        row for row in elements if row.get("role") == "AXTextField" and row.get("value") == basename
    ]
    if len(names) != 1 or names[0].get("unreadable") != []:
        raise ReplayError("Exact library name is absent, duplicated or unreadable")
    name_path = _ax_path(names[0])
    rows = [
        row for row in elements if row.get("role") == "AXRow" and _ax_path(row) == name_path[:-1]
    ]
    if len(rows) != 1 or rows[0].get("unreadable") != []:
        raise ReplayError("Exact library row is absent or unreadable")
    return rows[0]


def _background_idle(payload: dict[str, object]) -> bool:
    elements = _elements(payload)
    labels = [row for row in elements if row.get("identifier") == "_NS:26"]
    progress = [row for row in elements if row.get("role") == "AXProgressIndicator"]
    if labels or progress:
        parents = [_ax_path(row)[:-1] for row in labels]
        names = [row.get("value") for row in labels]
        if (
            len(labels) != 9
            or len(set(parents)) != 9
            or len({path[:-1] for path in parents}) != 1
            or any(not isinstance(value, str) or not value for value in names)
            or len(set(names)) != 9
            or sum(value in ("Sharing", "共有") for value in names) != 1
            or any(row.get("unreadable") != [] for row in labels)
        ):
            raise ReplayError("Complete nine-category inventory is absent")
        if any(
            _ax_path(row)[:-1] not in parents
            for row in elements
            if row.get("identifier") == "_NS:166" or row in progress
        ):
            raise ReplayError("Native category state is outside the complete inventory")
        idle = True
        for parent in parents:
            groups = [
                row for row in elements if row.get("role") == "AXGroup" and _ax_path(row) == parent
            ]
            states = [
                row
                for row in elements
                if (row.get("identifier") == "_NS:166" or row in progress)
                and _ax_path(row)[:-1] == parent
            ]
            if (
                len(groups) != 1
                or groups[0].get("unreadable") != []
                or len(states) != 1
                or states[0].get("unreadable") != []
            ):
                raise ReplayError("Category state is absent, duplicated or unreadable")
            row = states[0]
            value = row.get("value")
            if row.get("role") == "AXProgressIndicator":
                if type(value) not in (str, int, float):
                    raise ReplayError("Category progress is malformed")
                try:
                    fraction = float(str(value))
                except ValueError as error:
                    raise ReplayError("Category progress is malformed") from error
                if not math.isfinite(fraction) or not 0 <= fraction <= 1:
                    raise ReplayError("Category progress is outside its finite range")
                # Even 100% progress is busy until the native category says idle.
                idle = False
            elif not isinstance(value, str) or not value:
                raise ReplayError("Category status is malformed")
            else:
                idle = idle and value in ("Idle", "待機中")
        return idle
    statuses = [row for row in elements if row.get("identifier") == "_NS:166"]
    paths = [_ax_path(row) for row in statuses]
    if (
        len(statuses) != 9
        or len(set(paths)) != 9
        or not any(row.get("value") in ("Sharing", "共有") for row in elements)
        or any(not isinstance(row.get("value"), str) or not row["value"] for row in statuses)
    ):
        raise ReplayError("Complete nine-category inventory is absent")
    return all(row.get("value") in ("Idle", "待機中") for row in statuses)


def _output_observation(part: Part, state: ObservationState, now: int) -> OutputObservation | None:
    try:
        value = part.movie.lstat()
    except FileNotFoundError:
        if state.output is not None:
            raise ReplayError("Observed native output disappeared") from None
        return None
    if not stat.S_ISREG(value.st_mode) or value.st_nlink != 1 or value.st_mtime_ns > now:
        raise ReplayError("Native output is linked, nonregular or newer than its observation")
    current = OutputObservation(
        value.st_dev, value.st_ino, value.st_size, value.st_mtime_ns, value.st_ctime_ns
    )
    previous = state.output
    if previous is not None and (
        (current.device, current.inode) != (previous.device, previous.inode)
        or current.size < previous.size
        or current.modified_ns < previous.modified_ns
        or current.changed_ns < previous.changed_ns
    ):
        raise ReplayError("Observed native output identity regressed or was replaced")
    return current


def _next_observations(
    phase: str,
    payload: dict[str, object],
    *,
    queue: Queue,
    part: Part,
    state: ObservationState,
    now: int,
) -> list[tuple[str, dict[str, object]]]:
    base: dict[str, object] = {"app": "finalCutPro", "bundleID": queue.bundle}
    if phase == "windows":
        state.chooser_title = _chooser_title(payload)
        if state.chooser_title:
            return [
                (
                    "chooser-inspect",
                    {
                        **base,
                        "window": {"title": state.chooser_title},
                        "maxDepth": 12,
                        "maxElements": 500,
                    },
                )
            ]
    base["window"] = {"title": state.chooser_title}
    if phase == "chooser-inspect":
        row = _library_row(payload, part.library_basename)
        state.selected_path = _ax_path(row)
        return [
            (
                "chooser-select",
                {
                    **base,
                    "element": {
                        "path": list(state.selected_path),
                        "role": "AXRow",
                        "containsText": part.library_basename,
                    },
                    "attribute": "selected",
                    "boolValue": True,
                },
            )
        ]
    if phase == "chooser-select":
        row = record(payload.get("element"))
        if (
            payload.get("effectVerified") is not True
            or row.get("selected") is not True
            or _ax_path(row) != state.selected_path
        ):
            raise ReplayError("Library row selection is not verified")
        return [("chooser-readback", {**base, "maxDepth": 12, "maxElements": 500})]
    if phase == "chooser-readback":
        row = _library_row(payload, part.library_basename)
        buttons = [
            item
            for item in _elements(payload)
            if item.get("role") == "AXButton" and item.get("title") in ("Choose", "選択")
        ]
        if (
            row.get("selected") is not True
            or _ax_path(row) != state.selected_path
            or len(buttons) != 1
            or buttons[0].get("enabled") is not True
            or buttons[0].get("unreadable") != []
            or "AXPress" not in _rows(buttons[0].get("actions"), 20)
        ):
            raise ReplayError("Selected library or enabled confirmation readback is absent")
        return [
            (
                "chooser-confirm",
                {
                    **base,
                    "element": {
                        "path": list(_ax_path(buttons[0])),
                        "role": "AXButton",
                        "title": buttons[0]["title"],
                    },
                    "action": "press",
                },
            )
        ]
    if phase == "chooser-confirm":
        return [
            (
                "chooser-wait",
                {
                    "app": "finalCutPro",
                    "bundleID": queue.bundle,
                    "condition": "absent",
                    "element": {"role": "AXSheet"},
                    "timeoutSeconds": 30,
                },
            )
        ]
    if phase == "chooser-wait":
        if payload.get("conditionMet") is not True:
            raise ReplayError("Import sheet disappearance is not verified")
        return [("chooser-absent", {"app": "finalCutPro", "bundleID": queue.bundle})]
    if phase == "background":
        idle = _background_idle(payload)
        current = _output_observation(part, state, now)
        settled = current is not None and current == state.output
        state.output = current
        if not idle or not settled:
            phases = ("background-wait", "background")
            return [
                (
                    item,
                    _arguments(queue, part, next(op for op in part.operations if op.phase == item)),
                )
                for item in phases
            ]
    return []


def _accept(phase: str, payload: dict[str, object], queue: Queue, part: Part, now: int) -> None:
    pid = payload.get("pid")
    if pid != queue.target.pid and not (
        phase in ("import", "background-wait", "chooser-wait", "observation-wait")
        and "pid" not in payload
    ):
        raise ReplayError("Observed native PID differs")
    if phase == "import":
        delivery = payload.get("delivery")
        if (
            payload.get("completionVerified") is not False
            or not isinstance(delivery, str)
            or not 0 < len(delivery) <= 4096
        ):
            raise ReplayError(
                "Open-document delivery acceptance is absent; import remains unverified"
            )
    elif phase in ("background-wait", "chooser-wait", "observation-wait"):
        elapsed = payload.get("elapsedSeconds")
        if (
            type(payload.get("conditionMet")) is not bool
            or type(elapsed) not in (int, float)
            or not isinstance(elapsed, (int, float))
            or not math.isfinite(elapsed)
            or elapsed < 0
        ):
            raise ReplayError("Malformed bounded wait observation")
        # Even conditionMet=false has a known observation outcome. Neither value
        # establishes idle/completion; the following fresh inventory decides.
    elif phase == "windows":
        _chooser_title(payload)
    elif phase in ("chooser-inspect", "chooser-readback"):
        _elements(payload)
    elif phase == "chooser-absent":
        if _chooser_title(payload):
            raise ReplayError("Chooser absence is not verified")
    elif phase == "project":
        project = record(record(payload.get("finalCut")).get("project"))
        if project.get("name") != part.project:
            raise ReplayError("Native exact project readback is absent or wrong")
    elif phase == "export":
        operation = next(item for item in part.operations if item.phase == "export")
        if _dispatch(_arguments(queue, part, operation), payload, part.project) != part.movie:
            raise ReplayError("Native export result differs from the reserved movie")
    elif phase == "background":
        if not _background_idle(payload):
            return
        if (
            not part.movie.is_file()
            or part.movie.stat().st_size == 0
            or part.movie.stat().st_mtime_ns > now
        ):
            raise ReplayError("Idle observation predates a nonempty completed output")
    elif payload.get("dispatched") is not True:
        raise ReplayError("Native dispatch is unverified")


def run(queue: Queue, transport: Transport, host: Host) -> Result:
    """One finite process owns a serial sequence. Existing reservations are never replayed."""
    owner = Identity(os.getpid(), host.birth(os.getpid()))
    reservation = queue.root / "owner.json"
    lease = GUI_LEASE_ROOT / ".fcp-native-queue-owner.json"
    token = {
        "pid": owner.pid,
        "birth": owner.birth,
        "queueSHA256": queue.digest,
        "ownerPath": str(reservation),
        "startedAtNs": host.now_ns(),
    }
    _save(lease, token)
    _save(reservation, token)
    owner_digest = file_digest(reservation)
    lease_digest = file_digest(lease)
    calls = 0
    observed_artifacts: dict[str, dict[str, str]] = {}
    for part_number, part in enumerate(queue.parts, 1):
        _guard(queue, host, owner, part, export=True)
        _save(
            part.movie.with_name(f".{part.movie.name}.native-queue-reservation.json"),
            {
                "queueSHA256": queue.digest,
                "owner": {"pid": owner.pid, "birth": owner.birth},
                "eventRoot": str(queue.root),
                "retrySafe": False,
            },
        )
        pending = deque(part.operations)
        occurrences: dict[str, int] = {}
        state = ObservationState()
        started = host.now_ns()
        deadline = started + queue.max_calls * queue.timeout * 1_000_000_000
        while pending:
            operation = pending.popleft()
            if calls >= queue.max_calls or host.now_ns() >= deadline:
                return Result(
                    "bounded-stop",
                    "Call or time budget exhausted; preserve ownership evidence",
                    calls,
                )
            _guard(queue, host, owner, part, export=operation.phase in ("import", "export"))
            if file_digest(reservation) != owner_digest or file_digest(lease) != lease_digest:
                raise ReplayError("Exclusive GUI owner token changed")
            args = _arguments(queue, part, operation)
            occurrences[operation.phase] = occurrences.get(operation.phase, 0) + 1
            occurrence = occurrences[operation.phase]
            suffix = f"-{occurrence:04d}" if occurrence > 1 else ""
            stem = queue.root / f"part-{part_number:04d}-{operation.phase}{suffix}"
            _save(
                stem.with_suffix(".intent.json"),
                {
                    "queueSHA256": queue.digest,
                    "owner": {"pid": owner.pid, "birth": owner.birth},
                    "requestSHA256": file_digest(operation.request),
                    "phase": operation.phase,
                    "output": str(part.movie),
                    "startedAtNs": host.now_ns(),
                },
            )
            calls += 1
            try:
                captured = transport.call(
                    f"{queue.provider}.{TOOLS[operation.phase]}", args, queue.timeout
                )
            except (subprocess.TimeoutExpired, OSError, ReplayError) as error:
                _save(
                    stem.with_suffix(".uncertain.json"),
                    {
                        "reason": str(error),
                        "outputExists": part.movie.exists(),
                        "retrySafe": False,
                    },
                )
                return Result("uncertain", "Transport outcome requires reconciliation", calls)
            _save(stem.with_suffix(".response.json"), captured)
            try:
                _save(stem.with_suffix(".native.json"), _envelope(captured))
                if (
                    host.birth(owner.pid) != owner.birth
                    or host.birth(queue.target.pid) != queue.target.birth
                ):
                    raise ReplayError("Process identity changed during Executor call")
                if file_digest(reservation) != owner_digest or file_digest(lease) != lease_digest:
                    raise ReplayError("GUI ownership changed during Executor call")
                now = host.now_ns()
                if not started <= now < deadline:
                    raise ReplayError("Native response is outside the finite execution interval")
                current = _output_observation(part, state, now) if state.exported else None
                if _readonly_busy(operation.phase, captured):
                    queue, wait = _observation_retry(
                        queue, operation=operation, calls=calls, stem=stem
                    )
                    pending.appendleft(operation)
                    pending.appendleft(wait)
                    continue
                payload = _payload(captured)
                _accept(operation.phase, payload, queue, part, now)
                if operation.phase == "export":
                    state.output = _output_observation(part, state, now)
                    current = state.output
                    state.exported = True
                if operation.phase == "background-close" and current != state.output:
                    raise ReplayError("Native output changed after the final idle observation")
                additions = _next_observations(
                    operation.phase, payload, queue=queue, part=part, state=state, now=now
                )
                generated: list[Operation] = []
                for index, (phase, arguments) in enumerate(additions):
                    request = queue.root / f"generated-{calls:04d}-{index:02d}-{phase}.json"
                    _save(request, arguments)
                    queue = replace(queue, pins=(*queue.pins, (request, file_digest(request))))
                    generated.append(Operation(phase, request))
                pending.extendleft(reversed(generated))
                native = stem.with_suffix(".native.json")
                observed_artifacts[operation.phase] = {
                    "path": str(native),
                    "sha256": file_digest(native),
                }
                _save(
                    stem.with_suffix(".identity.json"),
                    {
                        "nativePIDPresent": "pid" in payload,
                        "osTargetPID": queue.target.pid,
                        "osTargetBirth": queue.target.birth,
                        "beforeAndAfterMatched": True,
                        "nativePayloadModified": False,
                        "outputObservation": None
                        if current is None
                        else {
                            "device": current.device,
                            "inode": current.inode,
                            "bytes": current.size,
                            "modifiedAtNs": current.modified_ns,
                            "changedAtNs": current.changed_ns,
                        },
                    },
                )
            except (ReplayError, ValueError) as error:
                _save(
                    stem.with_suffix(".uncertain.json"),
                    {
                        "reason": str(error),
                        "outputExists": part.movie.exists(),
                        "retrySafe": False,
                    },
                )
                return Result("uncertain", "Native effect requires reconciliation", calls)
        # Phase one never releases another media allocation before the existing
        # followup factory has produced the actual frame/AV/publication proof.
        break
    if _output_observation(queue.parts[0], state, host.now_ns()) != state.output:
        raise ReplayError("Native output changed before followup readiness")
    _save(
        queue.root / "followup-ready.json",
        {
            "queueSHA256": queue.digest,
            "calls": calls,
            "completionVerified": False,
            "qualityApproved": False,
            "processedParts": 1,
            "remainingParts": len(queue.parts) - 1,
            "queueOrdinal": 1,
            "assemblyPart": queue.parts[0].assembly_part,
            "project": queue.parts[0].project,
            "movie": str(queue.parts[0].movie),
            "exportRequest": {
                "path": str(queue.parts[0].operations[3].request),
                "sha256": file_digest(queue.parts[0].operations[3].request),
            },
            "exportResponse": observed_artifacts["export"],
            "background": observed_artifacts["background"],
            "requiredNext": "Existing fcp_part_followup factory and actual full media proof",
        },
    )
    if file_digest(lease) != lease_digest:
        raise ReplayError("GUI lease changed before release")
    lease.unlink()
    return Result(
        "followup-ready", "Native observations retained; full media verification required", calls
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "operation", choices=("check", "run", "identity", "prepare-data", "validate-candidate")
    )
    parser.add_argument("queue", type=Path, nargs="?")
    parser.add_argument("output", type=Path, nargs="?")
    parser.add_argument("--pid", type=int)
    args = parser.parse_args()
    if args.operation == "identity":
        pid = _int(args.pid)
        print(json.dumps({"pid": pid, "birth": SystemHost().birth(pid)}))
        return
    if args.queue is None:
        parser.error("check/run requires a queue descriptor")
    if args.operation == "prepare-data":
        if args.output is None:
            parser.error("prepare-data requires an output directory")
        print(prepare_data(args.queue.resolve(), args.output.absolute()))
        return
    if args.operation == "validate-candidate":
        print(
            json.dumps(
                {"parts": len(validate_candidate(args.queue.resolve())), "canDispatch": False}
            )
        )
        return
    queue = load(args.queue)
    if args.operation == "check":
        for part in queue.parts:
            for operation in part.operations:
                _arguments(queue, part, operation)
        print(json.dumps({"parts": len(queue.parts), "appDispatched": False}))
        return
    result = run(queue, ExecutorTransport(queue.executable), SystemHost())
    print(json.dumps({"state": result.state, "reason": result.reason, "calls": result.calls}))


if __name__ == "__main__":
    main()
