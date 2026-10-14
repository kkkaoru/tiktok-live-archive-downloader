"""Finite Executor binding for observed FCP recovery; no import or export calls."""

import ctypes
import os
import stat
import subprocess
import sys
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Protocol

from replay import fcp_native_admission as admission
from replay import fcp_native_delivery_pump as pump
from replay import fcp_native_queue as queue
from replay.native_sequence import Checkpoint, ContractError
from replay.recordings import record
from replay.sequence_adapters import reference
from replay.sequence_journal import Pin, bounded_read, decode, encode, publish
from replay.sequence_recovery import Boundary, BoundedRecovery, Quit


class Host(admission.Host, Protocol):
    def birth(self, pid: int) -> str: ...
    def now_ns(self) -> int: ...
    def free(self, path: Path) -> int: ...
    def pressure(self) -> int: ...
    def footprint(self, pid: int) -> int: ...
    def absent(self, pid: int) -> bool: ...
    def writer_free(self, path: Path) -> bool: ...


class SystemHost(admission.SystemHost):
    def writer_free(self, path: Path) -> bool:
        result = subprocess.run(
            ["/usr/sbin/lsof", "-n", "-P", "-F", "pfa", "--", str(path)],
            capture_output=True,
            check=False,
            timeout=10,
        )
        if result.returncode == 1 and not result.stdout and not result.stderr:
            return True
        if result.returncode != 0 or result.stderr or len(result.stdout) > 1024 * 1024:
            raise ContractError("Exact output writer observation unavailable")
        modes: list[str] = []
        for line in result.stdout.decode("utf-8", errors="strict").splitlines():
            if line.startswith("a"):
                modes.append(line[1:])
            elif not line.startswith(("p", "f")):
                raise ContractError("Unexpected writer observation field")
        if not modes or any(mode not in {"r", "w", "u"} for mode in modes):
            raise ContractError("Output access mode is unknown")
        return all(mode == "r" for mode in modes)

    def footprint(self, pid: int) -> int:
        if sys.platform != "darwin":
            raise ContractError("Physical footprint requires Darwin libproc")
        # SDK sys/resource.h rusage_info_v0: UUID16 + ten uint64 fields;
        # ri_phys_footprint is uint64 field7, not resident size field6.
        library = ctypes.CDLL("/usr/lib/libproc.dylib", use_errno=True)
        function = library.proc_pid_rusage
        function.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_void_p]
        function.restype = ctypes.c_int
        buffer = ctypes.create_string_buffer(96)
        if function(pid, 0, buffer) != 0:
            raise ContractError("Actual physical footprint is unavailable")
        value = int.from_bytes(buffer.raw[72:80], sys.byteorder)
        if value <= 0:
            raise ContractError("Actual physical footprint must be positive")
        return value

    def absent(self, pid: int) -> bool:
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return True
        # EPERM and unknown errors propagate; they never mean actual absence.
        return False


class FutureFactory(Protocol):
    def prepare(self, checkpoint: Checkpoint, before: Boundary, after: Boundary) -> Pin: ...


@dataclass(frozen=True)
class Settings:
    root: Path
    provider: str
    bundle: str
    target: queue.Identity
    owner: queue.Identity
    # Existing original queue owner token/reservation; adapter never invents a grant.
    lease: Pin
    provider_evidence: tuple[Pin, ...]
    saved_project: Pin
    progress: Pin
    outputs: tuple[Pin, ...]
    project: str
    reason: str
    boot: Path
    background_title: str
    background_menu: tuple[str, ...]
    maximum_age_ns: int = 30_000_000_000
    pump_config: Pin | None = None


class ExecutorRecoveryRuntime:
    """Requires an actually held original owner lease; preserves uncertain effects."""

    def __init__(
        self, settings: Settings, transport: queue.Transport, host: Host, future: FutureFactory
    ) -> None:
        if (
            settings.root != settings.root.resolve()
            or not settings.root.is_dir()
            or not settings.provider
            or not settings.bundle
            or not settings.project
            or not settings.background_menu
            or not settings.background_title
            or not 0 < settings.maximum_age_ns <= 30_000_000_000
            or not settings.provider_evidence
        ):
            raise ContractError(
                "Explicit canonical recovery settings and provider evidence required"
            )
        self.settings, self.transport, self.host, self.future = settings, transport, host, future
        self.current = settings.target
        self.calls = 0
        self.observed: dict[tuple[int, str], int] = {}
        self.normal: Quit | None = None
        self.launched = False

    def _owner(self) -> None:
        cfg = self.settings
        cfg.lease.verify()
        raw = decode(bounded_read(cfg.lease.path))
        if (
            cfg.lease.path != queue.GUI_LEASE_ROOT / ".fcp-native-queue-owner.json"
            or raw.get("pid") != cfg.owner.pid
            or raw.get("birth") != cfg.owner.birth
            or cfg.owner.pid != os.getpid()
            or self.host.birth(cfg.owner.pid) != cfg.owner.birth
        ):
            raise ContractError("Exact current process must own the original global GUI lease")
        if cfg.pump_config is not None:
            cfg.pump_config.verify()
            plan = pump.load(cfg.pump_config.path)
            if raw.get("queueSHA256") != plan.native.digest:
                raise ContractError("Original pump GUI lease differs")
        else:
            owner_path = raw.get("ownerPath")
            if not isinstance(owner_path, str) or bounded_read(Path(owner_path)) != bounded_read(
                cfg.lease.path
            ):
                raise ContractError("Original GUI owner reservation differs")
        for pin in (*cfg.provider_evidence, cfg.saved_project, cfg.progress, *cfg.outputs):
            pin.verify()

    def _outputs(self) -> None:
        seen: set[Path] = set()
        for pin in self.settings.outputs:
            pin.verify()
            raw = decode(bounded_read(pin.path))
            name = raw.get("path")
            if (
                raw.get("kind") != "native-recovery-output-identity"
                or raw.get("version") != 1
                or not isinstance(name, str)
            ):
                raise ContractError("Owned outputs require small pinned identity records")
            path = Path(name)
            if path in seen or path != path.resolve() or path.is_symlink():
                raise ContractError("Owned output namespace is duplicated or noncanonical")
            seen.add(path)
            info = path.stat()
            identity = [info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns, info.st_ctime_ns]
            if (
                not stat.S_ISREG(info.st_mode)
                or raw.get("identity") != identity
                or not self.host.writer_free(path)
            ):
                raise ContractError("Owned output changed or has an active writer")
            after = path.stat()
            if (
                identity
                != [after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns, after.st_ctime_ns]
                or path.is_symlink()
            ):
                raise ContractError("Owned output changed during writer observation")

    def _target(self) -> None:
        self._owner()
        if self.host.birth(self.current.pid) != self.current.birth:
            raise ContractError("Exact FCP PID/birth changed before native call")

    def _call(
        self, tool: str, args: dict[str, object], *, terminal: bool = False
    ) -> tuple[dict[str, object], Pin]:
        self._target()
        if self.calls >= 24:
            raise ContractError("Finite native recovery call budget exhausted")
        self.calls += 1
        path = self.settings.root / f"call-{self.calls:02d}-{tool}"
        publish(
            path.with_suffix(".request.json"),
            encode(
                {
                    "tool": f"{self.settings.provider}.{tool}",
                    "arguments": args,
                    "targetPID": self.current.pid,
                    "targetBirth": self.current.birth,
                }
            ),
        )
        captured = self.transport.call(f"{self.settings.provider}.{tool}", args, 45)
        response = path.with_suffix(".response.json")
        publish(response, encode(captured))
        self._owner()
        if not terminal:
            self._target()
        return captured, Pin(response, reference(response)["sha256"])

    def _read(self, tool: str, arguments: dict[str, object]) -> tuple[dict[str, object], Pin]:
        raw, pin = self._call(tool, arguments)
        payload = queue._payload(raw)
        if payload.get("pid") != self.current.pid:
            raise ContractError("Native readback belongs to another FCP process")
        return payload, pin

    def observe(self, checkpoint: Checkpoint) -> Boundary:
        if checkpoint.phase != "prepared":
            raise ContractError("Native recovery cannot touch a started or uncertain effect")
        return self._observe()

    def _observe(self) -> Boundary:
        cfg = self.settings
        self._outputs()
        started = self.host.now_ns()
        args: dict[str, object] = {"app": "finalCutPro", "bundleID": cfg.bundle}
        windows, windows_pin = self._read("ui_windows", args)
        if queue._chooser_title(windows):
            raise ContractError("A library chooser is not saved-project readiness")
        timeline, timeline_pin = self._read(
            "fcp_timeline_read", {"bundleID": cfg.bundle, "project": cfg.project, "maxClips": 1}
        )
        cut = record(timeline.get("finalCut"))
        if (
            cut.get("vocabularyVerified") is not True
            or record(cut.get("project")).get("name") != cfg.project
        ):
            raise ContractError("Exact existing saved project was not read back")
        rows = windows.get("windows")
        if not isinstance(rows, list):
            raise ContractError("Bounded windows missing")
        already_open = any(record(row).get("title") == cfg.background_title for row in rows)
        if not already_open:
            opened, _ = self._read(
                "ui_menu_select", {**args, "menuPath": list(cfg.background_menu)}
            )
            if opened.get("dispatched") is not True:
                raise ContractError("Background menu effect is uncertain")
        idle, idle_pin = self._read(
            "ui_inspect",
            {**args, "window": {"title": cfg.background_title}, "maxDepth": 12, "maxElements": 500},
        )
        if not queue._background_idle(idle):
            raise ContractError("Recovery requires all nine native categories idle")
        closed, _ = self._read(
            "ui_perform",
            {
                **args,
                "window": {"title": cfg.background_title},
                "element": {"role": "AXButton", "subrole": "AXCloseButton"},
                "action": "press",
            },
        )
        if closed.get("dispatched") is not True:
            raise ContractError("Background close effect is uncertain")
        final, final_pin = self._read("ui_windows", args)
        final_rows = final.get("windows")
        if not isinstance(final_rows, list):
            raise ContractError("Final bounded windows missing")
        if queue._chooser_title(final) or any(
            record(row).get("title") == cfg.background_title for row in final_rows
        ):
            raise ContractError("Final windows are not idle and closed")
        self._target()
        now = self.host.now_ns()
        if not 0 <= now - started <= cfg.maximum_age_ns:
            raise ContractError("Native recovery observations became stale")
        self.observed[(self.current.pid, self.current.birth)] = started
        self._outputs()
        observation = (
            cfg.root / f"observation-{self.current.pid}-{self.current.birth.replace(':', '-')}.json"
        )
        publish(
            observation,
            encode(
                {
                    "version": 1,
                    "kind": "native-recovery-runtime-observation",
                    "target": {"pid": self.current.pid, "birth": self.current.birth},
                    "project": cfg.project,
                    "savedProject": reference(cfg.saved_project.path),
                    "progress": reference(cfg.progress.path),
                    "outputs": [reference(pin.path) for pin in cfg.outputs],
                    "observedAtNs": started,
                    "maximumAgeSeconds": cfg.maximum_age_ns // 1_000_000_000,
                    "providerEvidence": [reference(pin.path) for pin in cfg.provider_evidence],
                }
            ),
        )
        return Boundary(
            cfg.project,
            self.current.pid,
            self.current.birth,
            cfg.progress,
            cfg.reason,
            cfg.saved_project,
            (
                windows_pin,
                timeline_pin,
                idle_pin,
                final_pin,
                Pin(observation, reference(observation)["sha256"]),
            ),
            cfg.outputs,
            self.host.pressure(),
            self.host.footprint(self.current.pid),
            self.host.free(cfg.boot),
            True,
            True,
            True,
            True,
            True,
        )

    def verify_current(self, boundary: Boundary) -> None:
        self._target()
        self._outputs()
        if (
            (boundary.pid, boundary.birth) != (self.current.pid, self.current.birth)
            or boundary.project != self.settings.project
            or not 0
            <= self.host.now_ns() - self.observed.get((boundary.pid, boundary.birth), -(10**30))
            <= self.settings.maximum_age_ns
        ):
            raise ContractError("Current native boundary is stale or foreign")
        for pin in boundary.evidence:
            pin.verify()

    def normal_quit(self, boundary: Boundary) -> Quit:
        self.verify_current(boundary)
        raw, pin = self._call(
            "app_quit",
            {"app": "finalCutPro", "bundleID": self.settings.bundle, "force": False},
            terminal=True,
        )
        payload = queue._payload(raw)
        if payload.get("pid") != boundary.pid or payload.get("dispatched") is not True:
            raise ContractError(
                "Quit response identity/effect is uncertain; reconcile without retry"
            )
        gone = self.host.absent(boundary.pid)
        self.normal = Quit(pin, payload.get("terminated") is not True, False, gone)
        return self.normal

    def force_quit(self, boundary: Boundary) -> Quit:
        # Current native app_quit schema has no independently verified
        # unresponsive/PID-bound force contract. Never infer it from AX failure.
        raise ContractError("Force binding requires independent unresponsive evidence; unsupported")

    def launch_once(self, previous: Boundary) -> Boundary:
        self._owner()
        if (
            self.launched
            or self.normal is None
            or not self.normal.gone
            or not self.host.absent(previous.pid)
        ):
            raise ContractError("One launch requires independently absent original FCP")
        self.launched = True
        args: dict[str, object] = {
            "app": "finalCutPro",
            "bundleID": self.settings.bundle,
            "activate": False,
        }
        path = self.settings.root / "launch.request.json"
        publish(
            path,
            encode(
                {
                    "tool": f"{self.settings.provider}.app_launch",
                    "arguments": args,
                    "previousPID": previous.pid,
                    "previousBirth": previous.birth,
                }
            ),
        )
        raw = self.transport.call(f"{self.settings.provider}.app_launch", args, 45)
        publish(self.settings.root / "launch.response.json", encode(raw))
        self._owner()
        payload = queue._payload(raw)
        pid = payload.get("pid")
        if (
            type(pid) is not int
            or not isinstance(pid, int)
            or pid <= 0
            or payload.get("dispatched") is not True
            or payload.get("running") is not True
        ):
            raise ContractError("Launch effect is uncertain; no second launch")
        identity = queue.Identity(pid, self.host.birth(pid))
        if identity == self.current:
            raise ContractError("Launch returned the old process identity")
        self.current = identity
        return self._observe()

    def future_data(self, checkpoint: Checkpoint, before: Boundary, after: Boundary) -> Pin:
        self.verify_current(after)
        return self.future.prepare(checkpoint, before, after)


class OwnedRecovery:
    """Finite unleased controller callback using the original pump's granted lease."""

    def __init__(
        self, settings: Settings, transport: queue.Transport, host: Host, future: FutureFactory
    ) -> None:
        self.settings, self.transport, self.host, self.future = settings, transport, host, future

    def refused(self, checkpoint: Checkpoint) -> None:
        cfg = self.settings
        if checkpoint.phase != "prepared" or cfg.pump_config is None:
            raise ContractError("Recovery needs an unstarted original granted pump config")
        cfg.pump_config.verify()
        if (
            checkpoint.config.identity != str(cfg.pump_config.path)
            or checkpoint.config.digest != cfg.pump_config.sha256
        ):
            raise ContractError("Recovery pump does not match the prepared checkpoint")
        plan = pump.load(cfg.pump_config.path)
        if (
            plan.native.target != cfg.target
            or cfg.owner.pid != os.getpid()
            or self.host.birth(cfg.owner.pid) != cfg.owner.birth
        ):
            raise ContractError("Recovery owner/target differ from the actual granted pump")
        path = queue.GUI_LEASE_ROOT / ".fcp-native-queue-owner.json"
        publish(
            path,
            encode(
                {"pid": cfg.owner.pid, "birth": cfg.owner.birth, "queueSHA256": plan.native.digest}
            ),
        )
        pump._lease(plan, self.host)
        runtime = ExecutorRecoveryRuntime(
            replace(cfg, lease=Pin(path, reference(path)["sha256"])),
            self.transport,
            self.host,
            self.future,
        )
        # No finally-release: uncertain effects retain the exact lease for reconciliation.
        BoundedRecovery(cfg.root, runtime).refused(checkpoint)
        runtime._target()
        runtime.settings.lease.verify()
        path.unlink()
