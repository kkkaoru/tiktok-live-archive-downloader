"""Dependency-light Darwin process observations for bounded owned child jobs.

No application, caption, model or provider imports. Measurements are snapshots,
not kernel quotas; signals require the still-unreaped Popen child identity.
"""

from __future__ import annotations

import ctypes
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

from replay.core import ReplayError
from replay.native_invocation import require


class SystemHost:
    def birth(self, pid: int) -> str:
        require(sys.platform == "darwin", "Exact birth requires Darwin libproc")
        function = ctypes.CDLL("/usr/lib/libproc.dylib", use_errno=True).proc_pidinfo
        function.argtypes = [
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_uint64,
            ctypes.c_void_p,
            ctypes.c_int,
        ]
        function.restype = ctypes.c_int
        buffer = ctypes.create_string_buffer(136)
        require(function(pid, 3, 0, buffer, len(buffer)) == 136, "Exact process birth unavailable")
        seconds = int.from_bytes(buffer.raw[120:128], sys.byteorder)
        micros = int.from_bytes(buffer.raw[128:136], sys.byteorder)
        require(seconds > 0 and 0 <= micros < 1000000, "Invalid kernel birth")
        return f"{seconds}:{micros}"

    def footprint(self, pid: int) -> int:
        require(sys.platform == "darwin", "Physical footprint requires Darwin libproc")
        function = ctypes.CDLL("/usr/lib/libproc.dylib", use_errno=True).proc_pid_rusage
        function.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_void_p]
        function.restype = ctypes.c_int
        buffer = ctypes.create_string_buffer(96)
        require(function(pid, 0, buffer) == 0, "Physical footprint unavailable")
        # SDK rusage_info_v0: UUID16 + ten uint64 fields; field7 is phys_footprint.
        value = int.from_bytes(buffer.raw[72:80], sys.byteorder)
        require(value > 0, "Invalid physical footprint")
        return value

    def absent(self, pid: int) -> bool:
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return True
        return False


def sample_family(pid: int, birth: str, host: SystemHost) -> dict[str, object]:
    require(host.birth(pid) == birth, "Owned PID reused")
    result = subprocess.run(
        ["/bin/ps", "-axo", "pid=,ppid=,rss="],
        check=True,
        capture_output=True,
        text=True,
        timeout=2,
    )
    require(len(result.stdout) <= 4 * 1024**2, "Process inventory overflow")
    processes: dict[int, tuple[int, int]] = {}
    for line in result.stdout.splitlines():
        fields = line.split()
        require(
            len(fields) == 3 and all(value.isdecimal() for value in fields), "Malformed process row"
        )
        member, parent, rss = map(int, fields)
        require(member > 0 and member not in processes, "Duplicate process identity")
        processes[member] = (parent, rss * 1024)
    require(pid in processes, "Owned process vanished during inventory")
    members = {pid}
    for _ in range(len(processes)):
        new = {member for member, (parent, _) in processes.items() if parent in members} - members
        if not new:
            break
        members.update(new)
    births: dict[int, str] = {}
    physical = 0
    exited: list[int] = []
    for member in sorted(members):
        try:
            observed = host.birth(member)
            amount = host.footprint(member)
            require(host.birth(member) == observed, "Family identity changed")
        except ReplayError:
            if member == pid or not host.absent(member):
                raise
            exited.append(member)
            continue
        births[member] = observed
        physical += amount
    return {
        "observedAtNs": time.time_ns(),
        "members": [{"pid": member, "birth": value} for member, value in births.items()],
        "exitedDuringObservation": exited,
        "physicalBytes": physical,
        "rssBytes": sum(processes[member][1] for member in births),
    }


def cleanup_child(
    child: subprocess.Popen[bytes], birth: str | None, host: SystemHost
) -> dict[str, object]:
    result: dict[str, object] = {"signalSent": False, "terminalObserved": False}
    try:
        if child.poll() is None:
            require(birth is not None and host.birth(child.pid) == birth, "Cleanup owner changed")
            require(
                os.getpgid(child.pid) == child.pid and os.getsid(child.pid) == child.pid,
                "Cleanup group/session changed",
            )
            # No intervening reap: this child PID cannot be reused before killpg.
            os.killpg(child.pid, signal.SIGKILL)
            result["signalSent"] = True
        child.wait(timeout=5)
        result["terminalObserved"] = True
    except (OSError, ReplayError, subprocess.TimeoutExpired) as error:
        result.update(errorType=type(error).__name__, error=str(error))
    return result


def sandbox_profile(root: Path) -> str:
    require(
        root.is_absolute()
        and root == root.resolve()
        and not any(c in str(root) for c in '\\"\n\r'),
        "Sandbox root must be canonical and representable",
    )
    return (
        "(version 1)\n(allow default)\n(deny file-write*)\n"
        f'(allow file-write* (subpath "{root}") (literal "/dev/null"))\n'
    )
