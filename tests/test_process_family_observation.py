"""Kernel ABI and owned-family failure cases; no real signals or model imports."""

import ctypes
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from replay import process_family_observation as observe
from replay.core import ReplayError


@pytest.mark.parametrize("mode", ["ok", "count", "birth", "rusage", "footprint", "platform"])
def test_kernel_layout(monkeypatch: pytest.MonkeyPatch, mode: str) -> None:
    library = MagicMock()

    def info(
        pid: int, flavor: int, arg: int, buffer: ctypes.Array[ctypes.c_char], size: int
    ) -> int:
        raw = bytearray(136)
        raw[120:128] = (0 if mode == "birth" else 123).to_bytes(8, "little")
        raw[128:136] = (456).to_bytes(8, "little")
        buffer.raw = bytes(raw)
        return 0 if mode == "count" else 136

    def usage(pid: int, flavor: int, buffer: ctypes.Array[ctypes.c_char]) -> int:
        raw = bytearray(96)
        raw[72:80] = (0 if mode == "footprint" else 789).to_bytes(8, "little")
        buffer.raw = bytes(raw)
        return 1 if mode == "rusage" else 0

    library.proc_pidinfo.side_effect = info
    library.proc_pid_rusage.side_effect = usage
    monkeypatch.setattr(ctypes, "CDLL", MagicMock(return_value=library))
    monkeypatch.setattr(sys, "platform", "linux" if mode == "platform" else "darwin")
    host = observe.SystemHost()
    if mode in {"count", "birth", "platform"}:
        with pytest.raises(ReplayError):
            host.birth(10)
    else:
        assert host.birth(10) == "123:456"
    if mode in {"rusage", "footprint", "platform"}:
        with pytest.raises(ReplayError):
            host.footprint(10)
    else:
        assert host.footprint(10) == 789


@pytest.mark.parametrize("mode", ["present", "absent", "denied"])
def test_absence_is_not_permission_failure(monkeypatch: pytest.MonkeyPatch, mode: str) -> None:
    failure = (
        ProcessLookupError()
        if mode == "absent"
        else PermissionError()
        if mode == "denied"
        else None
    )
    monkeypatch.setattr(os, "kill", MagicMock(side_effect=failure))
    if mode == "denied":
        with pytest.raises(PermissionError):
            observe.SystemHost().absent(10)
    else:
        assert observe.SystemHost().absent(10) is (mode == "absent")


@pytest.mark.parametrize(
    "mode", ["ok", "exit", "unknown", "reuse", "malformed", "duplicate", "missing"]
)
def test_family_measurement(monkeypatch: pytest.MonkeyPatch, mode: str) -> None:
    host = MagicMock(spec=observe.SystemHost)

    def birth(pid: int) -> str:
        if pid == 11 and mode in {"exit", "unknown"}:
            raise ReplayError("gone")
        return "changed" if mode == "reuse" else str(pid)

    host.birth.side_effect = birth
    host.footprint.return_value = 100
    host.absent.return_value = mode == "exit"
    output = {"malformed": "bad", "duplicate": "10 1 2\n10 1 2", "missing": "11 1 3"}.get(
        mode, "10 1 2\n11 10 3\n12 1 9"
    )
    monkeypatch.setattr(subprocess, "run", MagicMock(return_value=SimpleNamespace(stdout=output)))
    if mode in {"ok", "exit"}:
        result = observe.sample_family(10, "10", host)
        assert result["physicalBytes"] == (200 if mode == "ok" else 100)
        assert result["rssBytes"] == (5120 if mode == "ok" else 2048)
    else:
        with pytest.raises(ReplayError):
            observe.sample_family(10, "10", host)


@pytest.mark.parametrize("mode", ["done", "kill", "reuse", "group", "denied", "timeout"])
def test_owned_cleanup(monkeypatch: pytest.MonkeyPatch, mode: str) -> None:
    child = MagicMock()
    child.pid = 10
    child.poll.return_value = 0 if mode == "done" else None
    host = MagicMock(spec=observe.SystemHost)
    host.birth.return_value = "other" if mode == "reuse" else "original"
    monkeypatch.setattr(os, "getpgid", lambda _: 20 if mode == "group" else 10)
    monkeypatch.setattr(os, "getsid", lambda _: 10)
    kill = MagicMock(side_effect=PermissionError() if mode == "denied" else None)
    monkeypatch.setattr(os, "killpg", kill)
    if mode == "timeout":
        child.wait.side_effect = subprocess.TimeoutExpired("owned", 5)
    result = observe.cleanup_child(child, "original", host)
    assert result["terminalObserved"] is (mode in {"done", "kill"})
    assert kill.call_count == (1 if mode in {"kill", "denied", "timeout"} else 0)


def test_sandbox() -> None:
    assert "(deny file-write*)" in observe.sandbox_profile(Path.cwd())
    with pytest.raises(ReplayError):
        observe.sandbox_profile(Path("relative"))
