"""Synthetic Executor contracts; never launches or controls an application."""

import ctypes
import json
import os
import subprocess
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from test_native_sequence import Fake, MemoryJournal

from replay import fcp_native_delivery_pump as pump
from replay import fcp_native_queue as queue
from replay import fcp_recovery_runtime as target
from replay.native_sequence import Checkpoint, Config, ContractError, run
from replay.sequence_adapters import reference
from replay.sequence_journal import Pin, encode
from replay.sequence_recovery import Boundary


def pin(path: Path) -> Pin:
    return Pin(path, reference(path)["sha256"])


class Host(target.SystemHost):
    def __init__(self) -> None:
        self.current = 101
        self.gone = False
        self.now = 10_000_000_000
        self.writable = False

    def birth(self, pid: int) -> str:
        return f"birth-{pid}"

    def now_ns(self) -> int:
        return self.now

    def absent(self, pid: int) -> bool:
        return self.gone

    def pressure(self) -> int:
        return 4 if self.current == 101 else 1

    def footprint(self, pid: int) -> int:
        return 12000 if pid == 101 else 1500

    def free(self, path: Path) -> int:
        return 100000

    def writer_free(self, path: Path) -> bool:
        return not self.writable


class Transport:
    def __init__(self, host: Host) -> None:
        self.host = host
        self.calls: list[str] = []
        self.override: dict[str, object] = {}

    def call(self, tool: str, arguments: dict[str, object], timeout: int) -> dict[str, object]:
        name = tool.split(".")[-1]
        self.calls.append(name)
        row: dict[str, object] = {"pid": self.host.current, "dispatched": True}
        if name == "app_quit":
            self.host.gone = True
            row.update(terminated=True, running=False)
        elif name == "app_launch":
            self.host.current = 202
            row.update(pid=202, running=True)
        elif name == "ui_windows":
            row["windows"] = [{"title": "Project", "sheets": []}]
        elif name == "fcp_timeline_read":
            row["finalCut"] = {"vocabularyVerified": True, "project": {"name": "Project"}}
        elif name == "ui_inspect":
            row.update(
                truncated=False,
                elements=[
                    {"identifier": "_NS:166", "path": [i], "value": "Idle", "unreadable": []}
                    for i in range(9)
                ]
                + [{"value": "Sharing", "unreadable": []}],
            )
        row.update(self.override)
        return {
            "exitCode": 0,
            "stdout": json.dumps(
                {"ok": True, "data": {"isError": False, "structuredContent": row}}
            ),
            "stderr": "",
        }


class Future:
    def __init__(self, value: Pin) -> None:
        self.value = value

    def prepare(self, checkpoint: Checkpoint, before: Boundary, after: Boundary) -> Pin:
        return self.value


class RuntimeTests(unittest.TestCase):
    def setUp(self) -> None:
        scratch = tempfile.TemporaryDirectory(
            dir=Path(__file__).resolve().parents[1] / "edits/work"
        )
        self.addCleanup(scratch.cleanup)
        self.root = Path(scratch.name)
        self.host = Host()
        self.transport = Transport(self.host)
        data = self.root / "data.json"
        data.write_bytes(b"{}")
        candidate = self.root / "candidate.json"
        candidate.write_bytes(b'{"candidateOnly":true,"canDispatch":false}')
        self.future = Future(pin(candidate))
        self.lease = self.root / ".fcp-native-queue-owner.json"
        self.reservation = self.root / "owner.json"
        raw = encode(
            {
                "pid": os.getpid(),
                "birth": self.host.birth(os.getpid()),
                "ownerPath": str(self.reservation),
            }
        )
        self.lease.write_bytes(raw)
        self.reservation.write_bytes(raw)
        self.settings = target.Settings(
            self.root,
            "provider",
            "bundle",
            queue.Identity(101, "birth-101"),
            queue.Identity(os.getpid(), self.host.birth(os.getpid())),
            pin(self.lease),
            (pin(data),),
            pin(data),
            pin(data),
            (),
            "Project",
            "memory-pressure",
            self.root,
            "Background",
            ("Window", "Background"),
        )
        self.patch = patch.object(queue, "GUI_LEASE_ROOT", self.root)
        self.patch.start()
        self.addCleanup(self.patch.stop)

    def runtime(self) -> target.ExecutorRecoveryRuntime:
        return target.ExecutorRecoveryRuntime(self.settings, self.transport, self.host, self.future)

    def test_actual_shapes_through_controller_and_new_lease(self) -> None:
        self.lease.unlink()
        config = self.settings.saved_project
        self.settings = replace(self.settings, pump_config=config)
        plan = SimpleNamespace(native=SimpleNamespace(target=self.settings.target, digest="queue"))

        class Steps(Fake):
            def materialize(inner, tail: tuple[int, ...]) -> Config:
                return Config(str(config.path), tail, config.sha256)

        steps = Steps()
        steps.allowed = False
        with patch.object(pump, "load", return_value=plan):
            result = run(
                (3, 4),
                MemoryJournal(),
                steps,
                recovery=target.OwnedRecovery(
                    self.settings, self.transport, self.host, self.future
                ),
            )
        self.assertEqual(result.status, "resource-refused")
        self.assertFalse(self.lease.exists())
        self.assertEqual(self.transport.calls.count("app_quit"), 1)
        self.assertEqual(self.transport.calls.count("app_launch"), 1)
        self.assertFalse(any("export" in call for call in self.transport.calls))

    def test_boundary_lifecycle_and_denials(self) -> None:
        runtime = self.runtime()
        checkpoint = Checkpoint(Config("x", (1,), "x"), "prepared")
        with self.assertRaises(ContractError):
            runtime.observe(replace(checkpoint, phase="started"))
        before = runtime.observe(checkpoint)
        runtime.verify_current(before)
        with self.assertRaises(ContractError):
            runtime.force_quit(before)
        with self.assertRaises(ContractError):
            runtime.launch_once(before)
        self.assertTrue(runtime.normal_quit(before).gone)
        after = runtime.launch_once(before)
        self.assertEqual(runtime.future_data(checkpoint, before, after), self.future.value)
        with self.assertRaises(ContractError):
            runtime.launch_once(before)
        self.host.now += 31_000_000_000
        with self.assertRaises(ContractError):
            runtime.verify_current(after)

    def test_output_identity_and_writers(self) -> None:
        media = self.root / "synthetic.bin"
        media.write_bytes(b"synthetic")
        info = media.stat()
        record = self.root / "output.json"
        record.write_bytes(
            encode(
                {
                    "version": 1,
                    "kind": "native-recovery-output-identity",
                    "path": str(media),
                    "identity": [
                        info.st_dev,
                        info.st_ino,
                        info.st_size,
                        info.st_mtime_ns,
                        info.st_ctime_ns,
                    ],
                }
            )
        )
        self.settings = replace(self.settings, outputs=(pin(record),))
        self.runtime()._outputs()
        self.host.writable = True
        with self.assertRaises(ContractError):
            self.runtime()._outputs()
        self.host.writable = False
        media.write_bytes(b"changed")
        with self.assertRaises(ContractError):
            self.runtime()._outputs()

    def test_settings_owner_target_and_call_budget(self) -> None:
        with self.assertRaises(ContractError):
            target.ExecutorRecoveryRuntime(
                replace(self.settings, maximum_age_ns=0), self.transport, self.host, self.future
            )
        runtime = self.runtime()
        runtime.calls = 24
        with self.assertRaises(ContractError):
            runtime._call("ui_windows", {})
        runtime.current = queue.Identity(101, "reused")
        with self.assertRaises(ContractError):
            runtime._target()
        self.reservation.write_bytes(b"changed")
        with self.assertRaises(ContractError):
            self.runtime()._owner()

    def test_system_absence_and_writer_modes(self) -> None:
        host = target.SystemHost()
        with patch.object(os, "kill", side_effect=ProcessLookupError):
            self.assertTrue(host.absent(1))
        with patch.object(os, "kill"):
            self.assertFalse(host.absent(1))
        for code, stdout, stderr, expected in (
            (1, b"", b"", True),
            (0, b"p1\nf3\nar\n", b"", True),
            (0, b"p1\nf3\naw\n", b"", False),
        ):
            with patch.object(
                subprocess,
                "run",
                return_value=subprocess.CompletedProcess([], code, stdout, stderr),
            ):
                self.assertEqual(host.writer_free(self.root / "file"), expected)
        for code, stdout, stderr in (
            (1, b"", b"error"),
            (0, b"xunknown", b""),
            (0, b"p1", b""),
            (0, b"ax", b""),
        ):
            with (
                patch.object(
                    subprocess,
                    "run",
                    return_value=subprocess.CompletedProcess([], code, stdout, stderr),
                ),
                self.assertRaises(ContractError),
            ):
                host.writer_free(self.root / "file")

    def test_sdk_footprint_uses_physical_field_and_rejects_failure(self) -> None:
        host = target.SystemHost()
        buffer = ctypes.create_string_buffer(96)
        ctypes.memmove(ctypes.addressof(buffer) + 72, (12345).to_bytes(8, "little"), 8)
        function = Mock(return_value=0)
        library = SimpleNamespace(proc_pid_rusage=function)
        with (
            patch("replay.fcp_recovery_runtime.sys.platform", "darwin"),
            patch.object(ctypes, "CDLL", return_value=library),
            patch.object(ctypes, "create_string_buffer", return_value=buffer),
        ):
            self.assertEqual(host.footprint(101), 12345)
            function.return_value = 1
            with self.assertRaises(ContractError):
                host.footprint(101)
            function.return_value = 0
            ctypes.memmove(ctypes.addressof(buffer) + 72, bytes(8), 8)
            with self.assertRaises(ContractError):
                host.footprint(101)
        with (
            patch("replay.fcp_recovery_runtime.sys.platform", "linux"),
            self.assertRaises(ContractError),
        ):
            host.footprint(101)


if __name__ == "__main__":
    unittest.main()
