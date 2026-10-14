import json
import os
import resource
import signal
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import pytest

from replay import stateful_audio, stateful_guard
from replay.core import ReplayError
from replay.native_invocation import read


@dataclass
class SyntheticGuardBoundary:
    """No method forwards a signal or reads the parent suite's resource history."""

    pid: int = 170000001
    group: int = 170000001
    sampled_rss: int = 1024
    child_peak_bytes: int = 1024
    stopped: list[tuple[int, int]] = field(default_factory=list)
    RUSAGE_CHILDREN = resource.RUSAGE_CHILDREN

    def getpid(self) -> int:
        return self.pid

    def getpgrp(self) -> int:
        return self.group

    def group_rss(self, group: int) -> int:
        assert group == self.group
        return self.sampled_rss

    def getrusage(self, who: int) -> resource.struct_rusage:
        assert who == resource.RUSAGE_CHILDREN
        peak = self.child_peak_bytes if sys.platform == "darwin" else self.child_peak_bytes // 1024
        return resource.struct_rusage((0, 0, peak, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0))

    def killpg(self, group: int, sig: int) -> None:
        self.stopped.append((group, sig))
        raise ReplayError("synthetic boundary: group stop")


@pytest.fixture
def unit_guard(monkeypatch: pytest.MonkeyPatch) -> SyntheticGuardBoundary:
    boundary = SyntheticGuardBoundary()
    # Replace only this module's dependencies, not the process-wide os/resource modules.
    monkeypatch.setattr(stateful_guard, "os", boundary)
    monkeypatch.setattr(stateful_guard, "resource", boundary)
    monkeypatch.setattr(stateful_guard, "group_rss", boundary.group_rss)
    return boundary


@pytest.fixture
def guard_request(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    # The checked-in fixture's env shebang must resolve the configured interpreter.
    monkeypatch.setenv("PATH", f"{Path(sys.executable).parent}{os.pathsep}{os.environ['PATH']}")
    request = tmp_path / "request.json"
    request.write_text(
        json.dumps(
            {
                "outputDirectory": str(tmp_path / "output"),
                "statefulSelection": {
                    "budget": {
                        "maxRSSBytes": 256 * 1024 * 1024,
                        "maxSavedBytes": 64 * 1024 * 1024,
                        "maxWallSeconds": 2,
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    return request


def test_supervisor_success_and_verification(
    guard_request: Path, unit_guard: SyntheticGuardBoundary
) -> None:
    report = guard_request.parent / "guard.json"
    stateful_guard.supervise(request=guard_request, native=Path("/usr/bin/true"), report=report)
    stateful_guard.verify_guard(guard_request, report)
    assert read(report)["hardCap"] is False
    assert read(report)["overshootBoundBytes"] is None
    assert read(report)["kernelChildPeakRSSBytes"] == 1024
    assert unit_guard.stopped == []
    with pytest.raises(FileExistsError):
        report.unlink()
        stateful_guard.supervise(request=guard_request, native=Path("/usr/bin/true"), report=report)


def test_failed_native_records_before_group_stop(
    guard_request: Path, unit_guard: SyntheticGuardBoundary
) -> None:
    report = guard_request.parent / "guard.json"
    with pytest.raises(ReplayError, match="group stop"):
        stateful_guard.supervise(
            request=guard_request, native=Path("/usr/bin/false"), report=report
        )
    assert unit_guard.stopped == [(170000001, signal.SIGKILL)]
    assert read(report)["status"] == "failed"
    with pytest.raises(ReplayError):
        stateful_guard.verify_guard(guard_request, report)


@pytest.mark.parametrize("observation", ["kernel", "sampled"])
def test_resource_failure_targets_only_synthetic_owned_group(
    guard_request: Path, unit_guard: SyntheticGuardBoundary, observation: str
) -> None:
    real_parent_group = os.getpgrp()
    if observation == "kernel":
        unit_guard.child_peak_bytes = 512 * 1024 * 1024
    else:
        unit_guard.sampled_rss = 512 * 1024 * 1024
    report = guard_request.parent / "guard.json"
    with pytest.raises(ReplayError, match="synthetic boundary"):
        stateful_guard.supervise(request=guard_request, native=Path("/usr/bin/true"), report=report)
    assert read(report)["status"] == "failed"
    assert unit_guard.stopped == [(170000001, signal.SIGKILL)]
    assert unit_guard.stopped[0][0] != real_parent_group
    assert os.getpgrp() == real_parent_group


def test_nonowner_cannot_launch_or_signal(
    guard_request: Path, unit_guard: SyntheticGuardBoundary, monkeypatch: pytest.MonkeyPatch
) -> None:
    unit_guard.pid = 170000002

    def forbidden_launch(*args: object, **kwargs: object) -> None:
        pytest.fail("Ownership refusal must precede native launch")

    monkeypatch.setattr(subprocess, "Popen", forbidden_launch)
    report = guard_request.parent / "guard.json"
    with pytest.raises(ReplayError, match="workflow-owned session"):
        stateful_guard.supervise(request=guard_request, native=Path("/usr/bin/true"), report=report)
    assert unit_guard.stopped == []
    assert not report.exists()


@pytest.mark.parametrize("kind", ["link", "hardlink", "directory-link"])
def test_owned_links_refused(tmp_path: Path, kind: str) -> None:
    directory = tmp_path / "owned"
    directory.mkdir()
    (directory / "nested").mkdir()
    source = directory / "nested/source"
    source.write_bytes(b"1234")
    assert stateful_guard.saved_bytes(directory) == 4
    if kind == "hardlink":
        (directory / "link").hardlink_to(source)
    else:
        (directory / "link").symlink_to(source if kind == "link" else source.parent)
    with pytest.raises(ReplayError):
        stateful_guard.saved_bytes(directory)


@pytest.mark.parametrize("reason", ["disk", "rss", "timeout", "link", "native"])
def test_real_guard_stops_owned_wait(guard_request: Path, reason: str) -> None:
    data = read(guard_request)
    selection = data["statefulSelection"]
    assert isinstance(selection, dict)
    budget = selection["budget"]
    assert isinstance(budget, dict)
    if reason == "rss":
        budget["maxRSSBytes"] = 1
    if reason == "disk":
        budget["maxSavedBytes"] = 1
    if reason == "timeout":
        budget["maxWallSeconds"] = 0.1
    data.update(testLink=reason == "link", testFailure=reason == "native")
    guard_request.write_text(json.dumps(data), encoding="utf-8")
    native = Path(__file__).with_name("stateful_guard_fixture.py").resolve()
    report = guard_request.parent / "guard.json"
    command = [
        sys.executable,
        "-m",
        "replay.stateful_guard",
        str(guard_request),
        str(native),
        str(report),
    ]
    result = subprocess.run(command, start_new_session=True, capture_output=True, timeout=5)
    assert result.returncode == -signal.SIGKILL
    assert read(report)["status"] == "failed"
    with pytest.raises(ReplayError):
        stateful_guard.verify_guard(guard_request, report)
    retry = subprocess.run(command, start_new_session=True, capture_output=True, timeout=5)
    assert retry.returncode != 0


def test_real_group_probe() -> None:
    assert stateful_guard.group_rss(os.getpgrp()) > 0


@pytest.mark.parametrize("cancel", [False, True])
def test_workflow_guard_cleanup_and_no_adoption(guard_request: Path, cancel: bool) -> None:
    root = guard_request.parent
    source = root / "source"
    source.write_bytes(b"synthetic")
    data = read(guard_request)
    data.update(source=str(source), startValue=0, durationValue=1, timescale=1)
    selection = data["statefulSelection"]
    assert isinstance(selection, dict)
    selection["checkpoints"] = [{"startValue": 0, "timescale": 48000, "sampleCount": 17}]
    budget = selection["budget"]
    assert isinstance(budget, dict)
    if not cancel:
        budget["maxSavedBytes"] = 1
    guard_request.write_text(json.dumps(data), encoding="utf-8")
    native = Path(__file__).with_name("stateful_guard_fixture.py").resolve()
    plan = stateful_audio.prepare(
        guard_request, native, Path("/usr/bin/true"), Path("/usr/bin/false"), root / "plan"
    )
    command = [sys.executable, "-m", "replay.workflow", str(plan), "--state", str(root / "state")]
    assert subprocess.run([*command, "--check"], capture_output=True, timeout=5).returncode == 0
    with subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE) as process:
        deadline = time.monotonic() + 3
        pid_file = root / "output/pid"
        while not pid_file.exists() and time.monotonic() < deadline:
            time.sleep(0.01)
        assert pid_file.exists()
        if cancel:
            process.send_signal(signal.SIGTERM)
        process.communicate(timeout=5)
        assert process.returncode == 1
    pid = int(pid_file.read_text(encoding="utf-8"))
    deadline = time.monotonic() + 2
    while time.monotonic() < deadline:
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            break
        time.sleep(0.01)
    with pytest.raises(ProcessLookupError):
        os.kill(pid, 0)
    assert read(root / "state/source-pts-observation.json")["status"] == "running"
    assert subprocess.run(command, capture_output=True, timeout=5).returncode == 1
    assert read(root / "state/source-pts-observation.json")["status"] == "running"
