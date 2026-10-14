from __future__ import annotations

import ctypes
import fcntl
import hashlib
import json
import os
import subprocess
import sys
from dataclasses import replace
from fractions import Fraction
from pathlib import Path

import pytest

from replay import disk_budget
from replay import measured_job_admission as admission
from replay.core import ReplayError
from replay.recordings import record


class FakeHost:
    def __init__(self, job_request: admission.Request) -> None:
        self.clock = 10.0
        self.epoch = 1_000_000_000
        self.calls = 0
        self.observations = 0
        self.code = 0
        self.error: OSError | None = None
        self.snapshots = [snapshot(job_request)]
        self.change_request = False

    def now_ns(self) -> int:
        return self.epoch

    def monotonic(self) -> float:
        return self.clock

    def sleep(self, seconds: float) -> None:
        self.clock += seconds

    def observe(self, job_request: admission.Request) -> admission.Snapshot:
        self.observations += 1
        result = self.snapshots[0]
        if len(self.snapshots) > 1:
            self.snapshots.pop(0)
        if self.change_request:
            job_request.path.write_bytes(b"changed")
        return result

    def execute(self, job_request: admission.Request) -> int:
        self.calls += 1
        if self.error is not None:
            raise self.error
        return self.code


class NativeFixture:
    def __init__(self) -> None:
        self.births = ["100:25"]

    def birth(self, _pid: int) -> str:
        if len(self.births) > 1:
            return self.births.pop(0)
        return self.births[0]

    def pressure(self) -> int:
        return 1


class PathFunction:
    argtypes: object = None
    restype: object = None

    def __init__(self, path: bytes, count: int | None = None) -> None:
        self.path = path
        self.count = count

    def __call__(self, _pid: int, buffer: ctypes.Array[ctypes.c_char], _length: int) -> int:
        buffer.value = self.path
        return len(self.path) if self.count is None else self.count


class PathLibrary:
    def __init__(self, function: PathFunction) -> None:
        self.proc_pidpath = function


def snapshot(job_request: admission.Request) -> admission.Snapshot:
    return admission.Snapshot(
        1_000_000_000,
        4096,
        2,
        3,
        100,
        1,
        (admission.FamilyUsage(job_request.families[0].parent, (10, 11), 100),),
        (disk_budget.Headroom(job_request.work, 100 * 1024**3, 90 * 1024**3, 30 * 1024**3, 0),),
    )


def write_request(path: Path, raw: dict[str, object]) -> admission.Request:
    data = json.dumps(raw).encode("utf-8")
    path.write_bytes(data)
    path.chmod(0o600)
    return admission.load(path, hashlib.sha256(data).hexdigest())


@pytest.fixture
def request_data(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict[str, object]:
    root = tmp_path / "checkout"
    work = root / "edits" / "work" / "admission"
    work.mkdir(parents=True)
    (work / "tmp").mkdir()
    (work / "cache").mkdir()
    tool = root / "python"
    tool.write_bytes(b"synthetic executable")
    tool.chmod(0o700)
    monkeypatch.setattr(admission, "ROOT", root)
    return {
        "version": 1,
        "kind": "measured-job-admission",
        "name": "future-stage",
        "workDirectory": str(work),
        "scratchDirectory": str(work / "tmp"),
        "cacheDirectory": str(work / "cache"),
        "diskPaths": [str(work)],
        "executable": {
            "path": str(tool),
            "sha256": hashlib.sha256(tool.read_bytes()).hexdigest(),
            "invocationPath": str(tool),
            "linkTarget": None,
        },
        "command": [str(tool), "-m", "replay.workflow", "retained-plan.json"],
        "families": [
            {
                "parent": {
                    "pid": 10,
                    "birth": "100:25",
                    "executable": str(tool),
                    "commandSHA256": "a" * 64,
                },
                "peakMemoryBytes": 300,
                "remainingDiskBytes": 20,
            }
        ],
        "policy": {
            "newMemoryBytes": 100,
            "newDiskBytes": 30,
            "osBurstMarginBytes": 40,
            "ocrBurstMarginBytes": 50,
            "minimumFreeBytes": 60,
            "maximumPressure": 1,
            "maximumSwapBytes": 500,
            "maximumSwapGrowthBytes": 20,
            "maximumSnapshotAgeSeconds": 5,
            "waitSeconds": 2,
            "pollSeconds": 1,
        },
    }


@pytest.fixture
def job_request(request_data: dict[str, object]) -> admission.Request:
    work = Path(str(request_data["workDirectory"]))
    return write_request(work / "job_request.json", request_data)


def test_readonly_check_records_reservations_and_does_not_write(
    job_request: admission.Request,
) -> None:
    result = admission.check(job_request, host=FakeHost(job_request))
    assert result.admitted is True
    assert result.reasons == ()
    assert result.evidence["planningAvailableBytes"] == 8192
    assert result.evidence["requiredBytes"] == 450
    assert result.evidence["familyRemainingBytes"] == 200
    assert result.evidence["diskRequiredBytes"] == 50
    assert not (job_request.work / "future-stage.lease").exists()


def test_inactive_credit_is_explicit_and_does_not_override_pressure(
    job_request: admission.Request,
) -> None:
    configured = replace(
        job_request, policy=replace(job_request.policy, inactive_credit=Fraction(1, 2))
    )
    observed = replace(snapshot(job_request), free_pages=0, pressure=2)
    result = admission.assess(configured, observed, now_ns=1_000_000_000)
    assert result.admitted is False
    assert result.reasons == ("pressure",)
    assert result.evidence["inactiveCreditedBytes"] == 6144
    assert result.evidence["inactiveCreditFraction"] == "1/2"


def test_no_inactive_credit_default_does_not_dispatch(job_request: admission.Request) -> None:
    host = FakeHost(job_request)
    host.snapshots = [replace(snapshot(job_request), free_pages=0)]
    assert admission.run(job_request, host=host) == 75
    assert host.calls == 0
    assert host.observations == 3
    assert host.clock == 12.0
    end = json.loads((job_request.work / "future-stage.end.json").read_text(encoding="utf-8"))
    assert end["dispatchAttempted"] is False
    assert end["reservationReleased"] is True
    assert end["exitCode"] == 75


def test_wait_admits_once_after_actual_headroom_improves(job_request: admission.Request) -> None:
    host = FakeHost(job_request)
    host.snapshots = [replace(snapshot(job_request), free_pages=0), snapshot(job_request)]
    assert admission.run(job_request, host=host) == 0
    assert host.calls == 1
    assert host.clock == 11.0
    with pytest.raises(FileExistsError):
        admission.run(job_request, host=host)
    assert host.calls == 1
    end = json.loads((job_request.work / "future-stage.end.json").read_text(encoding="utf-8"))
    assert end["completed"] is True
    assert end["terminalResources"]["freePages"] == 2
    assert end["reservationReleased"] is True


def test_command_failure_retains_exit_and_releases_kernel_lease(
    job_request: admission.Request,
) -> None:
    host = FakeHost(job_request)
    host.code = 7
    assert admission.run(job_request, host=host) == 7
    assert json.loads(
        (job_request.work / "future-stage.exit.json").read_text(encoding="utf-8")
    ) == {"exitCode": 7}
    end = json.loads((job_request.work / "future-stage.end.json").read_text(encoding="utf-8"))
    assert end["completed"] is False
    assert end["reservationReleased"] is True
    with (job_request.work / "future-stage.lease").open("rb") as stream:
        fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    assert host.calls == 1


def test_uncertain_launch_is_not_repeated_or_declared_released(
    job_request: admission.Request,
) -> None:
    host = FakeHost(job_request)
    host.error = OSError("possibly launched; confidential text")
    with pytest.raises(ReplayError, match="retained evidence"):
        admission.run(job_request, host=host)
    end = json.loads((job_request.work / "future-stage.end.json").read_text(encoding="utf-8"))
    assert end["uncertainLaunch"] is True
    assert end["reservationReleased"] is False
    assert end["errorKind"] == "OSError"
    assert "confidential" not in (job_request.work / "future-stage.end.json").read_text(
        encoding="utf-8"
    )
    with pytest.raises(FileExistsError):
        admission.run(job_request, host=host)
    assert host.calls == 1


def test_changed_request_during_wait_never_dispatches(job_request: admission.Request) -> None:
    host = FakeHost(job_request)
    host.change_request = True
    with pytest.raises(ReplayError, match="retained evidence"):
        admission.run(job_request, host=host)
    assert host.calls == 0
    assert (job_request.work / "future-stage.end.json").exists()


def test_in_memory_policy_cannot_bypass_pinned_request(job_request: admission.Request) -> None:
    altered = replace(job_request, policy=replace(job_request.policy, new_memory_bytes=1))
    with pytest.raises(ReplayError, match="in-memory policy changed"):
        admission.check(altered, host=FakeHost(job_request))
    assert not (job_request.work / "future-stage.lease").exists()


@pytest.mark.parametrize("field", ["birth", "executable", "command_sha256", "pid"])
def test_parent_identity_change_rejected(job_request: admission.Request, field: str) -> None:
    original = job_request.families[0].parent
    if field == "birth":
        parent = replace(original, birth="101:25")
    elif field == "executable":
        parent = replace(original, executable=Path("/wrong"))
    elif field == "command_sha256":
        parent = replace(original, command_sha256="b" * 64)
    else:
        parent = replace(original, pid=12)
    observed = replace(snapshot(job_request), families=(admission.FamilyUsage(parent, (12,), 100),))
    with pytest.raises(ReplayError, match="parent identity"):
        admission.assess(job_request, observed, now_ns=1_000_000_000)


def test_swap_growth_denies_launch_even_when_memory_recovers(
    job_request: admission.Request,
) -> None:
    host = FakeHost(job_request)
    host.snapshots = [
        replace(snapshot(job_request), free_pages=0),
        replace(snapshot(job_request), swap_bytes=121),
    ]
    assert admission.run(job_request, host=host) == 75
    assert host.calls == 0
    last = json.loads(
        (job_request.work / "future-stage.observation-0002.json").read_text(encoding="utf-8")
    )
    assert last["reasons"] == ["swap-growth"]


def test_absolute_swap_and_disk_floor_independent(job_request: admission.Request) -> None:
    observed = snapshot(job_request)
    disk = replace(observed.disks[0], free=30 * 1024**3)
    observed = replace(observed, swap_bytes=501, disks=(disk,))
    assert admission.assess(job_request, observed, now_ns=1_000_000_000).reasons == (
        "swap-limit",
        "disk-deficit",
    )


@pytest.mark.parametrize("observed_ns", [-5_000_000_001, 1_000_000_001])
def test_stale_or_future_snapshot_rejected(
    job_request: admission.Request, observed_ns: int
) -> None:
    with pytest.raises(ReplayError, match="stale or from the future"):
        admission.assess(
            job_request,
            replace(snapshot(job_request), observed_ns=observed_ns),
            now_ns=1_000_000_000,
        )


def test_peak_below_current_rss_reserves_zero_remaining(job_request: admission.Request) -> None:
    observed = replace(
        snapshot(job_request),
        families=(admission.FamilyUsage(job_request.families[0].parent, (10,), 999),),
    )
    result = admission.assess(job_request, observed, now_ns=1_000_000_000)
    assert result.evidence["familyRemainingBytes"] == 0
    assert result.evidence["requiredBytes"] == 250


@pytest.mark.parametrize(
    "mutation", ["numbers", "pressure", "disk-scope", "disk-floor", "no-parent", "overlap"]
)
def test_incomplete_observation_rejected(job_request: admission.Request, mutation: str) -> None:
    observed = snapshot(job_request)
    if mutation == "numbers":
        observed = replace(observed, free_pages=-1)
    elif mutation == "pressure":
        observed = replace(observed, pressure=3)
    elif mutation == "disk-scope":
        observed = replace(observed, disks=())
    elif mutation == "disk-floor":
        observed = replace(observed, disks=(replace(observed.disks[0], floor=1),))
    elif mutation == "no-parent":
        observed = replace(
            observed, families=(admission.FamilyUsage(job_request.families[0].parent, (), 100),)
        )
    else:
        observed = replace(
            observed,
            families=(admission.FamilyUsage(job_request.families[0].parent, (10, 10), 100),),
        )
    with pytest.raises(ReplayError):
        admission.assess(job_request, observed, now_ns=1_000_000_000)


def test_tree_uses_parent_and_all_descendants_only(job_request: admission.Request) -> None:
    rows = admission.parse_processes("10 1 1\n11 10 2\n12 11 3\n13 1 99\n")
    result = admission.family_usage(job_request.families, rows)
    assert result[0].pids == (10, 11, 12)
    assert result[0].rss_bytes == 6144


def test_overlapping_family_and_missing_parent_rejected(job_request: admission.Request) -> None:
    other = replace(job_request.families[0], parent=replace(job_request.families[0].parent, pid=11))
    with pytest.raises(ReplayError, match="Overlapping"):
        admission.family_usage(
            (job_request.families[0], other), admission.parse_processes("10 1 1\n11 10 2\n")
        )
    with pytest.raises(ReplayError, match="missing"):
        admission.family_usage(job_request.families, admission.parse_processes("11 1 1\n"))


def test_cycle_in_owned_tree_rejected(job_request: admission.Request) -> None:
    with pytest.raises(ReplayError, match="Cycle"):
        admission.family_usage(
            job_request.families, admission.parse_processes("10 11 1\n11 10 2\n")
        )


@pytest.mark.parametrize("text", ["", "1 0 x", "1 0 1\n1 0 2", "1 1 1", "0 1 1", "1 0 1 extra"])
def test_invalid_ps_rows(text: str) -> None:
    with pytest.raises(ReplayError):
        admission.parse_processes(text)


def test_process_budget(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(admission, "MAX_PROCESSES", 1)
    with pytest.raises(ReplayError, match="row budget"):
        admission.parse_processes("1 0 1\n2 1 2")


def test_measured_page_size_and_decimal_swap_are_not_guessed() -> None:
    assert admission.parse_vm_stat(
        "Mach Virtual Memory Statistics: (page size of 16384 bytes)\n"
        "Pages free: 4.\nPages inactive: 9.\n"
    ) == (16384, 4, 9)
    assert (
        admission.parse_swap("total = 8192.00M  used = 12.50M  free = 8179.50M (encrypted)\n")
        == 13107200
    )


@pytest.mark.parametrize(
    "text",
    [
        "unknown",
        "page size of 3 bytes\nPages free: 1.\nPages inactive: 1.",
        "page size of 4096 bytes\nPages free: 1.\nPages free: 2.\nPages inactive: 1.",
    ],
)
def test_invalid_vm_stat(text: str) -> None:
    with pytest.raises(ReplayError):
        admission.parse_vm_stat(text)


@pytest.mark.parametrize(
    "text", ["unknown", "total = 1M used = 2M free = 0M", "total = .M used = 1M free = 0M"]
)
def test_invalid_swap(text: str) -> None:
    with pytest.raises(ReplayError):
        admission.parse_swap(text)


@pytest.mark.parametrize("credit", [-1, 2, "nan", True, [], "1/0"])
def test_invalid_inactive_credit(request_data: dict[str, object], credit: object) -> None:
    policy = request_data["policy"]
    assert isinstance(policy, dict)
    policy["inactiveCreditFraction"] = credit
    path = Path(str(request_data["workDirectory"])) / "job_request.json"
    with pytest.raises(ReplayError, match="[Ii]nactive"):
        write_request(path, request_data)


@pytest.mark.parametrize("credit", [0, 1, 0.5, "1/4"])
def test_explicit_credit_loads(request_data: dict[str, object], credit: object) -> None:
    policy = request_data["policy"]
    assert isinstance(policy, dict)
    policy["inactiveCreditFraction"] = credit
    path = Path(str(request_data["workDirectory"])) / "job_request.json"
    assert write_request(path, request_data).policy.inactive_credit <= 1


@pytest.mark.parametrize(
    "mutation",
    [
        "schema",
        "kind",
        "name",
        "command",
        "families",
        "disks",
        "scratch",
        "bool",
        "pressure",
        "polls",
    ],
)
def test_invalid_request_does_not_dispatch(request_data: dict[str, object], mutation: str) -> None:
    if mutation == "schema":
        request_data["unknown"] = 1
    elif mutation == "kind":
        request_data["kind"] = "wrong"
    elif mutation == "name":
        request_data["name"] = "../bad"
    elif mutation == "command":
        request_data["command"] = ["/wrong"]
    elif mutation == "families":
        request_data["families"] = "wrong"
    elif mutation == "disks":
        request_data["diskPaths"] = []
    elif mutation == "scratch":
        request_data["scratchDirectory"] = request_data["workDirectory"]
    else:
        policy = request_data["policy"]
        assert isinstance(policy, dict)
        if mutation == "bool":
            policy["newMemoryBytes"] = True
        elif mutation == "pressure":
            policy["maximumPressure"] = 3
        else:
            policy["waitSeconds"] = 86400
    with pytest.raises(ReplayError):
        write_request(Path(str(request_data["workDirectory"])) / "job_request.json", request_data)


def test_duplicate_json_key_is_rejected(job_request: admission.Request) -> None:
    data = b'{"name":"first","name":"second"}'
    job_request.path.write_bytes(data)
    with pytest.raises(ReplayError, match="Duplicate"):
        admission.load(job_request.path, hashlib.sha256(data).hexdigest())


def test_request_symlink_and_hardlink_are_rejected(job_request: admission.Request) -> None:
    link = job_request.work / "alias.json"
    link.symlink_to(job_request.path)
    with pytest.raises(ReplayError, match="canonical"):
        admission.load(link, job_request.sha256)
    link.unlink()
    os.link(job_request.path, link)
    with pytest.raises(ReplayError, match="single-link"):
        admission.load(job_request.path, job_request.sha256)


def test_escaping_scratch_link_rejected(request_data: dict[str, object], tmp_path: Path) -> None:
    work = Path(str(request_data["workDirectory"]))
    (work / "linked").symlink_to(tmp_path)
    request_data["scratchDirectory"] = str(work / "linked")
    with pytest.raises(ReplayError, match="canonical"):
        write_request(work / "job_request.json", request_data)


def test_explicit_executable_alias_is_bound_and_mutation_rejected(
    request_data: dict[str, object],
) -> None:
    raw = request_data["executable"]
    assert isinstance(raw, dict)
    tool = Path(str(raw["path"]))
    alias = tool.parent / "python-alias"
    alias.symlink_to(tool.name)
    raw["invocationPath"] = str(alias)
    raw["linkTarget"] = tool.name
    request_data["command"] = [str(alias), "-m", "replay.workflow"]
    job_request = write_request(
        Path(str(request_data["workDirectory"])) / "job_request.json", request_data
    )
    alias.unlink()
    alias.symlink_to("missing")
    with pytest.raises(ReplayError, match="alias changed"):
        admission.check(job_request, host=FakeHost(job_request))


def test_tool_bytes_changed_rejected_before_lease(job_request: admission.Request) -> None:
    job_request.executable.path.write_bytes(b"changed")
    with pytest.raises(ValueError, match="changed"):
        admission.run(job_request, host=FakeHost(job_request))
    assert not (job_request.work / "future-stage.lease").exists()


def test_occupied_start_record_is_preserved_no_dispatch(job_request: admission.Request) -> None:
    path = job_request.work / "future-stage.start.json"
    path.write_bytes(b"preserved")
    host = FakeHost(job_request)
    with pytest.raises(FileExistsError):
        admission.run(job_request, host=host)
    assert path.read_bytes() == b"preserved"
    assert host.calls == 0


def test_system_probes_are_bounded_and_keep_raw_args_out_of_errors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def failed(command: list[str], **_kwargs: object) -> subprocess.CompletedProcess[bytes]:
        return subprocess.CompletedProcess(command, 7, b"secret", b"private")

    monkeypatch.setattr(subprocess, "run", failed)
    with pytest.raises(ReplayError, match="Bounded OS resource query failed"):
        admission.probe(["/bin/ps"])
    monkeypatch.setattr(
        subprocess,
        "run",
        lambda command, **_kwargs: subprocess.CompletedProcess(command, 0, b"valid", b""),
    )
    assert admission.probe(["/bin/ps"]) == "valid"
    monkeypatch.setattr(admission, "MAX_PROBE_BYTES", 1)
    with pytest.raises(ReplayError):
        admission.probe(["/bin/ps"])


def test_system_execute_reuses_timer_and_child_local_environment(
    job_request: admission.Request, monkeypatch: pytest.MonkeyPatch
) -> None:
    observed: dict[str, object] = {}

    def execute(command: list[str], **kwargs: object) -> subprocess.CompletedProcess[bytes]:
        observed["command"] = command
        observed.update(kwargs)
        return subprocess.CompletedProcess(command, 9)

    monkeypatch.setattr(subprocess, "run", execute)
    monkeypatch.setenv("TMPDIR", "original-parent-value")
    assert admission.SystemHost().execute(job_request) == 9
    command = observed["command"]
    assert isinstance(command, list)
    assert command[1:4] == ["-B", "-m", "replay.phase_timer"]
    environment = observed["env"]
    assert isinstance(environment, dict)
    assert environment["TMPDIR"] == str(job_request.scratch)
    assert os.environ["TMPDIR"] == "original-parent-value"
    assert observed["stderr"] == subprocess.STDOUT
    assert (job_request.work / "future-stage.child.log").stat().st_mode & 0o777 == 0o600
    with pytest.raises(FileExistsError):
        admission.SystemHost().execute(job_request)


def test_main_readonly_and_error_are_short(
    job_request: admission.Request,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    host = FakeHost(job_request)
    monkeypatch.setattr(admission, "SystemHost", lambda: host)
    assert admission.main(["check", str(job_request.path), "--sha256", job_request.sha256]) == 0
    assert json.loads(capsys.readouterr().out)["admitted"] is True
    assert admission.main(["check", str(job_request.path), "--sha256", "invalid"]) == 1
    assert (
        capsys.readouterr().err == "MEASURED_ADMISSION_ERROR retained evidence must be reconciled\n"
    )


def test_main_run_and_refused_check(
    job_request: admission.Request, monkeypatch: pytest.MonkeyPatch
) -> None:
    host = FakeHost(job_request)
    monkeypatch.setattr(admission, "SystemHost", lambda: host)
    host.snapshots = [replace(snapshot(job_request), free_pages=0)]
    assert admission.main(["check", str(job_request.path), "--sha256", job_request.sha256]) == 75
    host.snapshots = [snapshot(job_request)]
    assert admission.main(["run", str(job_request.path), "--sha256", job_request.sha256]) == 0
    assert host.calls == 1


def test_system_observe_binds_kernel_path_birth_command_and_all_resources(
    job_request: admission.Request, monkeypatch: pytest.MonkeyPatch
) -> None:
    command_text = "trusted configured parent\n"
    parent = replace(
        job_request.families[0].parent,
        command_sha256=hashlib.sha256(command_text.encode()).hexdigest(),
    )
    planned = replace(job_request, families=(replace(job_request.families[0], parent=parent),))
    fixture = NativeFixture()
    fixture.births = ["100:25"]
    monkeypatch.setattr(admission, "NativeHost", lambda: fixture)
    monkeypatch.setattr(admission, "process_executable", lambda _pid: job_request.executable.path)

    def output(command: list[str]) -> str:
        if command[-1] == "command=":
            return command_text
        if command[0] == "/usr/bin/vm_stat":
            return "page size of 16384 bytes\nPages free: 2.\nPages inactive: 3.\n"
        if command[0] == "/usr/sbin/sysctl":
            return "total = 100M used = 20M free = 80M"
        return "10 1 1\n11 10 2\n"

    monkeypatch.setattr(admission, "probe", output)
    monkeypatch.setattr(
        disk_budget,
        "measure",
        lambda path: disk_budget.Headroom(path, 100 * 1024**3, 90 * 1024**3, 30 * 1024**3, 0),
    )
    observed = admission.SystemHost().observe(planned)
    assert observed.page_size == 16384
    assert observed.free_pages == 2
    assert observed.inactive_pages == 3
    assert observed.swap_bytes == 20971520
    assert observed.families[0].rss_bytes == 3072
    assert (
        observed.families[0].parent.command_sha256
        == hashlib.sha256(b"trusted configured parent\n").hexdigest()
    )


def test_system_parent_identity_race_and_empty_command_fail(
    job_request: admission.Request, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = NativeFixture()
    fixture.births = ["100:25", "101:25"]
    monkeypatch.setattr(admission, "NativeHost", lambda: fixture)
    monkeypatch.setattr(admission, "process_executable", lambda _pid: job_request.executable.path)
    monkeypatch.setattr(admission, "probe", lambda _command: "parent\n")
    with pytest.raises(ReplayError, match="changed during identity"):
        admission.SystemHost().parent(job_request.families[0].parent)
    monkeypatch.setattr(admission, "probe", lambda _command: "")
    with pytest.raises(ReplayError, match="disappeared"):
        admission.SystemHost().parent(job_request.families[0].parent)


def test_kernel_executable_path_is_bounded_not_argv0(
    job_request: admission.Request, monkeypatch: pytest.MonkeyPatch
) -> None:
    function = PathFunction(os.fsencode(job_request.executable.path))
    monkeypatch.setattr(ctypes, "CDLL", lambda *_args, **_kwargs: PathLibrary(function))
    assert admission.process_executable(10) == job_request.executable.path
    function.count = 0
    with pytest.raises(ReplayError, match="unavailable"):
        admission.process_executable(10)
    monkeypatch.setattr(sys, "platform", "unsupported")
    with pytest.raises(ReplayError, match="macOS"):
        admission.process_executable(10)


def test_parent_directory_replacement_detected(
    job_request: admission.Request,
) -> None:
    with admission._lease(job_request) as lease:
        retained = job_request.work.with_name("retained-old-work")
        job_request.work.rename(retained)
        job_request.work.mkdir()
        with pytest.raises(ReplayError, match="work directory changed"):
            admission._parent_guard(lease.parent, job_request, lease.parent_identity)


def test_maximum_poll_budget_and_record_budget_are_finite(
    job_request: admission.Request, monkeypatch: pytest.MonkeyPatch
) -> None:
    host = FakeHost(job_request)
    host.snapshots = [replace(snapshot(job_request), free_pages=0)]
    monkeypatch.setattr(admission, "MAX_POLLS", 4)
    monkeypatch.setattr(host, "sleep", lambda _seconds: None)
    with pytest.raises(ReplayError, match="retained evidence"):
        admission.run(job_request, host=host)
    assert host.calls == 0
    monkeypatch.setattr(admission, "MAX_OBSERVATION_BYTES", 1)
    with (
        admission._lease(replace(job_request, name="other-stage")) as lease,
        pytest.raises(ReplayError, match="byte budget"),
    ):
        admission._save(lease.parent, "too-big.json", {"key": "value"})


def test_lease_name_replacement_is_detected_before_effect(
    job_request: admission.Request,
) -> None:
    with admission._lease(job_request) as lease:
        path = job_request.work / "future-stage.lease"
        path.rename(job_request.work / "retained-old-lease")
        path.write_bytes(b"replacement")
        path.chmod(0o600)
        with pytest.raises(ReplayError, match="lease changed"):
            admission._lease_guard(lease, job_request)


def test_slow_measurement_cannot_launch_after_deadline(
    job_request: admission.Request, monkeypatch: pytest.MonkeyPatch
) -> None:
    host = FakeHost(job_request)
    original = host.observe

    def observe(planned: admission.Request) -> admission.Snapshot:
        host.clock += 3.0
        return original(planned)

    monkeypatch.setattr(host, "observe", observe)
    assert admission.run(job_request, host=host) == 75
    assert host.calls == 0


class CaptureFixture(admission.SystemHost):
    def __init__(self, parent: admission.Parent) -> None:
        self.actor = parent
        self.changed = False
        self.calls = 0

    def now_ns(self) -> int:
        return 1_000_000_000

    def identity(self, _pid: int) -> admission.Parent:
        self.calls += 1
        if self.changed and self.calls > 1:
            return replace(self.actor, birth="101:1")
        return self.actor


def completion_files(job_request: admission.Request) -> tuple[Path, str, Path, Path, Path]:
    state = job_request.work / "existing-state"
    state.mkdir()
    workflow = job_request.work / "existing-workflow.json"
    data = json.dumps(
        {
            "version": 1,
            "stages": [
                {
                    "name": "owned-model-stage",
                    "command": ["trusted-model"],
                    "inputs": [str(job_request.executable.path)],
                    "outputs": [str(job_request.work / "result.json")],
                    "timeout_seconds": 30,
                    "verify_command": ["existing-domain-verifier"],
                }
            ],
        }
    ).encode()
    workflow.write_bytes(data)
    checkpoint = state / "owned-model-stage.json"
    checkpoint.write_text(json.dumps({"status": "running", "fingerprint": "b" * 64}))
    return (
        workflow,
        hashlib.sha256(data).hexdigest(),
        state,
        job_request.work / "owned-model.status",
        checkpoint,
    )


def capture_probe(command: list[str]) -> str:
    if command[-1] == "pid=,pgid=":
        return "1 1\n10 10\n11 11\n"
    return "1 0 1\n10 1 2\n11 10 3\n"


@pytest.fixture
def completion_request(
    job_request: admission.Request, monkeypatch: pytest.MonkeyPatch
) -> admission.Request:
    workflow, digest, state, status, _checkpoint = completion_files(job_request)
    output = job_request.work / "actor-capture.json"
    monkeypatch.setattr(admission, "probe", capture_probe)
    admission.capture(
        10,
        workflow,
        digest,
        state,
        "owned-model-stage",
        status,
        output,
        host=CaptureFixture(job_request.families[0].parent),
    )
    raw = record(json.loads(job_request.path.read_bytes()))
    families = raw["families"]
    assert isinstance(families, list)
    owner = record(families[0])
    owner["completion"] = {
        "capturePath": str(output),
        "sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
    }
    families[0] = owner
    return write_request(job_request.path, raw)


def complete_actor(job_request: admission.Request, *, code: int = 0) -> admission.Completion:
    completion = job_request.families[0].completion
    assert completion is not None
    completion.status.write_text(f"{code}\n")
    if code == 0:
        completion.checkpoint.write_text(
            json.dumps(
                {
                    "status": "complete",
                    "fingerprint": "b" * 64,
                    "outputs": {completion.outputs[0]: "c" * 64},
                }
            )
        )
    return completion


def absent_probe(command: list[str]) -> str:
    if command[-1] == "pid=,pgid=":
        return "1 1\n"
    if command[0] == "/usr/bin/vm_stat":
        return "page size of 16384 bytes\nPages free: 2.\nPages inactive: 3.\n"
    if command[0] == "/usr/sbin/sysctl":
        return "total = 100M used = 20M free = 80M"
    return "1 0 1\n"


def test_live_capture_records_real_identity_and_all_descendant_groups(
    completion_request: admission.Request,
) -> None:
    completion = completion_request.families[0].completion
    assert completion is not None
    parent, loaded = admission.load_completion_capture(completion.capture, completion.sha256)
    assert parent == completion_request.families[0].parent
    assert loaded == completion
    assert completion.process_groups == (10, 11)
    assert admission.terminal_evidence(completion) == ()
    assert "command" not in record(json.loads(completion.capture.read_bytes()))


def test_genuine_terminal_releases_resources_without_approving_quality(
    completion_request: admission.Request, monkeypatch: pytest.MonkeyPatch
) -> None:
    completion = complete_actor(completion_request)
    monkeypatch.setattr(admission, "probe", absent_probe)
    result = admission.verify_completion_capture(completion.capture, completion.sha256)
    assert result["resourceReleased"] is True
    assert result["stageQualityApproved"] is False
    assert len(record(result["terminalEvidence"])) == 2


def test_failed_parent_with_rotating_orphan_cannot_release_from_initial_group_absence(
    completion_request: admission.Request, monkeypatch: pytest.MonkeyPatch
) -> None:
    completion = complete_actor(completion_request, code=7)
    monkeypatch.setattr(admission, "probe", lambda _command: "1 1\n55 55\n")
    with pytest.raises(ReplayError, match="current writer reconciliation"):
        admission.verify_completion_capture(completion.capture, completion.sha256)


def test_absence_alone_does_not_release_and_retained_checkpoint_stays_unchanged(
    completion_request: admission.Request, monkeypatch: pytest.MonkeyPatch
) -> None:
    completion = completion_request.families[0].completion
    assert completion is not None
    original = completion.checkpoint.read_bytes()
    monkeypatch.setattr(admission, "probe", absent_probe)
    with pytest.raises(ReplayError, match="genuine terminal"):
        admission.verify_completion_capture(completion.capture, completion.sha256)
    assert completion.checkpoint.read_bytes() == original


@pytest.mark.parametrize("groups", ["1 1\n10 10\n", "1 1\n55 11\n"])
def test_parent_exit_does_not_release_a_live_child_group(
    completion_request: admission.Request, monkeypatch: pytest.MonkeyPatch, groups: str
) -> None:
    completion = complete_actor(completion_request)
    monkeypatch.setattr(admission, "probe", lambda _command: groups)
    with pytest.raises(ReplayError, match="still present"):
        admission.verify_completion_capture(completion.capture, completion.sha256)


def test_reused_parent_pid_outside_old_groups_is_not_counted_as_owned(
    completion_request: admission.Request, monkeypatch: pytest.MonkeyPatch
) -> None:
    completion = complete_actor(completion_request)
    monkeypatch.setattr(admission, "probe", lambda _command: "1 1\n10 77\n")
    assert (
        admission.verify_completion_capture(completion.capture, completion.sha256)[
            "resourceReleased"
        ]
        is True
    )


@pytest.mark.parametrize(
    "failure", ["status", "large-status", "fingerprint", "running", "outputs", "digest"]
)
def test_terminal_mutations_and_false_success_are_rejected(
    completion_request: admission.Request, failure: str
) -> None:
    completion = complete_actor(completion_request)
    raw = record(json.loads(completion.checkpoint.read_bytes()))
    if failure == "status":
        completion.status.write_text("999\n")
    elif failure == "large-status":
        completion.status.write_text("0" * 40)
    else:
        raw[
            "fingerprint"
            if failure == "fingerprint"
            else "status"
            if failure == "running"
            else "outputs"
        ] = (
            "a" * 64
            if failure == "fingerprint"
            else "running"
            if failure == "running"
            else {"unrelated": "c" * 64}
            if failure == "outputs"
            else {completion.outputs[0]: "invalid"}
        )
        completion.checkpoint.write_text(json.dumps(raw))
    with pytest.raises(ReplayError):
        admission.terminal_evidence(completion)


def test_system_absent_pending_then_complete_admits_without_missing_parent_error(
    completion_request: admission.Request, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(admission, "probe", absent_probe)
    monkeypatch.setattr(admission, "NativeHost", NativeFixture)
    monkeypatch.setattr(
        disk_budget,
        "measure",
        lambda path: disk_budget.Headroom(path, 100 * 1024**3, 90 * 1024**3, 30 * 1024**3, 0),
    )
    host = admission.SystemHost()
    pending = host.observe(completion_request)
    assert pending.families[0].state == "terminal-pending"
    before = admission.assess(completion_request, pending, now_ns=host.now_ns())
    assert before.admitted is False
    assert before.evidence["familyRemainingBytes"] == 300
    assert "family-terminal-pending" in before.reasons
    complete_actor(completion_request)
    released = host.observe(completion_request)
    after = admission.assess(completion_request, released, now_ns=host.now_ns())
    assert after.admitted is False  # Actual fake swap remains above this tiny request's limit.
    assert released.families[0].state == "released"
    assert after.evidence["familyRemainingBytes"] == 0
    assert after.evidence["diskRequiredBytes"] == 30
    assert "family-terminal-pending" not in after.reasons


def test_release_evidence_race_and_group_reappearance_are_rejected(
    completion_request: admission.Request, monkeypatch: pytest.MonkeyPatch
) -> None:
    completion = complete_actor(completion_request)
    calls = 0

    def observe(_command: list[str]) -> str:
        nonlocal calls
        calls += 1
        if calls == 2:
            completion.checkpoint.write_text(
                json.dumps(
                    {
                        "status": "complete",
                        "fingerprint": "b" * 64,
                        "outputs": {completion.outputs[0]: "d" * 64},
                    }
                )
            )
        return "1 1\n"

    monkeypatch.setattr(admission, "probe", observe)
    with pytest.raises(ReplayError, match="changed during"):
        admission.verify_completion_capture(completion.capture, completion.sha256)
    calls = 0
    monkeypatch.setattr(admission, "probe", lambda _command: "1 1\n55 11\n")
    with pytest.raises(ReplayError):
        admission.verify_completion_capture(completion.capture, completion.sha256)


@pytest.mark.parametrize("text", ["", "1 1\n1 2\n", "a 1\n", "1\n", "0 1\n"])
def test_group_schema_is_bounded_and_unambiguous(text: str) -> None:
    with pytest.raises(ReplayError):
        admission.parse_groups(text)


def test_capture_cli_and_collision_preserve_original_actor_record(
    job_request: admission.Request, monkeypatch: pytest.MonkeyPatch
) -> None:
    workflow, digest, state, status, _ = completion_files(job_request)
    output = job_request.work / "capture.json"
    monkeypatch.setattr(admission, "probe", capture_probe)
    monkeypatch.setattr(
        admission, "SystemHost", lambda: CaptureFixture(job_request.families[0].parent)
    )
    argv = [
        "capture",
        str(workflow),
        "--sha256",
        digest,
        "--pid",
        "10",
        "--state",
        str(state),
        "--stage",
        "owned-model-stage",
        "--status",
        str(status),
        "--output",
        str(output),
    ]
    assert admission.main(argv) == 0
    saved = output.read_bytes()
    assert admission.main(argv) == 1
    assert output.read_bytes() == saved
    assert admission.main(["capture", str(workflow), "--sha256", digest]) == 1


def test_capture_actor_identity_change_does_not_publish(
    job_request: admission.Request, monkeypatch: pytest.MonkeyPatch
) -> None:
    workflow, digest, state, status, _ = completion_files(job_request)
    output = job_request.work / "capture.json"
    host = CaptureFixture(job_request.families[0].parent)
    host.changed = True
    monkeypatch.setattr(admission, "probe", capture_probe)
    with pytest.raises(ReplayError, match="changed during capture"):
        admission.capture(
            10, workflow, digest, state, "owned-model-stage", status, output, host=host
        )
    assert not output.exists()


def test_capture_output_ancestor_replacement_is_not_accepted(
    job_request: admission.Request, monkeypatch: pytest.MonkeyPatch
) -> None:
    workflow, digest, state, status, _ = completion_files(job_request)
    destination = job_request.work / "capture-output"
    destination.mkdir()
    output = destination / "capture.json"

    def changed(command: list[str]) -> str:
        if command[-1] == "pid=,pgid=":
            destination.rename(job_request.work / "old-capture-output")
            destination.mkdir()
        return capture_probe(command)

    monkeypatch.setattr(admission, "probe", changed)
    with pytest.raises(ReplayError, match="ancestor changed"):
        admission.capture(
            10,
            workflow,
            digest,
            state,
            "owned-model-stage",
            status,
            output,
            host=CaptureFixture(job_request.families[0].parent),
        )
    assert not output.exists()
