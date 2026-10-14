"""Synthetic cooperative boundary handoff; no real jobs or media are controlled."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack
from dataclasses import replace
from pathlib import Path

import pytest

from replay import stages, workflow
from replay import workflow_stage_reservation as reservation
from replay.compressor_helper_guard import processes
from replay.core import ReplayError
from replay.workflow_stage_reservation import ProcessIdentity as Process


@pytest.fixture
def request_path(tmp_path: Path) -> Path:
    source = tmp_path / "source.txt"
    source.write_bytes(b"source")
    state = tmp_path / "state"
    state.mkdir()
    (state / ".workflow.lock").write_bytes(b"")
    (state / "current.lock").write_bytes(b"")
    (tmp_path / "job.launch").mkdir()
    (tmp_path / "job.log").write_bytes(b"initial\n")
    entries = [
        {
            "name": "current",
            "command": ["work"],
            "verify_command": ["verify"],
            "inputs": [str(source)],
            "outputs": [str(tmp_path / "current.txt")],
            "timeout_seconds": 10,
        },
        {
            "name": "future",
            "command": ["future-work"],
            "verify_command": ["future-verify"],
            "inputs": [str(tmp_path / "current.txt")],
            "outputs": [str(tmp_path / "future.txt")],
            "timeout_seconds": 10,
        },
        {
            "name": "later",
            "command": ["later-work"],
            "verify_command": ["later-verify"],
            "inputs": [str(tmp_path / "future.txt")],
            "outputs": [str(tmp_path / "later.txt")],
            "timeout_seconds": 10,
        },
    ]
    plan = tmp_path / "workflow.json"
    plan.write_text(json.dumps({"version": 1, "stages": entries}), encoding="utf-8")
    current = workflow.load_stages(plan)[0]
    fingerprint = stages._fingerprint(current)
    (state / "current.json").write_text(
        json.dumps({"status": "running", "fingerprint": fingerprint}), encoding="utf-8"
    )
    manifest = tmp_path / "reservation.json"
    manifest.write_text(
        json.dumps(
            {
                "version": 1,
                "workflow": {"path": str(plan), "sha256": stages.file_digest(plan)},
                "state": str(state),
                "job": {
                    "root": str(tmp_path),
                    "name": "job",
                    "pid": 111,
                    "ppid": 100,
                    "uid": os.getuid(),
                    "start": "Sun Oct 11 00:00:00 2026",
                    "executable": "/usr/bin/python3",
                },
                "current": {"stage": "current", "fingerprint": fingerprint},
                "boundary": "future",
                "transferStages": ["future", "later"],
                "timeoutSeconds": 3,
            }
        ),
        encoding="utf-8",
    )
    return manifest


@pytest.fixture
def lease_request(request_path: Path) -> reservation.Request:
    return reservation.load_request(request_path)


@pytest.fixture
def fake_observation(monkeypatch: pytest.MonkeyPatch, lease_request: reservation.Request) -> None:
    calls = 0

    def observed(_directory: Path, *, deadline: float) -> list[Process]:
        nonlocal calls
        calls += 1
        if calls == 1:
            return [
                lease_request.owner,
                Process(112, 111, os.getuid(), "child", "/usr/bin/worker"),
            ]
        if calls > 2:
            return []
        lease_request.current.outputs[0].write_bytes(b"completed")

        def verifier(_command: tuple[str, ...], _log: Path, _timeout: int) -> None:
            return None

        stages.run_stage(
            lease_request.current, state_directory=lease_request.state, execute=verifier
        )
        lease_request.status.write_text("1\n", encoding="utf-8")
        with lease_request.log.open("a", encoding="utf-8") as log:
            log.write("current: completed\nfuture: busy\n")
        return []

    monkeypatch.setattr(reservation, "_snapshot", observed)
    monkeypatch.setattr(reservation, "_lock_owner", lambda _path, _owner, _deadline: None)
    monkeypatch.setattr(
        reservation, "_lsof", lambda _path, **_kwargs: f"p111\nn{lease_request.log}\n"
    )


def test_load_binds_exact_future_suffix_and_job(lease_request: reservation.Request) -> None:
    assert lease_request.boundary == "future"
    assert [stage.name for stage in lease_request.transfer] == ["future", "later"]
    assert lease_request.current.name == "current"
    assert lease_request.owner.pid == 111
    assert lease_request.timeout_seconds == 3
    assert not lease_request.current.outputs[0].exists()


def test_handoff_keeps_old_failure_and_releases_leases(
    lease_request: reservation.Request, fake_observation: None, tmp_path: Path
) -> None:
    proof = reservation.reserve(lease_request, proof=tmp_path / "handoff.json")
    result = json.loads(proof.read_text(encoding="utf-8"))
    assert result["status"] == "verified-boundary-handoff"
    assert result["terminal"]["exitCode"] == 1
    assert result["terminal"]["reservedStageAttempted"] is True
    assert result["terminal"]["reservedStageProducerStarted"] is False
    assert result["precedingStage"]["status"] == "complete"
    assert result["mediaApproved"] is False
    assert result["oldWorkflowComplete"] is False
    assert result["activeOwnedProcesses"] == []
    assert lease_request.status.read_text(encoding="utf-8") == "1\n"
    assert not (tmp_path / "future.txt").exists()
    assert not (lease_request.state / "future.json").exists()
    assert (lease_request.state / "future.lock").is_file()
    assert proof.stat().st_mode & 0o777 == 0o600
    with stages.execution_lock(lease_request.state, "future") as acquired:
        assert acquired is True
    with stages.execution_lock(lease_request.state, ".workflow") as acquired:
        assert acquired is True


def test_busy_future_rejected_without_changing_current_receipt(
    lease_request: reservation.Request, tmp_path: Path
) -> None:
    before = (lease_request.state / "current.json").read_bytes()
    with (
        stages.execution_lock(lease_request.state, "future"),
        pytest.raises(ReplayError, match="already active or leased"),
    ):
        reservation.reserve(lease_request, proof=tmp_path / "handoff.json")
    assert (lease_request.state / "current.json").read_bytes() == before
    assert not (tmp_path / "handoff.json").exists()
    assert not lease_request.status.exists()


@pytest.mark.parametrize("artifact", ["future.txt", "later.txt", "state/future.json"])
def test_existing_transfer_artifact_is_never_adopted(
    lease_request: reservation.Request, tmp_path: Path, artifact: str
) -> None:
    (tmp_path / artifact).write_bytes(b"existing")
    with pytest.raises(ReplayError, match="already has a receipt or producer output"):
        reservation.reserve(lease_request, proof=tmp_path / "handoff.json")
    assert (tmp_path / artifact).read_bytes() == b"existing"
    with stages.execution_lock(lease_request.state, "future") as acquired:
        assert acquired is True


def test_future_attempt_rejected_even_without_output(lease_request: reservation.Request) -> None:
    (lease_request.state / "later-attempt").mkdir()
    with pytest.raises(ReplayError, match="execution attempt"):
        reservation._future_absent(lease_request)


def test_terminal_job_cannot_be_reserved(lease_request: reservation.Request) -> None:
    lease_request.status.write_text("1\n", encoding="utf-8")
    with pytest.raises(ReplayError, match="active job"):
        reservation._initial(lease_request, time.monotonic() + 3)


def test_current_complete_receipt_is_not_an_active_reservation(
    lease_request: reservation.Request,
) -> None:
    (lease_request.state / "current.json").write_text(
        json.dumps({"status": "complete", "fingerprint": lease_request.current_fingerprint}),
        encoding="utf-8",
    )
    with pytest.raises(ReplayError, match="expected active receipt"):
        reservation._initial(lease_request, time.monotonic() + 3)


def test_missing_or_changed_owner_is_rejected(
    lease_request: reservation.Request, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(reservation, "_snapshot", lambda _directory, **_kwargs: [])
    with pytest.raises(ReplayError, match="identity changed or disappeared"):
        reservation._initial(lease_request, time.monotonic() + 3)


def test_owner_parent_is_part_of_initial_identity(lease_request: reservation.Request) -> None:
    assert not reservation._same_process(replace(lease_request.owner, ppid=99), lease_request.owner)


def test_controller_log_must_be_open_by_exact_owner(
    lease_request: reservation.Request, fake_observation: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(reservation, "_lsof", lambda _path, **_kwargs: "p111\nn/other-log\n")
    with pytest.raises(ReplayError, match="does not write"):
        reservation._initial(lease_request, time.monotonic() + 3)


@pytest.mark.parametrize(
    ("body", "message"),
    [
        ("p111\nlW\nn{path}\n", None),
        ("p111\nl \nn{path}\n", None),
        ("p111\nlR\nn{path}\n", "does not exclusively own"),
        ("p112\nlW\nn{path}\n", "does not exclusively own"),
        ("p111\nlW\nn{path}\np112\nlW\nn{path}\n", "does not exclusively own"),
    ],
)
def test_exact_exclusive_lock_owner_fields(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, body: str, message: str | None
) -> None:
    path = tmp_path / "observed.lock"
    monkeypatch.setattr(reservation, "_lsof", lambda _path, **_kwargs: body.format(path=path))
    with stages.execution_lock(tmp_path, "observed"):
        if message is None:
            reservation._lock_owner(path, 111, time.monotonic() + 3)
        else:
            with pytest.raises(ReplayError, match=message):
                reservation._lock_owner(path, 111, time.monotonic() + 3)


def test_owner_open_fd_without_busy_kernel_lease_is_rejected(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    path = tmp_path / "observed.lock"
    path.write_bytes(b"")
    monkeypatch.setattr(reservation, "_lsof", lambda _path, **_kwargs: f"p111\nl \nn{path}\n")
    with pytest.raises(ReplayError, match="does not exclusively own"):
        reservation._lock_owner(path, 111, time.monotonic() + 3)


def test_scoped_lsof_uses_no_process_arguments(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    calls: list[list[str]] = []

    def run(command: list[str], **_kwargs: object) -> subprocess.CompletedProcess[str]:
        calls.append(command)
        return subprocess.CompletedProcess(command, 0, "p111\nlW\n", "")

    monkeypatch.setattr(subprocess, "run", run)
    assert (
        reservation._lsof(tmp_path / "lock", pid=111, deadline=time.monotonic() + 3) == "p111\nlW\n"
    )
    assert calls[0][:8] == ["/usr/sbin/lsof", "-n", "-P", "-a", "-p", "111", "-Fpln", "--"]


@pytest.mark.parametrize(
    ("code", "stdout", "stderr"), [(1, "", ""), (0, "", "warning"), (0, "x" * 65537, "")]
)
def test_failed_or_oversize_lock_observation_is_not_assumed_safe(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, code: int, stdout: str, stderr: str
) -> None:
    monkeypatch.setattr(
        subprocess,
        "run",
        lambda command, **_kwargs: subprocess.CompletedProcess(command, code, stdout, stderr),
    )
    with pytest.raises(ReplayError, match="observation failed"):
        reservation._lsof(tmp_path / "lock", deadline=time.monotonic() + 3)


def test_wait_retains_reparented_child_identity(
    lease_request: reservation.Request, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    orphan = Process(112, 1, os.getuid(), "child", "/usr/bin/worker")
    family = {111: lease_request.owner, 112: replace(orphan, ppid=111)}
    calls = 0

    def observed(_directory: Path, **_kwargs: object) -> list[Process]:
        nonlocal calls
        calls += 1
        if calls == 1:
            return [orphan]
        lease_request.status.write_text("1\n", encoding="utf-8")
        return []

    monkeypatch.setattr(reservation, "_snapshot", observed)
    monkeypatch.setattr(time, "sleep", lambda _seconds: None)
    with ExitStack() as stack:
        reservation._wait(lease_request, stack=stack, deadline=time.monotonic() + 3, family=family)
    assert calls == 2
    assert family[112].ppid == 111


def test_pid_reuse_is_not_confused_with_owned_process_exit(
    lease_request: reservation.Request, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        reservation,
        "_snapshot",
        lambda _directory, **_kwargs: [replace(lease_request.owner, start="new")],
    )
    with ExitStack() as stack, pytest.raises(ReplayError, match="PID was reused"):
        reservation._wait(
            lease_request,
            stack=stack,
            deadline=time.monotonic() + 3,
            family={111: lease_request.owner},
        )


def test_reused_descendant_pid_is_not_overwritten(
    lease_request: reservation.Request, monkeypatch: pytest.MonkeyPatch
) -> None:
    child = Process(112, 111, os.getuid(), "child", "/usr/bin/worker")
    monkeypatch.setattr(
        reservation, "_snapshot", lambda _directory, **_kwargs: [replace(child, start="new")]
    )
    with ExitStack() as stack, pytest.raises(ReplayError, match="PID was reused"):
        reservation._wait(
            lease_request,
            stack=stack,
            deadline=time.monotonic() + 3,
            family={111: lease_request.owner, 112: child},
        )


@pytest.mark.parametrize(
    ("code", "body", "expected"), [(0, "Z+\n", True), (0, "S\n", False), (1, "", True)]
)
def test_dying_process_executable_requires_independent_exit_state(
    lease_request: reservation.Request,
    monkeypatch: pytest.MonkeyPatch,
    code: int,
    body: str,
    expected: bool,
) -> None:
    calls: list[list[str]] = []

    def run(command: list[str], **_kwargs: object) -> subprocess.CompletedProcess[str]:
        calls.append(command)
        return subprocess.CompletedProcess(command, code, body, "")

    monkeypatch.setattr(subprocess, "run", run)
    actual = replace(lease_request.owner, executable="(python3)")
    assert (
        reservation._unchanged_owned(actual, lease_request.owner, time.monotonic() + 3) is expected
    )
    assert calls == [["/bin/ps", "-p", "111", "-o", "state="]]


@pytest.mark.parametrize(
    ("code", "body", "error"), [(2, "", ""), (0, "invalid state\n", ""), (0, "Z\n", "warning")]
)
def test_failed_exit_state_observation_is_not_assumed_dead(
    monkeypatch: pytest.MonkeyPatch, code: int, body: str, error: str
) -> None:
    monkeypatch.setattr(
        subprocess,
        "run",
        lambda command, **_kwargs: subprocess.CompletedProcess(command, code, body, error),
    )
    with pytest.raises(ReplayError, match="exit-state observation failed"):
        reservation._ended_or_zombie(111, time.monotonic() + 3)


@pytest.mark.parametrize("clock", ["00:00:00", "01:00:00", "23:59:59"])
def test_process_birth_parser_uses_wall_clock_and_omits_elapsed(clock: str) -> None:
    birth = f"Sun Oct 11 {clock} 2026"
    rows = reservation._parse_processes(f"123 100 501 {birth} /owned/path with spaces\n")
    assert rows == [Process(123, 100, 501, birth, "/owned/path with spaces")]


def test_signed_system_uid_is_not_confused_with_missing_processes() -> None:
    body = "123 100 -2 Sun Oct 11 00:00:00 2026 /system/service\n"
    assert reservation._parse_processes(body)[0].uid == -2


@pytest.mark.parametrize(
    "body",
    [
        "",
        "bad metadata",
        "0 100 501 Sun Oct 11 00:00:00 2026 /worker\n",
        "1 100 501 Sun Oct 11 00:00:00 2026 /worker\n" * 2,
        "1 100 501 Sun Oct 11 60:00:00 2026 /worker\n",
        "1 100 501 Sun Oct 11 00:00:00 2026 /" + "x" * 4096,
        "x" * 1048577,
    ],
)
def test_malformed_process_rows_are_never_treated_as_absence(body: str) -> None:
    with pytest.raises((ReplayError, ValueError)):
        reservation._parse_processes(body)


@pytest.mark.parametrize("elapsed", ["60:00", "01:00:00", "1-00:00:00"])
def test_scoped_snapshot_never_requests_unused_elapsed_field(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, elapsed: str
) -> None:
    calls: list[list[str]] = []

    def run(command: list[str], **_kwargs: object) -> subprocess.CompletedProcess[str]:
        calls.append(command)
        # An etime field would shift columns and be rejected. Valid identity
        # output stays the same at all elapsed-time display transitions.
        extra = f"{elapsed} " if "etime=" in command[-1] else ""
        body = f"123 100 501 {extra}Sun Oct 11 00:00:00 2026 /worker\n"
        return subprocess.CompletedProcess(command, 0, body, "")

    monkeypatch.setattr(subprocess, "run", run)
    rows = reservation._snapshot(tmp_path, deadline=time.monotonic() + 3)
    assert rows[0].pid == 123
    assert calls == [["/bin/ps", "-ww", "-ax", "-o", "pid=,ppid=,uid=,lstart=,comm="]]


@pytest.mark.parametrize(
    ("code", "body", "error"),
    [(1, "", ""), (0, "", ""), (0, "x" * 1048577, ""), (0, "", "warning")],
)
def test_failed_process_snapshot_does_not_mean_no_owned_processes(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, code: int, body: str, error: str
) -> None:
    monkeypatch.setattr(
        subprocess,
        "run",
        lambda command, **_kwargs: subprocess.CompletedProcess(command, code, body, error),
    )
    with pytest.raises(ReplayError, match="snapshot failed"):
        reservation._snapshot(tmp_path, deadline=time.monotonic() + 3)


def test_timeout_only_releases_reservation_and_keeps_active_job(
    lease_request: reservation.Request,
    fake_observation: None,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    clock = [0.0]
    monkeypatch.setattr(time, "monotonic", lambda: clock[0])
    monkeypatch.setattr(
        reservation, "_snapshot", lambda _directory, **_kwargs: [lease_request.owner]
    )

    def sleep(seconds: float) -> None:
        clock[0] += seconds

    monkeypatch.setattr(time, "sleep", sleep)
    before = (lease_request.state / "current.json").read_bytes()
    with pytest.raises(ReplayError, match="deadline reached"):
        reservation.reserve(lease_request, proof=tmp_path / "handoff.json")
    assert (lease_request.state / "current.json").read_bytes() == before
    assert not lease_request.status.exists()
    with stages.execution_lock(lease_request.state, "future") as acquired:
        assert acquired is True


@pytest.mark.parametrize("status", ["0\n", "2\n", "x" * 17])
def test_terminal_exit_zero_or_wrong_status_is_not_handoff_success(
    lease_request: reservation.Request, status: str
) -> None:
    before = reservation._identity(lease_request.log)
    lease_request.status.write_text(status, encoding="utf-8")
    with pytest.raises(ReplayError, match="expected busy terminal exit"):
        reservation._terminal(lease_request, before, time.monotonic() + 3)


def test_old_busy_text_cannot_prove_new_refusal(lease_request: reservation.Request) -> None:
    lease_request.log.write_text("future: busy\n", encoding="utf-8")
    before = reservation._identity(lease_request.log)
    lease_request.status.write_text("1\n", encoding="utf-8")
    with pytest.raises(ReplayError, match="No new reserved-stage"):
        reservation._terminal(lease_request, before, time.monotonic() + 3)


def test_replaced_or_truncated_log_is_rejected(lease_request: reservation.Request) -> None:
    before = reservation._identity(lease_request.log)
    lease_request.status.write_text("1\n", encoding="utf-8")
    lease_request.log.write_bytes(b"x")
    with pytest.raises(ReplayError, match="log changed identity"):
        reservation._terminal(lease_request, before, time.monotonic() + 3)


def test_domain_completion_and_output_hash_are_required(
    lease_request: reservation.Request, fake_observation: None
) -> None:
    with pytest.raises((ReplayError, FileNotFoundError), match="nonempty|No such file"):
        reservation._completed(lease_request, time.monotonic() + 3)


def test_real_lease_owner_observation(tmp_path: Path) -> None:
    with stages.execution_lock(tmp_path, "observed") as acquired:
        assert acquired is True
        body = reservation._lsof(tmp_path / "observed.lock", deadline=time.monotonic() + 3)
        (tmp_path / "lock-observation.txt").write_text(body, encoding="utf-8")
        reservation._lock_owner(tmp_path / "observed.lock", os.getpid(), time.monotonic() + 3)


def test_modified_completed_output_invalidates_proof(
    lease_request: reservation.Request, fake_observation: None, tmp_path: Path
) -> None:
    reservation.reserve(lease_request, proof=tmp_path / "handoff.json")
    lease_request.current.outputs[0].write_bytes(b"changed")
    with pytest.raises(ReplayError, match="did not complete unchanged"):
        reservation._completed(lease_request, time.monotonic() + 3)


def test_future_output_appearing_during_wait_refuses_proof(
    lease_request: reservation.Request,
    fake_observation: None,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    wait = reservation._wait

    def changed(value: reservation.Request, **kwargs: object) -> None:
        stack = kwargs["stack"]
        deadline = kwargs["deadline"]
        family = kwargs["family"]
        assert isinstance(stack, ExitStack)
        assert isinstance(deadline, float)
        assert isinstance(family, dict)
        wait(value, stack=stack, deadline=deadline, family={111: value.owner})
        (tmp_path / "future.txt").write_bytes(b"unexpected")

    monkeypatch.setattr(reservation, "_wait", changed)
    with pytest.raises(ReplayError, match="already has a receipt"):
        reservation.reserve(lease_request, proof=tmp_path / "handoff.json")
    assert not (tmp_path / "handoff.json").exists()


@pytest.mark.parametrize("namespace", [False, True])
def test_replaced_lease_or_namespace_refuses_handoff(
    lease_request: reservation.Request,
    fake_observation: None,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    namespace: bool,
) -> None:
    wait = reservation._wait

    def replaced(
        value: reservation.Request, *, stack: ExitStack, deadline: float, family: dict[int, Process]
    ) -> None:
        wait(value, stack=stack, deadline=deadline, family=family)
        path = value.job_root / "job.launch" if namespace else value.state / "future.lock"
        path.rename(path.with_suffix(".retained"))
        if namespace:
            path.mkdir()
        else:
            path.write_bytes(b"")

    monkeypatch.setattr(reservation, "_wait", replaced)
    with pytest.raises(ReplayError, match="identity changed"):
        reservation.reserve(lease_request, proof=tmp_path / "handoff.json")
    assert lease_request.status.read_text(encoding="utf-8") == "1\n"
    assert not (tmp_path / "handoff.json").exists()


def test_owned_process_present_again_at_final_boundary_refuses_proof(
    lease_request: reservation.Request,
    fake_observation: None,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    observed = reservation._snapshot
    calls = 0

    def returned(directory: Path, *, deadline: float) -> list[Process]:
        nonlocal calls
        calls += 1
        if calls > 2:
            return [lease_request.owner]
        return observed(directory, deadline=deadline)

    monkeypatch.setattr(reservation, "_snapshot", returned)
    with pytest.raises(ReplayError, match="present at final handoff"):
        reservation.reserve(lease_request, proof=tmp_path / "handoff.json")
    assert not (tmp_path / "handoff.json").exists()


def test_request_change_after_load_is_rejected(
    lease_request: reservation.Request, tmp_path: Path
) -> None:
    raw = json.loads(lease_request.path.read_text(encoding="utf-8"))
    raw["timeoutSeconds"] = 4
    lease_request.path.write_text(json.dumps(raw), encoding="utf-8")
    with pytest.raises(ReplayError, match="changed after validation"):
        reservation.reserve(lease_request, proof=tmp_path / "handoff.json")


@pytest.mark.parametrize("proof_name", ["reservation.json", "state/proof.json", "future.txt"])
def test_proof_cannot_overwrite_or_overlap_existing_namespaces(
    lease_request: reservation.Request, tmp_path: Path, proof_name: str
) -> None:
    with pytest.raises(ReplayError, match="distinct, absent"):
        reservation._proof_path(tmp_path / proof_name, lease_request)


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ({"version": 2}, "request fields"),
        ({"version": True}, "request fields"),
        ({"timeoutSeconds": 0}, "integer"),
        ({"timeoutSeconds": True}, "integer"),
        ({"timeoutSeconds": 3601}, "finite budget"),
        ({"boundary": "current"}, "preceding stage"),
        ({"boundary": "unknown"}, "preceding stage"),
        ({"transferStages": ["future"]}, "complete ordered future suffix"),
        ({"transferStages": []}, "explicit bounded"),
        ({"transferStages": [1]}, "explicit bounded"),
        ({"current": {"stage": "current"}}, "current-stage reservation fields"),
        ({"workflow": {}}, "workflow reservation pin"),
        ({"job": {}}, "job identity fields"),
        ({"extra": True}, "request fields"),
    ],
)
def test_invalid_manifest_fields_do_not_take_leases(
    request_path: Path, change: dict[str, object], message: str
) -> None:
    raw = json.loads(request_path.read_text(encoding="utf-8"))
    raw.update(change)
    request_path.write_text(json.dumps(raw), encoding="utf-8")
    with pytest.raises(ReplayError, match=message):
        reservation.load_request(request_path)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("name", "../job", "job namespace"),
        ("pid", 0, "integer"),
        ("ppid", True, "integer"),
        ("uid", -1, "integer"),
        ("executable", "", "reservation text"),
        ("executable", "python\n", "reservation text"),
        ("executable", "x" * 4097, "text exceeds"),
        ("start", "", "reservation text"),
        ("start", "x" * 4097, "text exceeds"),
    ],
)
def test_invalid_expected_job_identity_is_rejected(
    request_path: Path, field: str, value: object, message: str
) -> None:
    raw = json.loads(request_path.read_text(encoding="utf-8"))
    raw["job"][field] = value
    request_path.write_text(json.dumps(raw), encoding="utf-8")
    with pytest.raises(ReplayError, match=message):
        reservation.load_request(request_path)


@pytest.mark.parametrize("comm", [".venv/bin/python", "python", "/absolute/python with spaces"])
def test_observed_executable_is_opaque_and_never_resolved(request_path: Path, comm: str) -> None:
    raw = json.loads(request_path.read_text(encoding="utf-8"))
    raw["job"]["executable"] = comm
    request_path.write_text(json.dumps(raw), encoding="utf-8")
    value = reservation.load_request(request_path)
    assert value.owner.executable == comm
    assert reservation._same_process(value.owner, replace(value.owner, ppid=value.owner.ppid))
    assert not reservation._same_process(replace(value.owner, executable=comm + "x"), value.owner)


def test_foreign_owner_uid_is_rejected(request_path: Path) -> None:
    raw = json.loads(request_path.read_text(encoding="utf-8"))
    raw["job"]["uid"] = os.getuid() + 1
    request_path.write_text(json.dumps(raw), encoding="utf-8")
    with pytest.raises(ReplayError, match="owned exact controller"):
        reservation.load_request(request_path)


def test_workflow_pin_mismatch_is_rejected(request_path: Path) -> None:
    raw = json.loads(request_path.read_text(encoding="utf-8"))
    raw["workflow"]["sha256"] = "a" * 64
    request_path.write_text(json.dumps(raw), encoding="utf-8")
    with pytest.raises(ReplayError, match="workflow pin changed"):
        reservation.load_request(request_path)


def test_domain_verifier_is_required(request_path: Path) -> None:
    raw = json.loads(request_path.read_text(encoding="utf-8"))
    plan = Path(raw["workflow"]["path"])
    graph = json.loads(plan.read_text(encoding="utf-8"))
    graph["stages"][0].pop("verify_command")
    plan.write_text(json.dumps(graph), encoding="utf-8")
    raw["workflow"]["sha256"] = stages.file_digest(plan)
    request_path.write_text(json.dumps(raw), encoding="utf-8")
    with pytest.raises(ReplayError, match="domain verifier"):
        reservation.load_request(request_path)


def test_boundary_requires_true_dependency_not_just_list_order(request_path: Path) -> None:
    raw = json.loads(request_path.read_text(encoding="utf-8"))
    plan = Path(raw["workflow"]["path"])
    graph = json.loads(plan.read_text(encoding="utf-8"))
    graph["stages"][1]["inputs"] = graph["stages"][0]["inputs"]
    plan.write_text(json.dumps(graph), encoding="utf-8")
    raw["workflow"]["sha256"] = stages.file_digest(plan)
    request_path.write_text(json.dumps(raw), encoding="utf-8")
    with pytest.raises(ReplayError, match="must depend"):
        reservation.load_request(request_path)


@pytest.mark.parametrize("value", ["relative", "/tmp/forbidden", "x\n", "x\0", None])
def test_paths_cannot_escape_repo_or_use_control_characters(value: object) -> None:
    with pytest.raises(ReplayError, match="path|work storage"):
        reservation._path(value)


def test_parent_link_inside_checkout_is_rejected(tmp_path: Path) -> None:
    directory = tmp_path / "real"
    directory.mkdir()
    alias = tmp_path / "alias"
    alias.symlink_to(directory, target_is_directory=True)
    with pytest.raises(ReplayError, match="traverse links"):
        reservation._path(str(alias / "proof.json"))


def test_linked_and_empty_json_evidence_is_rejected(tmp_path: Path) -> None:
    original = tmp_path / "original"
    original.write_bytes(b"{}")
    alias = tmp_path / "alias"
    os.link(original, alias)
    with pytest.raises(ReplayError, match="single-link"):
        reservation._json(alias)


def test_empty_json_is_rejected(tmp_path: Path) -> None:
    empty = tmp_path / "empty"
    empty.write_bytes(b"")
    with pytest.raises(ReplayError, match="single-link"):
        reservation._json(empty)


def test_bounded_json_does_not_read_oversize_artifact(tmp_path: Path) -> None:
    large = tmp_path / "large"
    large.write_bytes(b"x" * 1048577)
    with pytest.raises(ReplayError, match="byte budget"):
        reservation._json(large)


@pytest.mark.parametrize("value", [None, "", "A" * 64, "a" * 63])
def test_invalid_sha_never_becomes_expected_fingerprint(value: object) -> None:
    with pytest.raises(ReplayError, match="SHA-256"):
        reservation._sha(value)


def test_cwd_must_be_the_real_repository(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.chdir(tmp_path)
    with pytest.raises(ReplayError, match="repository root"):
        reservation._repository()


def test_descendant_enumeration_covers_nested_children() -> None:
    rows = [
        Process(113, 112, 0, "grandchild", "/worker"),
        Process(112, 111, 0, "child", "/worker"),
        Process(999, 1, 0, "unrelated", "/worker"),
    ]
    assert [row.pid for row in reservation._descendants(rows, 111)] == [112, 113]


def test_cli_reports_unfulfilled_without_creating_success_proof(
    lease_request: reservation.Request, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    lease_request.status.write_text("1\n", encoding="utf-8")
    assert (
        reservation.main([str(lease_request.path), "--proof", str(tmp_path / "handoff.json")]) == 1
    )
    assert "Reservation unfulfilled" in capsys.readouterr().err
    assert not (tmp_path / "handoff.json").exists()


def test_cli_check_only_loads_without_leases_or_state_changes(
    lease_request: reservation.Request, capsys: pytest.CaptureFixture[str]
) -> None:
    before = (lease_request.state / "current.json").read_bytes()
    assert reservation.main([str(lease_request.path), "--check"]) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["status"] == "loaded-request"
    assert result["reservationDispatched"] is False
    assert (lease_request.state / "current.json").read_bytes() == before
    assert not (lease_request.state / "future.lock").exists()
    assert not lease_request.status.exists()


def test_cli_returns_exclusive_proof_path(
    lease_request: reservation.Request,
    fake_observation: None,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    assert (
        reservation.main([str(lease_request.path), "--proof", str(tmp_path / "handoff.json")]) == 0
    )
    assert capsys.readouterr().out.endswith("/handoff.json\n")


def _wait_ready(path: Path) -> None:
    deadline = time.monotonic() + 5
    while not path.exists():
        if time.monotonic() >= deadline:
            raise RuntimeError("Synthetic controller did not start")
        time.sleep(0.02)


@pytest.mark.parametrize("terminal_failure", [False, True])
def test_real_subprocess_finishes_domain_then_stops_at_reserved_boundary(
    request_path: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, terminal_failure: bool
) -> None:
    script = Path(__file__).resolve()
    raw = json.loads(request_path.read_text(encoding="utf-8"))
    plan = Path(raw["workflow"]["path"])
    graph = json.loads(plan.read_text(encoding="utf-8"))
    ready = tmp_path / "ready"
    release = tmp_path / "release"
    graph["stages"][0]["command"] = [
        sys.executable,
        str(script),
        "fixture-work",
        str(tmp_path / "current.txt"),
        str(ready),
        str(release),
    ]
    graph["stages"][0]["verify_command"] = [
        sys.executable,
        str(script),
        "fixture-verify",
        str(tmp_path / "current.txt"),
    ]
    plan.write_text(json.dumps(graph), encoding="utf-8")
    (tmp_path / "state/current.json").unlink()
    with (
        (tmp_path / "job.log").open("wb") as log,
        subprocess.Popen(
            [
                sys.executable,
                str(script),
                "fixture-controller",
                str(plan),
                str(tmp_path / "state"),
                str(tmp_path / "job.status"),
            ],
            stdin=subprocess.DEVNULL,
            stdout=log,
            stderr=subprocess.STDOUT,
        ) as child,
        ThreadPoolExecutor(max_workers=1) as reaper,
    ):
        terminal = reaper.submit(child.wait, 5)
        _wait_ready(ready)
        actual = processes(tmp_path, deadline=time.monotonic() + 3, pid=child.pid)[0]
        raw["job"].update(
            {
                "pid": actual.pid,
                "ppid": actual.ppid,
                "uid": actual.uid,
                "start": actual.start,
                "executable": actual.executable,
            }
        )
        raw["workflow"]["sha256"] = stages.file_digest(plan)
        raw["current"]["fingerprint"] = json.loads(
            (tmp_path / "state/current.json").read_text(encoding="utf-8")
        )["fingerprint"]
        request_path.write_text(json.dumps(raw), encoding="utf-8")
        lease_request = reservation.load_request(request_path)
        initial = reservation._initial

        def release_after_observation(
            value: reservation.Request, deadline: float
        ) -> tuple[reservation.FileIdentity, dict[int, Process]]:
            observed = initial(value, deadline)
            release.write_bytes(b"continue")
            return observed

        monkeypatch.setattr(reservation, "_initial", release_after_observation)
        if terminal_failure:

            def fail_after_busy(_value: reservation.Request, _deadline: float) -> dict[str, object]:
                raise ReplayError("Synthetic final validation failed after controller busy exit")

            monkeypatch.setattr(reservation, "_completed", fail_after_busy)
        proof = tmp_path / "handoff.json"
        try:
            if terminal_failure:
                with pytest.raises(ReplayError, match="final validation failed"):
                    reservation.reserve(lease_request, proof=proof)
            else:
                reservation.reserve(lease_request, proof=proof)
        finally:
            if not release.exists():
                release.write_bytes(b"continue")
        assert terminal.result(timeout=5) == 1
    assert (tmp_path / "current.txt").read_bytes() == b"verified fixture"
    assert not (tmp_path / "future.txt").exists()
    assert not (tmp_path / "state/future.json").exists()
    assert (tmp_path / "job.status").read_text(encoding="utf-8") == "1\n"
    assert "future: busy" in (tmp_path / "job.log").read_text(encoding="utf-8")
    if terminal_failure:
        assert not proof.exists()
    else:
        assert json.loads(proof.read_text(encoding="utf-8"))["terminal"]["exitCode"] == 1
    with stages.execution_lock(lease_request.state, "future") as acquired:
        assert acquired is True


def _fixture_main(arguments: list[str]) -> int:
    mode = arguments[0]
    if mode == "fixture-controller":
        code = workflow.main([arguments[1], "--state", arguments[2]])
        Path(arguments[3]).write_text(f"{code}\n", encoding="utf-8")
        return code
    if mode == "fixture-verify":
        return 0 if Path(arguments[1]).read_bytes() == b"verified fixture" else 2
    if mode == "fixture-work":
        Path(arguments[2]).write_bytes(b"ready")
        deadline = time.monotonic() + 5
        while not Path(arguments[3]).exists():
            if time.monotonic() >= deadline:
                return 2
            time.sleep(0.02)
        Path(arguments[1]).write_bytes(b"verified fixture")
        return 0
    return 2


if __name__ == "__main__":
    raise SystemExit(_fixture_main(sys.argv[1:]))
