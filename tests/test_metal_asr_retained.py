"""Completed-only adoption and historical PID reuse; no model/process launches."""

import hashlib
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from test_metal_asr_resume import Runtime, replace
from test_metal_asr_resume import setup as setup

from replay import asr_assembly, asr_assembly_verification
from replay import isolated_metal_asr as lane
from replay import metal_asr_resume as retained
from replay import metal_asr_retained as completed
from replay.core import ReplayError
from replay.measured_job_admission import Parent
from replay.native_invocation import pin


def test_full_completion_without_original_collection_terminal(
    setup: tuple[Path, Path, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    request, source, output = setup
    lane.collect(request, source, runtime=Runtime())
    (source / "complete.json").unlink()
    original = pin(source / "000000480000.metal-runtime/complete.json")
    host = MagicMock(spec=completed.CurrentHost)
    host.absent.return_value = True
    monkeypatch.setattr(completed, "CurrentHost", lambda: host)
    completed.prepare(request, source, output)
    completed.run(request, output)
    completed.run(request, output, verify_only=True)
    assert pin(source / "000000480000.metal-runtime/complete.json") == original
    assert not (source / "complete.json").exists()
    assert lane._read(output / "complete.json")["windows"] == 2
    assert (
        lane._read(output / "inputs.json")["processPolicy"]
        == "retained-verified-completed-children-only; no-worker-dispatch"
    )
    labels = output.parent / "labels.json"
    assert asr_assembly.assemble(output, labels) == 0
    assert asr_assembly_verification.verify(output, labels) == 0
    assert not list(output.glob("*.metal-runtime"))


def test_pid_reuse_does_not_block_or_signal_unrelated_actor(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    host = MagicMock(spec=completed.CurrentHost)
    host.absent.return_value = False
    host.identity.return_value = Parent(42, "new:2", Path("/synthetic/new"), "a" * 64)
    assert completed._actor_gone(host, {"pid": 42, "birth": "old:1"}, launch=True) is True
    host.identity.assert_has_calls([((42,),), ((42,),)])


def test_same_birth_command_and_descendants_block() -> None:
    host = MagicMock(spec=completed.CurrentHost)
    host.absent.return_value = False
    digest = hashlib.sha256(b"/synthetic/worker --worker\n").hexdigest()
    host.identity.return_value = Parent(42, "old:1", Path("/synthetic/worker"), digest)
    with pytest.raises(ReplayError, match="still present"):
        completed._actor_gone(
            host,
            {"pid": 42, "birth": "old:1", "command": ["/synthetic/worker", "--worker"]},
            launch=True,
        )
    with pytest.raises(ReplayError, match="command identity"):
        completed._actor_gone(
            host, {"pid": 42, "birth": "old:1", "command": ["/synthetic/different"]}, launch=True
        )
    with pytest.raises(ReplayError, match="still present"):
        completed._actor_gone(host, {"pid": 42, "birth": "old:1"})
    with pytest.raises(ReplayError, match="launch command"):
        completed._actor_gone(host, {"pid": 42, "birth": "old:1"}, launch=True)
    host.identity.side_effect = [
        Parent(42, "new:2", Path("/synthetic/new"), "a" * 64),
        Parent(42, "newer:3", Path("/synthetic/new"), "a" * 64),
    ]
    with pytest.raises(ReplayError, match="changed during"):
        completed._actor_gone(host, {"pid": 42, "birth": "old:1"})


def test_reused_group_is_not_attached_to_original_receipt(
    setup: tuple[Path, Path, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    request, source, output = setup
    lane.collect(request, source, runtime=Runtime())
    host = MagicMock(spec=completed.CurrentHost)
    host.absent.return_value = False
    host.identity.return_value = Parent(99999999, "reused:2", Path("/synthetic/new"), "a" * 64)
    group = MagicMock(side_effect=ReplayError("unrelated new group"))
    monkeypatch.setattr(retained, "_group_gone", group)
    completed.prepare(request, source, output, host=host)
    completed.run(request, output, host=host)
    group.assert_not_called()
    assert lane._read(output / "complete.json")["windows"] == 2


def test_incomplete_source_is_not_completed(setup: tuple[Path, Path, Path]) -> None:
    request, source, output = setup
    with pytest.raises(ReplayError, match="every completed"):
        completed.prepare(request, source, output)
    assert not output.exists()


@pytest.mark.parametrize("change", ["raw", "resources", "complete"])
def test_original_corruption_rejected(setup: tuple[Path, Path, Path], change: str) -> None:
    request, source, output = setup
    lane.collect(request, source, runtime=Runtime())
    replace(source / "000000480000.metal-runtime" / f"{change}.json", {"corrupt": True})
    with pytest.raises(ReplayError):
        completed.prepare(request, source, output)
    assert not output.exists()


@pytest.mark.parametrize(
    "name",
    [
        "inputs.json",
        "000000000000.raw.json",
        "000000480000.clock.json",
        "000000000000.receipt.json",
        "complete.json",
    ],
)
def test_changed_new_publication_rejected(
    setup: tuple[Path, Path, Path], monkeypatch: pytest.MonkeyPatch, name: str
) -> None:
    request, source, output = setup
    lane.collect(request, source, runtime=Runtime())
    host = MagicMock(spec=completed.CurrentHost)
    host.absent.return_value = True
    monkeypatch.setattr(completed, "CurrentHost", lambda: host)
    completed.prepare(request, source, output)
    completed.run(request, output)
    replace(output / name, {"corrupt": True})
    with pytest.raises(ReplayError):
        completed.run(request, output, verify_only=True)


def test_source_lease_and_missing_verify_publication(
    setup: tuple[Path, Path, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    request, source, output = setup
    lane.collect(request, source, runtime=Runtime())
    host = MagicMock(spec=completed.CurrentHost)
    host.absent.return_value = True
    monkeypatch.setattr(completed, "CurrentHost", lambda: host)
    with retained._lease(source), pytest.raises(BlockingIOError):
        completed.prepare(request, source, output)
    completed.prepare(request, source, output)
    with pytest.raises(ReplayError, match="Missing retained"):
        completed.run(request, output, verify_only=True)


def test_actual_host_adapter(monkeypatch: pytest.MonkeyPatch) -> None:
    process = MagicMock()
    process.absent.return_value = True
    identity = MagicMock()
    identity.identity.return_value = Parent(42, "new:2", Path("/synthetic/new"), "a" * 64)
    monkeypatch.setattr(completed, "ProcessHost", lambda: process)
    monkeypatch.setattr(completed, "IdentityHost", lambda: identity)
    assert completed.CurrentHost().absent(42) is True
    assert completed.CurrentHost().identity(42).birth == "new:2"


def test_cli(monkeypatch: pytest.MonkeyPatch) -> None:
    prepare = MagicMock()
    run = MagicMock()
    monkeypatch.setattr(completed, "prepare", prepare)
    monkeypatch.setattr(completed, "run", run)
    assert completed.main(["prepare", "request", "output", "--source", "source"]) == 0
    prepare.assert_called_once_with(Path("request"), Path("source"), Path("output"))
    assert completed.main(["run", "request", "output"]) == 0
    assert completed.main(["verify", "request", "output"]) == 0
    with pytest.raises(ReplayError, match="requires"):
        completed.main(["prepare", "request", "output"])
    with pytest.raises(ReplayError, match="bound"):
        completed.main(["run", "request", "output", "--source", "source"])
