"""Owned entrypoint/alias scopes and future actor-release verification plans."""

import json
from pathlib import Path

import pytest

from replay import code_dependency_inventory as inventory
from replay import code_inventory_memo as memo
from replay import code_inventory_memo_run as runner
from replay import recovery_source_finalization as recovery
from replay import retained_cleanup_finalization as retained
from replay import subtitle_session_cleanup_finalization as finalization
from replay import subtitle_session_cleanup_wave as wave
from replay import subtitle_session_observations as observations
from replay import workflow
from replay.sequence_journal import encode
from replay.stages import execution_lock


def test_explicit_aliases_match_memoize_and_restore(monkeypatch: pytest.MonkeyPatch) -> None:
    ordinary = inventory.capture
    expected = ordinary(("core",))
    calls: list[tuple[str, ...]] = []

    def counted(
        modules: tuple[str, ...], *, package: Path = inventory.PACKAGE
    ) -> tuple[inventory.Source, ...]:
        calls.append(modules)
        return ordinary(modules, package=package)

    monkeypatch.setattr(inventory, "capture", counted)
    monkeypatch.setitem(observations.__dict__, "capture", counted)
    with runner.capture_scope():
        for _ in range(5):
            assert inventory.capture(("core",)) == expected
            assert runner._observation_alias_is(inventory.capture)
        with pytest.raises(ValueError, match="already active"), runner.capture_scope():
            pass
        with pytest.raises(ValueError, match="source package"):
            inventory.capture(("core",), package=Path.cwd())
    assert calls == [("core",)]
    assert inventory.capture is counted
    assert runner._observation_alias_is(counted)


def test_restore_on_body_error_and_ownership_loss(monkeypatch: pytest.MonkeyPatch) -> None:
    original = inventory.capture
    with pytest.raises(RuntimeError, match="body failed"), runner.capture_scope():
        raise RuntimeError("body failed")
    assert inventory.capture is original and runner._observation_alias_is(original)
    with pytest.raises(ValueError, match="ownership changed"), runner.capture_scope():
        inventory.capture = original
    assert inventory.capture is original and runner._observation_alias_is(original)
    monkeypatch.setitem(observations.__dict__, "capture", None)
    with pytest.raises(ValueError, match="not owned"), runner.capture_scope():
        pass
    assert observations.__dict__["capture"] is None
    assert not runner._SCOPE_LOCK.locked()


@pytest.mark.parametrize("fail_constructor", [False, True])
def test_initial_identity_failure_releases_process_lease(
    monkeypatch: pytest.MonkeyPatch, fail_constructor: bool
) -> None:
    ordinary = memo._identity

    def failed(path: Path, *, directory: bool = False) -> memo.Identity:
        if fail_constructor or path.name == "code_inventory_memo_run.py":
            raise ValueError("source guard failed")
        return ordinary(path, directory=directory)

    monkeypatch.setattr(memo, "_identity", failed)
    with pytest.raises(ValueError, match="source guard failed"), runner.capture_scope():
        pass
    assert not runner._SCOPE_LOCK.locked()


@pytest.mark.parametrize("name", tuple(runner.ALLOWED))
def test_typed_dispatch_and_target_exit(name: str, monkeypatch: pytest.MonkeyPatch) -> None:
    entries = {
        "replay.subtitle_session_cleanup_finalization": finalization,
        "replay.recovery_source_finalization": recovery,
        "replay.retained_cleanup_finalization": retained,
        "replay.subtitle_session_cleanup_wave": wave,
    }
    received: list[list[str] | None] = []

    def entry(args: list[str] | None = None) -> int:
        received.append(args)
        assert inventory.capture(("core",)) == inventory.capture(("core",))
        return 7

    monkeypatch.setattr(entries[name], "main", entry)
    mode = "verify-prepare-child"
    assert runner.main(["run", name, mode, "synthetic-operation.json"]) == 7
    assert received == [[mode, "synthetic-operation.json"]]


@pytest.mark.parametrize(
    "name,args",
    [
        ("replay.cli", ["download-all"]),
        ("replay.retained_cleanup_finalization", ["run-child"]),
        ("replay.retained_cleanup_finalization", []),
    ],
)
def test_unknown_entry_and_producer_modes_rejected(name: str, args: list[str]) -> None:
    with pytest.raises(ValueError, match="allowed"):
        runner.run(name, args)
    assert not runner._SCOPE_LOCK.locked()


def test_prior_actor_kernel_lease_blocks_entry(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    state = tmp_path / "actor-state"
    with execution_lock(state, ".workflow"):
        lease = state / ".workflow.lock"
        with pytest.raises(ValueError, match="active"):
            runner.run(
                "replay.retained_cleanup_finalization",
                ["verify-finalize", "unused"],
                leases=(lease,),
            )
    monkeypatch.setattr(retained, "main", lambda args: 0)
    assert (
        runner.run(
            "replay.retained_cleanup_finalization", ["verify-finalize", "unused"], leases=(lease,)
        )
        == 0
    )
    with pytest.raises(ValueError, match="distinct"):
        runner.run(
            "replay.retained_cleanup_finalization", ["verify-finalize"], leases=(lease, lease)
        )
    with pytest.raises(ValueError, match="already exist"):
        runner.run(
            "replay.retained_cleanup_finalization",
            ["verify-finalize"],
            leases=(state / "missing.lock",),
        )


def request(tmp_path: Path) -> Path:
    path = tmp_path / "request.json"
    path.write_bytes(
        encode(
            {
                "version": 1,
                "module": "replay.retained_cleanup_finalization",
                "arguments": ["verify-run-child", str(tmp_path / "operation.json")],
                "inputs": [str(tmp_path / "operation.json")],
                "proof": str(tmp_path / "completion.json"),
                "leases": [str(tmp_path / "actor-state/.workflow.lock")],
                "reason": "Verify genuine domain proof for downstream reuse after owner release",
            }
        )
    )
    return path


def test_future_plan_binds_helpers_and_actual_domain_verifier(tmp_path: Path) -> None:
    config = request(tmp_path)
    output = tmp_path / "future"
    assert runner.main(["prepare-check-plan", str(config), "--output", str(output)]) == 0
    graph = output / "workflow.json"
    assert workflow.main([str(graph), "--state", str(tmp_path / "future-state"), "--check"]) == 0
    stage = workflow.load_stages(graph)[0]
    assert stage.adopt_existing
    assert stage.command == stage.verify_command
    assert "--check" not in stage.verify_command
    assert Path(runner.__file__).resolve() in stage.inputs
    assert Path(memo.__file__).resolve() in stage.inputs
    assert stage.outputs == (tmp_path / "completion.json",)
    with pytest.raises(ValueError, match="previous"):
        runner.prepare_check_plan(config, output)


@pytest.mark.parametrize(
    "mutation",
    [
        "extra",
        "version",
        "arguments",
        "mode",
        "reason",
        "inputs",
        "leases",
        "duplicate",
        "proof-input",
    ],
)
def test_invalid_future_plan_refused(tmp_path: Path, mutation: str) -> None:
    config = request(tmp_path)
    data = json.loads(config.read_text())
    if mutation == "extra":
        data["extra"] = True
    elif mutation == "version":
        data["version"] = True
    elif mutation == "mode":
        data["arguments"][0] = "finalize"
    elif mutation == "duplicate":
        data["leases"] *= 2
    elif mutation == "proof-input":
        data["inputs"].append(data["proof"])
    else:
        data[mutation] = None
    config.write_bytes(encode(data))
    with pytest.raises(ValueError):
        runner.prepare_check_plan(config, tmp_path / "future")


def test_benchmark_cli_is_source_only(tmp_path: Path) -> None:
    output = tmp_path / "benchmark"
    assert runner.main(["benchmark", "--output", str(output), "--iterations", "1"]) == 0
    result = json.loads((output / "comparison.json").read_text())
    assert result["producerLaunched"] is False and result["liveProcessesModified"] is False
    assert all(row["exactInventories"] for row in result["results"])
    assert runner.main(["benchmark", "--output", str(output)]) == 1
    assert runner.main(["run", "replay.retained_cleanup_finalization", "run-child", "unused"]) == 1
