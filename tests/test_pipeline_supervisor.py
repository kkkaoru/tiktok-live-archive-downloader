from __future__ import annotations

import hashlib
import json
import os
import sys
from collections.abc import Callable, Sequence
from pathlib import Path

import pytest

from replay import delivery_retention, workflow
from replay import pipeline_supervisor as supervisor
from replay import whole_technical_batch as technical_batch
from replay.core import ReplayError
from replay.stages import Stage, StageError, _fingerprint, execution_lock


def scripted(results: list[supervisor.Outcome], seen: list[Sequence[str]]) -> supervisor.Runner:
    def run(
        command: Sequence[str], *, log: Path, timeout: int, started: Callable[[int], None]
    ) -> supervisor.Outcome:
        seen.append(command)
        log.write_text("diagnostic", encoding="utf-8")
        started(123456)
        return results.pop(0)

    return run


@pytest.fixture
def config(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    path = tmp_path / "config.json"
    path.write_text('{"version":1}', encoding="utf-8")
    monkeypatch.setattr(workflow, "main", lambda args: 0)
    return path


def test_legacy_failure_never_skips_or_retries(config: Path, tmp_path: Path) -> None:
    seen: list[Sequence[str]] = []
    assert (
        supervisor.supervise(
            config,
            state=tmp_path / "state",
            run=scripted([supervisor.Outcome(1, ("combine",))], seen),
        )
        == 1
    )
    assert len(seen) == 1
    assert "--start" not in seen[0]
    with pytest.raises(ValueError, match="uncertain"):
        supervisor.supervise(config, state=tmp_path / "state", run=scripted([], seen))


def test_repeated_diagnostics_do_not_reset_stalls(config: Path, tmp_path: Path) -> None:
    seen: list[Sequence[str]] = []
    results = [supervisor.Outcome(1, ("combine",))] * 3
    assert (
        supervisor.supervise(
            config,
            state=tmp_path / "state",
            workflow_state=tmp_path / "jobs",
            run=scripted(results, seen),
            sleep=lambda seconds: None,
        )
        == 1
    )
    assert len(seen) == 3
    assert len({tuple(command) for command in seen}) == 1
    assert (
        supervisor.supervise(
            config,
            state=tmp_path / "state",
            workflow_state=tmp_path / "jobs",
            run=scripted([], seen),
        )
        == 1
    )


def test_workflow_replays_then_succeeds(config: Path, tmp_path: Path) -> None:
    seen: list[Sequence[str]] = []
    assert (
        supervisor.supervise(
            config,
            state=tmp_path / "state",
            workflow_state=tmp_path / "jobs",
            targets=("chosen",),
            run=scripted([supervisor.Outcome(1), supervisor.Outcome(0)], seen),
            sleep=lambda seconds: None,
        )
        == 0
    )
    assert seen[0][-2:] == ["--target", "chosen"]
    assert seen[0] == seen[1]


def test_signal_is_uncertain(config: Path, tmp_path: Path) -> None:
    seen: list[Sequence[str]] = []
    assert (
        supervisor.supervise(
            config,
            state=tmp_path / "state",
            workflow_state=tmp_path / "jobs",
            run=scripted([supervisor.Outcome(-9, uncertain=True)], seen),
        )
        == 1
    )
    assert len(seen) == 1


def test_config_change_refuses_reuse(config: Path, tmp_path: Path) -> None:
    supervisor.supervise(
        config, state=tmp_path / "state", run=scripted([supervisor.Outcome(0)], [])
    )
    config.write_text("changed", encoding="utf-8")
    with pytest.raises(ValueError, match="changed"):
        supervisor.supervise(config, state=tmp_path / "state", run=scripted([], []))


def test_partial_launch_is_retained(config: Path, tmp_path: Path) -> None:
    def failed(
        command: Sequence[str], *, log: Path, timeout: int, started: Callable[[int], None]
    ) -> supervisor.Outcome:
        started(123456)
        raise OSError("launch uncertain")

    with pytest.raises(OSError):
        supervisor.supervise(config, state=tmp_path / "state", run=failed)
    with pytest.raises(ValueError, match="Unfinished"):
        supervisor.supervise(config, state=tmp_path / "state", run=scripted([], []))


def test_lease_refuses_duplicate(config: Path, tmp_path: Path) -> None:
    state = tmp_path / "state"
    with execution_lock(state, "supervisor"), pytest.raises(StageError):
        supervisor.supervise(config, state=state, run=scripted([], []))


@pytest.mark.parametrize(
    "line",
    [
        "STEP unknown: completed",
        "STEP motion-export: completed",
        "STEP combine: completed\nSTEP combine: completed",
        "STEP combine: pending",
    ],
)
def test_invalid_step_sequence(tmp_path: Path, line: str) -> None:
    log = tmp_path / "log"
    log.write_text(line, encoding="utf-8")
    with pytest.raises(ValueError):
        supervisor._steps(log)


def test_run_real_safe_child(tmp_path: Path) -> None:
    pids: list[int] = []
    result = supervisor._run(
        [sys.executable, "-c", "print('STEP combine: completed'); print('noise')"],
        log=tmp_path / "log",
        timeout=5,
        started=pids.append,
    )
    assert result == supervisor.Outcome(0, ("combine",), False)
    assert len(pids) == 1


def test_run_timeout_owns_only_its_group(tmp_path: Path) -> None:
    result = supervisor._run(
        [sys.executable, "-c", "import time; time.sleep(10)"],
        log=tmp_path / "log",
        timeout=1,
        started=lambda pid: None,
    )
    assert result == supervisor.Outcome(1, (), True)


def test_group_alive_current_group() -> None:
    assert supervisor._group_alive(os.getpgrp()) is True


@pytest.mark.parametrize("timeout", [0, True, 86401])
def test_invalid_deadline(config: Path, timeout: int) -> None:
    with pytest.raises(ValueError):
        supervisor.supervise(config, timeout=timeout)


def test_paths_and_modes(config: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    with pytest.raises(ValueError):
        supervisor.supervise(config, state=tmp_path / "same", workflow_state=tmp_path / "same")
    with pytest.raises(ValueError):
        supervisor.supervise(config, targets=("name",))
    monkeypatch.setattr(workflow, "main", lambda args: 1)
    with pytest.raises(ValueError, match="preflight"):
        supervisor.supervise(config, state=tmp_path / "state", workflow_state=tmp_path / "jobs")


def test_declared_pins(config: Path) -> None:
    with pytest.raises(ValueError, match="changed"):
        supervisor._pins(config, ((config, "0" * 64),))
    with pytest.raises(ValueError, match="Duplicate"):
        supervisor._pins(config, ((config, "0" * 64), (config, "0" * 64)))


def test_metadata_limits(tmp_path: Path) -> None:
    path = tmp_path / "metadata"
    path.write_text("[]", encoding="utf-8")
    with pytest.raises(ValueError):
        supervisor._metadata(path)
    path.write_bytes(b"x" * (supervisor.MAX_METADATA + 1))
    with pytest.raises(ValueError):
        supervisor._metadata(path)
    with pytest.raises(ValueError):
        supervisor._steps(path)


def test_alias_and_outside_rejected(config: Path, tmp_path: Path) -> None:
    alias = tmp_path / "alias"
    alias.symlink_to(config)
    with pytest.raises(ValueError):
        supervisor._canonical(alias)
    with pytest.raises(ValueError):
        supervisor._canonical(Path("relative"))
    with pytest.raises(ValueError):
        supervisor._canonical(Path("/"))


def test_main_usage_and_failure(config: Path) -> None:
    assert supervisor.main([]) == 2
    assert supervisor.main([str(config), "--timeout", "0"]) == 2


def test_shared_config_lease_blocks_different_state(config: Path, tmp_path: Path) -> None:
    key = hashlib.sha256(str(config).encode("utf-8")).hexdigest()
    with (
        execution_lock(supervisor.WORK / "pipeline-supervisor-leases", key),
        pytest.raises(StageError, match="configuration"),
    ):
        supervisor.supervise(config, state=tmp_path / "other", run=scripted([], []))


def test_successful_declared_input(config: Path, tmp_path: Path) -> None:
    expected = hashlib.sha256(config.read_bytes()).hexdigest()
    assert (
        supervisor.supervise(
            config,
            state=tmp_path / "state",
            declared=((config, expected),),
            run=scripted([supervisor.Outcome(0)], []),
        )
        == 0
    )


def test_unregistered_log_preserved(config: Path, tmp_path: Path) -> None:
    state = tmp_path / "state"
    supervisor.supervise(config, state=state, run=scripted([supervisor.Outcome(0)], []))
    (state / "attempt-0002.log").write_text("unknown child", encoding="utf-8")
    with pytest.raises(ValueError, match="readonly domain"):
        supervisor.supervise(config, state=state, run=scripted([], []))


def test_history_gap_rejected(config: Path, tmp_path: Path) -> None:
    state = tmp_path / "state"
    supervisor.supervise(config, state=state, run=scripted([supervisor.Outcome(0)], []))
    (state / "attempt-0001.json").rename(state / "attempt-0002.json")
    with pytest.raises(ValueError, match="contiguous"):
        supervisor.supervise(config, state=state, run=scripted([], []))


def test_input_mutation_marks_attempt_uncertain(config: Path, tmp_path: Path) -> None:
    def changed(
        command: Sequence[str], *, log: Path, timeout: int, started: Callable[[int], None]
    ) -> supervisor.Outcome:
        started(123456)
        config.write_text("changed", encoding="utf-8")
        return supervisor.Outcome(0)

    assert supervisor.supervise(config, state=tmp_path / "state", run=changed) == 1


@pytest.fixture
def technical(
    config: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> supervisor.Technical:
    monkeypatch.setattr(technical_batch, "load", lambda path, digest: None)
    return supervisor.Technical(
        "run",
        hashlib.sha256(config.read_bytes()).hexdigest(),
        tmp_path / "receipt.json",
        tmp_path / "technical-logs",
    )


def test_technical_original_run_success(
    config: Path, tmp_path: Path, technical: supervisor.Technical
) -> None:
    seen: list[Sequence[str]] = []

    def execute(
        command: Sequence[str], *, log: Path, timeout: int, started: Callable[[int], None]
    ) -> supervisor.Outcome:
        seen.append(command)
        started(123456)
        technical.receipt.write_text('{"verified":true}', encoding="utf-8")
        return supervisor.Outcome(0)

    assert (
        supervisor.supervise(config, state=tmp_path / "state", technical=technical, run=execute)
        == 0
    )
    assert seen[0][3:5] == ["replay.whole_technical_batch", "run"]
    assert Path(seen[0][-1]).name == "run-0001"
    assert "--start" not in seen[0]


def test_receipt_then_failure_reconciles_readonly(
    config: Path, tmp_path: Path, technical: supervisor.Technical
) -> None:
    seen: list[Sequence[str]] = []

    def execute(
        command: Sequence[str], *, log: Path, timeout: int, started: Callable[[int], None]
    ) -> supervisor.Outcome:
        seen.append(command)
        started(123456)
        technical.receipt.write_text('{"verified":true}', encoding="utf-8")
        return supervisor.Outcome(1 if len(seen) == 1 else 0)

    assert (
        supervisor.supervise(
            config,
            state=tmp_path / "state",
            technical=technical,
            run=execute,
            sleep=lambda seconds: None,
        )
        == 0
    )
    assert [command[4] for command in seen] == ["run", "verify"]
    assert [Path(command[-1]).name for command in seen] == ["run-0001", "verify-0002"]


def test_existing_receipt_only_verified(
    config: Path, tmp_path: Path, technical: supervisor.Technical
) -> None:
    technical.receipt.write_text('{"verified":true}', encoding="utf-8")
    seen: list[Sequence[str]] = []
    assert (
        supervisor.supervise(
            config,
            state=tmp_path / "state",
            technical=technical,
            run=scripted([supervisor.Outcome(0)], seen),
        )
        == 0
    )
    assert seen[0][4] == "verify"
    technical.receipt.write_text('{"verified":false}', encoding="utf-8")
    with pytest.raises(ValueError, match="receipt changed"):
        supervisor.supervise(
            config, state=tmp_path / "state", technical=technical, run=scripted([], [])
        )


def test_failed_readonly_receipt_not_accepted(
    config: Path, tmp_path: Path, technical: supervisor.Technical
) -> None:
    technical.receipt.write_text('{"forged":true}', encoding="utf-8")
    seen: list[Sequence[str]] = []
    assert (
        supervisor.supervise(
            config,
            state=tmp_path / "state",
            technical=technical,
            run=scripted([supervisor.Outcome(1)] * 3, seen),
            sleep=lambda seconds: None,
        )
        == 1
    )
    assert [command[4] for command in seen] == ["verify", "verify", "verify"]


def test_technical_receipt_and_logs_refusal(
    config: Path, tmp_path: Path, technical: supervisor.Technical
) -> None:
    technical.receipt.write_text("malformed", encoding="utf-8")
    with pytest.raises(ValueError):
        supervisor.supervise(
            config, state=tmp_path / "state", technical=technical, run=scripted([], [])
        )
    technical.receipt.unlink()
    technical.logs.mkdir()
    (technical.logs / "run-0001").mkdir()
    with pytest.raises(ValueError, match="occupied"):
        supervisor.supervise(
            config, state=tmp_path / "state", technical=technical, run=scripted([], [])
        )


def test_technical_manifest_fails_before_launch(
    config: Path, tmp_path: Path, technical: supervisor.Technical, monkeypatch: pytest.MonkeyPatch
) -> None:
    def invalid(path: Path, digest: str) -> None:
        raise ValueError("Manifest changed")

    monkeypatch.setattr(technical_batch, "load", invalid)
    with pytest.raises(ValueError, match="Manifest"):
        supervisor.supervise(
            config, state=tmp_path / "state", technical=technical, run=scripted([], [])
        )


def test_technical_options_require_explicit_mode(
    config: Path, technical: supervisor.Technical
) -> None:
    assert supervisor.main([str(config), "--technical-mode", "run"]) == 2
    assert supervisor.main([str(config), "--receipt", str(technical.receipt)]) == 2


def test_explicit_verify_requires_receipt(
    config: Path, tmp_path: Path, technical: supervisor.Technical
) -> None:
    verification = supervisor.Technical(
        "verify", technical.sha256, technical.receipt, technical.logs
    )
    with pytest.raises(FileNotFoundError):
        supervisor.supervise(
            config, state=tmp_path / "state", technical=verification, run=scripted([], [])
        )


def test_technical_does_not_mix_workflow(
    config: Path, tmp_path: Path, technical: supervisor.Technical
) -> None:
    with pytest.raises(ValueError, match="separate"):
        supervisor.supervise(
            config,
            state=tmp_path / "state",
            workflow_state=tmp_path / "jobs",
            technical=technical,
            run=scripted([], []),
        )


def test_technical_logs_do_not_overlap_state(
    config: Path, tmp_path: Path, technical: supervisor.Technical
) -> None:
    with pytest.raises(ValueError, match="separate"):
        supervisor.supervise(
            config, state=technical.logs, technical=technical, run=scripted([], [])
        )


def test_controller_closure_excludes_unrelated_code(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    code = tmp_path / "src/replay"
    code.mkdir(parents=True)
    (code / "__init__.py").write_text("# package\n", encoding="utf-8")
    (code / "pipeline_supervisor.py").write_text("from replay import helper\n", encoding="utf-8")
    (code / "helper.py").write_text("VALUE = 1\n", encoding="utf-8")
    config = tmp_path / "config.json"
    config.write_text("{}", encoding="utf-8")
    monkeypatch.setattr(supervisor, "ROOT", tmp_path)
    before = supervisor._pins(config, ())
    (code / "unrelated.py").write_text("VALUE = 3\n", encoding="utf-8")
    assert supervisor._pins(config, ()) == before
    (code / "unrelated.py").write_text("VALUE = 4\n", encoding="utf-8")
    assert supervisor._pins(config, ()) == before
    (code / "helper.py").write_text("VALUE = 2\n", encoding="utf-8")
    assert supervisor._pins(config, ()) != before


def test_declared_source_mutation_refused(config: Path, tmp_path: Path) -> None:
    source = tmp_path / "declared.py"
    source.write_text("VALUE = 1\n", encoding="utf-8")
    expected = hashlib.sha256(source.read_bytes()).hexdigest()
    supervisor._pins(config, ((source, expected),))
    source.write_text("VALUE = 2\n", encoding="utf-8")
    with pytest.raises(ValueError, match="changed"):
        supervisor._pins(config, ((source, expected),))


def test_technical_completed_reentry_does_not_consume_attempts(
    config: Path, tmp_path: Path, technical: supervisor.Technical
) -> None:
    technical.receipt.write_text('{"verified":true}', encoding="utf-8")
    state = tmp_path / "state"
    seen: list[Sequence[str]] = []
    supervisor.supervise(
        config, state=state, technical=technical, run=scripted([supervisor.Outcome(0)], seen)
    )
    assert (
        supervisor.supervise(
            config, state=state, technical=technical, run=scripted([supervisor.Outcome(0)], seen)
        )
        == 0
    )
    assert len(list(state.glob("attempt-*.json"))) == 1
    assert seen[1][4] == "verify"
    assert Path(seen[1][-1]).name == "reentry-verify-0001"
    assert (state / "reentry-0001.json").is_file()


def test_failed_completed_reentry_stays_uncertain(
    config: Path, tmp_path: Path, technical: supervisor.Technical
) -> None:
    technical.receipt.write_text('{"verified":true}', encoding="utf-8")
    state = tmp_path / "state"
    supervisor.supervise(
        config, state=state, technical=technical, run=scripted([supervisor.Outcome(0)], [])
    )
    assert (
        supervisor.supervise(
            config, state=state, technical=technical, run=scripted([supervisor.Outcome(1)], [])
        )
        == 1
    )
    with pytest.raises(ValueError, match="reentry is uncertain"):
        supervisor.supervise(config, state=state, technical=technical, run=scripted([], []))


def test_workflow_reentry_only_verifies_original_commands(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source, output = tmp_path / "input", tmp_path / "output"
    source.write_text("input", encoding="utf-8")
    output.write_text("output", encoding="utf-8")
    stage = Stage(
        name="chosen",
        command=("producer-must-not-run",),
        inputs=(source,),
        outputs=(output,),
        timeout_seconds=5,
        verify_command=("readonly-domain",),
    )
    config = tmp_path / "graph.json"
    config.write_text(
        json.dumps({"version": 1, "stages": [technical_batch._stage_json(stage)]}), encoding="utf-8"
    )
    jobs = tmp_path / "jobs"
    jobs.mkdir()
    receipt = jobs / "chosen.json"
    receipt.write_text(
        json.dumps(
            {
                "status": "complete",
                "fingerprint": _fingerprint(stage),
                "outputs": {str(output): hashlib.sha256(b"output").hexdigest()},
            }
        ),
        encoding="utf-8",
    )
    seen: list[tuple[str, ...]] = []

    def verify(command: tuple[str, ...], log: Path, timeout: int) -> None:
        seen.append(command)

    monkeypatch.setattr(supervisor, "execute_command", verify)
    state = tmp_path / "supervisor"
    assert (
        supervisor.supervise(
            config, state=state, workflow_state=jobs, run=scripted([supervisor.Outcome(0)], [])
        )
        == 0
    )
    assert supervisor.supervise(config, state=state, workflow_state=jobs, run=scripted([], [])) == 0
    assert seen == [("readonly-domain",)]
    assert len(list(state.glob("attempt-*.json"))) == 1
    output.write_text("changed", encoding="utf-8")
    with pytest.raises(ReplayError, match="changed"):
        supervisor.supervise(config, state=state, workflow_state=jobs, run=scripted([], []))
    assert seen == [("readonly-domain",)]


def test_a_retired_delivery_is_never_rendered_again(
    config: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(delivery_retention, "is_retired", lambda _config: True)
    seen: list[Sequence[str]] = []
    assert supervisor.supervise(config, state=tmp_path / "state", run=scripted([], seen)) == 0
    assert seen == []
