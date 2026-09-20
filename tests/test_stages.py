"""Synthetic stage execution, immutable resume and interrupted-output recovery."""

import fcntl
import json
import os
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import pytest

from replay.stages import Stage, StageError, execute_command, file_digest, run_stage


@pytest.fixture
def stage(tmp_path: Path) -> Stage:
    source = tmp_path / "input"
    source.write_bytes(b"source")
    return Stage(
        name="encode",
        command=("work",),
        inputs=(source,),
        outputs=(tmp_path / "output",),
        timeout_seconds=10,
    )


def test_complete_and_resume_without_reexecution(stage: Stage, tmp_path: Path) -> None:
    calls: list[tuple[str, ...]] = []

    def execute(command: tuple[str, ...], _log: Path, _timeout: int) -> None:
        calls.append(command)
        stage.outputs[0].write_bytes(b"result")

    assert run_stage(stage, state_directory=tmp_path / "state", execute=execute) == "completed"
    assert run_stage(stage, state_directory=tmp_path / "state", execute=execute) == "reused"
    assert calls == [("work",)]
    assert stage.outputs[0].read_bytes() == b"result"


def test_changed_completed_input_stops(stage: Stage, tmp_path: Path) -> None:
    def execute(_command: tuple[str, ...], _log: Path, _timeout: int) -> None:
        stage.outputs[0].write_bytes(b"result")

    run_stage(stage, state_directory=tmp_path / "state", execute=execute)
    stage.inputs[0].write_bytes(b"changed")
    with pytest.raises(StageError, match="changed"):
        run_stage(stage, state_directory=tmp_path / "state", execute=execute)


def test_changed_completed_output_stops(stage: Stage, tmp_path: Path) -> None:
    def execute(_command: tuple[str, ...], _log: Path, _timeout: int) -> None:
        stage.outputs[0].write_bytes(b"result")

    run_stage(stage, state_directory=tmp_path / "state", execute=execute)
    stage.outputs[0].write_bytes(b"changed")
    with pytest.raises(StageError, match="changed"):
        run_stage(stage, state_directory=tmp_path / "state", execute=execute)


def test_missing_completed_output_is_not_recreated(stage: Stage, tmp_path: Path) -> None:
    calls: list[tuple[str, ...]] = []

    def execute(command: tuple[str, ...], _log: Path, _timeout: int) -> None:
        calls.append(command)
        stage.outputs[0].write_bytes(b"result")

    run_stage(stage, state_directory=tmp_path / "state", execute=execute)
    stage.outputs[0].unlink()
    with pytest.raises(FileNotFoundError):
        run_stage(stage, state_directory=tmp_path / "state", execute=execute)
    assert calls == [("work",)]


def test_partial_output_cannot_trigger_a_mutating_retry(stage: Stage, tmp_path: Path) -> None:
    def interrupted(_command: tuple[str, ...], _log: Path, _timeout: int) -> None:
        stage.outputs[0].write_bytes(b"partial")
        raise StageError("Interrupted")

    with pytest.raises(StageError, match="Interrupted"):
        run_stage(stage, state_directory=tmp_path / "state", execute=interrupted)
    with pytest.raises(StageError, match="provenance verifier"):
        run_stage(stage, state_directory=tmp_path / "state", execute=interrupted)
    assert stage.outputs[0].read_bytes() == b"partial"


def test_interrupted_stage_is_recovered_only_by_verification(stage: Stage, tmp_path: Path) -> None:
    checked = replace(stage, verify_command=("check",))

    def interrupted(_command: tuple[str, ...], _log: Path, _timeout: int) -> None:
        stage.outputs[0].write_bytes(b"result")
        raise StageError("Interrupted")

    with pytest.raises(StageError, match="Interrupted"):
        run_stage(checked, state_directory=tmp_path / "state", execute=interrupted)
    calls: list[tuple[str, ...]] = []

    def verify(command: tuple[str, ...], _log: Path, _timeout: int) -> None:
        calls.append(command)
        assert stage.outputs[0].read_bytes() == b"result"

    assert run_stage(checked, state_directory=tmp_path / "state", execute=verify) == "adopted"
    assert calls == [("check",)]


def test_unregistered_output_requires_explicit_adoption(stage: Stage, tmp_path: Path) -> None:
    stage.outputs[0].write_bytes(b"result")
    with pytest.raises(StageError, match="explicit verified adoption"):
        run_stage(replace(stage, verify_command=("check",)), state_directory=tmp_path / "state")


def test_registered_adoption_validates_without_mutation(stage: Stage, tmp_path: Path) -> None:
    stage.outputs[0].write_bytes(b"result")
    calls: list[tuple[str, ...]] = []

    def verify(command: tuple[str, ...], _log: Path, _timeout: int) -> None:
        calls.append(command)

    checked = replace(stage, verify_command=("check",), adopt_existing=True)
    assert run_stage(checked, state_directory=tmp_path / "state", execute=verify) == "adopted"
    assert calls == [("check",)]


def test_verifier_failure_preserves_running_receipt(stage: Stage, tmp_path: Path) -> None:
    stage.outputs[0].write_bytes(b"partial")

    def fail(_command: tuple[str, ...], _log: Path, _timeout: int) -> None:
        raise StageError("Unverified")

    checked = replace(stage, verify_command=("check",), adopt_existing=True)
    with pytest.raises(StageError, match="Unverified"):
        run_stage(checked, state_directory=tmp_path / "state", execute=fail)
    saved: object = json.loads((tmp_path / "state/encode.json").read_text(encoding="utf-8"))
    assert isinstance(saved, dict)
    assert saved["status"] == "running"


def test_new_stage_runs_then_verifies(stage: Stage, tmp_path: Path) -> None:
    calls: list[tuple[str, ...]] = []

    def execute(command: tuple[str, ...], _log: Path, _timeout: int) -> None:
        calls.append(command)
        if command == ("work",):
            stage.outputs[0].write_bytes(b"result")

    checked = replace(stage, verify_command=("check",))
    assert run_stage(checked, state_directory=tmp_path / "state", execute=execute) == "completed"
    assert calls == [("work",), ("check",)]


def test_mutating_input_is_detected(stage: Stage, tmp_path: Path) -> None:
    def execute(_command: tuple[str, ...], _log: Path, _timeout: int) -> None:
        stage.inputs[0].write_bytes(b"changed")
        stage.outputs[0].write_bytes(b"result")

    with pytest.raises(StageError, match="immutable inputs"):
        run_stage(stage, state_directory=tmp_path / "state", execute=execute)


def test_cross_process_lock_refuses_duplicate(stage: Stage, tmp_path: Path) -> None:
    state = tmp_path / "state"
    state.mkdir()
    with (state / "encode.lock").open("a+b") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        assert run_stage(stage, state_directory=state) == "busy"
    assert not stage.outputs[0].exists()


def test_state_directory_symlink_is_refused(stage: Stage, tmp_path: Path) -> None:
    target = tmp_path / "target"
    target.mkdir()
    alias = tmp_path / "state"
    alias.symlink_to(target, target_is_directory=True)
    with pytest.raises(StageError, match="symlink"):
        run_stage(stage, state_directory=alias)


def test_lock_symlink_is_not_followed(stage: Stage, tmp_path: Path) -> None:
    state = tmp_path / "state"
    state.mkdir()
    (state / "encode.lock").symlink_to(stage.inputs[0])
    with pytest.raises(OSError):
        run_stage(stage, state_directory=state)
    assert stage.inputs[0].read_bytes() == b"source"


@pytest.mark.parametrize("text", ["null", "[]", "{}", '{"status":"other"}'])
def test_invalid_receipt_cannot_look_like_a_new_stage(
    stage: Stage, tmp_path: Path, text: str
) -> None:
    state = tmp_path / "state"
    state.mkdir()
    (state / "encode.json").write_text(text, encoding="utf-8")
    with pytest.raises(StageError, match="schema"):
        run_stage(stage, state_directory=state)


def test_invalid_receipt_fingerprint(stage: Stage, tmp_path: Path) -> None:
    state = tmp_path / "state"
    state.mkdir()
    (state / "encode.json").write_text('{"status":"running","fingerprint":1}', encoding="utf-8")
    with pytest.raises(StageError, match="fingerprint"):
        run_stage(stage, state_directory=state)


def test_receipt_symlink_is_refused(stage: Stage, tmp_path: Path) -> None:
    state = tmp_path / "state"
    state.mkdir()
    (state / "encode.json").symlink_to(stage.inputs[0])
    with pytest.raises(StageError, match="receipt"):
        run_stage(stage, state_directory=state)


def test_oversized_receipt_is_refused(stage: Stage, tmp_path: Path) -> None:
    state = tmp_path / "state"
    state.mkdir()
    (state / "encode.json").write_bytes(b"x" * 1_048_577)
    with pytest.raises(StageError, match="receipt"):
        run_stage(stage, state_directory=state)


@pytest.mark.parametrize("name", ["../escape", "", "Upper", "x" * 65])
def test_invalid_stage_name(stage: Stage, name: str) -> None:
    with pytest.raises(StageError, match="name"):
        replace(stage, name=name)


@pytest.mark.parametrize("command", [(), ("",), ("bad\0value",)])
def test_invalid_command(stage: Stage, command: tuple[str, ...]) -> None:
    with pytest.raises(StageError, match="argument vector"):
        replace(stage, command=command)


def test_invalid_verifier(stage: Stage) -> None:
    with pytest.raises(StageError, match="verification"):
        replace(stage, verify_command=("",))


@pytest.mark.parametrize("timeout", [0, -1, True, 86401])
def test_invalid_timeout(stage: Stage, timeout: int) -> None:
    with pytest.raises(StageError, match="timeout"):
        replace(stage, timeout_seconds=timeout)


def test_empty_inputs(stage: Stage) -> None:
    with pytest.raises(StageError, match="Explicit"):
        replace(stage, inputs=())


def test_empty_outputs(stage: Stage) -> None:
    with pytest.raises(StageError, match="Explicit"):
        replace(stage, outputs=())


def test_relative_paths(stage: Stage) -> None:
    with pytest.raises(StageError, match="absolute"):
        replace(stage, outputs=(Path("relative"),))


def test_input_output_aliases_are_refused(stage: Stage) -> None:
    with pytest.raises(StageError, match="distinct"):
        replace(stage, outputs=stage.inputs)


def test_lexical_aliases_are_refused(stage: Stage, tmp_path: Path) -> None:
    with pytest.raises(StageError, match="distinct"):
        replace(stage, outputs=(tmp_path / "sub/../input",))


def test_adoption_without_verifier_is_refused(stage: Stage) -> None:
    with pytest.raises(StageError, match="Adoption"):
        replace(stage, adopt_existing=True)


def test_nonempty_regular_file_digest(tmp_path: Path) -> None:
    path = tmp_path / "data"
    path.write_bytes(b"abc")
    assert file_digest(path) == "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"


def test_empty_artifact_is_refused(tmp_path: Path) -> None:
    path = tmp_path / "empty"
    path.touch()
    with pytest.raises(StageError, match="nonempty"):
        file_digest(path)


def test_artifact_symlink_is_refused(tmp_path: Path) -> None:
    path = tmp_path / "data"
    path.write_bytes(b"abc")
    alias = tmp_path / "alias"
    alias.symlink_to(path)
    with pytest.raises(StageError, match="regular"):
        file_digest(alias)


def test_hash_rejects_opened_file_identity_race(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "data"
    path.write_bytes(b"abc")
    original = path.stat()
    changed = os.stat_result((original.st_mode, 0, 0, 1, 0, 0, 3, 0, 0, 0))
    monkeypatch.setattr(os, "fstat", lambda _fd: changed)
    with pytest.raises(StageError, match="before hashing"):
        file_digest(path)


def test_hash_rejects_file_change_after_read(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "data"
    path.write_bytes(b"abc")
    original = path.stat()
    changed = os.stat_result(
        (original.st_mode, original.st_ino, original.st_dev, 1, 0, 0, 4, 0, 0, 0)
    )
    states = [original, changed]
    monkeypatch.setattr(Path, "lstat", lambda _path: states.pop(0))
    with pytest.raises(StageError, match="during hashing"):
        file_digest(path)


def test_execute_command_captures_output(tmp_path: Path) -> None:
    log = tmp_path / "run.log"
    execute_command((sys.executable, "-c", "print('ok')"), log, 10)
    assert log.read_text(encoding="utf-8") == "ok\n"
    assert log.stat().st_mode & 0o777 == 0o600


def test_execute_command_failure_is_explicit(tmp_path: Path) -> None:
    with pytest.raises(StageError, match="command failed"):
        execute_command((sys.executable, "-c", "raise SystemExit(7)"), tmp_path / "run.log", 10)


def test_execute_command_deadline(tmp_path: Path) -> None:
    with pytest.raises(subprocess.TimeoutExpired):
        execute_command(
            (sys.executable, "-c", "import time; time.sleep(20)"), tmp_path / "run.log", 1
        )


def test_cached_manifest_still_verifies_referenced_artifacts(stage: Stage, tmp_path: Path) -> None:
    piece = tmp_path / "referenced-piece"
    piece.write_bytes(b"piece")
    calls: list[tuple[str, ...]] = []

    def execute(command: tuple[str, ...], _log: Path, _timeout: int) -> None:
        calls.append(command)
        if command == ("work",):
            stage.outputs[0].write_bytes(b"manifest")
        elif piece.read_bytes() != b"piece":
            raise StageError("Referenced artifact changed")

    checked = replace(stage, verify_command=("check",))
    run_stage(checked, state_directory=tmp_path / "state", execute=execute)
    assert run_stage(checked, state_directory=tmp_path / "state", execute=execute) == "reused"
    piece.write_bytes(b"changed")
    with pytest.raises(StageError, match="Referenced artifact"):
        run_stage(checked, state_directory=tmp_path / "state", execute=execute)
    assert calls == [("work",), ("check",), ("check",), ("check",)]


def test_working_directory_is_part_of_checkpoint_identity(
    stage: Stage, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def execute(_command: tuple[str, ...], _log: Path, _timeout: int) -> None:
        stage.outputs[0].write_bytes(b"result")

    run_stage(stage, state_directory=tmp_path / "state", execute=execute)
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    monkeypatch.chdir(elsewhere)
    with pytest.raises(StageError, match="changed"):
        run_stage(stage, state_directory=tmp_path / "state", execute=execute)
