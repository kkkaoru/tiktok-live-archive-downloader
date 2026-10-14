import json
import sys
from pathlib import Path

import pytest

from replay import stage_recovery, workflow


def stage(tmp_path: Path, name: str, output: Path, *, fail: bool) -> dict[str, object]:
    source = tmp_path / f"{name}.input"
    source.write_text(name, encoding="utf-8")
    script = (
        "import sys; from pathlib import Path; p=Path(sys.argv[1]); "
        "p.parent.mkdir(exist_ok=True); p.write_text('partial' if sys.argv[2]=='1' else 'ok'); "
        "sys.exit(int(sys.argv[2]))"
    )
    return {
        "name": name,
        "command": [sys.executable, "-c", script, str(output), "1" if fail else "0"],
        "inputs": [str(source)],
        "outputs": [str(output)],
        "timeout_seconds": 10,
    }


def test_recover_keeps_completed_receipts_and_preserves_partial_outputs(tmp_path: Path) -> None:
    plan = tmp_path / "plan.json"
    good = stage(tmp_path, "good", tmp_path / "good" / "out.txt", fail=False)
    bad = stage(tmp_path, "bad", tmp_path / "bad" / "out.txt", fail=True)
    plan.write_text(json.dumps({"version": 1, "stages": [good, bad]}), encoding="utf-8")
    state = tmp_path / "state"
    assert workflow.main([str(plan), "--state", str(state)]) == 1
    assert stage_recovery.unfinished(plan, state) == ["bad"]
    preserved = stage_recovery.recover(plan, state)
    assert [path.name for path in preserved] == ["state.failed-0", "bad.failed-0"]
    assert (state / "good.json").is_file()
    assert not (state / "bad.json").exists()
    assert (tmp_path / "bad.failed-0" / "out.txt").read_text(encoding="utf-8") == "partial"
    assert stage_recovery.unfinished(plan, state) == []


def test_shared_output_directory_renames_only_the_file(tmp_path: Path) -> None:
    plan = tmp_path / "plan.json"
    first = stage(tmp_path, "first", tmp_path / "shared" / "a.txt", fail=False)
    second = stage(tmp_path, "second", tmp_path / "shared" / "b.txt", fail=True)
    plan.write_text(json.dumps({"version": 1, "stages": [first, second]}), encoding="utf-8")
    state = tmp_path / "state"
    workflow.main([str(plan), "--state", str(state)])
    preserved = stage_recovery.recover(plan, state)
    assert [path.name for path in preserved] == ["state.failed-0", "b.txt.failed-0"]
    assert (tmp_path / "shared" / "a.txt").is_file()


@pytest.mark.parametrize("nested", [False, True])
def test_recovery_preserves_workflow_inputs_and_existing_asides(
    tmp_path: Path, nested: bool
) -> None:
    plan = tmp_path / "plan.json"
    directory = tmp_path / "part"
    directory.mkdir()
    input_dir = directory / "nested" if nested else directory
    input_dir.mkdir(exist_ok=True)
    request = input_dir / "request.json"
    request.write_text("immutable request", encoding="utf-8")
    response = directory / "render.response.json"
    good = stage(tmp_path, "good", tmp_path / "good" / "out.txt", fail=False)
    bad = stage(tmp_path, "bad", response, fail=True)
    bad["inputs"] = [str(request)]
    plan.write_text(json.dumps({"version": 1, "stages": [good, bad]}), encoding="utf-8")
    state = tmp_path / "state"
    assert workflow.main([str(plan), "--state", str(state)]) == 1
    completed = (state / "good.json").read_bytes()
    aside = response.with_name("render.response.json.failed-0")
    aside.write_text("older evidence", encoding="utf-8")
    response.with_name("render.response.json.failed-1").symlink_to(directory / "missing")
    preserved = stage_recovery.recover(plan, state)
    assert [path.name for path in preserved] == ["state.failed-0", "render.response.json.failed-2"]
    assert request.read_text(encoding="utf-8") == "immutable request"
    assert aside.read_text(encoding="utf-8") == "older evidence"
    assert (state / "good.json").read_bytes() == completed
    assert (tmp_path / "good" / "out.txt").read_text(encoding="utf-8") == "ok"
    assert not response.exists()


def test_recovery_keeps_nested_completed_output(tmp_path: Path) -> None:
    plan = tmp_path / "plan.json"
    directory = tmp_path / "shared"
    directory.mkdir()
    good = stage(tmp_path, "good", directory / "nested" / "out.txt", fail=False)
    bad = stage(tmp_path, "bad", directory / "partial.txt", fail=True)
    plan.write_text(json.dumps({"version": 1, "stages": [good, bad]}), encoding="utf-8")
    state = tmp_path / "state"
    assert workflow.main([str(plan), "--state", str(state)]) == 1
    completed = (state / "good.json").read_bytes()
    preserved = stage_recovery.recover(plan, state)
    assert [path.name for path in preserved] == ["state.failed-0", "partial.txt.failed-0"]
    assert (directory / "nested" / "out.txt").read_text(encoding="utf-8") == "ok"
    assert (state / "good.json").read_bytes() == completed


def test_recovery_keeps_lexical_symlink_input(tmp_path: Path) -> None:
    plan = tmp_path / "plan.json"
    directory = tmp_path / "part"
    directory.mkdir()
    request = tmp_path / "request.json"
    request.write_text("immutable", encoding="utf-8")
    link = directory / "request.json"
    link.symlink_to(request)
    response = directory / "render.response.json"
    bad = stage(tmp_path, "bad", response, fail=True)
    bad["inputs"] = [str(link)]
    plan.write_text(json.dumps({"version": 1, "stages": [bad]}), encoding="utf-8")
    state = tmp_path / "state"
    state.mkdir()
    (state / "bad.json").write_text('{"status": "running"}', encoding="utf-8")
    response.write_text("partial", encoding="utf-8")
    preserved = stage_recovery.recover(plan, state)
    assert [path.name for path in preserved] == ["state.failed-0", "render.response.json.failed-0"]
    assert link.is_symlink()
    assert link.read_text(encoding="utf-8") == "immutable"


def test_nothing_to_recover(tmp_path: Path) -> None:
    plan = tmp_path / "plan.json"
    plan.write_text(json.dumps({"version": 1, "stages": []}), encoding="utf-8")
    assert stage_recovery.recover(plan, tmp_path / "missing-state") == []


def test_invalid_workflow_is_reported(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    plan = tmp_path / "plan.json"
    plan.write_text('{"version": 1}', encoding="utf-8")
    state = tmp_path / "state"
    state.mkdir()
    assert stage_recovery.main([str(plan), "--state", str(state)]) == 1
    assert "STAGE_RECOVERY_ERROR" in capsys.readouterr().err


def test_cli_prints_preserved_paths(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    plan = tmp_path / "plan.json"
    bad = stage(tmp_path, "bad", tmp_path / "bad" / "out.txt", fail=True)
    plan.write_text(json.dumps({"version": 1, "stages": [bad]}), encoding="utf-8")
    state = tmp_path / "state"
    workflow.main([str(plan), "--state", str(state)])
    assert stage_recovery.main([str(plan), "--state", str(state)]) == 0
    assert capsys.readouterr().out.splitlines()[0].endswith("state.failed-0")
