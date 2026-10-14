"""Graph selection, preflight isolation and deterministic bounded compilation."""

import fcntl
import io
import json
import os
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import pytest

from replay import workflow
from replay.stages import Stage, StageError


@pytest.fixture
def diamond(tmp_path: Path) -> tuple[Path, list[dict[str, object]]]:
    (tmp_path / "source").write_text("source", encoding="utf-8")
    definitions = [
        ("finish", ["left", "right"]),
        ("right", ["root"]),
        ("left", ["root"]),
        ("root", ["source"]),
        ("unrelated", ["source"]),
    ]
    entries: list[dict[str, object]] = [
        {
            "name": name,
            "inputs": [str(tmp_path / parent) for parent in parents],
            "outputs": [str(tmp_path / name)],
            "command": [
                sys.executable,
                "-c",
                "import sys; from pathlib import Path; Path(sys.argv[1]).write_text('ok')",
                str(tmp_path / name),
            ],
            "timeout_seconds": 10,
        }
        for name, parents in definitions
    ]
    plan = tmp_path / "plan.json"
    plan.write_text(json.dumps({"version": 1, "stages": entries}), encoding="utf-8")
    return plan, entries


def test_target_keeps_entire_diamond_and_reuses_receipts(
    diamond: tuple[Path, list[dict[str, object]]],
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    plan, entries = diamond
    entries[4]["inputs"] = [str(tmp_path / "missing")]
    plan.write_text(json.dumps({"version": 1, "stages": entries}), encoding="utf-8")
    workflow.run_plan(plan, state_directory=tmp_path / "state", targets=("finish",))
    assert capsys.readouterr().out == (
        "root: completed\nright: completed\nleft: completed\nfinish: completed\n"
    )
    workflow.run_plan(
        plan, state_directory=tmp_path / "state", targets=("finish", "right", "finish")
    )
    assert capsys.readouterr().out == (
        "root: reused\nright: reused\nleft: reused\nfinish: reused\n"
    )
    assert not (tmp_path / "unrelated").exists()
    assert not (tmp_path / "state/unrelated.json").exists()
    assert (tmp_path / "finish").read_text(encoding="utf-8") == "ok"


def test_target_cli_check_and_execution_agree(
    diamond: tuple[Path, list[dict[str, object]]],
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    plan, _entries = diamond
    arguments = [
        str(plan),
        "--state",
        str(tmp_path / "state"),
        "--target",
        "left",
        "--target",
        "right",
    ]
    assert workflow.main([*arguments, "--check"]) == 0
    assert capsys.readouterr().out == '{"left": ["root"], "right": ["root"], "root": []}\n'
    assert not (tmp_path / "state").exists()
    assert workflow.main(arguments) == 0
    assert capsys.readouterr().out == "root: completed\nright: completed\nleft: completed\n"
    assert not (tmp_path / "finish").exists()


def test_unknown_target_creates_nothing(
    diamond: tuple[Path, list[dict[str, object]]], tmp_path: Path
) -> None:
    plan, _entries = diamond
    with pytest.raises(StageError, match="Unknown workflow target"):
        workflow.run_plan(plan, state_directory=tmp_path / "state", targets=("typo",))
    assert not (tmp_path / "state").exists()
    assert not (tmp_path / "root").exists()


def test_unselected_cycle_still_fails_whole_graph(
    diamond: tuple[Path, list[dict[str, object]]], tmp_path: Path
) -> None:
    plan, entries = diamond
    entries[4]["inputs"] = [str(tmp_path / "finish")]
    entries[0]["inputs"] = [str(tmp_path / "unrelated")]
    plan.write_text(json.dumps({"version": 1, "stages": entries}), encoding="utf-8")
    with pytest.raises(StageError, match="cycle"):
        workflow.run_plan(plan, state_directory=tmp_path / "state", targets=("right",))
    assert not (tmp_path / "state").exists()


@pytest.mark.parametrize("kind", ["missing", "empty", "directory", "symlink", "fifo"])
def test_external_input_fails_before_any_mutation(
    diamond: tuple[Path, list[dict[str, object]]], tmp_path: Path, kind: str
) -> None:
    plan, entries = diamond
    source = tmp_path / "bad"
    if kind == "empty":
        source.touch()
    elif kind == "directory":
        source.mkdir()
    elif kind == "symlink":
        source.symlink_to(tmp_path / "source")
    elif kind == "fifo":
        os.mkfifo(source)
    entries[4]["inputs"] = [str(source)]
    plan.write_text(json.dumps({"version": 1, "stages": entries}), encoding="utf-8")
    with pytest.raises(StageError, match="external input"):
        workflow.run_plan(plan, state_directory=tmp_path / "state")
    assert not (tmp_path / "root").exists()
    assert not (tmp_path / "state").exists()


@pytest.mark.parametrize("field", ["inputs", "outputs"])
def test_artifacts_cannot_alias_state_namespace(
    diamond: tuple[Path, list[dict[str, object]]], tmp_path: Path, field: str
) -> None:
    plan, entries = diamond
    entries[4][field] = [str(tmp_path / "state/root.json")]
    plan.write_text(json.dumps({"version": 1, "stages": entries}), encoding="utf-8")
    with pytest.raises(StageError, match="separate from state"):
        workflow.run_plan(plan, state_directory=tmp_path / "state", targets=("root",))
    assert not (tmp_path / "state").exists()


def test_state_cannot_be_below_a_file_artifact(
    diamond: tuple[Path, list[dict[str, object]]], tmp_path: Path
) -> None:
    plan, _entries = diamond
    with pytest.raises(StageError, match="separate from state"):
        workflow.run_plan(plan, state_directory=tmp_path / "root/state")
    assert not (tmp_path / "root").exists()


def test_output_cannot_overwrite_control_manifest(
    diamond: tuple[Path, list[dict[str, object]]], tmp_path: Path
) -> None:
    plan, entries = diamond
    entries[4]["outputs"] = [str(plan)]
    plan.write_text(json.dumps({"version": 1, "stages": entries}), encoding="utf-8")
    before = plan.read_bytes()
    with pytest.raises(StageError, match="overwrite its manifest"):
        workflow.run_plan(plan, state_directory=tmp_path / "state")
    assert plan.read_bytes() == before
    assert not (tmp_path / "state").exists()


@pytest.mark.parametrize("field", ["inputs", "outputs"])
def test_file_ancestor_collisions_are_rejected(
    diamond: tuple[Path, list[dict[str, object]]], tmp_path: Path, field: str
) -> None:
    plan, entries = diamond
    entries[4][field] = [str(tmp_path / "root/child")]
    plan.write_text(json.dumps({"version": 1, "stages": entries}), encoding="utf-8")
    with pytest.raises(StageError, match="ancestor and descendant"):
        workflow.run_plan(plan, state_directory=tmp_path / "state")
    assert not (tmp_path / "root").exists()
    assert not (tmp_path / "state").exists()


@pytest.mark.parametrize("kind", ["hardlink", "directory", "symlink", "fifo"])
def test_unsafe_output_is_preserved_before_any_execution(
    diamond: tuple[Path, list[dict[str, object]]], tmp_path: Path, kind: str
) -> None:
    plan, _entries = diamond
    output = tmp_path / "unrelated"
    if kind == "hardlink":
        output.hardlink_to(tmp_path / "source")
    elif kind == "directory":
        output.mkdir()
    elif kind == "symlink":
        output.symlink_to(tmp_path / "unclaimed")
    else:
        os.mkfifo(output)
    identity = output.lstat().st_ino
    with pytest.raises(StageError, match="Workflow output"):
        workflow.run_plan(plan, state_directory=tmp_path / "state", targets=("root",))
    assert output.lstat().st_ino == identity
    assert (tmp_path / "source").read_bytes() == b"source"
    assert not (tmp_path / "state").exists()


def test_workflow_lease_prevents_interleaved_stage_execution(
    diamond: tuple[Path, list[dict[str, object]]], tmp_path: Path
) -> None:
    plan, _entries = diamond
    state = tmp_path / "state"
    state.mkdir()
    with (state / ".workflow.lock").open("wb") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with pytest.raises(StageError, match="Another process owns this workflow"):
            workflow.run_plan(plan, state_directory=state)
        assert not (tmp_path / "root").exists()
        assert not (state / "root.lock").exists()
        result = subprocess.run(
            [sys.executable, "-m", "replay.workflow", str(plan), "--state", str(state)],
            check=False,
            capture_output=True,
            text=True,
            timeout=10,
        )
        assert result.returncode == 1
        assert result.stdout == ""
        assert "StageError" in result.stderr
        assert not (state / "root.lock").exists()
    workflow.run_plan(plan, state_directory=state, targets=("root",))
    assert (tmp_path / "root").read_bytes() == b"ok"
    assert (state / ".workflow.lock").stat().st_mode & 0o777 == 0o600


def test_target_does_not_bypass_changed_completed_output(
    diamond: tuple[Path, list[dict[str, object]]], tmp_path: Path
) -> None:
    plan, _entries = diamond
    workflow.run_plan(plan, state_directory=tmp_path / "state", targets=("root",))
    (tmp_path / "root").write_bytes(b"changed")
    with pytest.raises(StageError, match="changed"):
        workflow.run_plan(plan, state_directory=tmp_path / "state", targets=("finish",))
    assert (tmp_path / "root").read_bytes() == b"changed"
    assert not (tmp_path / "right").exists()
    assert not (tmp_path / "finish").exists()


def test_manifest_growth_is_bounded(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    plan = tmp_path / "plan.json"
    plan.write_text("{}", encoding="utf-8")
    monkeypatch.setattr(workflow, "MAX_PLAN_BYTES", 4)
    monkeypatch.setattr(Path, "open", lambda *_args, **_kwargs: io.BytesIO(b"123456"))
    with pytest.raises(StageError, match="manifest file"):
        workflow.load_stages(plan)


def test_graph_compiles_once_per_cli_invocation(
    diamond: tuple[Path, list[dict[str, object]]],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    plan, _entries = diamond
    original = workflow._dependency_graph
    calls: list[int] = []

    def counted(stages: tuple[Stage, ...], paths: dict[Path, Path]) -> dict[str, tuple[str, ...]]:
        calls.append(len(stages))
        return original(stages, paths)

    monkeypatch.setattr(workflow, "_dependency_graph", counted)
    assert workflow.main([str(plan), "--state", str(tmp_path / "state")]) == 0
    assert calls == [5]


def test_shared_paths_are_resolved_once_per_graph(
    diamond: tuple[Path, list[dict[str, object]]], monkeypatch: pytest.MonkeyPatch
) -> None:
    plan, _entries = diamond
    stages = workflow.load_stages(plan)
    original = Path.resolve
    calls: list[Path] = []

    def counted(path: Path, strict: bool = False) -> Path:
        calls.append(path)
        return original(path, strict=strict)

    monkeypatch.setattr(Path, "resolve", counted)
    assert workflow.stage_graph(stages) == {
        "finish": ("left", "right"),
        "right": ("root",),
        "left": ("root",),
        "root": (),
        "unrelated": (),
    }
    assert len(calls) == 6
    assert len(set(calls)) == 6


def test_public_graph_rejects_duplicate_names(
    diamond: tuple[Path, list[dict[str, object]]], tmp_path: Path
) -> None:
    plan, _entries = diamond
    first = workflow.load_stages(plan)[0]
    with pytest.raises(StageError, match="unique"):
        workflow.stage_graph((first, replace(first, outputs=(tmp_path / "other",))))


def test_long_chain_selection_is_iterative(tmp_path: Path) -> None:
    (tmp_path / "source").write_text("source", encoding="utf-8")
    entries = [
        {
            "name": f"s{i}",
            "command": ["unused"],
            "inputs": [str(tmp_path / (f"o{i - 1}" if i else "source"))],
            "outputs": [str(tmp_path / f"o{i}")],
            "timeout_seconds": 10,
        }
        for i in range(512)
    ]
    stages = workflow.parse_stages({"version": 1, "stages": entries})
    graph = workflow.stage_graph(stages)
    selected = workflow._selected_graph(graph, ("s511",))
    assert len(selected) == 512
    assert selected["s0"] == ()
    assert selected["s511"] == ("s510",)


@pytest.mark.parametrize("kind", ["directory", "fifo"])
def test_nonregular_manifest_does_not_block(tmp_path: Path, kind: str) -> None:
    plan = tmp_path / "plan.json"
    if kind == "directory":
        plan.mkdir()
    else:
        os.mkfifo(plan)
    with pytest.raises(StageError, match="manifest file"):
        workflow.load_stages(plan)
