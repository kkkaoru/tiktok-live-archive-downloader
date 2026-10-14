"""Trusted manifest parsing and headless CLI behavior with synthetic artifacts."""

import json
import subprocess
import sys
from pathlib import Path

import pytest

from replay import workflow
from replay.core import ReplayError
from replay.stages import Stage, StageError


@pytest.mark.parametrize(
    "declaration",
    [
        None,
        {},
        {"bindings": {}},
        {"bindings": []},
        {"bindings": {"unknown": "binding.json"}},
        {"bindings": {"create": ""}},
        {"bindings": {"create": "missing.json"}},
    ],
)
def test_invalid_source_caption_binding_declarations(
    entry: dict[str, object], declaration: object
) -> None:
    with pytest.raises(StageError):
        workflow.parse_stages(
            {"version": 1, "stages": [entry], "sourceCaptionProduction": declaration}
        )


def test_all_known_render_consumers_require_bindings(
    entry: dict[str, object], tmp_path: Path
) -> None:
    binding = str(tmp_path / "binding.json")
    entry = {**entry, "inputs": [binding]}
    for module in workflow.SOURCE_CAPTION_CONSUMERS:
        consumer = {
            **entry,
            "name": "render",
            "command": [sys.executable, "-m", module],
            "outputs": [str(tmp_path / "different-output")],
        }
        with pytest.raises(StageError, match="lacks a source-caption binding"):
            workflow.parse_stages(
                {
                    "version": 1,
                    "stages": [entry, consumer],
                    "sourceCaptionProduction": {"bindings": {"create": binding}},
                }
            )


@pytest.mark.parametrize("outcome", ["new", "reused", "adopted", "failed"])
def test_source_caption_binding_runs_before_every_consumer(
    entry: dict[str, object], tmp_path: Path, monkeypatch: pytest.MonkeyPatch, outcome: str
) -> None:
    from replay import source_caption_binding

    binding = tmp_path / "binding.json"
    clean = tmp_path / "clean.mov"
    events: list[str] = []
    producer = Stage(
        name="repair",
        command=("repair",),
        inputs=(tmp_path / "input",),
        outputs=(binding, clean),
        timeout_seconds=10,
    )
    consumer = Stage(
        name="render",
        command=("render", str(clean)),
        inputs=(binding, clean),
        outputs=(tmp_path / "render.mov",),
        timeout_seconds=10,
        verify_command=("verify",),
        adopt_existing=outcome == "adopted",
    )

    def validate(
        path: Path, *, consumer_inputs: tuple[Path, ...], consumer_command: tuple[str, ...]
    ) -> None:
        events.append("validate")
        assert path == binding and binding.is_file() and clean.is_file()
        assert consumer_inputs == (binding, clean)
        assert consumer_command == consumer.command
        if outcome == "failed":
            raise ReplayError("repair failed")

    def execute(stage: Stage, *, state_directory: Path) -> str:
        events.append(stage.name)
        if stage.name == "repair":
            binding.write_text("synthetic binding")
            clean.write_bytes(b"synthetic media")
        return outcome

    monkeypatch.setattr(source_caption_binding, "validate_binding", validate)
    monkeypatch.setattr(workflow, "run_stage", execute)
    kwargs = {"render": binding}
    if outcome == "failed":
        with pytest.raises(StageError, match="repair binding failed"):
            workflow._run_selected(
                (producer, consumer),
                graph={"repair": (), "render": ("repair",)},
                state=tmp_path / "state",
                source_caption_bindings=kwargs,
            )
        assert events == ["repair", "validate"]
    else:
        workflow._run_selected(
            (producer, consumer),
            graph={"repair": (), "render": ("repair",)},
            state=tmp_path / "state",
            source_caption_bindings=kwargs,
        )
        assert events == ["repair", "validate", "render"]


def test_source_caption_check_accepts_a_binding_produced_upstream(tmp_path: Path) -> None:
    binding = str(tmp_path / "binding.json")
    plan = tmp_path / "plan.json"
    plan.write_text(
        json.dumps(
            {
                "version": 1,
                "sourceCaptionProduction": {"bindings": {"render": binding}},
                "stages": [
                    {
                        "name": "repair",
                        "command": ["repair"],
                        "inputs": [str(tmp_path / "source")],
                        "outputs": [binding],
                        "timeout_seconds": 10,
                    },
                    {
                        "name": "render",
                        "command": [sys.executable, "-m", "replay.part_render"],
                        "inputs": [binding],
                        "outputs": [str(tmp_path / "output")],
                        "timeout_seconds": 10,
                    },
                ],
            }
        )
    )
    assert workflow.main([str(plan), "--state", str(tmp_path / "state"), "--check"]) == 0
    assert not Path(binding).exists()


@pytest.fixture
def entry(tmp_path: Path) -> dict[str, object]:
    source = tmp_path / "input"
    source.write_bytes(b"source")
    return {
        "name": "create",
        "command": [
            sys.executable,
            "-c",
            "import sys; from pathlib import Path; Path(sys.argv[1]).write_text('result')",
            str(tmp_path / "output"),
        ],
        "inputs": [str(source)],
        "outputs": [str(tmp_path / "output")],
        "timeout_seconds": 10,
    }


def test_run_whole_plan_and_idempotent_resume(entry: dict[str, object], tmp_path: Path) -> None:
    plan = tmp_path / "plan.json"
    plan.write_text(json.dumps({"version": 1, "stages": [entry]}), encoding="utf-8")
    workflow.run_plan(plan, state_directory=tmp_path / "state")
    assert (tmp_path / "output").read_text(encoding="utf-8") == "result"
    workflow.run_plan(plan, state_directory=tmp_path / "state")
    assert (tmp_path / "output").read_text(encoding="utf-8") == "result"


def test_cli_success(entry: dict[str, object], tmp_path: Path) -> None:
    plan = tmp_path / "plan.json"
    plan.write_text(json.dumps({"version": 1, "stages": [entry]}), encoding="utf-8")
    assert workflow.main([str(plan), "--state", str(tmp_path / "state")]) == 0


def test_cli_failure_does_not_print_configuration(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    plan = tmp_path / "plan.json"
    plan.write_text('{"private":"never-log-this"}', encoding="utf-8")
    assert workflow.main([str(plan), "--state", str(tmp_path / "state")]) == 1
    captured = capsys.readouterr()
    assert "StageError" in captured.err
    assert "never-log-this" not in captured.err


def test_timeout_arguments_are_not_logged(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    def fail(_plan: Path, *, state_directory: Path, targets: tuple[str, ...]) -> None:
        raise subprocess.TimeoutExpired(("command", "never-log-this"), 1)

    monkeypatch.setattr(workflow, "run_plan", fail)
    assert workflow.main([str(tmp_path / "plan.json"), "--state", str(tmp_path / "state")]) == 1
    captured = capsys.readouterr()
    assert "deadline exceeded" in captured.err
    assert "never-log-this" not in captured.err


def test_busy_stage_stops_without_duplicate(
    entry: dict[str, object], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    plan = tmp_path / "plan.json"
    plan.write_text(json.dumps({"version": 1, "stages": [entry]}), encoding="utf-8")
    monkeypatch.setattr(workflow, "run_stage", lambda *_args, **_kwargs: "busy")
    with pytest.raises(StageError, match="Another process"):
        workflow.run_plan(plan, state_directory=tmp_path / "state")


def test_plan_symlink_is_refused(tmp_path: Path) -> None:
    original = tmp_path / "original"
    original.write_text("{}", encoding="utf-8")
    alias = tmp_path / "alias"
    alias.symlink_to(original)
    with pytest.raises(StageError, match="manifest file"):
        workflow.run_plan(alias, state_directory=tmp_path / "state")


def test_oversized_plan_is_refused(tmp_path: Path) -> None:
    plan = tmp_path / "plan.json"
    plan.write_bytes(b"x" * 1_048_577)
    with pytest.raises(StageError, match="manifest file"):
        workflow.run_plan(plan, state_directory=tmp_path / "state")


@pytest.mark.parametrize("value", [None, [], {}, {"version": 1, "stages": [], "extra": 1}])
def test_invalid_manifest_fields(value: object) -> None:
    with pytest.raises(StageError, match="manifest fields"):
        workflow.parse_stages(value)


@pytest.mark.parametrize("version", [True, 0, 2, "1"])
def test_invalid_version(version: object) -> None:
    with pytest.raises(StageError, match="version"):
        workflow.parse_stages({"version": version, "stages": []})


@pytest.mark.parametrize("stages", [None, [], [{}] * 513])
def test_invalid_stage_count(stages: object) -> None:
    with pytest.raises(StageError, match="stage count"):
        workflow.parse_stages({"version": 1, "stages": stages})


def test_duplicate_names_are_refused(entry: dict[str, object]) -> None:
    with pytest.raises(StageError, match="unique"):
        workflow.parse_stages({"version": 1, "stages": [entry, entry]})


@pytest.mark.parametrize("item", [None, {"unknown": 1}])
def test_invalid_stage_fields(item: object) -> None:
    with pytest.raises(StageError, match="stage fields"):
        workflow.parse_stages({"version": 1, "stages": [item]})


@pytest.mark.parametrize(
    "changes",
    [{"name": 1}, {"timeout_seconds": True}, {"adopt_existing": 1}],
)
def test_invalid_scalar_fields(entry: dict[str, object], changes: dict[str, object]) -> None:
    with pytest.raises(StageError, match="name, timeout or adoption"):
        workflow.parse_stages({"version": 1, "stages": [entry | changes]})


@pytest.mark.parametrize("command", [None, "command", ["x"] * 257])
def test_invalid_argument_array(entry: dict[str, object], command: object) -> None:
    with pytest.raises(StageError, match="bounded"):
        workflow.parse_stages({"version": 1, "stages": [entry | {"command": command}]})


@pytest.mark.parametrize("command", [[1], [""], ["bad\0value"]])
def test_invalid_argument_text(entry: dict[str, object], command: object) -> None:
    with pytest.raises(StageError, match="nonempty text"):
        workflow.parse_stages({"version": 1, "stages": [entry | {"command": command}]})


def test_optional_verification_policy_is_retained(entry: dict[str, object]) -> None:
    stages = workflow.parse_stages(
        {"version": 1, "stages": [entry | {"verify_command": ["check"], "adopt_existing": True}]}
    )
    assert stages[0].verify_command == ("check",)
    assert stages[0].adopt_existing is True


def test_graph_orders_producer_before_consumer(entry: dict[str, object], tmp_path: Path) -> None:
    consumer = entry | {
        "name": "consume",
        "inputs": [str(tmp_path / "output")],
        "outputs": [str(tmp_path / "final")],
        "command": [
            sys.executable,
            "-c",
            "import sys; from pathlib import Path; "
            "Path(sys.argv[2]).write_bytes(Path(sys.argv[1]).read_bytes())",
            str(tmp_path / "output"),
            str(tmp_path / "final"),
        ],
    }
    plan = tmp_path / "plan.json"
    plan.write_text(json.dumps({"version": 1, "stages": [consumer, entry]}), encoding="utf-8")
    workflow.run_plan(plan, state_directory=tmp_path / "state")
    assert (tmp_path / "final").read_bytes() == b"result"
    assert workflow.stage_graph(workflow.load_stages(plan)) == {
        "consume": ("create",),
        "create": (),
    }


def test_graph_preserves_valid_manifest_priority(
    entry: dict[str, object], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    consumer = entry | {
        "name": "consume",
        "inputs": [str(tmp_path / "output")],
        "outputs": [str(tmp_path / "final")],
    }
    independent = entry | {"name": "independent", "outputs": [str(tmp_path / "other")]}
    plan = tmp_path / "plan.json"
    plan.write_text(
        json.dumps({"version": 1, "stages": [entry, consumer, independent]}), encoding="utf-8"
    )
    calls: list[str] = []

    def execute(stage: Stage, *, state_directory: Path) -> str:
        calls.append(stage.name)
        return "completed"

    monkeypatch.setattr(workflow, "run_stage", execute)
    workflow.run_plan(plan, state_directory=tmp_path / "state")
    assert calls == ["create", "consume", "independent"]


def test_graph_refuses_two_output_owners_before_execution(
    entry: dict[str, object], tmp_path: Path
) -> None:
    plan = tmp_path / "plan.json"
    plan.write_text(
        json.dumps({"version": 1, "stages": [entry, entry | {"name": "other"}]}),
        encoding="utf-8",
    )
    with pytest.raises(StageError, match="one producer"):
        workflow.run_plan(plan, state_directory=tmp_path / "state")
    assert not (tmp_path / "output").exists()
    assert not (tmp_path / "state").exists()


def test_graph_refuses_cycle_before_execution(entry: dict[str, object], tmp_path: Path) -> None:
    first = entry | {"inputs": [str(tmp_path / "second")]}
    second = entry | {
        "name": "second",
        "inputs": [str(tmp_path / "output")],
        "outputs": [str(tmp_path / "second")],
    }
    plan = tmp_path / "plan.json"
    plan.write_text(json.dumps({"version": 1, "stages": [first, second]}), encoding="utf-8")
    with pytest.raises(StageError, match="cycle"):
        workflow.run_plan(plan, state_directory=tmp_path / "state")
    assert not (tmp_path / "state").exists()


def test_check_only_has_no_side_effects_or_command_disclosure(
    entry: dict[str, object], tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    plan = tmp_path / "plan.json"
    plan.write_text(json.dumps({"version": 1, "stages": [entry]}), encoding="utf-8")
    assert workflow.main([str(plan), "--state", str(tmp_path / "state"), "--check"]) == 0
    assert capsys.readouterr().out == '{"create": []}\n'
    assert not (tmp_path / "output").exists()
    assert not (tmp_path / "state").exists()


def test_failed_producer_never_runs_consumer(entry: dict[str, object], tmp_path: Path) -> None:
    producer = entry | {"command": [sys.executable, "-c", "raise SystemExit(2)"]}
    consumer = entry | {
        "name": "consumer",
        "inputs": [str(tmp_path / "output")],
        "outputs": [str(tmp_path / "final")],
    }
    plan = tmp_path / "plan.json"
    plan.write_text(json.dumps({"version": 1, "stages": [consumer, producer]}), encoding="utf-8")
    with pytest.raises(StageError, match="command failed"):
        workflow.run_plan(plan, state_directory=tmp_path / "state")
    assert not (tmp_path / "state/consumer.json").exists()
    assert not (tmp_path / "final").exists()


def test_graph_resolves_output_aliases(entry: dict[str, object], tmp_path: Path) -> None:
    alias = tmp_path / "alias"
    alias.symlink_to(tmp_path / "output")
    with pytest.raises(StageError, match="one producer"):
        workflow.parse_stages(
            {"version": 1, "stages": [entry, entry | {"name": "other", "outputs": [str(alias)]}]}
        )
