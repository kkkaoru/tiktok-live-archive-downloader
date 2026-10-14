"""Pipeline recovery must preserve native sidecars for the same workflow verifier."""

import json
from collections.abc import Sequence
from pathlib import Path

import pytest
from test_native_invocation import native_request as native_request

from replay import delivery_pipeline as pipeline
from replay import native_invocation as invocation
from replay import stage_recovery, workflow
from replay.stages import Stage, run_stage


@pytest.mark.parametrize("outcome", ["complete", "partial", "unknown", "reservation"])
def test_pipeline_preserves_and_verifies_wrapper(
    native_request: Path, monkeypatch: pytest.MonkeyPatch, outcome: str
) -> None:
    request = native_request
    native, output = request.parent / "native", request.parent / "outputs"
    plan = request.parent / "render-0-0000.json"
    state = plan.with_name("render-0-0000-state")
    plan.write_text(
        json.dumps(
            {
                "version": 1,
                "stages": [
                    {
                        "name": "render",
                        "command": list(invocation.command(native, request, output)),
                        "verify_command": list(
                            invocation.command(native, request, output, verify=True)
                        ),
                        "inputs": list(map(str, invocation.inputs(native, request))),
                        "outputs": list(map(str, invocation.paths(request))),
                        "timeout_seconds": 1800,
                    }
                ],
            }
        )
    )
    calls: list[str] = []

    def execute(argv: tuple[str, ...], log: Path, timeout: int) -> None:
        calls.append(argv[3])
        if argv[3] == "verify":
            raise RuntimeError("simulate interrupted stage after native binding")
        assert invocation.main(list(argv[3:])) == 0

    stage = workflow.load_stages(plan)[0]
    if outcome in {"complete", "partial"}:
        with pytest.raises(RuntimeError, match="interrupted"):
            run_stage(stage, state_directory=state, execute=execute)
    elif outcome == "unknown":
        invocation.run(native, request, output)
    else:
        request.with_name("render.attempt").mkdir()
    if outcome == "partial":
        request.with_name("render.binding.json").write_text('{"status":"incomplete"}')
    sidecars = {p: p.read_bytes() for p in invocation.paths(request) if p.exists()}

    def verify_only(argv: tuple[str, ...], log: Path, timeout: int) -> None:
        calls.append(argv[3])
        if invocation.main(list(argv[3:])):
            raise ValueError("read-only verification/no-resubmit failure")

    def run(stage: Stage, *, state_directory: Path) -> str:
        return run_stage(stage, state_directory=state_directory, execute=verify_only)

    monkeypatch.setattr(workflow, "run_stage", run)

    def launch(batch: Sequence[Sequence[str]]) -> list[int]:
        return [workflow.main(list(argv[3:])) for argv in batch]

    step = pipeline.Step("render", "render-*.json")
    assert workflow.main([str(plan), "--state", str(state), "--check"]) == 0
    result = pipeline._attempt(step, [plan], launch)
    assert result == ([0] if outcome == "complete" else [1])
    assert {p: p.read_bytes() for p in sidecars} == sidecars
    assert request.with_name("render.attempt").is_dir()
    assert not list(request.parent.glob("*.failed-*"))
    assert not list(state.glob("*.failed-*"))
    if outcome == "complete":
        assert pipeline._attempt(step, [plan], launch) == [0]
        assert calls == ["run", "verify", "verify", "verify"]
        assert json.loads((state / "render.json").read_text())["status"] == "complete"
    elif outcome == "partial":
        assert calls == ["run", "verify", "verify"]
    elif outcome == "unknown":
        assert calls == []
    else:
        assert calls == ["run"]  # wrapper refuses its reservation before any native call


def test_malformed_wrapper_never_enters_generic_recovery(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "workflow.json"
    path.write_text(json.dumps({"stages": [{"command": ["replay.native_invocation"]}]}))
    monkeypatch.setattr(stage_recovery, "recover", lambda *_: pytest.fail("must preserve"))
    assert pipeline._attempt(pipeline.Step("render", ""), [path], lambda _: [1]) == [1]
    assert pipeline._has_native_wrapper({"stages": [None, {"command": "legacy"}]}) is False
