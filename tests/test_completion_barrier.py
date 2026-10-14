from __future__ import annotations

import hashlib
import json
import time
from dataclasses import replace
from pathlib import Path

import pytest

from replay import completion_barrier as barrier
from replay import measured_job_admission as admission
from replay.core import ReplayError
from replay.recordings import record
from replay.workflow import _stage


class FakeHost:
    def __init__(self) -> None:
        self.clock = 0.0
        self.calls = 0
        self.groups_values = [((1, 1),)]
        self.release_value: dict[str, object] = {
            "resourceReleased": True,
            "stageQualityApproved": False,
            "kind": "synthetic-release",
        }
        self.on_sleep: tuple[Path, str] | None = None
        self.slow = False

    def now_ns(self) -> int:
        return 1_000_000_000

    def monotonic(self) -> float:
        return self.clock

    def sleep(self, seconds: float) -> None:
        self.clock += seconds
        if self.on_sleep is not None:
            path, data = self.on_sleep
            path.write_text(data)
            self.on_sleep = None

    def groups(self) -> tuple[tuple[int, int], ...]:
        if self.slow:
            self.clock += 10
        result = self.groups_values[0]
        if len(self.groups_values) > 1:
            self.groups_values.pop(0)
        return result

    def release(self, _pin: barrier.Pin) -> dict[str, object]:
        self.calls += 1
        return self.release_value


def pin(path: Path) -> dict[str, object]:
    return {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def write_request(path: Path, raw: dict[str, object]) -> barrier.Request:
    data = json.dumps(raw).encode()
    path.write_bytes(data)
    return barrier.load(path, hashlib.sha256(data).hexdigest())


@pytest.fixture
def raw_request(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict[str, object]:
    root = tmp_path / "checkout"
    work = root / "edits" / "work" / "completion"
    state = work / "producer-state"
    state.mkdir(parents=True)
    code = root / "src" / "source.py"
    code.parent.mkdir()
    code.write_text("# synthetic source")
    executable = root / "python"
    executable.write_text("synthetic tool")
    input_path = work / "original.wav"
    workflow = work / "workflow.json"
    stages = [
        {
            "name": "chunk",
            "command": ["trusted-producer"],
            "inputs": [str(code), str(input_path)],
            "outputs": [str(work / "part.wav")],
            "verify_command": ["original-domain-verifier"],
            "timeout_seconds": 10,
        },
        {
            "name": "whole",
            "command": ["trusted-assembler"],
            "inputs": [str(code), str(work / "part.wav")],
            "outputs": [str(work / "whole.wav")],
            "verify_command": ["original-domain-verifier"],
            "timeout_seconds": 10,
        },
    ]
    workflow.write_text(json.dumps({"version": 1, "stages": stages}))
    (state / "chunk.json").write_text(json.dumps({"status": "running", "fingerprint": "a" * 64}))
    capture = work / "capture.json"
    capture.write_text(
        json.dumps(
            {
                "version": 1,
                "kind": "owned-family-completion-capture",
                "observedAtNs": 100,
                "parent": {
                    "pid": 10,
                    "birth": "100:1",
                    "executable": str(executable),
                    "commandSHA256": "b" * 64,
                },
                "processGroups": [10, 11],
                "workflow": pin(workflow),
                "stage": "chunk",
                "checkpoint": str(state / "chunk.json"),
                "statusPath": str(work / "producer.status"),
                "initialCheckpointSHA256": "c" * 64,
                "initialCheckpoint": {"status": "running", "fingerprint": "a" * 64},
            }
        )
    )
    monkeypatch.setattr(admission, "ROOT", root)
    return {
        "version": 1,
        "kind": "completion-barrier",
        "workflow": pin(workflow),
        "capture": pin(capture),
        "stateDirectory": str(state),
        "terminalStatus": str(work / "producer.status"),
        "lastStage": "whole",
        "codeInputs": [pin(code)],
        "waitSeconds": 2,
        "pollSeconds": 1,
        "output": str(work / "proof.json"),
    }


@pytest.fixture
def job(raw_request: dict[str, object]) -> barrier.Request:
    return write_request(Path(str(raw_request["output"])).parent / "request.json", raw_request)


@pytest.fixture
def nested_raw_request(raw_request: dict[str, object]) -> dict[str, object]:
    work = Path(str(raw_request["output"])).parent
    workflow = work / "workflow.json"
    original = record(json.loads(workflow.read_bytes()))
    entries = original["stages"]
    assert isinstance(entries, list)
    first = record(entries[0])
    original_inputs = first["inputs"]
    assert isinstance(original_inputs, list)
    child = work / "child-workflow.json"
    child.write_text(
        json.dumps(
            {
                "version": 1,
                "stages": [
                    {
                        "name": "repair",
                        "command": ["trusted-child"],
                        "inputs": [original_inputs[0]],
                        "outputs": [str(work / "child.wav")],
                        "verify_command": ["original-child-verifier"],
                        "timeout_seconds": 10,
                    }
                ],
            }
        )
    )
    first["command"] = ["python", "-B", "-m", "replay.workflow", str(child)]
    first["inputs"] = [str(child), str(work / "original.wav")]
    second = record(entries[1])
    second["command"] = ["python", "-B", "-m", "replay.workflow", str(child)]
    second["inputs"] = [str(child), str(work / "part.wav")]
    original["stages"] = [first, second]
    workflow.write_text(json.dumps(original))
    raw_request["workflow"] = pin(workflow)
    capture = work / "capture.json"
    captured = record(json.loads(capture.read_bytes()))
    captured["workflow"] = pin(workflow)
    capture.write_text(json.dumps(captured))
    raw_request["capture"] = pin(capture)
    raw_request["nestedWorkflows"] = [pin(child)]
    return raw_request


@pytest.fixture
def nested_job(nested_raw_request: dict[str, object]) -> barrier.Request:
    return write_request(
        Path(str(nested_raw_request["output"])).parent / "request.json", nested_raw_request
    )


def complete(job: barrier.Request) -> None:
    job.status.write_text("0\n")
    for stage in job.stages:
        (job.state / f"{stage.name}.json").write_text(
            json.dumps(
                {
                    "status": "complete",
                    "fingerprint": "a" * 64,
                    "outputs": {str(path): "d" * 64 for path in stage.outputs},
                }
            )
        )


def test_no_status_waits_finitely_and_never_calls_release_or_writes_proof(
    job: barrier.Request,
) -> None:
    host = FakeHost()
    with pytest.raises(ReplayError, match="deadline"):
        barrier.wait(job, host=host)
    assert host.clock == 2
    assert host.calls == 0
    assert not job.output.exists()


def test_complete_whole_graph_proof_is_private_reusable_and_never_dispatches(
    job: barrier.Request,
) -> None:
    complete(job)
    host = FakeHost()
    assert barrier.wait(job, host=host) == job.output
    data = job.output.read_bytes()
    proof = record(json.loads(data))
    assert proof["qualityApproved"] is False
    assert proof["downstreamDispatched"] is False
    assert job.output.stat().st_mode & 0o777 == 0o600
    producer = record(proof["producer"])
    checkpoints = producer["checkpoints"]
    assert isinstance(checkpoints, list)
    assert len(checkpoints) == 2
    barrier.verify(job, host=host)
    assert barrier.wait(job, host=host) == job.output
    assert job.output.read_bytes() == data


def test_prefix_success_does_not_replace_actual_whole_stage_completion(
    job: barrier.Request,
) -> None:
    complete(job)
    (job.state / "whole.json").write_text(
        json.dumps({"status": "running", "fingerprint": "a" * 64})
    )
    with pytest.raises(ReplayError):
        barrier.wait(job, host=FakeHost())
    assert not job.output.exists()


@pytest.mark.parametrize("status", ["1\n", "143\n", "unknown", "0 0", "0\n\n"])
def test_failed_or_ambiguous_status_stops_without_retry(job: barrier.Request, status: str) -> None:
    complete(job)
    job.status.write_text(status)
    host = FakeHost()
    with pytest.raises(ReplayError, match="failed or"):
        barrier.wait(job, host=host)
    assert host.clock == 0
    assert host.calls == 0


def test_completed_producer_with_live_family_waits_until_group_released(
    job: barrier.Request,
) -> None:
    complete(job)
    host = FakeHost()
    host.groups_values = [((11, 11),), ((1, 1),)]
    barrier.wait(job, host=host)
    assert host.clock == 1
    assert host.calls == 1


def test_completion_status_written_during_wait_is_adopted_without_rerun(
    job: barrier.Request,
) -> None:
    complete(job)
    job.status.unlink()
    host = FakeHost()
    host.on_sleep = (job.status, "0\n")
    barrier.wait(job, host=host)
    assert host.clock == 1
    assert host.calls == 1


def test_slow_observation_cannot_publish_after_deadline(job: barrier.Request) -> None:
    complete(job)
    host = FakeHost()
    host.slow = True
    with pytest.raises(ReplayError, match="deadline"):
        barrier.wait(job, host=host)
    assert not job.output.exists()


@pytest.mark.parametrize("failure", ["missing", "running", "outputs", "hash", "fingerprint"])
def test_checkpoint_corruption_is_not_promoted(job: barrier.Request, failure: str) -> None:
    complete(job)
    path = job.state / "whole.json"
    if failure == "missing":
        path.unlink()
    else:
        data = record(json.loads(path.read_bytes()))
        if failure == "running":
            data["status"] = "running"
        elif failure == "outputs":
            data["outputs"] = {"unrelated": "d" * 64}
        elif failure == "hash":
            data["outputs"] = {str(job.stages[-1].outputs[0]): "bad"}
        else:
            data["fingerprint"] = "bad"
        path.write_text(json.dumps(data))
    with pytest.raises((ReplayError, FileNotFoundError)):
        barrier.wait(job, host=FakeHost())
    assert not job.output.exists()


def test_code_mutation_during_wait_is_rejected_before_any_proof(job: barrier.Request) -> None:
    host = FakeHost()
    host.on_sleep = (job.code[0].pin.path, "changed source")
    with pytest.raises(ReplayError, match="identity changed"):
        barrier.wait(job, host=host)
    assert not job.output.exists()


def test_in_memory_deadline_changes_are_rejected(job: barrier.Request) -> None:
    with pytest.raises(ReplayError, match="identity changed"):
        barrier.wait(replace(job, wait_seconds=5), host=FakeHost())


@pytest.mark.parametrize(
    "field,value",
    [
        ("version", 2),
        ("kind", "other"),
        ("lastStage", "missing"),
        ("codeInputs", []),
        ("waitSeconds", -1),
        ("pollSeconds", 0),
        ("waitSeconds", 86400),
        ("extra", True),
    ],
)
def test_invalid_requests_are_rejected(
    raw_request: dict[str, object], field: str, value: object
) -> None:
    raw_request[field] = value
    with pytest.raises(ReplayError):
        write_request(Path(str(raw_request["output"])).parent / "bad-request.json", raw_request)


def test_missing_or_repeated_declared_code_closure_is_rejected(
    raw_request: dict[str, object],
) -> None:
    entries = raw_request["codeInputs"]
    assert isinstance(entries, list)
    raw_request["codeInputs"] = [entries[0], entries[0]]
    with pytest.raises(ReplayError, match="incomplete or repeated"):
        write_request(Path(str(raw_request["output"])).parent / "request.json", raw_request)


@pytest.mark.parametrize(
    "change", ["state", "status", "output-state", "output-request", "media-code"]
)
def test_wrong_capture_scope_and_escaping_code_or_output_are_rejected(
    raw_request: dict[str, object], change: str
) -> None:
    work = Path(str(raw_request["output"])).parent
    request_path = work / "request.json"
    if change == "state":
        state = work / "wrong-state"
        state.mkdir()
        raw_request["stateDirectory"] = str(state)
    elif change == "status":
        raw_request["terminalStatus"] = str(work / "other.status")
    elif change == "output-state":
        raw_request["output"] = str(Path(str(raw_request["stateDirectory"])) / "proof.json")
    elif change == "output-request":
        raw_request["output"] = str(request_path)
    else:
        media = work / "not-code.wav"
        media.write_bytes(b"not a media read")
        raw_request["codeInputs"] = [pin(media)]
    with pytest.raises(ReplayError):
        write_request(request_path, raw_request)


def test_occupied_bad_proof_is_preserved_not_overwritten(job: barrier.Request) -> None:
    complete(job)
    job.output.write_text("occupied")
    with pytest.raises(ValueError):
        barrier.wait(job, host=FakeHost())
    assert job.output.read_text() == "occupied"


def test_terminal_receipt_changes_during_release_reject_the_proof(
    job: barrier.Request, monkeypatch: pytest.MonkeyPatch
) -> None:
    complete(job)
    host = FakeHost()
    original = host.release

    def changed(pin: barrier.Pin) -> dict[str, object]:
        (job.state / "whole.json").write_text(
            json.dumps(
                {
                    "status": "complete",
                    "fingerprint": "b" * 64,
                    "outputs": {str(job.stages[-1].outputs[0]): "d" * 64},
                }
            )
        )
        return original(pin)

    monkeypatch.setattr(host, "release", changed)
    with pytest.raises(ReplayError, match="changed during"):
        barrier.wait(job, host=host)
    assert not job.output.exists()


def test_wrong_release_approval_and_output_directory_replacement_are_rejected(
    job: barrier.Request,
) -> None:
    complete(job)
    host = FakeHost()
    host.release_value["stageQualityApproved"] = True
    with pytest.raises(ReplayError, match="claims media approval"):
        barrier.wait(job, host=host)
    with pytest.raises(ReplayError, match="identity changed"):
        barrier.wait(replace(job, output_directory=(0, 0)), host=FakeHost())


def test_proof_content_and_checkpoint_change_prevent_reuse(job: barrier.Request) -> None:
    complete(job)
    barrier.wait(job, host=FakeHost())
    proof = record(json.loads(job.output.read_bytes()))
    proof["qualityApproved"] = True
    job.output.write_text(json.dumps(proof))
    with pytest.raises(ReplayError, match="Invalid completion"):
        barrier.verify(job, host=FakeHost())
    proof["qualityApproved"] = False
    job.output.write_text(json.dumps(proof))
    job.status.unlink()
    with pytest.raises(ReplayError, match="changed or is incomplete"):
        barrier.verify(job, host=FakeHost())


def test_finite_loop_budget_and_output_size_are_not_relaxed(
    job: barrier.Request, monkeypatch: pytest.MonkeyPatch
) -> None:
    host = FakeHost()
    monkeypatch.setattr(host, "sleep", lambda _seconds: None)
    monkeypatch.setattr(barrier, "MAX_POLLS", 4)
    with pytest.raises(ReplayError, match="exhausted"):
        barrier.wait(job, host=host)
    monkeypatch.setattr(barrier, "MAX_BYTES", 1)
    with pytest.raises(ReplayError, match="byte budget"):
        barrier._publish(job, {"tooLarge": "true"})


def test_system_host_reuses_existing_queries_and_release_verifier(
    job: barrier.Request, monkeypatch: pytest.MonkeyPatch
) -> None:
    host = barrier.SystemHost()
    monkeypatch.setattr(admission, "probe", lambda _command: "1 1\n")
    monkeypatch.setattr(
        admission, "verify_completion_capture", lambda _path, _sha: {"resourceReleased": True}
    )
    assert host.groups() == ((1, 1),)
    assert host.release(job.capture) == {"resourceReleased": True}
    assert host.now_ns() > 0
    assert host.monotonic() > 0
    observed: list[float] = []
    monkeypatch.setattr(time, "sleep", observed.append)
    host.sleep(0.25)
    assert observed == [0.25]


def test_public_cli_wait_verify_and_error_are_bounded(
    job: barrier.Request, monkeypatch: pytest.MonkeyPatch
) -> None:
    complete(job)
    host = FakeHost()
    monkeypatch.setattr(barrier, "SystemHost", lambda: host)
    argv = ["wait", str(job.path), "--sha256", job.sha256]
    assert barrier.main(argv) == 0
    argv[0] = "verify"
    assert barrier.main(argv) == 0
    assert barrier.main(["wait", str(job.path), "--sha256", "invalid"]) == 1


def test_nested_parent_with_no_direct_code_keeps_child_closure_and_proof(
    nested_job: barrier.Request,
) -> None:
    assert len(nested_job.nested) == 1
    assert len(nested_job.code) == 1
    complete(nested_job)
    barrier.wait(nested_job, host=FakeHost())
    proof = record(json.loads(nested_job.output.read_bytes()))
    producer = record(proof["producer"])
    nested = producer["nestedWorkflows"]
    assert isinstance(nested, list)
    assert len(nested) == 1
    barrier.verify(nested_job, host=FakeHost())


@pytest.mark.parametrize("value", [None, {}, [], "child", ["invalid"]])
def test_nested_missing_or_invalid_exact_child_set_is_rejected(
    nested_raw_request: dict[str, object], value: object
) -> None:
    nested_raw_request["nestedWorkflows"] = value
    with pytest.raises(ReplayError):
        write_request(
            Path(str(nested_raw_request["output"])).parent / "request.json", nested_raw_request
        )


def test_nested_field_cannot_be_omitted_for_actual_child_command(
    nested_raw_request: dict[str, object],
) -> None:
    nested_raw_request.pop("nestedWorkflows")
    with pytest.raises(ReplayError, match="incomplete"):
        write_request(
            Path(str(nested_raw_request["output"])).parent / "request.json", nested_raw_request
        )


def test_nested_repeated_or_unlisted_children_cannot_supply_code(
    nested_raw_request: dict[str, object],
) -> None:
    nested = nested_raw_request["nestedWorkflows"]
    assert isinstance(nested, list)
    nested_raw_request["nestedWorkflows"] = [nested[0], nested[0]]
    with pytest.raises(ReplayError, match="repeated"):
        write_request(
            Path(str(nested_raw_request["output"])).parent / "request.json", nested_raw_request
        )


def test_nested_code_cannot_be_omitted_or_replaced_by_other_source(
    nested_raw_request: dict[str, object],
) -> None:
    other = Path(str(nested_raw_request["output"])).parent / "unlisted.py"
    other.write_text("# unrelated")
    nested_raw_request["codeInputs"] = [pin(other)]
    with pytest.raises(ReplayError, match="outside"):
        write_request(
            Path(str(nested_raw_request["output"])).parent / "request.json", nested_raw_request
        )


def test_nested_json_changes_during_wait_fail_before_proof(nested_job: barrier.Request) -> None:
    host = FakeHost()
    host.on_sleep = (nested_job.nested[0].pin.path, "{}")
    with pytest.raises(ValueError, match="changed"):
        barrier.wait(nested_job, host=host)
    assert not nested_job.output.exists()


def test_nested_total_budget_remains_finite(
    nested_raw_request: dict[str, object], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(barrier, "MAX_NESTED_BYTES", 1)
    with pytest.raises(ReplayError, match="byte budget"):
        write_request(
            Path(str(nested_raw_request["output"])).parent / "request.json", nested_raw_request
        )


@pytest.mark.parametrize(
    "command",
    [
        ["-m", "replay.workflow"],
        ["-m", "replay.workflow", "child.json"],
        ["-m", "replay.workflow", "/unlisted.json"],
        ["-m", "replay.workflow", "/a.json", "-m", "replay.workflow", "/b.json"],
    ],
)
def test_nested_command_must_bind_one_exact_absolute_input(command: list[str]) -> None:
    stage = _stage(
        {
            "name": "parent",
            "command": command,
            "inputs": ["/input.json"],
            "outputs": ["/output.json"],
            "verify_command": ["verify"],
            "timeout_seconds": 10,
        }
    )
    with pytest.raises(ReplayError):
        barrier._workflow_paths((stage,))


def test_nested_grandchildren_are_rejected(nested_job: barrier.Request) -> None:
    child = nested_job.nested[0].pin.path
    raw = record(json.loads(child.read_bytes()))
    stages = raw["stages"]
    assert isinstance(stages, list)
    stage = record(stages[0])
    stage["command"] = ["python", "-m", "replay.workflow", str(child)]
    stage["inputs"] = [str(child)]
    child.write_text(json.dumps({"version": 1, "stages": [stage]}))
    with pytest.raises(ReplayError, match="one child level"):
        barrier._nested_workflows([pin(child)], nested_job.stages)


def test_missing_nested_workflow_is_not_recreated(nested_job: barrier.Request) -> None:
    nested_job.nested[0].pin.path.unlink()
    with pytest.raises(FileNotFoundError):
        barrier.wait(nested_job, host=FakeHost())
    assert not nested_job.output.exists()


def test_nested_identity_replacement_with_identical_bytes_is_rejected(
    nested_job: barrier.Request,
) -> None:
    path = nested_job.nested[0].pin.path
    replacement = path.with_name("replacement.json")
    replacement.write_bytes(path.read_bytes())
    replacement.replace(path)
    with pytest.raises(ReplayError, match="identity changed"):
        barrier.wait(nested_job, host=FakeHost())
    assert not nested_job.output.exists()


def test_nested_workflow_tampering_during_terminal_release_is_rejected(
    nested_job: barrier.Request, monkeypatch: pytest.MonkeyPatch
) -> None:
    complete(nested_job)
    host = FakeHost()
    original = host.release

    def tampered(pin: barrier.Pin) -> dict[str, object]:
        nested_job.nested[0].pin.path.write_text("{}")
        return original(pin)

    monkeypatch.setattr(host, "release", tampered)
    with pytest.raises(ValueError, match="changed"):
        barrier.wait(nested_job, host=host)
    assert not nested_job.output.exists()
