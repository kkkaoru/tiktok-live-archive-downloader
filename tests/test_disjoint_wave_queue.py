"""Synthetic fixed-source waves; admission/workflow keep their original decisions."""

from __future__ import annotations

import hashlib
import json
import subprocess
import time
from dataclasses import replace
from pathlib import Path
from threading import Event
from unittest.mock import MagicMock

import pytest

from replay import completion_barrier, disk_budget, workflow
from replay import disjoint_wave_queue as queue
from replay import measured_job_admission as admission
from replay.core import ReplayError
from replay.native_invocation import durable, pin
from replay.stages import StageError


@pytest.fixture
def plan_path(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    root = tmp_path / "checkout"
    work = root / "edits/work/queue"
    work.mkdir(parents=True)
    mutex = work / "shared-mutex"
    mutex.mkdir()
    tool = root / "python"
    tool.write_bytes(b"synthetic executable")
    tool.chmod(0o700)
    source = root / "source.py"
    source.write_text("# synthetic pinned source\n", encoding="utf-8")
    monkeypatch.setattr(queue, "ROOT", root)
    monkeypatch.setattr(queue, "WORK", root / "edits/work")
    monkeypatch.setattr(admission, "ROOT", root)

    def reference(path: Path) -> dict[str, str]:
        return {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}

    def wave(name: str) -> dict[str, object]:
        directory = work / name
        directory.mkdir()
        (directory / "tmp").mkdir()
        (directory / "cache").mkdir()
        state = work / f"state-{name}"
        manifest = directory / "workflow.json"
        manifest.write_text(
            json.dumps(
                {
                    "version": 1,
                    "stages": [
                        {
                            "name": name,
                            "command": [str(tool)],
                            "inputs": [str(source)],
                            "outputs": [str(directory / "media.bin")],
                            "timeout_seconds": 2,
                            "verify_command": [str(tool)],
                        }
                    ],
                }
            ),
            encoding="utf-8",
        )
        request = directory / "admission.json"
        request.write_text(
            json.dumps(
                {
                    "version": 1,
                    "kind": "measured-job-admission",
                    "name": name,
                    "workDirectory": str(directory),
                    "scratchDirectory": str(directory / "tmp"),
                    "cacheDirectory": str(directory / "cache"),
                    "diskPaths": [str(directory)],
                    "executable": {
                        **reference(tool),
                        "invocationPath": str(tool),
                        "linkTarget": None,
                    },
                    "command": [
                        str(tool),
                        "-B",
                        "-m",
                        "replay.workflow",
                        str(manifest),
                        "--state",
                        str(state),
                    ],
                    "families": [],
                    "policy": {
                        "newMemoryBytes": 100,
                        "newDiskBytes": 30,
                        "osBurstMarginBytes": 40,
                        "ocrBurstMarginBytes": 50,
                        "minimumFreeBytes": 60,
                        "maximumPressure": 1,
                        "maximumSwapBytes": 500,
                        "maximumSwapGrowthBytes": 20,
                        "maximumSnapshotAgeSeconds": 5,
                        "waitSeconds": 0,
                        "pollSeconds": 1,
                    },
                }
            ),
            encoding="utf-8",
        )
        return {
            "name": name,
            "admission": reference(request),
            "workflow": reference(manifest),
            "stateDirectory": str(state),
            "completedPeers": [],
        }

    path = work / "plan.json"
    path.write_text(
        json.dumps(
            {
                "version": 1,
                "kind": "disjoint-wave-queue",
                "waves": [wave("first"), wave("second")],
                "sources": [reference(source)],
                "mutex": {
                    "directory": str(mutex),
                    "name": "priority",
                    "waitSeconds": 0,
                    "pollSeconds": 1,
                },
                "minimumBootFreeBytes": 1,
            }
        ),
        encoding="utf-8",
    )
    return path


@pytest.fixture
def plan(plan_path: Path) -> queue.Plan:
    return queue.load(plan_path, hashlib.sha256(plan_path.read_bytes()).hexdigest())


@pytest.fixture
def normal_runner(plan: queue.Plan, monkeypatch: pytest.MonkeyPatch) -> MagicMock:
    def run(request: admission.Request) -> int:
        durable(request.work / f"{request.name}.launch-intent.json", {"startedAtNs": 200})
        durable(
            request.work / f"{request.name}.end.json",
            {
                "kind": "measured-admission-end",
                "requestSHA256": request.sha256,
                "endedAtNs": 100,
                "reservationReleased": True,
                "completed": True,
                "dispatchAttempted": True,
                "uncertainLaunch": False,
                "exitCode": 0,
                "errorKind": None,
            },
        )
        return 0

    runner = MagicMock(side_effect=run)
    monkeypatch.setattr(admission, "run", runner)
    monkeypatch.setattr(
        disk_budget,
        "measure",
        MagicMock(return_value=disk_budget.Headroom(plan.mutex.directory, 1000, 900, 100, 0)),
    )
    return runner


def test_continuous_waves_have_actual_release_and_wall_gap(
    plan: queue.Plan, normal_runner: MagicMock
) -> None:
    output = plan.reference.path.with_name("result")
    assert queue.run(plan, output) == 0
    assert normal_runner.call_count == 2
    first = json.loads((output / "wave-0000-end.json").read_bytes())
    second = json.loads((output / "wave-0001-end.json").read_bytes())
    assert first["gapSincePreviousReleaseNs"] is None
    assert second["gapSincePreviousReleaseNs"] == 100
    assert second["release"]["reservationReleased"] is True
    assert json.loads((output / "queue-complete.json").read_bytes())["waves"] == 2


def test_failure_preserves_partial_state_and_stops_next_wave(
    plan: queue.Plan, normal_runner: MagicMock
) -> None:
    partial = plan.waves[0].request.work / "media.bin"
    partial.write_bytes(b"retained partial output")
    normal_runner.side_effect = None
    normal_runner.return_value = 1
    output = plan.reference.path.with_name("failed")
    assert queue.run(plan, output) == 1
    assert normal_runner.call_count == 1
    assert partial.read_bytes() == b"retained partial output"
    assert (output / "queue-complete.json").exists() is False


def test_completed_domain_state_still_delegates_to_normal_workflow(
    plan: queue.Plan, normal_runner: MagicMock
) -> None:
    plan.waves[0].state.mkdir()
    checkpoint = plan.waves[0].state / "first.json"
    checkpoint.write_text('{"status":"complete","fingerprint":"old"}', encoding="utf-8")
    assert queue.run(plan, plan.reference.path.with_name("reuse")) == 0
    assert normal_runner.call_args_list[0].args[0].command[3] == "replay.workflow"
    assert checkpoint.read_text(encoding="utf-8") == '{"status":"complete","fingerprint":"old"}'


def test_missing_peer_blocks_before_dispatch(
    plan: queue.Plan, monkeypatch: pytest.MonkeyPatch
) -> None:
    wave = replace(plan.waves[0], peers=(plan.reference,))
    current = replace(plan, waves=(wave,))
    runner = MagicMock()
    monkeypatch.setattr(admission, "run", runner)
    monkeypatch.setattr(
        admission,
        "verify_completion_capture",
        MagicMock(side_effect=ReplayError("peer unavailable")),
    )
    with pytest.raises(ReplayError, match="peer unavailable"):
        queue.run(current, plan.reference.path.with_name("missing-peer"))
    assert runner.call_count == 0


def test_prior_release_rewrite_stops_next_wave(
    plan: queue.Plan, normal_runner: MagicMock, monkeypatch: pytest.MonkeyPatch
) -> None:
    original = queue._peers

    def peers(wave: queue.Wave) -> list[dict[str, object]]:
        if wave.name == "second":
            end = plan.waves[0].request.work / "first.end.json"
            end.write_text("changed previous release", encoding="utf-8")
        return original(wave)

    monkeypatch.setattr(queue, "_peers", peers)
    with pytest.raises(ReplayError, match="Prior normal release evidence changed"):
        queue.run(plan, plan.reference.path.with_name("rewritten-release"))
    assert normal_runner.call_count == 1


def test_queue_journal_cannot_overlap_state(plan: queue.Plan, normal_runner: MagicMock) -> None:
    with pytest.raises(ReplayError, match="Queue journal overlaps"):
        queue.run(plan, plan.waves[0].state)
    assert normal_runner.call_count == 0


@pytest.mark.parametrize(
    "field",
    [
        "reservationReleased",
        "completed",
        "dispatchAttempted",
        "uncertainLaunch",
        "exitCode",
        "errorKind",
        "requestSHA256",
        "endedAtNs",
    ],
)
def test_invalid_normal_release_rejects_next_wave(plan: queue.Plan, field: str) -> None:
    wave = plan.waves[0]
    end = {
        "kind": "measured-admission-end",
        "requestSHA256": wave.request.sha256,
        "endedAtNs": 100,
        "reservationReleased": True,
        "completed": True,
        "dispatchAttempted": True,
        "uncertainLaunch": False,
        "exitCode": 0,
        "errorKind": None,
    }
    end[field] = {
        "reservationReleased": False,
        "completed": False,
        "dispatchAttempted": False,
        "uncertainLaunch": True,
        "exitCode": 1,
        "errorKind": "error",
        "requestSHA256": "bad",
        "endedAtNs": 0,
    }[field]
    durable(wave.request.work / f"{wave.request.name}.end.json", end)
    with pytest.raises(ReplayError):
        queue._release(wave)


def test_completed_peer_reservations_are_zero_in_normal_admission(plan: queue.Plan) -> None:
    request = plan.waves[0].request
    parent = admission.Parent(10, "123:456", request.executable.path, "a" * 64)
    completion = admission.Completion(
        plan.reference.path,
        "a" * 64,
        plan.reference.path,
        "a" * 64,
        plan.reference.path,
        plan.reference.path,
        "a" * 64,
        (),
        (10,),
    )
    request = replace(request, families=(admission.Family(parent, 100000, 100000, completion),))
    snapshot = admission.Snapshot(
        100,
        4096,
        10,
        0,
        0,
        1,
        (
            admission.FamilyUsage(
                parent, (), 0, "released", (("status", "a" * 64), ("checkpoint", "b" * 64))
            ),
        ),
        (disk_budget.Headroom(request.work, 100 * 1024**3, 90 * 1024**3, 30 * 1024**3, 0),),
    )
    result = admission.assess(request, snapshot, now_ns=100)
    assert result.evidence["familyRemainingBytes"] == 0
    assert result.evidence["diskRequiredBytes"] == 30


@pytest.mark.parametrize("mode", ["occupied", "lowdisk", "changedsource", "request"])
def test_pre_dispatch_guards(
    plan: queue.Plan, normal_runner: MagicMock, monkeypatch: pytest.MonkeyPatch, mode: str
) -> None:
    output = plan.reference.path.with_name("blocked")
    if mode == "occupied":
        output.mkdir()
    if mode == "lowdisk":
        plan = replace(plan, minimum_boot_free_bytes=1000)
    if mode == "changedsource":
        plan.sources[0].path.write_text("changed", encoding="utf-8")
    if mode == "request":
        monkeypatch.setattr(
            admission,
            "load",
            MagicMock(return_value=replace(plan.waves[0].request, name="changed")),
        )
    with pytest.raises(ReplayError):
        queue.run(plan, output)
    assert normal_runner.call_count == 0


@pytest.mark.parametrize("mode", ["busy", "wait"])
def test_priority_mutex_is_finite(
    plan: queue.Plan, normal_runner: MagicMock, monkeypatch: pytest.MonkeyPatch, mode: str
) -> None:
    acquired = MagicMock()
    acquired.__enter__.return_value = False
    monkeypatch.setattr(queue, "execution_lock", MagicMock(return_value=acquired))
    monkeypatch.setattr(time, "monotonic", MagicMock(side_effect=[0, 0, 2]))
    sleeper = MagicMock()
    monkeypatch.setattr(time, "sleep", sleeper)
    if mode == "wait":
        plan = replace(plan, mutex=replace(plan.mutex, wait_seconds=1))
    assert queue.run(plan, plan.reference.path.with_name("mutex-blocked")) == 75
    assert normal_runner.call_count == 0
    assert sleeper.call_count == (1 if mode == "wait" else 0)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("version", True),
        ("kind", "unknown"),
        ("waves", []),
        ("sources", []),
        ("minimumBootFreeBytes", 0),
        ("extra", 1),
    ],
)
def test_external_queue_schema(plan_path: Path, field: str, value: object) -> None:
    raw = json.loads(plan_path.read_bytes())
    raw[field] = value
    plan_path.write_text(json.dumps(raw), encoding="utf-8")
    with pytest.raises(ReplayError):
        queue.load(plan_path, hashlib.sha256(plan_path.read_bytes()).hexdigest())


@pytest.mark.parametrize(
    "mode",
    [
        "name",
        "command",
        "peers",
        "verifier",
        "preflight",
        "duplicate",
        "states",
        "outputs",
        "sources",
        "mutex",
        "mutexdir",
    ],
)
def test_wave_and_source_contract(
    plan_path: Path, monkeypatch: pytest.MonkeyPatch, mode: str
) -> None:
    raw = json.loads(plan_path.read_bytes())
    wave = raw["waves"][0]
    if mode == "name":
        wave["name"] = "bad name"
    if mode == "command":
        monkeypatch.setattr(
            admission,
            "load",
            MagicMock(
                return_value=replace(
                    admission.load(Path(wave["admission"]["path"]), wave["admission"]["sha256"]),
                    command=("bad",),
                )
            ),
        )
    if mode == "peers":
        wave["completedPeers"] = [wave["workflow"]]
    if mode == "verifier":
        manifest = Path(wave["workflow"]["path"])
        data = json.loads(manifest.read_bytes())
        data["stages"][0]["verify_command"] = []
        manifest.write_text(json.dumps(data), encoding="utf-8")
        wave["workflow"]["sha256"] = hashlib.sha256(manifest.read_bytes()).hexdigest()
    if mode == "preflight":
        monkeypatch.setattr(workflow, "main", MagicMock(return_value=1))
    if mode == "duplicate":
        raw["waves"] = [wave, wave]
    if mode == "states":
        monkeypatch.setattr(
            queue,
            "_wave",
            MagicMock(
                side_effect=[
                    queue.Wave(
                        "one",
                        admission.load(
                            Path(wave["admission"]["path"]), wave["admission"]["sha256"]
                        ),
                        queue.Reference(Path(wave["workflow"]["path"]), wave["workflow"]["sha256"]),
                        Path(wave["stateDirectory"]),
                        (),
                    ),
                    queue.Wave(
                        "two",
                        admission.load(
                            Path(raw["waves"][1]["admission"]["path"]),
                            raw["waves"][1]["admission"]["sha256"],
                        ),
                        queue.Reference(
                            Path(raw["waves"][1]["workflow"]["path"]),
                            raw["waves"][1]["workflow"]["sha256"],
                        ),
                        Path(wave["stateDirectory"]),
                        (),
                    ),
                ]
            ),
        )
    if mode == "outputs":
        monkeypatch.setattr(
            queue, "_disjoint", MagicMock(side_effect=[None, ReplayError("output overlap")])
        )
    if mode == "sources":
        raw["sources"] *= 2
    if mode == "mutex":
        raw["mutex"]["extra"] = 1
    if mode == "mutexdir":
        raw["mutex"]["directory"] = str(plan_path.parent / "absent")
    plan_path.write_text(json.dumps(raw), encoding="utf-8")
    with pytest.raises(ReplayError):
        queue.load(plan_path, hashlib.sha256(plan_path.read_bytes()).hexdigest())


@pytest.mark.parametrize(
    "mode", ["digest", "reference", "outside", "duplicatejson", "size", "wavefields"]
)
def test_bounded_pins(plan_path: Path, mode: str) -> None:
    raw = json.loads(plan_path.read_bytes())
    if mode == "digest":
        raw["sources"][0]["sha256"] = "invalid"
    if mode == "reference":
        raw["sources"][0]["extra"] = True
    if mode == "outside":
        raw["sources"][0]["path"] = "/outside/source.py"
    if mode == "wavefields":
        raw["waves"][0]["extra"] = True
    plan_path.write_text(json.dumps(raw), encoding="utf-8")
    if mode == "duplicatejson":
        plan_path.write_text('{"version":1,"version":1}', encoding="utf-8")
    if mode == "size":
        plan_path.write_bytes(b"x" * (1048576 + 1))
    with pytest.raises(ReplayError):
        queue.load(plan_path, hashlib.sha256(plan_path.read_bytes()).hexdigest())


@pytest.mark.parametrize("mode", ["check", "run", "missingoutput", "error"])
def test_cli(plan_path: Path, normal_runner: MagicMock, mode: str) -> None:
    args = [
        "check" if mode == "check" else "run",
        str(plan_path),
        "--sha256",
        hashlib.sha256(plan_path.read_bytes()).hexdigest(),
    ]
    if mode == "run":
        args += ["--output", str(plan_path.with_name("cli-result"))]
    if mode == "error":
        args[3] = "bad"
    assert queue.main(args) == (0 if mode in {"check", "run"} else 1)


@pytest.fixture
def group_path(plan: queue.Plan, monkeypatch: pytest.MonkeyPatch) -> Path:
    path = plan.reference.path.with_name("group.json")
    path.write_text(
        json.dumps(
            {
                "version": 1,
                "kind": "parallel-workflow-group",
                "lanes": [
                    {
                        "workflow": {
                            "path": str(plan.waves[0].workflow.path),
                            "sha256": plan.waves[0].workflow.sha256,
                        },
                        "stateDirectory": str(plan.waves[0].state),
                    },
                    {
                        "workflow": {
                            "path": str(plan.waves[1].workflow.path),
                            "sha256": plan.waves[1].workflow.sha256,
                        },
                        "stateDirectory": str(plan.waves[1].state),
                    },
                ],
                "maximumWorkers": 2,
                "outputDirectory": str(path.with_name("group-result")),
                "sources": [{"path": str(plan.sources[0].path), "sha256": plan.sources[0].sha256}],
            }
        ),
        encoding="utf-8",
    )
    host = MagicMock()
    host.birth.return_value = "123:456"
    monkeypatch.setattr(queue, "ProcessHost", MagicMock(return_value=host))
    return path


@pytest.fixture
def group(group_path: Path) -> queue.Group:
    return queue.load_group(group_path, hashlib.sha256(group_path.read_bytes()).hexdigest())


@pytest.fixture
def group_runner(monkeypatch: pytest.MonkeyPatch) -> MagicMock:
    def run(manifest: Path, *, state_directory: Path) -> None:
        state_directory.mkdir()
        stage = json.loads(manifest.read_bytes())["stages"][0]["name"]
        durable(state_directory / f"{stage}.json", {"status": "complete", "fingerprint": "a" * 64})

    runner = MagicMock(side_effect=run)
    monkeypatch.setattr(workflow, "run_plan", runner)
    return runner


def test_parallel_group_waits_and_verifies_normal_checkpoints(
    group: queue.Group, group_runner: MagicMock
) -> None:
    assert queue.run_group(group) == 0
    assert group_runner.call_count == 2
    queue.verify_group(group)
    report = json.loads((group.output / "result.json").read_bytes())
    assert report["allOwnedLanesWaited"] is True
    assert report["completed"] is True
    assert report["lanes"][0]["status"] == "complete"
    assert report["lanes"][1]["status"] == "complete"


def test_group_self_identity_never_spawns_ps(
    group: queue.Group, group_runner: MagicMock, monkeypatch: pytest.MonkeyPatch
) -> None:
    denied_ps = MagicMock(side_effect=PermissionError("Synthetic sandbox denies ps"))
    monkeypatch.setattr(subprocess, "run", denied_ps)
    assert queue.run_group(group) == 0
    assert group_runner.call_count == 2
    assert denied_ps.call_count == 0
    start = json.loads((group.output / "start.json").read_bytes())
    assert start["birth"] == "123:456"
    assert "commandSHA256" not in start
    assert len(start["interpreter"]["sha256"]) == 64


def test_changed_group_self_birth_preserves_real_lane_results(
    group: queue.Group, group_runner: MagicMock, monkeypatch: pytest.MonkeyPatch
) -> None:
    host = MagicMock()
    host.birth.side_effect = ["123:456", "changed"]
    monkeypatch.setattr(queue, "ProcessHost", MagicMock(return_value=host))
    with pytest.raises(ReplayError, match="producer identity changed"):
        queue.run_group(group)
    assert group_runner.call_count == 2
    assert (group.output / "result.json").exists() is False
    assert (group.output / "lane-0000.json").exists() is True
    assert (group.output / "lane-0001.json").exists() is True


def test_failed_lane_waits_for_other_normal_lane_and_preserves_partial(
    group: queue.Group, monkeypatch: pytest.MonkeyPatch
) -> None:
    failed = Event()
    completed = Event()
    partial = group.lanes[0].workflow.path.with_name("media.bin")

    def run(manifest: Path, *, state_directory: Path) -> None:
        if manifest == group.lanes[0].workflow.path:
            partial.write_bytes(b"retained synthetic partial")
            failed.set()
            raise StageError("synthetic failed stage")
        if not failed.wait(timeout=2):
            raise StageError("Other lane did not actually start")
        state_directory.mkdir()
        durable(state_directory / "second.json", {"status": "complete"})
        completed.set()

    monkeypatch.setattr(workflow, "run_plan", run)
    assert queue.run_group(group) == 1
    assert completed.is_set() is True
    assert partial.read_bytes() == b"retained synthetic partial"
    assert json.loads((group.output / "result.json").read_bytes())["completed"] is False
    with pytest.raises(ReplayError, match="lacks verified complete"):
        queue.verify_group(group)


@pytest.mark.parametrize(
    "mode",
    ["occupied", "source", "checkpoint", "incomplete", "lanecount", "laneidentity", "booleanindex"],
)
def test_parallel_group_guards(group: queue.Group, group_runner: MagicMock, mode: str) -> None:
    if mode == "occupied":
        group.output.mkdir()
        with pytest.raises(ReplayError, match="occupied"):
            queue.run_group(group)
        assert group_runner.call_count == 0
        return
    if mode == "source":
        group.sources[0].path.write_text("changed", encoding="utf-8")
        with pytest.raises(ReplayError, match="generation changed"):
            queue.run_group(group)
        assert group_runner.call_count == 0
        return
    assert queue.run_group(group) == 0
    report_path = group.output / "result.json"
    report = json.loads(report_path.read_bytes())
    if mode == "checkpoint":
        (group.lanes[0].state / "first.json").write_text('{"status":"running"}', encoding="utf-8")
    if mode == "incomplete":
        report["completed"] = False
    if mode == "lanecount":
        report["lanes"] = []
    if mode == "laneidentity":
        report["lanes"][0]["status"] = "failed"
    if mode == "booleanindex":
        report["lanes"][1]["index"] = True
    report_path.write_text(json.dumps(report), encoding="utf-8")
    with pytest.raises(ReplayError):
        queue.verify_group(group)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("version", True),
        ("kind", "unknown"),
        ("lanes", []),
        ("maximumWorkers", 0),
        ("sources", []),
        ("extra", 1),
    ],
)
def test_external_group_schema(group_path: Path, field: str, value: object) -> None:
    raw = json.loads(group_path.read_bytes())
    raw[field] = value
    group_path.write_text(json.dumps(raw), encoding="utf-8")
    with pytest.raises(ReplayError):
        queue.load_group(group_path, hashlib.sha256(group_path.read_bytes()).hexdigest())


@pytest.mark.parametrize(
    "mode", ["lanefields", "preflight", "verifier", "stages", "outputparent", "sources"]
)
def test_group_layout_and_bounds(
    group_path: Path, monkeypatch: pytest.MonkeyPatch, mode: str
) -> None:
    raw = json.loads(group_path.read_bytes())
    if mode == "lanefields":
        raw["lanes"][0]["extra"] = True
    if mode == "preflight":
        monkeypatch.setattr(workflow, "main", MagicMock(return_value=1))
    if mode == "verifier":
        stages = workflow.load_stages(Path(raw["lanes"][0]["workflow"]["path"]))
        monkeypatch.setattr(
            workflow,
            "load_stages",
            MagicMock(return_value=(replace(stages[0], verify_command=()),)),
        )
    if mode == "stages":
        monkeypatch.setattr(workflow, "MAX_STAGES", 1)
    if mode == "outputparent":
        raw["outputDirectory"] = str(group_path.parent / "absent/result")
    if mode == "sources":
        raw["sources"] *= 2
    group_path.write_text(json.dumps(raw), encoding="utf-8")
    with pytest.raises(ReplayError):
        queue.load_group(group_path, hashlib.sha256(group_path.read_bytes()).hexdigest())


def test_group_cli_and_cancellation(group_path: Path, group_runner: MagicMock) -> None:
    digest = hashlib.sha256(group_path.read_bytes()).hexdigest()
    assert queue.main(["group", str(group_path), "--sha256", digest]) == 0
    assert group_runner.call_count == 2
    assert queue.main(["verify-group", str(group_path), "--sha256", digest]) == 0
    with pytest.raises(StageError, match="cancelled"):
        queue._cancel_group(15, None)


def test_rigid_sandbox_wrapper_and_interpreter_pins(
    plan_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    raw = json.loads(plan_path.read_bytes())
    wave = raw["waves"][0]
    request = admission.load(Path(wave["admission"]["path"]), wave["admission"]["sha256"])
    profile = plan_path.with_name("sandbox.sb")
    profile.write_text("(version 1)\n(allow default)\n", encoding="utf-8")
    wave["sandboxProfile"] = {"path": str(profile), "sha256": pin(profile)["sha256"]}
    wave["pythonExecutable"] = {
        "path": str(request.executable.path),
        "sha256": request.executable.sha256,
        "invocationPath": str(request.executable.invocation),
        "linkTarget": None,
    }
    wrapped = replace(
        request,
        executable=replace(
            request.executable,
            path=Path("/usr/bin/sandbox-exec"),
            invocation=Path("/usr/bin/sandbox-exec"),
        ),
        command=("/usr/bin/sandbox-exec", "-f", str(profile), *request.command),
    )
    monkeypatch.setattr(
        admission,
        "load",
        MagicMock(
            side_effect=[
                wrapped,
                admission.load(
                    Path(raw["waves"][1]["admission"]["path"]),
                    raw["waves"][1]["admission"]["sha256"],
                ),
            ]
        ),
    )
    plan_path.write_text(json.dumps(raw), encoding="utf-8")
    current = queue.load(plan_path, hashlib.sha256(plan_path.read_bytes()).hexdigest())
    assert current.waves[0].sandbox_profile is not None
    assert current.waves[0].python is not None
    assert any(Path(str(item["path"])).name == "sandbox.sb" for item in queue._pins(current))
    request.executable.path.write_bytes(b"changed synthetic interpreter")
    with pytest.raises((ReplayError, ValueError)):
        queue._pins(current)


def test_wrong_sandbox_wrapper_is_rejected(
    plan_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    raw = json.loads(plan_path.read_bytes())
    wave = raw["waves"][0]
    request = admission.load(Path(wave["admission"]["path"]), wave["admission"]["sha256"])
    wave["sandboxProfile"] = wave["workflow"]
    wave["pythonExecutable"] = {
        "path": str(request.executable.path),
        "sha256": request.executable.sha256,
        "invocationPath": str(request.executable.invocation),
        "linkTarget": None,
    }
    monkeypatch.setattr(admission, "load", MagicMock(return_value=request))
    plan_path.write_text(json.dumps(raw), encoding="utf-8")
    with pytest.raises(ReplayError, match="sandbox-exec wrapper"):
        queue.load(plan_path, hashlib.sha256(plan_path.read_bytes()).hexdigest())


def test_predecessor_barriers_complete_before_peer_checks_or_admission(
    plan: queue.Plan, normal_runner: MagicMock, monkeypatch: pytest.MonkeyPatch
) -> None:
    proof = plan.reference.path.with_name("predecessor-proof.json")
    proof.write_text('{"resourceReleased":true}', encoding="utf-8")
    waited = Event()
    verified = Event()

    def wait(_request: completion_barrier.Request) -> Path:
        assert normal_runner.call_count == 0
        waited.set()
        return proof

    def verify(_request: completion_barrier.Request) -> None:
        assert waited.is_set() is True
        verified.set()

    def peers(_wave: queue.Wave) -> list[dict[str, object]]:
        assert verified.is_set() is True
        return []

    current = replace(plan, barriers=(plan.reference,))
    monkeypatch.setattr(completion_barrier, "load", MagicMock())
    monkeypatch.setattr(completion_barrier, "wait", wait)
    monkeypatch.setattr(completion_barrier, "verify", verify)
    monkeypatch.setattr(queue, "_peers", peers)
    output = plan.reference.path.with_name("barrier-queue")
    assert queue.run(current, output) == 0
    assert normal_runner.call_count == 2
    assert (output / "predecessor-0000-release.json").exists() is True


def test_failed_predecessor_barrier_never_dispatches(
    plan: queue.Plan, normal_runner: MagicMock, monkeypatch: pytest.MonkeyPatch
) -> None:
    current = replace(plan, barriers=(plan.reference,))
    monkeypatch.setattr(completion_barrier, "load", MagicMock())
    monkeypatch.setattr(
        completion_barrier, "wait", MagicMock(side_effect=ReplayError("Prior producer failed"))
    )
    with pytest.raises(ReplayError, match="Prior producer failed"):
        queue.run(current, plan.reference.path.with_name("failed-predecessor"))
    assert normal_runner.call_count == 0


def test_barrier_schema_uses_existing_typed_loader(
    plan_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    raw = json.loads(plan_path.read_bytes())
    raw["barriers"] = [raw["waves"][0]["workflow"]]
    loader = MagicMock()
    monkeypatch.setattr(completion_barrier, "load", loader)
    plan_path.write_text(json.dumps(raw), encoding="utf-8")
    current = queue.load(plan_path, hashlib.sha256(plan_path.read_bytes()).hexdigest())
    assert len(current.barriers) == 1
    assert loader.call_count == 1


def test_formal_pump_mutex_name_matches_existing_lock_protocol(plan_path: Path) -> None:
    raw = json.loads(plan_path.read_bytes())
    raw["mutex"]["name"] = ".pump"
    plan_path.write_text(json.dumps(raw), encoding="utf-8")
    current = queue.load(plan_path, hashlib.sha256(plan_path.read_bytes()).hexdigest())
    assert current.mutex.name == ".pump"
    with queue._mutex(current.mutex) as acquired:
        assert acquired is True
        assert (current.mutex.directory / ".pump.lock").is_file() is True


@pytest.mark.parametrize("name", [".", "..", "../pump", "/pump", "bad name", "pump\n"])
def test_mutex_name_rejects_traversal_and_invalid_leaf(plan_path: Path, name: str) -> None:
    raw = json.loads(plan_path.read_bytes())
    raw["mutex"]["name"] = name
    plan_path.write_text(json.dumps(raw), encoding="utf-8")
    with pytest.raises(ReplayError, match="mutex leaf name"):
        queue.load(plan_path, hashlib.sha256(plan_path.read_bytes()).hexdigest())


def test_job_name_still_rejects_leading_dot() -> None:
    with pytest.raises(ReplayError, match="queue name"):
        queue._name(".pump")
