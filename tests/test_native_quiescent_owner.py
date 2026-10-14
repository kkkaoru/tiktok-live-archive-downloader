import json
import subprocess
from dataclasses import dataclass
from pathlib import Path

import pytest

from replay import native_quiescent_owner as failed
from replay import native_retired_hardlink_verification as refs
from replay.core import ReplayError
from replay.stages import Stage, file_digest


def write(path: Path, value: object) -> Path:
    path.write_text(json.dumps(value), encoding="utf-8")
    return path


def pin(path: Path) -> dict[str, str]:
    return {"path": str(path), "sha256": file_digest(path)}


@dataclass
class Host:
    active: bool = False
    opened: tuple[str, ...] = ()

    def groups(self) -> tuple[tuple[int, int], ...]:
        return ((42, 42),) if self.active else ()

    def birth(self, pid: int) -> str:
        return "123:4"

    def open_paths(self, paths: tuple[Path, ...]) -> tuple[str, ...]:
        return self.opened


@dataclass(frozen=True)
class Case:
    root: Path
    request: Path
    output: Path
    endpoint: Path
    host: Host


@pytest.fixture
def case(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Case:
    monkeypatch.setattr(refs, "WORK", tmp_path)
    source = tmp_path / "source.py"
    source.write_text("original source", encoding="utf-8")
    identities = write(tmp_path / "source-map.json", {"inputs": {str(source): file_digest(source)}})
    workflow = write(tmp_path / "workflow.json", {"version": 1})
    endpoint = tmp_path / "never-created.json"
    stage = Stage(
        name="reader",
        command=("python", "reader"),
        inputs=(source, identities),
        outputs=(endpoint,),
        timeout_seconds=10,
    )
    monkeypatch.setattr(failed, "load_stages", lambda path: (stage,))
    checkpoint = write(tmp_path / "reader.json", {"status": "running", "fingerprint": "a" * 64})
    status = tmp_path / "status"
    status.write_text("1\n", encoding="utf-8")
    capture = write(
        tmp_path / "capture.json",
        {
            "version": 1,
            "kind": "owned-family-completion-capture",
            "observedAtNs": 124_000_000_000,
            "parent": {"pid": 42, "birth": "123:4"},
            "processGroups": [42],
            "workflow": pin(workflow),
            "stage": "reader",
            "checkpoint": str(checkpoint),
            "statusPath": str(status),
            "initialCheckpointSHA256": file_digest(checkpoint),
            "initialCheckpoint": json.loads(checkpoint.read_text()),
        },
    )
    log = tmp_path / "failure.log"
    log.write_text("Actual synthetic failure", encoding="utf-8")
    release = write(
        tmp_path / "release.json",
        {
            "version": 1,
            "kind": "terminal-failed-writer-release-observation",
            "capture": pin(capture),
            "parent": {"pid": 42, "birth": "123:4"},
            "processGroups": [42],
            "matches": [],
            "parentAbsent": True,
            "capturedGroupsAbsent": True,
            "status": {**pin(status), "exit": 1},
            "checkpoint": {**pin(checkpoint), "status": "running"},
            "failureLog": pin(log),
            "completedStage": False,
            "qualityApproved": False,
        },
    )
    registry = write(
        tmp_path / "registry.json",
        {
            "complete": True,
            "owners": [
                {"id": "reader", "role": "reader", "process": {"pid": 42, "birth": "123:4"}}
            ],
            "outputs": [
                {
                    "path": str(endpoint),
                    "owner": "reader",
                    "state": "planned",
                    "reservedAdditionalBytes": 67108864,
                }
            ],
        },
    )
    request = write(
        tmp_path / "request.json",
        {
            "version": 1,
            "kind": "native-failed-owner-request",
            "owner": "reader",
            "originalRegistry": pin(registry),
            "capture": pin(capture),
            "failureRelease": pin(release),
            "sourceEvidence": [pin(identities)],
        },
    )
    return Case(tmp_path, request, tmp_path / "seal.json", endpoint, Host())


def test_failed_stage_is_only_inactive_without_completion(case: Case) -> None:
    proof = failed.seal(case.request, case.output, host=case.host)
    assert proof.owner == "reader"
    raw = json.loads(case.output.read_text())
    assert raw["completedStage"] is False
    assert raw["completionCredit"] is False
    assert raw["futureAllocationBytes"] == 0
    assert raw["protectedReservationsReleased"] is False
    assert raw["globalTransientDescriptorCoverage"] is False
    assert (
        json.loads((case.root / "registry.json").read_text())["outputs"][0][
            "reservedAdditionalBytes"
        ]
        == 67108864
    )


@pytest.mark.parametrize(
    "fault",
    [
        "success",
        "status",
        "checkpoint",
        "capture",
        "source",
        "missing-source",
        "endpoint",
        "active",
        "fd",
        "owner",
        "role",
        "state",
        "schema",
    ],
)
def test_reject_false_quiescence(case: Case, fault: str) -> None:
    request = json.loads(case.request.read_text())
    release = json.loads((case.root / "release.json").read_text())
    registry = json.loads((case.root / "registry.json").read_text())
    if fault == "success":
        release["status"]["exit"] = 0
    elif fault == "status":
        (case.root / "status").write_text("2\n", encoding="utf-8")
    elif fault == "checkpoint":
        release["checkpoint"]["sha256"] = "f" * 64
    elif fault == "capture":
        release["parent"]["birth"] = "124:5"
    elif fault == "source":
        request["sourceEvidence"] = [pin(case.root / "registry.json")]
    elif fault == "missing-source":
        request["sourceEvidence"] = []
    elif fault == "endpoint":
        case.endpoint.write_text("appeared", encoding="utf-8")
    elif fault == "active":
        case.host.active = True
    elif fault == "fd":
        case.host.opened = (str(case.endpoint),)
    elif fault == "owner":
        registry["owners"][0]["process"]["pid"] = 43
    elif fault == "role":
        registry["owners"][0]["role"] = "native"
    elif fault == "state":
        registry["outputs"][0]["state"] = "active"
    else:
        request["kind"] = "completed"
    write(case.root / "release.json", release)
    write(case.root / "registry.json", registry)
    request["failureRelease"] = pin(case.root / "release.json")
    request["originalRegistry"] = pin(case.root / "registry.json")
    write(case.request, request)
    with pytest.raises(ReplayError):
        failed.seal(case.request, case.output, host=case.host)
    assert not case.output.exists()


def test_changed_seal_and_later_appearing_endpoint_rejected(case: Case) -> None:
    failed.seal(case.request, case.output, host=case.host)
    raw = json.loads(case.output.read_text())
    raw["completionCredit"] = True
    write(case.output, raw)
    with pytest.raises(ReplayError, match="seal"):
        failed.verify(
            case.output, owner="reader", process={"pid": 42, "birth": "123:4"}, host=case.host
        )


def test_historical_source_not_executed_or_claimed_current(case: Case) -> None:
    (case.root / "source.py").write_text("new generation", encoding="utf-8")
    assert failed.seal(case.request, case.output, host=case.host).owner == "reader"


@pytest.mark.parametrize(
    "code,out,err,raises",
    [
        (1, b"", b"", False),
        (0, b"p7\nnOTHER\n", b"", False),
        (2, b"", b"", True),
        (1, b"", b"denied", True),
        (0, b"\xff", b"", True),
        (0, b"x" * (1024 * 1024 + 1), b"", True),
    ],
)
def test_scoped_fd_observation(
    case: Case, monkeypatch: pytest.MonkeyPatch, code: int, out: bytes, err: bytes, raises: bool
) -> None:
    def run(*args: object, **kwargs: object) -> subprocess.CompletedProcess[bytes]:
        return subprocess.CompletedProcess(["lsof"], code, out, err)

    monkeypatch.setattr(subprocess, "run", run)
    if raises:
        with pytest.raises(ReplayError):
            failed.SystemHost().open_paths((case.endpoint,))
    else:
        assert failed.SystemHost().open_paths((case.endpoint,)) == ()


def test_scoped_fd_finds_exact_deleted_endpoint(
    case: Case, monkeypatch: pytest.MonkeyPatch
) -> None:
    def run(*args: object, **kwargs: object) -> subprocess.CompletedProcess[bytes]:
        return subprocess.CompletedProcess(
            ["lsof"], 0, ("p7\nn" + str(case.endpoint) + " (deleted)\n").encode(), b""
        )

    monkeypatch.setattr(subprocess, "run", run)
    assert len(failed.SystemHost().open_paths((case.endpoint,))) == 1
    with pytest.raises(ReplayError, match="count"):
        failed.SystemHost().open_paths(())


def test_cli_success_and_error(case: Case, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(failed, "SystemHost", lambda: case.host)
    assert failed.main([str(case.request), str(case.output)]) == 0
    assert failed.main([str(case.request), str(case.output)]) == 1
