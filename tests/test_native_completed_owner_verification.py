import json
from dataclasses import dataclass
from pathlib import Path

import pytest

from replay import native_completed_owner_verification as completed
from replay import native_retired_hardlink_verification as refs
from replay.core import ReplayError
from replay.native_completed_owner_verification import load_capture
from replay.stages import Stage, _fingerprint, file_digest


def write(path: Path, value: object) -> Path:
    path.write_text(json.dumps(value), encoding="utf-8")
    return path


def pin(path: Path) -> dict[str, str]:
    return {"path": str(path), "sha256": file_digest(path)}


@dataclass
class Host:
    rows: tuple[tuple[int, int], ...] = ()

    def groups(self) -> tuple[tuple[int, int], ...]:
        return self.rows

    def birth(self, pid: int) -> str:
        return "123:4"


@dataclass
class ReuseHost:
    snapshots: tuple[tuple[tuple[int, int], ...], ...]
    births: tuple[dict[int, str], ...]
    index: int = 0

    def groups(self) -> tuple[tuple[int, int], ...]:
        rows = self.snapshots[self.index]
        self.index += 1
        return rows

    def birth(self, pid: int) -> str:
        try:
            return self.births[self.index - 1][pid]
        except KeyError as error:
            raise ReplayError("Exact process birth is unknown") from error


@dataclass(frozen=True)
class Case:
    request: Path
    seal: Path
    registry: Path
    checkpoint: Path
    media: Path
    status: Path
    host: Host


@pytest.fixture
def case(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Case:
    monkeypatch.setattr(refs, "WORK", tmp_path)
    source = tmp_path / "source"
    source.write_bytes(b"source")
    media = tmp_path / "media"
    media.write_bytes(b"actual completed output")
    workflow = write(tmp_path / "workflow.json", {"version": 1})
    checkpoint = tmp_path / "repair.json"
    stage = Stage(
        name="repair",
        command=("python", "repair"),
        inputs=(source,),
        outputs=(media,),
        timeout_seconds=10,
    )
    write(
        checkpoint,
        {
            "status": "complete",
            "fingerprint": _fingerprint(stage),
            "outputs": {str(media): file_digest(media)},
        },
    )
    status = tmp_path / "status"
    status.write_text("1\n", encoding="utf-8")
    capture = write(tmp_path / "capture.json", {"genuineCaptured": True})
    registry = write(
        tmp_path / "registry.json",
        {
            "complete": True,
            "owners": [{"id": "lane", "role": "reader", "process": {"pid": 42, "birth": "123:4"}}],
            "outputs": [
                {
                    "path": str(media),
                    "owner": "lane",
                    "state": "active",
                    "reservedAdditionalBytes": 999,
                }
            ],
        },
    )
    request = write(
        tmp_path / "request.json",
        {
            "version": 1,
            "kind": "native-completed-owner-request",
            "originalRegistry": pin(registry),
            "owner": "lane",
            "capture": pin(capture),
            "stateDirectory": str(tmp_path),
        },
    )
    monkeypatch.setattr(completed, "load_stages", lambda path: (stage,))
    monkeypatch.setattr(
        completed,
        "load_capture",
        lambda path, digest: (
            completed.Parent(42, "123:4"),
            completed.Completion(workflow, file_digest(workflow), status, (42,), 124_000_000_000),
        ),
    )
    return Case(request, tmp_path / "seal.json", registry, checkpoint, media, status, Host())


def test_complete_child_with_failed_outer_preserves_status(case: Case) -> None:
    result = completed.seal(case.request, case.seal, host=case.host)
    assert result.owner == "lane"
    raw = json.loads(case.seal.read_text(encoding="utf-8"))
    assert raw["outerStatus"]["exitCode"] == 1
    assert raw["qualityApproved"] is False
    assert raw["allCapturedChildStagesComplete"] is True


def test_success_outer(case: Case) -> None:
    case.status.write_text("0\n", encoding="utf-8")
    assert completed.seal(case.request, case.seal, host=case.host).process == {
        "pid": 42,
        "birth": "123:4",
    }


@pytest.mark.parametrize("change", ["running", "fingerprint", "output"])
def test_bad_checkpoint(case: Case, change: str) -> None:
    raw = json.loads(case.checkpoint.read_text(encoding="utf-8"))
    raw.update(
        {
            "running": {"status": "running"},
            "fingerprint": {"fingerprint": "b" * 64},
            "output": {"outputs": {}},
        }[change]
    )
    write(case.checkpoint, raw)
    with pytest.raises(ReplayError, match="incomplete"):
        completed.seal(case.request, case.seal, host=case.host)
    assert not case.seal.exists()


@pytest.mark.parametrize("rows", [((42, 99),), ((50, 42),)])
def test_active_family(case: Case, rows: tuple[tuple[int, int], ...]) -> None:
    case.host.rows = rows
    with pytest.raises(ReplayError, match="still present|newly born leader"):
        completed.seal(case.request, case.seal, host=case.host)


def test_missing_retained_endpoint(case: Case) -> None:
    case.media.unlink()
    with pytest.raises(ReplayError, match="missing"):
        completed.seal(case.request, case.seal, host=case.host)


def test_terminal_status_required(case: Case) -> None:
    case.status.write_text("running", encoding="utf-8")
    with pytest.raises(ReplayError, match="terminal"):
        completed.seal(case.request, case.seal, host=case.host)


def test_wrong_process(case: Case) -> None:
    raw = json.loads(case.registry.read_text(encoding="utf-8"))
    raw["owners"][0]["process"]["birth"] = "other"
    write(case.registry, raw)
    request = json.loads(case.request.read_text(encoding="utf-8"))
    request["originalRegistry"] = pin(case.registry)
    write(case.request, request)
    with pytest.raises(ReplayError, match="identity"):
        completed.seal(case.request, case.seal, host=case.host)


@pytest.mark.parametrize(
    "field,value",
    [
        ("owner", "other"),
        ("qualityApproved", True),
        ("allCapturedChildStagesComplete", False),
        ("capturedGroups", [99]),
        ("endpoints", {}),
        ("pins", []),
    ],
)
def test_seal_mutation(case: Case, field: str, value: object) -> None:
    completed.seal(case.request, case.seal, host=case.host)
    raw = json.loads(case.seal.read_text(encoding="utf-8"))
    raw[field] = value
    write(case.seal, raw)
    with pytest.raises(ReplayError):
        completed.verify(
            case.seal, owner="lane", process={"pid": 42, "birth": "123:4"}, host=case.host
        )


def test_endpoint_change(case: Case) -> None:
    completed.seal(case.request, case.seal, host=case.host)
    case.media.write_bytes(b"changed endpoint")
    with pytest.raises(ReplayError, match="endpoint changed"):
        completed.verify(
            case.seal, owner="lane", process={"pid": 42, "birth": "123:4"}, host=case.host
        )


def test_historical_metadata_change(case: Case) -> None:
    completed.seal(case.request, case.seal, host=case.host)
    write(case.checkpoint, {"status": "running"})
    with pytest.raises(ReplayError, match="metadata"):
        completed.verify(
            case.seal, owner="lane", process={"pid": 42, "birth": "123:4"}, host=case.host
        )


def test_status_change(case: Case) -> None:
    completed.seal(case.request, case.seal, host=case.host)
    case.status.write_text("0", encoding="utf-8")
    with pytest.raises(ReplayError, match="terminal status changed"):
        completed.verify(
            case.seal, owner="lane", process={"pid": 42, "birth": "123:4"}, host=case.host
        )


def test_cli_failure(case: Case, capsys: pytest.CaptureFixture[str]) -> None:
    case.status.write_text("bad", encoding="utf-8")
    assert completed.main([str(case.request), str(case.seal)]) == 1
    assert capsys.readouterr().out == "NATIVE_COMPLETED_OWNER_ERROR\n"


def test_absent_planned_endpoint(case: Case) -> None:
    assert completed._endpoint(case.request.parent / "absent") == {"exists": False}


def test_symlink_endpoint(case: Case) -> None:
    alias = case.request.parent / "alias"
    alias.symlink_to(case.media)
    with pytest.raises(ReplayError, match="canonical regular"):
        completed._endpoint(alias)


@pytest.mark.parametrize("registry", [{}, {"complete": True, "owners": []}])
def test_invalid_owner_inventory(registry: dict[str, object]) -> None:
    with pytest.raises(ReplayError):
        completed._owner(registry, "lane")


@pytest.mark.parametrize("registry", [{}, {"outputs": []}, {"outputs": [{"owner": "other"}]}])
def test_invalid_output_inventory(registry: dict[str, object]) -> None:
    with pytest.raises(ReplayError):
        completed._paths(registry, "lane")


def test_invalid_request(case: Case) -> None:
    write(case.request, {})
    with pytest.raises(ReplayError, match="exact completed"):
        completed.seal(case.request, case.seal, host=case.host)


def test_cli_success(case: Case, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(completed.SystemHost, "groups", lambda self: ())
    assert completed.main([str(case.request), str(case.seal)]) == 0


def test_real_capture_parser(case: Case) -> None:
    path = write(
        case.request.parent / "actual-capture.json",
        {
            "version": 1,
            "kind": "owned-family-completion-capture",
            "parent": {"pid": 42, "birth": "123:4"},
            "workflow": pin(case.registry),
            "processGroups": [42],
            "observedAtNs": 124_000_000_000,
            "initialCheckpoint": {"status": "running"},
            "statusPath": str(case.status),
        },
    )
    assert load_capture(path, file_digest(path)) == (
        completed.Parent(42, "123:4"),
        completed.Completion(
            case.registry, file_digest(case.registry), case.status, (42,), 124_000_000_000
        ),
    )


@pytest.mark.parametrize("groups", [[42, 42], [42, 1], [0]])
def test_invalid_capture_groups(case: Case, groups: list[int]) -> None:
    path = write(
        case.request.parent / "actual-capture.json",
        {
            "version": 1,
            "kind": "owned-family-completion-capture",
            "parent": {"pid": 42, "birth": "123:4"},
            "workflow": pin(case.registry),
            "processGroups": groups,
            "observedAtNs": 124_000_000_000,
            "initialCheckpoint": {"status": "running"},
            "statusPath": str(case.status),
        },
    )
    with pytest.raises(ReplayError):
        load_capture(path, file_digest(path))


def test_actual_readonly_process_observation() -> None:
    assert isinstance(completed.SystemHost().groups(), tuple)


def test_revived_family_after_seal(case: Case) -> None:
    completed.seal(case.request, case.seal, host=case.host)
    case.host.rows = ((42, 42),)
    with pytest.raises(ReplayError, match="still present|newly born leader"):
        completed.verify(
            case.seal, owner="lane", process={"pid": 42, "birth": "123:4"}, host=case.host
        )


def test_pruned_checkpoint_pin_rejected(case: Case) -> None:
    completed.seal(case.request, case.seal, host=case.host)
    raw = json.loads(case.seal.read_text(encoding="utf-8"))
    raw["pins"] = [pin for pin in raw["pins"] if pin["path"] != str(case.checkpoint)]
    write(case.seal, raw)
    with pytest.raises(ReplayError, match="checkpoint pin is missing"):
        completed.verify(
            case.seal, owner="lane", process={"pid": 42, "birth": "123:4"}, host=case.host
        )


def test_young_recycled_group_leader(case: Case) -> None:
    host = ReuseHost(
        (((42, 42), (99, 42)), ((42, 42), (99, 42))),
        ({42: "250:0", 99: "251:0"}, {42: "250:0", 99: "251:0"}),
    )
    completed._absent(
        host,
        parent=completed.Parent(42, "123:4"),
        completion=completed.Completion(
            case.registry, file_digest(case.registry), case.status, (42,), 124_000_000_000
        ),
    )
    assert host.index == 2


def test_unrelated_recycled_parent_pid(case: Case) -> None:
    host = ReuseHost((((42, 99),), ((42, 99),)), ({42: "250:0"}, {42: "250:0"}))
    completed._absent(
        host,
        parent=completed.Parent(42, "123:4"),
        completion=completed.Completion(
            case.registry, file_digest(case.registry), case.status, (42,), 124_000_000_000
        ),
    )
    assert host.index == 2


@pytest.mark.parametrize(
    "rows,births,reason",
    [
        (((99, 42),), {99: "250:0"}, "newly born leader"),
        (((42, 42), (99, 42)), {42: "250:0", 99: "200:0"}, "older member"),
        (((42, 42),), {}, "birth is unknown"),
        (((42, 42),), {42: "invalid"}, "birth is unknown"),
        (((42, 42), (42, 42)), {42: "250:0"}, "Duplicate family"),
    ],
)
def test_unsafe_recycled_group(
    case: Case, rows: tuple[tuple[int, int], ...], births: dict[int, str], reason: str
) -> None:
    host = ReuseHost((rows, rows), (births, births))
    with pytest.raises(ReplayError, match=reason):
        completed._absent(
            host,
            parent=completed.Parent(42, "123:4"),
            completion=completed.Completion(
                case.registry, file_digest(case.registry), case.status, (42,), 124_000_000_000
            ),
        )


@pytest.mark.parametrize(
    "after,births",
    [
        (((42, 42),), {42: "260:0"}),
        (((42, 42), (99, 42)), {42: "250:0", 99: "251:0"}),
    ],
)
def test_recycled_group_changes_during_observation(
    case: Case, after: tuple[tuple[int, int], ...], births: dict[int, str]
) -> None:
    host = ReuseHost((((42, 42),), after), ({42: "250:0"}, births))
    with pytest.raises(ReplayError, match="changed during observation"):
        completed._absent(
            host,
            parent=completed.Parent(42, "123:4"),
            completion=completed.Completion(
                case.registry, file_digest(case.registry), case.status, (42,), 124_000_000_000
            ),
        )


def test_successor_preserves_historical_producer_without_executing_it(case: Case) -> None:
    completed.seal(case.request, case.seal, host=case.host)
    old = json.loads(case.seal.read_text(encoding="utf-8"))
    old["producerSHA256"] = "f" * 64
    write(case.seal, old)
    raw = json.loads(case.request.read_text(encoding="utf-8"))
    raw["predecessor"] = pin(case.seal)
    request = write(case.request.parent / "successor-request.json", raw)
    successor = case.request.parent / "successor.json"
    result = completed.seal(request, successor, host=case.host)
    assert result.predecessor == refs.reference(pin(case.seal))
    assert (
        json.loads(case.seal.read_text(encoding="utf-8"))["producerSHA256"]
        == "ffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff"
    )


@pytest.mark.parametrize(
    "field,value",
    [
        ("owner", "other"),
        ("endpoints", {}),
        ("capture", {}),
        ("process", {}),
        ("pins", []),
        ("unsupported", True),
    ],
)
def test_successor_rejects_changed_predecessor_generation(
    case: Case, field: str, value: object
) -> None:
    completed.seal(case.request, case.seal, host=case.host)
    old = json.loads(case.seal.read_text(encoding="utf-8"))
    old[field] = value
    write(case.seal, old)
    raw = json.loads(case.request.read_text(encoding="utf-8"))
    raw["predecessor"] = pin(case.seal)
    request = write(case.request.parent / "successor-request.json", raw)
    with pytest.raises(ReplayError):
        completed.seal(request, case.request.parent / "successor.json", host=case.host)


def test_successor_rejects_predecessor_hash_change(case: Case) -> None:
    completed.seal(case.request, case.seal, host=case.host)
    raw = json.loads(case.request.read_text(encoding="utf-8"))
    raw["predecessor"] = {"path": str(case.seal), "sha256": "a" * 64}
    request = write(case.request.parent / "successor-request.json", raw)
    with pytest.raises(ReplayError):
        completed.seal(request, case.request.parent / "successor.json", host=case.host)
