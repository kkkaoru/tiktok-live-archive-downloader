import json
import os
import tempfile
import time
from dataclasses import dataclass, replace
from pathlib import Path

import pytest
import test_fcp_native_queue as native_tests
from test_fcp_native_queue import FakeTransport, envelope

from replay import fcp_native_admission as admission
from replay import fcp_native_delivery_pump as pump
from replay import fcp_native_queue as queue
from replay import fcp_native_queue_followup as followup
from replay import native_retired_hardlink_verification as hardlinks
from replay import workflow
from replay.artifact_json import _path
from replay.core import ReplayError
from replay.recordings import record
from replay.stages import execution_lock, file_digest

native_case = native_tests.case


def write(path: Path, value: dict[str, object]) -> Path:
    path.write_text(json.dumps(value), encoding="utf-8")
    return path


def read(path: Path) -> dict[str, object]:
    return record(json.loads(path.read_text(encoding="utf-8")))


def pin(path: Path) -> dict[str, str]:
    return {"path": str(path), "sha256": file_digest(path)}


class Host:
    pressure_value = 1
    available = 10**12
    birth_value = "100:200"

    def __init__(self) -> None:
        self.observer = admission.SystemHost()

    def now_ns(self) -> int:
        return time.time_ns()

    def birth(self, pid: int) -> str:
        return self.birth_value

    def free(self, path: Path) -> int:
        return self.available

    def floor(self, path: Path) -> int:
        return 30 * 1024**3

    def pressure(self) -> int:
        return self.pressure_value

    def observe(self, path: Path) -> admission.FileState:
        return self.observer.observe(path)


@dataclass(frozen=True)
class Case:
    config: Path
    root: Path
    host: Host
    template: Path
    runtime: Path


@pytest.fixture(params=[("portrait", 7), ("landscape", 21)])
def case(
    native_case: queue.Queue,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    request: pytest.FixtureRequest,
) -> Case:
    assert isinstance(request.param, tuple)
    orientation, part = request.param
    assert isinstance(orientation, str) and isinstance(part, int)
    monkeypatch.setattr(pump, "WORK", tmp_path)
    monkeypatch.setattr(admission, "WORK", tmp_path)
    root = tmp_path / "pump"
    root.mkdir()
    scratch = tmp_path / "scratch"
    scratch.mkdir()
    monkeypatch.setenv("TMPDIR", str(scratch))
    monkeypatch.setenv("TMP", str(scratch))
    monkeypatch.setenv("TEMP", str(scratch))
    monkeypatch.setenv("XDG_CACHE_HOME", str(scratch))
    monkeypatch.setattr(tempfile, "gettempdir", lambda: str(scratch))
    descriptor = read(native_case.config)
    descriptor["inventory"] = str(root / "current-inventory.json")
    rows = descriptor["parts"]
    assert isinstance(rows, list)
    row = record(rows[0])
    row["assemblyPart"] = part
    write(native_case.config, descriptor)
    target = {"pid": 123, "birth": "100:200"}
    registry = write(
        tmp_path / "registry.json",
        {
            "version": 1,
            "kind": "native-admission-registry",
            "generation": orientation,
            "previous": None,
            "complete": True,
            "approvedOutputRoots": [str(tmp_path)],
            "owners": [
                {
                    "id": "native-owner",
                    "process": {"pid": os.getpid(), "birth": "100:200"},
                    "role": "native",
                }
            ],
            "outputs": [
                {
                    "path": str(native_case.parts[0].movie),
                    "owner": "native-owner",
                    "state": "planned",
                    "reservedAdditionalBytes": 0,
                }
            ],
        },
    )
    template = write(
        tmp_path / "admission.json",
        {
            "version": 1,
            "kind": "native-admission-preparation-request",
            "registry": pin(registry),
            "nativeObservation": None,
            "target": target,
            "nativeOwner": "native-owner",
            "releases": [],
            "requested": {
                "path": str(native_case.parts[0].movie),
                "peakAdditionalBytes": native_case.peak_bytes,
            },
            "policy": {
                "maximumAgeSeconds": 30,
                "maximumStagingCount": 4,
                "maximumStagingBytes": 10**12,
                "maximumActiveRenders": 1,
                "maximumPressure": 2,
                "minimumFreeBytes": 1,
                "bootVolume": str(tmp_path),
            },
        },
    )
    capabilities = write(
        tmp_path / "capabilities.json",
        {
            "ok": True,
            "data": {
                "isError": False,
                "structuredContent": {"observed": "synthetic capability evidence"},
            },
        },
    )
    runtime = write(
        tmp_path / "runtime.json",
        {
            "version": 1,
            "kind": "native-delivery-runtime-evidence",
            "executor": pin(native_case.executable),
            "provider": native_case.provider,
            "target": target,
            "capabilities": pin(capabilities),
            "scratch": {
                "roots": [str(scratch)],
                "observedPaths": [str(scratch / "actual-child-cache")],
            },
        },
    )
    binding = write(tmp_path / "binding.json", {"version": 1, "synthetic": orientation})
    config = write(
        tmp_path / "pump.json",
        {
            "version": 1,
            "kind": "native-delivery-pump",
            "queue": pin(native_case.config),
            "admissionTemplate": pin(template),
            "runtimeEvidence": pin(runtime),
            "outputRoot": str(root),
            "followup": {
                "assembly": pin(binding),
                "publishedAssembly": pin(binding),
                "sourceCleanup": pin(binding),
                "ffmpeg": pin(native_case.executable),
                "ffprobe": pin(native_case.executable),
                "audioSampleRate": 48000,
                "audioChannels": 2,
                "receiptOutput": str(root / "publication.json"),
            },
        },
    )
    return Case(config, root, Host(), template, runtime)


@pytest.fixture
def delegated(case: Case, monkeypatch: pytest.MonkeyPatch) -> list[str]:
    calls: list[str] = []

    def prepare(request: Path, output: Path) -> Path:
        calls.append("prepare")
        raw = read(request)
        assert (
            raw["exportResponse"]
            == read(case.root.parent / "events" / "followup-ready.json")["exportResponse"]
        )
        output.mkdir()
        prepared = output / "prepared"
        prepared.mkdir()
        graph = write(prepared / "workflow.json", {"synthetic": True})
        write(output / "preparation.json", {"synthetic": "prepared actual request"})
        write(prepared / "operation.json", {"request": pin(request)})
        return graph

    def check(arguments: list[str]) -> int:
        calls.append("check")
        assert arguments[-1] == "--check"
        return 0

    def timed(log: Path, phase: str, command: list[str]) -> int:
        calls.append("workflow")
        assert phase == "native-followup"
        assert command[1:3] == ["-m", "replay.workflow"]
        graph = Path(command[3])
        write(graph.parent / "verified-followup.json", {"kind": "synthetic actual domain terminal"})
        write(case.root / "publication.json", {"qualityApproved": False})
        return 0

    monkeypatch.setattr(followup, "prepare", prepare)
    monkeypatch.setattr(workflow, "main", check)
    monkeypatch.setattr(pump, "run_timed", timed)
    return calls


def test_real_queue_real_admission_delegates_followup_and_resume(
    case: Case, delegated: list[str]
) -> None:
    loaded = pump.load(case.config)
    transport = FakeTransport(loaded.native.parts[0])
    result = pump.run(loaded, transport=transport, host=case.host)
    assert result.state == "published-candidate"
    assert result.quality_approved is False
    assert delegated == ["prepare", "check", "workflow"]
    assert transport.calls.count("native.org.default.fcp_export") == 1
    assert transport.calls.count("native.org.default.ui_inspect") == 3
    initial = read(case.root / "refresh-initial" / "snapshot.json")
    project = read(case.root / "refresh-project" / "snapshot.json")
    assert initial["admitted"] is True
    assert project["admitted"] is True
    assert (case.root / "current-inventory.json").read_bytes() == (
        case.root / "refresh-project" / "snapshot.json"
    ).read_bytes()
    pump.run(loaded, transport=transport, host=case.host)
    assert transport.calls.count("native.org.default.fcp_export") == 1
    assert delegated == ["prepare", "check", "workflow", "check", "workflow"]


def approved_original(case: Case) -> Path:
    original = case.root.parent / "approved-original.mov"
    original.write_bytes(b"synthetic explicitly approved source")
    config = read(case.config)
    record(config["followup"])["approvedOriginal"] = pin(original)
    write(case.config, config)
    return original


def test_optional_approved_original_is_propagated_and_pinned_across_reentry(
    case: Case, delegated: list[str]
) -> None:
    original = approved_original(case)
    loaded = pump.load(case.config)
    assert any(
        ref.path == original and ref.sha256 == file_digest(original) for ref in loaded.inputs
    )
    transport = FakeTransport(loaded.native.parts[0])
    pump.run(loaded, transport=transport, host=case.host)
    assert read(case.root / "followup-request.json")["approvedOriginal"] == pin(original)
    pump.run(loaded, transport=transport, host=case.host)
    assert transport.calls.count("native.org.default.fcp_export") == 1
    assert delegated == ["prepare", "check", "workflow", "check", "workflow"]
    completed = read(case.root / "followup-complete.json")
    before = (case.root / "followup-complete.json").read_bytes()
    assert completed["qualityApproved"] is False
    original.write_bytes(b"changed source after completed native export")
    with pytest.raises(ReplayError, match="changed"):
        pump.run(loaded, transport=transport, host=case.host)
    assert transport.calls.count("native.org.default.fcp_export") == 1
    assert delegated == ["prepare", "check", "workflow", "check", "workflow"]
    assert (case.root / "followup-complete.json").read_bytes() == before


@pytest.mark.parametrize("mode", ["bad-sha", "linked", "extra-field", "missing", "unknown-field"])
def test_optional_approval_rejects_malformed_or_unpinned_inputs(case: Case, mode: str) -> None:
    original = approved_original(case)
    raw = read(case.config)
    fixed = record(raw["followup"])
    if mode == "bad-sha":
        fixed["approvedOriginal"] = {"path": str(original), "sha256": "0" * 64}
    elif mode == "linked":
        alias = case.root.parent / "linked-original.mov"
        alias.symlink_to(original)
        fixed["approvedOriginal"] = {"path": str(alias), "sha256": file_digest(original)}
    elif mode == "extra-field":
        fixed["approvedOriginal"] = {**pin(original), "trusted": True}
    elif mode == "missing":
        original.unlink()
    else:
        fixed["approvalOverride"] = pin(original)
    write(case.config, raw)
    expected = FileNotFoundError if mode == "missing" else ReplayError
    with pytest.raises(expected):
        pump.load(case.config)
    assert not (case.root / "native-complete.json").exists()


def test_approved_source_changed_before_native_is_not_dispatched(case: Case) -> None:
    original = approved_original(case)
    loaded = pump.load(case.config)
    transport = FakeTransport(loaded.native.parts[0])
    original.write_bytes(b"changed source before initial admission")
    with pytest.raises(ReplayError, match="changed"):
        pump.run(loaded, transport=transport, host=case.host)
    assert transport.calls == []


@pytest.mark.parametrize("field,value", [("kind", "other"), ("version", 2), ("unknown", True)])
def test_invalid_top_schema(case: Case, field: str, value: object) -> None:
    raw = read(case.config)
    raw[field] = value
    write(case.config, raw)
    with pytest.raises(ReplayError):
        pump.load(case.config)


@pytest.mark.parametrize(
    "field,value", [("audioSampleRate", 0), ("audioChannels", True), ("approvedOriginal", {})]
)
def test_followup_template_rejects_unsupported_clock_or_alias(
    case: Case, field: str, value: object
) -> None:
    raw = read(case.config)
    record(raw["followup"])[field] = value
    write(case.config, raw)
    with pytest.raises(ReplayError):
        pump.load(case.config)


@pytest.mark.parametrize(
    "field,value",
    [
        ("provider", "wrong"),
        ("target", {"pid": 124, "birth": "100:200"}),
        ("scratch", {}),
        ("scratch", {"roots": [], "observedPaths": []}),
    ],
)
def test_runtime_evidence_mismatch(case: Case, field: str, value: object) -> None:
    runtime = read(case.runtime)
    runtime[field] = value
    write(case.runtime, runtime)
    raw = read(case.config)
    raw["runtimeEvidence"] = pin(case.runtime)
    write(case.config, raw)
    with pytest.raises(ReplayError):
        pump.load(case.config)


def test_scratch_escape_and_link_rejected(case: Case) -> None:
    runtime = read(case.runtime)
    record(runtime["scratch"])["observedPaths"] = [str(case.root.parent.parent / "outside")]
    write(case.runtime, runtime)
    raw = read(case.config)
    raw["runtimeEvidence"] = pin(case.runtime)
    write(case.config, raw)
    with pytest.raises(ReplayError):
        pump.load(case.config)
    linked = case.root / "linked.json"
    linked.symlink_to(case.config)
    with pytest.raises(ReplayError, match="canonical"):
        pump.load(linked)


def test_duplicate_key_rejected(case: Case) -> None:
    case.config.write_text('{"version":1,"version":1}', encoding="utf-8")
    with pytest.raises(ReplayError, match="Duplicate"):
        pump.load(case.config)


@pytest.mark.parametrize("change", ["target", "path", "peak", "capacity", "allocation"])
def test_admission_identity_mismatch(case: Case, change: str) -> None:
    raw = read(case.template)
    if change == "target":
        raw["target"] = {"pid": 124, "birth": "100:200"}
    elif change == "path":
        record(raw["requested"])["path"] = str(case.root / "wrong.mov")
    elif change == "peak":
        record(raw["requested"])["peakAdditionalBytes"] = 12345
    elif change == "allocation":
        record(raw["requested"])["diskAllocation"] = pin(case.runtime)
    else:
        record(raw["policy"])["maximumActiveRenders"] = 2
    write(case.template, raw)
    config = read(case.config)
    config["admissionTemplate"] = pin(case.template)
    write(case.config, config)
    with pytest.raises(ReplayError):
        pump.load(case.config)


def test_current_index_cannot_be_historical_pin(case: Case) -> None:
    config = read(case.config)
    native_path = Path(str(record(config["queue"])["path"]))
    native = read(native_path)
    index = write(case.root / "current-inventory.json", {"old": True})
    native["corePins"] = [pin(index)]
    write(native_path, native)
    config["queue"] = pin(native_path)
    write(case.config, config)
    with pytest.raises(ReplayError, match="historical queue pin"):
        pump.load(case.config)


@pytest.mark.parametrize("variable", ["TMPDIR", "TMP", "TEMP", "XDG_CACHE_HOME"])
def test_child_scratch_must_be_real_local(
    case: Case, monkeypatch: pytest.MonkeyPatch, variable: str
) -> None:
    loaded = pump.load(case.config)
    monkeypatch.delenv(variable)
    with pytest.raises(ReplayError, match="configured locally"):
        pump.run(loaded, transport=FakeTransport(loaded.native.parts[0]), host=case.host)
    assert not (case.root / "native-started.json").exists()


def test_actual_temp_cache_is_checked(case: Case, monkeypatch: pytest.MonkeyPatch) -> None:
    loaded = pump.load(case.config)
    monkeypatch.setattr(tempfile, "gettempdir", lambda: str(case.root))
    with pytest.raises(ReplayError, match="Actual Python"):
        pump.run(loaded, transport=FakeTransport(loaded.native.parts[0]), host=case.host)


def test_pressure_denial_retained_no_export(case: Case) -> None:
    loaded = pump.load(case.config)
    case.host.pressure_value = 4
    transport = FakeTransport(loaded.native.parts[0])
    with pytest.raises(ReplayError, match="memory-pressure"):
        pump.run(loaded, transport=transport, host=case.host)
    assert read(case.root / "refresh-initial" / "snapshot.json")["admitted"] is False
    assert "native.org.default.fcp_export" not in transport.calls
    with pytest.raises(ReplayError, match="occupied or uncertain"):
        pump.run(loaded, transport=transport, host=case.host)


def test_other_gui_owner_cannot_be_taken_over(case: Case) -> None:
    loaded = pump.load(case.config)
    lease = case.root.parent / ".fcp-native-queue-owner.json"
    lease.write_bytes(b"other live owner")
    with pytest.raises(FileExistsError):
        pump.run(loaded, transport=FakeTransport(loaded.native.parts[0]), host=case.host)
    assert lease.read_bytes() == b"other live owner"


def test_lease_identity_change_prevents_refresh(case: Case) -> None:
    loaded = pump.load(case.config)
    write(
        case.root.parent / ".fcp-native-queue-owner.json",
        {"pid": 1, "birth": "wrong", "queueSHA256": loaded.native.digest},
    )
    with pytest.raises(ReplayError, match="does not own"):
        pump._refresh(loaded, FakeTransport(loaded.native.parts[0]), case.host, "refresh")


def test_uncertain_transport_retains_intent_and_no_retry(case: Case) -> None:
    loaded = pump.load(case.config)
    transport = FakeTransport(loaded.native.parts[0])
    transport.failure = ReplayError("Uncertain actual native call")
    with pytest.raises(ReplayError, match="Uncertain"):
        pump.run(loaded, transport=transport, host=case.host)
    assert (case.root / "refresh-initial" / "background-open-intent.json").exists()
    with pytest.raises(ReplayError, match="occupied or uncertain"):
        pump.run(loaded, transport=transport, host=case.host)
    assert len(transport.calls) == 1


def test_changed_input_before_effects(case: Case) -> None:
    loaded = pump.load(case.config)
    case.template.write_text("changed", encoding="utf-8")
    with pytest.raises(ReplayError, match="reader bytes changed"):
        pump.run(loaded, transport=FakeTransport(loaded.native.parts[0]), host=case.host)


def test_project_refresh_denied_does_not_dispatch_share(
    case: Case, delegated: list[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    loaded = pump.load(case.config)
    transport = FakeTransport(loaded.native.parts[0])
    original = transport.call

    def call(tool: str, arguments: dict[str, object], timeout: int) -> dict[str, object]:
        result = original(tool, arguments, timeout)
        if tool.endswith("fcp_project_open"):
            case.host.pressure_value = 4
        return result

    monkeypatch.setattr(transport, "call", call)
    with pytest.raises(ReplayError, match="uncertain or bounded"):
        pump.run(loaded, transport=transport, host=case.host)
    assert "native.org.default.fcp_export" not in transport.calls
    assert (case.root.parent / "events" / "part-0001-project.uncertain.json").exists()
    assert delegated == []


def test_bad_project_readback_never_refreshes(case: Case) -> None:
    loaded = pump.load(case.config)
    transport = FakeTransport(loaded.native.parts[0])
    transport.override = envelope({"pid": 123, "finalCut": {"project": {"name": "wrong"}}})
    wrapper = pump.RefreshTransport(loaded, transport, case.host)
    with pytest.raises(ReplayError, match="exact project"):
        wrapper.call("native.org.default.fcp_project_open", {}, 30)
    assert not (case.root / "refresh-project").exists()


def test_complete_native_resume_never_reexports_after_prepare_failure(
    case: Case, delegated: list[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    loaded = pump.load(case.config)
    transport = FakeTransport(loaded.native.parts[0])

    def fail(request: Path, output: Path) -> Path:
        raise ReplayError("Partial preparation failure")

    monkeypatch.setattr(followup, "prepare", fail)
    with pytest.raises(ReplayError, match="Partial preparation"):
        pump.run(loaded, transport=transport, host=case.host)
    assert (case.root / "native-complete.json").exists()
    with pytest.raises(ReplayError, match="uncheckpointed"):
        pump.run(loaded, transport=transport, host=case.host)
    assert transport.calls.count("native.org.default.fcp_export") == 1
    assert delegated == []


@pytest.mark.parametrize(
    "name", ["native-complete.json", "preparation-complete.json", "followup-complete.json"]
)
def test_changed_checkpoint_never_replays(case: Case, delegated: list[str], name: str) -> None:
    loaded = pump.load(case.config)
    transport = FakeTransport(loaded.native.parts[0])
    pump.run(loaded, transport=transport, host=case.host)
    raw = read(case.root / name)
    raw["qualityApproved"] = True
    write(case.root / name, raw)
    with pytest.raises(ReplayError, match="checkpoint identity"):
        pump.run(loaded, transport=transport, host=case.host)
    assert transport.calls.count("native.org.default.fcp_export") == 1


def test_changed_preparation_artifact_is_not_recreated(case: Case, delegated: list[str]) -> None:
    loaded = pump.load(case.config)
    transport = FakeTransport(loaded.native.parts[0])
    pump.run(loaded, transport=transport, host=case.host)
    graph = case.root / "followup" / "prepared" / "workflow.json"
    graph.write_bytes(b"changed prepared workflow")
    with pytest.raises(ReplayError, match="artifact changed"):
        pump.run(loaded, transport=transport, host=case.host)
    assert transport.calls.count("native.org.default.fcp_export") == 1


def test_incomplete_checkpoint_outputs(case: Case) -> None:
    loaded = pump.load(case.config)
    pump._checkpoint(loaded, "bad.json", (case.template,))
    raw = read(case.root / "bad.json")
    raw["outputs"] = []
    write(case.root / "bad.json", raw)
    with pytest.raises(ReplayError, match="inventory is invalid"):
        pump._completed(loaded, "bad.json")


def test_workflow_failed_can_resume_only_existing_state(
    case: Case, delegated: list[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    loaded = pump.load(case.config)
    transport = FakeTransport(loaded.native.parts[0])
    with monkeypatch.context() as patch:
        patch.setattr(pump, "run_timed", lambda log, phase, command: 1)
        with pytest.raises(ReplayError, match="followup stopped"):
            pump.run(loaded, transport=transport, host=case.host)
    pump.run(loaded, transport=transport, host=case.host)
    assert transport.calls.count("native.org.default.fcp_export") == 1
    assert delegated.count("prepare") == 1


def test_preflight_failure_does_not_run_media(
    case: Case, delegated: list[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    loaded = pump.load(case.config)
    monkeypatch.setattr(workflow, "main", lambda argv: 1)
    with pytest.raises(ReplayError, match="preflight failed"):
        pump.run(loaded, transport=FakeTransport(loaded.native.parts[0]), host=case.host)
    assert delegated == ["prepare"]


def test_concurrent_pump_does_not_call_apps(case: Case) -> None:
    loaded = pump.load(case.config)
    transport = FakeTransport(loaded.native.parts[0])
    with execution_lock(case.root, ".pump") as owned:
        assert owned is True
        with pytest.raises(ReplayError, match="Another process"):
            pump.run(loaded, transport=transport, host=case.host)
    assert transport.calls == []


def test_cli_check_and_error(case: Case, capsys: pytest.CaptureFixture[str]) -> None:
    assert pump.main(["check", str(case.config)]) == 0
    assert '"canDispatch": false' in capsys.readouterr().out
    case.config.write_bytes(b"invalid JSON")
    assert pump.main(["run", str(case.config)]) == 1
    assert "pump stopped" in capsys.readouterr().err


@pytest.mark.parametrize("inactive", [True, False])
def test_failed_reader_inactivity_requires_verified_quiescence(
    case: Case,
    monkeypatch: pytest.MonkeyPatch,
    inactive: bool,
) -> None:
    loaded = pump.load(case.config)
    registry = admission._registry(read(_path(record(loaded.template["registry"])["path"])))
    reader = admission.Owner(
        "failed-reader",
        admission.Identity(555, "different-birth"),
        "reader",
        quiescence=hardlinks.reference(pin(case.runtime)) if inactive else None,
    )
    revised = replace(registry, owners=(*registry.owners, reader))
    checks: list[str] = []

    def verify(evidence: admission.Evidence, value: admission.Registry) -> None:
        checks.append("verified")

    monkeypatch.setattr(admission, "_registry", lambda raw: revised)
    monkeypatch.setattr(admission, "_quiescent_owners", verify)
    if inactive:
        pump._reader_capacity(loaded, case.host)
        assert checks == ["verified", "verified"]
    else:
        with pytest.raises(ReplayError, match="identity changed"):
            pump._reader_capacity(loaded, case.host)


def test_cli_real_orchestration_only_mocked_native(
    case: Case, delegated: list[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    loaded = pump.load(case.config)
    transport = FakeTransport(loaded.native.parts[0])
    monkeypatch.setattr(queue, "ExecutorTransport", lambda executable: transport)
    monkeypatch.setattr(admission, "SystemHost", lambda: case.host)
    assert pump.main(["run", str(case.config)]) == 0
    assert delegated == ["prepare", "check", "workflow"]


def test_index_mutation_or_rejected_snapshot(case: Case) -> None:
    loaded = pump.load(case.config)
    lease = case.root.parent / ".fcp-native-queue-owner.json"
    write(lease, {"pid": os.getpid(), "birth": "100:200", "queueSHA256": loaded.native.digest})
    snapshot = write(case.root / "snapshot.json", {"admitted": False})
    with pytest.raises(ReplayError, match="accepted real snapshot"):
        pump._index(loaded, snapshot, None, host=case.host)
    write(loaded.native.inventory, {"other": "writer"})
    with pytest.raises(ReplayError, match="changed concurrently"):
        pump._index(loaded, snapshot, None, host=case.host)


def test_missing_root_or_outside_work(case: Case) -> None:
    raw = read(case.config)
    raw["outputRoot"] = str(case.root / "missing")
    write(case.config, raw)
    with pytest.raises(ReplayError, match="existing root"):
        pump.load(case.config)
    with pytest.raises(ReplayError, match="work storage"):
        pump._work(str(case.root.parent.parent / "outside.json"))


def test_scratch_raw_guard(case: Case) -> None:
    loaded = pump.load(case.config)
    corrupted = replace(loaded, runtime={"scratch": {"roots": "wrong"}})
    with pytest.raises(ReplayError, match="explicit scratch roots"):
        pump._scratch(corrupted)


@pytest.mark.parametrize(
    "occupied", ["followup", "followup-request.json", "publication.json", "refresh-initial"]
)
def test_known_occupied_namespace_rejected_before_any_native_call(
    case: Case, occupied: str
) -> None:
    loaded = pump.load(case.config)
    (case.root / occupied).write_bytes(b"pre-existing evidence")
    transport = FakeTransport(loaded.native.parts[0])
    with pytest.raises(ReplayError, match="occupied or uncertain"):
        pump.run(loaded, transport=transport, host=case.host)
    assert transport.calls == []


def test_app_birth_changed_before_observation(case: Case) -> None:
    loaded = pump.load(case.config)
    write(
        case.root.parent / ".fcp-native-queue-owner.json",
        {"pid": os.getpid(), "birth": "100:200", "queueSHA256": loaded.native.digest},
    )
    case.host.birth_value = "changed-birth"
    with pytest.raises(ReplayError, match="does not own"):
        pump._refresh(loaded, FakeTransport(loaded.native.parts[0]), case.host, "refresh")


def test_same_identity_lease_byte_mutation_during_call(
    case: Case, monkeypatch: pytest.MonkeyPatch
) -> None:
    loaded = pump.load(case.config)
    lease = write(
        case.root.parent / ".fcp-native-queue-owner.json",
        {"pid": os.getpid(), "birth": "100:200", "queueSHA256": loaded.native.digest},
    )
    transport = FakeTransport(loaded.native.parts[0])
    original = transport.call

    def call(tool: str, arguments: dict[str, object], timeout: int) -> dict[str, object]:
        captured = original(tool, arguments, timeout)
        raw = read(lease)
        raw["changed"] = True
        write(lease, raw)
        return captured

    monkeypatch.setattr(transport, "call", call)
    with pytest.raises(ReplayError, match="ownership bytes changed"):
        pump._refresh(loaded, transport, case.host, "refresh")
    assert len(transport.calls) == 1


def test_busy_actual_initial_background_never_allocates(case: Case) -> None:
    loaded = pump.load(case.config)
    lease = case.root.parent / ".fcp-native-queue-owner.json"
    write(lease, {"pid": os.getpid(), "birth": "100:200", "queueSHA256": loaded.native.digest})
    transport = native_tests.RenderTransport(loaded.native.parts[0])
    with pytest.raises(ReplayError, match="background is busy"):
        pump._refresh(loaded, transport, case.host, "refresh")
    assert not loaded.native.parts[0].movie.exists()


@pytest.mark.parametrize("resource", ["pressure", "disk"])
def test_followup_resource_guard(case: Case, resource: str) -> None:
    loaded = pump.load(case.config)
    if resource == "pressure":
        case.host.pressure_value = 4
    else:
        case.host.available = 1
    with pytest.raises(ReplayError, match="capacity|pressure"):
        pump._reader_capacity(loaded, case.host)


def test_global_two_other_readers_rejects_new_third(case: Case) -> None:
    raw = read(case.template)
    registry_path = Path(str(record(raw["registry"])["path"]))
    registry = read(registry_path)
    owners = registry["owners"]
    assert isinstance(owners, list)
    owners.extend(
        [
            {"id": "other-reader", "process": {"pid": 124, "birth": "100:200"}, "role": "reader"},
            {
                "id": "other-followup",
                "process": {"pid": 125, "birth": "100:200"},
                "role": "followup",
            },
        ]
    )
    write(registry_path, registry)
    raw["registry"] = pin(registry_path)
    write(case.template, raw)
    config = read(case.config)
    config["admissionTemplate"] = pin(case.template)
    write(case.config, config)
    loaded = pump.load(case.config)
    with pytest.raises(ReplayError, match="global read capacity"):
        pump._reader_capacity(loaded, case.host)


def test_real_phase_timer_current_interpreter_workflow_and_domain_reentry(
    case: Case, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A real child imports replay and runs only tiny synthetic cp/cmp stages."""
    loaded = pump.load(case.config)
    terminal_source = write(
        case.root / "synthetic-terminal.json", {"synthetic": True, "qualityApproved": False}
    )
    publication_source = write(
        case.root / "synthetic-publication.json", {"synthetic": True, "qualityApproved": False}
    )

    def prepare(request: Path, output: Path) -> Path:
        output.mkdir()
        prepared = output / "prepared"
        prepared.mkdir()
        terminal = prepared / "verified-followup.json"
        publication = case.root / "publication.json"
        graph = write(
            prepared / "workflow.json",
            {
                "version": 1,
                "stages": [
                    {
                        "name": "synthetic-terminal",
                        "timeout_seconds": 30,
                        "command": ["/bin/cp", str(terminal_source), str(terminal)],
                        "inputs": [str(terminal_source)],
                        "outputs": [str(terminal)],
                        "verify_command": ["/usr/bin/cmp", str(terminal_source), str(terminal)],
                    },
                    {
                        "name": "synthetic-publication",
                        "timeout_seconds": 30,
                        "command": ["/bin/cp", str(publication_source), str(publication)],
                        "inputs": [str(publication_source), str(terminal)],
                        "outputs": [str(publication)],
                        "verify_command": [
                            "/usr/bin/cmp",
                            str(publication_source),
                            str(publication),
                        ],
                    },
                ],
            },
        )
        write(output / "preparation.json", {"synthetic": True, "request": pin(request)})
        write(prepared / "operation.json", {"synthetic": True, "request": pin(request)})
        return graph

    monkeypatch.setattr(followup, "prepare", prepare)
    transport = FakeTransport(loaded.native.parts[0])
    result = pump.run(loaded, transport=transport, host=case.host)
    assert result.state == "published-candidate"
    assert (case.root / "publication.json").read_bytes() == publication_source.read_bytes()
    rows = (case.root / "phases.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(rows) == 1
    assert read(case.root / "followup-complete.json")["qualityApproved"] is False
    assert record(json.loads(rows[0]))["exit"] == 0
    before = (case.root / "publication.json").stat().st_mtime_ns
    pump.run(loaded, transport=transport, host=case.host)
    assert (case.root / "publication.json").stat().st_mtime_ns == before
    assert transport.calls.count("native.org.default.fcp_export") == 1
    assert len((case.root / "phases.jsonl").read_text(encoding="utf-8").splitlines()) == 2


def test_verified_completed_readers_do_not_consume_live_capacity(
    case: Case, monkeypatch: pytest.MonkeyPatch
) -> None:
    raw = read(case.template)
    registry_path = Path(str(record(raw["registry"])["path"]))
    registry = read(registry_path)
    owners = registry["owners"]
    assert isinstance(owners, list)
    proof = write(case.root / "completed-reader.json", {"synthetic": True})
    owners.extend(
        [
            {
                "id": "completed-one",
                "process": {"pid": 124, "birth": "old:1"},
                "role": "reader",
                "completedOwner": pin(proof),
            },
            {
                "id": "completed-two",
                "process": {"pid": 125, "birth": "old:2"},
                "role": "reader",
                "completedOwner": pin(proof),
            },
        ]
    )
    write(registry_path, registry)
    raw["registry"] = pin(registry_path)
    write(case.template, raw)
    config = read(case.config)
    config["admissionTemplate"] = pin(case.template)
    write(case.config, config)
    calls: list[str] = []

    def verified(evidence: admission.Evidence, inventory: admission.Registry) -> None:
        assert inventory.owners[-1].completion is not None
        evidence.add(pin(proof))
        calls.append("verified")

    monkeypatch.setattr(admission, "_completed_owners", verified)
    monkeypatch.setattr(admission, "_completed_transition", lambda previous, current: None)
    pump._reader_capacity(pump.load(case.config), case.host)
    assert calls == ["verified", "verified"]


def test_stale_completed_reader_proof_denies_followup(
    case: Case, monkeypatch: pytest.MonkeyPatch
) -> None:
    def stale(evidence: admission.Evidence, inventory: admission.Registry) -> None:
        raise ReplayError("Completed-owner retained endpoint changed")

    monkeypatch.setattr(admission, "_completed_owners", stale)
    with pytest.raises(ReplayError, match="endpoint changed"):
        pump._reader_capacity(pump.load(case.config), case.host)
