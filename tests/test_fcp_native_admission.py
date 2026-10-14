import ctypes
import json
import os
import sys
from dataclasses import dataclass, replace
from pathlib import Path

import pytest

from replay import fcp_native_admission as admission
from replay import fcp_native_queue_followup as followup
from replay import native_completed_owner_verification as owners
from replay import native_quiescent_owner as quiescent
from replay import native_retired_hardlink_verification as hardlinks
from replay.core import ReplayError
from replay.stages import file_digest


def preparation_request(case: "Case") -> Path:
    raw = json.loads(case.request.read_text())
    raw["kind"] = "native-admission-preparation-request"
    raw["nativeObservation"] = None
    return write(case.root / "prepare.json", raw)


def test_prepare_admission_without_host_or_native(
    case: "Case", monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(admission, "WORK", case.root)
    request = preparation_request(case)
    output = case.root / "candidate.json"
    admission.prepare_data(request, output)
    before = output.read_bytes()
    admission.prepare_data(request, output)
    assert output.read_bytes() == before
    admission.validate_candidate(output)
    assert json.loads(before)["hostObserved"] is False
    with pytest.raises(ReplayError, match="Unsupported"):
        admission.capture(output, case.root / "live.json", host=Host())


@pytest.mark.parametrize(
    "field,value",
    [("nativeObservation", {}), ("kind", "native-admission-request"), ("nativeOwner", "foreign")],
)
def test_prepare_admission_rejects_claims(
    case: "Case", monkeypatch: pytest.MonkeyPatch, field: str, value: object
) -> None:
    monkeypatch.setattr(admission, "WORK", case.root)
    request = preparation_request(case)
    raw = json.loads(request.read_text())
    raw[field] = value
    write(request, raw)
    with pytest.raises(ReplayError):
        admission.prepare_data(request, case.root / "candidate.json")


@pytest.mark.parametrize(
    "field,value",
    [("maximumAgeSeconds", 31), ("maximumPressure", True), ("maximumStagingCount", 0)],
)
def test_prepare_admission_keeps_policy_guards(
    case: "Case", monkeypatch: pytest.MonkeyPatch, field: str, value: object
) -> None:
    monkeypatch.setattr(admission, "WORK", case.root)
    request = preparation_request(case)
    raw = json.loads(request.read_text())
    raw["policy"][field] = value
    write(request, raw)
    with pytest.raises(ReplayError):
        admission.prepare_data(request, case.root / "candidate.json")


def test_admission_prepared_change_refused(case: "Case", monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(admission, "WORK", case.root)
    request = preparation_request(case)
    output = case.root / "candidate.json"
    admission.prepare_data(request, output)
    raw = json.loads(request.read_text())
    raw["requested"]["peakAdditionalBytes"] = 500
    write(request, raw)
    with pytest.raises(ReplayError):
        admission.validate_candidate(output)
    with pytest.raises(ReplayError):
        admission.prepare_data(request, output)


def test_retired_preparation_is_structural_not_release(
    case: "Case", monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(admission, "WORK", case.root)
    previous = write(case.root / "previous.json", json.loads(case.registry.read_text()))
    registry = json.loads(case.registry.read_text())
    registry["generation"] = "other-video-after-retirement"
    registry["previous"] = pin(previous)
    registry["outputs"] = [registry["outputs"][1]]
    write(case.registry, registry)
    request = preparation_request(case)
    metadata = write(case.root / "release.json", {"unverified": True})
    raw = json.loads(request.read_text())
    raw["registry"] = pin(case.registry)
    raw["releases"] = [
        {
            "path": str(case.retained),
            "assemblyPart": 42,
            "publication": pin(metadata),
            "retirement": pin(metadata),
            "terminal": pin(metadata),
            "postDeleteStatus": pin(metadata),
            "retirementAudit": pin(metadata),
            "consumer": pin(metadata),
            "publishedAssembly": pin(metadata),
            "history": [pin(metadata)],
        }
    ]
    write(request, raw)
    result = admission.prepare_data(request, case.root / "candidate.json")
    assert json.loads(result.read_text())["releaseVerified"] is False
    assert case.retained.exists()
    raw["releases"] = []
    write(request, raw)
    with pytest.raises(ReplayError, match="every retirement"):
        admission.prepare_data(request, case.root / "other.json")


def test_admission_candidate_rejects_claimed_readiness(
    case: "Case", monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(admission, "WORK", case.root)
    result = admission.prepare_data(preparation_request(case), case.root / "candidate.json")
    raw = json.loads(result.read_text())
    raw["readiness"] = {"ready": True}
    write(result, raw)
    with pytest.raises(ReplayError, match="unobserved"):
        admission.validate_candidate(result)


def test_admission_prepare_cli(case: "Case", monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(admission, "WORK", case.root)
    request = preparation_request(case)
    output = case.root / "candidate.json"
    monkeypatch.setattr(sys, "argv", ["admission", str(request), str(output), "--prepare-data"])
    admission.main()
    monkeypatch.setattr(sys, "argv", ["admission", str(output), "--validate-candidate"])
    admission.main()


def write(path: Path, data: object) -> Path:
    path.write_text(json.dumps(data), encoding="utf-8")
    return path


def pin(path: Path) -> dict[str, str]:
    return {"path": str(path), "sha256": file_digest(path)}


def synthetic_native_observation() -> dict[str, object]:
    return {
        "ok": True,
        "data": {
            "isError": False,
            "structuredContent": {
                "pid": 333,
                "truncated": False,
                "elements": [
                    {"value": "Sharing"},
                    *[
                        {"identifier": "_NS:166", "value": "Idle", "path": [index]}
                        for index in range(9)
                    ],
                ],
            },
        },
    }


class Host:
    now = 2_000_000_000_000_000_000
    available = 100_000
    level = 1
    birth_value = "100:200"

    def now_ns(self) -> int:
        return self.now

    def birth(self, pid: int) -> str:
        return self.birth_value

    def free(self, path: Path) -> int:
        return self.available

    def floor(self, path: Path) -> int:
        return 100

    def observe(self, path: Path) -> admission.FileState:
        return admission.SystemHost().observe(path)

    def pressure(self) -> int:
        return self.level


@dataclass(frozen=True)
class Case:
    root: Path
    request: Path
    registry: Path
    native: Path
    retained: Path
    planned: Path


@pytest.fixture
def case(tmp_path: Path) -> Case:
    retained = tmp_path / "retained.mov"
    retained.write_bytes(b"x" * 100)
    planned = tmp_path / "next.mov"
    registry = write(
        tmp_path / "registry.json",
        {
            "version": 1,
            "kind": "native-admission-registry",
            "generation": "generation-a",
            "previous": None,
            "complete": True,
            "approvedOutputRoots": [str(tmp_path)],
            "owners": [
                {
                    "id": "native-owner",
                    "process": {"pid": 111, "birth": "100:200"},
                    "role": "native",
                }
            ],
            "outputs": [
                {
                    "path": str(retained),
                    "owner": "native-owner",
                    "state": "retained",
                    "reservedAdditionalBytes": 0,
                },
                {
                    "path": str(planned),
                    "owner": "native-owner",
                    "state": "planned",
                    "reservedAdditionalBytes": 0,
                },
            ],
        },
    )
    native = write(
        tmp_path / "native.json",
        {
            "ok": True,
            "data": {
                "isError": False,
                "structuredContent": {
                    "pid": 333,
                    "truncated": False,
                    "elements": [
                        {"value": "Sharing"},
                        *[
                            {"identifier": "_NS:166", "value": "Idle", "path": [index]}
                            for index in range(9)
                        ],
                    ],
                },
            },
        },
    )
    os.utime(native, ns=(Host.now - 1_000_000_000, Host.now - 1_000_000_000))
    request = write(
        tmp_path / "request.json",
        {
            "version": 1,
            "kind": "native-admission-request",
            "registry": pin(registry),
            "nativeObservation": pin(native),
            "target": {"pid": 333, "birth": "100:200"},
            "nativeOwner": "native-owner",
            "policy": {
                "maximumAgeSeconds": 30,
                "maximumStagingCount": 2,
                "maximumStagingBytes": 1000,
                "maximumActiveRenders": 1,
                "maximumPressure": 1,
                "minimumFreeBytes": 100,
                "bootVolume": str(tmp_path),
            },
            "requested": {"path": str(planned), "peakAdditionalBytes": 400},
            "releases": [],
        },
    )
    return Case(tmp_path, request, registry, native, retained, planned)


def repin_registry(case: Case) -> None:
    raw = json.loads(case.request.read_text())
    raw["registry"] = pin(case.registry)
    write(case.request, raw)


@pytest.fixture
def inactive_reader(case: Case, monkeypatch: pytest.MonkeyPatch) -> Path:
    registry = json.loads(case.registry.read_text())
    registry["owners"].append(
        {"id": "failed-reader", "role": "reader", "process": {"pid": 555, "birth": "99:9"}}
    )
    endpoint = case.root / "never-reflow.json"
    registry["outputs"].append(
        {
            "path": str(endpoint),
            "owner": "failed-reader",
            "state": "planned",
            "reservedAdditionalBytes": 67108864,
        }
    )
    previous = write(case.root / "before-reader.json", registry)
    proof_path = write(case.root / "reader-proof.json", {"syntheticQuiescence": True})
    proof = quiescent.QuiescentOwner(
        "failed-reader",
        {"pid": 555, "birth": "99:9"},
        hardlinks.reference(pin(previous)),
        {str(endpoint): {"exists": False}},
        (),
    )

    def verify(
        path: Path, *, owner: str, process: dict[str, object], host: quiescent.Host
    ) -> quiescent.QuiescentOwner:
        return proof

    monkeypatch.setattr(quiescent, "verify", verify)
    registry = json.loads(previous.read_text())
    registry.update(generation="quiescent-generation", previous=pin(previous))
    registry["owners"][-1]["quiescentOwner"] = pin(proof_path)
    write(case.registry, registry)
    raw = json.loads(case.request.read_text())
    raw["policy"].update(maximumStagingBytes=100000000, maximumStagingCount=3)
    raw["registry"] = pin(case.registry)
    write(case.request, raw)
    return proof_path


def test_quiescent_reader_keeps_reservation_and_skips_live_birth(
    case: Case, inactive_reader: Path
) -> None:
    refs = admission._preparation(json.loads(preparation_request(case).read_text()))
    assert inactive_reader in refs
    host = Host()
    host.available = 100000000
    result = admission.capture(case.request, case.root / "quiescent-capture.json", host=host)
    assert result.admitted is True
    assert json.loads(result.path.read_text())["reservedAdditionalBytes"] == 67108864


@pytest.mark.parametrize(
    "fault", ["role", "both", "identity", "row", "initial", "baseline", "remove"]
)
def test_quiescent_transition_rejects_resets(case: Case, inactive_reader: Path, fault: str) -> None:
    raw = json.loads(case.registry.read_text())
    if fault == "role":
        raw["owners"][-1]["role"] = "native"
    elif fault == "both":
        raw["owners"][-1]["completedOwner"] = pin(inactive_reader)
    elif fault == "identity":
        raw["owners"][-1]["process"]["birth"] = "changed"
    elif fault == "row":
        raw["outputs"][-1]["reservedAdditionalBytes"] = 0
    elif fault == "initial":
        raw["previous"] = None
    elif fault == "baseline":
        other = write(
            case.root / "wrong-baseline.json",
            json.loads((case.root / "before-reader.json").read_text()),
        )
        raw["previous"] = pin(other)
    else:
        previous = write(case.root / "already-quiescent.json", raw)
        raw["previous"] = pin(previous)
        raw["generation"] = "removed-proof"
        raw["owners"][-1].pop("quiescentOwner")
    write(case.registry, raw)
    repin_registry(case)
    with pytest.raises(ReplayError):
        admission._preparation(json.loads(preparation_request(case).read_text()))


@pytest.fixture
def extended(case: Case) -> tuple[Path, Path]:
    previous = write(case.root / "previous.json", json.loads(case.registry.read_text()))
    raw = json.loads(case.request.read_text())
    raw["registry"] = pin(previous)
    write(case.request, raw)
    accepted = admission.capture(case.request, case.root / "accepted.json", host=Host())
    assert accepted.admitted is True
    ssd = case.root.parent / (case.root.name + "-ssd")
    ssd.mkdir()
    authorization = write(case.root / "authorization.json", {"syntheticUserSource": True})
    extension = write(
        case.root / "extension.json",
        {
            "version": 1,
            "kind": "native-output-root-extension",
            "previous": pin(previous),
            "acceptedPrevious": pin(accepted.path),
            "oldRoots": [str(case.root)],
            "addedRoots": [str(ssd)],
            "authorization": {
                "source": pin(authorization),
                "scope": "append-native-output-roots",
                "roots": [str(ssd)],
            },
            "evidence": [pin(authorization)],
        },
    )
    registry = json.loads(previous.read_text())
    registry.update(
        generation="extension-generation",
        previous=pin(previous),
        approvedOutputRoots=[str(case.root), str(ssd)],
        rootExtension=pin(extension),
    )
    registry["outputs"].append(
        {
            "path": str(ssd / "new.mov"),
            "owner": "native-owner",
            "state": "planned",
            "reservedAdditionalBytes": 0,
        }
    )
    write(case.registry, registry)
    raw["registry"] = pin(case.registry)
    raw["requested"]["path"] = str(ssd / "new.mov")
    write(case.request, raw)
    return previous, extension


def test_append_root_preparation_and_capture(case: Case, extended: tuple[Path, Path]) -> None:
    refs = admission._preparation(json.loads(preparation_request(case).read_text()))
    assert extended[1] in refs
    result = admission.capture(case.request, case.root / "extended-result.json", host=Host())
    assert result.admitted is True
    assert str(extended[1]) in json.loads(result.path.read_text())["inputs"]


@pytest.mark.parametrize(
    "fault",
    [
        "missing",
        "old-roots",
        "baseline",
        "scope",
        "owner",
        "row",
        "active",
        "initial",
        "conflict",
        "evidence",
        "unaccepted",
        "shrink",
        "symlink",
        "changed-authorization",
    ],
)
def test_root_extension_rejects_unapproved_or_reset_state(
    case: Case,
    extended: tuple[Path, Path],
    fault: str,
) -> None:
    registry = json.loads(case.registry.read_text())
    extension = json.loads(extended[1].read_text())
    if fault == "missing":
        registry.pop("rootExtension")
    elif fault == "old-roots":
        extension["oldRoots"] = []
    elif fault == "baseline":
        extension["previous"] = pin(case.request)
    elif fault == "scope":
        extension["authorization"]["scope"] = "unrelated"
    elif fault == "owner":
        registry["owners"][0]["process"]["birth"] = "changed"
    elif fault == "row":
        registry["outputs"][0]["reservedAdditionalBytes"] = 1
    elif fault == "active":
        registry["outputs"][-1]["state"] = "active"
    elif fault == "initial":
        registry["previous"] = None
    elif fault == "conflict":
        extension["addedRoots"] = [str(case.root)]
    elif fault == "unaccepted":
        accepted = write(case.root / "unaccepted.json", {"admitted": False})
        extension["acceptedPrevious"] = pin(accepted)
    elif fault == "shrink":
        registry["approvedOutputRoots"] = [str(case.root)]
    elif fault == "symlink":
        link = case.root / "linked-root"
        link.symlink_to(extension["addedRoots"][0], target_is_directory=True)
        extension["addedRoots"] = [str(link)]
        registry["approvedOutputRoots"][-1] = str(link)
    elif fault == "changed-authorization":
        (case.root / "authorization.json").write_text("changed", encoding="utf-8")
    else:
        extension["evidence"] = []
    write(extended[1], extension)
    if fault != "missing":
        registry["rootExtension"] = pin(extended[1])
    write(case.registry, registry)
    repin_registry(case)
    with pytest.raises(ReplayError):
        admission._preparation(json.loads(preparation_request(case).read_text()))
    with pytest.raises(ReplayError):
        admission.capture(case.request, case.root / "rejected.json", host=Host())


@pytest.mark.parametrize("available,accepted", [(505, True), (504, False)])
def test_explicit_allocation_capture_and_preparation(
    case: Case, available: int, accepted: bool
) -> None:
    observation = write(case.root / "disk-observation.json", {"synthetic": True})
    allocation = write(
        case.root / "allocation.json",
        {
            "version": 1,
            "kind": "native-disk-allocation",
            "movie": str(case.planned),
            "bootVolume": str(case.root),
            "allocations": [
                {
                    "role": "movie",
                    "directory": str(case.root),
                    "device": case.root.stat().st_dev,
                    "peakAdditionalBytes": 400,
                    "observedPeakBytes": 390,
                    "evidence": [pin(observation)],
                },
                {
                    "role": "local-metadata",
                    "directory": str(case.root),
                    "device": case.root.stat().st_dev,
                    "peakAdditionalBytes": 2,
                    "observedPeakBytes": 1,
                    "evidence": [pin(observation)],
                },
                {
                    "role": "fcp-cache",
                    "directory": str(case.root),
                    "device": case.root.stat().st_dev,
                    "peakAdditionalBytes": 3,
                    "observedPeakBytes": 2,
                    "evidence": [pin(observation)],
                },
            ],
        },
    )
    raw = json.loads(case.request.read_text())
    raw["requested"]["diskAllocation"] = pin(allocation)
    write(case.request, raw)
    refs = admission._preparation(json.loads(preparation_request(case).read_text()))
    assert allocation in refs
    host = Host()
    host.available = available
    result = admission.capture(case.request, case.root / "allocation-result.json", host=host)
    assert result.admitted is accepted
    evidence = json.loads(result.path.read_text())
    assert str(observation) in evidence["inputs"]


def repin_native(case: Case) -> None:
    raw = json.loads(case.request.read_text())
    raw["nativeObservation"] = pin(case.native)
    write(case.request, raw)
    os.utime(case.native, ns=(Host.now - 1_000_000_000, Host.now - 1_000_000_000))


def test_actual_snapshot_counts_retained_and_preserves_full_paths(case: Case) -> None:
    result = admission.capture(case.request, case.root / "snapshot.json", host=Host())
    assert result.admitted is True
    raw = json.loads(result.path.read_text())
    assert raw["stagingCount"] == 1
    assert raw["stagingBytes"] == 100
    assert raw["activeRenders"] == 0
    assert raw["capturedAtNs"] == 1_999_999_999_000_000_000
    assert raw["outputs"][0]["after"]["regular"] is True
    assert Path(raw["outputs"][0]["path"]).name == "retained.mov"
    assert raw["qualityApproved"] is False
    assert not case.planned.exists()
    with pytest.raises(FileExistsError):
        admission.capture(case.request, result.path, host=Host())


def test_compound_shell_exit_zero_does_not_make_existing_output_absent(case: Case) -> None:
    registry = json.loads(case.registry.read_text())
    registry["outputs"][1]["priorObservation"] = {
        "exitCode": 0,
        "claimedAbsent": True,
        "deletionPerformed": False,
    }
    write(case.registry, registry)
    repin_registry(case)
    case.planned.write_bytes(b"still present")
    result = admission.capture(case.request, case.root / "snapshot.json", host=Host())
    assert result.admitted is False
    assert "requested-output-already-present" in result.reasons
    assert "staging-capacity" in result.reasons
    assert json.loads(result.path.read_text())["stagingCount"] == 2
    assert case.planned.read_bytes() == b"still present"


def test_dangling_symlink_is_present_and_never_followed(case: Case) -> None:
    case.planned.symlink_to(case.root / "missing.mov")
    result = admission.capture(case.request, case.root / "snapshot.json", host=Host())
    assert result.admitted is False
    assert "output-not-regular" in result.reasons
    assert "requested-output-already-present" in result.reasons
    assert json.loads(result.path.read_text())["outputs"][1]["after"]["symlink"] is True


def test_directory_cannot_be_treated_as_movie(case: Case) -> None:
    case.planned.mkdir()
    result = admission.capture(case.request, case.root / "snapshot.json", host=Host())
    assert "output-not-regular" in result.reasons


def test_slow_import_requires_new_observation_not_timestamp_rewrite(case: Case) -> None:
    host = Host()
    host.now += 73_000_000_000
    before = file_digest(case.native)
    stale = admission.capture(case.request, case.root / "stale.json", host=host)
    assert stale.admitted is False
    assert "native-observation-stale-or-future" in stale.reasons
    fresh = write(case.root / "new-native-observation.json", synthetic_native_observation())
    os.utime(fresh, ns=(host.now, host.now))
    raw = json.loads(case.request.read_text())
    raw["nativeObservation"] = pin(fresh)
    new_request = write(case.root / "new-request.json", raw)
    result = admission.capture(new_request, case.root / "fresh.json", host=host)
    assert result.admitted is True
    assert file_digest(case.native) == before
    assert case.native.stat().st_mtime_ns == 1_999_999_999_000_000_000


@pytest.mark.parametrize(
    "change", ["duplicate-owner", "duplicate-output", "unknown-owner", "state"]
)
def test_registry_rejects_ambiguous_identity(case: Case, change: str) -> None:
    raw = json.loads(case.registry.read_text())
    if change == "duplicate-owner":
        raw["owners"].append(raw["owners"][0])
    elif change == "duplicate-output":
        raw["outputs"].append(raw["outputs"][0])
    elif change == "unknown-owner":
        raw["outputs"][0]["owner"] = "foreign"
    else:
        raw["outputs"][0]["state"] = "deleted-because-status-zero"
    write(case.registry, raw)
    repin_registry(case)
    with pytest.raises(ReplayError):
        admission.capture(case.request, case.root / "snapshot.json", host=Host())


@pytest.mark.parametrize("change", ["pid", "truncated", "missing", "busy", "future"])
def test_native_observation_is_fresh_complete_and_same_app(case: Case, change: str) -> None:
    raw = json.loads(case.native.read_text())
    payload = raw["data"]["structuredContent"]
    if change == "pid":
        payload["pid"] = 444
    elif change == "truncated":
        payload["truncated"] = True
    elif change == "missing":
        payload["elements"].pop()
    elif change == "busy":
        payload["elements"][-1]["value"] = "Working"
    write(case.native, raw)
    repin_native(case)
    if change == "future":
        os.utime(case.native, ns=(Host.now + 1, Host.now + 1))
    if change in ("busy", "future"):
        result = admission.capture(case.request, case.root / "snapshot.json", host=Host())
        assert result.admitted is False
    else:
        with pytest.raises(ReplayError):
            admission.capture(case.request, case.root / "snapshot.json", host=Host())


def test_registered_owner_pid_reuse_is_rejected(case: Case) -> None:
    host = Host()
    host.birth_value = "200:300"
    with pytest.raises(ReplayError, match="birth changed"):
        admission.capture(case.request, case.root / "snapshot.json", host=host)


def test_registry_mutation_during_measurement_is_rejected(case: Case) -> None:
    class Mutating(Host):
        def pressure(self) -> int:
            case.registry.write_text("changed", encoding="utf-8")
            return 1

    with pytest.raises(ReplayError, match="changed during"):
        admission.capture(case.request, case.root / "snapshot.json", host=Mutating())
    assert not (case.root / "snapshot.json").exists()


def test_growing_output_keeps_reserved_remaining_peak(case: Case) -> None:
    raw = json.loads(case.registry.read_text())
    raw["outputs"][0]["reservedAdditionalBytes"] = 450
    write(case.registry, raw)
    repin_registry(case)

    class Growing(Host):
        def pressure(self) -> int:
            case.retained.write_bytes(b"x" * 200)
            return 1

    result = admission.capture(case.request, case.root / "snapshot.json", host=Growing())
    assert result.admitted is False
    assert result.reasons == ("staging-capacity",)
    raw = json.loads(result.path.read_text())
    assert raw["stagingBytes"] == 200
    assert raw["reservedAdditionalBytes"] == 450
    assert raw["outputs"][0]["before"]["bytes"] == 100
    assert raw["outputs"][0]["after"]["bytes"] == 200


def test_active_uncertain_outputs_count_even_when_native_reports_idle(case: Case) -> None:
    raw = json.loads(case.registry.read_text())
    raw["outputs"][0]["state"] = "uncertain"
    write(case.registry, raw)
    repin_registry(case)
    result = admission.capture(case.request, case.root / "snapshot.json", host=Host())
    assert result.reasons == ("native-active-capacity",)
    assert json.loads(result.path.read_text())["observedActiveRenders"] == 1
    assert json.loads(result.path.read_text())["activeRenders"] is None


def test_unproven_disappearance_never_releases_slot(case: Case) -> None:
    case.retained.unlink()
    result = admission.capture(case.request, case.root / "snapshot.json", host=Host())
    assert "protected-output-missing-without-release" in result.reasons
    assert json.loads(result.path.read_text())["stagingCount"] == 1


@pytest.mark.parametrize("level,free", [(2, 100_000), (1, 450)])
def test_pressure_and_disk_measured_not_declared(case: Case, level: int, free: int) -> None:
    host = Host()
    host.level, host.available = level, free
    result = admission.capture(case.request, case.root / "snapshot.json", host=host)
    assert result.admitted is False


def test_replaced_inode_is_not_same_growing_output(case: Case) -> None:
    class Replacing(Host):
        def pressure(self) -> int:
            replacement = case.root / "replacement.mov"
            replacement.write_bytes(b"replacement")
            os.replace(replacement, case.retained)
            return 1

    result = admission.capture(case.request, case.root / "snapshot.json", host=Replacing())
    assert "output-identity-changed" in result.reasons


def test_duplicate_inode_paths_are_rejected(case: Case) -> None:
    os.link(case.retained, case.planned)
    result = admission.capture(case.request, case.root / "snapshot.json", host=Host())
    assert "duplicate-output-inode" in result.reasons


def test_removed_output_requires_retirement_and_consumer_proof(case: Case) -> None:
    previous = write(case.root / "previous.json", json.loads(case.registry.read_text()))
    raw = json.loads(case.registry.read_text())
    raw["generation"], raw["previous"] = "generation-b", pin(previous)
    raw["outputs"] = raw["outputs"][1:]
    write(case.registry, raw)
    repin_registry(case)
    with pytest.raises(ReplayError, match="exact release"):
        admission.capture(case.request, case.root / "snapshot.json", host=Host())


def test_retirement_producer_status_zero_cannot_release_existing_file(case: Case) -> None:
    output = admission.Output(case.retained, "native-owner", "retained", 0)
    with pytest.raises(ReplayError, match="not actual output absence"):
        admission._released(
            admission.Evidence(),
            {"deletionPerformed": False, "status": 0, "path": str(case.retained)},
            output,
            roots=(case.root,),
        )


def test_absence_alone_cannot_release_without_proof(case: Case) -> None:
    output = admission.Output(case.planned, "native-owner", "retained", 0)
    with pytest.raises(ReplayError, match="post-delete consumer evidence"):
        admission._released(
            admission.Evidence(), {"path": str(case.planned)}, output, roots=(case.root,)
        )


@pytest.mark.parametrize("reappears", [False, True])
def test_removed_output_delegates_retirement_and_postdelete_domain_verification(
    case: Case, monkeypatch: pytest.MonkeyPatch, reappears: bool
) -> None:
    previous = write(case.root / "previous.json", json.loads(case.registry.read_text()))
    registry = json.loads(case.registry.read_text())
    registry["previous"], registry["generation"] = pin(previous), "generation-b"
    registry["outputs"] = registry["outputs"][1:]
    write(case.registry, registry)
    repin_registry(case)
    case.retained.unlink()
    proof = write(case.root / "frame-proof.json", {"synthetic": True})
    survivor = case.root / "published.mov"
    survivor.write_bytes(b"synthetic surviving bytes")
    published = write(
        case.root / "published.json", {"frameProof": pin(proof), "movie": pin(survivor)}
    )
    retirement = write(case.root / "retirement.json", {"deletionPerformed": False})
    config = json.loads(case.request.read_text())
    config["releases"] = [
        {
            "path": str(case.retained),
            "assemblyPart": 2,
            "publication": pin(published),
            "retirement": pin(retirement),
            "terminal": pin(proof),
            "postDeleteStatus": pin(proof),
            "retirementAudit": pin(proof),
            "history": [pin(proof)],
            "consumer": pin(proof),
            "publishedAssembly": pin(proof),
        }
    ]
    write(case.request, config)
    calls: list[str] = []

    def retired(
        inputs: followup.Inputs, raw: dict[str, object], prep: dict[str, object]
    ) -> followup.Ref:
        assert prep["assemblyPart"] == 2
        calls.append("retirement")
        return inputs.add(raw["retirement"])

    def consumer(
        inputs: followup.Inputs,
        value: object,
        *,
        config: dict[str, object],
        retirement: followup.Ref,
    ) -> None:
        assert "publishedAssembly" in config
        assert retirement.path.name == "retirement.json"
        inputs.add(value)
        calls.append("post-delete-consumer")

    def published_verify(
        path: Path, proof_path: Path, *, part: int, retirement: Path | None = None
    ) -> tuple[Path, Path]:
        assert part == 2
        assert retirement is not None
        calls.append("published-media")
        return path, proof_path

    monkeypatch.setattr(followup, "_retirement", retired)
    monkeypatch.setattr(followup, "_consumer", consumer)
    monkeypatch.setattr(admission, "verify_published", published_verify)

    class Measured(Host):
        def observe(self, path: Path) -> admission.FileState:
            if reappears and path == case.retained:
                return admission.FileState(True, True, False, 1, 2, 100)
            return super().observe(path)

    result = admission.capture(case.request, case.root / "snapshot.json", host=Measured())
    assert calls == ["retirement", "post-delete-consumer", "published-media"]
    if reappears:
        assert result.reasons == ("released-output-reappeared",)
    else:
        assert result.admitted is True
        assert json.loads(result.path.read_text())["stagingCount"] == 0


@pytest.mark.parametrize("change", ["same-generation", "reset-state", "initial-release"])
def test_registry_transition_cannot_hide_protected_history(case: Case, change: str) -> None:
    previous = write(case.root / "previous.json", json.loads(case.registry.read_text()))
    raw = json.loads(case.registry.read_text())
    raw["previous"] = pin(previous)
    if change == "reset-state":
        raw["generation"] = "new-generation"
        raw["outputs"][0]["state"] = "planned"
    elif change == "initial-release":
        raw["previous"] = None
        request = json.loads(case.request.read_text())
        request["releases"] = [{"path": str(case.retained)}]
        write(case.request, request)
    write(case.registry, raw)
    repin_registry(case)
    with pytest.raises(ReplayError):
        admission.capture(case.request, case.root / "snapshot.json", host=Host())


@pytest.mark.parametrize("change", ["incomplete", "duplicate-root", "unbounded-owners"])
def test_explicit_registry_boundaries(case: Case, change: str) -> None:
    raw = json.loads(case.registry.read_text())
    if change == "incomplete":
        raw["complete"] = False
    elif change == "duplicate-root":
        raw["approvedOutputRoots"] *= 2
    else:
        raw["owners"] = []
    write(case.registry, raw)
    repin_registry(case)
    with pytest.raises(ReplayError):
        admission.capture(case.request, case.root / "snapshot.json", host=Host())


@pytest.mark.parametrize("change", ["owner", "pressure", "reserved", "schema"])
def test_request_cannot_weaken_admission_contract(case: Case, change: str) -> None:
    raw = json.loads(case.request.read_text())
    if change == "owner":
        raw["nativeOwner"] = "another-owner"
    elif change == "pressure":
        raw["policy"]["maximumPressure"] = 3
    elif change == "reserved":
        raw["requested"]["path"] = str(case.retained)
    else:
        raw["unknown"] = True
    write(case.request, raw)
    with pytest.raises(ReplayError):
        admission.capture(case.request, case.root / "snapshot.json", host=Host())


def test_duplicate_json_and_invalid_pin_are_not_silent(case: Case) -> None:
    with pytest.raises(ReplayError, match="Duplicate"):
        admission._unique([("version", 1), ("version", 2)])
    with pytest.raises(ReplayError, match="immutable"):
        admission.Evidence().add({"path": str(case.registry), "sha256": "not-a-hash"})


def test_path_parent_escape_and_outside_snapshot_are_rejected(case: Case) -> None:
    with pytest.raises(ReplayError, match="canonical approved parent"):
        admission._output_path(str(case.root / ".."), (case.root,))
    with pytest.raises(ReplayError, match="repository work"):
        admission._publish(Path("relative.json"), {})


def test_measurement_duration_and_process_identity_are_rechecked(case: Case) -> None:
    class Slow(Host):
        def pressure(self) -> int:
            self.now += 31_000_000_000
            return 1

    result = admission.capture(case.request, case.root / "snapshot.json", host=Slow())
    assert "measurement-exceeds-freshness" in result.reasons


def test_process_restart_during_measurement_is_rejected(case: Case) -> None:
    class Restart(Host):
        def pressure(self) -> int:
            self.birth_value = "300:400"
            return 1

    with pytest.raises(ReplayError, match="changed during measurement"):
        admission.capture(case.request, case.root / "snapshot.json", host=Restart())


def test_invalid_native_pressure_value_is_not_admitted(case: Case) -> None:
    host = Host()
    host.level = 0
    with pytest.raises(ReplayError, match="pressure value"):
        admission.capture(case.request, case.root / "snapshot.json", host=host)


@pytest.mark.parametrize("value", [None, True, -1, 2**64])
def test_external_integer_validation(value: object) -> None:
    with pytest.raises(ReplayError):
        admission._integer(value)


@pytest.mark.parametrize("value", [None, "", "a\x00b", "x" * 513])
def test_external_text_validation(value: object) -> None:
    with pytest.raises(ReplayError):
        admission._text(value)


def test_non_macos_pressure_fails_explicitly(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("replay.fcp_native_admission.sys.platform", "linux")
    with pytest.raises(ReplayError, match="macOS"):
        admission.SystemHost().pressure()


@pytest.mark.parametrize("status", [0, -1])
def test_sysctl_pressure_boundary_is_typed(monkeypatch: pytest.MonkeyPatch, status: int) -> None:
    class Measure:
        argtypes: object = None
        restype: object = None

        def __call__(
            self,
            name: bytes,
            value: ctypes.c_void_p,
            size: ctypes.c_void_p,
            new: object,
            length: int,
        ) -> int:
            assert name == b"kern.memorystatus_vm_pressure_level"
            assert length == 0
            assert new is None
            ctypes.memmove(value, ctypes.byref(ctypes.c_uint32(2)), 4)
            return status

    class Library:
        sysctlbyname = Measure()

    monkeypatch.setattr("replay.fcp_native_admission.sys.platform", "darwin")
    monkeypatch.setattr(
        "replay.fcp_native_admission.ctypes.CDLL", lambda *args, **kwargs: Library()
    )
    if status == -1:
        with pytest.raises(ReplayError, match="unavailable"):
            admission.SystemHost().pressure()
    else:
        assert admission.SystemHost().pressure() == 2


def test_cli_denial_returns_nonzero_with_retained_snapshot(
    case: Case, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    result = admission.Capture(case.root / "snapshot.json", False, ("staging-capacity",))
    monkeypatch.setattr(admission, "capture", lambda *args, **kwargs: result)
    monkeypatch.setattr("sys.argv", ["admission", str(case.request), str(result.path)])
    with pytest.raises(SystemExit) as error:
        admission.main()
    assert error.value.code == 1
    assert json.loads(capsys.readouterr().out)["admitted"] is False


def test_file_state_change_distinguishes_growth_from_identity() -> None:
    before = admission.FileState(True, True, False, 1, 2, 100, 10)
    assert admission._changed(before, replace(before, size=200)) is False
    assert admission._changed(before, replace(before, size=20)) is True
    assert admission._changed(before, replace(before, inode=3)) is True


@pytest.fixture
def alias_case(case: Case, monkeypatch: pytest.MonkeyPatch) -> Case:
    previous = write(case.root / "alias-previous.json", json.loads(case.registry.read_text()))
    keeper = case.root / "alias-keeper.mov"
    os.link(case.retained, keeper)
    case.retained.unlink()
    seal = write(case.root / "alias-seal.json", {"immutableSyntheticSeal": True})
    original_ref = hardlinks.reference(pin(previous))
    verified = hardlinks.VerifiedRetirement(case.retained, keeper, "native-owner", original_ref, ())

    def verify(path: Path, *, alias: Path, owner: str) -> hardlinks.VerifiedRetirement:
        if path != seal or alias != case.retained or owner != "native-owner":
            raise ReplayError("Wrong alias verification request")
        if alias.exists() or alias.is_symlink():
            raise ReplayError("Retired alias reappeared")
        return verified

    monkeypatch.setattr(hardlinks, "verify", verify)
    registry = json.loads(case.registry.read_text())
    registry["generation"] = "alias-retired-generation"
    registry["previous"] = pin(previous)
    registry["outputs"][0]["state"] = "retired-hardlink"
    registry["outputs"][0]["verifiedRetirement"] = pin(seal)
    registry["outputs"].append(
        {
            "path": str(keeper),
            "owner": "native-owner",
            "state": "retained",
            "reservedAdditionalBytes": 64,
        }
    )
    write(case.registry, registry)
    raw = json.loads(case.request.read_text())
    raw["registry"] = pin(case.registry)
    raw["policy"]["maximumStagingCount"] = 4
    write(case.request, raw)
    return case


def test_alias_retirement_keeps_absent_row_and_counts_keeper_reserve(alias_case: Case) -> None:
    result = admission.capture(
        alias_case.request, alias_case.root / "alias-snapshot.json", host=Host()
    )
    assert result.admitted is True
    snapshot = json.loads(result.path.read_text())
    assert snapshot["outputs"][0]["state"] == "retired-hardlink"
    assert snapshot["outputs"][0]["owner"] == "native-owner"
    assert snapshot["outputs"][0]["after"]["exists"] is False
    assert snapshot["outputs"][0]["reservedAdditionalBytes"] == 0
    assert snapshot["reservedAdditionalBytes"] == 64


@pytest.mark.parametrize("change", ["first", "owner", "reserved", "proof", "keeper"])
def test_alias_retirement_rejects_forged_transition(alias_case: Case, change: str) -> None:
    registry = json.loads(alias_case.registry.read_text())
    if change == "first":
        registry["previous"] = None
    elif change == "owner":
        registry["outputs"][0]["owner"] = "unknown"
    elif change == "reserved":
        registry["outputs"][0]["reservedAdditionalBytes"] = 1
    elif change == "proof":
        del registry["outputs"][0]["verifiedRetirement"]
    else:
        registry["outputs"] = registry["outputs"][:2]
    write(alias_case.registry, registry)
    request = json.loads(alias_case.request.read_text())
    request["registry"] = pin(alias_case.registry)
    write(alias_case.request, request)
    with pytest.raises(ReplayError):
        admission.capture(alias_case.request, alias_case.root / "denied.json", host=Host())


def test_alias_retired_row_cannot_be_removed_or_reset(alias_case: Case) -> None:
    previous = write(
        alias_case.root / "retired-previous.json", json.loads(alias_case.registry.read_text())
    )
    registry = json.loads(alias_case.registry.read_text())
    registry["generation"] = "after-retired"
    registry["previous"] = pin(previous)
    registry["outputs"] = registry["outputs"][1:]
    write(alias_case.registry, registry)
    request = json.loads(alias_case.request.read_text())
    request["registry"] = pin(alias_case.registry)
    write(alias_case.request, request)
    with pytest.raises(ReplayError, match="cannot be removed or reset"):
        admission.capture(alias_case.request, alias_case.root / "removed.json", host=Host())


def test_alias_reappearance_is_not_missing_output_exception(alias_case: Case) -> None:
    alias_case.retained.write_bytes(b"new unexpected alias")
    with pytest.raises(ReplayError, match="reappeared"):
        admission.capture(alias_case.request, alias_case.root / "reappeared.json", host=Host())


def test_alias_retirement_does_not_relax_capacity(alias_case: Case) -> None:
    host = Host()
    host.available = 500
    result = admission.capture(
        alias_case.request, alias_case.root / "capacity-denied.json", host=host
    )
    assert result.admitted is False
    assert "disk-capacity" in result.reasons


def test_completed_owner_preserves_endpoints_and_releases_future_buffer(
    case: Case, monkeypatch: pytest.MonkeyPatch
) -> None:
    proof_path = write(case.root / "completed.json", {"fixture": True})
    ref = hardlinks.reference(pin(proof_path))
    identity = admission.Identity(987654, "old:birth")
    owner = admission.Owner("ended", identity, "reader", ref)
    output = admission.Output(case.retained, "ended", "retained", 0)
    registry = admission.Registry("new", (owner,), (output,), (case.root,), {})
    proof = owners.CompletedOwner(
        "ended",
        {"pid": 987654, "birth": "old:birth"},
        ref,
        {str(case.retained): {"exists": True}},
        (),
    )
    monkeypatch.setattr(owners, "verify", lambda *args, **kwargs: proof)
    admission._completed_owners(admission.Evidence(), registry)
    with pytest.raises(ReplayError, match="no future allocation"):
        admission._completed_owners(
            admission.Evidence(),
            admission.Registry(
                "bad",
                (owner,),
                (admission.Output(case.retained, "ended", "retained", 1),),
                (case.root,),
                {},
            ),
        )
    with pytest.raises(ReplayError, match="preserve its original"):
        admission._completed_transition(None, registry)


@pytest.mark.parametrize("change", ["none", "fork", "row"])
def test_completed_owner_successor_requires_exact_predecessor_and_unchanged_rows(
    case: Case, monkeypatch: pytest.MonkeyPatch, change: str
) -> None:
    before_path = write(case.root / "before-owner.json", {"fixture": "before"})
    after_path = write(case.root / "after-owner.json", {"fixture": "after"})
    before_ref = hardlinks.reference(pin(before_path))
    after_ref = hardlinks.reference(pin(after_path))
    identity = admission.Identity(987654, "old:birth")
    previous_owner = admission.Owner("ended", identity, "reader", before_ref)
    current_owner = admission.Owner("ended", identity, "reader", after_ref)
    before_row = admission.Output(case.retained, "ended", "retained", 0)
    after_row = admission.Output(case.retained, "ended", "retained", 1 if change == "row" else 0)
    previous = admission.Registry("previous", (previous_owner,), (before_row,), (case.root,), {})
    current = admission.Registry("current", (current_owner,), (after_row,), (case.root,), {})
    proof = owners.CompletedOwner(
        "ended",
        {"pid": 987654, "birth": "old:birth"},
        before_ref,
        {str(case.retained): {"exists": True}},
        (),
        after_ref if change == "fork" else before_ref,
    )
    monkeypatch.setattr(owners, "verify", lambda *args, **kwargs: proof)
    if change == "none":
        admission._completed_transition(previous, current)
    else:
        with pytest.raises(ReplayError, match="exact predecessor and rows"):
            admission._completed_transition(previous, current)
