import ctypes
import json
import os
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import pytest

from replay import fcp_native_queue as queue
from replay.core import ReplayError
from replay.stages import file_digest


@pytest.fixture
def preparation(case: queue.Queue, tmp_path: Path) -> Path:
    export = json.loads((tmp_path / "export.json").read_text())
    manifest = write(
        tmp_path / "actions.json",
        {
            "parts": [
                {
                    "part": 21,
                    "xml": pin(tmp_path / "source.fcpxml"),
                    "export": export,
                    "duration": "101/25s",
                    "globalStart": "0s",
                }
            ]
        },
    )
    return write(
        tmp_path / "prepare.json",
        {
            "version": 1,
            "kind": "native-queue-preparation-request",
            "nativeActions": pin(manifest),
            "assemblyParts": [21],
            "libraryBasename": "Synthetic library.fcpbundle",
            "event": "Synthetic event",
            "bundleID": "com.apple.FinalCutApp",
            "backgroundTitle": "Background Tasks",
        },
    )


def test_prepare_candidate_has_no_grant_and_reuses_partial(
    preparation: Path, tmp_path: Path
) -> None:
    result = queue.prepare_data(preparation, tmp_path / "prepared")
    before = result.read_bytes()
    result.unlink()
    resumed = queue.prepare_data(preparation, tmp_path / "prepared")
    assert resumed.read_bytes() == before
    assert len(queue.validate_candidate(resumed)) == 1
    assert json.loads(resumed.read_text())["canDispatch"] is False
    with pytest.raises(ReplayError, match="Unsupported native queue"):
        queue.load(resumed)


@pytest.mark.parametrize("background", ["Background Tasks", "バックグラウンドタスク"])
def test_prepare_other_namespace_and_clock(
    preparation: Path, tmp_path: Path, background: str
) -> None:
    raw = json.loads(preparation.read_text())
    raw["backgroundTitle"] = background
    write(preparation, raw)
    manifest = tmp_path / "actions.json"
    actions = json.loads(manifest.read_text())
    actions["parts"][0]["export"]["project"] = "Wide landscape alternate"
    actions["parts"][0]["export"]["fileName"] = "alternate.mov"
    actions["parts"][0]["duration"] = "1001/240s"
    write(manifest, actions)
    raw["nativeActions"] = pin(manifest)
    write(preparation, raw)
    result = queue.prepare_data(preparation, tmp_path / "prepared")
    assert queue.validate_candidate(result)[0].project == "Wide landscape alternate"


@pytest.mark.parametrize("background_title", ["Background Tasks", "バックグラウンドタスク"])
@pytest.mark.parametrize(
    "phase,required_app,expected_fields",
    [
        ("import", "finalCutPro", {"app", "bundleID", "path"}),
        ("windows", "finalCutPro", {"app", "bundleID"}),
        ("project", None, {"bundleID", "project", "library", "event"}),
        (
            "export",
            None,
            {"bundleID", "project", "directory", "fileName", "codec", "format", "allowForeground"},
        ),
        ("background-open", "finalCutPro", {"app", "bundleID", "menuPath"}),
        (
            "background-wait",
            "finalCutPro",
            {"app", "bundleID", "window", "condition", "element", "timeoutSeconds"},
        ),
        ("background", "finalCutPro", {"app", "bundleID", "window", "maxDepth"}),
        ("background-close", "finalCutPro", {"app", "bundleID", "window", "element", "action"}),
    ],
)
def test_prepared_initial_deployed_request_contract(
    preparation: Path,
    tmp_path: Path,
    background_title: str,
    phase: str,
    required_app: str | None,
    expected_fields: set[str],
) -> None:
    raw = json.loads(preparation.read_text())
    raw["backgroundTitle"] = background_title
    write(preparation, raw)
    candidate = queue.prepare_data(preparation, tmp_path / "prepared")
    part = queue.validate_candidate(candidate)[0]
    operation = next(item for item in part.operations if item.phase == phase)
    arguments = json.loads(operation.request.read_text())
    assert set(arguments) == expected_fields
    assert arguments.get("app") == required_app
    assert arguments["bundleID"] == "com.apple.FinalCutApp"
    assert "maxElements" not in arguments


@pytest.mark.parametrize("change", ["request", "operation", "manifest", "candidate"])
def test_prepared_candidate_refuses_changed_bytes(
    preparation: Path, tmp_path: Path, change: str
) -> None:
    result = queue.prepare_data(preparation, tmp_path / "prepared")
    target = {
        "request": preparation,
        "operation": result.parent / "part-0021-export.json",
        "manifest": tmp_path / "actions.json",
        "candidate": result,
    }[change]
    raw = json.loads(target.read_text())
    raw["canDispatch"] = True
    write(target, raw)
    with pytest.raises(ReplayError):
        queue.validate_candidate(result)
    with pytest.raises(ReplayError):
        queue.prepare_data(preparation, tmp_path / "prepared")


@pytest.mark.parametrize(
    "field,value",
    [
        ("assemblyParts", [21, 21]),
        ("assemblyParts", [22]),
        ("libraryBasename", "wrong"),
        ("backgroundTitle", "unknown"),
        ("kind", "other"),
    ],
)
def test_prepare_invalid_input(
    preparation: Path, tmp_path: Path, field: str, value: object
) -> None:
    raw = json.loads(preparation.read_text())
    raw[field] = value
    write(preparation, raw)
    with pytest.raises(ReplayError):
        queue.prepare_data(preparation, tmp_path / "prepared")


def test_prepare_cli(preparation: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        sys, "argv", ["queue", "prepare-data", str(preparation), str(tmp_path / "prepared")]
    )
    queue.main()
    monkeypatch.setattr(
        sys, "argv", ["queue", "validate-candidate", str(tmp_path / "prepared" / "candidate.json")]
    )
    queue.main()


def write(path: Path, data: dict[str, object]) -> Path:
    path.write_text(json.dumps(data), encoding="utf-8")
    return path


def pin(path: Path) -> dict[str, str]:
    return {"path": str(path), "sha256": file_digest(path)}


class FakeHost:
    birth_value = "100:200"
    available = 10**12
    timestamp = 2_000_000_000_000_000_000

    def birth(self, pid: int) -> str:
        return self.birth_value

    def now_ns(self) -> int:
        return self.timestamp

    def free(self, path: Path) -> int:
        return self.available

    def floor(self, path: Path) -> int:
        return 30 * 1024**3


@pytest.fixture
def allocation(case: queue.Queue, tmp_path: Path) -> Path:
    evidence = write(tmp_path / "disk-observation.json", {"synthetic": True})
    return write(
        tmp_path / "disk-allocation.json",
        {
            "version": 1,
            "kind": "native-disk-allocation",
            "movie": str(case.parts[0].movie),
            "bootVolume": str(case.boot),
            "allocations": [
                {
                    "role": "movie",
                    "directory": str(tmp_path),
                    "device": tmp_path.stat().st_dev,
                    "peakAdditionalBytes": 10,
                    "observedPeakBytes": 9,
                    "evidence": [pin(evidence)],
                },
                {
                    "role": "local-metadata",
                    "directory": str(tmp_path),
                    "device": tmp_path.stat().st_dev,
                    "peakAdditionalBytes": 2,
                    "observedPeakBytes": 1,
                    "evidence": [pin(evidence)],
                },
                {
                    "role": "fcp-cache",
                    "directory": str(tmp_path),
                    "device": tmp_path.stat().st_dev,
                    "locationKnown": False,
                    "peakAdditionalBytes": 3,
                    "observedPeakBytes": 2,
                    "evidence": [pin(evidence)],
                },
            ],
        },
    )


@pytest.mark.parametrize("extra,accepted", [(-1, False), (0, True), (1, True)])
def test_allocation_same_device_floor(
    case: queue.Queue,
    allocation: Path,
    extra: int,
    accepted: bool,
) -> None:
    host = FakeHost()
    host.available = 30 * 1024**3 + 15 + extra
    assert queue.disk_capacity(host, case.boot, case.parts[0].movie, 10, 1, allocation) is accepted


@pytest.mark.parametrize(
    "field,value",
    [
        ("device", -1),
        ("device", True),
        ("role", "unknown"),
        ("evidence", []),
        ("peakAdditionalBytes", 8),
    ],
)
def test_disk_allocation_rejects_unbound_evidence(
    case: queue.Queue,
    allocation: Path,
    field: str,
    value: object,
) -> None:
    raw = json.loads(allocation.read_text())
    raw["allocations"][0][field] = value
    write(allocation, raw)
    with pytest.raises(ReplayError):
        queue.disk_allocation(pin(allocation), case.boot, case.parts[0].movie, 10)


def test_allocation_pinned_by_queue_and_no_app_check(
    case: queue.Queue,
    allocation: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    raw = json.loads(case.config.read_text())
    raw["limits"]["diskAllocation"] = pin(allocation)
    write(case.config, raw)
    loaded = queue.load(case.config)
    assert loaded.disk_allocation == allocation
    assert allocation in dict(loaded.pins)
    monkeypatch.setattr(sys, "argv", ["queue", "check", str(case.config)])
    queue.main()
    (allocation.parent / "disk-observation.json").write_text("changed", encoding="utf-8")
    with pytest.raises(ReplayError):
        queue.load(case.config)


def test_allocation_binds_movie_and_required_roles(case: queue.Queue, allocation: Path) -> None:
    with pytest.raises(ReplayError, match="exact movie"):
        queue.disk_allocation(pin(allocation), case.boot, case.boot / "other.mov", 10)
    raw = json.loads(allocation.read_text())
    raw["allocations"].pop()
    write(allocation, raw)
    with pytest.raises(ReplayError, match="All native"):
        queue.disk_allocation(pin(allocation), case.boot, case.parts[0].movie, 10)


def test_allocation_cross_device_movie_not_charged_to_boot(
    case: queue.Queue,
    allocation: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ssd = case.boot / "ssd"
    ssd.mkdir()
    original_stat = Path.stat

    def device_stat(path: Path, *, follow_symlinks: bool = True) -> os.stat_result:
        actual = original_stat(path, follow_symlinks=follow_symlinks)
        if path == ssd:
            return os.stat_result(
                (
                    actual.st_mode,
                    actual.st_ino,
                    12345,
                    actual.st_nlink,
                    actual.st_uid,
                    actual.st_gid,
                    actual.st_size,
                    actual.st_atime,
                    actual.st_mtime,
                    actual.st_ctime,
                )
            )
        return actual

    raw = json.loads(allocation.read_text())
    raw["movie"] = str(ssd / "output.mov")
    raw["allocations"][0].update(directory=str(ssd), device=12345)
    write(allocation, raw)
    monkeypatch.setattr(Path, "stat", device_stat)
    host = FakeHost()
    monkeypatch.setattr(host, "free", lambda path: 30 * 1024**3 + (10 if path == ssd else 5))
    assert queue.disk_capacity(host, case.boot, ssd / "output.mov", 10, 1, allocation)
    assert not queue.disk_capacity(host, case.boot, ssd / "output.mov", 10, 1, None)
    assert not queue.disk_capacity(
        host, case.boot, ssd / "output.mov", 10, 1, allocation, ((ssd, 1),)
    )


class FakeTransport:
    def __init__(self, part: queue.Part) -> None:
        self.part = part
        self.calls: list[str] = []
        self.failure: Exception | None = None
        self.override: dict[str, object] | None = None

    def call(self, tool: str, arguments: dict[str, object], timeout: int) -> dict[str, object]:
        self.calls.append(tool)
        if self.failure is not None:
            raise self.failure
        if self.override is not None:
            return self.override
        payload: dict[str, object] = {"pid": 123, "dispatched": True, "effectVerified": False}
        if tool.endswith("app_open_document"):
            payload = {"completionVerified": False, "delivery": "Synthetic open-document accepted"}
        elif tool.endswith("ui_wait"):
            payload = {
                "conditionMet": True,
                "elapsedSeconds": 21.5,
                "focus": {"frontmostChanged": False},
            }
        elif tool.endswith("ui_windows"):
            payload["windows"] = [{"title": "Editor", "sheets": []}]
        elif tool.endswith("fcp_project_open"):
            payload["finalCut"] = {"project": {"name": self.part.project}}
        elif tool.endswith("fcp_export"):
            self.part.movie.write_bytes(b"synthetic candidate, not decoded media")
            payload = {
                "pid": 123,
                "retrySafe": False,
                "finalCut": {
                    "project": {"name": self.part.project},
                    "vocabularyVerified": True,
                    "language": "en",
                    "export": {
                        "codec": "Apple ProRes 422 LT",
                        "format": "Video and Audio",
                        "action": "Save only",
                        "outputPath": str(self.part.movie),
                        "foregroundRestored": True,
                        "completionVerified": False,
                    },
                },
            }
        elif tool.endswith("ui_inspect"):
            payload.update(background())
        return envelope(payload)


def envelope(payload: dict[str, object]) -> dict[str, object]:
    return {
        "exitCode": 0,
        "stdout": json.dumps(
            {
                "ok": True,
                "data": {
                    "isError": False,
                    "structuredContent": payload,
                },
            }
        ),
        "stderr": "",
    }


def background() -> dict[str, object]:
    return {
        "pid": 123,
        "truncated": False,
        "elements": [
            {"role": "AXStaticText", "value": "Sharing"},
            *[{"identifier": "_NS:166", "value": "Idle", "path": [index]} for index in range(9)],
        ],
    }


class ChooserTransport(FakeTransport):
    selected = False
    confirmed = False
    disabled = False
    stale_row = False
    wait_met = True
    stays_open = False
    duplicate_name = False

    def call(self, tool: str, arguments: dict[str, object], timeout: int) -> dict[str, object]:
        if tool.endswith("ui_windows"):
            self.calls.append(tool)
            title = "Editor" if self.confirmed and not self.stays_open else "Open Library"
            return envelope({"pid": 123, "windows": [{"title": title, "sheets": []}]})
        window = arguments.get("window")
        if tool.endswith("ui_inspect") and window == {"title": "Open Library"}:
            self.calls.append(tool)
            row = [2, 1] if self.selected and self.stale_row else [2, 3]
            names = [
                {
                    "role": "AXTextField",
                    "value": self.part.library_basename,
                    "path": [*row, 0],
                    "unreadable": [],
                }
            ]
            return envelope(
                {
                    "pid": 123,
                    "truncated": False,
                    "elements": [
                        {
                            "role": "AXRow",
                            "path": row,
                            "selected": self.selected,
                            "unreadable": [],
                        },
                        *names,
                        *(names if self.duplicate_name else []),
                        {
                            "role": "AXButton",
                            "title": "Choose",
                            "enabled": self.selected and not self.disabled,
                            "path": [8],
                            "actions": ["AXPress"],
                            "unreadable": [],
                        },
                    ],
                }
            )
        if tool.endswith("ui_set_value"):
            self.calls.append(tool)
            self.selected = True
            return envelope(
                {
                    "pid": 123,
                    "dispatched": True,
                    "effectVerified": True,
                    "element": {"path": [2, 3], "selected": True},
                }
            )
        if tool.endswith("ui_perform") and window == {"title": "Open Library"}:
            self.calls.append(tool)
            self.confirmed = True
            return envelope({"pid": 123, "dispatched": True, "effectVerified": False})
        if tool.endswith("ui_wait") and arguments.get("element") == {"role": "AXSheet"}:
            self.calls.append(tool)
            return envelope({"conditionMet": self.wait_met, "elapsedSeconds": 7.8})
        return super().call(tool, arguments, timeout)


def busy_envelope(detail: object = None) -> dict[str, object]:
    failure = (
        {
            "retrySafe": False,
            "error": (
                "Unavailable: The app is busy or still launching (AXError -25204); "
                "wait or observe before retrying"
            ),
        }
        if detail is None
        else detail
    )
    return {
        "exitCode": 0,
        "stdout": json.dumps(
            {
                "ok": False,
                "error": {"code": "mcp_tool_error", "message": json.dumps(failure)},
            }
        ),
        "stderr": "",
    }


class BusyObservationTransport(ChooserTransport):
    def __init__(self, part: queue.Part, phase: str = "chooser-absent") -> None:
        super().__init__(part)
        self.phase = phase
        self.busy_remaining = 2
        self.readiness_requests: list[dict[str, object]] = []

    def call(self, tool: str, arguments: dict[str, object], timeout: int) -> dict[str, object]:
        phase = ""
        if tool.endswith("ui_windows"):
            phase = "chooser-absent" if self.confirmed else "windows"
        elif tool.endswith("ui_inspect"):
            if arguments.get("window") == {"title": "Background Tasks"}:
                phase = "background"
            else:
                phase = "chooser-readback" if self.selected else "chooser-inspect"
        if phase == self.phase and self.busy_remaining:
            self.calls.append(tool)
            self.busy_remaining -= 1
            return busy_envelope()
        if tool.endswith("ui_wait") and arguments.get("element") == {"role": "AXWindow"}:
            self.readiness_requests.append(arguments)
        return super().call(tool, arguments, timeout)


@pytest.mark.parametrize(
    "phase", ["windows", "chooser-inspect", "chooser-readback", "chooser-absent", "background"]
)
def test_busy_readonly_observation_waits_without_repeating_mutations(
    case: queue.Queue, phase: str
) -> None:
    transport = BusyObservationTransport(case.parts[0], phase)
    result = queue.run(replace(case, max_calls=32), transport, FakeHost())
    assert result.state == "followup-ready"
    assert transport.calls.count("native.org.default.app_open_document") == 1
    assert transport.calls.count("native.org.default.ui_set_value") == 1
    assert transport.calls.count("native.org.default.ui_perform") == 2
    assert transport.calls.count("native.org.default.fcp_project_open") == 1
    assert transport.calls.count("native.org.default.fcp_export") == 1
    assert transport.readiness_requests == [
        {
            "app": "finalCutPro",
            "bundleID": "com.apple.FinalCutApp",
            "condition": "exists",
            "element": {"role": "AXWindow"},
            "timeoutSeconds": 30,
        },
        {
            "app": "finalCutPro",
            "bundleID": "com.apple.FinalCutApp",
            "condition": "exists",
            "element": {"role": "AXWindow"},
            "timeoutSeconds": 30,
        },
    ]
    retained = sorted(case.root.glob("*.readonly-retry.json"))
    assert len(retained) == 2
    assert json.loads(retained[0].read_text())["retrySafe"] is False
    native = retained[0].with_name(
        retained[0].name.removesuffix(".readonly-retry.json") + ".native.json"
    )
    assert json.loads(native.read_text())["ok"] is False
    assert (case.root / "followup-ready.json").is_file()


def test_sustained_readonly_busy_is_bounded_and_keeps_original_effects(case: queue.Queue) -> None:
    transport = BusyObservationTransport(case.parts[0])
    transport.busy_remaining = 100
    result = queue.run(replace(case, max_calls=12), transport, FakeHost())
    assert result.state == "bounded-stop"
    assert result.calls == 12
    assert transport.calls.count("native.org.default.app_open_document") == 1
    assert transport.calls.count("native.org.default.ui_set_value") == 1
    assert transport.calls.count("native.org.default.ui_perform") == 1
    assert transport.calls.count("native.org.default.fcp_export") == 0
    assert not (case.root / "followup-ready.json").exists()
    assert len(list(case.root.glob("*.readonly-retry.json"))) == 3
    assert (case.root / "part-0001-chooser-absent.native.json").is_file()
    assert (case.root / "part-0001-chooser-absent-0003.native.json").is_file()


@pytest.mark.parametrize("change", ["pid", "input", "lease", "late"])
def test_readonly_retry_rechecks_identity_inputs_lease_and_deadline(
    case: queue.Queue, change: str
) -> None:
    host = FakeHost()

    class ChangedObservation(BusyObservationTransport):
        def call(self, tool: str, arguments: dict[str, object], timeout: int) -> dict[str, object]:
            captured = super().call(tool, arguments, timeout)
            if tool.endswith("ui_wait") and arguments.get("element") == {"role": "AXWindow"}:
                if change == "pid":
                    host.birth_value = "changed:birth"
                elif change == "input":
                    case.parts[0].operations[2].request.write_text("changed", encoding="utf-8")
                elif change == "lease":
                    (queue.GUI_LEASE_ROOT / ".fcp-native-queue-owner.json").write_text(
                        "changed", encoding="utf-8"
                    )
                else:
                    host.timestamp += 10_000_000_000_000
            return captured

    transport = ChangedObservation(case.parts[0])
    if change == "input":
        with pytest.raises(ReplayError, match="core pins changed"):
            queue.run(replace(case, max_calls=32), transport, host)
    else:
        result = queue.run(replace(case, max_calls=32), transport, host)
        assert result.state == "uncertain"
    assert transport.calls.count("native.org.default.app_open_document") == 1
    assert transport.calls.count("native.org.default.ui_perform") == 1
    assert transport.calls.count("native.org.default.fcp_export") == 0
    assert not (case.root / "followup-ready.json").exists()


def test_busy_then_reappeared_chooser_refuses_export(case: queue.Queue) -> None:
    transport = BusyObservationTransport(case.parts[0])
    transport.stays_open = True
    result = queue.run(replace(case, max_calls=32), transport, FakeHost())
    assert result.state == "uncertain"
    assert transport.calls.count("native.org.default.app_open_document") == 1
    assert transport.calls.count("native.org.default.ui_perform") == 1
    assert transport.calls.count("native.org.default.fcp_export") == 0


def test_busy_then_wrong_observed_pid_refuses_export(case: queue.Queue) -> None:
    class WrongPID(BusyObservationTransport):
        def call(self, tool: str, arguments: dict[str, object], timeout: int) -> dict[str, object]:
            if tool.endswith("ui_windows") and self.confirmed and not self.busy_remaining:
                self.calls.append(tool)
                return envelope({"pid": 999, "windows": [{"title": "Editor", "sheets": []}]})
            return super().call(tool, arguments, timeout)

    transport = WrongPID(case.parts[0])
    result = queue.run(replace(case, max_calls=32), transport, FakeHost())
    assert result.state == "uncertain"
    assert transport.calls.count("native.org.default.app_open_document") == 1
    assert transport.calls.count("native.org.default.ui_perform") == 1
    assert transport.calls.count("native.org.default.fcp_export") == 0


@pytest.mark.parametrize(
    "detail",
    [
        {"retrySafe": False, "error": "Accessibility permission denied"},
        {"retrySafe": True, "error": queue.BUSY_ERROR},
        {"retrySafe": False, "error": queue.BUSY_ERROR, "dispatched": True},
        {"retrySafe": False, "error": "AXError -25204: unrelated malformed error"},
        ["invalid"],
    ],
)
def test_unrecognized_readonly_errors_are_not_retryable(case: queue.Queue, detail: object) -> None:
    class OtherError(ChooserTransport):
        def call(self, tool: str, arguments: dict[str, object], timeout: int) -> dict[str, object]:
            if tool.endswith("ui_windows") and self.confirmed:
                self.calls.append(tool)
                return busy_envelope(detail)
            return super().call(tool, arguments, timeout)

    transport = OtherError(case.parts[0])
    result = queue.run(replace(case, max_calls=32), transport, FakeHost())
    assert result.state == "uncertain"
    assert transport.calls.count("native.org.default.fcp_export") == 0
    assert not list(case.root.glob("*.readonly-retry.json"))


@pytest.mark.parametrize(
    "phase", ["import", "chooser-select", "chooser-confirm", "project", "export", "background-open"]
)
def test_busy_error_never_replays_mutating_phase(phase: str) -> None:
    assert queue._readonly_busy(phase, busy_envelope()) is False


class RenderTransport(FakeTransport):
    busy_remaining = 2

    def call(self, tool: str, arguments: dict[str, object], timeout: int) -> dict[str, object]:
        if tool.endswith("ui_inspect") and self.busy_remaining:
            self.calls.append(tool)
            self.busy_remaining -= 1
            return envelope(
                {
                    "pid": 123,
                    "truncated": False,
                    "elements": [
                        {"value": "Sharing"},
                        *[
                            {"identifier": "_NS:166", "value": "Working", "path": [index]}
                            for index in range(9)
                        ],
                    ],
                }
            )
        return super().call(tool, arguments, timeout)


@pytest.fixture
def case(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> queue.Queue:
    monkeypatch.setattr(queue, "GUI_LEASE_ROOT", tmp_path)
    source = tmp_path / "source.fcpxml"
    source.write_text('<fcpxml><project name="Synthetic portrait"/></fcpxml>', encoding="utf-8")
    executable = tmp_path / "executor"
    executable.write_bytes(b"synthetic executor identity")
    movie = tmp_path / "output.mov"
    requests: dict[str, dict[str, object]] = {
        "import": {"path": str(source)},
        "windows": {},
        "project": {
            "project": "Synthetic portrait",
            "library": "Synthetic library",
            "event": "Synthetic event",
        },
        "export": {
            "project": "Synthetic portrait",
            "directory": str(tmp_path),
            "fileName": "output.mov",
            "codec": "Apple ProRes 422 LT",
            "format": "videoAndAudio",
            "allowForeground": True,
        },
        "background-open": {"menuPath": ["Window", "Background Tasks"]},
        "background-wait": {
            "window": {"title": "Background Tasks"},
            "condition": "absent",
            "element": {"role": "AXProgressIndicator"},
            "timeoutSeconds": 30,
        },
        "background": {"window": {"title": "Background Tasks"}},
        "background-close": {
            "window": {"title": "Background Tasks"},
            "element": {"subrole": "AXCloseButton"},
            "action": "press",
        },
    }
    operations = [
        {
            "phase": phase,
            "request": pin(
                write(
                    tmp_path / f"{phase}.json",
                    {
                        "bundleID": "com.apple.FinalCutApp",
                        **({} if phase in ("project", "export") else {"app": "finalCutPro"}),
                        **args,
                    },
                )
            ),
        }
        for phase, args in requests.items()
    ]
    inventory = write(
        tmp_path / "inventory.json",
        {
            "capturedAtNs": FakeHost.timestamp,
            "activeRenders": 0,
        },
    )
    events = tmp_path / "events"
    events.mkdir()
    grant = write(
        tmp_path / "grant.json",
        {
            "version": 1,
            "kind": "native-gui-idle-handoff",
            "released": True,
            "eventRoot": str(events),
            "previousOwner": "synthetic GUI owner",
            "target": {"pid": 123, "birth": "100:200", "bundleID": "com.apple.FinalCutApp"},
        },
    )
    descriptor = write(
        tmp_path / "queue.json",
        {
            "version": 1,
            "kind": "executor-native-part-queue",
            "provider": "native.org.default",
            "corePins": [pin(source)],
            "guiGrant": pin(grant),
            "executor": pin(executable),
            "eventRoot": str(events),
            "bootVolume": str(tmp_path),
            "inventory": str(inventory),
            "target": {"pid": 123, "birth": "100:200", "bundleID": "com.apple.FinalCutApp"},
            "parts": [
                {
                    "project": "Synthetic portrait",
                    "movie": str(movie),
                    "operations": operations,
                    "assemblyPart": 21,
                    "libraryBasename": "Synthetic library.fcpbundle",
                }
            ],
            "limits": {
                "maxCalls": 10,
                "callSeconds": 60,
                "observationSeconds": 30,
                "peakAdditionalBytes": 10,
                "minimumFreeBytes": 1,
                "capacity": 1,
            },
        },
    )
    return queue.load(descriptor)


@pytest.mark.parametrize("basename", ["Synthetic library.fcpbundle", "別のライブラリ.fcpbundle"])
def test_exact_chooser_then_full_idle_ready(case: queue.Queue, basename: str) -> None:
    request = case.parts[0].operations[2].request
    body = json.loads(request.read_text())
    body["library"] = basename.removesuffix(".fcpbundle")
    write(request, body)
    part = replace(case.parts[0], library_basename=basename)
    revised = replace(
        case,
        parts=(part,),
        max_calls=20,
        pins=tuple((path, file_digest(path)) for path, _ in case.pins),
    )
    transport = ChooserTransport(part)
    result = queue.run(revised, transport, FakeHost())
    assert result.state == "followup-ready"
    assert result.calls == 14
    assert transport.calls.count("native.org.default.fcp_export") == 1
    assert transport.calls.count("native.org.default.ui_set_value") == 1
    select = json.loads((case.root / "generated-0003-00-chooser-select.json").read_text())
    assert select["element"]["path"] == [2, 3]
    assert select["app"] == "finalCutPro"
    ready = json.loads((case.root / "followup-ready.json").read_text())
    assert ready["assemblyPart"] == 21
    assert ready["queueOrdinal"] == 1
    assert Path(ready["exportResponse"]["path"]).name == "part-0001-export.native.json"
    assert Path(ready["exportRequest"]["path"]).name == "export.json"


@pytest.mark.parametrize(
    "failure", ["disabled", "stale_row", "wait_met", "stays_open", "duplicate_name"]
)
def test_chooser_uncertainty_never_exports(case: queue.Queue, failure: str) -> None:
    transport = ChooserTransport(case.parts[0])
    setattr(transport, failure, failure != "wait_met")
    result = queue.run(replace(case, max_calls=20), transport, FakeHost())
    assert result.state == "uncertain"
    assert "native.org.default.fcp_export" not in transport.calls
    assert transport.calls.count("native.org.default.ui_set_value") <= 1
    assert not case.parts[0].movie.exists()


def test_finite_render_poll_retains_final_idle_artifact(case: queue.Queue) -> None:
    transport = RenderTransport(case.parts[0])
    result = queue.run(replace(case, max_calls=20), transport, FakeHost())
    assert result.state == "followup-ready"
    assert result.calls == 12
    assert transport.calls.count("native.org.default.fcp_export") == 1
    assert transport.calls.count("native.org.default.ui_wait") == 3
    ready = json.loads((case.root / "followup-ready.json").read_text())
    assert Path(ready["background"]["path"]).name == "part-0001-background-0003.native.json"
    assert (case.root / "part-0001-background.native.json").is_file()


def test_busy_render_budget_preserves_unknown_without_resubmission(case: queue.Queue) -> None:
    transport = RenderTransport(case.parts[0])
    transport.busy_remaining = 100
    result = queue.run(case, transport, FakeHost())
    assert result.state == "bounded-stop"
    assert result.calls == 10
    assert transport.calls.count("native.org.default.fcp_export") == 1
    assert case.parts[0].movie.is_file()
    assert not (case.root / "followup-ready.json").exists()
    assert (case.root / "owner.json").is_file()


@pytest.mark.parametrize(
    "field,value",
    [("assemblyPart", 0), ("assemblyPart", True), ("libraryBasename", "../x.fcpbundle")],
)
def test_required_assembly_and_library_identity(
    case: queue.Queue, field: str, value: object
) -> None:
    descriptor = json.loads(case.config.read_text())
    descriptor["parts"][0][field] = value
    write(case.config, descriptor)
    with pytest.raises(ReplayError):
        queue.load(case.config)


def test_duplicate_assembly_identity_is_not_another_part(case: queue.Queue) -> None:
    descriptor = json.loads(case.config.read_text())
    another = {**descriptor["parts"][0], "movie": str(case.root / "another.mov")}
    descriptor["parts"].append(another)
    write(case.config, descriptor)
    with pytest.raises(ReplayError, match="Assembly part"):
        queue.load(case.config)


def test_project_library_does_not_match_selected_chooser(case: queue.Queue) -> None:
    operation = case.parts[0].operations[2]
    body = json.loads(operation.request.read_text())
    body["library"] = "An unrelated library"
    write(operation.request, body)
    revised = replace(case, pins=tuple((path, file_digest(path)) for path, _ in case.pins))
    with pytest.raises(ReplayError, match="Exact library"):
        queue._arguments(revised, revised.parts[0], operation)


def test_unverified_row_selection_does_not_confirm(case: queue.Queue) -> None:
    state = queue.ObservationState("Open Library", (2, 3))
    with pytest.raises(ReplayError, match="selection is not verified"):
        queue._next_observations(
            "chooser-select",
            {"effectVerified": False, "element": {"selected": True, "path": [2, 3]}},
            queue=case,
            part=case.parts[0],
            state=state,
            now=FakeHost.timestamp,
        )


@pytest.mark.parametrize("path", [[], [-1], [True], [4097], ["0"]])
def test_invalid_observed_paths_are_not_gui_targets(path: object) -> None:
    with pytest.raises(ReplayError, match="AX element path"):
        queue._ax_path({"path": path})


def test_chooser_ambiguous_windows_stops_before_selection() -> None:
    with pytest.raises(ReplayError, match="Ambiguous"):
        queue._chooser_title({"windows": [{"title": "Open Library"}, {"title": "Open Library"}]})


def test_unreadable_library_row_is_not_selected() -> None:
    with pytest.raises(ReplayError, match="row is absent"):
        queue._library_row(
            {
                "truncated": False,
                "elements": [
                    {
                        "role": "AXTextField",
                        "value": "Library.fcpbundle",
                        "path": [3, 0],
                        "unreadable": [],
                    },
                    {"role": "AXRow", "path": [3], "unreadable": ["AXSelected"]},
                ],
            },
            "Library.fcpbundle",
        )


def test_finite_run_preserves_false_effect_without_quality_claim(case: queue.Queue) -> None:
    transport = FakeTransport(case.parts[0])
    result = queue.run(case, transport, FakeHost())
    assert result.state == "followup-ready"
    assert result.calls == 8
    assert len(transport.calls) == 8
    assert (
        json.loads((case.root / "followup-ready.json").read_text())["completionVerified"] is False
    )
    assert json.loads((case.root / "owner.json").read_text())["pid"] == os.getpid()


def test_reservation_forbids_replay_and_new_root(case: queue.Queue, tmp_path: Path) -> None:
    transport = FakeTransport(case.parts[0])
    transport.failure = subprocess.TimeoutExpired("executor", 5)
    assert queue.run(case, transport, FakeHost()).state == "uncertain"
    with pytest.raises(FileExistsError):
        queue.run(case, transport, FakeHost())
    new_root = tmp_path / "new-generation"
    new_root.mkdir()
    with pytest.raises(FileExistsError):
        queue.run(replace(case, root=new_root), transport, FakeHost())
    assert len(transport.calls) == 1


def test_failed_envelope_with_actual_output_is_uncertain(case: queue.Queue) -> None:
    class FailedExport(FakeTransport):
        def call(self, tool: str, arguments: dict[str, object], timeout: int) -> dict[str, object]:
            result = super().call(tool, arguments, timeout)
            if tool.endswith("fcp_export"):
                return {"exitCode": 0, "stdout": '{"ok":false,"data":{}}'}
            return result

    result = queue.run(case, FailedExport(case.parts[0]), FakeHost())
    assert result.state == "uncertain"
    assert result.calls == 4
    failure = json.loads((case.root / "part-0001-export.uncertain.json").read_text())
    assert failure["outputExists"] is True
    assert failure["retrySafe"] is False


@pytest.mark.parametrize("failure", [OSError("lost transport"), ReplayError("oversized")])
def test_transport_failure_retains_intent(case: queue.Queue, failure: Exception) -> None:
    transport = FakeTransport(case.parts[0])
    transport.failure = failure
    assert queue.run(case, transport, FakeHost()).state == "uncertain"
    assert (case.root / "part-0001-import.intent.json").exists()


def test_call_budget_stops_without_replay(case: queue.Queue) -> None:
    result = queue.run(replace(case, max_calls=1), FakeTransport(case.parts[0]), FakeHost())
    assert result.state == "bounded-stop"
    assert result.calls == 1


@pytest.mark.parametrize(
    "field,value",
    [
        ("version", 2),
        ("provider", "bad;tool"),
        ("parts", []),
        ("corePins", []),
        ("eventRoot", "/nonexistent-queue-root"),
    ],
)
def test_descriptor_rejections(case: queue.Queue, field: str, value: object) -> None:
    data = json.loads(case.config.read_text())
    data[field] = value
    write(case.config, data)
    with pytest.raises(ReplayError):
        queue.load(case.config)


@pytest.mark.parametrize("value", [0, True, "1", 10001])
def test_integer_bounds(value: object) -> None:
    with pytest.raises(ReplayError):
        queue._int(value, maximum=10000)


@pytest.mark.parametrize("value", [None, "", "x\0y", "x" * 513])
def test_text_bounds(value: object) -> None:
    with pytest.raises(ReplayError):
        queue._text(value)


def test_sequence_and_duplicate_outputs(case: queue.Queue) -> None:
    data = json.loads(case.config.read_text())
    data["parts"] *= 2
    write(case.config, data)
    with pytest.raises(ReplayError, match="unique"):
        queue.load(case.config)
    data["parts"] = data["parts"][:1]
    data["parts"][0]["operations"] = data["parts"][0]["operations"][:1]
    write(case.config, data)
    with pytest.raises(ReplayError, match="ordered"):
        queue.load(case.config)


@pytest.mark.parametrize(
    "field,value",
    [
        ("capturedAtNs", 1),
        ("capturedAtNs", FakeHost.timestamp + 1),
        ("activeRenders", 1),
        ("activeRenders", True),
    ],
)
def test_resource_inventory_gate(case: queue.Queue, field: str, value: object) -> None:
    data = json.loads(case.inventory.read_text())
    data[field] = value
    write(case.inventory, data)
    with pytest.raises(ReplayError):
        queue.run(case, FakeTransport(case.parts[0]), FakeHost())


def test_live_identity_and_disk_gate(case: queue.Queue) -> None:
    host = FakeHost()
    host.birth_value = "100:201"
    with pytest.raises(ReplayError, match="birth"):
        queue._guard(case, host, queue.Identity(1, "100:200"), case.parts[0], export=True)
    host.birth_value = "100:200"
    host.available = 100
    with pytest.raises(ReplayError, match="disk"):
        queue._guard(case, host, queue.Identity(1, "100:200"), case.parts[0], export=True)


def test_changed_pins_and_occupied_output(case: queue.Queue) -> None:
    case.parts[0].movie.write_bytes(b"existing")
    with pytest.raises(ReplayError, match="already exists"):
        queue._guard(case, FakeHost(), queue.Identity(1, "100:200"), case.parts[0], export=True)
    case.executable.write_bytes(b"changed")
    with pytest.raises(ReplayError, match="pins changed"):
        queue._guard(case, FakeHost(), queue.Identity(1, "100:200"), case.parts[0], export=False)


@pytest.mark.parametrize(
    "captured",
    [
        {"exitCode": 1},
        {"exitCode": 0},
        {"exitCode": 0, "stdout": "{"},
        {"exitCode": 0, "stdout": '{"ok":true,"data":{"isError":true}}'},
    ],
)
def test_failed_payload(captured: dict[str, object]) -> None:
    with pytest.raises((ReplayError, ValueError)):
        queue._payload(captured)


@pytest.mark.parametrize(
    "phase,payload",
    [
        ("import", {"pid": 124, "dispatched": True}),
        ("import", {"pid": 123, "dispatched": False}),
        ("windows", {"pid": 123, "windows": [{"title": "License", "subrole": "AXDialog"}]}),
        ("windows", {"pid": 123, "windows": [{"sheets": [{}]}]}),
        ("windows", {"pid": 123}),
        ("project", {"pid": 123, "finalCut": {"project": {"name": "Wrong"}}}),
    ],
)
def test_wrong_or_unknown_effect(case: queue.Queue, phase: str, payload: dict[str, object]) -> None:
    with pytest.raises(ReplayError):
        queue._accept(phase, payload, case, case.parts[0], FakeHost.timestamp)


@pytest.mark.parametrize("change", ["truncated", "missing", "active", "empty-movie", "chronology"])
def test_idle_inventory_and_chronology(case: queue.Queue, change: str) -> None:
    payload = background()
    elements = [
        {"value": "Sharing"},
        *[{"identifier": "_NS:166", "value": "Idle", "path": [index]} for index in range(9)],
    ]
    case.parts[0].movie.write_bytes(b"candidate")
    now = FakeHost.timestamp
    if change == "truncated":
        payload["truncated"] = True
    elif change == "missing":
        payload["elements"] = elements[:-1]
    elif change == "active":
        payload["elements"] = [*elements[:-1], {"identifier": "_NS:166", "value": "Working"}]
    elif change == "empty-movie":
        case.parts[0].movie.write_bytes(b"")
    else:
        now = 1
    with pytest.raises(ReplayError):
        queue._accept("background", payload, case, case.parts[0], now)


def test_transport_uses_argv_and_preserves_stderr(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    captured: list[object] = []

    def fake(args: list[str], **kwargs: object) -> subprocess.CompletedProcess[bytes]:
        captured.append(args)
        return subprocess.CompletedProcess(args, 2, b"response", b"diagnostic")

    monkeypatch.setattr(subprocess, "run", fake)
    response = queue.ExecutorTransport(tmp_path / "executor").call(
        "provider.ui_windows", {"name": "$(no shell)"}, 5
    )
    assert response == {"exitCode": 2, "stdout": "response", "stderr": "diagnostic"}
    assert len(captured) == 1


def test_cli_check_has_no_app_dispatch(
    case: queue.Queue, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr("sys.argv", ["queue", "check", str(case.config)])
    queue.main()
    assert json.loads(capsys.readouterr().out) == {"parts": 1, "appDispatched": False}
    assert not (case.root / "owner.json").exists()


@pytest.mark.parametrize(
    "phase,patch",
    [
        ("project", {"project": "Other"}),
        ("export", {"fileName": "other.mov"}),
        ("windows", {"bundleID": "other.app"}),
        ("background-open", {"menuPath": ["File", "Delete"]}),
        ("background-close", {"element": {"subrole": "AXButton"}}),
        ("background", {"window": {"title": "Other"}}),
        ("import", {"path": "/unexpected.fcpxml"}),
        ("project", {"app": "finalCutPro"}),
        ("export", {"app": "finalCutPro"}),
    ],
)
def test_argument_contract(case: queue.Queue, phase: str, patch: dict[str, object]) -> None:
    operation = next(item for item in case.parts[0].operations if item.phase == phase)
    data = json.loads(operation.request.read_text())
    data.update(patch)
    write(operation.request, data)
    fresh = replace(case, pins=tuple((path, file_digest(path)) for path, _ in case.pins))
    with pytest.raises(ReplayError):
        queue._arguments(fresh, fresh.parts[0], operation)


@pytest.mark.parametrize(
    "phase",
    ["import", "windows", "background-open", "background-wait", "background", "background-close"],
)
@pytest.mark.parametrize("application", [None, "motion"])
def test_initial_application_contract_fails_before_dispatch(
    case: queue.Queue,
    monkeypatch: pytest.MonkeyPatch,
    phase: str,
    application: str | None,
) -> None:
    operation = next(item for item in case.parts[0].operations if item.phase == phase)
    arguments = json.loads(operation.request.read_text())
    if application is None:
        del arguments["app"]
    else:
        arguments["app"] = application
    write(operation.request, arguments)
    raw = json.loads(case.config.read_text())
    raw["parts"][0]["operations"] = [
        {"phase": item.phase, "request": pin(item.request)} for item in case.parts[0].operations
    ]
    write(case.config, raw)
    monkeypatch.setattr(sys, "argv", ["queue", "check", str(case.config)])
    with pytest.raises(ReplayError, match="require app finalCutPro"):
        queue.main()
    assert not (case.root / "owner.json").exists()
    assert not (case.root / "followup-ready.json").exists()
    assert not case.parts[0].movie.exists()


def test_default_inspection_partial_inventory_cannot_release_followup(case: queue.Queue) -> None:
    class PartialTransport(FakeTransport):
        def call(self, tool: str, arguments: dict[str, object], timeout: int) -> dict[str, object]:
            response = super().call(tool, arguments, timeout)
            if tool.endswith("ui_inspect"):
                assert "maxElements" not in arguments
                return envelope(
                    {
                        "pid": 123,
                        "truncated": False,
                        "elements": [
                            {"value": "Sharing"},
                            *[
                                {"identifier": "_NS:166", "value": "Idle", "path": [index]}
                                for index in range(8)
                            ],
                        ],
                    }
                )
            return response

    transport = PartialTransport(case.parts[0])
    result = queue.run(case, transport, FakeHost())
    assert result.state == "uncertain"
    assert case.parts[0].movie.exists()
    assert not (case.root / "followup-ready.json").exists()
    assert transport.calls.count("native.org.default.fcp_export") == 1


def test_native_identity_uses_microseconds(monkeypatch: pytest.MonkeyPatch) -> None:
    class FakeFunction:
        argtypes: object = None
        restype: object = None

        def __call__(
            self, pid: int, flavor: int, arg: int, buffer: ctypes.Array[ctypes.c_char], size: int
        ) -> int:
            payload = (
                bytes(120) + (100).to_bytes(8, sys.byteorder) + (234567).to_bytes(8, sys.byteorder)
            )
            buffer.raw = payload
            return 136

    class FakeLibrary:
        proc_pidinfo = FakeFunction()

    monkeypatch.setattr("replay.fcp_native_queue.sys.platform", "darwin")
    monkeypatch.setattr(
        "replay.fcp_native_queue.ctypes.CDLL", lambda *args, **kwargs: FakeLibrary()
    )
    assert queue.SystemHost().birth(123) == "100:234567"


def test_non_native_host_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("replay.fcp_native_queue.sys.platform", "linux")
    with pytest.raises(ReplayError, match="macOS"):
        queue.SystemHost().birth(123)


def test_host_clock_and_disk(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("replay.fcp_native_queue.time.time_ns", lambda: 123)
    assert queue.SystemHost().now_ns() == 123
    assert queue.SystemHost().free(tmp_path) > 0
    assert queue.SystemHost().floor(tmp_path) >= 30 * 1024**3


def test_output_symlink_guard(case: queue.Queue) -> None:
    case.parts[0].movie.symlink_to(case.root / "missing")
    with pytest.raises(ReplayError, match="symlink"):
        queue._guard(case, FakeHost(), queue.Identity(1, "100:200"), case.parts[0], export=True)


def test_oversize_transport(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    def fake(args: list[str], **kwargs: object) -> subprocess.CompletedProcess[bytes]:
        return subprocess.CompletedProcess(args, 0, b"x" * (queue.MAX_BYTES + 1), b"")

    monkeypatch.setattr(subprocess, "run", fake)
    with pytest.raises(ReplayError, match="budget"):
        queue.ExecutorTransport(tmp_path / "executor").call("provider.ui_windows", {}, 5)


def test_cli_run_uses_finite_owner(
    case: queue.Queue, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr("sys.argv", ["queue", "run", str(case.config)])
    monkeypatch.setattr(queue, "SystemHost", FakeHost)
    monkeypatch.setattr(queue, "ExecutorTransport", lambda executable: FakeTransport(case.parts[0]))
    queue.main()
    assert json.loads(capsys.readouterr().out)["state"] == "followup-ready"


@pytest.mark.parametrize(
    "name,geometry",
    [
        ("Unknown landscape", "1920 1080"),
        ("別の縦配信", "720 1280"),
    ],
)
def test_geometry_independent_and_next_part_waits_for_proof(
    case: queue.Queue,
    name: str,
    geometry: str,
) -> None:
    source = case.pins[0][0]
    source.write_text(f'<fcpxml geometry="{geometry}"/>', encoding="utf-8")
    project_request = case.parts[0].operations[2].request
    export_request = case.parts[0].operations[3].request
    project = json.loads(project_request.read_text())
    project["project"] = name
    write(project_request, project)
    export = json.loads(export_request.read_text())
    export["project"] = name
    write(export_request, export)
    part = replace(case.parts[0], project=name)
    second = replace(part, movie=part.movie.with_name("later.mov"))
    fresh = replace(
        case, parts=(part, second), pins=tuple((path, file_digest(path)) for path, _ in case.pins)
    )
    transport = FakeTransport(part)
    assert queue.run(fresh, transport, FakeHost()).calls == 8
    assert not second.movie.exists()
    assert json.loads((fresh.root / "followup-ready.json").read_text())["remainingParts"] == 1


def test_identity_cli_is_read_only(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr("sys.argv", ["queue", "identity", "--pid", "123"])
    monkeypatch.setattr(queue, "SystemHost", FakeHost)
    queue.main()
    assert json.loads(capsys.readouterr().out) == {"pid": 123, "birth": "100:200"}


def test_no_gui_grant_no_run(case: queue.Queue) -> None:
    data = json.loads(case.config.read_text())
    grant_path = Path(data["guiGrant"]["path"])
    grant = json.loads(grant_path.read_text())
    grant["released"] = False
    write(grant_path, grant)
    data["guiGrant"] = pin(grant_path)
    write(case.config, data)
    with pytest.raises(ReplayError, match="ownership grant"):
        queue.load(case.config)


def test_long_export_does_not_refresh_or_block_on_old_resource_inventory(case: queue.Queue) -> None:
    host = FakeHost()
    original_inventory = case.inventory.read_bytes()

    class SlowExport(FakeTransport):
        def call(self, tool: str, arguments: dict[str, object], timeout: int) -> dict[str, object]:
            result = super().call(tool, arguments, timeout)
            if tool.endswith("fcp_export"):
                host.timestamp += 73_000_000_000
            return result

    result = queue.run(case, SlowExport(case.parts[0]), host)
    assert result.state == "followup-ready"
    assert result.calls == 8
    assert case.inventory.read_bytes() == original_inventory
    with pytest.raises(ReplayError, match="stale"):
        queue._guard(case, host, queue.Identity(1, "100:200"), case.parts[0], export=True)


def test_pidless_tool_evidence_remains_unmodified(case: queue.Queue) -> None:
    assert queue.run(case, FakeTransport(case.parts[0]), FakeHost()).state == "followup-ready"
    imported = json.loads((case.root / "part-0001-import.native.json").read_text())
    identity = json.loads((case.root / "part-0001-import.identity.json").read_text())
    assert "pid" not in imported["data"]["structuredContent"]
    assert "dispatched" not in imported["data"]["structuredContent"]
    assert identity["nativePIDPresent"] is False
    assert identity["beforeAndAfterMatched"] is True


@pytest.mark.parametrize("condition", [True, False])
def test_pidless_wait_is_observation_not_completion(case: queue.Queue, condition: bool) -> None:
    queue._accept(
        "background-wait",
        {"conditionMet": condition, "elapsedSeconds": 30.01},
        case,
        case.parts[0],
        FakeHost.timestamp,
    )
    assert not (case.root / "followup-ready.json").exists()


@pytest.mark.parametrize(
    "phase,payload",
    [
        ("import", {"delivery": "accepted", "completionVerified": True}),
        ("import", {"delivery": "", "completionVerified": False}),
        ("background-wait", {"conditionMet": True, "elapsedSeconds": True}),
        ("background-wait", {"conditionMet": True, "elapsedSeconds": float("nan")}),
        ("background-wait", {"conditionMet": "true", "elapsedSeconds": 1}),
        ("background-wait", {"conditionMet": True, "elapsedSeconds": -1}),
    ],
)
def test_actual_shape_rejects_unknown_values(
    case: queue.Queue, phase: str, payload: dict[str, object]
) -> None:
    with pytest.raises(ReplayError):
        queue._accept(phase, payload, case, case.parts[0], FakeHost.timestamp)


@pytest.mark.parametrize("value", [0, 31, True, float("inf")])
def test_wait_request_is_finite(case: queue.Queue, value: object) -> None:
    operation = next(op for op in case.parts[0].operations if op.phase == "background-wait")
    request = json.loads(operation.request.read_text())
    request["timeoutSeconds"] = value
    write(operation.request, request)
    fresh = replace(case, pins=tuple((path, file_digest(path)) for path, _ in case.pins))
    with pytest.raises(ReplayError, match="bounded"):
        queue._arguments(fresh, fresh.parts[0], operation)


def category_inventory(progress: object = None, *, japanese: bool = False) -> dict[str, object]:
    rows: list[dict[str, object]] = []
    for index in range(9):
        parent = [2, 4, index]
        rows.extend(
            [
                {"path": parent, "role": "AXGroup", "unreadable": []},
                {
                    "path": [*parent, 0],
                    "role": "AXStaticText",
                    "identifier": "_NS:26",
                    "value": ("共有" if japanese else "Sharing")
                    if index == 5
                    else f"Synthetic category {index}",
                    "unreadable": [],
                },
                {
                    "path": [*parent, 2],
                    "role": "AXProgressIndicator"
                    if index == 5 and progress is not None
                    else "AXStaticText",
                    "identifier": "_NS:77" if index == 5 and progress is not None else "_NS:166",
                    "value": progress
                    if index == 5 and progress is not None
                    else ("待機中" if japanese else "Idle"),
                    "unreadable": [],
                },
            ]
        )
    return {"pid": 123, "truncated": False, "elements": rows}


class ProgressTransport(FakeTransport):
    def __init__(self, part: queue.Part, host: FakeHost, states: list[object]) -> None:
        super().__init__(part)
        self.host = host
        self.states = states

    def call(self, tool: str, arguments: dict[str, object], timeout: int) -> dict[str, object]:
        if tool.endswith("ui_wait"):
            self.calls.append(tool)
            self.host.timestamp += 30_000_000_000
            return envelope({"conditionMet": False, "elapsedSeconds": 30.01})
        if tool.endswith("ui_inspect"):
            self.calls.append(tool)
            value = self.states.pop(0) if self.states else None
            if value is not None:
                with self.part.movie.open("ab") as stream:
                    stream.write(b"native writer growth")
            return envelope(category_inventory(value))
        return super().call(tool, arguments, timeout)


@pytest.mark.parametrize("japanese", [False, True])
@pytest.mark.parametrize("progress", [0, "0.5820066928863525", 1.0])
def test_real_category_progress_is_busy_until_status_idle(japanese: bool, progress: object) -> None:
    assert queue._background_idle(category_inventory(progress, japanese=japanese)) is False
    assert queue._background_idle(category_inventory(japanese=japanese)) is True


@pytest.mark.parametrize("progress", [True, "invalid", "nan", float("inf"), -0.1, 1.1, {}, []])
def test_corrupt_category_progress_is_not_completion(progress: object) -> None:
    with pytest.raises(ReplayError, match="progress"):
        queue._background_idle(category_inventory(progress))


@pytest.mark.parametrize(
    "change", ["missing", "truncated", "label", "duplicate-label", "group", "state", "unreadable"]
)
def test_partial_busy_category_inventory_is_refused(change: str) -> None:
    payload = category_inventory(0.58)
    elements = payload["elements"]
    assert isinstance(elements, list)
    if change == "truncated":
        payload["truncated"] = True
    elif change == "missing":
        elements.pop()
    elif change == "label":
        elements[1]["value"] = []
    elif change == "duplicate-label":
        elements.append(elements[1])
    elif change == "group":
        elements[0]["role"] = "AXButton"
    elif change == "state":
        elements.append(elements[2])
    else:
        elements[2]["unreadable"] = ["value"]
    with pytest.raises(ReplayError):
        queue._background_idle(payload)


def test_progress_busy_busy_idle_uses_only_readonly_polls(case: queue.Queue) -> None:
    host = FakeHost()
    transport = ProgressTransport(case.parts[0], host, [0.58, 1.0, None])
    result = queue.run(replace(case, max_calls=20), transport, host)
    assert result.state == "followup-ready"
    assert result.calls == 12
    assert transport.calls.count("native.org.default.app_open_document") == 1
    assert transport.calls.count("native.org.default.fcp_export") == 1
    assert transport.calls.count("native.org.default.ui_wait") == 3
    ready = json.loads((case.root / "followup-ready.json").read_text())
    assert ready["background"]["path"].endswith("background-0003.native.json")
    assert not (case.root / "part-0001-background.uncertain.json").exists()


def test_progress_sustained_busy_stops_with_original_effects_preserved(case: queue.Queue) -> None:
    host = FakeHost()
    transport = ProgressTransport(case.parts[0], host, [0.5] * 30)
    result = queue.run(replace(case, max_calls=20), transport, host)
    assert result.state == "bounded-stop"
    assert result.calls == 20
    assert transport.calls.count("native.org.default.fcp_export") == 1
    assert not (case.root / "followup-ready.json").exists()
    assert case.parts[0].movie.is_file()
    assert (case.root / "owner.json").is_file()
    with pytest.raises(FileExistsError):
        queue.run(case, transport, host)


def test_late_wait_response_does_not_claim_native_completion(case: queue.Queue) -> None:
    host = FakeHost()
    transport = ProgressTransport(case.parts[0], host, [0.5] * 30)
    result = queue.run(replace(case, max_calls=20, timeout=1), transport, host)
    assert result.state == "uncertain"
    assert transport.calls.count("native.org.default.fcp_export") == 1
    reason = json.loads((case.root / "part-0001-background-wait.uncertain.json").read_text())
    assert "finite execution interval" in reason["reason"]
    assert not (case.root / "followup-ready.json").exists()


@pytest.mark.parametrize(
    "change",
    ["pid", "replacement", "shrink", "disappearance", "symlink", "hardlink", "future", "close"],
)
def test_postexport_mutation_never_replays_effects(case: queue.Queue, change: str) -> None:
    part = case.parts[0]
    host = FakeHost()

    class ChangedOutput(FakeTransport):
        def call(self, tool: str, arguments: dict[str, object], timeout: int) -> dict[str, object]:
            captured = super().call(tool, arguments, timeout)
            if tool.endswith("ui_inspect"):
                if change == "pid":
                    return envelope({**category_inventory(), "pid": 999})
                if change == "replacement":
                    replacement = part.movie.with_name("replacement.mov")
                    replacement.write_bytes(part.movie.read_bytes())
                    os.replace(replacement, part.movie)
                elif change == "shrink":
                    part.movie.write_bytes(b"x")
                elif change == "disappearance":
                    part.movie.unlink()
                elif change == "symlink":
                    part.movie.unlink()
                    part.movie.symlink_to(case.config)
                elif change == "hardlink":
                    part.movie.with_name("linked.mov").hardlink_to(part.movie)
                elif change == "future":
                    os.utime(part.movie, ns=(host.timestamp + 1, host.timestamp + 1))
            if tool.endswith("ui_perform") and change == "close":
                with part.movie.open("ab") as stream:
                    stream.write(b"late native write")
            return captured

    transport = ChangedOutput(part)
    result = queue.run(case, transport, host)
    assert result.state == "uncertain"
    assert transport.calls.count("native.org.default.app_open_document") == 1
    assert transport.calls.count("native.org.default.fcp_export") == 1
    assert not (case.root / "followup-ready.json").exists()


def test_idle_with_new_output_size_waits_for_settled_output(case: queue.Queue) -> None:
    class FinishingWriter(FakeTransport):
        inspections = 0

        def call(self, tool: str, arguments: dict[str, object], timeout: int) -> dict[str, object]:
            if tool.endswith("ui_inspect"):
                self.inspections += 1
                if self.inspections == 1:
                    with self.part.movie.open("ab") as stream:
                        stream.write(b"buffered final write")
            return super().call(tool, arguments, timeout)

    transport = FinishingWriter(case.parts[0])
    result = queue.run(case, transport, FakeHost())
    assert result.state == "followup-ready"
    assert result.calls == 10
    assert transport.inspections == 2
    assert transport.calls.count("native.org.default.fcp_export") == 1


@pytest.mark.parametrize("change", ["foreign-progress", "empty-status", "nontext-status"])
def test_unbound_or_corrupt_category_state_is_refused(change: str) -> None:
    payload = category_inventory()
    elements = payload["elements"]
    assert isinstance(elements, list)
    if change == "foreign-progress":
        elements.append(
            {"path": [9, 9], "role": "AXProgressIndicator", "value": 0.7, "unreadable": []}
        )
    else:
        elements[2]["value"] = "" if change == "empty-status" else None
    with pytest.raises(ReplayError):
        queue._background_idle(payload)


def test_output_first_appearing_at_idle_needs_another_observation(case: queue.Queue) -> None:
    class DelayedFile(FakeTransport):
        def call(self, tool: str, arguments: dict[str, object], timeout: int) -> dict[str, object]:
            captured = super().call(tool, arguments, timeout)
            if tool.endswith("fcp_export"):
                self.part.movie.unlink()
            if tool.endswith("ui_inspect") and not self.part.movie.exists():
                self.part.movie.write_bytes(b"completed later native candidate")
            return captured

    transport = DelayedFile(case.parts[0])
    result = queue.run(case, transport, FakeHost())
    assert result.state == "followup-ready"
    assert result.calls == 10
    assert transport.calls.count("native.org.default.fcp_export") == 1
    final = json.loads((case.root / "part-0001-background-close.identity.json").read_text())
    assert final["outputObservation"]["bytes"] == case.parts[0].movie.stat().st_size


def test_elapsed_total_budget_stops_before_another_readonly_poll(case: queue.Queue) -> None:
    class DeadlineHost(FakeHost):
        remaining_reads = -1

        def now_ns(self) -> int:
            if self.remaining_reads >= 0:
                self.remaining_reads -= 1
                if self.remaining_reads < 0:
                    self.timestamp += 20 * case.timeout * 1_000_000_000
            return self.timestamp

    host = DeadlineHost()

    class ClockTransport(FakeTransport):
        def call(self, tool: str, arguments: dict[str, object], timeout: int) -> dict[str, object]:
            if tool.endswith("ui_inspect"):
                self.calls.append(tool)
                host.remaining_reads = 1
                return envelope(category_inventory(0.7))
            return super().call(tool, arguments, timeout)

    transport = ClockTransport(case.parts[0])
    result = queue.run(replace(case, max_calls=20), transport, host)
    assert result.state == "bounded-stop"
    assert result.calls == 7
    assert "time budget" in result.reason
    assert transport.calls.count("native.org.default.fcp_export") == 1
    assert not (case.root / "followup-ready.json").exists()


def test_write_after_close_cannot_create_ready_marker(
    case: queue.Queue, monkeypatch: pytest.MonkeyPatch
) -> None:
    original = queue._output_observation
    observations = 0

    def late_write(
        part: queue.Part, state: queue.ObservationState, now: int
    ) -> queue.OutputObservation | None:
        nonlocal observations
        observations += 1
        if observations == 7:
            with part.movie.open("ab") as stream:
                stream.write(b"late writer after close")
        return original(part, state, now)

    monkeypatch.setattr(queue, "_output_observation", late_write)
    transport = FakeTransport(case.parts[0])
    with pytest.raises(ReplayError, match="before followup readiness"):
        queue.run(case, transport, FakeHost())
    assert transport.calls.count("native.org.default.fcp_export") == 1
    assert not (case.root / "followup-ready.json").exists()
