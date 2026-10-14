import json
import os
import sys
from pathlib import Path

import pytest

from replay import fcp_native_parts as native
from replay.core import ReplayError
from replay.stages import file_digest


@pytest.mark.parametrize(
    "wrapper",
    [
        {"ok": True, "data": {"isError": False, "structuredContent": {"pid": 123}}},
        {"ok": True, "result": {"pid": 123}},
        {"response": {"pid": 123}},
    ],
)
def test_observation_wrappers(tmp_path: Path, wrapper: dict[str, object]) -> None:
    path = tmp_path / "observation.json"
    path.write_text(json.dumps(wrapper), encoding="utf-8")
    assert native._observation(
        {"path": str(path), "sha256": file_digest(path)}, historical=True
    ) == (path, {"pid": 123})


@pytest.mark.parametrize(
    "wrapper",
    [
        {"ok": False, "data": {}},
        {"ok": True, "data": {"isError": True, "structuredContent": {}}},
        {"ok": True, "data": []},
        {"ok": True, "data": {"isError": False, "structuredContent": []}},
        {"ok": True, "data": {"structuredContent": {}}},
    ],
)
def test_observation_rejects_failed_v4(tmp_path: Path, wrapper: dict[str, object]) -> None:
    path = tmp_path / "observation.json"
    path.write_text(json.dumps(wrapper), encoding="utf-8")
    with pytest.raises(ReplayError):
        native._observation({"path": str(path), "sha256": file_digest(path)}, historical=True)


@pytest.fixture
def case(tmp_path: Path) -> tuple[Path, Path, Path]:
    part = tmp_path / "part.fcpxml"
    part.write_text(
        '<fcpxml><resources><format id="f" frameDuration="1/30s"/></resources>'
        '<library><event><project name="Example Part 0001"><sequence format="f" '
        'duration="1s"><spine><asset-clip start="10s" offset="0s" duration="1s">'
        '<title offset="10s" start="3600s" duration="1s"><text>hello</text></title>'
        "</asset-clip>"
        "</spine></sequence></project></event></library></fcpxml>",
        encoding="utf-8",
    )
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps({"version": 1, "parts": [{"path": str(part), "sha256": file_digest(part)}]}),
        encoding="utf-8",
    )
    project = tmp_path / "project.json"
    project.write_text(
        json.dumps({"finalCut": {"project": {"name": "Example Part 0001"}}}), encoding="utf-8"
    )
    background = tmp_path / "background.json"
    background.write_text(
        json.dumps(
            {
                "elements": [
                    {"role": "AXStaticText", "value": "Sharing"},
                    {"identifier": "_NS:166", "value": "Idle"},
                ]
            }
        ),
        encoding="utf-8",
    )
    panel = tmp_path / "panel.json"
    panel.write_text('{"elements": []}', encoding="utf-8")
    config = tmp_path / "config.json"
    config.write_text(
        json.dumps(
            {
                "version": 1,
                "partition": {"path": str(manifest), "sha256": file_digest(manifest)},
                "part": 1,
                "directory": str(tmp_path),
                "masterPrefix": "export",
            }
        ),
        encoding="utf-8",
    )
    observations = tmp_path / "observations.json"
    observations.write_text(
        json.dumps(
            {
                "version": 1,
                "observedDirectory": str(tmp_path),
                "windowTitle": "Save",
                "project": {"path": str(project), "sha256": file_digest(project)},
                "background": {"path": str(background), "sha256": file_digest(background)},
                "savePanel": {"path": str(panel), "sha256": file_digest(panel)},
            }
        ),
        encoding="utf-8",
    )
    return config, observations, tmp_path / "request.json"


def test_bound_request_never_dispatches(case: tuple[Path, Path, Path]) -> None:
    native.prepare(*case)
    request = json.loads(case[2].read_text())
    assert request["context"]["fileName"] == "export-0001-master.mov"
    assert request["context"]["saveDispatched"] is False
    assert request["historicalInspection"] is False
    receipt = json.loads(case[2].with_suffix(".binding.json").read_text())
    assert receipt["project"] == "Example Part 0001"
    assert receipt["completionVerified"] is False
    assert receipt["dispatched"] is False
    assert len(receipt["inputs"]) == 7


@pytest.mark.parametrize(
    "field,value",
    [
        ("version", 2),
        ("part", 0),
        ("part", True),
        ("part", 2),
        ("masterPrefix", "../escape"),
        ("directory", "/missing-directory"),
    ],
)
def test_invalid_config(case: tuple[Path, Path, Path], field: str, value: object) -> None:
    raw = json.loads(case[0].read_text())
    raw[field] = value
    case[0].write_text(json.dumps(raw))
    with pytest.raises(ReplayError):
        native.prepare(*case)


@pytest.mark.parametrize(
    "field,value",
    [
        ("historicalInspection", "yes"),
        ("observedDirectory", "/wrong"),
        ("windowTitle", ""),
        ("version", 2),
    ],
)
def test_invalid_observations(case: tuple[Path, Path, Path], field: str, value: object) -> None:
    raw = json.loads(case[1].read_text())
    raw[field] = value
    case[1].write_text(json.dumps(raw))
    with pytest.raises(ReplayError):
        native.prepare(*case)


@pytest.mark.parametrize(
    "kind,payload",
    [
        ("project", {"finalCut": {"project": {"name": "Wrong"}}}),
        ("background", {"elements": []}),
        ("background", {"elements": [{"role": "AXStaticText", "value": "Sharing"}]}),
        (
            "background",
            {
                "elements": [
                    {"role": "AXStaticText", "value": "Sharing"},
                    {"identifier": "_NS:166", "value": "10%"},
                ]
            },
        ),
        ("savePanel", {"response": {"elements": "invalid"}}),
        ("savePanel", {"elements": [], "truncated": True}),
        ("project", {"ok": False, "error": "failed", "result": {}}),
    ],
)
def test_native_identity_and_active_jobs(
    case: tuple[Path, Path, Path], kind: str, payload: object
) -> None:
    raw = json.loads(case[1].read_text())
    path = Path(raw[kind]["path"])
    path.write_text(json.dumps(payload))
    raw[kind]["sha256"] = file_digest(path)
    case[1].write_text(json.dumps(raw))
    with pytest.raises(ReplayError):
        native.prepare(*case)


def test_stale_requires_historical_mode(case: tuple[Path, Path, Path]) -> None:
    raw = json.loads(case[1].read_text())
    os.utime(raw["project"]["path"], (1, 1))
    with pytest.raises(ReplayError, match="stale"):
        native.prepare(*case)
    raw["historicalInspection"] = True
    case[1].write_text(json.dumps(raw))
    native.prepare(*case)
    assert json.loads(case[2].read_text())["historicalInspection"] is True


def test_existing_movie_refuses_resubmission(case: tuple[Path, Path, Path]) -> None:
    (case[0].parent / "export-0001-master.mov").write_bytes(b"existing")
    with pytest.raises(ReplayError, match="resubmit"):
        native.prepare(*case)


def test_successful_executor_envelope(case: tuple[Path, Path, Path]) -> None:
    raw = json.loads(case[1].read_text())
    project = Path(raw["project"]["path"])
    project.write_text(
        json.dumps(
            {
                "ok": True,
                "error": None,
                "result": {"finalCut": {"project": {"name": "Example Part 0001"}}},
            }
        )
    )
    raw["project"]["sha256"] = file_digest(project)
    case[1].write_text(json.dumps(raw))
    native.prepare(*case)
    assert case[2].with_suffix(".snapshot.json").exists()


def test_no_overwrite_and_cli(
    case: tuple[Path, Path, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(sys, "argv", ["native-parts", str(case[0]), str(case[1]), str(case[2])])
    native.main()
    with pytest.raises(ReplayError, match="overwritten"):
        native.prepare(*case)


@pytest.fixture
def transition_case(case: tuple[Path, Path, Path]) -> tuple[Path, Path, Path]:
    observations = json.loads(case[1].read_text())
    project = Path(observations["project"]["path"])
    background = Path(observations["background"]["path"])
    panel = Path(observations["savePanel"]["path"])
    project.write_text(
        json.dumps({"pid": 42, "finalCut": {"project": {"name": "Example Part 0001"}}})
    )
    background.write_text(
        json.dumps(
            {
                "pid": 42,
                "elements": [
                    {"role": "AXStaticText", "value": "Sharing"},
                    {"identifier": "_NS:166", "value": "Idle"},
                ],
            }
        )
    )
    os.utime(project, (1, 1))
    os.utime(background, (2, 2))
    transition = case[0].parent / "transition.json"
    transition.write_text(
        json.dumps(
            {
                "ok": True,
                "data": {
                    "_meta": {
                        "dispatch_state": "dispatched",
                        "effect": "unverifiable",
                        "target_receipt": {"pid": 42, "process_start_identity": 123},
                    }
                },
            }
        )
    )
    os.utime(transition, (3, 3))
    panel.write_text(
        json.dumps(
            {
                "pid": 42,
                "elements": [
                    {"identifier": "saveAsNameTextField", "value": "Example Part 0001.mov"}
                ],
            }
        )
    )
    observations["project"]["sha256"] = file_digest(project)
    observations["background"]["sha256"] = file_digest(background)
    observations["savePanel"]["sha256"] = file_digest(panel)
    observations["shareTransition"] = {"path": str(transition), "sha256": file_digest(transition)}
    case[1].write_text(json.dumps(observations))
    return case


def test_transition_binds_older_premodal_observations(
    transition_case: tuple[Path, Path, Path],
) -> None:
    native.prepare(*transition_case)
    receipt = json.loads(transition_case[2].with_suffix(".binding.json").read_text())
    assert len(receipt["inputs"]) == 8
    assert receipt["completionVerified"] is False


@pytest.mark.parametrize(
    "payload",
    [
        {"ok": False, "data": {}},
        {
            "_meta": {
                "dispatch_state": "unknown",
                "target_receipt": {"pid": 42, "process_start_identity": 123},
            }
        },
        {
            "_meta": {
                "dispatch_state": "dispatched",
                "target_receipt": {"pid": 43, "process_start_identity": 123},
            }
        },
    ],
)
def test_transition_rejects_unbound_dispatch(
    transition_case: tuple[Path, Path, Path], payload: object
) -> None:
    raw = json.loads(transition_case[1].read_text())
    path = Path(raw["shareTransition"]["path"])
    path.write_text(json.dumps(payload))
    raw["shareTransition"]["sha256"] = file_digest(path)
    transition_case[1].write_text(json.dumps(raw))
    with pytest.raises(ReplayError):
        native.prepare(*transition_case)


def test_transition_rejects_wrong_fresh_filename(transition_case: tuple[Path, Path, Path]) -> None:
    raw = json.loads(transition_case[1].read_text())
    path = Path(raw["savePanel"]["path"])
    path.write_text('{"pid":42,"elements":[]}')
    raw["savePanel"]["sha256"] = file_digest(path)
    transition_case[1].write_text(json.dumps(raw))
    with pytest.raises(ReplayError, match="filename"):
        native.prepare(*transition_case)


def test_transition_executor_wrapper(transition_case: tuple[Path, Path, Path]) -> None:
    raw = json.loads(transition_case[1].read_text())
    path = Path(raw["shareTransition"]["path"])
    document = json.loads(path.read_text())
    path.write_text(json.dumps({"ok": True, "error": None, "result": document}))
    os.utime(path, (3, 3))
    raw["shareTransition"]["sha256"] = file_digest(path)
    transition_case[1].write_text(json.dumps(raw))
    native.prepare(*transition_case)
    assert transition_case[2].exists()


def test_capture_raw_paths_cli(
    transition_case: tuple[Path, Path, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    raw = json.loads(transition_case[1].read_text())
    observation_output = transition_case[0].parent / "captured.json"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "parts",
            str(transition_case[0]),
            str(observation_output),
            str(transition_case[2]),
            "--capture-paths",
            raw["project"]["path"],
            raw["background"]["path"],
            raw["savePanel"]["path"],
            raw["observedDirectory"],
            raw["windowTitle"],
            "--share-transition",
            raw["shareTransition"]["path"],
        ],
    )
    native.main()
    assert json.loads(observation_output.read_text()) == raw
    assert json.loads(transition_case[2].read_text())["context"]["saveDispatched"] is False


def test_transition_cli_requires_capture(
    case: tuple[Path, Path, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "parts",
            str(case[0]),
            str(case[1]),
            str(case[2]),
            "--share-transition",
            "missing",
        ],
    )
    with pytest.raises(SystemExit):
        native.main()
