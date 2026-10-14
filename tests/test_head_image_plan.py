import json
import sys
from dataclasses import replace
from fractions import Fraction
from pathlib import Path
from xml.etree import ElementTree as ET

import pytest

from replay import head_image_plan as plan
from replay import tracking_observations as storage
from replay.core import ReplayError
from replay.head_observations import TrackedFace, TrackingFrame, TrackingMethod
from replay.stages import file_digest
from replay.subtitle_clock import VideoFrameClock
from replay.tracking_observations import TrackingJob
from replay.vertical_captions import Box


def _frame(time: Fraction, box: Box | None = None) -> TrackingFrame:
    return TrackingFrame(
        time, TrackingMethod.DETECTION, True, () if box is None else (TrackedFace(box, 0.9),)
    )


def test_current_cells_round_face_outward_and_keep_missing_unknown() -> None:
    cells = plan.planned_cells(
        (_frame(Fraction(0), Box(1.2, 2.3, 4.1, 6.7)), _frame(Fraction(1, 30))),
        (Fraction(1, 30), Fraction(1, 15)),
        ((Fraction(1, 60), Fraction(1, 15)),),
    )
    assert cells[0].face == (1, 2, 4, 5)
    assert cells[1].face is None
    assert plan.image_cells([cell.encoded() for cell in cells], width=100, height=200) == cells
    face = cells[0].request_face()
    assert face is not None
    assert face["presentation"] == {"value": 0, "timescale": 1}


@pytest.mark.parametrize(
    "value",
    [
        [],
        None,
        [{"start": "0", "end": "1", "face": None, "held": True}],
        [{"start": "x", "end": "1", "face": None}],
        [{"start": "0", "end": "1/1000000001", "face": None}],
        [{"start": 0, "end": "1", "face": None}],
        [{"start": "1", "end": "0", "face": None}],
        [{"start": "0", "end": "2", "face": None}],
    ],
)
def test_unbounded_or_invalid_source_cells_rejected(value: object) -> None:
    with pytest.raises(ReplayError):
        plan.image_cells(value, width=100, height=200)


@pytest.mark.parametrize(
    "patch",
    [
        {"x": True},
        {"x": -1},
        {"width": 101},
        {"presentation": {"value": 1, "timescale": 1}},
        {"minimumFaceCoverage": 0.1},
    ],
)
def test_roi_must_be_observed_same_clock_in_canvas(patch: dict[str, object]) -> None:
    cell = plan.ImageFrameCell(Fraction(0), Fraction(1, 30), (1, 2, 4, 5))
    face = cell.request_face()
    assert face is not None
    with pytest.raises(ReplayError):
        plan.image_cells([{**cell.encoded(), "face": face | patch}], width=100, height=200)


def test_overlapping_cells_rejected() -> None:
    with pytest.raises(ReplayError, match="independent"):
        plan.image_cells(
            [{"start": "0", "end": "1", "face": None}, {"start": "1/2", "end": "1", "face": None}],
            width=100,
            height=200,
        )


def test_no_interpolation_across_source_cut_gap() -> None:
    frames = (_frame(Fraction(0)), _frame(Fraction(1)))
    assert (
        len(
            plan.planned_cells(
                frames,
                (Fraction(1, 30), Fraction(31, 30)),
                ((Fraction(0), Fraction(1, 30)), (Fraction(1), Fraction(31, 30))),
            )
        )
        == 2
    )
    with pytest.raises(ReplayError, match="complete actual"):
        plan.planned_cells(
            frames, (Fraction(1, 30), Fraction(31, 30)), ((Fraction(0), Fraction(31, 30)),)
        )
    with pytest.raises(ReplayError, match="cell ends"):
        plan.planned_cells(frames, (), ())
    with pytest.raises(ReplayError, match="frame budget"):
        plan.planned_cells((), (), ((Fraction(0), Fraction(1)),))


def _root() -> ET.Element:
    return ET.fromstring(
        "<fcpxml><library><event><project><sequence><spine>"
        '<asset-clip start="0s" offset="0s" duration="1/15s">'
        '<title offset="0s" start="3600s" duration="1/15s"/></asset-clip>'
        "</spine></sequence></project></event></library></fcpxml>"
    )


@pytest.mark.parametrize("change", ["timeMap", "outside", "missing"])
def test_caption_clock_cannot_be_reinterpreted(change: str) -> None:
    root = _root()
    clip = root.find(".//asset-clip")
    title = root.find(".//title")
    assert clip is not None and title is not None
    if change == "timeMap":
        ET.SubElement(clip, "timeMap")
    elif change == "outside":
        title.set("duration", "1s")
    else:
        clip.remove(title)
    with pytest.raises(ReplayError):
        plan.caption_spans(root)


@pytest.fixture
def planning(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[Path, TrackingJob]:
    files: dict[str, Path] = {}
    for name in ("xml", "index", "native", "sandbox"):
        path = tmp_path / name
        path.write_bytes(ET.tostring(_root()) if name == "xml" else b"{}")
        files[name] = path
    evidence = tmp_path / "evidence"
    evidence.mkdir()
    for name in ("request.json", "response.json", "provenance.json"):
        (evidence / name).write_text("{}", encoding="utf-8")
    config = tmp_path / "plan.json"
    config.write_text(
        json.dumps(
            {
                "version": 1,
                **{
                    ("faceIndex" if name == "index" else name): {
                        "path": str(path),
                        "sha256": file_digest(path),
                    }
                    for name, path in files.items()
                },
            }
        ),
        encoding="utf-8",
    )
    job = TrackingJob(
        files["index"],
        file_digest(files["index"]),
        files["native"],
        files["native"],
        file_digest(files["native"]),
        file_digest(files["native"]),
        VideoFrameClock((0, 1), Fraction(1, 30), 1),
        100,
        200,
        0.5,
    )
    monkeypatch.setattr(plan, "face_references", lambda _path: [(job.config, evidence)])
    monkeypatch.setattr(storage, "load_job", lambda _path: job)
    monkeypatch.setattr(storage, "verify", lambda _config, _output: None)
    monkeypatch.setattr(
        plan,
        "tracking_frames",
        lambda _raw, _clock, **_kwargs: (
            _frame(Fraction(0), Box(1, 2, 5, 7)),
            _frame(Fraction(1, 30)),
        ),
    )
    return config, job


def test_planning_pins_actual_cells_and_cli(
    planning: tuple[Path, TrackingJob], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config, _job = planning
    output = tmp_path / "collection.json"
    monkeypatch.setattr(sys, "argv", ["plan", str(config), str(output)])
    plan.main()
    raw = json.loads(output.read_text())
    assert len(raw["cells"]) == 2
    assert raw["cells"][0]["face"]["x"] == 1
    assert raw["cells"][1]["face"] is None
    assert raw["inputs"][str(config)] == file_digest(config)
    with pytest.raises(FileExistsError):
        plan.plan(config, output)


def test_inconsistent_window_source_rejected(
    planning: tuple[Path, TrackingJob], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config, job = planning
    monkeypatch.setattr(
        plan,
        "face_references",
        lambda _path: [(job.config, tmp_path / "evidence"), (config, tmp_path / "evidence")],
    )
    monkeypatch.setattr(
        storage,
        "load_job",
        lambda path: job if path == job.config else replace(job, source_hash="b" * 64),
    )
    with pytest.raises(ReplayError, match="one pinned"):
        plan.plan(config, tmp_path / "output.json")


def test_no_selected_windows_rejected(
    planning: tuple[Path, TrackingJob], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config, _job = planning
    monkeypatch.setattr(plan, "face_references", lambda _path: [])
    with pytest.raises(ReplayError, match="window budget"):
        plan.plan(config, tmp_path / "output.json")


def test_bad_plan_fields(tmp_path: Path) -> None:
    config = tmp_path / "bad.json"
    config.write_text("{}", encoding="utf-8")
    with pytest.raises(ReplayError, match="planning configuration"):
        plan.plan(config, tmp_path / "output.json")


def test_relative_cli_config_becomes_an_absolute_pin(
    planning: tuple[Path, TrackingJob], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config, _job = planning
    monkeypatch.chdir(tmp_path)
    result = plan.plan(Path(config.name), Path("collection.json"))
    inputs = result["inputs"]
    assert isinstance(inputs, dict)
    assert all(Path(path).is_absolute() for path in inputs)
