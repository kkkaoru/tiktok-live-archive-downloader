"""A fixed origin per run for a time range, from verified inputs, with failures recorded."""

import json
from fractions import Fraction
from pathlib import Path

import pytest

from replay import caption_fixed_y as fixed
from replay import caption_fixed_y_job as job
from replay.core import ReplayError
from replay.fcp_tracking_caption import FaceObservation, TrackingPosition
from replay.vertical_captions import Box

CUTS = [
    {"output_start_frame": 0, "source_start_frame": 10, "source_end_frame": 40},
    {"output_start_frame": 30, "source_start_frame": 100, "source_end_frame": 160},
]


def test_output_frames_map_into_their_cut_and_stay_inside_it() -> None:
    mapping = job.cut_map(CUTS)
    assert mapping.source_frame(0) == 10
    assert mapping.source_frame(29) == 39
    assert mapping.source_frame(30) == 100
    assert mapping.source_frame(60) == 130
    assert mapping.source_frame(500) == 159


def test_a_frame_before_the_plan_or_unordered_cuts_are_refused() -> None:
    shifted = [{"output_start_frame": 5, "source_start_frame": 0, "source_end_frame": 10}]
    with pytest.raises(ReplayError, match="precedes the cut plan"):
        job.cut_map(shifted).source_frame(0)
    with pytest.raises(ReplayError, match="ordered by output frame"):
        job.cut_map([CUTS[1], CUTS[0]])
    with pytest.raises(ReplayError, match="ordered by output frame"):
        job.cut_map([])


def test_a_page_envelope_grows_by_one_line_advance_per_line_and_uses_the_widest_line() -> None:
    sizes = {"あ": (100.0, 120.0), "いい": (300.0, 110.0)}
    envelope = job.top_envelope(sizes, padding=24.0, ascent=96.0, line_advance=121.0)
    one = envelope("あ")
    two = envelope("あ\nいい")
    assert one == Box(-50.0, -120.0, 50.0, 0.0)
    assert two == Box(-150.0, -120.0, 150.0, 121.0)


def pages() -> list[dict[str, object]]:
    return [
        {"text": "あ", "start": 0.0, "end": 0.5},
        {"text": "あい", "start": 0.5, "end": 1.0},
        {"text": "う\nえ\nお", "start": 5.0, "end": 6.0},
        {"text": "か", "start": 9.0, "end": 10.0},
    ]


def test_cues_are_selected_by_time_range_and_line_limit() -> None:
    cues = job.select_cues(pages(), start=Fraction(0), end=Fraction(7), max_lines=2)
    assert [cue.text for cue in cues] == ["あ", "あい"]
    three = job.select_cues(pages(), start=Fraction(0), end=Fraction(7), max_lines=3)
    assert [cue.text for cue in three] == ["あ", "あい", "う\nえ\nお"]


def test_a_range_without_a_matching_page_or_a_bad_limit_is_refused() -> None:
    with pytest.raises(ReplayError, match="No page matches"):
        job.select_cues(pages(), start=Fraction(20), end=Fraction(30), max_lines=2)
    with pytest.raises(ReplayError, match="Invalid cue selection"):
        job.select_cues(pages(), start=Fraction(0), end=Fraction(7), max_lines=4)
    with pytest.raises(ReplayError, match="Invalid cue selection"):
        job.select_cues(pages(), start=Fraction(7), end=Fraction(7), max_lines=2)


def observed() -> list[TrackingPosition]:
    return [
        TrackingPosition(Fraction(n, 30), 540.0, y)
        for n, y in (
            (0, 900.0),
            (20, 880.0),
            (40, 700.0),
            (70, 690.0),
            (100, 800.0),
            (120, 760.0),
            (150, 740.0),
        )
    ]


def test_person_top_uses_observed_samples_after_mapping_output_to_source_time() -> None:
    mapping = job.cut_map(CUTS)
    person_top = job.person_top_from(observed(), mapping, rate=30, maximum_drift=300.0)
    assert person_top(Fraction(0), Fraction(1, 3)) == 880.0
    assert person_top(Fraction(1), Fraction(11, 10)) == 800.0
    assert person_top(Fraction(1), Fraction(2)) == 760.0


def test_a_run_with_no_observation_holds_the_neighbours_or_fails_when_they_differ_too_much() -> (
    None
):
    mapping = job.cut_map(CUTS)
    gap = (Fraction(31, 30), Fraction(32, 30))
    held = job.person_top_from(
        observed()[:5] + observed()[6:], mapping, rate=30, maximum_drift=100.0
    )
    assert held(*gap) == 740.0
    strict = job.person_top_from(
        observed()[:5] + observed()[6:], mapping, rate=30, maximum_drift=10.0
    )
    with pytest.raises(ReplayError, match="differs too much"):
        strict(*gap)


def test_a_run_after_the_last_observation_cannot_be_held() -> None:
    mapping = job.cut_map(CUTS)
    person_top = job.person_top_from(observed()[:5], mapping, rate=30, maximum_drift=300.0)
    with pytest.raises(ReplayError, match="one side"):
        person_top(Fraction(36, 30), Fraction(37, 30))


def envelope(text: str) -> Box:
    return Box(-100.0, -50.0, 100.0, 50.0 * len(text.split("\n")))


def test_runs_that_fit_are_placed_and_runs_that_do_not_are_recorded_as_failures() -> None:
    cues = [
        fixed.RunCue("あ", Fraction(0), Fraction(1, 2)),
        fixed.RunCue("あい", Fraction(1, 2), Fraction(1)),
        fixed.RunCue("う", Fraction(3), Fraction(7, 2)),
    ]

    def person_top(start: Fraction, end: Fraction) -> float:
        if start >= 3:
            raise ReplayError(
                "Observed person top differs too much across an unobserved caption run"
            )
        return 1000.0

    placed, failures = job.plan_runs(
        cues, rate=30, region=Box(0.0, 100.0, 1080.0, 1800.0), center_x=540.0, clearance=20.0,
        envelope=envelope, person_top=person_top,
    )  # fmt: skip
    assert [(item.cue.text, item.run, item.y) for item in placed] == [
        ("あ", 0, 930.0),
        ("あい", 0, 930.0),
    ]
    assert failures == [
        {
            "run": 1,
            "cues": 1,
            "start": 3.0,
            "error": "Observed person top differs too much across an unobserved caption run",
        }
    ]


def test_rebasing_moves_every_time_by_the_same_offset_and_keeps_the_origins() -> None:
    cue = fixed.FixedCue(fixed.RunCue("あ", Fraction(10), Fraction(11)), 0, 540.0, 930.0)
    moved = job.rebased([cue], offset=Fraction(10))
    assert moved[0].cue == fixed.RunCue("あ", Fraction(0), Fraction(1))
    assert (moved[0].run, moved[0].x, moved[0].y) == (0, 540.0, 930.0)


def test_the_candidate_is_bound_to_the_calibration_and_every_input(tmp_path: Path) -> None:
    calibration = tmp_path / "cal.json"
    calibration.write_text("{}", encoding="utf-8")
    reflow = tmp_path / "reflow.json"
    reflow.write_text("[]", encoding="utf-8")
    cue = fixed.FixedCue(fixed.RunCue("あ", Fraction(0), Fraction(1, 3)), 0, 540.0, 930.0)
    document = job.candidate_for(
        [cue], canvas=(1080, 1920), rate=30, calibration=calibration, inputs=[reflow]
    )
    assert document["coordinateSystem"] == "top-left canvas; calibrated text origin"
    inputs = document["inputs"]
    assert isinstance(inputs, dict)
    assert set(inputs) == {str(calibration), str(reflow)}
    assert document["renderCalibration"] == {
        "path": str(calibration),
        "sha256": job.sha256(calibration),
    }
    assert document["candidateOnly"] is True and document["deliveryComplete"] is False


def test_the_observed_series_keeps_only_frames_with_a_face_in_time_order() -> None:
    samples = [
        FaceObservation(Fraction(2, 30), TrackingPosition(Fraction(2, 30), 5.0, 800.0)),
        FaceObservation(Fraction(1, 30), None, "missing"),
        FaceObservation(Fraction(0), TrackingPosition(Fraction(0), 6.0, 810.0)),
        FaceObservation(Fraction(3, 30), None, "ambiguous"),
    ]
    assert job.observed_only(samples) == [
        TrackingPosition(Fraction(0), 6.0, 810.0),
        TrackingPosition(Fraction(2, 30), 5.0, 800.0),
    ]


def test_an_index_without_windows_has_no_observations(tmp_path: Path) -> None:
    index = tmp_path / "index.json"
    index.write_text(json.dumps({"windows": []}), encoding="utf-8")
    assert job.load_observed(index) == []


def test_line_sizes_are_read_from_every_cached_response(tmp_path: Path) -> None:
    for batch, texts in ((0, ["あ", "い"]), (1, ["う"])):
        folder = tmp_path / f"metrics-{batch:04d}"
        folder.mkdir()
        bounds = [{"text": text, "width": 10.0 + batch, "height": 20.0} for text in texts]
        (folder / f"batch-{batch:06d}.response.json").write_text(
            json.dumps({"bounds": bounds}), encoding="utf-8"
        )
    assert job.line_sizes(tmp_path) == {"あ": (10.0, 20.0), "い": (10.0, 20.0), "う": (11.0, 20.0)}


def test_a_paging_directory_without_measurements_is_refused(tmp_path: Path) -> None:
    with pytest.raises(ReplayError, match="holds no cached line measurements"):
        job.line_sizes(tmp_path)


def test_the_command_line_refuses_to_overwrite_an_output(tmp_path: Path) -> None:
    output = tmp_path / "candidate.json"
    output.write_text("x", encoding="utf-8")
    with pytest.raises(ReplayError, match="Refusing to overwrite the output"):
        job.main(
            [str(tmp_path), "w.json", "c.json", "cal.json", str(output),
             "--start-seconds", "0", "--end-seconds", "1", "--region-top", "0",
             "--region-bottom", "1", "--clearance", "0", "--maximum-drift", "0"]
        )  # fmt: skip


def build_inputs(root: Path) -> list[str]:
    paging = root / "paging"
    (paging / "metrics-0000").mkdir(parents=True)
    pages = [
        {"text": "あ", "start": 0.0, "end": 0.5},
        {"text": "あい", "start": 0.5, "end": 1.0},
        {"text": "う", "start": 3.0, "end": 3.5},
    ]
    (paging / "reflow.json").write_text(json.dumps({"captions": pages}), encoding="utf-8")
    bounds = [{"text": t, "width": 100.0, "height": 50.0} for t in ("あ", "あい", "う")]
    (paging / "metrics-0000" / "batch-000000.response.json").write_text(
        json.dumps({"bounds": bounds}), encoding="utf-8"
    )
    cuts = root / "cuts.json"
    cuts.write_text(
        json.dumps(
            {"cuts": [{"output_start_frame": 0, "source_start_frame": 0, "source_end_frame": 300}]}
        ),
        encoding="utf-8",
    )
    calibration = root / "cal.json"
    calibration.write_text(
        json.dumps(
            {
                "canvas": [1080, 1920],
                "padding": 24,
                "nativeFont": {"renderedAscent": 96.0},
                "style": {"motion_line_advance": 121},
            }
        ),
        encoding="utf-8",
    )
    index = root / "index.json"
    index.write_text("{}", encoding="utf-8")
    return [
        str(paging), str(index), str(cuts), str(calibration), str(root / "out.json"),
        "--start-seconds", "0", "--end-seconds", "10", "--region-top", "100",
        "--region-bottom", "1800", "--clearance", "20", "--maximum-drift", "50", "--rebase",
    ]  # fmt: skip


def test_a_whole_range_is_planned_written_and_bound_to_its_inputs(tmp_path: Path) -> None:
    arguments = build_inputs(tmp_path)
    series = [TrackingPosition(Fraction(n, 30), 540.0, 1000.0) for n in range(0, 300, 10)]
    summary = job.run_plan(job.parse(arguments), series)
    assert summary == {"placedCues": 3, "failedRuns": 0}
    document = json.loads((tmp_path / "out.json").read_text())
    assert [c["y"] for c in document["candidates"]] == [1050.0, 1050.0, 1050.0]
    assert document["candidates"][0]["cue"] == {"text": "あ", "start": 0.0, "end": 0.5}
    assert set(document["inputs"]) == {
        str(tmp_path / "cal.json"),
        str(tmp_path / "paging" / "reflow.json"),
        str(tmp_path / "index.json"),
        str(tmp_path / "cuts.json"),
    }
    assert document["plannedRuns"] == {"placedCues": 3, "failedRuns": []}


def test_a_second_plan_into_the_same_output_is_refused(tmp_path: Path) -> None:
    arguments = build_inputs(tmp_path)
    series = [TrackingPosition(Fraction(n, 30), 540.0, 1000.0) for n in range(0, 300, 10)]
    job.run_plan(job.parse(arguments), series)
    with pytest.raises(ReplayError, match="Refusing to overwrite the output"):
        job.run_plan(job.parse(arguments), series)


def track_file(path: Path, rows: list[dict[str, object]], *, rate: int = 30) -> Path:
    path.write_text(
        json.dumps(
            {
                "protocol": "yolo-face-caption-track-v1",
                "provenance": "yolo",
                "rate": rate,
                "frames": rows,
            }
        ),
        encoding="utf-8",
    )
    return path


def test_yolo_samples_are_the_box_centre_and_top_of_single_confident_faces(tmp_path: Path) -> None:
    rows: list[dict[str, object]] = [
        {"frame": 60, "face": [100.0, 800.0, 300.0, 1000.0, 0.9], "faceCount": 1, "captions": 0},
        {"frame": 0, "face": [110.0, 810.0, 310.0, 1010.0, 0.8], "faceCount": 1, "captions": 0},
        {"frame": 30, "face": None, "faceCount": 0, "captions": 0},
        {"frame": 90, "face": [0.0, 1.0, 2.0, 3.0, 0.9], "faceCount": 2, "captions": 0},
        {"frame": 120, "face": [0.0, 1.0, 2.0, 3.0, 0.3], "faceCount": 1, "captions": 0},
    ]
    found = job.load_yolo(track_file(tmp_path / "t.json", rows), rate=30, minimum_confidence=0.5)
    assert found == [
        TrackingPosition(Fraction(0), 210.0, 810.0),
        TrackingPosition(Fraction(2), 200.0, 800.0),
    ]


def test_a_track_of_another_protocol_or_rate_is_refused(tmp_path: Path) -> None:
    other = tmp_path / "other.json"
    other.write_text(json.dumps({"protocol": "x", "provenance": "yolo"}), encoding="utf-8")
    with pytest.raises(ReplayError, match="Not a YOLO face track"):
        job.load_yolo(other, rate=30, minimum_confidence=0.5)
    slow = track_file(tmp_path / "slow.json", [], rate=24)
    with pytest.raises(ReplayError, match="different frame rate"):
        job.load_yolo(slow, rate=30, minimum_confidence=0.5)


def test_yolo_fills_only_the_gaps_between_vision_samples() -> None:
    vision = [TrackingPosition(Fraction(0), 1.0, 10.0), TrackingPosition(Fraction(10), 1.0, 12.0)]
    yolo = [
        TrackingPosition(Fraction(1, 4), 2.0, 20.0),
        TrackingPosition(Fraction(5), 2.0, 21.0),
        TrackingPosition(Fraction(10) - Fraction(1, 5), 2.0, 22.0),
        TrackingPosition(Fraction(20), 2.0, 23.0),
    ]
    merged = job.merge_observed(vision, yolo, gap=Fraction(1, 2))
    assert [(sample.time, sample.y) for sample in merged] == [
        (Fraction(0), 10.0),
        (Fraction(5), 21.0),
        (Fraction(10), 12.0),
        (Fraction(20), 23.0),
    ]


def test_without_vision_every_yolo_sample_is_kept() -> None:
    yolo = [TrackingPosition(Fraction(1), 2.0, 20.0), TrackingPosition(Fraction(2), 2.0, 21.0)]
    assert job.merge_observed([], yolo, gap=Fraction(1, 2)) == yolo


def test_the_command_line_plans_from_a_yolo_track_without_a_vision_index(tmp_path: Path) -> None:
    arguments = build_inputs(tmp_path)
    arguments[1] = "-"
    rows: list[dict[str, object]] = [
        {"frame": n, "face": [100.0, 1000.0, 300.0, 1200.0, 0.9], "faceCount": 1, "captions": 0}
        for n in range(0, 300, 10)
    ]
    track = track_file(tmp_path / "track.json", rows)
    assert job.main([*arguments, "--yolo-track", str(track)]) == 0
    document = json.loads((tmp_path / "out.json").read_text())
    assert [c["y"] for c in document["candidates"]] == [1050.0, 1050.0, 1050.0]
    assert set(document["inputs"]) == {
        str(tmp_path / "cal.json"),
        str(tmp_path / "paging" / "reflow.json"),
        str(tmp_path / "cuts.json"),
        str(track),
    }
