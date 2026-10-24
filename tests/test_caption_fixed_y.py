"""A caption run keeps one fixed origin above the person's worst-case top edge."""

from fractions import Fraction

import pytest

from replay import caption_fixed_y as fixed
from replay.caption_fixed_y import RunCue
from replay.core import ReplayError
from replay.fcp_tracking_caption import TrackingPosition
from replay.vertical_captions import Box

REGION = Box(0.0, 100.0, 1080.0, 1800.0)


def cue(text: str, start: str, end: str) -> RunCue:
    return RunCue(text, Fraction(start), Fraction(end))


def envelope(text: str) -> Box:
    lines = len(text.split("\n"))
    return Box(-300.0, -100.0 * lines, 300.0, 10.0)


def person(top: float) -> fixed.PersonTop:
    return lambda start, end: top


def test_a_growing_page_that_continues_without_a_gap_is_one_run() -> None:
    cues = [cue("あ", "0", "1/3"), cue("あい", "1/3", "2/3"), cue("あい\nう", "2/3", "1")]
    assert fixed.split_runs(cues, rate=30) == ((0, 1, 2),)


def test_a_new_text_or_a_gap_starts_a_new_run() -> None:
    cues = [cue("あい", "0", "1/3"), cue("う", "1/3", "2/3"), cue("うえ", "1", "4/3")]
    assert fixed.split_runs(cues, rate=30) == ((0,), (1,), (2,))


def test_a_shorter_or_equal_page_is_not_an_extension() -> None:
    cues = [cue("あいう", "0", "1/3"), cue("あい", "1/3", "2/3"), cue("あい", "2/3", "1")]
    assert fixed.split_runs(cues, rate=30) == ((0,), (1,), (2,))


def test_split_runs_rejects_unordered_cues_and_bad_input() -> None:
    with pytest.raises(ReplayError, match="ordered"):
        fixed.split_runs([cue("あ", "1", "2"), cue("あい", "0", "1/2")], rate=30)
    with pytest.raises(ReplayError, match="positive duration"):
        fixed.split_runs([cue("あ", "0", "1"), cue("あい", "1", "1")], rate=30)
    with pytest.raises(ReplayError, match="Invalid caption run input"):
        fixed.split_runs([], rate=30)


def test_every_cue_of_a_run_gets_the_same_origin() -> None:
    cues = [cue("あ", "0", "1/3"), cue("あい", "1/3", "2/3"), cue("あい\nう", "2/3", "1")]
    result = fixed.fixed_origins(
        cues,
        rate=30,
        region=REGION,
        center_x=540.0,
        clearance=20.0,
        envelope=envelope,
        person_top=person(1000.0),
    )
    assert [item.y for item in result] == [970.0, 970.0, 970.0]
    assert [item.run for item in result] == [0, 0, 0]
    assert [item.x for item in result] == [540.0, 540.0, 540.0]


def test_separate_runs_may_use_different_origins_but_each_stays_fixed() -> None:
    cues = [cue("あ", "0", "1/3"), cue("あい", "1/3", "2/3"), cue("う", "1", "4/3")]

    def top(start: Fraction, end: Fraction) -> float:
        return 1000.0 if start < 1 else 800.0

    result = fixed.fixed_origins(
        cues,
        rate=30,
        region=REGION,
        center_x=540.0,
        clearance=20.0,
        envelope=envelope,
        person_top=top,
    )
    assert [item.y for item in result] == [970.0, 970.0, 770.0]


def test_a_run_that_does_not_fit_above_the_person_is_an_error() -> None:
    cues = [cue("あ\nい\nう", "0", "1/3")]
    with pytest.raises(ReplayError, match="cannot fit above foreground support"):
        fixed.fixed_origins(
            cues,
            rate=30,
            region=REGION,
            center_x=540.0,
            clearance=20.0,
            envelope=envelope,
            person_top=person(150.0),
        )


def test_worst_person_top_uses_the_samples_bracketing_the_interval() -> None:
    series = [
        TrackingPosition(Fraction(0), 540.0, 900.0),
        TrackingPosition(Fraction(1), 540.0, 700.0),
        TrackingPosition(Fraction(2), 540.0, 950.0),
        TrackingPosition(Fraction(3), 540.0, 400.0),
    ]
    assert fixed.worst_person_top(series, Fraction(1, 2), Fraction(3, 2)) == 700.0
    assert fixed.worst_person_top(series, Fraction(2), Fraction(5, 2)) == 400.0
    assert fixed.worst_person_top(series, Fraction(1), Fraction(2)) == 700.0


def test_worst_person_top_requires_coverage_of_the_whole_interval() -> None:
    series = [
        TrackingPosition(Fraction(1), 540.0, 900.0),
        TrackingPosition(Fraction(2), 540.0, 900.0),
    ]
    with pytest.raises(ReplayError, match="does not cover"):
        fixed.worst_person_top(series, Fraction(1, 2), Fraction(3, 2))
    with pytest.raises(ReplayError, match="does not cover"):
        fixed.worst_person_top(series, Fraction(3, 2), Fraction(5, 2))
    with pytest.raises(ReplayError, match="does not cover"):
        fixed.worst_person_top([], Fraction(0), Fraction(1))


def test_worst_person_top_rejects_a_non_finite_sample() -> None:
    series = [
        TrackingPosition(Fraction(0), 540.0, float("nan")),
        TrackingPosition(Fraction(2), 540.0, 900.0),
    ]
    with pytest.raises(ReplayError, match="Non-finite"):
        fixed.worst_person_top(series, Fraction(1, 2), Fraction(1))


def test_candidate_document_is_candidate_only_with_all_approvals_false() -> None:
    cues = [cue("あ", "0", "1/3")]
    result = fixed.fixed_origins(
        cues,
        rate=30,
        region=REGION,
        center_x=540.0,
        clearance=20.0,
        envelope=envelope,
        person_top=person(1000.0),
    )
    document = fixed.candidate_document(
        result,
        canvas=(1080, 1920),
        rate=30,
        coordinate_system="top-left canvas; calibrated text origin",
        inputs={"/cal.json": "1" * 64},
        calibration=("/cal.json", "1" * 64),
    )
    assert document["candidateOnly"] is True
    assert document["renderCalibration"] == {"path": "/cal.json", "sha256": "1" * 64}
    assert document["canvas"] == [1080, 1920]
    assert document["policy"] == {"frame_rate": 30}
    assert document["candidates"] == [
        {"cue": {"text": "あ", "start": 0.0, "end": 1 / 3}, "x": 540.0, "y": 970.0}
    ]
    assert document["motionRenderVerified"] is False
    assert document["deliveryComplete"] is False


def observed() -> list[TrackingPosition]:
    return [
        TrackingPosition(Fraction(0), 540.0, 900.0),
        TrackingPosition(Fraction(1), 540.0, 880.0),
        TrackingPosition(Fraction(5), 540.0, 860.0),
        TrackingPosition(Fraction(6), 540.0, 700.0),
    ]


def test_an_observed_run_uses_its_own_worst_case_top() -> None:
    estimate = fixed.person_top_with_hold(
        observed(), Fraction(0), Fraction(3, 2), maximum_drift=50.0
    )
    assert estimate == fixed.PersonTopEstimate(880.0, False)


def test_an_unobserved_run_holds_the_nearest_observations_and_says_so() -> None:
    estimate = fixed.person_top_with_hold(observed(), Fraction(2), Fraction(3), maximum_drift=50.0)
    assert estimate == fixed.PersonTopEstimate(860.0, True)


def test_an_unobserved_run_fails_when_the_person_moved_too_much() -> None:
    with pytest.raises(ReplayError, match="differs too much"):
        fixed.person_top_with_hold(observed(), Fraction(2), Fraction(3), maximum_drift=10.0)


def test_an_unobserved_run_fails_without_an_observation_on_both_sides() -> None:
    with pytest.raises(ReplayError, match="one side"):
        fixed.person_top_with_hold(observed(), Fraction(7), Fraction(8), maximum_drift=50.0)
    with pytest.raises(ReplayError, match="one side"):
        fixed.person_top_with_hold(
            observed()[1:], Fraction(1, 2), Fraction(3, 4), maximum_drift=50.0
        )


def test_person_top_with_hold_rejects_invalid_requests() -> None:
    with pytest.raises(ReplayError, match="Invalid person-top hold request"):
        fixed.person_top_with_hold(observed(), Fraction(2), Fraction(1), maximum_drift=50.0)
    with pytest.raises(ReplayError, match="Invalid person-top hold request"):
        fixed.person_top_with_hold(observed(), Fraction(1), Fraction(2), maximum_drift=-1.0)


def test_a_calibration_missing_from_or_different_in_the_inputs_is_refused() -> None:
    cues = [cue("あ", "0", "1/3")]
    result = fixed.fixed_origins(
        cues, rate=30, region=REGION, center_x=540.0, clearance=20.0,
        envelope=envelope, person_top=person(1000.0),
    )  # fmt: skip
    with pytest.raises(ReplayError, match="bound among the document inputs"):
        fixed.candidate_document(
            result, canvas=(1080, 1920), rate=30, coordinate_system="c",
            inputs={}, calibration=("/cal.json", "1" * 64),
        )  # fmt: skip
    with pytest.raises(ReplayError, match="bound among the document inputs"):
        fixed.candidate_document(
            result, canvas=(1080, 1920), rate=30, coordinate_system="c",
            inputs={"/cal.json": "2" * 64}, calibration=("/cal.json", "1" * 64),
        )  # fmt: skip


def ties() -> list[TrackingPosition]:
    """Two samples share t=1 and two share t=3: ties at both interval edges."""
    return [
        TrackingPosition(Fraction(0), 540.0, 900.0),
        TrackingPosition(Fraction(1), 540.0, 800.0),
        TrackingPosition(Fraction(1), 540.0, 600.0),
        TrackingPosition(Fraction(2), 540.0, 850.0),
        TrackingPosition(Fraction(3), 540.0, 500.0),
        TrackingPosition(Fraction(3), 540.0, 950.0),
        TrackingPosition(Fraction(5), 540.0, 990.0),
    ]


def test_samples_tied_at_the_interval_edges_count_inside() -> None:
    series = ties()
    times = [point.time for point in series]
    assert fixed.person_top_with_hold(
        series, Fraction(1), Fraction(3), maximum_drift=0.0, times=times
    ) == fixed.PersonTopEstimate(500.0, False)
    assert fixed.person_top_with_hold(
        series, Fraction(3, 2), Fraction(3), maximum_drift=0.0
    ) == fixed.PersonTopEstimate(500.0, False)
    assert fixed.person_top_with_hold(
        series, Fraction(1), Fraction(3, 2), maximum_drift=0.0, times=times
    ) == fixed.PersonTopEstimate(600.0, False)


def test_an_empty_inside_holds_the_last_tied_sample_before_and_the_first_after() -> None:
    series = ties()
    times = [point.time for point in series]
    # Nothing in [3.5, 4.5]: the last of the t=3 pair (950) and t=5 (990).
    assert fixed.person_top_with_hold(
        series, Fraction(7, 2), Fraction(9, 2), maximum_drift=50.0, times=times
    ) == fixed.PersonTopEstimate(950.0, True)
    # Nothing in [1.5, 1.75]: the last of the t=1 pair (600) and t=2 (850).
    assert fixed.person_top_with_hold(
        series, Fraction(3, 2), Fraction(7, 4), maximum_drift=300.0, times=times
    ) == fixed.PersonTopEstimate(600.0, True)


def test_no_sample_before_or_after_an_empty_inside_fails() -> None:
    series = ties()
    times = [point.time for point in series]
    with pytest.raises(ReplayError, match="one side"):
        fixed.person_top_with_hold(
            series, Fraction(-2), Fraction(-1), maximum_drift=50.0, times=times
        )
    with pytest.raises(ReplayError, match="one side"):
        fixed.person_top_with_hold(series, Fraction(6), Fraction(7), maximum_drift=50.0)


def test_the_worst_top_brackets_with_ties_at_both_edges() -> None:
    series = ties()
    times = [point.time for point in series]
    # The first sample at or after end=3 is the first of the t=3 pair; the last at or before
    # start=1 is the second of the t=1 pair.
    assert fixed.worst_person_top(series, Fraction(1), Fraction(3), times=times) == 500.0
    assert fixed.worst_person_top(series, Fraction(1), Fraction(2)) == 600.0
    assert fixed.worst_person_top(series, Fraction(7, 2), Fraction(4), times=times) == 950.0
