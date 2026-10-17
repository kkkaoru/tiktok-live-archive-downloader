"""Region filling is conservative: unions only, explicit gaps, no invented text."""

import pytest

from replay import subtitle_region_fill as fill


def test_union_is_the_bounding_rectangle() -> None:
    assert fill.union([(0, 10, 50, 30), (20, 5, 60, 20)]) == (0, 5, 60, 30)


def test_union_rejects_no_regions() -> None:
    with pytest.raises(ValueError, match="zero"):
        fill.union([])


def test_widen_unites_neighbours_and_keeps_empty_frames_empty() -> None:
    regions = [None, (0, 0, 10, 10), None, None, (20, 0, 30, 10), None, None, None]
    assert fill.widen(regions, 1) == (
        (0, 0, 10, 10),
        (0, 0, 10, 10),
        (0, 0, 10, 10),
        (20, 0, 30, 10),
        (20, 0, 30, 10),
        (20, 0, 30, 10),
        None,
        None,
    )


def test_widen_rejects_a_negative_radius() -> None:
    with pytest.raises(ValueError, match="negative"):
        fill.widen([None], -1)


def test_bracket_unites_previous_and_next_sample() -> None:
    sampled = {0: (0, 0, 10, 10), 3: (5, 0, 20, 10), 4: None}
    assert fill.bracket(5, sampled) == (
        (0, 0, 10, 10),
        (0, 0, 20, 10),
        (0, 0, 20, 10),
        (5, 0, 20, 10),
        None,
    )


def test_bracket_requires_first_and_last_samples() -> None:
    with pytest.raises(ValueError, match="first and last"):
        fill.bracket(4, {1: None, 3: None})


def test_combine_unites_both_sources_per_frame() -> None:
    sampled = [(0, 0, 10, 10), None, None]
    detected = [(5, 5, 20, 20), (1, 1, 2, 2), None]
    assert fill.combine(sampled, detected) == ((0, 0, 20, 20), (1, 1, 2, 2), None)


def test_combine_rejects_length_mismatch() -> None:
    with pytest.raises(ValueError, match="different"):
        fill.combine([None], [])


def test_coverage_gaps_report_uncontained_required_frames() -> None:
    required = [(0, 0, 10, 10), (0, 0, 10, 10), None, (0, 0, 10, 10)]
    filled = [(0, 0, 20, 20), (0, 0, 5, 10), None, None]
    assert fill.coverage_gaps(required, filled) == (1, 3)


def test_coverage_gaps_reject_length_mismatch() -> None:
    with pytest.raises(ValueError, match="different"):
        fill.coverage_gaps([None], [])
