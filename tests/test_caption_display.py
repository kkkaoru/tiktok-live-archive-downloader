import pytest

from replay.caption_display import native_interval, owned_captions
from replay.caption_projection import CutSpan
from replay.caption_timing import CaptionCue, SpokenToken


def test_half_open_endpoint() -> None:
    assert native_interval(0, 1, first_frame=0, end_frame=60) == (0, 0.9999833333333333)


def test_empty_numerical_intersection() -> None:
    assert native_interval(10.099999999999909, 10.3, first_frame=0, end_frame=303) is None


def test_large_clock_mask_start_is_not_after_its_native_frame() -> None:
    assert native_interval(
        2603.5333333333333,
        2605.5333333333333,
        first_frame=77870,
        end_frame=78188,
        frame_end_tick=False,
    ) == (7.866666666666666, 9.866666666666667)


def test_adjacent_masks_have_no_tick_hole() -> None:
    left = native_interval(
        2601.5333333333333,
        2603.5333333333333,
        first_frame=77870,
        end_frame=78188,
        frame_end_tick=False,
    )
    right = native_interval(
        2603.5333333333333,
        2605.5333333333333,
        first_frame=77870,
        end_frame=78188,
        frame_end_tick=False,
    )
    assert left == (5.866666666666666, 7.866666666666666)
    assert right == (7.866666666666666, 9.866666666666667)


def test_title_start_is_also_frame_exact_but_retains_endpoint_tick() -> None:
    assert native_interval(
        2603.5333333333333, 2605.5333333333333, first_frame=77870, end_frame=78188
    ) == (7.866666666666666, 9.86665)


def test_unchanged_nonframe_clock() -> None:
    assert native_interval(0.01, 0.02, first_frame=0, end_frame=60) == (0.01, 0.02)


def test_disjoint_interval() -> None:
    assert native_interval(4, 5, first_frame=0, end_frame=60) is None


@pytest.mark.parametrize("start,end", [(float("nan"), 1), (2, 1), (-1, 1)])
def test_invalid_interval(start: float, end: float) -> None:
    with pytest.raises(ValueError, match="Invalid display"):
        native_interval(start, end, first_frame=0, end_frame=60)


def test_invalid_part() -> None:
    with pytest.raises(ValueError, match="render part"):
        native_interval(0, 1, first_frame=2, end_frame=1)


def test_real_subtick_interval_is_not_silently_discarded() -> None:
    with pytest.raises(ValueError, match="tick budget"):
        native_interval(0.01, 0.010001, first_frame=0, end_frame=60)


def test_tiny_nonframe_area_is_not_rounding_at_a_frame_boundary() -> None:
    with pytest.raises(ValueError, match="unrepresentable"):
        native_interval(0.01, 0.01000000001, first_frame=0, end_frame=60)


def test_hold_cannot_bleed_into_next_edit() -> None:
    assert owned_captions(
        (SpokenToken("話", 0.8, 0.99, "a:quiet-cuts-0-0"),), (CutSpan(0, 30, 0),)
    ) == (CaptionCue("話", 0.8, 1.0),)


@pytest.mark.parametrize("owner", ["missing", "a:quiet-cuts-x-0", "a:quiet-cuts-0-2"])
def test_missing_or_invalid_ownership(owner: str) -> None:
    with pytest.raises(ValueError, match="ownership"):
        owned_captions((SpokenToken("話", 0.8, 0.99, owner),), (CutSpan(0, 30, 0),))


def test_word_outside_owned_cut() -> None:
    with pytest.raises(ValueError, match="exceeds"):
        owned_captions((SpokenToken("話", 0.8, 1.01, "a:quiet-cuts-0-0"),), (CutSpan(0, 30, 0),))


def test_punctuation_is_not_a_new_owned_word() -> None:
    assert owned_captions(
        (
            SpokenToken("話", 0.7, 0.8, "a:quiet-cuts-0-0"),
            SpokenToken("。", 0.8, 0.9, "a:quiet-cuts-0-0"),
        ),
        (CutSpan(0, 30, 0),),
    ) == (CaptionCue("話。", 0.7, 0.88),)
