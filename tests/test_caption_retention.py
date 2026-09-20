import pytest

from replay.caption_boundaries import AudioSpan
from replay.caption_projection import CutSpan, MeasuredQuietSpan
from replay.caption_retention import RetentionRequest, retain_nonquiet_support


def test_extra_support_is_explicit_not_a_changed_core() -> None:
    result = retain_nonquiet_support(
        (CutSpan(30, 60, 0),), (AudioSpan(0.5, 1.5),), (), source_frames=90
    )
    assert result.cuts == (CutSpan(15, 60, 0),)
    assert result.requests == (RetentionRequest(0, 0.5, 1, 15, 30),)
    assert result.added_frames == 15


def test_independent_quiet_does_not_need_retention() -> None:
    result = retain_nonquiet_support(
        (CutSpan(30, 60, 0),),
        (AudioSpan(0.5, 1.5),),
        (MeasuredQuietSpan(0.5, 1),),
        source_frames=90,
    )
    assert result.cuts == (CutSpan(30, 60, 0),)
    assert result.requests == ()
    assert result.added_frames == 0


def test_only_measured_quiet_is_removed_from_missing_support() -> None:
    result = retain_nonquiet_support(
        (CutSpan(30, 60, 0),),
        (AudioSpan(0.5, 1.5),),
        (MeasuredQuietSpan(0.7, 0.9),),
        source_frames=90,
    )
    assert result.cuts == (CutSpan(15, 21, 0), CutSpan(27, 60, 6))
    assert result.added_frames == 9


def test_rounding_preserves_weak_subframe_onset() -> None:
    result = retain_nonquiet_support((), (AudioSpan(0.001, 0.011),), (), source_frames=30)
    assert result.cuts == (CutSpan(0, 1, 0),)
    assert result.requests == (RetentionRequest(0, 0.001, 0.011, 0, 1),)


def test_overlapping_original_and_new_supports_are_not_double_counted() -> None:
    result = retain_nonquiet_support(
        (), (AudioSpan(0, 1), AudioSpan(0.5, 1.5)), (), source_frames=60
    )
    assert result.cuts == (CutSpan(0, 45, 0),)
    assert result.added_frames == 45
    assert len(result.requests) == 2


def test_empty_supports_do_not_imply_silence() -> None:
    result = retain_nonquiet_support((CutSpan(0, 30, 0),), (), (), source_frames=30)
    assert result.cuts == (CutSpan(0, 30, 0),)


@pytest.mark.parametrize("frames,fps", [(True, 30), (0, 30), (30, 0), (30, 121), (648001, 30)])
def test_invalid_clock(frames: int, fps: int) -> None:
    with pytest.raises(ValueError, match="budget"):
        retain_nonquiet_support((), (), (), source_frames=frames, fps=fps)


@pytest.mark.parametrize(
    "cuts", [(CutSpan(0, 31, 0),), (CutSpan(0, 10, 1),), (CutSpan(0, 10, 0), CutSpan(9, 20, 10))]
)
def test_invalid_base(cuts: tuple[CutSpan, ...]) -> None:
    with pytest.raises(ValueError, match="base cut"):
        retain_nonquiet_support(cuts, (), (), source_frames=30)


def test_support_source_overrun() -> None:
    with pytest.raises(ValueError, match="exceeds source"):
        retain_nonquiet_support((), (AudioSpan(0, 1.1),), (), source_frames=30)


def test_quiet_source_overrun() -> None:
    with pytest.raises(ValueError, match="exceeds source"):
        retain_nonquiet_support((), (), (MeasuredQuietSpan(0, 1.1),), source_frames=30)


def test_support_budget() -> None:
    with pytest.raises(ValueError, match="budget"):
        retain_nonquiet_support((), (AudioSpan(0, 1),) * 100001, (), source_frames=30)


def test_request_budget() -> None:
    with pytest.raises(ValueError, match="request budget"):
        retain_nonquiet_support(
            (), (AudioSpan(0, 1),) * 100000, (MeasuredQuietSpan(0.4, 0.6),), source_frames=30
        )


def test_final_cut_budget() -> None:
    supports = tuple(map(lambda index: AudioSpan(index * 2, index * 2 + 0.1), range(3001)))
    with pytest.raises(ValueError, match="native cut budget"):
        retain_nonquiet_support((), supports, (), source_frames=180060)
