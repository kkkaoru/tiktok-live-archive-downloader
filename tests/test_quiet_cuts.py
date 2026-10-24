import pytest

from replay.caption_projection import CutSpan, MeasuredQuietSpan
from replay.caption_repair import ClockLabel
from replay.quiet_cuts import quiet_cut_frames


def test_quiet_only_inward_rounding_and_output_offsets() -> None:
    assert quiet_cut_frames((MeasuredQuietSpan(1.01, 2.99),), (), source_frames=120) == (
        CutSpan(0, 31, 0),
        CutSpan(89, 120, 31),
    )


def test_point_protection_does_not_change_native_clock() -> None:
    label = ClockLabel("声", 2, 2, "raw")
    assert quiet_cut_frames((MeasuredQuietSpan(1, 3),), (label,), source_frames=120) == (
        CutSpan(0, 30, 0),
        CutSpan(57, 63, 30),
        CutSpan(90, 120, 36),
    )
    assert (label.start, label.end) == (2, 2)


def test_zero_guard_still_protects_point_frame() -> None:
    assert quiet_cut_frames(
        (MeasuredQuietSpan(0, 4),),
        (ClockLabel("声", 2, 2, "raw"),),
        source_frames=120,
        speech_guard_seconds=0,
    ) == (CutSpan(60, 61, 0),)


def test_terminal_point_is_conservatively_retained() -> None:
    assert quiet_cut_frames(
        (MeasuredQuietSpan(0, 4),),
        (ClockLabel("声", 4, 4, "raw"),),
        source_frames=120,
        speech_guard_seconds=0,
    ) == (CutSpan(119, 120, 0),)


def test_overlapping_words_and_quiet_cells_share_protection() -> None:
    assert quiet_cut_frames(
        (MeasuredQuietSpan(0, 1.5), MeasuredQuietSpan(2, 4)),
        (ClockLabel("後", 1.4, 3.2, "b"), ClockLabel("前", 1, 2.1, "a")),
        source_frames=150,
        speech_guard_seconds=0,
    ) == (CutSpan(30, 96, 0), CutSpan(120, 150, 66))


def test_short_quiet_does_not_cut_and_unclassified_sound_stays() -> None:
    assert quiet_cut_frames((MeasuredQuietSpan(1, 1.1),), (), source_frames=120) == (
        CutSpan(0, 120, 0),
    )
    assert quiet_cut_frames((), (), source_frames=120) == (CutSpan(0, 120, 0),)


def test_protection_cursor_moves_past_old_words() -> None:
    assert quiet_cut_frames(
        (MeasuredQuietSpan(3, 4),),
        (ClockLabel("前", 0, 1, "a"), ClockLabel("後", 2, 2.5, "b")),
        source_frames=150,
    ) == (CutSpan(0, 90, 0), CutSpan(120, 150, 90))


def test_multiple_protected_islands_in_one_quiet_run() -> None:
    assert quiet_cut_frames(
        (MeasuredQuietSpan(0, 5),),
        (ClockLabel("前", 1, 2, "a"), ClockLabel("後", 3, 4, "b")),
        source_frames=150,
        speech_guard_seconds=0,
    ) == (CutSpan(30, 60, 0), CutSpan(90, 120, 30))


def test_all_quiet_is_not_an_empty_video() -> None:
    with pytest.raises(ValueError, match="entire"):
        quiet_cut_frames((MeasuredQuietSpan(0, 4),), (), source_frames=120)


@pytest.mark.parametrize(
    "quiet", [(MeasuredQuietSpan(2, 3), MeasuredQuietSpan(1, 2)), (MeasuredQuietSpan(1, 5),)]
)
def test_bad_quiet_clock(quiet: tuple[MeasuredQuietSpan, ...]) -> None:
    with pytest.raises(ValueError, match="ordered"):
        quiet_cut_frames(quiet, (), source_frames=120)


def test_quiet_under_a_frame_past_the_video_end_is_clipped() -> None:
    assert quiet_cut_frames((MeasuredQuietSpan(3, 4.014),), (), source_frames=120) == (
        CutSpan(0, 90, 0),
    )


def test_bad_label_clock() -> None:
    with pytest.raises(ValueError, match="Recognition"):
        quiet_cut_frames((), (ClockLabel("声", 3, 5, "raw"),), source_frames=120)


@pytest.mark.parametrize("frames", [0, True, 30 * 21600 + 1])
def test_bad_frame_budget(frames: int) -> None:
    with pytest.raises(ValueError, match="policy"):
        quiet_cut_frames((), (), source_frames=frames)


@pytest.mark.parametrize(
    "minimum,guard",
    [(float("nan"), 0.08), (0.3, float("nan")), (0.3, -1), (0.3, True), (True, 0.08)],
)
def test_bad_policy(minimum: float, guard: float) -> None:
    with pytest.raises(ValueError, match="policy"):
        quiet_cut_frames(
            (), (), source_frames=120, minimum_cut_seconds=minimum, speech_guard_seconds=guard
        )
