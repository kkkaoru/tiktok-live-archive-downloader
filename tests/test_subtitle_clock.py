"""Validated integer presentation clocks and private native filter lifetime."""

from fractions import Fraction
from pathlib import Path

import pytest

from replay.subtitle_clock import VideoFrameClock, concat_durations


@pytest.mark.parametrize(
    "pts,duration",
    [
        ((), 1),
        ((-1,), 1),
        ((True,), 1),
        ((0, 0), 1),
        ((1, 0), 1),
        ((0, 1.5), 1),
        ((0,), 0),
        ((0,), -1),
        ((0,), True),
        (tuple(range(50_001)), 1),
    ],
)
def test_invalid_presentation_clocks(pts: tuple[int, ...], duration: int) -> None:
    with pytest.raises(ValueError, match="frame clocks"):
        VideoFrameClock(pts, Fraction(1, 90_000), duration)


@pytest.mark.parametrize(
    "base", [Fraction(0), Fraction(-1), Fraction(1, 1_000_000_001), Fraction(86_401)]
)
def test_invalid_presentation_timebase(base: Fraction) -> None:
    with pytest.raises(ValueError, match="time base or extent"):
        VideoFrameClock((0,), base, 1)


def test_terminal_packet_support_preserves_reordered_frames_and_real_tail_gap() -> None:
    clock = VideoFrameClock((0, 3000, 15000), Fraction(1, 90000), 3000)
    clock.verify_terminal_packet_support(
        packet_pts=(0, 15000, 3000), packet_durations=(3000, 3000, 3000)
    )
    assert clock.pts == (0, 3000, 15000)
    assert (clock.pts[-1] + clock.last_duration) * clock.time_base == Fraction(1, 5)


@pytest.mark.parametrize(
    "packet_pts,packet_durations",
    [
        ((), ()),
        ((0, 3000), (3000,)),
        ((0, 0), (3000, 3000)),
        ((0, 6000), (3000, 3000)),
        ((False, 3000), (3000, 3000)),
        ((-1, 3000), (3000, 3000)),
        ((0, 3000), (0, 3000)),
        ((0, 3000), (True, 3000)),
    ],
)
def test_terminal_packet_support_rejects_missing_duplicate_or_invalid_packets(
    packet_pts: tuple[int, ...], packet_durations: tuple[int, ...]
) -> None:
    clock = VideoFrameClock((0, 3000), Fraction(1, 90000), 3000)
    with pytest.raises(ValueError, match="complete decoded tail"):
        clock.verify_terminal_packet_support(
            packet_pts=packet_pts, packet_durations=packet_durations
        )


def test_terminal_packet_support_does_not_substitute_a_shorter_header_duration() -> None:
    clock = VideoFrameClock((0, 3000), Fraction(1, 90000), 2000)
    with pytest.raises(ValueError, match="Terminal decoded duration"):
        clock.verify_terminal_packet_support(packet_pts=(0, 3000), packet_durations=(3000, 3000))


def test_private_balanced_filter_has_exact_local_ticks_and_terminal_extent() -> None:
    clock = VideoFrameClock((270000, 272970, 276030), Fraction(1, 90000), 2970)
    with clock.encoding_options() as options:
        assert options[0] == "-/filter:v"
        path = Path(options[1])
        assert path.stat().st_mode & 0o777 == 0o600
        assert path.read_text(encoding="utf-8") == (
            r"settb=1/90000,setpts=if(lt(N\,1)\,0\,if(lt(N\,2)\,2970\,6030))"
        )
        assert options[2:] == (
            "-fps_mode",
            "passthrough",
            "-enc_time_base",
            "1:90000",
            "-video_track_timescale",
            "90000",
            "-bsf:v",
            "setts=duration='if(eq(N,2),2970,DURATION)'",
        )
    assert not path.exists()


def test_vfr_requests_and_actual_indices_do_not_use_average_fps() -> None:
    clock = VideoFrameClock((90000, 91170, 92970, 630000), Fraction(1, 90000), 2970)
    assert clock.observation_times() == (0.00325, 0.018, 1.52475, 6.00825)
    assert clock.observed_indices((6, 0.033, 0.013, 0, 0)) == (0, 1, 2, 3)
    assert clock.observed_indices((0, 0, 0.033)) == (0, 2)


@pytest.mark.parametrize("time", [-1, -0.0000001, True, float("nan"), float("inf"), 0.008, 0.067])
def test_invalid_actual_presentation_clock(time: float) -> None:
    clock = VideoFrameClock((0, 1170, 2970), Fraction(1, 90000), 2970)
    with pytest.raises(ValueError, match="clock"):
        clock.observed_indices((time,))


def test_missing_actual_observations_are_not_coverage() -> None:
    with pytest.raises(ValueError, match="observation count"):
        VideoFrameClock((0,), Fraction(1, 30), 1).observed_indices(())


def test_ambiguous_serialized_clock_is_not_a_frame_identity() -> None:
    with pytest.raises(ValueError, match="Ambiguous"):
        VideoFrameClock((0, 1), Fraction(1, 1_000_000_000), 1).observed_indices((0,))


def test_concat_rounds_absolute_boundaries_not_each_duration() -> None:
    assert concat_durations((Fraction(1, 30),) * 6, time_base=Fraction(1, 90000)) == (
        "0.033333",
        "0.033334",
        "0.033333",
        "0.033333",
        "0.033334",
        "0.033333",
    )
    assert concat_durations(
        (Fraction(6000, 90000), Fraction(5970, 90000), Fraction(542910, 90000)),
        time_base=Fraction(1, 90000),
    ) == ("0.066667", "0.066333", "6.032333")


@pytest.mark.parametrize("duration", [Fraction(0), Fraction(-1), Fraction(1, 7), 0.5])
def test_concat_rejects_durations_outside_native_grid(duration: Fraction) -> None:
    with pytest.raises(ValueError, match="mux clock grid"):
        concat_durations((duration,), time_base=Fraction(1, 90000))


@pytest.mark.parametrize("base", [Fraction(0), Fraction(-1), Fraction(1, 1000000001), 0.5])
def test_concat_rejects_invalid_time_base(base: Fraction) -> None:
    with pytest.raises(ValueError, match="clock budget"):
        concat_durations((Fraction(1),), time_base=base)


def test_concat_rejects_empty_budget() -> None:
    with pytest.raises(ValueError, match="clock budget"):
        concat_durations((), time_base=Fraction(1, 30))


@pytest.mark.parametrize(
    "duration,base",
    [
        (Fraction(86401), Fraction(1)),
        (Fraction(1, 2000000), Fraction(1, 2000000)),
        (Fraction(1, 1500000), Fraction(1, 1500000)),
    ],
)
def test_concat_rejects_unrepresentable_offsets(duration: Fraction, base: Fraction) -> None:
    with pytest.raises(ValueError, match="cannot preserve"):
        concat_durations((duration,), time_base=base)


def test_numerator_is_preserved_in_filter_ticks() -> None:
    clock = VideoFrameClock((8, 9), Fraction(1001, 30000), 2)
    with clock.encoding_options() as options:
        assert Path(options[1]).read_text(encoding="utf-8") == (
            r"settb=1/30000,setpts=if(lt(N\,1)\,0\,1001)"
        )
        assert options[-1] == "setts=duration='if(eq(N,1),2002,DURATION)'"
