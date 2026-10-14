from dataclasses import replace
from fractions import Fraction
from pathlib import Path

import pytest

from replay import caption_linguistics
from replay import caption_source_support as module
from replay.caption_measured_plan import measured_plan
from replay.caption_source_support import OutputSourceSupport, RetainedSourceSpan
from replay.caption_timing import CaptionPolicy, SpokenToken
from replay.core import ReplayError
from replay.foreground_collection import ForegroundJob
from replay.foreground_evidence import ForegroundWindow
from replay.foreground_observations import ForegroundFrame, ForegroundMask
from replay.subtitle_clock import VideoFrameClock
from replay.vertical_captions import Box


@pytest.fixture
def window(tmp_path: Path) -> ForegroundWindow:
    job = ForegroundJob(
        tmp_path / "config",
        "a" * 64,
        tmp_path / "source",
        tmp_path / "native",
        "b" * 64,
        "c" * 64,
        VideoFrameClock((100, 101, 107), Fraction(1, 10), 2),
        1080,
        1920,
    )
    return ForegroundWindow(
        job,
        (
            ForegroundFrame(Fraction(10), (1,), ForegroundMask(10, Box(0, 900, 20, 1000))),
            ForegroundFrame(Fraction(101, 10), (1,), ForegroundMask(10, Box(0, 700, 20, 1000))),
            ForegroundFrame(Fraction(107, 10), (1,), ForegroundMask(10, Box(0, 800, 20, 1000))),
        ),
        "d" * 64,
    )


@pytest.fixture
def second(window: ForegroundWindow) -> ForegroundWindow:
    return replace(
        window,
        job=replace(window.job, clock=VideoFrameClock((109,), Fraction(1, 10), 1)),
        frames=(
            ForegroundFrame(Fraction(109, 10), (1,), ForegroundMask(10, Box(0, 600, 20, 1000))),
        ),
    )


def test_absolute_vfr_source_times_are_not_output_frame_ordinals(window: ForegroundWindow) -> None:
    support = OutputSourceSupport(
        (window,), (RetainedSourceSpan(Fraction(10), Fraction(0), Fraction(9, 10)),)
    )
    assert support(Fraction(0), Fraction(1, 10)) == 900
    assert support(Fraction(1, 20), Fraction(7, 10)) == 700
    assert support(Fraction(7, 10), Fraction(9, 10)) == 800


def test_removed_unknown_source_interval_is_not_queried(window: ForegroundWindow) -> None:
    unknown = replace(
        window,
        frames=(window.frames[0], ForegroundFrame(Fraction(101, 10), None, None), window.frames[2]),
    )
    support = OutputSourceSupport(
        (unknown,),
        (
            RetainedSourceSpan(Fraction(10), Fraction(0), Fraction(1, 10)),
            RetainedSourceSpan(Fraction(107, 10), Fraction(1, 10), Fraction(1, 5)),
        ),
    )
    assert support(Fraction(0), Fraction(3, 10)) == 800
    assert support(Fraction(0), Fraction(1, 10)) == 900
    assert support(Fraction(1, 10), Fraction(3, 10)) == 800
    retained = OutputSourceSupport(
        (unknown,), (RetainedSourceSpan(Fraction(10), Fraction(0), Fraction(9, 10)),)
    )
    with pytest.raises(ReplayError, match="unknown"):
        retained(Fraction(0), Fraction(9, 10))


def test_adjacent_evidence_windows_and_nonzero_output_origin(
    window: ForegroundWindow,
    second: ForegroundWindow,
) -> None:
    support = OutputSourceSupport(
        (window, second), (RetainedSourceSpan(Fraction(107, 10), Fraction(5), Fraction(3, 10)),)
    )
    assert support(Fraction(5), Fraction(26, 5)) == 800
    assert support(Fraction(26, 5), Fraction(53, 10)) == 600
    assert support(Fraction(5), Fraction(53, 10)) == 600


def test_gaps_in_evidence_remain_unknown_unless_removed(
    window: ForegroundWindow,
    second: ForegroundWindow,
) -> None:
    later = replace(
        second,
        job=replace(second.job, clock=VideoFrameClock((110,), Fraction(1, 10), 1)),
        frames=(replace(second.frames[0], time=Fraction(11)),),
    )
    support = OutputSourceSupport(
        (window, later), (RetainedSourceSpan(Fraction(107, 10), Fraction(0), Fraction(2, 5)),)
    )
    with pytest.raises(ValueError, match="lacks complete"):
        support(Fraction(0), Fraction(2, 5))
    cuts = OutputSourceSupport(
        (window, later),
        (
            RetainedSourceSpan(Fraction(107, 10), Fraction(0), Fraction(1, 5)),
            RetainedSourceSpan(Fraction(11), Fraction(1, 5), Fraction(1, 10)),
        ),
    )
    assert cuts(Fraction(0), Fraction(3, 10)) == 600
    assert cuts(Fraction(1, 5), Fraction(3, 10)) == 600


@pytest.mark.parametrize("source", [Fraction(9), Fraction(109, 10)])
def test_source_coverage_outside_available_windows_fails(
    window: ForegroundWindow,
    source: Fraction,
) -> None:
    support = OutputSourceSupport(
        (window,), (RetainedSourceSpan(source, Fraction(0), Fraction(1, 10)),)
    )
    with pytest.raises(ValueError, match="lacks complete"):
        support(Fraction(0), Fraction(1, 10))


@pytest.mark.parametrize(
    "start,end",
    [
        (Fraction(-1), Fraction(0)),
        (Fraction(0), Fraction(1)),
        (Fraction(1, 10), Fraction(1, 10)),
        (Fraction(1, 10), Fraction(0)),
        (Fraction(1, 1_000_000_001), Fraction(1, 2)),
    ],
)
def test_invalid_output_intervals_fail(
    window: ForegroundWindow,
    start: Fraction,
    end: Fraction,
) -> None:
    support = OutputSourceSupport(
        (window,), (RetainedSourceSpan(Fraction(10), Fraction(0), Fraction(9, 10)),)
    )
    with pytest.raises(ValueError, match="outside"):
        support(start, end)


@pytest.mark.parametrize(
    "source,output,duration",
    [
        (Fraction(-1), Fraction(0), Fraction(1)),
        (Fraction(86400), Fraction(0), Fraction(1)),
        (Fraction(10), Fraction(0), Fraction(0)),
        (Fraction(10), Fraction(0), Fraction(121)),
        (Fraction(10), Fraction(1, 1_000_000_001), Fraction(1)),
    ],
)
def test_invalid_retained_span_fields_fail(
    source: Fraction,
    output: Fraction,
    duration: Fraction,
) -> None:
    with pytest.raises(ValueError, match="bounded exact rational"):
        RetainedSourceSpan(source, output, duration)


def test_corrupt_record_cannot_pass_rational_validation() -> None:
    span = RetainedSourceSpan(Fraction(10), Fraction(0), Fraction(1))
    object.__setattr__(span, "source_start", 0.0)
    with pytest.raises(ValueError, match="bounded exact rational"):
        span.__post_init__()


@pytest.mark.parametrize("field", ["source_hash", "native_hash", "width"])
def test_conflicting_window_bindings_fail(
    window: ForegroundWindow,
    second: ForegroundWindow,
    field: str,
) -> None:
    jobs = {
        "source_hash": replace(second.job, source_hash="e" * 64),
        "native_hash": replace(second.job, native_hash="f" * 64),
        "width": replace(second.job, width=720),
    }
    changed = replace(second, job=jobs[field])
    with pytest.raises(ValueError, match="conflicting identities"):
        OutputSourceSupport(
            (window, changed), (RetainedSourceSpan(Fraction(10), Fraction(0), Fraction(1)),)
        )


def test_overlapping_evidence_is_not_silently_selected(window: ForegroundWindow) -> None:
    with pytest.raises(ValueError, match="overlap"):
        OutputSourceSupport(
            (window, window), (RetainedSourceSpan(Fraction(10), Fraction(0), Fraction(1)),)
        )


@pytest.mark.parametrize(
    "source,output",
    [
        (Fraction(21, 2), Fraction(1)),
        (Fraction(11), Fraction(2)),
        (Fraction(11), Fraction(1, 2)),
    ],
)
def test_reordered_source_or_noncontiguous_output_fails(
    window: ForegroundWindow,
    source: Fraction,
    output: Fraction,
) -> None:
    with pytest.raises(ValueError, match="source-ordered and output-contiguous"):
        OutputSourceSupport(
            (window,),
            (
                RetainedSourceSpan(Fraction(10), Fraction(0), Fraction(1)),
                RetainedSourceSpan(source, output, Fraction(1)),
            ),
        )


def test_structural_budgets(window: ForegroundWindow, monkeypatch: pytest.MonkeyPatch) -> None:
    cuts = (RetainedSourceSpan(Fraction(10), Fraction(0), Fraction(1)),)
    with pytest.raises(ValueError, match="required"):
        OutputSourceSupport((), cuts)
    with pytest.raises(ValueError, match="required"):
        OutputSourceSupport((window,), ())
    with pytest.raises(ValueError, match="duration budget"):
        OutputSourceSupport(
            (window,),
            (
                RetainedSourceSpan(Fraction(10), Fraction(0), Fraction(60)),
                RetainedSourceSpan(Fraction(70), Fraction(60), Fraction(61)),
            ),
        )
    monkeypatch.setattr(module, "MAX_WINDOWS", 0)
    with pytest.raises(ValueError, match="required"):
        OutputSourceSupport((window,), cuts)


def test_clock_map_drives_measured_planner_without_retiming(
    window: ForegroundWindow,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(caption_linguistics, "_atomic_ends", lambda text: frozenset({len(text)}))
    monkeypatch.setattr(caption_linguistics, "semantic_break", lambda previous, following: False)
    monkeypatch.setattr(
        caption_linguistics, "boundaries", lambda text: ((len(text),), (len(text),))
    )
    support = OutputSourceSupport(
        (window,), (RetainedSourceSpan(Fraction(107, 10), Fraction(2), Fraction(1, 5)),)
    )
    result = measured_plan(
        (SpokenToken("aa", 2, 2.1, "a"),),
        policy=CaptionPolicy(frame_rate=30, linguistic_boundaries=True),
        region=Box(0, 0, 200, 1000),
        center_x=100,
        clearance=4,
        output_support_top=support,
        envelope=lambda text: Box(-12, -10, 12, 2),
    )
    assert tuple((p.cue.start, p.cue.end, p.cue.text, p.x, p.y) for p in result) == (
        (2, 2.18, "aa", 100, 794),
    )
