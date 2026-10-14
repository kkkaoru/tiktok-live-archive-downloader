from fractions import Fraction

import pytest

from replay import caption_linguistics, caption_measured_plan
from replay.caption_measured_plan import measured_plan
from replay.caption_timing import CaptionPolicy, SpokenToken
from replay.core import ReplayError
from replay.foreground_geometry import interval_top, place_envelope
from replay.foreground_observations import ForegroundFrame, ForegroundMask
from replay.subtitle_clock import VideoFrameClock
from replay.vertical_captions import Box


@pytest.fixture(autouse=True)
def synthetic_morphology(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        caption_linguistics, "_atomic_ends", lambda text: frozenset(range(1, len(text) + 1))
    )
    monkeypatch.setattr(caption_linguistics, "semantic_break", lambda previous, following: False)
    monkeypatch.setattr(
        caption_linguistics,
        "boundaries",
        lambda text: (
            tuple(range(2, len(text) + 1, 2)),
            tuple(range(1, len(text) + 1)),
        ),
    )


def envelope(text: str) -> Box:
    lines = text.split("\n")
    width = max(len(line) for line in lines) * 10
    height = 12 + (len(lines) - 1) * 14
    return Box(-width / 2, -height / 2, width / 2, height / 2)


def test_real_measured_route_grows_one_two_three_then_resets() -> None:
    def baseline_envelope(text: str) -> Box:
        lines = text.split("\n")
        half_width = max(map(len, lines)) * 5
        return Box(-half_width, -10, half_width, 2 + 14 * (len(lines) - 1))

    result = measured_plan(
        (
            SpokenToken("ab", 0, 0.5, "one"),
            SpokenToken("cd", 0.5, 1, "one"),
            SpokenToken("ef", 1, 1.5, "one"),
            SpokenToken("gh", 1.5, 2, "two"),
        ),
        policy=CaptionPolicy(frame_rate=30, linguistic_boundaries=True, hold_seconds=0),
        region=Box(0, 0, 20, 100),
        center_x=10,
        clearance=4,
        output_support_top=lambda start, end: 80,
        envelope=baseline_envelope,
    )
    assert [(p.cue.text, p.cue.start, p.cue.end, p.x, p.y) for p in result] == [
        ("ab", 0, 0.5, 10, 74),
        ("ab\ncd", 0.5, 1, 10, 60),
        ("ab\ncd\nef", 1, 1.5, 10, 46),
        ("gh", 1.5, 2, 10, 74),
    ]


def test_plan_prepares_phrase_candidates_without_changing_result() -> None:
    prepared: list[tuple[str, ...]] = []
    words = (SpokenToken("abcdefghijkl", 0, 1, "one"),)
    expected = measured_plan(
        words,
        policy=CaptionPolicy(frame_rate=30, linguistic_boundaries=True),
        region=Box(0, 0, 60, 100),
        center_x=30,
        clearance=2,
        output_support_top=lambda start, end: 100,
        envelope=envelope,
    )
    result = measured_plan(
        words,
        policy=CaptionPolicy(frame_rate=30, linguistic_boundaries=True),
        region=Box(0, 0, 60, 100),
        center_x=30,
        clearance=2,
        output_support_top=lambda start, end: 100,
        envelope=envelope,
        prepare_lines=prepared.append,
    )
    assert result == expected
    assert prepared[0] == (
        "ab",
        "cdefghijkl",
        "abcd",
        "efghijkl",
        "abcdef",
        "ghijkl",
        "abcdefgh",
        "ijkl",
        "abcdefghij",
        "kl",
    )


@pytest.fixture
def tokens() -> tuple[SpokenToken, ...]:
    return (
        SpokenToken("aa", 0, 0.15, "a"),
        SpokenToken("bb", 0.2, 0.35, "a"),
        SpokenToken("cc", 0.4, 0.5, "a"),
    )


def test_measured_two_lines_anchor_only_when_displayed(tokens: tuple[SpokenToken, ...]) -> None:
    result = measured_plan(
        tokens,
        policy=CaptionPolicy(frame_rate=30, linguistic_boundaries=True),
        region=Box(0, 0, 50, 200),
        center_x=25,
        clearance=4,
        output_support_top=lambda start, end: 30,
        envelope=envelope,
    )
    assert tuple((p.cue.text, p.cue.start, p.cue.end, p.x, p.y) for p in result) == (
        ("aa", 0, 0.2, 25, 20),
        ("aabb", 0.2, 0.4, 25, 20),
        ("aa\nbbcc", 0.4, 0.58, 25, 13),
    )


def test_prefix_matches_standalone_and_reset_restores_single_line(
    tokens: tuple[SpokenToken, ...],
) -> None:
    def calibrated(text: str) -> Box:
        # Fixed calibrated first baseline; only measured line advance adds height.
        width = max(len(line) for line in text.split("\n")) * 10
        return Box(-width / 2, -8, width / 2, 4 + 14 * text.count("\n"))

    result = measured_plan(
        (*tokens, SpokenToken("dd", 1, 1.1, "b")),
        policy=CaptionPolicy(frame_rate=30, linguistic_boundaries=True),
        region=Box(0, 0, 50, 200),
        center_x=25,
        clearance=4,
        output_support_top=lambda start, end: 30,
        envelope=calibrated,
    )
    standalone = measured_plan(
        tokens[:1],
        policy=CaptionPolicy(frame_rate=30, linguistic_boundaries=True),
        region=Box(0, 0, 50, 200),
        center_x=25,
        clearance=4,
        output_support_top=lambda start, end: 30,
        envelope=calibrated,
    )
    assert result[0].y == standalone[0].y == 22
    assert tuple((p.cue.text, p.x, p.y) for p in result) == (
        ("aa", 25, 22),
        ("aabb", 25, 22),
        ("aa\nbbcc", 25, 8),
        ("dd", 25, 22),
    )
    assert tuple(p.y + calibrated(p.cue.text).bottom for p in result) == (26, 26, 26, 26)


def test_support_changes_inside_growth_follow_each_display_interval(
    tokens: tuple[SpokenToken, ...],
) -> None:
    def top(start: Fraction, end: Fraction) -> float:
        return 30 if start < Fraction(2, 5) and end > Fraction(1, 5) else 100

    result = measured_plan(
        tokens,
        policy=CaptionPolicy(frame_rate=30, linguistic_boundaries=True),
        region=Box(0, 0, 50, 200),
        center_x=25,
        clearance=4,
        output_support_top=top,
        envelope=envelope,
    )
    assert tuple((p.cue.text, p.cue.start, p.cue.end, p.x, p.y) for p in result) == (
        ("aa", 0, 0.2, 25, 90),
        ("aabb", 0.2, 0.4, 25, 20),
        ("aa\nbbcc", 0.4, 0.58, 25, 83),
    )


@pytest.mark.parametrize(
    "bounds,last_baseline,expected_y",
    [(Box(-20, -8, 20, 4), 0, 92), (Box(-20, -8, 20, 18), 14, 78), (Box(-20, -8, 20, 32), 28, 64)],
)
def test_pure_anchor_three_line_contract(
    bounds: Box,
    last_baseline: float,
    expected_y: float,
) -> None:
    # Synthetic measured advance 14; production wrapping remains limited to two lines.
    x, y = place_envelope(
        bounds,
        region=Box(0, 0, 50, 200),
        center_x=25,
        foreground_top=100,
        clearance=4,
    )
    assert x == 25
    assert y == expected_y
    assert y + last_baseline == 92
    assert y + bounds.bottom == 96
    assert y + bounds.top >= 0


def test_all_state_envelope_cannot_overflow_upper_region() -> None:
    with pytest.raises(ReplayError, match="cannot fit above"):
        place_envelope(
            Box(-20, -8, 20, 32),
            region=Box(0, 0, 50, 200),
            center_x=25,
            foreground_top=43,
            clearance=4,
        )


@pytest.mark.parametrize("invalid_top", [float("nan"), float("inf"), -1])
def test_invalid_cue_support_fails(
    tokens: tuple[SpokenToken, ...],
    invalid_top: float,
) -> None:
    def top(start: Fraction, end: Fraction) -> float:
        return 30 if end == Fraction(29, 50) else invalid_top

    with pytest.raises((ValueError, ReplayError)):
        measured_plan(
            tokens,
            policy=CaptionPolicy(frame_rate=30, linguistic_boundaries=True),
            region=Box(0, 0, 50, 200),
            center_x=25,
            clearance=4,
            output_support_top=top,
            envelope=envelope,
        )


def test_shorter_height_switches_at_existing_word_clock(tokens: tuple[SpokenToken, ...]) -> None:
    result = measured_plan(
        tokens,
        policy=CaptionPolicy(frame_rate=30, linguistic_boundaries=True),
        region=Box(0, 0, 50, 200),
        center_x=25,
        clearance=4,
        output_support_top=lambda start, end: 29,
        envelope=envelope,
    )
    assert tuple((p.cue.text, p.cue.start, p.cue.end, p.x, p.y) for p in result) == (
        ("aa", 0, 0.2, 25, 19),
        ("aabb", 0.2, 0.4, 25, 19),
        ("cc", 0.4, 0.58, 25, 19),
    )
    assert tokens == (
        SpokenToken("aa", 0, 0.15, "a"),
        SpokenToken("bb", 0.2, 0.35, "a"),
        SpokenToken("cc", 0.4, 0.5, "a"),
    )


def test_real_pauses_keep_existing_context_reset() -> None:
    result = measured_plan(
        (SpokenToken("aa", 0, 0.1, "a"), SpokenToken("bb", 0.2, 0.3, "a")),
        policy=CaptionPolicy(frame_rate=30, linguistic_boundaries=True),
        region=Box(0, 0, 50, 200),
        center_x=25,
        clearance=4,
        output_support_top=lambda start, end: 30,
        envelope=envelope,
    )
    assert tuple((p.cue.text, p.cue.start, p.cue.end) for p in result) == (
        ("aa", 0, 0.18),
        ("bb", 0.2, 0.38),
    )


def test_fixed_x_follows_higher_support_between_runs() -> None:
    def top(start: Fraction, end: Fraction) -> float:
        return 100 if end < 1 else 30

    result = measured_plan(
        (SpokenToken("aa", 0, 0.1, "a"), SpokenToken("bb", 1, 1.1, "b")),
        policy=CaptionPolicy(frame_rate=30, linguistic_boundaries=True),
        region=Box(0, 0, 50, 200),
        center_x=25,
        clearance=4,
        output_support_top=top,
        envelope=envelope,
    )
    assert tuple((p.x, p.y) for p in result) == ((25, 90), (25, 20))


def test_fixed_x_returns_to_lower_support_between_runs() -> None:
    def top(start: Fraction, end: Fraction) -> float:
        return 30 if start < 1 else 100

    result = measured_plan(
        (SpokenToken("aa", 0, 0.1, "a"), SpokenToken("bb", 1, 1.1, "b")),
        policy=CaptionPolicy(frame_rate=30, linguistic_boundaries=True),
        region=Box(0, 0, 50, 200),
        center_x=25,
        clearance=4,
        output_support_top=top,
        envelope=envelope,
    )
    assert tuple((p.cue.text, p.cue.start, p.x, p.y) for p in result) == (
        ("aa", 0, 25, 20),
        ("bb", 1, 25, 90),
    )
    assert tuple(p.cue.end for p in result) == pytest.approx((0.18, 1.18))


def test_vfr_source_mapping_is_explicit_not_nominal(tokens: tuple[SpokenToken, ...]) -> None:
    clock = VideoFrameClock((100, 101, 107), Fraction(1, 10), 2)
    frames = (
        ForegroundFrame(Fraction(10), (1,), ForegroundMask(10, Box(0, 90, 10, 100))),
        ForegroundFrame(Fraction(101, 10), (1,), ForegroundMask(10, Box(0, 30, 10, 100))),
        ForegroundFrame(Fraction(107, 10), (1,), ForegroundMask(10, Box(0, 90, 10, 100))),
    )
    calls: list[tuple[Fraction, Fraction]] = []

    def top(start: Fraction, end: Fraction) -> float:
        calls.append((start, end))
        return interval_top(frames, clock=clock, start=start + 10, end=end + 10)

    result = measured_plan(
        tokens,
        policy=CaptionPolicy(frame_rate=30, linguistic_boundaries=True),
        region=Box(0, 0, 50, 200),
        center_x=25,
        clearance=4,
        output_support_top=top,
        envelope=envelope,
    )
    assert calls == [
        (Fraction(0), Fraction(29, 50)),
        (Fraction(0), Fraction(1, 5)),
        (Fraction(1, 5), Fraction(2, 5)),
        (Fraction(2, 5), Fraction(29, 50)),
    ]
    assert tuple(p.y for p in result) == (20, 20, 13)


def test_unknown_source_support_does_not_become_layout_reset(
    tokens: tuple[SpokenToken, ...],
) -> None:
    def unknown(start: Fraction, end: Fraction) -> float:
        raise ReplayError("Required source evidence unknown")

    with pytest.raises(ReplayError, match="unknown"):
        measured_plan(
            tokens,
            policy=CaptionPolicy(frame_rate=30, linguistic_boundaries=True),
            region=Box(0, 0, 50, 200),
            center_x=25,
            clearance=4,
            output_support_top=unknown,
            envelope=envelope,
        )


@pytest.mark.parametrize(
    "policy",
    [
        CaptionPolicy(),
        CaptionPolicy(frame_rate=30),
        CaptionPolicy(frame_rate=30, linguistic_boundaries=True, measured_layout=lambda text: text),
    ],
)
def test_requires_explicit_output_policy(
    tokens: tuple[SpokenToken, ...], policy: CaptionPolicy
) -> None:
    with pytest.raises(ValueError, match="output-clock POS policy"):
        measured_plan(
            tokens,
            policy=policy,
            region=Box(0, 0, 50, 200),
            center_x=25,
            clearance=4,
            output_support_top=lambda start, end: 30,
            envelope=envelope,
        )


@pytest.mark.parametrize("top,gap", [(float("nan"), 4), (-1, 4), (30, -1), (30, float("inf"))])
def test_invalid_constraints_fail(tokens: tuple[SpokenToken, ...], top: float, gap: float) -> None:
    with pytest.raises(ValueError, match="region constraints"):
        measured_plan(
            tokens,
            policy=CaptionPolicy(frame_rate=30, linguistic_boundaries=True),
            region=Box(0, 0, 50, 200),
            center_x=25,
            clearance=gap,
            output_support_top=lambda start, end: top,
            envelope=envelope,
        )


def test_coarse_window_refused_before_measurement() -> None:
    with pytest.raises(ValueError, match="duration budget"):
        measured_plan(
            (SpokenToken("aa", 0, 121, "a"),),
            policy=CaptionPolicy(frame_rate=30, linguistic_boundaries=True),
            region=Box(0, 0, 50, 200),
            center_x=25,
            clearance=4,
            output_support_top=lambda start, end: 30,
            envelope=envelope,
        )


def test_indivisible_tall_token_is_not_shrunk() -> None:
    with pytest.raises(ValueError, match="Native token exceeds display budget"):
        measured_plan(
            (SpokenToken("aa", 0, 0.1, "a"),),
            policy=CaptionPolicy(frame_rate=30, linguistic_boundaries=True),
            region=Box(0, 0, 50, 200),
            center_x=25,
            clearance=4,
            output_support_top=lambda start, end: 15,
            envelope=envelope,
        )


def test_envelopes_are_measured_once_per_exact_text(tokens: tuple[SpokenToken, ...]) -> None:
    calls: list[str] = []

    def measure(text: str) -> Box:
        calls.append(text)
        return envelope(text)

    measured_plan(
        tokens,
        policy=CaptionPolicy(frame_rate=30, linguistic_boundaries=True),
        region=Box(0, 0, 50, 200),
        center_x=25,
        clearance=4,
        output_support_top=lambda start, end: 30,
        envelope=measure,
    )
    assert len(calls) == len(set(calls))
    assert "aa\nbbcc" in calls


def test_cache_budget_is_bounded(
    tokens: tuple[SpokenToken, ...], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(caption_measured_plan, "MAX_MEASUREMENTS", 1)
    with pytest.raises(ValueError, match="measurement budget"):
        measured_plan(
            tokens,
            policy=CaptionPolicy(frame_rate=30, linguistic_boundaries=True),
            region=Box(0, 0, 50, 200),
            center_x=25,
            clearance=4,
            output_support_top=lambda start, end: 30,
            envelope=envelope,
        )


def test_contradictory_support_provider_fails(tokens: tuple[SpokenToken, ...]) -> None:
    calls = 0

    def top(start: Fraction, end: Fraction) -> float:
        nonlocal calls
        calls += 1
        return 30 if calls == 1 else 29

    with pytest.raises(ValueError, match="contradicts"):
        measured_plan(
            tokens,
            policy=CaptionPolicy(frame_rate=30, linguistic_boundaries=True),
            region=Box(0, 0, 50, 200),
            center_x=25,
            clearance=4,
            output_support_top=top,
            envelope=envelope,
        )


def test_unretained_output_prefix_fails(
    monkeypatch: pytest.MonkeyPatch, tokens: tuple[SpokenToken, ...]
) -> None:
    monkeypatch.setattr(
        caption_measured_plan, "unretained_frame_prefixes", lambda tokens, policy: (0,)
    )
    with pytest.raises(ValueError, match="unretained"):
        measured_plan(
            tokens,
            policy=CaptionPolicy(frame_rate=30, linguistic_boundaries=True),
            region=Box(0, 0, 50, 200),
            center_x=25,
            clearance=4,
            output_support_top=lambda start, end: 30,
            envelope=envelope,
        )


def test_equal_length_strings_use_actual_width_not_character_count() -> None:
    def proportional(text: str) -> Box:
        lines = text.split("\n")
        width = max(line.count("i") * 3 + line.count("W") * 13 for line in lines)
        height = 12 + 14 * (len(lines) - 1)
        return Box(-width / 2, -height / 2, width / 2, height / 2)

    narrow = measured_plan(
        (SpokenToken("iiii", 0, 0.1, "a"),),
        policy=CaptionPolicy(frame_rate=30, linguistic_boundaries=True),
        region=Box(0, 0, 50, 200),
        center_x=25,
        clearance=4,
        output_support_top=lambda start, end: 16,
        envelope=proportional,
    )
    assert tuple((p.cue.text, p.x, p.y) for p in narrow) == (("iiii", 25, 6),)
    with pytest.raises(ValueError, match="Native token exceeds display budget"):
        measured_plan(
            (SpokenToken("WWWW", 0, 0.1, "a"),),
            policy=CaptionPolicy(frame_rate=30, linguistic_boundaries=True),
            region=Box(0, 0, 50, 200),
            center_x=25,
            clearance=4,
            output_support_top=lambda start, end: 16,
            envelope=proportional,
        )


def test_each_complete_animation_envelope_fits_without_future_union() -> None:
    def asymmetric(text: str) -> Box:
        return Box(-10, -15, 10, 5) if len(text) == 2 else Box(-20, -5, 20, 15)

    result = measured_plan(
        (SpokenToken("aa", 0, 0.15, "a"), SpokenToken("bb", 0.2, 0.3, "a")),
        policy=CaptionPolicy(frame_rate=30, linguistic_boundaries=True),
        region=Box(0, 0, 50, 200),
        center_x=25,
        clearance=4,
        output_support_top=lambda start, end: 29,
        envelope=asymmetric,
    )
    assert tuple((p.cue.text, p.x, p.y) for p in result) == (
        ("aa", 25, 20),
        ("aabb", 25, 10),
    )
    assert tuple(
        (
            p.x + asymmetric(p.cue.text).left,
            p.y + asymmetric(p.cue.text).top,
            p.x + asymmetric(p.cue.text).right,
            p.y + asymmetric(p.cue.text).bottom,
        )
        for p in result
    ) == ((15, 5, 35, 25), (5, 5, 45, 25))


def test_empty_or_excessive_tokens_fail(tokens: tuple[SpokenToken, ...]) -> None:
    with pytest.raises(ValueError, match="bounded tokens"):
        measured_plan(
            (),
            policy=CaptionPolicy(frame_rate=30, linguistic_boundaries=True),
            region=Box(0, 0, 50, 200),
            center_x=25,
            clearance=4,
            output_support_top=lambda start, end: 30,
            envelope=envelope,
        )
    with pytest.raises(ValueError, match="bounded tokens"):
        measured_plan(
            tokens * 4097,
            policy=CaptionPolicy(frame_rate=30, linguistic_boundaries=True),
            region=Box(0, 0, 50, 200),
            center_x=25,
            clearance=4,
            output_support_top=lambda start, end: 30,
            envelope=envelope,
        )
