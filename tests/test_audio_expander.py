"""Synthetic regression tests for per-video expander planning."""

import pytest

from replay.audio_expander import (
    ExpanderPolicy,
    parse_ebur128_frames,
    parse_srt_cues,
    plan_from_texts,
    policy_from_mapping,
    select_threshold,
    split_levels,
    transfer_points,
    verify_expansion,
    verify_from_texts,
)

SRT = """1
00:00:01,000 --> 00:00:04,000
hello

2
00:00:06,000 --> 00:00:09,000
world
"""


def _log(levels: tuple[float, ...]) -> str:
    lines = ["Input #0, wav, from 'x.wav':", "  Duration: 00:00:10.00"]
    lines += [
        f"[Parsed_ebur128_0 @ 0x1] t: {0.1 * (i + 1):.1f} M: {level:.1f} S: -70.0"
        for i, level in enumerate(levels)
    ]
    lines.append("  Summary:")
    return "\n".join(lines) + "\n"


def test_parse_skips_headers_and_reads_frames() -> None:
    frames = parse_ebur128_frames(_log((-20.0, -21.5)).splitlines())
    assert [(round(frame.time_seconds, 1), frame.momentary_lufs) for frame in frames] == [
        (0.1, -20.0),
        (0.2, -21.5),
    ]


def test_parse_rejects_empty_and_regressing_and_wild() -> None:
    with pytest.raises(ValueError, match="No ebur128"):
        parse_ebur128_frames(["nothing here"])
    with pytest.raises(ValueError, match="strictly increasing"):
        parse_ebur128_frames(_log((-20.0, -20.0)).splitlines()[2:4][::-1])
    with pytest.raises(ValueError, match="out of range"):
        parse_ebur128_frames(["[Parsed_ebur128_0 @ 0x1] t: 0.1 M: 99.0"])


def test_parse_srt_intervals() -> None:
    cues = parse_srt_cues(SRT)
    assert [(cue.start_seconds, cue.end_seconds) for cue in cues] == [(1.0, 4.0), (6.0, 9.0)]


def test_parse_srt_rejects_bad_blocks() -> None:
    with pytest.raises(ValueError, match="index"):
        parse_srt_cues("x\n00:00:01,000 --> 00:00:02,000\ntext\n")
    with pytest.raises(ValueError, match="timestamp"):
        parse_srt_cues("1\nnot a range\ntext\n")
    with pytest.raises(ValueError, match="empty or reversed"):
        parse_srt_cues("1\n00:00:02,000 --> 00:00:01,000\ntext\n")
    with pytest.raises(ValueError, match="No SRT cues"):
        parse_srt_cues("lone line\n")
    assert parse_srt_cues("lone line\n\n" + SRT)[0].start_seconds == 1.0


def _masked(t: float) -> bool:
    return (0.5 <= t <= 4.9) or (5.5 <= t <= 9.9)


def test_split_levels_masks_with_padding() -> None:
    levels = tuple(-20.0 if _masked(0.1 * (i + 1)) else -50.0 for i in range(100))
    frames = parse_ebur128_frames(_log(levels).splitlines())
    cues = parse_srt_cues(SRT)
    speech, residual = split_levels(frames, cues, ExpanderPolicy())
    assert min(speech) == -20.0
    assert max(residual) == -50.0
    assert len(speech) + len(residual) == 100


def test_split_levels_requires_both_sides() -> None:
    frames = parse_ebur128_frames(_log(tuple(-20.0 for _ in range(100))).splitlines())
    full = parse_srt_cues("1\n00:00:00,000 --> 00:00:10,000\nall\n")
    with pytest.raises(ValueError, match="no residual"):
        split_levels(frames, full, ExpanderPolicy())
    quiet_frames = parse_ebur128_frames(_log(tuple(-60.0 for _ in range(100))).splitlines())
    distant = parse_srt_cues("1\n01:00:00,000 --> 01:00:01,000\nfar\n")
    with pytest.raises(ValueError, match="covers no loudness"):
        split_levels(quiet_frames, distant, ExpanderPolicy())


def test_select_strongest_safe_threshold() -> None:
    plan = select_threshold(
        tuple(-20.0 for _ in range(9)), tuple(-50.0 for _ in range(9)), ExpanderPolicy()
    )
    assert plan.feasible
    assert plan.threshold_db == pytest.approx(-35.0)
    assert plan.predicted_suppression_db == pytest.approx(15.0)


def test_select_caps_at_ceiling_and_reports_infeasible() -> None:
    capped = select_threshold(
        tuple(-10.0 for _ in range(9)),
        tuple(-50.0 for _ in range(9)),
        ExpanderPolicy(maximum_threshold_above_floor_db=5.0),
    )
    assert capped.feasible and capped.threshold_db == pytest.approx(-45.0)
    stuck = select_threshold(
        tuple(-32.0 for _ in range(9)), tuple(-30.0 for _ in range(9)), ExpanderPolicy()
    )
    assert not stuck.feasible and stuck.threshold_db == 0.0
    with pytest.raises(ValueError, match="both speech and residual"):
        select_threshold((), (-50.0,), ExpanderPolicy())


def test_verify_gates_suppression_and_speech() -> None:
    assert verify_expansion(-20.0, -50.0, -20.2, -60.0, ExpanderPolicy()).passed
    weak = verify_expansion(-20.0, -50.0, -20.0, -51.0, ExpanderPolicy())
    assert not weak.passed and "suppression" in weak.reason
    harsh = verify_expansion(-20.0, -50.0, -25.0, -60.0, ExpanderPolicy())
    assert not harsh.passed and "speech" in harsh.reason
    with pytest.raises(ValueError, match="numbers"):
        verify_expansion(True, -50.0, -20.0, -60.0, ExpanderPolicy())


def test_budget_guards_reject_oversized_inputs(monkeypatch: pytest.MonkeyPatch) -> None:
    import replay.audio_expander as expander

    monkeypatch.setattr(expander, "MAXIMUM_FRAMES", 1)
    with pytest.raises(ValueError, match="exceeds budget"):
        parse_ebur128_frames(_log((-20.0, -21.0)).splitlines())
    monkeypatch.setattr(expander, "MAXIMUM_CUES", 1)
    with pytest.raises(ValueError, match="exceeds budget"):
        parse_srt_cues(SRT)


def test_policy_rejects_bad_values() -> None:
    assert (
        ExpanderPolicy(maximum_threshold_above_floor_db=60.0).maximum_threshold_above_floor_db
        == 60.0
    )
    with pytest.raises(ValueError, match="Invalid expander policy"):
        ExpanderPolicy(maximum_threshold_above_floor_db=81.0)
    with pytest.raises(ValueError, match="Invalid expander policy"):
        ExpanderPolicy(ratio=1.0)
    with pytest.raises(ValueError, match="Invalid expander policy"):
        ExpanderPolicy(maximum_speech_loss_db=99.0)
    with pytest.raises(ValueError, match="Invalid expander policy"):
        ExpanderPolicy(margin_above_floor_db=True)


def test_policy_from_mapping() -> None:
    policy = policy_from_mapping({"ratio": 3.0})
    assert policy.ratio == 3.0 and policy.margin_above_floor_db == 3.0
    with pytest.raises(ValueError, match="Unknown"):
        policy_from_mapping({"perVideoHack": 1.0})
    with pytest.raises(ValueError, match="numbers"):
        policy_from_mapping({"ratio": "loud"})


def test_transfer_points_curve_and_compand_domain() -> None:
    assert transfer_points(-50.2) == "-90/-90|-70.20/-89.90|-50.20/-50.20|0/0"
    # The rollout's first failure: a quiet bed plans T below the old
    # acompressor-era guard yet well inside the compand domain.
    assert transfer_points(-62.5) == "-90/-90|-82.50/-89.90|-62.50/-62.50|0/0"
    for bad in (-70.0, -75.0, 0.0, 3.0, True):
        with pytest.raises(ValueError, match="compand domain|number"):
            transfer_points(bad)
    with pytest.raises(ValueError, match="ratio"):
        transfer_points(-50.0, 1.0)


def test_bridges_end_to_end() -> None:
    levels = tuple(-20.0 if i < 50 else -50.0 for i in range(100))
    log = _log(levels)
    talk = "1\n00:00:00,000 --> 00:00:05,000\ntalk\n"
    plan = plan_from_texts(log, talk)
    assert plan["feasible"] and plan["thresholdDb"] == pytest.approx(-35.0)
    after = _log(tuple(-20.2 if i < 50 else -62.0 for i in range(100)))
    result = verify_from_texts(log, after, talk)
    assert result["passed"] and result["suppressionDb"] == pytest.approx(12.0)
