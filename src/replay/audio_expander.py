"""Per-video downward-expander planning from loudness evidence and transcript mask.

A residual music bed left by vocal separation sits below speech level. This
module derives, for one video at a time, the strongest downward-expansion
threshold that keeps speech untouched: the non-speech floor and the speech
reference come from the video's own ebur128 momentary loudness tagged by its
transcript cues. Anything that cannot be proven safe is reported infeasible
instead of guessed.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from statistics import median
from typing import TypedDict


def _stamp(prefix: str) -> str:
    return (
        rf"(?P<{prefix}h>[0-9]+):(?P<{prefix}m>[0-5][0-9]):"
        rf"(?P<{prefix}s>[0-5][0-9]),(?P<{prefix}ms>[0-9]{{3}})"
    )


_FRAME = re.compile(r"t:\s+(?P<time>[0-9]+\.[0-9]+)\b.*\bM:\s+(?P<momentary>-?[0-9]+\.[0-9]+)\b")
_RANGE = re.compile(_stamp("start_") + r"\s+-->\s+" + _stamp("end_"))

MAXIMUM_FRAMES = 1_100_000
MAXIMUM_CUES = 100_000


@dataclass(frozen=True)
class LoudnessFrame:
    """One ebur128 momentary measurement covering the previous 400 ms."""

    time_seconds: float
    momentary_lufs: float


@dataclass(frozen=True)
class SpeechCue:
    """One transcript cue interval; overlapping cues are allowed."""

    start_seconds: float
    end_seconds: float


@dataclass(frozen=True)
class ExpanderPolicy:
    """Generic guardrails; identical for every video, never per-ID tuned."""

    ratio: float = 2.0
    margin_above_floor_db: float = 3.0
    minimum_speech_clearance_db: float = 9.0
    maximum_threshold_above_floor_db: float = 15.0
    minimum_suppression_db: float = 3.0
    maximum_speech_loss_db: float = 1.5
    speech_padding_seconds: float = 0.5

    def __post_init__(self) -> None:
        values = (
            self.ratio,
            self.margin_above_floor_db,
            self.minimum_speech_clearance_db,
            self.maximum_threshold_above_floor_db,
            self.minimum_suppression_db,
            self.maximum_speech_loss_db,
            self.speech_padding_seconds,
        )
        if (
            any(isinstance(value, bool) for value in values)
            or not all(isinstance(value, (int, float)) for value in values)
            or not 1.0 < self.ratio <= 20.0
            or not 0.0 <= self.margin_above_floor_db <= 30.0
            or not 0.0 < self.minimum_speech_clearance_db <= 40.0
            or not 0.0 < self.maximum_threshold_above_floor_db <= 80.0
            or not 0.0 < self.minimum_suppression_db <= 40.0
            or not 0.0 <= self.maximum_speech_loss_db <= 6.0
            or not 0.0 <= self.speech_padding_seconds <= 5.0
        ):
            raise ValueError("Invalid expander policy")


@dataclass(frozen=True)
class ExpanderPlan:
    """Strongest feasible threshold, or infeasible with the blocking reason."""

    feasible: bool
    threshold_db: float
    floor_db: float
    speech_db: float
    predicted_suppression_db: float
    reason: str


@dataclass(frozen=True)
class ExpansionVerification:
    """Re-measured proof that residual dropped and speech survived."""

    passed: bool
    suppression_db: float
    speech_loss_db: float
    reason: str


class ExpanderPlanDict(TypedDict):
    feasible: bool
    thresholdDb: float
    floorDb: float
    speechDb: float
    predictedSuppressionDb: float
    reason: str


class ExpansionVerificationDict(TypedDict):
    passed: bool
    suppressionDb: float
    speechLossDb: float
    reason: str


def parse_ebur128_frames(lines: Iterable[str]) -> tuple[LoudnessFrame, ...]:
    """Extract strictly increasing momentary frames; skip non-frame log lines."""
    frames: list[LoudnessFrame] = []
    for line in lines:
        match = _FRAME.search(line[:320] if len(line) > 320 else line)
        if match is None:
            continue
        time = float(match["time"])
        level = float(match["momentary"])
        if not time >= 0.0 or not -130.0 <= level <= 10.0:
            raise ValueError("ebur128 frame measurement out of range")
        if frames and time <= frames[-1].time_seconds:
            raise ValueError("ebur128 frame clock is not strictly increasing")
        frames.append(LoudnessFrame(time, level))
    if not frames:
        raise ValueError("No ebur128 momentary frames found")
    if len(frames) > MAXIMUM_FRAMES:
        raise ValueError("ebur128 frame count exceeds budget")
    return tuple(frames)


def _seconds(groups: Mapping[str, str], prefix: str) -> float:
    return (
        int(groups[prefix + "h"]) * 3600.0
        + int(groups[prefix + "m"]) * 60.0
        + int(groups[prefix + "s"])
        + int(groups[prefix + "ms"]) / 1000.0
    )


def parse_srt_cues(text: str) -> tuple[SpeechCue, ...]:
    """Parse cue intervals; cue text is irrelevant to the speech mask."""
    cues: list[SpeechCue] = []
    for block in text.split("\n\n"):
        rows = [row.strip() for row in block.strip().splitlines() if row.strip()]
        if len(rows) < 2:
            continue
        if not rows[0].isdigit():
            raise ValueError("SRT cue index is invalid")
        match = _RANGE.fullmatch(rows[1][:128])
        if match is None:
            raise ValueError("SRT cue timestamp range is invalid")
        start = _seconds(match.groupdict(), "start_")
        end = _seconds(match.groupdict(), "end_")
        if not end > start:
            raise ValueError("SRT cue interval is empty or reversed")
        cues.append(SpeechCue(start, end))
    if not cues:
        raise ValueError("No SRT cues found")
    if len(cues) > MAXIMUM_CUES:
        raise ValueError("SRT cue count exceeds budget")
    return tuple(cues)


def split_levels(
    frames: tuple[LoudnessFrame, ...],
    cues: tuple[SpeechCue, ...],
    policy: ExpanderPolicy,
) -> tuple[tuple[float, ...], tuple[float, ...]]:
    """Split momentary levels into speech and non-speech evidence.

    Each frame covers the 400 ms before its timestamp, widened by the policy
    padding so attack/release smear never mislabels speech as residual.
    """
    speech: list[float] = []
    residual: list[float] = []
    pad = policy.speech_padding_seconds
    for frame in frames:
        window_start = frame.time_seconds - 0.4
        is_speech = any(
            cue.start_seconds - pad <= frame.time_seconds and cue.end_seconds + pad >= window_start
            for cue in cues
        )
        (speech if is_speech else residual).append(frame.momentary_lufs)
    if not speech:
        raise ValueError("Transcript mask covers no loudness frames")
    if not residual:
        raise ValueError("Transcript mask leaves no residual evidence")
    return tuple(speech), tuple(residual)


def select_threshold(
    speech_levels: tuple[float, ...],
    residual_levels: tuple[float, ...],
    policy: ExpanderPolicy,
) -> ExpanderPlan:
    """Take the strongest threshold that keeps every guardrail satisfied."""
    if not speech_levels or not residual_levels:
        raise ValueError("Expander selection needs both speech and residual evidence")
    speech = median(speech_levels)
    floor = median(residual_levels)
    lowest = floor + policy.margin_above_floor_db
    highest = min(
        speech - policy.minimum_speech_clearance_db,
        floor + policy.maximum_threshold_above_floor_db,
    )
    if lowest > highest:
        return ExpanderPlan(
            feasible=False,
            threshold_db=0.0,
            floor_db=floor,
            speech_db=speech,
            predicted_suppression_db=0.0,
            reason="No threshold clears both the residual margin and the speech clearance",
        )
    predicted = (highest - floor) * (policy.ratio - 1.0)
    return ExpanderPlan(
        feasible=True,
        threshold_db=highest,
        floor_db=floor,
        speech_db=speech,
        predicted_suppression_db=predicted,
        reason="Strongest threshold satisfying every guardrail",
    )


def verify_expansion(
    before_speech: float,
    before_floor: float,
    after_speech: float,
    after_floor: float,
    policy: ExpanderPolicy,
) -> ExpansionVerification:
    """Require measured residual suppression with negligible speech loss."""
    for value in (before_speech, before_floor, after_speech, after_floor):
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError("Expansion verification levels must be numbers")
    suppression = before_floor - after_floor
    loss = before_speech - after_speech
    if suppression < policy.minimum_suppression_db:
        return ExpansionVerification(
            passed=False,
            suppression_db=suppression,
            speech_loss_db=loss,
            reason="Measured residual suppression is below the minimum",
        )
    if loss > policy.maximum_speech_loss_db:
        return ExpansionVerification(
            passed=False,
            suppression_db=suppression,
            speech_loss_db=loss,
            reason="Measured speech loss exceeds the maximum",
        )
    return ExpansionVerification(
        passed=True,
        suppression_db=suppression,
        speech_loss_db=loss,
        reason="Residual suppressed with speech preserved",
    )


def plan_from_texts(
    ebur128_log: str,
    srt_text: str,
    policy: ExpanderPolicy | None = None,
) -> ExpanderPlanDict:
    """Shell bridge: loudness log plus transcript in, JSON-ready plan out."""
    active = policy if policy is not None else ExpanderPolicy()
    frames = parse_ebur128_frames(ebur128_log.splitlines())
    cues = parse_srt_cues(srt_text)
    speech_levels, residual_levels = split_levels(frames, cues, active)
    plan = select_threshold(speech_levels, residual_levels, active)
    return ExpanderPlanDict(
        feasible=plan.feasible,
        thresholdDb=plan.threshold_db,
        floorDb=plan.floor_db,
        speechDb=plan.speech_db,
        predictedSuppressionDb=plan.predicted_suppression_db,
        reason=plan.reason,
    )


def verify_from_texts(
    before_log: str,
    after_log: str,
    srt_text: str,
    policy: ExpanderPolicy | None = None,
) -> ExpansionVerificationDict:
    """Shell bridge: re-measured proof from the before/after loudness logs."""
    active = policy if policy is not None else ExpanderPolicy()
    cues = parse_srt_cues(srt_text)

    def medians(log: str) -> tuple[float, float]:
        speech_levels, residual_levels = split_levels(
            parse_ebur128_frames(log.splitlines()), cues, active
        )
        return median(speech_levels), median(residual_levels)

    before_speech, before_floor = medians(before_log)
    after_speech, after_floor = medians(after_log)
    result = verify_expansion(before_speech, before_floor, after_speech, after_floor, active)
    return ExpansionVerificationDict(
        passed=result.passed,
        suppressionDb=result.suppression_db,
        speechLossDb=result.speech_loss_db,
        reason=result.reason,
    )


def transfer_points(threshold_db: float, ratio: float = 2.0) -> str:
    """Render a compand downward-expansion curve through (T, T).

    Unity above T so speech passes untouched, ratio slope below T, and a
    silent gate one gate span down. The compand level domain needs every
    point above -90 dB, so T must clear a full gate span below zero; deeper
    thresholds fail loudly instead of emitting a collapsed curve.
    """
    if not isinstance(threshold_db, (int, float)) or isinstance(threshold_db, bool):
        raise ValueError("Expander threshold must be a number")
    if not -70.0 < threshold_db < 0.0:
        raise ValueError("Expander threshold outside the compand domain (-70, 0) dB")
    if not isinstance(ratio, (int, float)) or isinstance(ratio, bool) or ratio <= 1.0:
        raise ValueError("Expander ratio must exceed 1.0")
    gate_db = threshold_db - 20.0
    gate_out = max(threshold_db + ratio * (gate_db - threshold_db), -89.9)
    return f"-90/-90|{gate_db:.2f}/{gate_out:.2f}|{threshold_db:.2f}/{threshold_db:.2f}|0/0"


def policy_from_mapping(values: Mapping[str, object]) -> ExpanderPolicy:
    """Build a policy from caller overrides; unknown keys are rejected."""
    names = (
        "ratio",
        "margin_above_floor_db",
        "minimum_speech_clearance_db",
        "maximum_threshold_above_floor_db",
        "minimum_suppression_db",
        "maximum_speech_loss_db",
        "speech_padding_seconds",
    )
    unknown = set(values) - set(names)
    if unknown:
        raise ValueError("Unknown expander policy keys")
    selected: dict[str, float] = {}
    for name in names:
        if name in values:
            value = values[name]
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError("Expander policy overrides must be numbers")
            selected[name] = float(value)
    return ExpanderPolicy(**selected)
