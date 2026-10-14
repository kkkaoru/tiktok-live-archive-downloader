"""Bounded hysteresis for derived caption positions, never raw face observations."""

from dataclasses import dataclass
from fractions import Fraction
from math import isfinite

from replay.core import ReplayError


@dataclass
class CaptionYStability:
    """Suppress small reversals without delaying motion beyond a spatial bound.

    The deadband is in the caller's coordinate units. There is no frame-count
    window or velocity limit: real movement follows immediately within deadband.
    Reset between source cuts or unsupported gaps; retain across caption changes.
    """

    deadband: float
    time: Fraction | None = None
    value: float | None = None

    def __post_init__(self) -> None:
        if not isfinite(self.deadband) or self.deadband < 0:
            raise ReplayError("Caption deadband must be finite and nonnegative")

    def sample(self, time: Fraction, value: float) -> float:
        if time < 0 or not isfinite(value):
            raise ReplayError("Caption position requires finite Y and a nonnegative clock")
        if self.time is not None and time < self.time:
            raise ReplayError("Caption stabilization clock must not move backwards")
        previous = self.value
        result = (
            value
            if previous is None
            else min(max(previous, value - self.deadband), value + self.deadband)
        )
        self.time, self.value = time, result
        return result
