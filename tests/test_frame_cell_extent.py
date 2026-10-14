from fractions import Fraction

import pytest

from replay.core import ReplayError
from replay.frame_cell_extent import (
    terminal_cell_end,
    terminal_cell_receipt,
    validate_terminal_cell,
)


def test_explicit_last_partial_cell_keeps_physical_end() -> None:
    assert (
        validate_terminal_cell(
            start=Fraction(59, 30),
            end=Fraction(2),
            physical_end=Fraction(197, 100),
            frame_duration=Fraction(1, 30),
            allow_terminal_cell=True,
        )
        is True
    )
    assert (
        validate_terminal_cell(
            start=Fraction(0),
            end=Fraction(1),
            physical_end=Fraction(197, 100),
            frame_duration=Fraction(1, 30),
        )
        is False
    )
    assert terminal_cell_receipt(
        physical_end=Fraction(197, 100), frame_duration=Fraction(1, 30)
    ) == {
        "policy": "last-valid-display-cell",
        "physicalEnd": "197/100",
        "displayGridEnd": "2",
        "derivedDisplayTail": "3/100",
        "frameDuration": "1/30",
        "observedCoverageExtended": False,
    }


@pytest.mark.parametrize(
    "start,end,physical,frame,enabled",
    [
        (Fraction(0), Fraction(2), Fraction(197, 100), Fraction(1, 30), False),
        (Fraction(0), Fraction(61, 30), Fraction(197, 100), Fraction(1, 30), True),
        (Fraction(2), Fraction(61, 30), Fraction(197, 100), Fraction(1, 30), True),
        (Fraction(0), Fraction(199, 100), Fraction(197, 100), Fraction(1, 30), True),
        (Fraction(0), Fraction(0), Fraction(1), Fraction(1, 30), True),
        (Fraction(0), Fraction(1), Fraction(0), Fraction(1, 30), True),
        (Fraction(0), Fraction(1), Fraction(1), Fraction(0), True),
    ],
)
def test_no_arbitrary_extension(
    start: Fraction, end: Fraction, physical: Fraction, frame: Fraction, enabled: bool
) -> None:
    with pytest.raises(ReplayError):
        validate_terminal_cell(
            start=start,
            end=end,
            physical_end=physical,
            frame_duration=frame,
            allow_terminal_cell=enabled,
        )


def test_exact_grid_needs_no_tail() -> None:
    assert terminal_cell_end(physical_end=Fraction(2), frame_duration=Fraction(1, 30)) == 2
