from fractions import Fraction

import pytest

from replay.core import ReplayError
from replay.delivery_media import FrameClock
from replay.proxy_clock_correspondence import verify_clocks


def test_quantized_original_preserves_every_frame_and_exact_extent() -> None:
    result = verify_clocks(
        FrameClock((0, 33, 67), Fraction(1, 1000), 33),
        FrameClock((0, 512, 1024), Fraction(1, 15360), 512),
    )
    assert result == {
        "frames": 3,
        "originalTimeBase": "1/1000",
        "proxyTimeBase": "1/15360",
        "frameDuration": "1/30",
        "physicalEnd": "1/10",
        "maximumTimestampDifference": "1/3000",
        "clockIdentical": False,
        "mapping": "same-frame-index-with-source-tick-quantization",
        "pixelIdentityVerified": False,
    }


def test_exact_clock_is_not_pixel_identity() -> None:
    clock = FrameClock((0, 1, 2), Fraction(1, 25), 1)
    fine = FrameClock((0, 40, 80), Fraction(1, 1000), 40)
    assert verify_clocks(fine, clock)["clockIdentical"] is True
    assert verify_clocks(fine, clock)["pixelIdentityVerified"] is False


@pytest.mark.parametrize(
    ("source", "proxy", "message"),
    [
        (
            FrameClock((0, 33), Fraction(1, 1000), 33),
            FrameClock((0, 1, 2), Fraction(1, 30), 1),
            "counts",
        ),
        (
            FrameClock((1, 34), Fraction(1, 1000), 33),
            FrameClock((0, 1), Fraction(1, 30), 1),
            "counts",
        ),
        (
            FrameClock((0,), Fraction(1, 1000), 100),
            FrameClock((0,), Fraction(1, 30), 3),
            "two frames",
        ),
        (
            FrameClock((0, 33, 67), Fraction(1, 1000), 33),
            FrameClock((0, 1, 3), Fraction(1, 30), 1),
            "constant",
        ),
        (
            FrameClock((0, 1, 2), Fraction(1, 30), 1),
            FrameClock((0, 1, 2), Fraction(1, 30), 1),
            "precision",
        ),
        (
            FrameClock((0, 33, 67), Fraction(1, 1000), 33),
            FrameClock((0, 1, 2), Fraction(1, 30), 2),
            "precision",
        ),
        (
            FrameClock((0, 34, 67), Fraction(1, 1000), 33),
            FrameClock((0, 1, 2), Fraction(1, 30), 1),
            "quantization",
        ),
        (
            FrameClock((0, 33, 67), Fraction(1, 1000), 34),
            FrameClock((0, 1, 2), Fraction(1, 30), 1),
            "extents",
        ),
    ],
)
def test_mismatches_are_not_accepted(source: FrameClock, proxy: FrameClock, message: str) -> None:
    with pytest.raises(ReplayError, match=message):
        verify_clocks(source, proxy)
