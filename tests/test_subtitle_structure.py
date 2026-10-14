"""Classical kernel limits with known synthetic line structure, no media fixtures."""

from typing import Literal

import numpy as np
import pytest

from replay.subtitle_inpaint import InpaintSettings, repair_glyphs


@pytest.mark.parametrize("algorithm", ["telea", "navier-stokes"])
def test_visible_endpoints_do_not_prove_line_continuation(
    algorithm: Literal["telea", "navier-stokes"],
) -> None:
    truth = np.full((192, 192, 3), 180, np.uint8)
    truth[:, 94:98] = 30
    ink = np.zeros((192, 192), np.uint8)
    ink[80:112, 80:112] = 255
    source = truth.copy()
    source[ink > 0] = (200, 240, 255)
    repaired, support = repair_glyphs(
        source, settings=InpaintSettings(algorithm=algorithm), text_mask=ink
    )
    # This records a known negative, not an acceptance criterion for restoration.
    assert repaired[96, 96, 0] > 100
    assert truth[96, 96, 0] == 30
    assert not np.any(repaired[support == 0] != source[support == 0])


@pytest.mark.parametrize("algorithm", ["telea", "navier-stokes"])
def test_masked_ink_cannot_leak_through_the_kernel(
    algorithm: Literal["telea", "navier-stokes"],
) -> None:
    source = np.random.default_rng(8).integers(20, 200, (192, 192, 3), dtype=np.uint8)
    ink = np.zeros((192, 192), np.uint8)
    ink[80:112, 80:112] = 255
    original = source.copy()
    first, support = repair_glyphs(
        source, settings=InpaintSettings(algorithm=algorithm), text_mask=ink
    )
    source[support > 0] = (0, 255, 0)
    second, second_support = repair_glyphs(
        source, settings=InpaintSettings(algorithm=algorithm), text_mask=ink
    )
    np.testing.assert_array_equal(first, second)
    np.testing.assert_array_equal(support, second_support)
    assert not np.any(first[support == 0] != original[support == 0])
