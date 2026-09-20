"""Synthetic local glyph repair; not proof of naturalness on real footage."""

import numpy as np
import pytest

from replay.subtitle_inpaint import (
    Image,
    InpaintSettings,
    pink_glyph_mask,
    ramp_softness,
    repair_glyphs,
    validate_roi,
)


@pytest.mark.parametrize("expansion", [0, 7])
def test_invalid_expansion(expansion: int) -> None:
    with pytest.raises(ValueError, match="expansion"):
        InpaintSettings(expansion=expansion)


@pytest.mark.parametrize("softness", [-1, 2, float("nan")])
def test_invalid_softness(softness: float) -> None:
    with pytest.raises(ValueError, match="Softness"):
        InpaintSettings(softness=softness)


def test_invalid_channels() -> None:
    with pytest.raises(ValueError, match="BGR"):
        validate_roi(np.zeros((16, 16), dtype=np.uint8))


def test_small_roi() -> None:
    with pytest.raises(ValueError, match="ROI"):
        validate_roi(np.zeros((7, 16, 3), dtype=np.uint8))


def test_no_glyph_preserves_image() -> None:
    image = np.full((32, 32, 3), 100, dtype=np.uint8)
    repaired, mask = repair_glyphs(image, settings=InpaintSettings())
    np.testing.assert_array_equal(repaired, np.full((32, 32, 3), 100, dtype=np.uint8))
    assert np.count_nonzero(mask) == 0
    assert not np.shares_memory(image, repaired)


def glyph_fixture() -> Image:
    image = np.full((48, 48, 3), 100, dtype=np.uint8)
    image[20:25, 20:25] = (180, 50, 220)
    return image


def test_mask_is_glyph_local() -> None:
    mask = pink_glyph_mask(glyph_fixture())
    assert mask[22, 22] == 255
    assert mask[0, 0] == 0
    assert np.count_nonzero(mask) < 200


def test_repairs_color_and_preserves_unmasked_pixels() -> None:
    image = glyph_fixture()
    result, support = repair_glyphs(image, settings=InpaintSettings())
    np.testing.assert_array_equal(result[support == 0], image[support == 0])
    assert int(result[22, 22, 2]) < 130
    assert image[22, 22, 2] == 220


def test_softening_does_not_replace_rectangle() -> None:
    image = glyph_fixture()
    result, support = repair_glyphs(image, settings=InpaintSettings(softness=0.7))
    np.testing.assert_array_equal(result[support == 0], image[support == 0])
    assert np.count_nonzero(support) < 400
    assert result.dtype == np.uint8
    assert int(result[22, 22, 2]) < 130


def test_ramp_starts_at_zero() -> None:
    assert ramp_softness(frame=0, frames=60, target=0.8) == 0


def test_ramp_ends_at_zero() -> None:
    assert ramp_softness(frame=59, frames=60, target=0.8) == 0


def test_ramp_plateau() -> None:
    assert ramp_softness(frame=30, frames=60, target=0.8) == 0.8


def test_ramp_halfway() -> None:
    assert ramp_softness(frame=3, frames=60, target=0.8) == 0.4


@pytest.mark.parametrize("frame,frames", [(-1, 60), (60, 60), (0, 0)])
def test_invalid_clock(frame: int, frames: int) -> None:
    with pytest.raises(ValueError, match="clocks"):
        ramp_softness(frame=frame, frames=frames, target=0.8)


def test_invalid_target() -> None:
    with pytest.raises(ValueError, match="target"):
        ramp_softness(frame=0, frames=60, target=float("nan"))
