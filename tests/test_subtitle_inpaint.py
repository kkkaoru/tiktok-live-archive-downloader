"""Synthetic local glyph repair; not proof of naturalness on real footage."""

from typing import Literal
from unittest.mock import patch

import numpy as np
import pytest

from replay.subtitle_inpaint import (
    Image,
    InpaintSettings,
    pink_glyph_mask,
    ramp_softness,
    registered_glyph_samples,
    repair_glyphs,
    unchanged_outside_support,
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


def test_white_highlights_without_pink_cannot_seed_repair() -> None:
    image = np.full((32, 32, 3), 255, dtype=np.uint8)
    result, mask = repair_glyphs(image, settings=InpaintSettings())
    assert np.count_nonzero(mask) == 0
    np.testing.assert_array_equal(result, np.full((32, 32, 3), 255, dtype=np.uint8))


@pytest.mark.parametrize("value", [0, None, "yes"])
def test_invalid_enclosed_white_option(value: bool) -> None:
    with pytest.raises(ValueError, match="boolean"):
        InpaintSettings(fill_enclosed_white=value)


@pytest.mark.parametrize("algorithm", ["unknown", None, True])
def test_invalid_inpainting_algorithm(algorithm: Literal["telea", "navier-stokes"]) -> None:
    with pytest.raises(ValueError, match="algorithm"):
        InpaintSettings(algorithm=algorithm)


def test_navier_stokes_is_explicit_and_preserves_glyph_support() -> None:
    image = glyph_fixture()
    original = image.copy()
    telea, original_support = repair_glyphs(image, settings=InpaintSettings())
    result, support = repair_glyphs(image, settings=InpaintSettings(algorithm="navier-stokes"))
    np.testing.assert_array_equal(image, original)
    np.testing.assert_array_equal(support, original_support)
    np.testing.assert_array_equal(result[support == 0], image[support == 0])
    assert int(result[22, 22, 2]) < 130
    assert telea.dtype == np.uint8
    assert result.dtype == np.uint8


def test_enclosed_white_fill_is_opt_in_and_preserves_input() -> None:
    image = np.full((96, 96, 3), 100, dtype=np.uint8)
    image[20:76, 20:76] = (180, 50, 220)
    image[25:71, 25:71] = 240
    before = image.copy()
    legacy = pink_glyph_mask(image)
    candidate = pink_glyph_mask(image, fill_enclosed_white=True)
    assert legacy[48, 48] == 0
    assert candidate[48, 48] == 255
    assert candidate[0, 0] == 0
    np.testing.assert_array_equal(image, before)


def test_enclosed_fill_does_not_seed_standalone_white() -> None:
    image = np.full((48, 48, 3), 240, dtype=np.uint8)
    result, mask = repair_glyphs(image, settings=InpaintSettings(fill_enclosed_white=True))
    assert np.count_nonzero(mask) == 0
    np.testing.assert_array_equal(result, np.full((48, 48, 3), 240, dtype=np.uint8))


def test_enclosed_fill_preserves_dark_letter_counter() -> None:
    image = np.full((96, 96, 3), 100, dtype=np.uint8)
    image[20:76, 20:76] = (180, 50, 220)
    image[25:71, 25:71] = 240
    image[38:58, 38:58] = 30
    mask = pink_glyph_mask(image, fill_enclosed_white=True)
    assert mask[48, 48] == 0
    assert mask[30, 48] == 255


def test_enclosed_fill_does_not_flood_an_open_outline() -> None:
    image = np.full((96, 96, 3), 100, dtype=np.uint8)
    image[20:76, 20:76] = (180, 50, 220)
    image[25:71, 25:71] = 240
    image[20:35, 38:58] = 240
    mask = pink_glyph_mask(image, fill_enclosed_white=True)
    assert mask[48, 48] == 0


def test_enclosed_fill_repair_remains_within_glyph_support() -> None:
    image = np.full((96, 96, 3), 100, dtype=np.uint8)
    image[20:76, 20:76] = (180, 50, 220)
    image[25:71, 25:71] = 240
    result, support = repair_glyphs(image, settings=InpaintSettings(fill_enclosed_white=True))
    assert support[48, 48] == 255
    assert int(result[48, 48, 2]) < 130
    np.testing.assert_array_equal(result[support == 0], image[support == 0])


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


@pytest.mark.parametrize("channel", [0, 1, 2])
def test_exact_native_comparison_detects_one_byte_outside_support(channel: int) -> None:
    source = np.zeros((32, 48, 3), dtype=np.uint8)[::2, ::2]
    result = source.copy()
    support = np.zeros((32, 48), dtype=np.uint8)[::2, ::2]
    support[0, 0] = 127
    result[0, 0] = 255
    source.flags.writeable = False
    assert unchanged_outside_support(source, result, support)
    result[-1, -1, channel] = 1
    assert not unchanged_outside_support(source, result, support)


@pytest.mark.parametrize("mask_value", [0, 1, 255])
def test_native_comparison_matches_empty_and_full_support(mask_value: int) -> None:
    source = np.zeros((16, 16, 3), dtype=np.uint8)
    result = np.full((16, 16, 3), 255, dtype=np.uint8)
    support = np.full((16, 16), mask_value, dtype=np.uint8)
    assert unchanged_outside_support(source, result, support) == bool(
        np.array_equal(source[support == 0], result[support == 0])
    )


@pytest.mark.parametrize("shape", [(8, 16), (16, 16, 1)])
def test_native_comparison_rejects_wrong_support_shape(shape: tuple[int, ...]) -> None:
    source = np.zeros((16, 16, 3), dtype=np.uint8)
    with pytest.raises(ValueError, match="support comparison"):
        unchanged_outside_support(source, source, np.zeros(shape, dtype=np.uint8))


def test_native_comparison_rejects_mismatched_images() -> None:
    with pytest.raises(ValueError, match="support comparison"):
        unchanged_outside_support(
            np.zeros((16, 16, 3), dtype=np.uint8),
            np.zeros((8, 16, 3), dtype=np.uint8),
            np.zeros((16, 16), dtype=np.uint8),
        )


def test_native_copy_matches_legacy_gather_on_strided_input() -> None:
    gray = np.random.default_rng(42).integers(0, 256, (96, 96, 1), dtype=np.uint8)
    source = np.repeat(gray, 3, axis=2)[::2, ::2]
    source[20:25, 20:25] = (180, 50, 220)
    before = source.copy()

    def legacy_copy(image: Image, mask: Image, destination: Image) -> Image:
        destination[mask != 0] = image[mask != 0]
        return destination

    with patch("replay.subtitle_inpaint.cv2.copyTo", side_effect=legacy_copy):
        legacy, legacy_support = repair_glyphs(source, settings=InpaintSettings(softness=0.7))
    native, native_support = repair_glyphs(source, settings=InpaintSettings(softness=0.7))
    np.testing.assert_array_equal(native, legacy)
    np.testing.assert_array_equal(native_support, legacy_support)
    np.testing.assert_array_equal(source, before)
    assert np.count_nonzero(native_support == 0) > 0


def test_ramp_continues_across_processing_boundaries() -> None:
    assert ramp_softness(frame=0, frames=10, target=0.8, start_frame=10, total_frames=30) == 0.8
    assert ramp_softness(frame=9, frames=10, target=0.8, start_frame=10, total_frames=30) == 0.8
    assert ramp_softness(frame=9, frames=10, target=0.8, start_frame=20, total_frames=30) == 0


@pytest.mark.parametrize("start,total", [(-1, 30), (1, None), (25, 30), (True, 30), (0, True)])
def test_invalid_global_ramp_clock(start: int, total: int | None) -> None:
    with pytest.raises(ValueError, match="clocks"):
        ramp_softness(frame=0, frames=10, target=0.8, start_frame=start, total_frames=total)


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


def test_registered_glyph_translation_preserves_source_and_unselected_pixels() -> None:
    source = np.full((16, 16, 3), 20, dtype=np.uint8)
    donor = np.full((16, 16, 3), 80, dtype=np.uint8)
    donor[5, 7] = (10, 30, 50)
    mask = np.zeros((16, 16), dtype=np.uint8)
    mask[5, 5] = 255
    forward = np.zeros((16, 16, 2), dtype=np.float32)
    forward[:, :, 0] = 2
    source.flags.writeable = False
    result, accepted = registered_glyph_samples(
        source,
        donor,
        source_mask=mask,
        donor_mask=np.zeros_like(mask),
        forward=forward,
        backward=-forward,
    )
    assert result[5, 5].tolist() == [10, 30, 50]
    assert np.count_nonzero(accepted) == 1
    np.testing.assert_array_equal(source, np.full((16, 16, 3), 20, dtype=np.uint8))
    np.testing.assert_array_equal(result[accepted == 0], source[accepted == 0])


def test_registered_glyph_rejects_fractional_donor_mask_contamination() -> None:
    source = np.full((16, 16, 3), 20, dtype=np.uint8)
    donor = np.full_like(source, 80)
    mask = np.zeros((16, 16), dtype=np.uint8)
    mask[5, 5] = 255
    donor_mask = np.zeros_like(mask)
    donor_mask[5, 6] = 1
    forward = np.zeros((16, 16, 2), dtype=np.float32)
    forward[:, :, 0] = 0.5
    result, accepted = registered_glyph_samples(
        source,
        donor,
        source_mask=mask,
        donor_mask=donor_mask,
        forward=forward,
        backward=-forward,
    )
    assert np.count_nonzero(accepted) == 0
    np.testing.assert_array_equal(result, source)


def test_registered_glyph_rejects_inconsistent_round_trip() -> None:
    source = np.zeros((16, 16, 3), dtype=np.uint8)
    flow = np.ones((16, 16, 2), dtype=np.float32)
    result, accepted = registered_glyph_samples(
        source,
        np.full_like(source, 80),
        source_mask=np.full((16, 16), 255, dtype=np.uint8),
        donor_mask=np.zeros((16, 16), dtype=np.uint8),
        forward=flow,
        backward=flow,
    )
    assert np.count_nonzero(accepted) == 0
    np.testing.assert_array_equal(result, source)


def test_registered_glyph_rejects_outside_interpolation_footprint() -> None:
    source = np.zeros((16, 16, 3), dtype=np.uint8)
    mask = np.zeros((16, 16), dtype=np.uint8)
    mask[:, 0] = 255
    flow = np.zeros((16, 16, 2), dtype=np.float32)
    flow[:, :, 0] = -0.5
    result, accepted = registered_glyph_samples(
        source,
        np.full_like(source, 80),
        source_mask=mask,
        donor_mask=np.zeros_like(mask),
        forward=flow,
        backward=-flow,
    )
    assert np.count_nonzero(accepted) == 0
    np.testing.assert_array_equal(result, source)


@pytest.mark.parametrize("value", [float("nan"), float("inf"), 17.0])
def test_registered_glyph_rejects_invalid_flow_values(value: float) -> None:
    source = np.zeros((16, 16, 3), dtype=np.uint8)
    mask = np.zeros((16, 16), dtype=np.uint8)
    flow = np.full((16, 16, 2), value, dtype=np.float32)
    with pytest.raises(ValueError, match="flows"):
        registered_glyph_samples(
            source, source, source_mask=mask, donor_mask=mask, forward=flow, backward=flow
        )


def test_registered_glyph_rejects_flow_shape() -> None:
    source = np.zeros((16, 16, 3), dtype=np.uint8)
    mask = np.zeros((16, 16), dtype=np.uint8)
    flow = np.zeros((16, 16), dtype=np.float32)
    with pytest.raises(ValueError, match="flows"):
        registered_glyph_samples(
            source, source, source_mask=mask, donor_mask=mask, forward=flow, backward=flow
        )


def test_registered_glyph_rejects_mask_shape() -> None:
    source = np.zeros((16, 16, 3), dtype=np.uint8)
    mask = np.zeros((8, 16), dtype=np.uint8)
    flow = np.zeros((16, 16, 2), dtype=np.float32)
    with pytest.raises(ValueError, match="masks"):
        registered_glyph_samples(
            source, source, source_mask=mask, donor_mask=mask, forward=flow, backward=flow
        )


def test_registered_glyph_rejects_different_donor_shape() -> None:
    source = np.zeros((16, 16, 3), dtype=np.uint8)
    mask = np.zeros((16, 16), dtype=np.uint8)
    flow = np.zeros((16, 16, 2), dtype=np.float32)
    with pytest.raises(ValueError, match="shapes differ"):
        registered_glyph_samples(
            source, source[:8], source_mask=mask, donor_mask=mask, forward=flow, backward=flow
        )
