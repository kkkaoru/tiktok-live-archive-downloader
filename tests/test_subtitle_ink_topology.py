import cv2
import numpy as np
import pytest

from replay.subtitle_ink_topology import bright_component_proposals, bright_fill_proposals


def test_open_rim_fill_and_disjoint_white_decoration() -> None:
    image = np.full((32, 48, 3), 80, dtype=np.uint8)
    core = np.zeros((32, 48), dtype=np.uint8)
    core[8:24, 8:24] = 255
    core[10:22, 10:22] = 0
    core[8:10, 15] = 0
    image[core != 0] = (80, 80, 200)
    image[10:22, 10:22] = 250
    image[10:22, 32:44] = 250
    region = np.full((32, 48), 255, dtype=np.uint8)
    proposed = bright_component_proposals(image, core=core, region=region)
    assert np.count_nonzero(proposed) == 144
    assert proposed[16, 16] == 255
    assert proposed[16, 36] == 0


@pytest.mark.parametrize("background", [80, 255])
def test_component_counters_compare_actual_exterior(background: int) -> None:
    image = np.full((48, 48, 3), background, dtype=np.uint8)
    core = np.zeros((48, 48), dtype=np.uint8)
    cv2.circle(core, (24, 24), 17, 255, 2)
    cv2.circle(core, (24, 24), 8, 255, 2)
    cv2.circle(image, (24, 24), 15, (255, 255, 255), -1)
    cv2.circle(image, (24, 24), 9, (background, background, background), -1)
    image[core != 0] = (20, 160, 20)
    region = np.zeros((48, 48), dtype=np.uint8)
    region[4:44, 4:44] = 255
    proposed = bright_component_proposals(image, core=core, region=region)
    assert proposed[24, 24] == 0
    if background == 255:
        assert np.count_nonzero(proposed) == 0
    else:
        assert proposed[24, 36] == 255


@pytest.mark.parametrize("kind", ["empty", "uniform", "invalid", "excessive"])
def test_component_evidence_boundaries(kind: str) -> None:
    image = np.full((160, 160, 3), 80, dtype=np.uint8)
    core = np.zeros((160, 160), dtype=np.uint8)
    region = np.full((160, 160), 255, dtype=np.uint8)
    if kind == "empty":
        region[:] = 0
    elif kind == "invalid":
        core[0, 0] = 1
    elif kind == "excessive":
        image[::2, ::2] = 255
    if kind in ("invalid", "excessive"):
        with pytest.raises(ValueError):
            bright_component_proposals(image, core=core, region=region)
    else:
        assert np.count_nonzero(bright_component_proposals(image, core=core, region=region)) == 0


def test_thin_outer_rim_is_not_a_fill_and_may_cross_localization() -> None:
    image = np.full((32, 48, 3), 80, dtype=np.uint8)
    core = np.zeros((32, 48), dtype=np.uint8)
    core[10:22, 12:18] = 255
    image[8:24, 10:20] = 245
    image[core != 0] = (80, 80, 200)
    region = np.zeros((32, 48), dtype=np.uint8)
    region[9:23, 8:22] = 255
    proposal = bright_component_proposals(image, core=core, region=region)
    assert np.count_nonzero(proposal) == 88
    assert proposal[8, 10] == 255
    assert proposal[12, 12] == 0


def test_bright_fill_does_not_authorize_touching_white_scene_highlight() -> None:
    image = np.full((32, 48, 3), 80, dtype=np.uint8)
    core = np.zeros((32, 48), dtype=np.uint8)
    core[10:22, 12:24] = 255
    core[12:20, 14:22] = 0
    image[core != 0] = (80, 80, 200)
    image[12:20, 14:22] = 250
    image[8:10, 14:22] = 250
    proposed = bright_component_proposals(
        image, core=core, region=np.full((32, 48), 255, dtype=np.uint8)
    )
    assert np.count_nonzero(proposed) == 64
    assert proposed[8, 14] == 0
    assert proposed[12, 14] == 255


def test_nearby_large_white_background_is_not_a_thin_rim() -> None:
    image = np.full((32, 48, 3), 80, dtype=np.uint8)
    core = np.zeros((32, 48), dtype=np.uint8)
    core[10:22, 12:16] = 255
    image[core != 0] = (80, 80, 200)
    image[8:24, 16:40] = 250
    proposed = bright_component_proposals(
        image, core=core, region=np.full((32, 48), 255, dtype=np.uint8)
    )
    assert np.count_nonzero(proposed) == 0


def test_white_fill_is_pixel_selected_not_whole_cavity() -> None:
    image = np.full((32, 40, 3), 80, dtype=np.uint8)
    core = np.zeros((32, 40), dtype=np.uint8)
    core[8:24, 8:32] = 255
    core[11:21, 11:29] = 0
    image[core != 0] = (80, 80, 210)
    image[11:21, 11:29] = 250
    image[15:17, 16:18] = 40
    fill, rows = bright_fill_proposals(image, core=core)
    assert np.count_nonzero(fill) == 176
    assert fill[15, 16] == 0
    assert fill[11, 11] == 255
    assert len(rows) == 1
    assert rows[0].threshold == 80.0
    assert rows[0].pixels == 180


@pytest.mark.parametrize("background", [50, 255])
def test_nested_o_preserves_dark_and_white_counterholes(background: int) -> None:
    image = np.full((48, 48, 3), background, dtype=np.uint8)
    core = np.zeros((48, 48), dtype=np.uint8)
    cv2.circle(core, (24, 24), 17, 255, 2)
    cv2.circle(core, (24, 24), 8, 255, 2)
    cv2.circle(image, (24, 24), 15, (255, 255, 255), -1)
    cv2.circle(image, (24, 24), 9, (background, background, background), -1)
    image[core != 0] = (20, 160, 20)
    fill, rows = bright_fill_proposals(image, core=core)
    assert fill[24, 24] == 0
    assert len(rows) == 2
    if background == 255:
        assert np.count_nonzero(fill) == 0
    else:
        assert fill[24, 36] == 255


def test_white_decoration_without_outline_does_not_seed() -> None:
    image = np.zeros((32, 32, 3), dtype=np.uint8)
    image[8:24, 8:24] = 255
    fill, rows = bright_fill_proposals(image, core=np.zeros((32, 32), dtype=np.uint8))
    assert np.count_nonzero(fill) == 0
    assert rows == ()


def test_missing_exterior_is_unknown() -> None:
    image = np.full((32, 32, 3), 255, dtype=np.uint8)
    core = np.full((32, 32), 255, dtype=np.uint8)
    core[8:24, 8:24] = 0
    fill, rows = bright_fill_proposals(image, core=core)
    assert np.count_nonzero(fill) == 0
    assert rows[0].exterior_median is None
    assert rows[0].threshold is None


def test_bright_outlined_decoration_is_only_a_proposal() -> None:
    # Geometry alone cannot distinguish this decoration from a glyph.
    image = np.full((32, 32, 3), (90, 110, 160), dtype=np.uint8)
    core = np.zeros((32, 32), dtype=np.uint8)
    core[8:24, 8:24] = 255
    core[10:22, 10:22] = 0
    image[10:22, 10:22] = 250
    fill, rows = bright_fill_proposals(image, core=core)
    assert np.count_nonzero(fill) == 144
    assert rows[0].proposed_pixels == 144


@pytest.mark.parametrize("kind", ["shape", "dtype", "nonbinary", "excessive"])
def test_bad_core_rejected(kind: str) -> None:
    image = np.zeros((32, 32, 3), dtype=np.uint8)
    core = np.zeros((32, 32), dtype=np.uint8)
    if kind == "shape":
        core = core[:8]
    elif kind == "dtype":
        core = core.astype(np.float32)
    elif kind == "nonbinary":
        core[0, 0] = 1
    else:
        image = np.zeros((160, 160, 3), dtype=np.uint8)
        core = np.zeros((160, 160), dtype=np.uint8)
        core[::2, ::2] = 255
    with pytest.raises(ValueError):
        bright_fill_proposals(image, core=core)
