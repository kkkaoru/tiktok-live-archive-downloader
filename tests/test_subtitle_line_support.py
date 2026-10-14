from dataclasses import asdict

import cv2
import numpy as np
import pytest
from pytest import MonkeyPatch

from replay import subtitle_line_support as support
from replay.caption_masks import TextDetection, TextRectangle
from replay.subtitle_inpaint import Image


@pytest.fixture
def envelopes() -> tuple[Image, ...]:
    return support.line_masks(
        (
            TextDetection("UI", 1, TextRectangle(4, 1, 20, 8)),
            TextDetection("First caption", 1, TextRectangle(4, 12, 24, 12)),
            TextDetection("Second caption", 1, TextRectangle(4, 36, 24, 12)),
        ),
        region=TextRectangle(0, 0, 32, 56),
    )


def test_separate_lines_do_not_authorize_union(envelopes: tuple[Image, ...]) -> None:
    mask = np.zeros((56, 32), dtype=np.uint8)
    mask[16:18, 8:10] = 255
    mask[40:42, 8:10] = 255
    rows = support.attribute_components(mask, envelopes=envelopes, selected=(1, 2))
    assert tuple(asdict(row) for row in rows) == (
        {
            "component": 1,
            "pixels": 4,
            "box": (8, 16, 2, 2),
            "overlapping_lines": (1,),
            "selected_line": 1,
            "outside_pixels": 0,
        },
        {
            "component": 2,
            "pixels": 4,
            "box": (8, 40, 2, 2),
            "overlapping_lines": (2,),
            "selected_line": 2,
            "outside_pixels": 0,
        },
    )


@pytest.mark.parametrize("kind", ["union", "bridge", "ui", "padding", "unknown"])
def test_noncaption_components_remain_unknown(envelopes: tuple[Image, ...], kind: str) -> None:
    mask = np.zeros((56, 32), dtype=np.uint8)
    if kind == "union":
        mask[1:48, 4:28] = 255
    elif kind == "bridge":
        mask[16:42, 8:10] = 255
    elif kind == "ui":
        mask[3:5, 8:10] = 255
    elif kind == "padding":
        mask[26:28, 8:10] = 255
    else:
        mask[50:52, 1:3] = 255
    rows = support.attribute_components(mask, envelopes=envelopes, selected=(1, 2))
    assert len(rows) == 1
    assert rows[0].selected_line is None
    assert rows[0].outside_pixels > 0
    with pytest.raises(ValueError, match="Unknown"):
        support.repair_reviewed_ink(
            np.zeros((56, 32, 3), dtype=np.uint8),
            ink=mask,
            allowed=mask,
            envelopes=envelopes,
            selected=(1, 2),
        )


def test_overlapping_unselected_line_is_not_authorized(envelopes: tuple[Image, ...]) -> None:
    mask = np.zeros((56, 32), dtype=np.uint8)
    mask[16:18, 8:10] = 255
    rows = support.attribute_components(mask, envelopes=(*envelopes, envelopes[1]), selected=(1, 2))
    assert rows[0].overlapping_lines == (1, 3)
    assert rows[0].selected_line is None


def test_empty_is_not_absence(envelopes: tuple[Image, ...]) -> None:
    assert (
        support.attribute_components(
            np.zeros((56, 32), dtype=np.uint8), envelopes=envelopes, selected=(1, 2)
        )
        == ()
    )


def test_no_fixed_hue_and_no_automatic_interior_fill(envelopes: tuple[Image, ...]) -> None:
    image = np.full((56, 32, 3), 80, dtype=np.uint8)
    image[16:20, 8:12] = (220, 20, 20)
    image[17:19, 9:11] = 255
    image[40:44, 8:12] = (20, 220, 20)
    masks, thresholds = support.chroma_proposals(
        image, regions=np.maximum(envelopes[1], envelopes[2]), envelopes=envelopes, selected=(1, 2)
    )
    assert thresholds == (0.0, 0.0)
    assert int(np.count_nonzero(masks[0])) == 12
    assert masks[0][17, 9] == 0
    assert int(np.count_nonzero(masks[1])) == 16


def test_uniform_and_empty_proposals_are_unresolved(envelopes: tuple[Image, ...]) -> None:
    masks, thresholds = support.chroma_proposals(
        np.full((56, 32, 3), 80, dtype=np.uint8),
        regions=envelopes[1],
        envelopes=envelopes,
        selected=(1, 2),
    )
    assert thresholds == (0.0, 0.0)
    assert np.count_nonzero(masks) == 0


def test_reject_whole_edge_component_not_clip(envelopes: tuple[Image, ...]) -> None:
    ink = np.zeros((56, 32), dtype=np.uint8)
    ink[16:18, 8:10] = 255
    ink[12:14, 16:18] = 255
    kept, rejected = support.bounded_proposals(ink, envelopes=envelopes, selected=(1, 2))
    assert np.count_nonzero(kept) == 4
    assert np.count_nonzero(rejected) == 4
    assert kept[12, 16] == 0
    assert rejected[12, 16] == 255
    assert np.array_equal(kept | rejected, ink)


def test_actual_glyph_kernel_changes_zero_outside_permission(envelopes: tuple[Image, ...]) -> None:
    image = np.full((56, 32, 3), 80, dtype=np.uint8)
    image[16:18, 8:10] = (250, 10, 10)
    ink = np.zeros((56, 32), dtype=np.uint8)
    ink[16:18, 8:10] = 255
    allowed = np.asarray(
        cv2.dilate(ink, np.ones((3, 3), dtype=np.uint8), iterations=2), dtype=np.uint8
    )
    candidate, actual = support.repair_reviewed_ink(
        image, ink=ink, allowed=allowed, envelopes=envelopes, selected=(1, 2)
    )
    assert np.count_nonzero(actual) == 36
    assert np.count_nonzero(np.any(candidate != image, axis=2) & (allowed == 0)) == 0
    assert np.count_nonzero(candidate != image) > 0
    assert np.array_equal(actual, allowed)


@pytest.mark.parametrize("kind", ["empty", "outside", "shape", "expansion"])
def test_repair_requires_complete_reviewed_permission(
    envelopes: tuple[Image, ...], kind: str
) -> None:
    ink = np.zeros((56, 32), dtype=np.uint8)
    ink[16:18, 8:10] = 255
    allowed = ink.copy()
    image = np.zeros((56, 32, 3), dtype=np.uint8)
    if kind == "empty":
        ink[:] = 0
    elif kind == "outside":
        allowed[:] = 0
    elif kind == "shape":
        image = np.zeros((16, 16, 3), dtype=np.uint8)
    with pytest.raises(ValueError):
        support.repair_reviewed_ink(
            image, ink=ink, allowed=allowed, envelopes=envelopes, selected=(1, 2)
        )


def test_broken_kernel_cannot_escape(
    envelopes: tuple[Image, ...], monkeypatch: MonkeyPatch
) -> None:
    ink = np.zeros((56, 32), dtype=np.uint8)
    ink[16:18, 8:10] = 255
    allowed = np.asarray(
        cv2.dilate(ink, np.ones((3, 3), dtype=np.uint8), iterations=2), dtype=np.uint8
    )
    monkeypatch.setattr(
        support,
        "repair_glyphs",
        lambda *args, **kwargs: (np.full((56, 32, 3), 255, dtype=np.uint8), allowed),
    )
    with pytest.raises(ValueError, match="escaped"):
        support.repair_reviewed_ink(
            np.zeros((56, 32, 3), dtype=np.uint8),
            ink=ink,
            allowed=allowed,
            envelopes=envelopes,
            selected=(1, 2),
        )


@pytest.mark.parametrize("selected", [(), (-1,), (3,), (1, 1), (2, 1)])
def test_invalid_selection(envelopes: tuple[Image, ...], selected: tuple[int, ...]) -> None:
    with pytest.raises(ValueError):
        support.attribute_components(
            np.zeros((56, 32), dtype=np.uint8), envelopes=envelopes, selected=selected
        )


def test_invalid_support_and_envelopes(envelopes: tuple[Image, ...]) -> None:
    with pytest.raises(ValueError):
        support.attribute_components(
            np.full((56, 32), 1, dtype=np.uint8), envelopes=envelopes, selected=(1, 2)
        )
    with pytest.raises(ValueError):
        support.attribute_components(
            np.zeros((56, 32), dtype=np.uint8),
            envelopes=(np.zeros((8, 8), dtype=np.uint8),),
            selected=(0,),
        )
    with pytest.raises(ValueError):
        support.chroma_proposals(
            np.zeros((8, 8, 3), dtype=np.uint8),
            regions=np.zeros((56, 32), dtype=np.uint8),
            envelopes=envelopes,
            selected=(1, 2),
        )


@pytest.mark.parametrize("region", [TextRectangle(0.5, 0, 32, 56), TextRectangle(0, 0, 2, 2)])
def test_invalid_canvas(region: TextRectangle) -> None:
    with pytest.raises(ValueError):
        support.line_masks((TextDetection("Caption", 1, TextRectangle(1, 1, 2, 2)),), region=region)


def test_offset_and_off_roi_lines_are_not_padding() -> None:
    masks = support.line_masks(
        (
            TextDetection("UI", 1, TextRectangle(1, 1, 2, 2)),
            TextDetection("Caption", 1, TextRectangle(12, 12, 2, 2)),
        ),
        region=TextRectangle(10, 10, 8, 8),
    )
    assert np.count_nonzero(masks[0]) == 0
    assert np.count_nonzero(masks[1]) == 4
    assert masks[1][2, 2] == 255


def test_reviewed_permission_can_exceed_localization_not_bridge(
    envelopes: tuple[Image, ...],
) -> None:
    ink = np.zeros((56, 32), dtype=np.uint8)
    ink[12:14, 8:10] = 255
    allowed = np.asarray(
        cv2.dilate(ink, np.ones((3, 3), dtype=np.uint8), iterations=2), dtype=np.uint8
    )
    image = np.zeros((56, 32, 3), dtype=np.uint8)
    with pytest.raises(ValueError, match="Unknown"):
        support.reviewed_support(
            image, ink=ink, allowed=allowed, envelopes=envelopes, selected=(1,)
        )
    result = support.reviewed_support(
        image, ink=ink, allowed=allowed, envelopes=envelopes, selected=(1,), localization_only=True
    )
    assert np.count_nonzero(result) == 36
    allowed[14:40, 8:10] = 255
    with pytest.raises(ValueError, match="Unknown"):
        support.reviewed_support(
            image,
            ink=ink,
            allowed=allowed,
            envelopes=envelopes,
            selected=(1, 2),
            localization_only=True,
        )


def test_reviewed_halo_crosses_coarse_neighbour_box_not_neighbour_ink() -> None:
    envelopes = support.line_masks(
        (
            TextDetection("Upper", 1, TextRectangle(2, 2, 28, 12)),
            TextDetection("Lower", 1, TextRectangle(2, 16, 28, 12)),
        ),
        region=TextRectangle(0, 0, 32, 32),
    )
    ink = np.zeros((32, 32), dtype=np.uint8)
    ink[15:19, 8:10] = 255
    allowed = np.asarray(
        cv2.dilate(ink, np.ones((3, 3), dtype=np.uint8), iterations=2), dtype=np.uint8
    )
    result = support.reviewed_support(
        np.zeros((32, 32, 3), dtype=np.uint8),
        ink=ink,
        allowed=allowed,
        envelopes=envelopes,
        selected=(1,),
        localization_only=True,
    )
    assert result[13, 8] == 255
    # Ink itself crossing two localizations still has no unique line owner.
    ink[13:16, 8:10] = 255
    allowed = np.asarray(
        cv2.dilate(ink, np.ones((3, 3), dtype=np.uint8), iterations=2), dtype=np.uint8
    )
    with pytest.raises(ValueError, match="Unknown"):
        support.reviewed_support(
            np.zeros((32, 32, 3), dtype=np.uint8),
            ink=ink,
            allowed=allowed,
            envelopes=envelopes,
            selected=(0, 1),
            localization_only=True,
        )


def test_connected_chroma_recovers_clipped_outline_not_another_line(
    envelopes: tuple[Image, ...],
) -> None:
    image = np.full((56, 32, 3), 80, dtype=np.uint8)
    image[10:16, 8:10] = (180, 50, 80)
    image[38:44, 8:10] = (180, 50, 80)
    seed = np.zeros((56, 32), dtype=np.uint8)
    seed[12:16, 8:10] = 255
    proposed = support.connected_chroma_proposal(
        image,
        regions=np.full((56, 32), 255, dtype=np.uint8),
        seed=seed,
        threshold=60,
        envelopes=envelopes,
        selected=(1,),
    )
    assert np.count_nonzero(proposed) == 12
    assert proposed[10, 8] == 255
    assert proposed[38, 8] == 0
    # A skin-coloured bridge cannot be clipped into a valid line component.
    image[16:39, 8:10] = (180, 50, 80)
    bridged = support.connected_chroma_proposal(
        image,
        regions=np.full((56, 32), 255, dtype=np.uint8),
        seed=seed,
        threshold=60,
        envelopes=envelopes,
        selected=(1, 2),
    )
    assert np.count_nonzero(bridged) == 0


@pytest.mark.parametrize("kind", ["shape", "threshold", "seed"])
def test_connected_chroma_validates_evidence(envelopes: tuple[Image, ...], kind: str) -> None:
    image = np.zeros((56, 32, 3), dtype=np.uint8)
    seed = np.zeros((56, 32), dtype=np.uint8)
    threshold = 60.0
    if kind == "shape":
        image = image[:8]
    elif kind == "threshold":
        threshold = float("nan")
    else:
        seed[0, 0] = 255
    with pytest.raises(ValueError):
        support.connected_chroma_proposal(
            image,
            regions=envelopes[1],
            seed=seed,
            threshold=threshold,
            envelopes=envelopes,
            selected=(1,),
        )


def test_excessive_components_rejected() -> None:
    mask = np.zeros((160, 160), dtype=np.uint8)
    mask[::2, ::2] = 255
    with pytest.raises(ValueError, match="Excessive"):
        support.attribute_components(
            mask, envelopes=(np.full((160, 160), 255, dtype=np.uint8),), selected=(0,)
        )
