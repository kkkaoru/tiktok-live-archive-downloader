"""Detected support follows moving, differently sized subtitle lines."""

import numpy as np
import pytest

from replay.subtitle_inpaint import InpaintSettings, repair_glyphs


@pytest.mark.parametrize(
    "left,top,width,lines", [(12, 16, 20, 1), (70, 20, 42, 2), (35, 64, 60, 3)]
)
@pytest.mark.parametrize("softness", [0.0, 0.8])
def test_local_repair_follows_detected_lines_without_rectangle_fallback(
    left: int, top: int, width: int, lines: int, softness: float
) -> None:
    image = np.full((160, 160, 3), 80, dtype=np.uint8)
    detected = np.zeros((160, 160), dtype=np.uint8)
    for line in range(lines):
        y = top + line * 24
        for x in range(left, left + width, 10):
            detected[y : y + 4, x : x + 3] = 255
    image[detected != 0] = (180, 50, 220)
    original = image.copy()
    result, support = repair_glyphs(
        image, settings=InpaintSettings(expansion=2, softness=softness), text_mask=detected
    )
    np.testing.assert_array_equal(image, original)
    np.testing.assert_array_equal(result[support == 0], original[support == 0])
    assert np.any(result[detected != 0] != original[detected != 0])
    assert np.count_nonzero(support) < image.shape[0] * image.shape[1] // 4
    # A gap between detected lines remains original even inside their bounding box.
    if lines > 1:
        assert support[top + 14, left + 1] == 0
        np.testing.assert_array_equal(result[top + 14], original[top + 14])


def test_empty_detection_does_not_invent_a_fixed_subtitle_band() -> None:
    image = np.full((64, 96, 3), 90, dtype=np.uint8)
    image[35:42] = (180, 50, 220)
    result, support = repair_glyphs(
        image,
        settings=InpaintSettings(expansion=4, softness=0.8),
        text_mask=np.zeros((64, 96), dtype=np.uint8),
    )
    np.testing.assert_array_equal(result, image)
    assert not np.any(support)
