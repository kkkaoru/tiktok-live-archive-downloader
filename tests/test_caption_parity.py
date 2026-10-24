"""Caption render parity metrics on synthetic RGBA caption frames."""

import json
from collections.abc import Iterator
from pathlib import Path

import numpy as np
import pytest
from numpy.typing import NDArray

from replay import caption_parity
from replay.caption_parity import (
    MAX_DISTANCE,
    compare,
    crossing,
    delta_e2000,
    edge_profile,
    flip,
    glyph_lags,
    lag,
    main,
    ssim,
    summary,
    text_lines,
    verdict,
)
from replay.core import ReplayError

FILL = (47, 176, 238, 255)


def caption(shift: int = 0, *, lines: int = 2) -> NDArray[np.uint8]:
    """A style-20-like block: rim, white band, keyline, fill lines of 9-pixel glyph strokes
    and a soft shadow below."""
    frame = np.zeros((140, 220, 4), dtype=np.uint8)
    frame[104:114, 20 + shift : 190 + shift] = (0, 0, 0, 90)
    frame[20:104, 20 + shift : 190 + shift] = (0, 0, 0, 255)
    frame[26:98, 26 + shift : 184 + shift] = (255, 255, 255, 255)
    frame[32:92, 32 + shift : 178 + shift] = (0, 0, 0, 255)
    frame[34:58, 34 + shift : 176 + shift] = FILL
    frame[34:38, 34 + shift : 176 + shift] = (151, 216, 247, 255)
    if lines == 2:
        frame[64:90, 34 + shift : 176 + shift] = FILL
    for gap in range(43, 176, 12):
        frame[34:90, gap + shift : gap + 3 + shift] = (0, 0, 0, 255)
    return frame


def test_ciede2000_matches_sharma_wu_dalal_reference_pairs() -> None:
    first = np.array([[50, 2.6772, -79.7751], [50, 0, 0], [50, 2.5, 0]])
    second = np.array([[50, 0, -82.7485], [50, -1, 2], [73, 25, -18]])
    assert np.round(delta_e2000(first, second), 4).tolist() == [2.0425, 2.3669, 27.1492]


def test_identical_images_have_full_ssim_and_no_flip_error() -> None:
    image = caption_parity.premultiplied(caption())
    support = np.ones(image.shape[:2], dtype=bool)
    assert ssim(image, image, support) == pytest.approx(1.0)
    grey = caption_parity.over_grey(caption())
    assert float(flip(grey, grey).max()) == pytest.approx(0.0)


def test_flip_sees_a_missing_caption() -> None:
    blank = caption_parity.over_grey(np.zeros((140, 220, 4), dtype=np.uint8))
    error = flip(caption_parity.over_grey(caption()), blank)
    assert round(float(error.max()), 2) == 0.97


def test_text_lines_measures_each_line_and_its_fill_colours() -> None:
    lines = text_lines(caption())
    assert [(line.top, line.bottom) for line in lines] == [(34, 57), (64, 89)]
    assert sorted(lines[0].colours) == [10, 20, 30, 50]
    assert np.round(lines[0].colours[50] * 255).tolist() == [47.0, 176.0, 238.0]


def test_an_empty_frame_has_no_lines_and_an_empty_edge_profile() -> None:
    empty = np.zeros((20, 20, 4), dtype=np.uint8)
    assert text_lines(empty) == []
    assert float(edge_profile(empty).sum()) == 0.0


def test_lag_finds_a_sub_pixel_displacement() -> None:
    axis = np.arange(60, dtype=np.float64)
    first = np.exp(-((axis - 30.0) ** 2) / 20)
    second = np.exp(-((axis - 31.4) ** 2) / 20)
    assert round(lag(first, second), 2) == 1.4
    assert round(lag(second, first), 2) == -1.4


def test_lag_at_the_search_edge_is_whole_pixels() -> None:
    first = np.zeros(30)
    first[10] = 1.0
    second = np.zeros(30)
    second[25] = 1.0
    assert lag(first, second) == -6.0


def test_glyph_windows_of_a_shifted_line_move_by_the_shift() -> None:
    reference, moved = text_lines(caption()), text_lines(caption(shift=2))
    assert [round(value, 2) for value in glyph_lags(reference[0], moved[0])] == [2.0] * 6


def test_crossing_interpolates_and_reports_profiles_that_never_fall() -> None:
    distances = np.array([0.5, 1.5, 2.5, 3.5])
    assert crossing(distances, np.array([1.0, 1.0, 0.0, 0.0]), 0.5) == 2.0
    assert crossing(distances, np.array([0.2, 0.1, 0.0, 0.0]), 0.5) == 0.5
    assert crossing(distances, np.array([1.0, 1.0, 1.0, 1.0]), 0.5) == MAX_DISTANCE


def test_identical_renders_differ_in_nothing() -> None:
    result = summary(compare([caption(), caption()], [caption(), caption()]))
    assert set(result.values()) == {0.0}


def test_a_shifted_render_is_measured_as_displaced() -> None:
    result = summary(compare([caption()], [caption(shift=2)]))
    assert (result["glyphX"], result["glyphY"], result["baseline"]) == (1.916667, 0.0, 0.0)
    assert result["ssimLoss"] > 0.1


def test_a_render_with_another_line_count_is_counted() -> None:
    result = summary(compare([caption()], [caption(lines=1)]))
    assert result["lineMismatch"] == 1.0


def test_frames_without_captions_are_skipped_but_one_caption_is_needed() -> None:
    empty = np.zeros((140, 220, 4), dtype=np.uint8)
    assert compare([empty, caption()], [empty, caption()]).frames == 1
    with pytest.raises(ReplayError, match="Neither render"):
        summary(compare([empty], [empty]))


def test_renders_of_different_length_or_size_are_refused() -> None:
    with pytest.raises(ReplayError, match="frame counts"):
        compare([caption()], [caption(), caption()])
    with pytest.raises(ReplayError, match="frame sizes"):
        compare([caption()], [np.zeros((10, 10, 4), dtype=np.uint8)])


def test_verdict_allows_the_floor_or_the_absolute_minimum() -> None:
    floor = {name: 0.0 for name in caption_parity.MINIMUM_ERROR} | {"ssimLoss": 0.1}
    close = {name: 0.0 for name in caption_parity.MINIMUM_ERROR} | {"ssimLoss": 0.1}
    far = close | {"glyphX": 0.3}
    assert verdict(floor, close)["pass"] is True
    assert verdict(floor, far)["failed"] == ["glyphX"]


def test_main_compares_against_the_floor_and_exits_on_failure(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    movies = {"motion.mov": 0, "shifted.mov": 1, "close.mov": 0, "far.mov": 4}

    def frames(path: str, *, step: int) -> Iterator[NDArray[np.uint8]]:
        yield caption(shift=movies[path] * step)

    monkeypatch.setattr(caption_parity, "frames", frames)
    assert main(["motion.mov", "close.mov", "--floor", "shifted.mov"]) == 0
    assert json.loads(capsys.readouterr().out)["failed"] == []
    assert main(["motion.mov", "far.mov", "--floor", "shifted.mov"]) == 1
    assert "glyphX" in json.loads(capsys.readouterr().out)["failed"]


def test_the_absolute_guard_caps_a_loose_floor() -> None:
    floor = {name: 0.0 for name in caption_parity.MINIMUM_ERROR} | {
        "deltaE20": 3.0,
        "alphaHalf": 1.0,
    }
    allowed = caption_parity.limits(floor)
    assert (allowed["deltaE20"], allowed["alphaHalf"], allowed["ssimLoss"]) == (2.0, 0.5, 0.002)


def test_a_half_alpha_white_edge_stored_premultiplied_is_seen() -> None:
    straight = caption()
    straight[30:32, 26:184] = (255, 255, 255, 128)
    premultiplied_by_mistake = straight.copy()
    premultiplied_by_mistake[30:32, 26:184] = (128, 128, 128, 128)
    result = summary(compare([straight], [premultiplied_by_mistake]))
    assert result["ssimLoss"] > 0.0
    assert result["flipMean"] > 0.0


def test_main_reuses_a_stored_floor(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    def frames(path: str, *, step: int) -> Iterator[NDArray[np.uint8]]:
        yield caption(shift=step * (path == "far.mov") * 4)

    monkeypatch.setattr(caption_parity, "frames", frames)
    stored = tmp_path / "floor.json"
    floor = {name: 0.0 for name in caption_parity.MINIMUM_ERROR}
    stored.write_text(json.dumps({"floor": floor}), encoding="utf-8")
    assert main(["motion.mov", "close.mov", "--floor-json", str(stored)]) == 0
    assert json.loads(capsys.readouterr().out)["floor"] == floor
    assert main(["motion.mov", "far.mov", "--floor-json", str(stored)]) == 1
    capsys.readouterr()
    stored.write_text(json.dumps({"floor": {"ssimLoss": 0.1}}), encoding="utf-8")
    with pytest.raises(ReplayError, match="no complete parity floor"):
        main(["motion.mov", "close.mov", "--floor-json", str(stored)])
