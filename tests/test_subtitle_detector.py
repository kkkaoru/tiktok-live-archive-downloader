"""Synthetic geometry and mocked, local-only model loading."""

from pathlib import Path
from unittest.mock import Mock

import cv2
import numpy as np
import pytest
from pytest import MonkeyPatch

from replay.local_asr import fingerprint
from replay.subtitle_detector import LearnedTextDetector, polygon_mask


def test_polygon_pixels_not_a_guessed_envelope() -> None:
    polygon = np.array([[4, 2], [6, 4], [4, 6], [2, 4]], dtype=np.int32)
    mask = polygon_mask((polygon,), np.array([0.9]), width=736, height=736)
    assert np.count_nonzero(mask) == 13
    assert mask[2, 2] == 0
    assert mask[4, 4] == 255


def test_outer_grid_edges_and_subpixel_projection() -> None:
    polygon = np.array([[368, 184], [736, 184], [736, 552], [368, 552]], dtype=np.int32)
    mask = polygon_mask((polygon,), np.array([0.9]), width=8, height=8)
    assert np.count_nonzero(mask) == 20
    assert mask[2, 4] == 255
    assert mask[6, 7] == 255
    assert mask[2, 3] == 0


def test_unclip_expansion_is_intersected_without_adding_support() -> None:
    polygon = np.array([[368, 184], [739, 184], [739, 552], [368, 552]], dtype=np.int32)
    mask = polygon_mask((polygon,), np.array([0.9]), width=8, height=8)
    assert np.count_nonzero(mask) == 20
    assert mask[2, 3] == 0
    assert mask[6, 7] == 255


def test_oblique_intersection_does_not_clamp_corners() -> None:
    polygon = np.array([[700, 100], [780, 150], [780, 200], [700, 150]], dtype=np.int32)
    mask = polygon_mask((polygon,), np.array([0.9]), width=736, height=736)
    assert mask[140, 735] == 255
    assert mask[190, 735] == 0


def test_wholly_off_canvas_is_not_accepted() -> None:
    polygon = np.array([[740, 2], [750, 2], [750, 6], [740, 6]], dtype=np.int32)
    with pytest.raises(ValueError, match="no canvas support"):
        polygon_mask((polygon,), np.array([0.9]), width=8, height=8)


def test_empty_is_a_zero_mask_not_an_absence_claim() -> None:
    mask = polygon_mask((), np.array([], dtype=np.float32), width=8, height=8)
    assert not np.any(mask)


@pytest.mark.parametrize("width,height", [(7, 8), (2001, 1000)])
def test_canvas_budget(width: int, height: int) -> None:
    with pytest.raises(ValueError, match="canvas"):
        polygon_mask((), np.array([]), width=width, height=height)


@pytest.mark.parametrize("polygons", [None, (), np.zeros((1, 4, 2)), [None] * 65])
def test_polygon_list_boundary(polygons: object) -> None:
    with pytest.raises(ValueError):
        polygon_mask(polygons, np.array([0.9]), width=8, height=8)


@pytest.mark.parametrize(
    "confidence",
    [
        None,
        np.array([1]),
        np.array([np.nan]),
        np.array([0.49]),
        np.array([1.1]),
        np.zeros((1, 1, 1)),
        np.array([]),
    ],
)
def test_confidence_boundary(confidence: object) -> None:
    with pytest.raises(ValueError, match="confidence"):
        polygon_mask(
            (np.array([[2, 2], [4, 2], [4, 4], [2, 4]], dtype=np.int32),),
            confidence,
            width=8,
            height=8,
        )


@pytest.mark.parametrize(
    "polygon",
    [
        None,
        np.zeros((4, 2)),
        np.zeros((3, 2), dtype=np.int32),
        np.array([[-737, 2], [4, 2], [4, 4], [0, 4]], dtype=np.int32),
        np.array([[2, 2], [1473, 2], [1473, 4], [2, 4]], dtype=np.int32),
        np.array([[2, 2], [4, 2], [4, 1473], [2, 1473]], dtype=np.int32),
        np.zeros((4, 2), dtype=np.int32),
        np.array([[2, 2], [4, 4], [4, 2], [2, 4]], dtype=np.int32),
    ],
)
def test_invalid_geometry_is_rejected_not_clipped(polygon: object) -> None:
    with pytest.raises(ValueError, match="polygon"):
        polygon_mask((polygon,), np.array([0.9]), width=8, height=8)


@pytest.fixture
def native(tmp_path: Path, monkeypatch: MonkeyPatch) -> tuple[Path, Mock, Mock]:
    weights = tmp_path / "model.onnx"
    weights.write_bytes(b"synthetic weights")
    network = Mock()
    detector = Mock()
    detector.detect.return_value = (
        (np.array([[184, 184], [368, 184], [368, 368], [184, 368]], dtype=np.int32),),
        (0.9,),
    )
    monkeypatch.setattr(cv2.dnn, "readNet", Mock(return_value=network))
    monkeypatch.setattr(cv2.dnn, "TextDetectionModel_DB", Mock(return_value=detector))
    threads = Mock()
    monkeypatch.setattr(cv2, "setNumThreads", threads)
    return weights, detector, threads


def test_bounded_sequential_model_reuse_preserves_source(native: tuple[Path, Mock, Mock]) -> None:
    model = LearnedTextDetector(native[0], sha256=fingerprint(native[0]))
    source = np.zeros((8, 8, 3), dtype=np.uint8)
    masks = model.detect((source, source))
    assert len(masks) == 2
    assert np.count_nonzero(masks[0]) == 9
    assert not np.any(source)
    assert model.observed_regions == [1, 1]
    assert model.intersected_regions == [0, 0]
    assert native[1].detect.call_count == 2
    assert native[1].detect.call_args.args[0].shape == (736, 736, 3)
    native[2].assert_called_once_with(2)
    native[1].setInputSize.assert_called_with((736, 736))
    assert native[1].setInputSize.call_count == 3
    native[1].setBinaryThreshold.assert_called_once_with(0.3)
    native[1].setPolygonThreshold.assert_called_once_with(0.5)
    native[1].setUnclipRatio.assert_called_once_with(2.0)


def test_detector_grid_follows_source_aspect(native: tuple[Path, Mock, Mock]) -> None:
    model = LearnedTextDetector(native[0], sha256=fingerprint(native[0]))
    wide = np.zeros((16, 32, 3), dtype=np.uint8)
    tall = np.zeros((32, 16, 3), dtype=np.uint8)
    masks = model.detect((wide, tall))
    assert model.grid_sizes == [(736, 384), (384, 736)]
    assert masks[0].shape == (16, 32)
    assert masks[1].shape == (32, 16)
    assert not np.any(wide)
    assert not np.any(tall)
    assert native[1].detect.call_args_list[0].args[0].shape == (384, 736, 3)
    assert native[1].detect.call_args_list[1].args[0].shape == (736, 736, 3)
    assert native[1].detect.call_args_list[2].args[0].shape == (736, 384, 3)
    assert model.square_context_regions == [1, 1]
    assert model.observed_regions == [2, 2]


@pytest.mark.parametrize("grid", [(0, 736), (736, 737), (736, 127)])
def test_invalid_model_grid(grid: tuple[int, int]) -> None:
    with pytest.raises(ValueError, match="model grid"):
        polygon_mask((), np.array([]), width=8, height=8, grid=grid)


def test_complementary_views_union_only_measured_support(native: tuple[Path, Mock, Mock]) -> None:
    native[1].detect.side_effect = [
        ((np.array([[0, 0], [184, 0], [184, 184], [0, 184]], dtype=np.int32),), (0.9,)),
        ((np.array([[552, 552], [690, 552], [690, 690], [552, 690]], dtype=np.int32),), (0.9,)),
    ]
    model = LearnedTextDetector(native[0], sha256=fingerprint(native[0]))
    mask = model.detect((np.zeros((16, 32, 3), dtype=np.uint8),))[0]
    assert mask[2, 2] == 255
    assert mask[12, 24] == 255
    assert mask[9, 16] == 0
    assert model.observed_regions == [2]
    assert model.square_context_regions == [1]


def test_combined_views_keep_the_region_budget(native: tuple[Path, Mock, Mock]) -> None:
    polygon = np.array([[0, 0], [184, 0], [184, 184], [0, 184]], dtype=np.int32)
    native[1].detect.return_value = ((polygon,) * 33, (0.9,) * 33)
    model = LearnedTextDetector(native[0], sha256=fingerprint(native[0]))
    with pytest.raises(ValueError, match="Combined detector"):
        model.detect((np.zeros((16, 32, 3), dtype=np.uint8),))


def test_empty_native_confidences(native: tuple[Path, Mock, Mock]) -> None:
    native[1].detect.return_value = ((), ())
    model = LearnedTextDetector(native[0], sha256=fingerprint(native[0]))
    assert not np.any(model.detect((np.zeros((8, 8, 3), dtype=np.uint8),))[0])


@pytest.mark.parametrize("sha", ["bad", "0" * 64, "X" * 64])
def test_no_unverified_weights(native: tuple[Path, Mock, Mock], sha: str) -> None:
    with pytest.raises(ValueError, match="hash-verified"):
        LearnedTextDetector(native[0], sha256=sha)
    native[1].detect.assert_not_called()


def test_missing_linked_and_oversize_weights(
    native: tuple[Path, Mock, Mock], tmp_path: Path
) -> None:
    linked = tmp_path / "link.onnx"
    linked.symlink_to(native[0])
    with pytest.raises(ValueError):
        LearnedTextDetector(linked, sha256=fingerprint(native[0]))
    with pytest.raises(ValueError):
        LearnedTextDetector(tmp_path / "missing", sha256="0" * 64)
    with native[0].open("wb") as stream:
        stream.truncate(32 * 1024 * 1024 + 1)
    with pytest.raises(ValueError):
        LearnedTextDetector(native[0], sha256="0" * 64)


def test_roi_count_and_shape_guards(native: tuple[Path, Mock, Mock]) -> None:
    model = LearnedTextDetector(native[0], sha256=fingerprint(native[0]))
    with pytest.raises(ValueError, match="1–8"):
        model.detect(())
    with pytest.raises(ValueError, match="BGR"):
        model.detect((np.zeros((8, 8), dtype=np.uint8),))
    native[1].detect.assert_not_called()


def test_model_mutation_during_loading_fails(native: tuple[Path, Mock, Mock]) -> None:
    digest = fingerprint(native[0])
    native[1].setInputScale.side_effect = lambda value: native[0].write_bytes(b"changed")
    with pytest.raises(ValueError, match="changed during loading"):
        LearnedTextDetector(native[0], sha256=digest)


def test_native_errors_are_not_hidden(native: tuple[Path, Mock, Mock]) -> None:
    native[1].detect.side_effect = cv2.error("synthetic inference failure")
    model = LearnedTextDetector(native[0], sha256=fingerprint(native[0]))
    with pytest.raises(cv2.error, match="synthetic"):
        model.detect((np.zeros((8, 8, 3), dtype=np.uint8),))
