"""Synthetic character geometry and mocked native process boundaries."""

import json
import logging
import subprocess
from pathlib import Path
from unittest.mock import Mock

import numpy as np
import pytest

from replay.caption_masks import TextRectangle
from replay.subtitle_inpaint import InpaintSettings, repair_glyphs
from replay.subtitle_repair import repair_with_retries
from replay.subtitle_retry import RepairEvidence
from replay.subtitle_vision import (
    _contrast_image,
    character_mask,
    text_region_mask,
    vision_text_mask,
    vision_text_masks,
)


def test_degenerate_native_envelope_uses_only_measured_characters(
    caplog: pytest.LogCaptureFixture,
) -> None:
    document = {
        "width": 16,
        "height": 16,
        "characterBoxes": [{"x": 2, "y": 3, "width": 2, "height": 3}],
        "textBoxes": [],
        "diagnostics": {"degenerateTextObservationCount": 1},
    }
    mask = text_region_mask(document, width=16, height=16)
    assert np.count_nonzero(mask) == 6
    assert np.array_equal(mask, character_mask(document, width=16, height=16))
    assert "quality unapproved" in caplog.text


@pytest.mark.parametrize("count", [True, -1, 65, None, "1"])
def test_invalid_degenerate_count_fails_closed(count: object) -> None:
    with pytest.raises(ValueError, match="degenerate"):
        text_region_mask(
            {
                "width": 16,
                "height": 16,
                "textBoxes": [],
                "diagnostics": {"degenerateTextObservationCount": count},
            },
            width=16,
            height=16,
        )


def test_degenerate_envelope_requires_real_character_support() -> None:
    with pytest.raises(ValueError, match="lacks measured"):
        text_region_mask(
            {
                "width": 16,
                "height": 16,
                "textBoxes": [],
                "characterBoxes": [],
                "diagnostics": {"degenerateTextObservationCount": 1},
            },
            width=16,
            height=16,
        )


def test_degenerate_envelope_does_not_accept_invalid_character_geometry() -> None:
    with pytest.raises(ValueError, match="outside"):
        text_region_mask(
            {
                "width": 16,
                "height": 16,
                "textBoxes": [],
                "characterBoxes": [{"x": 15, "y": 0, "width": 2, "height": 2}],
                "diagnostics": {"degenerateTextObservationCount": 1},
            },
            width=16,
            height=16,
        )


def test_measured_text_regions_fill_gaps_only_in_explicit_mode() -> None:
    document = {
        "width": 16,
        "height": 16,
        "characterBoxes": [{"x": 2, "y": 3, "width": 2, "height": 3}],
        "textBoxes": [{"x": 2, "y": 3, "width": 10, "height": 3}],
    }
    characters = character_mask(document, width=16, height=16)
    regions = text_region_mask(document, width=16, height=16)
    assert np.count_nonzero(characters) == 6
    assert np.count_nonzero(regions) == 30
    assert np.count_nonzero(regions[:, :2]) == 0
    assert np.count_nonzero(regions[:, 12:]) == 0
    assert regions[3:6, 4:10].tolist() == [[255] * 6] * 3


@pytest.mark.parametrize("boxes", [None, "invalid", [{}] * 65])
def test_text_regions_do_not_fall_back_to_characters(boxes: object) -> None:
    with pytest.raises(ValueError, match="Vision text boxes"):
        text_region_mask(
            {"width": 16, "height": 16, "characterBoxes": [], "textBoxes": boxes},
            width=16,
            height=16,
        )


def test_explicit_text_region_request_uses_observed_native_geometry(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    executable = tmp_path / "native"
    executable.touch()
    monkeypatch.setattr(
        "replay.subtitle_vision.subprocess.run",
        Mock(
            return_value=subprocess.CompletedProcess(
                [],
                0,
                b'[{"width":8,"height":8,"characterBoxes":[],"textBoxes":[{"x":1,"y":2,"width":6,"height":2}]}]',
                b"",
            )
        ),
    )
    mask = vision_text_mask(
        np.zeros((8, 8, 3), dtype=np.uint8),
        executable=executable,
        work_directory=tmp_path,
        text_regions=True,
    )
    assert np.count_nonzero(mask) == 12
    assert np.count_nonzero(mask[:2]) == 0
    assert np.count_nonzero(mask[4:]) == 0


def test_reference_contrast_preserves_letters_without_changing_source() -> None:
    image = np.full((32, 32, 3), 140, dtype=np.uint8)
    image[8:16, 8:12] = 160
    image[8:16, 12:16] = 240
    before = image.copy()
    prepared = _contrast_image(image, TextRectangle(8, 8, 8, 8))
    assert prepared.shape == (32, 32, 3)
    assert np.count_nonzero(prepared[8:16, 8:12]) == 0
    assert np.count_nonzero(prepared[8:16, 12:16] == 255) == 96
    assert np.count_nonzero(prepared[:8]) == 0
    np.testing.assert_array_equal(image, before)


@pytest.mark.parametrize("region", [TextRectangle(0, 0, 8, 8), TextRectangle(7, 0, 8, 8)])
def test_invalid_or_uniform_contrast_reference(region: TextRectangle) -> None:
    with pytest.raises(ValueError, match="Contrast reference"):
        _contrast_image(np.zeros((8, 8, 3), dtype=np.uint8), region)


def test_contrast_reference_count_must_match_batch() -> None:
    with pytest.raises(ValueError, match="reference count"):
        vision_text_masks((np.zeros((8, 8, 3), dtype=np.uint8),), contrast_regions=())


def test_observed_contrast_uses_vision_geometry_not_threshold_as_mask(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    executable = tmp_path / "native"
    executable.touch()
    monkeypatch.setattr(
        "replay.subtitle_vision.subprocess.run",
        Mock(
            return_value=subprocess.CompletedProcess(
                [], 0, b'[{"width":8,"height":8,"characterBoxes":[]}]', b""
            )
        ),
    )
    image = np.full((8, 8, 3), 160, dtype=np.uint8)
    image[:, 4:] = 240
    before = image.copy()
    with caplog.at_level(logging.DEBUG, logger="replay.subtitle_vision"):
        mask = vision_text_mask(
            image,
            executable=executable,
            work_directory=tmp_path,
            contrast_region=TextRectangle(0, 0, 8, 8),
        )
    assert np.count_nonzero(mask) == 0
    assert "threshold=160.000000" in caplog.text
    assert "round_trip_equal=true" in caplog.text
    np.testing.assert_array_equal(image, before)


def test_character_boxes_round_outwards_without_filling_between_letters() -> None:
    mask = character_mask(
        {
            "width": 16,
            "height": 16,
            "characterBoxes": [
                {"x": 2.5, "y": 3.25, "width": 1.5, "height": 2.25},
                {"x": 10, "y": 3, "width": 2, "height": 3},
            ],
        },
        width=16,
        height=16,
    )
    assert np.count_nonzero(mask) == 12
    assert mask[3:6, 2:4].tolist() == [[255, 255], [255, 255], [255, 255]]
    assert np.count_nonzero(mask[:, 4:10]) == 0


@pytest.mark.parametrize(
    "value",
    [
        {},
        {"width": True, "height": 16},
        {"width": 16, "height": 15},
        {"width": 16, "height": 16},
        {"width": 16, "height": 16, "characterBoxes": "missing"},
        {"width": 16, "height": 16, "characterBoxes": [{}] * 4097},
    ],
)
def test_invalid_detection_document(value: object) -> None:
    with pytest.raises(ValueError):
        character_mask(value, width=16, height=16)


@pytest.mark.parametrize("x,w", [(True, 1), ("0", 1), (float("nan"), 1), (-1, 1), (0, 0), (15, 2)])
def test_invalid_character_coordinates(x: object, w: object) -> None:
    with pytest.raises(ValueError):
        character_mask(
            {
                "width": 16,
                "height": 16,
                "characterBoxes": [{"x": x, "y": 0, "width": w, "height": 1}],
            },
            width=16,
            height=16,
        )


def test_native_process_uses_exact_png_and_explicit_request(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    executable = tmp_path / "native"
    executable.touch()
    process = Mock(
        return_value=subprocess.CompletedProcess(
            [], 0, json.dumps([{"width": 16, "height": 16, "characterBoxes": []}]).encode(), b""
        )
    )
    monkeypatch.setattr("replay.subtitle_vision.subprocess.run", process)
    mask = vision_text_mask(
        np.zeros((16, 16, 3), dtype=np.uint8), executable=executable, work_directory=tmp_path
    )
    assert np.count_nonzero(mask) == 0
    assert process.call_args.args[0][0:3] == [
        str(executable),
        "measure-media",
        "image_text_rectangles_batch",
    ]
    assert process.call_args.kwargs == {"check": True, "capture_output": True, "timeout": 60}
    assert sorted(p.name for p in tmp_path.iterdir()) == ["native"]


def test_missing_native_tool_does_not_use_color_fallback(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        vision_text_mask(
            np.zeros((16, 16, 3), dtype=np.uint8),
            executable=tmp_path / "missing",
            work_directory=tmp_path,
        )


def test_invalid_image_rejected_before_native_call(tmp_path: Path) -> None:
    executable = tmp_path / "native"
    executable.touch()
    with pytest.raises(ValueError, match="uint8 BGR"):
        vision_text_mask(
            np.zeros((4, 4), dtype=np.uint8), executable=executable, work_directory=tmp_path
        )


def test_png_encoding_failure(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    executable = tmp_path / "native"
    executable.touch()
    monkeypatch.setattr(
        "replay.subtitle_vision.cv2.imencode",
        Mock(return_value=(False, np.zeros(1, dtype=np.uint8))),
    )
    with pytest.raises(ValueError, match="encode"):
        vision_text_mask(
            np.zeros((16, 16, 3), dtype=np.uint8), executable=executable, work_directory=tmp_path
        )


def test_excessive_response_rejected(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    executable = tmp_path / "native"
    executable.touch()
    monkeypatch.setattr(
        "replay.subtitle_vision.subprocess.run",
        Mock(return_value=subprocess.CompletedProcess([], 0, b" " * 1048577, b"")),
    )
    with pytest.raises(ValueError, match="Excessive"):
        vision_text_mask(
            np.zeros((16, 16, 3), dtype=np.uint8), executable=executable, work_directory=tmp_path
        )


def test_vision_mask_removes_white_text_without_selecting_other_pink_pixels() -> None:
    image = np.full((32, 32, 3), 100, dtype=np.uint8)
    image[10:14, 10:14] = 255
    image[25:28, 25:28] = (180, 50, 220)
    mask = np.zeros((32, 32), dtype=np.uint8)
    mask[10:14, 10:14] = 255
    detector = Mock(return_value=mask)
    measure = Mock(
        side_effect=[RepairEvidence(True, 0.2, 0, True), RepairEvidence(True, 0, 0, False)]
    )
    result = repair_with_retries(image, measure=measure, softness=0.5, detect=detector)
    assert detector.call_count == 1
    assert len(result.attempts) == 2
    assert np.max(result.image[10:14, 10:14]) < 120
    np.testing.assert_array_equal(result.image[25:28, 25:28], image[25:28, 25:28])
    assert not detector.call_args.args[0].flags.writeable
    assert np.count_nonzero(mask) == 16


@pytest.mark.parametrize(
    "mask",
    [
        np.zeros((8, 8), dtype=np.uint8),
        np.zeros((16, 16), dtype=np.float32),
        np.ones((16, 16), dtype=np.uint8),
    ],
)
def test_invalid_supplied_mask(mask: object) -> None:
    # Object values deliberately cross the runtime boundary through the detector.
    detector = Mock(return_value=mask)
    with pytest.raises(ValueError, match="mask"):
        repair_with_retries(
            np.zeros((16, 16, 3), dtype=np.uint8), measure=Mock(), softness=0.5, detect=detector
        )


def test_empty_measured_mask_does_not_modify_pixels() -> None:
    source = np.full((16, 16, 3), 200, dtype=np.uint8)
    result, mask = repair_glyphs(
        source, settings=InpaintSettings(), text_mask=np.zeros((16, 16), dtype=np.uint8)
    )
    np.testing.assert_array_equal(result, source)
    assert np.count_nonzero(mask) == 0


@pytest.mark.parametrize("count", [0, 9])
def test_native_batch_budget(count: int) -> None:
    with pytest.raises(ValueError, match="1–8"):
        vision_text_masks((np.zeros((8, 8, 3), dtype=np.uint8),) * count)


def test_native_batch_preserves_order(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    executable = tmp_path / "native"
    executable.touch()
    process = Mock(
        return_value=subprocess.CompletedProcess(
            [],
            0,
            json.dumps(
                [
                    {
                        "width": 8,
                        "height": 8,
                        "characterBoxes": [{"x": 2, "y": 3, "width": 1, "height": 1}],
                    },
                    {"width": 16, "height": 8, "characterBoxes": []},
                ]
            ).encode(),
            b"",
        )
    )
    monkeypatch.setattr("replay.subtitle_vision.subprocess.run", process)
    masks = vision_text_masks(
        (np.zeros((8, 8, 3), dtype=np.uint8), np.zeros((8, 16, 3), dtype=np.uint8)),
        executable=executable,
        work_directory=tmp_path,
    )
    assert process.call_count == 1
    assert masks[0][3, 2] == 255
    assert np.count_nonzero(masks[0]) == 1
    assert masks[1].shape == (8, 16)
    assert np.count_nonzero(masks[1]) == 0


def test_debug_trace_checks_pixels_and_distinguishes_empty_detection(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    executable = tmp_path / "native"
    executable.touch()
    monkeypatch.setattr(
        "replay.subtitle_vision.subprocess.run",
        Mock(
            return_value=subprocess.CompletedProcess(
                [],
                0,
                json.dumps(
                    [
                        {
                            "width": 8,
                            "height": 8,
                            "characterBoxes": [],
                            "diagnostics": {"observationCount": 0, "requestRevision": 1},
                        }
                    ]
                ).encode(),
                b"",
            )
        ),
    )
    image = np.full((8, 8, 3), (17, 63, 241), dtype=np.uint8)
    with caplog.at_level(logging.DEBUG, logger="replay.subtitle_vision"):
        mask = vision_text_mask(image, executable=executable, work_directory=tmp_path)
    assert np.count_nonzero(mask) == 0
    assert "round_trip_equal=true channel_min=[17, 63, 241]" in caplog.text
    assert "'observationCount': 0" in caplog.text
    assert "mask_pixels=0 text_absence_certified=false" in caplog.text
    assert "vision native end seconds=" in caplog.text
    assert str(tmp_path) not in caplog.text


@pytest.mark.parametrize("failure", [OSError("failure"), subprocess.TimeoutExpired("native", 60)])
def test_debug_trace_retains_failure_stage_without_swallowing_error(
    failure: Exception,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    executable = tmp_path / "native"
    executable.touch()
    monkeypatch.setattr("replay.subtitle_vision.subprocess.run", Mock(side_effect=failure))
    with (
        caplog.at_level(logging.DEBUG, logger="replay.subtitle_vision"),
        pytest.raises(type(failure)),
    ):
        vision_text_mask(
            np.zeros((8, 8, 3), dtype=np.uint8), executable=executable, work_directory=tmp_path
        )
    assert "vision native failed kind=" in caplog.text
    assert "vision batch end" not in caplog.text


@pytest.mark.parametrize("decoded", [None, np.ones((8, 8, 3), dtype=np.uint8)])
def test_debug_trace_rejects_changed_png_pixels(
    decoded: object,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    executable = tmp_path / "native"
    executable.touch()
    process = Mock()
    monkeypatch.setattr("replay.subtitle_vision.subprocess.run", process)
    monkeypatch.setattr("replay.subtitle_vision.cv2.imdecode", Mock(return_value=decoded))
    with (
        caplog.at_level(logging.DEBUG, logger="replay.subtitle_vision"),
        pytest.raises(ValueError, match="round-trip"),
    ):
        vision_text_mask(
            np.zeros((8, 8, 3), dtype=np.uint8), executable=executable, work_directory=tmp_path
        )
    process.assert_not_called()


@pytest.mark.parametrize("response", [b"{}", b"[]"])
def test_native_batch_rejects_count_mismatch(
    response: bytes, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    executable = tmp_path / "native"
    executable.touch()
    monkeypatch.setattr(
        "replay.subtitle_vision.subprocess.run",
        Mock(return_value=subprocess.CompletedProcess([], 0, response, b"")),
    )
    with pytest.raises(ValueError, match="result count"):
        vision_text_masks(
            (np.zeros((8, 8, 3), dtype=np.uint8),), executable=executable, work_directory=tmp_path
        )
