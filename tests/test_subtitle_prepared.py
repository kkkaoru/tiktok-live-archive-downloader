import json
import subprocess
from pathlib import Path

import cv2
import numpy as np
import pytest
from pytest import MonkeyPatch

from replay import subtitle_prepared
from replay.subtitle_inpaint import Image


def image() -> Image:
    result = np.zeros((64, 64, 3), dtype=np.uint8)
    result[16:32, 16:32] = 255
    return result


@pytest.fixture
def native(tmp_path: Path, monkeypatch: MonkeyPatch) -> tuple[Path, dict[str, object]]:
    executable = tmp_path / "native"
    executable.write_bytes(b"synthetic native executable")
    document: dict[str, object] = {
        "width": 64,
        "height": 64,
        "lines": [
            {
                "text": "字幕",
                "confidence": 0.9,
                "boundsClipped": False,
                "region": {"x": 16, "y": 16, "width": 16, "height": 16},
            }
        ],
    }

    def run(
        command: list[str], *, capture_output: bool, check: bool, timeout: int
    ) -> subprocess.CompletedProcess[bytes]:
        assert command[1:3] == ["measure-media", "image_text_recognize_batch"]
        assert capture_output and check and timeout == 60
        request = json.loads(Path(command[3]).read_text())
        assert all(Path(path).is_file() for path in request["paths"])
        return subprocess.CompletedProcess(
            command, 0, json.dumps([document for _ in request["paths"]]).encode(), b""
        )

    monkeypatch.setattr(subprocess, "run", run)
    return executable, document


@pytest.mark.parametrize("clipped", [False, True])
def test_preparation_uses_measured_support_not_ocr_rectangle(
    native: tuple[Path, dict[str, object]], tmp_path: Path, monkeypatch: MonkeyPatch, clipped: bool
) -> None:
    native[1]["lines"] = [
        {
            "text": "字幕",
            "confidence": 0.9,
            "boundsClipped": clipped,
            "region": {"x": 16, "y": 16, "width": 16, "height": 16},
        }
    ]
    calls: list[int] = []

    def masks(
        images: tuple[Image, ...], *, executable: Path, work_directory: Path, text_regions: bool
    ) -> tuple[Image, ...]:
        assert executable == native[0] and text_regions
        calls.append(len(images))
        result: list[Image] = []
        for prepared in images:
            assert prepared.shape == (64, 64, 3)
            assert set(np.unique(prepared)) <= {0, 255}
            mask = np.zeros((64, 64), dtype=np.uint8)
            mask[8:12, 8:12] = 255
            result.append(mask)
        return tuple(result)

    monkeypatch.setattr(subtitle_prepared, "vision_text_masks", masks)
    source = image()
    before = source.copy()
    result = subtitle_prepared.prepared_text_masks(
        (source,) * 5, executable=native[0], work_directory=tmp_path / "tmp"
    )
    assert calls == [8, 2]
    expected = np.zeros((64, 64), dtype=np.uint8)
    expected[12:14, 12:14] = 255
    assert all(np.array_equal(mask, expected) for mask in result.masks)
    assert result.observed_lines == (1, 1, 1, 1, 1)
    assert result.clipped_lines == (int(clipped),) * 5
    assert np.array_equal(source, before)
    assert tuple((tmp_path / "tmp").iterdir()) == ()


@pytest.mark.parametrize(
    "lines",
    [
        [],
        [
            {
                "text": "123",
                "confidence": 1,
                "boundsClipped": False,
                "region": {"x": 16, "y": 16, "width": 16, "height": 16},
            }
        ],
    ],
)
def test_no_context_does_not_invent_mask(
    native: tuple[Path, dict[str, object]], tmp_path: Path, lines: list[dict[str, object]]
) -> None:
    native[1]["lines"] = lines
    masks = subtitle_prepared.prepared_text_masks(
        (image(),), executable=native[0], work_directory=tmp_path / "tmp"
    )
    assert not np.any(masks.masks[0])


def test_uniform_observed_context_is_not_silently_replaced(
    native: tuple[Path, dict[str, object]], tmp_path: Path
) -> None:
    with pytest.raises(ValueError, match="contrast"):
        subtitle_prepared.prepared_text_masks(
            (np.zeros((64, 64, 3), dtype=np.uint8),),
            executable=native[0],
            work_directory=tmp_path / "tmp",
        )


@pytest.mark.parametrize("field,value", [("width", 63), ("lines", [{"boundsClipped": "unknown"}])])
def test_observation_geometry_fails_closed(
    native: tuple[Path, dict[str, object]], tmp_path: Path, field: str, value: object
) -> None:
    native[1][field] = value
    with pytest.raises(ValueError):
        subtitle_prepared.image_observations(
            (image(),), executable=native[0], work_directory=tmp_path / "tmp"
        )


def test_input_budget(native: tuple[Path, dict[str, object]], tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="1–8"):
        subtitle_prepared.image_observations(
            (), executable=native[0], work_directory=tmp_path / "tmp"
        )
    with pytest.raises(ValueError, match="excessive subtitle ROI"):
        subtitle_prepared.image_observations(
            (np.zeros((2001, 1000, 3), dtype=np.uint8),),
            executable=native[0],
            work_directory=tmp_path / "tmp",
        )


def test_prepared_context_budget(native: tuple[Path, dict[str, object]], tmp_path: Path) -> None:
    native[1].update(
        width=6400,
        height=200,
        lines=[
            {
                "text": "字幕",
                "confidence": 0.9,
                "boundsClipped": False,
                "region": {"x": 0, "y": 40, "width": 6400, "height": 96},
            }
        ],
    )
    with pytest.raises(ValueError, match="context exceeds"):
        subtitle_prepared.prepared_text_masks(
            (np.zeros((200, 6400, 3), dtype=np.uint8),),
            executable=native[0],
            work_directory=tmp_path / "tmp",
        )


@pytest.mark.parametrize("payload", [b"[]", b"x" * (8 * 1024 * 1024 + 1)])
def test_response_budget(
    native: tuple[Path, dict[str, object]], tmp_path: Path, monkeypatch: MonkeyPatch, payload: bytes
) -> None:
    def run(command: list[str], **kwargs: object) -> subprocess.CompletedProcess[bytes]:
        return subprocess.CompletedProcess(command, 0, payload, b"")

    monkeypatch.setattr(subprocess, "run", run)
    with pytest.raises(ValueError):
        subtitle_prepared.image_observations(
            (image(),), executable=native[0], work_directory=tmp_path / "tmp"
        )
    assert tuple((tmp_path / "tmp").iterdir()) == ()


def test_encoding_failure(
    native: tuple[Path, dict[str, object]], tmp_path: Path, monkeypatch: MonkeyPatch
) -> None:
    monkeypatch.setattr(cv2, "imencode", lambda *args: (False, np.zeros(0, dtype=np.uint8)))
    with pytest.raises(ValueError, match="encode"):
        subtitle_prepared.image_observations(
            (image(),), executable=native[0], work_directory=tmp_path / "tmp"
        )
