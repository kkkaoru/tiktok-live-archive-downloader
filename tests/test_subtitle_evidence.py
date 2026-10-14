from fractions import Fraction
from pathlib import Path

import numpy as np
import pytest
from pytest import MonkeyPatch

from replay import subtitle_evidence
from replay.subtitle_clock import VideoFrameClock
from replay.subtitle_evidence import RepairCapture
from replay.subtitle_stream import SubtitleSpan, repair_stream
from replay.subtitle_video import spatial_evidence


def test_real_repair_writer_capture(tmp_path: Path, monkeypatch: MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    capture = RepairCapture(
        tmp_path / "capture", [0], VideoFrameClock((40, 43), Fraction(1, 30), 3)
    )
    source = np.full((16, 16, 3), 80, dtype=np.uint8)
    source[7:9, 7:9] = 255
    mask = np.zeros((16, 16), dtype=np.uint8)
    mask[7:9, 7:9] = 255
    written: list[np.ndarray[tuple[int, ...], np.dtype[np.uint8]]] = []
    repair_stream(
        (source, source),
        spans=(SubtitleSpan(0, 2, 0, 0, 16, 16),),
        expected_frames=2,
        measure=spatial_evidence,
        quality_available=False,
        detect=lambda image: mask,
        write=written.append,
        receipt=lambda *args: None,
        evidence=capture,
    )
    manifest = capture.manifest()
    assert manifest["qualityApproval"] is False
    assert manifest["temporalRegisteredSupport"] is None
    assert capture.records[0]["sourcePTS"] == 40
    assert capture.records[0]["candidatePTS"] == 0
    assert capture.records[0]["glyphPixels"] == 4
    with np.load(tmp_path / "capture/00000000.npz", allow_pickle=False) as arrays:
        np.testing.assert_array_equal(arrays["candidate"], written[0])
        np.testing.assert_array_equal(arrays["source"], source)
        np.testing.assert_array_equal(arrays["glyph"], mask)
        assert np.count_nonzero(arrays["repair"]) >= 4
        assert not np.array_equal(arrays["candidate"], source)
    assert (tmp_path / "capture/00000000.npz").stat().st_mode & 0o777 == 0o600
    (tmp_path / "capture/00000000.npz").write_bytes(b"changed")
    with pytest.raises(ValueError, match="changed"):
        capture.manifest()


@pytest.mark.parametrize("frames", [None, [], [True], [-1], [2], [0, 0], [1, 0]])
def test_invalid_selection(tmp_path: Path, monkeypatch: MonkeyPatch, frames: object) -> None:
    monkeypatch.chdir(tmp_path)
    with pytest.raises(ValueError):
        RepairCapture(tmp_path / "capture", frames, VideoFrameClock((0, 1), Fraction(1, 30), 1))


def test_missing_and_escaping(tmp_path: Path, monkeypatch: MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    clock = VideoFrameClock((0,), Fraction(1, 30), 1)
    with pytest.raises(ValueError, match="checkout-local"):
        RepairCapture(tmp_path.parent / "escape", [0], clock)
    capture = RepairCapture(tmp_path / "capture", [0], clock)
    with pytest.raises(ValueError, match="not all"):
        capture.manifest()


@pytest.mark.parametrize("budget,geometry", [(1, False), (1024, True)])
def test_capture_rejects_budget_and_geometry(
    tmp_path: Path, monkeypatch: MonkeyPatch, budget: int, geometry: bool
) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(subtitle_evidence, "MAX_EVIDENCE_BYTES", budget)
    capture = RepairCapture(tmp_path / "capture", [0], VideoFrameClock((0,), Fraction(1, 30), 1))
    image = np.zeros((8, 8, 3), dtype=np.uint8)
    if geometry:
        capture.frames = (1,)
    with pytest.raises(ValueError):
        repair_stream(
            (image,),
            spans=(SubtitleSpan(0, 1, 0, 0, 8, 8),),
            expected_frames=1,
            measure=spatial_evidence,
            quality_available=False,
            detect=lambda image: np.full((8, 8), 255, dtype=np.uint8),
            write=lambda image: None,
            receipt=lambda *args: None,
            evidence=capture,
        )
        capture.manifest()
