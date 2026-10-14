"""Synthetic signal interpretations; no current Apple renderer acceptance."""

import json
import runpy
import sys
from pathlib import Path

import numpy as np
import pytest
from numpy.typing import NDArray

from replay import caption_signal
from replay.caption_signal import SignalModel
from replay.core import ReplayError


@pytest.mark.parametrize("model", ["bt709-camera", "bt1886-display", "apple-video-1.961"])
def test_endpoints_and_low_signal(model: SignalModel) -> None:
    assert caption_signal.signal_to_srgb((0, 0, 0), model=model) == (0, 0, 0)
    assert caption_signal.signal_to_srgb((1, 1, 1), model=model) == pytest.approx((1, 1, 1))
    assert caption_signal.signal_to_srgb((0.001, 0.001, 0.001), model=model)[0] > 0


def test_camera_toe_is_not_display_gamma() -> None:
    assert caption_signal.signal_to_srgb(
        (0.009, 0.009, 0.009), model="bt709-camera"
    ) == pytest.approx((0.02584, 0.02584, 0.02584))
    assert caption_signal.signal_to_srgb((0.5, 0.5, 0.5), model="bt1886-display") == pytest.approx(
        (0.4725, 0.4725, 0.4725), abs=1e-12
    )


@pytest.mark.parametrize(
    "signal",
    [
        (0, 0),
        (True, 0, 0),
        (-0.01, 0, 0),
        (1.001, 0, 0),
        (float("nan"), 0, 0),
        (float("inf"), 0, 0),
    ],
)
def test_invalid_or_out_of_gamut_signal_is_not_clipped(signal: tuple[float, ...]) -> None:
    with pytest.raises(ReplayError, match="do not clip"):
        caption_signal.signal_to_srgb(signal, model="bt709-camera")


def test_unknown_model_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(caption_signal, "MODEL_GAMMAS", {})
    with pytest.raises(ReplayError, match="Unknown"):
        caption_signal.signal_to_srgb((0, 0, 0), model="bt709-camera")


def test_counts_preserve_16bit_codes_and_never_approve() -> None:
    image = np.array(
        [[[1, 2, 3, 65535], [1, 2, 3, 65535], [4, 5, 6, 65534]]], dtype=caption_signal.SIGNAL_DTYPE
    )
    report = caption_signal.measure_signal(image)
    assert report["opaquePixels"] == 2
    assert report["reportedPixels"] == 2
    assert report["uniqueOpaqueCodes"] == 1
    assert report["transferInterpretationVerified"] is False
    assert report["renderedContrastVerified"] is False
    assert report["captionQualityApproved"] is False
    records = report["dominantOpaqueCodes"]
    assert isinstance(records, list)
    assert records[0]["rgb16"] == [1, 2, 3]
    assert records[0]["count"] == 2
    assert records[0]["models"]["bt709-camera"]["suppliedSampleCorrespondence"] is None
    assert records[0]["models"]["bt709-camera"]["sRGB"][0] > 0


def test_intent_is_checked_but_decorations_do_not_abort_diagnostic() -> None:
    image = np.array(
        [[[0, 0, 0, 65535], [65535, 65535, 65535, 65535]]], dtype=caption_signal.SIGNAL_DTYPE
    )
    report = caption_signal.measure_signal(image, intent=(0, 0, 0))
    records = report["dominantOpaqueCodes"]
    assert isinstance(records, list)
    assert records[0]["models"]["bt1886-display"]["suppliedSampleCorrespondence"] is True
    assert records[1]["models"]["bt1886-display"]["suppliedSampleCorrespondence"] is False
    with pytest.raises(ReplayError, match="below 7"):
        caption_signal.measure_signal(image, intent=(1, 1, 1))


def test_empty_opaque_region_is_explicit() -> None:
    report = caption_signal.measure_signal(np.zeros((1, 2, 4), dtype=caption_signal.SIGNAL_DTYPE))
    assert report["opaquePixels"] == 0
    assert report["dominantOpaqueCodes"] == []
    assert report["captionQualityApproved"] is False


def test_report_cap_does_not_imply_complete_coverage(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(caption_signal, "MAX_REPORTED_CODES", 1)
    image = np.array(
        [[[1, 2, 3, 65535], [1, 2, 3, 65535], [4, 5, 6, 65535]]], dtype=caption_signal.SIGNAL_DTYPE
    )
    report = caption_signal.measure_signal(image)
    assert report["opaquePixels"] == 3
    assert report["reportedPixels"] == 2
    assert report["uniqueOpaqueCodes"] == 2


@pytest.mark.parametrize(
    "image",
    [
        np.zeros((2, 3), dtype=np.uint16),
        np.zeros((1, 2, 3), dtype=np.uint16),
        np.zeros((0, 2, 4), dtype=np.uint16),
    ],
)
def test_invalid_image(image: NDArray[np.uint16]) -> None:
    with pytest.raises(ReplayError, match="RGBA64LE"):
        caption_signal.measure_signal(image)


def test_image_budget(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(caption_signal, "MAX_PIXELS", 1)
    with pytest.raises(ReplayError):
        caption_signal.measure_signal(np.zeros((1, 2, 4), dtype=caption_signal.SIGNAL_DTYPE))


def test_cli_is_private_exclusive_and_retains_exact_codes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source, output = tmp_path / "signal.rgba64", tmp_path / "out.json"
    source.write_bytes(b"\x01\0\x02\0\x03\0\xff\xff")
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "caption_signal",
            str(source),
            str(output),
            "--width",
            "1",
            "--height",
            "1",
            "--intent",
            "24556A",
        ],
    )
    runpy.run_path(caption_signal.__file__, run_name="__main__")
    report = json.loads(output.read_text(encoding="utf-8"))
    assert report["dominantOpaqueCodes"][0]["rgb16"] == [1, 2, 3]
    assert (
        report["dominantOpaqueCodes"][0]["models"]["bt709-camera"]["suppliedSampleCorrespondence"]
        is False
    )
    assert len(report["sourceSHA256"]) == 64
    assert output.stat().st_mode & 0o777 == 0o600
    with pytest.raises(FileExistsError):
        caption_signal.main()
    assert source.read_bytes() == b"\x01\0\x02\0\x03\0\xff\xff"


@pytest.mark.parametrize("width,height", [(0, 1), (-1, -1), (9000, 9000)])
def test_cli_invalid_dimensions(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, width: int, height: int
) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "caption_signal",
            str(tmp_path / "none"),
            str(tmp_path / "out"),
            "--width",
            str(width),
            "--height",
            str(height),
        ],
    )
    with pytest.raises(ReplayError, match="pixel budget"):
        caption_signal.main()


@pytest.mark.parametrize("kind", ["missing", "short", "link", "changed-size", "changed-digest"])
def test_cli_bad_input(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, kind: str) -> None:
    source, output = tmp_path / "signal.rgba64", tmp_path / "out.json"
    if kind == "link":
        source.symlink_to(tmp_path / "absent")
    elif kind != "missing":
        source.write_bytes(b"x" if kind == "short" else b"12345678")
    if kind == "changed-size":
        monkeypatch.setattr(Path, "read_bytes", lambda _: b"12")
    if kind == "changed-digest":
        digests = iter(["a" * 64, "b" * 64])
        monkeypatch.setattr(caption_signal, "file_digest", lambda _: next(digests))
    monkeypatch.setattr(
        sys, "argv", ["caption_signal", str(source), str(output), "--width", "1", "--height", "1"]
    )
    with pytest.raises(ReplayError):
        caption_signal.main()
    assert not output.exists()
