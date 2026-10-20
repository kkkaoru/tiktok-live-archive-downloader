"""The band audit reads tiled caption bands and maps every line back to its own audit frame."""

import json
import subprocess
from fractions import Fraction
from pathlib import Path

import numpy as np
import pytest

from replay import subtitle_band_audit as audit
from replay.caption_io import FrameObservation
from replay.caption_masks import TextDetection, TextRectangle
from replay.core import ReplayError
from replay.subtitle_clock import VideoFrameClock
from replay.subtitle_sampled_chunk import OutputLimitError

CLOCK = VideoFrameClock(tuple(range(0, 3000 * 12, 3000)), Fraction(1, 90000), 3000)


def test_the_band_holds_the_zone_grown_by_half_its_height_and_the_margin() -> None:
    assert audit.audit_band((0, 1283, 1077, 167), (1080, 1920)) == (1120, 492)


def test_the_band_is_clipped_to_the_canvas_with_an_even_height() -> None:
    assert audit.audit_band((0, 1800, 1077, 101), (1080, 1920)) == (1670, 250)


def line(y: float, text: str = "字幕") -> TextDetection:
    return TextDetection(text, 0.95, TextRectangle(100.0, y, 300.0, 40.0))


def test_lines_return_to_their_own_frame_in_canvas_coordinates() -> None:
    observed = {
        0: FrameObservation(0.0, 0.0, 1080, 300, (line(20.0, "a"), line(120.0, "b"))),
        1: FrameObservation(0.0, 0.0, 1080, 300, (line(220.0, "c"),)),
    }
    result = audit.untile(
        observed, [0, 5, 10, 11], tile=3, band=(1000, 100), canvas=(1080, 1920), clock=CLOCK
    )
    assert [d.text for d in result[0].detections] == ["a"]
    assert [d.text for d in result[5].detections] == ["b"]
    assert result[10].detections == ()
    assert result[5].detections[0].rectangle == TextRectangle(100.0, 1020.0, 300.0, 40.0)
    assert result[5].requested_time == pytest.approx(5 / 30)


def test_a_line_on_the_black_padding_after_the_last_frame_is_ignored() -> None:
    observed = {0: FrameObservation(0.0, 0.0, 1080, 300, (line(220.0),))}
    result = audit.untile(
        observed, [0, 5], tile=3, band=(1000, 100), canvas=(1080, 1920), clock=CLOCK
    )
    assert result[0].detections == () and result[5].detections == ()


def video(tmp_path: Path) -> Path:
    path = tmp_path / "v.mp4"
    subprocess.run(
        ["ffmpeg", "-nostdin", "-v", "error", "-f", "lavfi", "-i",
         "testsrc2=size=320x240:rate=30:duration=1", "-pix_fmt", "yuv420p", str(path)],
        check=True, timeout=60,
    )  # fmt: skip
    return path


def test_band_frames_keep_only_the_audited_frames(tmp_path: Path) -> None:
    frames = audit.band_frames(
        video(tmp_path), [0, 5, 29], canvas=(320, 240), band=(100, 40), ffmpeg="ffmpeg"
    )
    assert [frame.shape for frame in frames] == [(40, 320, 3)] * 3


def test_band_frames_refuse_unordered_or_missing_indices(tmp_path: Path) -> None:
    source = video(tmp_path)
    with pytest.raises(ReplayError, match="ascending and unique"):
        audit.band_frames(source, [5, 0], canvas=(320, 240), band=(100, 40), ffmpeg="ffmpeg")
    with pytest.raises(ReplayError, match="every audited band"):
        audit.band_frames(source, [0, 99], canvas=(320, 240), band=(100, 40), ffmpeg="ffmpeg")


def test_tiles_are_stacked_padded_and_timed(tmp_path: Path) -> None:
    bands = [np.full((40, 320, 3), value, dtype=np.uint8) for value in (10, 20, 30, 40)]
    movie = tmp_path / "t.mov"
    assert audit.write_tiles(bands, 3, movie, ffmpeg="ffmpeg") == 2
    clock = audit.movie_clock(movie, ffprobe="ffprobe")
    assert len(clock.pts) == 2
    with pytest.raises(ReplayError, match="Invalid audit tile"):
        audit.write_tiles(bands, 5, tmp_path / "x.mov", ffmpeg="ffmpeg")


def test_a_single_image_movie_has_a_clock(tmp_path: Path) -> None:
    movie = tmp_path / "one.mov"
    audit.write_tiles([np.zeros((40, 320, 3), dtype=np.uint8)], 1, movie, ffmpeg="ffmpeg")
    assert len(audit.movie_clock(movie, ffprobe="ffprobe").pts) == 1


def test_the_band_audit_recognises_tiles_and_records_its_band(tmp_path: Path) -> None:
    seen: list[tuple[int, int]] = []

    def recognise(movie: Path, tiles: VideoFrameClock, height: int) -> dict[int, FrameObservation]:
        seen.append((len(tiles.pts), height))
        return {0: FrameObservation(0.0, 0.0, 320, height, (line(10.0, "x"),))}

    result = audit.band_audit(
        video(tmp_path), [0, 5, 10], clock=CLOCK, canvas=(320, 240), envelope=(0, 100, 320, 20),
        tile=2, work=tmp_path / "audit", recognise=recognise,
    )  # fmt: skip
    assert seen == [(2, 2 * 200)]
    assert [d.text for d in result[0].detections] == ["x"]
    assert json.loads((tmp_path / "audit" / "band.json").read_text(encoding="utf-8")) == {
        "top": 10,
        "height": 200,
        "tile": 2,
        "indices": [0, 5, 10],
    }


def test_the_band_audit_hands_its_decoded_bands_on(tmp_path: Path) -> None:
    decoded: list[audit.Image] = []

    def recognise(movie: Path, tiles: VideoFrameClock, height: int) -> dict[int, FrameObservation]:
        return {0: FrameObservation(0.0, 0.0, 320, height, (line(10.0, "x"),))}

    audit.band_audit(
        video(tmp_path), [0, 5, 10], clock=CLOCK, canvas=(320, 240), envelope=(0, 100, 320, 20),
        tile=2, work=tmp_path / "audit", recognise=recognise, decoded=decoded,
    )  # fmt: skip
    assert [frame.shape for frame in decoded] == [(200, 320, 3)] * 3


def test_a_tiled_image_over_the_capture_bound_is_read_again_untiled(tmp_path: Path) -> None:
    seen: list[tuple[Path, int]] = []

    def recognise(movie: Path, tiles: VideoFrameClock, height: int) -> dict[int, FrameObservation]:
        seen.append((movie, len(tiles.pts)))
        if height > 200:
            raise OutputLimitError("Native OCR session failed; evidence retained")
        return {2: FrameObservation(0.0, 0.0, 320, height, (line(10.0, "x"),))}

    result = audit.band_audit(
        video(tmp_path), [0, 5, 10], clock=CLOCK, canvas=(320, 240), envelope=(0, 100, 320, 20),
        tile=2, work=tmp_path / "audit", recognise=recognise,
    )  # fmt: skip
    assert seen == [
        (tmp_path / "audit" / "band-tiles.mov", 2),
        (tmp_path / "audit" / "untiled" / "band-tiles.mov", 3),
    ]
    assert [d.text for d in result[10].detections] == ["x"]


def test_an_untiled_image_over_the_capture_bound_still_fails(tmp_path: Path) -> None:
    def recognise(movie: Path, tiles: VideoFrameClock, height: int) -> dict[int, FrameObservation]:
        raise OutputLimitError("Native OCR session failed; evidence retained")

    with pytest.raises(OutputLimitError):
        audit.band_audit(
            video(tmp_path), [0, 5], clock=CLOCK, canvas=(320, 240), envelope=(0, 100, 320, 20),
            tile=1, work=tmp_path / "audit", recognise=recognise,
        )  # fmt: skip
