"""Real synthetic FFmpeg streaming tests; no private media or network."""

import subprocess
from pathlib import Path

import numpy as np
import pytest

from replay.subtitle_stream import SubtitleSpan
from replay.subtitle_video import VideoRepairJob, render_repaired_video, spatial_evidence


@pytest.mark.parametrize("width,frames,fps", [(0, 1, 30), (4097, 1, 30), (16, 1, 121)])
def test_invalid_video_budget(width: int, frames: int, fps: int) -> None:
    with pytest.raises(ValueError):
        VideoRepairJob(Path("source"), Path("output"), width, 4096, frames, (), fps)


def test_source_replacement_refused() -> None:
    with pytest.raises(ValueError, match="replace source"):
        VideoRepairJob(Path("source"), Path("source"), 16, 16, 1, ())


def test_missing_input_refused(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="Missing"):
        render_repaired_video(VideoRepairJob(tmp_path / "absent", tmp_path / "out", 16, 16, 1, ()))


def test_existing_output_preserved(tmp_path: Path) -> None:
    source, output = tmp_path / "source", tmp_path / "output"
    source.write_bytes(b"source")
    output.write_bytes(b"keep")
    with pytest.raises(FileExistsError):
        render_repaired_video(VideoRepairJob(source, output, 16, 16, 1, ()))
    assert output.read_bytes() == b"keep"


@pytest.mark.parametrize("hardware", [False, True])
def test_real_stream_repairs_and_decodes(tmp_path: Path, hardware: bool) -> None:
    source, output = tmp_path / "source.mp4", tmp_path / "output.mp4"
    subprocess.run(
        [
            "ffmpeg",
            "-v",
            "error",
            "-nostdin",
            "-f",
            "lavfi",
            "-i",
            "color=gray:s=64x64:r=30,drawbox=x=20:y=20:w=8:h=8:color=magenta:t=fill",
            "-frames:v",
            "4",
            "-c:v",
            "libx264",
            str(source),
        ],
        check=True,
        capture_output=True,
    )
    receipt = render_repaired_video(
        VideoRepairJob(
            source,
            output,
            64,
            64,
            4,
            (SubtitleSpan(0, 4, 8, 8, 48, 48),),
            hardware_encode=hardware,
        )
    )
    assert receipt.frames == 4
    assert receipt.softened_frames == 4
    assert receipt.attempts == 4
    assert not receipt.quality_measured
    assert receipt.seconds > 0
    result = subprocess.run(
        ["ffmpeg", "-v", "error", "-xerror", "-nostdin", "-i", str(output), "-f", "null", "-"],
        check=True,
        capture_output=True,
    )
    assert result.stderr == b""


@pytest.mark.parametrize("expected,message", [(1, "excess"), (3, "before expected")])
def test_wrong_frame_count_refused(tmp_path: Path, expected: int, message: str) -> None:
    source, output = tmp_path / "source.mp4", tmp_path / "output.mp4"
    subprocess.run(
        [
            "ffmpeg",
            "-v",
            "error",
            "-nostdin",
            "-f",
            "lavfi",
            "-i",
            "color=gray:s=16x16:r=30",
            "-frames:v",
            "2",
            "-c:v",
            "libx264",
            str(source),
        ],
        check=True,
        capture_output=True,
    )
    with pytest.raises(ValueError, match=message):
        render_repaired_video(VideoRepairJob(source, output, 16, 16, expected, ()))


def test_explicit_native_padding_trim(tmp_path: Path) -> None:
    source, output = tmp_path / "source.mp4", tmp_path / "output.mp4"
    subprocess.run(
        [
            "ffmpeg",
            "-v",
            "error",
            "-nostdin",
            "-f",
            "lavfi",
            "-i",
            "color=gray:s=16x16:r=30",
            "-frames:v",
            "2",
            "-c:v",
            "libx264",
            str(source),
        ],
        check=True,
        capture_output=True,
    )
    receipt = render_repaired_video(
        VideoRepairJob(source, output, 16, 16, 1, (), allow_padded_tail=True)
    )
    assert receipt.frames == 1
    assert receipt.attempts == 0


def test_spatial_measurement_does_not_claim_ocr() -> None:
    image = np.zeros((16, 16, 3), dtype=np.uint8)
    evidence = spatial_evidence(image, image, np.zeros((16, 16), dtype=np.uint8))
    assert evidence.technically_valid
    assert evidence.readable_text is None
    assert evidence.unstable_fraction is None
