"""Synthetic native decode and malformed clock receipts, without private media."""

import io
import subprocess
from fractions import Fraction
from pathlib import Path
from unittest.mock import Mock

import pytest

from replay.subtitle_clock import VideoFrameClock
from replay.subtitle_decode import verify_native_decoded_window


@pytest.mark.parametrize(
    "trace",
    [
        "",
        "0,0,0,1,1,0x0\n",
        "#tb 0: 1/30\n0,0,0,1,1,0x0\n",
        "#tb 0: 0\n",
        "#tb 0: 1/30\n#tb 0: 1/30\n",
        "#tb 0: 1/30\n#codec_id 0: rawvideo\n0,0,0,1,1,0x0\n",
        "#tb 0: 1/30\n#codec_id 0: wrapped_avframe\n0,0,1,1,1,0x0\n",
        "#tb 0: 1/30\n#codec_id 0: wrapped_avframe\n1,0,0,1,1,0x0\n",
        "#tb 0: 1/30\n#codec_id 0: wrapped_avframe\n0,0,0\n",
        "#tb 0: 1/30\n#codec_id 0: wrapped_avframe\n0,0,0,1,1,0x0\n0,0,0,1,1,0x0\n",
    ],
)
def test_bad_decoded_trace_is_not_coverage(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, trace: str
) -> None:
    source = tmp_path / "source.mp4"
    source.write_bytes(b"mock source")

    def execute(_command: object, **options: object) -> None:
        stream = options["stdout"]
        assert isinstance(stream, io.BufferedWriter)
        stream.write(trace.encode("utf-8"))

    monkeypatch.setattr("replay.subtitle_decode.subprocess.run", execute)
    with pytest.raises(ValueError):
        verify_native_decoded_window(
            source,
            VideoFrameClock((0,), Fraction(1, 30), 1),
            evidence_directory=tmp_path / "evidence",
            timeout_seconds=30,
        )
    assert (tmp_path / "evidence" / "frames.framecrc").is_file()


@pytest.mark.parametrize("deadline", [0, 86401, True])
def test_invalid_deadline_does_not_create_evidence(tmp_path: Path, deadline: int) -> None:
    with pytest.raises(ValueError, match="deadline"):
        verify_native_decoded_window(
            tmp_path / "source",
            VideoFrameClock((0,), Fraction(1, 30), 1),
            evidence_directory=tmp_path / "evidence",
            timeout_seconds=deadline,
        )
    assert not (tmp_path / "evidence").exists()


def test_symlink_source_is_rejected(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.symlink_to(tmp_path / "missing")
    with pytest.raises(ValueError, match="regular local source"):
        verify_native_decoded_window(
            source,
            VideoFrameClock((0,), Fraction(1, 30), 1),
            evidence_directory=tmp_path / "evidence",
            timeout_seconds=30,
        )
    assert not (tmp_path / "evidence").exists()


def test_nonzero_diagnostics_are_not_suppressed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "source"
    source.write_bytes(b"mock source")

    def execute(_command: object, **options: object) -> None:
        stream = options["stderr"]
        assert isinstance(stream, io.BufferedWriter)
        stream.write(b"decoder reported an error\n")

    monkeypatch.setattr("replay.subtitle_decode.subprocess.run", execute)
    with pytest.raises(ValueError, match="diagnostics"):
        verify_native_decoded_window(
            source,
            VideoFrameClock((0,), Fraction(1, 30), 1),
            evidence_directory=tmp_path / "evidence",
            timeout_seconds=30,
        )
    assert (tmp_path / "evidence" / "decode.stderr").read_bytes() == b"decoder reported an error\n"


@pytest.mark.parametrize(
    "error", [subprocess.CalledProcessError(1, "ffmpeg"), subprocess.TimeoutExpired("ffmpeg", 1)]
)
def test_native_failure_retains_attempt_without_retry(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    error: subprocess.CalledProcessError | subprocess.TimeoutExpired,
) -> None:
    source = tmp_path / "source"
    source.write_bytes(b"mock source")
    execute = Mock(side_effect=error)
    monkeypatch.setattr("replay.subtitle_decode.subprocess.run", execute)
    with pytest.raises(type(error)):
        verify_native_decoded_window(
            source,
            VideoFrameClock((0,), Fraction(1, 30), 1),
            evidence_directory=tmp_path / "evidence",
            timeout_seconds=30,
        )
    with pytest.raises(FileExistsError):
        verify_native_decoded_window(
            source,
            VideoFrameClock((0,), Fraction(1, 30), 1),
            evidence_directory=tmp_path / "evidence",
            timeout_seconds=30,
        )
    assert execute.call_count == 1
    assert (tmp_path / "evidence" / "frames.framecrc").is_file()


def test_real_native_decode_retains_close_frames_and_nonzero_window(tmp_path: Path) -> None:
    source = tmp_path / "source.mp4"
    clock = VideoFrameClock((0, 2970, 2971, 6000, 11970, 551970), Fraction(1, 90000), 2910)
    with clock.encoding_options() as options:
        subprocess.run(
            [
                "ffmpeg",
                "-v",
                "error",
                "-nostdin",
                "-f",
                "lavfi",
                "-i",
                "testsrc2=s=128x128:r=30:d=0.2",
                "-c:v",
                "hevc_videotoolbox",
                "-b:v",
                "0",
                "-q:v",
                "65",
                "-bf",
                "0",
                "-tag:v",
                "hvc1",
                *options,
                str(source),
            ],
            capture_output=True,
            check=True,
            timeout=30,
        )
    assert (
        verify_native_decoded_window(
            source,
            clock,
            evidence_directory=tmp_path / "all",
            timeout_seconds=30,
        )
        == 6
    )
    assert (
        verify_native_decoded_window(
            source,
            VideoFrameClock((11970, 551970), Fraction(1, 90000), 2910),
            evidence_directory=tmp_path / "tail",
            timeout_seconds=30,
        )
        == 2
    )
    assert (tmp_path / "all").stat().st_mode & 0o777 == 0o700
    assert (tmp_path / "all" / "frames.framecrc").stat().st_mode & 0o777 == 0o600
    assert (tmp_path / "all" / "decode.stderr").read_bytes() == b""
