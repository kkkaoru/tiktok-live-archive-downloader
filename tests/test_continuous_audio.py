"""Synthetic command and real-FFmpeg checks for uncorrected sample decoding."""

import json
import subprocess
import wave
from operator import itemgetter
from pathlib import Path

import numpy as np
import pytest

from replay.continuous_audio import continuous_decode_command
from replay.core import ReplayError


def test_command_does_not_add_audio_correction(tmp_path: Path) -> None:
    source = tmp_path / "source.wav"
    source.write_bytes(b"fixture")
    command = continuous_decode_command(source, tmp_path / "new.wav")
    assert command[:8] == [
        "ffmpeg",
        "-hide_banner",
        "-nostdin",
        "-v",
        "error",
        "-n",
        "-i",
        str(source),
    ]
    assert command[8:-1] == [
        "-map",
        "0:a:0",
        "-vn",
        "-af",
        "asetpts=N/SR/TB",
        "-c:a",
        "pcm_f32le",
    ]
    assert Path(command[-1]).name == "new.wav"


def test_missing_source_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(ReplayError, match="regular source"):
        continuous_decode_command(tmp_path / "absent", tmp_path / "out.wav")


def test_linked_source_is_rejected(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.write_bytes(b"fixture")
    linked = tmp_path / "linked"
    linked.symlink_to(source)
    with pytest.raises(ReplayError, match="regular source"):
        continuous_decode_command(linked, tmp_path / "out.wav")


@pytest.mark.parametrize("linked", [False, True])
def test_existing_or_dangling_output_is_rejected(tmp_path: Path, linked: bool) -> None:
    source = tmp_path / "source"
    source.write_bytes(b"fixture")
    target = tmp_path / "output"
    if linked:
        target.symlink_to(tmp_path / "absent")
    else:
        target.write_bytes(b"preserved")
    with pytest.raises(ReplayError, match="new output"):
        continuous_decode_command(source, target)


def test_missing_output_parent_is_rejected(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.write_bytes(b"fixture")
    with pytest.raises(ReplayError, match="existing directory"):
        continuous_decode_command(source, tmp_path / "absent" / "out.wav")


def test_native_timestamp_gap_does_not_insert_silence(tmp_path: Path) -> None:
    original = tmp_path / "original.wav"
    with wave.open(str(original), "wb") as writer:
        writer.setparams((1, 2, 48000, 0, "NONE", "not compressed"))
        writer.writeframes(b"\x00\x20" * 24000 + b"\x00\xe0" * 24000)
    discontinuous = tmp_path / "gapped.nut"
    subprocess.run(
        [
            "ffmpeg",
            "-nostdin",
            "-v",
            "error",
            "-n",
            "-i",
            str(original),
            "-af",
            "asetpts=PTS+gte(T\\,0.4)*0.25/TB",
            "-c:a",
            "pcm_s16le",
            str(discontinuous),
        ],
        check=True,
        capture_output=True,
        timeout=15,
    )
    timing = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "frame=pts_time,nb_samples",
            "-of",
            "json",
            str(discontinuous),
        ],
        check=True,
        capture_output=True,
        timeout=15,
    )
    frames = json.loads(timing.stdout)["frames"]
    pts = np.fromiter(map(float, map(itemgetter("pts_time"), frames)), dtype=float)
    samples = np.fromiter(map(int, map(itemgetter("nb_samples"), frames)), dtype=int)
    assert np.max(np.diff(pts) - samples[:-1] / 48000) == pytest.approx(0.25, abs=1e-6)
    output = tmp_path / "continuous.wav"
    subprocess.run(
        continuous_decode_command(discontinuous, output),
        check=True,
        capture_output=True,
        timeout=15,
    )
    decoded = subprocess.run(
        [
            "ffmpeg",
            "-nostdin",
            "-v",
            "error",
            "-i",
            str(output),
            "-c:a",
            "pcm_s16le",
            "-f",
            "s16le",
            "-",
        ],
        check=True,
        capture_output=True,
        timeout=15,
    )
    assert decoded.stderr == b""
    assert decoded.stdout == b"\x00\x20" * 24000 + b"\x00\xe0" * 24000
