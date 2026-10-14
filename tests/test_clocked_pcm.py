import json
import shutil
import subprocess
import wave
from dataclasses import replace
from fractions import Fraction
from pathlib import Path

import numpy as np
import pytest

from replay.clocked_pcm import (
    ClockedSpan,
    observed_pcm_mov_command,
    reader_placements,
    validate_placements,
    write_clocked_pcm,
)
from replay.stages import file_digest


def test_actual_reader_support_is_not_projected_packet_metadata() -> None:
    spans = reader_placements(
        {
            "mode": "track",
            "buffers": [
                {
                    "value": 48000,
                    "timescale": 48000,
                    "durationValue": 1600,
                    "durationTimescale": 48000,
                    "samples": 1600,
                },
                {
                    "value": 51200,
                    "timescale": 48000,
                    "durationValue": 1600,
                    "durationTimescale": 48000,
                    "samples": 1600,
                },
            ],
        },
        sample_rate=48000,
        source_frames=3200,
        origin=Fraction(1),
        input_clock="reader-pcm-ordinal",
    )
    assert spans == (
        ClockedSpan(0, 1600, Fraction(1), Fraction(0)),
        ClockedSpan(1600, 1600, Fraction(16, 15), Fraction(1, 15)),
    )
    assert validate_placements(
        spans, sample_rate=48000, source_frames=3200, end=Fraction(1, 10)
    ) == ((Fraction(1, 30), Fraction(1, 15)),)


@pytest.mark.parametrize(
    "clock", ["native-reader-presentation", "native-source-presentation-already-projected"]
)
def test_duplicate_reader_projection_is_rejected(clock: str) -> None:
    with pytest.raises(ValueError, match="unprojected"):
        reader_placements(
            {}, sample_rate=48000, source_frames=1, origin=Fraction(0), input_clock=clock
        )


@pytest.mark.parametrize(
    "observation",
    [
        {"packets": []},
        {"mode": "track", "buffers": []},
        {"mode": "track", "buffers": [{"value": 0, "timescale": 0}]},
        {
            "mode": "track",
            "buffers": [
                {
                    "value": 0,
                    "timescale": 48000,
                    "durationValue": 2,
                    "durationTimescale": 48000,
                    "samples": 1,
                }
            ],
        },
        {
            "mode": "audio-mix",
            "buffers": [
                {
                    "value": 0,
                    "timescale": 48000,
                    "durationValue": 2,
                    "durationTimescale": 48000,
                    "samples": 2,
                }
            ],
        },
    ],
)
def test_reader_rejects_ambiguous_or_unmatched_pcm(observation: object) -> None:
    with pytest.raises(ValueError):
        reader_placements(
            observation,
            sample_rate=48000,
            source_frames=1,
            origin=Fraction(0),
            input_clock="reader-pcm-ordinal",
        )


def test_playable_support_keeps_sub_movie_tick_tail(tmp_path: Path) -> None:
    ffmpeg = shutil.which("ffmpeg")
    ffprobe = shutil.which("ffprobe")
    assert ffmpeg is not None
    assert ffprobe is not None
    source, target = tmp_path / "short.wav", tmp_path / "short.mov"
    with wave.open(str(source), "wb") as writer:
        writer.setparams((1, 2, 48000, 0, "NONE", "not compressed"))
        writer.writeframes(b"\x01\x00" * 17)
    command = observed_pcm_mov_command(
        source,
        target,
        spans=(ClockedSpan(0, 17, Fraction(1), Fraction(0)),),
        ffmpeg=Path(ffmpeg),
    )
    subprocess.run(command, check=True, capture_output=True, timeout=10)
    probe = subprocess.run(
        [ffprobe, "-v", "error", "-show_streams", "-of", "json", str(target)],
        check=True,
        capture_output=True,
        timeout=10,
    )
    assert json.loads(probe.stdout)["streams"][0]["duration_ts"] == 17
    readback = subprocess.run(
        [ffmpeg, "-v", "error", "-i", str(target), "-f", "s16le", "-"],
        check=True,
        capture_output=True,
        timeout=10,
    )
    assert readback.stdout == b"\x01\x00" * 17
    with pytest.raises(ValueError, match="new repository-local"):
        observed_pcm_mov_command(source, target, spans=(), ffmpeg=Path(ffmpeg))


def test_playable_support_rejects_native_holes_even_when_output_is_contiguous(
    pcm: Path,
    tmp_path: Path,
) -> None:
    with pytest.raises(ValueError, match="sparse"):
        observed_pcm_mov_command(
            pcm,
            tmp_path / "bad.mov",
            ffmpeg=Path("/unused"),
            spans=(
                ClockedSpan(0, 1600, Fraction(0), Fraction(0)),
                ClockedSpan(1600, 1600, Fraction(2, 30), Fraction(1, 30)),
            ),
        )


def test_playable_support_rejects_non_pcm16(tmp_path: Path) -> None:
    source = tmp_path / "eight-bit.wav"
    with wave.open(str(source), "wb") as writer:
        writer.setparams((1, 1, 48000, 0, "NONE", "not compressed"))
        writer.writeframes(b"\x01" * 17)
    with pytest.raises(ValueError, match="PCM16"):
        observed_pcm_mov_command(source, tmp_path / "bad.mov", spans=(), ffmpeg=Path("/unused"))


@pytest.fixture
def pcm(tmp_path: Path) -> Path:
    source = tmp_path / "source.wav"
    with wave.open(str(source), "wb") as writer:
        writer.setparams((1, 2, 48000, 0, "NONE", "not compressed"))
        writer.writeframes(b"\x01\x00" * 1600 + b"\x02\x00" * 1600)
    return source


def test_gap_keeps_second_support_at_its_native_time(pcm: Path, tmp_path: Path) -> None:
    receipt = write_clocked_pcm(
        pcm,
        tmp_path / "clocked",
        source_sha256=file_digest(pcm),
        spans=(
            ClockedSpan(0, 1600, Fraction(0), Fraction(0)),
            ClockedSpan(1600, 1600, Fraction(2, 30), Fraction(2, 30)),
        ),
        end=Fraction(3, 30),
    )
    document = json.loads(receipt.read_text())
    assert document["missingSupport"] == [["1/30", "1/15"]]
    assert document["pieces"][1]["outputStartSample"] == 3200
    assert document["pieces"][1]["sourceStartSample"] == 1600
    assert document["pieces"][1]["sourceTime"] == "1/15"
    assert document["pieces"][1]["outputTime"] == "1/15"
    assert document["playbackPolicy"] is None
    assert document["sourceCorrespondenceVerified"] is False
    assert document["deliveryApproved"] is False
    with wave.open(document["pieces"][1]["path"]) as reader:
        assert reader.getnframes() == 1600
        assert reader.readframes(1600) == b"\x02\x00" * 1600
    assert file_digest(pcm) == document["source"]["sha256"]
    with pytest.raises(FileExistsError):
        write_clocked_pcm(
            pcm,
            tmp_path / "clocked",
            source_sha256=file_digest(pcm),
            spans=(ClockedSpan(0, 3200, Fraction(0), Fraction(0)),),
            end=Fraction(1, 15),
        )


def test_ffmpeg_resampler_impulses_keep_integer_clock_without_priming_shift(tmp_path: Path) -> None:
    ffmpeg = shutil.which("ffmpeg")
    assert ffmpeg is not None
    source, output = tmp_path / "44100.wav", tmp_path / "48000.wav"
    samples = np.zeros(44100, dtype="<i2")
    samples[[4410, 10001, 11025, 22050, 33075]] = 16000
    with wave.open(str(source), "wb") as writer:
        writer.setparams((1, 2, 44100, 0, "NONE", "not compressed"))
        writer.writeframes(samples.tobytes())
    command = [
        ffmpeg,
        "-nostdin",
        "-v",
        "error",
        "-n",
        "-i",
        str(source),
        "-af",
        "aresample=48000",
        "-c:a",
        "pcm_s16le",
        str(output),
    ]
    subprocess.run(command, check=True, capture_output=True, timeout=20)
    (tmp_path / "resampler-command.json").write_text(json.dumps(command), encoding="utf-8")
    with wave.open(str(output)) as reader:
        assert reader.getnframes() == 48000
        data = np.frombuffer(reader.readframes(48000), dtype="<i2")
    assert int(np.argmax(abs(data[4700:4900]))) == 100
    assert int(np.argmax(abs(data[10800:10950]))) == 85
    assert int(np.argmax(abs(data[11900:12100]))) == 100
    assert int(np.argmax(abs(data[23900:24100]))) == 100
    assert int(np.argmax(abs(data[35900:36100]))) == 100
    with pytest.raises(ValueError, match="fractional"):
        validate_placements(
            (ClockedSpan(0, 1600, Fraction(10001, 44100), Fraction(0)),),
            sample_rate=48000,
            source_frames=48000,
            end=Fraction(1),
        )


def test_cut_translation_is_exact_and_tail_is_missing() -> None:
    assert validate_placements(
        (ClockedSpan(1600, 1600, Fraction(1, 30), Fraction(0)),),
        sample_rate=48000,
        source_frames=3200,
        end=Fraction(1, 15),
    ) == ((Fraction(1, 30), Fraction(1, 15)),)


@pytest.mark.parametrize(
    "second",
    [
        ClockedSpan(1600, 1600, Fraction(1, 60), Fraction(1, 30)),
        ClockedSpan(1600, 1600, Fraction(1, 30), Fraction(1, 60)),
        ClockedSpan(1600, 1600, Fraction(1, 30), Fraction(1601, 48001)),
        ClockedSpan(1600, 1600, Fraction(1601, 48001), Fraction(1, 30)),
        ClockedSpan(1599, 1600, Fraction(1, 30), Fraction(1, 30)),
        ClockedSpan(1600, 1601, Fraction(1, 30), Fraction(1, 30)),
        ClockedSpan(1600, 0, Fraction(1, 30), Fraction(1, 30)),
    ],
)
def test_ambiguous_support_is_not_written(second: ClockedSpan) -> None:
    with pytest.raises(ValueError, match="Ambiguous"):
        validate_placements(
            (ClockedSpan(0, 1600, Fraction(0), Fraction(0)), second),
            sample_rate=48000,
            source_frames=3200,
            end=Fraction(1),
        )


@pytest.mark.parametrize(
    "rate,frames,end",
    [(0, 3200, Fraction(1)), (48000, 0, Fraction(1)), (48000, 3200, Fraction(1, 48001))],
)
def test_invalid_budgets(rate: int, frames: int, end: Fraction) -> None:
    with pytest.raises(ValueError, match="budget"):
        validate_placements((), sample_rate=rate, source_frames=frames, end=end)


def test_extent_cannot_trim_support() -> None:
    with pytest.raises(ValueError, match="exceeds"):
        validate_placements(
            (ClockedSpan(0, 3200, Fraction(0), Fraction(0)),),
            sample_rate=48000,
            source_frames=3200,
            end=Fraction(1, 30),
        )


def test_changed_source_and_external_output_fail(pcm: Path, tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="digest"):
        write_clocked_pcm(
            pcm,
            tmp_path / "bad",
            source_sha256="0" * 64,
            spans=(ClockedSpan(0, 3200, Fraction(0), Fraction(0)),),
            end=Fraction(1, 15),
        )
    with pytest.raises(ValueError):
        write_clocked_pcm(
            pcm,
            Path("/outside"),
            source_sha256=file_digest(pcm),
            spans=(ClockedSpan(0, 3200, Fraction(0), Fraction(0)),),
            end=Fraction(1, 15),
        )


def test_wrong_format(tmp_path: Path) -> None:
    source = tmp_path / "eight.wav"
    with wave.open(str(source), "wb") as writer:
        writer.setparams((1, 1, 48000, 0, "NONE", "not compressed"))
        writer.writeframes(b"\x80" * 100)
    with pytest.raises(ValueError, match="PCM16"):
        write_clocked_pcm(
            source,
            tmp_path / "bad",
            source_sha256=file_digest(source),
            spans=(ClockedSpan(0, 100, Fraction(0), Fraction(0)),),
            end=Fraction(1, 480),
        )


def test_missing_initial_support() -> None:
    span = ClockedSpan(0, 1600, Fraction(0), Fraction(0))
    assert validate_placements(
        (replace(span, output_time=Fraction(1, 30)),),
        sample_rate=48000,
        source_frames=1600,
        end=Fraction(1, 15),
    ) == ((Fraction(0), Fraction(1, 30)),)
