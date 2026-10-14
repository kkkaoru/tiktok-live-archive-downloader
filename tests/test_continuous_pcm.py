"""Byte-exact PCM selection tests without private media or correction filters."""

import wave
from pathlib import Path

import pytest

from replay.continuous_cuts import write_pcm_spans


@pytest.fixture
def pcm(tmp_path: Path) -> Path:
    path = tmp_path / "source.wav"
    with wave.open(str(path), "wb") as writer:
        writer.setparams((2, 2, 44100, 0, "NONE", "not compressed"))
        writer.writeframes(b"\x01\x00\x02\x00" * 3 + b"\x03\x00\x04\x00" * 3)
    return path


def test_selected_samples_are_byte_identical(pcm: Path, tmp_path: Path) -> None:
    output = tmp_path / "selected.wav"
    assert write_pcm_spans(pcm, output, ((1, 2), (4, 6))) == 3
    with wave.open(str(output), "rb") as reader:
        assert reader.getparams()[:4] == (2, 2, 44100, 3)
        assert reader.readframes(4) == b"\x01\x00\x02\x00\x03\x00\x04\x00\x03\x00\x04\x00"
    with wave.open(str(pcm), "rb") as reader:
        assert reader.getnframes() == 6


def test_large_selection_streams_across_blocks(tmp_path: Path) -> None:
    source = tmp_path / "source.wav"
    with wave.open(str(source), "wb") as writer:
        writer.setparams((1, 2, 44100, 0, "NONE", "not compressed"))
        writer.writeframes(b"\x01\x02" * 70000)
    output = tmp_path / "copy.wav"
    assert write_pcm_spans(source, output, ((0, 70000),)) == 70000
    with wave.open(str(output), "rb") as reader:
        assert reader.readframes(70001) == b"\x01\x02" * 70000


@pytest.mark.parametrize(
    "spans", [(), ((0, 7),), ((2, 2),), ((-1, 1),), ((0, 3), (2, 4)), ((True, 3),)]
)
def test_invalid_selections_create_no_output(
    pcm: Path, tmp_path: Path, spans: tuple[tuple[int, int], ...]
) -> None:
    output = tmp_path / "invalid.wav"
    with pytest.raises(ValueError, match="Invalid ordered"):
        write_pcm_spans(pcm, output, spans)
    assert not output.exists()


def test_source_link_is_rejected(pcm: Path, tmp_path: Path) -> None:
    source = tmp_path / "linked.wav"
    source.symlink_to(pcm)
    with pytest.raises(ValueError, match="regular source"):
        write_pcm_spans(source, tmp_path / "out.wav", ((0, 1),))


def test_missing_source_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="regular source"):
        write_pcm_spans(tmp_path / "missing.wav", tmp_path / "out.wav", ((0, 1),))


def test_existing_output_is_preserved(pcm: Path, tmp_path: Path) -> None:
    output = tmp_path / "existing.wav"
    output.write_bytes(b"keep")
    with pytest.raises(ValueError, match="new output"):
        write_pcm_spans(pcm, output, ((0, 1),))
    assert output.read_bytes() == b"keep"


def test_non_pcm16_is_rejected(tmp_path: Path) -> None:
    source = tmp_path / "pcm8.wav"
    with wave.open(str(source), "wb") as writer:
        writer.setparams((1, 1, 8000, 0, "NONE", "not compressed"))
        writer.writeframes(b"\x80" * 16)
    with pytest.raises(ValueError, match="PCM16 only"):
        write_pcm_spans(source, tmp_path / "out.wav", ((0, 16),))


def test_truncated_source_retains_partial_output(pcm: Path, tmp_path: Path) -> None:
    pcm.write_bytes(pcm.read_bytes()[:-4])
    output = tmp_path / "partial.wav"
    with pytest.raises(ValueError, match="truncated"):
        write_pcm_spans(pcm, output, ((0, 6),))
    assert output.exists()
