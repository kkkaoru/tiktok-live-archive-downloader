import json
import wave
from pathlib import Path

import pytest

from replay import asr_window
from replay.core import ReplayError
from replay.local_asr import fingerprint


def pcm(path: Path, *, rate: int = 16000) -> Path:
    with wave.open(str(path), "wb") as audio:
        audio.setparams((1, 2, rate, 0, "NONE", "not compressed"))
        audio.writeframes(b"\x01\x00\x02\x00\x03\x00\x04\x00")
    return path


def test_extract_preserves_samples_and_source_clock(tmp_path: Path) -> None:
    source = pcm(tmp_path / "source.wav")
    before = source.read_bytes()
    output = tmp_path / "new"
    clip = asr_window.extract(source, output, start_sample=1, sample_count=2)
    with wave.open(str(clip), "rb") as audio:
        assert audio.getnframes() == 2
        assert audio.readframes(2) == b"\x02\x00\x03\x00"
    receipt = json.loads((output / "window.json").read_text(encoding="utf-8"))
    assert receipt["startSample"] == 1
    assert receipt["endSample"] == 3
    assert receipt["sourceSHA256"] == fingerprint(source)
    assert receipt["windowSHA256"] == fingerprint(clip)
    assert receipt["sourceClockProjected"] is False
    assert source.read_bytes() == before
    assert clip.stat().st_mode & 0o777 == 0o600
    with pytest.raises(FileExistsError):
        asr_window.extract(source, output, start_sample=1, sample_count=2)


@pytest.mark.parametrize("start,count", [(-1, 2), (0, 0), (True, 1)])
def test_invalid_window(tmp_path: Path, start: int, count: int) -> None:
    with pytest.raises(ValueError):
        asr_window.extract(
            tmp_path / "missing", tmp_path / "out", start_sample=start, sample_count=count
        )


def test_over_budget(tmp_path: Path) -> None:
    with pytest.raises(ReplayError, match="60 seconds"):
        asr_window.extract(
            tmp_path / "missing", tmp_path / "out", start_sample=0, sample_count=960001
        )


@pytest.mark.parametrize("rate,count", [(8000, 2), (16000, 5)])
def test_invalid_pcm_or_bounds(tmp_path: Path, rate: int, count: int) -> None:
    source = pcm(tmp_path / "source.wav", rate=rate)
    with pytest.raises(ReplayError, match="in-range"):
        asr_window.extract(source, tmp_path / "out", start_sample=0, sample_count=count)
    assert not (tmp_path / "out").exists()


def test_changed_source(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    source = pcm(tmp_path / "source.wav")
    values = iter(["before", "after"])
    monkeypatch.setattr(asr_window, "fingerprint", lambda _path: next(values))
    with pytest.raises(ReplayError, match="Source changed"):
        asr_window.extract(source, tmp_path / "out", start_sample=0, sample_count=2)


def test_truncated_source(tmp_path: Path) -> None:
    source = pcm(tmp_path / "source.wav")
    with source.open("r+b") as stream:
        stream.truncate(source.stat().st_size - 2)
    with pytest.raises(ReplayError, match="Truncated"):
        asr_window.extract(source, tmp_path / "out", start_sample=0, sample_count=4)


def test_cli(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    source = pcm(tmp_path / "source.wav")
    assert (
        asr_window.main(
            [str(source), str(tmp_path / "out"), "--start-sample", "0", "--sample-count", "4"]
        )
        == 0
    )
    assert "window.wav" in capsys.readouterr().out
