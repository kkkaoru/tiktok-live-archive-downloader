"""External quiet cuts reuse shared measurement and planning on synthetic PCM."""

import hashlib
import json
import math
import shutil
import struct
import wave
from pathlib import Path

import pytest

from replay.core import ReplayError
from replay.external_quiet_cuts import run

RATE = 8000
FFMPEG = shutil.which("ffmpeg")


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _tone(seconds: float, frequency: float = 440.0) -> bytes:
    frames = int(seconds * RATE)
    return b"".join(
        struct.pack("<h", int(20000 * math.sin(2 * math.pi * frequency * i / RATE)))
        for i in range(frames)
    )


def _write(tmp_path: Path) -> Path:
    total = 8.0
    audio = _tone(2.0) + b"\x00\x00" * int(4.0 * RATE) + _tone(2.0)
    assert len(audio) == int(total * RATE) * 2
    stereo = tmp_path / "stereo.wav"
    with wave.open(str(stereo), "wb") as stream:
        stream.setnchannels(2)
        stream.setsampwidth(2)
        stream.setframerate(RATE)
        stream.writeframes(b"".join(a + a for a in [audio[i : i + 2] for i in range(0, len(audio), 2)]))
    per = 1024
    count = (int(total * RATE) + per - 1) // per
    clock = tmp_path / "audio-clock.json"
    clock.write_text(
        json.dumps(
            {
                "streams": [{"sample_rate": str(RATE), "time_base": f"1/{RATE}"}],
                "frames": [
                    {
                        "pts": index * per,
                        "nb_samples": min(per, int(total * RATE) - index * per),
                    }
                    for index in range(count)
                ],
            }
        ),
        encoding="utf-8",
    )
    video = tmp_path / "video-clock.json"
    video.write_text(
        json.dumps(
            {
                "streams": [{"time_base": "1/30"}],
                "frames": [{"pts": index, "duration": 1} for index in range(int(total * 30))],
            }
        ),
        encoding="utf-8",
    )
    labels = tmp_path / "labels.json"
    labels.write_text(
        json.dumps(
            {
                "version": 1,
                "clock": "source-presentation",
                "labels": [
                    {"text": "あ", "start": 0.5, "end": 1.5, "provenance": "w:0"},
                    {"text": "い", "start": 6.5, "end": 7.5, "provenance": "w:1"},
                ],
            }
        ),
        encoding="utf-8",
    )
    config = tmp_path / "config.json"
    config.write_text(
        json.dumps(
            {
                "version": 1,
                "pcmClock": "continuous-decoded-pcm",
                "fps": 30,
                "artifacts": {
                    name: {"path": str(path), "sha256": _sha(path)}
                    for name, path in (
                        ("labels", labels),
                        ("stereoPCM", stereo),
                        ("audioClock", clock),
                        ("videoClock", video),
                    )
                },
            }
        ),
        encoding="utf-8",
    )
    return config


@pytest.mark.skipif(FFMPEG is None, reason="FFmpeg fixture is unavailable")
def test_plans_cut_in_measured_quiet(tmp_path: Path) -> None:
    output = tmp_path / "cuts"
    count = run(_write(tmp_path), output, ffmpeg=str(FFMPEG))
    result = json.loads((output / "cuts.json").read_text(encoding="utf-8"))
    assert result["clock"] == "source-presentation"
    assert result["lineageIsCallerDeclared"] is True
    assert count >= 1
    kept = [
        (cut["source_start_frame"] / 30, cut["source_end_frame"] / 30)
        for cut in result["cuts"]
    ]
    assert any(start <= 0.5 and end >= 1.5 for start, end in kept)
    assert any(start <= 6.5 and end >= 7.5 for start, end in kept)
    assert not any(start < 5.0 and end > 3.0 for start, end in kept)
    assert result["sourceQuiet"]


@pytest.mark.skipif(FFMPEG is None, reason="FFmpeg fixture is unavailable")
def test_rejects_digest_mismatch(tmp_path: Path) -> None:
    config = _write(tmp_path)
    data = json.loads(config.read_text(encoding="utf-8"))
    data["artifacts"]["stereoPCM"]["sha256"] = "0" * 64
    config.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(ReplayError, match="digest"):
        run(config, tmp_path / "out", ffmpeg=str(FFMPEG))
