"""Low-energy cue review is conservative and never rewrites recognized words."""

import json
import wave
from pathlib import Path

import numpy as np
import pytest

from replay.caption_speech_review import main, select
from replay.core import ReplayError


def test_silence_is_flagged_without_claiming_semantics() -> None:
    result = select({"captions": [{"text": "笑い声かも", "start": 0, "end": 0.2}]}, np.zeros(100))
    assert result["zeroActivityCues"] == 1
    assert result["semanticAlignmentVerified"] is False
    assert result["samples"] == [
        {
            "index": 0,
            "start": 0.0,
            "end": 0.2,
            "text": "笑い声かも",
            "activeFraction": 0.0,
            "peakDb": -240.0,
        }
    ]


@pytest.mark.parametrize("values", [[], [-1.0], [float("nan")]])
def test_invalid_energy_is_rejected(values: list[float]) -> None:
    with pytest.raises(ReplayError, match="energy"):
        select({"captions": []}, np.asarray(values))


def test_wrong_audio_clock_is_rejected() -> None:
    with pytest.raises(ReplayError, match="clock"):
        select({"captions": [{"text": "はい", "start": 2, "end": 3}]}, np.ones(100))


def test_cli_pins_artifacts_and_refuses_overwrite(tmp_path: Path) -> None:
    plan, audio, output = tmp_path / "plan.json", tmp_path / "audio.wav", tmp_path / "review.json"
    plan.write_text(json.dumps({"captions": [{"text": "はい", "start": 0, "end": 0.2}]}))
    with wave.open(str(audio), "wb") as wav:
        wav.setparams((1, 2, 1000, 0, "NONE", "none"))
        wav.writeframes(np.zeros(1000, dtype="<i2").tobytes())
    assert main([str(plan), str(audio), str(output)]) == 0
    assert json.loads(output.read_text())["measuredCues"] == 1
    with pytest.raises(ReplayError, match="overwrite"):
        main([str(plan), str(audio), str(output)])
