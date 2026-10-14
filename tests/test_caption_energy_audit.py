"""Energy measurements expose unsupported continuous voice and real onset lag."""

import json
import wave
from pathlib import Path

import numpy as np
import pytest

from replay.caption_energy_audit import audit, main, onset_lag, refine, rms_windows
from replay.core import ReplayError


def test_sustained_rise_and_delayed_caption() -> None:
    energy = np.concatenate((np.zeros(100), np.ones(100)))
    assert onset_lag(energy, 1.05) == pytest.approx(0.05)
    result = audit({"captions": [{"start": 1.05}]}, energy)
    assert result["thresholdsPassed"] is True
    assert result["allCuesMeasured"] is True


def test_continuous_voice_and_silence_are_not_invented_onsets() -> None:
    assert onset_lag(np.ones(200), 1) is None
    assert onset_lag(np.zeros(200), 1) is None
    result = audit({"captions": [{"start": 1}]}, np.ones(200))
    assert result["unsupportedCues"] == 1
    assert result["thresholdsPassed"] is False


def test_impulse_is_not_a_sustained_voice_onset() -> None:
    energy = np.zeros(200)
    energy[100] = 1
    assert onset_lag(energy, 1) is None


def test_invalid_onset_and_plan_fail() -> None:
    with pytest.raises(ReplayError):
        onset_lag(np.zeros(100), -1)
    with pytest.raises(ReplayError):
        audit({"captions": []}, np.zeros(100))
    with pytest.raises(ReplayError):
        audit({"captions": [{"start": True}]}, np.zeros(100))


def test_pcm_windows_and_cli(tmp_path: Path) -> None:
    vocals = tmp_path / "vocals.wav"
    with wave.open(str(vocals), "wb") as audio:
        audio.setnchannels(1)
        audio.setsampwidth(2)
        audio.setframerate(1000)
        audio.writeframes(
            np.concatenate((np.zeros(1000), np.full(1000, 10000))).astype("<i2").tobytes()
        )
    assert rms_windows(vocals).shape == (200,)
    plan = tmp_path / "plan.json"
    plan.write_text(json.dumps({"captions": [{"start": 1.05}]}))
    output = tmp_path / "audit.json"
    assert main([str(plan), str(vocals), str(output)]) == 0
    assert json.loads(output.read_text())["thresholdsPassed"] is True
    with pytest.raises(ReplayError):
        main([str(plan), str(vocals), str(output)])


def test_non_pcm16_fails(tmp_path: Path) -> None:
    vocals = tmp_path / "vocals.wav"
    with wave.open(str(vocals), "wb") as audio:
        audio.setnchannels(1)
        audio.setsampwidth(1)
        audio.setframerate(1000)
        audio.writeframes(bytes(100))
    with pytest.raises(ReplayError, match="PCM16"):
        rms_windows(vocals)


def test_missing_local_onset_can_be_measured_in_bounded_larger_context() -> None:
    energy = np.concatenate((np.zeros(100), np.ones(200)))
    assert onset_lag(energy, 1.9) is None
    assert onset_lag(energy, 1.9, search_seconds=1.5) == pytest.approx(0.9)
    assert audit({"captions": [{"start": 1.9}]}, energy)["allCuesMeasured"] is True


def test_refine_uses_measured_onsets_and_retains_recognition() -> None:
    result = refine(
        {"captions": [{"text": "字幕", "start": 1.05, "end": 2.0}]},
        {"records": [{"start": 1.05, "lagSeconds": 0.05}]},
        ((0, 90, 0),),
    )
    assert result["captions"] == [
        {
            "text": "字幕",
            "start": 1.0,
            "end": 2.0,
            "recognizedStart": 1.05,
            "energyOnsetSelected": True,
        }
    ]


def test_refine_rejects_cross_cut_anchor_and_reports_unsupported() -> None:
    result = refine(
        {
            "captions": [
                {"text": "字幕", "start": 1.0, "end": 2.0},
                {"text": "次", "start": 2.1, "end": 3.0},
            ]
        },
        {"records": [{"start": 1.0, "lagSeconds": 0.9}, {"start": 2.1, "lagSeconds": None}]},
        ((15, 90, 0),),
    )
    assert result["onsetRefinement"] == {
        "changedCues": 0,
        "unchangedAnchors": [
            {
                "index": 0,
                "reason": "cut/order/minimum-duration boundary",
                "candidate": pytest.approx(0.1),
            },
            {"index": 1, "reason": "no supported energy onset"},
        ],
        "recognitionClocksPreserved": True,
        "semanticAlignmentVerified": False,
        "evaluationUsesSameEnergyDetector": True,
    }


def test_refine_trims_hold_only_at_the_next_measured_display_onset() -> None:
    result = refine(
        {
            "captions": [
                {"text": "前", "start": 0.5, "end": 1.5},
                {"text": "後", "start": 1.5, "end": 2.5},
            ]
        },
        {"records": [{"start": 0.5, "lagSeconds": 0}, {"start": 1.5, "lagSeconds": 0.1}]},
        ((0, 90, 0),),
    )
    assert result["captions"] == [
        {
            "text": "前",
            "start": 0.5,
            "end": 1.4,
            "recognizedStart": 0.5,
            "energyOnsetSelected": False,
        },
        {
            "text": "後",
            "start": 1.4,
            "end": 2.5,
            "recognizedStart": 1.5,
            "energyOnsetSelected": True,
        },
    ]


def test_refine_rejects_mismatched_measurements() -> None:
    with pytest.raises(ReplayError):
        refine({"captions": []}, {"records": [{}]}, ())
    with pytest.raises(ReplayError):
        refine({"captions": [{"start": "bad"}]}, {"records": [{}]}, ())
    with pytest.raises(ReplayError):
        refine({"captions": [{"start": 1, "end": 2}]}, {"records": [{"start": 2}]}, ())
    with pytest.raises(ReplayError):
        refine(
            {"captions": [{"start": 1, "end": 2}]}, {"records": [{"start": 1, "lagSeconds": 0}]}, ()
        )
