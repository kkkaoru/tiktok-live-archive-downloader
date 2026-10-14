"""Measured removal of Whisper silence and laughter hallucinations."""

import json
import wave
from pathlib import Path

import numpy as np
import pytest

from replay import asr_hallucination as hallucination
from replay.core import ReplayError

RATE = 16000


def _write_wav(path: Path, samples: np.ndarray) -> Path:
    pcm = np.clip(samples * 32767.0, -32768, 32767).astype("<i2")
    with wave.open(str(path), "wb") as stream:
        stream.setnchannels(1)
        stream.setsampwidth(2)
        stream.setframerate(RATE)
        stream.writeframes(pcm.tobytes())
    return path


def _silence(seconds: float) -> np.ndarray:
    return np.zeros(int(seconds * RATE))


def _voiced(seconds: float) -> np.ndarray:
    return 0.3 * np.sin(2 * np.pi * 220.0 * np.arange(int(seconds * RATE)) / RATE)


def _label(text: str, start: float, end: float) -> hallucination.Label:
    return hallucination.Label(text, start, end)


def _phrase_labels(start: float) -> list[hallucination.Label]:
    return [
        _label("ご", start, start + 0.2),
        _label("視聴", start + 0.2, start + 0.6),
        _label("ありがとうございました", start + 0.6, start + 1.6),
    ]


def test_activity_is_zero_over_silence_and_full_over_voice(tmp_path: Path) -> None:
    wav = _write_wav(tmp_path / "a.wav", np.concatenate([_silence(2.0), _voiced(2.0)]))
    envelope = hallucination.measure(wav)
    assert (envelope.activity(0.2, 1.8), envelope.activity(2.2, 3.8)) == (0.0, 1.0)


def test_label_over_digital_silence_is_removed(tmp_path: Path) -> None:
    wav = _write_wav(tmp_path / "a.wav", np.concatenate([_silence(3.0), _voiced(3.0)]))
    labels = [_label("幻", 0.5, 1.0), _label("こんにちは", 4.0, 4.8)]
    kept, removals = hallucination.filter_labels(labels, hallucination.measure(wav))
    assert ([item.text for item in kept], [item.reason for item in removals]) == (
        ["こんにちは"],
        ["silent"],
    )


def test_label_beside_energy_survives_alignment_slack(tmp_path: Path) -> None:
    wav = _write_wav(tmp_path / "a.wav", np.concatenate([_silence(3.0), _voiced(3.0)]))
    kept, removals = hallucination.filter_labels(
        [_label("はい", 2.4, 2.6)], hallucination.measure(wav)
    )
    assert ([item.text for item in kept], removals) == (["はい"], [])


def test_label_beyond_the_alignment_slack_of_any_energy_is_removed(tmp_path: Path) -> None:
    wav = _write_wav(tmp_path / "a.wav", np.concatenate([_silence(5.0), _voiced(3.0)]))
    kept, removals = hallucination.filter_labels(
        [_label("はい", 3.0, 3.2)], hallucination.measure(wav)
    )
    assert (kept, [item.reason for item in removals]) == ([], ["silent"])


def test_stock_phrase_over_voiced_laughter_is_removed(tmp_path: Path) -> None:
    wav = _write_wav(tmp_path / "a.wav", _voiced(200.0))
    labels = [*_phrase_labels(10.0), _label("次の話題", 20.0, 21.0)]
    kept, removals = hallucination.filter_labels(labels, hallucination.measure(wav))
    assert ([item.text for item in kept], [item.reason for item in removals]) == (
        ["次の話題"],
        ["stock-phrase"],
    )


def test_stock_phrase_run_is_reported_with_its_evidence(tmp_path: Path) -> None:
    wav = _write_wav(tmp_path / "a.wav", _voiced(200.0))
    _, removals = hallucination.filter_labels(_phrase_labels(10.0), hallucination.measure(wav))
    assert (removals[0].text, removals[0].indices, removals[0].activity) == (
        "ご視聴ありがとうございました",
        (0, 1, 2),
        1.0,
    )


def test_closing_phrase_on_speech_energy_is_kept(tmp_path: Path) -> None:
    wav = _write_wav(tmp_path / "a.wav", _voiced(200.0))
    labels = [_label("本編", 100.0, 101.0), *_phrase_labels(195.0)]
    kept, removals = hallucination.filter_labels(labels, hallucination.measure(wav))
    assert (len(kept), removals) == (4, [])


def test_closing_phrase_over_silence_is_removed(tmp_path: Path) -> None:
    wav = _write_wav(tmp_path / "a.wav", np.concatenate([_voiced(100.0), _silence(100.0)]))
    kept, removals = hallucination.filter_labels(
        [_label("本編", 50.0, 51.0), *_phrase_labels(195.0)], hallucination.measure(wav)
    )
    assert ([item.text for item in kept], len(removals)) == (["本編"], 1)


def test_label_straddling_the_phrase_boundary_is_left_alone(tmp_path: Path) -> None:
    wav = _write_wav(tmp_path / "a.wav", _voiced(200.0))
    labels = [
        _label("今日はご", 10.0, 10.5),
        _label("視聴", 10.5, 10.9),
        _label("ありがとうございました", 10.9, 11.9),
    ]
    kept, removals = hallucination.filter_labels(labels, hallucination.measure(wav))
    assert (len(kept), removals) == (3, [])


def test_non_mono_audio_is_refused(tmp_path: Path) -> None:
    path = tmp_path / "stereo.wav"
    with wave.open(str(path), "wb") as stream:
        stream.setnchannels(2)
        stream.setsampwidth(2)
        stream.setframerate(RATE)
        stream.writeframes(b"\x00\x00" * 2 * RATE)
    with pytest.raises(ReplayError, match="mono 16-bit"):
        hallucination.measure(path)


def test_audio_shorter_than_a_frame_is_refused(tmp_path: Path) -> None:
    path = _write_wav(tmp_path / "tiny.wav", np.zeros(10))
    with pytest.raises(ReplayError, match="shorter than one analysis frame"):
        hallucination.measure(path)


def _candidate(path: Path, rows: list[dict[str, object]]) -> Path:
    path.write_text(
        json.dumps({"version": 1, "clock": "input-pcm", "labels": rows}), encoding="utf-8"
    )
    return path


def test_apply_writes_a_new_candidate_with_a_report(tmp_path: Path) -> None:
    wav = _write_wav(tmp_path / "a.wav", np.concatenate([_silence(3.0), _voiced(3.0)]))
    candidate = _candidate(
        tmp_path / "c.json",
        [
            {"text": "幻", "start": 0.5, "end": 1.0, "provenance": "w0"},
            {"text": "声", "start": 4.0, "end": 4.4, "provenance": "w1"},
        ],
    )
    report = hallucination.apply(candidate, wav, tmp_path / "out.json")
    written = json.loads((tmp_path / "out.json").read_text(encoding="utf-8"))
    assert (
        [row["text"] for row in written["labels"]],
        written["clock"],
        report["keptLabels"],
        written["hallucinationFilter"]["removed"][0]["reason"],
    ) == (["声"], "input-pcm", 1, "silent")


def test_apply_never_overwrites_an_existing_output(tmp_path: Path) -> None:
    wav = _write_wav(tmp_path / "a.wav", _voiced(5.0))
    candidate = _candidate(tmp_path / "c.json", [{"text": "声", "start": 1.0, "end": 1.4}])
    (tmp_path / "out.json").write_text("keep", encoding="utf-8")
    with pytest.raises(FileExistsError):
        hallucination.apply(candidate, wav, tmp_path / "out.json")


def test_apply_refuses_to_remove_every_label(tmp_path: Path) -> None:
    wav = _write_wav(tmp_path / "a.wav", _silence(5.0))
    candidate = _candidate(tmp_path / "c.json", [{"text": "幻", "start": 1.0, "end": 1.4}])
    with pytest.raises(ReplayError, match="every label"):
        hallucination.apply(candidate, wav, tmp_path / "out.json")


@pytest.mark.parametrize(
    "document",
    [[], {"labels": []}, {"labels": ["x"]}, {"labels": [{"text": "a", "start": "0", "end": 1}]}],
)
def test_malformed_candidates_are_refused(tmp_path: Path, document: object) -> None:
    wav = _write_wav(tmp_path / "a.wav", _voiced(5.0))
    candidate = tmp_path / "c.json"
    candidate.write_text(json.dumps(document), encoding="utf-8")
    with pytest.raises(ReplayError):
        hallucination.apply(candidate, wav, tmp_path / "out.json")


def test_label_with_end_before_start_is_refused(tmp_path: Path) -> None:
    wav = _write_wav(tmp_path / "a.wav", _voiced(5.0))
    candidate = _candidate(tmp_path / "c.json", [{"text": "a", "start": 2.0, "end": 1.0}])
    with pytest.raises(ReplayError, match="ordered start and end"):
        hallucination.apply(candidate, wav, tmp_path / "out.json")


def test_cli_prints_the_counts(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    wav = _write_wav(tmp_path / "a.wav", np.concatenate([_silence(3.0), _voiced(3.0)]))
    candidate = _candidate(
        tmp_path / "c.json",
        [{"text": "幻", "start": 0.5, "end": 1.0}, {"text": "声", "start": 4.0, "end": 4.4}],
    )
    code = hallucination.main([str(candidate), str(wav), str(tmp_path / "out.json")])
    assert (code, json.loads(capsys.readouterr().out)) == (
        0,
        {"inputLabels": 2, "keptLabels": 1, "removedRuns": 1},
    )


def test_cli_reports_a_refused_input(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    wav = _write_wav(tmp_path / "a.wav", _voiced(5.0))
    candidate = tmp_path / "c.json"
    candidate.write_text("[]", encoding="utf-8")
    code = hallucination.main([str(candidate), str(wav), str(tmp_path / "out.json")])
    assert (code, "ASR_HALLUCINATION_ERROR" in capsys.readouterr().err) == (1, True)
