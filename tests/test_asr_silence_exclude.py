"""Labels wholly inside exactly-zero input are set aside with their evidence; others stay."""

import json
import wave
from pathlib import Path

import numpy as np
import pytest

from replay import asr_silence_exclude as silence
from replay.asr_silence_exclude import Span
from replay.core import ReplayError

RATE = 10


def block(*values: int) -> np.ndarray:
    return np.array(values, dtype="<i2")


def zeros(seconds: int) -> list[int]:
    return [0] * (seconds * RATE)


def test_a_run_of_zeros_between_voiced_samples_is_found() -> None:
    samples = block(5, *zeros(3), 7)
    assert silence.zero_runs([samples], rate=RATE, minimum_seconds=2) == [Span(1, 31, RATE)]


def test_a_run_shorter_than_the_minimum_is_not_reported() -> None:
    assert silence.zero_runs([block(5, *zeros(1), 7)], rate=RATE, minimum_seconds=2) == []


def test_a_run_at_the_start_and_one_at_the_end_are_found() -> None:
    samples = block(*zeros(2), 9, 9, *zeros(2))
    assert silence.zero_runs([samples], rate=RATE, minimum_seconds=2) == [
        Span(0, 20, RATE),
        Span(22, 42, RATE),
    ]


def test_a_run_that_crosses_a_block_boundary_is_one_run() -> None:
    first, second = block(4, *zeros(2)), block(*zeros(2), 4)
    assert silence.zero_runs([first, second], rate=RATE, minimum_seconds=3) == [Span(1, 41, RATE)]


def test_one_voiced_sample_splits_a_run() -> None:
    samples = block(*zeros(2), 1, *zeros(2))
    assert silence.zero_runs([samples], rate=RATE, minimum_seconds=2) == [
        Span(0, 20, RATE),
        Span(21, 41, RATE),
    ]
    assert silence.zero_runs([samples], rate=RATE, minimum_seconds=3) == []


def test_all_zero_input_is_one_run_and_empty_input_has_none() -> None:
    assert silence.zero_runs([block(*zeros(4))], rate=RATE, minimum_seconds=2) == [
        Span(0, 40, RATE)
    ]
    assert silence.zero_runs([], rate=RATE, minimum_seconds=2) == []


def test_invalid_rates_and_minimums_are_refused() -> None:
    with pytest.raises(ReplayError, match="Invalid rate or minimum silence"):
        silence.zero_runs([], rate=0, minimum_seconds=2)
    with pytest.raises(ReplayError, match="Invalid rate or minimum silence"):
        silence.zero_runs([], rate=RATE, minimum_seconds=0)


def label(text: str, start: float, end: float) -> dict[str, object]:
    return {"text": text, "start": start, "end": end}


def test_a_label_wholly_inside_a_run_is_excluded_with_its_span() -> None:
    labels = [label("あ", 0.5, 1.0), label("ご視聴", 10.0, 12.0), label("い", 30.0, 31.0)]
    kept, excluded = silence.exclude(labels, [Span(80, 200, RATE)])
    assert [item["text"] for item in kept] == ["あ", "い"]
    assert excluded == [
        {
            "index": 1,
            "label": label("ご視聴", 10.0, 12.0),
            "span": {
                "startSeconds": 8.0,
                "endSeconds": 20.0,
                "rule": "every input sample in the span is exactly zero",
            },
        }
    ]


def test_a_label_that_only_partly_overlaps_a_run_stays() -> None:
    labels = [label("あ", 7.5, 9.0), label("い", 19.0, 21.0), label("う", 30.0, 31.0)]
    kept, excluded = silence.exclude(labels, [Span(80, 200, RATE)])
    assert [item["text"] for item in kept] == ["あ", "い", "う"]
    assert excluded == []


def test_a_label_exactly_filling_a_run_is_excluded() -> None:
    kept, excluded = silence.exclude(
        [label("あ", 8.0, 20.0), label("い", 25.0, 26.0)], [Span(80, 200, RATE)]
    )
    assert [item["text"] for item in kept] == ["い"]
    assert [item["index"] for item in excluded] == [0]


def test_a_point_label_inside_a_run_is_excluded() -> None:
    kept, excluded = silence.exclude(
        [label("あ", 10.0, 10.0), label("い", 25.0, 26.0)], [Span(80, 200, RATE)]
    )
    assert [item["text"] for item in kept] == ["い"]
    assert len(excluded) == 1


def test_the_input_labels_are_not_modified() -> None:
    labels = [label("あ", 10.0, 12.0), label("い", 30.0, 31.0)]
    silence.exclude(labels, [Span(80, 200, RATE)])
    assert labels == [label("あ", 10.0, 12.0), label("い", 30.0, 31.0)]


def test_excluding_every_label_or_having_none_is_refused() -> None:
    with pytest.raises(ReplayError, match="Every label lies in digital silence"):
        silence.exclude([label("あ", 10.0, 12.0)], [Span(80, 200, RATE)])
    with pytest.raises(ReplayError, match="Invalid label count"):
        silence.exclude([], [])


def write_wav(path: Path, samples: np.ndarray, *, channels: int = 1) -> None:
    with wave.open(str(path), "wb") as stream:
        stream.setparams((channels, 2, RATE, 0, "NONE", "not compressed"))
        stream.writeframes(samples.tobytes())


def test_the_command_line_writes_a_derived_file_and_leaves_the_parent(tmp_path: Path) -> None:
    pcm = tmp_path / "mono.wav"
    write_wav(pcm, block(7, *zeros(5), 7, 7))
    labels = tmp_path / "labels.json"
    parent = {"version": 1, "labels": [label("あ", 0.0, 0.1), label("ご視聴", 2.0, 4.0)]}
    labels.write_text(json.dumps(parent), encoding="utf-8")
    output = tmp_path / "out.json"
    assert silence.main([str(labels), str(pcm), str(output), "--minimum-seconds", "3"]) == 0
    document = json.loads(output.read_text())
    assert [item["text"] for item in document["labels"]] == ["あ"]
    assert document["excludedAsDigitalSilence"][0]["label"]["text"] == "ご視聴"
    assert document["digitalSilenceSpans"] == [{"startSeconds": 0.1, "endSeconds": 5.1}]
    assert document["parentLabelsFile"] == str(labels)
    assert document["minimumSilenceSeconds"] == 3
    assert json.loads(labels.read_text()) == parent


def test_the_command_line_refuses_stereo_input_and_overwriting(tmp_path: Path) -> None:
    stereo = tmp_path / "stereo.wav"
    write_wav(stereo, block(1, 2, 3, 4), channels=2)
    labels = tmp_path / "labels.json"
    labels.write_text(json.dumps({"labels": [label("あ", 0.0, 0.1)]}), encoding="utf-8")
    with pytest.raises(ReplayError, match="Expected mono PCM16"):
        silence.main([str(labels), str(stereo), str(tmp_path / "o.json"), "--minimum-seconds", "1"])
    existing = tmp_path / "exists.json"
    existing.write_text("x", encoding="utf-8")
    with pytest.raises(ReplayError, match="Refusing to overwrite"):
        silence.main([str(labels), str(stereo), str(existing), "--minimum-seconds", "1"])
