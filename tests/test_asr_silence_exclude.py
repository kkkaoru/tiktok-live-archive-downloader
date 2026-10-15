"""Labels wholly inside exactly-zero input are set aside with their evidence; others stay."""

import json
import wave
from fractions import Fraction
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
    with pytest.raises(ReplayError, match="Invalid rate, minimum silence or amplitude"):
        silence.zero_runs([], rate=0, minimum_seconds=2)
    with pytest.raises(ReplayError, match="Invalid rate, minimum silence or amplitude"):
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


SPAN = [Span(80, 200, RATE)]
FRAME = Fraction(1, 30)


def test_a_label_that_starts_inside_a_run_and_ends_after_it_starts_at_the_run_end() -> None:
    result, clamps, slivers = silence.clamp_partial(
        [label("聴", 10.0, 25.0)], SPAN, minimum_kept=FRAME
    )
    assert result[0]["start"] == 20.0 and result[0]["end"] == 25.0
    assert result[0]["silenceClamp"] == {
        "side": "start",
        "originalStart": 10.0,
        "originalEnd": 25.0,
        "spanStartSeconds": 8.0,
        "spanEndSeconds": 20.0,
        "droppedSeconds": 10.0,
        "rule": "label time inside digital silence is not delivered speech",
    }
    assert clamps == [{"index": 0, "text": "聴", "side": "start"}]
    assert slivers == []


def test_a_label_that_starts_before_a_run_and_ends_inside_it_ends_at_the_run_start() -> None:
    result, clamps, slivers = silence.clamp_partial(
        [label("あ", 5.0, 15.0)], SPAN, minimum_kept=FRAME
    )
    assert result[0]["start"] == 5.0 and result[0]["end"] == 8.0
    silence_clamp = result[0]["silenceClamp"]
    assert isinstance(silence_clamp, dict)
    assert silence_clamp["side"] == "end" and silence_clamp["droppedSeconds"] == 7.0
    assert clamps == [{"index": 0, "text": "あ", "side": "end"}]


def test_a_label_starting_exactly_at_the_run_start_is_moved_to_the_run_end() -> None:
    result, _, _ = silence.clamp_partial([label("い", 8.0, 25.0)], SPAN, minimum_kept=FRAME)
    assert result[0]["start"] == 20.0


def test_labels_with_signal_on_both_sides_or_outside_every_run_are_untouched() -> None:
    labels = [label("う", 5.0, 25.0), label("え", 1.0, 3.0), label("お", 20.0, 22.0)]
    result, clamps, slivers = silence.clamp_partial(labels, SPAN, minimum_kept=FRAME)
    assert result == labels
    assert clamps == []


def test_the_clamped_boundaries_never_fall_inside_the_run_after_rounding() -> None:
    third = [Span(1, 3, 3)]
    start_case, _, _ = silence.clamp_partial([label("あ", 0.4, 2.0)], third, minimum_kept=FRAME)
    end_case, _, _ = silence.clamp_partial(
        [label("い", 0.1, 0.9)], [Span(1, 4, 3)], minimum_kept=FRAME
    )
    assert Fraction(str(start_case[0]["start"])) >= Fraction(1, 1)
    assert Fraction(str(end_case[0]["end"])) <= Fraction(1, 3)


def test_clamping_does_not_modify_its_input() -> None:
    labels = [label("聴", 10.0, 25.0)]
    silence.clamp_partial(labels, SPAN, minimum_kept=FRAME)
    assert labels == [label("聴", 10.0, 25.0)]


def test_the_command_line_can_clamp_partial_overlaps(tmp_path: Path) -> None:
    pcm = tmp_path / "mono.wav"
    write_wav(pcm, block(7, *zeros(5), 7, 7, 7))
    labels = tmp_path / "labels.json"
    parent = {
        "version": 1,
        "labels": [label("あ", 0.0, 0.1), label("聴", 2.0, 8.0), label("ご視聴", 2.2, 4.0)],
    }
    labels.write_text(json.dumps(parent), encoding="utf-8")
    output = tmp_path / "out.json"
    arguments = [str(labels), str(pcm), str(output), "--minimum-seconds", "3", "--clamp-partial"]
    assert silence.main(arguments) == 0
    document = json.loads(output.read_text())
    assert [item["text"] for item in document["labels"]] == ["あ", "聴"]
    assert document["labels"][1]["start"] == 5.1
    assert document["silenceClamps"] == [{"index": 1, "text": "聴", "side": "start"}]
    assert json.loads(labels.read_text()) == parent


def test_a_label_between_two_runs_keeps_only_the_gap_and_a_sliver_is_set_aside() -> None:
    two_runs = [Span(80, 200, RATE), Span(201, 300, RATE)]
    kept, clamps, slivers = silence.clamp_partial(
        [label("聴", 10.0, 25.0), label("い", 40.0, 41.0)], two_runs, minimum_kept=Fraction(1, 5)
    )
    assert [item["text"] for item in kept] == ["い"]
    assert [item["side"] for item in clamps] == ["start", "end"]
    assert slivers == [{"index": 0, "label": label("聴", 10.0, 25.0), "kept": pytest.approx(0.1)}]
    long_enough, _, none = silence.clamp_partial(
        [label("聴", 10.0, 25.0)], two_runs, minimum_kept=Fraction(1, 30)
    )
    assert long_enough[0]["start"] == 20.0 and long_enough[0]["end"] == 20.1
    assert none == []


def test_clamping_every_label_away_is_refused() -> None:
    with pytest.raises(ReplayError, match="Every label lies in digital silence"):
        silence.clamp_partial(
            [label("聴", 10.0, 25.0)], [Span(80, 200, RATE), Span(201, 300, RATE)],
            minimum_kept=Fraction(1, 5),
        )  # fmt: skip


def test_a_quiet_tail_and_isolated_samples_do_not_split_one_silence() -> None:
    samples = block(500, *zeros(3), 6, 8, *zeros(3), 1, *zeros(2), 400)
    assert silence.zero_runs([samples], rate=RATE, minimum_seconds=5) == []
    assert silence.zero_runs([samples], rate=RATE, minimum_seconds=5, maximum_amplitude=8) == [
        Span(1, 84, RATE, 8)
    ]


def test_a_sample_louder_than_the_amplitude_still_ends_the_run() -> None:
    samples = block(500, *zeros(3), 9, *zeros(3), 400)
    assert silence.zero_runs([samples], rate=RATE, minimum_seconds=3, maximum_amplitude=8) == [
        Span(1, 31, RATE, 8),
        Span(32, 62, RATE, 8),
    ]


def test_negative_samples_are_measured_by_magnitude() -> None:
    samples = block(500, *zeros(2), -8, -9, *zeros(2), 400)
    assert silence.zero_runs([samples], rate=RATE, minimum_seconds=2, maximum_amplitude=8) == [
        Span(1, 22, RATE, 8),
        Span(23, 43, RATE, 8),
    ]


def test_the_most_negative_sample_does_not_overflow() -> None:
    samples = block(-32768, *zeros(3), 400)
    assert silence.zero_runs([samples], rate=RATE, minimum_seconds=3, maximum_amplitude=8) == [
        Span(1, 31, RATE, 8)
    ]


def test_an_amplitude_outside_the_supported_range_is_refused() -> None:
    with pytest.raises(ReplayError):
        silence.zero_runs([], rate=RATE, minimum_seconds=2, maximum_amplitude=-1)
    with pytest.raises(ReplayError):
        silence.zero_runs([], rate=RATE, minimum_seconds=2, maximum_amplitude=65)


def test_the_span_rule_names_the_amplitude_when_it_is_not_zero() -> None:
    labels = [{"text": "a", "start": 0.0, "end": 1.0}, {"text": "b", "start": 5.0, "end": 6.0}]
    kept, excluded = silence.exclude(labels, [Span(0, 40, RATE, 8)])
    assert [item["text"] for item in kept] == ["b"]
    assert excluded[0]["span"] == {
        "startSeconds": 0.0,
        "endSeconds": 4.0,
        "rule": "every input sample in the span has magnitude at most 8 LSB",
    }


def test_the_command_line_records_the_amplitude(tmp_path: Path) -> None:
    pcm = tmp_path / "a.wav"
    with wave.open(str(pcm), "wb") as stream:
        stream.setnchannels(1)
        stream.setsampwidth(2)
        stream.setframerate(RATE)
        stream.writeframes(block(500, *zeros(3), 6, *zeros(3), 400).tobytes())
    labels = tmp_path / "labels.json"
    labels.write_text(
        json.dumps(
            {
                "labels": [
                    {"text": "a", "start": 0.0, "end": 0.1},
                    {"text": "x", "start": 0.5, "end": 0.9},
                ]
            }
        ),
        encoding="utf-8",
    )
    assert (
        silence.main(
            [
                str(labels),
                str(pcm),
                str(tmp_path / "o.json"),
                "--minimum-seconds",
                "5",
                "--maximum-amplitude",
                "8",
            ]
        )
        == 0
    )
    document = json.loads((tmp_path / "o.json").read_text(encoding="utf-8"))
    assert document["maximumSilenceAmplitude"] == 8
    assert [item["label"]["text"] for item in document["excludedAsDigitalSilence"]] == ["x"]
